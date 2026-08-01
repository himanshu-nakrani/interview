### What do tensor cores actually do, and why might my kernel silently not be using them?

A tensor core is a fixed-function unit that executes a small **matrix multiply-accumulate** as one instruction: it consumes tiles of A and B, multiplies them, and accumulates into a wider-precision C, all in a handful of cycles. The programming interface is the `mma`/`wgmma` instruction family at warp or warpgroup granularity — you do not issue it per element, you issue it per tile.

The performance gap is the whole point. On H100 SXM5, dense BF16 through tensor cores is **989 TFLOP/s**; general FP32 arithmetic on the CUDA cores is around **67 TFLOP/s**. That is roughly **15×**. A kernel that misses the tensor-core path is not "somewhat slower," it is on a different machine.

The silent misses, in the order I actually find them:

**Wrong dtype.** Tensor cores want fp16/bf16/fp8/int8 inputs (with TF32 as the automatic path for fp32 matmuls when enabled). A stray `.float()` in your model code — someone casting to fp32 "for stability" in a place that did not need it — moves that GEMM to the CUDA cores.

**TF32 disabled.** For fp32 matmuls, PyTorch will use TF32 tensor cores only if allowed. `torch.backends.cuda.matmul.allow_tf32` and `torch.set_float32_matmul_precision("high")` control it, and defaults have changed across PyTorch versions. If someone set it to `"highest"` for numerical debugging, your fp32 matmuls dropped ~8×.

**Shapes not tile-friendly.** Tensor-core tiles have fixed dimensions; a matmul with an inner dimension of 17, or a batch dimension that leaves the last tile 90% padded, wastes proportionally. This is why model dimensions are always multiples of 128 and vocabularies get padded — a vocab of 32,000 versus 32,768 is a real and measurable difference in the output projection.

**It is not a matmul.** LayerNorm, softmax, activations, rotary embedding, element-wise residual adds — none of these touch tensor cores. If your profile shows most time in these, tensor-core utilization is low *correctly*, and the fix is fusion (fewer HBM round-trips), not precision.

**⚠ Trap:** reading a low "Tensor Core utilization" number in `ncu` and concluding the kernel is broken. Ask what the kernel *is* first. A fused RMSNorm kernel should have zero tensor-core utilization; that is not a bug, that is a kernel doing element-wise work. The metric is only meaningful on kernels whose dominant work is matmul.

### CUDA graphs — what exactly do they eliminate, and show me the arithmetic for why decode cares?

They eliminate **CPU-side kernel launch overhead**, by recording a fixed sequence of kernel launches (with their arguments and dependency structure) once, and then replaying the whole DAG with a single submission.

The mechanism: each `cudaLaunchKernel` costs the host a few microseconds — argument marshalling, driver validation, pushing a command into the stream's queue. Call it **~5 µs** as a working figure. That cost is per launch and independent of how much work the kernel does, which is exactly why it dominates on small kernels.

Do the decode arithmetic. A 32-layer model issues on the order of 15 kernels per layer once you count RMSNorm, the QKV projection, rotary embedding, attention, the output projection, the residual add, the second norm, three MLP GEMMs, and the activation — call it 480 launches per forward pass, plus sampling.

- **8B model, batch 32:** weight read = 16 GB (bf16) ÷ 3.35 TB/s = 4.8 ms at 100% MBU, ~7 ms realistic. Launch overhead = 480 × 5 µs = **2.4 ms**, which is **26% of a 9.4 ms step**. Eliminating it takes you from 9.4 ms to 7.0 ms — a **1.34× throughput gain from a config flag**.
- **70B model, TP4, batch 128:** step is ~19 ms. The same 2.4 ms is 11%. Still worth having, less dramatic.
- **1B draft model in a speculative-decoding loop:** step is ~1.2 ms of real work. 2.4 ms of launch overhead means the host is the bottleneck by 2×, and your draft model — the thing you added to make decoding faster — is now the slow part. CUDA graphs are not optional here.

Replaying a graph costs one submission plus very cheap in-graph node dispatch, so the 2.4 ms collapses to well under 100 µs.

**🗣 Say this in the room:** "Launch overhead is a fixed per-kernel CPU cost, roughly five microseconds, and a decode step is hundreds of tiny kernels. On a 70B it's 10% of the step; on an 8B it's a quarter; on a 1B draft model it's the entire bottleneck. CUDA graphs turn hundreds of launches into one replay. The first thing I check on a small-model deployment is whether someone left `--enforce-eager` on."

### CUDA graphs need static shapes, but batch size changes every iteration. How do engines reconcile that?

By **bucketing and padding**, plus a graceful fallback — and the trade-offs of that choice are a good interview conversation because they mirror a JIT warm-up problem you have already dealt with.

At startup the engine captures a graph for each of a set of batch-size buckets — powers of two and some intermediate values, e.g. `{1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256}`. At each decode step the scheduler rounds the actual running-batch size **up** to the nearest captured bucket, pads the batch tensors with dummy sequences, and replays that graph. The padded lanes compute garbage which is discarded.

Three costs you should name unprompted:

**Warm-up time and memory.** Every captured graph pins its own memory pool for intermediate buffers. Thirteen buckets on a large model can be several GB of graph memory, which is memory *not* available for KV. This is a real capacity trade and it is why `gpu_memory_utilization` interacts with graph capture in ways that surprise people. Capture also adds tens of seconds to startup, which matters for autoscaling responsiveness.

**Wasted work on padding.** At a real batch of 33 you replay the 48-bucket graph and waste 31% of the decode compute. Since decode is bandwidth-bound and the padding lanes do not add weight reads, the waste is smaller than it looks — mostly it is KV reads and attention work for the dummy lanes — but it is not zero.

**The fallback cliff.** Prefill, and any batch larger than the largest captured bucket, runs in eager mode. So a request pattern that pushes you past the top bucket silently loses graph acceleration, and the symptom is a step-time discontinuity at a specific concurrency. Modern engines mitigate this with **piecewise graphs** — capturing the parts of the model that are shape-stable while leaving the attention operation (whose shape genuinely varies with sequence length) outside the graph.

**⚠ Trap:** capturing a graph while any tensor address in it can change. A graph records **pointers**, not variables. If a buffer is reallocated between capture and replay — a caching allocator handing out a different address, a resized KV pool, a LoRA adapter swapped in — the graph reads freed or wrong memory and produces silently wrong output with no error. This is one of the few bug classes in the stack that corrupts results without crashing, and it is why engines allocate a dedicated memory pool for graph capture and pin every buffer the graph touches.

### Write me a fused softmax in Triton and tell me what makes it faster than PyTorch's.

The speedup argument is entirely about HBM round-trips. Naive PyTorch softmax on an `[M, N]` tensor is roughly five separate kernels: `max`, `sub`, `exp`, `sum`, `div`. Each reads the input from HBM and writes its output back — so you move on the order of **8 × M × N × 4 bytes** through HBM to do work that fundamentally requires reading the input once and writing the output once, i.e. `2 × M × N × 4 bytes`. Fusing gets you a ~4× reduction in traffic on a kernel that is purely bandwidth-bound, and therefore roughly a 4× speedup.

```python
import torch, triton
import triton.language as tl

@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)                       # one program per row
    cols = tl.arange(0, BLOCK)                   # BLOCK is a power of two >= n_cols
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                    # numerically safe, stays in SRAM
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK < 2048 else (8 if BLOCK < 4096 else 16)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols,
                              BLOCK=BLOCK, num_warps=num_warps)
    return y

x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)
assert torch.allclose(softmax(x), torch.softmax(x, dim=-1), atol=1e-6)
```

Three things to point at when you present it. `BLOCK` is a `tl.constexpr`, so Triton specializes and unrolls per value — this is why you round up to a power of two and why a different `n_cols` triggers a recompile. The `mask` on load and store is how you handle a non-power-of-two row without a tail loop, and `other=-inf` is load-bearing: pad with `0.0` and your padded lanes win the max. And the whole row lives in registers/SRAM between load and store, which is the entire point.

The same skeleton is the answer if they ask for a fused LayerNorm or RMSNorm — one program per row, load the row once, reduce, transform, store. For RMSNorm you replace the max/exp/sum with `rstd = 1/tl.sqrt(tl.sum(x*x, axis=0)/n_cols + eps)` and store `x * rstd * w`; for LayerNorm you additionally compute the mean and subtract it. The traffic argument is identical: eager PyTorch does four or five HBM round-trips over the row, the fused kernel does one in and one out. The one extra thing a production norm kernel does is **fuse the residual add and write the residual back** in the same pass, because the residual stream is read and written on every block and that round-trip is pure waste.

**⚠ Trap:** this kernel assumes **one row fits in one program's block**. For a 128k-token vocabulary at fp32 that is 512 KB per row, far beyond an SM's shared memory, so it will fail or spill catastrophically. The production version streams the row in tiles with the *online* softmax recurrence from FlashAttention — same identity, applied to a 1-D reduction. If an interviewer asks "what if `n_cols` is 128,000," this is the answer they are fishing for.

### Now write a tiled matmul in Triton, and tell me the two things everyone gets wrong.

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)          # accumulate in fp32
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc += tl.dot(a, b)                             # lowers to tensor-core MMA
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

The structure to narrate: each program computes one `BM × BN` output tile, holding its accumulator in registers for the entire K loop, and streams `BM × BK` and `BK × BN` operand tiles. Arithmetic intensity of the tile is `2·BM·BN·BK` FLOPs over `(BM·BK + BK·BN)·2` bytes = `BM·BN/(BM+BN)` FLOP/byte. For 128×128 tiles that is 64 FLOP/byte; for 256×128 it is 85. **This formula is why tiles are square-ish and large** — it is the entire reason tiling works, and being able to write it down is what separates "I read the tutorial" from "I understand the tutorial."

**Mistake one: accumulating in fp16.** `acc = tl.zeros(..., dtype=tl.float16)` looks like a memory optimization and is a correctness bug. Summing `K` products in fp16 accumulates rounding error that grows with `K`; at `K = 8192` you can lose several significant digits. Tensor cores natively accumulate fp16 inputs into fp32 for exactly this reason. Always accumulate in fp32 and cast on store.

**Mistake two: ignoring L2 locality in the program ordering.** A naive row-major program id assignment means programs running concurrently span an entire row of the output, so they collectively touch all of B — thrashing L2. The standard fix is **grouped ordering** (the "supergrouping" in Triton's own matmul tutorial): reorder program ids so that concurrently-scheduled programs form a compact 2-D block of the output, whose operand footprint fits in L2. This is worth a large single-digit percentage and it is invisible in the math.

**⚠ Trap:** shipping this. Do not — `torch.matmul` dispatches to cuBLAS/CUTLASS kernels that are autotuned per shape per architecture by people who do only this, and you will not beat them on a plain GEMM. Write Triton matmuls when the matmul is **fused with something else** (a GEMM with a bias, an activation, a dequantization, or a custom epilogue), because that is where you save an HBM round-trip that cuBLAS cannot save for you.

### When is Triton the right tool, when is it the wrong one, and when should you write CUDA C++?

The decision rule I use, in order:

**Use a library.** cuBLAS/CUTLASS for GEMMs, cuDNN where applicable, FlashAttention for attention, and the fused kernels already shipping in vLLM/SGLang. These are autotuned per architecture by full-time specialists. If your kernel is a plain instance of a solved problem, writing it yourself is a negative-value activity.

**Use `torch.compile`.** For element-wise chains, normalizations, activation+residual patterns, and anything Inductor can fuse. This is Triton, generated for you, with no maintenance burden. Reach for hand-written Triton only after `torch.compile` has failed to fuse something you can see it should have fused.

**Write Triton** when you have a **fusion opportunity a library cannot express**: a GEMM with a nonstandard epilogue, a dequantize-and-matmul for a custom quantization format, a custom attention variant with a mask FlexAttention cannot express, a fused optimizer step. Triton's value proposition is that it gives you tile-level control (blocks, `tl.dot`, masks) while the compiler handles the hard parts — thread-level assignment inside the block, shared-memory allocation, vectorization, and software pipelining. You get maybe 80–95% of an expert CUDA kernel for 10% of the effort and a tenth of the code.

**Write CUDA C++** when you need something Triton abstracts away from you: warp-specialized producer/consumer pipelines, explicit TMA descriptors, `wgmma` scheduling, cluster-level cooperation, custom swizzling. This is the FlashAttention-3 regime. It is also where the last 5–20% lives. **📅 Volatile:** Triton's coverage of Hopper/Blackwell async features has been expanding release over release, so the line between "Triton can do this" and "you need CUDA" moves — check before asserting it.

**🎯 Targeted:** the honest scoping note. If you are interviewing at Nvidia, Together, Fireworks, Baseten, Modal, or a frontier lab's inference team, you need to be able to *write* the Triton kernels above unaided and discuss warp specialization. If you are interviewing at Cursor, Perplexity, Notion, Sierra, Harvey, Glean, or a big-tech applied team, you need to be able to *read* them, explain FlashAttention as a memory-hierarchy argument, and know when a kernel is the bottleneck — and you will almost never be asked to write one.

### My Triton kernel is slower than PyTorch eager. Walk me through debugging it.

Five checks, cheapest first, and the ordering matters because four of the five are configuration rather than code.

**1. Are you benchmarking correctly?** GPU work is asynchronous, so a naive `time.time()` around a kernel launch measures the launch, not the work. Use `triton.testing.do_bench` (which handles warm-up, `torch.cuda.synchronize`, and L2 cache flushing between reps) or at minimum synchronize and discard the first several iterations. I would say half of "my kernel is slower" reports die here. The L2 flush matters more than people expect: without it, a repeated benchmark serves the input from a warm 50 MB L2 and reports a bandwidth number the kernel will never see in production.

**2. Did you include compilation time?** Triton JIT-compiles per `constexpr` specialization on first call. If your benchmark loop varies `BLOCK`, you are timing the compiler.

**3. Is `num_warps` sane?** Too few warps and you cannot hide memory latency; too many and register pressure per thread drops, causing spills to local memory (which is HBM wearing a costume). Sweep `num_warps` in `{1, 2, 4, 8, 16}` and `num_stages` in `{2, 3, 4}` — or just wrap the kernel in `@triton.autotune` with a config list and let it search. On a bandwidth-bound kernel this alone frequently swings 2×.

**4. Are your loads masked but unaligned or strided?** Print the generated PTX/SASS (`kernel.asm['ptx']` on a compiled handle) or check `ncu`'s `l1tex__t_sectors_per_request`. A pointer expression with a non-unit stride in the fastest-varying dimension produces uncoalesced loads and an easy 4–8× loss on a bandwidth-bound kernel.

**5. Is the kernel actually the right thing to fuse?** Compute the theoretical floor: `bytes_that_must_move / peak_bandwidth`. If your kernel is already at 70% of that, it is not slow, and the eager version being faster means eager is doing *less work* — usually because PyTorch dispatched to a fused library kernel you did not know existed, or because your version has an extra materialization.

**⚠ Trap:** register spilling. Triton will happily compile a kernel whose tile does not fit in the register file; it spills to local memory, which is backed by HBM, and your carefully-tiled kernel now does more HBM traffic than the naive version. The symptom is a kernel that is fine at `BM=64` and catastrophically worse at `BM=128`. `ncu` reports local memory traffic explicitly — if it is nonzero on a kernel that should have none, that is your answer.

### `torch.compile` — what does it actually do, and where do graph breaks destroy the win?

Three stages, and you should be able to name all three because interviewers use it to check whether you have used it or read about it.

**TorchDynamo** hooks CPython's frame evaluation API and traces Python bytecode into an FX graph, with **guards** — runtime checks (this tensor is still fp16 on cuda with this shape, this Python int is still 4) that decide whether a compiled artifact is still valid. **AOTAutograd** captures the backward graph ahead of time so the backward can be compiled and fused too. **TorchInductor** lowers the graph to fused kernels — generating **Triton** for GPU and C++/OpenMP for CPU — doing loop fusion, memory planning, and buffer reuse. `mode="reduce-overhead"` additionally wraps the result in CUDA graphs; `mode="max-autotune"` searches kernel configurations.

The win is fusion: a sequence like `norm → matmul → bias → SiLU → mul` in eager is five kernels and five HBM round-trips of the activation; compiled, it can be one or two. On element-wise-heavy models this is routinely 1.3–2×.

**Graph breaks** are where Dynamo cannot trace and falls back to the interpreter, splitting your one graph into several with eager code in between. Each break means: the fusion window closes, intermediate tensors get materialized to HBM, and any CUDA graph must be re-entered. Two or three breaks in a transformer block can erase the entire benefit.

The reliable causes: **data-dependent control flow on tensor values** (`if loss > 0:`, `while not converged:`), **`.item()` / `.tolist()` / `float(tensor)`** — anything forcing a device-to-host sync, **printing or logging a tensor**, **numpy interop**, **calling an unsupported C extension or a custom op without a registered meta/fake implementation**, and **mutating global state** mid-graph.

How to find them: `TORCH_LOGS="graph_breaks,recompiles" python train.py` prints each break with the offending line, and `torch._dynamo.explain(fn)(*args)` gives you a structured report. For a kernel you *believe* is fully traceable, compile with `fullgraph=True` so a break becomes a hard error rather than a silent regression — that is the setting I require in review for anything in the serving hot path.

**⚠ Trap:** **recompilation thrash**, which is worse than a graph break and much harder to see. Guards include shapes, so if your batch size varies over a wide range, Dynamo recompiles per shape until it hits `torch._dynamo.config.cache_size_limit` (default 8), then gives up and falls back to eager **permanently for that frame**. The symptom is a service that is fast in the load test (one shape) and slow in production (many shapes), with no error anywhere. Fixes: mark the dynamic dimension with `torch._dynamo.mark_dynamic`, or `dynamic=True`, or bucket and pad your batch sizes the way an inference engine does. `TORCH_LOGS="recompiles"` shows you which guard failed and why.

### Nsight Systems, Nsight Compute, or the PyTorch profiler — give me the decision procedure.

They answer three different questions and using the wrong one wastes a day.

**PyTorch profiler** (`torch.profiler.profile` with `record_shapes=True`, `with_stack=True`, exported via `export_chrome_trace`) answers **"which PyTorch operator is expensive, and what Python called it?"** It attributes GPU time to `aten` ops and gives you a stack trace back to your model code. This is where you start, always, because it is the only tool that speaks your source code's language. It will tell you "62% of the time is in `aten::scaled_dot_product_attention`" or "there are 4,000 tiny `aten::add` calls," which usually ends the investigation.

**Nsight Systems** (`nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o out python serve.py`) answers **"where is the time going across CPU, GPU, memory transfers and communication, and what is overlapping with what?"** It is a timeline, not a counter dump. This is the tool for launch-bound diagnosis (gaps in the CUDA stream), for CPU/GPU overlap questions, for seeing NCCL collectives serialize against compute in a tensor-parallel deployment, and for seeing that your dataloader is starving the GPU. Annotate with `torch.cuda.nvtx.range_push/pop` (or NVTX ranges around your scheduler phases) so the timeline is labelled with your own concepts instead of raw kernel names.

**Nsight Compute** (`ncu --set full -k regex:my_kernel -o report python bench.py`) answers **"why is this one kernel slow?"** Per-kernel hardware counters: achieved occupancy, memory throughput against peak, the roofline chart, sectors per request, shared-memory bank conflicts, warp stall reasons, register spills. It is expensive — it serializes and replays kernels to collect counters, so a full profile can be 100× slower than the real run — and you only point it at a kernel you have *already* identified as the problem.

**The procedure:** PyTorch profiler → find the expensive op or the pathological op count. If the GPU has gaps or you suspect overlap issues → `nsys` for the timeline. If one kernel is dominant and you want to know why → `ncu` on that kernel alone. Never start with `ncu`.

**🗣 Say this in the room:** "Torch profiler for attribution, Nsight Systems for the timeline, Nsight Compute for one kernel's counters. The mistake is starting with Nsight Compute — it tells you a kernel achieved 41% of peak bandwidth, which is useless if that kernel is 3% of your step time. I want to know where the time is before I want to know why a kernel is slow."

### 🏋 Give me the drill set for this material. What should I be able to do unaided, and how do I know I've passed?

Five drills, each timed, each with a hard pass criterion. Do them with no autocomplete — Anthropic, DeepMind, xAI and several quant shops prohibit AI tools in live rounds, so practicing with one trains a skill you cannot use.

**Drill 1 — KV and concurrency arithmetic. 5 minutes, no calculator beyond mental math.** Given: 80 layers, 8 KV heads, head_dim 128, fp16 KV, TP=4 on 80 GB cards, 8k context. Produce: bytes per token, KV per sequence, weights per GPU, usable KV pool, maximum concurrency, and the same numbers with FP8 KV. **Pass:** 0.3125 MB/token, 2.56 GB/seq, 35 GB/GPU, ~156 GB pool, ~60 sequences, doubling to ~120 with FP8 KV — all within 10%, out loud, no notes.

**Drill 2 — online softmax from scratch. 20 minutes.** Write the tiled attention loop in NumPy and assert `allclose` against the naive reference including a causal mask. **Pass:** correct on the first run, `m` initialized to `−inf`, the accumulator rescaled by `exp(m_old − m_new)`, and the final division done once outside the inner loop.

**Drill 3 — continuous-batching scheduler. 40 minutes.** The one earlier in this section. **Pass:** the block-accounting invariant holds after every step, LIFO preemption terminates, and the admission loop `break`s rather than `continue`s at the head of the queue.

**Drill 4 — roofline placement. 10 minutes, whiteboard.** Draw the roofline for H100 in BF16 and FP8, mark the ridge points, place prefill at T=2,048 and decode at batch 1, 32, and 256, and explain which optimizations move which point and in which direction. **Pass:** ridge points of ~295 and ~591 FLOP/byte derived not recalled, decode's intensity identified as ≈ batch size, and the observation that FP8 moves the ridge *right*, making decode relatively more bandwidth-bound.

**Drill 5 — the deploy-regression tree. 15 minutes, spoken.** Someone reads you "throughput down 40% after a deploy, model unchanged." Produce the four-metric discriminator (cached-token ratio, preemption count, KV pool occupancy, prefill:decode token ratio), map each fingerprint to a cause, and say which log line you would read first. **Pass:** you check the engine startup log (KV block count, launch args) *before* the metrics, and you finish by naming the canary that would have caught it.

**🗣 Say this in the room** (when asked how you prepared): "I can size a KV cache and a concurrency ceiling on a napkin, I can write online softmax and a continuous-batching scheduler from memory, and I have a written decision tree for a throughput regression. Those three cover most of what I've actually needed on-call."

### I'm interviewing at an AI product company, not Nvidia. Why should I know any of this?

Because the failures that wake you up are engine failures, and because the questions you *will* be asked are engine questions wearing product clothing.

Three concrete places it shows up in an applied loop. **Cost**: a product engineer who does not understand prefix caching writes a system prompt with a timestamp in it and burns five figures a month invisibly — I showed that arithmetic earlier and it is a real, common, undetected regression. **Latency**: a product engineer who does not understand the TTFT/ITL trade will tune `max_num_batched_tokens` for throughput, watch p99 triple, and have no vocabulary to explain it to their PM. **Capacity**: "can we support 10,000 concurrent users?" is answerable in ninety seconds from `bytes_per_token × context × concurrency` versus pool size, and unanswerable without it — and that question comes up in every applied-AI system design round I have seen.

The calibration I would actually apply, honestly:

**Must be automatic, regardless of employer.** The KV bytes-per-token formula and a concurrency ceiling. Prefill compute-bound versus decode bandwidth-bound and what follows from it. Continuous batching and why static batching is a head-of-line-blocking bug. PagedAttention as virtual memory, including the 60–80% → <4% number. Prefix caching mechanics and its Merkle-chain invalidation semantics. FlashAttention as an IO argument, with "it's exact, not approximate" said out loud. The two scheduler knobs and the latency/throughput curve. The four-metric regression discriminator.

**Should be conversational.** Roofline and ridge points. CUDA graphs and launch overhead. Preemption swap-versus-recompute. RadixAttention versus block-hash caching. FA2 and FA3's contributions at one sentence each. What `torch.compile` does and what a graph break costs.

**🎯 Only for AI-infra and inference-provider loops.** Writing Triton unaided. Warp-level work partitioning. TMA/WGMMA and warp specialization. Bank-conflict swizzling. Reading SASS. Nsight Compute counter interpretation.

**🗣 Say this in the room:** "I don't write kernels. But the KV cache is my capacity model, the scheduler is my tail-latency model, and prefix caching is my unit-economics model — so I need to be able to read the engine and argue with its defaults. The three production incidents I'd expect on this system are a prefix-cache invalidation that shows up as a bill, a preemption thrash loop that shows up as throughput collapse, and a long prefill that shows up only in p99 ITL. I'd rather be able to name those than be able to write a matmul."
