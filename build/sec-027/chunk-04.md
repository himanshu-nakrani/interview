### Count the HBM traffic for a naive attention implementation and tell me why it's memory-bound.

Take one head: Q, K, V each `[T, d]` with `d = 128`, `T = 8192`, fp16. The naive implementation does exactly what the math says.

Compute `S = QKᵀ / √d` → an `[8192, 8192]` matrix, which at 2 bytes/element is **134 MB, written to HBM**. Read it back to compute the row max (134 MB read). Read it back again to subtract and exponentiate and write `P` (134 MB read + 134 MB write). Read `P` again for the row sum, and again for `P @ V`. Even a decent implementation that fuses the softmax passes touches the `T×T` matrix at least three times: write it, read it, write the normalized version, read that. Call it **~4 passes × 134 MB = 536 MB of HBM traffic per head per layer**.

Now count the FLOPs for that same head: `QKᵀ` is `2·T²·d` and `P@V` is another `2·T²·d`, so `4·T²·d` = 4 × 8192² × 128 = 4 × 6.71e7 × 128 = **3.44e10 FLOPs**. Arithmetic intensity = 3.44e10 FLOPs ÷ 5.36e8 bytes = **64 FLOP/byte**.

H100's ridge point in bf16 is 989 TFLOP/s ÷ 3.35 TB/s = **295 FLOP/byte**. At 64 you are at 22% of the ridge — deeply memory-bound. The kernel spends most of its time shuttling a matrix you never wanted to HBM and back.

And the memory footprint is the other half of the disaster. That `[T, T]` matrix must be *materialized*, so attention memory is **O(T²)**. For 32 heads at T=8192 in fp16 that is 32 × 134 MB = **4.3 GB for one layer's attention scores**, which is why long-context training used to OOM before it got slow.

**🗣 Say this in the room:** "Naive attention writes and re-reads a T-by-T matrix that never needs to exist. Its arithmetic intensity is about 64 FLOP per byte against an H100 ridge point near 295, so it's memory-bound by a factor of four or five — and its memory is O(T²). FlashAttention is not a micro-optimization of that kernel, it's the observation that you can compute the same answer without ever writing that matrix down."

### Teach me FlashAttention. Not "it's optimized" — the actual argument.

FlashAttention is a **memory-hierarchy argument**, and the way to make it feel inevitable is to notice that the `T×T` score matrix is an *intermediate*, not an output. Nobody consumes `S`; they consume `softmax(S)V`. Every byte you spend writing `S` to HBM is a byte spent on a value you are going to read back once and throw away. The IO-aware framing is: treat HBM as disk, SRAM as RAM, and ask what the minimum number of "disk" reads is to compute the output. Answer: read Q, K, V once each and write O once. Everything above that is waste.

The obstacle is softmax, which is a **global** operation over a row — you cannot normalize until you have seen every score in the row, and you cannot see every score without materializing the row. FlashAttention's engineering contribution is showing that the obstacle is fake: you *can* compute softmax incrementally, keeping a running max and a running sum, and retroactively rescale the partial output when a larger max appears.

So the algorithm is a two-level tile loop entirely in SRAM. Load a block of `Br` query rows into SRAM. Loop over blocks of `Bc` key/value rows: compute the `Br×Bc` score tile (small — it fits in SRAM), update the running max `m`, rescale the running accumulator `O` and the running denominator `ℓ` by `exp(m_old − m_new)`, add the new tile's contribution, discard the tile. After the last K/V block, divide `O` by `ℓ` and write `O` to HBM once.

HBM traffic drops from `O(T²)` to `O(T·d)` for Q and O, plus K and V re-read `T/Br` times — which is why block sizes are chosen so K/V tiles fit in the SM's ~100–228 KB of shared memory. Arithmetic intensity goes from ~64 FLOP/byte to roughly `T/2`, which for T=8192 is 4,096 — an order of magnitude *past* the ridge point. The kernel becomes compute-bound, which is the goal: a compute-bound kernel is one you can actually feed with tensor cores. And attention memory becomes **O(T)** — you store only `m` and `ℓ` per row — which is what made 128k-context training possible at all.

**📄 Paper:** Dao, Fu, Ermon, Rudra, Ré (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS. Contributed the tiled, IO-aware, exactly-equivalent attention kernel using online softmax and backward-pass recomputation. It replaced materialized-score attention and the approximate-attention research program that had been trying to solve the wrong problem (FLOPs) for three years.

**⚠ Trap:** calling FlashAttention an approximation. It is **numerically exact** — bit-comparable up to floating-point reassociation. This distinction matters because the entire prior literature (Linformer, Performer, Reformer, sparse attention) *was* approximate and traded quality for speed. FlashAttention's headline result is that no such trade was necessary; the bottleneck was IO, not FLOPs. Saying "FlashAttention approximates attention" in an interview marks you as having read a blog post rather than the paper.

### Derive online softmax for me and implement it. You have twenty lines.

The identity you need: softmax is invariant to subtracting any constant from the logits, and if you have processed a prefix with max `m` and denominator `ℓ`, and then see new values with max `m'`, you can convert the old partial sums to the new reference point by multiplying by `exp(m − m_new)` where `m_new = max(m, m')`. That correction factor is the whole trick, and it applies identically to the unnormalized output accumulator.

Formally, for the first `j` blocks: `m_j = max(m_{j-1}, rowmax(S_j))`, `ℓ_j = ℓ_{j-1}·e^{m_{j-1}−m_j} + rowsum(e^{S_j − m_j})`, and `O_j = O_{j-1}·e^{m_{j-1}−m_j} + e^{S_j − m_j} V_j`. Divide `O` by `ℓ` once at the end.

```python
import numpy as np

def flash_attention(Q, K, V, Br=64, Bc=64):
    """Exact attention, O(T) memory. Q,K,V: [T, d]."""
    T, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    O = np.zeros((T, d), dtype=np.float32)
    for i in range(0, T, Br):                       # outer: query tile stays in SRAM
        q = Q[i:i+Br] * scale
        m = np.full((q.shape[0], 1), -np.inf)       # running row max
        l = np.zeros((q.shape[0], 1))               # running denominator
        acc = np.zeros((q.shape[0], d), np.float32) # running UNNORMALIZED output
        for j in range(0, T, Bc):                   # inner: stream K/V tiles
            k, v = K[j:j+Bc], V[j:j+Bc]
            s = q @ k.T                             # [Br, Bc] -- never leaves SRAM
            s = np.where(np.arange(i, i+q.shape[0])[:, None]
                         >= np.arange(j, j+k.shape[0])[None, :], s, -np.inf)  # causal
            m_new = np.maximum(m, s.max(axis=1, keepdims=True))
            alpha = np.exp(m - m_new)               # correction for everything so far
            p = np.exp(s - m_new)
            l = l * alpha + p.sum(axis=1, keepdims=True)
            acc = acc * alpha + p @ v
            m = m_new
        O[i:i+Br] = acc / l
    return O

# check against the naive reference
T, d = 256, 64
rng = np.random.default_rng(0)
Q, K, V = (rng.normal(size=(T, d)) for _ in range(3))
S = Q @ K.T / np.sqrt(d)
S = np.where(np.arange(T)[:, None] >= np.arange(T)[None, :], S, -np.inf)
P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
assert np.allclose(flash_attention(Q, K, V), P @ V, atol=1e-5)
```

**📄 Paper:** Milakov & Gimelshein (2018), *Online normalizer calculation for softmax*, showed softmax computable in one pass with a running max/denominator. It is the numerical core FlashAttention tiles on top of. Rabe & Staats (2021), *Self-attention Does Not Need O(n²) Memory*, independently established the memory result; FlashAttention added the IO-aware tiling that made it *fast* as well as small.

**⚠ Trap:** initialising `m` to `0` instead of `−inf`. With `m = 0`, the first block's correction factor `exp(0 − m_new)` is wrong whenever the true max is negative, and you get a subtly incorrect result that is close enough to pass an `atol=1e-2` test and wrong enough to shift your logits. Always `−inf`, and handle the fully-masked-row case (all `−inf`) explicitly or you produce `0/0 = NaN` — which is exactly the bug that bites people implementing custom masks.

### If FlashAttention never stores the attention matrix, how does the backward pass work?

By **recomputation**, which is the same time-for-memory trade as gradient checkpointing, applied at the granularity of one kernel.

The backward pass needs `P = softmax(S)` to compute `dV = Pᵀ dO` and `dS = P ⊙ (dP − rowsum(dP ⊙ P))`. FlashAttention does not store `P`; it stores only the per-row **log-sum-exp** statistic `L = m + log(ℓ)`, which is `T` floats per head instead of `T²`. In the backward kernel it re-runs the tiled forward loop, recomputing each `Br×Bc` score tile in SRAM, and because it already knows the final `L` it can normalize each tile *directly* — `P_tile = exp(S_tile − L)` — without needing a second pass or another running max.

The accounting: you pay roughly one extra forward's worth of attention FLOPs in the backward, so attention's backward cost goes from ~2× forward to ~2.5× forward. You save `O(T²)` of activation memory per head per layer. At T=8192, 32 heads, 80 layers in fp16 that saved memory is 80 × 32 × 134 MB = **343 GB** — which is not "a saving," it is the difference between the run existing and not existing.

**⚠ Trap:** assuming this means FlashAttention is only a training optimization. Inference prefill is a forward pass over thousands of tokens and gets the full IO win with no recomputation cost at all. What inference *decode* does not get is the parallelism, which is a separate problem with a separate fix.

### What did FlashAttention-2 change, and why was FlashAttention-1 leaving performance on the table?

FA1 reached roughly 25–40% of an A100's theoretical peak. That is a good kernel and a bad utilization number, and FA2's contribution was diagnosing why, in three parts.

**Non-matmul FLOPs.** Tensor cores do matmul at ~16× the rate the CUDA cores do everything else, so any non-matmul work in the inner loop is disproportionately expensive. FA1 rescaled the output accumulator by `1/ℓ` on *every* inner iteration. FA2 keeps the accumulator unnormalized throughout and divides exactly once at the end, and defers the max-rescaling bookkeeping similarly. Fewer element-wise ops per matmul op.

**Parallelism over sequence length.** FA1 parallelized over `batch × heads` only — one thread block per (batch, head) pair, looping over the whole sequence inside it. For long sequences with small batch (exactly the training regime people cared about) that leaves SMs idle: batch 1 × 32 heads = 32 thread blocks on a 108-SM A100 is 30% SM occupancy. FA2 adds a **query-block dimension** to the grid: each thread block owns a slice of query rows, and — crucially — query blocks are *independent* under online softmax because each row's running max and denominator are private. No cross-block reduction needed. Now the grid is `batch × heads × ceil(T/Br)`, which saturates the machine.

**Warp-level work partitioning.** Inside a thread block, FA1 split the **K** dimension across the four warps ("split-K"): each warp computed a slice of the scores for all query rows, which meant every warp's partial result had to be written to shared memory and reduced across warps, with `__syncthreads()` between. FA2 flips it to **split-Q**: each warp owns a subset of query rows and iterates over all of K/V for them. A warp's rows are entirely its own, so there is no cross-warp communication and no synchronization in the inner loop. This is the change with the least intuitive name and the largest constant-factor effect.

**📄 Paper:** Dao (2023), *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. Roughly 2× over FA1, reaching 50–73% of theoretical peak on A100. It replaced FA1's batch-and-head-only parallelism and split-K warp layout.

**🗣 Say this in the room:** "FA1 solved the IO problem and left the occupancy problem. FA2 is three fixes: stop rescaling the accumulator every iteration because non-matmul FLOPs are 16× more expensive per unit than matmul FLOPs on tensor cores; parallelize over query blocks so long-sequence small-batch training saturates the SMs; and switch from split-K to split-Q inside the block so warps don't have to synchronize. Same algorithm, roughly double the throughput."

### What does FlashAttention-3 exploit that's specific to Hopper?

Three Hopper features, and the through-line is **asynchrony** — on Hopper the tensor cores and the memory system can run genuinely concurrently with the CUDA cores, and a kernel that keeps them in lockstep wastes most of the machine.

**TMA (Tensor Memory Accelerator)** is a dedicated hardware unit that performs bulk asynchronous copies between global and shared memory from a descriptor, without burning warp cycles on address arithmetic. FA3 uses it to prefetch the next K/V tile while the current one is being consumed.

**WGMMA (warpgroup MMA)** is Hopper's asynchronous matmul instruction operating at warpgroup granularity (four warps, 128 threads), taking operands directly from shared memory and completing asynchronously. It replaces Ampere's synchronous `mma` and is what lets the tensor cores run while other warps do something else.

**Warp specialization and software pipelining.** FA3 splits the warps in a block into **producers** (issuing TMA loads) and **consumers** (issuing WGMMA and doing softmax), and pipelines them so that the softmax of tile `j` — which runs on CUDA cores as exponentials and reductions — overlaps with the WGMMA of tile `j+1` on tensor cores. The paper calls the two-warpgroup variant "ping-pong" scheduling. This is the interesting bit conceptually: softmax is the non-matmul work FA2 tried to *minimize*, and FA3 instead **hides** it behind matmul.

**FP8.** FA3 also supports FP8 attention, using block quantization and incoherent processing (a random orthogonal/Hadamard transform applied to Q and K to spread outliers before quantizing) to keep the error down, and reports substantially lower error than a naive per-tensor FP8 attention baseline.

**📐 Numbers you must know:** FA3 reports 1.5–2.0× over FA2 on H100, reaching roughly **740 TFLOP/s in FP16 (~75% of H100's 989 TFLOP/s dense peak)** and approaching **1.2 PFLOP/s with FP8**. **📅 Volatile:** these are the paper's numbers for its evaluated shapes; kernel performance is shape-dependent and the library has moved. **📄 Paper:** Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao (2024), *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*.

**⚠ Trap:** assuming FA3 is a drop-in speedup on your workload. It targets Hopper specifically — on Ampere you get FA2 — and its wins are largest on long sequences with big head dimensions. On a decode-heavy serving workload with query length 1, FA3's pipelining has almost nothing to pipeline; you want FlashDecoding-style kernels instead. Knowing which kernel your *phase* needs is the actual skill.

### Why isn't FlashAttention enough for decode, and what does FlashDecoding do about it?

Because decode has a query length of **one**, and FlashAttention's parallelism comes from splitting the *query* dimension.

Work it out. In decode, each sequence contributes one query token per head. The natural grid is `batch × heads` thread blocks. At batch 8 with 32 query heads that is 256 thread blocks — fine. At batch 1 with 8 KV heads under GQA, it is **8 thread blocks on a 132-SM H100**: you are using 6% of the machine while attending over a 100,000-token cache. The kernel is not slow because of arithmetic; it is slow because 124 SMs are idle.

The fix is to split the dimension that *is* long: the **KV sequence**. FlashDecoding partitions the KV cache into chunks, assigns each chunk to its own thread block, and has each block compute a partial output plus its partial log-sum-exp. A tiny second kernel then combines the partials — using exactly the online-softmax rescaling identity, since combining `(O_i, m_i, ℓ_i)` pairs is associative. This is a split-K reduction, and it is only sound because softmax has that associative decomposition.

The gain is proportional to how starved you were: at batch 1 with 128k context, splitting into 16 chunks takes you from 8 blocks to 128 blocks and gives a large multiple. At batch 256 you already had 2,048 blocks and splitting buys you nothing but reduction overhead — so engines pick the split factor dynamically from `batch × heads` versus SM count. **FlashDecoding++** (a separate line of work) attacks the remaining synchronization by using a precomputed/unified maximum so that partial softmaxes need no cross-block max agreement.

**📐 Numbers you must know:** the rule is `blocks = batch × kv_heads × split`; you want `blocks ≳ 2 × SM_count` for decent occupancy. H100 SXM5 has **132 SMs**, so target ≳ 264 thread blocks. That inequality tells you immediately whether splitting will help: at batch 64 × 8 KV heads = 512, it will not; at batch 2 × 8 = 16, it will help enormously.

**⚠ Trap:** benchmarking your attention kernel at batch 64 and concluding decode attention is fine, then shipping a product whose p99 path is a single long-context request at batch 1 — an agent doing a 200k-token codebase read, say. The low-batch long-context case is the one that is pathological, and it is exactly the case that AI-product companies (Cursor, Glean, Harvey) hit constantly.

### What problem does FlexAttention solve, and when would you reach for it?

The **attention-variant combinatorial explosion**. Causal, sliding-window, ALiBi, soft-capping, document-boundary masking for packed sequences, prefix-LM bidirectional-then-causal, tree attention for speculative decoding, per-head different window sizes — each is a small modification to the score matrix, and each historically required either a hand-written CUDA kernel (weeks, and now you maintain it) or falling back to materialized `[T, T]` attention with a mask, which throws away everything FlashAttention bought you. That fallback is the real cost: teams silently take a 5–10× regression and an O(T²) memory blowup because they needed a custom mask.

FlexAttention (PyTorch) lets you express the variant as a small Python function over indices and compiles it into a fused Triton kernel. Two hooks: a `score_mod(score, batch, head, q_idx, kv_idx)` that returns a modified score (this is how you write ALiBi, soft-capping, relative position bias), and a `mask_mod(batch, head, q_idx, kv_idx)` returning a boolean (this is how you write causal, sliding-window, document masking). You precompute a **BlockMask** with `create_block_mask`, which records which coarse blocks are entirely masked out so the kernel can **skip them entirely** rather than computing and discarding — that block-sparsity is where the speed comes from on sliding-window and document-packed workloads. Backward is generated automatically.

Reach for it when: you have a genuinely custom mask or score modification, you are in PyTorch, and you would otherwise be choosing between a CUDA rewrite and a dense fallback. Do **not** reach for it when a stock kernel covers your case — plain causal attention through FlashAttention-2/3 will beat a compiled generic kernel, and the specialized library kernels get more tuning attention than the generic path.

**⚠ Trap:** writing a `mask_mod` that is data-dependent on *values* rather than indices. The block-mask precomputation assumes the mask is a function of positions, so it can be computed once and reused; a value-dependent mask cannot be block-skipped and you lose the sparsity win. Same trap as writing a predicate a database index cannot use.

### Derive the roofline model for me, then place prefill and decode on it.

Roofline is one inequality: `achievable FLOP/s = min(peak_FLOP/s, arithmetic_intensity × memory_bandwidth)`, where arithmetic intensity `I = FLOPs / bytes moved from HBM`. Plot `I` on a log x-axis and achievable FLOP/s on a log y-axis: you get a diagonal line of slope 1 (the bandwidth roof, `I × BW`) meeting a horizontal line (the compute roof, peak FLOP/s). The corner is the **ridge point**, `I* = peak_FLOP/s ÷ bandwidth`. Left of it you are bandwidth-bound and adding FLOPs is free; right of it you are compute-bound and reducing bytes is free.

**📐 Numbers you must know:** H100 SXM5 — 989 TFLOP/s dense BF16, 3.35 TB/s HBM3. Ridge point = 989e12/3.35e12 = **295 FLOP/byte**. In FP8 (1,979 TFLOP/s dense) the ridge moves to **591 FLOP/byte** — note that *quantizing pushes you further into the bandwidth-bound regime*, because you doubled compute and left bandwidth alone. A100 80GB SXM: 312 TFLOP/s BF16, 2.04 TB/s → ridge 153 FLOP/byte. Memorize the H100 pair; everything else you derive.

**Prefill.** Processing `T` prompt tokens through a model of `N` parameters: FLOPs ≈ `2·N·T`; bytes moved ≈ `N × dtype_bytes` (the weights, read once, since the whole prompt reuses them). Intensity ≈ `2·N·T / (2N)` = **T** for bf16 — the number of tokens in the batch, near enough. At T = 2,048, intensity is 2,048, which is 7× past the ridge. **Prefill is compute-bound.**

**Decode.** One token per sequence, batch `B`: FLOPs ≈ `2·N·B`; bytes ≈ `N × dtype_bytes` (same weights) `+ KV bytes`. Ignoring KV, intensity ≈ **B**. At batch 1, intensity is 1 — you are 295× to the left of the ridge, using 0.3% of the machine's arithmetic capability. At batch 64, intensity 64, still 4.6× left of the ridge. **Decode is memory-bandwidth-bound until batch sizes in the hundreds**, and that single sentence explains continuous batching, PagedAttention, speculative decoding, MoE serving economics, and why quantizing weights speeds up decode but not prefill.

**🗣 Say this in the room:** "Arithmetic intensity for prefill is roughly the number of tokens in the batch; for decode it's roughly the batch size. The H100 ridge point is about 295 FLOP per byte. So prefill sits far right of the ridge and is compute-bound, decode sits far left and is bandwidth-bound, and every serving design decision — batching, paging, speculation, quantization — is an attempt to move decode rightward."

### Give me the decision procedure for whether a kernel is compute-bound, bandwidth-bound, or launch-bound.

Three measurements, in this order, because each is cheaper than the next.

**Step 1 — is it launch-bound?** Run `nsys profile` and look at the CUDA timeline for gaps between kernels. If the sum of kernel durations is materially less than wall clock, the GPU is idle waiting on the host. Corroborate with the arithmetic: `kernels_per_step × ~5 µs` versus step time. A 32-layer model at ~15 kernels/layer is ~500 launches; at 5 µs that is 2.5 ms of launch overhead, which is trivial next to a 70B's 20 ms step and catastrophic next to a 1B's 1.5 ms step. **Fix:** CUDA graphs, kernel fusion, `torch.compile`, or a bigger batch.

**Step 2 — is it bandwidth-bound?** Compute achieved bandwidth: `bytes_moved / kernel_time`, where `bytes_moved` you get from first principles (weights + activations + KV) or from `ncu`'s `dram__bytes.sum`. Divide by the card's peak. If you are above ~65–70% of peak HBM bandwidth, you are bandwidth-bound and you are doing well — the only remaining moves are **move fewer bytes** (quantize weights, quantize KV, fuse to avoid round-trips) or **reuse more** (raise batch size, prefix cache).

**Step 3 — is it compute-bound?** Compute achieved FLOP/s and compare to the card's peak *for the precision you are actually using*. Above ~60% of peak is a good kernel. If you are below 40% of peak on *both* roofs simultaneously, you are neither — you are **latency-bound**, and the culprit is usually low occupancy (too few thread blocks), a serialized dependency chain, or shared-memory bank conflicts. That is where `ncu --set full` and its "Warp State" section earn their keep: it tells you what warps were stalled *on*.

**⚠ Trap:** using `nvidia-smi` utilization as any of these three. It reports the fraction of sampling intervals in which at least one kernel was resident, so a kernel using 2% of the SMs reads as 100% utilization. It cannot distinguish any of the three cases above. The metrics that mean something are **MBU** (achieved bandwidth ÷ peak) and **MFU** (achieved FLOP/s ÷ peak); build your dashboards on those.

**🔍 Failure taxonomy — "the kernel got slower after a change":** (1) Did bytes moved go up? Check for an added HBM round-trip from a broken fusion. (2) Did occupancy drop? Check register pressure — one added local array can spill and halve the blocks per SM. (3) Did launch count go up? A graph break in `torch.compile` splits one fused kernel into dozens. (4) Did precision change under you? An `.float()` somewhere drops you off the tensor-core path entirely.

### Give me the CUDA execution model in the terms a backend engineer already thinks in.

The hierarchy, mapped onto things you own.

A **thread** is not a thread in your sense — it is a lane. It has private registers and it executes one instruction stream, but it never executes alone.

A **warp** is 32 threads that execute **in lockstep** on one instruction. This is the single most important structural fact, and it has no CPU analogue you have used: if threads in a warp take different branches, the warp executes *both* paths with the inactive lanes masked off — "warp divergence" — so a branch inside a warp costs you the sum of both sides, not the max. The closest analogy is SIMD, but with a scheduler that pretends it is threads.

A **thread block** is 1–1,024 threads (32 warps) that are guaranteed to be resident on **one SM** simultaneously, can share that SM's shared memory, and can synchronize with `__syncthreads()`. This is your unit of cooperation and your unit of scheduling. Blocks are dispatched to SMs by hardware; you do not control placement and you cannot assume any ordering between blocks.

An **SM (Streaming Multiprocessor)** is the actual core: its own register file, its own L1/shared memory, its own warp schedulers, its own tensor cores. H100 SXM5 has **132 SMs**, each supporting up to 64 resident warps (2,048 threads), with 65,536 32-bit registers and roughly 228 KB of combined L1/shared memory per SM on Hopper.

**Occupancy** is resident warps ÷ maximum resident warps per SM. It exists because the GPU hides memory latency by **switching warps at zero cost** — while one warp waits on a 400-cycle HBM load, the scheduler issues from another. Low occupancy means nothing to switch to, so latency is exposed. Occupancy is limited by whichever resource runs out first: registers per thread, shared memory per block, or the hard warp-slot cap.

**⚠ Trap:** treating occupancy as the optimization target. It is not — throughput is. Hand-tuned matmul and attention kernels (CUTLASS, FlashAttention) deliberately run at *low* occupancy, using enormous register tiles and shared-memory buffers per block, because they hide latency with instruction-level parallelism and software pipelining rather than with warp switching. "Increase occupancy" is a reasonable first move on a memory-latency-bound kernel and an actively wrong move on a well-pipelined tensor-core kernel. If someone says "we should raise occupancy" without saying what the warps are stalled on, push back.

### Explain memory coalescing and shared-memory bank conflicts, and tell me how you'd actually see them.

Both are the same lesson at two levels of the hierarchy: **the hardware serves a warp's 32 accesses in a small number of fixed-shape transactions, and your access pattern decides how many.**

**Coalescing (global memory).** The memory system serves requests in 32-byte sectors (128-byte cache lines). If the 32 threads of a warp read 32 consecutive `float32`s, that is 128 contiguous bytes — one 128-byte transaction, four sectors, full efficiency. If the same warp reads with a stride of 32 floats, each thread touches a different sector: 32 transactions to deliver 128 useful bytes out of 1,024 fetched. You have used **12.5% of the bandwidth you paid for**, and on a workload that is bandwidth-bound by definition, that is an 8× slowdown. This is why tensor layouts in kernels are chosen so the fastest-varying index maps to the thread index — it is the same instinct as making your hot query hit a covering index instead of a heap scan.

**Bank conflicts (shared memory).** Shared memory is divided into **32 banks of 4 bytes**, striped so that consecutive 4-byte words live in consecutive banks. A warp can service one access per bank per cycle. If two threads in the warp hit different addresses in the *same* bank, the accesses serialize. The pathological case is a column read of a `float32` array with row stride 32: every thread hits bank `(i·32) mod 32 = 0`, so all 32 accesses serialize — a **32-way conflict**, 32× slower than it should be. The standard fix is one line: **pad the row stride by one element** (`__shared__ float tile[32][33]`), which rotates the bank assignment per row and makes the column read conflict-free. Broadcast — all threads reading the *same* address — is free; the hardware special-cases it.

**How you see them.** `ncu` is the tool, not guesswork. For coalescing, look at `l1tex__t_sectors_per_request` — for a fully coalesced 4-byte-per-thread access you expect ~4 sectors per request; 32 means you are doing scattered loads. For bank conflicts, `ncu` reports shared-memory conflict counters directly, and `ncu --set full` surfaces both in the Memory Workload Analysis section with a plain-English hint. **⚠ Trap:** trying to reason about these from the source. Strides interact with the compiler's vectorization, with `float4` loads, and with the swizzling patterns CUTLASS-style kernels apply deliberately. Measure the counters; do not infer them from a loop nest.
