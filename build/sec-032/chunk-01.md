### Before we talk about any parallelism strategy — tell me exactly why a 7B model will not train on a single 80GB H100.

Because in training, the parameters are the *smallest* thing you store. That is the single mental flip that makes every distributed-training technique feel inevitable rather than arbitrary. In inference you hold weights plus a KV cache. In training you hold weights, gradients, optimizer state, activations for the whole backward pass, and communication buffers — and the optimizer state alone is typically 6× the size of the bf16 weights.

Do the accounting for standard mixed-precision Adam training, per parameter:

- bf16 parameter copy used for the forward/backward: **2 bytes**
- bf16 gradient produced by backward: **2 bytes**
- fp32 master weight held by the optimizer: **4 bytes**
- Adam first moment `m` in fp32: **4 bytes**
- Adam second moment `v` in fp32: **4 bytes**

That is **16 bytes per parameter**, conventionally written `2 + 2 + 12`. For 7B: 7e9 × 16 = 112e9 bytes = **112 GB**. On an 80 GB H100 you are 32 GB short before a single activation tensor exists. Add activations and you are typically 150–200 GB deep.

Memorize the decomposition, not the total. It tells you immediately what each technique buys: ZeRO-1 shards the 12 bytes of optimizer state, ZeRO-2 additionally shards the 2 bytes of gradients, ZeRO-3/FSDP additionally shards the 2 bytes of parameters. On 8 GPUs, ZeRO-1 takes you from 16 B/param to 2 + 2 + 12/8 = 5.5 B/param (38.5 GB for 7B); ZeRO-3 takes you to 16/8 = 2 B/param (14 GB). That is the whole ladder.

**📐 Numbers you must know:** mixed-precision Adam = **16 bytes/param**. SGD with momentum = 2 + 2 + 4 + 4 = 12. Pure bf16 weights with 8-bit optimizer states (bitsandbytes `AdamW8bit`) = 2 + 2 + 4 + 1 + 1 = 10. LoRA on a frozen base = 2 bytes/param for the frozen weights plus 16 bytes/param on the ~0.5% of params that are trainable — which is why LoRA on 7B fits in 24 GB and full fine-tuning does not.

**⚠ Trap:** candidates say "7B in bf16 is 14 GB, it fits." That is the *inference* number, and saying it out loud in a training round is an immediate signal you have never launched a training job. The follow-up that catches people is "okay, and what if I use SGD instead of Adam?" — you must be able to answer 12 bytes/param instantly, and note that in practice nobody trains transformers with plain SGD because it needs an order of magnitude more steps.

**🗣 Say this in the room:** "Training memory is 16 bytes per parameter with mixed-precision Adam — 2 for the bf16 weight, 2 for the gradient, 12 for the fp32 master weight plus the two Adam moments. So 7B is 112 GB of state before any activations. Everything from ZeRO-1 to FSDP is just a decision about which of those five buckets you shard and what communication you pay for the privilege."

### Where does activation memory actually come from? Derive it for me.

Activations are the tensors the backward pass needs but the forward pass has already finished with. That is the whole definition, and it explains their behavior: they scale with `batch × sequence × hidden × layers`, not with parameter count, which means they are the term that explodes when you push context length rather than model size.

The reference derivation is Korthikanti et al. (2022). For one standard transformer layer, storing every intermediate needed by backward, with no recomputation and no fused attention:

```
bytes_per_layer = s·b·h · (34 + 5·a·s/h)
```

where `s` = sequence length, `b` = microbatch size, `h` = hidden size, `a` = number of attention heads, assuming 2-byte activations. The `34·s·b·h` term is the sum of all the per-token vectors (QKV projections, attention output, the two MLP tensors, layernorm inputs, dropout masks). The `5·a·s²·b` term is the **attention score matrix** — it is quadratic in sequence length and it is the reason long context was memory-infeasible before FlashAttention.

Work it for a 13B-class model: `h=5120, a=40, L=40, s=4096, b=1`.

- `s·b·h` = 4096 × 1 × 5120 = 20.97e6 bytes
- `5·a·s/h` = 5 × 40 × 4096 / 5120 = 160
- per layer = 20.97e6 × (34 + 160) = 20.97e6 × 194 = **4.07 GB**
- × 40 layers = **163 GB** for a *single* microbatch of one sequence.

Now switch on FlashAttention. It never materializes the `s × s` score matrix — it recomputes tiles of it in the backward pass from Q, K, V in SRAM. The `5·a·s/h` term disappears:

- per layer = 20.97e6 × 34 = **0.71 GB**, × 40 = **28.5 GB**.

Now switch on full activation checkpointing. You keep only the *input* to each transformer block and recompute the interior during backward:

- per layer = 2·s·b·h = 42 MB, × 40 = **1.7 GB**.

**📐 Numbers you must know:** `34·s·b·h` bytes per layer is the FlashAttention-era activation footprint; `2·s·b·h` per layer is the full-checkpointing footprint. The ratio is 17×. Memorize that going from "no recompute" to "full recompute" buys you roughly 17× on activations at a cost of ~33% more compute.

**⚠ Trap:** people quote activation memory per *step* and forget it is per *microbatch in flight*. Under pipeline parallelism with 1F1B you have up to `p` microbatches alive simultaneously on the first stage, so the first stage's activation memory is `p ×` the single-microbatch number. This is the number one reason a pipeline-parallel job OOMs on rank 0 and nowhere else.

### Walk me through DDP at the level of the collective. What is actually on the wire?

DistributedDataParallel is the simplest possible answer to "I have more data than one GPU can chew": replicate the entire model on every GPU, feed each a different microbatch, and after the backward pass average the gradients so every replica applies an identical update and the weights never diverge. Every rank holds a full 16 bytes/param — DDP saves you *time*, not memory.

The collective is an **all-reduce** over the gradient tensors, and the implementation NCCL picks for large messages is ring all-reduce, which is bandwidth-optimal. The ring runs in two phases: a reduce-scatter where each of `N` ranks ends up owning the fully-reduced `1/N` slice of the buffer, then an all-gather where those slices are circulated back. Each phase sends `(N−1)/N × S` bytes per rank, so the total bytes each rank pushes onto the wire is:

```
2 · (N−1)/N · S   ≈  2S  for large N
```

That "≈ 2S" is the number to have in your head. **All-reduce costs twice the payload, independent of world size** — that is precisely why it scales and why naive parameter-server designs did not.

Work the arithmetic for a 7B model on an InfiniBand NDR cluster. Gradients in bf16 are 14 GB. Bytes on the wire per rank ≈ 28 GB. NDR is 400 Gb/s = 50 GB/s per port; assume you achieve 45 GB/s of effective bus bandwidth. Communication time = 28 / 45 = **0.62 s**.

Now the compute you have to hide it behind. Per-GPU microbatch of 8 sequences × 4096 tokens = 32,768 tokens. Model FLOPs ≈ 6N per token forward+backward = 6 × 7e9 × 32,768 = 1.38 PFLOP. At an achieved 400 TFLOPS on an H100 that is **3.44 s** of compute. So the all-reduce is 18% of step time — comfortably overlappable, and DDP will hide essentially all of it. Halve the microbatch to 16k tokens and compute drops to 1.72 s while communication stays at 0.62 s: now it is 36% and you will see the ragged edge. That ratio is the entire content of "when does data parallelism stop scaling."

**🗣 Say this in the room:** "DDP is an all-reduce of the gradient buffer, ring-implemented, so each rank moves about 2× the gradient bytes regardless of world size. Whether it is free depends on one ratio: gradient bytes over interconnect bandwidth, versus tokens-per-GPU times 6N over achieved FLOPS. If communication is under ~20% of compute, bucketed overlap hides it entirely."

### Why does DDP bucket gradients, and what goes wrong when the bucketing is wrong?

If you waited for the backward pass to finish and then fired one giant all-reduce, you would serialize communication behind compute and eat the full 0.6 s in the example above. Bucketing exists to *overlap* them. The mental model is a producer/consumer pipeline you already know: backward produces gradients layer by layer in reverse order, and DDP flushes them to the network in fixed-size batches as soon as each batch is complete, so the wire is busy while the GPU is still computing earlier layers.

Mechanically: at construction, DDP walks `model.parameters()` in *reverse* registration order and packs them into contiguous buckets of `bucket_cap_mb` (default 25 MB). It registers an autograd hook on each parameter's `AccumulateGrad` node. When the last parameter in a bucket has its gradient ready, DDP launches an async all-reduce on that bucket's flat buffer on a side CUDA stream. Reverse order is deliberate — the last layers finish backward first, so bucket 0 (holding the last layers) is ready earliest.

Three failure modes, all of which I have actually debugged:

**Unused parameters.** If a parameter does not participate in the forward pass for a given batch — a conditional branch, a frozen adapter path, an auxiliary head you disabled — its gradient hook never fires, its bucket never completes, and every rank hangs on the collective until the NCCL watchdog kills the job at 30 minutes with a useless timeout message. `find_unused_parameters=True` fixes it by doing a graph traversal each iteration to mark unused params ready, but it costs a full autograd-graph walk per step. The rule I enforce in review is: never ship `find_unused_parameters=True` as a permanent fix. It is a diagnostic. Fix the model so all parameters are used, or explicitly exclude them from the DDP-managed set.

**Bucket size mismatched to the interconnect.** 25 MB is tuned for 10–25 GB/s links. On NVLink at 900 GB/s a 25 MB bucket takes 28 µs and you are entirely launch-latency-bound — larger buckets (100–200 MB) win. On a slow Ethernet cluster, smaller buckets start the wire sooner. This is a measurable knob, not a religion: sweep it.

**Gradient-order nondeterminism.** If your model's execution order varies across ranks (data-dependent control flow, MoE routing), bucket-ready order diverges, ranks issue collectives in different orders, and NCCL deadlocks. Non-obvious, and the reason MoE does not use vanilla DDP for the expert weights.

**⚠ Trap:** `gradient_as_bucket_view=True` is nearly free memory savings (it makes `param.grad` a view into the bucket instead of a separate allocation, saving one full gradient copy — 14 GB on a 7B model) but it silently breaks any code that does `param.grad = something` or holds a reference to `.grad` across a step. Optimizers that detach and cache gradients will read garbage. Check your optimizer before enabling it.

### Gradient accumulation with DDP — what is the subtle correctness bug people ship?

Gradient accumulation is how you get a large effective batch without the memory for it: run `k` microbatches, accumulate `.grad`, then step once. The bug is that vanilla DDP all-reduces on *every* backward, so you pay `k` all-reduces per optimizer step when you only need one. That is not a correctness bug — it is a 4× waste on `k=4` — and the fix is `model.no_sync()` around the first `k−1` microbatches.

The actual correctness bug is subtler and it is about **loss normalization**. Under DDP, the all-reduce computes a *mean* across ranks. Under accumulation, gradients are *summed* across microbatches. So you must divide each microbatch loss by `k` yourself:

```python
for i, batch in enumerate(microbatches):
    ctx = model.no_sync() if i < k - 1 else contextlib.nullcontext()
    with ctx:
        loss = compute_loss(model, batch) / k     # <-- the line people forget
        loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

Forget the `/ k` and your gradients are `k×` too large. This does not crash. It does not produce NaN. It silently makes your effective learning rate `k×` your configured one, and you discover it two days later when the loss curve is worse than the single-GPU baseline and you cannot reproduce the paper's numbers.

The second correctness bug, which is worse because it survives review: **token-count normalization with variable-length sequences.** If you compute mean-over-tokens loss per microbatch and then average across microbatches, microbatches with fewer real (non-padding) tokens get overweighted. The correct reduction is `sum(token_losses) / total_real_tokens_across_all_microbatches_and_ranks`, which requires an all-reduce of the token count. Most training loops in the wild get this wrong and it costs a fraction of a point of loss. Frameworks like torchtitan do it correctly; hand-rolled loops usually do not.

**🗣 Say this in the room:** "Accumulation needs `no_sync()` on all but the last microbatch to avoid `k` redundant all-reduces, the loss divided by `k` because accumulation sums while all-reduce means, and — if sequences are variable length — normalization by globally-reduced real token count rather than by microbatch."

### Explain ZeRO's three stages and give me the communication volume for each.

ZeRO's insight is that DDP's replication of optimizer state, gradients, and parameters across `N` ranks is pure redundancy: at any instant, only one rank *needs* a given slice. So shard the state and materialize the pieces just in time. The genius of the paper is that stages 1 and 2 are free — they cost the same communication as DDP — and only stage 3 costs extra.

Let Ψ be the number of parameters, and count communication volume in "Ψ-units of the dtype you communicate in."

**ZeRO-1 — optimizer state sharded.** Rank `i` owns Adam's `m`, `v` and the fp32 master weights for `1/N` of the parameters. Backward produces full gradients everywhere. Instead of all-reduce, you do a **reduce-scatter** (Ψ), each rank updates its slice, then an **all-gather** of the updated bf16 parameters (Ψ). Total = **2Ψ**. Identical to DDP's all-reduce (which is itself reduce-scatter + all-gather). Memory: 16 → 2 + 2 + 12/N bytes/param.

**ZeRO-2 — gradients also sharded.** Same collectives, but gradients are reduce-scattered *incrementally* during backward and the non-owned portions freed immediately, so you never hold the full gradient buffer. Total = **2Ψ**, still free. Memory: 2 + (2+12)/N.

**ZeRO-3 — parameters also sharded.** Now no rank holds the full parameters, so you must all-gather each layer's weights right before you use it, in forward *and* again in backward, then discard. Volume: all-gather in forward (Ψ) + all-gather in backward (Ψ) + reduce-scatter of gradients (Ψ) = **3Ψ**. Memory: 16/N bytes/param.

So the trade is stated in one line: **ZeRO-3 gives you `N×` memory reduction for 1.5× the communication of DDP.** For 7B on 8 GPUs that is 112 GB → 14 GB per rank.

**📄 Paper:** Rajbhandari et al. (2020), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. It replaced the prior orthodoxy that scaling past one-GPU-worth of state required model parallelism with its awkward code rewrites; ZeRO kept the data-parallel programming model and sharded the state underneath it.

**⚠ Trap:** people say "ZeRO-3 is 3× the communication of DDP." It is 3Ψ versus DDP's 2Ψ, so **1.5×**, not 3×. Getting this wrong is a tell. The related trap is thinking ZeRO-1 is a compromise you take when you cannot afford ZeRO-3 — no: ZeRO-1 is strictly better than DDP with zero downside, so if your model fits under ZeRO-1 you should never run plain DDP.

### ZeRO-3 costs 1.5× the communication. When is that a bad trade, and what do you do instead?

When the extra Ψ of all-gather traffic cannot hide behind compute — which happens precisely when your per-GPU token count is small or your interconnect is slow. This is the single most common misconfiguration I see: a team turns on FULL_SHARD because "it's the memory-safe default," runs at a microbatch of 1, and burns 40% of their cluster on collectives that a less aggressive sharding strategy would not have issued.

Make it concrete. 7B model, 8 GPUs *within one node* on NVLink 4 (~400 GB/s achievable per-GPU all-gather bandwidth) versus 8 GPUs spread across 8 nodes on 400G InfiniBand (~45 GB/s).

- ZeRO-3 extra traffic vs ZeRO-2 = 1Ψ of bf16 params = 14 GB per rank per step.
- Intra-node: 14 / 400 = **0.035 s**. Irrelevant.
- Cross-node: 14 / 45 = **0.31 s**. Against 3.4 s of compute at 32k tokens/GPU that is 9% — acceptable. Against 0.43 s of compute at 4k tokens/GPU it is 72% — catastrophic.

So the decision rule I use:

1. If the model + optimizer state fits per-GPU under ZeRO-1, use ZeRO-1 / `SHARD_GRAD_OP`. Never pay for sharding you do not need.
2. If it does not fit, use `HYBRID_SHARD`: shard fully *within* a node (across the fast NVLink domain) and replicate *across* nodes with a plain all-reduce. You get 8× memory reduction with DDP-level cross-node traffic. This is almost always the right answer for models in the 7B–70B range on 8-GPU nodes.
3. Only go FULL_SHARD across the whole world when one node's worth of HBM genuinely cannot hold a shard — i.e. very large models — and then pair it with tensor parallelism so the world size the shard is spread over is smaller.

**💰 Math:** 512 H100s at a blended $2.50/GPU-hour (**📅 Volatile** — verify current on-demand and reserved rates before your loop) is $1,280/hour. Choosing FULL_SHARD where HYBRID_SHARD would do, at the 4k-tokens/GPU operating point above, wastes roughly 0.31/(0.43+0.31) = 42% of step time on avoidable communication. That is $537/hour, **$12,900/day, $387k/month**, for one config flag. This is why "which sharding strategy" is a real interview question and not trivia.

### When does offloading optimizer state to CPU actually pay, and when is it a trap?

CPU offload converts an HBM capacity problem into a PCIe bandwidth problem, and whether that is a good deal is decided entirely by one ratio: how many bytes must cross PCIe per step versus how many FLOPs you do per step. It pays when you are compute-rich and memory-poor and cannot add GPUs; it is a trap when you are already communication-bound.

The mechanics (ZeRO-Offload, Ren et al. 2021): gradients are reduce-scattered on GPU, copied to host, the Adam update runs on the CPU in fp32 against host-resident `m`, `v` and master weights, and the updated bf16 parameters are copied back. ZeRO-Infinity (Rajbhandari et al. 2021) extends the same idea to NVMe, adding a second tier.

The arithmetic. 7B model, ZeRO-3 over 8 GPUs, so each rank owns 7e9/8 = 875M params of state. Per step across PCIe per rank: gradients down (875M × 2 B = 1.75 GB) + updated params up (1.75 GB) = 3.5 GB. PCIe Gen5 x16 is 64 GB/s theoretical, ~50 GB/s achieved — but *eight GPUs share the host's PCIe root complexes and memory bandwidth*, so budget 10–15 GB/s per GPU realistically. 3.5 / 12 = **0.29 s per step**, plus the CPU-side Adam update itself, which for 875M params on a 64-core host runs roughly 0.2–0.4 s because it is memory-bandwidth-bound on host DRAM (~200 GB/s for a dual-socket server, and Adam touches ~14 bytes/param × 875M = 12 GB of host memory per step → 12/200 = 0.06 s at best, in practice 3–5× that).

So offload adds ~0.5 s per step. If your step is 3.4 s of compute, that is a **15% slowdown to unlock 12 bytes/param of HBM** — a fine trade if the alternative is not training at all. If your step is 0.5 s, you have doubled it.

**⚠ Trap:** offload is often "free" in a micro-benchmark and disastrous at scale because everyone benchmarks it on one node where the host is uncontended. On a real 8-GPU node with a dataloader also hammering host memory and NIC DMA competing for PCIe lanes, the achieved copy bandwidth can be 3× worse than your benchmark. Always measure offload with the real dataloader running.

**🗣 Say this in the room:** "Offload is the right call when you are HBM-bound and have compute headroom — a single-node fine-tune of a model that is one tier too big. At cluster scale I would rather add GPUs and shard, because offload puts the host PCIe and DRAM on the critical path of every step and they are the least observable part of the system."

### bf16, fp16, tf32, fp32 — walk me through the choice and the failure mode of each.

The choice is entirely about **dynamic range versus precision**, and for training, range wins. That is the one-sentence reason bf16 displaced fp16 as the default and it is worth being able to state crisply.

fp16 is 1 sign / 5 exponent / 10 mantissa. Max ~65,504, min normal ~6e-5. bf16 is 1 / 8 / 7 — the *same exponent field as fp32*, so the same ~1e38 range, but only 8 bits of mantissa (7 explicit + implicit). tf32 is a compute-only format inside the tensor cores: 19 bits (1/8/10), used for fp32-declared matmuls, giving fp32 range with fp16-ish precision and ~8× the throughput of true fp32 on Ampere and later.

The failure mode of fp16 is **gradient underflow**. Gradients in deep transformers routinely sit at 1e-6 to 1e-8. Below 6e-5 fp16 goes subnormal and below ~6e-8 it flushes to exactly zero. Those parameters simply stop learning, silently. The fix is loss scaling (Micikevicius et al., 2018): multiply the loss by `S` (e.g. 2^16) before backward so all gradients shift up into fp16's representable band, then divide by `S` before the optimizer step. Dynamic loss scaling adjusts `S` automatically — double it every 2000 clean steps, halve it and *skip the step* whenever an inf/NaN is detected in the gradients. `torch.amp.GradScaler` implements exactly this.

bf16 needs none of that. A gradient of 1e-30 is perfectly representable. You pay in precision — 8 mantissa bits means a relative resolution of 2^-8 ≈ 0.39% — and you pay for it in one specific place, which is the next question.

The decision rule: **on Ampere or newer, use bf16 for everything.** Use fp16 only when you are on a V100 or Turing card that has no bf16 tensor cores, or when you are doing inference with a kernel that only ships an fp16 path. Set `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.set_float32_matmul_precision("high")` so any residual fp32 matmuls use tf32; the default in recent PyTorch is conservative and leaves a large factor on the table.

**⚠ Trap:** "bf16 has no numerical issues" is wrong. bf16 has a *worse* accumulation problem than fp16 — with 8 mantissa bits, summing 10,000 terms sequentially in bf16 loses catastrophic precision. This is fine only because tensor cores accumulate matmuls in fp32 internally and PyTorch's reductions (layernorm, softmax, loss) upcast. If you write a custom kernel that accumulates in bf16, you will get a subtly worse model and no error message.

**📄 Paper:** Micikevicius et al. (2018), *Mixed Precision Training*. It introduced the fp32-master-weight + loss-scaling recipe that made fp16 training viable, replacing full-fp32 training and roughly halving memory and doubling throughput on Volta.

### Why do you need an fp32 master weight? Can't you just keep the bf16 copy and update it in place?

No, and the reason is a single arithmetic fact you should be able to derive at the whiteboard: **the parameter update is typically smaller than the representable gap in bf16, so it rounds to zero and the model stops learning.**

bf16 has 8 bits of mantissa, so consecutive representable values near a weight `w` differ by about `w × 2^-8 = w × 0.0039`. Take a typical transformer weight magnitude `|w| ≈ 0.02`. The gap between adjacent bf16 values there is 0.02 × 0.0039 = **7.8e-5**.

Now the update. With AdamW, the step is `lr × m̂/(√v̂ + ε)`, and the bracketed term is normalized to roughly unit scale, so the update magnitude is ≈ `lr`. At a typical late-training learning rate of 1e-5, the update is **1e-5** — which is smaller than the 7.8e-5 gap. Round-to-nearest sends it to zero. Every step. The weight is frozen and nothing tells you.

The fp32 master weight fixes this by accumulating updates at 24 bits of mantissa (gap ≈ `w × 6e-8` ≈ 1.2e-9 at `w=0.02`), then casting down to bf16 only for the forward pass. The bf16 copy is a *lossy view* of the true parameter; the fp32 copy is the parameter.

This is why the memory equation is 2 + 2 + 12 and not 2 + 2 + 8: those 4 bytes of master weight are not optional bookkeeping.

The one legitimate escape is **stochastic rounding** — instead of round-to-nearest, round up with probability proportional to the fractional distance, so an update of 1e-5 against a 7.8e-5 gap bumps the weight one ULP about 13% of the time and the expectation is correct. This lets you drop the fp32 master and run pure-bf16 optimizer state, saving 4 bytes/param. It is used in production at some labs and is available in a few optimizer implementations, but it is not the PyTorch default and it interacts badly with gradient clipping thresholds. **📅 Volatile** — support here is moving; check what your framework actually implements before claiming it in an interview.

**🗣 Say this in the room:** "bf16 has 8 mantissa bits, so the ULP at a weight of 0.02 is about 8e-5. A late-training AdamW update is around 1e-5. Round-to-nearest makes that update vanish. The fp32 master weight exists so the accumulation happens at 24 bits; the bf16 copy is just what the tensor cores read."

### 🏋 Drill: I want to full-fine-tune a 13B model on 8×A100 80GB. It OOMs. Give me the complete memory accounting and tell me the smallest set of changes that makes it fit.

Twelve minutes, no calculator beyond mental arithmetic, no notes. Pass criterion: you produce the per-GPU byte budget under three configurations and name the specific config flags. Here is the worked answer.

**Step 1 — model state, DDP (the naive attempt).** 13e9 × 16 bytes = **208 GB per GPU**. An A100 is 80 GB. You are 2.6× over before activations. This is the answer to "why won't it fit": DDP *replicates*, so having 8 GPUs (640 GB aggregate) does not help at all. Candidates who divide 208 by 8 and say "26 GB, it fits" have not understood what DDP does.

**Step 2 — model state, ZeRO-1 / `SHARD_GRAD_OP`.** Params 2 + grads 2 + optimizer 12/8 = 5.5 bytes/param → 13e9 × 5.5 = **71.5 GB per GPU**. Technically under 80, but you have 8.5 GB left for activations, CUDA context (~1 GB), NCCL buffers (~1–2 GB), the all-gather staging buffer, and fragmentation. It will OOM on the first long batch. Not viable.

**Step 3 — model state, ZeRO-3 / `FULL_SHARD`.** 16/8 = 2 bytes/param → **26 GB per GPU**, plus a transient all-gather buffer for the largest wrapped unit. If you wrap per transformer block, one block of a 13B model is ~13e9/40 = 325M params = 0.65 GB in bf16, and with forward prefetch you hold two of them, so ~1.3 GB transient. Call it **27.5 GB**. Now you have ~50 GB for activations.

**Step 4 — activations.** From the derivation above with `h=5120, a=40, L=40, s=4096`: with FlashAttention and no recompute, 0.71 GB per layer per sequence × 40 = **28.5 GB per sequence**. So microbatch 1 fits in the 50 GB budget with room; microbatch 2 (57 GB) does not. With full activation checkpointing it is 1.7 GB per sequence, so microbatch 8 costs 13.6 GB and you comfortably fit — at ~33% more compute.

**The answer.** The minimum change set is:

1. `FSDP` with `ShardingStrategy.FULL_SHARD` (or `SHARD_GRAD_OP` only if you also add activation checkpointing and accept microbatch 1).
2. `transformer_auto_wrap_policy` keyed on the decoder-block class — *not* the default, which wraps the whole model as one unit and gives you zero sharding benefit during forward.
3. FlashAttention / SDPA memory-efficient backend on (this is the 17× activation lever, and on a 4k context it is worth more than everything else combined).
4. Activation checkpointing on the transformer blocks if you want a microbatch above 1.
5. `MixedPrecision(param_dtype=bf16, reduce_dtype=fp32)`.

**⚠ Trap:** the drill's hidden question is whether you reach for CPU offload first. Offload is the wrong first move here — you have 50 GB of headroom after FULL_SHARD, you just have not spent it correctly. Reaching for offload before checking your attention backend and wrap policy is a tell that you tune by trial and error rather than by budget.

**💰 Math:** 8×A100 at ~$1.50/GPU-hour (**📅 Volatile**) = $12/hour. Getting to microbatch 8 with checkpointing instead of microbatch 1 without it: tokens/step goes 4,096 → 32,768 (8×), step time goes up by ~1.33× from recompute, so throughput improves ~6×. A run that would take 30 days takes 5. That is $8,640 versus $1,440 — the config choice is worth **$7,200** on one fine-tune.
