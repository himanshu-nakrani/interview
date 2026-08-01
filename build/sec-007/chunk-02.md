### Where does "2N FLOPs per token" come from? Derive it rather than quoting it.

Mental model: **every parameter in a dense model is a multiply-accumulate that gets used exactly once per token, and a multiply-accumulate is two floating-point operations.** That's the whole derivation. Two FLOPs per parameter per token, so 2N per token forward.

Make it concrete on one linear layer. y = xW with x ∈ R^{d_in}, W ∈ R^{d_in×d_out}. Computing y requires d_in × d_out multiplications and about the same number of additions: 2·d_in·d_out FLOPs, and d_in·d_out is exactly the parameter count of W. Every matmul in the network has this property, so summing over all layers gives 2N FLOPs per token, where N is the total parameter count. For a batch of B sequences of length T you have B·T tokens, so the forward pass costs 2N·B·T.

Three things are being swept under the rug, and you should name them before the interviewer does. First, **elementwise operations are ignored** — softmax, GELU/SiLU, RMSNorm, the residual adds. They're O(B·T·d) rather than O(B·T·d²), so they're a fraction of a percent of the FLOPs. They are emphatically *not* a fraction of a percent of the runtime, because they're memory-bound and the matmuls are not; this is the entire reason kernel fusion exists. Second, **the attention score computation is not counted**, because its cost depends on T and not on N — I'll come back to when that matters. Third, the **embedding lookup is a gather, not a matmul**, so the embedding table's parameters cost ~0 FLOPs at the input side (they do cost 2·d·V at the output side, via the LM head, which is why the head is often the single most expensive matmul in a small model).

**📐 Numbers you must know:** forward = 2N per token, backward ≈ 4N per token, so training = 6N per token. Total training compute C ≈ 6ND for N parameters and D tokens. Inference prefill = 2N per token; inference decode = 2N per *generated* token. These five numbers let you estimate anything in this field on a napkin, and interviewers will ask you to.

**⚠ Trap:** applying 2N to a mixture-of-experts model using total parameter count. MoE decouples parameters from compute — that's the entire point of the architecture. A model with 400B total parameters and 30B active per token costs 2 × 30e9 = 6e10 FLOPs per token, not 8e11. When you cite a FLOP number for an MoE, say "active parameters" explicitly. And note the flip side: the *memory* footprint follows total parameters, so an MoE is cheap in compute and expensive in HBM — the opposite trade from what people assume.

### And why 6N for training? Where do the other 4N come from?

Backprop through a linear layer requires two matmuls where the forward required one, and each is the same size as the forward. That's the whole answer, but you should be able to name both matmuls.

For y = xW, given the incoming gradient ∂L/∂y you must compute two things:

- **∂L/∂x = (∂L/∂y)Wᵀ** — needed to keep propagating backward to the previous layer. Contracts over d_out, costs 2·d_in·d_out per token.
- **∂L/∂W = xᵀ(∂L/∂y)** — needed to actually update the weights. Contracts over the batch-and-time axis, costs 2·d_in·d_out per token.

Two matmuls, each the same FLOP cost as the forward's one. Forward 2N + backward 4N = **6N per token**, and multiplying by D tokens gives Kaplan's C ≈ 6ND. The optimizer step itself is elementwise — Adam is a handful of ops per parameter, so ~10N FLOPs per *step*, not per token, and at a batch of a million tokens that's five orders of magnitude below the matmuls. Ignore it in FLOP accounting; do not ignore it in memory accounting.

Two refinements worth having ready.

**Gradient checkpointing changes the constant.** If you discard activations in the forward and recompute them during the backward, you pay an extra forward pass: 8N per token instead of 6N, a 33% compute increase. In practice full recompute lands closer to +30–40% wall-clock, and *selective* recompute — recomputing only the cheap-to-recompute, expensive-to-store tensors like the attention softmax output, while keeping the matmul outputs — lands closer to +5–10% for most of the memory saving. Selective recompute is what modern training stacks actually do.

**The first layer's ∂L/∂x is wasted work.** You compute the gradient with respect to the input embeddings at layer 0 and then throw it away (unless you're doing input-gradient-based attribution). It's one layer out of 32; nobody optimizes it.

**🗣 Say this in the room:** "Forward is 2N per token because every parameter is one multiply-accumulate. Backward is 4N because each linear layer needs two gradient matmuls — one for the input gradient and one for the weight gradient — each the same size as the forward. Six total, so C ≈ 6ND. Gradient checkpointing pushes it to 8N with full recompute, or about 6.5N with selective."

### Estimate the GPU-days to pretrain a 7B model on 1 trillion tokens. Show me the arithmetic.

**💰 Math, step by step:**

1. **Total compute.** C = 6ND = 6 × 7×10⁹ × 1×10¹² = **4.2 × 10²² FLOPs**.
2. **Per-GPU throughput.** An H100 SXM does roughly 990 TFLOP/s dense bf16 (ignore the sparsity-doubled marketing number; you will not get it). Call it 9.9 × 10¹⁴ FLOP/s peak.
3. **Apply MFU.** Model FLOPs Utilization for a well-tuned dense pretraining run at this scale is typically 35–50%. Take **40%**: 9.9 × 10¹⁴ × 0.40 = 3.96 × 10¹⁴ FLOP/s achieved.
4. **Divide.** 4.2 × 10²² / 3.96 × 10¹⁴ = **1.06 × 10⁸ GPU-seconds** = 29,500 GPU-hours = **≈ 1,230 H100-days**.
5. **Wall-clock.** On a 256-GPU cluster: 1,230 / 256 ≈ **4.8 days**. On 1,024 GPUs: ~1.2 days, minus the scaling efficiency you lose to communication — assume 85–90% at that width, so call it 1.4 days.
6. **Dollars.** At a market rate of roughly $2/H100-hour on a reserved cluster: 29,500 × $2 = **≈ $59,000** for the compute alone. **📅 Volatile:** H100 hourly rates have moved a lot and continue to; verify the current number before quoting it.

Then add the things the naive estimate omits, because the interviewer's follow-up is always "and what did you leave out?" Data preparation and tokenization is a large CPU job. Failed runs — the honest multiplier on a first-time-through pretraining effort is 1.5–3× total spend, because you will restart from a loss spike, discover a data bug at 200B tokens, and re-tune the LR schedule. Checkpoint storage: a 7B checkpoint with optimizer state is ~112 GB; saving every 1,000 steps for 250,000 steps is not something you keep all of. Evaluation compute during the run. And the salaries, which dominate everything above at this scale.

**🗣 Say this in the room:** "6ND gives 4.2e22 FLOPs. An H100 at 40% MFU delivers about 4e14 FLOP/s, so that's roughly 1.1e8 GPU-seconds — about 1,200 H100-days, five days on 256 cards, and around $60k of compute at $2 a card-hour. I'd budget 2× that for failed runs, and the real cost is engineer-months, not GPU-hours."

**⚠ Trap:** quoting MFU above 60% for a dense pretraining run. That number gets reported for carefully-tuned specific configurations and it is not what you get. Quoting 70% signals you have read a blog post rather than watched a training dashboard. Conversely, if you're at 15% MFU, that is a real bug — usually a small micro-batch, a bad sequence-parallel config, or a data loader stalling the GPU, and the first thing I'd do is profile for GPU idle time before touching the model.

### When does the 2N approximation break down? At what context length does attention start to matter?

The 2N estimate counts weight FLOPs. Attention scores use no weights, so they're invisible to it — but they scale with T while weight FLOPs don't, so there is a crossover, and being able to find it is what separates "I memorized 6ND" from "I understand it."

Derive the attention term. For one query token attending over T keys, in one head: QKᵀ is T dot products of length d_h = 2·T·d_h FLOPs, and the AV weighted sum is the same, 2·T·d_h. Total 4·T·d_h per head; over H heads that's 4·T·d_model (since H·d_h = d_model); over L layers, **4·L·T·d_model FLOPs per token** in the forward pass. (Under causal masking, averaged over the whole sequence, the effective T is T/2 — I'll keep the conservative full-T version.)

Now the weight term. A standard transformer layer has ≈ 4d² attention parameters and ≈ 8d² MLP parameters, so N ≈ 12·L·d², and the weight FLOPs per token are 2N = 24·L·d². Take the ratio:

  attention / weights = (4·L·T·d) / (24·L·d²) = **T / (6·d_model)**.

That's the rule I keep in my head. **Attention FLOPs equal weight FLOPs when T ≈ 6 × d_model.**

Plug in numbers. For d_model = 4096 (a 7–8B model), crossover is at T ≈ 24,600 tokens. At T = 2,048, attention is 2048/24576 = **8% of FLOPs** — safely ignorable, which is exactly why "2N" became the standard approximation in the 2020–2022 era when 2k context was normal. At T = 32,768 it is 133% of the weight FLOPs — attention is now the *majority* of the compute. At 128k it is 5.3× the weight FLOPs, and your model's parameter count has become nearly irrelevant to prefill cost.

Two consequences that make this worth deriving rather than memorizing. First, **the crossover scales with d_model**, so bigger models tolerate longer context before attention dominates: a d = 8192 model doesn't cross over until ~49k tokens. Second, this is a *prefill* story. During decode with a KV cache, you attend over T keys but only for one new query, so the attention FLOPs per generated token are 4·L·T·d_h·H... which is the same expression, but now compared against 2N for a single token. At 128k context decode, attention is again 5× the weight compute — except decode is memory-bound, so what actually kills you is reading the cache, not the FLOPs. Different bottleneck, same T-dependence.

**⚠ Trap:** the FLOP crossover is *not* the same as the point where attention starts hurting you in production. Memory hits first. The `[B,H,T,T]` score tensor is quadratic in bytes as well as FLOPs, which is what motivated FlashAttention — it computes attention in tiles and never writes the T×T matrix to HBM, turning attention from O(T²) memory to O(T). **📄 Paper:** Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — an exact, tiled, recomputation-based attention kernel that removed the quadratic HBM traffic; it replaced both naive attention and the approximate-attention literature that had been trying to solve the wrong bottleneck.

### Give me the four-term memory equation for training, and size a full fine-tune of a 7B model.

**Mental model: model weights are the small part.** Every backend engineer's first instinct is "7B parameters × 2 bytes = 14 GB, fits on any card." The training-time footprint is eight times that, and the four terms are why.

The equation, per parameter, for standard mixed-precision AdamW:

1. **Weights** — bf16 copy used for forward/backward: **2 bytes/param**
2. **Gradients** — bf16 (sometimes fp32): **2 bytes/param**
3. **Optimizer state** — Adam's first and second moments in fp32: **8 bytes/param**; plus the fp32 master copy of the weights that the optimizer actually updates: **4 bytes/param**
4. **Activations** — everything saved in the forward for use in the backward: depends on batch × sequence × depth, *not* on parameter count

Terms 1–3 sum to **16 bytes per parameter**, and 18 if your framework also keeps fp32 gradients. For 7B:

- weights: 7e9 × 2 = **14 GB**
- gradients: 7e9 × 2 = **14 GB**
- fp32 master weights: 7e9 × 4 = **28 GB**
- Adam m and v: 7e9 × 8 = **56 GB**
- **subtotal: 112 GB of static state**, before a single activation.

An 80 GB H100 cannot hold that. This is the number to have memorized, because it instantly explains why LoRA/QLoRA became the default for anything under a research budget, and why ZeRO exists.

Activations, term 4, with gradient checkpointing at layer granularity: you store only each layer's input, so B·T·d·L·2 bytes. For B = 4, T = 4096, d = 4096, L = 32: 4 × 4096 × 4096 × 32 × 2 = **4.3 GB**. Without checkpointing you store many intermediates per layer — the norm outputs, the qkv projections, the attention output, the MLP's intermediate at d_ff = 3.5d — and the figure is an order of magnitude higher, comfortably 40–80 GB at this shape, which is why nobody trains long-sequence models without checkpointing.

**📐 Numbers you must know:** 16 bytes/param for AdamW mixed precision (2 + 2 + 4 + 8). 2 bytes/param for bf16 inference. 4 bytes/param at int32-free fp32 inference. 1 byte/param at int8, 0.5 at int4. Memorize the 16, because the follow-up question in every training round is "so how do you make it fit?"

**⚠ Trap:** forgetting the fp32 master weights. Candidates confidently recite "2 + 2 + 8 = 12 bytes" and get 84 GB, then conclude a 7B fine-tune fits on an 80 GB card — and it doesn't. Mixed precision keeps an fp32 master copy precisely because a bf16 weight has ~8 mantissa bits, and an update of relative size 10⁻⁴ against a weight of order 1 rounds to zero. That master copy is the difference between "fits" and "OOM," and knowing it is a strong tell that you've actually run a training job.

### So it doesn't fit on one 80 GB card. What do you actually change, in what order?

I'd work down this ladder, cheapest intervention first, and I would state the ladder out loud in an interview because the ordering *is* the answer.

**1. Don't full fine-tune.** For 90% of applied use cases, LoRA is the right call, and it's not a compromise — it's the correct default. Freeze the base (14 GB bf16), train r = 16 adapters over all linear layers (~40M params). Optimizer state and gradients now cost 40e6 × 16 = **0.64 GB** instead of 98 GB. Total: ~15 GB + activations, comfortably on one card with a real batch size. QLoRA quantizes the frozen base to NF4, taking it to ~4 GB and letting you fine-tune a 7B on a 24 GB consumer card.

**2. Gradient checkpointing.** As above: ~30% more compute (or ~5–10% with selective recompute) for roughly an order of magnitude less activation memory. **📄 Paper:** Chen et al. (2016), *Training Deep Nets with Sublinear Memory Cost* — showed you can train an n-layer net in O(√n) activation memory by checkpointing every √n layers and recomputing, at the cost of one extra forward pass.

**3. Cheaper optimizer state.** 8-bit Adam (bitsandbytes) cuts the moments from 8 bytes to 2, saving 42 GB on a 7B. Adafactor factorizes the second moment into row and column statistics, cutting it to O(d) per matrix instead of O(d²). Both cost some quality; 8-bit Adam costs very little in my experience and is underused.

**4. Shard across GPUs — ZeRO / FSDP.** **📄 Paper:** Rajbhandari et al. (2020), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* — partitions optimizer state (stage 1), then gradients (stage 2), then parameters themselves (stage 3) across data-parallel ranks instead of replicating them. On 8 GPUs, ZeRO-2 puts the 84 GB of gradient+optimizer state at 10.5 GB per card; ZeRO-3/FSDP shards the weights too, at the cost of all-gathering them per layer during forward and backward, which is bandwidth you need NVLink to afford.

**5. Reduce the batch, then recover it with gradient accumulation.** Activations scale linearly with micro-batch. Micro-batch 1 with 32 accumulation steps has the same effective batch and the same convergence as micro-batch 32, at a fraction of the activation memory and worse GPU utilization. This is the lever you pull last because it's the one that costs throughput most directly.

**⚠ Trap:** reaching for model parallelism (tensor or pipeline) before exhausting this list. Tensor parallelism introduces an all-reduce per layer and only makes sense inside a node with NVLink; pipeline parallelism introduces bubbles and a microbatch scheduling problem. For a 7B, if you're reaching for TP you have skipped four cheaper options. For a 70B+ you have no choice, and that's a different conversation.

### Derive the KV cache size formula and compute it for a 70B-class model at 128k context.

Mental model, and this is the bridge that lands with a backend engineer: **a KV cache is a per-request memo table whose eviction policy you do not control and whose size you cannot bound at admission time.** It exists because at decode step t, the keys and values for tokens 1…t−1 are byte-identical to what you computed at step t−1 — recomputing them turns an O(T) generation into O(T²). But unlike a Redis cache, it is not shared across requests, it grows monotonically for the life of the request, and it lives in the same HBM as your weights.

The formula, derived by counting. Per token, per layer, you store one K vector and one V vector, each of size n_kv_heads × head_dim, in your KV dtype:

  **bytes/token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element**

The leading 2 is K and V. Note what is *absent*: batch size (it's per token, so multiply later) and n_query_heads (with GQA, only the KV heads count).

Llama-3-70B-class config: L = 80, n_kv_heads = 8, head_dim = 128, bf16 (2 bytes).

  2 × 80 × 8 × 128 × 2 = 327,680 bytes = **320 KiB per token**.

At 128k context (131,072 tokens): 131,072 × 327,680 = 4.295 × 10¹⁰ bytes = **40 GiB for a single sequence**. The model weights are 140 GB in bf16, so on an 8-way tensor-parallel node with 8×80 = 640 GB total, weights take 140 GB and *one* 128k-context request takes another 40 GB. You can serve about twelve of them concurrently before HBM is gone, and that is with zero headroom for activations or fragmentation.

Now the comparison that makes the point about GQA. Llama-2-7B is MHA: L = 32, 32 KV heads, head_dim 128 → 2 × 32 × 32 × 128 × 2 = **512 KiB per token**. Llama-3-8B is GQA-8: 2 × 32 × 8 × 128 × 2 = **128 KiB per token**. **A 7B model with MHA has a 1.6× larger KV cache per token than a 70B model with GQA.** State that in an interview and watch the room reorder its priors: KV cache size is governed by architecture, not by parameter count.

**📐 Numbers you must know:** 128 KiB/token for an 8B GQA-8 model; 320 KiB/token for a 70B GQA-8 model; ×8 if the model is MHA; ÷2 if you quantize the cache to fp8. Multiply by context length, then by concurrency. This is the single most-asked arithmetic in an inference-serving round.

**💰 Math on why this is the capacity planner:** an 8B model on one H100 (80 GB) has 16 GB of weights, leaving ~60 GB for cache after activations and overhead. At 128 KiB/token that's 60e9/131072 ≈ **458,000 tokens of total cache**. Sixty-four concurrent users at 8k context each = 512k tokens — you're already over. At 4k each, 256k tokens, you fit with room. Your maximum concurrency is a division problem, and the number that comes out of it is what your autoscaler should be tracking instead of CPU.

### Follow-up: why does GQA cut the cache but not the compute? Show me.

Because grouped-query attention shrinks the K and V *projections*, and the projections were never where the FLOPs were.

Walk the two costs separately. With MHA at d = 4096, H = 32, d_h = 128, the four projections W_q, W_k, W_v, W_o are each `[4096, 4096]`, so attention holds 4d² = 67M parameters per layer. With GQA-8, W_k and W_v become `[4096, 1024]` — a quarter the size — so attention holds 2d² + 2(d²/4) = 2.5d² = 42M per layer. That is a real 37% cut in *attention* parameters, but the MLP at SwiGLU with d_ff = 14336 holds 3 × 4096 × 14336 = 176M per layer, so total per-layer parameters fall from 243M to 218M: about **10% fewer parameters and 10% fewer weight FLOPs**. Nice, not transformative.

Now the attention-score computation. You still have 32 query heads. Each still attends over all T keys. The 8 cached KV heads are *broadcast* — each is reused by 4 query heads — so the score matmul is still `[B, 32, T, 128] × [B, 32, T, 128]ᵀ`, exactly the same shape and exactly the same FLOP count as MHA. **The arithmetic is unchanged; only the operand storage shrank.** That is the entire trick: you trade a 4× reduction in bytes for zero reduction in flops, and since decode is memory-bandwidth-bound, bytes are what you were paying for.

The KV cache falls exactly 4×, from 512 KiB/token to 128 KiB/token at this config, and that is the number that shows up in your serving capacity.

**📄 Paper:** Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Query Checkpoints* — interpolated between MHA and Shazeer's (2019) multi-query attention by sharing K/V across groups of query heads, and showed you can uptrain an existing MHA checkpoint into GQA cheaply. MQA (1 KV head) was faster but lost measurable quality; GQA recovered nearly all of it at 8 heads.

**⚠ Trap:** claiming GQA "makes attention 4× faster." It does not touch attention FLOPs. It makes *decode throughput* faster, indirectly, by shrinking the bytes you read from HBM per step and by letting you hold more concurrent sequences — which raises batch size, which raises arithmetic intensity. The causal chain runs through memory, and stating it as a compute win is the tell that you memorized the outcome.

### Explain arithmetic intensity, and use it to tell me why decode is memory-bound and prefill is not.

Mental model: **a GPU is a machine with two separate budgets — FLOPs per second and bytes per second — and every kernel spends them in a fixed ratio determined by its algorithm. If your ratio is below the machine's ratio, the FLOP units sit idle waiting on memory, and no amount of a faster GPU helps.** This is the roofline model, and it is the single most useful mental tool in inference engineering.

Arithmetic intensity = FLOPs performed / bytes moved from HBM. The machine's balance point for an H100 SXM is 9.9×10¹⁴ FLOP/s ÷ 3.35×10¹² bytes/s ≈ **295 FLOPs per byte**. Below that intensity you are memory-bound; above it, compute-bound.

**Decode, batch size 1.** To generate one token you read every weight once — 2N bytes at bf16 — and perform 2N FLOPs. Intensity = 2N/2N = **1 FLOP per byte**. You are 295× below the machine balance. The tensor cores are idle 99.7% of the time and you are, precisely, a memory-copy engine that occasionally multiplies.

**Decode, batch size B.** The weights are read once and reused across all B sequences in the batch, so bytes stay ≈ 2N while FLOPs become 2N·B. Intensity ≈ **B FLOPs per byte**. So the batch size at which decode becomes compute-bound is *approximately the machine balance point*: **B ≈ 300 on an H100**. (Approximately, because KV-cache reads scale with B and don't amortize, which pushes the real crossover higher, especially at long context.) This is why continuous batching is not a nice-to-have — it is the only lever that moves you off the memory roofline.

**Prefill.** You process T tokens at once through the same weights: bytes ≈ 2N, FLOPs ≈ 2N·T. Intensity ≈ T. At T = 2,048 you are at 2,048 FLOPs/byte, far above 295, solidly compute-bound. Prefill and decode are *different workloads on the same weights*, and that asymmetry is why serving stacks separate them — chunked prefill, or full prefill/decode disaggregation onto different hardware pools.

**🗣 Say this in the room:** "Decode at batch 1 has an arithmetic intensity of about 1 FLOP per byte; an H100 needs about 295 to saturate its tensor cores. So single-stream decode uses well under 1% of the GPU's math throughput and is purely a bandwidth problem. Batching is what fixes it — intensity rises roughly linearly with batch size — which is why continuous batching and KV-cache capacity, not FLOPs, are the things I'd size a deployment around."

**⚠ Trap:** concluding "so buy a GPU with more FLOPs." If you are at intensity 1, doubling FLOP throughput changes your decode latency by zero. The levers that actually work are all bandwidth-or-bytes levers: higher-bandwidth memory (H200's 4.8 TB/s over H100's 3.35), fewer bytes per weight (int8/fp8/int4 quantization halves or quarters the read), fewer bytes per token of cache (GQA, MLA, fp8 KV), or more work per byte read (bigger batch, speculative decoding — which is the trick of verifying k draft tokens in one weight-read).

### Estimate single-stream decode throughput for a 7B model in bf16 on one H100. First principles, no benchmarks.

**💰 Math:**

1. **Bytes that must move per generated token.** At batch 1 you read every weight once: 7×10⁹ params × 2 bytes = **14 GB per token**.
2. **Time to move them.** H100 SXM HBM3 bandwidth is 3.35 TB/s = 3,350 GB/s. 14 / 3,350 = **4.18 ms per token**.
3. **Theoretical ceiling.** 1 / 0.00418 = **239 tokens/second**.
4. **Apply achieved-bandwidth efficiency.** Real kernels hit 70–85% of peak HBM bandwidth. At 78%: 239 × 0.78 ≈ **186 tok/s**.
5. **Add the KV cache read.** At 4k context with a GQA-8 7B-class model (128 KiB/token), the cache is 4,096 × 131,072 = 0.54 GB — about 3.8% on top of the 14 GB, so ~180 tok/s. At 128k context the cache is 16.8 GB, *larger than the weights*, and per-token traffic goes from 14 GB to 30.8 GB: throughput drops to roughly 186 × (14/30.8) ≈ **85 tok/s**. Same model, same GPU, 2.2× slower purely from cache traffic.
6. **Sanity check against reality.** Published single-stream numbers for 7–8B models on an H100 with a good engine land in the 100–180 tok/s range. The estimate is in the right place, which is what a first-principles estimate is for.

Two follow-ups worth pre-empting. **"How would you double it?"** Quantize the weights to fp8 or int8 — 7 GB instead of 14, so ~2× on the weight-read term, which is the dominant term at short context. That is the single highest-leverage change to single-stream latency, and it is why every latency-sensitive deployment is quantized. **"And if I need 10× ?"** You can't get it from one stream on one card; you change the problem — speculative decoding (verify k drafted tokens per weight-read, 2–3× realistic), or accept that per-stream latency is bandwidth-bound and optimize throughput per dollar via batching instead.

**⚠ Trap:** conflating tokens/second per stream with tokens/second per GPU. At batch 64 the *per-stream* rate might fall to 60 tok/s while the *aggregate* rises to 3,800 tok/s. Those are different SLOs owned by different stakeholders: the user experiences inter-token latency, finance experiences aggregate throughput per dollar. Every serving design question is a negotiation between those two, and an answer that doesn't name both is incomplete.

### Backprop is a chain of Jacobians. Explain why we never actually build one.

The mental model: **autograd computes vector-Jacobian products, never Jacobians.** Given a composite function L = f_n ∘ … ∘ f_1 (x), the chain rule says the gradient is a product of Jacobian matrices J_n J_{n−1} … J_1. Written that way it looks like you need those matrices. You don't, because you only ever need the product against a *vector* — and reverse-mode differentiation evaluates the product right-to-left starting from the scalar loss, so at every step you're computing vᵀJ, which is a vector, not a matrix.

The size argument makes it visceral. Consider the LM head: input `[T, 4096]`, output logits `[T, 128256]`. Its full Jacobian with respect to the input has 4096 × 128256 entries per token — 5.25 × 10⁸ per token, 4 bytes each, **2.1 GB per token**. For a 2,048-token sequence you'd need 4.3 TB to represent one layer's derivative. Instead, the vector-Jacobian product for y = xW is just `grad_x = grad_y @ W.T` — one matmul, no materialization, and the "Jacobian" only ever exists as the algorithm that applies it.

That's the abstraction PyTorch encodes: every op registers a `backward` that consumes an upstream gradient and returns downstream gradients. When you write `torch.autograd.Function`, the method you implement is exactly a VJP.

```python
class ScaledDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, scale):
        ctx.save_for_backward(q, k); ctx.scale = scale
        return (q @ k.transpose(-1, -2)) * scale

    @staticmethod
    def backward(ctx, g):                 # g is the upstream vector, not a Jacobian
        q, k = ctx.saved_tensors; s = ctx.scale
        return (g @ k) * s, (g.transpose(-1, -2) @ q) * s, None
```

The direction matters and is worth naming. **Reverse mode** (backprop) costs one pass per *output* and is cheap when outputs are few — a scalar loss — and inputs are many, which is exactly the training setting. **Forward mode** (JVPs) costs one pass per *input* and is cheap in the opposite regime. `torch.func.jvp` and `jacrev`/`jacfwd` expose both; you reach for forward mode when differentiating with respect to a handful of scalars, e.g. hyperparameter sensitivity.

**⚠ Trap:** using `torch.autograd.functional.jacobian` on anything but a toy. It works by running the backward pass once per output element. On a `[T, V]` output that's 262 million backward passes. People do this in notebooks, watch it hang, and conclude "autograd is slow." Autograd is not slow; asking for a Jacobian is.

### Quick matrix-calculus check. Y = XW. Give me dL/dW and dL/dX, with shapes.

This is a two-minute filter question, and the thing being tested is whether you check shapes reflexively rather than recalling a formula.

Setup: X is `[B, T, d_in]`, W is `[d_in, d_out]`, Y = XW is `[B, T, d_out]`. Let G = ∂L/∂Y, which has the same shape as Y: `[B, T, d_out]`.

**∂L/∂X = G Wᵀ.** Shape check: `[B,T,d_out] @ [d_out,d_in]` → `[B,T,d_in]`. Matches X. ✓

**∂L/∂W = Xᵀ G, summed over the batch and time axes.** Shape check: you need `[d_in, d_out]`, and you have X as `[B,T,d_in]` and G as `[B,T,d_out]`, so contract over both B and T: `torch.einsum('bti,bto->io', X, G)`. Matches W. ✓

The rule that makes this derivable rather than memorized, and the one I'd give a junior: **the gradient of a scalar loss with respect to any tensor has exactly that tensor's shape.** So there is only one way to arrange the available operands to produce the required shape, and if there's only one arrangement, you don't need to remember it — you derive it in five seconds by shape-matching. The only genuine choices are which axes to contract and which to transpose, and the shape constraint fixes both.

Two follow-ups that show up. **"Why is the W gradient summed over the batch?"** Because W is shared across all B·T token positions — every position contributed to the loss through the same weights, and the chain rule sums contributions over all paths. This is also why the weight-gradient matmul reduces over the biggest axis and is therefore the memory-traffic-heavy half of the backward. **"What if there's a bias?"** ∂L/∂b = G summed over B and T, giving `[d_out]` — same logic, same reduction.

**🏋 Drill:** 6 minutes, blank page, no autocomplete. Write the forward and backward for a two-layer MLP with GELU by hand — `h = gelu(xW1 + b1); y = hW2 + b2` — and produce all six gradients with their shapes. Then verify against autograd with `torch.autograd.gradcheck` on float64 inputs of shape `[4, 8]`. Pass criterion: gradcheck returns True on your first submission. Most people get W2 and b2 right and fumble the GELU derivative chain into W1 — that's the part worth rehearsing.
