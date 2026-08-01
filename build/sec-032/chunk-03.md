### Derive Megatron tensor parallelism for me. Count the collectives per transformer layer, forward and backward.

Tensor parallelism splits individual weight matrices across devices, and the entire design rests on one algebraic observation: a matmul can be cut two ways, and if you cut two consecutive matmuls in the two different ways, the intermediate communication cancels. That cancellation is why a transformer block needs one collective per sub-block instead of one per matmul, and it is the whole reason Megatron's scheme is usable.

**Column parallel.** `Y = XA` with `A = [A₁ | A₂]` split by columns. Rank `i` computes `XAᵢ` and holds a column-slice of `Y`. The input `X` is replicated; no communication in forward.

**Row parallel.** `Z = YB` with `B = [B₁ ; B₂]` split by rows, and `Y = [Y₁ | Y₂]` split by columns. Rank `i` computes `YᵢBᵢ`, which is a *partial sum* of the true `Z`. An **all-reduce** produces `Z`.

Now the MLP: `h → 4h → GeLU → h`. Make the first matrix column-parallel, the second row-parallel. Rank `i` holds columns of `W₁` and rows of `W₂`. The GeLU in between is elementwise, so it applies fine to a column-slice. No communication is needed until the row-parallel output — **one all-reduce per MLP**. If you had chosen row-then-column instead, you would need an all-reduce before the GeLU too, because GeLU is nonlinear and does not commute with summation. That is the trick, stated precisely: *the nonlinearity must sit where the tensor is sharded, not where it is partial.*

Attention is the same shape. QKV projection is column-parallel, split **by attention head**, so each rank owns a complete subset of heads. The attention computation — scores, softmax, weighted sum — is entirely within a head, so it runs locally with no communication. The output projection is row-parallel. **One all-reduce per attention block.**

**Two all-reduces per layer in forward.** In backward, the conjugate operators fire: the row-parallel layer's backward needs no collective (gradients w.r.t. its sharded input are already sharded), but the column-parallel layer's backward must all-reduce the gradient w.r.t. its replicated input, because each rank computed a partial contribution. So **two more all-reduces in backward: four per layer per step.**

In Megatron's notation these are the `f` and `g` operators: `f` is identity in forward and all-reduce in backward; `g` is all-reduce in forward and identity in backward. Being able to say that sentence is the shibboleth for this topic.

**📄 Paper:** Shoeybi et al. (2019), *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*. It replaced naive layer-wise model parallelism (which serializes devices) with intra-layer sharding that keeps every device busy on every layer, at the cost of two collectives per layer.

**🗣 Say this in the room:** "Column-parallel then row-parallel, so the GeLU and the softmax both land on sharded tensors and the only collective is at the row-parallel output. Two all-reduces forward, two backward, per layer — `f` is identity/all-reduce, `g` is all-reduce/identity."

### Why must tensor parallelism stay inside the NVLink domain? Show me the arithmetic.

Because TP's communication is per-layer, on the critical path, and proportional to the *activation* tensor, which means its volume is enormous and it cannot be hidden. This is not a preference — it is a hard constraint that shapes every cluster layout, and you should be able to prove it in 90 seconds.

Take a 70B-class model: `L = 80` layers, `h = 8192`, microbatch `b = 1`, sequence `s = 4096`, bf16, `TP = 8`.

The tensor being all-reduced is `[s, b, h]` = 4096 × 1 × 8192 × 2 bytes = **67 MB**.

Ring all-reduce moves `2(N−1)/N × S` per rank = 2 × (7/8) × 67 MB = **117 MB** per collective.

Four collectives per layer × 80 layers = 320 collectives, so per rank per step:

```
320 × 117 MB = 37.5 GB
```

Now divide by link bandwidth:

- **NVLink 4 inside an H100 node**, ~400 GB/s achievable all-reduce bus bandwidth: 37.5 / 400 = **94 ms**
- **InfiniBand NDR 400G across nodes**, ~45 GB/s: 37.5 / 45 = **833 ms**

Compare to compute. Model FLOPs for the microbatch = 6 × 70e9 × 4096 = 1.72 PFLOP, split over 8 TP ranks = 215 TFLOP each. At an achieved 400 TFLOPS that is **537 ms**.

So on NVLink, TP communication is 94/537 = **17.5% of compute** — noticeable, partially overlappable, acceptable. Across InfiniBand it is 833/537 = **155% of compute**: you would spend more time communicating than computing, and because these all-reduces sit between dependent matmuls there is almost nothing to overlap them with. Your MFU would fall below 20%.

**📐 Numbers you must know:** NVLink 4 (H100) is 900 GB/s bidirectional per GPU, ~450 GB/s each way, and a well-tuned intra-node all-reduce achieves 350–450 GB/s of bus bandwidth. InfiniBand NDR is 400 Gb/s = 50 GB/s per port, achieving ~45 GB/s. **The intra-node link is roughly 9–10× the inter-node link.** That single ratio determines the entire parallelism layout. NVLink 5 on Blackwell roughly doubles the intra-node number and GB200 NVL72 extends the NVLink domain to 72 GPUs, which is the architectural point of that product — it makes `TP > 8` viable for the first time. **📅 Volatile:** verify per-generation link rates before quoting.

**⚠ Trap:** "just use TP=16 across two nodes to fit a bigger model." Never. If you need more sharding than fits in the NVLink domain, add FSDP or pipeline parallelism, both of which communicate far less per FLOP. The rule: **TP degree ≤ the NVLink domain size**, full stop.

### What constrains the tensor-parallel degree besides the interconnect, and how do you handle the embedding and the loss?

Three structural constraints, and the embedding/loss handling is where an otherwise-correct TP implementation blows up memory.

**Constraint 1 — head divisibility.** Attention is sharded by head, so `n_heads % TP == 0`. With GQA you also need `n_kv_heads % TP == 0`, or you must replicate the KV heads across TP ranks. A model with 8 KV heads simply cannot use TP=16 without replication, and this is a real design constraint on modern GQA models: Llama-3-70B has 64 query heads and 8 KV heads, so TP=8 is the natural maximum before you start duplicating KV projections.

**Constraint 2 — hidden-dim divisibility and tensor-core alignment.** The column-parallel shard is `4h/TP` wide; you want that to remain a multiple of 128 (or at least 64) for tensor-core efficiency. Sharding `h = 5120` by TP=8 gives per-rank MLP width 2560, fine. Sharding an oddly-sized model can leave you with shapes that fall off the fast path and cost 20% throughput for no visible reason.

**Constraint 3 — the vocabulary.** This is the interesting one. A 128k-vocab, `h=8192` embedding is 128000 × 8192 = **1.05B parameters**, larger than any single transformer layer. Left unsharded and replicated across TP ranks it is 2.1 GB per rank in bf16 and 16.8 GB of full optimizer state.

Megatron shards it **vocab-parallel**: rank `i` owns rows `[i·V/t, (i+1)·V/t)` of the embedding table. For the input embedding, each rank looks up the tokens it owns and zeroes the rest, then an all-reduce sums the partial results. For the output projection, each rank produces logits only for its vocabulary slice.

And that is where the memory bomb is. The logits tensor is `[s, b, V]` — for `s=4096, b=1, V=128000` in fp32 that is 4096 × 128000 × 4 = **2.1 GB**, and the softmax + cross-entropy needs several such temporaries. Materializing full logits per microbatch, in fp32, is routinely the single largest allocation in a training step and is a top-three cause of OOM in long-context runs.

The fix is **parallel cross-entropy**: never gather the logits. Each rank has its `V/t` slice. Compute the local max, all-reduce a `[s, b]` max, subtract, exponentiate, compute the local sum-of-exps, all-reduce a `[s, b]` sum, and pick out the target logit from whichever rank owns that token id (all-reduce a `[s, b]` gather). Every collective is on a tensor of shape `[s, b]` — 4096 floats — instead of `[s, b, V]`. Megatron implements this as `vocab_parallel_cross_entropy`; PyTorch users typically get it from a fused chunked cross-entropy kernel.

**💰 Math:** on a 32k-context run with `b=4`, unsharded fp32 logits are 32768 × 4 × 128000 × 4 B = **67 GB**. That does not fit anywhere. Vocab-parallel with TP=8 gets you to 8.4 GB per rank, and chunked/fused cross-entropy (processing 1024 sequence positions at a time) gets you to ~260 MB. The difference between a run that trains and a run that OOMs at step 1 is one loss-function implementation.

### Explain sequence parallelism. It doesn't reduce communication volume — so what is it for?

Correct, and that framing is exactly right: **sequence parallelism is a pure memory optimization with identical communication volume.** It exists because Megatron TP leaves a surprising amount of activation memory replicated.

Look at where TP does *not* shard. Between the attention block and the MLP block sit the LayerNorm, the dropout, and the residual add. These are elementwise or per-token operations with no weight matrix to split, so in vanilla Megatron TP every rank holds the *full* `[s, b, h]` activation in those regions. On TP=8 that means eight identical copies of a tensor that only one of them needed.

Sequence parallelism shards those regions along the **sequence** dimension instead: rank `i` holds `[s/t, b, h]`. Since LayerNorm normalizes over the hidden dimension and dropout is elementwise, both are perfectly correct on a sequence-slice. No math changes.

The elegant part is the transition. Entering a TP region you need the full sequence replicated; leaving it you produced a partial sum. So:

- **All-reduce** (the old `g`) is decomposed into **reduce-scatter** (produce a sequence-sharded, fully-reduced tensor) entering the SP region.
- **Identity** (the old `f`) becomes **all-gather** (reassemble the full sequence) entering the TP region.

And since all-reduce *is* reduce-scatter followed by all-gather, the total bytes on the wire are unchanged. You get the memory for free.

The memory win, following Korthikanti et al. (2022): with TP alone, per-layer activations look like `sbh·(10 + 24/t + 5as/(ht))` — the `10` is the un-sharded LayerNorm/dropout/residual region that TP cannot touch. With TP+SP it becomes `sbh/t · (34 + 5as/h)` — everything divided by `t`. At `t=8`, the constant term goes from `10 + 3 = 13` down to `34/8 = 4.25`, roughly a **3× reduction** in the non-attention activation footprint.

**📄 Paper:** Korthikanti et al. (2022), *Reducing Activation Recomputation in Large Transformer Models*. It contributed both sequence parallelism and selective activation recomputation, replacing the previous practice of blanket full recomputation.

**⚠ Trap:** dropout under SP must use a *different* RNG offset per sequence shard, or you apply a correlated mask across shards and change the regularization. Megatron handles this with a separate "tensor-parallel RNG tracker" that seeds differently per rank. If you implement SP by hand and reuse the global RNG, your model still trains — slightly worse — and you will never find it.

### Explain selective activation recomputation. Why is it strictly better than full checkpointing for most runs?

Because the FLOPs you spend recomputing and the memory you save are wildly uncorrelated across a transformer's operations, so recomputing *everything* is a blunt instrument that pays a full 33% compute tax for savings you could have gotten for 4%.

Sort the layer's stored tensors by bytes-per-FLOP-to-regenerate. At the top of that list, by an enormous margin, sits the **attention score region**: the `[b, a, s, s]` softmax output and dropout mask. Their size is `5·a·s²·b` bytes — quadratic in sequence length — while regenerating them costs only the `QKᵀ` and `PV` matmuls, which are `4·b·a·s²·d` FLOPs, a small fraction of the layer's total because the linear projections dominate at typical `s`. At the bottom of the list sit the linear-layer inputs: linear in `s`, but expensive to regenerate because they *are* the big matmuls.

So: **recompute the attention region, store everything else.** With FlashAttention this is essentially automatic — FlashAttention never materializes the score matrix in the first place and recomputes tiles of it in backward from Q, K, V held in SRAM. So the modern statement is "use FlashAttention and store the rest," and selective recompute is the pre-FlashAttention formulation of the same insight plus a bit more.

The reported numbers from Korthikanti et al.: selective recomputation costs roughly **4% overhead** while full recomputation costs about **36%**, and selective recovers the large majority of the memory. That ratio — 9× cheaper for most of the benefit — is the whole argument.

The modern PyTorch mechanism is `torch.utils.checkpoint` with a **save policy** rather than an all-or-nothing wrapper: `create_selective_checkpoint_contexts` lets you supply a predicate over ATen ops saying which outputs to keep and which to recompute. Typical policy: always save the outputs of `aten::mm` and `aten::_scaled_mm` (expensive to redo), recompute everything cheap and elementwise.

**The decision ladder I actually use, in order:**

1. FlashAttention / SDPA memory-efficient backend. Free 17× on the quadratic term. Non-negotiable.
2. Sequence parallelism if you already run TP. Free 3× on the constant term.
3. Selective recompute of the remaining large-but-cheap tensors. ~4% compute.
4. Full block checkpointing. ~33% compute. Only when 1–3 are not enough.
5. Activation offload to host. Only when you are out of options, because it puts PCIe on the critical path.

**⚠ Trap:** people jump straight to step 4 because it is one flag, then report "activation checkpointing costs us 35% throughput" as if that were a law of nature. It is a law of *blanket* checkpointing. I push back on any PR that enables full recompute without first showing me the SDPA backend in use and the memory profile.

### Long context is blowing up my activation memory even with everything you just described. What is context parallelism and how does Ring Attention work?

At some point the sequence itself is the problem: activations scale linearly in `s` even with FlashAttention, and the KV tensors for a single 1M-token sequence do not fit on one device no matter how you shard the weights. Context parallelism shards the **sequence dimension across devices for the entire layer**, not just the LayerNorm regions — which immediately raises the question that makes it interesting: attention is all-to-all over positions, so how can a rank that holds only its slice of Q, K, V compute attention correctly?

The answer is **Ring Attention** (Liu et al., 2023), and the mental model is a distributed streaming aggregation with online rescaling — structurally identical to FlashAttention's tiling, but with the tiles living on different GPUs and moving over the network instead of moving between HBM and SRAM.

Each of `cp` ranks holds `Q_i, K_i, V_i` for its `s/cp` positions. Then, for `cp` steps:

1. Compute local attention of `Q_i` against the currently-held `K_j, V_j` block, producing a partial output plus a running max and running sum-of-exps.
2. **Simultaneously**, send your `K_j, V_j` block to rank `i+1` and receive the next block from rank `i−1` (a ring send/recv).
3. Merge the new partial into the running output using the online-softmax rescaling identity: rescale the accumulated output by `exp(m_old − m_new)` and add the new contribution.

After `cp` steps every rank has seen every K/V block and holds the exact attention output for its query positions. The point-to-point transfer of step 2 overlaps with the compute of step 1, so if `compute_per_block > transfer_per_block` the communication is entirely hidden — and since local attention compute is `O((s/cp)² · d)` while transfer is `O((s/cp) · d)`, longer sequences make the overlap *easier*. That is the property that makes it work at 1M context.

**The load-balance trap, which is the thing interviewers actually probe.** With causal masking, query position `p` attends only to positions `≤ p`. If you assign contiguous chunks, rank 0 holds the earliest positions and does almost no work while rank `cp−1` attends to the entire prefix. Work per rank is proportional to rank index, so the slowest rank does ~2× the average and your CP efficiency caps around 50%. The fix is **zigzag (striped) assignment**: give each rank two chunks, one from the front and one from the back — rank `i` gets chunk `i` and chunk `2·cp − 1 − i` — so every rank's total work is equal. Any CP implementation that does not do this is leaving half its throughput on the floor.

**📄 Paper:** Liu et al. (2023), *Ring Attention with Blockwise Transformers for Near-Infinite Context*. It replaced "shard the model, replicate the sequence" with "shard the sequence and stream the KV around a ring," making context length a scalable axis rather than a hard per-device ceiling.

**⚠ Trap:** CP does not reduce parameter or optimizer memory at all — it is orthogonal to FSDP and TP and must be composed with them. And it only pays when the sequence is long: at `s = 4096` the per-block compute is too small to hide the ring transfers and CP is pure overhead. My threshold is roughly `s ≥ 32k` before CP earns its place in the mesh.

### Derive the pipeline bubble for GPipe, then tell me what 1F1B changes.

Pipeline parallelism assigns contiguous *layers* to devices, so device 0 holds layers 0–19, device 1 holds 20–39, and so on. The immediate problem is that this is a sequential dependency: device 1 cannot start until device 0 finishes. Naively, `p` devices give you `1/p` utilization — strictly worse than one device.

The fix is to split the batch into `m` **microbatches** and stream them through, so device 0 starts microbatch 2 as soon as it finishes microbatch 1. Now the only idle time is the fill at the start and the drain at the end.

**GPipe's bubble.** Let one microbatch's forward on one stage take `t_f` and backward take `t_b`. GPipe runs all `m` forwards, then all `m` backwards. The pipeline fill costs `(p−1)·t_f` and the drain costs `(p−1)·t_b`. Useful work per device is `m·(t_f + t_b)`. So:

```
bubble_time / useful_time  = (p−1)(t_f + t_b) / (m(t_f + t_b)) = (p−1)/m

bubble_fraction_of_wall_clock = (p−1) / (m + p−1)
```

Both forms appear in the literature and you should quote whichever the interviewer's question implies. With `p = 16` stages and `m = 32` microbatches: bubble = 15/32 = **47% overhead**, or 15/47 = **32% of wall clock idle**. Push to `m = 128`: 15/128 = **12%**. The lesson: **PP efficiency is governed by `m/p`, and you want `m ≥ 4p`.**

**What 1F1B changes: memory, not the bubble.** The 1F1B schedule (from PipeDream, Narayanan et al. 2019, adopted in Megatron) runs a warmup of `p−1` forwards, then strictly alternates one forward and one backward, then drains. The bubble fraction is *identical* to GPipe. What changes is that under GPipe, every stage must hold the activations of all `m` in-flight microbatches — memory `∝ m`. Under 1F1B, a microbatch's backward runs as soon as possible, so stage `i` holds at most `p − i` microbatches of activations, and the **first stage holds `p`, the last holds 1** — memory `∝ p`, independent of `m`.

That decoupling is what makes large `m` affordable, which is what makes the bubble small. So the correct causal story is: 1F1B does not reduce the bubble directly; it removes the memory constraint that was forcing `m` to be small, which lets you reduce the bubble.

**⚠ Trap:** "1F1B has a smaller bubble than GPipe." No — same bubble, different memory. Saying it wrong is a clean tell. The related trap is forgetting that stage 0's memory is `p×` a single microbatch's activations, which is why a PP job OOMs on rank 0 specifically and nowhere else.

**📐 Numbers you must know:** bubble = `(p−1)/m`. Target `m ≥ 4p` for ≤25% overhead, `m ≥ 8p` for ≤12.5%. And `m` is bounded above by `global_batch_tokens / (dp_degree × microbatch_tokens)` — so PP degree and DP degree fight each other for the same budget.

### Interleaved and zero-bubble pipeline schedules — what do they buy and what do they cost?

**Interleaved 1F1B** (Narayanan et al., 2021) attacks the `(p−1)/m` bubble by making each physical device host `v` *non-contiguous* model chunks instead of one contiguous stage. With `p = 4` devices and `v = 2`, device 0 holds layers 0–9 and layers 40–49, device 1 holds 10–19 and 50–59, and so on. The pipeline now has `p·v` virtual stages, so each stage's work is `1/v` as long, so the fill and drain — measured in units of a *virtual* stage — cost the same number of stages but each stage is `v×` cheaper:

```
bubble = (p−1) / (v · m)
```

With `p=16, m=32, v=2` you go from 47% to **23%** overhead. Real gain.

The cost is communication. Each microbatch now crosses a device boundary `p·v` times instead of `p` times, so **point-to-point volume increases by `v×`**. Since PP's P2P traffic is small (one `[s_micro, b, h]` activation tensor per boundary), this is usually affordable, but it stops being affordable when your pipeline spans a slow link and `v` is large. And the schedule is genuinely complicated — the warmup/steady/drain structure with `v` chunks is where PP implementations have bugs.

**Zero-bubble** (Qi et al., 2024) attacks it from a different angle: it observes that the backward pass is really **two** independent computations — `B`, the gradient with respect to the *input* (which the upstream stage is blocked on), and `W`, the gradient with respect to the *weights* (which nothing downstream needs until the optimizer step). Standard schedules fuse them. If you split them, `W` becomes a free-floating unit of work with no dependency, and the scheduler can slot `W` chunks into what would otherwise be bubble.

The ZB-H1 variant reaches roughly half the 1F1B bubble at the same memory. ZB-H2 reaches essentially zero bubble but needs more in-flight activation memory and requires handling one nasty detail: to fully fill the bubble you must relax the synchronized optimizer step across stages, which the paper handles with a post-update validation that rolls back if a gradient clip or an inf/NaN check would have changed the decision.

**My honest read for an interview:** interleaved 1F1B is the production default in Megatron-style stacks and you should know it cold. Zero-bubble is real, implemented in several stacks, and the right answer to "how would you push PP efficiency further" — but it is newer, the implementations vary, and I would present it as "the direction the field is going" rather than "what I would deploy on Monday." **📅 Volatile:** which schedules are production-supported in Megatron-LM, DeepSpeed and `torch.distributed.pipelining` changes release to release; check before claiming availability.

**🗣 Say this in the room:** "Interleaved divides the bubble by the number of virtual chunks at the cost of `v×` the P2P traffic. Zero-bubble splits backward into input-gradient and weight-gradient, and since weight-gradient has no downstream dependency it can be scheduled into the bubble. Both are attacking `(p−1)/m` from different sides — one shrinks the numerator's effective unit, the other fills the gap with real work."

### Pipeline parallelism looks clean on the whiteboard. What actually goes wrong when you deploy it?

Four things, and none of them is the bubble.

**Stage imbalance.** The pipeline runs at the speed of its slowest stage, so a 5% imbalance costs 5% of the *entire* job. Transformers are not uniform: stage 0 carries the token embedding, and the last stage carries the LM head plus the loss. For a 128k vocab at `h=8192`, the LM head is 1.05B parameters — larger than a decoder layer at that width — and the cross-entropy over `[s, b, 128000]` logits is both compute-heavy and the biggest allocation in the model. So the naive "L/p layers per stage" split makes the last stage 30–50% slower than the middle stages. The fix is to give the first and last stages **fewer transformer layers**, and to always profile per-stage step time rather than trusting the layer count. A good stack exposes an explicit layer-to-stage map for exactly this reason.

**Memory imbalance.** As covered: under 1F1B stage 0 holds `p` microbatches of activations, the last stage holds 1. So stage 0 OOMs first, and the naive response — reduce microbatch size — increases `m` and is actually fine, but the naive response people *actually* take is to reduce `m`, which increases the bubble. Know which lever you are pulling.

**The `m` budget conflict.** `m = global_batch_tokens / (dp × micro_tokens)`. You want `m ≥ 4p` for a small bubble, but you also want `dp` large for throughput and `global_batch` bounded because past the critical batch size extra tokens per step buy you almost nothing in loss. These three constraints fight. On a 16k-GPU run with PP=16 you need `m ≥ 64`, which with `dp = 128` and a 4M-token global batch means microbatches of 4M/(128×64) = 488 tokens — very small, poor tensor-core utilization. This tension is the real reason PP degree is kept modest.

**Composition with the optimizer.** The loss exists only on the last stage, so every metric you log — loss, perplexity, grad norm — needs a broadcast or a reduce to be visible on rank 0. Teams routinely ship a training run where the logged loss is from one pipeline stage's rank and looks fine while another stage is producing NaN. And gradient clipping must be global across *all* stages, which is another instance of the collective-over-shards problem.

**🔍 Failure taxonomy — PP throughput is below the bubble prediction:**

1. Measure per-stage forward and backward wall time. If max/mean > 1.1, it is imbalance — rebalance the layer map before anything else.
2. If balanced, check P2P: is `NCCL_DEBUG=INFO` showing the send/recv going over the right transport? A pipeline boundary that accidentally falls back to TCP instead of IB will silently cost 10×.
3. If P2P is healthy, check `m/p`. Below 4 and the bubble is your answer.
4. If `m/p` is fine, look for a sync point — a `.item()`, a `print` of a GPU tensor, or a host-side barrier inside the loop will serialize the pipeline completely.

### Explain expert parallelism for MoE training, and why all-to-all is the bottleneck.

A Mixture-of-Experts layer replaces one dense MLP with `E` MLPs plus a router that sends each token to its top-`k` experts. The parameter count goes up by `E` while the FLOPs per token go up only by `k` — that is the entire value proposition, and it is why frontier models are MoE. But it creates a data-movement pattern that no other parallelism scheme has: **the destination of a tensor is data-dependent and changes every step.**

Expert parallelism places different experts on different devices. So a training step for one MoE layer is:

1. **Router** computes per-token expert assignments (a `[tokens, E]` gate, top-`k` selection).
2. **All-to-all dispatch**: every device sends each token's hidden vector to the device(s) holding its chosen experts.
3. **Expert compute**: each device runs its local experts on whatever tokens arrived.
4. **All-to-all combine**: results are sent back to the originating device and weighted by the gate values.

And again in backward, mirrored. So **four all-to-alls per MoE layer per step.**

All-to-all is the worst collective in the catalogue for three independent reasons. First, there is no ring or tree trick — it is inherently `N²` pairwise flows, so its cost is bounded by bisection bandwidth rather than by per-link bandwidth, and it stresses the fabric's worst dimension. Second, it is a **hard barrier**: expert compute cannot begin until dispatch completes, and combine cannot begin until expert compute completes, so there is very little natural overlap. Third, the per-pair messages are small and irregular — `tokens_to_expert_j × h × 2` bytes — so you are partly latency-bound, and the sizes are *data-dependent*, meaning every rank must first exchange counts before it can even post the receives.

**💰 Math.** `h = 4096`, top-2, bf16, 8192 tokens per rank per microbatch, 32 MoE layers.

- Dispatch out per rank per layer = 8192 tokens × 2 experts × 4096 × 2 B = **134 MB**
- Combine back = **134 MB**
- Forward total per layer = 268 MB; backward ≈ 268 MB → **536 MB per MoE layer per step**
- × 32 layers = **17.2 GB per rank per step** of all-to-all traffic.

If EP is confined inside a node (NVLink, ~300 GB/s effective for all-to-all): 17.2/300 = **57 ms**. If EP spans nodes over NDR with ~25 GB/s achieved all-to-all: 17.2/25 = **688 ms**. On a 1.5 s step, that is 4% versus 46%. This is why `EP ≤ GPUs-per-node` is the strong default and why NVL72's 72-GPU NVLink domain is genuinely strategically important for MoE training.

**The load-balance problem.** Routing is learned, and left alone it collapses: a few experts get most tokens, the rest get none. Two consequences — the popular expert's device becomes a straggler that everyone waits on, and the unused experts never train. Mitigations: a **capacity factor** `C` that caps each expert's buffer at `C × tokens/E` and **drops** overflow tokens (they skip the MoE and pass through the residual), plus an **auxiliary load-balancing loss** that penalizes the dot product of the fraction-of-tokens-routed and the mean-gate-probability per expert (Switch Transformer, Fedus et al. 2021; the formulation originates with Shazeer et al. 2017 and GShard, Lepikhin et al. 2020).

**⚠ Trap:** token dropping is invisible in your loss curve and visible only in eval. At `C = 1.0` with imperfect balance you can drop 5–15% of tokens; those tokens get a degraded representation and the model quietly learns less per step. Always log the drop rate as a first-class metric. The newer alternative is bias-based load balancing that adjusts per-expert routing biases without an auxiliary loss term, avoiding the gradient interference that the aux loss introduces — worth knowing as the direction of travel. **📅 Volatile.**

### I have 1,024 H100s in 128 nodes of 8. Design the parallelism layout for a 400B-parameter dense model and justify every axis.

The design procedure is always the same and I would say it out loud in this order: **rank the parallelism dimensions by communication volume per unit of compute, then map them onto the interconnect hierarchy from fastest link to slowest.** Everything else is arithmetic.

**The hierarchy, fastest to slowest:** NVLink within a node (~400 GB/s effective) → InfiniBand within a rack/leaf switch (~45 GB/s, one hop) → across spine switches (~45 GB/s but more hops and more congestion).

**The traffic ranking, heaviest to lightest per FLOP:**

1. **TP** — four all-reduces of `[s,b,h]` per layer. Enormous, on the critical path, unoverlappable. → NVLink only.
2. **EP** (if MoE) — four all-to-alls per MoE layer, hard barriers. → NVLink, or one IB hop at most.
3. **CP** — ring P2P of K/V per layer, but overlappable with attention compute. → NVLink preferred, one IB hop tolerable at long sequence.
4. **FSDP/DP-shard** — all-gather + reduce-scatter once per unit per step, highly overlappable with backward compute. → IB is fine.
5. **PP** — one activation tensor per microbatch per stage boundary. Tiny. → slowest link available, span racks freely.

**The layout.** 400B dense, bf16 + Adam = 6.4 TB of model state; 1,024 H100s give 81.9 TB of HBM, so state is ~8% of memory and activations dominate. Start:

- **TP = 8.** One full node. This shards the 400B weights 8× for compute, and it is free because it stays on NVLink. Requires `n_heads % 8 == 0` — check the config.
- **PP = 16.** 400B / 16 = 25B params per stage. Spans nodes; P2P traffic per boundary is one `[s_micro, b, h]` tensor ≈ tens of MB, trivial over IB. PP=16 needs `m ≥ 64` microbatches for a ≤23% bubble, so use interleaved 1F1B with `v = 2` to get to ~12%.
- **DP = 1024 / (8 × 16) = 8**, run as FSDP `HYBRID_SHARD`-style sharding across the 8 data-parallel replicas.
- **CP = 1** at 8k context. If the run has a long-context phase at 128k, take CP = 8 out of the DP budget for that phase only — which means the mesh is `(pp=16, dp=1, cp=8, tp=8)` in that phase and the global batch composition changes.

Mesh declaration, ordering the axes so TP varies fastest and therefore lands inside a node:

```python
mesh = init_device_mesh("cuda", (16, 8, 8), mesh_dim_names=("pp", "dp", "tp"))
```

**The check I would actually run before launching.** Per-GPU model state = 400e9 × 16 / (8 TP × 16 PP × 8 DP) — but note TP and PP shard parameters while FSDP shards what remains, so per-GPU state ≈ 400e9/(8×16) × 16 bytes / 8 = 6.25 GB. Comfortable. Then compute the expected step time from FLOPs: 6 × 400e9 × tokens_per_step / (1024 × achieved_TFLOPS) and compare against the measured first-100-step time. If measured is more than 1.3× predicted, something in the mesh is on the wrong link and I go to `nvidia-smi topo -m` and NCCL logs before touching anything else.

**🗣 Say this in the room:** "I rank the axes by bytes-per-FLOP and map them onto the link hierarchy: TP inside NVLink, expert parallelism next, then context, then FSDP over InfiniBand, then pipeline across racks because it moves the least. Then I sanity-check with a FLOPs-based step-time prediction — if the measured step is 30% off the prediction, an axis landed on the wrong fabric."
