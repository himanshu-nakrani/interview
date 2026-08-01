### DeepSeek-V3 is 671B total and 37B active. People say it "serves like a 37B model." Tell me precisely what that means and everything it does not mean.

It means exactly one thing: **the arithmetic per token is that of a 37B dense model.** Forward FLOPs per token ≈ `2 × 37e9 = 74 GFLOP` instead of `2 × 671e9 = 1.34 TFLOP`. That is an 18× reduction in compute, and it is real. In the compute-bound regime — prefill, and decode at very large batch — you genuinely get 37B-class speed with far better than 37B-class quality. That is the whole point of the architecture and it is a legitimate achievement.

Here is everything it does not mean, in the order the bill arrives.

**It does not mean 37B of memory.** Every one of the 671B parameters must be resident in HBM, because you do not know which experts the next token will need. At fp8 that is 671 GB. An H100 has 80 GB. You need a minimum of nine cards for weights alone, and realistically **16 GPUs** — two full nodes — once you leave room for KV cache, activations, all-to-all staging buffers and fragmentation. A dense 37B at fp8 is 37 GB and fits on one card. **You have an 8-to-16× hardware multiplier that active-parameter count is completely silent about.**

**It does not mean 37B of memory bandwidth per decode step**, and this is the subtler and more important point. At decode, each *token* touches 8 of 256 routed experts — but a *batch* touches the union. With 256 experts, top-8, and a batch of 512 decode tokens, you have 4,096 expert-token assignments over 256 experts: every expert is hit, several times. So the server reads **all 671 GB** off HBM every decode step, not 37 GB. Bandwidth-wise, at meaningful batch sizes, this model behaves like a **671B** model, not a 37B one.

**It does not mean 37B of communication.** A dense model's decode step involves tensor-parallel all-reduces. An expert-parallel MoE adds two all-to-all collectives per MoE layer — for DeepSeek-V3 that is 58 MoE layers × 2 = **116 collectives per token**. A dense model has zero of those.

**It does not mean 1/18th the cost.** It means roughly 1/18th the FLOPs, 16× the GPUs, and a communication pattern that punishes small batches.

**🗣 Say this in the room:** "Active parameters predict FLOPs. Total parameters predict your GPU count and your HBM bandwidth bill at realistic batch sizes. Those are different quantities, and for MoE they differ by 18×. When someone estimates serving cost from active parameters, they've priced the cheap resource and ignored the expensive one."

**⚠ Trap:** the interviewer says "so an MoE is cheaper to serve." The correct response is a question, not agreement: *"Cheaper than what, at what batch size, on what interconnect?"* An MoE is cheaper than the **dense model of equal quality** at **high batch**. It is dramatically more expensive than the **dense model of equal active size** at **any** batch, and worse than both at batch 1.

### Explain expert parallelism. Walk me through what actually crosses the wire during one decode step.

Expert parallelism is a sharding strategy in which each GPU owns a *disjoint subset of the experts*, in full, rather than a slice of every weight matrix. With 256 experts across 16 GPUs, GPU 0 holds experts 0–15 complete, GPU 1 holds 16–31, and so on. Contrast with tensor parallelism, where every GPU holds a column slice of *every* matrix. The consequence is that a token's hidden state must physically travel to the GPU that owns the expert it selected, and the result must come back.

Per MoE layer, per forward step, the sequence is:

1. **Attention and the router run replicated or tensor-parallel**, so every GPU ends up holding the tokens in its own data-parallel shard, plus their routing decisions.
2. **Dispatch all-to-all.** Every GPU packs each of its tokens into `k` copies addressed to the `k` GPUs owning its chosen experts, then a single `all_to_all_v` (variable-sized) collective delivers them. Payload per token: `k · d_model · bytes_per_elem`. For DeepSeek-V3 in fp8: `8 · 7168 · 1 B = 57 KB` per token per layer.
3. **Local expert GEMMs.** Each GPU runs a grouped GEMM over the ragged set of tokens it received, grouped by local expert id.
4. **Combine all-to-all.** Results travel back to their originating GPU, which weights them by the gates and sums. Roughly the same payload as dispatch, though implementations that pre-reduce on the sender can shrink it.
5. Add to the residual stream, move to the next layer.

Two collectives per MoE layer. In a 61-layer model with 58 MoE layers, that is 116 all-to-alls per forward pass.

**📐 Numbers you must know — the interconnect ratio that decides your topology.** On H100, NVLink gives ~900 GB/s bidirectional per GPU (~450 GB/s each way); a single InfiniBand NDR port gives 400 Gb/s = 50 GB/s each way. That is roughly a **9× cliff** the moment your expert-parallel group crosses a node boundary. Now compute the arithmetic intensity of an EP step: per token per MoE layer, expert FLOPs = `2 · k · 3 · d_model · d_ff_expert = 2 · 8 · 3 · 7168 · 2048 ≈ 705 MFLOP`; network bytes ≈ `2 · 57 KB = 114 KB`. Ratio ≈ **6,200 FLOP per network byte**. An H100 at ~990 TFLOP/s fp8 needs 990e12/450e9 ≈ **2,200 FLOP/byte** to stay compute-bound on NVLink — 6,200 comfortably clears it. Over InfiniBand it needs 990e12/50e9 ≈ **19,800 FLOP/byte** — 6,200 misses by roughly 3×. **Conclusion: expert parallelism belongs inside an NVLink domain; cross-node EP is communication-bound by about 3× and you must overlap or shrink it.** (Derate all of this by real achieved efficiency; the ratio is what survives.)

**⚠ Trap:** at decode, all-to-all is usually **latency**-bound, not bandwidth-bound, and people size it with bandwidth math. A batch of 32 decode tokens sends messages of a few kilobytes — far below the size where bandwidth matters. What you pay is fixed collective latency, on the order of 10–30 µs intra-node and worse across nodes. 116 collectives × 20 µs ≈ **2.3 ms per token of pure synchronization**, against an inter-token-latency target of maybe 25 ms. That is ~9% of your budget spent on nothing but barriers, and it does not shrink when you add GPUs — it grows. This is why production MoE stacks invest so heavily in overlapping communication with computation (compute layer `L`'s attention while layer `L-1`'s combine is in flight) and in specialized low-latency all-to-all kernels; DeepSeek open-sourced their expert-parallel communication library precisely because the stock collective was not good enough.

### Expert parallelism, tensor parallelism, data parallelism — how do you actually choose for an MoE, and what's your decision rule?

They are not alternatives; they compose, and the real question is which axis absorbs the experts.

**Tensor parallelism on everything (no EP).** Every GPU holds a column-slice of every expert. Simple, no all-to-all, reuses the dense code path. The killer is that expert matrices are already *small* — DeepSeek-V3's per-expert `d_ff` is 2048 — so slicing a `7168 × 2048` matrix eight ways gives you `7168 × 256` GEMMs that cannot fill a tensor core, and you pay an all-reduce per layer on top. Verdict: fine for coarse-grained MoEs (Mixtral's `d_ff` = 14336 slices well), bad for fine-grained ones.

**Expert parallelism.** Each GPU owns whole experts. Big, well-shaped local GEMMs; no weight slicing. Costs two all-to-alls per layer and exposes you to load imbalance, because now imbalance means *GPU* imbalance rather than a slightly ragged local GEMM.

**Data parallelism on the attention/dense path, EP on the expert path.** This is the modern default and it is worth being able to say crisply: attention (with its KV cache) is replicated or tensor-parallel across a DP group, while the MoE layers use a wide EP group spanning the same GPUs. Attention parallelism and expert parallelism have genuinely different optimal degrees — attention wants low TP degree to avoid all-reduce overhead and to keep KV cache local, experts want high EP degree to fit in memory — so decoupling them is the right design, and this decoupling (sometimes called attention-DP with expert-EP) is what the high-throughput MoE stacks converged on.

**Pipeline parallelism.** Layers split across devices. It works and it is memory-efficient, but it introduces bubbles that hurt at decode, where every micro-batch is one token deep. I reach for it last, and mainly to cross a node boundary where all-to-all would be worse.

**My decision rule, in order:**
1. Does the model fit in one node's aggregate HBM (8 × 80 GB = 640 GB, or 8 × 141 GB on H200)? If yes, **keep the entire EP group inside the node.** Do this before optimizing anything else; the 9× NVLink-versus-IB cliff dominates every other consideration.
2. Are experts fine-grained (`d_ff_expert` under ~4096)? Then **EP, not TP, on the experts** — TP would shred them into unusable GEMMs.
3. Is attention's KV cache the memory pressure rather than weights? Then **lower the TP degree on attention and raise DP**, so each replica keeps its KV local and you avoid cross-GPU KV chatter.
4. Must you cross nodes? Then place the EP boundary at the node edge, use pipeline or DP across nodes, and budget explicitly for overlap.

**⚠ Trap:** choosing TP degree without checking GQA head divisibility *and* expert-count divisibility simultaneously. With 8 KV heads you cannot cleanly do TP=16 on attention; with 256 experts and EP=16 you get exactly 16 per GPU, but EP=24 gives you a ragged split where some GPUs hold 11 experts and others 10 — and the straggler is now structural, permanent, and invisible in your config review.

### Why does one overloaded expert stall the whole batch? I want the mechanism, not "it's slower."

Because the all-to-all is a **barrier**, and a barrier converts a *distribution* into a *maximum*.

In a dense model, if one GPU is 20% slower, tensor-parallel all-reduce makes everyone wait 20% — but every GPU is doing identical work, so this only happens from hardware variation. In an expert-parallel MoE the work assignment is *data-dependent and unequal by construction*. GPU 3 might receive 400 tokens this step while GPU 11 receives 90. The combine all-to-all cannot begin until every GPU has finished its local expert GEMMs. So **step time = max over GPUs, not mean**, and every GPU except the busiest is idle for the difference. That idle time is not recoverable and it repeats 58 times per token.

The arithmetic on why this hurts even with a *perfect* router is the part that surprises people. Suppose routing is statistically ideal — assignments land uniformly at random. Batch of 128 decode tokens, top-8, 256 experts across 16 GPUs: 1,024 assignments, expected 64 per GPU, standard deviation ≈ √64 = 8 under a Poisson model. The expected maximum of 16 draws sits roughly 1.7–1.8 σ above the mean, so the busiest GPU has about **78 tokens against a mean of 64 — 22% more work.** You lose ~22% of your expert throughput to imbalance that no amount of load-balancing loss can fix, because it is sampling noise, not router bias. Add real router skew on top (a hot expert running at 3× uniform) and 50%+ losses are ordinary.

**🔍 Failure taxonomy — MoE straggler diagnosis, as a decision procedure:**
1. **Is the imbalance the same GPUs every step?** Yes → *structural*: ragged expert-to-GPU assignment, a genuinely hot expert, or a bad NIC/thermal-throttled card. Fix by re-placing experts or replicating the hot one.
2. **Does it move around every step?** → *stochastic* imbalance. No placement fix exists. Only larger batch (variance shrinks as `1/√n` relative to the mean) or capacity-padding helps.
3. **Does it correlate with request content?** → *semantic* skew: one tenant, one language, or one document type is monopolizing a specialist expert. Fix by mixing traffic across batches rather than batching per-tenant, which is a scheduler change, not a model change.
4. **Is the gap in the collective rather than the GEMM?** Check whether time is inside `all_to_all` or inside the grouped GEMM. If the collective, you have a topology or overlap problem, not a routing problem.

**🗣 Say this in the room:** "The all-to-all is a barrier, so step time is the max over GPUs, not the mean. And even with a perfectly unbiased router you eat the maximum of a random allocation — at 64 tokens per GPU expected, the busiest GPU sees about 78, so roughly 20% of expert throughput evaporates to pure sampling noise. Load-balancing losses fix bias; they do not fix variance. Only bigger batches fix variance."

### How would you measure per-expert utilization in production? What exactly do you instrument?

The router already computes everything you need; the work is getting it out cheaply and joining it to something meaningful. My instrumentation contract, in three tiers.

**Tier 1 — always on, near-zero cost.** In the routing kernel, accumulate a `[n_layers, n_experts]` int32 counter of dispatch counts on-device. Reduce and export it once every `N` steps (I use N such that export is under 1% of step time — typically every 100–1000 decode steps), not per step. Export as a histogram, not a mean. The derived scalars I alert on:
- `cv2 = E · Σ_i (f_i)²` — the squared coefficient of variation of load, equal to 1.0 at perfect balance. Alert above ~1.5.
- `max_i f_i / (1/E)` — the hot-expert ratio. Alert above ~3.
- Per-GPU aggregate load, since that is what actually determines step time.

**Tier 2 — sampled, joined to request metadata.** On a 1-in-1000 sample of requests, emit the full `[n_layers, k]` routing trace as a span attribute alongside tenant id, language, and task type. This is the tier that answers "*why* is expert 47 hot," and it is the difference between an alert and a fix. You already run distributed tracing; the routing trace is just another span attribute, and it compresses well because expert ids are small ints.

**Tier 3 — offline, on a fixed probe corpus.** Run a frozen 10k-token probe set through the model nightly and record the routing distribution. This gives you a *drift* signal that is independent of traffic mix — the single hardest thing to separate in production, because a change in `f_i` can mean the traffic changed or the model changed, and only a fixed corpus disentangles them. This is exactly the golden-dataset discipline you already apply to search relevance regressions.

**The joins that make the data actionable:** expert id × layer × tenant, expert id × language, expert id × step-time contribution. Without the tenant join you will chase a "routing regression" that is actually one enterprise customer onboarding a corpus in Japanese.

**⚠ Trap:** logging routing decisions per token per layer with full fidelity. For a 61-layer model at top-8, that is 488 int16s per token; at 10k tokens/sec that is ~10 MB/s of telemetry, forever, to answer a question you ask twice a month. Sample it. The rule I enforce in review: **on-device counters always, full traces sampled, and nothing per-token in the log pipeline.**

### Tell me about replicating hot experts. How does it work, when is it worth it, and what breaks?

The idea is straightforward once you see expert placement as a *sharding* problem: if expert 47 receives 4× its share of tokens, put a copy of expert 47 on a second GPU and split its traffic between the replicas. Placement stops being "16 experts per GPU, in order" and becomes a bin-packing problem over *expected load*, solved from the utilization telemetry you just built. Production systems do this dynamically — measure load over a window, recompute the placement, and shift expert weights between GPUs during a maintenance window or between batches.

Mechanically you need three things. A **placement table** mapping `(layer, expert_id) → [gpu_ids]`, consulted by the dispatch logic to pick a destination when there is more than one. A **tie-break policy** for choosing among replicas — round-robin is fine and stateless; least-loaded is better and requires a shared counter. And a **rebalancing procedure** that moves expert weights across GPUs without dropping requests, which in practice means holding both placements valid during a drain window.

**When it is worth it:** when your utilization histogram shows persistent, structural skew — the same experts hot across days and across tenants — and when the resulting straggler cost exceeds the memory you must give up. That is the trade: a replica is a *duplicate copy of expert weights in HBM*, and HBM is the resource you were already short of. For DeepSeek-V3, one expert is `3 · 7168 · 2048 ≈ 44 M` parameters ≈ 44 MB at fp8, so replicating the top 16 hottest experts across 58 layers costs `16 · 58 · 44 MB ≈ 41 GB` — half an H100. **💰 Math:** you would only spend that if it bought back more than 41 GB worth of throughput. If replication takes your straggler overhead from 40% to 15%, you gained 25% of expert throughput; on a 16-GPU deployment that is worth ~4 GPUs, which is far more than the half-GPU of HBM you spent. Do it. If it takes you from 22% to 18%, do not.

**What breaks:**
*Determinism.* Two replicas of the same expert should produce bit-identical output, and under bf16 accumulation with different tile schedules they may not. If you have promised a customer reproducible output, replica routing is now a source of nondeterminism you did not have before.
*Cache and graph invalidation.* If you use CUDA graphs, the placement table is baked into the captured graph. Changing placement means recapture, which is not free.
*Feedback loops.* Least-loaded routing plus a load-based rebalancer is a control system with two loops at different timescales. I have seen these oscillate. Damp the rebalancer heavily — hours, not seconds — and never let it react faster than the traffic pattern it is chasing.

### How does batching degrade for an MoE when different requests activate different experts? Give me the arithmetic.

This is the single most important serving fact about MoE and I want the formula memorized.

For a dense model, batching `B` tokens means each weight matrix is read once from HBM and reused `B` times. Arithmetic intensity scales linearly with `B`; that is why continuous batching works.

For an MoE, the batch's tokens **spread across experts**. With `B` tokens, top-`k` of `E` experts, the expected number of distinct experts touched is `E · (1 − (1 − k/E)^B)`, which saturates at `E` very quickly. Once you are near saturation you read *all* expert weights, but each expert's GEMM only serves the tokens routed to it. So the *effective* batch size per expert weight is:

```
B_eff = B · k / E
```

**📐 The number to memorize:** an MoE at decode batch `B` has the weight-reuse characteristics of a dense model at batch `B·k/E`. Mixtral (k=2, E=8): `B/4`. DeepSeek-V3 (k=8, E=256): `B/32`. Qwen3-class (k=8, E=128): `B/16`.

Now put numbers on it. Suppose you want the arithmetic intensity of a dense model at batch 256 — comfortably past the H100 ridge point, where you are finally using the tensor cores. For Mixtral you need `B = 1024` concurrent decode tokens. For DeepSeek-V3 you need `B = 8192`. **Eight thousand concurrent sequences in the decode batch.** That is the number that decides whether an MoE makes sense for your deployment, and it is a *product* question — do you have that much simultaneous traffic? — dressed as an architecture question.

And here is the cruelty that makes this self-defeating on fixed hardware: the batch size you can reach is capped by KV-cache memory, and the MoE's weights already ate that memory. Take Mixtral 8x22B (141 B total, 39 B active) on 4× H100 in fp8: 141 GB of weights leaves 179 GB for KV. At 224 KiB/token of GQA cache that is 800k tokens — about **25 concurrent sequences at 32k context**. But you needed `B ≈ 1024` tokens in flight to reach good expert utilization, and 25 sequences gives you 25. You are running at `B_eff = 25 · 2/8 = 6.25`. The model is being read at an effective batch of six.

**⚠ Trap:** benchmarking an MoE with a synthetic load of thousands of short concurrent prompts, seeing excellent throughput, and shipping it for a long-context product where KV capacity caps concurrency at a few dozen. Your benchmark measured the regime where MoE wins; production runs in the regime where it loses. **The rule I enforce: benchmark MoE serving at your actual context length, not at your actual request count.** Context length is what determines the batch you can reach, and the batch is what determines whether the architecture pays.

### Show me the crossover. At what concurrency does an MoE actually beat the dense model, on real hardware?

Let me do this fully, because the arithmetic *is* the answer. Compare **Llama-3-70B dense** against **Mixtral 8x22B** (141 B total, 39 B active), both fp8, both on 4× H100 (320 GB HBM, ~3.35 TB/s each, ~990 TFLOP/s dense fp8 each). I will use peak numbers; real MFU derates both sides similarly, so the crossover moves little.

**Bytes per decode step (the floor, independent of batch):**
- Dense 70B: 70 GB of weights, sharded 4 ways → 17.5 GB per GPU → `17.5e9 / 3.35e12 = 5.2 ms`.
- Mixtral 8x22B at any batch large enough to touch all 8 experts (which is any batch above ~15 tokens): 141 GB → 35.3 GB per GPU → `35.3e9 / 3.35e12 = 10.5 ms`.

**FLOPs per decode step (linear in batch):**
- Dense: `2 · 70e9 · B` FLOP over `4 · 990e12` FLOP/s → `35.4 ns · B`.
- MoE: `2 · 39e9 · B` over the same → `19.7 ns · B`.

**Where each becomes compute-bound:** dense at `B = 5.2e-3 / 35.4e-9 ≈ 147`; MoE at `B = 10.5e-3 / 19.7e-9 ≈ 533`.

**Throughput:**
| batch B | dense step | dense tok/s | MoE step | MoE tok/s | winner |
|---|---|---|---|---|---|
| 16 | 5.2 ms | 3,080 | 10.5 ms | 1,520 | dense, 2.0× |
| 128 | 5.2 ms | 24,600 | 10.5 ms | 12,200 | dense, 2.0× |
| 256 | 9.1 ms | 28,200 | 10.5 ms | 24,400 | dense, 1.2× |
| 300 | 10.6 ms | 28,300 | 10.5 ms | 28,600 | **crossover** |
| 512 | 18.1 ms | 28,300 | 10.5 ms | 48,800 | MoE, 1.7× |
| 1024 | 36.2 ms | 28,300 | 20.2 ms | 50,700 | MoE, 1.8× |

**The crossover is around 300 concurrent decode tokens.** Below it, the dense 70B is up to **2× faster** despite having 1.8× the active parameters, because decode is bandwidth-bound and the MoE reads twice as many bytes. Above it, the MoE is up to 1.8× faster because it does half the FLOPs.

Now the capacity check, which is the part people skip. Can you *reach* batch 300 on this hardware? KV budget for Mixtral 8x22B is `320 − 141 = 179 GB` at 224 KiB/token = 800k tokens. At 32k average context: 25 sequences. At 8k: 100 sequences. At 2k: 400 sequences. **So the MoE only wins if your workload is short-context and high-concurrency.** For a long-context enterprise assistant, you sit permanently on the wrong side of the crossover.

**🗣 Say this in the room:** "For that pair on 4 H100s, the crossover is around 300 concurrent decode tokens. Below it the dense model wins by up to 2× because decode is bandwidth-bound and the MoE reads 141 GB per step against the dense model's 70. Above it the MoE wins by ~1.8× because it does half the FLOPs. Then I'd check whether the KV budget even permits batch 300 at our context length — for 32k contexts it caps out around 25 sequences, so we'd never get there."

### Walk me through the memory profile of an MoE over a request's lifetime. What dominates when?

At `t = 0` — cold server, zero requests — HBM is essentially all weights. For Mixtral 8x22B in bf16 that is 282 GB of the 640 GB on an 8×H100 node: 44% of the machine gone before a single token exists. The dense comparison is stark: Llama-3-70B bf16 is 141 GB, 22%.

As requests arrive and sequences grow, KV cache grows linearly in total tokens resident. Mixtral 8x22B has GQA with 8 KV heads, `d_head` 128, 56 layers: `2 · 56 · 8 · 128 · 2 bytes = 229,376 B = 224 KiB per token`. So the crossover — the point at which KV cache exceeds expert weights — is at:

```
282e9 bytes / 229,376 bytes-per-token ≈ 1.23 M tokens resident
```

which is 38 sequences at 32k context, or 9.6 sequences at 128k. On an 8×H100 node you have `640 − 282 = 358 GB` for KV = 1.56 M tokens, so **you cross over just before you run out** — you spend the last third of your capacity in a regime where KV dominates and the weights are the smaller half. That shape is important: it means both levers matter, and neither alone saves you.

Now do the same for **DeepSeek-V3 with MLA**, and watch the profile invert. MLA stores a compressed latent plus a decoupled RoPE component — 512 + 64 = 576 elements per token per layer — so across 61 layers at fp8 that is `576 · 61 · 1 B ≈ 35 KB per token` (≈70 KB at bf16). Weights at fp8 are 671 GB. Crossover:

```
671e9 / 35e3 ≈ 19.2 M tokens resident
```

On 16 H100s you have `1280 − 671 = 609 GB` of KV budget = 17.4 M tokens — **you never reach the crossover.** For DeepSeek-V3 the memory profile is expert-weight-dominated over the *entire* operating envelope, at any context length you can afford. That is not an accident: MLA and a 671B MoE were co-designed. If DeepSeek had used standard GQA, the KV cache would have consumed the remaining HBM immediately and the model would have been unservable at useful concurrency. **📄 Paper:** the MLA/MoE pairing is the central architectural argument of the DeepSeek-V2 and V3 technical reports.

**🗣 Say this in the room:** "Expert weights are a fixed, up-front tax — 44% of an 8-GPU node for Mixtral 8x22B before you serve anything. KV cache grows linearly and overtakes them somewhere around a million resident tokens for a GQA model. Which side dominates tells you which lever to pull: weight-dominated means quantize or shard wider; KV-dominated means MLA, shorter contexts, or eviction. For DeepSeek-V3, MLA pushes the crossover to ~19 M tokens, so it's weight-dominated everywhere — which is exactly why they needed MLA to make the MoE servable."

### Does MoE sparsity help during prefill the same way it helps during decode?

No, and the asymmetry is instructive, because prefill is the one phase where MoE delivers exactly what the brochure promises.

**Prefill is compute-bound.** A 4k-token prompt pushes 4,096 tokens through each weight matrix in a single pass, so arithmetic intensity is enormous and you are limited by tensor-core throughput. Here, "37B active instead of 671B" translates directly: you do 18× fewer FLOPs and prefill is genuinely ~18× cheaper than the dense model of the same total size. Expert utilization is also excellent, because 4,096 tokens at top-8 over 256 experts gives 32,768 assignments — an average of 128 tokens per expert, which makes each expert's GEMM properly shaped. **MoE prefill is close to ideal.**

**Decode is bandwidth-bound**, and as established, at any realistic batch you read all the expert weights anyway. The FLOP saving buys you nothing because you were not FLOP-limited. So MoE's advantage evaporates in exactly the phase that dominates user-perceived latency for long generations.

The consequence is a workload-shape rule that I would lead with in a design review: **MoE is best for prefill-heavy workloads and worst for decode-heavy ones.** Concretely — RAG with a 20k-token context producing a 200-token answer has a 100:1 prefill:decode ratio and is a great MoE fit. An agent loop that generates 4,000 tokens of reasoning off a 500-token prompt is a 1:8 ratio and is a terrible MoE fit. A chat product with prefix caching is somewhere in between and shifts *toward* decode-heavy over a session, because the cached prefix removes prefill work but not decode work.

**💰 Math:** take a RAG request with 20k prompt tokens and 200 output tokens on the 4×H100 setup above. Dense 70B prefill: `2 · 70e9 · 20000 = 2.8 PFLOP` over 3.96 PFLOP/s = 707 ms at peak, realistically ~1.4 s at 50% MFU. Mixtral 8x22B prefill: `2 · 39e9 · 20000 = 1.56 PFLOP` → 394 ms at peak, ~790 ms realistic. The MoE saves ~600 ms of TTFT. Then decode 200 tokens at batch 32: dense `200 · 5.2 ms = 1.04 s`, MoE `200 · 10.5 ms = 2.1 s`. The MoE gives back 1.06 s. **Net: the MoE is slower end-to-end for this request, despite winning TTFT.** If your product is judged on time-to-first-token — a streaming chat UI — the MoE looks better. If it is judged on time-to-complete-answer — an agent step, a batch job — it looks worse. Know which metric your product is graded on before you pick the architecture.

**⚠ Trap:** running a prefill-only benchmark (or an MMLU-style eval where outputs are one token) to justify an MoE, then shipping it behind a chat product. Your benchmark measured the phase where MoE excels and never exercised the phase where it does not.

### What does an MoE do to your serving kernels? I'm thinking about dynamic shapes and CUDA graphs specifically.

It attacks the thing that makes low-latency decode fast: **static shapes**.

A dense decode step has a completely fixed shape graph. Batch size is fixed by the scheduler, every matmul has compile-time-known dimensions, so you capture the whole step as a CUDA graph and replay it — collapsing hundreds of kernel launches into one. At batch 32 with 80 layers, launch overhead is easily 30–50% of the step, so graph capture is not a micro-optimization; it is the difference between 10 ms and 15 ms per token.

An MoE decode step has shapes that **depend on the data**. How many tokens go to expert 7 is known only after the router runs, and it differs every step. That breaks graph capture outright, because a captured graph bakes in tensor sizes. Your options, and this is a real design menu:

**1. Pad to a fixed capacity per expert.** Allocate `capacity = CF · B · k / E` slots per expert unconditionally, pad the short ones with zeros, drop the overflow. Shapes are static again, graphs capture cleanly. The cost is wasted FLOPs on padding (at `CF = 1.25` and real imbalance, easily 30–50% waste) and, worse, **you have reintroduced token dropping at inference time**, which is a correctness compromise you must consciously accept.

**2. Dropless with device-side shapes.** Use kernels that read their group sizes from device memory — grouped GEMM with an offsets tensor, or MegaBlocks-style block-sparse matmul. Nothing is dropped and nothing is padded, but the shapes are only known on-device, so the launch configuration must be either worst-case or resolved by the kernel itself. This is where the field landed for quality reasons, and it is why MoE inference kernels are so much more complex than dense ones.

**3. Bucketed graphs.** Capture several graphs at quantized capacity levels (e.g. 64, 128, 256 tokens per expert) and dispatch to the smallest one that fits. Recovers most of the graph benefit with bounded padding waste. Memory cost is one captured graph per bucket per batch size, which multiplies fast.

**⚠ Trap:** enabling CUDA graphs on an MoE deployment, observing a 30% speedup in benchmarks, and not noticing that the padding path silently drops overflow tokens under skewed traffic. The symptom is not an error — it is a quality regression concentrated on exactly the traffic that causes skew, which is usually your most specialized and highest-value tenant. **Rule: if you pad, you must export an overflow counter, and it must be an alerting metric, not a debug log.**

**⚠ Trap:** assuming the vendor stack handles all of this. Engine support for MoE features — dropless kernels, expert parallelism, per-expert quantization, graph capture with MoE — varies substantially by engine and by version, and the matrix changes every few months. **📅 Volatile:** verify feature support against the current release notes of whichever engine you name in an interview, and say that you would.
