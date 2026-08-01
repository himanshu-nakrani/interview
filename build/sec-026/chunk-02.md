### Explain FP8 E4M3 versus E5M2, and tell me which one you'd point at the KV cache.

Both are 8-bit floats and the only question is how you spend the seven non-sign bits. **E4M3** is 4 exponent bits and 3 mantissa bits: dynamic range up to ±448, and 3 mantissa bits means 8 representable steps per binade, so worst-case relative error around 6.25%. **E5M2** is 5 exponent bits and 2 mantissa: range up to ±57,344, but only 4 steps per binade, worst-case relative error around 12.5%. The trade is stated exactly once and then applied everywhere: **E4M3 buys precision, E5M2 buys range.** In the standard OCP formulation E4M3 also gives up infinities to reclaim encodings, while E5M2 keeps IEEE-like inf/NaN behaviour.

For the KV cache, my default is **E4M3 with a calibrated scale**. The reason is that K and V are activations with a bounded, empirically stable distribution — you can measure their per-tensor (or per-head) absolute maximum on a calibration set, fold a scale factor in, and then you need precision inside that range far more than you need headroom above it. E5M2's extra three orders of dynamic range are insurance against outliers you have already handled with the scale.

The counter-argument, and it is a legitimate one: E5M2 is often usable **without** calibration, because its range absorbs outliers you never measured. If you are serving fifty fine-tuned variants and cannot run a calibration pass per checkpoint, E5M2 is the pragmatic choice, and the reported quality cost is small. Published KV-quantisation evaluations put the average benchmark delta at roughly **0.3 percentage points for a Llama-3.3-70B at E5M2** and around **0.6 points for a Qwen-72B-class model** (**📅 Volatile:** these are reported figures on specific benchmark suites and specific engine versions; re-measure on your own eval before quoting them as yours).

**⚠ Trap:** quantising the KV cache and the weights and reporting one aggregate quality number. They are independent decisions with independent failure signatures. Weight quantisation degrades uniformly across all inputs; KV quantisation degrades *disproportionately on long contexts*, because error compounds over more cached tokens and the attention softmax amplifies error in the tail. A KV-quant eval run at 2k context tells you almost nothing about behaviour at 128k, which is exactly where you deployed it to help.

**🗣 Say this in the room:** "E4M3 for precision, E5M2 for range. For KV I default to E4M3 with a calibrated per-tensor scale, because K and V are bounded activations and I'd rather have mantissa bits than exponent bits. I'd switch to E5M2 if I couldn't run calibration per checkpoint."

### What does KV quantisation actually cost in quality, and how would you measure it rather than trusting a blog post?

The mental model that keeps you honest: KV quantisation injects noise into the *keys* that determine attention weights and into the *values* that get mixed. Noise in values averages out — it's a weighted sum, and errors partially cancel. Noise in keys does not average out, because it passes through a softmax, which is an exponential; a small perturbation to a logit near the top of the distribution can reorder which token gets attended to. **Keys are the fragile side.** That single asymmetry drives every serious KV-quantisation design.

Given that, here is the measurement protocol I insist on, because generic benchmark averages actively hide the failure.

**One: evaluate at your deployed context length, not at 2k.** Build the eval set from real production context-length distribution. If your p95 is 60k tokens, your eval must contain 60k-token items. This is the step everybody skips and it is the step that catches the regression.

**Two: use tasks that require precise retrieval from mid-context.** Needle-in-a-haystack variants, multi-hop retrieval over a long document, exact citation of a line number. Aggregate reasoning benchmarks are insensitive to KV noise because they are robust to attending to a slightly-wrong token; exact-retrieval tasks are not.

**Three: measure the distributional shift, not just the mean.** Run both configurations on the same 2,000 items and look at per-item deltas. A KV quantisation that moves the mean by 0.3 points while flipping 4% of items from correct to incorrect and 3.7% from incorrect to correct is not "0.3% worse" — it is a 7.7% churn rate, and if any of those flipped items are in a regulated workflow you have a problem the mean does not show.

**Four: check the tail of the output distribution.** Compare the KL divergence between the fp16 and quantised next-token distributions on a held-out corpus. This catches degradation that task metrics miss, and it's cheap.

**⚠ Trap:** "we A/B'd it and quality was flat." KV quantisation regressions are context-length-dependent and often long-tail. An A/B on a chat product where the median conversation is 900 tokens will be flat and will tell you nothing about the 128k document-analysis path where the same flag is doing real damage. Segment every KV-quant A/B by context-length bucket or the result is uninterpretable.

### If I halve the KV cache, what do I actually get — more concurrency or more context? Show me.

You get to choose, and the choice is a product decision that should be made explicitly rather than falling out of a config default. The pool is a fixed token budget:

$$\text{live tokens} = \frac{\text{KV pool bytes}}{\text{bytes per token}}$$

Halving `bytes per token` doubles `live tokens`. What you spend that on is up to you. Take the 4×H100 / 70B / 200.4 GiB pool from earlier, bf16 KV at 320 KiB/token → 655k live tokens. Switch to FP8 KV at 160 KiB/token → **1.31M live tokens**. Three ways to spend it:

**Spend it on concurrency.** 8k sessions go from 80 → 160. If you are throughput-bound and your context distribution is stable, this roughly doubles your requests-per-node and halves your cost per request. At $10/hour per node and 80 → 160 concurrent 8k sessions, your per-session hourly cost goes from $0.125 to $0.0625.

**Spend it on context.** Keep concurrency at 80 and double `max_model_len` from 8k to 16k. Now every user gets twice the history, and your product does something it couldn't.

**Spend it on tail latency.** Keep both fixed, and hold the extra 655k tokens as headroom. Preemption — where the scheduler evicts a running sequence's cache because the pool filled — is what produces the p99 TTFT spikes that wreck your SLO. Running the pool at 50% instead of 90% occupancy makes preemption approximately never happen. This is the option nobody picks and it is frequently the right one.

**💰 Math:** the second-order effect is easy to miss. Doubling batch also moves you rightward on the arithmetic-intensity curve. Going from batch 80 to 160 at decode moves AI from 80 to 160 FLOP/byte against a bf16 ridge point of ~295 — still bandwidth-bound, so throughput scales close to linearly with batch. Going from 300 to 600 would *not*, because you've crossed the ridge and become compute-bound. Know which side of the ridge your batch increase lands on before you promise a 2× throughput win: below the ridge you get most of it, above the ridge you get almost none of it.

**🗣 Say this in the room:** "Halving the KV cache doubles the token budget. Whether that becomes 2× concurrency, 2× context, or 2× headroom against preemption is a product call, and I'd want it made deliberately. My default is to spend the first tranche on preemption headroom, because that's the one that shows up in p99 rather than in a dashboard nobody looks at."

### How do you quantise a key cache to INT4 without destroying it?

Naively — per-tensor, symmetric, round-to-nearest — you destroy it, and the reason is a structural property of the K cache that is worth knowing cold: **keys have persistent per-channel outliers.** Specific channels of the key vector carry values one to two orders of magnitude larger than their neighbours, consistently across tokens. A per-tensor scale is set by those outliers, so every ordinary channel gets quantised into a handful of levels at the bottom of the range and the attention logits go to noise.

Values do not have this structure. Value outliers are per-token and sporadic, not channel-aligned.

So the design falls out: **quantise K per-channel and V per-token.** For K, compute a scale for each channel dimension across tokens — outlier channels get their own generous scale, ordinary channels get a tight one. For V, compute a scale per token across channels. This asymmetric grouping is the core of KIVI, which showed 2-bit KV is viable with it and catastrophic without it.

The implementation wrinkle that makes this hard in a real serving engine, and which an interviewer will probe: per-channel K quantisation needs statistics *across* tokens, but tokens arrive one at a time during decode. You cannot compute a channel max over a dimension that is still growing. The practical resolution is to keep a small **residual window** of the most recent tokens in full precision — typically 32 to 128 tokens — and quantise a group only once it is complete. That window also happens to be where attention is densest, so you get accuracy for free at the position that needs it most.

A second design point: quantise keys **before** RoPE where you can. RoPE mixes channel pairs by rotation, which smears the channel-aligned outlier structure and makes per-channel scaling less effective. Applying RoPE after dequantisation preserves the structure quantisation depends on. This is one of the ideas in the KVQuant line of work.

**📄 Paper:** Liu et al. (2024), KIVI — tuning-free asymmetric 2-bit KV quantisation, per-channel for keys and per-token for values, replacing uniform per-tensor schemes that failed below ~4 bits. **📄 Paper:** Hooper et al. (2024), KVQuant — per-channel pre-RoPE key quantisation with non-uniform datatypes and explicit outlier handling, pushing usable KV precision lower still.

**⚠ Trap:** assuming INT4 KV gives you a clean 4× over bf16. It does not, because per-group scales and zero-points are metadata you also store. At group size 64 with fp16 scale and zero-point, that's 32 bits per 64 values = 0.5 extra bits per value, so INT4 is really 4.5 bits — an effective 3.6× not 4×. Small, but it is exactly the kind of arithmetic that separates people who have shipped this from people who have read about it.

### We turned on FP8 KV cache and throughput barely moved. Walk me through debugging that.

Good question, because the expected result is "roughly double the concurrency," and there are four distinct reasons it doesn't materialise.

**First — and check this before anything else — did the KV pool actually get bigger?** FP8 KV halves bytes per token, which doubles the *number of blocks* in a fixed-size pool. It does not change the pool's size in GiB. If your concurrency is not being limited by the pool but by `max_num_seqs`, you have just doubled a limit that wasn't binding. Look at whether `num_requests_waiting` is nonzero and whether cache-usage percentage was near 100% *before* the change. If the cache was sitting at 45%, KV quantisation was never going to help and you should have known that from the dashboard.

**Second: you are bandwidth-bound on weights, not on KV.** From the crossover derivation: KV traffic exceeds weight traffic only above ~214k live tokens for a 70B. If your workload is batch 32 at 2k context — 64k live tokens — then KV is under a third of the traffic and halving it improves the step time by maybe 15%. Compute the two terms before you predict the win.

**Third: the quantise/dequantise cost ate the gain.** The attention kernel must dequantise K and V on the fly. If your engine falls back to a non-fused path for the FP8 KV layout — which happens on certain head-dim, block-size, or GPU-architecture combinations — you pay a separate dequant kernel per layer, and 80 extra kernel launches per decode step at ~10 µs each is 0.8 ms on a 5 ms budget. Profile it; the signature is a large increase in kernel count with roughly unchanged bytes moved.

**Fourth: prefix cache hit rate collapsed.** If prefix caching keys blocks by content hash and you changed the cache dtype, every previously-cached block is invalid. Right after the deploy your hit rate is zero, TTFT spikes, and it takes however long your traffic pattern needs to rewarm. If you measured throughput in the first fifteen minutes, you measured a cold cache.

**🔍 Failure taxonomy:** cache usage was already below 80% → the pool was never the constraint, look at `max_num_seqs` and at your arrival rate. Cache usage was 100% and concurrency doubled but throughput didn't → you moved from KV-bound to compute-bound or to a scheduler limit. Concurrency doubled *and* per-token latency got worse by more than the batch effect predicts → dequantisation is not fused, profile the kernels. Throughput dipped then recovered over 30 minutes → prefix cache rewarming, not a regression.

### A 400-billion-parameter mixture-of-experts model — does it need more KV cache than a 70B dense model?

No, and this catches people out badly enough that it is worth stating as a rule: **the KV cache depends only on the attention configuration. Mixture-of-experts scales the feed-forward network, which contributes zero bytes to the cache.**

Concretely, Mixtral 8×7B has 32 layers, 8 KV heads, head-dim 128 — arithmetically identical to Llama-3.1-8B. Both cost `2 · 32 · 8 · 128 · 2 = 131,072` bytes/token = 128 KiB. The MoE has 46.7B parameters against the dense model's 8B, and exactly the same cache.

Now flip it around, because the *interesting* consequence runs the other way. MoE inflates the weight bucket without inflating the KV bucket, so on fixed hardware it leaves **less** room for KV. Compare per-GPU:

- **Llama-3.1-8B, FP8, 1×H100.** Weights $8\times10^9$ B = 7.5 GiB. Pool = 71.6 − 7.5 − 1.2 − 3 = 59.9 GiB → 59.9 × 2³⁰ / 131,072 = 491k tokens → **60 concurrent 8k sessions on one GPU.**
- **Mixtral 8×7B, FP8, 2×H100.** Weights 46.7 GB = 43.5 GiB, 21.8 per card. Pool = 2 × (71.6 − 21.8 − 1.2 − 3) = 91.2 GiB → 748k tokens → 91 sessions across two GPUs = **46 per GPU.**

Scale that to the 400B in the question. Take a hypothetical 400B-total / 40B-active MoE whose attention block resembles a 70B's — say 60 layers, 8 KV heads, head-dim 128, giving $2 \times 60 \times 8 \times 128 \times 2 = 245{,}760$ B/token = 240 KiB, *less* than the dense 70B's 320 KiB. But FP8 weights are $400\times10^9$ B = 372.5 GiB, so you need 8×H100 to hold them at all. Pool = $8 \times 71.6 - 372.5 - 8 \times 1.2 - 8 \times 3 = 166.7$ GiB → 728k live tokens → 89 concurrent 8k sessions across **eight** GPUs = 11 per GPU. Against the dense 70B's 80 sessions on four GPUs = 20 per GPU. The 400B MoE has a *smaller* cache per token and *half* the concurrency per GPU, purely because its weights ate the budget.

So the MoE gives you *fewer* concurrent sessions per GPU while being far cheaper per token in FLOPs (only ~40B of 400B parameters active). That is the actual MoE trade at inference and it is almost never stated correctly: **MoE trades HBM capacity for arithmetic.** If you are compute-bound, MoE is a gift. If you are KV-capacity-bound — which is what long-context and agentic workloads are — MoE makes your problem worse, and you will need more GPUs for memory reasons while the FLOPs sit idle.

**🗣 Say this in the room:** "MoE doesn't touch the KV cache — that's set by layers, KV heads and head dim. What MoE does is eat the memory budget with weights, which shrinks the residual KV pool. So on a fixed node an MoE gives you cheaper tokens but fewer concurrent long sessions. Whether that's a win depends entirely on whether I'm compute-bound or capacity-bound, and I'd want the crossover number before choosing."

### Does tensor parallelism give me proportionally more KV cache? Where does that break?

Two effects, one obvious and one that bites.

The obvious one: TP=$P$ gives you $P$ GPUs' worth of HBM, and shards the weights $P$ ways, so the KV pool grows *superlinearly* in $P$ at first. Per card the pool is $(\text{budget} - W/P - \text{overheads})$; the $W/P$ term shrinks as $P$ grows, so total pool $= P \cdot \text{budget} - W - P\cdot\text{overheads}$. Going 70B FP8 from TP=2 to TP=4: TP=2 gives $2 \times (71.6 - 32.6 - 1.2 - 4) = 67.6$ GiB; TP=4 gives $4 \times (71.6 - 16.3 - 1.2 - 4) = 200.4$ GiB. That is **3× the pool for 2× the GPUs**, because you stopped spending half of each card on weights.

The one that bites: **KV heads are sharded across TP ranks, and you cannot shard 8 heads across 16 ranks.** When TP exceeds `num_key_value_heads`, engines fall back to *replicating* KV heads across ranks. At TP=16 on a model with 8 KV heads, each pair of ranks holds a duplicate copy of the same KV head. Aggregate KV memory consumed **doubles** for the same logical cache. Your pool grows with more cards but your effective capacity does not grow with it, and the effective bytes-per-token in your capacity formula silently becomes `2 · n_layers · TP · d_head · dtype` instead of using `n_kv_heads`.

This is a genuine production surprise on models with aggressive GQA. Llama-3-70B has 8 KV heads. TP=8 is the sweet spot; TP=16 wastes memory on KV replication *and* doubles the collective count per layer. If you need more than 8-way parallelism for a model with 8 KV heads, the correct answer is usually to add pipeline parallelism on top rather than pushing TP further.

**📐 Numbers you must know:** TP is bounded above by `num_key_value_heads` before KV replication kicks in. For the Llama-3 family that ceiling is **8**. Check `num_key_value_heads` before you write a TP degree into a deployment manifest — it is a two-second check that prevents a silent 2× memory waste.

**⚠ Trap:** "we'll go TP=8 to halve latency versus TP=4." At TP=4 a 70B decode step already has 160 all-reduces and is partially launch-bound; at TP=8 you double the collectives and halve the per-GPU work, so the useful-work-to-overhead ratio degrades. Expect single-stream latency to improve well under 2×, sometimes not at all. TP is a capacity lever first, a latency lever a distant second.

### Compare tensor and pipeline parallelism from a KV-cache point of view.

Different axis, different consequences, and the KV story is the cleanest way to tell them apart.

**Tensor parallelism** splits every layer's weight matrices across ranks, and correspondingly splits the KV heads. Every rank holds a slice of *every* token's cache for a subset of heads. A sequence's cache is therefore spread horizontally across the whole TP group, and any operation on it — eviction, offload, migration to another node — must be coordinated across all $P$ ranks. Communication is two all-reduces per layer, which is fine over NVLink inside a node and awful over Ethernet between nodes. **Rule: TP inside a node, never across.**

**Pipeline parallelism** splits by *layer*. Stage 0 holds layers 0–19 and the KV for those layers only; stage 1 holds 20–39, and so on. Total KV bytes are identical; the partition is vertical rather than horizontal. Communication is point-to-point activation passing between adjacent stages — a `[tokens, hidden]` tensor, tiny compared to all-reduce traffic — which is why PP is the correct tool for crossing node boundaries.

The KV-specific consequences of PP that people underweight:

*Bubbles are worse at decode than at training.* Pipeline efficiency depends on having enough micro-batches in flight to fill the stages. During decode with continuous batching, the "micro-batch" is one iteration of the whole running batch, and a $P$-stage pipeline idles $(P-1)/P$ of each stage's time unless you overlap iterations — which means having multiple decode iterations in flight, which means more scheduler complexity and more subtle interactions with preemption.

*A sequence's cache is not co-located.* Under PP, evicting or offloading one sequence touches every stage. Under TP, it touches every rank. Neither is local, but PP additionally means the stages are on different *nodes* in the typical topology, so a KV migration is a cross-node operation.

*PP does not reduce per-stage weight memory as evenly as you'd like* if layers have different sizes, and the embedding and LM-head layers are large and land on the end stages. Uneven stages give you uneven KV pools, and your capacity is set by the smallest one.

**🗣 Say this in the room:** "TP shards KV by head, PP shards it by layer. Same total bytes. TP costs you two all-reduces per layer so it stays inside a node; PP costs you an activation handoff per stage so it crosses nodes. For a 70B on 8 H100s in one box I'd do TP=8. For a model that genuinely doesn't fit in one box I'd do TP inside each node and PP across them, and I'd budget for pipeline bubbles at decode."

### Someone opens a PR bumping `max_model_len` from 8k to 128k because "the model supports it." What do you say in review?

I block it, and I give three specific numbers rather than a vibe.

**Number one: this is a 16× reduction in worst-case concurrency, not a feature flag.** Our KV pool is 200.4 GiB. At 8k a session is 2.5 GiB and we fit 80. At 128k a session is 40 GiB and we fit 5. The engine will happily admit five 128k requests and then queue everything else. The change does not "allow longer contexts" — it allows five users to take the whole node hostage.

**Number two: the engine reserves against the declared maximum in several places, so you pay even when nobody sends a long request.** Block-table sizing, any preallocated scratch that scales with maximum sequence length, and — critically — the scheduler's admission decisions all reference `max_model_len`. You do not get "8k performance until someone sends 128k."

**Number three: prefill time is quadratic in the attention term and that lands on everyone's TTFT.** A 128k prefill for a 70B: linear part $2 \times 70\times10^9 \times 131{,}072 = 1.84\times10^{16}$ FLOP; attention part $2 \times 80 \times (131{,}072)^2 \times 8192 = 2.25\times10^{16}$ FLOP — already larger than the rest of the model, consistent with the ~107k crossover. Total $4.1\times10^{16}$ FLOP; on 4×H100 at FP8 and 40% MFU ($3.17\times10^{15}$ FLOP/s) that is **12.9 seconds** of the entire node. Without chunked prefill, every in-flight decode stalls for those 12.9 seconds. Your p99 inter-token latency becomes 12,900 ms.

What I'd approve instead: raise `max_model_len` to the p99.5 of *observed* context length plus headroom — probably 32k — put a hard rejection at the gateway above that with a clear error, route genuine long-document work to a **separate pool** with its own SLO and its own pricing, and enable chunked prefill so a long prefill is sliced into token-budget-sized pieces instead of monopolising an iteration. Then, and only then, consider FP8 KV to buy the capacity back.

**🗣 Say this in the room:** "`max_model_len` is a capacity declaration, not a capability flag. Going 8k to 128k takes worst-case concurrency from 80 to 5 and puts a 13-second prefill in front of everyone's next token. If we want long context we need a separate pool with a separate SLO, and we should price it differently, because it costs 16× more per session."

### When a request asks for four samples of the same prompt, what happens to the KV cache?

Naively it is 4× the memory, and the fix — copy-on-write over paged blocks — is one of the more elegant things in modern serving, and worth being able to explain because it is the same mechanism that powers prefix caching and beam search.

The observation: four samples of the same prompt share an identical prefix. Their K and V for the prompt tokens are bit-identical, because the computation is deterministic and causal. They diverge only when sampling produces different tokens. So if the cache is stored in fixed-size blocks with a per-sequence block table — the OS page-table analogy is exact — the four sequences can point their block tables at the *same* physical blocks for the shared prefix, with a reference count. When a sequence writes into a block that has refcount > 1, you copy that single block, decrement, and repoint. Copy-on-write, with a page table, in HBM.

Work the arithmetic. A 2,000-token prompt, $n=4$, 500 generated tokens each, block size 16, on a 70B at 320 KiB/token:

- Naive: $4 \times 2{,}500 = 10{,}000$ token-slots = 3.05 GiB.
- Shared: $2{,}000 + 4 \times 500 = 4{,}000$ token-slots = 1.22 GiB, **2.5× less**.
- Plus copy-on-write overhead: the prompt's last block is partially filled (2000 mod 16 = 0 here, so cleanly aligned; with a 2,003-token prompt, one block gets copied per branch — 3 extra blocks of 16 tokens = 48 token-slots, 0.4% overhead).

Beam search is the same mechanism with a tree instead of a fan-out, and it is where the saving is largest, because beams share long prefixes and prune constantly; copy-on-write means pruning a beam is a refcount decrement rather than a memory copy.

**⚠ Trap:** issuing $n$ separate API calls with the same prompt instead of one call with $n=4$. Separate calls arrive as separate sequences and — unless automatic prefix caching happens to catch them and they land in the same scheduling window — you prefill the prompt four times and store it four times. One request with $n=4$ guarantees the sharing. On a 2,000-token prompt at 70B scale that is 6,000 tokens of redundant prefill, $2 \times 70\times10^9 \times 6{,}000 = 8.4\times10^{14}$ wasted FLOP, = 0.265 s of a 4×H100 node at 40% MFU. At 10,000 such requests/day: $10{,}000 \times 0.265 \times 4 = 10{,}600$ GPU-seconds/day = 2.94 GPU-hours/day × $2.50 = **$7.36/day = $221/month**, thrown away by a client-side loop that should have been one parameter.

### How much do activations actually cost at inference, and what sets that floor?

At decode, essentially nothing — and that's the important half of the answer, because it means your entire activation budget is really a *prefill* budget, controlled by one knob.

Decode processes one token per sequence. The largest live tensor is the MLP intermediate: `[batch, d_ff]`. For a 70B at batch 256, `d_ff = 28,672`, bf16: $256 \times 28{,}672 \times 2 = 14.7$ MiB. Divided by TP=4, under 4 MiB per card. Rounding error.

Prefill processes `max_num_batched_tokens` at once, and the same tensor becomes `[T, d_ff]`. At T = 8,192: $8{,}192 \times 28{,}672 \times 2 = 448$ MiB, 112 MiB per card at TP=4, with several such buffers live across the gated-MLP's up/gate/down chain. Add attention workspace and cuBLAS scratch. That is where the 3–5 GiB per card in the budget comes from, and it is **roughly linear in `max_num_batched_tokens`** — which makes that knob the direct dial between activation memory and KV pool size.

The one non-linear term, and the one that has bitten every team that hits it: **logits**. Computing logits for all $T$ positions at vocab 128,256 in fp32 would be $8{,}192 \times 128{,}256 \times 4 = 4.2\times10^9$ bytes = **3.9 GiB** — larger than the entire rest of the activation budget. Engines avoid this by slicing the hidden states to only the positions that need sampling (the last token of each sequence in the batch) before the LM head. If you enable features that force full-sequence logits — returning logprobs for every prompt token, certain scoring or reranking endpoints — that 3.9 GiB comes back and you OOM. This is a real incident pattern: "we added a `/score` endpoint and the serving pod started crashing."

**📐 Numbers you must know:** activation memory ≈ linear in `max_num_batched_tokens`; halving it from 8192 to 4096 returns roughly 1.5–2.5 GiB per card to the KV pool, which on a 4-card 70B deployment is 6–10 GiB, or 20k–33k more live tokens, or 2–4 more concurrent 8k sessions. Small, but free if your prefill throughput has headroom.

### Your capacity spreadsheet says one thing and production says another. How do you measure real KV usage?

Stop estimating and read it out of the engine, then reconcile. Three layers of instrumentation, in the order I'd add them.

**Layer one: the engine's own counters.** vLLM exposes a Prometheus endpoint with the metrics that matter — KV cache utilisation as a fraction, running and waiting request counts, preemption counts, prefix-cache hit and query counters, and TTFT/inter-token histograms. (**📅 Volatile:** exact metric names have changed across major engine versions — verify against the version you deploy rather than copying names from a blog post.) The two that go on the primary dashboard are **cache utilisation** and **waiting requests**, because their joint state is diagnostic in a way neither is alone.

**Layer two: reconcile against startup logs.** Engines log the number of KV blocks allocated and the block size at startup. `blocks × block_size` is your exact token capacity, in the engine's own accounting. Compare it against your spreadsheet's `total_live_tokens`. If they disagree by more than 5%, your overhead assumptions are wrong and every downstream number you've quoted to a PM is wrong with them. Do this reconciliation on every model and every TP-degree change; it takes two minutes and it is the difference between a capacity model people trust and one they route around.

**Layer three: attribute usage to tenants.** The engine tells you the pool is 80% full; it does not tell you that one customer's document-analysis job is holding 60% of it. Emit per-request `prompt_tokens + generated_tokens` at completion, weight by request duration, and you have **token-seconds per tenant** — the correct unit for both chargeback and for finding the noisy neighbour. This is your own instrumentation at the gateway; no engine gives it to you.

The joint interpretation of layers one and two is the payoff:

**🔍 Failure taxonomy:** high utilisation + zero waiting → healthy, you are efficiently full; consider raising concurrency limits. High utilisation + nonzero waiting + rising preemptions → **genuinely KV-bound**; the fixes are KV quantisation, lower `max_model_len`, or more hardware. Low utilisation + nonzero waiting → you are *not* KV-bound; you are hitting `max_num_seqs`, or a token-budget limit, or the bottleneck is upstream in the gateway. Utilisation sawtoothing between 30% and 100% → arrival burstiness exceeding your headroom; fix with admission control, not capacity. Utilisation pinned at 100% with *zero* preemptions → suspicious; check that preemption metrics are actually wired up, because a pool that is genuinely full is preempting.

**⚠ Trap:** treating KV utilisation as a health metric where lower is better. It is a *utilisation* metric, and 30% means you bought hardware you are not using. The target is high-and-stable with headroom for your arrival burstiness — I aim for 70–80% steady-state on interactive traffic, which leaves room for a burst to be absorbed without preemption.
