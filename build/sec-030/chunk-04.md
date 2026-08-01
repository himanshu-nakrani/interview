### I give you a bottleneck, you give me a compression axis. Walk me through your whole decision tree.

This is the thing I'd want to say first in any compression interview, because it reframes the question from "which method is best" — there is no answer to that — to "which constraint binds," which has a defensible answer every time.

**Step 0: measure which of four things is binding.** Run your workload and answer: does the model *fit*? What is the arithmetic intensity at your real batch size? Where do the FLOPs go — prefill or decode? What is the latency budget and what is the floor you'd hit even with infinite hardware? Everything below follows from those four.

**Memory-capacity bound** — the model plus KV cache plus activations does not fit, or fits so tightly that concurrency is capped at an embarrassing number. → **Quantize weights.** This is the highest-leverage, lowest-risk axis in the whole field: 4-bit weight-only gets you 3.8× on the weights for ~1–2% aggregate quality, in an afternoon, with mature kernels. Do this before anything else. If it still doesn't fit, quantize the KV cache next, then consider a smaller model.

**Memory-bandwidth bound** — low batch, decode-dominated, GPU utilisation looks high but the SM occupancy tells you it's waiting on HBM. → **Quantize weights (fewer bytes per token) and quantize the KV cache (attention reads the whole cache every step).** Sparsity helps here only via its memory ratio (1.78× for 2:4), which is worse than 4-bit's 3.8×, so quantize first and add sparsity only if you need more and have a kernel.

**Compute bound** — high batch, prefill-heavy, the GEMMs are genuinely saturating the tensor cores. → **This is where quantizing weights alone does nothing.** You need fewer FLOPs or faster FLOPs. Faster FLOPs = W8A8 fp8 (2× on Hopper) or fp4 on Blackwell. Fewer FLOPs = structural: prune width and depth, or distill to a smaller model. This is the only branch where pruning and distillation are the primary answer rather than an afterthought.

**Latency-floor bound** — even at batch 1 with perfect kernels you cannot hit the SLO, because `n_layers × time_per_layer × n_output_tokens` exceeds your budget. → **No amount of quantization saves you**, because the sequential depth is fixed. Two real options: a **smaller model** (fewer layers = less sequential depth, which is why depth pruning and distillation matter here), or **speculative decoding**, which is the only technique that breaks the one-token-per-forward-pass barrier by verifying several draft tokens in one pass.

**Cost bound with latency headroom** — you're fine on latency, the bill is the problem. → Batch harder, cascade to a cheaper model, use provider batch tiers, and quantize for throughput. Compression is one lever among several here, and often not the biggest one.

**⚠ Trap:** the reflex of "we're slow, let's quantize." If you are compute-bound at batch 128, W4A16 will make you *slower*. If you are latency-floor-bound, quantization moves the number by 20% when you need 3×. Diagnose first — the diagnosis takes an hour and the wrong fix takes a sprint.

**🗣 Say this in the room:** "Memory-capacity bound, quantize weights. Bandwidth bound, quantize weights and KV. Compute bound, prune or distill or move to fp8/fp4 — weight-only quantization does nothing for you there. Latency-floor bound, you need a shallower model or speculative decoding, because no bit-width change shortens the sequential depth."

### Design this: serve Llama-3-70B-class quality at p95 under 2 seconds for a chat product, on 2×H100-80GB per replica. Show me the sizing.

Let me build it from the memory equation up, because that's what determines everything else.

**Weights.** fp16 is 140 GB against 160 GB of HBM — technically fits across two cards, but leaves 20 GB for KV, activations, workspace and CUDA context, which is not a serving configuration, it's a demo. At **4-bit group-128 (4.25 bpw): 70e9 × 4.25/8 = 37.2 GB**, so ~18.6 GB per card under TP=2. At fp8: 70 GB, 35 per card. I'd start at 4-bit weight-only.

**Overheads.** CUDA context + engine + activation workspace + graph capture: budget ~8 GB per card, 16 GB total. Available for KV: 160 − 37.2 − 16 = **106.8 GB**.

**KV.** 80 layers, 8 KV heads, head_dim 128 → 160 KB/token at fp16, **80 KB/token at fp8**. Assume a chat workload averaging 3,000 tokens of context: 3,000 × 80 KB = 240 MB per sequence. 106.8 GB / 0.24 GB = **445 concurrent sequences**. That is comfortably more concurrency than a 2-GPU replica can serve at 2s p95, so KV is *not* my binding constraint at this context length — good, that tells me fp8 KV is sufficient and I don't need to reach for int4 KV.

**Now latency, which is the actual constraint.** Decode is bandwidth-bound. Per card: 37.2/2 = 18.6 GB of weights, HBM3 at 3.35 TB/s, so the theoretical floor is 18.6/3350 = 5.6 ms/token; real kernels achieve maybe 65–75% of peak and TP=2 adds an all-reduce per block, so call it **~9 ms/token at low batch**. A 400-token answer is 3.6 s of decode — **over budget**. Batching makes it worse per-token, not better.

So the design has to attack output length and sequential depth, not bits:
- **Stream.** If p95 is measured to *first* token rather than to completion, the problem changes entirely. TTFT = prefill of 3,000 tokens: 2 × 70e9 × 3000 = 4.2e14 FLOPs at ~700 achieved TFLOPS across two cards = **0.3 s**. That's the number to sell to the product team, and I would push hard to define the SLO on TTFT + inter-token latency rather than total, because for a chat product that's what users actually feel.
- If total-completion p95 really is the SLO: **speculative decoding** with a ~1B draft. At an accepted-token rate of 2.5 per verification pass — realistic for chat continuation — 400 tokens takes 160 passes instead of 400. Verification is slightly more expensive than plain decode (it's a small batch of candidates), so call it 11 ms/pass → **1.76 s**. Now you're inside budget.
- Or **distill to a smaller model** and accept the quality trade, which is the honest alternative and should be on the table.

**Final config:** 4-bit W4A16 weights (GPTQ or AWQ, group 128, calibrated on production traffic), fp8 KV cache, TP=2, continuous batching with chunked prefill so long prefills don't stall decode, prefix caching on the system prompt, plus a 1B draft model for speculation. **The sentence that matters:** compression got the model to fit and made decode 3.8× faster; it did not by itself meet the SLO, and I'd say so up front rather than discovering it in load test.

**⚠ Trap:** sizing the replica on average context and getting paged at 2am. Size KV on the p95 or p99 of your context-length distribution, and configure the engine's preemption/swap policy explicitly, because a burst of 32k-context requests will otherwise evict in-flight sequences and your p99 will go vertical.

### Cursor-style inline code completion: sub-200ms end to end, high volume. What's your compression strategy?

The dominant fact is that this is a **latency-floor problem, not a memory or cost problem**, and it changes every answer. A completion is short — 20 to 60 tokens — and the user is typing, so the whole interaction has to feel like autocomplete rather than like a chat.

Budget the 200 ms: network round-trip 20–40 ms, request handling and tokenisation ~5 ms, prefill of the file context, decode of ~40 tokens, detokenisation and streaming. If you leave 130 ms for model time and need 40 tokens, that's **3.25 ms/token** including prefill. On an H100 at 3.35 TB/s, 3.25 ms of bandwidth is 10.9 GB of weight traffic per token. So the model has to be **under ~10 GB in its serving format** — that is a 3B at fp16, or a 7–13B at 4-bit, and nothing larger. This arithmetic, done in the room, is the answer to the question.

So the strategy, in order:

**1. Pick a small model first.** Compression cannot rescue a 70B here — 37 GB at 4-bit is 11 ms/token, 440 ms for 40 tokens, three times over budget before you've sent a packet. Start at 1–7B, ideally a code-specialised model, ideally distilled from something larger on your own completion data.

**2. Then quantize for bandwidth.** W4A16 on a 7B: 7e9 × 4.25/8 = 3.7 GB → 1.1 ms/token theoretical, ~1.6 ms realistic. Comfortable. This is exactly the regime W4A16 was built for: batch is small per replica because you're latency-optimising, decode-dominated, memory-bandwidth-bound.

**3. Attack prefill separately, because it's most of your time.** A code completion carries a big context — the current file, related files, symbols. 8,000 tokens of prefill on a 7B is 2 × 7e9 × 8000 = 1.1e14 FLOPs, ~160 ms at 700 achieved TFLOPS. That blows the budget by itself. The fixes are **prefix caching** (the file's prefix is nearly identical between keystrokes — this is the single highest-leverage optimisation in the entire design and it is not compression), plus fp8 activations to double prefill throughput, plus aggressive context truncation.

**4. Speculative decoding, with a caveat.** It helps, but a draft model adds its own latency per step and the win at 40 output tokens is modest. For code specifically, **n-gram / prompt-lookup speculation** — draft by copying from the prompt, since completions frequently repeat identifiers and lines already in the file — is nearly free and has a high acceptance rate on code. I'd try that before a draft model.

**💰 Math — why compression is also the cost answer here.** Say 50M completions/day, 8k prefill (95% prefix-cache hit so ~400 effective new tokens) + 40 decode. Decode dominates GPU-seconds: 40 × 1.6 ms = 64 ms/request. 50e6 × 0.064 = 3.2e6 GPU-seconds/day = 889 GPU-hours/day. At $3/GPU-hour that's **$2,667/day ≈ $80k/month** on the 4-bit 7B. At fp16 the same fleet is 2.4× the GPU-seconds → ~$192k/month. **Quantization is a $112k/month line item here**, and that's the framing to bring to a room. **📅 Volatile:** GPU pricing; redo with your actual rate.

**⚠ Trap:** optimising the model and ignoring that at 200 ms, a cross-region network hop is 15% of your budget and a cold replica is 100% of it. Latency-floor products are won on placement, prefix cache hit rate and warm capacity at least as much as on bit-width.

### Enterprise RAG at Harvey/Glean scale — 32k–128k contexts, high concurrency, quality is the product. Where does compression fit?

Here the binding constraint is **KV cache capacity**, and the answer looks nothing like the Cursor one.

**The arithmetic that frames it.** Take a 70B (80 layers, 8 KV heads, 128 head_dim): 160 KB/token fp16, 80 KB/token fp8. At 128k context that is **20.5 GB per sequence in fp16, 10.2 GB in fp8**. On a 4×H100 node (320 GB) with 4-bit weights (37.2 GB) and 32 GB of overhead, you have ~250 GB of KV. In fp16 that is **12 concurrent 128k sequences**. Twelve. On $12/hour of GPU. That is the entire business problem in one number, and it is why long-context enterprise RAG is expensive.

fp8 KV doubles it to 24. That is the single highest-leverage compression decision in this design — bigger than anything you do to the weights — and it's why fp8 KV has been a production default since roughly 2024.

**What I'd actually deploy, in priority order:**

1. **fp8 KV cache.** 2× concurrency. Validate on a length-stratified retrieval eval, because KV quantization error compounds with context length and this is exactly the workload where it shows.
2. **Prefix caching, aggressively.** In enterprise RAG the same document, the same system prompt and the same tool schema recur constantly. A cached prefix costs zero prefill. This is not compression but it dominates compression in impact — it is the difference between paying for 128k of prefill per request and paying for 2k.
3. **W8A8 fp8 weights, not W4A16.** This workload is *prefill-dominated*: 128k prefill on a 70B is 2 × 70e9 × 128e3 = 1.79e16 FLOPs, versus 500 output tokens of decode at 1.4e14 — prefill is **128× the FLOPs**. You are compute-bound, and 4-bit weight-only does nothing for a compute-bound GEMM while fp8 halves it. This is the cleanest example in the guide of why the decision tree matters more than the method.
4. **Reduce the context before you compress it.** A better reranker that gets you from 128k to 16k of context is a 8× win on prefill FLOPs and KV bytes simultaneously, with *better* quality (less distractor text) rather than worse. I'd fund that ahead of any quantization work. The senior move in this design round is saying so.
5. **MRL-truncated embeddings in the retrieval stage.** Search at 256 dims and rerank at full width: 6× less index memory and 6× less distance computation, negligible recall loss when you rerank. This is compression applied to the part of the system that isn't the LLM, and interviewers at retrieval-heavy companies notice when you bring it up.

**⚠ Trap:** quantizing the KV cache in a legal/medical product without a needle-in-a-haystack eval at your maximum context. Quality *is* the product here; a 1% aggregate drop that concentrates entirely on "find the indemnity clause on page 78" is a product failure, not a metric. Gate on retrieval-at-length, in every language you serve.

### Does quantization actually save money? Give me a real cost model, not a vibe.

It saves money in exactly two ways and it is worth separating them, because teams routinely claim one and deliver the other.

**Channel 1 — fewer GPUs for the same model (capacity).** A 70B at fp16 is 140 GB and needs 2×H100-80GB minimum. At 4-bit it is 37.2 GB and fits on **one** H100 with ~35 GB left for KV. If your traffic is served comfortably by one card's throughput, you just halved your fleet. At $3/GPU-hour, one card is $2,160/month versus $4,320. Across 40 replicas that's **$86,400/month saved**, and it's a step function, not a gradient — it only pays if crossing the capacity threshold actually removes a GPU.

**Channel 2 — more tokens per GPU-second (throughput).** This one requires care, because the win depends entirely on your batch regime. At batch 1–8 (bandwidth-bound), 4-bit gives close to the 3.8× bits ratio, discounted by non-weight time — call it 2.5–3× real. At batch 128+ (compute-bound), W4A16 gives approximately **zero**, and fp8 W8A8 gives up to 2× because it changes the math unit.

**💰 Worked example.** A service doing 500M output tokens/day on a 70B. At fp16 on 2×H100 with TP=2, suppose you measure 1,400 output tok/s per replica at your production batch size. Tokens per replica-day = 1,400 × 86,400 = 1.21e8. You need 500e6 / 1.21e8 = **4.13 → 5 replicas = 10 H100s**. At $3/hr: 10 × 3 × 720 = **$21,600/month**.

Now 4-bit on 1×H100 per replica. Single-card removes the TP all-reduce but halves the bandwidth available; measured throughput lands around 1,100 tok/s per single-card replica (fewer cards, but 3.8× less weight traffic each). Tokens/replica-day = 9.5e7. You need 500e6/9.5e7 = 5.26 → **6 replicas = 6 H100s** → **$12,960/month**. Saving: **$8,640/month, exactly 40%**, from a two-day quantization project. That is a real, defensible number and it is the shape of answer that passes.

**The costs on the other side of the ledger, which candidates omit:**
- **Engineering.** 2–5 days to quantize, calibrate, benchmark and eval, plus the eval suite you needed anyway.
- **Quality.** 1–2% aggregate. If that costs you 0.5% of task success and each failed task costs a human 4 minutes of rework, price it. At 100k tasks/day, 0.5% is 500 tasks × 4 min = 33 human-hours/day. **At a $60/hr loaded cost that's $2,000/day = $60k/month — seven times the GPU saving.** This is the calculation that should kill most aggressive quantization proposals in a high-value enterprise product, and being the person who runs it is how you get credited as senior.
- **Ops.** Another artifact in the registry, another thing that can drift from the fp16 reference, another axis in your A/B matrix.

**🗣 Say this in the room:** "Quantization saves money through capacity — fewer GPUs per replica — and through throughput, but only in the bandwidth-bound regime. I'd size it as a concrete fleet reduction, and I'd put the quality cost on the same ledger in dollars: a 0.5% task-success regression on a high-value workflow can dwarf the GPU saving. In a consumer product the GPU saving usually wins; in Harvey's product it usually doesn't."

### What's your validation protocol after compressing a model? Be specific — this is the rule you say you enforce.

The rule, stated as a rule: **a compressed model does not ship until it has passed your own eval, sliced, including a long-context slice and a non-English slice, against the uncompressed model on identical inputs with identical decoding parameters.** Never against a published benchmark number, never in aggregate only, never with a different sampling config.

The protocol I run, in order:

**1. A numerical sanity gate, before any semantic eval.** Run 64 fixed prompts through fp16 and quantized, capture per-layer hidden states, compute cosine similarity per layer. Healthy: ≥0.99 throughout with a slow monotone drift. A knee at any layer means a specific tensor broke, and you find it in ten minutes instead of chasing an eval score for two days. Also check: no NaN/inf anywhere, and the output logit distribution's entropy is in the same ballpark.

**2. Perplexity on a held-out slice of *your* data.** Necessary, cheap, and — as I'll argue separately — nowhere near sufficient. It catches gross breakage only.

**3. Sliced task eval. This is the gate.** The slices, non-negotiable:
- **Length**: your task at 2k / 8k / 32k / max context, reported as a curve. Long-context degradation is the signature failure of both KV and weight quantization.
- **Language**: every language you serve, at its production share or higher.
- **Reasoning/math**: a chain-of-thought slice, because that's where errors amplify instead of averaging.
- **Structured output**: JSON schema validity rate, tool-call argument correctness, code-compiles rate. Measured as a hard pass/fail rate, not a similarity score.
- **Safety/refusal**: refusal rate on a red-team set and on a benign-but-adjacent set, because compression moves both and moving either is a shipping blocker.
- **Format/template**: the exact chat template and system prompt you use in production. A model evaluated on raw completions and served with a tool schema was not evaluated.

**4. Paired, not independent.** Same prompts, same seeds, greedy decoding (or fixed seed and enough samples for a confidence interval). Report the *paired* difference and its confidence interval. Two independent samples at temperature 0.7 will differ by more than your quantization effect and you'll conclude nothing.

**5. Latency and throughput at production batch size and production length distribution.** The whole point was performance; measure it in the regime you'll deploy, not at batch 1 with 128-token prompts.

**6. A shadow or canary period.** Route 1–5% of live traffic, compare on your production quality signals (thumbs, retry rate, task completion, escalation rate). Aggregate offline evals miss slice regressions; live traffic doesn't.

**⚠ Trap:** reusing the calibration set as an eval set. You optimised against it. The score will be excellent and meaningless. Hold out strictly, and ideally hold out *by time* so you also catch distribution drift.

**🗣 Say this in the room:** "The rule I enforce in review is: re-run *our* eval after any compression, sliced by context length, by language, by reasoning-vs-extraction, and by structured-output validity, paired against the fp16 baseline on identical inputs and decoding params. Aggregate benchmark parity is not evidence — it's the exact place these regressions hide."

### Our quantized model scores the same on MMLU but our agent's tool calls started failing. What happened?

This is the most common real-world quantization incident and the mechanism is worth spelling out precisely, because the answer is structural rather than anecdotal.

**Mental model: multiple-choice benchmarks measure the argmax over four options; structured output requires the argmax to be correct at every single position of a long, low-entropy sequence.** Those are wildly different robustness requirements. On MMLU, the gap between the right answer's logit and the wrong ones is typically large, so a small perturbation almost never flips it. Emitting `{"user_id": 4471, "action": "refund"}` requires ~30 consecutive correct argmaxes, several of which — the closing brace, the exact key spelling, a specific enum value — have competitors that are very close in logit space. A perturbation that flips one token in a hundred is invisible on MMLU and produces a **3% tool-call failure rate**, which for an agent that chains five calls is a 14% task failure rate.

The second mechanism: **schema and enum tokens are rare**. Function names, argument keys, and enum values from your specific API appear rarely in pretraining and only in fine-tuning. Rare behaviours live in low-margin regions of the weight space. They're also almost certainly not in your calibration set, so GPTQ/AWQ explicitly deprioritised the channels that carry them.

**Diagnosis, in order:**
1. Measure the actual rate: JSON parse-success, schema-validation-success, correct-function-selected, correct-argument-values. Four separate numbers — they fail differently and the breakdown localises the problem.
2. Compare against fp16 on identical prompts. If fp16 is at 99.2% and quantized at 96.5%, that's your regression and it's real.
3. Check whether the calibration set contained tool-formatted examples with your chat template. It almost certainly didn't.
4. Look at the logit margin on the failing positions: dump top-5 logits at the divergence point for both models. You will typically see the correct token dropping from a comfortable margin to a coin flip.

**Fixes, cheapest first:**
- **Constrained decoding.** A grammar/FSM-constrained decoder masks invalid tokens at every step, so a schema violation becomes structurally impossible. This converts the failure from "invalid JSON" into "valid JSON with possibly-wrong values," which is a much better failure and is usually a config flag in your engine. Do this regardless of quantization; it's the highest-value fix in this whole answer.
- **Re-quantize with tool-call examples in the calibration mixture**, with the production chat template applied.
- **Keep the LM head and embeddings at higher precision.** The output projection is where the final argmax margins are set; on a 128k vocab it's ~2.1 GB in fp16 on a 70B, cheap insurance.
- **Escalate the format**: group 64 instead of 128, or fp8 W8A8 instead of int4, if the margin problem persists.

**⚠ Trap:** treating this as "the model got dumber" and reaching for a bigger model. It didn't get dumber in any general sense — it got *less sharp on low-margin, low-frequency tokens*. The fix is targeted, not a model swap.

### We quantized to 4-bit and throughput went down. Give me your hypothesis list.

Good — this happens often enough that having an ordered list is a genuine signal of experience.

**1. You're compute-bound, not bandwidth-bound.** The most likely cause by a wide margin. At batch 64+ with long prefills, the GEMMs saturate the tensor cores. W4A16 doesn't change the math — it *adds* a dequantization step in the kernel's inner loop. You paid a tax for a benefit you weren't eligible for. Check: what's your average running batch size and prefill:decode token ratio? If batch is high or prefill dominates, this is it, and the fix is fp8 W8A8, not int4.

**2. You're on a slow generic kernel.** Engines ship optimised kernels for a subset of (format × group size × act-order × shape × dtype) and a correctness-only fallback for everything else. A GPTQ checkpoint with act-order and a group size your fast kernel doesn't handle, or a shape that isn't tile-aligned, silently lands on the fallback. Check the engine's startup logs for which kernel was selected — every serious engine logs this — and benchmark a plain group-128 no-act-order checkpoint as a control.

**3. Shape misalignment from a pruning or TP interaction.** Group size must divide the per-rank K dimension. TP=8 on an FFN intermediate of 11,008 gives 1,376 per rank, which is not a multiple of 128 → padding or a slow path. This is the failure I flagged under width pruning and it bites here too.

**4. The bottleneck moved to attention.** Once weights are 3.8× cheaper to read, the attention kernel and the KV traffic become a larger share. At long context, KV reads can exceed weight reads entirely, so shrinking weights moves the needle less than you modelled and any regression elsewhere dominates. Check: profile the split between GEMM time and attention time before and after.

**5. You lost a fused path.** Quantized kernels sometimes don't compose with CUDA graphs, with chunked prefill, or with the engine's fused MLP, so you traded a fused fp16 path for an unfused int4 one and picked up dozens of extra launches per layer. Check kernel-launch counts per forward pass.

**6. Dequant overhead on small matrices.** Not all weights benefit equally. Tiny tensors (norms, biases, small projections) have dequant overhead that exceeds the bandwidth saved. Most pipelines skip these; verify yours did.

**7. You're not actually comparing like for like.** Different max batch size, different `gpu_memory_utilization`, different max-num-seqs, a warm versus a cold prefix cache. This is embarrassingly common. Re-run both configs with identical serving flags and an identical request trace.

**🏋 Drill (45 minutes with a GPU):** serve any 7B model in fp16 and in 4-bit on the same card. Measure output tokens/sec at batch sizes 1, 4, 16, 64, 256, with a fixed 512-token prompt and 128-token output. Plot both curves. Pass criterion: you can point to the crossover batch size where fp16 overtakes 4-bit, and explain it with the roofline argument rather than by narrating the graph.

### How does quantization interact with LoRA adapters when you're multiplexing many fine-tunes on one base?

One base model, dozens of per-customer adapters, one fleet — that shape is everywhere in enterprise AI, and the quantization interaction has one clean rule and one sharp edge.

**The clean rule:** the base weights and the adapter live in **different precisions and different kernels, and that's fine.** The forward pass for a layer is `y = dequant(W_q) x + (B A) x · (α/r)`. The base term runs through the quantized GEMM; the LoRA term is two small dense fp16/bf16 GEMMs of rank 8–64. Because the adapter is tiny (a rank-16 adapter on a 4096×4096 layer is 2 × 4096 × 16 = 131k params versus 16.8M, i.e. 0.8%), keeping it in bf16 costs essentially nothing and preserves full precision exactly where the customer-specific behaviour lives. Serving engines with multi-LoRA support batch requests for different adapters together by grouping the LoRA GEMMs — the base GEMM is shared across the whole batch regardless of adapter, which is what makes multiplexing efficient in the first place.

**The sharp edge: do not merge a LoRA adapter into quantized weights.** Merging computes `W' = W + BA(α/r)` and requires `W` in high precision. If you dequantize the 4-bit base, add the delta, and re-quantize, you get *two* rounds of quantization error, and worse — the delta is small relative to the quantization step size, so a substantial fraction of it rounds away entirely. A rank-16 adapter with typical magnitudes can simply vanish into the rounding at 4 bits. The model loads, runs, and behaves like the base model, which is a maddening bug to chase because nothing errors.

The correct pipelines are: **(a)** keep the base quantized and apply adapters at runtime, unmerged — this is what you want for multiplexing anyway; or **(b)** if you need a single merged artifact, merge into the **fp16** base and then quantize the merged result, with a fresh calibration pass on data representative of the fine-tuned behaviour. Never merge into an already-quantized checkpoint.

This is also the QLoRA hand-off point, and it's worth being explicit: QLoRA trains a bf16 adapter against an NF4-frozen base. Because the base is frozen and quantized *during training*, the adapter has already learned to compensate for that specific quantization. If you then serve the adapter against a **differently**-quantized base — say a GPTQ int4 checkpoint instead of the NF4 one it was trained on — the compensation is wrong. Match training-time and serving-time base quantization, or merge into fp16 and re-quantize as a unit.

**⚠ Trap:** benchmarking multi-LoRA throughput with one adapter. With `n` distinct adapters live in a batch, the LoRA GEMMs become `n` small ragged matmuls instead of one, and at high adapter cardinality that overhead is real. Load-test at your actual adapter-diversity, not with a single tenant.

### Why is perplexity a necessary but badly insufficient metric for evaluating a compressed model?

Perplexity is `exp` of the mean negative log-likelihood of the true next token over a corpus. It is the right first check and the wrong gate, and the reasons are structural rather than a matter of taste.

**Why it's necessary.** It is cheap, deterministic, requires no generation, no judge, no sampling variance, and it is exquisitely sensitive to gross breakage — a broken layer, a NaN scale, a bad mask shows up as a perplexity explosion immediately. If your quantized model's perplexity is more than a few percent worse, stop and debug; you have a bug, not a quality trade.

**Why it's insufficient**, in order of severity:

**It measures the whole distribution, not the argmax.** Generation samples or takes the argmax. A quantization that shifts probability mass among the tail tokens moves perplexity but never changes an output; a quantization that flips a narrow top-2 margin barely moves perplexity and changes every output. The correlation between perplexity delta and task-metric delta is positive but loose, and it gets looser exactly where you care — at small deltas.

**It's dominated by easy tokens.** Most tokens in natural text are highly predictable — whitespace, function words, the second half of a word. They carry most of the corpus mass and almost none of the task-relevant information. A model can preserve perplexity while degrading precisely on the rare, high-information tokens: a variable name, a numeric answer, an enum value.

**It's measured on the wrong distribution.** WikiText-2 perplexity is the field's default and it is *not your traffic*. And if it's also your calibration distribution, you're grading the exam you studied for.

**It's blind to everything sequential.** Perplexity is computed with teacher forcing — every position conditions on the *ground-truth* prefix. Autoregressive generation conditions on the model's own prefix, so errors compound. Chain-of-thought degradation, repetition loops, and drift over long generations are invisible to perplexity by construction. This is the deepest reason and the one to lead with.

**It's blind to format and safety.** JSON validity, tool-call correctness, refusal behaviour — none of these are functions of average next-token likelihood.

**🗣 Say this in the room:** "Perplexity is my smoke test, not my gate. It's teacher-forced, so it can't see error compounding in generation; it's dominated by easy high-frequency tokens; and it measures the full distribution when generation only cares about the top of it. I use it to catch bugs and I gate on sliced task metrics, structured-output validity rate, and a long-context slice."

### When would you use speculative decoding instead of compressing the model — or both?

They attack different terms in the latency equation and understanding that is the point of the question.

Time to generate `N` tokens ≈ `N × (weight_bytes / bandwidth + fixed_per_step_overhead)`. **Quantization reduces `weight_bytes`.** **Speculative decoding reduces the effective `N`** — the number of full forward passes — by having a cheap drafter propose `k` tokens and the target model verify all `k` in a single pass. Verification is a batch-of-`k` forward, which on a bandwidth-bound model costs almost the same as a batch-of-1 forward, because you're reading the same weights either way. That's the free lunch: on a bandwidth-bound decode, checking 5 tokens costs what checking 1 costs.

Crucially, speculative decoding with the standard rejection-sampling acceptance criterion is **distribution-preserving** — the accepted token sequence has the same distribution as sampling from the target model directly. It costs you *nothing* in quality, which quantization cannot claim. That asymmetry drives the decision:

**Prefer speculation when quality is non-negotiable and you're latency-bound at low batch.** A legal or medical product where a 1% regression is unacceptable: speculation gives 1.5–3× on decode latency with mathematically identical output distribution.

**Prefer quantization when you're memory-capacity-bound.** Speculation doesn't help you fit a model; in fact it *costs* memory for the draft model and for the extra KV.

**Both, usually.** They compose cleanly: quantize to reduce bytes per pass, speculate to reduce passes. My 70B chat design earlier used exactly this stack.

**Where speculation fails**, which is the part that separates a real answer: (1) **high batch** — at batch 64+, the verification forward is no longer nearly-free because you're compute-bound, and the wasted FLOPs on rejected tokens are real; speculation's benefit degrades sharply and can go negative, which is why some engines disable it above a batch threshold. (2) **Low acceptance rate** — if the drafter disagrees often, you burn draft compute and reject, and you can be slower than plain decode. Acceptance is very task-dependent: high on code and templated text, much lower on creative or highly-technical generation. (3) **Draft model overhead** — a 7B drafter for a 70B target is too expensive; you want roughly a 10–20× size ratio, or a draft-free method (n-gram/prompt-lookup, or a trained lightweight head on the target model itself).

**💰 Math:** a 70B at 4-bit on one H100, ~11 ms/token realistic. 300-token answer = 3.3 s. With speculation at `k=5` and 60% acceptance (≈2.5–3 accepted tokens per pass): 300/2.75 ≈ 109 passes, each ~13 ms (verification of 5 candidates plus draft cost) = **1.42 s**. A **2.3× latency reduction with identical output distribution** — better than what 4-bit gave you, and stackable with it.

**⚠ Trap:** quoting a speculative speedup without stating the acceptance rate and the batch size. Those two numbers determine everything, and an interviewer who has shipped this will ask for both immediately.

### Give me the full failure taxonomy — how do compressed models break in production, as a procedure I could hand to an on-call engineer?

**🔍 Failure taxonomy — compressed models in production.** Organise by the symptom you'd actually see on a dashboard.

**A. Loud failures — you find these in staging.**
- *NaN/inf in output.* Almost always a zero or near-zero quantization scale (an all-zero weight group, a padded tensor) or an fp8 scale that overflowed. Fix: clamp scales, assert finiteness in the quantization pipeline.
- *Model won't load.* Format/engine mismatch, or a checkpoint quantized with a different tool version. Not subtle, cheap to fix.
- *Garbage tokens from step one.* Usually a layout bug — packed nibbles in the wrong order, a transposed scale tensor, a group axis along N instead of K.

**B. Silent quality failures — the dangerous class.**
- *Degrades with context length.* Suspect KV quantization first, calibration sequence length second. Detect with a length-stratified eval, not an aggregate score.
- *Degrades in one language.* Calibration composition. Detect with per-language slices.
- *Degrades on math/reasoning only.* Error amplification through the CoT chain. Detect with a reasoning slice; fix by raising bit-width or protecting the LM head.
- *Structured output / tool calls fail intermittently.* Low-margin argmax on rare tokens. Detect with schema-validity rate as a first-class metric. Fix with constrained decoding.
- *Refusal behaviour shifted.* Alignment is a small perturbation and compression is a comparable one. Detect with a safety slice in both directions (over-refusal and under-refusal).
- *A specific customer got worse.* On a multi-tenant fleet, the calibration set matched the aggregate and not that tenant. Detect with per-tenant quality metrics; this is why tenant-level dashboards matter.

**C. Performance failures — it works but it's slower or costlier.**
- *Throughput dropped.* Compute-bound regime, slow fallback kernel, or lost fusion. Detect by profiling the GEMM/attention split and checking the engine's kernel-selection log.
- *p99 got much worse while p50 improved.* Ragged batching from a sparse or LoRA path, or KV pressure causing preemption. Detect with per-percentile latency and a preemption counter.
- *Cold start got worse.* Some quantized checkpoints require an unpacking or kernel-JIT step at load. Detect by measuring time-to-first-healthy-request, not just steady-state.

**D. Process failures — how you got here.**
- *Eval was the calibration set.* Score is meaningless.
- *Eval was aggregate only.* Slice regressions invisible.
- *Compared against a published benchmark instead of your own fp16 baseline on the same stack.* You are now measuring the delta between two unrelated things.
- *No canary.* Offline evals don't see the tail of real traffic.

**The on-call procedure, in one line:** confirm the fp16 baseline on the same stack and the same inputs first (rules out one third of reports), then bisect by axis — set KV back to fp16, then weights back to fp16 — and whichever restore fixes it names the culprit in two deploys.

### Last one: I hand you a model and a serving budget and say "make it cheaper." What do you do in the first week, and what would you push back on?

I'd push back on the framing first, and then execute in a specific order, because "make it cheaper" without a constraint is how teams optimise the wrong axis for a sprint.

**Day 1 — refuse to compress anything, and measure.** Get four numbers: the prefill:decode token ratio on real traffic, the running batch size distribution per replica, the KV-cache utilisation, and the context-length distribution at p50/p95/p99. Then get the cost breakdown: what fraction of spend is prefill, decode, embeddings, reranking, and idle capacity. In my experience the single largest line item is frequently **idle capacity from over-provisioned replicas or a bad autoscaling signal**, and no amount of quantization fixes that. I've seen a 35% cost reduction from fixing the autoscaler before anyone touched a bit-width.

**Day 1 also — establish the eval baseline.** If there isn't a sliced eval with a long-context slice, a non-English slice, a reasoning slice and a structured-output validity rate, building it *is* the first week's work, and I would say that out loud to the person asking. You cannot trade quality for cost without a quality number; you can only trade quality for cost *unknowingly*.

**Days 2–3 — the free wins, before compression.** Prefix caching on system prompts and repeated documents. Chunked prefill so long prefills don't stall decode. Correct max-batch and KV-utilisation-based autoscaling. Provider batch tier for anything asynchronous. Shorter retrieved context via a better reranker. Every one of these is cheaper and lower-risk than quantization and several are larger.

**Days 3–4 — quantize along the axis the measurement pointed at.** Decode-bound at low batch → W4A16 (AWQ or GPTQ, group 128, calibrated on 256 sequences of production traffic with the real chat template). Prefill-bound or high batch → fp8 W8A8. Long context or KV-capacity-capped → fp8 KV cache, which is often the biggest single win and the one people do last.

**Day 5 — eval, sliced, paired against fp16 on the same stack. Then canary at 5%.**

**What I'd push back on, explicitly:**
- **"Let's prune 50% of the weights."** Unstructured sparsity is not a systems win. Ask what shape the zeros are and which kernel exploits them.
- **"Let's go to 4-bit everywhere including the KV."** Not without a needle-in-a-haystack eval at max context in every language served.
- **"Let's fine-tune a small model to replace it."** Sometimes correct and often the biggest win available — but it's a multi-week project with a real eval and data pipeline, so it belongs in the plan as a project, not as this week's fix.
- **"The benchmark says only 1% loss."** That benchmark is not our traffic, and the 1% is an average over a distribution where our failure modes have almost no weight.

**🏋 Drill (60 minutes, unaided, whiteboard):** given a service description — model size, GPU type and count, traffic in requests/day, p50/p95 context lengths, output-length distribution, SLO — produce: the memory budget, the bound (capacity / bandwidth / compute / latency-floor), the chosen compression stack with justification, the expected fleet reduction in GPUs and dollars/month with arithmetic, the eval slices you'd gate on, and the one thing you'd push back on. Pass criterion: every number derived on the board from the four constants (bytes/param, KV bytes/token formula, HBM bandwidth, achieved TFLOPS), no figure quoted from memory without a derivation, and a stated bound that follows from the numbers rather than from the method you wanted to use.
