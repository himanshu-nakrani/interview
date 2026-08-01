### Explain lost-in-the-middle. Is it a positional-encoding bug, and what do you actually do about it?

The finding: give a model `k` retrieved documents and ask a question whose answer is in exactly one of them, then sweep which position holds it. Accuracy is high when the gold document is first, high when it is last, and measurably lower in the middle — a U-shaped curve. On some settings the middle-position accuracy fell below the model's closed-book accuracy with *no* documents at all, which is the result that made people pay attention.

It is not a positional-encoding bug, and saying so is the mark of a shallow answer. It is a **data and objective artifact** with a couple of reinforcing mechanisms. Pretraining text puts the thesis at the top and the conclusion at the bottom; instruction data puts the instruction first and the query last; RLHF rewards responses that track the beginning and end of the prompt. The model learns a prior that the ends matter. On the mechanism side, the first token is an attention sink that hoards probability mass, and recency is structurally favoured by causal attention and by any positional scheme with a recency prior. Middle positions have neither advantage.

The important corollary: **it does not go away with longer context windows.** A 1M-token model has the same U-shape, just stretched. Nor does it go away with better positional scaling — YaRN fixes out-of-distribution angles, not attentional priors.

What I actually do, in order of leverage:

1. **Order retrieved documents by relevance with the strongest at the extremes.** Do not concatenate in retrieval-score order top-to-bottom. Interleave so rank 1 is first, rank 2 is last, rank 3 is second, rank 4 is second-to-last, and so on. This is a five-line change to the context assembler and it is typically worth several points.
2. **Retrieve fewer documents.** The U-shape gets worse with `k`. Going from `k = 20` to `k = 8` with a reranker usually beats any positional trick, because it removes the middle entirely.
3. **Repeat the question after the documents** as well as before. Cheap, and it converts "the question is in a disadvantaged position relative to the evidence" into "the question is adjacent to the evidence."
4. **Measure it on your own corpus** — sweep gold-document position and plot accuracy. This is a 40-line eval and every team that has long-context RAG should have it in CI. If someone claims their reranker improved things, this curve tells you whether it improved retrieval or just moved the gold doc to position 1.

**📄 Paper:** Liu et al. (2023/2024, TACL), *Lost in the Middle: How Language Models Use Long Contexts* — established the U-shaped positional-accuracy curve for multi-document QA and key-value retrieval, and showed it persists across model families and sizes.

**⚠ Trap:** "we fixed lost-in-the-middle by upgrading to a 200k model." You did not. You moved the middle. The only structural fixes are putting less in the context and controlling where you put it.

### Give me the advertised-versus-usable context framing, with numbers.

**📐 Numbers you must know — always state context as two numbers.** Advertised context is the value in `max_position_embeddings` or the API docs; it is a statement about what the code will accept without erroring. Usable context is the length at which the model still hits your accuracy bar on *your* task, and it is routinely a fraction of the advertised figure. The rule I state in every design doc: **"advertised 128k, usable X on our corpus at Y% on multi-needle,"** where X and Y are measured, not quoted.

The evidence base for the gap is RULER, which built a synthetic battery of multi-needle, multi-hop, variable-tracking and aggregation tasks and defined "effective context length" as the longest length at which a model still beats a fixed baseline. Its headline finding was that most evaluated models degrade well before their advertised length, with a substantial number of effective lengths landing far below the claim — often at a half or less. Chroma's later "context rot" work made the same point on a broader task set: performance declines continuously with input length even on tasks that are trivially easy at short length, and the decline is worse when the distractors are semantically similar to the target. (**📅 Volatile — see §5:** specific model numbers move every release; the *methodology* is what is durable, not the leaderboard.)

**📄 Paper:** Hsieh et al. (2024, NVIDIA), *RULER: What's the Real Context Size of Your Long-Context Language Models?* — replaced single-needle NIAH with a synthetic battery spanning multi-needle retrieval, multi-hop tracing, aggregation and long-context QA, and defined effective context length against a baseline threshold.

**💰 Math — why the gap costs money, not just accuracy.** Suppose you sized a summarization pipeline for 128k-token inputs and the usable length turns out to be 40k. You either accept the quality loss or you chunk into four 32k passes plus a synthesis pass — five calls instead of one. At $3/Mtok input, one 128k call costs 128,000 × 3/1e6 = **$0.384**. Four 32k passes plus a 10k synthesis pass costs (4 × 32,000 + 10,000) × 3/1e6 = **$0.414**, plus four extra round trips of latency. Roughly cost-neutral per call, but at 50,000 calls/day you have gone from 50k requests to 250k requests against your provider rate limits, which is the constraint that actually bites. **Measuring usable context is capacity planning, not just eval hygiene.**

**🗣 Say this in the room:** "I never quote a single context number. Advertised is what the API accepts; usable is where multi-needle accuracy on our corpus is still above bar. In my experience those differ by a factor of two to four, and the design has to be sized on the second number."

### What are attention sinks, why does StreamingLLM keep four tokens, and what does it actually give you?

The mental model: softmax is a *forced* probability distribution — it must sum to 1 across the visible keys. But a head often has nothing it wants to attend to at a given position; its job might be "fire only when the previous token is an open paren." It has no way to output "attend to nothing." So it learns to dump its mass somewhere harmless. Token 0 is the ideal dumping ground because under causal masking it is the only token *every* query can see, at every position, in every sequence. So token 0 accumulates enormous attention mass across many heads and layers, while contributing essentially nothing semantically. It is a no-op sink, an escape valve for the softmax normalizer.

The discovery came from trying to build a rolling-window cache for infinite streaming. Naively, you keep the most recent `W` tokens and evict the oldest. The moment you evict token 0, perplexity explodes — not degrades, explodes. Every head that was parking its mass on token 0 now has to redistribute that mass across real tokens, which corrupts every attention pattern simultaneously.

**StreamingLLM's fix is embarrassingly simple and that is why it is a great result: keep the first 4 tokens permanently, plus a rolling window of recent tokens.** Four is empirical — one is not enough because different heads sink to slightly different early tokens, and beyond four there is no further gain. The second, less-quoted half of the design: positions are assigned **relative to the cache**, not to the original sequence. If your cache holds original positions {0,1,2,3} plus {50000...52047}, you encode them as positions 0,1,2,3,4,5,...,2051. This keeps rotation angles inside the trained range forever, which is what makes it truly unbounded.

**⚠ Trap — and this is the one interviewers use to separate people who read the abstract from people who understood it.** StreamingLLM gives you **infinite fluency, not infinite memory.** Evicted tokens are gone. The model will chat coherently for four million tokens and cannot tell you anything about token 1,000,000. It is the right tool for an always-on assistant where only recent context matters and the alternative is a hard session reset; it is the wrong tool for anything requiring recall over the full stream. If someone proposes StreamingLLM as a long-context solution, that is the pushback.

A useful downstream consequence: attention sinks are why some models now train with an explicit learned sink token or a per-head additive bias in the softmax denominator (a "softmax + 1" style off-ramp). Once you know the model needs a place to dump mass, giving it a dedicated one is strictly better than letting it hijack token 0 — and it removes a class of quantization and eviction hazards, since the sink is no longer a real token whose value vector is enormous and outlier-heavy.

### Test-time training and Titans-style memory keep coming up. What's the actual claim?

The claim is that the hidden state of a recurrent model does not have to be a *vector* updated by a linear rule — it can be a *small neural network* updated by **gradient descent on a self-supervised objective, at inference time**. That is the whole idea, and it is genuinely interesting because it attacks the exact bottleneck we identified with SSMs: a fixed-size linear state has limited capacity and suffers destructive interference, whereas a small MLP trained online can, in principle, store associations more efficiently and compress the history nonlinearly.

**TTT layers** (Sun et al., 2024, *Learning to (Learn at Test Time)*) make this concrete: the hidden state is the weights `W` of a small model; the update rule for each incoming token is one gradient step on a reconstruction loss for that token; the readout is a forward pass of that model on the query. It is still `O(1)` per token in sequence length, still linear-time overall, and it is still a "compress the past into fixed capacity" architecture — but with a much better compressor.

**Titans** (Behrouz, Zhong & Mirrokni, Google Research) adds an explicit long-term neural memory module governed by a *surprise* signal: how much a token updates the memory is proportional to the gradient magnitude of the associative loss — i.e. how badly the memory currently predicts it — with momentum so a surprising region keeps writing for a while, and a decay term acting as learned forgetting. The paper presents several ways to wire it in (memory as extra context, as a gating branch, or as its own layer) alongside short-term attention. Read it as: attention handles precise recent recall, the neural memory handles compressed long-range recall, surprise decides what is worth writing.

How I would talk about this in a room: **this is an active research direction, not a production choice.** The honest position is that the fixed-state family is trying to buy back read-time flexibility by making the write rule adaptive, and it is not yet settled whether that closes the exact-recall gap or merely narrows it. I would say exactly that, then give the decision rule: if your product needs verbatim recall of arbitrary spans, you need attention or an external retrieval index today, and no amount of clever state compression changes the counting argument. If your product needs *gist* over an unbounded stream, this family is where I would be watching.

**⚠ Trap:** describing TTT as "the model fine-tunes itself on your data." It is not learning across requests, it is not persisting anything, and nothing about it is a substitute for retrieval or fine-tuning. The gradient steps update a per-sequence hidden state that is discarded when the sequence ends. Getting this wrong sounds like a serious misunderstanding of the training/inference boundary.

### Why is needle-in-a-haystack a weak proxy? Be specific about what it fails to measure.

NIAH plants one sentence — classically something like "the best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day" — inside a large volume of unrelated filler, then asks for it back. It is a *lexical outlier detection* task dressed up as a retrieval task, and it is weak for five specific reasons that I would enumerate:

1. **The needle is semantically alien to the haystack.** Paul Graham essays with one sentence about sandwiches is a distribution-shift detector. Real retrieval is finding the *relevant* passage among *plausible* ones. The moment you make the distractors topically similar, scores fall dramatically — this is the single biggest gap, and it is why "context rot" work emphasizes distractor similarity as a controlled variable.
2. **One hop, one fact.** No composition. Real questions need two facts 60k tokens apart, combined.
3. **No aggregation.** "How many times does X appear" or "what is the total of all invoices" requires attending to many positions at once, and models that ace NIAH fail these badly.
4. **No negative case.** NIAH never tests whether the model correctly says "that information is not in this document." A model that always produces a confident guess scores 100% on NIAH and is unusable in production, where the false-positive rate is the metric that gets you sued.
5. **It has been optimized against.** NIAH is public, well-known, and has been in the loop long enough that models are effectively tuned to pass it. A saturated public benchmark stops carrying signal.

**🗣 Say this in the room:** "Single-needle NIAH is a smoke test — I treat a non-perfect score as a build failure and a perfect score as carrying no information. It's lexical outlier detection with an alien needle, one hop, no aggregation, no abstention case, and it's been optimized against. What I actually run is multi-needle with topically-similar distractors, multi-hop chains, and a negative set."

### Design a real long-context evaluation on our own corpus. We have a few million enterprise documents. Go.

I will lay this out as the artifact I would actually build, because the deliverable is a harness, not a number.

**Step 0 — define the decision the eval informs.** Ours is: "what context length do we serve at, and does model B beat model A for this product?" Every design choice below follows from that, and it rules out things like "run LongBench," which answers a different question.

**Step 1 — build the item bank from real documents, with programmatic ground truth.** Hand-authoring 500 long-context questions is a month of work and unmaintainable. Instead, generate items with *verifiable* answers:

- **Multi-fact extraction.** Pick a document set, use a cheap model plus a regex/parser pass to extract structured facts with known character offsets (dates, party names, amounts, section numbers). Assemble a context containing `k` documents, ask for a fact, and grade by exact match against the extracted value. Ground truth is derived, not judged.
- **Multi-hop chains.** Item: "What is the payment term in the contract signed by the party named in Exhibit B of document 14?" Construct it by walking a real cross-reference graph you built from the corpus, so the chain is genuine.
- **Aggregation.** "Sum all line items tagged `travel` across these 40 expense reports." Ground truth from the parser. This is the one that separates models most sharply.
- **Abstention / negatives.** ~20% of items must have no answer in the context. Grade false-positive rate separately and weight it heavily; in enterprise settings a confident wrong citation is worse than a refusal.
- **Citation grounding.** Require the model to return the document ID and span it used. Grade the citation independently of the answer — answer-right-citation-wrong is a distinct and important failure class.

**Step 2 — make position and length controlled variables, not incidental.** Every item gets rendered at multiple context lengths (8k, 16k, 32k, 64k, 128k) by padding with additional real corpus documents, and at multiple gold positions (first decile, middle, last decile). The output is not a scalar; it is a **surface over (length × position)**. That surface is what you take to a design review, and it is what tells you your usable context.

**Step 3 — control distractor hardness explicitly.** Build two padding pools: random corpus documents (easy) and top-`k` retrieval neighbours of the gold document (hard). Report both. The gap between them is the single most predictive number for real-world behavior, and it is invisible in NIAH.

**Step 4 — grading.** Prefer exact match and parser-verified equality wherever the item design permits, because a judge model at 128k context is expensive and is itself subject to everything in this section. Where you need a judge (free-form summarization), give it *only* the gold span and the answer, never the full context — a short-context judging task — and report judge-human agreement on a 100-item sample (Cohen's κ) before you trust any number it produces.

**Step 5 — statistics.** Report per-cell accuracy with a bootstrap CI. With 500 items across 5 lengths × 3 positions you have ~33 items per cell, and the 95% CI on a proportion near 0.8 with n=33 is roughly ±0.14 — far too wide to call a 3-point regression. Either report at the aggregate level with paired bootstrap over items, or budget for enough items per cell. **Say this number out loud in the design review**; the most common eval failure at this scale is a beautifully constructed surface plot whose every cell is noise.

**Step 6 — CI integration.** A fast tier (100 items, one length, one position) on every prompt or model change, and the full surface nightly and before any model swap. Store per-item results so you can diff *which* items regressed, not just the aggregate — the aggregate hides the case where you gained 5 points on extraction and lost 20 on abstention.

**🔍 Failure taxonomy — long-context evals that lie to you.** (1) Padding drawn from a different distribution than production → the model is doing outlier detection, scores are inflated. (2) Gold document always placed uniformly at random → you never measure the U-shape, and a reranker change that only helps position-1 looks like a general win. (3) No abstention items → you cannot see the model trading a higher false-positive rate for a higher hit rate. (4) Judge run over the full context → judge cost dominates and judge accuracy degrades with the same length effects you are trying to measure. (5) Items generated by the same model family you are evaluating → self-preference contamination.

### Long context or RAG? Give me the decision rule and the arithmetic.

The framing I use: **long context and retrieval are not competitors, they are different points on the same cost-latency-recall curve, and the right question is what fraction of the corpus you must consider per query.**

**The arithmetic first.** Suppose each query needs answers from a 150,000-token document set, at $3.00/Mtok input (**📅 Volatile — see §5**), 100,000 queries/day.

- **Stuff everything, uncached:** 150,000 × $3.00/1e6 = **$0.45/query** → $45,000/day → **$1.35M/month.**
- **Stuff everything, with prefix caching** and a 90% cached-input discount, assuming the 150k corpus is stable across queries and only the question varies: ≈ **$0.045/query** → **$135k/month.** A 10× cut for a prompt-ordering change.
- **RAG, retrieving 8,000 tokens:** 8,000 × $3.00/1e6 = **$0.024/query** → $2,400/day → **$72k/month**, plus a one-time index cost (embedding 500M corpus tokens at, say, $0.02/Mtok = $10) and a vector-search cost measured in single-digit milliseconds.

So: $1.35M → $135k → $72k. Note that the cache brings stuffing to within 2× of RAG on *cost* — which is the result that surprises people and which changes the decision. When the corpus is genuinely stable and shared, caching does most of RAG's work on the cost axis.

**Latency is where it stops being close.** Prefill is at best linear in input length and has a superlinear attention term. If your provider prefills at an effective 10,000 tokens/second for this model (measure it — this varies by an order of magnitude across models and providers), 150k tokens is **15 seconds of TTFT** versus 0.8 seconds for 8k. A cache hit removes most of that, but a cache *miss* — a new tenant, a changed document, an evicted entry — puts a 15-second tail latency into a product whose p50 is under a second. **The tail is the problem.** I have seen teams ship cached long-context stuffing, see a beautiful p50, and discover a 12-second p99 driven entirely by cache misses.

**The decision rule I actually use:**

- Corpus per query **under ~20k tokens and stable** → stuff it, use prefix caching, do not build retrieval. Retrieval infrastructure has real ongoing cost (chunking strategy, embedding model drift, reindexing, alias swaps) and you should not pay it for a 20k corpus.
- Corpus **large but a small, identifiable subset is relevant** → retrieval. This is most enterprise search, most support, most docs QA.
- Corpus **large and the task needs global structure** — "summarize the themes across all 400 tickets," "find every clause that conflicts with clause 12" — → retrieval alone fails, because there is no small relevant subset. Use map-reduce over chunks, or an agentic loop, or hierarchical summarization. This is the case people mishandle most often by throwing more context at it.
- **Latency budget under ~2 seconds TTFT** → hard constraint against large uncached prefills, full stop, regardless of the cost analysis.

And the answer that gets you the "senior" label: **the two compose.** Retrieve aggressively to get from 150k to 20k, then use long context to be sloppy about precision within that 20k — retrieve 15 chunks instead of 3 and let the model sort it out. Long context's real product value is that it makes your retriever's recall@k matter more than its precision@1, which is a much easier engineering target.

### A long-context feature was fine, we upgraded the model, and quality dropped with no errors. Debug it.

I would work this as a narrowing procedure, cheapest and most-likely first, because "model got worse" is almost never the actual cause.

**1. Verify the token count reaching the model.** Different model, different tokenizer. A corpus that was 118k tokens under the old tokenizer can be 131k under the new one, and if your framework silently truncates to the context limit you are now dropping the end of every document. Log the actual prompt token count and the truncation flag on every request; if you cannot, that is the first thing to fix. **Silent truncation is the single most common cause of this exact symptom** and it costs ten minutes to rule out.

**2. Check the rotary and attention config on the new checkpoint.** Diff `rope_theta`, `rope_scaling`, `sliding_window`, and `max_position_embeddings` between the two configs. If the new model uses a scaling type your serving stack does not implement, it will fall back silently. Generate `inv_freq` from the running framework and diff it against a reference computation.

**3. Run the per-position NLL curve on both models.** Ten minutes of GPU time and it localizes the problem to a length. A knee at a specific length points at config; a uniform offset points at a genuine capability difference; a curve that looks identical points away from the model entirely and toward your prompt.

**4. Sweep gold-document position on your eval set.** If the new model's U-shape is deeper, your context assembly order (which was tuned for the old model, whether you realized it or not) needs retuning. This is real and underappreciated: **prompt ordering is model-specific tuning that silently transfers as a regression.**

**5. Check for prompt-format drift.** New model family, new chat template, new system-role handling, new tool-call serialization. If you built the prompt string yourself rather than through the tokenizer's chat template, the new model may be seeing a subtly malformed conversation — which degrades much more at long context, because the model has more to be confused by.

**6. Check KV-cache dtype and quantization.** If you enabled fp8 KV cache and the new model has different activation outlier structure, quantization error accumulates over length. Symptom: fine at 8k, degrading progressively past 32k. Test by running bf16 KV at the same length.

**7. Only now consider that the model is genuinely worse at this task.** Confirm with a controlled A/B on the item bank at matched lengths and positions, with paired bootstrap CIs — and remember that "newer" is not "better on your task," particularly for models tuned toward reasoning, which frequently trade long-context extraction accuracy for chain-of-thought quality.

**🗣 Say this in the room:** "Before I conclude the model regressed, I rule out three things that produce this exact signature: silent truncation from a different tokenizer, a `rope_scaling` type the serving stack doesn't implement and is silently ignoring, and a chat-template mismatch. All three are free to check and all three are more likely than a capability regression."

### It works fine at 32k and produces garbage at 100k. No exceptions, no warnings. Give me your ladder.

**🔍 Failure taxonomy — long-context degradation, in the order I check.**

**Rung 1 — is the input intact?** Print the actual token IDs at the boundary. Look for (a) truncation at exactly the old context limit, (b) truncation from the *wrong end* — many frameworks truncate the head, which removes your system prompt, (c) a max-length parameter capping input rather than output. This rung catches maybe 40% of real cases.

**Rung 2 — is positional scaling actually applied?** Compute `inv_freq` from the live model and compare against the reference for the declared `rope_scaling`. Confirm the serving engine's version supports the declared `rope_type`. This is the "silently fell back to plain RoPE" case and its signature is a knee in per-position NLL right at `original_max_position_embeddings`.

**Rung 3 — is there a window you forgot about?** Check `sliding_window` in the config and whether the engine is enforcing it. A model with a 4k window will produce locally-fluent, globally-incoherent output at 100k and no error. Also check any `--max-model-len` or attention-backend flag that silently switches implementation past a length threshold.

**Rung 4 — precision.** Softmax over 100k keys with fp16 accumulation is a numerically different animal from 32k. Confirm bf16 (or fp32 softmax accumulation) rather than fp16. Separately: if KV cache is fp8 or int4, disable it and retest — quantization error in K compounds with the number of keys attended.

**Rung 5 — cache-position correctness.** In hand-rolled or heavily-customized inference paths, check that the absolute position passed to RoPE during decode equals `past_kv_len`, and that chunked prefill increments positions across chunk boundaries. The classic bug is that chunk 2 restarts positions at 0. Signature: fine for the first chunk length, then degrading. Test by comparing a single-shot prefill against a chunked prefill of the same prompt — logits should match to within numerical tolerance, and if they do not, you have found it.

**Rung 6 — attention-sink eviction.** If any cache-management feature is enabled (rolling window, H2O-style eviction, StreamingLLM mode), confirm the first few tokens are pinned. Evicting token 0 produces exactly this "sudden garbage past a length" signature.

**Rung 7 — genuine capability.** Only after all six. Run the multi-needle surface and locate the length where accuracy crosses your bar. If rungs 1–6 are clean and accuracy declines smoothly with length rather than falling off a cliff, this is context rot and the fix is architectural (retrieve less, chunk, or change models), not a bug fix.

The general principle: **cliff-edge failures are configuration; smooth declines are capability.** That single heuristic will route you to the right half of the ladder in seconds and it is worth saying out loud in an interview.

### How do you schedule a mixed workload of 2k and 200k requests on the same fleet?

This is the question where your backend instincts transfer almost perfectly, and you should say so. It is head-of-line blocking with a 100× variance in job size, and the standard remedies apply with LLM-specific twists.

**The core problem:** a 200k-token prefill occupies the GPU for seconds. Every 2k request queued behind it eats that as pure TTFT. With a naive FCFS scheduler, your p99 TTFT is set entirely by the largest prefill in flight, and adding replicas does not fix it because the long requests are spread across all of them.

**Fix one: chunked prefill.** Split a long prefill into fixed-size chunks (say 2k tokens) and interleave chunks with decode steps for other sequences in the same batch. The 200k prefill becomes 100 chunks that yield the GPU between them, so short requests interleave. This converts a 5-second head-of-line block into ~50ms of added latency per short request. **This is the single highest-leverage setting in a mixed-length deployment** and every major engine supports it. The cost is a small throughput reduction for the long request, because you lose some batching efficiency at chunk boundaries.

**Fix two: separate the fleets.** Route by input length at the gateway — short requests to a latency-tier deployment with small `max_model_len` and aggressive prefix caching, long requests to a throughput-tier deployment with large KV allocation and possibly a different parallelism strategy. This is the same argument as separating OLTP from analytics, and it has the same advantage: each pool's tuning is coherent instead of a compromise. The disadvantage is capacity fragmentation — you now have two pools that cannot absorb each other's spikes.

**Fix three: admission control on KV budget, not on request count.** Your concurrency limit must be expressed in KV bytes. A limit of "64 concurrent requests" is meaningless when one request is 16 GiB of KV and another is 256 MiB. Compute `bytes_per_token × requested_max_tokens` at admission and reject or queue when the projected total exceeds the pool. Without this you get preemption storms: the engine admits too many long sequences, runs out of KV, and starts evicting and recomputing, which burns compute on work it already did. **The metric to alert on is KV-cache utilization, not queue depth** — it is the LLM analogue of watching connection-pool saturation rather than request count.

**Fix four: prioritize by remaining work, not arrival time.** Once you have chunked prefill, you can implement shortest-remaining-first among queued prefills, which minimizes mean TTFT. Guard it with aging so long requests do not starve — the same fix you would apply to any priority queue.

**💰 Math:** the case for chunked prefill. Assume 95% short requests (2k in) and 5% long (200k in), 1,000 req/min, and a prefill rate of 20,000 tok/s per replica. Long requests consume 200,000/20,000 = 10 GPU-seconds each; at 50/min that is 500 GPU-seconds/min of prefill from long requests alone, i.e. you need at least 9 replicas just for their prefill. Under FCFS, a short request arriving during a long prefill waits on average 5 s. At 950 short req/min, roughly half of them land during some long prefill, so **p50 TTFT for short requests goes from ~0.1 s to multiple seconds** — a 20×+ regression caused entirely by 5% of traffic. Chunked prefill at 2k chunks reduces the worst-case wait to one chunk time (~0.1 s), restoring short-request p99 at a few percent throughput cost on the long ones.

### Multi-turn chat over a 100k-token document. Walk me through the caching economics turn by turn.

Set it up concretely: a 100,000-token document, 10 conversational turns, each user message ~100 tokens, each assistant reply ~300 tokens. Prices $3/Mtok input, $15/Mtok output, cached input at 10% of the input rate (**📅 Volatile — see §5** — cache-write surcharges, TTLs and discounts differ by provider and are the most commonly-misquoted numbers in this whole area).

**Naive, no caching.** Every turn resends the document plus the full conversation so far. Input tokens at turn `n` ≈ 100,000 + `n` × 400. Summing over 10 turns: 10 × 100,000 + 400 × (1+2+...+10) = 1,000,000 + 22,000 = **1,022,000 input tokens** → 1.022 × $3 = **$3.07 in input** for one conversation. Output: 10 × 300 = 3,000 → $0.045. Total **≈ $3.11 per conversation.** At 10,000 conversations/day: **$31,100/day = $933k/month.**

**With prefix caching on the document.** Turn 1 pays full price for the document (plus, on some providers, a cache-write surcharge — check it, it is often 1.25× the base input rate). Turns 2–10 hit the cached 100k prefix at 10%. Input cost ≈ 100,000 × $3/1e6 + 9 × 100,000 × $0.30/1e6 + 22,000 × $3/1e6 = $0.30 + $0.27 + $0.066 = **$0.636**, plus $0.045 output = **≈ $0.68 per conversation.** At 10,000/day: **$6,800/day = $204k/month.** A **4.6× reduction**, and the latency win is larger than the cost win because the 100k prefill is skipped on 9 of 10 turns.

**Three things that silently destroy this**, and they are the actual content of the answer:

1. **A timestamp or session ID in the system prompt.** If anything before the document varies per conversation, the cache key differs and the hit rate is zero. This is the most common own-goal in production LLM systems. The rule: **the prompt must be assembled in strictly decreasing order of stability**, and anything per-request goes at the end. No exceptions, enforce it in the context-builder API by making the stable part a separate typed argument.
2. **Cache TTL versus think time.** Provider prefix caches typically expire in minutes. A user who reads for ten minutes and then asks turn 3 gets a cold cache and pays full price *and* full latency. Model this: if 30% of turns fall outside the TTL, your effective saving drops from 4.6× to roughly 2.5×, and your p95 TTFT gets a multi-second tail. Some providers offer extended TTLs at a surcharge; the break-even is `P(expiry) × full_prefill_cost` versus the surcharge, and you should compute it from your own inter-turn time distribution rather than guessing.
3. **Growing conversation history invalidating nothing, but trimming it invalidating everything.** If you implement history truncation by dropping the *oldest* turns, you change the prefix and blow the cache on every subsequent turn. Truncate by summarizing the middle and keeping the prefix byte-identical, or accept the miss deliberately at a known point rather than on every turn.

**🗣 Say this in the room:** "For a 100k-doc, 10-turn conversation, caching takes it from about $3.11 to $0.68 — 4.6×, and a bigger win on latency since you skip a 100k prefill nine times out of ten. The two things that kill it are a per-request variable anywhere in the prefix and TTL expiry across user think time, so I'd measure the inter-turn time distribution before promising the saving."

### 🏋 Drill: RoPE and context extension, unaided, timed.

**Setup:** 25 minutes, no references, no autocomplete, blank file. This is the exact shape of a Cursor-, Perplexity- or Anthropic-style implementation round on this material.

**Task A (10 min).** Implement, in PyTorch:
- `rope_inv_freq(d_head, base)` returning `[d_head//2]`.
- `build_cos_sin(positions, d_head, base)` returning `cos, sin` of shape `[T, d_head]` in the half-split layout.
- `apply_rope(x, cos, sin)` for `x` of shape `[B, H, T, d_head]`.

**Task B (8 min).** Implement `llama3_inv_freq(...)` implementing the three-zone NTK-by-parts scaling: untouched above `high_freq_factor` rotations, fully divided by `factor` below `low_freq_factor` rotations, linear ramp between.

**Task C (7 min).** Write two tests and make them pass:
1. **Shift invariance.** For random `q, k` and any integer shift `c`, `⟨rope(q, m), rope(k, n)⟩ == ⟨rope(q, m+c), rope(k, n+c)⟩` to 1e-4 in fp32.
2. **Scaling boundaries.** For `d_head=128, base=500000, factor=8, low=1, high=4, orig=8192`, assert that the number of untouched frequency indices is 29 (indices 0–28), the number fully divided by 8 is 28 (indices 36–63), and the remaining 7 are strictly between.

**Pass criteria:** both tests green inside 25 minutes with no reference material. Then answer these three out loud, in under 60 seconds each, without notes:
- *What changes if you cache K before rotation instead of after?* (You must re-rotate the whole cache every step: `O(T)` work per decode token instead of `O(1)`, and prefix-cache blocks stop being position-bound — a real trade some designs make deliberately.)
- *Your model's `d_head` is 64 instead of 128. Do the boundary indices move?* (Yes — the boundaries are at fixed *wavelengths*, and with 32 pairs instead of 64 the same wavelengths land at proportionally scaled indices; the fraction of the spectrum in each zone is unchanged.)
- *You raise `base` from 500,000 to 1,000,000 on a deployed model. What is the first thing you check?* (Prefix cache namespacing — every cached K is now invalid; then a per-position NLL curve at both short and long lengths.)

If you cannot get Task A out in 10 minutes cold, that is the highest-value thing to drill in this entire section, because it is the single most likely thing to be asked at a whiteboard from this material.

### 🏋 Drill: the context-extension design defense.

**Setup:** 20 minutes, whiteboard or blank page, no references. This is the design-round shape.

**The prompt you are given:** "We have an internal 8B model, 32 layers, `d_model` 4096, 8 KV heads, `d_head` 128, trained at 8k context with RoPE base 500,000. Product wants 128k for contract analysis. We have 64 H100s for two weeks and one ML engineer. Design it."

**What a passing answer contains, and grade yourself against this list:**

1. **The KV arithmetic, unprompted.** `2 × 32 × 8 × 128 × 2 = 131,072 B = 128 KiB/token`; at 131,072 tokens that is **16 GiB per sequence**. With ~64 GB free after weights on an 80 GB card you get **4 concurrent 128k sessions per GPU**. State this before anything else, because it determines whether the product is even viable and it is the number nobody volunteers.
2. **The extension method and why.** `rope_scaling` of the llama3/YaRN NTK-by-parts family with `factor` around 16 (8k → 128k), applied *before* the continued-pretraining stage, not after. Say explicitly why not naive PI (uniform frequency squeeze costs short-context resolution) and why not just raising the base (needs training either way, and gives no per-frequency control).
3. **The length curriculum.** 8k → 16k → 32k → 64k → 128k, gated on short-context non-regression at each step.
4. **The data mix with the anti-forgetting term.** ~30–40% naturally long (contracts, repos, books), ~25% synthetic long-dependency (planted multi-hop facts at controlled positions), ~10% masked document packing, **~30% replay of the original pretraining mix**. If you omit replay, you fail.
5. **The compute estimate.** Per-token training cost at 128k = `6N + 6·n_layers·d_model·T` = 4.8e10 + 1.03e11 = **1.51e11 FLOP**. 64 H100s at 40% MFU = 64 × 3.96e14 = 2.53e16 FLOP/s. Two weeks = 1.21e6 s → **3.07e22 FLOP available → ~200B tokens** at 128k, or considerably more under the curriculum since early stages are cheaper. Conclusion: **compute is not the constraint; data curation and eval are.** Saying that is the senior move.
6. **The eval plan, before the training plan is finished.** Per-position NLL, RULER-style multi-needle and multi-hop, a domain item bank on real contracts with position and distractor-hardness as controlled variables, abstention items, and a short-context regression table. Name the acceptance gate: short-context within noise, multi-needle above bar at 128k with hard distractors.
7. **The serving consequence.** 4 sessions/GPU at 128k is probably not economic, so propose the mitigations: retrieve down to 32k where possible (4× more concurrency), fp8 KV (2×), and an interleaved local:global variant if a retrain is on the table anyway.
8. **The one-sentence risk you name unprompted:** "the most likely way this fails is that we ship a 128k model whose usable context on contracts is 40k, so the eval surface over length × position is the deliverable, not the checkpoint."

**Pass criterion:** you produce items 1, 4, 5 and 6 without being prompted for them. Producing the method (item 2) alone is a mid-level answer; the arithmetic and the eval plan are what get you the level.

### What's genuinely unsettled in long context right now, and what's your decision rule in the meantime?

I would name four open questions honestly rather than pretending the field has converged, because at this compensation level "it depends, and here is on what" is a stronger answer than a confident wrong consensus.

**One: whether fixed-state models can close the exact-recall gap.** The counting argument says no — you cannot losslessly recall an arbitrary item from 100k when your state holds 4k numbers. The counter-argument is that real tasks do not require lossless recall of arbitrary items, and that a sufficiently clever write rule (Mamba-2's larger state, TTT's gradient-based updates, Titans' surprise gating, delta-rule variants) recovers most of what matters. **My decision rule:** if the product requires verbatim reproduction of arbitrary spans, use attention or an external index today. If it requires gist over an unbounded stream, fixed-state is already the better engineering choice.

**Two: whether long context substitutes for retrieval or complements it.** Every context-length increase produces a round of "RAG is dead" posts, and the arithmetic above says otherwise on both cost and latency for large corpora. But the boundary genuinely moves. **My decision rule:** compute the tokens-per-query the task actually requires. Under ~20k and stable → stuff it and cache. Over that, or unstable → retrieve. Do not decide this by reading blog posts; decide it with the cost table for your traffic.

**Three: whether sparse attention becomes the default.** NSA and MoBA are the strongest evidence yet that learned, hardware-aligned sparsity works at pretraining scale, and the argument that 128k dense attention is mostly wasted computation is sound. But it requires pretraining from scratch with the mechanism in place, so adoption is gated on new model generations rather than on anyone's decision. **My decision rule:** treat it as a property of models you might adopt, not a lever you can pull.

**Four: whether "usable context" converges on "advertised context."** Each generation narrows the gap and none has closed it. **My decision rule, permanently:** measure it yourself, on your corpus, with hard distractors, and quote two numbers.

**🗣 Say this in the room:** "The three things I would not claim consensus on are whether fixed-state architectures close the recall gap, where the long-context-versus-retrieval boundary sits, and whether learned sparse attention becomes default. What I would claim is the measurement discipline: advertised context is a code property, usable context is an empirical one, and every design I sign off on states both numbers for our own corpus."
