### I keep hearing about a "constraint tax." What is it, and is it real?

The constraint tax is the empirical observation that turning on structured output makes the model *worse at the task* even while making it perfect at the format — and the reason it is worth a name is that it is invisible to every metric a team adds when they ship structured output. Your parse rate goes to 100%; your answer quality quietly drops; you have no instrument pointed at the second thing.

There are two distinct mechanisms and it is worth keeping them separate.

**Mechanism one: the format is off-distribution for the reasoning.** A model post-trained on chat data has learned to solve hard problems by writing prose — restating the question, working through cases, correcting itself. A JSON grammar forecloses all of that. The tokens the model would have spent reasoning are simply not reachable, and the model must jump straight to a value. This is not a subtle effect: it is removing chain-of-thought and then being surprised that performance drops on tasks that need chain-of-thought.

**Mechanism two: the constraint fights a trained channel.** Modern instruct models have specific token sequences they were trained to emit for particular behaviors — a tool-call opener, a thinking-block delimiter, a refusal preamble. A grammar that begins by forcing `{` masks all of them at position zero. The model's highest-probability action was to open a tool call; that action is now impossible, and it proceeds down its second-best path with a distribution that was never trained for this situation. **📅 Volatile:** the specific, widely-discussed 2026 version of this is that structured-output constraints measurably suppress *tool calling* in open-weight models — the model that would have called a tool instead fabricates an answer into the schema. Treat the magnitude as version- and model-specific and measure it yourself; treat the mechanism as durable, because it follows directly from masking a trained opener.

**⚠ Trap:** interpreting a quality drop after enabling structured outputs as "the constrained decoder is buggy." It almost never is. The mask is doing exactly what you asked. The bug is that you asked for something that removes the model's working memory.

**🗣 Say this in the room:** "The constraint tax is that format compliance and task accuracy move in opposite directions. Masking removes the model's ability to reason in prose and can mask the token that would have opened a tool call. It's not a decoder bug — it's the decoder doing precisely what you specified, and the fix is to constrain a later part of the output rather than all of it."

### Is the research settled on whether constrained decoding hurts reasoning?

No, and I would say so directly rather than pick a side, because this is a genuinely contested result and knowing the shape of the disagreement is more useful than knowing a number.

**📄 Paper:** Tam et al. (2024), *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models* (EMNLP industry track) — reported that format restrictions, and JSON-mode in particular, degraded performance on reasoning-heavy tasks relative to free-form generation, with the degradation growing as the restriction tightened. This is the paper everyone cites for "structured output makes models dumber."

The rebuttal came from the Outlines/dottxt authors, who argued the comparison was confounded: the constrained and unconstrained arms used different prompts, the unconstrained arm's answers were extracted with a parser that did its own quality filtering, and the constrained arm's schema forbade any reasoning field. Rerun with matched prompts and a reasoning field present, they reported structured generation matching or beating free-form. I have not seen this settled in a way I would call definitive, and both positions have obvious incentives.

Here is what I actually believe, and the decision rule I use, which is what an interviewer wants:

**The disagreement is mostly about the schema, not about masking.** If your schema has no room for reasoning, you have removed chain-of-thought and you will pay for it on reasoning-heavy tasks — that is Tam's finding and it is real. If your schema *has* a reasoning field before the answer, the masking overhead per se is small, because the mask leaves the model's preferred continuation available almost everywhere inside a free-text field. The variable that predicts the effect is "does the output shape permit derivation before conclusion," not "is a grammar attached."

So the rule: **structured output is close to free for extraction, classification and formatting; it is expensive for multi-step reasoning unless the schema contains a reasoning field emitted before the answer.** And on any task where you are unsure, run the A/B — 300 examples, constrained versus free-form-plus-parser, scored on task accuracy, not on parse rate. That experiment takes an hour and settles it for *your* model and *your* task, which is the only answer that matters.

**🗣 Say this in the room:** "It's contested. Tam et al. found format restrictions hurt reasoning; the Outlines authors argued the comparison was confounded by prompt mismatch and by schemas with no reasoning field. My read is that the effect tracks whether the schema permits derivation before conclusion, so I always emit reasoning first, and I A/B the specific task rather than trusting either side."

### Explain "constrain the output, not the thinking." How would you actually build that?

This is the single most useful design pattern in this section, and it resolves the constraint tax almost entirely. The idea: **the model does not have to be constrained for its whole generation. Constrain only the span you intend to parse.** Everything before that span is free-form, the model reasons in the mode it was trained in, and the constraint switches on for the final block.

There are three implementations, in increasing order of enforcement strength.

**One — reasoning as a schema field.** The cheapest and the one I reach for first. `{"reasoning": str, "answer": ...}`, with `reasoning` first and generously bounded (say 1,500 characters). The model reasons *inside* a JSON string. Cost: the JSON string escaping is mildly off-distribution — the model has to escape quotes and newlines while thinking, which is a small tax — but it works with every provider and needs no engine support. This is my default.

**Two — a two-phase call.** Call once unconstrained ("think through this, don't format anything"), then call again with the reasoning in context and the schema attached ("now emit the object"). Enforcement is perfect and the reasoning is fully unconstrained. Cost is one extra round trip *and* an extra prefill of the reasoning tokens — though prefix caching covers the shared document, so the marginal cost is the reasoning tokens only. **💰 Math:** a 6k-token document, 500 reasoning tokens, 200 structured tokens, at $3/Mtok in, $15/Mtok out, 90% cache discount on cached input. Single call: `6000×3/1e6 + 700×15/1e6 = $0.018 + $0.0105 = $0.0285`. Two calls: call one `$0.018 + 500×15/1e6 = $0.0255`; call two reads 6,000 cached tokens at `6000 × 0.30/1e6 = $0.0018` plus 500 new reasoning tokens at `$0.0015` plus 200 output at `$0.0030` = `$0.0063`. Total `$0.0318` — an **11.6% cost increase** and roughly a 40% latency increase from the second TTFT. That is the honest price of perfect separation.

**Three — a grammar with a free-text prefix region.** If you own the grammar, write it as `<thinking> [^<]* </thinking>` followed by the constrained JSON block. The automaton permits anything inside the thinking region and clamps down after. This is exactly what GBNF and llguidance let you express, and it gives you one call, unconstrained reasoning, and hard enforcement. It is the best answer technically and the least portable.

**⚠ Trap:** bounding the reasoning field too tightly. A `maxLength=200` reasoning field is worse than no reasoning field, because the model starts a derivation and gets masked into truncating it mid-thought, then must produce an answer from a half-finished argument. If you are going to include reasoning, give it real room — my floor is 800 characters — or leave it out.

**🗣 Say this in the room:** "Constrain the output, not the thinking. In practice that's a `reasoning` field emitted before the answer, generously bounded. If I need perfect separation I do two calls with prefix caching covering the document, which costs about 12% more and one extra TTFT. If I own the grammar I put a free-text region in front of the JSON block and get both properties in one call."

### How do reasoning models interact with structured outputs?

Reasoning models make the pattern above native rather than a workaround, and that changes the design in one specific way: **the thinking block is outside the constraint by construction.** The model emits reasoning tokens in a dedicated channel (delimited by special tokens, and often not returned to you verbatim), and the schema applies only to the final answer region. This is mechanism three from the previous question, implemented by the provider.

Practical consequences. First, adding a `reasoning` field to your schema on a reasoning model is usually redundant and sometimes harmful — you get double reasoning, once in the thinking channel and once in the string, at double the token cost. Strip it. Second, your `max_tokens` budget must account for thinking tokens, which are billed as output and which you frequently cannot see; a schema that expects 200 output tokens may consume 3,000 thinking tokens first, and if `max_tokens` is 500 the model gets truncated *inside the thinking block* and returns nothing parseable at all. That failure is confusing the first time you hit it because the response is empty and the finish reason says length.

**📐 Numbers you must know:** budget thinking tokens as a separate line item. A moderate reasoning effort on a hard extraction is commonly 1,000–4,000 thinking tokens. At $15/Mtok output that is `$0.015–$0.060` per call *before* your 200 answer tokens — so on a reasoning model the answer is often under 10% of the output bill. **📅 Volatile:** per-model thinking-token distributions and the pricing of thinking tokens move constantly; measure the distribution on your own traffic and set `max_tokens` at p99-of-thinking plus your answer budget, not at a guess.

Third, and this is the sharp edge: some engines and providers apply the constraint from token zero unless they know about the thinking channel. If you attach a JSON grammar to a model that wants to open a thinking block, you mask its thinking-block opener, which is precisely mechanism two of the constraint tax — you have paid for a reasoning model and then forbidden it to reason. Verify by sampling raw outputs with constraints on and confirming thinking tokens are still present; if they are not, the constraint start position is wrong and that is an engine configuration bug, not a model limitation.

**⚠ Trap:** shipping a schema with a `reasoning` field to a reasoning model and reading the field as if it were the model's actual reasoning. It is a *summary* the model writes after the fact, in a different channel, and it is no more faithful than any other post-hoc rationalization. Do not build a "show your work" UI on it and call it interpretability.

### Format-compliance rate — how do you define it and what do you put next to it?

Define it precisely, because the loose version is useless. **Format-compliance rate = fraction of responses that (a) parse as JSON, (b) validate against the schema, and (c) terminate with a natural stop reason rather than length truncation.** All three clauses matter: (a) alone is JSON-mode's metric, (b) alone misses the truncation class, and (c) is where the unbounded-string failures hide.

Then — and this is the part that separates people who have run this from people who have read about it — you never report it alone. The dashboard I insist on has five panels:

**Format-compliance rate**, broken down by tenant, schema version and model. Not aggregate. An aggregate 99.4% can be 100% on 40 tenants and 71% on the one tenant whose documents are Japanese.

**Repair rate and repair depth.** What fraction needed attempt 2, and attempt 3. This is the leading indicator: compliance stays at ~100% while repair rate climbs, because your retry loop is absorbing a degradation. Repair rate moving from 3% to 9% is the alert; compliance staying at 100% is the reason nobody noticed.

**Field-level null and placeholder rate.** Per field. This is the only instrument that catches "correct but unusable." A null rate that moves in *either* direction after a change is a signal — down means fabrication, up means the model stopped finding things.

**Value-distribution drift per enum field.** Compare the production distribution against a fixed labelled sample using something simple and robust — I use population stability index or just a chi-square against the reference — and alert on drift. This catches enum anchoring and model-swap regressions.

**Task accuracy on a golden set**, run on every schema change and every model change. 200–500 hand-labelled examples, scored field by field with exact match for enums and ids, normalized match for strings, and tolerance-based match for numerics.

**⚠ Trap:** treating format-compliance rate as a quality metric in a launch review. It is a *hygiene* metric, like HTTP 5xx rate. Nobody ships a service because its 5xx rate is zero. The number that justifies the launch is field-level accuracy on the golden set, and if a team presents only compliance I read it as a team that has not built the accuracy eval yet.

**💰 Math on the golden set:** 400 examples × 6k input tokens × $3/Mtok = `400 × 6000 × 3/1e6 = $7.20` per full run, plus output. Running it on every PR that touches a schema, at 60 PRs a month, is `$432/month`. That is roughly two hours of one engineer's time per month, for the ability to ship schema changes without a quality review. There is no cheaper insurance in the whole stack, and stating that arithmetic is how you win the argument for building it.

### Design the schema-compilation and caching layer for a latency-sensitive multi-tenant service.

Assume the product is enterprise document extraction — each tenant defines their own extraction template, so schemas are user-authored and unbounded in variety, and the SLO is p99 under 2 seconds. The naive design compiles at inference time and fails the SLO on every cold schema.

**The registration-time compile.** A schema enters the system exactly once, when a tenant saves a template. That write path does: validate the schema (reject recursion, enforce a depth cap of 3, enforce `maxLength` on every string, cap total properties at ~60), canonicalize it, hash it, and enqueue a compile job. The compile job builds the grammar, measures the compile time and the resulting index size, and writes the artifact to a shared store keyed by hash. The template is not marked usable until the compile succeeds. **Inference never compiles.** This is the whole design, and it is the same "do expensive work on the write path" instinct you already apply to materialized views.

**The two-tier cache.** Tier one is an in-process LRU of deserialized grammar objects, sized by memory not by count — this is important, because grammar index sizes vary by two orders of magnitude across schemas and a count-based LRU will happily hold 500 tiny grammars or OOM on 12 large ones. Tier two is the shared artifact store (S3 or Redis, depending on artifact size), so a cold replica loads a compiled artifact in tens of milliseconds instead of recompiling for seconds.

**Warmup.** On pod start, before the readiness probe passes, pull the top-N schemas by last-24h request volume and load them into the process cache. Without this, every deploy produces a burst of tier-one misses across every replica simultaneously — the classic thundering-herd-on-deploy shape, and it will show up as a p99 spike exactly at rollout time and nowhere else, which is maddening to diagnose from application traces alone.

**💰 Math on cache sizing:** suppose the p50 compiled artifact is 4 MB and p95 is 40 MB. A pod with 2 GB of headroom holds 500 p50 artifacts or 50 p95 ones. If your top 200 schemas cover 94% of traffic (they will — schema popularity is Zipfian just like every other tenant-keyed distribution you have seen), a 200-slot memory-bounded cache gets you a 94% tier-one hit rate, and the 6% miss costs a tier-two fetch of maybe 30 ms rather than a 1.4 s compile. Latency contribution: `0.06 × 30 ms = 1.8 ms` mean, and a p99 that never sees a compile.

**⚠ Trap:** letting tenants author schemas without a validator on the write path. A tenant with a 200-property schema containing eight `anyOf` branches and unbounded strings will produce a grammar that takes 40 seconds to compile and 900 MB of index, and they will discover this by hanging your compile workers. The schema-admission validator is not a nicety; it is the rate limiter for a resource nobody thinks of as a resource. Cap depth, cap properties, cap enum cardinality, require string bounds, reject recursion, and enforce a compile-time budget with a hard kill.

### Our p99 TTFT regressed 800 ms after we shipped per-request dynamic schemas. Debug it.

Good — this is the schema-compile pathology and the debugging path is short if you know where to look, and long if you look at the GPU first. I would work it in this order.

**Step one: confirm it is not GPU-side.** Check GPU utilization, batch size, queue depth and prefix-cache hit rate over the regression window. If all four are flat and TTFT moved, the added time is on the CPU path before the forward pass. That single check saves a day, because the instinct is always to look at the model.

**Step two: look at the shape of the regression, not the mean.** Pull the TTFT histogram. The signature of a schema-compile problem is **bimodality**: a tight mode at the old p50 and a second, much smaller, much slower mode a second or more out. A GPU-side regression shifts the whole distribution; a cache-miss regression adds a second mode. If you see bimodality, you are looking at a cache miss of some kind — schema compile or prefix cache.

**Step three: instrument the grammar cache.** You want three counters: compile count, cache hit rate, and compile duration histogram. If the compile count is roughly equal to the request count, your hit rate is zero and the cause is almost certainly cache-key instability, not schema variety. The usual culprits, in the order I check them: schema dicts serialized with non-deterministic key order; Pydantic `$defs` names varying because the model class is generated dynamically per request with a name like `Extraction_7f3a`; a `description` field containing a timestamp or a tenant name; or a `title` auto-added by Pydantic that varies. Any of these makes structurally identical schemas hash differently.

**Step four: measure schema entropy.** Count distinct canonical hashes per hour against request count. If you have 4,000 requests/hour and 3,800 distinct hashes, the schemas really are dynamic and no cache will save you — the fix is architectural (move to registration-time compile, or move to a backend like llguidance whose Earley parser has a much cheaper cold path than a full FSM index build). If you have 4,000 requests and 40 distinct schemas but 3,800 compiles, it is purely a keying bug and it is a one-line fix.

**💰 What the regression costs:** at 3% cold rate and a 1.4 s compile on a service with a 900 ms p50, the affected requests land at 2.3 s. If your SLO is p99 < 2 s, you are failing it on all of that 3%. If the service is customer-facing at 2M requests/month, that is 60,000 requests over SLO. And note the compile is CPU-bound and single-threaded — at 3% of 4,000 req/hour, that is 120 compiles/hour × 1.4 s = 168 CPU-seconds/hour, which will also starve your uvicorn workers and inflate *unrelated* request latencies. That coupling is why the regression often looks broader than its cause.

**🗣 Say this in the room:** "First I'd confirm GPU metrics are flat, which localizes it to the CPU path. Then I'd look for bimodality in the TTFT histogram — a compile miss adds a second mode rather than shifting the distribution. Then I'd check grammar-cache hit rate and distinct-schema-hash count; if compiles ≈ requests but distinct schemas are few, it's a cache-key canonicalization bug, and if distinct schemas really are many, the fix is compiling at schema-registration time instead of at inference."

### Since we turned on structured outputs, every field comes back as an empty string. What happened?

Empty-string-everything is a specific signature and it has three plausible causes; I would distinguish them in about ten minutes.

**Cause one: the prompt no longer asks for anything.** Someone enabled structured outputs and deleted the format instructions from the prompt on the reasonable-sounding theory that the schema now handles it. It does not. The schema constrains *shape*; the prompt supplies *intent*. With the instructions gone, the model's best guess at "what goes in these fields" degenerates, and under a mask that requires a string, the cheapest legal string is `""`. **Check by diffing the prompt** against the last known-good version. This is the most common cause by a wide margin and it is embarrassing, which is why people look for exotic explanations first.

**Cause two: the content the model needs is not in context.** The empty string is the model correctly signalling absence in the only way the schema permits. Check whether the document actually got attached — a truncation bug, a retrieval bug, or a `max_tokens` on the *input* side. The tell: the empty-string rate correlates with document length or with a specific ingestion path.

**Cause three: the constraint starts at the wrong position.** If the grammar is applied from token zero on a model that wants to emit a preamble or a thinking block, you have masked its natural opener and it recovers into a degenerate mode. The tell: raw output starts with `{` immediately with no whitespace or preamble, and the same prompt without constraints produces good content. Fix by moving the constraint start to after the model's preferred opener, or by switching to the reasoning-field pattern.

**⚠ Trap:** the fourth cause, which is not a bug: the schema requires fields the document genuinely does not contain, and because strict mode forces every property into `required`, the model must emit *something*. Empty string is the honest answer and your schema forbade `null`. If this is it, the fix is to make the fields nullable (`str | None`, present-but-nullable) and instruct null-on-absence, then treat the null rate as a data-quality metric. **The empty-string epidemic is frequently a correct model behaving under an incorrect contract.**

Whichever it is, the systematic detection is what I would build afterward: a per-field empty-and-placeholder rate on the dashboard, with an alert on any field whose rate moves more than a few points week over week. That metric would have caught this on the day of the deploy rather than whenever a customer complained.

### Compliance was 100% in staging and 94% in production on the same model. Explain the gap.

A compliance gap between environments on an identical model is almost always an *input* difference or a *configuration* difference, and there are five candidates I would check in this order.

**Input distribution.** Staging runs on a curated set of clean documents; production gets scans, multi-column layouts, other languages, and 90-page PDFs. The 6% is likely concentrated: group the failures by document length, language, source and tenant, and I would bet heavily that one or two buckets contain nearly all of them. The most common single bucket is *long inputs hitting `max_tokens` mid-generation*, which produces truncation, which is a compliance failure with a `length` stop reason. Check that first because it is one query.

**Non-ASCII content.** If the grammar has any `pattern` compiled against byte semantics, or the schema has length bounds you are measuring in characters while the constraint counts bytes, non-Latin scripts fail systematically. A `maxLength: 100` enforced over bytes cuts a Hindi string at roughly 33 characters, mid-codepoint, producing invalid UTF-8.

**Sampling configuration drift.** Production has a temperature or a penalty that staging does not — very often a frequency penalty inherited from a chat config, which as covered earlier progressively bans the JSON punctuation. Diff the full generation config, not just the prompt.

**Concurrency and truncation under load.** Under production batching, some engines apply a shorter effective `max_tokens` or preempt long sequences. A preempted-and-resumed sequence with a grammar must have its automaton state restored correctly; a bug there yields rare, load-correlated malformation. The tell is that the failure rate correlates with batch size or with queue depth, which is a correlation you should be able to plot in five minutes and which nobody thinks to plot.

**Version skew.** Staging pins one engine version, production runs another; grammar backends have had real behavioral differences across versions, particularly around recursion and around which schema keywords are honored. Compare the resolved package versions, not the requirement specs.

**🗣 Say this in the room:** "A 6% gap on an identical model is an input or config difference, not a model difference. I'd bucket the failures by length, language, tenant and stop reason first — truncation from `max_tokens` on long documents is the single most common cause, and it shows up as a `length` finish reason rather than as malformed JSON. Then diff the generation config for a stray frequency penalty, and diff engine versions."

### How does constrained decoding interact with prefix caching?

They are orthogonal in the happy path and interact badly in exactly one place, which is worth knowing precisely.

Prefix caching operates on the *input* — it reuses computed KV for a shared prompt prefix, so the second request with the same 6k-token system prompt and document skips that prefill. Constrained decoding operates on the *output* — it masks logits during decode. Different phases, no conflict. A constrained call gets the full prefix-cache benefit on its prompt.

The interaction is in **the repair loop**, and it is a positive one you should exploit deliberately. When you retry with error feedback, you have a choice: rewrite the prompt to incorporate the correction, or append the bad assistant turn plus a user correction. Rewriting mutates the prefix and destroys the cache hit; appending preserves it. **💰 Math:** on a 6k-token cached prefix at $3/Mtok with a 90% cache discount, an appended retry pays `6000 × 0.30/1e6 = $0.0018` for the prefix instead of `6000 × 3/1e6 = $0.018` — a `$0.0162` saving per retry. At 4% of 2M monthly calls repairing, that is `0.04 × 2e6 × 0.0162 = $1,296/month` for the discipline of appending rather than rewriting. That is a bigger number than the entire retry overhead in the earlier cost model, which is a nice illustration that in LLM systems the second-order effects of a code-structure choice frequently exceed the first-order effects of the feature.

The second interaction is on the schema side and it is a footgun. If you embed a per-request identifier — a request id, a timestamp, a tenant name — anywhere in the schema or in the system prompt region that carries the schema, you have made every prefix unique and your prefix-cache hit rate is zero. This is the same class of bug as the grammar-cache-key instability, and it very often has the same root cause: someone templated something that should have been static.

**⚠ Trap:** the schema goes in the *prefix* on most providers (it is part of the request configuration or the rendered system region), so a schema that changes per request costs you both the grammar-compile cache and the prefix cache simultaneously. When someone proposes per-request dynamic schemas, that is a double cache miss, and the design review question is "can these be a small enumerated set of schemas plus a data field, instead of an infinite family of schemas?" Usually the answer is yes.

### What does constrained decoding do to speculative decoding and to batched throughput?

Two separate answers, and both matter for anyone claiming serving depth.

**Speculative decoding.** The draft model proposes `k` tokens; the target model verifies them in one forward pass; accepted tokens are kept. Adding a grammar means every proposed token must also be *grammar-legal*, and the draft model does not know the grammar unless you mask it too. So you get one of two designs. Naive: draft unconstrained, then reject any proposal that violates the grammar at verification. This tanks the acceptance rate, because a draft model producing free-form text inside a JSON structure will violate constantly — you can lose most of your speculative speedup. Correct: apply the mask to the draft model as well, so proposals are legal by construction. That costs `k` extra mask computations per step, which is the reason mask cost matters at 40 µs rather than 2 ms.

The subtle part is **rollback**. When the target rejects proposal `j`, the grammar automaton has already advanced `j` steps and must be rewound. This is precisely why XGrammar uses a persistent execution stack with cheap fork and rollback rather than a scalar DFA state — a scalar state can be rewound only if you saved the history, and a stack-based PDA needs structural rollback. If you are writing a custom grammar backend and you have not implemented rollback, speculative decoding will silently corrupt your constraint. That is a great thing to raise unprompted in a serving-focused interview.

**Batched throughput.** The mask must be computed per *sequence*, because each request is at a different automaton state, and it must be applied to a `[B, V]` logit tensor. The cost model: mask computation is CPU work that scales linearly in batch size, mask application is a GPU kernel that is trivially cheap. So constrained decoding converts some of your batch into CPU-bound work, and at large batch sizes the CPU can become the bottleneck.

**💰 Math:** at 40 µs/token/sequence and batch 128, mask computation is `128 × 40 µs = 5.12 ms` per decode step of CPU time. If your decode step is 20 ms of GPU time, and the mask work is overlapped with the forward pass on another thread, you absorb it — 5.12 ms fits inside 20 ms. If the mask work is *not* overlapped (a naive Python logits processor, or a GIL-bound implementation), you have added 5.12 ms to a 20 ms step: a **25% throughput loss** at batch 128. And if the per-token cost is 2 ms instead of 40 µs — which is exactly what a pure-Python processor costs — then `128 × 2 ms = 256 ms` of CPU per step against a 20 ms GPU step, and your GPU is idle 92% of the time. **This is the entire reason the engines rewrote these in Rust and C++ and why the microsecond figure is quoted so loudly.**

**⚠ Trap:** benchmarking constrained decoding at batch 1 and concluding the overhead is negligible. At batch 1 the mask is 40 µs against a 20 ms step — 0.2%, genuinely nothing. The cost is superlinear in operational impact because it competes for the CPU that also runs your scheduler, detokenizer and API server. Benchmark at your production batch size or the number is meaningless.

### If everything is masked and only one token is legal, does the model even need to run?

No, and this is a real optimization with a name — token forcing, or "fast-forwarding" — and knowing about it signals that you have thought about the mechanism rather than just used the API.

Whenever the automaton's legal set is a singleton, the sampler's output is determined regardless of the logits. You can append that token to the sequence without a forward pass. In JSON this happens constantly: after `{` the next tokens `"` and then the first property name are fully determined by the schema; after a property name, `":` is determined; after the last property's value, `}` is determined. For a schema with fixed keys, a substantial fraction of the emitted tokens are pure structure that the model contributes nothing to.

You can extend this: if the next `m` tokens are determined, you can fast-forward all `m` at once and then run a single prefill-style forward pass over them to populate the KV cache — one forward over `m` positions is far cheaper than `m` forwards over one position each, since prefill is compute-bound and decode is bandwidth-bound.

**💰 Math:** take a 12-field extraction where key names and punctuation total ~90 tokens and the actual values total ~110 tokens, 200 total. Fast-forwarding the 90 structural tokens removes 90 of 200 decode steps — **45% of the generation**. At 20 ms/token that is `90 × 20 ms = 1.8 s` saved on a 4 s generation, minus one prefill over 90 tokens (call it 30 ms). And on cost, if your provider bills output tokens regardless, you save nothing on the bill — but on self-hosted serving you have just increased throughput on that request by nearly 2×.

This is why terse field names are a bad idea for accuracy but a *good* idea for cost, and why the two considerations pull opposite ways: long descriptive field names improve the model's grounding and are free to generate under fast-forwarding, but they cost input tokens in the schema. Descriptive names, fast-forwarded emission: that is the combination you want.

**⚠ Trap:** assuming your engine does this. Some do, some do it only for the trivial single-token case, and hosted providers give you no visibility. You can detect it empirically: generate a structured response with a heavily-structural schema and compare wall-clock time against an equivalent free-form generation of the same token count. If constrained generation is meaningfully *faster* per emitted token, fast-forwarding is on. If it is the same or slower, it is not, and there is real performance sitting on the table.
