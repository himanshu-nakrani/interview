### Here's the one that separates seniors: when does speculative decoding make throughput *worse*?

At high batch size, and the mechanism is worth stating precisely because most candidates know the fact and can't explain it.

The entire premise of speculation is that decode is memory-bandwidth-bound: a forward pass costs a weight read regardless of how many tokens flow through it, so extra tokens are free. **That premise is a function of batch size.** At batch `B`, one decode step processes `B` tokens through the same weight read, so arithmetic intensity scales roughly linearly with `B`. Somewhere around `B` in the low hundreds — the exact crossover depends on model shape, quantization, and GPU — you cross the roofline knee and become **compute-bound**. Past that point the free lunch is gone: tokens now cost FLOPs.

Now apply speculation at batch `B` with `K` draft tokens. Your verification pass processes `B × (K+1)` tokens instead of `B`. If you were already compute-saturated, that is a `(K+1)×` increase in verification FLOPs, of which only `E[accepted]/(K+1)` fraction is useful. At α=0.7, K=5, you emit 2.94 tokens per iteration but computed 6 — **you threw away 51% of the FLOPs you just spent.** Plus the draft passes, which at batch B are themselves no longer negligible. Aggregate throughput drops.

So the honest framing: **speculative decoding trades throughput for latency.** At low batch it's nearly free because the throughput you're spending was idle anyway. At high batch you are explicitly buying per-request latency with cluster-wide tokens/sec, and you should say so in exactly those words.

**📐 Numbers you must know:** the rough crossover for a dense 70B on H100-class hardware is a batch of order 32–64 for speculation to remain clearly net-positive on throughput, degrading through the low hundreds and turning negative beyond. **📅 Volatile and hardware-specific** — MoE models, quantized weights, and long contexts all move the knee, and you must measure your own. The *shape* of the curve is the durable knowledge: monotonically decreasing benefit in batch size, crossing zero.

**⚠ Trap:** the entire EAGLE/Medusa literature reports batch-size-1 speedups, because that's the regime where the technique shines and the regime a latency paper cares about. A team reads "4× speedup," enables it on a production fleet running batch 128, measures a **throughput regression**, and concludes the papers are wrong. The papers aren't wrong; they're measuring a different quantity. This is the single most valuable thing to say unprompted in a serving-internals interview.

**🗣 Say this in the room:** "Speculation exploits the fact that decode is bandwidth-bound, and that stops being true as batch size rises — at batch B you verify B×(K+1) tokens and discard the rejected fraction. So it's a latency-for-throughput trade. I'd enable it on low-batch, latency-SLO traffic and disable or shrink K on high-batch throughput traffic, and I'd expect the published 3–4× numbers to be batch-1 figures that don't transfer."

### So give me the decision rule. When do you turn it on, and can you make that decision dynamically?

**The static decision rule**, which I'd write into a design doc:

| Situation | Speculation |
|---|---|
| Batch 1–8, hard p95 ITL/TTFT SLO (voice, autocomplete, interactive agent) | **On, aggressive K.** This is what it's for. |
| Copy-heavy workload (RAG, code edit, extraction) at any batch | **On, n-gram drafting** — `c ≈ 0` means it rarely loses even at higher batch. |
| Batch 64+, throughput-oriented, generous latency budget (batch scoring, offline eval, bulk summarization) | **Off.** You are converting tokens/sec into latency nobody is measuring. |
| Anything where measured α < ~0.4 | **Off.** The formula says you're near or below 1× and you're burning draft compute. |
| Mixed traffic on one fleet | **Adaptive** — see below. |

**The adaptive answer**, which is what production engines converge on and what I'd propose. Make K a function of *current scheduler state*, not a static config:

```python
def choose_k(running_batch, alpha_ewma, k_max=6):
    if alpha_ewma < 0.40:           # drafter is not tracking this traffic
        return 0                     # disable speculation entirely
    if running_batch >= HIGH_BATCH:  # compute-saturated: verification isn't free
        return 0
    if running_batch >= MID_BATCH:
        return min(2, k_max)         # small K: keep some latency win, cap wasted FLOPs
    return min(k_max, 1 + int(6 * alpha_ewma))   # low batch: spend freely
```

Two signals, both of which you already have in a serving engine: the current running-batch size (the scheduler knows it every step) and the EWMA acceptance rate. This gives you the behavior you actually want across the day — aggressive speculation at 3 a.m. when the fleet is empty and one user's latency is all that matters, and automatic step-down at peak when the batch is full and throughput is the binding constraint. It also means a traffic spike degrades gracefully into higher throughput instead of collapsing into wasted verification.

**⚠ Trap:** doing this per-fleet with a config flag and a deploy. The batch size changes minute to minute; your config doesn't. If the decision is made at deploy time it will be wrong for most of the day. It belongs in the scheduler.

**💰 Math:** a fleet serving 60% interactive (batch ~8) and 40% bulk (batch ~192). Static-on: interactive gets 2.9×, bulk loses ~15% throughput. Static-off: interactive pays full latency. Adaptive: interactive keeps 2.9×, bulk keeps 100%. If interactive is 400,000 requests/day at 500 output tokens and 40 tok/s baseline, 2.9× saves 400,000 × 500 × (1/40 − 1/116) = 400,000 × 500 × 0.0164 = 3.28M GPU-seconds/day = 911 GPU-hours = **$2,278/day**, while the bulk fleet's 15% throughput loss on (say) 1,200 GPU-hours/day would have cost **$450/day**. Adaptive banks the first and avoids the second.

### The acceptance rate collapsed in production last night. Give me the failure taxonomy.

**🔍 Failure taxonomy** — worked as a decision procedure, ordered by how fast each check is.

**Is the drop global or segmented?** Split the acceptance metric by route, language, tenant, and model version. This one cut identifies the cause about half the time.

- **Segmented to one language or locale** → **OOD collapse.** A drafter distilled on English traffic sees a wave of Japanese, Hindi, or code-switched prompts and TV(p,q) explodes. This is the canonical failure and it's usually caused by a marketing launch in a new region, not by anything you deployed. Response: per-language K (or disable), and add that language to the distillation corpus.
- **Segmented to one route** → the route's *output distribution* changed. Someone added JSON-mode, or switched from summarization to code generation, or changed a system prompt in a way that shifts style. Speculation configuration is per-route; treat this as a config gap.
- **Segmented to one tenant** → that tenant's prompts are unusual — a domain-specific jargon corpus, an unusual language, an adversarial user. Per-tenant K, or exclude.
- **Global, sharp, coincides with a deploy** → **you changed the target model.** The drafter was distilled against the old target; its distributional alignment is now stale. This is the most expensive and most preventable one: **draft models are versioned artifacts coupled to a specific target checkpoint,** and a target upgrade without a matching drafter re-distillation is a silent 2× throughput regression. Enforce the coupling in your model registry.
- **Global, sharp, no deploy** → check whether decoding parameters changed. A temperature or top-p change from a client, a new default in your SDK, or a grammar/constrained-decoding path being enabled all shift `p`.
- **Global, gradual, over weeks** → traffic drift. Your user base is writing different things than when you distilled. This is normal and is why acceptance rate needs a slow-moving trend alert, not just a spike alert.

**The monitoring that catches all of these:** acceptance rate emitted per (route, language_tag, tenant_tier, target_model_version, drafter_version), with two alerts — a fast one on a >20% relative drop over 15 minutes, and a slow one on a >10% drop week-over-week. Plus a hard **guardrail**: if measured α drops below the break-even threshold implied by your K and c, the engine should disable speculation automatically rather than continuing to burn draft compute. Speculation failing should degrade to plain decoding, not to slower-than-plain decoding.

**💰 Math:** a fleet at α=0.8, K=5, c=0.05 (2.95× effective) dropping to α=0.35 (1.23×) costs you a factor of 2.4 in decode throughput. On a 200-GPU decode fleet at $2.50/hr that's 200 × 24 × $2.50 × (2.95/1.23 − 1) = 12,000 × 1.40 = **$16,800/day** if you were running at capacity — or, more likely, a latency SLO breach and an autoscaler stampede. The guardrail that disables speculation below break-even is worth more than the speculation itself.

### How does speculative decoding interact with constrained decoding — JSON schema, grammars, regex?

This is a genuinely subtle interaction and a great question to be asked, because the two techniques both modify the sampling distribution and they compose in a specific, correct way if — and only if — you're careful about *which* distribution the acceptance test uses.

**The correct composition.** Constrained decoding works by masking logits: at each position, the grammar's state machine determines which tokens are legal, and illegal tokens get `-inf`. The resulting constrained distribution `p_c = normalize(p ⊙ mask)` is the distribution the user actually wants. Speculative decoding's guarantee is that it reproduces whatever distribution you use in the acceptance test. So: **run the acceptance test against `p_c`, the masked target distribution, not the raw `p`.** Get that right and speculation is exactly lossless with respect to the constrained distribution.

**Where it goes wrong.** If your drafter proposes tokens without the mask applied — a separate small model that doesn't run the grammar — then `q` puts mass on tokens with `p_c = 0`, and those are rejected with probability 1. Every illegal proposal is a wasted draft step. In a tight grammar (JSON with a strict schema, where at many positions only one or two tokens are legal), an unmasked drafter's acceptance rate can collapse toward zero even though the target is *maximally* predictable at those positions.

**The fix, which is also the opportunity.** Apply the grammar mask to the drafter too. Now something nice happens: **at positions where the grammar admits exactly one token, acceptance is guaranteed.** Emitting `{"name": "` after `{` is forced by the schema — the drafter and target both have all mass on the same token, TV = 0, α = 1. Tight grammars make speculation *better*, not worse, because the constraint that makes generation predictable is exactly the thing acceptance rate measures.

Taken to the limit, this is why some engines implement **jump decoding / fast-forwarding**: when the grammar's state machine has only one legal continuation for several tokens (a long fixed key name, a closing bracket sequence), you can emit those tokens without running the model at all. That is speculation with α = 1 and c = 0 — the degenerate best case.

**⚠ Trap:** running the grammar state machine forward speculatively and then failing to **roll it back** on rejection. The FSM has state; if you advance it over 5 draft tokens and only 2 are accepted, the machine is now three transitions ahead of reality and will emit structurally invalid output from that point on. This is the same class of bug as forgetting to truncate the KV cache, it's harder to spot because output stays *syntactically* plausible for a while, and it is the first thing I'd look for if a constrained-output endpoint started producing malformed JSON after a speculation rollout.

**🗣 Say this in the room:** "They compose if you run the acceptance test against the *masked* target distribution and apply the same mask to the drafter. Done right, a tight grammar raises acceptance rather than lowering it — at forced positions TV is zero and acceptance is one, which is what makes fast-forwarding over deterministic grammar spans possible. Done wrong, the drafter proposes illegal tokens that are always rejected, and you must remember to roll back the grammar's FSM state on rejection or the output goes structurally invalid."

### How do prefix caching and speculative decoding interact? Do they compose?

They compose cleanly, because they attack **different phases**, and being able to say that crisply is the point of putting them in the same section.

Prefix caching removes **prefill** — it cuts TTFT and input cost. Speculative decoding accelerates **decode** — it cuts ITL/TPOT and output-phase GPU time. A request's latency is `TTFT + n_output × ITL`; caching shrinks the first term, speculation shrinks the second. They are multiplicative on total latency and neither interferes with the other's mechanism.

Three second-order interactions worth naming:

**One: the draft model needs its own prefix cache, and people forget it.** If the target has a warm 20k-token prefix but the drafter is prefilling 20k tokens from scratch on every request, you have added a serial 20k-token prefill of a small model to your TTFT. It's a small model, so it's fast — but at K sequential draft passes per iteration plus a cold prefill, it's not free. **Enable prefix caching on the drafter too**, and if your engine only caches the target, that's a real gap. Self-drafting methods (Medusa, EAGLE) sidestep this entirely, which is an underrated operational argument for them.

**Two: they compete for the same HBM.** The prefix-cache reuse pool, the running batch's KV, the draft model's weights, and the draft model's KV all come out of the same 80 GB. Adding a 7B drafter to a 70B deployment costs 14 GB of weights (FP16) — which is 14 GB not available for cached prefixes or batch capacity. **That memory is a real cost and belongs in the comparison**: a 7B drafter that buys 3× decode may cost you 20% of your prefix-cache capacity and therefore some TTFT. Do the budget.

**Three: prefix-affinity routing and speculation want the same thing.** Both favour smaller batches per replica and stable request-to-replica assignment. A router that already implements prefix affinity is also creating the low-batch conditions where speculation pays. That's a happy accident but worth naming as a design coherence argument.

**💰 Math:** a RAG endpoint, 24k-token cached context, 400 output tokens. Baseline: TTFT with cold prefill ≈ 900 ms, decode at 40 tok/s = 10.0 s, total **10.9 s**. Prefix caching alone: TTFT ≈ 180 ms, total **10.2 s** — a 6% win, because decode dominates. Speculation alone (n-gram, α ≈ 0.8 on quoted spans, effective 3.3×): TTFT 900 ms + 3.0 s = **3.9 s**. Both: 180 ms + 3.0 s = **3.2 s**, a 3.4× end-to-end improvement. Note the shape of that: **caching alone barely moved the number, but caching on top of speculation is a 19% further cut**, because once you've compressed decode, prefill is a much larger share of what's left. Optimizations that look marginal in isolation become significant after you fix the dominant term — measure the stack, not the components.

### How does speculation interact with continuous batching and the scheduler?

This is where a clean algorithm meets a messy scheduler, and the friction is real.

**The core problem: variable token yield per step breaks the scheduler's uniformity assumption.** Continuous batching schedules at the iteration level assuming each running sequence advances by exactly one token per step. With speculation, sequence A might advance 6 tokens this step and sequence B might advance 1. That has three consequences.

**Memory allocation becomes bursty.** A sequence that accepts 6 tokens may cross a KV block boundary and need a new block *mid-step*, and the block it needs must have been reserved before verification began — because verification writes KV for all K speculated positions before you know how many are accepted. So the engine must **speculatively allocate `K+1` token-slots per sequence per step and free the unaccepted tail afterward.** At batch 64 with K=5 that's 64 × 6 = 384 token-slots reserved per step versus 64 without speculation — a 6× increase in allocation churn and a 6× increase in the headroom you must keep free to avoid preemption. If your engine preempts on allocation failure, speculation makes preemption dramatically more likely at high KV utilization.

**Batching draft steps.** The `K` draft passes are themselves batched across running sequences, but sequences may have different `K` (if you're adapting K per request) and some may have speculation disabled. The scheduler either forces uniform K across the batch (simple, wastes work on sequences that would rather have K=2) or supports ragged K (correct, more kernel complexity). Real engines mostly do the former with a per-step global K.

**Fairness and tail latency.** A sequence in a high-α region of its generation races ahead; one in a low-α region crawls. Under a step-synchronous scheduler that's fine — everyone gets the same number of steps — but it means **per-request throughput becomes variance-heavy**, which shows up as a widened ITL distribution even when the mean improves.

**⚠ Trap:** measuring speculation's benefit as mean ITL. Mean ITL improves; the *distribution* gets wider and more bursty, because tokens now arrive in clumps of 1 to K+1. For a streaming UI that is often *worse perceptually* than a slower but smooth stream — text that arrives in visible bursts reads as janky. If you care about perceived smoothness, buffer output at the application layer and emit at a steady rate; you keep the total-time win and discard the burstiness. I'd raise this proactively for any consumer-facing streaming product.

**📐 Numbers you must know:** the KV headroom requirement. With speculation at K, your engine must keep `batch × K` extra token-slots free to avoid mid-step allocation failure. On an 80 GB card at 0.31 MB/token with batch 64 and K=5, that's 64 × 5 × 0.31 MB = **99 MB** — trivial. But with K=8 and batch 256 it's 256 × 8 × 0.31 = **635 MB**, and at long context with FP16 KV on a heavily-loaded card that's the difference between running and preempting.

### What is adaptive speculation in a disaggregated serving setup, and why does disaggregation change the calculus?

Disaggregation splits prefill and decode onto separate GPU pools with independent scaling, and that changes the speculation decision in a way that's genuinely favourable — which is why the two techniques are increasingly discussed together.

**Why it helps.** In an aggregated engine, prefill and decode share GPUs, so the decode batch is whatever's left after prefill takes its share, and it fluctuates. In a disaggregated setup the **decode pool is doing nothing but decode**, so (a) you can size it for the batch regime you want rather than accepting whatever the mix produces, and (b) the decode pool is unambiguously bandwidth-bound, which is exactly speculation's home turf. You can even choose *different hardware* for the decode pool — cards optimized for memory bandwidth rather than FLOPs — which sharpens the bandwidth/compute imbalance that speculation exploits.

**What "adaptive" means here.** The decode pool's controller knows its own running batch, its KV utilization, and its per-request SLO class. Adaptive speculation means the K for a given step is chosen from that state, per the decision function from earlier — but disaggregation makes the signal *cleaner*, because the decode pool's batch isn't being perturbed by prefill arrivals. You get a stable control signal instead of a noisy one, and control loops on noisy signals is how you end up oscillating.

**The additional lever disaggregation unlocks: per-SLO-class speculation.** With separate pools you can run two decode pools — a low-batch, high-K pool for interactive traffic under a tight ITL SLO, and a high-batch, no-speculation pool for background and batch traffic — and route by SLO class at admission. That is strictly better than one pool trying to serve both, because the two workloads want opposite points on the latency/throughput curve and any single configuration is wrong for one of them.

**The cost.** Disaggregation's own overhead is KV transfer between pools — moving a 2.5 GB cache for an 8k-context 70B request over the interconnect. Over NVLink that's sub-millisecond and irrelevant; over Ethernet it can dominate your TTFT and turn the whole thing into a regression. **📅 Volatile:** the reported gains for P/D-disaggregated serving (up to ~7× in the papers' settings for specific workloads) do not materialize when the interconnect can't carry the KV, and that's the first question to ask about any such claim.

**🗣 Say this in the room:** "Disaggregation makes speculation easier to reason about, because the decode pool is purely bandwidth-bound and its batch size is a control input rather than a byproduct of the prefill mix. That gives you a clean signal for adaptive K, and it lets you run separate decode pools per SLO class — high-K low-batch for interactive, no speculation for bulk — which one pool can never do well."

### What is input-time speculation, and where does it matter?

Input-time speculation is speculating on **what the user is going to say**, rather than on what the model is going to say — starting inference before the input is complete, so that the model's work overlaps the user's.

The canonical setting is **real-time voice**. In a voice agent the pipeline is ASR → LLM → TTS, and the naive design waits for a complete utterance before starting the LLM. But ASR emits partial hypotheses continuously, and the last 300–800 ms of a user's utterance usually doesn't change the meaning of what came before. So: **start the LLM prefill on the partial transcript**, and when the final transcript arrives, either (a) it matches the prefix you already processed and you have a warm KV cache with essentially zero remaining prefill, or (b) it differs and you discard the speculative work.

Note what that is: it is **prefix caching used as a speculation mechanism**. You are betting that the prefix you prefilled will be a prefix of the final prompt, and the cache is what makes the bet cheap to win and cheap to lose. The same mechanism appears in code assistants (start prefilling on the buffer state before the user finishes typing) and in chat UIs (prefill the system prompt and history the moment the user focuses the input box, before they've typed a word).

**Why it matters so much in voice specifically:** the perceptual budget is brutal. Human turn-taking latency in conversation is roughly 200 ms, and anything past ~500 ms of silence reads as the system being slow or the conversation being broken. Your budget has to cover endpointing (deciding the user stopped), LLM TTFT, and TTS first-audio. If LLM prefill on a 4,000-token conversation costs 200 ms, you have already spent most of the budget on something that could have been done while the user was still speaking.

**💰 Math:** a voice agent with a 6,000-token system prompt plus history. Cold prefill ≈ 250 ms. Endpointing ≈ 200 ms, TTS first audio ≈ 150 ms, LLM first token ≈ 250 ms + generation. Total to first audio ≈ **600 ms** — noticeably laggy. With the static prefix cached (removes ~230 ms of the 250) and input-time speculation on the partial transcript (overlaps the remaining prefill with the user still talking), first audio lands near **330 ms** — inside the range that reads as natural. Same model, same weights, two caching decisions.

**⚠ Trap:** speculating too eagerly on partial transcripts wastes real GPU capacity — every discarded speculation is a prefill you paid for and threw away. Gate it: only speculate when the ASR's partial hypothesis has been **stable for some window** (say 150–200 ms) and the partial is long enough that the prefill saving exceeds the expected waste. On a fleet serving thousands of concurrent calls, a naive "prefill on every partial" policy can multiply your prefill load several-fold. Measure speculation **waste rate** — discarded prefill tokens / total prefill tokens — as a first-class metric, exactly as you'd measure draft rejection rate.

### Design case: you own inline code edits at a Cursor-style product. Target p95 of 1.2 seconds end to end. Walk me through it and price it.

Let me fix the request shape first, because everything follows from it: a ~14,000-token prompt (system + repo context + open file + selection), and a ~350-token output that is mostly a modified copy of the input. That shape tells me immediately where the time goes and which levers apply.

**Baseline, no optimizations.** Prefill 14,000 tokens at ~30k tok/s effective ≈ **470 ms**. Decode 350 tokens at 45 tok/s ≈ **7,780 ms**. Plus network and gateway, ~80 ms. Total ≈ **8.3 s**. Four to seven times over budget; decode is 94% of it.

**Lever 1 — n-gram / prompt-lookup speculation.** This is the highest-leverage change and it costs nothing. Code edits copy heavily from the prompt; measure it, but expect α ≈ 0.85 on the copied spans with K=8, giving `E[tokens/iter] = (1 − 0.85⁹)/0.15 ≈ 5.1` with `c ≈ 0`. Decode drops to 7,780/5.1 ≈ **1,530 ms**. Total ≈ 2.1 s. Already a 4× win from a feature flag.

**Lever 2 — prefix caching, three tiers.** The system prompt and tool schemas (~2,500 tokens) are global and always warm. The repo context (~9,000 tokens) is session-scoped: cache it with session affinity so consecutive edits in the same session hit. The open file and selection (~2,500 tokens) change per request. Warm-case prefill becomes 2,500 tokens ≈ **85 ms**, down from 470 ms. Total ≈ **1.72 s**.

**Lever 3 — shrink the output.** This is where most teams stop thinking and shouldn't. If the model emits a full rewritten function, output is 350 tokens; if it emits a **diff or a structured edit** (a search/replace pair, or a line-range plus replacement), output is 60–90 tokens. Decode drops proportionally: 80 tokens at an effective 5.1 tokens/iter and 45 tok/s baseline ≈ **350 ms**. Total ≈ **520 ms**. **The biggest latency lever in a generation-bound workload is almost always generating fewer tokens**, and I'd push back on any design that hasn't tried it before reaching for a smaller model.

**Lever 4 — constrained decoding on the edit format**, which composes with speculation as discussed: the grammar makes structural tokens deterministic, raising α further and enabling fast-forwarding over fixed spans.

**Final budget:** gateway 40 ms + prefill 85 ms + decode ~350 ms + network 60 ms ≈ **535 ms p50**, with p95 around 900 ms once you account for cache misses on session start and occasional low-α generations. **Inside the 1.2 s target with margin**, which is what you want because the margin absorbs the cold-start cases.

**💰 Math:** at 3M edit requests/day. Decode GPU time per request: 350 ms of a 2×H100 decode slot ÷ effective batch. Rather than guess the batch, price the delta: the baseline design needs ~15.5× more decode GPU-seconds than the final one (8.3 s vs 0.535 s, decode-dominated). If the final design runs on 40 H100s at $2.50/hr = $2,400/day, the naive design needs roughly 620 GPUs = **$37,200/day**. That is **$10.8M/year**, and none of it required a different model.

**🗣 Say this in the room:** "Decode dominates, so I attack decode first: n-gram speculation for ~5 tokens per forward pass because code edits copy from the prompt, then a diff-style output format to cut the token count 4×, then three-tier prefix caching with session affinity to kill prefill, then a grammar on the edit format which raises acceptance further. That's 8.3 seconds to about 535 ms on the same weights."

### Design case: Perplexity-style grounded answer. 8 retrieved documents, streaming output, hard 800 ms TTFT. Go.

The constraint here is TTFT specifically, not total time — the product streams, so once the first token lands the user is reading and inter-token latency just has to beat reading speed (~30–50 tok/s is comfortably above it). So this is a **prefill and pipeline problem**, and speculation is the *second* lever, not the first.

**Where the 800 ms goes, budgeted:**

| Stage | Budget | Notes |
|---|---|---|
| Gateway + auth | 30 ms | |
| Query understanding / rewrite | 0 ms | Overlap it with retrieval, or use a tiny model |
| Retrieval (ANN + BM25 hybrid) | 120 ms | |
| Rerank (cross-encoder, top-50 → top-8) | 90 ms | Separate fleet, batched |
| **LLM prefill** | **≤ 450 ms** | The variable we control |
| First token emitted | 20 ms | |
| **Total** | **710 ms** | 90 ms of headroom |

**The prefill problem.** Eight documents at 2,000 tokens each is 16,000 tokens, plus a 3,000-token system prompt, plus the query. At ~30k tok/s that's ~630 ms — over budget on its own.

**Lever 1: cache the static prefix.** System prompt, tool schemas, citation-format instructions, few-shots — 3,000 tokens, globally shared, essentially always warm. Saves ~100 ms and costs nothing.

**Lever 2, the important one: the retrieved documents are the problem, and they're per-query.** Three options, in order of how much I like them:

- **(a) Document-level KV caching.** Documents come from a corpus with a **long tail and a very hot head** — popular pages are retrieved constantly. If you cache the KV of frequently-retrieved documents, a query whose top-8 includes 5 hot documents only prefills 3. This requires the engine to compose non-contiguous cached spans, which is not a stock feature everywhere and is the interesting engineering here. Where available, it's the biggest win.
- **(b) Retrieve less, rerank harder.** Eight 2,000-token documents is 16k tokens because the chunker is coarse. A better reranker over finer chunks might give you 8 × 600 = 4,800 tokens of *more relevant* context. Prefill drops to ~190 ms and answer quality typically **improves** — precision beats recall for grounded QA. This is the cheapest fix and the one I'd do first.
- **(c) Overlap prefill with retrieval.** Start prefilling the cached system prefix (and, if you can predict them, the top-1 or top-2 documents from a fast first-stage retrieval) while the reranker is still running. Same trick as input-time speculation in voice: bet on a prefix and eat the occasional discard.

**Lever 3: speculation for the decode phase.** Grounded answers quote heavily from context, so n-gram/prompt-lookup drafting is a strong fit — expect long accepted runs on quoted spans and entity names. This doesn't help TTFT but it materially cuts total time-to-complete-answer, which matters for the follow-up-question flow.

**Final:** with (b) plus caching, prefill ≈ 3,000 cached (10 ms read) + 4,800 fresh (160 ms) = **170 ms**, total TTFT ≈ **430 ms** — comfortably inside 800 ms with room for a p95 tail.

**⚠ Trap:** putting retrieved documents *before* the system prompt because that's how you'd naturally write it. Retrieval output is the most volatile layer in the prompt; anything you place after it is uncacheable. Order is stability, always: system → tools → few-shots → retrieved docs → query.

**💰 Math:** at 12M queries/day, 20k prompt tokens, the prefill compute for a 70B is 2 × 70e9 × 20,000 = 2.8e15 FLOPs/query ≈ 7 H100-seconds. Cutting the prompt from 20k to 8k tokens (levers 1 and 2b) is a 60% reduction: 12M × 4.2 H100-s = 50.4M GPU-s/day = 14,000 GPU-hours = **$35,000/day** of prefill compute reclaimed at $2.50/hr. Cutting context length is a cost lever before it's a latency lever.

### We enabled speculative decoding. p50 latency improved 40% but p99 got worse. Debug it.

Good symptom — it's diagnostic, because there are only a few mechanisms that produce that specific signature.

**Hypothesis 1: variance from α heterogeneity.** Speculation's benefit is a random variable. Requests in high-α regions fly; requests in low-α regions get the baseline rate *plus* the draft overhead. If 5% of your traffic is a domain the drafter doesn't cover (another language, math, an unusual format), those requests are now **slower than baseline** by roughly the draft cost — `K·c` per token — while everything else got faster. The mean and median improve; the tail, which is made of exactly those requests, degrades. **Check: plot the acceptance-rate distribution, not the mean.** If it's bimodal, this is your answer, and the fix is per-route or adaptive K with a hard disable below break-even.

**Hypothesis 2: KV pressure and preemption.** Speculation reserves `K+1` token-slots per sequence per step. At high KV utilization that extra reservation pushes the engine over the threshold where it must preempt a running sequence — swap it out or recompute it. A preempted request pays a full re-prefill and lands squarely in your p99. **Check: preemption counter, and KV utilization at p99 timestamps.** If preemptions rose when you enabled speculation, this is it. Fix: lower K, raise the free-block watermark, or lower `max_num_seqs`.

**Hypothesis 3: batch-size regime crossing at peak.** If p99 requests are the ones arriving at peak traffic — when batch is highest — then at those moments you're compute-bound, verification is not free, and speculation is a net cost. Everything at off-peak got faster; peak got slower. **Check: correlate p99 latency with concurrent-batch size.** Fix: the adaptive-K controller.

**Hypothesis 4: draft-model cold start.** If the drafter has no prefix cache, the first token of a long-prompt request pays a full draft-model prefill *serially before* the first draft step. On a 20k-token prompt with a 7B drafter, that's a real addition to TTFT on every cache-cold request — exactly the requests already in your tail. **Check: TTFT specifically, split by hit/miss, before and after enabling.** Fix: enable prefix caching on the drafter, or move to a self-drafting method.

**Hypothesis 5: it's not speculation at all.** Enabling a feature is a deploy, and a deploy is a cache flush plus a cold-start window plus possibly a config change. **Check: does p99 recover after one TTL window and one autoscale cycle?** If yes, you measured a deploy, not a feature.

**The order I'd actually check them:** 5 (free, rules out a whole class), then 1 (one dashboard query), then 3 (one correlation), then 2 (one counter), then 4. And the meta-point worth saying out loud: **speculative decoding widens the latency distribution by construction** — token yield per step is a random variable between 1 and K+1 — so a p99 regression alongside a p50 improvement is the *expected* shape, not an anomaly. The engineering question is whether the tail is inside your SLO, not whether it moved.

**🗣 Say this in the room:** "That signature is characteristic — speculation makes per-step token yield a random variable, so it compresses the median and stretches the tail. I'd check whether the acceptance-rate histogram is bimodal, whether preemptions rose from the extra K+1 KV reservation, and whether p99 requests correlate with peak batch size where verification stops being free. The fix is usually adaptive K with a hard disable below break-even, not turning the feature off."

### What's on your dashboard for a fleet running both prefix caching and speculative decoding?

One dashboard, organized by **the two phases and the two SLOs**, because that's how you reason about it and how you'll explain a regression to a VP.

**Prefill / TTFT half:**
- **Token-weighted cache hit rate** = `read / (read + write + uncached)`, per route, per model version.
- **p50/p95 TTFT split by cache hit and cache miss** on one axis. If the two lines converge, caching has stopped working regardless of what the hit-rate counter says.
- **Effective input price per 1M tokens** — the volume-normalized cost metric that alerts on every caching regression.
- **Distinct-prefix cardinality** per window — the leading indicator that someone added a per-request invalidator.
- **Prefix-cache eviction rate** and **KV pool utilization** — rising evictions with falling hit rate is cache thrash.

**Decode / ITL half:**
- **Acceptance rate** = `accepted / drafted`, segmented by route, language tag, drafter version, and target version. The segmentation *is* the alert.
- **Tokens per target forward pass** — the metric that maps directly to throughput, comparable against the theoretical `(1 − α^(K+1))/(1 − α)`. If measured is far below theoretical, your K is mistuned or your acceptance is correlated in a way the geometric model misses.
- **Distribution of accepted-run length**, as a histogram — bimodality means you need per-route config.
- **Current K** (if adaptive) and **fraction of steps with speculation disabled** by the guardrail.
- **p50/p95/p99 ITL**, and explicitly the **ITL variance** — speculation improves the mean and worsens the spread, and you need to see both.

**Shared / cost half:**
- **Running batch size distribution.** This is the variable that determines whether speculation is helping or hurting, and it's the one people don't chart.
- **Preemption rate.** Rises when speculation's K+1 reservation meets high KV utilization.
- **GPU-seconds per 1,000 completed requests**, split prefill/decode. This is the number that converts everything above into dollars and the one I'd put at the top.

**The two alerts I'd page on**, and only these two:
1. `effective_input_price_per_Mtok` up >20% week-over-week on any route → a caching regression.
2. `acceptance_rate` down >20% relative over 15 minutes on any segment → OOD collapse or a drafter/target version mismatch.

Everything else is a dashboard, not a page. **⚠ Trap:** alerting on raw GPU cost or raw token spend. Both are confounded by traffic volume, so they fire on a marketing campaign and stay silent on a 10× efficiency regression that coincides with a traffic dip. **Normalize every cost alert by volume** — that single discipline is the difference between an alert that gets acted on and one that gets muted in week two.

### Last one. An interviewer says: "our p95 is too slow and we can't change models." Give me your complete answer.

**🗣 Say this in the room** — this is the ladder, in the order I'd actually work it, and I'd open by asking which term of the latency equation dominates before proposing anything:

"Latency is `TTFT + n_output × ITL`, so the first question is which term owns the p95 — I'd want the TTFT/ITL split and the output-length distribution before I propose anything. Then:

**If TTFT dominates** (long prompt, short output — classifiers, routers, extraction, autocomplete):
1. **Prefix caching**, ordered by stability, breakpoints at tier boundaries. Free, ~90% off the prefix, and often the whole problem.
2. **Cut the prompt.** Better reranking over finer chunks usually improves quality *and* prefill. Context length is a latency lever.
3. **Chunked prefill** so a long prefill stops stalling in-flight decodes, and **prefix-affinity routing** so a warm cache is actually reachable.
4. **Input-time speculation** — start prefilling before the input is final where the product allows it.

**If ITL × n_output dominates** (long generation — chat, agents, code):
1. **Generate fewer tokens.** Diff instead of full rewrite, structured output instead of prose, a stricter length instruction. Almost always the biggest single win and almost always skipped.
2. **Speculative decoding.** N-gram/prompt-lookup first — zero cost, and if the workload copies from context it's 3–5×. A distilled draft model or an EAGLE-style head if n-gram acceptance is poor.
3. **Quantize the KV cache to FP8** — decode is bandwidth-bound, so halving the bytes moved per token is a direct ITL win and buys batch capacity too.
4. **Check the batch regime.** If we're compute-saturated, speculation costs throughput; that's a scheduler decision, not a config flag.

**Structural, if the above isn't enough:** disaggregate prefill and decode so each scales independently and the decode pool sits in the regime speculation likes; route by SLO class; and only then consider a smaller model or distillation, because that's the only step that requires re-running the eval suite."

**🏋 Drill — the capstone for this section, 45 minutes, unaided.** Given: a 70B on 2×H100, 40 tok/s baseline decode, ~28k prefill tok/s, an 11,000-token prompt (7,000 static / 4,000 per-request), 500-token output, 250,000 requests/day, $3/Mtok API alternative, $2.50/GPU-hr. Produce, on paper: (1) baseline p50 latency with the arithmetic; (2) the latency after prefix caching and after speculation at α=0.75, K=5, c=0.02, each computed separately and then combined; (3) daily GPU cost before and after; (4) the self-host vs API break-even in requests/day; (5) the three metrics you'd alert on and their thresholds. **Pass criteria:** every number has its derivation shown, you correctly note that speculation's benefit is batch-dependent and state the batch you assumed, and your break-even calculation includes the caching discount on the API side — most candidates forget that the API alternative also gets cheaper with caching, which moves the break-even substantially.
