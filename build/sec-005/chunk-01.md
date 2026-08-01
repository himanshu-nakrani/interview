### We're not going to test whether you memorized a model table. Tell me how you actually decide which model to put behind a feature.

The mental model I want you to hold: **model selection is a constraint-satisfaction problem with one hard constraint and eight soft ones, and the hard one is the only one you evaluate first.** The hard constraint is the capability floor — does this model clear the quality bar on *my* eval, on *my* data? Everything else (price, latency, context, caching, structured output, tool-calling, rate limits, retention terms) is a soft constraint you optimize *within* the set of models that clear the floor. Candidates fail this question by starting with price, because price is the number that's easiest to look up.

Here is the nine-axis rubric I actually run, in order:

1. **Capability floor.** Does it pass my task eval at the threshold the product needs? Binary gate. Everything below is a tiebreak.
2. **Latency budget.** TTFT and inter-token latency against the UX contract. A 200 ms TTFT product and a 4-second-TTFT product are different model markets.
3. **Price per million in / out**, weighted by *your* in:out ratio, not the vendor's marketing example.
4. **Context limit — advertised vs usable.** Two numbers, always.
5. **Caching semantics.** Prefix-cache discount, TTL, minimum cacheable prefix, what invalidates it. This routinely moves effective price by 5–10×, so it belongs above raw price in practice.
6. **Structured-output support.** Constrained decoding with a real schema guarantee, or "please output JSON" and a retry loop? The difference is a 3% malformed rate versus 0%.
7. **Tool-calling quality.** Not "does it emit valid JSON for a tool call" — does it pick the *right* tool, with the right arguments, and stop when it should.
8. **Rate limits and headroom.** Your requests-per-minute and tokens-per-minute ceiling at your current org tier, and how fast you can escalate.
9. **Data-retention and residency terms.** Often decided by someone who is not you, and it can veto axes 1–8.

**🗣 Say this in the room:** "I don't pick a model, I eliminate models. First I run my own eval to get the set that clears the capability floor — usually two or three. Then it's a constrained optimization over latency, effective price after caching, and context. The answer is almost never 'the best model'; it's 'the cheapest model that clears the floor with margin, plus a documented escalation path to a stronger one.'"

**⚠ Trap:** treating this as a static decision. It is a *routing policy*, re-evaluated on a cadence. The senior signal is saying "and here's the eval job that re-runs this decision every time a provider ships, so I find out from CI rather than from a support ticket."

**📅 Volatile:** every number in this section — lineups, prices, context limits, tier thresholds, engine feature matrices — has a shelf life measured in weeks. Everything is date-stamped. Re-verify before your loop; the arithmetic and the framework are what you're being tested on.

### "Best model wins" became "best fit wins" at some point. What actually changed?

Three things changed at once, and the interesting part is that none of them was a capability breakthrough.

**First, the capability floor for common tasks dropped below the cheap tier.** In 2023, if you wanted reliable JSON extraction from a messy invoice, there was exactly one model that could do it. By 2025 the small tier could do it, and the frontier tier was 5–20× the price for output you couldn't distinguish on your eval. Once a task's difficulty falls below the floor of the cheap tier, spending frontier money on it is pure waste — and *most production tasks are below that floor*. The frontier tier stopped being where you build and became where you escalate.

**Second, price stopped being a single number.** With prefix caching, batch tiers, and reasoning-token billing, "cost per request" became a function of your traffic shape rather than a property of the model. A model with a higher sticker price and a 90% cache discount on a 12k-token system prompt is cheaper for an agent than a model with a lower sticker price and no caching. You cannot rank models by price anymore; you can only rank *deployments*.

**Third, the axes became genuinely orthogonal.** The model with the best coding eval is not the one with the best structured-output guarantee, is not the one with the highest TPM ceiling on your account tier, is not the one your legal team will sign for. In 2023 those correlated because there was one good model. Now they don't.

**💰 Math:** a concrete illustration of why "best" is the wrong word. Take a classification endpoint at 5M requests/month, 800 input tokens and 40 output tokens each. On a frontier-tier model at $5/Mtok in and $25/Mtok out (**📅 Volatile:** Opus-class pricing as of mid-2026): (5e6 × 800 / 1e6 × $5) + (5e6 × 40 / 1e6 × $25) = $20,000 + $5,000 = **$25,000/month**. On a small-tier model at $1/$5: $4,000 + $1,000 = **$5,000/month**. If your eval shows the small model at 96.1% and the frontier at 96.4% on this task, you are paying $20,000/month for 0.3 points that is inside your confidence interval. That is the entire "best fit" argument in one calculation.

**🗣 Say this in the room:** "'Best' assumes a total ordering that stopped existing around the point where the cheap tier cleared most production tasks. What I optimize is cost per *resolved task* subject to a quality floor and a latency SLO — and those three quantities point at different models depending on the workload."

### Walk me through establishing the capability floor. How do you find it without just guessing?

The mental model: the capability floor is not a property of the model, it is a property of **the pair (your task, your quality threshold)** — so it can only be measured, never looked up. Public benchmarks tell you the ordering on *someone else's* task distribution, and the correlation with your task is unknown and usually weaker than you'd like.

Mechanism, as a procedure I'd actually run in a week:

**Step 1 — build the eval set before you touch a model.** 150–300 examples sampled from real traffic (or the closest proxy you have), stratified so the hard cases aren't drowned out by the easy head of the distribution. Label them. This is the expensive part and there is no shortcut; if you skip it, every subsequent decision is vibes.

**Step 2 — define the threshold as a product decision, in writing.** "≥95% exact match on the extracted fields, with ≤0.5% hallucinated fields" is a threshold. "Good quality" is not. The hallucination-rate side matters more than the accuracy side for most products, because a wrong-but-confident answer costs more than a refusal.

**Step 3 — descend, don't ascend.** Start at the strongest model you can afford, confirm the task is solvable *at all*. If the frontier model fails your threshold, the problem is your prompt, your retrieval, or your task decomposition — not your model choice, and swapping models will not save you. Once the frontier model passes, walk *down* the tier ladder until something fails. The cheapest passing model is your floor.

**Step 4 — measure the gap with statistics, not eyeballs.** On a 200-example set, a 96.4% vs 96.1% difference is 0.6 examples. Use a paired bootstrap over the per-example outcomes; if the 95% CI on the difference straddles zero, the models are indistinguishable *on your evidence* and you take the cheaper one. Report the CI in your decision doc.

**Step 5 — re-run on a cadence and on every provider release.** Wire it into CI as a scheduled job, not a manual notebook.

**⚠ Trap:** running the descent with a prompt tuned for the frontier model. Small models are more sensitive to prompt structure — they benefit from more explicit formatting instructions, fewer implicit inferences, and few-shot examples the big model didn't need. If you hand a small model a terse prompt written for a reasoning model and conclude "it can't do this," you've measured your prompt, not the model. Budget an hour of prompt adaptation per candidate before you call the floor.

**🏋 Drill (60 minutes, unaided):** take any task you've shipped. Write the threshold statement in one sentence with two numbers in it. Build a 50-example labeled set. Run three model tiers. Produce a table with accuracy, 95% CI, p50 latency, and cost per 1k requests. Pass criterion: you can state which model you'd ship and defend it against "why not the better one?" using only your table.

### How do you turn a latency SLO into a model constraint? Be specific about the metrics.

The mental model: **a chat product's latency contract is two numbers, not one, and they're bounded by different physics.** Time-to-first-token is dominated by queueing, prefill compute, and network round-trip; inter-token latency is dominated by memory bandwidth at the serving layer and is essentially fixed per model per deployment. Your product SLO must be decomposed onto both, because a model can be great at one and useless at the other.

The vocabulary swap you need to make from backend: p95 request latency becomes **TTFT** (time to first token), **ITL/TPOT** (inter-token latency / time per output token), and **total generation time** = TTFT + ITL × output_tokens. Throughput becomes **tokens/sec** and, more usefully, **goodput** — requests/sec that actually met the SLO, which is the only throughput number worth quoting.

The procedure:

1. **Write the UX contract.** Streaming chat: TTFT under ~500 ms feels instant, under ~1 s is fine, over 2 s users start reloading. Non-streaming API in a request path: total generation time is what matters and there is no hiding behind streaming. Autocomplete in an editor (Cursor-class): TTFT budget is 100–300 ms and the model tier is decided by that alone.
2. **Convert to a token budget.** If your SLO is 3 s total, TTFT measures 600 ms, and ITL is 15 ms/token, you have (3000 − 600)/15 = **160 output tokens**. That is your real max output length, and it is a *design constraint on the prompt*, not something you discover in prod.
3. **Check whether reasoning tokens fit.** A model that emits 2,000 reasoning tokens before its first visible token has a TTFT-equivalent of 2000 × 15 ms = 30 seconds from the user's perspective, even if the API's technical TTFT is 400 ms. For any interactive surface, a reasoning model is a different product, not a drop-in upgrade.
4. **Measure at your p99 input length, not your median.** Prefill is roughly linear in input tokens; a 32k-token context request has a TTFT several times your median. If your context is retrieval-augmented and unbounded, your TTFT distribution has a long tail by construction.

**📐 Numbers you must know:** ITL at 15–30 ms/token is the range a well-served mid-size model lands in, which is 30–65 tokens/sec — comfortably faster than human reading speed (~5 tokens/sec of *reading*, ~250 wpm). This is why streaming saves you: past roughly 10 tokens/sec, further ITL improvement is invisible to a reading user, and your entire remaining latency budget should be spent on TTFT. Derive it yourself: 250 words/min ÷ 60 ≈ 4.2 words/s × ~1.3 tokens/word ≈ 5.5 tokens/s.

**⚠ Trap:** quoting the provider's advertised tokens/sec. That figure is measured at their batch size on their hardware with a short prompt. Your number depends on your prompt length, your concurrency, and whether you got a cache hit. Measure it from your own client, at your own p95 input length, over a week, from the region you deploy in.

### Build me a per-request cost function from a pricing table. I want to see the code.

Mental model: **a provider's price list is not a price, it's a rate card with four or five distinct rates, and which one applies depends on the state of a cache you don't control.** Your cost function must therefore take the *usage object returned by the API*, not your estimate of the request, as its input. The single most common cost-modeling error I see in review is multiplying an estimated token count by a single price.

The rate structure you must model, per provider:

- **Uncached input** — full input rate.
- **Cache write** — a premium over base input (Anthropic charges ~1.25× base for the 5-minute TTL and ~2× for the 1-hour TTL; **📅 Volatile**).
- **Cache read** — a deep discount, roughly 0.1× base input on Anthropic, and providers differ (some are 0.25×, some 0.5×). This is the single highest-leverage number in the whole table.
- **Output** — typically 4–5× the input rate.
- **Reasoning/thinking tokens** — billed as output, and *invisible in your response text*. This is where budgets die.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Rates:
    """USD per 1M tokens. Verify against the provider pricing page before use."""
    input_uncached: float
    input_cache_write: float   # e.g. 1.25 * input_uncached for a 5m TTL
    input_cache_read: float    # e.g. 0.10 * input_uncached
    output: float              # reasoning tokens bill at this rate too

def request_cost_usd(usage, r: Rates) -> float:
    """`usage` is the provider's usage object, not your estimate."""
    uncached = getattr(usage, "input_tokens", 0)
    write    = getattr(usage, "cache_creation_input_tokens", 0)
    read     = getattr(usage, "cache_read_input_tokens", 0)
    out      = getattr(usage, "output_tokens", 0)   # includes thinking tokens
    return (uncached * r.input_uncached
            + write  * r.input_cache_write
            + read   * r.input_cache_read
            + out    * r.output) / 1_000_000
```

Two properties of this function that matter in review. First, **`input_tokens` on Anthropic is the uncached remainder only** — total prompt size is `input_tokens + cache_creation + cache_read`. If you log `input_tokens` as "prompt size" you will report an agent that ran for an hour as having a 4k-token prompt, and your dashboards will be quietly wrong. Second, the function must be applied per response and *summed over the whole agent trajectory*, because a single user-visible task is 10–40 API calls in an agent loop. Cost per API call is a meaningless unit; **cost per resolved task** is the unit your director cares about.

**💰 Math:** a 12k-token system prompt at $3/Mtok input, called 200k times/day. Uncached: 12,000 × $3/1e6 = $0.036/call → $7,200/day → **$216k/month**. With a prefix cache at 0.1× and a ~90% hit rate: hits cost 12,000 × $0.30/1e6 = $0.0036, misses cost 12,000 × $3.75/1e6 = $0.045 (write premium). Blended: 0.9 × $0.0036 + 0.1 × $0.045 = $0.00774/call → $1,548/day → **$46.4k/month**. That is a **$170k/month** delta from one `cache_control` marker, and it is entirely arithmetic you can do at a whiteboard.

**⚠ Trap:** forgetting that reasoning tokens bill at the output rate and are not returned to you as text. A request whose visible answer is 80 tokens can bill 3,000 output tokens. If your cost dashboard is built from `len(response_text)`, you will under-report by 30× on reasoning workloads and discover it on the invoice.

### Context limits — you said "advertised vs usable, always two numbers." Defend that.

The mental model: **the advertised context limit is a memory-allocation guarantee, not a comprehension guarantee.** The provider is telling you the request will not be rejected. It is not telling you the model will use the middle of that window. Those are different claims and only one of them is contractual.

The mechanism behind the gap is well documented. Retrieval accuracy over long contexts degrades with position — the "lost in the middle" effect, where facts placed near the beginning or end of a long prompt are recovered reliably and facts in the middle are not (**📄 Paper:** Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts* — showed a U-shaped accuracy curve over document position in multi-document QA and key-value retrieval, and reframed long context as a quality question rather than a capacity question). Beyond that, degradation compounds when the task requires *multiple* facts, or reasoning over retrieved facts rather than quoting them, or discriminating between similar distractors.

So the two numbers are:

- **Advertised**: what the API accepts. 200k, 1M, 2M — read off the docs, date-stamped.
- **Usable**: the length beyond which *your* task's accuracy drops below *your* threshold. Measured, on your corpus, with your task.

How you measure the second one: build a length-stratified eval. Same questions, same gold answers, but with the supporting evidence embedded in contexts of 4k / 16k / 64k / 200k tokens, with position of the evidence randomized (not always at the end — that's how you accidentally certify a model that only reads the tail). Plot accuracy against context length. The knee in that curve is your usable limit.

**⚠ Trap:** using needle-in-a-haystack as your long-context eval and declaring victory. NIAH asks the model to find one verbatim, semantically out-of-place string. It is a weak proxy: models that ace NIAH at 1M tokens can fail multi-fact synthesis at 32k. If a candidate tells me "the model gets 100% on needle-in-a-haystack at 1M," my follow-up is "and what's its accuracy when the answer requires combining three facts from three different positions?" That is the question the product actually depends on.

**💰 Math:** the cost argument against long context is usually stronger than the quality argument. Stuffing 200k tokens of context into every request at $3/Mtok costs $0.60 per call in input alone. At 50k calls/day that is $30,000/day — **$900k/month**. A retrieval step that selects 4k relevant tokens instead costs $0.012/call → $600/day → $18k/month, plus embedding and index cost of maybe $2k/month. Long context is not free RAG; it's RAG's expensive cousin that you use when the corpus genuinely doesn't decompose.

**🗣 Say this in the room:** "I quote context as two numbers: advertised and usable-on-my-corpus. Advertised is what the API accepts; usable is where my length-stratified eval's accuracy crosses my threshold with evidence positioned randomly. For most of my workloads those differ by a factor of two to eight, and the difference is the entire justification for the retrieval layer."

### Draw me the full price surface for a modern provider — every distinct token rate — and tell me which line dominates in an agent.

There are five rates, and engineers routinely model two.

| Rate | Typical multiple of base input | Why it exists |
|---|---|---|
| Uncached input | 1× | Prefill compute you actually did |
| Cache write | 1.25× (short TTL) → 2× (long TTL) | You occupied KV memory for future reuse |
| Cache read | ~0.1× (Anthropic) — verify per provider | You skipped prefill; provider only pays memory |
| Output | 4–5× input | Decode is bandwidth-bound and doesn't batch as well |
| Reasoning / thinking | Billed as output | It *is* output, you just don't see it |

**📐 Numbers you must know (📅 Volatile — Anthropic first-party rates as cached 2026-06-24, re-verify):** Opus-class $5 in / $25 out per Mtok; Sonnet-class $3 / $15; Haiku-class $1 / $5. Cache read ≈ 0.1×; cache write 1.25× at 5-minute TTL, 2× at 1-hour TTL. Batch tier ≈ 50% off. The *ratios* are the durable part: output ≈ 5× input, cache read ≈ 0.1× input, batch ≈ 0.5×. Memorize the ratios, look up the absolutes.

Now, which line dominates? It depends entirely on your architecture, and being able to say which is a strong seniority signal:

- **Single-turn classification/extraction** (short output): *uncached input* dominates. Optimization = shorter prompts, cheaper tier, batch tier.
- **Chat with a long system prompt**: *cache reads* dominate after warm-up, and your job is protecting the cache from invalidation. Optimization = frozen prefix.
- **Agent loop with tools**: **output tokens dominate, and it isn't close.** Every turn re-sends the whole growing transcript (cheap, cached) and generates new reasoning + tool arguments (expensive, uncached, at 5× input rate). A 20-turn agent generating 600 output tokens/turn burns 12,000 output tokens = 12,000 × $15/1e6 = **$0.18 per task in output alone**, versus perhaps $0.04 of cached input.
- **Batch document processing**: *input* dominates and the batch tier is free money.

**💰 Math (the break-even for prompt caching, which you should be able to derive live):** cache write costs 1.25×, cache read costs 0.1×, uncached costs 1×. Two calls with caching = 1.25 + 0.1 = 1.35×; two calls without = 2×. **Caching pays from the second call at 5-minute TTL.** At 1-hour TTL: 2 + 0.1 = 2.1× for two calls versus 2× uncached — that *loses*; you need three calls (2.2× vs 3×) to break even. So the TTL choice is a traffic-shape decision: bursty traffic with gaps longer than the short TTL justifies the long one; steady traffic never does.

**⚠ Trap:** enabling the 1-hour TTL "to be safe" on a high-traffic endpoint. You've just doubled your cache-write cost on a prefix that would have stayed warm anyway. The rule I enforce in review: long TTL requires evidence of inter-request gaps exceeding the short TTL, from your own latency-between-requests histogram.

### When do you use the batch tier, and when does the 50% discount lose you money?

Mental model: **the batch tier is a spot market for GPU capacity — you're telling the provider "schedule me whenever you have a trough," and they pay you ~50% for the option.** It is the exact same trade as spot instances, and you should reason about it the same way: great for work with a soft deadline, catastrophic for work in a request path.

Mechanism (Anthropic's Message Batches API as a concrete instance; OpenAI's Batch API is structurally similar): you `POST` a list of requests, each tagged with a `custom_id`. You poll for `processing_status == "ended"`. You stream results. Limits as of mid-2026: up to 100,000 requests or 256 MB per batch, most complete within an hour, **maximum 24 hours**, results retained 29 days (**📅 Volatile**). Every Messages API feature works inside it — tools, vision, prompt caching.

Three rules I enforce:

1. **Key results by `custom_id`, never by position.** Results come back in arbitrary order. Indexing by array position is a bug that passes review, passes staging with 10 items, and silently mismatches 40,000 labels in prod. This is the single most common batch bug.
2. **Handle four result types, not one.** `succeeded`, `errored`, `canceled`, `expired`. `expired` means the 24h window elapsed — those requests never ran and must be resubmitted. A pipeline that only branches on `succeeded` silently drops them.
3. **Never put the batch tier in a path with a user waiting.** "Most complete within an hour" is not an SLA you can build a UI on.

Where batch **loses** you money or time:

- **Iterating on prompts.** A 20-minute batch turnaround on a prompt-tuning loop costs you a day of engineering time to save $40 of API spend. Use the sync API while iterating; switch to batch when the prompt is frozen.
- **Anything with a retry-and-refine loop.** Two dependent batch rounds is potentially a 48-hour pipeline.
- **When it breaks your cache.** Batch requests may not hit the same warm prefix cache your sync traffic maintains, so a workload whose real saving was a 90% cache-read rate can be *more* expensive at "50% off" uncached. Do the arithmetic before assuming.

**💰 Math:** a nightly eval suite, 8,000 examples × (3,000 input + 400 output) tokens against a $3/$15 model. Sync: (8000 × 3000 / 1e6 × $3) + (8000 × 400 / 1e6 × $15) = $72 + $48 = **$120/night** = $3,600/month. Batch at 50%: **$60/night** = $1,800/month. Saving $1,800/month for a job that runs at 2am and is read at 9am — obviously correct. Now the counter-case: that same suite with a shared 20k-token cached preamble across all 8,000 requests. Cached-sync input = 8000 × 20,000 × $0.30/1e6 = $48 versus batch-uncached 8000 × 20,000 × $1.50/1e6 = $240. **The "discounted" path costs 5× more.** Always compute against your *cached* baseline, not your list-price baseline.

### How would you evaluate structured-output support as a selection axis? What are you actually comparing?

The mental model: **there are three qualitatively different things vendors all call "JSON mode," and only one of them gives you a guarantee.**

1. **Prompted JSON.** You ask for JSON in the prompt. No enforcement. Failure rate on hard schemas is single-digit percent and correlates with exactly the inputs you care about (long, weird, adversarial). You need a parse-and-retry loop, and retries cost you a full extra generation.
2. **JSON mode.** The decoder is constrained to emit syntactically valid JSON. You get well-formed JSON — but not *your schema*. Missing required fields, wrong types, and invented keys all still happen.
3. **Constrained decoding against your schema** (Anthropic's `output_config.format` with a JSON schema, or `strict: true` on a tool definition; OpenAI's Structured Outputs with `strict: true`). The provider compiles your schema into a grammar/automaton and masks the logits at each decode step so that only tokens keeping the output on a valid path have nonzero probability. **Schema conformance becomes a property of the sampler, not of the model.** This is the only one where "0% malformed" is a defensible claim.

What to compare, concretely:

- **Which JSON Schema features are supported.** Universally the constrained-decoding implementations restrict the schema language. Typically supported: object/array/string/integer/number/boolean/null, `enum`, `const`, `anyOf`, `$ref`/`$defs`, and common string `format`s. Typically *not*: recursive schemas, numeric bounds (`minimum`/`maximum`), string length bounds, complex array constraints. `additionalProperties: false` is usually mandatory on every object. If your schema uses `minimum: 0`, find out whether the provider drops it silently (and validates client-side) or rejects the request — those are very different failure modes.
- **First-request latency.** New schemas incur a one-time compilation cost, then get cached (24h on Anthropic; **📅 Volatile**). If you generate schemas dynamically per request you pay that penalty every time — a real, measurable regression that looks like random latency spikes.
- **What happens on `max_tokens` or a refusal.** Constrained decoding guarantees the output is *on a valid path*, not that it *completed*. Hitting the token cap gives you truncated-but-locally-valid JSON. Your parser must handle it.
- **Interaction with other features.** On Anthropic, structured output is incompatible with citations (400) and with message prefilling; strict tool use is incompatible with programmatic tool calling and forced `tool_choice`. These matrices are real and they bite in integration, not in the prototype.

**⚠ Trap:** believing schema conformance implies semantic correctness. Constrained decoding guarantees the model emits `{"amount": 4200, "currency": "USD"}` in the right shape. It guarantees nothing about whether 4200 is the right number. I've watched teams celebrate "we eliminated our JSON errors" while their field-level accuracy sat at 82%. Constrained decoding moves your failure mode from *loud* (parse error) to *silent* (wrong value), which is strictly worse for observability unless you add field-level evals at the same time.

**🗣 Say this in the room:** "I treat structured output as a sampler feature, not a model capability — the provider masks logits against a compiled grammar, so conformance is guaranteed and correctness isn't. I select on which schema features survive compilation, and I always pair it with a field-level accuracy eval, because the whole point of enabling it is that malformed output stops being the thing that alerts me."

### Tool-calling quality — how do you measure it, rather than trusting a leaderboard?

Mental model: **"tool calling" bundles four separable skills, and models fail at different ones.** A model that emits perfectly-typed tool arguments 100% of the time and calls `web_search` for a question it already knows the answer to is a bad tool-caller, and no schema-validity metric will tell you that.

The four skills, each of which needs its own metric:

1. **Selection** — given a tool set and a request, does it call the right tool (or correctly call none)? Metric: accuracy over a labeled set that deliberately includes *no-tool-needed* cases. The no-tool cases are where over-eager models fail.
2. **Argument construction** — are the arguments not just schema-valid but *semantically* right? Metric: exact match on arguments against gold, field by field.
3. **Composition** — multi-step: does it chain tools correctly, pass outputs forward, and recover when a tool returns an error? Metric: end-to-end task success over a set of trajectories requiring 2–5 calls, plus a specific recovery-rate metric on injected tool errors.
4. **Termination** — does it stop when the task is done, or loop? Metric: distribution of turns-to-completion, with a hard cap; the mean is less interesting than the 95th percentile and the fraction that hit the cap.

The eval harness is a mock tool server with deterministic responses. Do not evaluate against live tools; you cannot separate model failure from API flakiness, and you'll spend a week chasing a rate limit.

```python
# Sketch of the harness shape — deterministic mocks, per-skill scoring.
CASES = [
    {"q": "What's 17% of 4,280?",       "gold_tool": "calculator", "gold_args": {"expr": "0.17*4280"}},
    {"q": "What's the capital of Peru?", "gold_tool": None},        # no-tool case: over-eager models fail here
    {"q": "Email the Q3 report to Sam",  "gold_tool": "send_email", "gold_args": {"to": "sam@…", "attach": "q3.pdf"}},
]
# score: selection accuracy, arg exact-match | correct selection, turns-to-completion p95.
```

Two things that move tool-calling quality more than model choice, and which you should raise unprompted because they signal you've shipped this:

- **Tool descriptions that state *when* to call, not just what the tool does.** "Get current weather" versus "Get current weather. Call this when the user asks about conditions in a named location; do not call it for historical or forecast questions beyond 7 days." The second reliably raises should-call rate on models that are conservative about tools.
- **Tool-set size.** Past roughly 15–20 tools, selection accuracy degrades measurably on every model I've tested. The fix is not a better model, it's namespacing, a router, or deferred tool loading (tool search) so only relevant schemas are in context.

**⚠ Trap:** measuring tool calling only on happy paths. The production failure mode is a tool that returns an error or an empty result and a model that either ignores it and hallucinates the answer, or retries the identical call forever. Inject error results into 20% of your eval trajectories. This single change is what separates an eval that predicts production from one that doesn't.

### Where does data retention and residency sit in your selection framework, and how has it actually vetoed a choice for you?

Mental model: **retention terms are a hard constraint that arrives late and has no engineering workaround.** They belong at the top of your framework even though they're the least technical axis, because discovering them in week six of a project costs you the project.

The axes that actually get negotiated:

- **Zero data retention (ZDR).** Provider does not persist prompts/completions beyond the request. Standard for enterprise contracts. The trap: **ZDR is not universally compatible with every model or feature.** As of mid-2026, Anthropic's Fable-class model requires 30-day retention and returns `400 invalid_request_error` on every request from a ZDR org — a perfectly valid payload that fails for a contract reason (**📅 Volatile**). If you're debugging a blanket 400 with no obvious request problem, check the org's retention configuration before you touch the payload. Similar coupling exists elsewhere: abuse-monitoring caches, prompt-caching persistence, and batch-result retention (29 days on Anthropic) are all *storage*, and a strict ZDR reading can exclude them.
- **Training on your data.** All the major API providers state that business/API traffic is not used for training by default. Consumer tiers are a different contract. Your legal team will want this in the enterprise agreement, not a docs page.
- **Residency.** Which region the inference physically runs in. Some providers expose it as a request parameter or an endpoint; some only through a cloud partner (Bedrock/Vertex/Foundry in a specific region). If you're serving EU healthcare or EU financial customers, this decides your provider before any eval runs.
- **Subprocessor chain.** Going through a cloud reseller adds a party to your DPA. Sometimes that's how you get approved (the customer already trusts AWS); sometimes it's what blocks you.

The engineering consequence you should raise: **cloud-partner deployments are feature-lagged.** Running Claude on Bedrock or Vertex rather than first-party gets you the core Messages API, but a documented subset of everything else — as of mid-2026, Message Batches, the Models API, several server-side tools, and Managed Agents are first-party-only or partner-restricted (**📅 Volatile — check the platform availability matrix, it changes every release**). So "legal says Bedrock" isn't a procurement footnote, it's an architecture constraint that can delete a feature you designed around.

**🗣 Say this in the room:** "Retention and residency are the first thing I check, not the last, because they're the only axis with no engineering workaround. And I check them against the *feature* matrix, not just the model list — a cloud-partner deployment that satisfies legal may not support the batch endpoint or the caching semantics my cost model assumed, and I'd rather find that out in week one."

**⚠ Trap:** assuming the provider's headline retention policy applies to every surface. Prompt caching stores your prefix somewhere for the TTL. Batch results are retained for weeks. Abuse-monitoring pipelines have their own window. Each of those is a separate line in the DPA, and "we use ZDR" is not an answer to "where is that 20k-token system prompt for the next hour?"

### It's the week before your loop. Walk me through your refresh checklist.

Mental model: **treat the volatile layer like a dependency lockfile — you don't memorize it, you re-resolve it, and you timestamp the resolution.** The interviewer is not testing whether you know today's price. They're testing whether you have a *process* that would have known last month's and will know next month's. Saying "I re-verify this before every loop, here's my checklist, and as of last Tuesday it was X" is a stronger answer than a confidently-recited stale number.

The checklist, in the order I run it (budget: 90 minutes):

1. **Provider model lineups and IDs.** Pull the live list programmatically, not from a blog post — Anthropic exposes `GET /v1/models` returning `id`, `display_name`, `max_input_tokens`, `max_tokens`, and a `capabilities` tree. That's your ground truth for context window and feature support, and it's newer than any doc page.
2. **Pricing pages, all three providers.** Record input, output, cache-read, cache-write, and batch rates. Compute the ratios (output/input, cache-read/input) and check whether they moved — the ratios moving is more interesting than the absolutes moving.
3. **Deprecation and retirement schedules.** Which model IDs you depend on have a retirement date inside the next two quarters.
4. **Feature/parameter deltas.** The thinking-control surface, sampling parameters, prefill support, and structured-output parameter names have all changed under stable-looking model families. Read the migration guide for the newest model in each family — it is the single densest source of "what changed."
5. **Rate-limit tier tables.** Your org's current RPM/ITPM/OTPM and what the next tier requires.
6. **Serving-engine feature matrices** if you're touching infra: which of vLLM / SGLang / TensorRT-LLM currently support the quantization format, attention variant, speculative decoding mode, and structured-output backend you're planning to claim in a design round.
7. **Open-weight releases** in the last 90 days and their licenses.
8. **Regulatory dates** — the EU AI Act milestone calendar, since one of them is probably within six months of your loop.

**🗣 Say this in the room, verbatim, when you don't know a current number:** "I don't want to quote you a stale price — that table moves monthly and I re-verify it before I make a decision. What I can give you is the structure: output is roughly 5× input, cache reads run around a tenth of base input, and batch is about half. Those ratios have been stable for two years and they're what the architecture decision actually turns on. If you want, I'll do the arithmetic with your real numbers."

That answer converts "I don't know" into a demonstration of judgment, and interviewers reward it. The failure mode is the opposite: confidently quoting a 2024 price in a 2026 room. It signals you haven't shipped recently, and it's unrecoverable — every subsequent number you say gets discounted.

**🏋 Drill (25 minutes, timed, unaided):** from memory, write down the five distinct token rates a modern provider charges, the approximate multiple of base input for each, the caching break-even in number-of-calls for a short TTL and a long TTL, and the three most common reasons a prefix cache silently stops hitting. Then verify against the live docs and mark every place you were wrong. Pass criterion: at most one wrong ratio, and all three cache invalidators correct.
