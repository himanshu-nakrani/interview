### Make this endpoint idempotent. It calls a model and writes the result to a database.

This is the question where backend instinct misleads most cleanly, so let me name the misleading part first: **an idempotency key guarantees you perform the operation once. It does not guarantee that a retried request would have produced the same result, and with a model, it emphatically would not.** Everything else follows from taking that seriously.

Your normal contract is "same key, same effect, and the second call returns the first call's response." That contract still holds and you build it the same way — a table keyed on `(tenant_id, idempotency_key)` with a unique constraint, holding request-hash, status and the serialized response, written in the same transaction as the effect. Nothing about LLMs changes that machinery. What changes is what you are protecting.

**What is actually nondeterministic, precisely.** Temperature > 0 samples, so outputs differ run to run. But even at temperature 0 you should not promise determinism: floating-point reduction order varies with batch composition on the GPU, and your request's batch-mates change every call, so logits differ in the last bits and occasionally flip an argmax. Provider-side model updates, silent routing between hardware generations, and MoE routing under different batching all add variance. **📅 Volatile:** some providers expose a `seed` and a `system_fingerprint` for best-effort reproducibility; treat it as best-effort, never as a contract, and verify current behaviour.

**The consequences to design around:**

*The stored response is the source of truth, not a re-computation.* If the client retries with the same key, you return the stored bytes. You never re-call the model to "reproduce" it. This is standard, but the stakes are higher because re-calling is expensive and would produce a *different* answer, which is a far worse bug than a duplicate charge — the user sees the answer change under them.

*In-flight concurrency needs a real lock, not a check-then-act.* Two retries arriving 50 ms apart both find no row and both call the model. Insert the key row with status `in_progress` *first*, let the unique constraint reject the loser, and have the loser poll or return 409 with `Retry-After`. Otherwise you pay twice and race on the write.

*Idempotency at the wrong layer is worse than none.* Making the outer HTTP handler idempotent while the agent loop inside retries individual tool calls means the *effects* — the email sent, the refund issued, the row inserted — are not covered. Every tool with a side effect needs its own idempotency key, derived deterministically from the run ID and the step index, and passed through. This is the part candidates miss.

*Some effects are inherently non-idempotent and the model chooses them.* If the model decides to call `send_email` twice with slightly different bodies, no key at the HTTP layer saves you. The control is at the tool boundary: dedupe on a semantic key (`run_id + recipient + intent`), or require confirmation for irreversible actions, or make the tool a *proposal* that a deterministic executor applies once.

**⚠ Trap:** hashing the request body to detect "same request, same answer" and treating a hash miss as a client error. Prompts contain timestamps, session IDs and retrieved context that legitimately change between a request and its retry, so the hash differs while the intent is identical. Key on an explicit client-supplied idempotency key; use the body hash only to *detect conflict* (same key, materially different body → 422), never as the key itself.

**🗣 Say this in the room:** "Idempotency here protects the side effects, not the output — I can guarantee we only charge the card once, I cannot guarantee that a regenerated answer is byte-identical, and I wouldn't design anything that depends on it. So the key goes on the write and on each side-effecting tool call, and the stored response is authoritative on replay rather than something we recompute."

### An engineer on your team wraps every model call in a three-attempt retry with backoff. What do you say in code review?

I block it, and the reason is arithmetic rather than taste.

In your normal services a retry costs a little CPU and a little connection churn, so "retry 3× with exponential backoff" is close to free and is correctly the default. Here every attempt re-runs the full generation, and the failures that trigger retries are *correlated across all your callers* — a 429 or an overloaded upstream means everyone is retrying at once. Retries stop being resilience and become a load amplifier pointed at a resource that is already saturated, with a billing meter attached. This is congestion collapse, which you already know, with the novel feature that it also empties your budget.

**💰 Math on the amplification.** The killer is that retry layers *multiply*, and modern stacks stack them without anyone deciding to. Provider SDKs retry internally by default (commonly 2 retries = 3 attempts). Your `tenacity` decorator adds 3 attempts. Your workflow engine retries the activity. Nobody wrote "27 attempts" but that is the ceiling. In steady state with a 2% failure rate the expected multiplier is ≈ 1.02 and nobody notices. During an incident where the failure rate hits 50%, each layer's expected attempts is 1/(1−0.5) = 2, and two nested layers give **2 × 2 = 4× the spend, precisely during the window when the provider is telling you to back off.** At a normal $5,000/day that is $20,000/day burned making the outage worse. And the retried traffic is what keeps the failure rate at 50%.

**What I ask for instead, as a review checklist:**

1. **Exactly one retry layer.** Disable the SDK's internal retries or disable yours. Name in a comment which one is authoritative.
2. **Retry only what is retryable.** 429 and 5xx, yes. 400, 401, 403, content-policy refusals, schema violations and context-length errors: never — the identical request will fail identically and you have paid for nothing twice.
3. **Respect `retry-after`.** Exponential backoff that ignores the server's explicit instruction is not backoff.
4. **Full jitter.** `sleep(random.uniform(0, min(cap, base * 2**attempt)))`. Without jitter, retries from a thousand callers re-synchronize into a thundering herd on every cycle.
5. **A concurrency limiter that shrinks on failure.** AIMD: additively increase in-flight capacity on success, multiplicatively halve it on 429. A circuit breaker is the coarse version and is also acceptable; a fixed semaphore is not.
6. **A per-request cost ceiling, enforced.** Retries count against the same budget as the original. A run that has spent its budget does not get retried, it gets failed and recorded.
7. **A budget for the retry itself.** For a streaming response that failed at token 400 of 500, retrying from scratch costs a full generation to recover 20% of one. Prefer resuming with the partial output as context, or degrade to a cheaper model, or return the partial with a marker.

**⚠ Trap:** the timeout-abort double charge. If you set an aggressive client timeout and abort, the generation may have already run to completion server-side, and whether you are billed for tokens you never received depends on the provider and on whether you were streaming. Assume you are billed. That converts an aggressive timeout policy into a silent cost line that never appears in your success metrics — you pay full price for every request you threw away.

**🗣 Say this in the room:** "Retries here are a cost decision, not a resilience default. I allow exactly one retry layer, only on 429 and 5xx, with full jitter and `retry-after` respected, sitting behind an AIMD concurrency limiter — because with two nested layers and a 50% failure rate you're at 4× spend at exactly the moment the provider is asking you to send less."

### All our tests pass and CI is green, but users say the assistant got worse after last week's deploy. How is that possible?

It is not only possible, it is the expected state of an LLM system under a conventional test suite, and the reason is a category error in what your tests assert.

Your unit and integration tests assert **structure**: the endpoint returns 200, the response parses as JSON, the schema validates, the tool was called with the right arguments, the database row was written. Every one of those can pass while the *content* degrades from correct to confidently wrong. The set of behaviours your tests constrain and the set of behaviours users care about barely overlap. A green build is evidence that the plumbing works; it is zero evidence about quality, and treating it as such is how teams ship regressions invisibly.

Compounding that: the things that changed last week may not be in your diff at all. Candidate causes, in the order I would check them:

**🔍 Failure taxonomy — "it got worse and CI is green":**

1. *Did the model version change under you?* Provider aliases float. A pinned snapshot ID is the only defence, and even then snapshots retire. Check the model string actually being sent in production, not the one in the config file. **📅 Volatile** — deprecation windows differ per provider.
2. *Did the retrieval corpus change?* An ingest job added 400k low-quality documents, or a reindex changed chunking, or the staleness gauge you never built has been climbing for nine days. Retrieval regressions look exactly like model regressions from the user's seat.
3. *Did the prompt change through a path that isn't code?* Prompts in a database, a feature flag, an admin UI, a CMS. If your prompts are not versioned in git with the same review gate as code, this is your most likely cause and you cannot bisect it.
4. *Did context length grow past the effective window?* Adding two more few-shot examples or a bigger `k` can push the important instruction into the middle of a long context, where adherence measurably degrades even though the context "fits."
5. *Did a tool's output format change?* An upstream API added a field, the tool result got 3× longer, and the model now truncates or gets distracted. This is a contract change no test covered because the tool test asserted on schema, not on token count.
6. *Did nothing change, and the traffic mix shifted?* A marketing campaign brought a new query distribution your prompt was never tuned for. This is the one people never check and it is common.

**What replaces the green build.** An eval suite in CI, gated statistically. Concretely: a versioned dataset of 200–500 cases with expected outcomes or judge rubrics; a job that runs it on every prompt, model, retrieval-config or code change; a gate that fails the build on a *statistically significant* regression, not on any drop. And a separate always-on production sampler scoring live traffic, because your eval set ages and your traffic does not.

**📐 Numbers you must know — how big does the eval set need to be?** To detect a 3-percentage-point change around a 75% baseline with an *unpaired* comparison you need roughly `16·p(1−p)/δ² = 16 × 0.1875 / 0.0009 ≈ 3,300` examples per arm, which nobody can label. But you are comparing two systems **on the same examples**, so use a paired test — McNemar's on the discordant pairs, or a paired bootstrap. Paired designs need order-of-magnitude fewer examples, because agreement on the easy cases cancels out; with realistic discordance rates a few hundred examples is enough to detect a change of that size. **The practical rule: 200–500 cases, always paired, always with a confidence interval reported next to the point estimate.** A dashboard showing "74% → 78%" with no interval is how teams chase noise for a week.

**⚠ Trap:** writing eval cases by asking the model to generate them. You get cases drawn from the model's own distribution — exactly the inputs it already handles — and a suite that scores 96% and detects nothing. Eval sets come from production logs, stratified over intent and over whether the current system succeeded, with the hard cases deliberately over-sampled.

### Someone proposes a semantic cache to cut spend. Sell me on it, then tell me why you'd be nervous.

The proposal is sound in outline: embed the incoming query, search a store of previous `(query, answer)` pairs, and if the nearest neighbour's cosine similarity exceeds a threshold, return the stored answer without calling the model. On a product with a heavy head — support FAQ, docs Q&A, product search — this genuinely converts a $0.05, 900 ms call into a $0.00002, 8 ms vector lookup. At a 30% hit rate on 200k daily calls that is 60,000 × $0.05 = **$3,000/day = $90k/month saved**, and the latency win is more valuable than the money.

Now the nervousness, and it is specific rather than general. **An exact cache can only be stale. A semantic cache can be wrong.** Those are different risk classes and your existing cache intuition does not carry over, because every cache you have built before had an equality test at its core.

The failure mode is that cosine similarity in embedding space is a *topical* measure, not a *semantic-equivalence* measure. Embeddings are trained to put things that are about the same subject near each other, and the distinctions that flip the correct answer are exactly the low-magnitude ones:

- **Entity substitution.** "What's the refund policy for the Pro plan?" and "...for the Enterprise plan?" typically sit above 0.95 cosine. Same topic, different answer, and the wrong answer is fluent and specific.
- **Negation and polarity.** "Can I cancel after 30 days?" vs "Can I cancel *before* 30 days?" Embedding models handle negation poorly; this is a well-documented weakness, not an implementation defect you can tune away.
- **Numeric and temporal specificity.** "How much is the Q3 invoice?" and "How much is the Q4 invoice?" are near-identical vectors.
- **Tenant leakage — the incident that gets written up.** If the cache is not partitioned by tenant, by user permissions, and by every filter that scoped the retrieval, you will serve customer A's answer to customer B. This is not a quality bug, it is a data-breach class of bug, and it is the reason I require cache keys to include the full authorization context, not just the query embedding.

**💰 Math on why the threshold cannot be tuned to safety.** Suppose at threshold 0.92 you get a 34% hit rate with a 6% false-hit rate. That is 200,000 × 0.34 × 0.06 = **4,080 confidently wrong answers per day** in exchange for $3,400/day. Raise the threshold to 0.97: hit rate falls to 11%, false-hit rate to maybe 1.5%, so 330 wrong answers/day for $1,100/day of savings. **You cannot get the false-hit rate to zero without the hit rate going with it**, because the two are the same distribution viewed from opposite ends. The decision is therefore a product decision about the cost of a wrong answer, and the only defensible way to make it is to *measure* the false-hit rate on a labelled set — never to pick a threshold because it looked round.

**What I would actually ship**, in order of how much I trust it: exact-match cache on a normalized request first, because it is free of this entire risk class; then prefix caching, which saves more money than semantic caching in most systems and cannot be wrong at all; then semantic caching *only* for a curated, high-frequency, low-variance query set with a measured false-hit rate, partitioned by tenant, with a TTL tied to the underlying data's change rate, and with a "was this helpful" signal wired to invalidate.

**⚠ Trap:** reporting semantic cache hit rate as a win metric with no paired false-hit rate. It is the most common form of AI-adjacent metric fraud I see, usually unintentional, and it is exactly the thing an interviewer will probe when you mention semantic caching. Volunteer the false-hit number before they ask.

### We added a 10-second timeout to the model call and now users complain about failures on the long answers. Diagnose it.

The diagnosis is that you applied a timeout to the wrong quantity. **A total-duration timeout on a streaming generation is a bug, because duration is a function of output length, and output length is a function of the question.** A user asking for a 2,000-token migration plan will legitimately take 40 seconds while streaming perfectly. Your timeout kills a healthy request in progress, after the user has already watched half the answer appear — the worst possible failure presentation, and one your error rate metric probably records as an upstream failure rather than as self-harm.

The correct instrument is two timeouts on two different quantities, plus a cap:

**A TTFT timeout, tight.** If no first token has arrived in 5–10 seconds, something is genuinely wrong — you are queued behind a limiter, the provider is degraded, or a huge prefill is stalling. This is the timeout that maps to your usual "is the upstream alive" check, and it should fire fast because it is the only one that can fire before you have spent output tokens.

**An inter-token-gap timeout, tighter.** Once tokens are flowing, healthy streams do not pause. A gap of more than 2–5 seconds between consecutive tokens means the stream is dead or stalled behind another request's prefill. This is a *stall* detector, and it is the one people never implement. It is what actually distinguishes "slow because the answer is long" from "hung."

**A hard cap expressed in tokens, not seconds.** `max_tokens` is your real bound on cost and on duration, and it fails cleanly with a `stop_reason` you can detect and handle, rather than by severing a connection.

**📐 Numbers you must know:** convert a latency budget into a token budget rather than a second budget, because tokens are the thing you control. `max_duration ≈ TTFT + TPOT × max_tokens`. At a measured TTFT of 0.5 s and TPOT of 30 ms, a 20-second product budget allows (20 − 0.5) / 0.03 ≈ **650 output tokens**. If the feature genuinely needs 2,000 tokens, the budget is 0.5 + 0.03 × 2,000 = 60.5 s and no timeout will fix that — you either cut scope, cut TPOT (quantization, speculative decoding, smaller model), or change the UX so the wait is not dead time. This substitution — seconds into tokens — is one of the most useful reflexes in the whole discipline.

```python
# Timeout on onset and on stalls, never on total duration.
async def stream_with_stall_detection(client, params, ttft_s=8.0, gap_s=4.0):
    async with client.messages.stream(**params) as stream:
        it = stream.text_stream.__aiter__()
        deadline = ttft_s
        while True:
            try:
                chunk = await asyncio.wait_for(it.__anext__(), timeout=deadline)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                raise StreamStalled(f"no token within {deadline}s")
            deadline = gap_s          # after first token, switch to gap budget
            yield chunk
```

**The infrastructure that will undo you regardless of application code:** every proxy between you and the client has its own idle timeout and its own buffering, and the defaults are hostile to streaming. nginx buffers proxied responses by default (`proxy_buffering off` and `X-Accel-Buffering: no` are the fixes), load balancers have idle timeouts commonly in the 30–60 second range, and some CDNs will hold a response until they see enough bytes. The classic incident is that streaming works perfectly in dev and returns as one blob in production, and everyone blames the model. **📅 Volatile:** default idle timeouts differ per cloud and change; check yours rather than trusting a remembered number.

**⚠ Trap:** setting a total timeout derived from p95 latency measured on your current traffic. The moment someone ships a feature that produces longer answers — a summarize-this-document button, a code-generation mode — the p95 shifts and your timeout starts killing your newest, most valuable feature. Timeouts must be tied to a *rate* (onset, gap) which is stable, not to a *total* which is workload-dependent.

**🗣 Say this in the room:** "I'd never put a wall-clock timeout on a streaming generation, because duration scales with output length and I'd be killing my longest, most valuable answers. I put a tight timeout on time-to-first-token, a stall timeout on the inter-token gap, and a hard cap on `max_tokens` — that bounds cost and duration without ever cutting off a healthy stream."

### The model isn't good enough on our domain. Walk me up your escalation ladder — and tell me what has to be true at each rung before you climb to the next.

The ladder exists because the cheap interventions are also the fast, reversible ones, and because every rung you skip makes the next rung's evaluation harder to interpret. The rule I enforce is that **you may not climb a rung until you can state, with a number, what the previous rung achieved and why it plateaued.** Without that, you cannot tell whether the expensive thing worked.

**Rung 0 — build the eval.** Not a rung, a precondition for the whole ladder. 200–500 cases from production logs, a primary metric, a paired comparison procedure. If someone asks me to improve a model's domain performance and there is no eval, my answer is that we are not yet in a position to improve anything, only to change things.

**Rung 1 — prompt and context engineering.** Clearer task framing, explicit output contract, 3–8 well-chosen few-shot examples, and — most underrated — *removing* material rather than adding it. In my experience this recovers the majority of a domain gap, and it costs an afternoon. **Precondition to leave:** you have run at least a dozen prompt variants against the eval, the best one is significantly better than the baseline, and the remaining errors are no longer instruction-following errors. *Climb when:* the failures are now missing-knowledge failures, not misunderstanding failures.

**Rung 2 — retrieval.** The model does not know your data. Give it the data. This is the correct fix for anything that *changes*: policies, prices, tickets, code, documents. **Precondition to leave:** you have measured retrieval separately from generation. Recall@k on a labelled query→document set tells you whether the right chunk was even in the context. If Recall@10 is 55%, you have a retrieval problem and no amount of generation work will fix it. *Climb when:* retrieval recall is high and the model still gets it wrong with the right context in front of it.

**Rung 3 — tool design.** The model does not need to *know* it, it needs to *fetch or compute* it. Wrap the exact query, the calculator, the internal API. This is where the biggest wins in agentic products actually live and it is chronically skipped because it is unglamorous engineering. **Precondition to leave:** tool call error rate broken out per tool is low, tool descriptions have been revised at least once against observed misuse, and error strings are written as corrective instructions. *Climb when:* the model is calling the right tools correctly and still producing unusable output.

**Rung 4 — structured output and constrained decoding.** If the failure is format — malformed JSON, inconsistent enums, wrong field names — do not fine-tune for it and do not prompt harder. Constrain the decode. This is a near-total fix for a whole failure class and takes a day. **Precondition to leave:** output is now schema-valid at essentially 100% and the residual errors are semantic, not syntactic.

Those four rungs, in that order, close most gaps. The three below the line are the ones that require you to have earned them, and they are the next question.

**⚠ Trap:** running the rungs in parallel to "save time." You change the prompt, add retrieval and swap the model in one week, the score moves 9 points, and you have no idea which change did it — or that two of them helped and one hurt. Sequential with a paired eval between each is not bureaucracy; it is the only way to know what to keep when you later need to cut cost.

### Now the bottom of the ladder — routing, distillation, fine-tuning. What are the actual preconditions?

**Rung 5 — routing.** Send easy requests to a cheap small model and hard ones to a frontier model, with a classifier or a heuristic making the call. Sometimes the "route" is to no model at all — a cached answer or a deterministic path. This is a cost lever primarily and a latency lever secondarily; it rarely improves peak quality.

**Precondition:** you can *measure* difficulty in advance, and you have a validated fallback for misroutes. The classifier's errors are asymmetric — routing a hard query to the small model is a visible quality failure, routing an easy one to the big model just costs money — so tune the threshold to be conservative and monitor the escalation rate.

**💰 Math:** 1M requests/month, 3k input + 500 output each. All-frontier at $3/$15 per M: 1M × (3,000 × 3/1e6 + 500 × 15/1e6) = 1M × ($0.009 + $0.0075) = **$16,500/month**. Route 70% to a small model at roughly $0.25/$1.25 per M (**📅 Volatile**): 700k × (3,000 × 0.25/1e6 + 500 × 1.25/1e6) = 700k × ($0.00075 + $0.000625) = $963. Remaining 300k at frontier = $4,950. Plus routing classifier cost, negligible if it is an embedding model. Total **$5,913 — a 64% reduction, $10,587/month saved.** Now the honest part: if the router misroutes 4% of hard queries, that is 28,000 degraded answers a month, and whether that trade is correct is a product decision that requires the eval from rung 0 to even discuss.

**Rung 6 — distillation.** Use your frontier model to generate high-quality outputs on your traffic distribution, then train a small model on them. You get frontier-ish quality on *your* narrow task at small-model cost and latency. This is the right answer far more often than general fine-tuning, and it is under-used.

**Preconditions, all of them:** a task narrow enough that a small model can represent it; tens of thousands of teacher outputs, *filtered* by your eval rather than used raw; a licence that permits training on the teacher's outputs (several providers' terms restrict using outputs to train competing models — **📅 Volatile**, read the current terms, this is a real legal constraint not a formality); and a serving story for the student. **Precondition to justify:** volume high enough that the training and serving cost amortizes. At 1M requests/month the routing example above already saves $10.5k/month, so a distillation project costing $30k of engineering pays back in three months — at 20k requests/month it never does.

**Rung 7 — fine-tuning the base behaviour.** LoRA or full SFT on curated examples, and it is last for a reason. **What it is good at:** output format and structure, house style and tone, tool-calling conventions the base model keeps getting wrong, domain vocabulary and jargon, and compressing a very long prompt into weights so you stop paying for it every call. **What it is bad at:** installing facts. Facts change; weights do not. Anything that changes weekly must come from retrieval or it is stale the day after your training run.

**The named preconditions, and I would recite these as a list in an interview:** (1) a working eval with a baseline number; (2) rungs 1–4 exhausted, with the plateau documented; (3) at least a few thousand high-quality, consistent examples — consistency matters more than volume, and 2,000 clean examples beat 20,000 noisy ones; (4) a data-refresh plan, because the model will drift from your evolving product; (5) a serving plan — adapter hosting, versioning, rollback, and the fact that a custom model forfeits some provider features and cache economics; (6) an owner, because a fine-tuned model is a dependency with a maintenance cost forever.

**⚠ Trap — the reflex-fine-tuning rejection trigger, stated plainly.** Answering "the model doesn't know our domain" with "we'd fine-tune" is the single fastest way to be marked down in an applied AI loop, because it reveals that you have read about ML rather than shipped it. The interviewer is not testing whether you know what fine-tuning is; they are testing whether you know it is the seventh thing to try.

**🗣 Say this in the room:** "I'd hold fine-tuning until last and I'd want to be able to state what each earlier rung bought us. Fine-tuning teaches the model how to behave, not what's true — so format, style and tool conventions, yes; domain facts, no, those go in retrieval or they're stale immediately. Before I'd greenlight it I'd want an eval with a baseline, a documented plateau on prompting and retrieval, a few thousand consistent examples, and a named owner for the refresh."

### How would you actually find your own gaps before spending eight weeks studying? Give me the test.

Build a 40-question self-assessment, run it unaided and timed, and grade it honestly. The purpose is not a score, it is **localization** — the difference between "I need to study AI" and "I am fine on serving, weak on evaluation statistics, and cannot write attention." The first statement produces eight weeks of undirected reading; the second produces a plan.

**The structure: eight domains, five questions each, answered out loud or on paper in 90 minutes total with no references.**

*Transformer internals (5).* Write scaled dot-product attention with a causal mask. Derive √d_k. Give the parameter split between attention and FFN in a standard block. Explain what RoPE rotates and why relative position falls out. State the KV cache formula and say which quantities are absent from it.

*Serving and inference (5).* Explain why decode is memory-bandwidth-bound. Distinguish continuous batching from dynamic batching. Explain PagedAttention as a memory-allocator argument. Compute max concurrency for an 8B model on an 80 GB card at 8k context. Name the correct autoscaling signal and say why GPU utilization is not it.

*Retrieval (5).* Explain the chunking trade-off in terms of recall and precision. Contrast dense, sparse and hybrid retrieval and say when each wins. Describe reranking and its cost. Define Recall@k and nDCG@k. Describe a zero-downtime re-embedding migration.

*Evaluation (5).* Construct an eval dataset for a support agent, saying where cases come from. Choose a metric and defend it. Describe LLM-as-judge validation and name three of its biases. Compute the sample size to detect a 3-point change. Say what you would gate CI on.

*Agents (5).* Write the tool-calling loop from memory. Design a tool schema and say why it is coarse. Describe poison-trajectory detection. Give a context-budget accounting for a 10-step run. Name the four dominant agent failure causes in production.

*Post-training (5).* Contrast SFT, RLHF and DPO in one sentence each. Explain what RLVR needs that a chat task cannot supply. Explain LoRA's low-rank decomposition and its parameter count. Say when fine-tuning beats prompting. Say when it does not.

*Cost and capacity (5).* Compute monthly spend for a stated workload. Compute prefix-caching savings. Compute the routing trade with a misroute rate. Compute KV cache for a given model. Convert a latency budget into a token budget.

*Product and judgment (5).* Say when not to use an LLM, with the substitution. Name the failure mode users punish hardest. Describe the escalation ladder with preconditions. Critique an AI product you use. Say how you would know a feature is working.

**Grading, and this is the part that has to be brutal.** Score each question 0, 1 or 2. **2 = you produced it unaided, correctly, in time, including the arithmetic.** 1 = you knew the shape but needed a hint, fumbled a number, or could not write the code. 0 = you could not start. There is no partial credit for "I've read about that." A 1 is a fail for interview purposes, because a 1 under interview pressure becomes a 0.

**How to read the result.** Per-domain out of 10. **0–3: this domain is a rejection risk and gets a full week.** 4–6: real gaps, budget three focused days and re-test. 7–8: sharpen with drills only. 9–10: skip and re-test in week six to catch decay. Total out of 80 is nearly meaningless — the *shape* is the artifact, and the single lowest domain gets attacked first regardless of what the total says.

**⚠ Trap:** grading yourself while reading the answer. The entire diagnostic value of this exercise comes from the gap between recognition and production, and recognition feels exactly like knowledge. Write your answer down *first*, in full, then compare. Every engineer who skips this step scores themselves 15–20 points high and discovers the truth in a real loop.

**🏋 Drill — do this before you read another section.** 90 minutes, timer running, phone away, no autocomplete, blank paper and a blank editor. All 40 questions. Then grade against the guide over the following week, one domain at a time. **Pass criterion for the exercise itself (not for the score):** you have a per-domain vector of eight numbers and a written one-line plan for each domain scoring below 7, with a specific section to read and a date. If you finish with a total score and no plan, you did the exercise wrong — the score was never the point.

**🗣 Say this in the room,** when a hiring manager asks how you approached the transition: "I ran a written self-assessment across eight domains before I studied anything, graded it on whether I could *produce* the answer rather than recognize it, and it turned out my gaps were transformer internals and evaluation statistics — not the engineering. So I spent a third of my prep on those two and skipped the async, queueing and container material entirely, because I've been shipping it for years."
