### You've built Redis rate limiters before. Design the equivalent for an LLM gateway serving 40 internal teams.

Start from what changed, because the algorithm did not. A token bucket is still a token bucket. What changed is **the unit of the bucket and the fact that you cannot measure the withdrawal until after you have spent it.**

Your HTTP rate limiter counts requests, and a request's cost is known at admission. Here the provider limits you on three axes simultaneously — requests per minute, *input* tokens per minute, and *output* tokens per minute — and the dominant one is almost always tokens. A single request can consume 200 tokens or 200,000. So a `Semaphore(100)` or a 1,000-RPM limit is not merely imprecise, it is measuring the wrong quantity: one tenant sending 100k-token documents will saturate your token budget at 5 requests per minute while your request-counter reports 0.5% utilization.

The design that works:

**Bucket in tokens, admit on an estimate, reconcile on the actual.** At admission you know input tokens exactly (tokenize, or use the provider's counting endpoint, or estimate at ~4 characters/token for English and be conservative). You do *not* know output tokens, so you reserve `max_tokens` — the ceiling the caller requested — and refund the difference when the response completes. This is a two-phase commit against a bucket, and it is the whole trick.

```lua
-- Atomic reserve against a per-tenant token bucket. KEYS[1]=bucket, ARGV: now, rate, burst, want
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or tonumber(ARGV[3])
local ts     = tonumber(b[2]) or tonumber(ARGV[1])
tokens = math.min(tonumber(ARGV[3]), tokens + (tonumber(ARGV[1]) - ts) * tonumber(ARGV[2]))
if tokens < tonumber(ARGV[4]) then
  redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', ARGV[1])
  return {0, math.ceil((tonumber(ARGV[4]) - tokens) / tonumber(ARGV[2]))}  -- deny + retry_after_s
end
redis.call('HMSET', KEYS[1], 'tokens', tokens - tonumber(ARGV[4]), 'ts', ARGV[1])
redis.call('EXPIRE', KEYS[1], 600)
return {1, 0}
```

Then `HINCRBYFLOAT bucket tokens <refund>` when the actual output count arrives.

**Three things beyond the algorithm that the interviewer is actually probing:**

*The upstream bucket is shared and you do not own it.* Your per-tenant buckets must sum to less than your provider allocation, with headroom, or you have built a fair queue that fairly distributes 429s. Read the provider's rate-limit response headers (`x-ratelimit-remaining-tokens` / `anthropic-ratelimit-tokens-remaining` families, plus `retry-after` — **📅 Volatile**, header names differ per provider and change) and feed them back as a *measured* ceiling rather than trusting a configured constant.

*Queue, don't reject, for async traffic.* A 429 to a human-facing chat request is correct. A 429 to a batch enrichment job is waste — that work should sit in a priority queue and drain at whatever rate the bucket refills, with interactive traffic given strict priority. This is classic multi-class scheduling and it is where your background pays off.

*Fairness needs a work-conserving policy.* Strict per-tenant caps leave capacity idle when tenants are quiet. Deficit round-robin or weighted fair queueing over the shared token budget lets a quiet-hours tenant burst into unused capacity without letting them starve anyone at peak.

**💰 Math:** suppose your org tier gives 2,000,000 input tokens/minute. Forty teams, naive equal split = 50k TPM each. One team runs a document pipeline at 100k-token requests: they can issue 0.5 requests/second and are permanently throttled while 39 teams idle. Now the same budget under weighted fair queueing with a 200k burst allowance: the pipeline team drains unused capacity at 3–4× their nominal share overnight and is capped back at peak. Same money, ~4× the useful work out of the same tier.

**⚠ Trap:** rate-limiting on requests because that is what your middleware already does. It passes review, it looks correct on the dashboard, and it fails the first time a tenant changes their prompt length. **The rule I enforce: any limiter in front of a model call is denominated in tokens, or it is decorative.**

### Walk me through the LLM version of a dead-letter queue. What goes in it and how does it get there?

The mental model shift is that in your queue systems the poison is in the *message*, and here the poison is in the *trajectory*. Every individual step is well-formed and would pass any payload validator you wrote. The pathology is in the sequence: the model calls `search_tickets(status="opne")`, gets a validation error, apologizes, and calls `search_tickets(status="opne")` again. Sixty times. There is nothing malformed to quarantine on and no exception is ever raised.

So detection must be state-based, and there are exactly four signals worth wiring:

**Repeat-action detection.** Hash `(tool_name, canonicalized_args)` at every step into a per-run counter. Canonicalization matters — sort keys, normalize whitespace and casing, round floats — or the model's cosmetic variation defeats you. Threshold at 3 identical calls, or 5 near-identical ones by edit distance. This catches the majority of real loops.

**Budget exhaustion.** Per-run caps on steps (12–20 is typical), on cumulative tokens, and on wall-clock. Every one of these should be a *hard* kill with a recorded reason, not a soft warning. The step cap alone is insufficient because a single step with a 200k-token tool result can blow the whole budget.

**Non-progress detection.** Harder and worth building for high-value agents: track whether the agent's *state* changed — files written, rows returned, a distinct URL fetched. Ten steps with no state delta is a stall even if every call was distinct.

**Terminal-error classification.** Distinguish retryable (429, 503, transient tool timeout) from terminal (401, schema violation the model keeps repeating, tool that does not exist). Terminal errors should abort the run, not be fed back for another attempt, because the model cannot fix a missing credential by rewording.

What goes in the DLQ is different too. A poison message goes to the DLQ as a payload; a poison trajectory goes to a review queue as **the entire transcript plus the run's cost**, because the transcript *is* the debugging artifact and because the transcript is what you mine to build eval cases. This is the highest-value part of the whole pattern: your poison-trajectory queue is also your eval dataset generator. A run that looped is, by definition, a case your system handles badly and should be regression-tested against.

**🔍 Failure taxonomy — run this as a decision procedure when an agent misbehaves:**

1. *Did it loop on an identical call?* → tool error message is uninformative. Fix the error string first; it is the cheapest and most effective lever.
2. *Did it loop on varying calls to the same tool?* → schema is ambiguous. The model is guessing at parameter semantics. Fix the description and add an enum.
3. *Did it stop early with a confident wrong answer?* → termination condition too permissive, or the tool returned an empty result the model read as "no such thing exists" rather than "query failed."
4. *Did it blow the token budget in 2 steps?* → a tool result is unbounded. Truncate and paginate at the tool boundary, never at the context boundary.
5. *Did it call a tool that does not exist?* → too many tools in the registry, or names too similar. Consolidate.

**⚠ Trap:** retrying a poison trajectory from step zero. You just paid the full cost again for a deterministic failure with a nonzero chance of the same loop. Poison trajectories are not retried; they are killed, recorded, and — if the domain allows — resumed *from the last good state* with the failing tool disabled or with human input injected.

### Our RAG answers went stale after a reindex and nobody noticed for a week. What should the rollout have looked like?

This is your Postgres index migration, and the reason it bit you is that a vector index fails *silently* in a way a B-tree does not. If you break a B-tree, queries error or plans go sequential and latency screams at you. If you break a vector index — by mixing embedding spaces, by building HNSW with the wrong `ef_construction`, by rebuilding while writes are still landing — the query still returns exactly `k` rows, in a plausible order, with plausible scores. **The failure surface has no error signal at all.** That is why "nobody noticed for a week" is the expected outcome, not a team failure.

The rollout you want is the alias-swap you already know, with two additions that are specific to embeddings.

```
1. Freeze schema: new index/table/collection `docs_v2`, alias `docs_live` still → v1.
2. Backfill v2 from a durable queue with a watermark. Resumable. Idempotent per chunk_id.
   Track: chunks_total, chunks_embedded, chunks_failed, oldest_unembedded_updated_at.
3. Do NOT flip on "backfill complete." Flip on evidence.
4. Shadow read: for 1–5% of live queries, run both indexes, log
   overlap@10 (|topk_v1 ∩ topk_v2| / 10) and score distributions. Do not compare latency only.
5. Golden-set eval: a frozen set of ~200 (query, known-relevant-doc-ids) pairs.
   Measure Recall@10 and nDCG@10 on v1 and v2. v2 must be >= v1 minus a stated tolerance.
6. Flip alias. Keep v1 warm for the rollback window (I use 7 days).
7. Continuously: emit oldest_document_indexed_at as a gauge. This is your staleness SLI.
```

Step 4 and step 7 are the ones teams skip and the ones that would have caught your incident.

The staleness metric deserves emphasis because it is the single most under-instrumented signal in production RAG. You want a gauge — `now() − min(indexed_at)` over documents whose `updated_at > indexed_at` — alarmed at whatever your product's freshness promise is. It is exactly Kafka consumer lag, applied to an indexing pipeline, and framing it that way in an interview lands well because it shows you are reusing rather than reinventing.

**⚠ Trap:** the two-embedding-model incident, which is the specific version of this that keeps happening. You upgrade the embedding model, redeploy the *query* path, and the index still holds vectors from the old model. Cosine similarity between two different embedding spaces is not an error; it is a number, usually in a reasonable range, ranked meaninglessly. Retrieval quality collapses and every dashboard stays green. **Defence: store the embedding model ID and dimension as a column on every vector row, and have the query path assert equality at read time.** One cheap assertion prevents the whole class.

**💰 Math on the rollback window:** 5M chunks × 1536 dims × 4 bytes = 30.7 GB of raw vectors per version, plus HNSW graph overhead (call it 40% at `m=16`) ≈ 43 GB per version, ≈ 86 GB while both are live. On a managed Postgres with 128 GB RAM that is the difference between the index sitting in shared buffers and thrashing to disk — so a shadow window is a *capacity* decision you plan for, not a free safety net. If you cannot hold both, the honest alternative is a read-only maintenance window, and you should say so rather than pretending.

### You keep Kafka consumer lag on a dashboard. What's the equivalent leading indicator for an inference deployment?

Consumer lag works because it is a *leading* indicator with a physical meaning: unprocessed work is accumulating and you can see the backlog growing before latency degrades. Almost every metric people put on an LLM autoscaling dashboard is *lagging* — request latency, error rate, GPU utilization — which means you scale after users have already had a bad experience, and given 60–90 second cold starts, "after" is fatal.

The genuine equivalents, in order of usefulness:

**KV-cache utilization** is the primary one, and it maps to lag almost exactly. The serving engine knows what fraction of its KV block pool is allocated. As utilization climbs toward 100%, the scheduler stops admitting new sequences and starts *preempting* — vLLM will swap or recompute evicted sequences, which is a cliff, not a slope. Alarm at 70–80%, scale at 80%, because you need the new replica to be warm before you hit 95%. This is the number I would put in the middle of the dashboard.

**Waiting-queue depth and time-in-queue.** The engine's pending-request queue is literally a backlog. Time-in-queue rising while GPU-side step time stays flat means you are admission-limited, not compute-limited, and adding replicas will help. If step time is rising too, you are compute-limited and adding replicas helps differently (and you may instead need to cap batch size or context).

**Preemption / recompute rate.** Any nonzero preemption is capacity pain. It is the equivalent of seeing rebalances in a consumer group — the system is doing work that is not your work.

**⚠ Trap — and this is the one that most reliably separates candidates:** autoscaling on GPU utilization. `nvidia-smi`'s utilization figure reports the fraction of sampled intervals during which at least one kernel was executing. During memory-bandwidth-bound decode, kernels are essentially always resident, so it reads 90–100% whether you are serving 3 sequences or 200. **It is a boolean dressed as a percentage.** A deployment autoscaled on it will sit pinned at "fully utilized" and never scale, or will scale on nothing. Use `DCGM_FI_PROF_*` occupancy metrics if you must have a hardware signal, but prefer the engine's own scheduler metrics — they are the ones that know about queueing.

**💰 Math on why leading matters:** a 70B model at ~140 GB in bf16 pulled from an object store at 2 GB/s takes 70 s to load, plus container start and CUDA init — call it 90 s to serve traffic. Traffic doubles in 30 s. If you scale on p95 latency you begin scaling at t≈40 s and are healthy at t≈130 s, so you serve ~90 seconds of degraded traffic. If you scale at 80% KV utilization you begin at t≈12 s and are healthy at t≈102 s — but more importantly you can hold a warm pool sized off the same signal. The fix for cold start is never a faster loader; it is a leading indicator plus purchased idle capacity, and the interviewer wants to hear you say idle capacity is a *product* decision with a price tag.

### Translate "connection pool exhaustion" into this world for me.

It becomes **provider rate-limit saturation**, and the reason the analogy is worth carrying is that the observable symptoms are identical and the misdiagnosis is identical.

In your world: pool of 20, 200 concurrent requests, 180 coroutines blocked on `acquire()`. Your database looks healthy — low CPU, fast queries — and your app's p99 is catastrophic. The latency is entirely queueing at a resource whose depth is not in the trace. Every senior engineer has been burned by this once and then instruments pool wait time forever.

Here: your provider tier is 2M input TPM. You are at 1.9M. The provider is not erroring, is not slow, and its own latency metrics are perfect. But your client-side limiter (or the provider's) is queueing, so your measured TTFT includes several seconds of waiting for a token bucket to refill. The upstream looks healthy from *its* side. **You are measuring your own queue and attributing it to the model.**

The instrumentation that makes this diagnosable is the same instrumentation:

- Emit **time-in-limiter** as its own span, separate from time-to-first-token from the provider. Without this split, TTFT is an uninterpretable sum.
- Emit **429 rate and `retry-after` values** as a first-class metric, not as an error-log line. A rising `retry-after` is the provider telling you exactly how oversubscribed you are.
- Track **remaining-tokens headroom** from the response headers as a gauge. This is your "connections available" and it is the leading indicator.
- Tag every metric by tenant *and* by model, because the limits are per-model and one tenant's migration to a bigger model silently halves everyone's headroom.

The differences worth naming, because an interviewer will push:

*You cannot just raise the pool size.* Provider limits are a commercial negotiation with a tier-escalation process measured in days. So the mitigations are structural: route overflow to a second model or a second provider, shed low-priority traffic, move batchable work to a batch endpoint at roughly half price, and cache aggressively at the prefix level.

*Multi-account or multi-region load balancing is an ops decision, not a hack* — but it must be an explicit, contractually-clean one, and you should say that out loud rather than implying you would evade limits. The clean version is separate organizations for separate business units with separate quotas, which is how the provider expects large customers to operate.

*Failing open is worse here.* When a pool is exhausted you queue and the work eventually completes. When you are rate-limited and you "fail open" by retrying harder, you amplify the saturation — the classic congestion collapse. Retries against a 429 must be exponential with jitter and must respect `retry-after`, and your concurrency limiter must *decrease* on 429, AIMD-style, not stay constant.

**🗣 Say this in the room:** "This is connection-pool exhaustion with a different pool. The tell is that upstream latency looks perfect while our p99 is terrible, which means the wait is in our own admission queue. So the first thing I'd do is split TTFT into time-in-limiter and time-at-provider, because until those are separate spans nobody in the room can tell whether we have a capacity problem or a model problem."

### Your dashboards are all p50/p95/p99. What replaces that for a streaming LLM endpoint, and why isn't p95 latency enough?

Because a streaming response has no single latency. It has an onset and a rate, and users perceive those completely differently. A response that starts in 300 ms and streams for 8 seconds feels fast. A response that returns as one blob at 4 seconds feels slow. End-to-end p95 ranks the second one better. **Your existing metric literally inverts the user's preference ordering**, which is the sharpest reason to change vocabulary rather than just add jargon.

The three metrics that replace it:

**TTFT — time to first token.** Onset. Dominated by prefill, which is compute-bound and scales with input length, plus any queueing in your limiter or the engine's admission queue. This is the number that determines whether the product feels alive. For chat, I would set an SLO around 500 ms p95; for an agent step that a human is not watching, TTFT is nearly irrelevant and optimizing it is wasted effort.

**ITL — inter-token latency.** The distribution of gaps between consecutive tokens. This is what you alarm on for *stutter*: a p99 ITL spike means another request's long prefill just stalled your decode loop, which is the single most common cause of a "it froze mid-sentence" complaint.

**TPOT — time per output token.** The mean of that distribution, i.e. `(E2E − TTFT) / (output_tokens − 1)`. This is your throughput-per-user figure and it is bounded by memory bandwidth and batch size.

The relation you should be able to write instantly: `E2E ≈ TTFT + TPOT × (output_tokens − 1)`.

**📐 Numbers you must know:** human reading speed is roughly 250 words/minute ≈ 4 words/sec ≈ **5–6 tokens/sec**. So a TPOT of 200 ms (5 tok/s) is exactly reading speed; anything faster is invisible to a reading user and is only worth paying for when the output is being consumed by code, or when the user is skimming for a code block. This single number kills a lot of expensive optimization: **if a human reads the output, TPOT below ~50 ms buys nothing and you should spend that budget on TTFT or on quality instead.**

**💰 Math, worked:** a 500-token answer at TTFT 400 ms and TPOT 25 ms gives E2E = 0.4 + 0.025 × 499 = 12.9 s. Halving TPOT to 12.5 ms — which might cost you speculative decoding complexity or a second GPU — gives 6.6 s. But the user is reading at 5–6 tok/s and needs 500/5.5 ≈ 91 s to actually read it. **You spent a GPU to finish generating 6 seconds earlier into a 91-second reading task.** Halving TTFT from 400 ms to 200 ms, by contrast, is directly perceptible. That is the trade an interviewer wants you to make explicitly.

**⚠ Trap:** treating ITL and TPOT as synonyms in a room where someone serves models. TPOT is the mean; ITL is the distribution, and the whole point of tracking ITL separately is the tail. A deployment with excellent p50 ITL and terrible p99 ITL is a deployment with prefill/decode contention, and the fix is chunked prefill or prefill/decode disaggregation — a specific, nameable fix you can only reach if you kept the distribution.

### If QPS is the wrong throughput metric, what's the right one?

QPS is wrong because a "query" is not a unit of work here — one request might be 500 tokens of work and the next 500,000. Reporting QPS for an LLM service is like reporting "files per second" for a storage system without mentioning file size: technically a rate, operationally meaningless.

**Tokens per second** is the honest throughput unit, and it splits in two because the two phases have different economics. **Prefill tokens/sec** measures compute throughput and is roughly bounded by the GPU's matmul rate. **Decode tokens/sec** — aggregated across all concurrent sequences — measures how well you are amortizing weight reads across the batch, and it is bounded by memory bandwidth. A deployment can be excellent at one and terrible at the other, so a single aggregate number hides the actual bottleneck.

But the metric that actually belongs on the SLO dashboard is **goodput: the rate of requests completed *while meeting their latency SLO*.** This distinction is the one that gets you credit. Throughput and latency trade off through batch size: bigger batches raise total tokens/sec and raise per-request TPOT, because each sequence now waits behind more work per step. So you can always "improve throughput" by degrading everyone's experience, and a team optimizing raw tokens/sec will do exactly that and report it as a win. Goodput is throughput with the SLO as a constraint rather than an afterthought — it is the only number that cannot be gamed by making users wait.

**📄 Paper:** the goodput framing for LLM serving was made central by the prefill/decode disaggregation work — Zhong et al. (2024), DistServe — which argued that co-locating prefill and decode forces a single batch-size choice onto two workloads with opposite latency profiles, and that separating them onto distinct resource pools lets you tune each for its own SLO. Related: chunked prefill (Agrawal et al., 2024, Sarathi-Serve) splits a long prefill into slices interleaved with decode steps so a single 100k-token prompt does not stall every other user's stream.

**💰 Math showing the trade:** suppose at batch 8 your decode step takes 20 ms (TPOT 20 ms, aggregate 8 × 50 = 400 tok/s) and at batch 32 it takes 45 ms (TPOT 45 ms, aggregate 32 × 22 = 711 tok/s). Raw throughput is up 78%. But if your TPOT SLO is 30 ms, **goodput at batch 32 is zero** — every request violates. The correct answer is not "pick batch 8," it is "find the largest batch whose TPOT stays under 30 ms," which here might be 16, and then note that the remaining headroom should come from disaggregation or quantization, not from batching harder.

**🗣 Say this in the room:** "I'd stop reporting QPS. The unit of work is tokens, and the number I'd hold the team to is goodput — completed requests per second that met the TTFT and TPOT SLOs — because throughput and latency trade off through batch size, and any team measured on raw throughput will quietly buy it by degrading everyone's stream."

### Finance asks for cost per request. Why do you push back, and what do you give them instead?

Because cost per request is measurable, stable, and answers a question nobody cares about. A support agent that resolves a ticket in one $0.40 call is dramatically better business than one that resolves it in twelve $0.03 calls, and cost per request ranks them the wrong way round — it makes the twelve-call agent look 13× cheaper per unit while it burns more money and more of the user's time.

The number to hold is **cost per resolved task**: total spend divided by the count of tasks that reached a *verified* successful outcome. Everything hard about it is in the word "verified," which is exactly why it is a good metric — it forces you to define success, which forces you to build the eval, which is the discipline the whole loop is testing.

Build it as a chain:

```
cost_per_resolved_task = (Σ tokens_in·price_in + Σ tokens_out·price_out + retrieval + tools + human_escalation_cost)
                         ÷ (tasks × resolution_rate)
```

Two terms in there are the ones people forget and both are usually dominant. **Retries and failed trajectories are numerator, not noise** — a run that looped for 40 steps and was killed costs real money and resolved nothing. And **human escalation is a real cost line**: if 30% of tickets escalate to a human at a loaded cost of $6 each, that term swamps your token spend and it is the term your automation is actually trying to move.

**💰 Math, the comparison that makes the point:** two designs for a support agent, 100k tickets/month.

*Design A — single frontier call, big context.* 25k input tokens (full policy docs stuffed in), 600 output. At $3/M in, $15/M out: 25,000 × 3/1e6 + 600 × 15/1e6 = $0.0750 + $0.0090 = **$0.084/ticket**. Resolution rate 62%. Escalations: 38,000 × $6 = $228,000. Token spend 100,000 × $0.084 = $8,400. Total $236,400. **Cost per resolved task = $236,400 / 62,000 = $3.81.**

*Design B — retrieval + a 4-step agent.* Average 5 model calls, 4k input each, 300 output each: 5 × (4,000 × 3/1e6 + 300 × 15/1e6) = 5 × ($0.012 + $0.0045) = **$0.0825/ticket** — essentially identical token cost. Resolution rate 79%. Escalations: 21,000 × $6 = $126,000. Token spend $8,250. Total $134,250. **Cost per resolved task = $134,250 / 79,000 = $1.70.**

Cost per *request* says A is fractionally cheaper. Cost per *resolved task* says B is 2.2× better and saves **$102,150/month**. That is the argument, with digits, and it is the shape of answer that makes a hiring manager decide you have owned a budget. (**📅 Volatile:** unit prices move; recompute with current numbers before you cite them.)

**⚠ Trap:** defining resolution as "the user didn't reply again." That proxy is corrupted in both directions — users abandon when the answer is bad, and users stop replying when the answer is good. You need either an explicit signal (ticket closed by the customer, code merged, invoice matched) or a validated LLM-judge with a measured agreement rate against human labels. Say which one you would use and why; hand-waving here is the failure.

### Cache hit rate is second nature to you. What's the LLM-specific version and how do you move it?

Three different caches get called "cache" in this stack and conflating them is a common tell. Rank them by how much of the industry's actual savings they produce.

**Prefix cache hit rate** is the important one and the one you should name first. Your system prompt, tool schemas, few-shot examples and retrieved documents form a shared prefix across calls; the KV states for that prefix are deterministic given the tokens, so they can be computed once and reused. Self-hosted, that is vLLM's automatic prefix caching or SGLang's RadixAttention. Through an API, it is the provider's prompt caching, with cached input billed at a steep discount.

The lever that moves it is **prompt layout**, and this is the single most actionable piece of engineering in this whole answer: **order your prompt from most-static to most-dynamic.** System instructions, then tool definitions, then few-shot examples, then retrieved context, then conversation history, then the user turn. One timestamp, one user ID, one randomized example order near the top invalidates everything after it, because the cache is a prefix match — not a fuzzy match. I have seen a team's hit rate go from 8% to 91% by moving a `"Current time: ..."` line from the first sentence of the system prompt to the last message.

**💰 Math:** 12k-token system prompt + tools, 200k calls/day, $3/M input, cache reads at ~10% of base and cache writes at ~1.25× (**📅 Volatile** — Anthropic-style economics; OpenAI's automatic caching has used a different discount, verify both). Uncached: 12,000 × 3/1e6 = $0.036/call → 200,000 × $0.036 = **$7,200/day = $216k/month**. At a 90% hit rate: 10% of calls pay the write premium (0.1 × 200k × 12,000 × 3.75/1e6 = $90) and 90% pay the read rate (0.9 × 200k × 12,000 × 0.30/1e6 = $648), total ≈ **$738/day = $22.1k/month**. **Saving ≈ $194k/month from reordering a prompt.** Nothing else in applied LLM engineering has that ratio of effort to dollars, which is why prefix-cache hit rate belongs on the main dashboard next to error rate.

**Exact-response cache** — hash the full normalized request, return the stored response. Safe, boring, and effective on the head of the query distribution. Hit rates of 15–40% are realistic for consumer products with a heavy head and near zero for personalized or agentic traffic.

**Semantic cache** — embed the query, return a stored answer if cosine similarity exceeds a threshold. This is the dangerous one and it gets its own treatment shortly. Its "hit rate" is meaningless without a paired *false-hit rate*, and reporting the first without the second is the exact mistake I would flag in review.

**⚠ Trap:** measuring cache hit rate at the request level when the provider caches at a block granularity (commonly 128 tokens, and with a minimum cacheable prefix length). A 300-token prompt may be entirely uncacheable, and a prefix that matches for 1,900 of 2,000 tokens gives you the first 1,792 cached and the rest recomputed. Report *cached input tokens / total input tokens* from the provider's usage fields, not a boolean hit/miss you computed yourself.

### Last one on this: here's our current service dashboard — latency percentiles, error rate, QPS, cost per request. Redesign it for an LLM product and tell me what's missing entirely.

I would keep exactly one of those five unchanged — error rate — and even that gets split.

**Row 1: user-perceived latency.** TTFT p50/p95/p99 as the headline. ITL p99 next to it, because that is your stutter detector. TPOT p95 third, with the reading-speed reference line at 180 ms drawn on the chart so nobody optimizes past the point of human perceptibility. E2E stays, but demoted, because it is a derived quantity now.

**Row 2: capacity, leading indicators only.** KV-cache utilization (or, if you are API-only, provider token headroom from the rate-limit headers). Waiting-queue depth and time-in-queue. Preemption rate. Time-in-limiter as a distinct series from time-at-provider — that split is non-negotiable, it is the difference between a diagnosable and an undiagnosable p99.

**Row 3: throughput and economics.** Goodput — completions per second meeting SLO — not QPS. Prefill and decode tokens/sec separately. Prefix-cache hit rate as *cached input tokens ÷ total input tokens*. Cost per resolved task, with cost per request kept only as a debugging sub-metric.

**Row 4: quality, which does not exist on your current dashboard at all, and that is the real finding.** This is where the interview is actually won. Your existing dashboard cannot distinguish "working" from "returning fluent nonsense," because every LLM failure mode returns HTTP 200. So: refusal rate, empty-or-degenerate-output rate, schema-validation failure rate on structured outputs, retrieval staleness (`now() − oldest indexed document`), and a sampled online judge score on a fixed rubric with its human-agreement rate published next to it so nobody over-trusts it. Plus tool-call error rate broken down by tool, which is the highest-signal single chart in any agent system.

**Row 5: safety and abuse**, sized to your product — prompt-injection detector hits, PII-in-output detections, jailbreak-classifier rate.

**⚠ Trap:** the one that motivates the entire redesign — **every catastrophic LLM failure is an HTTP 200.** A hallucinated policy, a stale retrieval, a silently truncated context, a semantic cache serving the wrong tenant's answer: all of them are green on your current board. If you take one thing from this section into a design round, make it the sentence "my existing observability stack cannot see any of the failure modes that matter here, so the first thing I'd build is the quality row."

**🗣 Say this in the room:** "The honest answer is that four of those five metrics measure the wrong quantity and the fifth can't see the failures that matter. Latency becomes TTFT and inter-token latency because streaming has an onset and a rate. QPS becomes goodput because a request isn't a unit of work. Cost per request becomes cost per resolved task. And I'd add a whole row that doesn't exist today — refusal rate, schema failures, retrieval staleness, sampled judge scores — because every bad answer this system produces returns a 200."
