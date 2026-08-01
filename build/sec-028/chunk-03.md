### Apply Little's Law to an inference server for me. What does it actually tell you here?

`L = λ · W` — average number in the system equals arrival rate times average time in system. It is an identity, not a model: it holds for any stable system regardless of arrival distribution or service discipline, which is why it is the first thing I write on the whiteboard for a capacity question. In inference terms: **average concurrent requests in flight = requests per second × average end-to-end latency.**

The reason it matters more here than in a typical backend service is that in inference, `L` is not an abstraction — it is a physical resource you have already bought. `L` is the number of sequences resident in the KV cache, and KV capacity is finite and computable. So Little's Law lets you convert between three quantities you are given in different units by different stakeholders: the product manager gives you λ (QPS), the SLO gives you W (latency), and the hardware gives you L (KV budget). Any two determine the third.

**💰 Worked, both directions.** Product says 12.7 req/s at peak; the answer shape is 4k in, 600 out; SLO is 400 ms TTFT and 25 ms ITL, so W = 0.4 + 599 × 0.025 = **15.4 s**. Then L = 12.7 × 15.4 = **196 concurrent sequences**. Now check the hardware: at 4,600 tokens average residency (4,000 prompt + 600 generated) and 0.156 MB/token for a 70B at FP8 KV, that is 196 × 4,600 × 0.156 MB = **141 GB of KV**. An H100 node (8 × 80 GB = 640 GB) holding 70 GB of FP8 weights and ~50 GB of activations/CUDA context has ~520 GB free — so KV capacity is not the binding constraint here; decode step time at batch 196 will be. That is exactly the kind of conclusion you want to reach in 90 seconds.

Run it the other way for the more common interview framing: "we have 4 nodes, each can hold batch 64 without violating ITL, so L_max = 256. Our W is 15.4 s. Therefore λ_max = 256 ÷ 15.4 = **16.6 req/s**." Anything above that queues, and the queue is unbounded.

**⚠ Trap:** the most common misuse is applying Little's Law with `W` = TTFT. `W` must be *time in system*, which for a streaming response is the full duration the sequence occupies a KV slot — TTFT plus the entire decode. A request that streams for 15 seconds holds a slot for 15 seconds. I have seen a capacity plan off by 30× because someone used a 500 ms TTFT as W. **Output length, not TTFT, drives your concurrency requirement**, which is why a product change from 300-token to 900-token answers is a capacity event.

### Why can't I just run the fleet at 90% utilisation like I do for a stateless web service?

You cannot run *any* queueing system at 90% utilisation and expect a good tail, but LLM serving is worse than most, for three compounding reasons.

The baseline reason is the standard queueing result: for an M/M/1-ish system, expected waiting time is `W_q = ρ/(1−ρ) × S` where ρ is utilisation and S is service time. At ρ = 0.5 you wait 1× a service time; at 0.8, 4×; at 0.9, **9×**; at 0.95, **19×**. That hockey stick is the whole story, and it is why every SRE you have worked with targets 60–70%.

LLM serving compounds it three ways. **First, service time variance is enormous.** A stateless HTTP handler has a service-time distribution with a coefficient of variation near 1; LLM service time is proportional to output length, whose distribution routinely spans 50 to 4,000 tokens — a coefficient of variation well above 1. Queueing delay scales with `(1 + C²)/2`, so high variance multiplies your wait at any given ρ. **Second, service time is not independent of load**: as batch grows, per-step time grows, so service time *increases* with utilisation. That is positive feedback the classic model does not have. **Third, you cannot shed the cost of a partially-served request** — a request preempted at token 400 has consumed 400 tokens' worth of GPU and produced nothing billable and nothing useful.

**📐 Numbers you must know:** the utilisation target I defend in design review is **65–75% of the batch size at which ITL still meets SLO**, not 65–75% of peak throughput. Those are different numbers and the second one is a trap. Concretely: if ITL SLO is met up to batch 64 and throughput peaks at batch 192, plan to sit at ~45 average concurrent per replica, not at 130.

**🗣 Say this in the room:** "I'd size for 65–75% of the SLO-feasible batch, not of peak throughput. Queueing delay is ρ/(1−ρ) times service time, and LLM service time has both huge variance and a positive feedback loop with load, so the knee arrives earlier and harder than it does for a stateless service."

### Traffic is bursty. Write me the admission control policy you'd actually ship.

Admission control for inference is unusual because you can *estimate the cost of a request before you run it* — you know the prompt length, and for most products you can bound or predict the output length. That makes it strictly better-informed than the connection-count admission control you are used to. Here is the policy I ship, in order of evaluation:

```python
def admit(req, engine, now):
    # 1. Hard capacity gate: can the KV cache hold the worst case?
    need = (req.n_prompt_tokens + req.max_tokens) * BYTES_PER_TOKEN
    if engine.free_kv_bytes() - engine.reserved_bytes < need:
        return Defer("kv_capacity")

    # 2. Predictive SLO gate: would admitting this break the running set?
    projected_batch = engine.n_running + 1
    if predicted_itl(projected_batch) > SLO[req.tier].itl * 0.9:
        return Defer("itl_headroom")

    # 3. Queue-time gate: is this request already doomed?
    waited = now - req.arrival_ts
    est_prefill = req.n_prompt_tokens / PREFILL_TOK_PER_S
    if waited + est_prefill > SLO[req.tier].ttft:
        return Reject(429, "ttft_unattainable")   # shed, don't queue

    # 4. Per-tenant token-bucket on *tokens*, not requests
    if not engine.tenant_bucket[req.tenant].consume(req.n_prompt_tokens):
        return Reject(429, "tenant_rate_limit")

    return Admit()
```

Four things in there are the whole answer. **Gate 3 is the one nobody writes**: if a request has already waited long enough that it cannot meet its TTFT SLO even with instant service, admitting it burns GPU to produce a violation. Reject it immediately with a 429 and a `Retry-After`. This is load shedding as a *goodput* optimisation, not as a capacity backstop, and it is the difference between graceful degradation and a system where everyone gets a slow answer.

**Gate 4 meters tokens, not requests.** You already know this instinct from rate limiting, but the unit changes: a tenant sending ten 100k-token prompts consumes 200× the capacity of a tenant sending ten 500-token prompts. Rate limiting on requests-per-minute for an LLM gateway is the equivalent of rate limiting a database on connections while ignoring query cost. Meter on tokens-per-minute, ideally with separate input and output buckets since they have different costs.

**Gate 1 reserves for the worst case** (`max_tokens`), which is conservative and will under-admit. The refinement — and Mooncake does a version of this — is to *predict* output length from the request features and reserve the predicted p90 rather than the max, then preempt if you are wrong. That is a meaningful capacity win (often 20–40% more admitted concurrency) and a meaningful source of preemption incidents. I would ship the conservative version first and add prediction only with a preemption path already tested.

**⚠ Trap:** implementing admission control at the engine and not at the gateway. By the time a request reaches the engine you have already paid for TLS, auth, retrieval and reranking. Shedding at the engine wastes the 200 ms of retrieval you just did. The gate belongs at the front, informed by engine state pushed on a fast interval.

### Queue it, shed it, or degrade it? How do you choose under overload?

Ranked by what actually preserves user value, and the ranking is the answer:

**Degrade first.** Route the request to a smaller, faster model. A user who gets a good-enough answer from an 8B in 900 ms is served; a user who waits 12 s for the 70B is not, and a user who gets a 429 definitely is not. This is the option backend engineers systematically under-use because stateless services have no analogue — there is no "smaller version of your API handler." It exists here and it is your best overload response. Prerequisites: a routing layer, a second deployment kept warm (which costs money in steady state — that is the premium you pay for the option), and an eval that tells you *which request classes* degrade acceptably. Extractive RAG answers degrade well; multi-step reasoning and code generation degrade badly.

**Shed second.** For requests that cannot degrade and cannot meet SLO, a fast 429 with `Retry-After` is more honest than a slow 200. Shed by *tier* — background agent jobs and batch first, interactive last — and shed by *predicted cost*, dropping the 120k-token request before the 800-token one, because one of them is worth 150 of the other.

**Queue last, and bounded.** Queue only for the short window where you expect the burst to drain — tens of seconds, not minutes. An unbounded queue is a latency amplifier disguised as resilience: it converts "some users get errors" into "all users get timeouts," and then your clients retry, which is where the real outage starts. My rule: **queue depth capped at the number the fleet can drain within the TTFT SLO**, computed as `drain_rate × ttft_slo`, and everything beyond that is shed. With a fleet doing 16 req/s and a 400 ms SLO that is a queue of **6**. Yes, six. That number shocks people and it is correct.

**🗣 Say this in the room:** "Under overload I degrade before I shed and shed before I queue. The queue is capped at drain-rate times the TTFT SLO — for a 16 req/s fleet with a 400 ms budget that's a queue depth of six, and anything deeper is a request that's already failed, it just doesn't know it yet."

**💰 Math on why degrading wins:** an 8B model at TP1 prefills 4k tokens in 4,096 ÷ (400e12 ÷ 16e9) = 164 ms and decodes at ~120 tok/s at batch 32. Serving the overflow on a single 8B replica costs $3/hr and absorbs maybe 40 concurrent users; adding a 70B TP8 node to absorb the same load costs $24/hr. **Eight times cheaper**, at a quality cost you can measure on your eval and choose to accept during a spike.

### You've got interactive chat, background agents and nightly batch on the same fleet. How do you keep them from eating each other?

You give them different SLOs, different admission budgets, and different scheduling priority — and critically, you make the low-priority tiers *preemptible* rather than just deprioritised, because a long-running background generation that got admitted will hold its KV slot for minutes regardless of how low its priority is.

The tier design I use:

| Tier | TTFT SLO | ITL SLO | Admission | Preemptible |
|---|---|---|---|---|
| Interactive (chat, autocomplete) | 400 ms p95 | 25 ms p95 | always, up to reserved floor | no |
| Agent background (tool loops, long tasks) | 5 s p95 | 100 ms p95 | only above interactive floor | yes, recompute |
| Batch (nightly indexing, evals) | none | none | separate deployment | n/a |

Three mechanisms make it real. **A reserved floor for interactive**: a fixed fraction of KV capacity (say 60%) that lower tiers may not occupy, so a burst of agent traffic cannot lock interactive out. This is the same idea as reserving connections for a superuser role in Postgres, and for the same reason. **Priority in the scheduler's waiting queue**, so interactive requests jump the line — every serious engine supports request priority, and if yours does not, implement it in the gateway by holding low-tier requests back. **Preemption with recompute for the agent tier**: when interactive demand rises, evict an agent sequence's KV blocks and re-prefill it later. Recompute is the right preemption mode here because the agent tier's SLO is loose enough to absorb a re-prefill, and swap-to-host costs PCIe bandwidth you would rather spend elsewhere.

Batch gets a **separate deployment**, not a low priority on the shared one. Different config entirely: chunked prefill off, token budget maxed, `max_num_seqs` cranked, ITL irrelevant, running on spot/preemptible instances at 60–70% discount. Trying to run batch on the interactive fleet at low priority is a false economy — you get the interference without the throughput configuration that makes batch cheap.

**⚠ Trap:** treating "agent background" as low priority and then discovering it is user-facing. In products like Cursor's agent mode or a Sierra support agent, the "background" tool loop *is* the user experience; the user is watching a spinner. The tier boundary is not interactive-versus-background, it is **synchronously-observed versus not**. I have seen a whole tiering scheme invalidated by that distinction, and the tell is a support ticket saying "the agent is slow" from a tier you classified as latency-insensitive.

### What signal do you autoscale on, and how do you handle the fact that a 70B takes minutes to come up?

The signal first. CPU utilisation is meaningless. GPU utilisation is nearly meaningless because a decode-bound worker sits at 30–40% SM occupancy while being completely saturated on bandwidth. QPS is misleading because requests have wildly different costs. The three signals that work, in priority order:

**Queue wait time (p95).** This is the leading indicator and the one closest to the SLO. If p95 queue wait exceeds 25% of your TTFT budget, you are out of capacity *now*. It reacts fast and it directly encodes the thing you promised.

**KV cache utilisation.** The capacity indicator. Above ~80%, the engine starts preempting and your ITL tail degrades. It is smoother than queue time and better for scale-*down* decisions.

**Running-batch size versus SLO-feasible batch.** The ratio you actually want to hold at 0.65–0.75.

I scale up on queue wait (fast, aggressive, low threshold) and scale down on KV utilisation with a long stabilisation window (slow, conservative). Asymmetric policies are correct here because the costs are asymmetric: scaling up late costs SLO violations, scaling down late costs dollars, and dollars are cheaper than an incident.

**💰 The cold-start math, which is the real constraint.** Bringing up a 70B replica: cloud node provisioning 120–300 s; pulling a 70 GB FP8 checkpoint from object storage at ~1 GB/s = **70 s** (or 14 s at 5 GB/s from a node-local NVMe cache — this is why you bake or cache weights); loading into HBM and sharding across TP ranks 20–40 s; CUDA graph capture and warmup 30–90 s. Total: **4–8 minutes**. Compare that to the burst you are trying to absorb, which in a consumer product is often 60–120 seconds wide. **Autoscaling cannot respond to your bursts.** That is the sentence to say out loud.

So the policy has to be: (1) keep enough steady-state headroom to absorb the burst without scaling — sized from your measured burst amplitude, e.g. if p99 minute-over-minute traffic is 1.6× the mean, carry 60% headroom above mean; (2) keep a **warm pool** of loaded-but-idle replicas, which costs money and is the correct thing to spend it on; (3) shed and degrade during the gap rather than pretending the autoscaler will save you; (4) pre-scale on schedule for predictable diurnal patterns — 9 a.m. in each major timezone is not a surprise, and a cron-driven scale-up at 8:45 is more reliable than any reactive policy.

**⚠ Trap:** scaling on a rolling 5-minute average of anything. Add a 5-minute averaging window to a 6-minute cold start and your control loop has an 11-minute delay against a 90-second disturbance. That system does not converge; it oscillates. Use short windows for scale-up (30–60 s) and long ones only for scale-down.

### Walk me through capacity planning for a new LLM feature. What do you need to know before you can size anything?

Six inputs, and I refuse to size without them because every one of them moves the answer by more than 2×:

1. **Peak QPS, not mean.** With the peak-to-mean ratio and the width of the peak. A 5× peak that lasts 20 minutes is a different problem from a 5× peak that lasts 3 hours.
2. **Input token distribution** — p50, p90, p99. Not the mean. The p99 sets your TTFT tail and your prefill pool size.
3. **Output token distribution** — p50, p90, p99. This sets `W` in Little's Law and therefore your concurrency.
4. **Prefix cache hit rate**, or an estimate of the shared-prefix fraction. A 70% hit rate cuts effective prefill work by 70% and can halve your prefill fleet. This is the input people forget and it is often the largest single factor.
5. **The SLO pair** (TTFT and ITL, at a stated percentile) and whether it is tiered by prompt length.
6. **The model and precision**, because everything downstream is derived from bytes-per-token and FLOPs-per-token.

Then the derivation is mechanical: prefill token demand = QPS × mean input × (1 − cache hit rate); decode token demand = QPS × mean output; concurrency from Little's Law; KV bytes from concurrency × mean residency × bytes/token; replicas = max(demand ÷ per-replica capability, KV bytes ÷ per-replica KV budget); then multiply by (1 ÷ 0.7) for the utilisation target and add the burst headroom.

**⚠ Trap:** sizing from a benchmark run at a fixed 1k-in/1k-out shape. Every vendor benchmark and most internal ones do this, and it is a fiction. Real traffic's cost per request is dominated by a long right tail on input length; sizing on the mean shape under-provisions prefill by 2–4× for agentic workloads. Always re-run your capacity benchmark by *replaying a sample of production request shapes*, not synthetic ones. If you do not have production yet, take the p50/p90/p99 from the closest analogous product and say so explicitly.

**🗣 Say this in the room:** "Before I size anything I want six numbers: peak QPS with peak width, the input and output token distributions at p50/p90/p99, the expected prefix-cache hit rate, the SLO pair with its percentile, and the model plus precision. Give me those and I'll have a fleet size in five minutes. Without the cache hit rate and the input tail I'd be guessing by a factor of three."

### What are the streaming UX thresholds, and what do you do when the hardware won't hit them?

**📐 Numbers you must know, with derivations:**
- **TTFT under ~200 ms** is below the threshold where a UI transition feels instantaneous; 200–500 ms reads as responsive; beyond ~1 s the user disengages and re-reads their prompt. These are the same perceptual bands you already use for page interactions — LLM streaming did not change human perception.
- **Reading speed**: ~250 words/min silent reading = 4.2 words/s ≈ **5.5 tokens/s** at ~1.3 tokens/word for English. So 30 tok/s is 5.5× reading speed, 50 tok/s is 9×.
- The consequence: **anything past ~40 tok/s is invisible.** The user cannot read it. Spending GPU to go from 45 to 70 tok/s is spending money on a metric no user perceives, and that budget should go to batch size (cost) or to TTFT (perceived).

When you cannot hit them, you have four moves, and three of them are UX rather than systems:

**Fill the TTFT gap with real content.** Stream the retrieval sources, the plan, or the tool the model is about to call, before the answer tokens exist. Perplexity-shaped products do this: sources appear at ~250 ms, answer tokens at ~800 ms, and perceived TTFT is 250 ms. This is the highest-leverage thing on the list and it costs no GPU.

**Stream at the semantic unit the user consumes.** For code completion, a 300 ms wait for a complete, correct 20-token suggestion beats a token-by-token dribble — nobody reads a completion character by character. For chat, token-level streaming is right. Match the streaming granularity to the artifact.

**Cut prefill work rather than buying compute.** Prefix caching for the static preamble, dropping retrieved documents from 12 to 6 (measure the recall cost), summarising conversation history. Each of these is a linear reduction in prefill and therefore in TTFT.

**Accept a slower stream and say so.** For a 90-second deep-research task, TTFT of 3 s is fine if the UI shows what it is doing. The threshold is not universal; it is set by what the user believes they asked for.

**⚠ Trap:** optimising ITL because it is easy to measure while TTFT is the thing users feel. I have reviewed a project that spent six weeks and a hardware upgrade taking ITL from 28 ms to 19 ms — invisible — while TTFT sat at 1.4 s because nobody had turned on prefix caching for a 3,000-token system prompt.

### Timeouts, retries and hedging on a streaming endpoint — what breaks?

Everything you know about HTTP retries, because the request is stateful, expensive, non-idempotent in effect, and partially delivered.

**Timeouts.** A single request timeout is wrong: a 3,000-token answer legitimately takes 75 s at 25 ms ITL. If you set a 30 s deadline you kill healthy long generations. The correct instrument is **two timeouts**: a TTFT deadline (kill if no first token within, say, 5 s — the request is stuck in queue or the worker is wedged) and an **idle-gap deadline** (kill if the gap between consecutive tokens exceeds, say, 10 s — the stream is dead even though the connection is open). Total-duration deadlines should be generous and exist only as a backstop against runaway loops.

**Retries.** Retrying a request that has already streamed tokens is not a retry, it is a second answer. If the client has rendered 200 tokens and you retry, you must either discard the rendered text (jarring) or splice (incoherent, because the new generation does not know what the old one said). My rule: **retries are only safe before the first token.** After the first token, the only correct behaviours are to finish, to fail visibly, or to hand the emitted prefix to the retry as context and tell the model to continue — which is an application-level decision with its own quality risk. Encode this in the client library, not in each caller.

**Retries also amplify cost superlinearly under load.** A retry on a saturated system adds a full prefill of load to a system that is failing *because* it is saturated. This is the classic retry storm, but each retry costs 4,000 tokens of prefill rather than 200 bytes of TCP. Always: exponential backoff, jitter, a **retry budget** (I cap total retries at 10% of requests fleet-wide via a token bucket, so a broad failure cannot double the load), and circuit breaking.

**Hedging.** Sending a duplicate request to a second replica after a delay is a beautiful tail-latency technique for idempotent reads and a **bad** one here: you have doubled prefill cost for the hedged fraction, and if both complete you have paid twice for one answer and must discard one. It is defensible only when (a) hedge rate is capped at low single-digit percent, (b) you hedge *before* prefill starts (i.e. on queue-wait, not on slow decode), and (c) you cancel the loser immediately and the engine actually honours cancellation by freeing KV blocks. That third condition fails in more engines than you would expect — verify that a cancelled request releases its blocks promptly, or hedging leaks capacity.

**🗣 Say this in the room:** "I set a TTFT deadline and an inter-token idle deadline rather than one total timeout, because a long generation is not a hung one. Retries are safe only before the first token, they're capped by a fleet-wide budget because each retry costs a full prefill, and I'd hedge only on queue wait with cancellation verified to actually free KV blocks."

### Our p95 TTFT is 380 ms and our p95 ITL is 22 ms — both green — but users say it feels slow. What's going on?

Green dashboards with unhappy users almost always means you are measuring the wrong boundary, the wrong aggregation, or the wrong cohort. Here are the five causes in the order I check them:

**You are measuring at the engine, not the edge.** Engine TTFT excludes gateway, auth, guardrail, retrieval and rerank. In the waterfall I use, that is 270 ms of work in front of the model. Engine p95 TTFT of 380 ms is edge p95 TTFT of 650 ms, and if there is an SSE-buffering proxy in between it could be worse. **Fix: measure at the outermost hop you control, and reconcile the two numbers on the same dashboard.**

**Percentiles by request, not by user or session.** A user in a 20-turn conversation experiences 20 draws from your latency distribution. The probability that at least one of them lands in the p95 tail is `1 − 0.95²⁰ = 64%`. **Nearly two-thirds of your multi-turn sessions contain a p95 event.** Your per-request p95 can be beautiful while the per-session experience is routinely bad. This is the single most important statistical point in latency SLOs for chat products and almost nobody raises it unprompted.

**Pooled ITL hides stalls.** If you average ITL across a request, a stream that emits 400 tokens at 15 ms and then stalls for 900 ms has a mean ITL of 17 ms. Green. The user saw the text freeze. **Chart per-request max inter-token gap at p95 and p99 — that is your stall metric and it is the one that correlates with complaints.**

**A cohort is being hidden by aggregation.** Slice by input-length bucket, by tenant, by tier, by model, and by cache-hit-versus-miss. It is routine to find aggregate p95 at 380 ms while the >16k-prompt bucket sits at 3.2 s, and your loudest customers are entirely inside that bucket.

**Time-to-*useful*-token, not time-to-first-token.** If the model emits a 40-token preamble ("Certainly! Let me look into that...") before the substance, or if extended thinking emits reasoning tokens the user does not see, first-token time is meeting SLO while first-*content* time is 2 s later. With a thinking model this can be tens of seconds. **Instrument time-to-first-visible-content separately** and treat it as the real SLO for a thinking-enabled path.

**⚠ Trap:** trusting a p95 computed over a 1-hour window. Latency incidents are minutes wide; a 1-hour p95 smooths a 4-minute total-collapse into a shrug. Compute SLO compliance on 1-minute windows and alert on consecutive bad windows.
