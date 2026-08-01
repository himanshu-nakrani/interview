### Your team autoscales the inference fleet on GPU utilization. Tell me why that's wrong.

Because GPU utilization, as reported by `nvidia-smi` and DCGM's `DCGM_FI_DEV_GPU_UTIL`, measures *whether at least one kernel was resident on the SMs during the sample interval*. It does not measure how full the machine is. An LLM decode step is memory-bandwidth-bound — the GPU spends its time streaming weights from HBM, with SMs occupied but underfed — so a single request generating one token at a time will show 90–100% "utilization" while using a fraction of the available FLOPs and a fraction of the KV cache.

The consequence is that the signal saturates long before the system does, and then stays pinned while everything you actually care about degrades. You scale up at 80% utilization, you hit 100% at low load, and from there the metric is flat and uninformative as your queue grows from 0 to 400 requests and p95 goes from 1.2 s to 40 s. The autoscaler sees "still 100%" and does nothing. It is a broken thermometer that reads 100°C at every temperature above room.

The backend analogue you already trust: you do not autoscale a Postgres-backed service on "is the CPU doing something," you scale on connection-pool saturation and queue wait. Same idea, different resource.

**What to use instead**, in the order I prefer them:

1. **Pending-request queue depth** (or, better, *waiting time* of the oldest queued request). This is the most direct proxy for "requests are not being served now," it responds instantly, and every engine exports it — vLLM exposes `num_requests_waiting` alongside `num_requests_running`. If you can pick only one signal, pick this.
2. **KV-cache utilization** (`gpu_cache_usage_perc`). This is your real capacity gauge: when the paged cache is 90% full, the scheduler starts refusing to admit new sequences or preempting existing ones, and latency degrades regardless of what the SMs are doing. It is also *leading* — cache fills before the queue backs up.
3. **Tokens/second throughput against a known ceiling.** If you have benchmarked that a replica sustains 2,000 output tok/s at your SLO, then current tok/s ÷ 2,000 is a clean utilization fraction with meaning.

**📐 Numbers you must know:** the thresholds I start from and then tune — scale up when `num_requests_waiting > 2 × replicas` sustained for 30 s, **or** when KV-cache utilization exceeds 0.85, **or** when p95 TTFT exceeds 70% of the SLO. Scale down only when all three are below half their up-thresholds for 10+ minutes. The asymmetry is deliberate: scaling up is cheap and reversible, scaling down on a GPU fleet is expensive to undo because of cold start.

**🗣 Say this in the room:** "GPU utilization pins at 100% for a single decode stream because the workload is memory-bandwidth-bound, so it carries no information above trivial load. I autoscale on pending-queue depth and KV-cache utilization, with p95 TTFT as a guard, and I use utilization only as a health check for 'is this replica doing anything at all.'"

### Design the autoscaler concretely. KEDA on what, with what thresholds and windows?

I will describe it as a config, because the specifics are the answer.

**Metrics pipeline.** vLLM (or whichever engine) exports Prometheus metrics on `/metrics`. A ServiceMonitor scrapes each pod every 15 s. KEDA's Prometheus scaler queries an aggregate expression against Prometheus and drives a HorizontalPodAutoscaler under the hood.

**The scaling expression.** I use queue depth *per replica*, because that is the quantity whose target is invariant to fleet size:

```
sum(vllm:num_requests_waiting{deployment="llm-70b"})
  / count(up{deployment="llm-70b"} == 1)
```

with a KEDA `threshold` of 2 — meaning "target two waiting requests per replica." KEDA converts that to a desired replica count with the standard HPA ratio: `desired = ceil(current × currentValue / targetValue)`.

Add a second trigger on KV utilization as the leading indicator:

```
max(vllm:gpu_cache_usage_perc{deployment="llm-70b"})
```

with threshold `0.85`. KEDA takes the **maximum** of the replica counts requested by all triggers, which is exactly the behavior you want — any one signal saying "we're full" scales up.

**Timing knobs, and these matter more than the thresholds.** KEDA's `pollingInterval` at 15 s (default 30 is too slow for a bursty product). `cooldownPeriod` at 600 s before scaling to zero, if you use zero at all. Then the HPA behavior policies, which KEDA passes through:

- **Scale-up:** `stabilizationWindowSeconds: 0` — react immediately, no averaging window. `policies: [{type: Percent, value: 100, periodSeconds: 60}]` — allow doubling every 60 s. On a fleet with a 90-second cold start, being slow to start is the dominant cost.
- **Scale-down:** `stabilizationWindowSeconds: 600` — require ten minutes of sustained low demand. `policies: [{type: Pods, value: 1, periodSeconds: 300}]` — remove at most one replica every five minutes.

**`minReplicaCount`** is not zero for interactive traffic. Set it to your p10 concurrency, so you never cold-start on a user's request.

**Graceful drain on scale-down.** `terminationGracePeriodSeconds` must exceed your longest generation. If your `max_tokens` is 4,096 and you generate at 40 tok/s, that is 102 s, so set 180. And the engine must handle SIGTERM by removing itself from the endpoints, finishing in-flight requests, and *then* exiting. A 30-second default grace period truncates long generations on every scale-down, and it will show up in your metrics as a mysterious cluster of `max_tokens`-shaped truncations correlated with traffic dropping.

**⚠ Trap:** aggressive symmetric scale-down. GPU pods take 60–120 s to become useful; a fleet that flaps between 4 and 6 replicas every ten minutes spends a meaningful fraction of its capacity cold-starting, and each removal drops that replica's prefix cache on the floor, so your cache hit rate degrades every time you scale in. Asymmetric policies — fast up, slow down — are the correct shape, and the 10-minute stabilization window is not a rounding detail, it is the design.

### Derive for me how much headroom I need given my scaling lag and my traffic burst shape.

The mental model: an autoscaler is a control loop with dead time, and dead time is unrecoverable. Between the moment demand rises and the moment new capacity serves a request, your existing fleet absorbs 100% of the excess. Headroom is simply the capacity that must be pre-provisioned to survive that window without violating the SLO.

**Enumerate the lag terms**, because candidates usually count one and there are five:

| Term | Typical |
|---|---|
| Metric scrape + Prometheus ingest | 15–30 s |
| KEDA poll + HPA reconcile | 15–30 s |
| Pod scheduling (existing node) | 1–5 s |
| Node provisioning (if scale-up needed) | 60–240 s |
| Image pull (if not cached) | 30–180 s |
| Weight load + engine warm | 30–120 s |

Best case, warm node pool, cached image: **60–90 s**. Worst case, new node from the cloud: **5–10 minutes**. Measure yours; do not assume.

**Now the arithmetic.** Let baseline demand be *D* requests/s, each replica serve *C* req/s at SLO, current replicas *R* with *R·C ≥ D*. A burst raises demand to *D·k* at time 0, and capacity does not change until time *L*.

During the lag you must satisfy `D·k ≤ R·C`. Since you provisioned `R = ceil(D/C) · (1 + h)` with headroom fraction *h*, the requirement is **`1 + h ≥ k`** — i.e. *headroom must cover the entire burst multiplier*, because the autoscaler contributes nothing during *L*.

That is the sobering result: if your traffic can double in under your scaling lag, you need 100% headroom, full stop. No autoscaler configuration fixes it.

What actually saves you is that requests **queue** rather than fail, and queueing converts a capacity shortfall into a latency cost. If your SLO allows a p95 of 2 s and a request normally takes 1.2 s, you have ~0.8 s of queue budget per request. Excess arrivals during the lag accumulate as backlog `(D·k − R·C) · L` requests, which drains at rate `(R'·C − D·k)` after new capacity lands. Keeping the peak queue wait under 0.8 s at a lag of 90 s and a service rate of, say, 8 req/s/replica means the shortfall must stay under roughly `0.8 × 8 / 90 ≈ 0.07` replicas' worth — effectively zero. **Queueing does not rescue you at a 90-second lag; it only rescues you at a 5-second lag.**

**💰 Math, made concrete.** Baseline 40 req/s, replica capacity 8 req/s → 5 replicas. Observed p99 burst multiplier from a month of traffic: 1.6×. Required headroom = 60% → 8 replicas standing. At 8 GPUs/replica × $2.50/hr, 5 replicas = $100/hr, 8 replicas = $160/hr — **$43,800/month of headroom** to survive a burst you see a few times a week. That number is the honest price of the SLO, and putting it in front of a product owner usually produces a productive conversation about whether p95 2 s is really the requirement, or whether a queue-with-a-spinner at p99 is acceptable.

**The cheaper levers,** which I would propose before buying headroom: (1) cut the lag — pre-pulled images, warm node pool, NVMe weight cache, taking you from 300 s to 75 s; (2) predictive scaling on the diurnal curve so you are already scaled before the daily peak, leaving reactive scaling to handle only unpredictable bursts; (3) a burst overflow path to a managed API, paying a per-token premium for the few minutes per week you need it — this is often the cheapest of the three by an order of magnitude.

**🗣 Say this in the room:** "Headroom has to cover the full burst multiplier over the scaling lag window, because the autoscaler contributes zero capacity during that window. So I attack the lag first — warm pools, pre-pulled images, cached weights — then buy the residual headroom, and I'd wire a managed-API overflow for the tail rather than standing up GPUs for a burst that happens twice a week."

### Design a per-tenant rate limiter for an LLM gateway. What's different from the Redis token bucket I've already built ten times?

Structurally, nothing — it is still a token bucket in Redis with a Lua script for atomicity. Three things are different, and each of them is where a naive port breaks.

**First: you must limit tokens, not requests.** A request is not a unit of work; one request can be 200 tokens or 200,000. Providers rate-limit on TPM (tokens per minute) and RPM simultaneously for exactly this reason, and your gateway should too. So the bucket's currency is tokens.

**Second: you don't know the cost until after the call.** The prompt-token count you know at admission time (tokenize it, or estimate at ~4 characters/token for English and reconcile later). The *completion* token count you do not. The standard pattern is **reserve-then-reconcile**: at admission, debit `prompt_tokens + max_tokens` (the worst case); on completion, credit back `max_tokens − actual_completion_tokens`. This is pessimistic — a tenant that sets `max_tokens=4096` and generates 200 gets throttled as though they generated 4,096 — so you either educate tenants to set realistic ceilings or you reserve `prompt_tokens + p95_completion_for_this_tenant` and accept occasional overshoot. I ship the pessimistic version first because overshoot on a shared GPU fleet means someone else's SLO breach.

**Third: streaming means the debit happens over time.** A long generation holds a KV slot for a minute. If your limiter only debits at the start, a tenant can open 500 concurrent streams within one refill window and occupy the entire fleet. So you need a **concurrency limit as well as a rate limit** — a semaphore on in-flight requests per tenant, which is really a limit on KV-cache footprint. In practice I enforce three ceilings per tenant: tokens/minute, requests/minute, and concurrent in-flight requests.

Sketch of the admission path:

```lua
-- KEYS[1] = bucket key; ARGV = capacity, refill_per_sec, now, cost
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or tonumber(ARGV[1])
local ts     = tonumber(b[2]) or tonumber(ARGV[3])
tokens = math.min(tonumber(ARGV[1]),
                  tokens + (tonumber(ARGV[3]) - ts) * tonumber(ARGV[2]))
if tokens < tonumber(ARGV[4]) then
  redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', ARGV[3])
  return -1                                  -- caller returns 429 + Retry-After
end
tokens = tokens - tonumber(ARGV[4])
redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', ARGV[3])
redis.call('EXPIRE', KEYS[1], 3600)
return tokens
```

Return `Retry-After` computed from the deficit and the refill rate, plus `X-RateLimit-Remaining-Tokens` headers, because clients that can see their budget behave much better than clients that guess.

**⚠ Trap:** rate-limiting only at the gateway and not admitting against actual fleet capacity. A tenant can be within their TPM budget while the fleet is saturated, and letting the request in just moves the queue from your gateway to the engine, where you have less control and worse visibility. The gateway needs a global admission control on top of per-tenant limits: if fleet KV utilization is above 95%, shed the lowest-priority traffic with a 429 rather than accepting it into a queue you cannot drain. Load shedding at the edge is the same discipline you already apply to a saturated thread pool.

**⚠ Trap:** counting tokens with `len(text.split())` or a character heuristic and then being surprised at a 30% error. Use the actual tokenizer for the model in question — and cache the tokenizer, do not construct it per request, because instantiating a tokenizer is milliseconds and it will show up in your gateway's p99.

### One tenant starts sending 200k-token prompts and everyone's TTFT triples. Walk me through the diagnosis and the fix.

The diagnosis is fast if you have the right metrics and impossible if you do not, so let me start there: I want prompt-token-count histograms **sliced by tenant**, TTFT and ITL as separate metrics, and per-step batch composition. With those, the shape is unmistakable — one tenant's p50 prompt length jumps two orders of magnitude at time T, and everyone's TTFT steps up at the same T.

The mechanism, precisely. A 200k-token prefill is a single enormous compute unit. Prefill FLOPs are roughly `2 × params × tokens`; for a 70B that is 2 × 70e9 × 200,000 = **2.8 × 10¹⁶ FLOPs = 28 PFLOPs**. On 8 H100s at a realistic 400 TFLOPS effective each (3.2 PFLOPS), that is **8.75 seconds of solid GPU occupancy**. Without chunked prefill, that prefill runs as one scheduler step and every decode step for every other user is blocked behind it — so all their inter-token latencies get an 8.75-second hole punched in them. Users experience it as the stream freezing.

It also eats the cache: 200k tokens × 320 KB/token = **64 GB of KV**, which on a fleet with ~420 GB of cache is 15% consumed by one request, and it will trigger preemption of other sequences.

**The fixes, in the order I would apply them:**

1. **Turn on chunked prefill.** This is the config-flag fix and it addresses the ITL stutter directly: the scheduler breaks the 200k prefill into chunks of a few hundred tokens and interleaves them with everyone's decode steps. The big prompt's TTFT gets slightly worse; everyone else's ITL becomes smooth. This is the right default and it should already be on.
2. **Cap `max_model_len` at the product's real requirement.** If your product never legitimately needs 200k, do not accept it. The engine's ceiling is not your product's contract.
3. **Per-tenant token quotas** — as in the previous question, limit prompt tokens per minute and concurrent in-flight tokens, so no tenant can occupy an unbounded fraction of the cache.
4. **Route long-context traffic to a separate pool.** This is the structural fix. Long prompts have a fundamentally different cost profile and belong on their own replicas — possibly with higher TP for faster prefill, possibly with context parallelism. Then long-prompt users get consistent (slower) service and short-prompt users are unaffected. Segmenting a fleet by request shape is the same instinct as separating OLTP and analytics traffic, and it is usually right.
5. **Priority-aware scheduling**, if your engine supports it: give interactive short requests scheduling priority over bulk long ones.

**🔍 Failure taxonomy — noisy neighbor, as a decision procedure.** Symptom: latency degrades for many tenants with no deploy and no traffic-volume change. Check in order: (a) prompt-length histogram by tenant — a shape change identifies the culprit immediately; (b) KV-cache utilization and preemption counter — if preemptions are non-zero, you are thrashing the cache, not just queueing; (c) batch-composition metrics — is one request dominating steps; (d) adapter diversity, if you run multi-LoRA, since a tenant using a cold adapter causes swap-in stalls; (e) only then look at the fleet and the hardware. The order matters: four of those five are workload-shape problems and none of them are fixed by adding GPUs.

**🗣 Say this in the room:** "This is a workload-shape problem, not a capacity problem. One 200k prefill is ~28 PFLOPs and occupies the whole fleet for ~9 seconds, so without chunked prefill it blocks every other user's decode step. I'd enable chunked prefill immediately, cap max_model_len to what the product actually needs, put a per-tenant token quota in the gateway, and then move long-context traffic to its own pool because its cost profile is genuinely different."

### How do you build a realistic load test for an LLM service? Why does a 512-in/128-out benchmark lie to you?

Because the ratio of input to output tokens is the single biggest determinant of a serving system's behavior, and a fixed 512/128 test picks one point on that axis — usually a flattering one — and tells you nothing about the rest.

Consider three real workloads on the same model and hardware:

- **Chat:** ~600 in, ~300 out. Balanced. Continuous batching works beautifully.
- **Coding assistant with repo context:** ~25,000 in, ~200 out. Prefill dominates by two orders of magnitude; your throughput is essentially a prefill-FLOPs problem, and prefix caching is worth more than every kernel optimization combined.
- **Reasoning/agent:** ~2,000 in, ~8,000 out (thinking tokens). Decode dominates; you are memory-bandwidth-bound and KV-cache capacity is the binding constraint.

The same fleet will show wildly different tokens/second, different cost per request, and a different bottleneck on each. A benchmark that reports "4,200 tok/s" without stating the length distribution is not a measurement, it is a marketing number.

**How I build the test:**

**Sample lengths from production, not from a config.** Pull a week of request logs, build empirical distributions of prompt length and completion length, and *sample jointly* — they are correlated, and sampling them independently produces requests that do not exist (a 50-token prompt with a 6,000-token completion). If you have no production yet, use your best synthetic corpus and state the assumption loudly.

**Preserve prefix structure.** This is the one people always miss. If 60% of your production traffic shares a system prompt, your load test must too, or you will measure a prefix-cache hit rate near zero and conclude you need twice the hardware. Conversely if your test replays the *same* prompt 10,000 times you will measure a 99% hit rate and under-provision by 2×. Reproduce the *distribution of prefix sharing*, ideally by replaying real prompt prefixes.

**Model arrival as a real process.** Closed-loop load generators (N workers, each sending the next request as soon as the last returns) systematically understate latency under overload, because they self-throttle — this is coordinated omission, and it is the same bug you already know from HTTP benchmarking. Use an open-loop generator with a Poisson (or replayed) arrival process at a fixed target rate, and let the queue grow if the system cannot keep up. That is what production does.

**Measure the right things, separately:** TTFT (p50/p95/p99), inter-token latency / TPOT (p50/p95), end-to-end latency, output tokens/second aggregate, and **goodput** — requests completed *within SLO* per second, which is the only number that matters and the one almost nobody reports. A system doing 5,000 tok/s with 40% of requests over SLO has lower goodput than one doing 3,500 tok/s with 99% in-SLO.

**Tooling:** vLLM ships `benchmark_serving.py` with dataset-driven length distributions (ShareGPT and similar) and both request-rate and concurrency modes; `genai-perf` and `llmperf` occupy similar ground. Any of them is fine as a harness; the value is entirely in feeding it your distribution.

**⚠ Trap:** running the load test against a freshly-started server and reporting the numbers. The prefix cache is empty, the CUDA graphs may not be captured for all shapes, and the page cache is cold. Always run a warm-up phase and discard it — and separately measure the cold phase deliberately, because that is what your users see after every scale-up.

### Walk me through the experiment that finds the knee of the throughput/latency curve.

The mental model: a serving system has a classic queueing response curve. Throughput rises with offered load until the bottleneck saturates, then flattens; latency is roughly flat until saturation, then rises without bound. The **knee** is where throughput stops rising meaningfully and latency starts rising sharply, and the *operating point* is a chosen distance below the knee. You cannot pick an operating point you have not measured, and you cannot capacity-plan without one.

**The experiment, step by step:**

1. Fix everything except load: one replica, production model, production engine config, production-shaped request distribution with realistic prefix sharing.
2. Warm up for 2–3 minutes. Discard.
3. Run an **open-loop** generator at a fixed request rate for 5 minutes. Record TTFT p50/p95/p99, TPOT p50/p95, completed requests/s, output tok/s, goodput at SLO, and — crucially — engine-side KV-cache utilization, queue depth, and preemption count.
4. Step the rate up: 1, 2, 4, 6, 8, 10, 12, 16 req/s. Between steps, drain to idle for 60 s so each step is independent.
5. Stop when p95 exceeds 2× your SLO or when the queue grows monotonically without bound (that is past saturation and further data is noise).
6. Plot output tok/s on x, p95 end-to-end latency on y. The knee is visually obvious.

**What you read off the plot:**

- **Maximum throughput** (the flat asymptote) — useful for batch capacity planning.
- **Maximum throughput at SLO** — this is your replica capacity *C*, and it is the number that goes into every autoscaling and cost calculation. It is always well below the asymptote.
- **The operating point:** I run at 70–75% of *C*. That 25–30% is the burst headroom from the earlier derivation.
- **Which resource saturated.** If KV utilization hit ~100% and the preemption counter started climbing before latency rose, you are cache-bound → more memory, shorter contexts, fp8 KV, or higher TP. If KV stayed at 40% and latency rose anyway, you are compute- or bandwidth-bound → more replicas or faster GPUs. **These two situations have opposite fixes and the plot alone does not distinguish them** — you need the engine-side metrics on the same timeline, which is why step 3 lists them.

Then repeat the sweep across the two or three configurations you are choosing between (TP degree, chunked-prefill chunk size, `max_num_seqs`, quantization) and compare curves. `max_num_seqs` in particular is a direct latency/throughput dial: raising it raises the asymptote and moves the knee left in latency terms.

**💰 Math:** suppose the sweep says one replica sustains 9 req/s at p95 = 1.9 s (SLO 2 s), and the asymptote is 14 req/s at p95 = 11 s. Operating at 75% of 9 = 6.75 req/s per replica. For a 60 req/s peak you need ceil(60 / 6.75) = **9 replicas** = 72 H100s = $180/hr = $131k/month. If you had capacity-planned off the 14 req/s asymptote you would have provisioned 5 replicas and been 44% short at peak, discovering it during your launch. That gap — 5 versus 9 — is the entire practical value of running the sweep, and it is why I will not sign off on a capacity plan that cites a throughput number without a latency number attached.

### How do you validate that your autoscaling actually works before a launch?

You test the *control loop*, not the steady state, and the assertion is on time-to-recovery rather than on throughput. Most teams load-test capacity and never test scaling, then discover during launch that the autoscaler needed six minutes to react to a two-minute burst.

**The test.** Start at baseline load with the fleet in steady state. Step the load — not ramp, *step* — to 2.5× baseline instantaneously and hold for 15 minutes. Record on one timeline: offered load, replica count (desired and ready), queue depth, p95 TTFT, and error rate.

**What you are asserting:**

- **Time to first new replica *Ready*** — this is your true lag *L*. Compare it against the number you assumed when you sized headroom. In my experience the assumed number is optimistic by 2–3× because nobody counted the metric-scrape delay or the readiness-probe warm-up.
- **Peak p95 during the transient**, and for how long it exceeded SLO. Convert that to a number of affected requests: if p95 exceeded 2 s for 4 minutes at 60 req/s, that is 14,400 requests degraded. That is the number you show a product owner.
- **Error rate during the transient** — should be zero if you are queueing, non-zero if you are shedding. Both are valid designs; you need to know which one you built.
- **Whether it converged or oscillated.** Plot desired-replica-count over time. If it saws — 5, 9, 6, 10, 7 — your stabilization windows are wrong and you are burning capacity on churn.

**Then the harder half, which people skip: test scale-*down*.** Drop load to baseline and assert that in-flight generations complete rather than being truncated. Grep for a spike in `finish_reason == "length"` or aborted streams correlated with the scale-in. A too-short `terminationGracePeriodSeconds` produces exactly this, and it is silent — no error, just a user whose answer stopped mid-sentence.

**Also test scale-from-zero if you use it**, and test the node-provisioning path specifically by cordoning the existing nodes first so the scale-up is forced to request new hardware. That is the 5–10 minute path, and it is the one that actually happens during an unusual traffic event.

**⚠ Trap:** running the burst test in a staging cluster with a pre-warmed node pool and concluding your lag is 45 seconds. Production at 3am has no spare nodes. Run the test at least once in an environment where the autoscaler must actually ask the cloud for hardware, and record the p95 of *that* — including the case where the first request for capacity is rejected and retried.

**🏋 Drill (30 min, unaided):** write the load-step script and the assertion list for a burst test of a service you have shipped. Pass criterion: your assertions include time-to-Ready, peak p95, duration over SLO expressed as affected request count, oscillation check, and truncation check on scale-down. If you only asserted "it scaled up," you failed.

### Your p95 was 2 seconds in the load test and it's 6 seconds in production at the same QPS. What's different?

Same QPS is doing a lot of work in that sentence, and my first move is to stop trusting it. Here is the checklist I walk, roughly in order of how often it is the answer.

**Token-length distribution differs.** The load test used a distribution you constructed; production's is real, has a fat tail, and the tail is where p95 lives. Compare the prompt- and completion-length histograms directly — not the means, the p95s. A production p95 prompt of 18k against a test p95 of 4k explains a 3× TTFT difference entirely.

**Prefix-cache hit rate differs.** The test either replayed one prompt (hit rate ~100%, unrealistically fast) or fully random prompts (hit rate ~0%). Check `gpu_prefix_cache_hit_rate` in both. This is my most common finding.

**Concurrency distribution differs.** Same mean QPS, different arrival burstiness. A closed-loop test at 40 req/s produces smooth arrivals; production is Poisson at best and bursty at worst, and p95 is set by the bursts. Compare the distribution of instantaneous in-flight requests, not the mean rate.

**Multi-tenancy and adapter diversity.** Production has 50 LoRA adapters in play, cold swap-ins, and a mix of models on shared nodes. The test had one adapter and one model.

**The rest of the request path.** The load test hit the engine directly; production goes through a gateway, an auth check, a retrieval step, a reranker, and a guardrail model. Instrument end-to-end with distributed tracing and check what fraction of the 6 s is even inside the LLM — I have seen "the model got slower" turn out to be a vector-database p99 and a synchronous safety classifier.

**Fleet composition and neighbors.** Production replicas may be on shared nodes, on a different GPU SKU (mixed A100/H100 pools), or on spot instances that got reclaimed and replaced with something else. Slice p95 by pod and by node — if one pod is the whole story you have a hardware or placement problem, possibly a thermally-throttled GPU or one sitting on a bad NUMA/PCIe path.

**Cold replicas.** If the fleet scales, a fraction of your traffic is always hitting a freshly-started replica with an empty prefix cache and un-captured graphs. Slice p95 by pod age.

**🔍 Failure taxonomy — "prod is slower than the test," as a decision procedure:** (1) compare length histograms, (2) compare prefix-cache hit rates, (3) compare in-flight-concurrency distributions, (4) slice latency by pod, node, SKU and pod-age, (5) decompose the trace end-to-end and confirm the LLM is the slow span at all, (6) only then suspect the model or the hardware. Steps 1–3 resolve this in the large majority of cases, and they cost fifteen minutes if you already emit the metrics — which is the real lesson.

### How do you set SLOs for a streaming chat product versus an agent? Which latency metric actually matters?

The mental model: users do not experience "latency," they experience *responsiveness*, and for a streaming interface that is dominated by when text starts appearing and whether it keeps appearing faster than they read. So decompose:

**TTFT (time to first token)** — how long the user stares at a spinner. This is the metric that governs perceived responsiveness and it is set almost entirely by prefill work: prompt length, prefix-cache hit, queue wait, and prefill TP degree.

**ITL / TPOT (inter-token latency, or time per output token)** — the streaming rate. This is set by decode, which is memory-bandwidth-bound, so it is governed by model size, quantization, TP degree and batch size. The perceptual bar is reading speed: a fast adult reads roughly 250 words/minute ≈ 4 words/s ≈ **5–6 tokens/s**. Anything above ~15 tok/s (67 ms/token) feels comfortably fast; below ~8 tok/s (125 ms/token) feels like waiting.

**End-to-end latency** — matters only when the output is not streamed to a human.

**For a streaming chat product** (Notion AI, Perplexity, a Sierra agent's user-visible turn), I would set: TTFT p95 ≤ 800 ms, p99 ≤ 2 s; ITL p95 ≤ 60 ms (≈17 tok/s sustained); and no end-to-end SLO at all, because a longer good answer is not a worse experience. The trap here is that optimizing end-to-end pushes you toward shorter answers, which is a quality regression dressed as a latency win.

**For an agent doing multi-step tool use where the user waits for a final result**, TTFT of the individual model calls is nearly irrelevant — nobody is reading the intermediate tokens. What matters is **end-to-end task latency** and, underneath it, total tokens generated across all steps, since decode time dominates and scales with token count. So the SLO is on task completion time (e.g. p95 ≤ 25 s for a 4-tool-call task) and the lever is *reducing the number of steps and the tokens per step*, not shaving TTFT. This is the reason a good agent design round talks about step budgets and context trimming rather than kernel performance.

**For a batch/offline job**, none of these. The SLO is a deadline and a cost per document.

**⚠ Trap:** setting one SLO for the whole product. A single "p95 < 2 s" rolled across chat, agent steps, autocomplete and document analysis is unachievable and uninformative — autocomplete needs 200 ms, document analysis legitimately needs 30 s. Define an SLO per *request class*, tag every request with its class at the gateway, and dashboard them separately. Otherwise your one number is dominated by whichever class has the most traffic and tells you nothing about the others.

**🗣 Say this in the room:** "For anything streamed to a human I hold TTFT and inter-token latency separately and don't set an end-to-end SLO, because a longer answer isn't a worse one. The ITL bar comes from reading speed — around 15 tokens/second is where it stops feeling like waiting. For agents nobody reads the intermediate stream, so the SLO is end-to-end task time and the lever is fewer steps and fewer tokens, not faster first tokens."

### What goes on the dashboard on day one for an LLM serving fleet? Assume I already have RED metrics and traces.

You already have request rate, error rate, duration and distributed tracing, and none of them tell you anything specific about this system. The delta is four groups of signals, and I would refuse to launch without them because every debugging procedure in this section reads them.

**Group 1 — latency, decomposed.** Not "duration." TTFT and inter-token latency as separate histograms, each sliced by request class and by prompt-length bucket. The bucketing is the part people skip and it is what makes the dashboard diagnostic instead of decorative: a TTFT p95 of 3 s is meaningless, while "TTFT p95 is 400 ms for the under-2k bucket and 6 s for the over-16k bucket" tells you exactly what to do.

**Group 2 — token economics.** Prompt tokens, cached prompt tokens, and completion tokens per request as histograms, plus their sums per tenant per hour. This is simultaneously your cost meter, your capacity forecast and your regression detector — a prompt-template change that adds 800 tokens shows up here hours before it shows up in the invoice. Emit these at the gateway so self-hosted and API-backed models report identically.

**Group 3 — engine internals**, scraped from the serving engine and not derivable from outside it: running and waiting request counts, KV-cache utilization percentage, prefix-cache hit rate, preemption counter, and the current batch size per step. These four are the ones that distinguish "we need more GPUs" from "our workload shape changed," and no amount of client-side observability substitutes for them.

**Group 4 — hardware, via DCGM:** per-GPU memory used, SM clock (to catch thermal/power throttling), ECC and XID error counts, NVLink and PCIe error counters. Alert on XID errors specifically — they are how you learn a GPU is dying before it takes a TP group with it.

**Then the two derived numbers that go at the top of the dashboard**, because they are what you actually manage:

- **Goodput** — requests completed within SLO per second, per request class. Not throughput. A fleet doing 5,000 tok/s with 40% of requests over SLO is failing, and only this metric says so.
- **Cost per 1k requests**, computed hourly as fleet spend ÷ requests, sliced by class and tenant. Putting a live dollar figure on the wall changes engineering behavior more than any other single thing I have done on these systems.

**⚠ Trap:** averaging TTFT across a fleet that includes cold replicas and warm ones, and across cache hits and misses. Those are three distinct populations with means an order of magnitude apart, and their aggregate mean describes none of them. Slice by pod age and by cache-hit boolean, or the number on your dashboard is a fiction that happens to be numerically stable.

**🗣 Say this in the room:** "RED metrics don't survive contact with this system. I add TTFT and ITL as separate histograms bucketed by prompt length, token counts per tenant as the cost meter, and four engine-internal signals — queue depth, KV utilization, prefix-cache hit rate and preemption count — because those are what distinguish a capacity problem from a workload-shape problem. Then goodput and live cost-per-1k-requests go at the top, because those are the two numbers anyone actually manages."

### It's 3am and p99 latency just tripled with no deploy. Give me your decision procedure.

I want a procedure, not a hunch, because at 3am hunches cost twenty minutes each. Here is the one I run, ordered by (probability × speed to check).

**Step 0 — establish blast radius, 60 seconds.** Is it all traffic or one slice? Slice p99 by: request class, tenant, model, pod, node, GPU SKU, and pod age. If it is one pod, this is a hardware or placement problem and you cordon it. If it is one tenant, it is a workload-shape problem. If it is one request class, it is a routing or content problem. Global degradation is the rarest and most alarming case.

**Step 1 — is it demand or supply?** Look at offered load and at ready-replica count on the same timeline. Three cases: load rose and replicas did not (autoscaler problem — check KEDA/HPA events and whether it is capped at `maxReplicas`, or whether the cloud is rejecting instance requests); replicas fell and load did not (preemption of spot nodes, an OOMKill loop, a node going NotReady, or a failing readiness probe flapping pods out of the endpoint list); neither moved (workload shape or an upstream dependency — go to step 2).

**Step 2 — check workload shape.** Prompt-length p95 by tenant, prefix-cache hit rate, and adapter distribution. A batch job that someone kicked off at 2am against the interactive fleet is a classic 3am cause, as is a customer's nightly ingestion hitting the shared embedding fleet. Also check: did the cache hit rate drop? A router misconfiguration or a scale event that reshuffled consistent hashing will drop hit rate to near zero and triple TTFT with no other symptom.

**Step 3 — check the engine's internal state.** KV-cache utilization, `num_requests_waiting`, and the **preemption counter**. Preemptions climbing means the cache is thrashing: sequences are being evicted and recomputed, which burns compute and inflates latency superlinearly. That points at either a long-context influx or too-aggressive `gpu_memory_utilization`.

**Step 4 — check the hardware.** DCGM: per-GPU SM clocks (thermal or power throttling — a failed fan or a hot aisle shows here), ECC error counts, XID errors, PCIe replay counts, NVLink error counters. A single degraded GPU in a TP group slows the *entire group* to its speed, because every all-reduce waits for the slowest rank. This is the failure mode most specific to GPU serving and the one backend intuition does not prepare you for: one sick device does not degrade 1/8 of your capacity, it degrades all of it.

**Step 5 — check downstream and upstream.** Vector DB p99, embedding-service p99, the reranker, any guardrail model, the object store. And the token-count distribution of *outputs* — a model that started producing longer outputs (because a prompt template changed upstream, or a retrieval change stuffed more context in) raises decode time proportionally with no infra cause at all.

**Mitigations available before root cause**, which you should be willing to pull immediately: scale out manually past the autoscaler; shed the lowest-priority tenant class at the gateway; drop `max_tokens` globally for non-critical classes; route a fraction of traffic to the managed-API fallback; cordon a suspect node. **Mitigate first, diagnose second** — the same discipline as any incident, and the only difference here is that "scale out" takes 90 seconds instead of 10.

**⚠ Trap:** restarting the pods as the first move. On a GPU fleet, a restart costs 60–120 seconds of cold start *per pod*, drops every prefix cache, and destroys the evidence in the engine's internal counters. It is the reflex from stateless backend services and it is actively harmful here. Capture the metrics snapshot first, and restart only when you have a reason to believe a process is in a bad state.
