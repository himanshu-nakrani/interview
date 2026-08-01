### 🏋 Drill: hit p95 TTFT 400 ms and p95 ITL 25 ms for a 70B at 200 concurrent users. Size the fleet and price it. Forty minutes, no laptop.

**Pass criterion:** you produce a node count, a parallelism choice, a configuration, and a monthly dollar figure, with every intermediate number derived rather than asserted, and you state your assumptions before you use them. Here is the reference solution.

**Assumptions stated up front:** Llama-3.3-70B, FP8 weights (70 GB) and FP8 KV (0.156 MB/token), H100 SXM5 80 GB at 3.35 TB/s, achieved ~400 TFLOPS and ~80% of nominal bandwidth. Request shape 4,000 in / 600 out. Prefix-cache hit rate 50% (a 2k static system prompt). Chunked prefill on, co-located pools.

**Step 1 — arrival rate from Little's Law.** W = 0.400 + 599 × 0.025 = **15.4 s**. L = 200 → λ = 200 ÷ 15.4 = **13.0 req/s**.

**Step 2 — can TTFT close?** Per-GPU prefill rate = 400e12 ÷ (2 × 70e9) = 2,857 tok/s. TP8 at 85% scaling efficiency = **19,400 tok/s per node**. A cold 4,096-token prefill takes 4,096 ÷ 19,400 = **211 ms**; with a 50% cache hit, the real work is 2,048 tokens = **106 ms**. That leaves 400 − 106 = **294 ms** for gateway, queue and chunking overhead. It closes, with room. If I had chosen TP4 the prefill would be ~212 ms and the margin would be uncomfortably thin — so **TP8 is chosen by the TTFT SLO, not by memory.**

**Step 3 — what batch does ITL allow?** Per-GPU weight streaming = 8.75 GB ÷ 2.68 TB/s = **3.3 ms**. Per-GPU KV read at batch B with 4,300-token average residency = B × 4,300 × 0.0195 MB. At B = 64: 5.37 GB ÷ 2.68 TB/s = **2.0 ms**. Bandwidth floor = **5.3 ms**. Measured ITL lands 2–3× the floor once you count 160 all-reduces per step, sampling, detokenisation and non-GEMM ops → **~13–16 ms at batch 64**. At B = 128 the KV term doubles to 4.0 ms, floor 7.3 ms, measured ~18–22 ms. So the SLO-feasible batch is roughly **96–128**; I design to **96**.

**Step 4 — the interference correction, which is the step people skip.** Prefill demand = 13.0 × 4,096 × (1 − 0.5) = **26,600 prefill tok/s**. Across N nodes with 19,400 tok/s each, the fraction of scheduler time spent on prefill is 26,600 ÷ (N × 19,400). At N = 4 that is **34%** — meaning decode gets only 66% of iterations and effective ITL inflates by 1/0.66 = 1.5×, pushing batch-64 ITL from ~14 ms to ~21 ms. Under a 25 ms p95 that is real but uncomfortable. At N = 5 the prefill fraction drops to 27% and ITL lands ~19 ms. **Take 5 nodes = 40 H100s.**

**Step 5 — verify capacity.** 5 nodes × 96 SLO-feasible batch = 480 slots for 200 average concurrent = **42% occupancy**, which is the right side of the queueing knee and leaves room for a 2× burst. KV: 200 × 4,300 × 0.156 MB = **134 GB** against 5 × (640 − 70 − 50) = 2,600 GB available. KV is nowhere near binding; step time is.

**💰 Step 6 — price it.** 40 GPUs × $3.00/GPU-hr on-demand = **$120/hr** = $2,880/day = **$87,600/month**. On a 1-year commit at ~$2.00/GPU-hr: $80/hr = **$58,400/month**. Cost per request at full load = $120 ÷ (13.0 × 3,600) = $120 ÷ 46,800 = **$0.00256/request**. At a plausible frontier-API price of $3/Mtok in and $15/Mtok out, the same request costs 4,096 × $3e-6 + 600 × $15e-6 = 0.0123 + 0.0090 = **$0.0213**. Self-hosting is **8.3× cheaper per request at full utilisation.**

**💰 Step 7 — the break-even, which is the answer that gets you hired.** Real fleets are not at full utilisation. At 30% average utilisation over a diurnal cycle, effective self-host cost is $0.00256 ÷ 0.30 = $0.0085/request — still 2.5× cheaper than the API. Break-even *utilisation* is 0.00256 ÷ 0.0213 = **12%**. Break-even *volume*, including say $25k/month of amortised platform engineering: (87,600 + 25,000) ÷ 0.0213 = **5.3 million requests/month.** Below that, buy the API. Above it, own the fleet.

**📅 Volatile:** GPU hourly rates and frontier API prices both move fast and token prices have been deflating sharply. The arithmetic is durable; re-source the two price inputs the week of your interview.

### Our p95 TTFT went from 350 ms to 4 seconds at 9 a.m. GPU utilisation is 60% and unchanged. Debug it in front of me.

Sixty percent utilisation with a 11× TTFT regression tells me immediately that the time is not being spent computing — it is being spent **waiting**. So my first move is not a hypothesis, it is a decomposition: split TTFT into `queue_time` and `prefill_time` on the same chart. That single split eliminates half the hypothesis space in one look.

**If it is queue-dominated** (which at 60% util it almost certainly is), then requests are arriving faster than the scheduler will admit them, and the question is why the scheduler is refusing to admit at 60% utilisation. Four causes, checked in this order:

*KV capacity, not compute, is the limiter.* Utilisation measures SM activity; admission is gated on free KV blocks. If average context length rose at 9 a.m. — different user cohort, a new customer with long documents, a prompt-template change your own team deployed at 8:50 — then each sequence occupies more KV, `max_num_seqs` is no longer reachable, and you admit fewer requests while the SMs idle. **Check: KV utilisation percentage and mean prompt length, both by hour.** This is my leading hypothesis and it is right more often than not.

*`max_num_seqs` is the cap and it is set too low.* A hard concurrency ceiling looks exactly like this: utilisation plateaus below 100%, queue grows. Check whether running-batch is pinned at exactly the configured limit — a flat line at a round number is a configuration limit, not a physics limit.

*Prefix cache hit rate collapsed.* If a deploy changed the system prompt, every request now prefills 4k instead of 2k, doubling prefill demand with no change in QPS. Utilisation may barely move because prefill was never the utilisation driver. **Check: cache hit rate, hour over hour, correlated against deploy timestamps.** This is the highest-embarrassment cause because it is self-inflicted and invisible without the metric.

*A dependency in front of the model got slow.* If TTFT is measured at the edge and your reranker degraded, you will see a TTFT regression with a completely healthy engine. Check the waterfall stages independently.

**If it is prefill-dominated**, the input length distribution shifted — a batch job started pasting large documents, or an agent tier began sending 60k-token contexts. Confirm with a p99 input-length chart, then decide between tiering the SLO, raising prefill parallelism, and rate-limiting the offending tenant by tokens.

**⚠ Trap:** reaching for "scale up" first. At 60% utilisation, adding nodes may not help at all — if you are KV-limited by long contexts, more nodes do help; if you are capped by `max_num_seqs`, they help but you paid money for a config change; if your cache hit rate collapsed, they help by exactly the amount you regressed and you still have the bug. **Diagnose the constraint before spending.**

**🗣 Say this in the room:** "First I split TTFT into queue time and prefill time. Eleven-x TTFT at flat 60% GPU means we're waiting, not computing, so I'd check KV utilisation, the `max_num_seqs` ceiling, prefix-cache hit rate against the deploy log, and mean prompt length by hour — in that order. Scaling up before knowing which one is guessing with money."

### p50 ITL is 15 ms and p99 ITL is 900 ms. Nothing on the dashboard correlates. Where do you look?

A p99 that is 60× the p50 is not load — load moves the whole distribution. A 60× outlier with a healthy median is a **discrete event**, and there are only about six discrete events that stall a decode step. I would go through them in this order because it is roughly the order of frequency:

**A long prefill was co-scheduled.** The classic. If chunked prefill is off, or its token budget is large, one 32k prompt injects a multi-hundred-millisecond gap into every running stream. **Check: correlate per-request max-ITL-gap timestamps against prefill admission timestamps.** If they line up, you found it, and the fix is chunked prefill with a budget derived from your ITL SLO.

**Preemption.** The engine ran out of KV blocks and evicted a running sequence, then recomputed it later. Recompute of a 4k prefix is ~200 ms of stall for that sequence. **Check: the engine's preemption counter.** It should be near zero in steady state; any nonzero rate is a capacity or admission-control bug, not a normal operating condition.

**CUDA graph fallback.** A batch shape outside the captured buckets forces eager execution; 80 layers × ~10 kernels × ~5 µs of launch overhead is ~4 ms, not 900, so this alone is not your p99 — but combined with a Python-side scheduling hiccup it contributes.

**Host-side stalls.** Detokenisation, response serialisation, SSE flushing, and — in Python engines — a garbage-collection pause or a blocked event loop. You know this one from asyncio: one synchronous call on the loop thread stalls everything. In an inference server the symptom is identical and the blast radius is every stream. **Check: is the API-server process's event loop lag instrumented?** If not, instrument it; this is the cause that "correlates with nothing on the dashboard" because the dashboard has no panel for it.

**A single slow GPU or a straggler rank.** Under tensor parallelism every step is a synchronisation point, so the slowest rank sets the step time. A card that is thermally throttling, has ECC retries, or sits behind a degraded NVLink makes every step slow for everyone on that node. **Check: per-GPU SM clock, temperature, and NCCL collective duration percentiles.** A p99 that is localised to one node is the tell.

**Client-side or network buffering.** If a proxy buffers SSE, tokens arrive at the client in bursts and *your* measurement at the edge shows gaps that the engine never produced. Verify by comparing engine-side and edge-side gap distributions.

**⚠ Trap:** concluding "it's just the tail, p99 is always noisy." A 60× ratio is not noise; it is a mechanism. And per-request p99 understates the user impact, because a single 900 ms freeze mid-sentence is more noticeable than a uniformly slower stream — the eye tracks the discontinuity, not the rate.

### We turned on prefill/decode disaggregation. TTFT improved 30% but goodput went *down*. Explain that.

Entirely plausible, and there are four mechanisms. This is exactly the outcome the papers warn about when the preconditions are not met, and being able to enumerate the causes is the difference between having read the abstract and having run the system.

**You partitioned a pool that was benefiting from multiplexing.** Before, every GPU could serve either phase; the workload's natural burstiness was absorbed by the whole fleet. Now a decode-pool GPU sits idle during a prefill burst while prefill queues, and vice versa. If your P:D ratio is even slightly wrong for the current traffic mix, you have stranded capacity on one side and a queue on the other. **Check: per-pool utilisation. A large divergence is the signature.** The fix is dynamic re-ratioing, not more nodes.

**You halved your effective KV pool.** Decode capacity is now bounded by the decode pool's memory alone. If you took an 8-node fleet and split it 4/4, decode-side KV went from 8 nodes' worth to 4, so your maximum concurrency roughly halved, so `L` halved, so by Little's Law at fixed W your sustainable λ halved. TTFT looks great because prefill is uncontended; goodput fell because you cannot hold as many sequences.

**The transfer is on the critical path and is not overlapped.** If the implementation does a bulk copy after the last layer rather than layer-wise streaming, you added 64 ms (single NIC) or 284 ms (Ethernet) to every request. TTFT can still improve if the interference you removed was larger than the transfer you added — and goodput falls because the transfer occupies the prefill worker's memory and the decode worker's reserved blocks for that whole window, reducing both pools' effective concurrency.

**Backpressure inversion.** Prefill workers complete faster than decode workers accept, so completed KV piles up in prefill HBM waiting for destinations. Prefill's own memory fills, admission stops, and you have built a queue inside GPU memory — the most expensive place in your stack to store a queue.

**🔍 The decision procedure I would run:** chart per-pool utilisation (divergence → ratio problem), decode-pool KV utilisation (high → you shrank the pool), transfer duration p95 as a fraction of TTFT (high → not overlapped), and prefill-worker "waiting for destination" time (nonzero → backpressure inversion). One of those four will be lit. If none are, the honest conclusion is that your workload did not have a prefill-interference problem in the first place and you should roll back to co-located with chunked prefill.

**🗣 Say this in the room:** "TTFT up and goodput down is the classic disaggregation regression: you removed interference but you also removed statistical multiplexing and halved the decode-side KV pool. I'd chart per-pool utilisation divergence, decode KV utilisation, transfer time as a fraction of TTFT, and prefill-side wait-for-destination — one of those four explains it, and if none do, the workload never needed disaggregation."

### Design the serving tier for a code editor: inline autocomplete plus an agent mode, on one fleet. What are your SLOs and how do you enforce them?

Two workloads that look similar and are not. I would start by writing the SLOs, because they are the design.

**Inline completion.** The user is typing. TTFT is everything and total latency must be under ~300 ms end-to-end or the suggestion arrives after the user has moved on. Output is short — 10 to 60 tokens. Context is large and *highly repetitive*: the same file, the same repository prefix, the same system preamble, changing only at the cursor. So: **TTFT SLO 200 ms p95, ITL irrelevant, cancellation rate is enormous** (every keystroke invalidates the in-flight request).

**Agent mode.** The user issued a task and is watching a plan execute. TTFT SLO 1–2 s is fine. But the request is a *loop* — 8 to 40 model calls, each with a growing context that accumulates tool outputs and file contents — so per-turn latency multiplies. Contexts run 20k–200k tokens. ITL matters because the user reads the streamed reasoning and edits.

The architecture that follows:

**Different models, obviously.** Completion runs a small specialised model (1B–8B class) at TP1 on cheap cards; agent runs the frontier model. These do not share a fleet at all, and saying so is the first correct answer. Do not let an interviewer's "one fleet" framing push you into a bad design — push back and say why.

**Completion path: prefix caching is the entire architecture.** The repository context and system preamble are stable across keystrokes, so with block-level prefix caching a completion request prefills only the tokens that changed. That takes prefill from 8,000 tokens to maybe 30. **This requires prefix-affinity routing** — the request must land on the GPU holding that session's blocks, via consistent hashing on session ID, which fights your load balancer. Accept the imbalance; a cache miss costs 100× more than a slightly hot node.

**Completion path: aggressive cancellation.** With a keystroke invalidating requests, your cancellation rate can exceed your completion rate. The engine must free KV blocks on cancel *immediately*, and the gateway must not have already committed the request downstream. Measure "wasted prefill tokens due to cancellation" as a first-class metric — I have seen it be 40% of total prefill work.

**Agent path: it is a KV-residency problem, not a throughput problem.** A 40-turn agent loop with a 100k-token context holds 100,000 × 0.156 MB = **15.6 GB of FP8 KV** for the duration of the task, possibly minutes. Ten concurrent agent sessions consume 156 GB — two H100s' worth of memory doing nothing but *remembering*. Your capacity model for the agent tier is driven by concurrent sessions × context size, not by QPS. And the moment you evict that KV, the next turn re-prefills 100k tokens at ~5 s and $0.03 of GPU time.

**Enforcement:** separate deployments, a KV reservation floor for the completion tier, token-based per-user rate limits on the agent tier (an agent loop can consume 2M tokens in five minutes and one runaway user should not take the fleet), and a hard turn cap with a cost ceiling per task.

**💰 Math on the agent tier:** 40 turns averaging 60k input tokens with 90% prefix-cache hits = 40 × 60,000 × 0.1 = 240,000 tokens of actual prefill, plus 40 × 400 = 16,000 output tokens. On a 5-node fleet at $120/hr with 19,400 prefill tok/s per node, that prefill is 12.4 s of one node = 12.4 × ($24/3600) = **$0.083 per agent task**. Without prefix caching it is 2.4M prefill tokens = 124 s = **$0.83 per task, 10× more.** Prefix caching is not an optimisation for agent workloads; it is the business model.

### Perplexity-style: you have one second from the user pressing enter to the first token, including retrieval. Design it.

One second is generous compared to the 400 ms case, but it has to cover a search stack, so it is a scheduling problem more than a serving problem. My budget:

```
edge + auth + rate limit ............   15 ms
query understanding (rewrite/classify)  90 ms   [small model, TP1]
   ├─ web/index search (parallel) ...  180 ms
   ├─ embedding + ANN (parallel) ....   60 ms
   └─ guardrail scan (parallel) .....   25 ms
merge + dedupe + rerank (top 8) .....  110 ms
prompt assembly .....................    8 ms
LLM queue ...........................   30 ms
prefill (6k tokens, 60% cached) .....  124 ms
---------------------------------------------
first token .........................  557 ms
```

Four design decisions are doing the work here.

**Everything that can be parallel is parallel.** Search, embedding and guardrail run concurrently; the stage cost is the max (180 ms), not the sum (265 ms). This is the single biggest structural saving and it is free.

**The query-rewrite model is small and separate.** Putting a 90 ms hop in front of retrieval is only affordable because it runs on a 1B-class model on its own replica with a tiny prompt. If you route it through the main 70B fleet it costs 300 ms and queues behind agent traffic.

**The static preamble is cache-warm.** Of the 6k prompt tokens, the ~3.5k of system prompt, tool schemas and citation-format instructions are a prefix-cache hit; only the ~2.5k of retrieved snippets are new. Prefill drops from 6,144 ÷ 19,400 = 317 ms to **124 ms**. That 193 ms is the difference between the design closing and not closing.

**The UI does not wait for the LLM.** Sources render at ~450 ms, the moment rerank completes. Perceived TTFT is 450 ms even though token TTFT is 557 ms, and if the LLM path degrades to 1.4 s the user still saw progress at 450 ms. **Streaming the intermediate artifacts is the cheapest latency work in the entire stack** and it is a product decision, not an infra one.

**⚠ Trap:** putting the reranker on the critical path at full width. Reranking 100 candidates with a cross-encoder is ~100 forward passes of a small model; reranking 30 is 30. The recall difference between top-100 and top-30 candidates is usually small if your first-stage retrieval is decent, and the latency difference is 70–100 ms. **Measure recall@k for your first stage before you pay for a wide rerank.**

**🔍 What breaks at p99:** one of the parallel retrieval legs times out. Design for it explicitly — a per-leg deadline (180 ms) with graceful degradation to whatever legs returned, rather than a global failure. Answering from 4 of 8 sources beats a 3-second timeout, and you should log the degradation so it shows up in quality analysis rather than silently poisoning your eval.

### A legal-research product needs to answer questions over a 200,000-token document. TTFT is 17 seconds. What do you do?

First, confirm the 17 seconds is real physics and not a bug, because the number is derivable and you should derive it live. Dense prefill FLOPs = 2 × 70e9 × 200,000 = 28 PFLOP. Causal attention = 2 × 80 layers × (200,000)² × 8,192 ÷ 2 = 26.2 PFLOP — note that at 200k, attention has grown to *equal* the dense term, which it does not at 4k. Total ≈ 54 PFLOP ÷ 3.2 PFLOPS (TP8, achieved) = **17 seconds**. That is not a misconfiguration; that is what the model costs. Any answer that starts with "tune the scheduler" is wrong.

Four real options, in the order I would evaluate them:

**Amortise it across the session — cache the document's KV.** The lawyer will ask 8–15 questions of the same document. Prefill it once, keep the 200,000 × 0.156 MB = **31 GB of FP8 KV** resident, and every subsequent question is a ~500-token prefill on top of a cache hit: **TTFT drops from 17 s to ~200 ms for questions 2 through 15.** This is cache-augmented generation, and for a bounded corpus that fits in memory it beats retrieval on both quality and latency.

**💰 Is holding 31 GB worth it?** On an 8×80 GB node at $24/hr, 31 GB is 31 ÷ 520 usable = 6% of the node. Holding it for a 30-minute session costs 0.06 × $24 × 0.5 = **$0.72**. Re-prefilling for 12 questions costs 12 × 17 s × ($24 ÷ 3,600) = 12 × $0.113 = **$1.36**. Caching wins by ~2×, and it wins by 85× on latency. Tier it to host DRAM or NVMe between questions and the memory cost drops further — a 31 GB transfer over PCIe 5 at ~50 GB/s achieved is 0.6 s, still 28× better than re-prefilling.

**Retrieve instead of stuffing.** Chunk the document, embed it once, retrieve the 8 relevant passages, prefill 6k instead of 200k. TTFT ~300 ms, cost 30× lower. The objection — and it is a real one in legal — is that retrieval can miss a clause that a full read would catch, and "the model didn't see the indemnity section" is a career-ending failure mode in this domain. **The honest engineering answer is to run both and measure**: build an eval of 200 real questions with known ground-truth spans and compare full-context against retrieval on recall of the governing clause. Do not choose on principle.

**Prefill once, offline, at upload time.** The document is uploaded minutes before anyone queries it. Prefill it in the background on the batch tier at spot pricing, persist the KV to an NVMe/object tier, and load it on first query. The user's first question then costs a 31 GB load rather than 54 PFLOP of compute. This turns an interactive latency problem into a pipeline problem, which is the trade you want.

**Tier the SLO and tell the truth in the UI.** "Reading a 340-page document — about 20 seconds" with a progress indicator is an acceptable product for question one, if questions two through fifteen are instant. Users forgive a stated wait; they do not forgive an unexplained one.

**🗣 Say this in the room:** "Seventeen seconds is physics — 54 petaFLOPs at 3.2 achieved petaFLOPS. So I don't tune it, I amortise it: prefill the document once at upload on the batch tier, persist the 31 GB of FP8 KV, and every question after the first is a 200 ms TTFT. The alternative is retrieval, which is 30× cheaper but risks missing a governing clause, and in legal I'd only ship that after an eval on ground-truth spans."

### Build me the dashboard. What panels, and what alerts on each?

Six panels, and I would defend every one of them as load-bearing rather than decorative.

**1. SLO compliance, by tier, on 1-minute windows.** The fraction of requests meeting both TTFT and per-request-p95-ITL. This is the headline. **Alert:** three consecutive minutes below target (page); a 15-minute burn-rate alert against the monthly error budget (ticket). Slice by input-length bucket in a drill-down — aggregate compliance hides the long-prompt cohort completely.

**2. TTFT decomposed into queue time and prefill time, stacked.** This panel routes every latency incident: queue-dominated means capacity, prefill-dominated means input shape or cache. **Alert:** p95 queue time above 25% of the TTFT budget.

**3. Per-request max inter-token gap, p95 and p99.** Your stall detector. Mean ITL will not show you a 900 ms freeze; this will. **Alert:** p99 max-gap above 4× the ITL SLO.

**4. KV utilisation and preemption rate.** KV utilisation is the capacity signal and the autoscale input for scale-down. Preemption rate should be **zero** in steady state. **Alert:** KV above 85% for 5 minutes; *any* sustained preemption rate above ~0.1% of requests.

**5. Prefix-cache hit rate, with deploy markers overlaid.** This is simultaneously a latency metric and a cost metric, and a drop is the fastest cost regression detector you have. **Alert:** hit rate drops more than 15 percentage points in an hour. Overlaying deploys is not a nicety — the cause is a prompt change nine times out of ten.

**6. Cost per completed request and per 1M output tokens, by tier and tenant.** Derived from GPU-hours ÷ requests. **Alert:** 30% week-over-week increase. This is the panel that catches an agent loop that started making 40 calls instead of 8.

Two things I deliberately keep *off* the primary dashboard: raw GPU utilisation (structurally misleading — decode workers sit at 30–40% while fully saturated) and aggregate tokens/second (throughput, not goodput; it rises as you get worse). Both belong in a diagnostic view, clearly labelled, not on the wall.

**⚠ Trap:** alerting on latency averages. An average TTFT alert will fire after the incident is over and will not fire during a partial outage affecting 20% of traffic. Alert on percentile compliance over short windows, and always with a burn-rate companion so you distinguish "a bad minute" from "we will exhaust the month's budget by Thursday."

### 🏋 Drill: I give you a model, an SLO pair and a QPS. Ten minutes, whiteboard, no calculator — produce a fleet size.

**Pass criterion:** a defensible node count in under ten minutes with every step derived, plus one stated risk. The point of this drill is not the arithmetic — it is having the *sequence* memorised so you never freeze. Rehearse it until it is muscle memory. The sequence is:

1. **Bytes and FLOPs per token.** KV bytes/token = `2 · n_layers · n_kv_heads · d_head · dtype_bytes`. FLOPs/token = `2 · n_params` for the dense part. Write both down before anything else.
2. **W from the SLO.** `W = TTFT_slo + (n_out − 1) · ITL_slo`.
3. **L from Little's Law.** `L = λ · W`. This is your concurrency requirement.
4. **ITL floor per GPU.** `weights_per_gpu ÷ achieved_bandwidth`, plus the KV term at your target batch. Multiply by 2–3× for the real world. Compare to ITL SLO → **SLO-feasible batch**.
5. **Prefill rate per node.** `achieved_TFLOPS ÷ FLOPs_per_token × n_gpus × scaling_efficiency`. Divide input length by it → prefill latency. Compare to TTFT budget → **choose TP degree.**
6. **Node count, twice.** From concurrency: `L ÷ (SLO-feasible batch × 0.7)`. From prefill demand: `λ · T_in · (1 − cache_hit) ÷ prefill_rate_per_node`. **Take the max**, then add burst headroom.
7. **KV sanity check.** `L × mean_residency × bytes_per_token` against `n_nodes × (HBM − weights − overhead)`. If KV binds before step time, you have a different design.
8. **Price it.** Nodes × GPUs × $/GPU-hr × 730.
9. **State one risk.** Mine is almost always the same: "this assumes the input-length p99 is close to the mean; if the tail is heavy, prefill demand and TTFT compliance both degrade faster than this model predicts."

**⚠ Trap under time pressure:** forgetting step 6's *max*. Candidates compute concurrency, get a node count, and stop — then the interviewer points out that prefill demand alone needs twice that many nodes. Both constraints are real and either can bind. Say "I need to check this from both sides" out loud before you compute either.

### A PM asks for 200 ms TTFT and 15 ms ITL and won't take no. How do you have that conversation?

I do not argue about feasibility; I show them the frontier and let them choose a point on it. Refusing sounds like resistance; drawing the curve sounds like engineering, and it usually converts the conversation from a negotiation into a design decision within five minutes.

Concretely, I bring three artifacts. **First, the measured curve** — a plot of p95 TTFT and p95 ITL against batch size for our actual model on our actual hardware, with the two SLO lines drawn on it. If there is no batch size where both lines are satisfied, that is not an opinion, it is a measurement, and it ends the debate about whether the target is achievable *on this configuration*.

**Second, the menu of ways to move the frontier, priced.** Each one is a real option with a real number: raise TP from 4 to 8 (TTFT −45%, cost +0%, ITL slightly worse, available today); move to a smaller model (TTFT −80%, cost −70%, quality delta X on our eval — here is the eval); enable prefix caching properly (TTFT −45% on cached paths, cost −40%, needs a prompt refactor, two weeks); buy newer higher-bandwidth cards (ITL −30%, cost +25%, six-week lead time). The PM is now choosing between quality, money and time, which is a conversation they are qualified to have — rather than between "yes" and "no," which they are not.

**Third, the perception data.** 15 ms ITL is 67 tok/s, which is 12× reading speed. I would show that no user can perceive the difference between 15 ms and 25 ms, and that the same GPU budget spent on TTFT — where the user *is* waiting — produces a measurable satisfaction difference. Reframing an ITL request as a TTFT investment is usually the win, and it is usually free.

**🗣 Say this in the room:** "I wouldn't say no. I'd bring the measured TTFT-versus-ITL curve for our model on our hardware with both SLO lines drawn on it, a priced menu of four ways to move that curve, and the fact that 15 ms ITL is twelve times reading speed. Then it's their call which trade to buy — and usually they'd rather spend it on TTFT, which is the part users actually experience as waiting."

### Last one. Ninety seconds at the whiteboard: "make this hit its latency SLO." What comes out of your mouth?

"First I'd decompose the SLO into TTFT and ITL and get the percentile and the cohort — p95 over all requests, or p95 over the long-prompt cohort, because those are different systems.

Then I'd decompose measured TTFT into queue time and prefill time at the edge, not at the engine. Queue-dominated means I'm out of capacity and the fix is admission control plus more replicas. Prefill-dominated means the fix is prefix caching, a higher tensor-parallel degree, or fewer input tokens.

For ITL, I'd check whether the tail is caused by co-scheduled prefills. If it is, chunked prefill with a token budget derived from the ITL headroom — take my decode-only step time, subtract it from the ITL SLO, divide by per-prefill-token cost. If that budget comes out under about 512 tokens, chunking can't save me and I'd look at prefill/decode disaggregation, but only if I've got long prompts, a hot fleet and RDMA between the pools. Otherwise the honest answer is a smaller model or a looser SLO.

Then I'd size from Little's Law — concurrency equals arrival rate times end-to-end time, where end-to-end is dominated by output length, not TTFT — and target about 70% of the SLO-feasible batch, not 70% of peak throughput, because queueing delay is rho over one-minus-rho.

And I'd optimise for goodput under SLO rather than throughput, because throughput rises monotonically with batch size while goodput has a peak, and I want to sit just left of that peak.

Finally, I'd tell you how I'd know it worked: percentage of requests meeting both SLOs, on one-minute windows, sliced by input-length bucket and tier, with per-request max inter-token gap charted separately because mean ITL hides freezes. If that number isn't on a dashboard before we ship, we don't have an SLO — we have an aspiration."
