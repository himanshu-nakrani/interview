### Why does one long prefill stall every in-flight decode? Walk me through the mechanism, not the metaphor.

Because the GPU runs one batch at a time, and a prefill step is not a small unit of work — it is a step that can occupy the device for hundreds of milliseconds while every decoding sequence in the engine waits its turn. This is head-of-line blocking, but at a level below the one you are used to: not a request blocking a request, but a *scheduler iteration* blocking every other sequence's next token. Your intuition for a thread pool where a long task blocks one worker is wrong here. There is one worker, and prefill grabs it.

Mechanically: a modern engine's scheduler loop picks a set of sequences, builds one fused forward pass, launches it, and only then can it emit tokens for anyone. If the scheduler admits a 32k-token prefill, that forward pass processes 32,768 positions through 80 layers. Every decode sequence that was producing a token every 20 ms now produces nothing until that pass returns. The stall shows up in exactly one metric — inter-token latency — and it shows up as a spike in the tail, not the mean, which is why it survives dashboards that only chart averages.

**💰 Math:** Llama-3.3-70B, TP8 on an H100 node. Prefill FLOPs ≈ 2·P·T for the dense parts = 2 × 70e9 × 32,768 = 4.59 PFLOP, plus causal attention ≈ 2·n_layers·T²·d_model = 2 × 80 × (32,768)² × 8192 ÷ 2 (causal halving) = 1.41 PFLOP. Total ≈ 6.0 PFLOP. Eight H100s at a realistic 400 TFLOPS achieved each = 3.2 PFLOPS → **1.9 seconds**. During those 1.9 s, a decode sequence whose SLO is 25 ms ITL missed **76 token deadlines** in a row. One request did that to all 64 of its batchmates.

**⚠ Trap:** candidates say "so we should run prefill and decode in separate streams." CUDA streams do not help. The bottleneck is not launch concurrency, it is that both phases contend for the same SMs and the same HBM bandwidth on the same device; overlapping them without a token budget just makes both slower and adds jitter. The two real answers are *slice the prefill* (chunked prefill) or *move it to a different machine* (disaggregation). Everything in this section is one of those two.

**🗣 Say this in the room:** "A long prefill is a monolithic scheduler iteration. Until it returns, no sequence in the engine gets a token, so a 1.9-second prefill injects a 1.9-second ITL spike into every concurrent stream. That's why TTFT and ITL are not independent SLOs — they're competing claims on the same forward pass."

### Define TTFT, ITL and TPOT precisely for me, and tell me why you can't optimise all of them at once.

**TTFT** (time to first token) is wall-clock from the request hitting your gateway to the first token byte reaching the client. It includes queue time, prefill compute, and every non-model hop in front of it. **ITL** (inter-token latency) is the gap between consecutive streamed tokens after the first — a distribution, one sample per token gap. **TPOT** (time per output token) is usually the *mean* ITL for a request, computed as (e2e latency − TTFT) ÷ (output tokens − 1). The distinction matters: TPOT is an average and hides stalls; ITL is the distribution and is where you see them.

End-to-end is then `E2E = TTFT + (n_out − 1) × TPOT`. Note that for a 600-token answer at 25 ms TPOT, the streaming phase is 15.0 s and TTFT is 0.4 s — decode dominates the wall clock by 37×, but TTFT dominates the *perception*, because the user is staring at nothing during TTFT and reading during decode.

They compete because they want opposite batching decisions. TTFT wants your prefill to run alone at maximum parallelism, admitted the instant it arrives, with no queueing. ITL wants the decode batch never to be interrupted and never to grow so large that the per-step time inflates. Every scheduler knob moves you along that curve: raise `max_num_batched_tokens` and prefills complete faster (TTFT down) but decode steps get longer and lumpier (ITL up); shrink the prefill token budget and decodes stay smooth (ITL down) while prefill takes more iterations (TTFT up). There is no setting that is best at both; there is only the setting that is best for *your* SLO pair.

**⚠ Trap:** reporting "p95 ITL" without saying how you aggregated. If you pool every token gap from every request into one histogram, a single 2,000-token generation contributes 2,000 samples and drowns out a hundred short requests that each stalled once. If you take per-request p95 and then p95 across requests, you get a different — and usually more honest — number. The rule I enforce is: **SLO on per-request p95 ITL, alert on the pooled distribution.** State which one you mean before anyone asks.

**📐 Numbers you must know:** the perceptual thresholds. TTFT under ~200 ms reads as instant; 200–500 ms reads as responsive; past ~1 s users start re-reading their own prompt. Human silent reading is roughly 250 words/minute ≈ 4.2 words/s ≈ **5–6 tokens/s**. So 30–50 tok/s (ITL 20–33 ms) is 6–10× reading speed — anything faster is invisible to the user and should be spent on batch size instead. That last clause is the senior move: **once you clear ~40 tok/s, further ITL improvement has zero product value and negative cost value.**

### If decode dominates wall-clock time by 37×, why does anyone optimise TTFT at all?

Because TTFT is the only latency the user experiences as *waiting*, and everything after it they experience as *reading*. This is a UX claim with an engineering consequence, and it is the reason the whole prefill/decode literature exists.

There is a second reason that is less obvious and more important for capacity: TTFT is where the queue lives. Decode time is a property of the model and the batch; it is roughly constant and roughly predictable. Queue time is unbounded and is what actually blows up under load. When your p95 TTFT goes from 400 ms to 4 s during a traffic spike, essentially none of that is prefill compute — it is 3.6 s of sitting in the waiting queue. So p95 TTFT is doing double duty as your saturation signal. I chart TTFT decomposed into `queue_time + prefill_time` on the same panel for exactly this reason; the ratio between them tells you whether to add replicas (queue-dominated) or change parallelism (prefill-dominated).

**🗣 Say this in the room:** "TTFT is the only part of the response the user spends waiting rather than reading, and it's also where queueing time hides. I always chart it decomposed into queue versus prefill — queue-dominated means scale out, prefill-dominated means change TP degree or enable prefix caching."

### Explain chunked prefill. What is it, mechanically, and what problem is it solving?

Chunked prefill says: stop treating a prefill as one indivisible forward pass. Slice it into fixed-size pieces of, say, 2,048 tokens, and process one piece per scheduler iteration, carrying the KV you produced forward into the next iteration. The sequence's KV cache is built incrementally, exactly as it would be by decode, just 2,048 positions at a time instead of 1. This is the contribution of Sarathi-Serve.

**📄 Paper:** Agrawal et al. (OSDI 2024), *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* — chunked prefills plus stall-free batching. It replaced the prevailing choice between prefill-prioritising schedulers (good TTFT, terrible ITL tails) and decode-prioritising schedulers (good ITL, terrible TTFT), by making the prefill unit small enough that neither has to wait long for the other.

The mechanism has two halves and the second is the one candidates forget. First half: cap the tokens processed per iteration at a **token budget** (in vLLM this is `max_num_batched_tokens`). A 32k prefill with a 2,048-token budget becomes 16 iterations rather than 1. Instead of one 1.9-second stall, decodes now see 16 stalls of ~120 ms each — still bad, until the second half. Second half: **piggyback the decodes into the same batch.** Each iteration's budget is filled first with the decode tokens of every running sequence (one token each), and the *remainder* of the budget is given to prefill chunks. With 64 running decodes and a 2,048-token budget, 64 tokens go to decode and 1,984 go to prefill. Now every iteration emits a token for every decoding sequence, and no sequence ever waits more than one iteration. That is "stall-free scheduling": the guarantee is not that iterations are fast, it is that *no iteration is decode-free*.

**⚠ Trap:** thinking the win comes from the chunking. It does not. Chunking alone converts one big stall into many medium stalls and slightly *reduces* total throughput. The win comes from the fusion — because the mixed batch reads the model weights from HBM exactly once and uses them for both the decode tokens and the prefill chunk, and the decode was going to pay that read anyway. The prefill chunk rides along on bandwidth that was already spent.

### Quantify the fusion win for me. Why is a piggybacked prefill chunk nearly free?

Because decode is memory-bandwidth-bound and prefill is compute-bound, and fusing them puts idle compute to work during a step whose cost is already dominated by weight streaming.

Work it. Llama-3.3-70B, FP8 weights (70 GB), TP8 on H100 SXM5 at 3.35 TB/s per GPU, assume 80% achieved bandwidth = 2.68 TB/s. Each GPU holds 70 ÷ 8 = 8.75 GB of weights. A pure decode step must stream all of them: 8.75 GB ÷ 2.68 TB/s = **3.3 ms**, and during those 3.3 ms the tensor cores are doing 2 × 70e9 × 64 = 8.96 TFLOP of work for a batch of 64, spread over 8 GPUs = 1.12 TFLOP each — against a per-GPU capability of ~400 TFLOPS × 3.3 ms = 1.32 PFLOP of *available* compute in that window. The GPU is running at roughly 0.1% of its arithmetic capability during decode. That is not a typo; that is what "memory-bandwidth-bound" means.

Now add a 1,984-token prefill chunk to the same step. Its dense FLOPs are 2 × 70e9 × 1,984 = 278 TFLOP, ÷ 8 GPUs = 34.7 TFLOP per GPU, which at 400 TFLOPS takes 87 ms. So the fused step costs ~87 ms instead of ~3.3 ms — the prefill is *not* free in wall-clock. What is free is the **weight read**: the chunk did not have to stream 8.75 GB of its own. Run them separately and you pay 3.3 ms of bandwidth twice; run them fused and you pay it once.

**💰 Math on the throughput cost:** with a token budget of 2,048 versus one monolithic 32k prefill, you re-stream the weights 16 times instead of once: 15 extra passes × 3.3 ms = 49.5 ms of pure overhead, against a total prefill compute of ~1.9 s. That is **2.6% prefill-throughput loss** for turning a 1.9 s ITL stall into a bounded ~90 ms one. At a token budget of 512 the arithmetic gets worse: 64 passes → 63 × 3.3 = 208 ms overhead on 1.9 s = **11%**, plus GEMM efficiency falls because the M dimension is small. That is the shape of the whole tuning problem in one calculation.

**🗣 Say this in the room:** "Chunked prefill trades a few percent of prefill throughput for a bounded ITL tail, and the trade is cheap because the fused batch amortises the weight read the decode step was already paying for. At a 2k budget I measure about 3% throughput cost; at 512 it's more like 10% and the GEMMs start losing tensor-core efficiency."

### How do you actually pick the chunk size? Tell me what breaks at 256 and what breaks at 8192.

I pick it from the ITL SLO, backwards, and then sanity-check the throughput cost. The governing identity is simple: **worst-case ITL ≈ the duration of one scheduler iteration**, and iteration duration is roughly (token budget × per-token prefill cost) + (decode weight-stream cost). So if your ITL SLO is 25 ms p95 and your decode-only step is 12 ms, you have ~13 ms of headroom for the prefill portion, which at a 70B TP8 (2 × 70e9 ÷ 8 GPUs ÷ 400 TFLOPS = 43.75 µs per token per GPU) buys you 13 ms ÷ 43.75 µs ≈ **297 prefill tokens per iteration.** That is a shockingly small budget, and it is the honest answer for a tight ITL SLO on a big model.

At **256**: the prefill GEMM has M=256, which is small enough that you leave tensor-core throughput on the table (you want M in the low thousands to saturate the MMA pipeline on Hopper), you re-stream weights on every one of the many iterations, and Python/host scheduling overhead per iteration — which is fixed, on the order of 1–3 ms in a well-optimised engine — becomes a double-digit percentage of the step. Your prefill throughput can drop 20–30% and TTFT for long inputs gets meaningfully worse.

At **8192**: prefill is efficient and TTFT is great, but one iteration is now ~8192 × 43.75 µs = 358 ms, so your worst-case ITL is 358 ms — fourteen times your SLO. You have effectively turned chunked prefill off.

**🔍 Failure taxonomy — how to tune it in production:**
1. Measure decode-only step time at your target batch size. That is your ITL floor.
2. Budget = (ITL SLO at p95 − ITL floor − 20% safety) ÷ per-prefill-token cost.
3. If that budget comes out under ~512, your ITL SLO and your model size are in conflict — go to disaggregation, a smaller model, or a looser SLO. Do not try to win it with chunk size.
4. If it comes out over ~4,096, chunking is not your binding constraint; set it at 4,096 and go optimise something else.
5. Re-measure after any change to TP degree, quantisation, or prefix-cache hit rate, because all three move the ITL floor.

**📅 Volatile:** vLLM's default `max_num_batched_tokens` and whether chunked prefill is on by default have both changed across versions (V1 turned it on by default with a larger budget than the old 512-token default). Verify against the version you are actually deploying before your loop.

### Does chunked prefill interact badly with anything else in the stack?

Yes, three things, and the third one has bitten me personally.

**Prefix caching.** They compose cleanly but you must get the ordering right: the engine should resolve cached blocks *first*, then chunk only the uncached remainder. If a 30k-token prompt has 28k cached, the actual prefill work is 2k and does not need chunking at all. An engine that chunks before consulting the cache does 15 iterations of lookup instead of 1 — same result, more host overhead. Modern engines do this correctly; older forks and homegrown schedulers do not.

**CUDA graphs.** Decode steps are graph-captured because they have a fixed shape and the per-launch overhead of 80 layers × ~10 kernels is otherwise material. A fused prefill+decode batch has a *variable* shape (the chunk size varies as it fills the budget), so it cannot use the pure-decode graph. Engines handle this by padding to bucketed shapes or by falling back to eager for mixed steps. The consequence is real: on a workload where 100% of steps are mixed, you can lose the CUDA-graph win entirely and see host-bound behaviour at high batch. If your GPU utilisation is oddly low with chunked prefill on, this is the first thing to check.

**Speculative decoding.** Speculation wants a decode-only step so it can verify K draft tokens per sequence in one pass. A mixed batch complicates the verification bookkeeping and some engines simply disable speculation on mixed steps. If you enabled both and your measured acceptance-driven speedup vanished, check whether your mixed-step fraction is high.

**⚠ Trap:** enabling chunked prefill and then benchmarking with a synthetic workload of 128-token prompts. Every prefill fits in one chunk, nothing is exercised, and you conclude the feature is free. Benchmark with your *actual* input-length distribution, including the p99 long tail — that tail is the entire reason the feature exists.

### Would you turn chunked prefill on by default? Give me the decision rule.

Yes, by default, with three named exceptions. The default is right because most production traffic has a long-tailed input-length distribution, and without chunking a single p99 input holds the whole engine hostage. It costs single-digit percent of prefill throughput to buy a bounded ITL tail; that is a good trade in almost every interactive product.

The exceptions:

**Pure offline batch.** If you are scoring 40 million documents overnight and nobody is watching a stream, ITL is meaningless and every percent of prefill throughput is money. Turn chunking off, set the token budget huge, prioritise prefill. This is the batch tier of your fleet and it should be configured as a different deployment, not the same one with different traffic.

**Uniformly short inputs.** A classification endpoint where p99 input is 400 tokens has nothing to chunk. Leaving it on is harmless but you should know it is a no-op, not a safety net.

**When you have already disaggregated.** If prefill runs on a separate pool, the decode pool never sees a prefill and chunking has nothing to protect. Chunking inside the *prefill* pool is then only about fairness between concurrent prefills, which is a much weaker motivation. This is the real relationship between the two techniques: chunked prefill is interference mitigation on shared hardware; disaggregation is interference *elimination* by not sharing.

**🗣 Say this in the room:** "On by default for anything interactive, off for the offline batch tier, and largely redundant once you disaggregate. Chunked prefill and P/D disaggregation are the same idea at two price points — one costs you a few percent of throughput, the other costs you a second GPU pool and a KV transfer."

### What is goodput under SLO, and why do you insist on it over throughput?

Goodput is throughput counted only over requests that met their SLO. A request that returned in 9 seconds against a 2-second budget did not contribute value; it consumed capacity and produced a user who left. Counting it in your tokens-per-second makes your system look better exactly as it gets worse, which is the single most misleading property of raw throughput as a serving metric.

The mechanism by which raw throughput lies is worth being precise about. Increase batch size and tokens/s rises monotonically for a long way — you are amortising the weight read over more sequences, which is free until you saturate bandwidth or memory. But per-step latency rises too, so ITL rises, so at some batch size every request in flight is violating the ITL SLO. Throughput is at its maximum and goodput is **zero**. The curve of throughput against batch is monotonic; the curve of goodput against batch is a hump. You want to sit just left of the peak of the hump, and you cannot find that point by maximising throughput.

**📄 Paper:** Zhong et al. (OSDI 2024), *DistServe* — made goodput-under-SLO the explicit optimisation objective for LLM serving and used it to argue for prefill/decode disaggregation. Its framing (optimise goodput per GPU, not throughput per GPU) is the one to quote; it replaced the implicit "maximise tokens/s" objective that every earlier serving benchmark used.

**💰 Math:** suppose at batch 64 you get 2,000 tok/s with 98% of requests meeting a 25 ms ITL SLO → goodput 1,960 tok/s. At batch 192 you get 3,200 tok/s but only 45% meet the SLO → goodput 1,440 tok/s. Raw throughput went up 60%; useful work went down 27%; and your cost per *satisfied* request went up 36% because you are paying for the same GPUs. That is the whole argument in four numbers.

**⚠ Trap:** defining goodput per-token instead of per-request. If a request violates its SLO at token 300 of 600, none of its 600 tokens counted — the user got a bad experience regardless of how many tokens arrived on time. Count whole requests. I have seen a team report 85% goodput on a token basis when the request-level number was 61%.

### How do you actually measure goodput in production? Give me the instrumentation.

You need per-request records with the three timestamps that let you reconstruct the whole latency budget, and you need them at the point closest to the user, not at the engine.

Emit, per request: `arrival_ts` (gateway ingress), `admitted_ts` (scheduler dequeue), `first_token_ts`, `last_token_ts`, `n_prompt_tokens`, `n_output_tokens`, `n_cached_prefix_tokens`, `model`, `tier`, `tenant`, and the vector of inter-token gaps or at least its p50/p95/max. From those, `queue_time = admitted − arrival`, `prefill_time = first_token − admitted`, `ttft = first_token − arrival`, and per-request ITL stats. Store the gap vector's max — that is your stall detector and it is the one field people omit.

The SLO evaluation is then a boolean per request: `met = (ttft ≤ ttft_slo) AND (p95_of_this_request's_ITL ≤ itl_slo)`. Goodput is `sum(output_tokens where met) / window`, and the headline number I put on the dashboard is the *fraction of requests meeting SLO*, sliced by tier and by input-length bucket. Slicing by input-length bucket is non-negotiable: aggregate SLO compliance can sit at 97% while the >16k-token bucket sits at 30%, and that bucket is your enterprise customer.

The trap in instrumentation is measuring TTFT at the wrong boundary. If you time from when the engine dequeues the request, you have measured prefill and deleted queueing — which is exactly the quantity that degrades under load. Measure from gateway ingress. If you have a streaming proxy in front (SSE termination, a token-level guardrail, a Cloudflare hop), measure from the outermost hop you control and account for the rest as a fixed budget line.

**🗣 Say this in the room:** "I define met-SLO as a boolean per request over TTFT and per-request p95 ITL, measured from gateway ingress so queueing is included, and I slice compliance by input-length bucket — aggregate compliance hides the long-prompt cohort completely."

### Draw me the latency waterfall for a real production request. Where does the time actually go?

Here is the one I use, for a retrieval-augmented answer in an enterprise search product — Glean-shaped, 4k-token assembled prompt, 600-token answer:

```
auth + JWT validation ...................    5 ms
gateway routing, rate limit, tenant lookup   10 ms
input guardrail (PII scan, injection check)  25 ms
query embedding ........................    18 ms
vector search (ANN, top-100) ...........   120 ms
reranker (cross-encoder, 100 → 12) .....    80 ms
permission filter + prompt assembly ....    12 ms
LLM queue wait .........................    40 ms
prefill / model TTFT ...................   400 ms
---------------------------------------------------
first token to user ....................   710 ms
600 output tokens @ 25 ms ITL ..........  15,000 ms
total ..................................  15.7 s
```

Two readings of this table, and the second one is what gets you credit. First reading: the model is 62% of time-to-first-token (440 of 710 ms including queue) and retrieval plus rerank is 28% (200 ms). Second reading: **the 200 ms of retrieval is the cheapest 200 ms to remove and the 400 ms of prefill is the most expensive.** Cutting rerank from 80 ms to 25 ms by moving to a smaller cross-encoder or reranking only the top 30 is an afternoon of work; cutting prefill from 400 ms to 200 ms means doubling your TP degree, which doubles your GPU bill for the prefill phase. Optimise in cost-per-millisecond order, not in size order — that is the reflex most candidates do not have.

Third thing to notice: the guardrail is 25 ms *in series*. Guardrails belong in parallel with the query embedding wherever the semantics allow, because they are independent. That is 25 ms free. And permission filtering after retrieval rather than as a pre-filter in the vector index costs you recall as well as latency; the right architecture pushes the ACL into the ANN filter.

**⚠ Trap:** presenting a waterfall with no queue-wait line. Every real system has one and it is the line that moves under load. A waterfall without queue time is a waterfall measured on an idle staging cluster.

### Our TTFT SLO is 400 ms but retrieval alone takes 200 ms. How do you think about splitting an end-to-end budget?

I treat it as a budget-allocation problem with a hard constraint and explicit slack, exactly like an error budget. Write down the target, subtract the irreducible hops, and see what is left for the model. If the number left is smaller than your measured p95 prefill, the design does not close and you must change the architecture, not tune it.

For a 400 ms first-token target with the waterfall above, the model gets 400 − 5 − 10 − 25 − 18 − 120 − 80 − 12 = 130 ms, and you need queue time inside that too. A 4k-token prefill on a 70B at TP8 is ~220 ms of pure compute. The design does not close. The honest options, in the order I would present them:

1. **Overlap, don't shrink.** Start the LLM request's static prefix (system prompt + tool schemas) prefilling while retrieval is still running, so the retrieved documents only add an incremental prefill. With prefix caching this is nearly automatic — the 2k-token static preamble is a cache hit and only the 2k of retrieved context is actually prefilled, halving prefill to ~110 ms. This alone can close the gap and it costs nothing.
2. **Stream the retrieval stage to the user.** Show the sources as they resolve. This is a UX answer to a latency problem and it is often the right one — perceived TTFT drops to 200 ms even though the token TTFT does not move.
3. **Shrink the model for this path.** A 8B model prefills 4k tokens roughly 9× faster than a 70B (2 × 8e9 vs 2 × 70e9 FLOPs per token). If the eval says an 8B is adequate for extractive answers over retrieved context — and for many RAG answers it is — this is the biggest single lever.
4. **Loosen the SLO for the long-prompt cohort.** Publish 400 ms p95 for prompts under 4k and 900 ms for prompts over 16k. Tiered SLOs by input size are honest and defensible; a single number that you miss 30% of the time is not.

**🗣 Say this in the room:** "I'd write the budget as a subtraction, and if the remainder for the model is smaller than measured prefill I'd say the design doesn't close and name the four levers: overlap the static prefix via prefix caching, stream retrieval into the UI, drop to a smaller model for this path, or tier the SLO by prompt length. I would not promise a single 400 ms number across all prompt sizes."
