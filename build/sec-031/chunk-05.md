### When would you use a provider's batch API, and what do you have to build around it?

The mental model: a batch API is the provider selling you their trough. Interactive traffic is diurnal, so their fleet is half-idle at 4am; by accepting a 24-hour completion window you let them schedule your work into that idle capacity, and they pass roughly half the saving back. It is the same trade as spot instances, expressed in tokens.

**📐 Numbers you must know:** the major providers' batch tiers price at approximately **50% of synchronous rates with a 24-hour completion SLA** (Anthropic's Message Batches, OpenAI's Batch API, and equivalents; **📅 Volatile** — verify the current discount, window, per-batch size limits and whether prompt caching composes with batching before you quote it). Batch requests typically do not consume your synchronous rate limits, which is often as valuable as the discount.

**💰 Math:** classify 2 million support tickets. Average 800 input tokens, 120 output tokens. At $3/Mtok in and $15/Mtok out: input 2e6 × 800 = 1.6e9 tokens = $4,800; output 2e6 × 120 = 2.4e8 = $3,600. Total **$8,400 synchronous, $4,200 batch** — a $4,200 saving on one job. And you avoid building the rate-limit backpressure machinery that a synchronous run of 2M requests would demand.

**What you build around it**, because the API is deliberately minimal:

**Idempotent custom IDs.** Every request in the batch carries a `custom_id` you choose, and results come back keyed by it, out of order. Make that ID a deterministic function of the input — I use `sha256(document_id + ':' + prompt_version + ':' + model_id)`. That makes re-submission safe, makes results joinable without a fragile order assumption, and makes it trivial to compute the set of work still outstanding.

**A durable job ledger.** Rows of `(custom_id, batch_id, status, submitted_at, result_uri)` in Postgres. You need this because the batch may partially fail, because your process will restart during the 24 hours, and because you need to answer "are we done" without re-reading a multi-gigabyte JSONL.

**Chunking and submission control.** Batches have size and byte limits, so a 2M-request job is many batches. Submit them with a bound on concurrent open batches, and record the mapping.

**Result reconciliation with per-item error handling.** Individual items in a batch can fail — content filter, context overflow, transient error — while the batch succeeds. Reconcile the returned `custom_id` set against the submitted set, route the difference to a retry batch, and after N attempts to a dead-letter table. This is exactly your Celery DLQ discipline; the only new thing is that the failure unit is inside a bulk response.

**When *not* to use it:** anything with a human waiting; anything where a 24-hour turnaround makes the result stale; and anything where you need to iterate on the prompt, because a 20-hour feedback loop makes prompt development impossible. My workflow is: iterate on 200 examples synchronously, freeze the prompt, then run the 2M in batch.

**⚠ Trap:** assuming batch means "guaranteed within 24 hours, so I can schedule downstream work at hour 24." The window is an upper bound with no lower bound and no progress guarantee; jobs often finish in minutes but you cannot depend on it. Your pipeline must be event-driven on completion, not scheduled on an assumed duration.

### Design a pipeline to extract structured fields from 20 million PDFs on self-hosted GPUs. Walk me through it.

I will structure this as: unit of work, ordering, execution, durability, cost.

**Unit of work and idempotency first**, because everything else depends on it. One document, one deterministic output key: `s3://out/{prompt_version}/{model_digest}/{sha256(doc_bytes)}.json`. Three properties fall out of that key: re-running the job skips completed work with an existence check (or better, a manifest join); changing the prompt or the model produces a new key space, so you never mix outputs from two prompt versions in one dataset; and a worker that dies after writing but before recording success causes a harmless overwrite of identical content. **This single design choice is what makes the pipeline resumable, and it is the first thing I would want a candidate to say.**

**Ingestion and pre-processing are a separate, CPU-bound stage.** PDF parsing, OCR where needed, chunking, and token counting. This is 20M documents of CPU work and it should not sit on GPU nodes. Output of this stage is a manifest: `(doc_id, text_uri, token_count)`. Do the token counting here — the next stage needs it.

**Length bucketing is the throughput lever.** Sort the manifest by token count and form batches of similar-length documents. Why it matters: in a batch of mixed lengths the prefill work is set by the sum, but padding and scheduler granularity mean a batch containing one 30k document and 63 × 500-token documents runs at the pace of the outlier. Bucketing — say, buckets at 1k/2k/4k/8k/16k/32k+ tokens — lets you set a per-bucket `max_num_seqs` that keeps every batch near the memory ceiling. In my experience this is worth 20–40% on a mixed corpus, and it is nearly free to implement. Route the 32k+ bucket to its own replicas with a different config.

**Execution: Ray Data is the right tool** for this shape. It gives you streaming execution over a dataset (so 20M documents never materialize), actor-based workers holding a vLLM engine each (so the model loads once per actor, not per batch), backpressure between stages, and automatic recovery of failed tasks. Spark is a reasonable alternative if your organization already lives there and your pre-processing is Spark-shaped — the pattern is `mapPartitions` with a model loaded once per partition, and the risk is that Spark's task model fights you on long-lived GPU state. The anti-pattern is a Celery task per document: you pay model-load or at least HTTP overhead per item, you lose batching entirely, and 20M queue messages is its own problem.

**Concretely, the Ray shape:**

```python
import ray, hashlib, json
ds = ray.data.read_parquet("s3://manifest/")          # doc_id, text, token_count
ds = ds.sort("token_count")                            # length bucketing

class Extractor:
    def __init__(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=MODEL, tensor_parallel_size=1,
                       max_model_len=32768, enable_prefix_caching=True)
        self.sp  = SamplingParams(max_tokens=512, temperature=0)

    def __call__(self, batch):
        prompts = [render(t) for t in batch["text"]]
        outs = self.llm.generate(prompts, self.sp)       # engine batches internally
        return {"doc_id": batch["doc_id"],
                "out": [o.outputs[0].text for o in outs]}

ds.map_batches(Extractor, batch_size=64, concurrency=8,
               num_gpus=1).write_parquet("s3://out/v3/")
```

**Durability and resumability.** Before the run, left-anti-join the manifest against the existing output keys so a restart processes only what is missing. During the run, write outputs in reasonably-sized parquet files with atomic put semantics. Keep a failures table with the error class; a document that overflows context or fails schema validation goes there rather than killing the stage.

**Validation as a first-class stage.** Parse every output with a Pydantic model. Schema failures go to the retry path — first a retry with the same prompt (nondeterminism sometimes fixes it even at temperature 0, because batching changes numerics), then a retry with a repair prompt, then dead-letter. Track the schema-failure rate as a health metric; a sudden rise means the input distribution shifted.

**Spot instances.** This job is the ideal spot workload — see the drain discussion earlier. Keep the unit of work small (a batch of 64 documents, tens of seconds) so a 30-second preemption notice is enough to finish or abandon cleanly.

**⚠ Trap:** running the 20M job against the same fleet that serves interactive traffic, with a plan to "give it low priority." Unless your engine supports genuine request priority *and* you have separate KV-cache accounting, a batch job with 512 in-flight requests will fill the paged cache and force preemption of interactive sequences. Separate fleets, or separate nodes with priority classes so the batch pods are evicted rather than merely deprioritized.

### Give me the cost per document for that job, both self-hosted and via an API.

Assume: 20M documents, average 3,000 input tokens after chunking, 400 output tokens. Total 6e10 input tokens and 8e9 output tokens.

**API path**, taking a mid-tier model at $3/Mtok input and $15/Mtok output with the 50% batch discount:

- Input: 6e10 / 1e6 × $3 × 0.5 = **$90,000**
- Output: 8e9 / 1e6 × $15 × 0.5 = **$60,000**
- Total **$150,000**, or **$0.0075 per document**.

With a cheap small model at, say, $0.25/$1.25 per Mtok and the same batch discount: 6e10/1e6 × 0.25 × 0.5 = $7,500 plus 8e9/1e6 × 1.25 × 0.5 = $5,000 = **$12,500**, or $0.000625/doc. **📅 Volatile:** all four prices move; the structure of the calculation is what you memorize, not the digits.

**Self-hosted path.** Serve an open 8B-class model on a single H100 per replica. Measured (you must measure — do not assume) throughput for this mixed prefill-heavy workload: say 12,000 total tokens/s per GPU when running batch-style with no latency SLO, of which the prefill share dominates. Total tokens = 6.8e10. GPU-seconds = 6.8e10 / 12,000 = 5.67e6 s = **1,574 GPU-hours**. At $2.50/hr on-demand that is **$3,935**; on spot at $0.75/hr it is **$1,180**.

Add the parts people forget: pre-processing (PDF parse/OCR) for 20M documents on CPU — say 0.5 CPU-seconds each = 2,778 CPU-hours at $0.04/hr = $111, though OCR-heavy documents can be 20× that; object storage reads and writes; and the engineering time to build and babysit the pipeline, which for a first build is realistically two engineer-weeks.

**Self-hosted total ≈ $1,300–4,100 of compute**, or **$0.000065–0.0002 per document** — 30–100× cheaper than the mid-tier API path.

**The honest conclusion**, and this is the judgment the interviewer is after: for *large offline batch jobs on a model you can self-host, with no latency SLO*, self-hosting wins decisively, because batch throughput is exactly the regime where a GPU is most efficient and where the API's margin is least justified. That is the opposite of the conclusion for interactive serving, which is the next question. **The break-even is not a property of self-hosting versus APIs; it is a property of your duty cycle.** Batch jobs run at ~100% duty cycle by construction. Interactive fleets run at 20–40%.

**The remaining question is quality**, and it is the one that decides the job: an 8B open model may or may not extract these fields as accurately as a frontier model. Build the eval on 500 hand-labeled documents *first*, measure field-level accuracy for both, and price the difference. If the frontier model is 4 points more accurate on a field that feeds a financial system, $150k may be cheap. If it is 0.5 points better on a field a human reviews anyway, it is not.

### Derive the self-host versus API break-even for interactive serving. Show me the formula and then use it.

The formula is short and you should be able to write it on a whiteboard:

```
cost_per_Mtok_selfhosted = (GPUs × $/GPU-hr) / (achievable_tok_s × 3600 / 1e6) / duty_cycle
```

Read it as: dollars per hour, divided by millions of tokens per hour, divided by the fraction of hours you actually have load. The three inputs are hardware price (known), achievable throughput at your SLO (**must be measured, and it is where every wrong answer comes from**), and duty cycle (from your traffic curve).

**Worked, for a 70B-class model:**

Fleet: 8×H100 at $2.50/GPU-hr = **$20/hr**. Measured throughput at p95 ≤ 2 s on your production length distribution: **2,000 output tok/s** plus incidental prefill (I will price on output tokens, which is the conservative and standard framing).

At 100% duty cycle: $20 / (2,000 × 3,600 / 1e6) = $20 / 7.2 = **$2.78 per Mtok**.

At a realistic interactive duty cycle of 30% (business-hours product): **$9.26 per Mtok**.

Compare to a hosted open-70B endpoint at roughly $0.90/Mtok blended (**📅 Volatile**). Self-hosting loses by 3× at 30% duty cycle and by 3× even at 100%... which tells you something important: **at these price points, self-hosting a mid-size open model to beat a commodity open-model API is usually a losing trade on cost alone.**

**Solve for the break-even throughput.** Set $20/hr ÷ (X × 3600/1e6) = $0.90 → X = 20e6 / (0.90 × 3600) = **6,173 output tok/s sustained, 24/7**. If your 8-GPU box cannot sustain 6,200 output tok/s inside your SLO at full duty cycle, the API is cheaper. Most cannot, at 70B and a 2-second p95.

**So when does self-hosting actually win?** Not on the commodity path. It wins when:

- **The comparison is against a frontier model, not an open-model host.** At $3/$15 per Mtok, break-even throughput drops by roughly 5–15×, and a self-hosted open model clears it easily. The real question then becomes quality, not cost.
- **Duty cycle is high** — batch, or a global product with round-the-clock load.
- **Non-cost constraints dominate:** data residency, on-prem requirements, a custom fine-tune nobody hosts, a modified engine, or a latency floor you cannot achieve over the public internet.
- **You need cost *predictability*** for a fixed-price enterprise contract; a GPU reservation is a known monthly number, per-token spend is not.

**🗣 Say this in the room:** "The formula is fleet dollars-per-hour divided by achievable tokens-per-hour at SLO, divided by duty cycle. For a 70B at 2,000 output tok/s on 8 H100s that's $2.78 per million at full utilization and about $9 at a realistic 30% interactive duty cycle — so self-hosting to beat a $0.90 open-model API is usually a losing trade. I self-host when I'm displacing a frontier-priced model, when duty cycle is near 100% like batch, or when the constraint isn't cost at all."

**⚠ Trap:** the break-even calculation that omits duty cycle and the one that omits headroom. You do not run at your knee — you run at 70–75% of it — and you do not run 24 hours of peak. Multiply the naive number by roughly 1.4 for headroom and divide by duty cycle, and self-hosting's advantage shrinks by 3–5×. I have seen a self-hosting proposal approved on a 100%-utilization spreadsheet and come in at 4× the projected cost.

### Where does speculative decoding fit into a capacity plan?

The mental model: decode is memory-bandwidth-bound, which means that generating one token and generating four tokens cost almost the same, because both require reading the entire model's weights from HBM once. Speculative decoding exploits that slack — a cheap draft model proposes *k* tokens, the target model verifies all *k* in a single forward pass, and every accepted token is essentially free. You are converting unused arithmetic intensity into latency.

Mechanically: draft *k* tokens autoregressively with a small model (or a draft head, or n-gram lookup from the prompt), run the target model once over the k+1 positions, and accept the longest prefix consistent with the target's distribution using a rejection-sampling rule that preserves the target's exact output distribution. That last property is why it is not a quality trade: correctly-implemented speculative decoding is **distributionally lossless**.

**📄 Paper:** Leviathan et al. (2023) and Chen et al. (2023) independently introduced speculative decoding with the acceptance rule that preserves the target distribution. Medusa (Cai et al., 2024) replaced the separate draft model with multiple decoding heads on the target itself, removing the second-model deployment burden.

**The capacity-planning consequences, which are not what people expect:**

**It helps latency, not throughput — and can hurt throughput.** At low batch size the GPU is idle-ish and verification is nearly free, so you get real speedups (commonly 1.5–2.5× on inter-token latency, workload-dependent). At high batch size the GPU is already compute-saturated, and the extra verification FLOPs for rejected tokens are pure waste. So speculative decoding is a **low-concurrency, latency-critical** technique. Enabling it fleet-wide on a high-throughput deployment can reduce your tokens/second.

**Acceptance rate is everything and it is workload-specific.** If the draft agrees with the target 70% of the time you win big; at 30% you are burning FLOPs to produce rejects. Acceptance is high on predictable text — code with strong local structure, JSON matching a schema, extractive summarization quoting the source — and low on creative or high-entropy generation. **N-gram / prompt-lookup drafting** is the underrated variant here: for tasks where the output copies heavily from the input (editing a document, answering from retrieved context, refactoring code), drafting from an n-gram index over the prompt costs zero GPU and achieves high acceptance. For a Cursor- or Notion-shaped product that is often the best speculation strategy available.

**💰 Math:** suppose your ITL is 40 ms and speculation with an acceptance-weighted average of 2.2 accepted tokens per verification step takes it to 20 ms. On a 600-token answer that is 24 s → 12 s of streaming time. If that moves you from below to above the reading-speed threshold, it is a product win. But if throughput drops 15% at your operating batch size, you need 15% more replicas: on a 9-replica fleet at $180/hr that is $27/hr = **$19,700/month** for the latency improvement. Whether that trades well is a product decision, and I would want the A/B on engagement before committing.

**⚠ Trap:** measuring speculative decoding's speedup at batch size 1 and projecting it onto a production fleet. Every published speedup figure is essentially a low-batch measurement. Benchmark at your operating concurrency, and be prepared for the answer to be "it makes things worse here."

### How does quantization change the parallelism and cost picture? Does int4 let me drop from TP=8 to TP=4?

Often yes, and that is one of the largest cost levers available — but the arithmetic has to include KV cache and the throughput consequences, not just weight bytes.

**Weights.** A 70B: bf16 = 140 GB, fp8 = 70 GB, int4 (with group-128 scales, ~4.25 effective bits) = 70e9 × 4.25 / 8 = **37.2 GB**. So bf16 needs ≥2 H100s and realistically 8 for cache room; fp8 fits on 1 GPU for weights alone and comfortably on 4; int4 fits on one H100 with 43 GB left over.

**But KV cache does not shrink with weight quantization.** That is the point candidates miss. At 320 KB/token, a single H100 holding a 37 GB int4 model with ~33 GB of usable cache supports 33e9 / 320e3 ≈ **103,000 tokens** of KV — about 26 concurrent 4k-token sequences. That may be fine for a low-concurrency deployment and is nowhere near enough for a high-traffic one. So the real question is not "does the model fit" but "does the model plus my concurrency target fit," and the answer usually keeps TP above 1.

**Throughput consequences.** Weight-only 4-bit accelerates *decode*, because decode is bandwidth-bound and you are reading 4× fewer weight bytes per step. It does **not** accelerate prefill, because prefill is compute-bound and weight-only int4 kernels dequantize to bf16 before the GEMM — you do the same FLOPs plus dequantization overhead. On a prefill-heavy workload (RAG, long context), int4 can be *slower*. fp8 is different: on H100 the tensor cores execute fp8 natively at roughly 2× bf16 throughput, so fp8 helps both phases. That asymmetry is the practical reason I default to fp8 on Hopper-class hardware for serving and reserve int4 for memory-constrained or decode-heavy cases.

**💰 Math, the real comparison.** Serving a 70B at a fixed 60 req/s peak:

- bf16 on 8×H100, measured 6.75 req/s per replica at SLO → 9 replicas → 72 GPUs → $180/hr → **$131k/month**.
- fp8 on 4×H100. Weights 70 GB across 4 GPUs = 17.5 GB each, leaving ~52 GB × 4 = 208 GB of cache (half the bf16 fleet's per-replica cache, so lower per-replica concurrency), but faster decode. Suppose measurement gives 5.5 req/s per replica → 11 replicas → 44 GPUs → $110/hr → **$80k/month**. A **39% saving**.
- The saving is real only if the eval holds. fp8 weight+activation quantization on a 70B typically costs very little on standard benchmarks, but "typically" is not a launch criterion — you run your own eval, including on long-context and structured-output tasks where degradation concentrates.

**⚠ Trap:** quantizing the KV cache to fp8 *and* the weights to int4 and reporting the combined memory saving without a long-context eval. Each is individually defensible; together they compound error through the attention computation over long sequences, and the failure shows up specifically as degraded recall from the middle of a long context — precisely the capability your document-analysis product sells. Gate on a long-context retrieval eval, not on MMLU.

### 🎯 Design a job queue for a 100,000-GPU cluster with preemption and checkpoint-aware eviction.

This is the frontier-lab and hyperscaler infra question. I will answer it as a system with five components and be explicit about the invariants each one holds.

**The mental model.** At this scale the scheduler is not allocating GPUs, it is allocating **contiguous, topologically-coherent blocks of GPUs for bounded time**, and the scarce resource is not GPU-hours but *network-adjacent* GPU-hours. Two 512-GPU jobs placed across the same set of leaf switches will contend on the fabric and both run slowly, so a placement that looks fine in a resource accountant is a 30% goodput loss in reality. Everything below follows from "topology is a first-class resource."

**1. Job model.** Every job declares: GPU count, a *minimum* and *maximum* world size if it is elastic, expected duration, priority tier, checkpoint interval, restart-from-checkpoint capability (a hard boolean — it changes everything), and topology requirement (must-fit-in-one-NVLink-domain / prefer-same-rack / same-island / anywhere). Jobs that cannot checkpoint are a separate, expensive class and I would price them accordingly internally, because they cannot be evicted cheaply.

**2. Hierarchical queues with fair share and quota borrowing.** Organizations get guaranteed quota; unused quota is borrowable by other organizations at a lower effective priority, and is reclaimable on demand. This is DRF-style fair-share (as in YARN/Volcano/Kueue cohorts), and it is what makes a shared cluster politically survivable — teams tolerate lending capacity only if reclaim is guaranteed and fast.

**3. Topology-aware gang placement.** The placer maintains a tree (GPU → node → rack → leaf switch → island) and allocates the smallest subtree that satisfies the request. Best-fit against the tree, with a defragmentation objective: prefer placements that leave large contiguous blocks intact. Without this, a week of small jobs fragments the cluster so that no 4,096-GPU job can ever start even though 30,000 GPUs are free. **Fragmentation, not utilization, is the metric that kills large clusters.**

**4. Checkpoint-aware eviction — the heart of the question.** When a high-priority job needs capacity, do not evict by priority alone. Score each candidate victim by the *work destroyed*:

```
lost_work(job) = (now − last_checkpoint_time) × gpus
evict_cost(job) = lost_work(job)
                + restart_overhead(job)      # requeue + weight reload + warmup
                + progress_penalty(job)      # near-completion jobs cost more to kill
```

Evict the set minimizing total `evict_cost` subject to freeing enough topologically-suitable GPUs. Then two refinements that matter enormously:

- **Checkpoint-then-evict.** Instead of killing immediately, signal the victim to checkpoint now and exit. If its checkpoint interval is 20 minutes and it is 18 minutes in, waiting 30 seconds for an on-demand checkpoint saves 18 × N GPU-minutes. With async checkpointing (staging to host memory, flushing to storage in the background) the pause is seconds, so this is almost always the right move. Give victims a preemption grace window and a signal contract; kill hard only on timeout.
- **Elastic shrink before eviction.** If the victim declared `min_world_size < current`, shrink it instead of killing it. A 512-GPU job dropping to 256 loses throughput but loses *zero* work, and elastic frameworks (torchrun elastic, Ray) can rendezvous at a new world size. Shrink beats evict every time it is available, and it is a strong reason to require elasticity from large training jobs as a platform policy.

**5. Reliability and goodput accounting.** At 100k GPUs, node failures are continuous — with a per-node MTBF of, say, 10,000 hours and 12,500 nodes, expected failures are 12,500/10,000 = **1.25 per hour**. So the scheduler must treat failure as normal: health-check and drain nodes automatically, maintain a hot spare pool (2–5% of the fleet) so a failed rank is replaced in seconds rather than requeued, and detect stragglers (a job whose step time drifts up is often one sick GPU — evict that rank, not the job). The metric the platform is graded on is **goodput**: GPU-hours that produced retained progress, divided by GPU-hours allocated. Queue wait, fragmentation, eviction losses, restart overhead and straggler drag all show up in that one number, and it is the number I would put on the team's dashboard.

**💰 Math for the checkpoint interval, which the scheduler should influence:** for a job on *N* GPUs with per-node failure rate λ and checkpoint cost *C* seconds, the classic Young/Daly result gives an optimal interval ≈ `sqrt(2 × C × MTBF)`. With MTBF for a 512-GPU (64-node) job = 10,000/64 = 156 hours = 5.6e5 s and C = 60 s, the interval is sqrt(2 × 60 × 5.6e5) = **8,200 s ≈ 2.3 hours**. But if the platform preempts more often than hardware fails — say every 40 minutes under contention — the *effective* MTBF is 2,400 s and the optimal interval collapses to sqrt(2 × 60 × 2400) = **537 s ≈ 9 minutes**. The scheduler should publish the observed preemption rate so jobs can tune their interval to it; a fixed hard-coded interval is wrong in both directions.

**⚠ Trap:** designing this purely as a priority queue. Priority tells you *who* wins; it does not tell you *whom to evict*, and evicting a job that is 19 minutes past its last checkpoint when a lower-priority one checkpointed 30 seconds ago destroys hundreds of GPU-hours for no reason. Eviction policy is an optimization over destroyed work, and stating that distinction is the thing that separates a strong answer from a generic one in this round.

### The thesis question: serve a 70B at 2-second p95 on a fixed GPU budget, and defend the dollar figure.

I will answer this the way I would run it: requirements, sizing, configuration, validation, cost, and what I would cut.

**Requirements I would extract before designing anything.** What is the 2-second p95 measuring — TTFT or end-to-end? (I will assume TTFT, since end-to-end p95 on a streaming product is not a meaningful target.) What is the token-length distribution? What is peak concurrency and its shape? How much prefix sharing? Is 70B a requirement or a guess — has anyone evaluated a 32B or an 8B on this task? That last question is the highest-leverage one in the room, and I ask it every time: **a 70B is 4–8× the serving cost of a 14B, and on a well-scoped extraction or routing task the smaller model often ties.** If the answer is "we tested and 70B is required," fine, proceed.

Assume the answers: TTFT p95 ≤ 2 s; 4,000 input / 500 output tokens median, with a p95 input of 16,000; peak 60 req/s; 60% of traffic shares a 3,000-token system prompt.

**Sizing.** 70B in fp8 = 70 GB of weights. TP=4 on H100 puts 17.5 GB/GPU, leaving ~52 GB × 4 = 208 GB of KV cache per replica. KV at fp8 is 160 KB/token (bf16 would be 320); I will start with bf16 KV at 320 KB/token pending a long-context eval, giving 208e9/320e3 = **650,000 tokens** of cache per replica ≈ 145 concurrent 4.5k-token sequences. Plenty.

TP=4 rather than 8 because the model fits and 4 gives me two replicas per 8-GPU node — better GPU efficiency than one TP=8 replica, and the collectives are cheaper. I would benchmark TP=2 as well.

**Configuration.** vLLM with prefix caching on (that 3,000-token shared system prompt is 60% of traffic), chunked prefill on (to protect ITL against the 16k p95 prompts), `max_model_len` set to 32,768 not 128k, `gpu_memory_utilization` 0.88, `max_num_seqs` tuned by the sweep. Prefix-aware routing at the gateway hashing on the leading 2,000 tokens plus session ID, with a load-balance term so no replica hot-shards.

**Validation — the sweep.** Open-loop, production-shaped lengths, realistic prefix sharing, warm cache. Suppose it yields 6.75 req/s per replica at TTFT p95 = 1.85 s, with the asymptote at 14 req/s and p95 = 11 s. Operating at 75% → 5.06 req/s per replica.

**Fleet.** ceil(60 / 5.06) = **12 replicas** × 4 GPUs = **48 H100s**. Then a burst test measures a 90-second scaling lag; the p99 observed burst multiplier is 1.4×, so `minReplicas` = 12 with autoscaling to 18, and I hold 12 as always-on since peak is a daily event.

**💰 The dollar figure, with arithmetic.** 48 H100s at $2.50/GPU-hr = $120/hr. Always-on: 120 × 730 = **$87,600/month**. With a 40% one-year commitment on the 48-GPU floor: $1.50 × 48 × 730 = **$52,560/month**, plus on-demand burst above it — say 4 extra GPUs' worth for 2 hours/day = 4 × 2 × 30 × $2.50 = $600. Call it **$53,200/month**, or $638k/year.

Per token: at 60 req/s × 500 output tokens × 86,400 s/day = 2.59e9 output tokens/day = 7.78e10/month. $53,200 / 77,800 Mtok = **$0.68 per million output tokens** — which is competitive with a hosted open-model endpoint, and that is only true because the duty cycle here is high and the commitment discount is real. If peak were two hours a day instead of steady, this number would be 5× worse and I would recommend the API.

**What I would cut if the budget were half.** In order: (1) evaluate a 32B — likely a 2× fleet reduction for a small quality delta, and it is the first thing I would test; (2) fp8 KV cache after a long-context eval, roughly doubling per-replica concurrency; (3) raise the TTFT SLO from 2 s to 3 s, which moves the operating point up the curve and can be worth 25–30% of the fleet; (4) route the easy 50% of traffic to an 8B with a quality classifier. Every one of those is a quality/cost trade with a measurable eval, and I would present them as a menu with numbers rather than as a recommendation.

**🗣 Say this in the room:** "48 H100s as twelve TP=4 replicas, sized from a measured 6.75 req/s per replica at a 1.85-second TTFT p95, run at 75% of that with a 12-to-18 autoscale range. That's $53k a month on a one-year commit, about $0.68 per million output tokens. The number I'd challenge first isn't the infrastructure — it's whether this needs a 70B at all."

### 🏋 Drill: size a serving fleet in twenty minutes, no calculator, no internet.

**The prompt.** You are given: model = 32B dense, 64 layers, 8 KV heads, head_dim 128, hidden 5,120. Hardware = H100 80 GB, NVLink node of 8. Traffic = peak 120 req/s; input tokens p50 1,200 / p95 9,000; output tokens p50 400 / p95 1,500; 45% of requests share a 2,500-token prefix. SLO = TTFT p95 ≤ 1.2 s, ITL p95 ≤ 60 ms. Budget question: what does it cost per month and per million output tokens?

**Deliver, in twenty minutes, on paper:**

1. Weight bytes at bf16, fp8 and int4, and the minimum GPU count for each.
2. KV bytes per token, and per-replica KV capacity at your chosen TP degree.
3. Your chosen TP degree with the reason (fit, KV-head divisibility, NVLink domain, latency-vs-throughput).
4. Concurrent-sequence capacity implied by the KV budget at the p50 and p95 sequence lengths — both numbers, because the p95 is what breaks you.
5. An estimated decode-bound ITL floor from HBM bandwidth (weight bytes per rank ÷ 3.35 TB/s), and whether the ITL SLO is even achievable.
6. A stated throughput assumption per replica, **explicitly labelled as an assumption to be measured**, and the replica count it implies at 75% operating utilization.
7. Monthly cost at on-demand and at a 40% commit, and cost per million output tokens.
8. The two configuration flags you would set first and why.
9. The one number you would measure before believing any of the above.

**Pass criteria:** every figure has its arithmetic beside it; you flagged that TP > 8 would replicate KV heads; you used the p95 sequence length somewhere and not only the p50; you labelled the throughput figure as an assumption rather than asserting it; and your answer to (9) is a throughput-at-SLO sweep, not a vendor benchmark. Failing to label the throughput assumption is an automatic fail — it is the single number that determines the whole answer and the only one you cannot derive from first principles.

**Redo it under variation:** repeat with an MoE where total params are 8× active params, and with a workload whose input p95 is 60,000 tokens. The second variant should change your TP degree and probably introduce a second fleet.

### 🏋 Drill: build the engine bake-off harness in forty-five minutes.

**The prompt.** You have two candidate serving configurations and one week to pick. Write — from memory, no framework docs — the harness and the decision procedure.

**What you must produce:**

1. **A trace sampler.** Given a parquet of production request logs, emit a replayable workload preserving the *joint* distribution of input and output lengths and the *prefix-sharing structure*. State how you preserve prefix sharing (replay actual prompt prefixes, or synthesize with a controlled sharing rate) and why independent sampling of the two length distributions is wrong.
2. **An open-loop generator.** Poisson arrivals at a target rate, not a fixed worker pool. Explain coordinated omission in one sentence and show which line of your code avoids it.
3. **The metric set:** TTFT p50/p95/p99, ITL p50/p95, end-to-end p95, output tok/s, **goodput at SLO**, plus engine-side prefix-cache hit rate, KV utilization and preemption count scraped on the same timeline.
4. **The sweep plan:** rates, hold duration, drain between steps, warm-up discard, and how many repetitions per point to make the comparison statistically meaningful (I want to hear you say you ran each point at least three times and report a spread, not a single number).
5. **The decision rule, written before you look at results.** For example: "pick the config with higher goodput at SLO; if within 10%, pick the one with lower p99 TTFT; if still within 10%, pick the one with lower operational burden." Pre-registering the rule is what stops you from rationalizing toward the config you already liked.
6. **The cost translation.** Convert each config's goodput into replicas needed at peak, then into monthly dollars, and report the comparison in dollars rather than in tokens/second. That is the sentence a director remembers.

**Pass criteria:** the harness is open-loop; prefix sharing is explicitly reproduced; goodput-at-SLO is among your metrics; you scrape engine-internal counters alongside client-side latency; you pre-registered the decision rule; and your final output is a dollar figure per config with the arithmetic shown. Forty-five minutes is enough for the design and the skeleton, not the implementation — write the structure and the decision rule, and be prepared to defend why each metric is on the list.

**Why this drill and not another:** almost every serving question in a real loop terminates in "how would you know?" A candidate who can produce this harness from memory has answered that question permanently, and can attach it to the parallelism, engine, quantization and autoscaling decisions in every preceding question rather than arguing from priors.
