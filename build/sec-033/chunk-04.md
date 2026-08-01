### Why does upgrading an embedding model force a full reindex? Isn't there a cheaper path?

Because **every embedding model defines its own coordinate system, and there is no canonical alignment between two of them.** This is the single most consequential operational fact about embeddings and it deserves a precise statement rather than a hand-wave.

The training objective only ever constrains *relative* geometry — it says "query `i` should be closer to passage `i` than to passage `j`." Any global rotation, reflection or rescaling of the whole space satisfies that constraint equally well. Two models trained on the same data with different random seeds converge to spaces that are equally good and completely incompatible: dimension 47 means something in model A and something unrelated in model B, and even the "same concept" occupies an arbitrarily different direction. So a dot product between a model-A document vector and a model-B query vector is not a degraded similarity — it is **an uncorrelated number**, statistically indistinguishable from comparing against a random vector.

This is different from every migration instinct a backend engineer has. A schema migration can be done column by column. A cache can be warmed incrementally with a fallback to the origin. An index rebuild in Postgres serves reads from the old index while `CREATE INDEX CONCURRENTLY` runs. Here there is no partial-compatibility regime: at any instant, a query vector is either in the old space or the new one, and every document it is compared against must be in the same space.

The other half of the trap: even a **same-model version bump** can break this. If a provider ships `v3.1` of a hosted embedding model, or you upgrade a `sentence-transformers` version that changes the default pooling or normalization, or a tokenizer update changes how a rare token is split, your new vectors may be subtly misaligned with your old ones. Subtly is worse than completely, because completely fails loudly.

**The rule I enforce in review:** every vector row carries an `embedding_model_version` column, and the query path asserts that the version it is using matches the version stamped on the index it is querying. That assert costs nothing and it converts a silent quality collapse into a loud startup failure.

**🗣 Say this in the room:** "Embedding spaces are only defined up to an arbitrary rotation — the contrastive objective constrains relative distances, not absolute directions — so vectors from two models are not 'a bit different,' they're uncorrelated. That means a model upgrade is a full reindex, not an incremental migration, and I stamp a model-version tag on every vector and assert it at query time so a mismatch fails loudly instead of quietly returning noise."

### Suppose part of my index got re-embedded with a new model and part didn't. What does that look like in production, and how would I catch it?

This is the failure mode I would explicitly design against, because it is **quiet, partial, and self-consistent enough to survive every smoke test.**

What it looks like: queries still return results — the index happily returns *something* for any vector. But the results are drawn almost entirely from **one** of the two populations. Whichever population happens to sit closer to the query's region of the new space dominates the top-k, and the other population becomes effectively invisible. So the symptom reported by users is not "search is broken." It is "search doesn't find anything from before March" or "the new documents never come up." Support triages it as a data problem. Meanwhile your aggregate nDCG on a benchmark drawn from the *old* documents looks fine, because those still match.

A second, meaner variant: the two populations have different **score distributions**. Model B's vectors might produce systematically higher cosines with model-B queries. If you have a relevance threshold anywhere in the stack — a cutoff below which you don't include a chunk — the mismatched population gets filtered out entirely, so it isn't even ranked low, it's gone.

**Detection, in order of how fast it gets you an answer:**

1. **Version tag as a hard invariant.** Every vector row has `embedding_model_version`. Query path asserts equality. This is prevention, not detection, and it is where you should spend the effort.
2. **A canary query set that spans the corpus.** 50 queries whose known-correct documents are deliberately spread across ingestion dates, document sources and tenants. Run them every 15 minutes and alert on recall. A partial-reindex bug shows as recall dropping for a *subset* of canaries while others are fine — which is diagnostic in itself.
3. **Distributional monitoring on the vectors themselves.** Sample 10k vectors per hour, compute mean norm and the mean pairwise cosine to a fixed set of 100 random probe vectors. Two model versions produce visibly different distributions on both. This is a cheap statistical tripwire that catches drift you didn't anticipate — and it costs one small job.
4. **Coverage audit.** `SELECT embedding_model_version, count(*), min(updated_at), max(updated_at) GROUP BY 1`. If that returns more than one row outside a planned migration window, page someone.

**⚠ Trap:** relying on aggregate retrieval metrics to catch this. Your benchmark is 200 queries and their labeled documents were probably all indexed at the same time. If they were all indexed *before* the partial migration, your metric is perfect and half your corpus is dark. **Stratify your canary set by ingestion cohort** for exactly this reason.

**🔍 Failure taxonomy — how a partial reindex happens in the first place:** a backfill job that crashed at 60% and nobody rechecked; a Celery/Kafka consumer processing new documents with the *new* model config while the backfill of old ones was still queued; a rollback of the embedding service that didn't roll back the vectors already written; a per-tenant rollout where one tenant's shard was missed. Every one of these is prevented by the same discipline: **write to a new index, not into the existing one.**

### Design the zero-downtime re-embedding of a 100M-chunk index. Give me cost, wall clock, and the rollback plan.

The design principle is the one you already use for a schema migration you cannot take downtime for: **build the new thing beside the old thing, verify it, flip an alias, keep the old thing warm.** The only new wrinkle is that the two artifacts are mutually unreadable, so there is no dual-read fallback per-query — the flip has to be atomic at the *alias* level.

**Architecture.**

- Two named indexes, `chunks_v7` (live) and `chunks_v8` (building), behind a **logical alias** that the query path resolves at request time — Elasticsearch/OpenSearch aliases, a Qdrant collection alias, or in a Postgres/pgvector world just two tables and a view you can `CREATE OR REPLACE` inside a transaction.
- A **dual-write** path: from the moment the backfill starts, every new or updated document is written to *both* indexes, each with its own model. This is what prevents the backfill from racing with live ingestion. Your Kafka/Celery consumer becomes a fan-out to two embedders.
- A **backfill job** reading the canonical chunk store (which must be your source of truth — the vector DB is a derived index, never the system of record) in `id` order, checkpointed, idempotent by `chunk_id`, resumable.

**💰 Cost and wall clock.** 100M chunks × 400 tokens = 40 Btok.

- *Self-hosted*, planning at ~100k tokens/s per H100 measured (derived earlier; the FLOP-bound upper limit is ~296k tok/s at 20% MFU, and measured is 2–3× below that): `40e9 / 1e5` = 400,000 GPU-seconds = **111 GPU-hours**. On 8 H100s that is **~14 hours wall clock**, at 8 × $3/hr × 14 = **$336**. On 32 H100s: ~3.5 hours, same total dollars. This is embarrassingly parallel — shard by `chunk_id` range.
- *API path* at $0.02/Mtok: `40,000 × $0.02` = **$800**, and the wall clock is set by your rate limit, not your money. If your tier allows 5M tokens/minute, `40e9 / 5e6` = 8,000 minutes = **5.6 days**. That rate limit, not the price, is the thing that will surprise you, and it is worth pre-negotiating with the provider before you plan the migration.
- *Index build.* HNSW construction is often the hidden cost — for 100M vectors it can rival or exceed the embedding time. Budget for it explicitly and build with a lower `efConstruction` if you must, then measure the recall cost.

**Cutover.** Backfill completes → run the full 200-query benchmark against `v8` and compare to `v7` on the same queries, requiring nDCG@10 not to regress beyond the bootstrap interval → run **shadow reads** for 24–48 hours, sending a copy of live query traffic to `v8` and logging both result sets without serving them, so you can measure overlap and inspect the disagreements → flip the alias → keep `v7` alive and dual-written for at least a week.

**Rollback** is then a single alias flip back, which is why `v7` must stay dual-written. The moment you stop dual-writing, rollback becomes "re-run a backfill," and your recovery time goes from 30 seconds to 14 hours.

**⚠ Trap:** treating the vector store as the source of truth. If your chunk text only lives in the vector DB, you cannot re-embed without extracting from it, and any corruption is unrecoverable. **The chunk store — Postgres, S3, whatever — is authoritative; the vector index is a cache you can always rebuild.** I would push back hard on any design that violates this, because it converts a routine migration into an archaeology project.

**⚠ Trap:** forgetting that **deletes** must apply to both indexes during the dual-write window. A document deleted upstream that is still live in `v8` gets served to a customer after the flip, and "we quoted a deleted document" is a compliance incident, not a bug.

### Could I train a small linear map from the old embedding space to the new one and skip the reindex?

You can, it sometimes works surprisingly well, and I would still almost never ship it. Being able to say both halves of that is the interesting answer.

**The mechanism.** Embed a sample of `n` chunks with both models — say 100k chunks — giving you paired matrices `X` (old, `n × d_old`) and `Y` (new, `n × d_new`). Solve for the linear map `W` minimizing `‖XW − Y‖²`, which is ordinary least squares: `W = (XᵀX)⁻¹XᵀY`, or the orthogonal-constrained version (the **orthogonal Procrustes** solution, `W = UVᵀ` from the SVD of `XᵀY`) if both spaces are unit-normalized and you want to preserve angles. This is the same machinery used in the cross-lingual word-embedding literature to align two monolingual spaces. Then push all 100M old vectors through `W` — a single `100e6 × d_old × d_new` matmul, which at 1024→1024 is `100e6 × 1024 × 1024 × 2` = 2.1e14 FLOPs, **under a second of H100 time.** The appeal is obvious: seconds instead of 14 hours.

**Why I wouldn't ship it.** The map is linear; the relationship between two independently-trained transformer representation spaces is not. The residual error is not uniform — it is largest exactly where the new model *differs most* from the old one, which is to say, precisely on the cases where you were upgrading in order to get better. You keep the old model's mistakes, projected into new coordinates. You have spent the migration and captured maybe 60–80% of the quality gain, with no way to know which 20% you lost without... running the full evaluation you were trying to avoid.

There is also an ugly operational property: your index now contains *approximated* vectors while newly-ingested documents contain *true* new-model vectors, so you have re-created the two-population problem from the previous question, just with a smaller gap.

**Where it is genuinely useful:** as a **bridge during a long backfill**. Project the not-yet-re-embedded portion through `W` so that the new index is queryable and roughly sane from hour one, while the true re-embedding proceeds and overwrites rows progressively. That gives you a graceful degradation curve instead of a cliff, and it is legitimate engineering. It is also useful for **cheap evaluation**: before committing to a migration, project a sample and get a rough read on whether the new model is worth it.

**🗣 Say this in the room:** "You can fit an orthogonal Procrustes map between the two spaces from ~100k paired samples and project the whole index in under a second of GPU time. I'd use that as a bridge during a long backfill, never as the destination — the linear residual is largest exactly on the examples where the new model differs from the old one, which is the quality you were migrating for."

### How do you actually serve embeddings at scale? Walk me through the deployment.

Embedding serving is a **separate fleet with fundamentally different scaling behavior from LLM serving**, and treating it as "just another model server" is the design mistake.

The key structural difference: an embedding forward pass is **pure prefill, no decode.** There is no KV cache, no autoregressive loop, no token-by-token streaming. It is one batched forward pass, so it is **compute-bound and trivially batchable** — which means, unlike LLM decode, throughput scales close to linearly with batch size until you saturate the GPU. That has three consequences: (1) GPU utilization is a meaningful autoscaling signal here, unlike for LLM serving; (2) latency is a deterministic function of `(batch_size, max_seq_len_in_batch)`; (3) you should batch aggressively.

**The stack.** **TEI** (Hugging Face Text Embeddings Inference, Rust, token-based dynamic batching, Flash-Attention-backed kernels, supports rerankers too) and **Infinity** (Python, broad model support, multi-model serving on one process) are the two purpose-built servers. vLLM also serves embedding models. For CPU-only or edge, ONNX Runtime with a quantized model is genuinely viable for small encoders.

**The batching detail that matters most: bucket by length.** A naive batch pads everything to the longest member, so one 512-token chunk in a batch of 20-token queries wastes 96% of your compute. Token-budget batching — fill a batch until you hit a total-token cap rather than a fixed count — plus sorting the incoming queue by length gives you a large multiple in real throughput. TEI does this natively; if you hand-roll a service you must implement it or your measured throughput will be a fraction of your FLOP estimate.

**Two fleets, not one.** Ingestion embedding and query embedding have opposite requirements. Ingestion is **throughput-oriented**: huge batches, long sequences, latency-insensitive, perfect for spot instances and for preemption. Query embedding is **latency-oriented**: batch size 1–8, 20-token inputs, sits directly in your TTFT budget, must never be preempted. Running them on the same replicas means a backfill batch of 512 × 512-token chunks lands in front of a user query and adds 200 ms to its p99. **Separate deployments, separate autoscaling, separate priority classes.**

**📐 Numbers you must know:** for a 335M-parameter encoder, the FLOP-derived ceiling on an H100 at 20% MFU is ~296k tokens/s (`1.98e14 / (2 × 0.335e9)`); plan production capacity at **~100k tokens/s per H100** or **~15–25k tokens/s on an L4**, and measure. Query-side latency for a 20-token query at batch 8 should be **2–5 ms of GPU time**; if you are seeing 40 ms, your bottleneck is tokenization, HTTP, or a cold Python path — not the model.

### Your embedding service p99 tripled overnight and query latency spiked with it. Nothing was deployed. Debug it.

"Nothing was deployed" plus "throughput-side and latency-side both degraded" points me immediately at **contention between a batch workload and the interactive workload**, and my first question is what job started running.

Here is the ladder I'd walk, in order, with what each step rules out.

**1. Is there a backfill running?** Check for a bulk ingestion job — a new customer onboarded, a scheduled reindex, a CDC replay after a consumer lag spike. If ingestion and query traffic share a fleet, a backfill's 512-item batches of 512-token chunks queue ahead of 20-token queries, and every query waits behind a batch that takes 200 ms. **This is the single most likely cause and the fix is fleet separation, not more GPUs.** The tell: GPU utilization is pinned at 100%, throughput in tokens/s is *up*, and only latency is bad. High utilization plus high throughput plus bad latency is a queueing problem, not a capacity problem.

**2. Did the input length distribution change?** Compute time scales with tokens, and attention scales quadratically within a sequence. If someone started sending 2,000-token queries (a "search with this whole document as the query" feature, a pasted email, a retry loop that concatenates history), your per-request cost jumped 10–100× with no code change. Plot p50/p99 **input token count**, not just request count — if you are not exporting that histogram today, add it, because request-rate dashboards are blind to this entire class of incident.

**3. Batch composition / padding waste.** Related but distinct: if length variance within batches increased, padding waste rose even at constant mean length. Export **effective tokens ÷ padded tokens** as a gauge. Below ~0.6 you have a bucketing problem.

**4. Is it actually the GPU?** Profile the split. Tokenization is CPU work and is genuinely a bottleneck for short inputs — a fast tokenizer on one core can be slower than the GPU forward pass at batch 1. If CPU is saturated and GPU is at 30%, add CPU or move tokenization off the request path. This is a very common misdiagnosis: people scale GPUs to fix a tokenizer bottleneck.

**5. Autoscaling lag or a replica churn loop.** If HPA scaled down overnight on low traffic and then traffic returned, cold starts (weight load, CUDA context init, first-batch kernel autotuning) can take 30–90 s per replica, during which the surviving replicas are overloaded, which triggers more scaling, which... Check replica count over time against latency.

**6. Downstream backpressure.** If the vector store is rejecting or slowing writes, the embedding service's write queue backs up, memory grows, and its own request handling degrades. The symptom appears in the embedder; the cause is in the database.

**🔍 The decision rule that shortcuts all of this:** GPU util high + tokens/s high + latency bad = **queueing/contention** → separate the fleets or add priority. GPU util high + tokens/s *flat* + latency bad = **per-request cost went up** → input lengths changed. GPU util low + latency bad = **it isn't the GPU** → tokenizer, HTTP, network, or a downstream dependency.

### How do you cache embeddings? What is the key and what invalidates it?

An embedding is a **pure function of (model version, model config, exact input string)** — which makes it one of the most cacheable things in the entire stack, and one of the most commonly cached wrong.

**The cache key must be the hash of the tuple, not of the text.** Concretely: `sha256(model_id || model_revision || prompt_prefix || pooling || normalize_flag || text)`. Every element of that is load-bearing. Drop `model_revision` and a silent provider-side model update poisons your cache with vectors from a different space. Drop `prompt_prefix` and the `query:`-prefixed and unprefixed versions of the same string collide — which is exactly the recall-collapse bug from earlier, now cached and persistent across deploys. Drop `normalize_flag` and you get magnitude-inconsistent vectors depending on which code path warmed the entry.

**What it's worth, on the two sides:**

- **Ingestion side.** This is where it actually pays. Documents get re-processed constantly — a chunking-parameter change, a metadata backfill, a pipeline retry, a CDC event for a row where only `updated_at` changed. Content-hash the chunk text; if the hash is unchanged, reuse the vector. In a real pipeline with a 5%-per-month document change rate, this turns a monthly full reprocess from 40 Btok into 2 Btok — **a 95% reduction, $800 → $40 per pass** at $0.02/Mtok.
- **Query side.** Query distributions are Zipfian: in most products the top 1% of distinct queries are 20–40% of traffic. Caching query embeddings in Redis with a 24-hour TTL gets you a hit rate in that range and removes 3–5 ms plus one network hop from those requests. Worth doing, but the win is latency, not money — query text is short and query embedding is a trivial share of your bill.

**Invalidation.** Content-addressed keys mean you never invalidate on content change — a changed chunk simply has a different key, and the old entry ages out. What you *do* invalidate deliberately is on **model version bump**: include the version in the key so a bump makes the entire old namespace unreachable rather than requiring a flush, and let TTL reclaim the space.

**⚠ Trap:** caching by `chunk_id` instead of by content hash. The chunk's text changes, the id doesn't, and you now serve a vector for text that no longer exists — a document that says one thing and retrieves as though it says another. This is the embedding-layer version of a stale cache, except that nothing downstream can detect it, because the vector is perfectly well-formed. **Content-address or don't cache.**

**⚠ Trap:** an unbounded in-process dict as the "cache." For 1024-dim fp32 that is 4 KB per entry plus Python object overhead — realistically ~4.5 KB — so a million entries is **4.5 GB of RSS** in a process you sized for 2 GB, and it will OOM on the day traffic doubles. Use Redis, or an LRU with an explicit item cap you computed from the byte size.

### Query embedding sits directly in the TTFT critical path. How do you keep it out of your latency budget?

First, size the problem honestly, because the answer differs by an order of magnitude depending on the number. For a small encoder embedding a 20-token query, GPU time is **2–5 ms**. Add HTTP, serialization and a network hop inside the cluster and you are at **8–15 ms**. Against a RAG TTFT budget of, say, 800 ms, that is 1–2% and **you should not spend engineering time on it.** For a 7B decoder-based embedder over an HTTP hop it can be 40–80 ms, which is 5–10% and worth attacking.

So the first answer is: **measure it, and if it is under 2% of your budget, say so and move on.** Optimizing a 10 ms component of an 800 ms path is a junior instinct and interviewers notice.

When it *is* worth attacking, in order of leverage:

**1. Colocate.** Run the embedding model in-process or as a sidecar rather than as a network service. Removes the hop and the serialization entirely — often half the wall-clock. For a 100–350M model, in-process on the API server with ONNX Runtime on CPU is genuinely viable and eliminates a whole tier from your architecture.

**2. Overlap it with something.** In an agentic or multi-turn system you frequently know the query several hundred milliseconds before you need the retrieval — the moment the user's message arrives, before you have finished assembling the prompt or checking auth. Fire the embedding as an `asyncio` task at message receipt and `await` it only where you actually need the vector. This is the same structured-concurrency move you already make everywhere else, and it makes the cost genuinely zero when there is any other I/O to hide behind.

**3. Cache the head.** Per the previous question, 20–40% of queries are repeats. A Redis hit is sub-millisecond.

**4. Use a smaller model on the query side.** This is legal only if the query and document encoders are the *same* model, which for a bi-encoder they must be — so this really means choosing a smaller model overall, and it is a quality decision, not a latency optimization. Do not let anyone talk you into embedding queries with a different model than the documents.

**5. Speculative retrieval.** In a chat UI, embed and retrieve on the partial query as the user types (debounced at ~300 ms), so the retrieval is often already warm on submit. Real technique, real complexity, and it multiplies your embedding QPS by the number of debounce fires per query — do the arithmetic before proposing it: at 3 fires per submitted query you have tripled your embedding fleet to save 10 ms.

**🗣 Say this in the room:** "Query embedding is 8–15 ms end-to-end for a small encoder, which is 1–2% of a typical RAG TTFT budget — so my first move is to measure it and probably not optimize it. If it's a 7B embedder at 60 ms, I'd colocate it to kill the network hop, cache the Zipfian head of the query distribution, and kick the embedding off as an asyncio task the moment the message arrives so it overlaps with auth and prompt assembly."

### Give me the full failure taxonomy for the embedding layer. How do I triage a retrieval quality complaint?

Here is the decision procedure I actually run, ordered so that the cheapest checks that eliminate the most possibilities come first.

**Tier 0 — is the content even there?** Query the chunk store directly for text you know should match. If it isn't there: ingestion. Check parse failures (`SELECT status, count(*) FROM ingest_log`), check the quarantine queue, check whether a source connector's token expired three weeks ago. **In my experience this is the top cause of "search doesn't find X" and people spend days on the retriever first.**

**Tier 1 — is the embedding function correct?** Symptoms: uniform degradation across all query types, no errors, index untouched. Causes, in likelihood order: missing or swapped instruction prefix; wrong pooling (mean vs CLS vs last-token); missing normalization; a tokenizer or library version bump; silent truncation of long inputs. **Detection: the golden-vector CI test.** Twenty (text → reference embedding) fixtures, assert cosine > 0.9999. This one test covers this entire tier.

**Tier 2 — is the index consistent with the embedder?** Symptoms: some documents findable, others invisible; recall varies by ingestion date, tenant, or source. Causes: partial reindex, mixed model versions, a crashed backfill, deletes not propagated. **Detection: `GROUP BY embedding_model_version` should return exactly one row, and a cohort-stratified canary set.**

**Tier 3 — is the ANN index losing things the embedder found?** Symptoms: the correct document scores well under brute force but doesn't appear in ANN results. Causes: `efSearch`/`nprobe` too low, tombstone accumulation after heavy deletes degrading graph connectivity, a filter applied post-hoc collapsing recall. **Detection: sample 200 queries, run exact brute-force top-10 and ANN top-10, and report the overlap. If ANN recall against exact is below ~0.95, tune the index — this is a knob, not a model problem.**

**Tier 4 — is the model wrong for these queries?** Symptoms: failures cluster by query *type*. Exact identifiers failing → dense retrieval structurally cannot do this; add lexical/hybrid. Domain jargon failing → fine-tune or pick a domain model. Negation failing → no embedding handles this; needs query understanding or a reranker. Non-English failing → language coverage. **Detection: cluster your failing queries and read 30 of them. The cluster labels are your work plan.**

**Tier 5 — is it hubness or duplication?** Symptoms: the same few chunks appear for unrelated queries, or the top-5 is five copies of one paragraph. **Detection: retrieved-document frequency histogram over 5,000 queries.** Fix at ingestion (dedupe, strip boilerplate) before you fix it at ranking (MMR).

**Tier 6 — is it actually generation?** The correct chunk was in the top 5 and the answer was still wrong. Nothing in this section helps you; go work on the prompt, the grounding instructions, or the model.

**🗣 Say this in the room:** "I triage top-down: is the content in the store at all, is the embedding function right, is the index consistent with the embedder, is ANN recall losing what the embedder found, is the model wrong for this query class, and only then is it generation. The first four are one-hour checks with concrete detectors — a golden-vector test, a version-count query, and exact-vs-ANN overlap on 200 queries — and they eliminate most of the search space before I've touched a model."

### Timed drill: build a domain fine-tune and prove it worked, in 90 minutes.

**🏋 Drill.** No AI assistance, no copying from a previous project. You get a corpus of 20k chunks and a list of 200 real user queries with graded relevance judgments. 90 minutes on a single GPU.

**The tasks:**

1. **(15 min) Baseline.** Load a general open embedder. Embed all 20k chunks and all 200 queries. Compute exact (brute-force) nDCG@10 and recall@50. Write the numbers down. If you cannot produce a baseline in 15 minutes you will never win an argument about whether something helped.
2. **(20 min) Synthetic training data.** Sample 3,000 chunks stratified by cluster. Generate 3 queries per chunk with an LLM. Filter with round-trip retrieval: drop any query whose source chunk ranks 1 with a very large margin, and drop any whose source chunk isn't in the top 50. **Split by document ID** so no document appears in both the training pairs and the 200-query eval.
3. **(15 min) Hard negatives.** For each surviving synthetic query, retrieve top-50 with the baseline model, exclude the source chunk and anything from its parent document, sample 4 negatives from ranks 10–50.
4. **(20 min) Train.** `CachedMultipleNegativesRankingLoss`, batch 512 (GradCache), LR 2e-5, warmup 10%, 2 epochs, evaluate on the 200-query set every 25 steps, keep the best checkpoint by nDCG@10.
5. **(20 min) Prove it.** Re-run the baseline evaluation with the fine-tuned model. Report the delta **with a paired bootstrap 95% interval**. Also report a general-retrieval regression number on a fixed public slice. Write three sentences: what improved, whether it is significant, and what you gave up.

**Pass criteria — all four:**
- You produced a baseline number before you trained anything.
- Your train/eval split is by document, and you can state why splitting by pair would have been wrong.
- Your reported delta carries a confidence interval, and if that interval straddles zero you said "no result" rather than reporting the point estimate.
- You reported a general-retrieval number alongside the in-domain number.

**Fail conditions that look like success:** an eval set derived from the same synthetic generator as the training set; a 12-point improvement that turns out to be near-duplicate chunks spanning the split; reporting only the best of six runs.

The point of the drill is not the model. It is that **you can construct an experiment that is capable of telling you that you were wrong** — which is the specific thing take-home defenses grade for and the specific thing most candidates cannot demonstrate.

### Whiteboard exercise: 300M chunks, a 2 TB memory budget, p95 retrieval under 50 ms, multilingual. Choose the embedding stack and defend every number.

**🏋 Drill / 🎯 design round.** 25 minutes, whiteboard, no notes. This is a composite of the questions this section has been building toward and it is the shape a Glean, Notion, Harvey or Databricks retrieval round actually takes.

Here is the answer I would give, with the arithmetic exposed at every step — the arithmetic is the deliverable, not the model name.

**Step 1 — the storage constraint dictates the representation before you pick a model.** 300M chunks. At 1024 dims fp32: `300e6 × 1024 × 4` = **1.23 TB** of raw vectors, plus HNSW graph at `M=16` ≈ 128 B/vector = 38 GB, total ~1.27 TB. That fits in 2 TB, but with no headroom for growth, replicas or the OS page cache, and you will be paying for ~2 TB of RAM (roughly $16–20/hr of memory-optimized instances ≈ **$12–15k/month**). So fp32 is technically feasible and economically dumb.

**int8**: `300e6 × 1024 × 1` = **307 GB** + 38 GB graph = 345 GB. Fits on one large instance with room to spare, >99% of full-precision recall, no rescoring stage needed. **This is my answer.**

**binary + rescore**: `300e6 × 128 B` = **38 GB** hot, with the int8 or fp32 vectors on NVMe for rescoring. This is what I'd propose if the budget were 64 GB rather than 2 TB, or if I needed many tenant indexes resident simultaneously.

**Step 2 — model choice follows from "multilingual" plus "I want to fine-tune."** Multilingual is a hard gate, so: a multilingual-E5 / BGE-M3 / multilingual-Cohere-class model. Self-hosted, because I want the option to fine-tune and I need to pin the version for the lifetime of a 300M-vector index. 1024 dims, which is where these families live and which I chose above for the storage math. If BGE-M3, I take the sparse head as well and run hybrid, because a 300M-chunk enterprise corpus is guaranteed to contain identifiers that dense retrieval will miss.

**Step 3 — latency budget.** p95 < 50 ms total. Allocate: query embedding **5 ms** (small encoder, colocated, cached head), ANN search **20 ms** (int8 HNSW with `efSearch` tuned to hit ≥0.97 recall against exact — measure it, don't guess), reranking **20 ms** (cross-encoder over top 50, batched on GPU), **5 ms** slack. Sharding: 345 GB fits on one node so I do not shard for memory — but I would shard by 4 for *latency*, because HNSW search on 300M vectors in one graph is slower than on 4 × 75M in parallel, and I can fan out and merge. State that the merge is trivial because scores are comparable within one model version.

**Step 4 — index build cost.** 300M × 400 tokens = 120 Btok. At ~100k tok/s/H100: `120e9 / 1e5` = 1.2M GPU-seconds = **333 GPU-hours** = **$1,000** at $3/hr, or **~10.5 hours wall clock on 32 GPUs**. Say this number out loud, because it is the number that governs how often you can afford to change your mind about the model, and therefore how much evaluation you should do *before* you commit.

**Step 5 — what I'd measure before shipping.** Recall@50 and nDCG@10 on a 200-query stratified in-domain set, per language, weighted by traffic mix. ANN-vs-exact overlap at the chosen `efSearch`. Retrieved-document frequency histogram for hubs. And the shortlist-recall ceiling: what fraction of the exact top-10 survives into the top-50 that the reranker sees.

**Pass criterion for the drill:** every number above is derived on the board from `count × dims × bytes` or `FLOPs ÷ throughput`, you named the two hard gates (multilingual, fine-tunability) before discussing quality, and you stated at least one thing you would *measure* rather than assume. Candidates who name a model first and do arithmetic never lose this round on the model choice — they lose it on not having a budget.
