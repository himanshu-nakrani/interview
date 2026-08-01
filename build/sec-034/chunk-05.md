### Design the write path for a live index — upserts, soft deletes, and compaction. What are the invariants?

The write path is where most retrieval systems accumulate their unpayable debt, so I design it around three invariants and everything else follows.

**Invariant 1: chunk IDs are deterministic functions of content, not database sequences.** I use `chunk_id = hash(doc_id, chunk_index, content_hash)` or, more usefully, `chunk_id = f(doc_id, chunk_index)` with a separate stored `content_hash`. Deterministic IDs make the entire pipeline idempotent: reprocessing the same document twice produces the same IDs and the second write is a no-op upsert. You already know why this matters — it is the same reason you use an idempotency key on a payment — but here the consequence of getting it wrong is duplicate chunks, and duplicate chunks are *invisible* because the system returns plausible results. They just quietly fill your context window with five copies of the same paragraph.

**Invariant 2: the write path never re-embeds unchanged content.** Store the `content_hash` of each chunk alongside the vector. On reprocessing, compare hashes and skip. In a typical corpus a document edit changes 1–3 chunks out of 40, so hash-gating turns a 40-embedding update into a 3-embedding update — a **13× reduction in the dominant cost of your pipeline**. This single check is the highest-ROI line of code in an ingestion system and it is routinely absent.

**Invariant 3: a document's chunk set is replaced atomically, including shrinkage.** This is the subtle one. Document had 10 chunks, the author deleted three paragraphs, now it has 6. If you upsert chunks 0–5 and stop, chunks 6–9 remain in the index forever, containing deleted text, and they will be retrieved. The fix is a **generation stamp**: every document write carries an incrementing `doc_version`; you upsert all current chunks with the new version, then issue `DELETE WHERE doc_id = $1 AND doc_version < $2`. Same pattern as a garbage-collected write of a new file version followed by a sweep of the old — and it is correct under retries and out-of-order delivery, which naive "delete then insert" is not (a crash between the two leaves the document unretrievable).

**Soft deletes and compaction.** As established earlier, deletes in a graph index are tombstones. So the write path has a fourth component: a compaction policy. Concretely I want (a) a `deleted_at` marker so the tombstone is queryable and auditable, (b) a per-segment or per-shard tombstone-ratio metric, (c) a background compactor triggered on ratio not on a fixed schedule, and (d) — because compaction is expensive — a design where compaction rebuilds *one segment* rather than the whole index.

That last point is the architectural argument for **segment-based storage with background merge**, which is Lucene's model and which Elasticsearch, Qdrant, Milvus and Vespa all use in some form. Writes append to a small new segment; a merge policy periodically combines small segments into larger ones, dropping tombstoned entries during the merge. You get amortised compaction with bounded work per merge, exactly the way an LSM tree handles the same problem. If you build on a raw HNSW library, you are choosing to own this yourself, and my strong advice is do not — segment-and-merge is a solved problem and reimplementing it badly is a six-month tax.

**⚠ Trap:** treating "the vector store accepted my upsert" as "the read path will see it." Most vector stores are eventually consistent between write and searchable, with a flush or refresh interval measured in seconds to minutes. A user who edits a document and immediately searches for the edit gets the old version, and files a bug that your integration tests cannot reproduce because they sleep. Know your engine's refresh semantics, expose them in your API contract, and — if you need read-your-writes for a specific flow — serve that flow from the source of truth rather than the index.

### How do you measure index bloat, and what triggers compaction?

Bloat is the gap between the resources your index consumes and the resources its *live* data would consume. It has three components and you should measure all three, because they grow at different rates and the mitigation differs.

**Tombstone ratio** = `deleted_vectors / total_vectors_in_index`. The primary number. Emit per segment/shard, not just globally, because compaction operates per segment and a global average of 12% can hide one segment at 70%.

**Memory amplification** = `resident_bytes / (live_vectors × bytes_per_vector)`. This catches things the tombstone ratio does not: fragmentation in the allocator, per-segment overhead when you have too many small segments, ID-mapping tables that grew and never shrank.

**Traversal amplification** = `distance_computations_per_query / (efSearch × M)`. The theoretical minimum is roughly `efSearch × M`. If you are doing 3× that, you are walking through dead nodes and re-expanding regions. This is the metric that directly explains latency drift, and almost nobody has it. It requires the engine to expose a per-query distance-computation counter; if yours does not, ask the vendor for it, because without it you are inferring index health from latency, which conflates it with load.

**The trigger policy.** I do *not* schedule compaction by cron. Cron compaction runs when it does not need to (wasting IO and CPU on a healthy index) and does not run when it does (a bulk delete at 2 p.m. Tuesday). I trigger on the metric:

- Tombstone ratio > 20% on a segment → merge that segment.
- Memory amplification > 1.4 → investigate; usually too many small segments, so force-merge.
- Traversal amplification > 2.0 → the graph is degraded; this is the "rebuild, don't merge" signal.

And I put a rate limit on the compactor, because the failure mode of aggressive compaction is the noisy-neighbour write-contention case from earlier: your background optimizer eats the IO budget and every tenant's p99 doubles. Compaction should be a low-priority, throttled, preemptible background job — the same posture you already give to autovacuum and to LSM compaction.

**💰 Math — what bloat costs.** Take a 50M-vector index at d=1536, int8-quantized: live data is `50e6 × (1536 + 130) = 83 GB`. At 35% tombstones, resident is `83 / 0.65 = 128 GB` — you have provisioned 45 GB of RAM to store deleted documents, at ~$4/GB-month that is **$180/month of pure waste**, plus the recall degradation, plus the latency inflation. That is not a scary number on its own, which is exactly why it never gets fixed; the argument that wins is the recall one, so measure recall against a flat shadow index and show the line going down.

### Walk me through a zero-downtime reindex using versioned indexes and an alias swap.

This is a pattern you already own from relational schema migrations and blue-green deploys; the only new parts are that the "migration" takes days and that correctness is statistical rather than binary. The trigger is usually one of: embedding model upgrade, chunking-strategy change, dimension change, index-parameter change requiring a rebuild, or a corpus-wide reprocessing.

**The core invariant: readers never name a concrete index.** They name an alias — `chunks_current` — and the alias points at `chunks_v7`. Every vector store worth using supports either a real alias (Elasticsearch aliases, Qdrant collection aliases, Weaviate) or something you can emulate (a Postgres view, or a config-service value read per request with a short TTL). If your store supports neither, you emulate it in your retrieval client with a value fetched from a config source — but it must be a *runtime* lookup, not a deploy-time constant, or your "swap" is a deploy and your "rollback" is another deploy.

**The phases:**

**Phase 0 — freeze the contract.** Write down exactly what is changing and what the acceptance criteria are, as numbers, before you start. "nDCG@10 on the golden set must not regress by more than 1 point, and p95 retrieval latency must stay under 40 ms." If you have not written this down you will end up arguing about vibes at 11 p.m. on swap night.

**Phase 1 — dual write.** Start writing *both* indexes from the ingestion pipeline. New index `v8` gets the new embedding model / chunking / parameters; `v7` keeps taking the old format. Dual writes must be idempotent and must not fail the request if `v8` errors — `v8` is not yet serving, so its write path is best-effort with a DLQ. Run this from the moment you start the backfill so that live changes during the backfill are not lost. **This is the piece people skip and it is what makes their backfill produce a stale index.**

**Phase 2 — backfill.** Re-embed and load the historical corpus into `v8`, ordered newest-first so that the most-queried content is correct earliest. Track progress as a percentage and a watermark timestamp. Because you are dual-writing, a document that changes mid-backfill is written twice — once by the backfill with old content, once by the live path with new content — so **the backfill must not overwrite a newer live write**. Enforce with a version/timestamp comparison on write, or by having the backfill skip any doc whose `updated_at` is newer than the backfill start.

**Phase 3 — shadow reads.** Serve from `v7`, but asynchronously issue the same query against `v8` and log both result sets. Do not block the user; do not put it in the critical path. Collect for at least a full weekly cycle, because query distributions are weekly-seasonal. Details in the next question.

**Phase 4 — canary.** Route 1% of traffic to `v8` for real, with the quality metrics you actually track (answer acceptance, thumbs, citation-click rate, task success). Then 5%, 25%, 50%. This is where you catch the failures shadow reads cannot: latency under real concurrency, cache behaviour, and downstream effects of different chunk boundaries on the prompt.

**Phase 5 — swap the alias.** One atomic operation. Traffic moves.

**Phase 6 — hold `v7` hot for a full rollback window** — I use one week — dual-writing the whole time. Rollback is re-pointing the alias, which is seconds. Only then delete `v7` and stop the dual write.

**⚠ Trap:** deleting the old index the day after the swap because it is expensive. The failures you actually get from a reindex are *slow* failures — a specific query class that degraded, a customer segment whose documents chunk differently, a language that the new model handles worse. Those surface over days, from support tickets, not from your dashboards. Carrying two indexes for a week is the insurance premium, and you should price it into the project from the start: for a 100M-vector index that is roughly a week of double storage. On the binary-quantized configuration priced earlier at $533/month, one extra week is `533 / 4.3 = $124`. **You are arguing about $124 to protect a multi-week migration.** Say the number out loud and the argument ends.

### Price and schedule the re-embedding of 100 million chunks. Show me all the arithmetic.

This is the question I would use to separate people who have done a migration from people who have read about one, because the surprising part is that **money is not the constraint — throughput and rate limits are.**

**Corpus.** 100M chunks, averaging 500 tokens → `100e6 × 500 = 5e10 = 50 billion tokens = 50,000 Mtok`.

**Option A — hosted embedding API.** **📅 Volatile: rates move; re-derive.** At a small-model rate of $0.02/Mtok: `50,000 × 0.02 = $1,000`. At a large-model rate of $0.13/Mtok: `50,000 × 0.13 = $6,500`. So the *money* is between one and seven thousand dollars, which is nothing.

The constraint is the rate limit. At a 5M tokens/minute organisation limit: `5e10 / 5e6 = 10,000 minutes = 167 hours ≈ 7 days` of continuous saturation, assuming zero throttling and zero retries — realistically 9–10 days. **That is your schedule, and it is set by a quota, not by your engineering.** So the first phone call in this project is to your account team about a temporary limit increase, and the first line of your design doc is the rate-limit math.

**Option B — self-hosted encoder.** Take a ~110M-parameter encoder (BERT-base scale). Forward-pass FLOPs ≈ `2 × N` per token, so per 500-token chunk: `2 × 110e6 × 500 = 1.1e11 FLOP`. For 100M chunks: `1.1e19 FLOP = 11 EFLOP`. Attention adds roughly 8% at T=500 for this shape; ignore it for the estimate.

One H100 at ~300 TFLOP/s achieved (bf16, realistic MFU for a well-batched encoder): `1.1e19 / 3e14 = 36,700 seconds ≈ 10.2 hours`. That is `100e6 / 36,700 = 2,725 chunks/sec` on one GPU, which is the right order for a batched encoder served through TEI or Infinity.

Eight H100s: **1.3 hours**. At ~$3/GPU-hour: `8 × 1.3 × 3 = $31` of GPU. Add data movement and orchestration and call it **under $100 and under two hours.**

**Compare.** Self-hosting is roughly **10–65× cheaper and 100× faster in wall clock** than the API path for a bulk backfill. That inversion surprises people because per-request the API is convenient and cheap; in bulk, you are paying for someone else's rate limiter. Note the caveat that decides it: if you need a *specific proprietary* embedding model, Option B does not exist, and you are back to negotiating quota. And if your target is a 7B-parameter embedder, multiply Option B by ~64: `650 GPU-hours`, 81 hours on 8 GPUs, `650 × 3 = $1,950`. Still cheap, no longer instant.

**The actual long pole is neither.** Writing 100M vectors into the index is: `100e6 × 6144 = 614 GB` of vector payload, plus index build. If your store ingests 20,000 vectors/sec you need `100e6 / 20e3 = 5,000 s = 1.4 hours`; at 2,000/sec it is 14 hours. And HNSW build for 100M vectors, from the earlier arithmetic, is tens of hours single-node — so **shard the build across 10–20 nodes and merge, or build offline and load a prebuilt index**. Most managed stores support bulk import that is dramatically faster than the streaming upsert path; use it, and price it separately because it is often billed differently.

**💰 Total project cost, honestly stated:** embeddings ~$100 (self-hosted) to ~$6,500 (API); one week of duplicate storage ~$120–500 depending on configuration; sharded build compute maybe 200 node-hours ≈ $200; **and roughly three engineer-weeks of dual-write, shadow-read, canary and reconciliation work, which at a $300k loaded cost is `3/52 × 300,000 ≈ $17,300`.** The engineering is 60–95% of the cost. That is the number that should drive the decision about whether the new embedding model is worth it, and it is the number nobody puts in the doc.

**🗣 Say this in the room:** "Re-embedding 100M chunks is about 50 billion tokens. On a hosted API that's one to seven thousand dollars — trivial — but at a 5M-token-per-minute limit it's seven days of saturated quota, so the schedule is set by the rate limit, not the budget. Self-hosted on eight H100s it's about eleven exaFLOPs, roughly 1.3 hours and under a hundred dollars of GPU. The real long pole is index build and the three engineer-weeks of dual-write, shadow-read and rollback machinery."

### How do you actually compare the old and new index during shadow reads? What are you looking for?

Shadow reads are worthless if all you log is "both returned 10 results." The comparison has to be designed, and there are four levels of it, in increasing order of value and cost.

**Level 1 — overlap (free, weak).** For each query, compute Jaccard or top-k overlap between the two result sets. This tells you *how much changed*, not whether it got better. It is genuinely useful as a tripwire: an overlap of 0.95 when you expected a big improvement means your change did nothing; an overlap of 0.05 means something is broken (wrong opclass, wrong normalisation, dimension mismatch). **Overlap is a smoke test, not a quality metric**, and presenting it as a quality metric is a common junior mistake.

**Level 2 — the labelled golden set (the real answer).** You need a set of queries with human relevance judgements — I would want 300–500 queries drawn from the real production distribution, each with graded relevance on a pool of candidates. Then compute **nDCG@10 and recall@k for both indexes** and run a **paired statistical test**, because the same queries hit both systems and paired tests are far more powerful. With 400 paired queries you can detect roughly a 2-point nDCG difference at 80% power; with 50 you cannot detect anything and your "the new index is better" claim is noise. Build the golden set once; it pays for itself across every future retrieval change.

**Level 3 — LLM-as-judge on the diff, which is where the leverage is.** Judging 400 queries × 10 results × 2 systems by hand is expensive. But **you only need to judge where they disagree.** Compute the symmetric difference of the two result sets — typically 3–5 chunks per query — and have a judge model rate only those for relevance to the query. Now: `400 queries × 4 chunks × 2 systems = 3,200 judgements`. At ~1,500 tokens each and $3/Mtok input: `3,200 × 1500 / 1e6 × 3 = $14.40`. **Fourteen dollars to evaluate a migration.** Sample 10% of those for human review to validate the judge. This is the technique that makes continuous retrieval evaluation affordable and I would lead with it.

**Level 4 — end-to-end, offline.** Run the full RAG pipeline on both retrieval sets and compare final answer quality, because retrieval quality is instrumental, not terminal. Often the honest finding is that a retrieval improvement does not move answer quality, and that is a *result* — it tells you the bottleneck is elsewhere and saves you the migration.

**What you are looking for, specifically, beyond the aggregate:** (a) **per-segment regressions** — slice by tenant, language, document type, query length and query intent, because a 1-point aggregate gain frequently hides a 10-point loss on one segment that happens to be your most important customer; (b) **length and duplication shifts** — a chunking change alters how many tokens you stuff into the prompt, which changes cost and can change answer quality independently of relevance; (c) **result-count distribution** — is the new index returning fewer than k more often, which is the post-filter-collapse detector; (d) **latency at real concurrency**, which shadow reads *understate* because they run off the critical path and often at lower parallelism than production.

**⚠ Trap:** running shadow reads for two days and declaring victory. Query distributions are weekly-seasonal (weekday work queries versus weekend, month-end reporting, release cycles) and the queries that expose a regression are frequently the rare ones. Run for a full week minimum, and explicitly check the **tail of the query distribution**, not just the head — the head is easy for both systems and tells you nothing.

### Design CDC-driven ingestion from Postgres into your vector index.

This is a stream-processing problem you already know how to solve; the parts that are new are that one of the "transformations" costs money and has a rate limit, and that the sink is eventually consistent. I would draw it as five stages.

**1. Capture.** Postgres logical replication (a publication plus a replication slot), read by Debezium or by your own `pgoutput` consumer, producing change events onto Kafka. Key the topic by `doc_id` so all changes to one document land on one partition and are therefore **ordered**. This matters more than usual: out-of-order processing of an update and a delete for the same document leaves a resurrected deleted document, which is the incident from the previous chunk.

The operational caveat you must volunteer: **a replication slot that falls behind pins WAL and will fill the primary's disk.** So `pg_replication_slots.confirmed_flush_lsn` lag is a page-worthy alert, and you need a documented decision for what happens when the consumer is down for hours — drop the slot and do a full reconciliation, or accept the WAL growth. This is the single most common way a CDC pipeline takes down the database it is reading.

**2. Decide.** The consumer computes chunking and content hashes and compares against the current state. Three outcomes: unchanged (drop the event — and this will be the majority, because CDC fires on every column update including `last_viewed_at`), changed (proceed), deleted (emit tombstones for all chunks of that doc). **The hash comparison is the cost control**; without it a bot that touches a `view_count` column re-embeds your corpus.

**3. Embed.** Batch aggressively — embedding throughput is dominated by batch size, and single-document embedding wastes 90% of the GPU or of your API quota. I buffer by time-or-size (e.g. flush at 256 chunks or 500 ms) and I make the embedding stage the natural place for **backpressure**: when the embedding provider throttles, the consumer stops committing offsets, Kafka absorbs the backlog, and lag rises. That is correct behaviour and it is exactly the Celery/Kafka intuition you already have — the only new part is that the retry budget is expensive, so retries must be capped and failures dead-lettered rather than retried forever.

**4. Write.** Upsert with the deterministic IDs and the generation-stamp delete described earlier. Batch the writes; most vector stores are dramatically faster at batches of 100–1,000 than at singles.

**5. Reconcile.** A periodic job that diffs source IDs against index IDs and repairs both directions (missing documents, orphaned documents). **CDC pipelines drift. All of them.** Slots get dropped, consumers get reset, DLQs get ignored, a schema migration breaks a deserializer for six hours. The reconciliation job is not optional and its `orphaned` and `missing` counts are alerts, not dashboard decorations.

**The metrics I put on the dashboard:** consumer lag (in events and in seconds), replication slot lag in bytes, hash-skip rate (should be high — if it drops, something is touching rows unnecessarily), embedding cost per hour, DLQ depth, index-write error rate, and end-to-end freshness measured by the synthetic probe described earlier. **Freshness measured by probe, not by lag**, because lag being zero while a downstream flush is stuck tells you nothing.

**⚠ Trap:** treating the vector index as a Kafka sink with at-least-once semantics and calling it done. At-least-once plus deterministic IDs plus idempotent upserts genuinely does give you correctness for creates and updates. It does **not** give you correctness for deletes, because a redelivered delete is fine but a *lost* delete is permanent — and deletes are the events most likely to be lost, since a soft delete upstream may not even produce a change event your filter recognises. Deletes need the reconciliation backstop; creates mostly do not.

### How do you handle TTL and freshness without wrecking recall?

Freshness is a ranking signal, not a retrieval filter — that is the mental model, and getting it backwards is the common error. If you bake recency into the ANN search you corrupt the geometry; if you apply it after retrieval you have a clean, tunable, measurable knob.

**Three mechanisms, in increasing order of aggression:**

**Hard TTL / date filter.** `WHERE created_at > now() - interval '2 years'`. This is a filter, with all the pre/post/filtered-traversal consequences from earlier, so implement it as a filter properly — and be aware that a date range is often a *low*-selectivity filter (most of the corpus is recent) which is the friendly case. Use this when stale content is *wrong*, not merely less useful: expired promotions, superseded policies, deprecated API docs.

**Recency decay applied at rerank.** After retrieval, before the model, adjust: `final = relevance_score × exp(-age_days / τ)` or `final = relevance + w × recency_score`. Pick `τ` from your content's actual half-life — news is hours, engineering docs are quarters, legal precedent is decades.

**Freshness as a reranker feature.** If you have a cross-encoder or a learned ranker in the pipeline, age is just another feature and the model learns the weight from data instead of you guessing it. This is strictly better if you have the labelled data to train on.

**⚠ Trap — the one I see most:** applying multiplicative decay directly to cosine similarity. Cosine scores from a modern embedding model live in a **narrow band**, typically 0.6–0.9 for anything plausibly relevant. The *difference* between a great match and a mediocre one might be 0.05. Multiply by a decay factor that ranges over `[0.1, 1.0]` and the recency term dominates relevance by an order of magnitude: a barely-relevant document from yesterday beats a perfect match from last year. Work the numbers: perfect old match `0.90 × 0.30 = 0.27`; weak new match `0.68 × 1.00 = 0.68`. The weak one wins by 2.5×. **You have built a recency sorter with a relevance tiebreak, which is not what you designed and not what you told anyone you built.**

The fix: normalise relevance to a comparable scale *first* — min-max or z-score within the candidate set, or better, use reciprocal-rank-style fusion where you combine *ranks* rather than raw scores, so the incommensurability problem disappears. `score = 1/(60 + rank_relevance) + w × 1/(60 + rank_recency)` is crude and works far better than naive multiplication.

**TTL implementation.** Two options: a background sweeper that periodically deletes expired vectors (simple, predictable, adds to your tombstone load — schedule it to run just before compaction so the merge reclaims in one pass), or lazy expiry where the query filters on the expiry timestamp and the sweeper is slower (returns correct results immediately, but expired vectors still occupy memory and still get traversed). **I take lazy expiry for correctness plus a sweeper for hygiene** — correctness at query time, cost reclaimed in the background.

**The measurement that makes this defensible:** add a freshness dimension to your golden set. Tag each query with whether the correct answer is time-sensitive, and report retrieval quality separately for time-sensitive and evergreen queries. Without that split, tuning `τ` improves one at the expense of the other and your aggregate metric stays flat while both user populations get worse.

### Build me the capacity and cost model for one billion vectors. Take your time.

I will lay this out the way I would on a whiteboard, because the structure is the answer.

**Step 1 — the invariant.** `1e9 vectors × 1536 dims`. At float32: `1e9 × 6144 = 6.14 TB`. At float16/`halfvec`: 3.07 TB. At int8: 1.54 TB. At 1-bit: 192 GB. HNSW graph at M=16: `1e9 × 130 = 130 GB` regardless of vector precision. Those five numbers are the whole model; everything else is which storage medium holds which of them.

**Step 2 — the configurations, with monthly cost. 📅 Volatile: instance prices; re-derive from the current rate card.**

*(a) In-memory HNSW, float32.* Needs 6.27 TB RAM. Seven 1 TB-RAM instances at ~$8/hr: `7 × 8 × 730 = $40,880/month`. Latency ~1–2 ms, recall ~0.98. This is the "money is no object" configuration and I would never approve it.

*(b) In-memory HNSW, int8 scalar quantization.* 1.54 TB + 0.13 TB = 1.67 TB. Two 1 TB instances: `2 × 8 × 730 = $11,680/month`. Recall loss typically under a point. **A 3.5× saving for essentially free** — int8 SQ is the default I would start from, not an optimisation.

*(c) Binary quantization in RAM, float rescore from NVMe.* 192 GB codes + 130 GB graph = 322 GB RAM, 6.14 TB on local NVMe. One instance with ~512 GB RAM and 15 TB NVMe at ~$5.50/hr: `5.50 × 730 = $4,015/month`, ×2 for HA = **$8,030/month**. Recall ~0.93–0.96 with over-fetch 200.

*(d) DiskANN on NVMe.* 192 GB PQ codes in RAM, graph + full vectors on SSD. An instance with 256 GB RAM and 7.5 TB NVMe at ~$2.75/hr: `2.75 × 730 = $2,008`, ×2 for HA = **$4,016/month**. Latency 2–5 ms p50, 10 ms p99. Recall ~0.92–0.96.

*(e) Object-storage tiered.* 6.14 TB on S3 at $0.023/GB-month = `6,140 × 0.023 = $141/month` for storage, plus cache/query nodes — say two 128 GB instances at ~$0.86/hr = `2 × 0.86 × 730 = $1,255` — plus request charges. **≈ $1,500/month.** Cold-query latency 100–500 ms; hot-tier latency a few ms. Only viable if your access distribution is skewed, which you must verify.

**Step 3 — the spread.** `$40,880` down to `$1,500` is **27×**, for the same billion vectors, and the recall spread across those configurations is roughly 6 points. **The architecture decision is worth 27× and the index-parameter decisions are worth maybe 20%.** That ratio is the reason this question is asked.

**Step 4 — the costs people forget.**
- **Ingest/backfill:** 1B chunks × 500 tokens = 500 Gtok. Self-hosted on the earlier arithmetic: `1.1e20 FLOP`, ~102 GPU-hours on H100s ≈ `102 × 3 = $306` — one-time, but you will do it every time you change embedding models. On an API at $0.02/Mtok it is `500,000 Mtok × 0.02 = $10,000` and, at 5M tokens/min, **70 days** — which is not a schedule, it is a reason to self-host.
- **Index build:** 1B-vector HNSW builds are hundreds of node-hours. Sharded across 50 nodes, call it $500–1,500 per full build.
- **Replication:** everything above ×2 for HA, and ×3 if you want multi-AZ read capacity. I priced HA into (c) and (d) and not into (a), (b) and (e); be explicit about which you are quoting, because "our vector store costs $2k/month" and "our vector store costs $4k/month" is usually just this.
- **Duplicate index during migration:** budget 2× for one week per reindex.
- **Egress**, if your query path crosses a region or a VPC boundary. At 500 KB read per query and 1,000 QPS that is `500e3 × 1000 × 86400 = 43 TB/day`; at $0.02/GB cross-AZ that is **$860/day**. Keep it in-AZ.

**Step 5 — sharding.** At 1B you are sharding regardless of configuration. Shard **randomly**, not semantically: query all shards, merge top-k. Random sharding gives balanced load and zero recall loss. Semantic sharding (by topic, by tenant, by cluster) is tempting because it promises you only query one shard, and it fails because query-to-shard routing is itself an ANN problem with its own recall, and because it produces hotspots. The exception is genuine tenant isolation, where "one shard" is a correctness requirement rather than an optimisation.

**🗣 Say this in the room:** "A billion 1536-dim vectors is 6.1 TB at fp32, 1.5 TB at int8, 192 GB at one bit, plus about 130 GB of HNSW graph — those five numbers drive everything. In-memory fp32 is roughly forty thousand a month; DiskANN on NVMe is about four; object-storage-tiered with a hot cache is about fifteen hundred. That's 27× for six points of recall, so the storage-tier decision dominates every parameter I could tune, and I'd make it by measuring whether my access distribution is skewed enough for tiering to work."

### Recall dropped from 0.94 to 0.71 overnight. No deploy went out. Debug it.

"No deploy" is the interesting constraint, because it eliminates the first thing everyone checks and forces a real differential diagnosis. I would work it as an ordered decision procedure — cheapest and most likely first — and I would say out loud that the first move is to determine **whether the index changed or the queries changed**, because those are two entirely different investigations.

**Step 0 — is the measurement real?** Check the recall harness itself. Did the ground-truth flat index get rebuilt? Did the golden query set change? Did someone re-embed the *evaluation* queries with a different model than the index? A drop of exactly this shape — large, sudden, no deploy — is frequently the measurement, and it takes two minutes to rule out. I have been burned by not checking this first.

**Step 1 — did the query distribution shift?** Compare today's query embeddings against last week's: mean pairwise similarity, cluster assignments, the distribution of query lengths, the fraction of queries carrying each filter. A marketing campaign, a new customer onboarding, a bot, or a new product surface can change the query mix overnight. If the *new* queries have low recall and the old ones are fine, the index is healthy and your problem is that a new query class is out of distribution — probably a filter class you have never post-filtered on before.

**Step 2 — did filter selectivity change?** This is my leading hypothesis for a sudden, large, deploy-free drop. A new tenant onboarded, an ACL rollout narrowed everyone's visible set, a feature flag turned on a `document_type` filter, a customer archived 90% of their corpus. Any of these can take a filter from 40% selectivity to 0.5% with no code change, and post-filter collapse does exactly this — a large, sudden, uniform-looking recall drop. **Check the joint distribution of (filter predicate, result count).** If `len(results) < k` got more common, you have found it.

**Step 3 — did the corpus change?** A bulk ingest (recall degrades if the index is now much larger than its IVF centroids were trained for), a bulk delete (tombstone ratio spike), or a bulk update (which in an HNSW is a bulk insert). Check corpus size, tombstone ratio, and segment count over the last 48 hours. Segment count is the sneaky one: a burst of writes creates many small segments, each with its own graph, and per-segment ANN search plus merge can lose recall relative to one well-built graph.

**Step 4 — did anything in the index's environment change?** No deploy of *your* code does not mean no change. A managed vector store upgraded its version. An autoscaler added replicas and the new ones are serving cold or from an older snapshot — check whether recall differs *by replica*, which is a five-minute check that finds this instantly. A node was replaced and its index was rebuilt with default parameters instead of yours. Memory pressure caused the engine to spill or to reduce cache, silently changing behaviour.

**Step 5 — embedding model drift.** If you call a hosted embedding API, the model behind a version alias can change. Deterministic check: keep 100 fixed reference strings and their embeddings from index-build time; re-embed them today and compute cosine against the stored vectors. **If that is not ~1.0, your query embeddings are no longer in the same space as your index**, and recall against your index will collapse in exactly this way. This check costs cents per day and I would put it in every production retrieval system. It is also the argument for pinning explicit model versions rather than aliases.

**Step 6 — only now, index parameters.** Did `efSearch` change? Is something overriding it per request? Did a client library upgrade change a default?

**🔍 The compressed version I would say out loud:** "Is the metric real → did queries change → did filter selectivity change → did the corpus change → did the environment change → did the embedding model change → did the parameters change. Four of those six happen without a deploy, and the two I'd bet on are filter selectivity and a replica serving a different index than I think it is."

### How would you benchmark a vector database candidate honestly, before committing to it?

Published benchmarks are near-useless for this decision and I want to say why precisely, because the reasons tell you what to build instead. They run on **static** corpora (no writes, no deletes, no compaction), **unfiltered** queries, **single-tenant**, with the vendor tuning their own system and nobody tuning the comparison, on **public datasets** (SIFT, GIST, GloVe, DEEP) whose geometry does not resemble modern text embeddings. Every one of those differences is load-bearing.

So I build a harness. It takes about a week and it is reusable forever, which is why I would push to build it before the first evaluation rather than during.

**Use your own data.** Ten million of your own chunks, embedded with the model you will actually ship, and a query set sampled from your production logs including the tail. If you have no production yet, generate queries from your corpus with an LLM and hand-check a sample — imperfect, but far better than SIFT.

**Reproduce the filter distribution.** For each sampled query, attach the filter it would carry in production, so your benchmark sees the real selectivity histogram, not one convenient value. This alone eliminates half the candidates.

**Benchmark under write load.** Run the query benchmark while sustaining your production write and delete rate. Then run it again after 24 hours of that load. **The number that matters is not day-zero performance; it is day-thirty performance**, and the gap between them is entirely about compaction quality. Most published benchmarks measure day zero.

**Sweep the recall/latency curve, do not report a point.** For each system, sweep the runtime knob and produce (recall@10, p50, p95, p99) at each setting. Compare systems **at equal recall**, not at default settings. Reporting "system A did 5,000 QPS and system B did 3,000 QPS" without stating their recall is the most common dishonest comparison in this space, and calling it out is itself a good interview signal.

**Measure the tail and the resources.** p99 and p99.9, not just p50 — the tail is where compaction, GC and cache misses live. And record memory, disk and CPU at each operating point, so you can convert the curve into dollars.

**Test the operations, not just the queries.** Time a full rebuild. Kill a node and time the recovery. Delete 30% of the corpus and re-measure recall and memory. Restore from a backup. Do a rolling upgrade. **These are the things that determine whether you are happy in eighteen months**, and they are absent from every vendor benchmark.

**🏋 Drill:** 90 minutes. Take 1M of your own embeddings and a 200-query set with ground truth from flat search. Build the same data in two systems. Produce one chart: recall@10 on the y-axis, p95 latency on the x-axis, one curve per system, swept over the runtime knob, with memory footprint annotated at three points on each curve. Pass criterion: the chart is readable by someone who has never used either system, and you can state the crossover point in one sentence.

**⚠ Trap:** benchmarking with `k=10` and shipping with `k=100`, or benchmarking with 100-token queries and shipping with 8-token ones. Both `k` and query length materially change the shape of the curve — larger `k` requires larger `efSearch` and flattens the advantage of graph methods; short queries have different embedding geometry and often lower recall. Benchmark the configuration you will ship, including the boring parameters.

### What is on the dashboard and in the runbook for a production vector index? Assume I'm the on-call.

If I hand you the pager, here is what I want you to be able to see and do. I organise it as four dashboards and four runbook entries, because that is how it gets used at 3 a.m.

**Dashboard 1 — quality (the one nobody builds).** Daily ANN recall@10 against a flat shadow index over 1,000 sampled production queries, with a confidence band. Retrieval nDCG@10 against the golden set, run on every index change. The **distribution of `len(retrieved_chunks)`** — the post-filter-collapse detector. The distribution of top-1 and top-10 similarity scores. The reference-embedding drift check (100 fixed strings, cosine against their stored vectors, alerts if below 0.999). **Without this dashboard you cannot detect the failures that matter, because none of them produce errors.**

**Dashboard 2 — latency and load.** p50/p95/p99/p99.9 for the vector search stage specifically, separated from embedding and reranking. **Sliced by tenant** — the noisy-neighbour and cold-tier failures are invisible in the aggregate. QPS, `efSearch` distribution (someone will override it), and distance-computations-per-query if your engine exposes it.

**Dashboard 3 — index health.** Vector count (live and tombstoned), tombstone ratio per segment, segment count, memory amplification, index size versus available RAM, time since last compaction, and time since last full rebuild.

**Dashboard 4 — freshness.** The synthetic probe's three histograms — grant latency, revocation latency, deletion latency, per source system. CDC consumer lag and replication-slot lag. DLQ depth. Reconciliation job's `orphaned` and `missing` counts, both of which alert on any non-zero value.

**Runbook 1 — "recall dropped."** The six-step differential diagnosis from earlier, as a numbered list with the exact queries to run. First action: check per-replica recall, because a divergent replica is both the most common cause and the fastest fix (drain it).

**Runbook 2 — "latency spiked."** Check compaction/merge activity first (it is usually a background merge), then per-tenant slicing (a whale arrived), then cache hit rate (something cold), then `efSearch` overrides. Mitigation ladder: throttle the compactor, rate-limit the offending tenant on cost units, lower `efSearch` globally as a load-shed (**this is your circuit breaker — it trades recall for latency and it should be a runtime config value, not a deploy**), add replicas.

**Runbook 3 — "we're serving deleted or unauthorised content."** This is a sev-1. Immediate mitigation: enable or tighten the late-binding verification and flush the semantic cache. Then work the four layers — source, pipeline, index, downstream caches and derived structures. Then run reconciliation. Then write the incident up, because this one has legal exposure and the timeline matters.

**Runbook 4 — "reindex/rollback."** The alias swap procedure and, critically, the **rollback**: repoint the alias, confirm within 60 seconds, and keep the old index dual-written for a week. The single most important line in this runbook is that rollback is an alias write, not a deploy — and if that is not true in your system, fix it before you need it.

**🗣 Say this in the room:** "The thing I'd insist on that most teams don't have is a daily recall measurement against a flat shadow index, plus the distribution of how many chunks each query actually returned. Every serious retrieval failure I've dealt with — tombstone rot, centroid drift, post-filter collapse, an embedding model silently changing under a version alias — is silent: no errors, no latency change, just quietly worse answers. If you can't see those two charts, you're not operating the index, you're hoping."

**🏋 Drill:** 15 minutes, no notes. Write down the five metrics whose alert you would *page* on for a production vector index — not warn, page — and for each, the exact first mitigation. Pass criterion: at least two of your five are quality metrics rather than latency or availability metrics, and your reindex rollback is an alias write measured in seconds.
