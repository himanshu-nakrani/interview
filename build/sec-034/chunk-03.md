### Defend the claim "use Postgres until you can measure why you cannot." I want the argument, not the slogan.

The argument is not that pgvector is the best ANN implementation. It is not. The argument is that **a vector index is a derived structure over data you already own transactionally, and every dedicated vector database forces you to give up that transactionality in exchange for index quality you probably cannot yet measure a need for.**

Think about what a separate vector store actually is in your architecture: a second copy of your data, in a second system, with a second consistency model, updated by a pipeline you now have to build, monitor, backfill, and reconcile. You have taken a `JOIN` and turned it into a distributed transaction. Concretely, here is what you lose the day you move out of Postgres:

**Atomicity between content and vector.** In Postgres, `UPDATE documents SET body = $1, embedding = $2 WHERE id = $3` is one transaction. Either both change or neither does. Split them and there is now a window — usually seconds, sometimes hours if the pipeline is queued — where the vector index says one thing and the source of truth says another. Every "why did the model quote an old version of this doc" incident lives in that window.

**Deletes.** `DELETE FROM documents WHERE id = $1` removes the row and the index entry in the same transaction, and `ON DELETE CASCADE` handles the chunks. In a split architecture, a delete that succeeds in Postgres and fails in the vector store leaves you retrieving and quoting a document that legally no longer exists. That is a compliance incident, not a bug.

**Filtering and joins.** Your real query is almost never "nearest neighbours." It is "nearest neighbours among documents this user can see, in these workspaces, not archived, updated in the last year, and join me the title and the author's display name." In Postgres that is one query with a planner that knows the cardinalities. Split, it becomes: fetch 200 IDs from the vector store, `WHERE id = ANY($1)` in Postgres, discover that only 6 survive the filters, go back and fetch 2,000. You have hand-rolled a query planner, badly.

**Operational surface.** Backups, PITR, replication, failover, connection pooling, schema migrations, access control, an audit log — all of that exists and is battle-tested for your Postgres. For the vector store it is new work.

**🗣 Say this in the room:** "A vector index is a derived structure over data I already own transactionally. Moving it out of Postgres converts an atomic write into a distributed transaction and a `JOIN` into a hand-rolled two-phase fetch — and buys me index quality I usually can't yet prove I need. So my default is pgvector, with an explicit list of tripwires that would move me off it: index size past what fits in RAM alongside my OLTP working set, write rate that makes autovacuum the bottleneck, or filtered-search quality I can measure pgvector losing. I move when a tripwire fires, not before."

The version of this answer that fails is the dogmatic one in either direction. "Always Postgres" ignores that filtered HNSW in Qdrant genuinely is better than what pgvector does under a selective predicate. "Always a real vector DB" ignores that most companies asking the question have four million chunks, which is 25 GB, which is a laptop.

### Walk me through what pgvector actually does. How does its HNSW index interact with MVCC and vacuum?

pgvector is a Postgres extension providing vector types and index access methods. The types: `vector` (float32), `halfvec` (float16, half the storage at typically negligible recall cost for text embeddings), `bit` (for binary quantization with Hamming distance), and `sparsevec`. The distance operators are `<->` (L2), `<=>` (cosine), `<#>` (negative inner product), with `<~>` for Hamming on bit vectors. Two index types: `ivfflat` and `hnsw`.

```sql
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 128);
SET hnsw.ef_search = 100;              -- session-scoped runtime knob
SELECT id, body FROM chunks
 WHERE workspace_id = $1
 ORDER BY embedding <=> $2
 LIMIT 10;
```

Two operational details on build. First, set `maintenance_work_mem` large enough that the whole graph fits — pgvector builds in memory if it can and falls back to a much slower on-disk path if it cannot, and the difference is often an order of magnitude in build time. Second, `max_parallel_maintenance_workers` controls build parallelism, and its default (typically 2) is far below what your box can do. Both of those turn a "pgvector HNSW builds are unusably slow" complaint into a non-issue, and both are the first thing I check when someone reports that.

**The MVCC story is the interesting part**, because it is where a vector index is structurally different from a B-tree. In Postgres, an `UPDATE` is a delete plus an insert: a new heap tuple is written and the old one becomes dead. Every index on that table gets a new entry pointing at the new tuple, and the old entry stays until vacuum reclaims it. For a B-tree that is cheap. For HNSW, **an update to any column of the row inserts a whole new node into the graph** — with a full `ef_construction` search and neighbour-list repair — even if you did not touch the embedding.

Two consequences you should state without prompting. **One: HOT updates do not save you here.** Heap-only-tuple optimisation avoids index writes when no indexed column changed *and* there is room on the page — but vector rows are large (a d=1536 `vector` is ~6 KB and will be TOASTed), so page-level fill factor behaviour is unusual and HOT is unreliable. Assume every update to a row with a vector index costs an index insert. **Two: this makes update-heavy vector tables an anti-pattern.** The rule I enforce is to put embeddings in their own narrow table — `(chunk_id, embedding)` — so that updates to the document's metadata, status flags, view counts, or ACLs do not touch the vector index at all. That one schema decision eliminates most pgvector performance complaints I have seen.

On vacuum: pgvector's HNSW implements a vacuum path that unlinks dead entries from the graph and repairs neighbour lists, which is genuinely more than a bare HNSW library gives you. But it is still a full index scan, it is driven by autovacuum thresholds tuned for B-trees, and on a large vector index it is slow. Tune `autovacuum_vacuum_scale_factor` down for that table specifically (`ALTER TABLE chunks SET (autovacuum_vacuum_scale_factor = 0.02)`) so it runs often and briefly instead of rarely and catastrophically — the same reasoning you already apply to a high-churn OLTP table, just with a much more expensive index.

**⚠ Trap:** the dimension limit. pgvector's HNSW index supports `vector` columns up to 2000 dimensions. A 3072-dimensional embedding **cannot be indexed as `vector`** — the column is fine, the index build errors. The fixes are to cast to `halfvec` (which supports up to 4000 dimensions for HNSW) or to truncate with Matryoshka to 1024/1536. People discover this at the `CREATE INDEX` step after they have already loaded 40M rows.

### What is pgvectorscale and when does it change the Postgres calculus?

pgvectorscale is a separate extension (from Timescale, written in Rust) that adds a **StreamingDiskANN** index alongside pgvector's HNSW, plus a binary-quantization scheme. It is the answer to "pgvector is fine until my index exceeds RAM."

The three things it changes:

**DiskANN-style disk residency.** The index is designed to live on SSD with a compressed in-memory representation driving traversal, which is exactly the DiskANN split described earlier — but running inside Postgres, so you keep the transactional story. This is the load-bearing feature: pgvector's HNSW wants the vectors in shared buffers or the OS page cache, and once your index exceeds RAM, random-access graph traversal against a cold cache produces latency that is not a little worse, it is 50× worse. StreamingDiskANN is built for that regime instead of degrading into it.

**Statistical binary quantization.** A binarization scheme with rescoring, giving the 32×-storage / small-recall-loss trade discussed earlier without leaving Postgres.

**Streaming results — the "filtered search" fix.** This is the one that matters most architecturally and it is worth understanding precisely. The classic Postgres-vector problem is: `ORDER BY embedding <=> $1 LIMIT 10` with an extra `WHERE` clause. An index scan that returns a fixed candidate set and *then* applies the filter will under-return when the filter is selective. A "streaming" index can be asked for more results incrementally until the filter has been satisfied — the planner pulls tuples until it has 10 that pass. pgvector addressed the same problem from the other side in 0.8.0 with **iterative index scans** (`SET hnsw.iterative_scan = strict_order;` or `relaxed_order`, bounded by `hnsw.max_scan_tuples`), which is the feature to reach for if you would rather not add a second extension.

**My decision rule.** Stay on plain pgvector while the index fits comfortably in RAM and your filters are not brutally selective. Add pgvectorscale when either (a) index size crosses ~50–70% of available RAM, or (b) you have measured filtered-query recall collapsing and iterative scans are too slow. Leave Postgres entirely when you need per-tenant physical isolation at thousands of tenants, or a write rate where autovacuum on the vector table cannot keep up, or hybrid lexical+vector ranking sophisticated enough that you want a real ranking engine.

**📅 Volatile:** extension feature sets here move fast — pgvector's iterative scans, halfvec, sparsevec and binary support all landed across a handful of releases, and pgvectorscale's parameter names and quantization defaults have changed. Check the current docs before quoting a parameter name in an interview; quote the mechanism confidently and the parameter name with a hedge.

### My pgvector query has a WHERE clause and it's suddenly doing a sequential scan. Debug it.

This is the most common pgvector production bug and it has a specific structure, so I debug it in a fixed order.

**First, confirm what the planner is actually doing.** `EXPLAIN (ANALYZE, BUFFERS)` on the real query with real parameter values. You are looking for whether the node is `Index Scan using chunks_embedding_idx` or `Seq Scan on chunks` followed by a `Sort`. If it is the latter, the planner decided the index was not worth using — and it is often right, which is the part people miss.

**Second, check whether the ORDER BY can use the index at all.** An HNSW index is only usable for `ORDER BY column <op> constant LIMIT n` where `<op>` matches the operator class the index was built with. Three ways this silently breaks: (1) you built with `vector_l2_ops` and you query with `<=>` (cosine) — different operator class, index unusable, no warning; (2) there is no `LIMIT`, so the planner must produce a full sorted set and an approximate index cannot help; (3) the query vector is not a constant from the planner's perspective — it is the result of a subquery or a function call — so the index scan cannot be planned. Fix (1) by matching opclass to operator, (2) by always having a LIMIT, (3) by passing the vector as a bound parameter.

**Third — and this is usually it — the filter changed the cost estimate.** With `WHERE workspace_id = $1`, Postgres has two plans available: scan the HNSW index in distance order and filter each candidate (`Index Scan` with a `Filter`), or use the `workspace_id` B-tree to get the matching rows and sort them by distance exactly (`Bitmap Heap Scan` + `Sort`). The planner picks based on estimated selectivity. If `workspace_id = $1` is estimated at 200 rows, the exact plan is *correct and better* — 200 exact distance computations is nothing. If it is estimated at 200 rows but actually matches 2,000,000, you get a catastrophic plan. So: compare `rows=` estimated against `actual rows=` in the EXPLAIN ANALYZE output. If they diverge by more than ~10×, your statistics are the bug — `ANALYZE chunks`, raise `default_statistics_target` on that column, or add an extended statistics object if the filter involves correlated columns.

**Fourth, if the plan is the index scan and it is still slow or under-returning:** you have hit the filtered-search problem head on. The HNSW scan walks the graph in distance order and discards non-matching rows, so if only 1 in 500 rows matches, it must walk 5,000 graph nodes to yield 10 results — or give up early and return 3. This is what `hnsw.iterative_scan` exists for; set it to `relaxed_order` (faster, results not strictly distance-ordered) or `strict_order`, and raise `hnsw.max_scan_tuples` from its default. If that is still too slow, the answer is **partial indexes**: `CREATE INDEX ... ON chunks USING hnsw (embedding vector_cosine_ops) WHERE workspace_id = 42` for your handful of enormous tenants, or **table partitioning by tenant** so each partition has its own small index. Partitioning by tenant is the pgvector equivalent of namespace-per-tenant and it works well up to a few thousand partitions.

**⚠ Trap:** benchmarking with a filter that matches everything, then shipping to production where it matches 0.2%. The recall and latency curves are entirely different regimes. Your load test must reproduce the *selectivity distribution* of production filters, not just the query shape. I have watched a team ship a p95 of 8 ms from staging and see 900 ms in production purely because staging had one tenant.

### Give me the tripwires. When does Postgres genuinely stop being the right home for your vectors?

I want these written down before you start, so the migration is a planned event triggered by a measurement rather than an argument someone wins in a meeting. Five tripwires, each with a number:

**1. Index size crosses ~60% of RAM available to Postgres.** Compute it: `N × (d × bytes_per_elem + ~130 graph bytes)`. At d=1536 float32 that is `6274 bytes/vector`, so 10M vectors is 63 GB. On a 128 GB instance shared with your OLTP working set, you are already in trouble at 10M. The symptom before the tripwire fires is a rising `shared_buffers` miss rate on the index relation and p99 latency that is bimodal — fast when warm, 50× slower when a query lands on a cold region. If you see bimodal vector latency, you are paging.

**2. Write rate where autovacuum on the vector table cannot keep up.** Measure `n_dead_tup` on the chunks table over time in `pg_stat_user_tables`. If it is monotonically rising across days, vacuum is losing. HNSW vacuum is expensive, so the threshold is much lower than your B-tree intuition — I get nervous above a few hundred vector row-updates per second sustained.

**3. Filtered-search recall you can measure losing.** Build the flat ground truth for your production filter distribution and measure recall@10 *with the filter applied*. If iterative scans plus partial indexes cannot keep you above your threshold at acceptable latency, an engine with native filtered graph traversal is a real capability you do not have.

**4. Tenant count where partitioning stops working.** Postgres handles a few thousand partitions; past roughly ten thousand, planning time itself becomes the bottleneck and `pg_class` bloat is real. If you have 100k tenants who each need physical isolation, you want an engine designed for namespace-per-tenant.

**5. Ranking sophistication.** The moment you want true hybrid ranking — BM25 and dense fused, with per-field boosts, a learned second-phase ranker, and tensor-valued late interaction — you want Vespa or an ES/OpenSearch stack. Postgres full-text search is real but it is not a ranking engine, and bolting RRF onto `ts_rank` in SQL is a project that always looks smaller than it is.

Notice what is *not* on this list: "we have a lot of vectors." Ten million vectors is not a reason. Neither is "vector databases are what serious teams use."

### Walk me through the vector database landscape and tell me what you'd actually pick on.

I do not evaluate these on QPS benchmarks, because published benchmarks are run on static, unfiltered, single-tenant workloads that resemble nobody's production. I evaluate on seven axes, and I would write them on the board in this order:

**1. Filtered-search quality.** Does it do native filtered graph traversal, or does it pre-filter to a scan / post-filter and hope? This is the axis that decides most real deployments and the one benchmarks never test.
**2. Write and update characteristics.** Sustained write rate, delete handling, whether compaction is automatic and observable, and whether reads see writes (and how fast).
**3. Multi-tenancy primitives.** Namespaces with physical isolation, or just a metadata field? What is the cost of 100k of them?
**4. Consistency with your source of truth.** Transactional (Postgres), read-your-writes, or eventually-consistent with an unspecified lag?
**5. Hybrid ranking.** BM25 in the same engine, fusion built in, reranking hooks, or dense-only?
**6. Cost model shape.** Per-GB-month, per-read-unit, per-pod, or per-node? Serverless cost models are usage-shaped and can be wonderful or ruinous depending on your read/write ratio.
**7. Ops burden.** Managed vs self-hosted, backup/restore story, upgrade story, and whether you can observe recall at all.

Against those axes, the landscape as I hold it:

**Qdrant** — Rust, and the strongest *filtered* HNSW in the open-source tier. Payload indexes with cardinality estimation let it choose between filtered graph traversal and a direct scan of the matching set, which is the correct adaptive behaviour. Good quantization support (scalar, product, binary), segment-based storage with a background optimizer so deletes actually get compacted. My default recommendation when leaving Postgres for a filter-heavy workload.

**Weaviate** — Go, with genuine multi-tenancy as a first-class concept (per-tenant shards that can be individually offloaded to cold storage), hybrid BM25+dense with an `alpha` blend, and a module ecosystem that will embed for you. Pick it when tenant isolation and hybrid-out-of-the-box matter more than raw knob control.

**Milvus / Zilliz** — the most *distributed* of the open-source options: separate coordinator, query, data and index nodes, object storage underneath, a log broker in the middle. That architecture is right at billion scale and is a lot of moving parts at 10M. It supports the widest index menu (HNSW, IVF variants, DiskANN, ScaNN-family). Pick it when you are genuinely at billions and have platform engineers.

**Pinecone** — serverless, storage/compute separated, namespaces, metadata filtering, and essentially zero ops. The reason to pick it is that you have no platform team and want to stop thinking about this. The reason not to is cost opacity at scale and the fact that you cannot inspect or tune what it is doing.

**Vespa** — the most capable ranking engine in the list and the least like a "vector database." Tensor types as first-class values, multi-phase ranking (a cheap first phase over many candidates, an expensive second phase over few, and an optional global phase), real-time partial updates, and native hybrid. If your problem is *ranking* — a search product, a feed, a recommender — Vespa is frequently the correct answer and is underused because its learning curve is steep.

**Elasticsearch / OpenSearch** — Lucene HNSW inside a mature search engine. You get BM25, aggregations, the segment-merge model (which means deletes are handled by a compaction system that has been production-hardened for fifteen years), and operational tooling everyone already knows. You give up some ANN performance and some filtered-search quality. If your org already runs ES, the marginal cost of putting vectors in it is near zero and that usually wins. Recent releases added int8 and binary quantization modes built on the RaBitQ line of work — **📅 Volatile:** verify names and availability.

**Turbopuffer** — object-storage-native with a hot/warm/cold tier structure, so your cost per vector at rest is S3 pricing rather than RAM pricing. Structurally SPANN-shaped: few large sequential reads rather than many small random ones, which is the only IO pattern that works against object storage. The trade is cold-query latency measured in hundreds of milliseconds. Excellent for enormous, rarely-queried corpora — per-user archives, long-tail tenants.

**LanceDB** — embedded, built on the Lance columnar format, designed to sit directly on object storage with zero servers. Beautiful for local/edge and for analytical workflows where the vectors live next to the rest of your columns. **Chroma** — the developer-experience option; excellent for prototypes and notebooks, and I would not run a large production index on it. **Redis** — vector search in RediSearch, in-memory, so latency is superb and cost per vector is the worst in the list; the right call when your vectors are small in number and hot, e.g. a semantic cache. **ClickHouse / DuckDB** — brute force (and increasingly indexed) over a column that lives beside your analytics; the right call when the query is "find similar rows and then aggregate over them," which OLAP engines do and vector databases do not.

**🗣 Say this in the room:** "I'd pick on filtered-search quality, write/delete handling, multi-tenancy primitives and cost-model shape — not on published QPS, because those benchmarks are unfiltered single-tenant static workloads that resemble no production system. Concretely: Postgres until a tripwire fires, Qdrant for filter-heavy workloads, Vespa if the real problem is ranking rather than retrieval, Elasticsearch if the org already runs it, and object-storage-native tiering for a large cold corpus."

### What actually changes when the index lives on object storage instead of local disk?

Everything about the IO budget, and therefore everything about the index design. This is worth reasoning about from first principles because the object-storage-native tier is where the cost curve is moving.

The physics: a local NVMe random read is ~80 µs and you can do 500,000 per second. An S3 `GET` is **~20–100 ms of first-byte latency** and each request costs money — on the order of $0.0004 per 1,000 GETs — while bandwidth is effectively unlimited and storage is ~$0.023/GB-month against roughly $3–5/GB-month for RAM. So object storage is **250–1000× worse on latency, 100× cheaper on capacity**, and charges per request.

A graph index issuing 100 dependent random reads per query is therefore catastrophic: 100 sequential dependent round trips at 30 ms each is 3 seconds, and you cannot parallelise them because each hop's destination depends on the previous hop's contents. **Graph traversal is a latency-bound pointer chase, and object storage is the worst possible medium for a pointer chase.** This is the single most important structural fact in this design space.

So object-storage-native systems do three things:

**Restructure toward few large reads.** A partition/posting-list layout (SPANN-shaped) lets you decide *up front*, from a small in-memory centroid index, which 4–16 blocks you need, then fetch them all **in parallel** in one round trip's worth of wall clock. Latency becomes `one RTT + bytes/bandwidth` rather than `hops × RTT`.

**Tier aggressively.** Hot data in RAM/NVMe on the query node, warm in a local cache, cold in the bucket. Because real access distributions are brutally skewed — in a multi-tenant product, typically the top 1–5% of tenants generate most queries — a cache holding 5% of the corpus can serve 80%+ of traffic from the hot tier. That single fact is what makes the economics work.

**Make cost per query explicit.** Every query has a request count and a byte count, and both are billed. A design that does 16 GETs of 512 KB per query at 100 QPS is `16 × 100 × 86400 = 138M GETs/day = $55/day` in request charges alone, plus `16 × 512 KB × 100 × 86400 = 70 TB/day` of reads — which, if it crosses a region boundary, is a bill that will get you a meeting. Keep the reads in-region and in-VPC.

**💰 Math — the storage line item.** 1B vectors at d=1536 float32 = 6.14 TB. In RAM at ~$4/GB-month: `6,140 GB × 4 = $24,560/month`. On S3 Standard at $0.023/GB-month: `6,140 × 0.023 = $141/month`. That is **174×**, and it is why this architecture exists. You then spend some of that saving back on caching and request charges, but even at a 10× add-back you are ahead by an order of magnitude. **📅 Volatile:** storage prices; re-derive from current rate cards.

**⚠ Trap:** assuming the cold tier's latency is acceptable because your p50 is fine. In a skewed multi-tenant system, the *tenants* on the cold tier are a fixed set of customers, and for them p50 *is* the cold latency. Your dashboard shows a healthy p50 and a small tail; three specific customers experience a uniformly slow product. Always slice retrieval latency **by tenant**, not just globally — the same lesson you already know from shared database clusters.

### Build me the cost model for Pinecone serverless versus self-hosted Qdrant at 50M vectors.

I will build the *structure* and mark the rates, because the rates move and quoting a stale one is a worse answer than showing the model. **📅 Volatile:** every rate below — re-derive from current price sheets before your loop.

**The workload.** 50M vectors, d=1536. Say 20M queries/month (≈7.7 QPS average, call it 40 QPS peak) and 2M upserts/month. Raw vector bytes: `50e6 × 1536 × 4 = 307 GB`.

**💰 Math — self-hosted Qdrant.** With int8 scalar quantization the vectors are `50e6 × 1536 = 77 GB`; the HNSW graph at M=16 adds `50e6 × 130 = 6.5 GB`; payloads and overhead, call it 15 GB. Total ~100 GB resident, so I want 192 GB of RAM for headroom and merge activity, with the full-precision vectors on NVMe for rescoring. Two nodes for HA:

- 2 × `r7gd.6xlarge`-class (192 GB RAM, local NVMe) at ~$1.60/hr → `2 × 1.60 × 730 = $2,336/month`
- Backups to S3: 307 GB × $0.023 = $7/month
- **Engineer time: this is the line item people omit.** Upgrades, monitoring, capacity, an on-call rotation that now includes a stateful system. Call it 10% of one senior engineer: at a $300k fully-loaded cost, `0.10 × 300,000 / 12 = $2,500/month`.
- **Total ≈ $4,850/month**, of which more than half is human.

**Pinecone serverless.** The model has three components — storage per GB-month, read units consumed per query (which scale with how much data the query touches), and write units per upsert. The shape of the arithmetic:

- Storage: 307 GB × (storage rate $/GB-month)
- Reads: 20M queries × (read units per query) × (rate per read unit). Read units scale with the size of the *namespace being searched*, which is the crucial modelling fact: **the same query costs more in a 50M-vector namespace than in a 100k-vector one**. If you partition by tenant into namespaces, your read cost drops substantially. If you keep one flat 50M namespace and filter, it does not.
- Writes: 2M upserts × (write units) × rate.

**The decision rule, which is what is actually being asked.** Serverless usage-based pricing wins decisively when your traffic is **spiky, low-duty-cycle, or heavily namespaced**, because you pay for the queries you run rather than for RAM sitting idle at 3 a.m. Self-hosting wins when traffic is **steady and high**, because a box costs the same at 40 QPS and at 400 QPS while a per-query bill scales linearly. The crossover is a real calculation: find the QPS at which `20M queries × per-query cost` exceeds `$2,336 + $2,500`. If a query costs $0.0004, that is `20e6 × 0.0004 = $8,000/month` and self-hosting wins today; if it costs $0.00005, that is `$1,000/month` and serverless wins by 5×. **Do that division with current rates and put the crossover QPS on the slide.** An interviewer is testing whether you know the shape of the answer — "usage-based wins on spiky, capacity-based wins on steady, and here is where they cross" — not whether you memorised a price.

**⚠ Trap:** modelling only steady state and forgetting the **backfill**. A one-time load of 50M vectors into a per-write-unit system is 50M write units in a burst; at even $0.00001 per write unit that is a $500 one-off, and at a less friendly rate it can exceed a year of steady-state cost. I have seen a migration budget blown entirely on the initial ingest. Price the backfill separately, always, and ask the vendor whether bulk import is billed differently — it usually is.

### Vespa keeps coming up in search-heavy shops. When is it the right call, and what does it cost you?

Vespa is the right call when your problem is **ranking with multiple signals under a latency budget**, rather than "find nearest neighbours." That distinction is the whole answer.

The capability that nothing else in the list matches is **multi-phase ranking with a real expression language over tensors**. You define a first phase that is cheap and runs over a large candidate set (a linear blend of BM25, vector similarity, freshness and a business boost, say), a second phase that is expensive and runs over the top few hundred per node (a GBDT model, or a cross-encoder-ish scoring function), and optionally a global phase after merging across nodes. Vespa evaluates all of that *inside the content nodes*, next to the data, so you are not shipping 500 candidates over the network to a separate ranking service and paying a round trip. For a product like a search engine, a feed, or an e-commerce ranker, that architecture is simply correct, and reproducing it on top of a vector database means building a ranking service and eating the network hop.

The second differentiator is that **tensors are first-class**. Vespa can store a per-document tensor — a ColBERT-style per-token matrix, a multi-vector representation, a set of embeddings for a document's sections — and compute over it in the ranking expression. Late-interaction retrieval is natural there and awkward almost everywhere else.

Third: real-time partial updates. You can update a single field (a freshness score, a click-through rate, an ACL) without reindexing the document or touching the vector index. In most vector databases, "update the metadata" means "upsert the record," which means "reinsert into the graph."

What it costs you: **a steep learning curve and real operational weight.** You write application packages with schema definitions, rank profiles, and deployment configs; you learn YQL; you learn the content-node/container-node topology. It is a JVM-plus-C++ system with a distribution layer. A team of three shipping a RAG feature in a quarter should not pick Vespa. A team of fifteen building a search product that will run for five years frequently should — and I have seen more than one org spend two years building, on top of a vector DB and a pile of Python, a worse version of what Vespa gives you configured.

**My decision rule, stated as a question:** *is the hard part finding the candidates, or ordering them?* If finding — a vector database is fine. If ordering, with more than two signals, under a tight budget, at scale — look hard at Vespa before you build a ranking tier yourself.

### Elasticsearch or OpenSearch for vectors — give me the honest assessment.

The honest assessment starts with the thing nobody says out loud: **for most RAG systems, the lexical half of hybrid retrieval matters at least as much as the dense half, and Elasticsearch is the best lexical engine in this list by a wide margin.** If you are going to end up doing BM25 anyway — and you are, because dense retrieval fails badly on exact identifiers, SKUs, error codes and proper nouns — then a system that does both in one query, in one engine, with one relevance framework, is starting from a structural advantage.

What you get: Lucene's HNSW implementation, `knn` queries with filters, the segment-merge model (immutable segments, background merging), BM25 with full analyzer control, aggregations, and an operational surface your org probably already staffs. The segment model is genuinely important for the deletion problem discussed earlier: deletes are marked in a per-segment liveness bitmap and reclaimed by merges, which is a compaction system that has been hardened over fifteen years and that you do not have to think about. That is a real advantage over a monolithic in-process HNSW where you own compaction yourself.

What you give up. **ANN performance is a notch below the purpose-built engines** — Lucene's HNSW is good, not best-in-class, and per-segment graphs mean a query searches each segment's graph and merges, so recall and latency depend on your segment count and merge policy in a way that is a genuine operational subtlety. Filtered vector search is better than post-filtering (Lucene can pass an acceptance bitset into the graph traversal and falls back to exact search when the filter is very selective, which is the right adaptive behaviour) but is not as tuned as Qdrant's. Memory management is JVM memory management, with all that implies. And the cluster is heavy — you are running a distributed search cluster to do a job a single Postgres box might do.

Where it clearly wins: you already run it; you need hybrid; you need faceting, aggregations and structured search alongside vectors; you have documents with rich metadata and complex boolean filters; you need a mature security model with document-level and field-level security — which, for ACL-aware enterprise retrieval, is a genuinely differentiating feature that most vector databases simply do not have.

Where it clearly loses: pure dense retrieval at very high QPS where every millisecond counts; billion-scale on a tight budget; teams with no Elasticsearch operational experience who would be adopting a cluster to store four million vectors.

**📅 Volatile:** quantization support here has moved quickly — int8 and binary quantization modes (Elasticsearch's BBQ line, derived from the RaBitQ family of work) have landed and changed the memory arithmetic substantially. Verify what your version supports rather than quoting a memory figure from an older release.

**🗣 Say this in the room:** "If the org already runs Elasticsearch, putting vectors in it is usually correct, because hybrid retrieval is the actual requirement and it's the best lexical engine here — and Lucene's segment-merge model solves the tombstone-compaction problem I'd otherwise own myself. I'd give up maybe a notch of pure ANN latency and some filtered-search tuning to get BM25, aggregations, document-level security and an ops story my team already knows."
