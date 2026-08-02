# PART X — Data and Enterprise Integration Platform

Half the job at Glean, Harvey, Sierra, Notion and Ramp, and the whole job in an FDE deployment — yet a distinct engineering discipline that retrieval sections never cover.

## Contents

1. [61. The AI Data Platform: Embedding Pipelines, Lakehouse Integration, Contracts and Versioning](#61-the-ai-data-platform-embedding-pipelines-lakehouse-integration-contracts-and-versioning) — 44 questions
2. [62. The Enterprise Integration Surface: Connectors, Identity, Permission Mirroring and Tenant Configuration](#62-the-enterprise-integration-surface-connectors-identity-permission-mirroring-and-tenant-configuration) — 44 questions


---

## 61. The AI Data Platform: Embedding Pipelines, Lakehouse Integration, Contracts and Versioning

*Mastering this proves the Databricks/Snowflake fusion round: architecting a retrieval pipeline and defending its data engineering in the same breath.*

### Let's start broad. Why do you keep calling the embedding pipeline a data engineering problem rather than an ML problem?

Because nothing about it is statistically interesting and everything about it is operationally brutal. The model is a fixed pure function `bytes -> float32[d]`. It has no state, no gradients, no drift of its own. What makes an embedding pipeline hard is exactly what makes any large batch pipeline hard: partitioning, skew, checkpointing, idempotency, exactly-once semantics against a downstream that does not support transactions, and the fact that you will run it again — many times — over data that is simultaneously being mutated by the source system.

The mental model I use, and the one that reframes the whole section: **the vector index is a derived dataset, and every rule you already enforce about derived datasets applies unchanged.** You would never let a Postgres materialized view be the only copy of the data that produced it. You would never rebuild a nightly aggregate without a way to compare it to yesterday's. You would never let a downstream consumer read a half-built table. Yet teams routinely write embeddings directly into a Pinecone or a Qdrant namespace with no source-of-truth table behind it, no way to recompute, and no version stamp — and then discover six months later that they cannot answer "which model produced this vector" for 40% of their corpus.

The ML content of the pipeline is one line: which encoder, at which dimension. Everything else is your existing job. That is genuinely good news for you in a Databricks or Snowflake loop, because that round is a fusion round — they will ask you to architect a retrieval system and then immediately interrogate the data engineering underneath it, and most candidates who can talk fluently about HNSW parameters cannot tell you what happens when the ingestion job dies 60% through a 100M-chunk run.

**⚠ Trap:** treating the vector store as the system of record. The moment you cannot rebuild the index from an upstream table, you have lost the ability to change embedding models, change chunking, re-run with a fixed parser, or answer a deletion request completely. I enforce this in review as a hard rule: **every vector in the index must be reconstructible from a row in a governed table by a pinned, versioned function.** If you cannot write that function down, the design is not done.

**🗣 Say this in the room:** "I treat the vector index as a materialized view, never as a source of truth. The source of truth is a Delta or Iceberg table of chunks with content hashes and version columns; the index is a build artifact I can throw away and regenerate. That one constraint is what makes model migrations, deletion requests, and incident rollback tractable instead of heroic."

### Design the embedding generation job for me. 100 million chunks, one shot. What framework, what partitioning, what does a task actually do?

Start from the shape of the work: it is embarrassingly parallel over chunks, GPU-bound in the middle, and I/O-bound at both ends. That means the framework question reduces to "which one lets me put a GPU stage in the middle of a CPU pipeline without the GPU idling."

**Spark** is the right answer when your organization already runs Spark and the corpus already lives in Delta/Iceberg — you get the catalog, the ACID writes, the lineage, and the ops team for free. The GPU stage is a `mapInPandas` or `mapInArrow` over partitions, with the model loaded once per executor in a module-level singleton, not per row. The weakness is real: Spark's scheduler thinks in CPU cores, so a GPU executor with 8 cores will happily run 8 tasks that each want the whole GPU. You pin this with `spark.task.resource.gpu.amount` and a partition count sized so one task ≈ one GPU's worth of work.

**Ray Data** is the right answer when the pipeline is heterogeneous — CPU parse, GPU embed, CPU write — because it pipelines those stages with independent autoscaling and streams batches through, so the GPU stage stays fed while the CPU stage is still parsing. That is the actual win: not throughput per GPU, but **GPU utilization across the whole job**. (**📄 Paper:** Moritz et al. (2018), *Ray: A Distributed Framework for Emerging AI Applications* — introduced the actor + task model that lets a stateful GPU actor sit inside a streaming dataflow, which is precisely what a "load model once, embed many batches" stage needs.)

My default in 2026: Ray Data for the compute, Delta or Iceberg for both input and output, orchestrated by whatever your platform already runs. If the shop is Databricks-native, Spark with GPU executors is less friction and I would not fight it.

Partitioning is where people lose 3× throughput. Three rules:

1. **Partition by source document, not by chunk.** Chunks from the same doc share tokenizer state and locality; more importantly, retry granularity should match your idempotency key.
2. **Sort within a partition by token length.** Encoder inference pads to the longest sequence in the batch. A batch of 32 chunks where 31 are 120 tokens and one is 512 tokens costs the same as 32 × 512. Length-sorting cuts padded FLOPs by 40–60% on a realistic length distribution, for free.
3. **Size partitions so a task is 30–120 seconds.** Shorter and you drown in scheduling and model-warmup overhead; longer and a single straggler or preemption costs you real wall-clock on retry.

A task, concretely:

```python
# Ray Data: one GPU actor per replica, model loaded once in __init__
class Embedder:
    def __init__(self, model_id: str, dim: int):
        self.model = SentenceTransformer(model_id, device="cuda")
        self.model_id, self.dim = model_id, dim

    def __call__(self, batch: dict) -> dict:
        order = np.argsort([len(t) for t in batch["text"]])   # length-sort
        texts = [batch["text"][i] for i in order]
        vecs = self.model.encode(texts, batch_size=64,
                                 normalize_embeddings=True,
                                 convert_to_numpy=True)
        inv = np.argsort(order)                                # restore order
        return {
            "chunk_id": batch["chunk_id"],
            "content_sha": batch["content_sha"],
            "vector": vecs[inv],
            "embed_model": [self.model_id] * len(texts),
            "embed_version": [EMBED_PIPELINE_VERSION] * len(texts),
        }

ds = ray.data.read_delta("s3://lake/chunks_v7")   # or read_parquet
ds.map_batches(Embedder, fn_constructor_args=(MODEL_ID, 1024),
               concurrency=32, num_gpus=1, batch_size=512) \
  .write_delta("s3://lake/embeddings_v7", mode="append")
```

(**📅 Volatile:** Ray Data's Delta reader/writer names have moved across releases. If `read_delta`/`write_delta` aren't in the version you're on, `read_parquet`/`write_parquet` over the table's data path plus a catalog registration is the portable fallback — verify the spelling before you write it on a whiteboard. The actor-per-replica shape is the part that matters and hasn't changed.)

**⚠ Trap:** loading the model inside the map function instead of the actor constructor. It works, the job runs, and you silently pay 3–8 seconds of model load per task. At 20,000 tasks that is 16–44 GPU-hours of pure warmup — a 30–80% tax on a 56 GPU-hour job, and if your tasks are shorter than the model load itself the warmup exceeds the embedding compute outright. This passes code review constantly because the code is correct; it is only slow.

### How many chunks per second can one GPU actually embed? Derive it, don't quote me a vendor benchmark.

Derive it from FLOPs, because that number transfers across hardware and models and a vendor benchmark does not.

A transformer encoder does roughly **2 FLOPs per parameter per token** in a forward pass (one multiply, one add, per weight, per token). Attention adds a term quadratic in sequence length, but at 512 tokens with a hidden size of 768 it is a modest fraction, so ignore it for a back-of-envelope and note that you're ignoring it.

**📐 Numbers you must know:**
- A **110M-parameter encoder** (BERT-base class — most of the fast embedding models are in this family) on a **512-token chunk**: 2 × 110×10⁶ × 512 = **1.13 × 10¹¹ FLOPs ≈ 113 GFLOP per chunk**.
- An **A100 80GB** does ~312 TFLOP/s dense bf16 at peak. Encoder inference with good batching realistically lands at **35–45% MFU**, so call it **125 TFLOP/s effective**.
- Ceiling: 125×10¹² ÷ 1.13×10¹¹ = **~1,100 chunks/sec** compute-bound. In practice, with padding waste, tokenization on CPU, and I/O, **400–700 chunks/sec** is the honest sustained figure for a well-tuned job. If you are getting 60, something is wrong — almost always per-task model loading, batch size 1, or an un-sorted length distribution.
- The same arithmetic for a **7B-parameter LLM-based embedder**: 2 × 7×10⁹ × 512 = 7.17 × 10¹² FLOPs = **7.2 TFLOP per chunk** → 125 ÷ 7.2 = **~17 chunks/sec**. That is a **65× throughput cliff** for maybe 3–6 points of nDCG@10.

That 65× is the whole judgment. **📅 Volatile:** specific model quality rankings move monthly; verify the current MTEB-style leaderboard before your loop, and verify it on *your* data because leaderboard rank and your-corpus rank diverge badly on domain text.

**💰 Math:** 100M chunks on one A100 at 500 chunks/s = 200,000 s = **55.6 GPU-hours**. Spread over 32 GPUs that is **1.7 hours wall-clock**. At ~$1.10/GPU-hr for A100 spot (**📅 Volatile:** GPU spot pricing swings 2–3× by region and quarter), that is 55.6 × $1.10 = **$61 of compute**. With the 7B embedder: 100M ÷ 17 = 5.88M seconds = **1,634 GPU-hours** = $1,797, and 51 hours of wall-clock on the same 32 GPUs. Same corpus, 29× the bill and a job that no longer fits in a maintenance window.

**🗣 Say this in the room:** "Two FLOPs per parameter per token, times chunk length, divided by roughly 40% of the GPU's peak — that's the ceiling, and it tells me a 7B embedder costs about 65× a 110M one per chunk. I'd need to see the retrieval win in an offline eval on our own corpus before I'd sign up for a 51-hour backfill instead of a 2-hour one."

### What's the idempotency key for an embedding job, and what actually breaks if you get it wrong?

The key must be a **content hash, not a row id**, because the thing you are memoizing is the function `embed(text)`, and the function does not care which row the text came from.

Concretely I write the chunk table with:

```
chunk_id        = sha256(f"{source_id}:{doc_id}:{chunker_version}:{chunk_ordinal}")
content_sha     = sha256(normalized_chunk_text)
embed_key       = sha256(f"{content_sha}:{embed_model_id}:{embed_params_hash}")
```

Three keys, three jobs. `chunk_id` is the stable address a citation points at. `content_sha` is the dedup and change-detection key. `embed_key` is the memo key — if a row with that `embed_key` already exists in the embeddings table, skip the GPU call entirely.

What breaks with each wrong choice:

**Keying on `doc_id` alone:** a document is edited, you re-embed it, and you now have two generations of chunks in the index with no way to tell which is current. Retrieval returns the old paragraph. This is the single most common cause of "our RAG system returns stale answers after a reindex" — the old vectors were never deleted because nothing tied them to the new write.

**Keying on `chunk_id` without `content_sha`:** you cannot detect that a document changed but produced the same chunk boundaries, so you re-embed 100% of a corpus where 2% actually changed. That is the difference between a $61 nightly job and a $1.20 one.

**Keying on `content_sha` without the model id:** you migrate to a new embedding model, the memo table says "already embedded," and you silently ship a hybrid index where 30% of vectors are from model A and 70% from model B. Cosine similarity across two different embedding spaces is *not an error* — it returns a number, a plausible-looking number, and retrieval quality degrades by 15–30 points of Recall@10 with no exception in any log. This is my favourite question to ask candidates because the failure is completely silent.

**⚠ Trap:** normalizing the text for the hash but not for the embed call, or vice versa. If you strip whitespace before hashing but embed the raw string, two logically identical chunks get one memo entry and two different vectors depending on which one won the race. Hash exactly the bytes you embed. I put that in a single function and forbid callers from doing their own normalization.

**🏋 Drill (12 minutes, unaided):** write the three key definitions above from memory, then write the `MERGE INTO` (or Iceberg `MERGE`) that upserts embeddings and *deletes* rows whose `chunk_id` no longer appears in the current chunk table. Pass criterion: your merge handles the delete side. Most people write only the upsert, and the delete side is where staleness lives.

### The embedding job dies 60% of the way through a 12-hour run. What did you build so that isn't a disaster?

Nothing exotic — the memo table above *is* the checkpoint. If the write of `(embed_key, vector)` is atomic and the read path skips existing keys, then "resume" and "run again" are the same operation, and you have converted a stateful long-running job into an idempotent one you can run in a loop until it converges. That is the design goal: **no separate checkpoint mechanism, because the output is the checkpoint.**

Three details make that actually work at 100M scale.

**Commit granularity.** Do not write once at the end. Write in commits of 100k–1M rows so a failure loses at most one commit. On Delta/Iceberg, each commit is a table version, so you also get free progress observability: `DESCRIBE HISTORY` (Delta) or the snapshots metadata table (Iceberg) tells you exactly how far you got and when. The cost of small commits is metadata bloat — 100M rows in 1k-row commits gives you 100,000 snapshots and a manifest read that takes minutes. I size commits so a run produces **50–500 commits**, not 5 and not 50,000.

**The skip-scan must be cheap.** On resume you need "which of these 100M `embed_key`s are already present." Do not do 100M point lookups. Do an anti-join: read the distinct `embed_key` column from the output table (one column, 32 bytes/row = 3.2 GB, trivially scanned), broadcast or shuffle-join against the input, and process the difference. On Delta this is a single `LEFT ANTI JOIN` and it runs in a couple of minutes.

**Spot preemption as the normal case, not the exception.** At $1.10/hr spot vs ~$3.70/hr on-demand for A100-class capacity (**📅 Volatile:** verify current rates), you are choosing between $61 and $206 for the 55.6 GPU-hour job. Spot is obviously right *if and only if* a preemption costs you at most one commit. So I size the commit interval against the preemption rate: if you see a preemption every ~40 minutes per node, a 10-minute commit interval loses on average half a commit — 5 minutes, or **12.5% of a node's work**, 25% in the worst case — which is well inside the 3.4× price gap.

**💰 Math:** 55.6 GPU-hours. On-demand: 55.6 × $3.70 = **$206**. Spot with a 15% average re-do penalty from preemptions: 55.6 × 1.15 × $1.10 = **$70**. Saving $136 on one run is uninteresting; saving it on a nightly incremental plus four full backfills a year is the difference between a $2k and a $7k line item, and the real argument is that spot with checkpointing also means you *can* run the job at 3am when capacity exists.

**⚠ Trap:** treating a partially-written output table as valid input for the index build. Your DAG must not have "build index" read the embeddings table on a schedule; it must read a **specific committed version** that a preceding task declared complete. Otherwise a resumed job and an index build race, and you ship an index that is missing 40% of the corpus with no error anywhere.

### API embeddings or self-hosted? Give me the actual decision, with the money.

The decision is not primarily cost — it is **the cost of the migration you will be forced into**, which is where most teams get surprised.

**💰 Math, one full pass over 100M chunks at ~512 tokens each = 51.2 billion tokens:**
- Small hosted embedding model at **$0.02 per million input tokens**: 51,200 Mtok × $0.02 = **$1,024**.
- Large hosted embedding model at **$0.13 per million**: 51,200 × $0.13 = **$6,656**.
- Self-hosted 110M-param encoder on spot A100s: **~$70** in GPU time (derived above), plus engineering.

(**📅 Volatile:** those per-token prices are the shape of the market, not a quote — verify before your loop. The *ratio* — hosted small is ~15× self-hosted compute, hosted large ~95× — is the durable part, and it is stable because you are paying for someone else's margin plus a better model.)

So a 15–95× cost multiple on compute, but on a base of ~$70. For a one-time 100M-chunk backfill, $1,024 is not a decision — it is a rounding error against two engineer-weeks of building a GPU pipeline. The number that actually decides it is **steady-state volume**:

- 5M new chunks/day (an active enterprise corpus with CDC) = 2.56 Btok/day. Hosted small: 2,560 × $0.02 = **$51/day = $1,530/month**. Self-hosted: 5M ÷ 500/s = 10,000 s = 2.8 GPU-hours/day = $3/day = **$92/month**. Now the gap is $1,438/month and self-hosting starts to pay back an engineer-week in about four months.
- 200k chunks/day (a normal internal wiki): hosted = $2/day. Never self-host. You are burning a headcount to save $700/year.

The three non-cost factors that usually dominate:

1. **Reproducibility.** A hosted embedding endpoint can change behind a stable model name. If your provider silently re-tunes `embedding-model-v3`, half your index is in a slightly different space than the other half and nothing tells you. Self-hosted weights with a pinned hash cannot do this. For a regulated customer this alone forces self-hosting.
2. **Rate limits as a wall-clock constraint.** A backfill is not cost-limited on a hosted API, it is TPM-limited. If your tier gives you 5M tokens/minute, 51.2 Btok takes 51,200,000,000 ÷ 5,000,000 = 10,240 minutes = **7.1 days**. Your self-hosted job took 1.7 hours on 32 GPUs. That is the argument that actually wins the room.
3. **Data residency.** If chunks cannot leave a VPC, the decision is made for you.

**🗣 Say this in the room:** "For a one-off backfill I use the hosted API — a thousand dollars is cheaper than two engineer-weeks, and the rate limit is the real constraint, not the price. At sustained volume above roughly a billion tokens a month, or when I need bit-reproducible vectors across a year, I self-host a small encoder on spot GPUs and the pipeline becomes a $100/month line item."

### Where do the vectors live — a Delta or Iceberg table, a dedicated vector database, or both? Defend your answer.

Both, and the reason is that they answer different queries and have different durability requirements.

The lakehouse table is the **system of record**: append-only-ish, ACID, time-travelable, governed by the same catalog and access policy as the rest of your data, cheap per GB on object storage, and joinable to everything else you own. What it cannot do is answer "give me the 50 nearest neighbours of this 1024-dim vector in 20ms." A brute-force scan over 100M × 1024 fp32 vectors is 409.6 GB of reads per query — Spark will do it in minutes, which is fine for an offline eval and useless for a product.

The vector index is the **serving artifact**: an HNSW or IVF-PQ or DiskANN structure that trades memory and build time for sub-linear search. (**📄 Paper:** Malkov & Yashunin (2018), *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* — the multi-layer skip-list-over-a-proximity-graph that most production vector search still runs; **📄 Paper:** Subramanya et al. (2019), *DiskANN* — showed a graph index can serve billion-scale from SSD with a small memory footprint, which is what made 100M+ corpora affordable outside pure-RAM systems.)

The architecture I default to:

```
source systems → raw table (Iceberg/Delta, governed)
               → chunks table (chunk_id, content_sha, text, metadata, chunker_version)
               → embeddings table (embed_key, chunk_id, vector, embed_model, embed_version)
               → [BUILD] → vector index (versioned name) → [ALIAS SWAP] → serving alias
```

Everything left of `[BUILD]` is durable, governed, replayable. Everything right of it is disposable. If the vector database loses a node and cannot recover, the correct response is to rebuild from the embeddings table, not to page someone about backups.

**When is the index legitimately the *only* store?** Two cases. First, small corpora — under ~1M vectors, the "table" can just be a Parquet file and the ceremony is not worth it. Second, when your vector database *is* backed by the lakehouse: Databricks Vector Search's Delta Sync index and similar managed products maintain the index as a genuine materialized view over a Delta table, with the sync as the vendor's problem (**📅 Volatile:** the exact product surface and sync semantics change; verify). That is the ideal shape, and when it is available I take it, because it eliminates the class of bug where the table and index disagree.

**⚠ Trap:** "we'll just use pgvector, it's one less system." pgvector is genuinely the right answer more often than vector-DB vendors want you to believe — under ~5–10M vectors with HNSW, colocated with your relational data so filters are real SQL joins rather than a metadata sublanguage, it is simpler and faster to operate. The trap is the other direction: teams pick it at 5M and then discover the index build takes hours, blocks vacuum, and the whole thing lives in the same instance as their transactional workload. My rule: **pgvector until the index no longer fits comfortably in RAM alongside your OLTP working set, then a dedicated store.** Decide it on memory arithmetic, not on taste.

### You said "the index is a materialized view of the table." Push on that — where does the analogy break?

It holds in three places and breaks in two, and knowing which is which is the actual senior signal.

**Where it holds.** (1) *Derivability*: the index content is a pure function of the table content plus a pinned build config. (2) *Staleness as the core property*: like any matview, the interesting question is not "is it correct" but "how far behind is it, and does the product notice." (3) *Refresh strategy*: full rebuild vs incremental maintenance, with exactly the same trade-off — incremental is cheap but drifts, full is expensive but self-healing. I schedule a full rebuild weekly for the same reason DBAs schedule a full refresh: it is the only thing that repairs drift you failed to detect.

**Where it breaks, first: the view has internal state that the query doesn't determine.** An HNSW graph built by inserting documents in order A,B,C is not the same graph as one built C,B,A. Both are valid; both give slightly different recall. So "rebuild and diff" does not produce byte equality, and any promotion gate you write must be **statistical** — Recall@k against a fixed probe set, not a checksum. This surprises people with a strong data-engineering background, because their entire instinct is "recompute and compare hashes."

Worse: HNSW **degrades under deletion**. Most implementations tombstone rather than truly remove, so the graph keeps the deleted node's edges as routing structure while filtering it from results. A corpus with heavy churn accumulates dead nodes, recall drifts down, and latency drifts up, and the only fix is a rebuild. That is a maintenance property no Postgres matview has, and it is the honest reason "just do incremental updates forever" fails.

**Where it breaks, second: the refresh is not transactional with the read path.** A Postgres `REFRESH MATERIALIZED VIEW CONCURRENTLY` gives you a consistent switch. A vector index rebuild takes 40 minutes and produces a *new* index; making it live is a separate operation you have to design (the alias swap, later in this section). During those 40 minutes you are serving the old index, which is fine, but any writes that landed during the build are missing from the new one — so the build must record its input version and the incremental path must replay from that version forward. Miss that and every rebuild silently drops a build-window's worth of documents.

**🗣 Say this in the room:** "I use the materialized-view framing for everything except two things: the index isn't byte-deterministic, so my promotion gate is Recall@k on a fixed probe set rather than a checksum, and refresh isn't atomic, so I need an explicit alias swap plus a replay of everything written during the build window."

### Walk me through chunking as a pipeline stage. Why do you make such a fuss about chunk id stability?

Chunking is the stage everyone treats as a prompt-engineering detail and it is actually a **schema decision**, because `chunk_id` is a foreign key that citations, feedback, evals, and audit logs all point at. Change the chunker and you break every one of those references at once.

The mechanism: a chunker is `(document_bytes, config) -> list[(offset_start, offset_end, text)]`. I insist it emit byte offsets into the *original* document, not just text, for three reasons — a citation can be rendered as a highlight in the source viewer, a re-chunk can be diffed against the old one in offset space, and a PII scan on the source can be projected onto chunks without rescanning.

`chunk_id` stability is the fuss because of what depends on it:

- **Human feedback.** A user thumbs-down an answer citing chunk `abc123`. Six weeks later you tune the chunker. If `abc123` no longer exists, you have thrown away labeled data — and labeled data on your own corpus is the scarcest asset in the whole system.
- **Eval sets.** Your retrieval eval is a list of (query, relevant_chunk_ids). Re-chunk and your eval set silently measures nothing; Recall@10 drops to near zero and looks like a model regression.
- **Deletion.** "Delete everything derived from document X" is a query on `chunk_id` prefix or a join to `doc_id`. If ids are content-derived only, you cannot enumerate a document's chunks without re-parsing it — and if the source is already deleted you cannot.

So the id includes the chunker version *by design*: `sha256(source_id:doc_id:chunker_version:ordinal)`. That means changing the chunker mints new ids — which is correct and honest. The discipline is that a chunker change is then a **migration**, with a mapping table `old_chunk_id -> new_chunk_id[]` computed by offset overlap, so feedback and eval sets can be forward-ported instead of lost. Computing that mapping is ten lines:

```python
def remap(old_chunks, new_chunks, min_overlap=0.5):
    """Map each old chunk to new chunks sharing >= min_overlap of its byte span."""
    out = {}
    for o in old_chunks:
        hits = []
        for n in new_chunks:
            lo, hi = max(o.start, n.start), min(o.end, n.end)
            if hi > lo and (hi - lo) / (o.end - o.start) >= min_overlap:
                hits.append(n.chunk_id)
        out[o.chunk_id] = hits
    return out
```

**⚠ Trap:** hashing the chunk *text* to make the id. It feels elegant — content-addressed, dedupes for free — and it destroys you the first time two documents contain the same boilerplate paragraph, because now one id maps to two source documents with two different ACLs. I have seen this produce a genuine cross-tenant leak: a shared legal disclaimer chunk deduped across tenants, and the permission filter matched on the chunk's single stored `tenant_id`. Content hash is the *dedup* key. It is not the *identity* key. Keep them separate.

### Your embeddings table has 100,000,000 rows and the index reports 99,412,338. Find the missing 588k.

Good — this is the debugging question I actually ask, because the answer is a procedure, not a guess. The gap is 0.59%, which is too big to be rounding and too small to be a broken stage. That size profile almost always means **per-row rejection**, not a systemic failure.

**🔍 Failure taxonomy — run these in order, cheapest first:**

1. **Is it a version mismatch, not a loss?** Does the index build read a pinned table version? Compare `count(*)` in the embeddings table *at the snapshot the build read* against the index count. Very often nothing is missing at all; 588k rows landed after the build started. Delta: `SELECT count(*) FROM embeddings VERSION AS OF <v>`. Iceberg: query the snapshot. If this explains it, the bug is that the build window's writes were never replayed — a real bug, but a different one.

2. **Null or malformed vectors.** `SELECT count(*) FROM embeddings WHERE vector IS NULL OR size(vector) != 1024`. Embedding calls that returned an error and got written as null are the single most common cause. Also check for NaN: a chunk that tokenized to zero tokens (an empty string after normalization, or a chunk that is pure whitespace after a PDF parse failure) produces a NaN or zero vector in some libraries, and most vector stores reject NaN on insert — silently, in a batch upsert, with a partial-success response nobody checked.

3. **Duplicate primary keys.** The index dedupes on id; the table does not. `SELECT count(*) - count(DISTINCT chunk_id) FROM embeddings`. If your job ran twice with an append (not merge) write mode, you have duplicates and the index count is the *correct* one.

4. **Metadata that violates the index's schema.** Vector stores enforce constraints your lakehouse does not: a metadata string over 40 KB, a field with mixed types across rows, a null in a field declared as a filterable keyword. These reject per-row inside a batch. The tell: rejections cluster by source system, because one connector emits a field the others don't.

5. **Batch partial failure that was never surfaced.** Most vector-store bulk APIs return a per-item status array. If your writer does `if response.ok:` and moves on, you have been dropping rows for months. This is the one I'd bet on if steps 1–4 come back clean.

The fix that prevents recurrence is not better logging — it is **a reconciliation job**, run after every build and nightly in steady state:

```sql
-- anti-join the source of truth against a dumped id list from the index
SELECT e.chunk_id, e.embed_model, e.updated_at
FROM embeddings_v7 e
LEFT ANTI JOIN index_ids_snapshot i ON e.chunk_id = i.chunk_id
```

Emit the count as a metric, alert on `missing_ratio > 0.001`, and dump a sample of 100 ids into the alert so the on-call has evidence in the page rather than a number. The reverse anti-join matters just as much — ids in the index with no row in the table are **orphans that will never be deleted**, which is how you fail a GDPR audit.

**📐 Numbers you must know:** at 100M chunks, a 0.1% silent drop is 100,000 chunks — roughly 5,000 documents at 20 chunks each. If those 5,000 documents are the ones your newest connector ingested, an entire customer's data is invisible in search and no dashboard shows red. That is why I alert on ratio, and why the alert includes which `source_id`s the missing rows belong to.

### Talk me through the write path from "a document changed in Confluence" to "the new text is searchable." What are the stages and where does latency come from?

Trace it end to end, because the interesting answer is the latency budget, not the box diagram.

**Stage 1 — change detection (seconds to hours).** Either a webhook from the source, a polled cursor, or log-based CDC if the source is a database you control. This is the dominant term in freshness and it is almost never the part people optimize. A Confluence webhook fires in seconds; a polling connector on a 15-minute cursor has a mean detection latency of 7.5 minutes and a p99 of 15.

**Stage 2 — fetch and parse (100ms–30s).** Pull the document, run the extractor (HTML, PDF, DOCX). PDFs with OCR are the tail: a 200-page scanned PDF is minutes, not milliseconds, and it must not be on the same queue as a 4 KB wiki page or one bad document blocks a thousand good ones. **Separate queues by expected cost class** — the same reason you would not put a report-generation task and a webhook-ack on the same Celery queue.

**Stage 3 — normalize, chunk, hash (milliseconds).** Compute `content_sha` per chunk and diff against the existing chunk set for that `doc_id`. This is where you win: a typical wiki edit changes one paragraph, so 1 of 20 chunks has a new `content_sha` and 19 are skipped. **The diff is the optimization**, and it is a pure data-engineering move — 95% of the GPU work on an edit-heavy corpus disappears.

**Stage 4 — embed (10–200ms per chunk, batched).** Only for changed chunks. If you're on a hosted API, this is a network call with a rate limit; if self-hosted, it's a queue in front of a GPU. Either way, batch it: embedding one chunk costs nearly the same as embedding 32.

**Stage 5 — upsert to table, then upsert to index (100ms–seconds).** Two writes, and they must be ordered table-first so the index is never ahead of the source of truth. Deletes of superseded chunks go here too.

**Stage 6 — index visibility (0–60s).** This is the stage backend engineers don't expect. Many vector stores do not make an upsert searchable immediately; there is a refresh or commit interval, exactly like Elasticsearch's `refresh_interval`. Default is often ~1s but is sometimes configured to 30–60s for throughput. If your product promises "edits are searchable immediately," you must know this number and possibly force a refresh on the write path for interactive edits.

**💰/📐 The budget, added up:** webhook 5s + parse 2s + chunk 0.1s + embed 0.2s + upsert 0.5s + refresh 1s ≈ **~9 seconds p50** for a wiki page. With a 15-minute polling connector instead of a webhook, p50 becomes **~7.5 minutes** and p99 ~15 minutes. Same pipeline, 50× worse freshness, and the difference is entirely in stage 1. When a customer says "search is stale," the answer is in stage 1 about 80% of the time, and the fix is a webhook or a tighter cursor — not a faster GPU.

**⚠ Trap:** measuring freshness as "time from our ingestion job starting." The SLA the customer experiences is **source-modified-time to searchable-time**, which requires you to carry the source's `modified_at` through every stage and emit `now() - source_modified_at` as a histogram at the moment the chunk becomes searchable. If you don't propagate that timestamp from stage 1, you literally cannot measure the thing you're being judged on, and I have seen teams instrument the whole pipeline and still be blind to a broken connector because every internal stage was green.
### Design CDC-driven index maintenance for me. The source is a Postgres table of 40 million support tickets that changes constantly.

The mental model: **the vector index is a downstream consumer of the write-ahead log, and every property you already rely on in a Kafka-to-Postgres sink applies unchanged — at-least-once delivery, ordering per key, and a compaction story for deletes.** The only new part is that the "transform" in the middle costs a GPU call, so you care about deduplicating redundant work in a way a normal sink does not.

The pipeline:

```
Postgres WAL → logical replication slot → Debezium → Kafka topic (key = ticket_id)
   → dedupe/debounce window → chunk + content_sha diff → embed changed chunks only
   → MERGE into embeddings table → upsert/delete into vector index
```

Concretely on Postgres: create a publication for the table, a logical replication slot, and set `REPLICA IDENTITY FULL` on the table if you need the before-image on updates and deletes. Debezium emits a record per change with an `op` of `c`/`u`/`d`/`r` (create, update, delete, read-during-snapshot) plus source metadata including the LSN, which is your monotonic position marker and the thing you will use for replay.

Three engineering decisions that separate a working design from a demo:

**1. Debounce, aggressively.** A support ticket gets 12 updates in an hour as an agent types. Naively you embed it 12 times. I window by key — a 60-second tumbling or session window keyed on `ticket_id`, take the last state, then diff `content_sha` per chunk. On a ticketing corpus this collapses embedding volume by 5–15×. **💰 Math:** 40M tickets with an average 8 updates/day = 320M change events/day. At 20 chunks/ticket and no debounce that is 6.4B embed calls/day, which is absurd. With last-write-wins debounce at 60s you drop to ~40M ticket-versions/day, and with `content_sha` diffing you only embed the ~1.5 chunks that actually changed per edit — call it 60M chunk-embeds/day. From 6.4B to 60M is a **107× reduction**, and both numbers come from the same pipeline with two lines of difference.

**2. `REPLICA IDENTITY` determines whether deletes are even possible.** With the default (`DEFAULT` = primary key only), a delete event carries only the PK. That is enough if `doc_id -> chunk_id` is derivable or stored; it is *not* enough if you need the old content to compute which chunks to remove. My rule: store the `doc_id -> [chunk_id]` mapping in your own table so a delete never requires the source's before-image. Do not depend on `REPLICA IDENTITY FULL`, because it doubles WAL volume and a DBA will turn it off.

**3. The replication slot is a liability.** If your consumer stops, the slot holds WAL and the source's disk fills. That is a Sev-1 in *someone else's* database caused by your AI pipeline. Alert on `pg_replication_slots.confirmed_flush_lsn` lag in bytes, with a hard threshold well below the WAL volume the disk can hold, and have a documented "drop the slot and re-snapshot" runbook. Every senior data engineer has been burned by this; saying it unprompted is a strong signal.

**⚠ Trap:** treating CDC as a replacement for periodic full reconciliation. CDC is at-least-once with a replay boundary, and it *will* miss things — a slot dropped during an incident, a topic retention expiry, a schema change that broke the connector for two hours. I run a nightly reconciliation that anti-joins source PKs against index ids and repairs the difference. CDC gives you seconds of freshness; reconciliation gives you correctness. You need both, and the design that has only CDC drifts invisibly.

### CDC events arrive out of order for the same document. Walk me through how that corrupts the index and how you prevent it.

The corruption is straightforward and permanent: update v5 lands, then delayed update v3 lands, the index now holds v3's vectors, and no future event will fix it because v5 will never be re-sent. You have a silently stale document forever. This is worse than a lag, because lag heals and this doesn't.

Two mechanisms cause it. First, **partition-level parallelism**: if you consume a Kafka topic with 32 partitions across 32 workers, ordering is guaranteed *within* a partition only, which is fine as long as the partition key is `doc_id`. If someone "improved throughput" by keying on a random UUID or round-robin, ordering is gone. Check this first. Second, **retry reordering**: event v3 fails, goes to a retry queue with backoff, and lands after v5 succeeded. This is the sneaky one, because the partition key was correct and the reordering happened in *your* error handling.

The fix that actually works is to make writes **idempotent and monotonic** rather than to try to enforce global ordering — the same conclusion you'd reach designing any at-least-once sink.

Carry a monotonic version on every record: the source LSN, or the source's `updated_at` as a hybrid logical clock, or a per-doc sequence number. Then the write is conditional:

```sql
MERGE INTO embeddings t
USING staged s ON t.chunk_id = s.chunk_id
WHEN MATCHED AND s.source_version > t.source_version THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
-- note: no unconditional UPDATE. A stale event matches but does not apply.
```

Now v3 arriving after v5 is a no-op instead of a corruption, and you have made the operation **commutative** — apply the events in any order and you converge on the highest version. That is the property you want, because it also makes replay safe.

The vector index side is harder, because most vector stores have no conditional upsert. Two workable options: (a) do the version check in your writer by reading the current `source_version` from the *table* (which does support conditional merge) and only calling the index when the merge actually applied — the table becomes your ordering authority; or (b) encode the version in the stored metadata and run a periodic sweep that finds index entries whose version is behind the table. I strongly prefer (a): one authority, checked in one place.

**⚠ Trap:** using wall-clock `updated_at` from the source as the version when the source has multiple writers or clock skew. Two updates in the same millisecond, or a NTP correction, and your monotonic guard silently drops a real update. Postgres LSN is genuinely monotonic and is the right choice when you have it; a database-generated sequence is second best; wall clock is a last resort and should carry a tiebreaker.

**🗣 Say this in the room:** "I don't try to guarantee global ordering — I make the write monotonic. Every record carries the source LSN, the merge only applies when the incoming version is strictly greater, and the vector index is only touched when the merge actually changed a row. That makes the pipeline commutative, which means out-of-order delivery and full replay are the same operation."

### A row is hard-deleted in the source. Trace the tombstone all the way to a user's search results and tell me every place it can get stuck.

The deletion path is the single most under-tested path in every RAG system I have reviewed, and it is the one with legal consequences. Trace it stage by stage and name the stall points.

**Source → CDC.** A hard delete emits `op: d`. Two ways this dies: a **soft** delete in the source (`deleted_at` set) emits `op: u` and your handler treats it as an update, re-embedding a row that should vanish. And if the source deletes via a bulk `TRUNCATE` or a partition drop, logical decoding may emit nothing per row at all. Both are real. My rule: the ingestion contract must state explicitly whether deletes are hard or soft, and the soft-delete predicate is part of the contract.

**CDC → Kafka.** Debezium emits the delete event and then a **tombstone** — a record with the same key and a `null` value — specifically so log compaction can eventually drop the key entirely. If your consumer deserializes with a schema that rejects nulls, it throws, retries forever, and blocks the partition. I have seen a partition stuck for eleven days on exactly this.

**Consumer → chunk enumeration.** You need every `chunk_id` for this `doc_id`. If you rely on re-parsing the document you are dead, because the document is gone. Hence the earlier rule: keep your own `doc_id -> [chunk_id]` mapping table, written at ingest.

**Table delete.** A `DELETE FROM embeddings WHERE doc_id = ?` in Delta or Iceberg does not remove bytes; it writes a delete vector or a positional-delete file and the old Parquet remains readable via time travel. **That is a compliance problem, not a bug** — a "deleted" record is recoverable by anyone with `VERSION AS OF`. You must additionally run `VACUUM` (Delta) or expire snapshots plus rewrite data files (Iceberg) inside your retention window, and you must be able to prove you did.

**Index delete.** Vector stores tombstone rather than remove. The vector is filtered from results but the bytes are on disk and in a memory-mapped graph, and depending on implementation may be recoverable. For a corpus with a real deletion SLA, tombstoning is not sufficient and you need a scheduled compaction or rebuild that physically drops it. Know your store's behaviour and write it down.

**Caches.** The document text is now in: the semantic cache, the prefix cache on your inference provider, the retrieval-result cache, the eval dataset, the trace store, the logs, and possibly a fine-tuning dataset. **This is where the deletion actually leaks.** A retrieval cache keyed on query hash will happily serve the deleted chunk for another 24 hours.

**🔍 Failure taxonomy — the five places a delete stalls, in the order I check them:** (1) source emitted an update, not a delete, because the delete is soft; (2) consumer crashed on the null-valued tombstone; (3) `doc_id -> chunk_id` mapping was never persisted, so nothing knew what to delete; (4) index deleted but caches not invalidated; (5) everything deleted but time-travel and trace stores still hold the text.

**🏋 Drill (20 minutes, unaided):** write the deletion runbook for one document as a checklist of concrete operations against: chunks table, embeddings table, vector index, retrieval cache, semantic cache, trace store, eval datasets, and object-storage originals — plus the verification query for each that proves it's gone. Pass criterion: eight items and eight verification queries. If you list fewer than six stores you have not thought about where the text actually went, and that is exactly the gap a Harvey or Glean interviewer is probing for.

### You shipped a broken chunker at 09:00 and noticed at 15:00. Six hours of bad data is in the index. What do you do?

First, stop the bleeding and *do not* start repairing until you have. Disable the ingestion DAG, not just the deploy — a rollback of the code with the DAG still running means the next scheduled run reprocesses with good code but the bad rows stay. Then answer one question before anything else: **is the bad data serving?** If the alias points at an index containing bad chunks, flip the alias to the previous index immediately. Repair is a datafix; serving bad answers is an incident. Separate them.

Now the repair, and this is where the earlier design decisions pay off or don't.

**Step 1 — bound the blast radius by version, not by time.** If `chunk_id` includes `chunker_version` and every row carries `ingested_at` plus the git SHA of the pipeline, then the affected set is `WHERE pipeline_sha = '<bad>'` — exact, not approximate. If you only have timestamps you have to use `ingested_at BETWEEN '09:00' AND '15:00'`, which over-selects (rows that were fine) and under-selects (retries that landed at 15:20). **This is the entire argument for stamping a build identifier on every derived row**, and it is a two-column change that saves you a day during an incident.

**Step 2 — decide repair vs rebuild by arithmetic.** Suppose six hours of ingestion produced 3.2M chunks. Targeted repair: re-chunk and re-embed 3.2M chunks = 3.2M ÷ 500/s = 6,400 GPU-seconds = **1.8 GPU-hours ≈ $2**, plus the merge and index upserts. Full rebuild of 100M: 55.6 GPU-hours ≈ $61 and ~2 hours wall-clock on 32 GPUs. Both are cheap; the deciding factor is **confidence**, not cost. If I can prove the bad rows are exactly identifiable, I repair. If the chunker bug also corrupted the `doc_id -> chunk_id` mapping — meaning I cannot enumerate what to delete — I rebuild, because a repair that leaves orphans is worse than a rebuild that costs $60. I say that out loud in interviews: **at these volumes compute is cheap enough that I buy certainty with it.**

**Step 3 — replay from the source, not from the intermediate.** The bad chunker's output is poison; you cannot fix chunks by transforming chunks. Replay must start at the raw documents table (which you kept — this is why you keep it) or, if you didn't, re-fetch from the source connectors with a cursor rewound to 09:00. Rewinding a Kafka consumer group offset or resetting a Debezium slot position is the mechanism; if retention is 7 days you're fine, if it's 24 hours and you noticed on Monday about Friday you are re-crawling the source system and apologizing to their rate limiter.

**Step 4 — verify with a diff, not a vibe.** Before promoting: run the retrieval eval probe set against the repaired index and compare Recall@10 to the pre-09:00 baseline. Then sample 50 repaired chunks and diff their text against a freshly-parsed source. A repair that "looks done" and is missing 4% is a repeat incident.

**⚠ Trap:** repairing forward by re-running the pipeline in append mode. You now have both the bad chunks and the good chunks in the index, because the bad ones have different `chunk_id`s (different chunker version!) and nothing deletes them. The repair must be **delete-then-insert scoped by the bad build id**, and the delete must run against both the table and the index. I have watched a team celebrate a repair that doubled their index size.

**🗣 Say this in the room:** "First I flip the alias back — repair and serving are different problems. Then I scope the blast radius by pipeline build SHA rather than by timestamp, delete-then-reinsert that exact set from the raw document table, and gate promotion on the retrieval eval probe set matching the pre-incident baseline. The reason that's fast is that every derived row carries the build id that produced it."

### Airflow, Dagster, or Prefect for this pipeline? I want an actual opinion, not a comparison table.

Dagster, if I am choosing greenfield, and the reason is not features — it is that **Dagster's unit of scheduling is an asset, and my unit of correctness is also an asset.** The whole thesis of this section is that the index is a derived dataset with a version and a lineage; a framework whose primitive is "the chunks table for source X at version N" rather than "run this Python function at 3am" removes an entire class of translation error. When a downstream eval fails I want to ask "which upstream asset materialization produced this," and in an asset-oriented orchestrator that is a first-class query rather than a log grep.

Airflow is the right answer when the organization already runs Airflow at scale, which is often, and the honest gap has narrowed — Airflow's dataset/asset scheduling gives you data-aware triggering, so "run the index build when the embeddings asset updates" is expressible rather than "run at 04:00 and hope the 03:00 job finished." (**📅 Volatile:** Airflow's asset/dataset API surface has moved across recent major versions; verify the current spelling before you write it on a whiteboard.) The thing I would *not* do is pick Airflow and then express data dependencies as cron offsets, because that is how you get a 04:00 index build reading a 60%-complete embeddings table twice a month.

Prefect I reach for when the workload is dynamic and Python-native — fan-out over a number of documents you don't know until runtime, with retries and concurrency limits expressed in code. It is the lightest to adopt and the weakest on lineage, so it fits a team of three shipping fast and fits a data platform team poorly.

The decision rule I'd state:

- Already on Airflow with a platform team → **Airflow**, but use asset-based triggering and stop scheduling by clock.
- Greenfield, lineage and data-quality gates are the point → **Dagster**.
- Small team, dynamic fan-out, no central data platform → **Prefect**.
- On Databricks and staying there → **Databricks Workflows/DLT**, because fighting the native scheduler to get lineage the platform already gives you is a bad trade.

What matters far more than the choice: the orchestrator must express **"this run consumed table version 4,127 and produced index build 8829"** as data, not as a log line. Whichever tool you pick, if you cannot answer "what exactly went into this index" from a query, you picked wrong regardless of the logo.

**⚠ Trap:** running the embedding job *as* an orchestrator task. Airflow and Prefect workers are not GPU schedulers and their retry semantics are wrong for a 6-hour distributed job. The orchestrator's job is to submit the Ray or Spark job, poll it, and record the resulting version — a thin control-plane task. Teams that run heavy compute inside the scheduler get scheduler outages, and the failure mode is that the whole platform's DAGs stop while one embedding job eats the worker pool.

### Describe the promotable unit. What exactly gets promoted, and what's the gate?

The promotable unit is **one immutable bundle**: `(chunks table version, embeddings table version, embed model id + weights hash, chunker version, index build id, index build parameters)`. Not a file, not a job run — a manifest row, written to a registry table, that names every input and every artifact. Promotion is a single write that points the serving alias at that manifest. Rollback is a single write that points it back.

That framing matters because the failure it prevents is subtle: teams promote *stages* independently. The embeddings job succeeds so the embeddings land in the serving table; the index build fails; now the table and the index disagree and there is no single thing to roll back. Bundle it, and you inherit the property you already rely on in a container deploy — the artifact you tested is bit-identically the artifact you shipped.

The pipeline stages inside the unit:

```
ingest → parse/chunk → embed → build index → EVAL → alias swap
                                                ↑
                                        the gate lives here
```

The gate is not "the job exited 0." It is three checks, all of which must be automated and blocking:

**1. Volume and completeness.** Row counts within tolerance of the previous build (I use ±2% and require a human ack outside that), reconciliation anti-join returns fewer than 0.1% missing, zero null/NaN vectors, zero duplicate `chunk_id`s.

**2. Retrieval quality on a frozen probe set.** A fixed set of 300–1,000 (query, relevant_chunk_ids) pairs, ideally harvested from real traffic and real feedback. Compute Recall@10 and nDCG@10 against the new index. Gate: **no more than a 1-point absolute drop from the currently-serving index**, computed on the *same* probe set. Because index builds are not byte-deterministic, this is a statistical gate and you need to know its noise floor — run the same build twice and measure the variance, or you will chase 0.3-point "regressions" that are just graph construction randomness.

**3. End-to-end answer quality on a smaller set.** 50–150 cases through the actual RAG chain with an LLM judge or exact-match on extractable answers. Slower and noisier, so it is a second-tier gate: a drop blocks, but the primary signal is #2.

**⚠ Trap:** evaluating the new index against a probe set whose `relevant_chunk_ids` came from the *old* chunker. If the promotable unit includes a chunker change, the ids don't exist in the new index and Recall@10 reads ~0. Your gate fires, everyone assumes the pipeline is broken, and after a day someone disables the gate — which is much worse than never having had it. The fix is the remap table from the chunking question, applied to the probe set as part of the promotion.

**📐 Numbers you must know:** with a 300-query probe set, the standard error on a Recall@10 of 0.80 is √(0.8 × 0.2 ÷ 300) ≈ **0.023**, so a 95% CI is roughly ±4.5 points. That means a 300-query probe set **cannot** reliably detect a 2-point regression. To detect 2 points you need n ≈ 0.8×0.2 ÷ (0.01)² ≈ **1,600 queries** for a ±2-point CI. Say this number out loud when someone proposes a 50-query gate: at n=50 the CI is ±11 points and the gate is decorative.

### Give me the alias swap. How do you actually cut over an index with zero downtime and a real rollback?

Alias indirection, exactly like a blue-green deploy or a Postgres index rebuild followed by a rename inside a transaction. The serving layer never names a physical index; it resolves a logical name through a pointer it can re-read.

The mechanism, concretely:

```
Physical:  chunks_prod_2026_08_02_a  (embed_model=E5, chunker=v7, build=8829)
           chunks_prod_2026_07_29_a  (previous, kept warm)
Alias:     chunks_prod  ->  chunks_prod_2026_08_02_a
```

Where the alias lives depends on your store. Elasticsearch/OpenSearch have native aliases with an atomic multi-action update — the ideal case, one API call swaps and it is transactional. Most dedicated vector databases have collections but no alias primitive, so the alias is **a row in your own control-plane store** (Postgres, or a Redis key with a Postgres backstop), read by the query service.

The three details that make it real:

**1. Readers must resolve the alias per request, with a short cache.** If the query service resolves at startup, a swap requires a rolling restart and you have converted a 1-second cutover into a 5-minute one. I cache the alias for 5–10 seconds and accept that a swap takes up to 10 seconds to fully propagate. Crucially, an in-flight request must **finish against the index it started on** — resolve once at the top of the request, pass it down. Otherwise a multi-hop agent does its first retrieval against the old index and its second against the new one, and if the chunker changed, the `chunk_id`s it collected in step one no longer resolve in step two.

**2. Keep N−1 hot, not just present.** Rollback is only fast if the previous index is still loaded and serving-warm. A cold HNSW index at 100M vectors takes minutes to page in from disk, so a "rollback" that requires a load is a 10-minute outage. I keep the previous build fully resident for 48 hours. **💰 Math:** 100M × 1024 dims × 4 bytes = **409.6 GB** of raw fp32 vectors, plus HNSW graph overhead of roughly M × 4 bytes/neighbour — at M=32 that's ~128–256 bytes/vector = 12.8–25.6 GB. Call it **~435 GB** per copy. Two copies = 870 GB, so you're provisioning ~1 TB of RAM instead of 512 GB. On r6i-class instances at roughly $4/hr for 512 GB (**📅 Volatile:** verify current pricing), that's ~$2,900/month per copy, so the rollback insurance costs about **$2,900/month**. Store the vectors as fp16 and it halves to 205 GB and ~$1,450 — which is usually the right call, since fp16 costs you well under a point of recall on normalized embeddings.

**3. Warm before you swap.** Run the probe set (and ideally a replay of the last hour of real queries) against the new index *before* the alias moves. This both validates and pages the graph in. Swapping to a cold index turns p99 from 40ms into 800ms for ten minutes and looks exactly like an outage.

**🗣 Say this in the room:** "The serving layer never names a physical index. It resolves a logical alias, once per request, from a control-plane row. Promotion is one write, rollback is one write, and I keep the previous build hot for 48 hours so rollback is a second rather than a cold-start. The cost of that insurance is one extra copy of RAM, which at 100M vectors in fp16 is about 205 GB."

### Re-embed 100 million chunks with a new model, in production, without a quality dip. Give me the whole plan with cost and wall-clock.

This is the set-piece question for a Databricks or Snowflake loop, so I'd answer it as a plan with numbers at every step.

**Premise:** current index is model A at 768 dims; the new model B is 1024 dims and offline benchmarks suggest +4 points nDCG@10 on our own probe set. Corpus: 100M chunks, ~512 tokens each, growing at 5M chunks/day.

**Phase 0 — prove it's worth it (2 days, ~$50).** Sample 2M chunks stratified by source. Embed with B, build a throwaway index, run the frozen probe set. Gate: **≥3 points nDCG@10 improvement on our data**, not on MTEB. If a 2M-chunk sample shows +0.5, stop — you just saved a week and a five-figure migration. Cost: 2M ÷ 500/s = 4,000 GPU-s = 1.1 GPU-hr ≈ $1.20 of compute, and two days of an engineer, which is the real cost.

**Phase 1 — dual-write the new model from now on (1 day of work).** Before touching history, make every *new* chunk get embedded by both A and B and written to two tables (`embeddings_a`, `embeddings_b`) or one table with an `embed_model` column and a composite key. This means the backfill's finish line stops moving. Without dual-write you are chasing a corpus growing at 5M/day with a backfill running at some rate, and if your rate is close to the growth rate you never converge. **Cost:** doubles the incremental embed spend from ~$3/day to ~$6/day of GPU. Irrelevant. Do it first.

**Phase 2 — backfill history (wall-clock is the number that matters).**

**💰 Math:** 100M chunks at 500 chunks/s/GPU. Total = 200,000 GPU-seconds = **55.6 GPU-hours**.
- On 32 A100s: 55.6 ÷ 32 = **1.74 hours** wall-clock. Spot at $1.10: 55.6 × 1.10 × 1.15 (preemption redo) = **$70**.
- On 8 GPUs: **7 hours**. Same dollar cost, different window.
- If you must use a hosted API at $0.02/Mtok: 51.2 Btok × $0.02 = **$1,024**, but the binding constraint is the rate limit — at 5M TPM it's 51.2×10⁹ ÷ 5×10⁶ = 10,240 minutes = **7.1 days**. Get the limit raised or self-host; do not plan a 7-day critical path.
- **Storage:** 100M × 1024 × 4 bytes = **409.6 GB** for the new vectors in the table (fp32 in Parquet, compressing poorly — floats don't compress). At $0.023/GB-month for object storage that's **$9.42/month**. Storage is never the issue; RAM for the index is.
- **Index build:** at 100M vectors, HNSW construction is typically 3–10× the cost of the embedding pass in CPU-hours and is the stage people forget. Budget a separate 4–12 hours on a large-memory machine, or use a sharded parallel build.

Total for phase 2: **~$70–120 of compute, ~2 hours of GPU wall-clock, plus 4–12 hours of index build.** Say those numbers; the total being small is itself the insight — the expensive part of this migration is validation, not compute.

**Phase 3 — shadow reads (3–7 days).** Serve from A. For a sampled 5–20% of live queries, also query the B index, log both result sets, and score offline. This is where you find the regressions your probe set missed. Compare: overlap@10 between A and B results, per-source recall, and LLM-judge answer quality on the subset. Cost is one extra vector search per sampled query — pure CPU, negligible.

**Phase 4 — canary by tenant, then alias swap.** Route 1% → 10% → 50% → 100% of tenants to B, watching the online proxy SLIs (empty-retrieval rate, citation-click rate, thumbs-down rate, escalation rate) for 24 hours at each step. Keep A hot the whole time.

**Phase 5 — decommission A after 14 days.** Not 1 day. Quality regressions in retrieval surface on a weekly cycle because they hit rare query classes.

**⚠ Trap:** the dimension change. 768 → 1024 means a different index, a different memory footprint (409.6 GB vs 307.2 GB for the vectors alone), and — critically — **you cannot mix**. Any code path that reads a vector without checking `embed_model` will compare across spaces and return garbage silently. I make the model id part of the index name and part of every stored vector's metadata, and I add an assertion in the query path that the query embedder's model id equals the index's model id. That assertion has caught real bugs for me; it costs a string compare per query.

**🗣 Say this in the room:** "Dual-write first so the finish line stops moving, then backfill — which for 100M chunks is about 56 GPU-hours, under two hours on 32 GPUs and roughly $70 on spot. The compute is cheap; the schedule is dominated by shadow reads and canary, which I run for a week and two weeks respectively, because retrieval regressions show up on rare query classes and a 24-hour canary won't see them."

### During that migration you're running dual writes and shadow reads. What exactly do you compare, and what's your promote/rollback threshold?

Comparing "is B better than A" is the easy half. The half candidates miss is that **you must compare on the distribution you actually serve, segmented, because a migration that improves the mean and destroys one source is a worse outcome than no migration.**

The four comparisons, in order of how much I trust them:

**1. Offline retrieval metrics on the frozen probe set (highest trust, lowest coverage).** Recall@10, nDCG@10, MRR. You have ground-truth labels here, so this is the only measurement that is unambiguously *correct*. Its weakness is coverage — a 1,000-query probe set does not contain your long tail.

**2. Shadow-read result overlap (highest coverage, no ground truth).** For each sampled live query, compute Jaccard overlap of the top-10 chunk sets from A and B. Note this needs the chunk-id remap if chunking changed. Overlap tells you *magnitude of change*, not direction. My rule of thumb: overlap@10 in the **0.55–0.80** band is a healthy model upgrade — the model genuinely changed something. Above 0.95 means you did a lot of work for nothing and should ask whether the migration is worth the risk. **Below 0.35 is a red flag**: either you have a bug (wrong normalization, wrong dimension, mismatched query embedder) or the change is so large that your existing prompt engineering, reranker thresholds, and chunk-count settings are all now mistuned.

**3. LLM-judge on the divergent subset only.** Do not judge all 20,000 shadowed queries — judge the ~30% where the top-3 differ, because the rest are ties by construction. **💰 Math:** 20,000 sampled queries × 30% divergent = 6,000 judgments, each ~3k input / 300 output tokens. At $3/Mtok in and $15/Mtok out: 6,000 × (0.003 × $3 + 0.0003 × $15) = 6,000 × ($0.009 + $0.0045) = **$81**. That is nothing, and it's the strongest evidence you'll get. Judging all 20,000 costs $270 and adds no information.

**4. Online proxy SLIs during canary.** Empty-retrieval rate, mean top-1 similarity score, citation-click-through, thumbs-down rate, human-escalation rate, and answer latency. These are noisy and lagging but they're the only things measuring the actual product.

**The segmentation is mandatory.** Break every metric by `source_id`, by document language, by document age, and by query length bucket. The classic failure: the new model is trained with more multilingual data and improves aggregate nDCG by 4 points while dropping recall on your German legal corpus by 9, because the old model happened to be better tuned for that domain's vocabulary. Aggregate metrics hide this perfectly.

**My thresholds, stated as a decision procedure:**
- Promote if: probe-set nDCG@10 up ≥2 points, **no segment** down more than 2 points, empty-retrieval rate not up more than 10% relative, and 24 hours of canary at 10% with no SLI regression outside noise.
- Hold if: aggregate up but a segment is down 2–5 points → investigate that segment, possibly ship a per-source hybrid (keep A for that source; yes, running two models for one corpus is acceptable and I've done it).
- Roll back immediately if: empty-retrieval rate up more than 25% relative, or any segment down more than 5 points, or p99 search latency up more than 50%.

**⚠ Trap:** running shadow reads with the *production* query embedder against the B index. The query must be embedded by model B to search index B. This sounds too obvious to state, and it is the single most common shadow-read bug, because the query-embedding call is usually buried in a shared retrieval service that takes a text and returns results. The tell is an overlap@10 near zero and a B-side result set that looks like random chunks. If you see near-zero overlap, check this before you check anything else.

### The alias is swapped, everything looked green, and three hours later quality reports come in. Roll it back — and tell me what you can't roll back.

Roll forward the pointer, backward in content: one write to the control-plane row, alias `chunks_prod` → previous build. Because the previous index is hot, search is serving old-but-good results within the alias cache TTL, call it 10 seconds. That part is easy and it's the whole reason for the design.

What you **cannot** roll back is the interesting half, and naming these unprompted is what separates a senior answer:

**1. Writes that happened during the window.** Everything ingested in those three hours was embedded with model B and written to `embeddings_b`. The A-side index has been diverging from the corpus since the swap. If you dual-wrote (you did, phase 1), the A-side is current and rollback is clean. **If you didn't dual-write, rollback means the last three hours of documents are invisible in search.** This is the concrete reason dual-write is not optional — it is not about the backfill, it is about making rollback lossless.

**2. Downstream artifacts derived from B's output.** Any extraction, summarization, or classification your pipeline ran using B-based retrieval during the window is now in tables that don't roll back. If you cached "the summary of ticket 4471" and it was generated from B-retrieved context, the alias flip does not invalidate it. **Version-stamp derived artifacts with the retrieval build id** and invalidate by that id, or you serve B-era outputs from an A-era index indefinitely.

**3. Caches.** The semantic cache and the retrieval-result cache are full of B-era results. Flip the alias without flushing and users keep getting the bad answers for the cache TTL. I include the index build id in every cache key precisely so a rollback invalidates by construction rather than by remembering to flush. That is a five-character change to a cache key that removes a whole class of incident.

**4. User-visible side effects.** If the three hours included an agent that filed tickets, sent emails, or updated a CRM based on bad retrieval, none of that reverses. This is why I gate irreversible actions on a separate confidence path, and why "quality regression" for an agentic product has a higher severity than for a search box.

**5. Feedback data.** Thumbs-downs collected during the window are labeled against B's results. Keep them — they're valuable — but they must carry the build id or they'll poison your probe set with "chunk X is irrelevant" labels that were true only under B's chunking.

**🔍 Failure taxonomy — why quality drops three hours after a green canary, in likelihood order:** (a) a query class that's rare per-minute but common per-day just showed up — the canary window was shorter than the query distribution's period, which is usually 24 hours with a weekly overlay; (b) cache warming masked it — the first hour served cached results from the old index and only hour two-plus hit the new one; (c) a segment you didn't split on, most often a single large tenant whose corpus is stylistically unlike the rest; (d) the index was still building/compacting during the canary and recall improved-then-degraded as the graph settled; (e) it isn't the index at all and something else shipped in the same window — check the deploy log before you blame yourself.

**📐 Numbers you must know:** a query distribution with a 24-hour period needs a canary of **at least 24 hours** to see every class once, and to see a class that appears 5 times/day with any statistical power you need **3–7 days**. A 1-hour canary tests deployment mechanics, not quality. State that distinction explicitly; people conflate them constantly.

### Incremental index maintenance forever, or periodic full rebuilds? Give me the rule.

Both, on different clocks, and the rule is set by three numbers you can actually measure.

**Incremental** (CDC-driven upserts and deletes) gives you freshness measured in seconds and costs nearly nothing per event. **Full rebuild** costs a bounded amount of compute and repairs every form of drift at once. The question is only how often you need the repair.

**The three numbers that set the cadence:**

**1. Churn rate.** Deletions and updates per day as a fraction of the corpus. HNSW accumulates tombstoned nodes; recall degrades roughly with the tombstone fraction because deleted nodes still occupy graph routing positions. My threshold: **rebuild when tombstones exceed ~10% of live vectors.** At 100M vectors with 2M deletes/day, you hit 10M tombstones in 5 days — so weekly. At 50k deletes/day you hit it in 200 days, so quarterly is fine and you're rebuilding for other reasons.

**2. Reconciliation drift.** The nightly anti-join count. If it's consistently under 0.01% you have a healthy incremental path; if it's 0.5% and growing, incremental is losing writes and a rebuild is masking a bug you should fix rather than a cadence you should tighten.

**3. Rebuild cost against the maintenance window.** **💰 Math:** 100M chunks, no re-embedding needed (vectors are in the table) — this is *just* the index build, so 4–12 hours on a big-memory box, maybe $30–100 of compute (**📅 Volatile:** instance pricing). If that fits in a weekend window, weekly rebuild is free insurance and I'd just do it. If it's 30 hours, you need sharded parallel builds before you can afford weekly.

**My default cadence for a 100M-chunk enterprise corpus:** incremental continuously, full rebuild weekly, reconciliation nightly, and a forced rebuild on any chunker or model change. The weekly rebuild is not primarily about recall — it is about **proving the rebuild path works**. A rebuild path you exercise once a quarter is a rebuild path that is broken when you need it during an incident, exactly like a backup you never restore.

**⚠ Trap:** assuming incremental and rebuild produce the same index. They do not — HNSW graph structure depends on insertion order, so an incrementally-maintained index and a freshly-built one over identical data will differ in recall by a point or two, sometimes in favour of the incremental one. If your promotion gate demands the rebuild match the incremental index exactly, it will never pass. Gate on absolute quality against the probe set, with a defined noise floor measured by building the same data twice, not on equality with the previous build.

**🗣 Say this in the room:** "Incremental for freshness, weekly full rebuild for drift repair, nightly reconciliation to detect what incremental lost. I set the rebuild cadence off the tombstone fraction — I rebuild before dead nodes exceed about 10% of the graph — and I rebuild weekly regardless of need, because a rebuild path you only exercise during incidents is a rebuild path that doesn't work."
### What is a data contract for an LLM input? Be specific — what's in the document and what enforces it?

A data contract is the same object you already know from event-driven systems — a producer's binding promise about a payload's shape, semantics, and quality, enforced in CI and at runtime — with three fields added that only matter because a language model is the consumer.

The standard fields, which I'll list quickly because you already own them: schema (Avro/Protobuf/JSON Schema), primary key and uniqueness, nullability, allowed enum values, ownership and on-call rotation, SLA on freshness and completeness, and a versioning policy with a deprecation window.

The three that are specific to an LLM consumer:

**1. Text quality guarantees, because the model cannot signal a parse error.** A `NOT NULL` constraint doesn't help when the field contains `"\n\n\n[Object object]\n\n"`. So the contract specifies: encoding is UTF-8 and validated; the field is plain text or a named markup dialect, not a mystery; boilerplate (nav chrome, email signatures, legal footers) is stripped by the producer or explicitly declared as present; and a minimum information threshold — I use "at least 20 non-whitespace characters and at least 3 distinct tokens," which cheaply catches the whole class of empty-after-parse rows.

**2. A declared PII and sensitivity class per field.** The consumer needs to know, at ingest, whether this text may be sent to a third-party API, whether it may be logged in a trace, and what its retention is. If that classification is not in the contract it will be decided ad hoc by whoever writes the retrieval service, and it will be decided wrong.

**3. A stability promise on the *identity* fields.** `doc_id`, `source_id`, and `modified_at` must be stable and monotonic respectively. This is the one producers break most often — a source system "improves" its ids during a migration and your entire index becomes orphaned rows with no deletion path.

Enforcement is in three places, and a contract with only one is theatre:

```python
# 1. CI: schema compatibility check on every producer PR (Buf/Avro compat, or
#    a JSON Schema diff gate). Blocks a breaking change from merging.
# 2. Ingest-time validation: quarantine, not drop.
class DocumentIn(BaseModel):
    doc_id: str
    source_id: str
    modified_at: datetime
    text: str = Field(min_length=20)
    sensitivity: Literal["public", "internal", "confidential", "restricted"]
    lang: str | None = None

    @field_validator("text")
    @classmethod
    def has_signal(cls, v: str) -> str:
        if len(set(v.split())) < 3:
            raise ValueError("insufficient distinct tokens")
        return v
# 3. Continuous: distributional checks on the landed table (next question).
```

**⚠ Trap:** validating and *dropping*. A dropped row is a silent hole in the index that nobody can find later. Every rejection goes to a quarantine table with the raw payload and the violated rule, the quarantine rate is a monitored metric, and reprocessing quarantine after a fix is a first-class DAG. You already do this with a Kafka DLQ; the only new thing is that the consequence of dropping is invisible instead of a missing order.

**🗣 Say this in the room:** "A data contract for an LLM input is a normal data contract plus three things: a text-quality floor because the model cannot raise a parse error, a per-field sensitivity class so the retrieval layer isn't guessing what may leave the VPC, and a stability promise on the identity columns so deletes stay possible. Violations quarantine with the payload; they never drop."

### Now the other direction — the LLM's *output* is a column in a table. What does a contract for extracted fields look like, and how do you evolve it?

This is the harder and more interesting case, because the producer is nondeterministic and the schema is defined by a prompt.

The mental model: **the extraction prompt is code that produces a schema-conforming record, and it must be versioned, tested, and released exactly like a serializer.** Once you say that out loud, the whole design follows — the contract is the JSON Schema you pass to structured output, the tests are a golden set of documents with expected extractions, and a change to the prompt is a release with a compatibility classification.

The contract for an extracted table has the normal shape plus:

- **A confidence or abstention field per extracted value.** Not the model's self-reported confidence, which is poorly calibrated — I mean an explicit `null` plus a `reason` enum (`not_present`, `ambiguous`, `low_quality_source`). If the schema doesn't allow "I don't know," the model will invent, and the downstream consumer cannot distinguish invented from extracted.
- **A provenance span per extracted value.** `(chunk_id, char_start, char_end)` for the text the value came from. This is what makes an extraction auditable and what lets you compute an automated "is this value actually supported by the cited span" check.
- **The extractor version:** `extraction_schema_version`, `prompt_version`, `model_id`, `model_params_hash`. Stamped on every row.

Compatibility classification, which I enforce the same way I'd enforce Avro compatibility:

- **Backward-compatible (safe, ship it):** adding a new optional field; widening an enum by adding a value that downstream code treats via a default branch; loosening a validation.
- **Breaking (needs a version bump and a migration):** removing a field, narrowing a type, renaming, changing an enum's *meaning*, or — the one people miss — **changing the extraction instruction such that the same document now yields a different value for an existing field.** That last one produces no schema error at all. It is a data-semantics break wearing a compatible-schema costume, and it is the dominant failure in this area.

The mechanism for handling a breaking change: **never mutate in place.** Write extracted values to a table partitioned or keyed by `extraction_schema_version`, run the new extractor alongside the old on new documents, backfill history in the background, and let consumers migrate by selecting a version — the exact shape of the index alias swap, applied to a column instead of an index.

**⚠ Trap:** letting a prompt change ship through the normal code-review path with no data diff. A one-word change to "extract the contract's *effective* date" from "extract the contract's date" is a two-character diff and re-labels a million rows. My review rule: **any diff touching an extraction prompt requires an attached before/after diff on a fixed 200-document sample, with the disagreement rate and 10 sampled disagreements in the PR description.** That is cheap — 200 docs × two extractions × ~$0.01 = ~$4 per PR — and it turns an invisible semantic change into a reviewable artifact.

### Someone tweaked the extraction prompt on Tuesday and the Wednesday finance dashboard moved 8%. Is that a bug or a release, and how do you architect so it never surprises anyone again?

It is a **release**, and the fact that it surprised people is the bug. Concretely: an unversioned, un-diffed, un-announced change to a data-producing transform shipped to production and silently re-labeled a table that a business dashboard reads. In any other data platform that would be an incident with a post-mortem, and the reason it wasn't treated as one is that the transform was "just a prompt" living in a Python string rather than in the dbt model where everyone would have recognized it as ETL.

Three architectural changes so it can't recur:

**1. The prompt is an artifact with a version and a hash, not a string literal.** It lives in a file, it has a semantic version, its SHA is stamped on every row it produces, and the DAG refuses to run if the loaded prompt's hash doesn't match the version in the manifest. This is the same discipline as pinning a container digest rather than a `:latest` tag, and the argument is identical.

**2. Every extraction change runs a shadow diff before merge.** New prompt runs over a frozen 200–1,000-document sample; you compute per-field disagreement rate against the current production extractor and block the merge if any field's disagreement exceeds a threshold without an explicit acknowledgement in the PR. **💰 Math:** 1,000 documents × ~4k input / 400 output tokens × 2 runs = 8M input + 0.8M output. At $3/Mtok in and $15/Mtok out: 8 × $3 + 0.8 × $15 = $24 + $12 = **$36 per gated PR**. Against an 8% swing in a finance dashboard, $36 is free.

**3. Downstream reads a version, not a table.** The extracted table is versioned and the dashboard's query pins `extraction_schema_version = 'v4'`. Shipping v5 does not move the dashboard; migrating the dashboard to v5 is a separate, deliberate, reviewed change with a backfill behind it. This is the single most important structural fix, because it converts "a prompt change silently moved a number" into "a prompt change created a new version nobody is reading yet."

**⚠ Trap:** believing you can avoid this by pinning the *model* and keeping the prompt stable. You cannot. A hosted model behind a stable alias can be updated by the provider, and even with a pinned snapshot, temperature-0 decoding is not bit-deterministic across batch sizes and hardware because floating-point reduction order changes. **📐 Numbers you must know:** in my experience re-running the same extraction prompt over the same 1,000 documents with the same pinned model and temperature 0 gives a field-level disagreement rate on the order of **0.5–3%** — small, real, and enough to move an aggregate. So the architecture must tolerate non-reproducibility rather than assume it away: store the extracted value, don't recompute it on read, and treat any recomputation as a new version.

**🗣 Say this in the room:** "That's a release that skipped release engineering. The prompt is a versioned artifact whose hash is stamped on every row, the change is gated on a before/after disagreement diff over a frozen sample, and downstream pins an extraction schema version — so shipping a new extractor creates a new version rather than moving somebody's dashboard overnight."

### An answer was wrong. Tell me every version you need pinned in that trace to reproduce it a month later.

Reproducibility here means "I can reconstruct the exact inputs and the exact configuration," not "I can reproduce the exact bytes of the output" — the model is nondeterministic and you should say so before someone catches you assuming otherwise. What you're actually reconstructing is the **retrieval and prompt state**, because that's where nearly all wrong answers come from.

The pin list, which I'd write on the whiteboard as a struct:

```
trace {
  # corpus state
  chunks_table_version        # Delta/Iceberg snapshot id
  embeddings_table_version
  index_build_id              # + physical index name resolved from the alias
  alias_resolved_at

  # transforms
  chunker_version
  embed_model_id + weights_sha   # the query embedder AND the index embedder
  extraction_schema_version + prompt_sha   # if extracted fields fed the answer

  # retrieval
  retrieval_config_version     # top_k, hybrid weights, filters, MMR params
  reranker_model_id + version
  retrieved_chunk_ids[] + scores[]   # the actual result, not just the config

  # generation
  generation_model_id          # pinned snapshot, not the moving alias
  prompt_template_version + sha
  sampling_params              # temperature, top_p, max_tokens, seed if offered
  tool_schema_version

  # policy
  tenant_id, policy_bundle_version, feature_flag_snapshot_id
}
```

Two subtleties that matter more than the list.

**First, store the retrieved chunk *ids and text hashes*, not just the ids.** A month later the chunk may have been re-chunked or the document edited; the id may resolve to different text. Storing `(chunk_id, content_sha)` lets you detect "the chunk changed since this answer" — which is frequently the entire explanation for a complaint.

**Second, the alias resolution must be recorded, not re-derived.** If you log `index_alias = "chunks_prod"` and nothing else, then a month later you cannot know which physical build served the query. Log the resolved physical name. This is a one-line change that people skip and then regret during exactly the incident it was for.

**💰 Math on what this costs:** the version block is roughly 400–600 bytes per trace; the retrieved chunk ids and hashes for k=20 are another ~1.5 KB. Call it 2 KB per request. At 10M requests/day that's **20 GB/day = 600 GB/month**. At $0.023/GB-month object storage that's **$13.80/month** if you archive to Parquet, versus a hot observability vendor at $0.10–0.50/GB ingested which would be **$60–300/month** for the same volume — a 4–20× gap on the pins alone, and the gap explodes once you add payloads (the full arithmetic is in the retention question later in this section). So: full-fidelity version pins to cheap columnar storage for 100% of traffic, and sample the *payloads* (prompts, outputs) at 1–5% plus 100% of errors. That split is the whole cost strategy and the arithmetic is what makes it defensible.

**⚠ Trap:** pinning the generation model as `"the-latest-frontier-model"` because that's what your config says. Aliases move. If a provider rolls a point release under the same name, your trace says the answer came from model X and the actual weights were different. Resolve to the concrete snapshot id at request time, log that, and alert when it changes — **the "our quality regressed and nothing shipped" incident is a provider model bump about a third of the time**, and this log line is how you prove it in twenty minutes instead of two days.

### You're running an LLM as a transform over 20 million rows in a nightly batch job. Design it.

The framing that makes this tractable: **it is a UDF with a 2-second p50, a 5% error rate, a per-invocation cost, and a rate-limited external dependency.** You have built exactly this before — an enrichment pipeline calling a third-party API — and every instinct you have is correct. What is different is only that failure is often *silent* rather than a 500, and that cost per row is high enough to appear on the finance dashboard.

The design:

**1. Batch tier, not the synchronous API.** Providers offer an asynchronous batch endpoint at roughly **50% of the synchronous price** with a turnaround measured in hours (**📅 Volatile:** availability, discount, and turnaround SLA vary by provider; verify). For a nightly job you do not need synchronous latency, so declining a 50% discount is just a mistake. **💰 Math:** 20M rows × 3k input / 300 output tokens = 60 Btok in, 6 Btok out. At $3/$15 per Mtok synchronous: 60,000 × $3 + 6,000 × $15 = $180,000 + $90,000 = **$270,000 per run**. At the 50% batch tier: **$135,000**. If that number horrifies you, good — it should, and it's the reason the next paragraph exists.

**2. Reduce the row count before you reduce the price.** $135k/night is not a pipeline, it's a business model failure. The levers, ranked: (a) **only process changed rows** — content-hash memoization takes 20M to maybe 400k/night, a 50× cut to **$2,700**; (b) **cascade** — route the easy 80% to a model at ~1/15th the price and only escalate the hard 20%, giving roughly 0.8/15 + 0.2 = 0.253 of the cost, another **4× cut to ~$680**; (c) **shrink the input** — 3k tokens per row is usually 2k of boilerplate, and trimming to 1.2k cuts input cost 60%. Stack all three and you're at a few hundred dollars a night. **Always show this ladder.** An interviewer asking about LLM-as-a-transform is testing whether you reach for the batch discount and stop, or whether you attack the row count first.

**3. Deterministic output keys.** `output_key = sha256(input_content_sha : prompt_sha : model_id : params_sha)`. Same three-part memo key as embeddings. A rerun with no changes does zero LLM calls and costs nothing.

**4. Retry semantics that distinguish three error classes.** Transport errors (429, 500, timeout) → exponential backoff with jitter, retry up to 5, and respect the provider's `retry-after`; these are free to retry. Content errors (the response didn't parse against the schema, or failed validation) → retry **once** with the validation error appended to the prompt, then quarantine; retrying these 5 times burns money for nothing because the model is deterministic-ish about its failures. Policy errors (refusal, content filter) → do not retry at all, quarantine with the reason, because retrying a refusal is a guaranteed loss.

**5. Per-row cost accounting.** Write `input_tokens`, `cached_input_tokens`, `output_tokens`, and computed `cost_usd` as columns on the output table, plus `tenant_id` and `feature`. This is what makes cost attribution a `GROUP BY` instead of an archaeology project, and it takes four columns.

**⚠ Trap:** unbounded concurrency into the provider. Your Spark job has 2,000 tasks and each opens 50 connections; you hit the rate limit, every task gets 429s, your retry logic amplifies the load, and you have DDoSed yourself into a 4× cost increase from retries that all failed. Put a **token-bucket rate limiter keyed on tokens-per-minute, not requests-per-minute**, in front of the whole job — the provider limits you on TPM and your rows have variable token counts, so an RPM limiter will let you sail past the actual limit. You already know how to build this in Redis; it's the same limiter, different unit.

### One document poisons your LLM transform — every retry fails, and the job never finishes. Handle it.

A poison row is a DLQ problem you already know, with two twists: the failure may be *expensive* rather than fast, and it may be *partial* rather than an exception.

The taxonomy, because the handling differs per class:

**Class 1 — oversized input.** A 400-page PDF whose extracted text is 900k tokens, submitted to a 200k-token context. Fails instantly with a clear error. Handling: pre-validate token count at ingest and route oversized documents to a map-reduce path (summarize chunks, then summarize summaries) or quarantine. **This should never reach the LLM** — it's a cheap `len(tokenizer.encode(text))` check, and a contract violation.

**Class 2 — persistent refusal.** The document contains content that trips the provider's safety filter (a security incident report, a medical record, a legal document quoting something ugly). Every retry refuses. Retrying five times costs five times as much for a guaranteed zero. Handling: detect refusal explicitly (a classifier on the response, or a stop-reason field if the provider gives one), mark `status='refused'` with the reason, and **do not retry**. Route to a fallback path — a different model, a stricter extraction that avoids the sensitive span, or human review.

**Class 3 — schema-unsatisfiable.** The document genuinely lacks the fields your schema requires as non-null, so structured output either fails or hallucinates. Handling: this is a schema bug. Every extracted field should be nullable with an abstention reason. If your schema requires `contract_value`, a document that is not a contract will fail forever.

**Class 4 — timeout on a long generation.** The row triggers a long output, the request times out at 60s, retries time out again. Handling: cap `max_tokens` explicitly, and treat a timeout as a signal to route to a longer-timeout lane rather than to retry in place.

**Class 5 — the silent one: it "succeeds" and returns garbage.** No exception, valid JSON, meaningless values. No retry logic catches this. Handling: post-hoc validation — cross-field consistency checks, provenance-span verification (does the cited span actually contain the extracted value?), and distributional monitoring on the output columns. This is the class that actually hurts you, and it is why the data-quality section of this pipeline monitors *outputs*, not just inputs.

The mechanical guard, which I insist on in review:

```python
MAX_ATTEMPTS = {"transport": 5, "schema": 2, "refusal": 0, "oversize": 0}

def process(row):
    key = output_key(row)
    if store.exists(key):            # memo: reruns are free
        return
    for attempt in range(1 + MAX_ATTEMPTS[classify_pre(row)]):
        try:
            out = call_llm(row)
            if validate(out, row):   # includes provenance-span check
                store.put(key, out); return
            err_class = "schema"
        except ProviderError as e:
            err_class = classify(e)
        if MAX_ATTEMPTS[err_class] == 0:
            break
    quarantine.put(row.id, err_class, last_error, attempts, cost_so_far)
```

**⚠ Trap:** a global retry cap that lets one bad partition consume the budget. I put a **circuit breaker per error class and per source**: if quarantine rate for one `source_id` exceeds 5% in a 10-minute window, stop processing that source and page, because it is almost always a connector regression producing garbage text rather than 5% of that source's documents being individually cursed. Without this, a broken PDF parser on one connector will happily spend $40,000 producing extractions of `"���"`.

**💰 Math on why the breaker pays for itself:** 200k rows/hour × 5% quarantine × 3 retries × 3k tokens × $3/Mtok = 200,000 × 0.05 × 3 × 0.003 × $3 = **$270/hour burned on rows that will never succeed**. Over a weekend nobody watches, that's $13,000. The breaker is twenty lines.

### How do you make a nondeterministic transform idempotent for reruns? The model gives different output each time.

You don't make the *model* deterministic — you make the *pipeline* idempotent by memoizing on inputs and treating the stored output as authoritative. The distinction matters and stating it clearly is the answer.

The memo key is `sha256(input_content_sha : prompt_sha : model_id : params_sha : schema_version)`. If that key exists in the output store, the pipeline does not call the model; it reads. So a rerun of a completed partition costs zero dollars and produces bit-identical output, which is exactly the idempotency property you need for an at-least-once orchestrator that will absolutely re-run tasks. Note what's in the key: any change to the prompt, the model, the sampling params, or the schema mints new keys, which is correct — those are different functions.

Three things people get wrong:

**1. Writing the memo before the write is durable.** Store the output and the memo entry in the *same* atomic write. If you `MERGE` the output row and separately mark it done, a crash between them leaves you either double-paying or, worse, marked-done-with-no-output. In Delta/Iceberg the memo *is* the presence of the row, keyed on `output_key` — one write, no second bookkeeping table. That's the clean design and I'd push back on any proposal that introduces a separate "processed" table.

**2. Assuming temperature=0 gives you reproducibility, so you can skip the memo.** It does not. Floating-point reduction order varies with batch size and kernel selection, so the same prompt against the same weights can produce a different token at a near-tie, and one different token can cascade. Providers do not guarantee bit-reproducibility even with a seed parameter, and where a seed exists it's usually documented as best-effort. **📅 Volatile:** determinism guarantees are a moving target across providers; verify rather than assume. Design as if there is none.

**3. Rerunning to "fix" a suspect row without changing the key.** If you delete the memo and rerun, you get a *different* value, and now the row's history has two answers with no version distinguishing them. My rule: **a rerun that is expected to produce a different value must change the key**, by bumping a `rerun_epoch` component that is stamped on the output row. Then "why did this value change" is answerable.

**What about when you genuinely want to reduce variance?** Self-consistency: run k=3 or 5 samples at temperature ~0.7 and take the majority per field. It measurably improves extraction accuracy on ambiguous documents, and it costs k×. **💰 Math:** at 400k rows/night, 3k in / 300 out, $3/$15 per Mtok, single-shot is 400,000 × (0.003×$3 + 0.0003×$15) = 400,000 × $0.0135 = **$5,400/night**. At k=3 it's **$16,200/night** — but note the input can be cached across the 3 calls, and with a 90% cache discount on input the marginal calls cost 400,000 × 2 × (0.003×$0.30 + 0.0003×$15) = 400,000 × 2 × $0.0054 = $4,320, so the real total is ~$9,720 not $16,200. **That's the number worth knowing: prefix caching makes self-consistency cost roughly 1.8×, not 3×.** Whether that 1.8× buys enough accuracy is an eval question, and I'd only pay it on the subset where the single-shot pass emitted low confidence or the fields disagree with a cheap rule-based extractor.

### What do you monitor on a corpus, as opposed to on the pipeline? Give me the metric list and what each one catches.

Pipeline monitoring tells you the job ran. Corpus monitoring tells you the job ran *correctly*, and it is the difference between finding a problem in an hour and finding it when a customer does. The mental model: these are **distributional assertions on a derived dataset**, exactly what you'd write in Great Expectations or dbt tests, and the reason they're worth enumerating is that the specific distributions that matter for a retrieval corpus are not the ones you'd instrument for a normal ETL job.

**Parse-failure rate, per source, per file type.** Fraction of documents where extraction produced text failing the quality floor. Catches: a connector auth change returning HTML error pages, a PDF library upgrade, a source that started serving a JS-rendered page. Baseline is source-specific — 0.3% for a wiki, 4% for a PDF-heavy legal corpus is normal. **Alert on relative change (2× over the 7-day median), not an absolute threshold**, because absolute thresholds get set once and never revisited.

**Chunk-length distribution.** Track p10/p50/p90 of tokens per chunk. Catches: a chunker regression (p50 collapsing from 380 to 40 means the splitter is splitting on the wrong delimiter — a classic when a source switches line endings or starts emitting `\n` between every word), and a source-format change. This one is high-value because a chunker regression is invisible in every count-based metric: the row count goes *up*, which looks healthy.

**Empty-or-degenerate chunk rate.** Chunks under the token floor, or with a token-to-character ratio suggesting garbage (base64 blobs, minified JS, OCR noise). These embed to near-identical vectors and pollute retrieval, because a degenerate vector sits near the centroid and gets retrieved for everything.

**Language mix, per source.** Fraction by detected language. Catches: a new region's data landing in a corpus whose embedding model is English-tuned. This changes retrieval quality for that segment without changing any aggregate.

**Duplicate and near-duplicate rate.** Exact via `content_sha`, near via MinHash or SimHash. Catches: a connector re-ingesting without dedup, a source with heavy templating. High duplicate rates directly damage retrieval — top-10 fills with five copies of the same paragraph and the answer loses breadth.

**Freshness per source.** `now() - max(source_modified_at)` and the p95 of `searchable_at - source_modified_at`. Catches: a dead connector. This is the one customers notice first and the one most often un-instrumented.

**Vector-space health.** Mean pairwise cosine on a random sample, and the fraction of vectors within ε of the corpus centroid. Catches: a broken embedding call returning a constant vector, a normalization bug, a model mismatch. Cheap: sample 10k vectors nightly.

**Permission-coverage rate.** Fraction of chunks with a resolvable ACL. Anything unresolvable must be treated as private, and a spike means a permission-sync failure — which is a security metric, not a quality one.

**⚠ Trap:** monitoring only the aggregate. Every one of these must be sliced by `source_id`, because a corpus is a union of sources with wildly different characteristics and a new source arriving at 2% of volume can be 100% broken while every global metric stays green. My rule in review: **a corpus metric without a `source_id` dimension is not a corpus metric.**

**🗣 Say this in the room:** "I monitor eight distributions on the corpus itself, all sliced by source: parse-failure rate, chunk-length percentiles, degenerate-chunk rate, language mix, duplicate rate, freshness, vector-space health, and permission-coverage. The single highest-yield one is chunk-length p50 by source, because a chunker regression raises the row count — so it looks healthy on every count-based dashboard while destroying retrieval."

### Parse-failure rate on one connector went from 0.4% to 6% overnight. Walk me through the debug.

Fifteen-fold jump, overnight, one connector — that shape means a *step change from a discrete cause*, not gradual drift. So I'm looking for something that changed at a point in time, and the first thing I do is find the time.

**🔍 Failure taxonomy, in the order I execute it:**

**1. Bound it in time and in space.** Query the quarantine table: failures per 5-minute bucket for the last 48 hours, and group by `source_id`, `file_type`, `error_class`, and `document_size_bucket`. You want a picture in ninety seconds. Three shapes and their meanings: a **clean step at a timestamp** → something deployed (yours or theirs); a **ramp over hours** → a rollout or a growing backlog of a new document type; **only new documents failing, old ones fine on retry** → the source changed what it serves.

**2. Correlate against deploy logs on both sides.** Your deploys, obviously. But also: did the source system upgrade? A Confluence or SharePoint version bump changing the export format is extremely common and you won't have it in your deploy log — you'll have it in their status page or a release note. My first question in the incident channel is always "did anything change on *their* side," because the answer is yes about half the time and nobody thinks to ask.

**3. Read five failures. Actually read them.** Not the error class — the raw bytes of the payload. This resolves it faster than any dashboard, and it's the step engineers skip in favour of more querying. The five most common things you'll see: an HTML login page (auth/token expiry — and note it's a *200 OK*, which is why no HTTP-level alert fired), a JSON error envelope where a document was expected (rate limiting, often with a 200 too), a valid document in an encoding you're not handling, a PDF that's a scan with no text layer, or an empty body with a `Content-Length: 0`.

**4. Check whether it's actually a *failure* or a *detection*.** If you tightened the text-quality floor in a recent deploy, you didn't break parsing — you started detecting a problem that was already there. This is genuinely good news and a different remediation: the 6% were previously being indexed as garbage. Check the validator's git history before you chase the connector.

**5. Check volume, not just rate.** 6% of a suddenly 10× larger denominator is a different story from 6% of the same denominator. A backfill of an old archive with different formatting will move the rate without anything being broken.

The two most likely answers, from experience: **token expiry returning an HTML page with a 200**, and **a new file type entering the corpus** — someone in that department started attaching `.msg` or `.pages` files and your extractor has no handler, so it returns empty text and the quality floor rejects it.

**⚠ Trap:** reprocessing the quarantine immediately after fixing the parser, without checking whether the *documents* also changed in the meantime. You'll re-ingest 6% of the corpus at whatever state they were in when you fetched them, which may be stale by two days. Quarantine reprocessing should re-fetch from source, not replay the stored payload, unless you've confirmed the payload was the problem rather than the parser.

**📐 Numbers you must know:** at 6% parse failure on a source contributing 500k documents/month, that's 30,000 documents invisible in search per month. If that source is one large customer's SharePoint, it is 100% of their data being 6% missing — and their perception is not "94% works," it's "search doesn't find the thing I'm looking for," because the missing 6% is not randomly distributed, it's concentrated in whatever format broke.

### How do you find near-duplicates across 100 million chunks, and why should I care about the ones that aren't exact?

Care first, because the "why" determines the algorithm.

Near-duplicates damage retrieval in a specific, measurable way: they **consume the top-k budget**. If your reranker gets 50 candidates and 30 are the same legal disclaimer from 30 documents, you've effectively retrieved 21 distinct things instead of 50, and the answer loses the breadth that makes it correct. They also skew evals — a probe query whose gold chunk has 12 near-copies scores as a hit no matter which copy comes back, which makes your Recall@10 look better than the product feels. And they cost money: 30 copies of a 500-token boilerplate block embedded 30 times, stored 30 times, and searched 30 times.

Exact duplicates are trivial — `content_sha`, group by, done. The interesting cases are the ones with a changed date, a different customer name, or one edited sentence.

**The algorithm: MinHash + LSH** for Jaccard similarity over shingles, or **SimHash** for cosine-ish similarity over weighted features. (**📄 Paper:** Broder (1997), *On the resemblance and containment of documents* — introduced min-wise hashing, which estimates Jaccard similarity from a constant-size signature instead of comparing full sets. **📄 Paper:** Charikar (2002) — the random-hyperplane sketch behind SimHash, later shown by Manku et al. (2007) to work at web scale for near-duplicate detection.)

The mechanism, because you should be able to state it: shingle the text into overlapping k-grams (k=5 words is a reasonable default for prose), hash each shingle, and for each of P independent hash permutations keep the minimum — that P-element signature estimates Jaccard similarity with standard error ≈ 1/√P. Then band the signature into b bands of r rows (P = b×r) and hash each band; two documents are candidates if any band collides. The probability of becoming a candidate at true similarity s is 1 − (1 − sʳ)ᵇ, which is an S-curve whose knee sits near (1/b)^(1/r) — that's your tuning knob.

**📐 Numbers you must know:** with P=128, b=16, r=8, the knee is at (1/16)^(1/8) = e^(ln(1/16)/8) = e^(−0.3466) ≈ **0.71**. So that configuration catches most pairs above ~0.75 Jaccard and mostly ignores those below ~0.6. Signature size: 128 × 4 bytes = 512 bytes/chunk × 100M = **51 GB** — one Spark job, not a research project.

The practical decision: **don't delete near-duplicates from the corpus — cluster them and pick a canonical.** Deletion is wrong because the copies have different ACLs, different sources, and different `doc_id`s that citations point at. Instead, assign a `dup_cluster_id`, mark one canonical, and apply the deduplication **at retrieval time**: retrieve normally, then collapse results sharing a `dup_cluster_id` to the highest-scoring member that the user is permitted to see. That preserves permissions and citations while fixing the top-k budget problem.

**⚠ Trap:** deduping at ingest and thereby deleting the only copy a particular tenant is allowed to see. In a multi-tenant corpus, "the same text" in tenant A and tenant B are different rows with different ACLs and must both exist. Cross-tenant dedup of *storage* is fine only if the permission filter is applied to the stored row's tenant list rather than to a single tenant column — and getting that wrong is a cross-tenant data leak, which is a Sev-0. My default is: never dedup across tenant boundaries, accept the storage cost, and revisit only with a security review.

### Define a freshness SLA per source and tell me how you'd enforce it when the source is a third-party API you don't control.

The SLA has to be stated in terms the customer experiences, which is **source-modified-time to searchable-time**, at a percentile, per source: "95% of Confluence edits are searchable within 5 minutes; 99% within 30 minutes." Not "we run every 15 minutes," which is an implementation detail that says nothing about whether the pipeline is working.

Measuring it requires that `source_modified_at` be carried untouched from stage 1 through to the moment the chunk becomes searchable, at which point you emit a histogram of `now() - source_modified_at`. If you don't propagate that field, you cannot measure the SLA at all — you can only measure your own internal stage latencies, all of which can be green while a connector has been dead for two days. That's the single most important mechanical point.

Then there's the part you don't control, and the honest answer is that you enforce it with **two independent detectors**, because the failure modes are different:

**Detector 1 — the freshness gauge:** `now() - max(source_modified_at)` per source, alerting when it exceeds a source-specific threshold. This catches a stopped connector. Its weakness is that it produces false alarms on genuinely quiet sources — a wiki nobody edits on a weekend looks identical to a dead connector. So the threshold must be derived from that source's *historical* inter-arrival distribution: alert when the current gap exceeds, say, the p99 of the trailing 30-day gap distribution. That is a five-line query and it eliminates the weekend false positives that get freshness alerts muted.

**Detector 2 — the canary document:** a document you own in each source, touched by a scheduled job every N minutes, whose round-trip through the pipeline you measure directly. This is a genuine synthetic probe and it catches everything the gauge can't: a connector that's running but has lost permission to a specific space, a source that accepts writes but stopped emitting change events, a webhook endpoint whose TLS cert expired. **Every connector integration I've owned has needed this**, and it's the thing candidates never mention.

On the parts genuinely outside your control — the source's rate limits and its own webhook reliability — the engineering move is to make the SLA *conditional and reported*: publish a per-source freshness dashboard the customer can see, with a documented dependency ("Slack's Events API delivery, our p99 5 min conditional on their delivery"). Enterprise customers accept a conditional SLA with visible measurement far better than an unconditional one that quietly misses.

**💰 Math on cursor tightening, since someone always proposes it as the fix:** dropping a polling connector from 15-minute to 1-minute cursors cuts mean detection latency from 7.5 min to 30 s — a 15× freshness win — at the cost of 15× the API calls. If the source allows 10,000 calls/hour and you have 40 tenants, 1-minute polling with 3 calls per poll is 40 × 60 × 3 = 7,200 calls/hour, inside the limit. At 100 tenants it's 18,000/hour and you're throttled, and *throttling makes freshness worse than the 15-minute cursor was*. So the honest answer is: use webhooks where available, poll adaptively (tighten the cursor for sources with recent activity, back off for quiet ones), and compute the tenant count at which your polling budget breaks — because that number is your scaling wall and you should know it before sales does.

**🗣 Say this in the room:** "Freshness is source-modified to searchable, at p95 and p99, per source — anything measured from our own job start is measuring the wrong thing. I enforce it with two detectors: a freshness gauge thresholded on that source's own historical inter-arrival p99 so quiet weekends don't page, and a canary document I edit on a schedule in every source, because a connector that's running but has lost permission to one space is invisible to every other signal."

### Chunk-length distribution drifted: p50 went from 380 tokens to 210 over three weeks. Nobody deployed a chunker change. What happened?

Gradual, no deploy, one-directional — that's composition change, not a code bug, and separating those two hypotheses is the whole answer.

The first thing I do is **decompose the aggregate by source and by ingestion date**, because a corpus-level p50 is a mixture and a mixture can move for two entirely different reasons: (a) the per-source distributions moved, or (b) the mixture weights moved. Plot p50 per source over the three weeks. If every source is flat and only the aggregate moved, it's the mixture — a new or growing source with shorter documents. That's not a bug at all, and the correct response is to stop alerting on the aggregate and alert per-source. If one source's own p50 moved, it's that source.

The realistic causes, roughly in order:

**Mixture shift.** A new connector went live three weeks ago — Slack messages, or Jira comments, or a ticketing system — and it produces short documents at high volume. Slack in particular will do this violently: a channel export is thousands of 30-token messages, and if you're chunking per message your p50 collapses. **The fix is not the chunker, it's the ingestion strategy for that source**: conversational sources should be chunked by thread or by time-window, not by message, because a single message is almost never a retrievable unit of meaning.

**Source format change.** The source started emitting content with different structure — a Confluence upgrade that adds more heading levels, or a documentation site migration to a format with shorter sections. Your recursive splitter respects headings, more headings means more splits.

**A dependency upgrade that isn't "your" deploy.** A tokenizer library bump changing token counts, or a text-extraction library that now emits `\n\n` where it used to emit `\n`. If your splitter treats double newline as a hard boundary, that's a silent chunker change delivered by `pip install -U`. **Check the lockfile diff over the window** — this is why the pipeline's dependency set should be pinned by hash and why I stamp the pipeline image digest on every row.

**A growing tail of degenerate documents.** If parse quality declined gradually, you're producing more short garbage chunks. Cross-check against the parse-failure and degenerate-chunk metrics for the same window; if they moved together, the length drift is a symptom, not the disease.

**Does it matter?** Sometimes not, and I'd say that rather than manufacture urgency. But three things do break, and I'd check each: retrieval Recall@k on the probe set (shorter chunks usually help precision and hurt recall of multi-fact answers); **cost per query**, because if top-k is fixed at 10 and chunks halved in size, you're feeding half the context to the generator and the answers get thinner — that shows up as a quality complaint with no metric attached; and **index size**, since halving chunk length roughly doubles chunk count, doubling vectors, memory, and search cost. **💰 Math:** 100M chunks at 435 GB of RAM in fp32-plus-graph becoming 200M chunks is another ~435 GB, which at ~$4/hr per 512 GB instance is roughly **$2,900/month of additional serving cost** appearing with no deploy and no ticket. That's the number that makes a length-distribution monitor worth having.

**⚠ Trap:** "fixing" this by forcing a minimum chunk length in the splitter, merging short chunks until they hit 380 tokens. On conversational data that concatenates unrelated messages from different authors into one chunk with one blended embedding — semantically incoherent, and it retrieves for everything and answers nothing. The right fix is source-aware chunking, which means the chunker takes `source_type` as a parameter and has genuinely different strategies for prose, conversation, code, and tables. That's more code, and it's the correct code.
### A compliance officer points at one sentence in one answer and asks "where did this come from?" Build me the plumbing that answers that in under a minute.

Lineage for an AI answer is a chain with five links, and the design job is making every link a stored foreign key rather than something you reconstruct by re-running.

**Sentence → citation.** The generator must emit citation markers bound to a `chunk_id`, and you must *verify* the binding rather than trust it — models emit citation markers that look right and point at the wrong chunk at a non-trivial rate. My check is mechanical: for each cited claim, compute whether the claim's key entities appear in the cited chunk, or run a small NLI-style entailment check on (chunk, sentence). Uncited or unverifiable sentences get flagged in the UI rather than silently rendered. If citations aren't verified, the rest of the chain is decoration.

**Citation → chunk.** `chunk_id` resolves in the chunks table to `(text, content_sha, doc_id, char_start, char_end, chunker_version)`. Because you stored offsets, you can render the exact highlighted span in the original document. Because you stored `content_sha`, you can detect that the chunk's text changed since the answer was produced, which is frequently the whole explanation.

**Chunk → document → source row.** `doc_id` resolves to the raw documents table, which carries `source_id`, the source system's native id and URL, the fetch timestamp, the connector version, and the original bytes' checksum. If the source is a database, this is a literal primary key in a literal table.

**Answer → configuration.** The trace's version block — index build id, embed model, prompt version, generation model snapshot, retrieval config — established earlier in this section. Without it you can say where the text came from but not why the system chose it.

**Answer → identity and policy.** Who asked, under which tenant, with which permission set evaluated at which time, and which policy bundle version was in force. A compliance officer's next question after "where did this come from" is always "was this person allowed to see it."

The implementation is unglamorous and that's the point: **one lineage table with foreign keys, written synchronously on the answer path**, not reconstructed from logs. I've seen teams try to rebuild lineage by joining OpenTelemetry spans after the fact, and it works until a span is sampled out or an async boundary loses the parent context, at which point the one trace the regulator asked for is the one that's missing. The version block and the retrieved chunk ids are small — around 2 KB per request — so writing them for 100% of traffic to a Parquet table costs about **$14/month at 10M requests/day** on object storage. Sample the *payloads*; never sample the lineage.

**⚠ Trap:** lineage that stops at the document. "This came from the Q3 policy PDF" is not an answer if the PDF is 200 pages. The unit of lineage is the **span**, with byte offsets, which is why the chunker must emit offsets into the original bytes and why you must keep the original bytes. A team that discards originals after extraction cannot ever produce a defensible audit trail, and discovers this during a customer's security review rather than during design.

**🗣 Say this in the room:** "Lineage is a stored chain, not a reconstruction: sentence to verified citation, citation to chunk with byte offsets, chunk to document to source primary key, plus the version block that says which index and which prompt produced it. I write it synchronously for 100% of requests because the one trace someone asks about is always the one that got sampled out."

### Where does PII classification happen in this pipeline, and what do you do with the label once you have it?

At ingest, before the text is written anywhere durable, and the reason is structural: **the earliest point at which a piece of text exists in your system is the only point at which it exists in exactly one place.** Classify later and you're chasing copies through chunks, embeddings, caches, traces, and index metadata.

The mechanism is layered, cheapest first, because running an LLM classifier over every document is both expensive and unnecessary:

**Layer 1 — source-level declaration.** The data contract says this connector's data is `confidential` and contains `customer_pii`. Most of your classification comes free from knowing which system the data came from. An HR system is PII by construction.

**Layer 2 — deterministic detectors.** Regex and validators for the structured stuff: national ids with checksum validation, card numbers with Luhn, IBANs, emails, phone numbers. High precision, near-zero cost, and it runs on every document. **Checksum-validate** — a bare 16-digit regex will flag every order number in your corpus and drown the signal.

**Layer 3 — an NER model for the unstructured stuff.** Person names, addresses, medical terms, employer names. A small purpose-built model, not a frontier LLM: 100M documents through a frontier API at 3k tokens each is 300 Btok × $3/Mtok = **$900,000**, versus a spaCy-class or small-transformer NER pass at a few documents per second per core on 3k-token text, which is a few thousand CPU-hours and **a few hundred dollars**. That arithmetic decides it — three orders of magnitude is not a close call. Reserve the LLM for a sampled audit of the classifier, not for the bulk pass.

**Layer 4 — LLM adjudication on a sample.** 1,000 documents/week through a strong model to measure the cheap classifiers' false-negative rate. **💰 Math:** 1,000 × 3k in / 200 out at $3/$15 = 3 × $3 + 0.2 × $15 = $9 + $3 = **$12/week**. That's how you keep the cheap layer honest for the price of a lunch.

What you do with the label is where the real design is, and it is four different decisions, not one:

1. **Routing.** Does this text may leave the VPC? A `restricted` label routes to a self-hosted model; everything else can use a hosted API. This must be enforced in the client, not in a doc.
2. **Storage.** Tokenize/redact-at-rest for the highest class, store a reversible mapping in a separate vault, keep the chunk text with placeholders. Note the embedding is then of the *redacted* text, which changes retrieval behaviour — you trade recall for containment, and you should say so rather than pretend it's free.
3. **Trace redaction.** Payload capture applies the same classification. A trace store holding raw payloads is a second, un-governed copy of your entire corpus, and it is the copy that leaks.
4. **Retention.** The class sets the deletion clock, which propagates to every derived table.

**⚠ Trap:** classifying at ingest and never re-classifying. Detectors improve, and a document classified `internal` in January under a weaker detector is still labeled `internal` in June. I keep `classifier_version` on every document and run a **re-classification sweep** when the detector changes, treating it exactly like a re-embedding backfill — same dual-write, same version pin. Without it, your PII coverage is frozen at the quality of the day the document arrived.

### A user exercises their right to erasure. Trace that deletion through every derived artifact and tell me which one you'd forget.

The one people forget is the **trace store**, and second is the **eval dataset**. I'll answer with the full propagation and then say why those two are the ones that bite.

The propagation, as an ordered runbook keyed on `subject_id`:

1. **Resolve subject → documents.** You need an index from person to documents mentioning them. If you only have documents-to-PII-flag and not the resolved entity, you cannot answer this request without a full corpus scan. Building a `subject_id → doc_id` mapping at ingest — as a byproduct of layer-3 NER — is the difference between a 2-minute deletion and a 20-hour one.
2. **Raw documents table + object storage originals.** Delete rows; then `VACUUM` (Delta) or expire snapshots and rewrite data files (Iceberg), because a logical delete leaves the bytes readable via time travel. Delete the original blobs, including any versioned-bucket previous versions — S3 versioning will silently retain them.
3. **Chunks table.** Delete by `doc_id`, using the persisted `doc_id → [chunk_id]` mapping.
4. **Embeddings table.** Delete by `chunk_id`, plus vacuum.
5. **Vector index.** Delete by id — and know whether your store tombstones or physically removes. If it tombstones, schedule the compaction/rebuild that physically drops it and record that it ran.
6. **Retrieval cache and semantic cache.** Keyed on query hash, so you cannot delete by document. This forces either a full flush or a secondary index from `chunk_id → cache_keys`. In practice I set cache TTLs short enough (hours) that a documented "purged within TTL" is defensible, and I flush on high-severity requests.
7. **Extracted-field tables.** Any derived table containing values extracted from those documents. This is the one that multiplies — every downstream table built off the extraction is in scope.
8. **Trace store.** Prompts, retrieved context, and model outputs, all containing the text verbatim. Often in a third-party observability vendor. **This is a full second copy of the deleted content, in a system with its own retention, run by a subprocessor.**
9. **Eval datasets and feedback records.** If a probe-set case cites a deleted chunk, or a golden answer quotes the person, it's in scope — and unlike everything above, it's usually in a git repo or a spreadsheet, where deletion means a force-push and a rewritten history.
10. **Fine-tuning datasets and any model trained on them.** If the text entered a training set, deletion of the artifact is straightforward; deletion from the *weights* is not, and the honest answer is that you must be able to say a model was trained on data as of date D and schedule a retrain. Say this plainly rather than implying weights can be surgically edited.

**Why traces and evals are the forgotten ones:** both are *engineering* artifacts, owned by engineers, outside the data platform's governance perimeter, and neither appears on the architecture diagram the privacy review looked at. I've watched a company pass a deletion audit on their production pipeline while holding six months of full prompt payloads in an observability SaaS with a 13-month retention.

**⚠ Trap:** treating deletion as a fire-and-forget job. Every deletion must produce a **verification artifact**: a per-store query proving zero rows remain, with a timestamp, retained as evidence. "We ran the deletion job" is not what an auditor accepts; "here are ten queries and their zero-row results, dated" is.

**🏋 Drill (25 minutes, unaided):** write the deletion DAG as ten tasks with the exact operation and the exact verification query per store, including the vacuum/snapshot-expiry steps and the trace-store call. Pass criterion: your DAG includes object-storage version deletion, index compaction, and the trace store. Most people write four tasks.

### Retention on derived data — how do you stop a trace store from becoming an ungoverned copy of your entire corpus?

By making the trace store a **structured, mostly-pointer** record rather than a payload dump. That single design choice fixes governance, cost, and deletion simultaneously.

The mechanism: a trace row stores the version block, `retrieved_chunk_ids[]` with `content_sha`s, token counts, timings, costs, and outcome — but *not* the chunk text and *not* the fully-rendered prompt. To replay, you resolve the ids against the governed chunks table. If a chunk was deleted, the replay honestly fails with "source deleted," which is the correct behaviour. Deletion propagates for free because you never held a second copy.

You do need payloads sometimes — debugging a prompt-rendering bug requires the rendered prompt. So the policy is tiered:

- **100% of traces:** pointers, versions, metrics. ~2 KB/request.
- **1–5% of successes + 100% of errors + 100% of user-reported-bad:** full payloads, redacted per the PII classification, with a short retention (7–30 days).
- **Anything flagged `restricted`:** never captured as payload, only as pointers, enforced in the tracing client so it cannot be forgotten at a call site.

**💰 Math, why this is also the cost answer:** 10M requests/day, full payloads averaging 25 KB (a 6k-token prompt plus context plus output) = 250 GB/day = **7,500 GB/month**. At a hot observability vendor's $0.10–0.50/GB ingested, that's **$750–3,750/month**. Pointer-only at 2 KB = 20 GB/day = 600 GB/month, and at 3% payload sampling you add 7.5 GB/day = 225 GB/month → **825 GB/month total**, so **$83–413/month at the same vendor rates** — a 9× cut — or **825 × $0.023 = $19/month** if you land it in Parquet on object storage and query it with your lakehouse engine, which is a **40–200× cut**. Same decision buys the cost win and the governance fix. **📅 Volatile:** observability vendor per-GB pricing moves; verify.

Retention itself is set by the strictest applicable rule per class and must be **enforced by the storage layer**, not by a cron job someone might disable: object-storage lifecycle rules, table partitioning by day so expiry is a partition drop rather than a delete-scan, and TTLs on cache keys. A retention policy implemented as a scheduled Python job is a retention policy that will be found not to have run for four months.

**⚠ Trap:** partitioning traces by `tenant_id` and then discovering you cannot expire by date without a full rewrite. Partition by **date first**, then cluster or sub-partition by tenant. Date-first makes expiry a metadata operation; tenant-first makes it a scan of the entire history. This is a five-minute decision at design time and a multi-week migration afterwards.

**🗣 Say this in the room:** "The trace stores pointers and versions, not payloads — chunk ids with content hashes, resolved against the governed table on replay. That means deletion propagates for free, and it drops trace storage from roughly 250 GB/day to 20 GB/day, which at typical observability pricing is roughly $750–3,750 a month becoming $83–413 — or nineteen dollars a month if I land it in Parquet instead. Payloads are sampled at a few percent plus all errors, redacted by the ingest-time PII class."

### Where does a semantic layer sit relative to the LLM? A PM wants natural-language questions over our warehouse.

Between them, always, and this is one of the few places in applied AI where I hold a genuinely strong opinion: **the LLM must not author metric definitions, it must select them.**

The mental model: text-to-SQL over a raw warehouse asks the model to do two jobs — understand the question, and re-derive your business logic. The second job it cannot do, because the business logic isn't in the schema. "Revenue" in your company means recognized revenue net of refunds excluding intercompany, over a fiscal calendar starting in February, and none of that is inferable from a table named `transactions`. A model asked to write that SQL will produce something plausible and wrong, and — this is the part that makes it dangerous — it will be wrong *consistently and confidently*, so it looks like a working feature.

The semantic layer (dbt's semantic models and MetricFlow, Cube, LookML, or a warehouse-native metrics layer) holds the definitions: entities, dimensions, measures, joins, time grains, and the metric expressions themselves, all version-controlled and reviewed by the people who own those numbers. **📅 Volatile:** the semantic-layer product landscape consolidates frequently; verify which one an employer runs before you name one.

The architecture I'd draw:

```
question → LLM: entity/metric/dimension/filter/grain resolution
        → structured query object (validated against the semantic layer's catalog)
        → semantic layer compiles → SQL → warehouse
        → results + the metric's canonical definition rendered alongside
```

The LLM's output is **not SQL**. It's a small validated object:

```python
class MetricQuery(BaseModel):
    metrics: list[str]          # must exist in the catalog
    dimensions: list[str] = []  # must be valid for those metrics
    filters: list[Filter] = []
    grain: Literal["day","week","month","quarter"] = "month"
    limit: int = Field(default=100, le=10_000)
```

Every field is validated against the catalog before anything executes. An invalid metric name is a clean, correctable error — you can even hand the model the k nearest catalog entries and let it retry — instead of a syntactically-valid query returning a wrong number.

The wins are concrete and worth enumerating because they're what makes this the senior answer: the model's output space collapses from "all of SQL" to "a few hundred catalog entries," which is the difference between a hard generation problem and an easy retrieval-and-fill problem; governance and row-level security live in the semantic layer where they already are; the answer can display the metric's canonical definition, so the user sees *what was computed*; and you get caching and cost control, because a metric query is parameterized and cacheable while free-form SQL is not.

**⚠ Trap:** "we'll give the LLM the full schema and some example queries, it'll be fine." It is fine in the demo, on the three tables you tested. At 400 tables with six overlapping revenue-ish columns, accuracy falls off a cliff and — worse — the errors are silent numbers rather than exceptions. My rule in review: **if there is no semantic layer, the first deliverable of the text-to-NL-analytics project is the semantic layer for the top 20 metrics, not the LLM.** That is an unpopular thing to say to a PM and it is correct.

**🗣 Say this in the room:** "The model resolves intent to a validated metric query; the semantic layer compiles it to SQL. It never writes SQL directly, because the definition of revenue lives in dbt and finance owns it, not in the model's priors. That also collapses the output space from all of SQL to a few hundred catalog entries, which is why accuracy holds at 400 tables instead of 4."

### Two teams' dashboards disagree by 8% and the culprit is an LLM-generated query. Fix it architecturally, not by patching the prompt.

The diagnosis first, because "patch the prompt" is the reflex you're being tested against. If an LLM-generated query produced a number that disagrees with the governed dashboard, the model authored a definition. That's a **capability boundary violation**, not a quality problem, and no amount of prompt tuning closes it — you're asking the model to guess a fact it doesn't have access to, and it will guess correctly most of the time, which is the worst possible outcome because it means the failures are rare and trusted.

The architectural fixes, in the order I'd ship them:

**1. Remove the capability.** The LLM loses raw SQL execution and gets the validated `MetricQuery` interface. This is the fix; everything else is defence in depth. If the metric it needs doesn't exist in the catalog, the correct product behaviour is "I can't compute that — here's who owns adding it," not a plausible number.

**2. Render provenance with every number.** Every figure in an AI-generated answer carries the metric name, its definition, the filters applied, and the grain. This makes the 8% discrepancy *visible at the point of disagreement* rather than in a meeting three weeks later. In my experience this alone resolves most "the AI is wrong" tickets, because usually one side applied a different filter and both numbers are correct for their definition.

**3. Assertion tests on the metric layer, in CI.** For each of your top metrics, a golden query with an expected value on a frozen data snapshot. This is dbt-test discipline applied to the thing the LLM reads, and it catches the case where the *catalog* drifted rather than the model.

**4. A reconciliation job.** Nightly, run the top 50 metric queries through both the governed path and the AI path and alert on any divergence over a threshold. This is a shadow read applied to analytics, and it's how you find the next instance before a customer does.

**5. An escape hatch with a different trust level.** Analysts will genuinely need ad-hoc SQL. Give them a generation path that is clearly labeled as **draft, unreviewed, not a source of truth**, that writes to a scratch schema, and that cannot power a dashboard. The failure isn't that the model wrote SQL; it's that a model-authored number entered a governed surface without a review step.

**💰 Math on why this is worth real engineering:** the cost of the incident isn't compute. It's a finance team reconciling two numbers for a week — say three people × 20 hours × a fully-loaded $100/hour = **$6,000** — plus the durable outcome that they now distrust every AI-produced number, which kills adoption of the feature you spent two quarters building. That's the number I'd put in the post-mortem, because "the model was 92% accurate" is not the relevant metric when the 8% lands in a board deck.

**⚠ Trap:** responding with an LLM-as-judge that checks the generated SQL. You've now got a second nondeterministic component validating the first, with correlated failure modes — both models share training data and share the same wrong priors about what "revenue" means. Validation must be **deterministic and external**: schema validation against a catalog, or comparison against a governed computation. Two models agreeing is not evidence.

### Explain train/serve skew for a system that has both a classical ML model and an LLM in the path. Where does it bite that it wouldn't in a pure-ML system?

Train/serve skew is the same disease you already know — the features at inference differ from the features at training — but a hybrid system adds three new vectors that a pure-ML system doesn't have, and those are the interesting part.

**Vector 1: the embedding model is a feature transformer that can change under you.** A router or reranker trained on model A's embeddings, then served embeddings from model B, is skew in its purest form — the feature space literally moved. Nothing errors. Accuracy drops silently, because vectors from two different encoders are both 1024 floats and both produce cosine similarities. The guard: **the embedding model id and weights hash are part of the trained model's manifest, and the serving path asserts equality at load time.** Ten lines, and it turns a silent 15-point accuracy loss into a startup failure — which is exactly the trade you want.

**Vector 2: the retrieval context is a feature, and it's built by a pipeline with its own version.** If you trained a reranker or a classifier on top of retrieved-context features, and since then the chunker changed or the index was rebuilt with different parameters, the feature distribution has shifted even though the embedding model is identical. This is the one I've seen bite hardest, because nobody thinks of "top-10 retrieval results" as a feature vector with a distribution. It is one. Monitor it like one: track the distribution of top-1 similarity, of score gaps, and of retrieved-source mix, and alert on drift.

**Vector 3: LLM-generated features.** If a classical model consumes an LLM-extracted field (`sentiment`, `intent`, `contract_type`), then the extraction prompt's version is part of that model's training manifest. Change the prompt and you've silently changed a feature's distribution for every row produced after the change — while the training data still reflects the old prompt. This is why extracted tables must be versioned rather than mutated in place: a versioned column lets the classical model pin `extraction_schema_version = 'v4'` and be immune to your v5 release until it's retrained.

**The general defence** is the one you already run for features — compute the feature once, in one code path, and serve it from a store to both training and inference — extended to cover embeddings, retrieval configuration, and extracted fields. Point-in-time-correct joins matter here exactly as much as they do in classical feature engineering: if you build a training set by joining today's extracted fields onto historical labels, you've leaked information the model wouldn't have had at prediction time, and your offline metrics will be beautiful and your online metrics will not.

**⚠ Trap:** the reranker trained on `top_k=50` candidates and served `top_k=20`. The candidate distribution differs — the model learned to discriminate against weak candidates that no longer appear — and its calibration is wrong. `top_k`, the hybrid search weights, the filter set, and the MMR diversity parameter are all **hyperparameters of the training data**, not runtime knobs, and changing one is a retraining trigger. I've seen a "harmless" top-k reduction made for cost reasons drop reranker precision by 6 points, with the cost saving proudly reported and the quality loss unattributed for a month.

**📐 Numbers you must know:** the two skew checks worth hard-coding are (1) embedding model id + weights hash equality between the model manifest and the serving encoder, and (2) `retrieval_config_version` equality. Both are string compares costing microseconds, and between them they catch the large majority of hybrid-system skew. If an interviewer asks how you'd prevent skew and you name those two assertions concretely, you're ahead of most candidates who answer "use a feature store."

### Is a feature store the right home for embeddings? Give me the actual boundary.

Partially, and the boundary is **cardinality and update pattern**, not technology preference.

A feature store solves two things: point-in-time-correct offline retrieval for training, and low-latency online lookup by entity key for serving, from one definition. That maps cleanly onto **entity embeddings** — a user embedding, a merchant embedding, a session embedding — where the key is a known id, the cardinality is millions not hundreds of millions, and the access pattern is "give me the vector for this key." That's a KV lookup, the feature store is genuinely the right home, and you get the point-in-time joins for free, which matter enormously because a user embedding computed today must not be joined onto a label from last March.

It maps badly onto **document chunk embeddings**, because the access pattern is fundamentally different: you don't look them up by key, you *search* them by similarity. A feature store has no ANN index. Forcing 100M chunk vectors into one gives you a very expensive KV store that cannot answer the only question you ask of it.

So the boundary I'd draw:

- **Entity embeddings, keyed lookup, needed at training and serving** → feature store. Materialize offline to the warehouse for training, online to Redis or DynamoDB for serving, from one definition.
- **Corpus/chunk embeddings, similarity search** → lakehouse table (source of truth) + vector index (serving), as designed throughout this section.
- **Both, when an entity embedding is also searched** — "find users similar to this one" — → compute once, land in the feature store for keyed access, and *also* build a vector index over the same table. Two serving artifacts, one source of truth. That's not duplication to be embarrassed about; it's the same pattern as an OLTP table with a search index.

The genuinely valuable thing the feature store contributes that a plain table doesn't is the **point-in-time-correct join**, and it's worth being precise about why: if you train a model on "user embedding at time T" you must retrieve the embedding **as of T**, not the current one. Doing that by hand with `AS OF` joins over a slowly-changing dimension is error-prone, and getting it wrong produces label leakage — offline AUC of 0.94 and online AUC of 0.71, the classic hybrid-system disappointment. If your embeddings are recomputed on a schedule, they are a slowly-changing dimension and you need the versioned history, not just the latest value. That means your entity-embedding table is append-only with a validity interval, which roughly multiplies storage by the number of retained versions — **💰** 10M users × 512 dims × 4 bytes = 20.5 GB per snapshot, × 52 weekly snapshots = **1.07 TB**, about $25/month on object storage. Cheap enough that "keep the history" is always the right call, and worth saying out loud because people assume it isn't.

**⚠ Trap:** storing only the current entity embedding and building training sets from it. Every training run silently leaks future information, the offline numbers look great, and the model underperforms in production by a margin nobody can explain. It's the single most common defect I've seen in hybrid systems, and it is invisible to every test you'd normally write.

### Design it end to end. Fifty enterprise tenants, a hundred million chunks, hard isolation, a nightly quality gate. Take your time.

I'll build it as five planes, because the isolation and versioning questions have different answers in each, and then give the numbers.

**Ingestion plane.** Per-tenant, per-source connectors emitting to a shared Kafka topic keyed on `(tenant_id, doc_id)` — shared topic, tenant in the key, so ordering is per-document and one noisy tenant doesn't need its own cluster. Every message carries `tenant_id`, `source_id`, `source_modified_at`, `sensitivity`, and the connector version. Ingest-time validation against the data contract; violations quarantine with the payload. Per-tenant rate limits so a 40M-document initial crawl doesn't starve everyone else's incremental — this is the concurrency problem you already solve for background jobs, and the answer is the same: separate queues by cost class plus a per-tenant token bucket.

**Storage plane.** Iceberg or Delta tables on object storage: `documents`, `chunks`, `embeddings`, `extractions`, `traces`, all **partitioned by `tenant_id` first, then by date** for the append-heavy ones. Tenant-first partitioning here is deliberate and the opposite of my advice for traces — for corpus tables the dominant operations are "rebuild this tenant's index" and "delete this tenant," both of which become partition operations. For traces, date-first, because expiry dominates. Being able to explain why the two differ is the answer to "how do you partition," not a single rule.

Isolation: separate storage prefixes with distinct IAM policies per tenant for anything above `internal`; a shared prefix with row-level policies for the rest. Say plainly that **row-level security is a weaker boundary than a separate bucket**, and that which one you use is a function of what you sold. A tenant paying for isolation gets a prefix and a key; the mid-market gets RLS.

**Compute plane.** Ray Data (or Spark on Databricks) for embedding, with a shared GPU pool and fair scheduling. Not per-tenant GPUs — at 50 tenants that's 50 idle GPUs. One pool, a queue with per-tenant weights, and backfills at lower priority than incremental so a new customer's crawl never degrades an existing customer's freshness.

**Serving plane.** This is where the real design decision sits: **one index with a tenant filter, or an index per tenant?**

- One index with a metadata filter: cheapest, one 435 GB HNSW graph, but filtered ANN search degrades as the filter gets more selective — you traverse a graph mostly full of other tenants' vectors to find the few you're allowed to see. At 50 tenants averaging 2% of the corpus each, a naive post-filter has to over-fetch by ~50× to fill top-10, and latency goes with it.
- Index per tenant: clean isolation, exact deletion, per-tenant rebuilds, no filter cost. But 50 graphs means 50 lots of index overhead, and the small tenants waste resources.
- **What I'd actually ship: shard by tenant, group the small ones.** Tenants above ~1M chunks get a dedicated index; everyone below shares a multi-tenant index with a partition-aware filter. That's roughly 8 dedicated + 1 shared for a typical 50-tenant distribution, and it gives you per-tenant rebuild and deletion for the customers who care while keeping the long tail cheap.

Alias per tenant: `idx:{tenant}:prod → chunks_t{tenant}_build_8829`. Promotion and rollback are per-tenant writes, which means a canary is "three tenants," which is the granularity you actually want.

**Control plane.** The manifest registry (build ids, versions, aliases), the quality gate results, per-tenant configuration, and the kill switches. This is a boring Postgres database and it is the most important component, because it's the thing that makes every other plane's state queryable.

**The nightly gate**, as a DAG: ingest → chunk → embed → build → **eval** → promote. The eval stage runs each tenant's frozen probe set against their new index; promotion is per-tenant and automatic only when the gate passes, held for human review otherwise. Crucially, **a failing tenant does not block the other 49** — this is why promotion must be per-tenant rather than one global cutover, and it's the design detail I'd push hardest on in review.

**💰 The numbers, since this is the part that gets graded:**
- Embedding backfill: 100M chunks ÷ 500 chunks/s = 55.6 GPU-hours ≈ **$70** on spot; **1.7 hours** on 32 GPUs.
- Incremental: 5M chunks/day, but with `content_sha` diffing only ~15% are genuinely new → 750k/day = 1,500 GPU-s = **0.42 GPU-hr/day ≈ $0.50/day**.
- Vector RAM: 100M × 1024 dims × 2 bytes (fp16) = 204.8 GB + ~25 GB graph = **230 GB**, doubled for the hot previous build = **460 GB**. One 512 GB instance at ~$4/hr = **$2,900/month**. (**📅 Volatile:** instance pricing.)
- Object storage: documents + chunks + embeddings ≈ 1.2 TB at $0.023/GB-month = **$28/month**. Storage is never the line item; RAM is.
- Traces: pointer-only at 2 KB × 10M/day = 20 GB/day = 600 GB/month ≈ **$14/month** in Parquet.
- Nightly eval: 50 tenants × 1,000 probe queries = 50,000 vector searches (free) + 50 tenants × 150 end-to-end judged cases × ~4k in / 400 out at $3/$15 = 7,500 × ($0.012 + $0.006) = **$135/night ≈ $4,050/month**.
- **Total platform run-rate ≈ $7,100/month** excluding the answer-serving inference itself. At 50 tenants that's $142/tenant/month of platform cost — a number you should be able to produce, because it's what determines whether the product has a gross margin.

**🗣 Say this in the room:** "Five planes — ingestion, storage, compute, serving, control. The two decisions I'd defend hardest are per-tenant index sharding above about a million chunks with a shared index for the long tail, so isolation and rebuild granularity match what customers paid for; and per-tenant promotion gated on a per-tenant probe set, so one tenant's failing eval doesn't block the other forty-nine. Platform run-rate is about seven thousand a month at that scale, dominated by index RAM and the nightly eval, not by embedding compute."

### Last one. You have one week to ship a v0 of this and twelve months to build the platform. What's in the week, and what's the thing you refuse to skip?

The week is: object storage for originals, one Parquet or Delta table of chunks with `chunk_id`, `content_sha`, `doc_id`, and byte offsets, a hosted embedding API, pgvector or a managed vector index, a cron job, and a fifty-query probe set with Recall@10 printed to the console. That's a working retrieval system and it's genuinely enough to learn from. I'd deliberately skip: CDC, Ray, orchestration beyond cron, dedup, a semantic layer, per-tenant sharding, and the entire feature-store conversation.

**The two things I refuse to skip even in the week**, and I'd fight for both:

**1. The chunks table as a source of truth, with content hashes and byte offsets.** Not because week one needs it, but because it's the only decision that is expensive to retrofit. If v0 writes straight into a vector store, then at month three when you need to change embedding models, answer a deletion request, or debug a stale answer, you cannot — and the fix is re-crawling every source, which is a month of work and a conversation with fifty customers' rate limiters. Everything else in this section can be added later. **This one cannot.**

**2. A frozen probe set with a printed number.** Fifty queries and their expected chunk ids, run on every change. Not because fifty is statistically adequate — at n=50 the CI on Recall@10 is roughly ±11 points, so it detects catastrophes, not regressions — but because the *habit* of gating on a number is what separates a system you can improve from a system you can only argue about. Teams that don't build this in week one never build it, because there's never a week where adding an eval is more urgent than the next feature.

The twelve-month arc, in the order the constraints actually arrive: content-hash memoization and incremental ingestion (month 1–2, when the nightly full re-embed stops fitting in the window); CDC and freshness SLAs (month 3–4, when a customer complains about staleness); versioned index builds with alias swap and a real promotion gate (month 4–5, when you need to change embedding models without downtime); data contracts and quarantine (month 5–6, when the third connector's format change silently drops 6% of a corpus); per-tenant sharding and isolation (month 6–9, when enterprise procurement asks); lineage, PII classification, and deletion propagation (month 6–9, driven by the same procurement conversation); dedup, semantic layer, and feature-store integration (month 9–12, when quality work has run out of cheaper wins).

Notice the ordering principle, because it's the thing I'd want an interviewer to take away: **every item is pulled by a specific failure or a specific customer conversation, not pushed by an architecture diagram.** I've watched teams build the CDC pipeline in month one for a corpus that changes twice a week, and then have no eval harness in month nine. The sequencing is the judgment.

**🏋 Drill (45 minutes, unaided, whiteboard):** given "10M documents, 5 sources, 20 tenants, quality regressions must be caught before users see them," draw the five planes, name every table with its partition key, name the promotable unit's fields, and produce the monthly cost with arithmetic for embedding compute, index RAM, object storage, and the eval budget. Pass criterion: your promotable unit has at least six fields, your partition keys differ between corpus tables and trace tables *and you can say why*, and every cost figure has its derivation next to it. If you can do that cold in 45 minutes, you're ready for the Databricks and Snowflake fusion round — which is, in the end, one question repeated: "you designed a retrieval system; now defend the data engineering underneath it."

**🗣 Say this in the room:** "In week one I ship a chunks table with content hashes and byte offsets, a hosted embedding API, and a fifty-query probe set that prints Recall@10 — and I skip CDC, orchestration and sharding entirely. The chunks table is the one thing I refuse to skip, because it's the only decision here you can't retrofit: without it you can't change embedding models, can't complete a deletion request, and can't debug a stale answer without re-crawling every source."


---

## 62. The Enterprise Integration Surface: Connectors, Identity, Permission Mirroring and Tenant Configuration

*Mastering this proves you can pass a Glean/Harvey/Sierra loop and an FDE deployment conversation, where retrieval quality is table stakes and integration is the actual work.*

### Let's start with framing. At a company like Glean or Harvey, what fraction of the engineering is retrieval quality and what fraction is integration, and why do you think that ratio surprises people?

Retrieval quality is maybe a fifth of it, and it is the fifth that a candidate can already read about. The other four-fifths is the integration surface: getting the customer's data out of eleven SaaS systems continuously, reproducing each system's authorization model faithfully enough that the answer engine never shows a person a document they could not have opened themselves, proving that to a security reviewer, and doing it per tenant with different policies, regions and retention rules.

The reason this surprises people is a category error. Engineers think of a connector as an ETL job — read from an API, write to an index — and ETL feels solved. But an enterprise search connector is not an ETL job; **it is a continuously-reconciling replica of a foreign system's content *and* its access-control graph, where the foreign system offers you no transactions, no consistent snapshot, incomplete change feeds, and rate limits tuned for a dashboard rather than a crawler.** Every hard distributed-systems property you know — ordering, exactly-once, read-your-writes, replication lag — shows up, and the consequence of getting the lag wrong is not a stale dashboard, it is a leaked salary spreadsheet.

The way I'd frame it to an interviewer: retrieval quality is a curve you push up over quarters with better chunking, better rerankers, better hybrid scoring. Integration is a set of *binary* gates. If your Salesforce connector cannot express record-level sharing rules, you cannot sell to that customer at all. If you cannot produce an audit log naming which model version answered a question, you fail procurement. There is no partial credit.

**🗣 Say this in the room:** "I think of an enterprise AI product as three planes: a content plane, an identity-and-permission plane, and a policy plane. Retrieval lives in the first. Most of the engineering risk, and essentially all of the deal risk, lives in the second and third — and the second is a distributed-systems replication problem with a security blast radius, which is exactly the kind of problem my backend background maps onto."

**⚠ Trap:** treating permissions as a filter you add at the end. Access control is a *data model* decision that constrains your chunking, your index layout, your cache keys and your eval sets. Bolting it on after you have shipped a single flat index is a rewrite, not a feature.

### Design the connector framework. What's the interface every connector implements, and what does the platform own versus what does each connector own?

The mental model: a connector is a **driver**, in exactly the sense a database driver is a driver. It translates one foreign system's idiosyncrasies into a small, boring, uniform vocabulary that the platform understands. If a connector contains retry logic, embedding calls, or index writes, you have built eleven half-platforms instead of one platform with eleven drivers. The rule I enforce in review: **a connector may not import the vector store, the embedding client, or the scheduler.**

The vocabulary I would standardize on has five verbs and two nouns.

The nouns are `Document` and `Principal`. A `Document` is `{external_id, tenant_id, source, url, title, body_or_blob_ref, mime, created_at, updated_at, version_token, acl, container_path, custom_fields}`. A `Principal` is `{external_id, kind: user|group|role|public|external, display, email_or_null}`. Every connector emits only these. Everything downstream — chunking, embedding, index build, permission expansion — is written once against these two types.

The verbs:

```python
class Connector(Protocol):
    async def validate(self, cfg: SourceConfig) -> HealthReport: ...
    async def full_crawl(self, cfg, cursor: Cursor | None
                         ) -> AsyncIterator[Batch[Document] | CursorCheckpoint]: ...
    async def delta(self, cfg, cursor: Cursor
                    ) -> AsyncIterator[Batch[Change] | CursorCheckpoint]: ...
    async def fetch_acl(self, cfg, ext_ids: list[str]) -> dict[str, Acl]: ...
    async def expand_group(self, cfg, group_ext_id: str
                           ) -> AsyncIterator[Principal]: ...
```

`Change` is `Upsert(Document) | Delete(external_id) | AclChange(external_id, Acl) | ContainerMoved(...)`. That last one matters more than people expect: in Drive, SharePoint and Confluence, moving a folder can silently change the effective permissions of ten thousand descendants without emitting a single per-document change event.

The platform owns: scheduling and concurrency, per-tenant per-source rate limiting, retry with jitter and circuit breaking, credential vaulting and OAuth refresh, cursor durability, checkpointed resumption, dedupe by content hash, chunking, embedding, index writes with alias swap, permission-graph materialization, deletion propagation, metrics, and the audit trail. The connector owns: pagination shape, the change-feed dialect, the ACL dialect, the throttling signal that source uses, and the mapping into `Document`/`Principal`.

**⚠ Trap:** letting each connector define its own cursor format as an opaque string and then storing it as a bare column. You will need to *invalidate* cursors — because you fixed a parser bug, because the source rotated its change-token domain, because a customer restored from backup. Store cursors as `{connector_version, schema_version, token, issued_at, source_epoch}` so a platform-side migration can reason about which cursors are still valid instead of you writing eleven ad-hoc SQL updates at 2am.

**🗣 Say this in the room:** "I'd give every connector the same five-verb interface — validate, full crawl, delta, fetch ACL, expand group — and keep all retry, rate-limit, embedding and index logic in the platform. The test I use for whether the boundary is right: adding a twelfth connector should be a few hundred lines and require zero changes to the indexing pipeline."

### Walk me through incremental sync across the sources you'd actually connect. How does the change feed differ between Google Drive, Slack, Confluence, Jira, Salesforce, SharePoint and GitHub?

They differ along one axis that determines everything: **does the source give you a totally-ordered, complete change log, or does it give you a query you poll?** A real change log gives you deleted-item detection and ordering for free. A poll gives you neither, and you have to reconstruct both.

**Google Drive** is the good case. `changes.list` takes a `startPageToken` and returns a stream of change records plus a `newStartPageToken` you persist as your cursor. It is a genuine log: it includes removals (`removed: true`) and trashing, and it covers permission changes on files you can see. Shared-drive content either folds into that feed via `includeItemsFromAllDrives`/`supportsAllDrives` or gets its own per-drive change stream via the `driveId` parameter — decide deliberately, because the two give you different cursor-management and coverage properties. The catch is that the token has a finite retention window — if your connector is down long enough, the token is rejected and your only recovery is a full re-crawl.

**Microsoft Graph / SharePoint / OneDrive** offers `delta` queries on a drive or a list: you follow `@odata.nextLink` through the pages, then persist the `@odata.deltaLink` as your cursor. Deleted items come back with a `deleted` facet rather than disappearing. Same failure mode: an expired delta token forces a resync.

**Slack** is the awkward one. `conversations.history` and `conversations.replies` are cursor-paginated over time, so you can poll per channel — but that scales as O(channels) per cycle and Slack's rate limits punish it hard. The real design is the Events API over a socket or webhook for `message`, `message_changed`, `message_deleted`, `channel_archive`, `member_joined_channel`, with polling only as a gap-filler after a disconnect. Org-wide historical access on Enterprise Grid goes through Slack's Discovery API, which is plan-gated — **📅 Volatile:** Slack has repeatedly retuned rate limits and access tiers for non-Marketplace apps; verify the current tiers and whether your app needs Marketplace approval before you promise a customer a sync SLA.

**Confluence** and **Jira** are poll-with-a-watermark. Confluence: CQL with `lastModified >= "<watermark>"` ordered by `lastModified`. Jira: JQL with `updated >= "<watermark>"` ordered by `updated`. Both give you *modifications* and neither reliably gives you deletions, so deletion detection is a separate reconciliation problem. Atlassian also ships webhooks, which are great for latency and terrible as a source of truth because they are at-most-once in practice — I treat webhooks as a *hint that lowers latency* and the poll as the correctness mechanism.

**Salesforce** is the best-instrumented of the lot. `SystemModstamp` gives you a reliable modification watermark in SOQL; the replication APIs (`getUpdated`/`getDeleted` over a time window) give you real deletion detection including the recycle bin; Change Data Capture and Platform Events give you a push feed; Bulk API 2.0 gives you a sane path for the initial extraction of millions of records. If a candidate tells me they'd poll `LastModifiedDate` in Salesforce I know they haven't shipped it — `LastModifiedDate` does not move on cascading updates, `SystemModstamp` does.

**GitHub** is push-first. Webhooks for `push`, `pull_request`, `issues`, `issue_comment`, plus REST/GraphQL with `since` for backfill. The Events API is not a durable log — it has a short retention window and drops under load — so for correctness you reconcile against refs: store the last-synced commit SHA per branch and diff.

**Email** (Gmail `history.list` from a `startHistoryId`; Graph mail `delta`) is a real log per mailbox, which means your sync unit is a *user*, not a tenant — a structurally different scaling problem I'd handle separately.

**⚠ Trap:** using `updated_at >= last_watermark` with a strictly-greater or naive-equal comparison and losing records. Two records can share a timestamp at the source's resolution, and clock skew inside a multi-node SaaS backend means a record written at T can become visible after you've already read past T. The fix is the same one you'd use on a Postgres CDC tail: **overlap the window** (`updated >= watermark - 5 minutes`) and rely on content-hash dedupe to make the reprocessing free, rather than trying to make the watermark exact. Cheap insurance: a 5-minute overlap on a 5-minute poll doubles read volume on the delta path, which is nothing compared to silently missing a document forever.

### How do you detect that a document was deleted at the source, when half these APIs never tell you?

You detect it three ways, in descending order of preference, and you must be honest about which one a given source supports because the *deletion SLA* is a contractual number in enterprise deals.

**Tier 1 — the source tells you.** Drive `removed`/`trashed`, Graph's `deleted` facet, Salesforce `getDeleted`, Slack `message_deleted`, GitHub webhook `deleted: true` on a ref push. Latency is seconds to minutes. Take it whenever it exists.

**Tier 2 — tombstone by absence within a bounded scope.** For sources with listable containers (a Confluence space, a Jira project, a Drive folder, a SharePoint list), enumerate IDs in the container cheaply — ID-only projection, no bodies — and diff against what you have indexed for that container. Anything you hold that the source no longer lists is deleted *or moved*, and you must distinguish those: check whether the ID reappears elsewhere in the same sweep before you tombstone it, or you will delete and re-add half a wiki every time someone reorganizes a space.

**Tier 3 — full reconciliation sweep.** A periodic ID-only crawl of the whole source, diffed against the index. This is your backstop for everything Tier 1 and 2 miss — permission-only changes that emit no content event, items moved into a space you can't see, items whose delete event you dropped during an outage.

The scheduling I'd defend: Tier 1 continuous, Tier 2 hourly-to-daily per container weighted by churn, Tier 3 weekly, plus an unconditional Tier 3 after any connector incident.

**💰 Math:** Tier 3 on a 4M-document Drive tenant, using ID-only `files.list` at 1,000 items per page, is 4,000 requests. At a conservative 10 requests/second budget (staying well inside a per-project quota you share with delta traffic) that is 400 seconds of API time — under 7 minutes of wall clock. The reason people skip it is not cost; it is that nobody wrote it. Write it.

**⚠ Trap:** treating "I got a 404 fetching this document" as a delete. A 404 from Drive or Graph frequently means *your service account lost access*, not that the file is gone. If you tombstone on 404 you will silently evict thousands of live documents the first time someone tightens a shared-drive membership, and the user-visible symptom is "search got worse" with no error anywhere. Distinguish: 404 on a fetch → mark `access_lost`, remove from *that principal's* visibility, keep the document; only tombstone on an explicit deletion signal or a Tier-2/3 absence confirmed twice.

**🔍 Failure taxonomy — "a deleted doc is still answerable."** Walk it in this order: (1) Was a delete event emitted by the source? Check the raw event log, not your parsed one. (2) Did the connector receive it — did the webhook 200 or was the cursor stalled? (3) Did the tombstone reach the index — is there a delete record in the write-ahead log for the index? (4) Did the index actually apply it — many vector stores mark deleted and only reclaim on compaction, and a *filtered* query can still return tombstoned vectors if your filter runs post-search. (5) Is it in a cache — semantic cache, prefix cache, a materialized answer, or a memory store? Deletion propagation to caches is the step teams forget, and it is the one a compliance reviewer will ask about by name.

### Your crawler is getting rate-limited by a customer's SharePoint tenant and the initial crawl is going to take three weeks. Talk me through what you do.

First, I refuse to treat this as a tuning problem until I know which of the four limits is binding, because the mitigations are disjoint: a per-app quota, a per-tenant service-protection limit, a per-resource limit (one busy site collection), or a *concurrency* limit rather than a rate limit. Graph tells you which by way of `Retry-After` and throttling reason headers; the shape of the 429s over time tells you the rest — a hard wall at a fixed RPS is a quota, sawtoothing is a token bucket, and 429s clustered on a few site IDs is per-resource.

The controls I'd apply, in order:

**Respect the server's own signal, exclusively.** `Retry-After` is not advisory. The single most common bug I see is exponential backoff *ignoring* `Retry-After` and retrying sooner than told, which converts throttling into a soft ban. And the retry must be per-token-bucket, not global — one 429 on one site collection should not stall the other 400.

**Adaptive concurrency, not a fixed rate.** Run an AIMD controller per (tenant, source): additive increase of one in-flight request per success window, multiplicative decrease to 50% on a 429. This is TCP congestion control and it is the right primitive, because you cannot know the customer's limit — it varies by their license tier, their other integrations, and the time of day.

**Reduce request count, not just rate.** This is where the real wins are and where most engineers don't look. Batch: Graph's `$batch` endpoint packs up to 20 subrequests into one HTTP call. Project: `$select` only the fields you need so pages are cheaper and you can raise page size. Use `delta` rather than re-listing. Fetch ACLs at the *container* level and inherit, instead of per-item — on a typical SharePoint site 95%+ of items inherit from their library, and per-item permission calls are usually the actual thing blowing your budget.

**Prioritize.** Three weeks of crawl is only unacceptable because the customer sees nothing for three weeks. Crawl in value order — recently-modified first, then most-viewed containers if the source exposes analytics, then the long tail — and make search live over the partial index immediately with an explicit "still indexing" affordance. I'll come back to this under onboarding, because it is a product decision more than an engineering one.

**Escalate the credential.** Often the real fix is not code: it is asking the customer's admin to register a dedicated app registration for you rather than sharing quota with their other integrations, or provisioning a second service principal. **📅 Volatile:** service-protection limits and batch sizes change; verify against current Microsoft Graph throttling guidance rather than a number you memorized.

**💰 Math:** suppose the tenant is 12M items and you're pulling 1 item per request at a sustained 8 req/s. That is 12,000,000 / 8 = 1.5M seconds ≈ 17.4 days. Now batch 20 subrequests per call and drop per-item ACL calls by inheriting from 40k containers: the request count falls to 12M/20 = 600k content calls plus 40k container-ACL calls = 640k requests, and if the throttle is per-*request* rather than per-item you can hold 8 req/s → 80,000 seconds ≈ 22 hours. Same rate limit, 19× faster, purely from request shaping. This arithmetic is the answer the interviewer is listening for; "I'd add backoff" is not.

**⚠ Trap:** parallelizing a crawl by sharding across many workers with independent rate limiters. The source limits you at the *tenant* level; your ten workers each politely staying under 10 req/s will hit the tenant at 100 req/s and get all ten banned. The rate limiter must be shared state — a Redis token bucket keyed `(tenant, source, credential)` — and the credential is part of the key because some sources meter per app registration and some per service account.

### Backfill and delta at the same time — how do you run a three-day initial crawl without missing changes that happen during it?

This is the standard "consistent snapshot plus log tail" problem you already know from Postgres logical replication, and the answer has the same shape: **acquire the log position first, then take your leisurely snapshot, then replay the log from the position you saved, and let idempotency absorb the overlap.**

Concretely, the sequence per source:

1. Call the source's "give me a cursor for right now" primitive *before* the backfill starts — `changes.getStartPageToken` for Drive, an initial `delta` link for Graph, `SystemModstamp = now()` for Salesforce, current commit SHAs for GitHub, the current `historyId` for Gmail. Persist it as `pending_cursor`.
2. Run the backfill. It is slow, it is unordered, and it sees a smear of the source's state across three days. That is fine.
3. When the backfill completes, start the delta stream from `pending_cursor` and let it catch up. During catch-up the index is *converging*, not correct.
4. Only when the delta stream reaches live tail (lag below your threshold) do you flip the tenant's `sync_state` to `live` and start honoring your freshness SLA.

Two properties make this safe. **Upserts must be idempotent and last-writer-wins on a source version token**, never on your ingest time — otherwise a slow backfill worker writing a 3-day-old version after the delta stream already applied a fresh one will resurrect stale content. Store `source_version` (Drive `version`, Graph `eTag`, Confluence `version.number`, Jira's `fields.updated` since issues expose no version counter, Salesforce `SystemModstamp`, GitHub blob SHA) on every row and drop writes whose version is not greater than what you hold. **And deletes must be tombstones, not row removals**, for the same reason: a backfill worker that read the doc before it was deleted must not be able to re-insert it. A tombstone with a version wins the comparison and the re-insert is rejected.

If the source has no version token, fall back to `(source_updated_at, content_sha)` with the timestamp as the ordering key and the hash to make equal-timestamp writes a no-op.

**⚠ Trap:** starting the delta stream *after* the backfill, taking the cursor at that moment. Everything that changed during the three days is lost forever, and you will never notice, because the index looks full. This is the single most common connector bug I've seen, and it is invisible until a customer says "I edited that doc on Tuesday and search still shows the old text" — three weeks later.

**🗣 Say this in the room:** "Take the change cursor before the snapshot, not after — same discipline as capturing an LSN before a `pg_dump` for logical replication. Then make every write idempotent under a source-provided version token so the overlap between backfill and delta is free rather than a race."

### Slack specifically. What's hard about it that isn't hard about a document store?

Slack breaks three assumptions that every document connector is built on.

**The unit of retrieval isn't the unit of ingestion.** A Slack message is 40 characters and useless alone; the meaningful unit is a thread, or a time-bounded window of a channel's conversation. So the connector's `Document` is a *synthesized* object — a thread rollup, or a rolling window — which means it is mutable in a way files are not: a reply arriving six months later must update an existing document rather than create a new one, and your `external_id` has to be the thread's parent `ts` so that update lands in the right place. Get this wrong and you get one embedding per message, retrieval returns "sounds good 👍", and everyone concludes the product is bad at Slack.

**Permissions are membership, and membership is high-churn.** A public channel is visible to everyone in the workspace; a private channel is visible to its current members; a DM or MPIM is visible to its participants; and Enterprise Grid adds shared channels spanning multiple workspaces plus Slack Connect channels with *external* organizations. Membership changes constantly. If you bake a member list into every message document, a single `member_joined_channel` event dirties every document in that channel. The right model is indirection: the document's ACL is `channel:C0123`, and the principal→channel membership lives in a separately-maintained, fast-changing table that the query path expands. This is the same reason you don't denormalize group membership into every row in a relational schema.

**Access is intrinsically per-user for a large slice of the corpus.** DMs and private channels the crawler's token can't see simply are not crawlable with a bot token, and pulling them requires either the Discovery API on Enterprise Grid or per-user OAuth. This forces an architectural decision early: are you an org-wide index with a privileged token and mirrored ACLs, or a per-user index built from delegated tokens? Both are legitimate; mixing them accidentally is how you get a leak.

**⚠ Trap:** indexing Slack Connect / shared channels into the tenant's index without marking the external organization. Messages from a partner company sitting in your customer's answer engine, attributed as internal knowledge, is both a retrieval-quality disaster and a genuine confidentiality incident. Every document needs an `origin_org` field and a tenant policy that says whether external-origin content is retrievable, quotable, or neither.

**📐 Numbers you must know:** for sizing, a 2,000-person Slack workspace at typical usage generates on the order of 1–3M messages per year across all channels. Rolled into threads and 15-minute windows, that is roughly 150k–400k retrievable documents per year — comparable to a mid-size wiki, but with 10–50× the write rate. Size your delta path for the write rate and your index for the document count; they lead to different bottlenecks.

### Google Drive and SharePoint both have inheritance and link-sharing. What does that do to your permission model?

It turns a per-document ACL into a *graph query*, and pretending otherwise is where leaks come from.

In Drive, a file's effective access is the union of: its own explicit permissions, the permissions inherited from every ancestor folder, the shared-drive's membership if it lives in a shared drive, any domain-wide sharing (a permission of `type: domain`, which the UI calls "anyone at <org> with the link"), and link-sharing (`type: anyone`, i.e. public on the internet). In SharePoint, an item inherits from its list/library, which inherits from the site, which inherits from the site collection — until someone breaks inheritance at any level, which creates a unique permission scope. Add sharing links (organization-wide, specific-people, anonymous), sensitivity labels that can override, and guest accounts from the B2B directory.

The design consequence is that I do not store a flat list of allowed principals on the document. I store **an ACL expression referencing container nodes plus per-item overrides**, and materialize the effective set through a permission service:

```
doc.acl = {
  inherits_from: "container:drive/0AB.../folder/1xY...",   # nullable
  explicit_grants: [ principal:user:u_88, principal:group:g_eng ],
  link_scope: "anyoneWithLink" | "domain" | "none",
  external_allowed: bool
}
```

Container ACLs are their own indexed objects with their own change feed. A single folder re-share then costs you one container write instead of ten thousand document rewrites — and, critically, it *takes effect immediately* for every descendant instead of racing a bulk re-index.

**⚠ Trap:** the link-sharing hole. A document set to "anyone with the link" is, from Drive's point of view, accessible to your service account and to every employee — so a naive mirror marks it visible to the whole tenant, which is technically true and completely wrong as a product behavior. Users do not expect a link-shared doc from a different department to surface in their AI answers, and a security reviewer will treat it as over-permissioning. My default: link-shared content is retrievable only if the requesting user's other grants already reach it, *or* the tenant explicitly opts in, and either way it is labeled in the citation. Ship the conservative default; make the permissive one a config flag with an audit entry.

**⚠ Trap:** the "moved folder" silent re-permission. Moving a folder in Drive or SharePoint changes the effective permissions of everything under it, and neither source reliably emits a per-descendant change event. If your model is flat per-document ACLs, you are now wrong for every descendant until your next full reconciliation — which could be a week. With container-referencing ACLs you're correct the instant you process the one container event. This is the strongest single argument for the indirection, and it is worth saying out loud in an interview because it demonstrates you've actually operated one of these.

### Jira, Confluence and Salesforce all have permission models that don't look like an ACL. How do you mirror them?

They don't look like an ACL because they're *rule engines*, and the honest answer is that you mirror the rules, not the outcomes — or, where you can't, you delegate the decision back to the source.

**Confluence** is the tractable one: space permissions plus page restrictions. A page is visible if the user has space-view permission and is not excluded by a page-level restriction (restrictions are subtractive and can be view or edit). Restrictions cascade to child pages. That maps cleanly onto container-plus-override, same as Drive.

**Jira** is harder because there are two independent systems. Project permission schemes decide who can browse a project at all, and *issue security schemes* decide, per-issue, which security level applies — and security levels grant to users, groups, project roles, or *reporter/assignee/custom-field-user*, which are per-issue dynamic references. So the ACL for a Jira issue is genuinely a small expression: `browse(project) AND (no_security_level OR user ∈ resolve(security_level, issue))`, where `resolve` may need the issue's own field values. I materialize what I can — the static grants — and I keep the dynamic references (`reporter`, `current assignee`, `value of custom field X`) as symbolic terms evaluated at query time against the issue's stored fields. That hybrid is the realistic answer.

**Salesforce** is the extreme case. Effective record access is org-wide defaults, plus role hierarchy (managers see subordinates' records), plus ownership, plus criteria-based and owner-based sharing rules, plus manual shares, plus territory management, plus team membership, plus Apex managed sharing — and on top of *record* access there's object-level CRUD, field-level security, and restriction rules. No sane person reimplements that. Two viable strategies:

The first is to mirror the underlying share tables. Salesforce exposes `__Share` objects for records and `UserRole`, `Group`, `GroupMember` for the hierarchy, so you can reconstruct effective access. It works, it is expensive to keep current, and it is fragile against configuration changes.

The second, which I'd default to, is **ask Salesforce at query time.** `UserRecordAccess` lets you query, for a given user, whether they have read access to a set of record IDs. So retrieval runs unfiltered-but-tenant-scoped, produces a candidate set of, say, 200 record IDs, and one batched access check prunes it before the LLM ever sees a body. You pay a round trip; you get *exactly* correct semantics including every rule you didn't implement.

**🗣 Say this in the room:** "For sources whose authorization is a rule engine rather than an ACL — Salesforce most of all — I don't reimplement the engine. I mirror a coarse, safe over-approximation for recall and then do a batched just-in-time access check against the source or a local Zanzibar-style service before anything reaches the model. Reimplementing Salesforce sharing rules is a permanent correctness liability and I'd push back on any design that proposes it."

**📄 Paper:** Pang et al. (2019), *Zanzibar: Google's Consistent, Global Authorization System* (USENIX ATC) — the reference design for a centralized relationship-based authorization service with explicit consistency tokens ("zookies"); it replaced per-service bespoke ACL checks across Google and is the direct intellectual ancestor of OpenFGA/SpiceDB, which is what I'd reach for if I were building the permission plane rather than buying it.

### Walk me through a GitHub connector for a code-assistant product. What's different about code?

Three things are different, and they compound.

**The document boundary is semantic, not positional.** A 4,000-line file split every 500 tokens produces chunks that begin mid-function and end mid-string, which destroys both retrieval and the model's ability to use the result. Chunk on the syntax tree — one chunk per function or class, with the file path, the enclosing class, and the import block prepended as a header so the chunk is self-describing. Tree-sitter is the standard tool because it parses broken code and covers dozens of grammars from one interface.

**The corpus is a DAG of versions, not a set of documents.** Which commit is the index built at? If the assistant answers from `main` while the developer is on a feature branch that renamed the function, the answer is confidently wrong. My model: index a canonical ref per repo (usually the default branch), stamp every chunk with its blob SHA, and let the *client* — the editor extension — supply working-tree state for the file the user has open, which is both fresher and cheaper than trying to index every branch. Incremental sync is then trivially correct: on a push webhook, diff `before..after`, and re-embed only the changed blobs. A push touching 12 files in a 40k-file monorepo costs 12 blob re-embeddings, not 40k.

**Permissions are org- and repo-shaped, with a nasty edge.** GitHub App installation tokens are scoped to selected repositories, which is clean. Repo access derives from org membership, team membership (nested teams exist), outside collaborators, and per-repo role. The edge case: **private forks**. A fork of a private repo inherits access from the *upstream* network, so a user who loses access upstream may still hold a fork, and content can exist in a fork that never existed upstream. If you index forks, you need the fork's own permission edges, not the parent's.

**⚠ Trap:** indexing secrets. Repos contain `.env` files, test fixtures with real tokens, and historical commits that leaked credentials before being rotated. Once a secret is in your vector store it is in your backups, your caches, and potentially in a model context window that gets logged. Run secret detection at ingest — before chunking, before embedding — and drop or redact. This is a question a customer's security reviewer will ask verbatim, and "the model probably won't repeat it" is a failing answer.

**💰 Math:** a 40k-file repo averaging 200 lines/file at roughly 10 tokens/line is 80M tokens. Function-level chunking at ~300 tokens per chunk gives ~270k chunks. Embedding 80M tokens at $0.02 per million (**📅 Volatile:** small-embedding-model pricing moves; verify) is $1.60 for a full index build. The full build is essentially free; what costs money is doing it wrong and rebuilding weekly for 500 customers — 500 × $1.60 × 52 = $41,600/year of pure rework you avoid by making pushes incremental.

### Email is on the connector list. Why is it structurally different from every other source, and how do you architect for it?

Because for every other source there is a plausible tenant-level view of the corpus, and for email there is not. Mail is per-mailbox by construction: there is no such thing as "the company's inbox." That flips your sharding, your crawl scheduling, your cost model, and your consent story all at once.

Architecturally it means the sync unit is a user. For a 5,000-person customer, you have 5,000 independent cursors (Gmail `historyId` per mailbox, or a Graph mail `delta` link per mailbox), 5,000 independent rate-limit budgets, and 5,000 independent failure domains. Your scheduler has to fair-share across them, because otherwise one executive with a 400k-message mailbox starves the other 4,999.

It also means the *index* is per-user, or at minimum the ACL is a single principal with no group indirection — which is actually simpler on the permission side and much worse on the storage side, since you lose all deduplication. The same 8MB deck attached to a 200-person all-hands email exists 200 times. Deduplicate attachments by content hash, store one copy, and attach 200 ACL edges to it. That is not an optimization, it is the difference between a viable and a non-viable storage bill.

And it means consent is a first-class flow. Most enterprises will not let you crawl employee mail on a domain-wide delegation without a specific legal basis, and several jurisdictions make it hard regardless. Design for per-user OAuth grants with a visible, revocable connection, and design for the case where 30% of the org never connects — the product has to be useful with a partial mail corpus.

**💰 Math:** 5,000 users × 60,000 messages each = 300M messages, at ~1.5 KB of extracted text apiece = 450 GB of text. Chunked at one chunk per message and embedded at 1,024 dimensions in float32, that is 300M × 1024 × 4 bytes = 1.23 TB of raw vectors before any index overhead, and HNSW typically adds 30–60% for graph edges — call it 1.7–2.0 TB resident. At that point you are not choosing a vector database, you are choosing quantization: int8 scalar quantization cuts the vector payload 4× to ~300 GB, or binary quantization with a float rerank stage cuts it 32× to ~38 GB. **This is why email is the connector that forces your quantization decision**, and saying that unprompted in an interview lands well.

**⚠ Trap:** indexing an email thread as one document per message and then serving all of them. Threads quote their entire history, so message N contains messages 1..N−1 verbatim. Naive per-message indexing means your top-10 results are ten near-duplicate copies of the same conversation, your context window fills with quoted text, and your answer quality collapses. Strip quoted blocks at parse time (the `>` / `On <date>, X wrote:` boundary, plus the provider's own quote markers) and index the *new* content of each message plus a thread-level rollup.

### How do you decide whether a connector should crawl with a privileged service account or with each end user's delegated token?

This is the single most consequential architectural choice in the whole integration surface, and it should be made per source, deliberately, not by accident.

**A privileged service account** — a Google domain-wide delegated service account, a Microsoft app registration with application permissions, a Slack bot with org-wide scopes, a GitHub App installation, a Salesforce integration user — sees everything, crawls once, and produces one shared index. You get: a single crawl amortized across all users, complete coverage regardless of who has logged in, one cursor per source, and a good story for global freshness. You take on: **the entire burden of permission mirroring**, because now *you* are the one deciding who may see what, and a bug in your ACL logic is a data breach rather than an API error. You also concentrate risk — one compromised credential reads the customer's entire Drive.

**Delegated user tokens** — each user OAuths individually and you crawl as them — means the source enforces permissions for you. A user's crawl can only ever see what that user can see, so the worst case of an ACL bug is that someone sees their own documents twice. You take on: N× the crawl cost, N× the token lifecycle, coverage gaps for users who never connect, per-user rate limits, and enormous storage duplication unless you dedupe content and attach per-user ACL edges (which you should).

My decision rule: **privileged account where the source's permission model is faithfully mirrorable and the corpus is shared; delegated tokens where the permission model is a rule engine or the corpus is intrinsically personal.** In practice that means service accounts for Drive/SharePoint/Confluence/Jira/GitHub/public Slack, and delegated tokens for email, calendar, DMs, and often Salesforce.

The hybrid that actually ships is: privileged crawl for content and coarse ACLs, plus **a just-in-time check using the user's own token or a per-user access API** as a final gate on anything you're about to send to the model. Belt and braces — the mirror gives you retrieval recall and speed, the JIT check gives you correctness you can defend in a security review.

**🗣 Say this in the room:** "I choose per source. Service account where the ACL model is mirrorable and the content is shared; per-user delegated tokens where authorization is a rule engine or the content is personal, like mail. And regardless of which, I put a just-in-time access check between retrieval and the model, because the mirror is an optimization for recall and the JIT check is the thing I'm willing to defend to a security reviewer."

**⚠ Trap:** using a service account for the *crawl* and then forgetting that the service account's own access can change. The integration user gets removed from a Salesforce role, or the domain-wide delegation scope gets trimmed during an admin cleanup, and your connector starts 404ing on documents it used to read. If your delete detection is naive you will now tombstone a third of the corpus. Monitor connector *coverage* — document count and container count per source per day — with an alert on a drop greater than a few percent, and make a coverage drop pause deletion propagation rather than execute it.
### Define permission mirroring for me precisely. What exactly are you replicating, and what invariant are you trying to hold?

Permission mirroring is maintaining a local, queryable replica of a foreign system's authorization relation — the set of `(principal, resource, action)` triples it would grant — accurate enough and fresh enough that you can pre-filter a search index by it without ever showing a user something the source would deny.

State the invariant precisely, because vagueness here is what gets people rejected. The invariant is **one-sided**: for every document `d` and user `u`, if your system returns `d` to `u`, then the source grants `u` read on `d`. The converse — if the source grants, you return — is *not* an invariant. It's a quality metric. This asymmetry is the whole design: **false positives are security incidents, false negatives are relevance bugs.** Every trade-off you make should be biased toward under-permissioning, and you should say that sentence out loud.

What you replicate is three distinct graphs, and conflating them is the classic mistake:

1. **The identity graph** — who is a person, and which foreign identities are the same person (Slack `U0123`, Google `alice@corp.com`, Jira `accountId:5f…`, Okta `00u…`).
2. **The group graph** — group membership, including nested groups, and the derived transitive closure.
3. **The resource graph** — containers, inheritance edges, and per-resource grants and denials.

Each has a different change rate and therefore a different sync strategy. The resource graph is huge and mostly static. The group graph is small and moderately dynamic. The identity graph is tiny and changes on employee events. Sync them on separate schedules with separate cursors and separate freshness SLOs; a single "permissions sync" job is a design smell.

**⚠ Trap:** materializing the fully-expanded `(user, document)` cross product. It's tempting because query-time is then a single index lookup. But for 20,000 users and 10M documents where an average document is visible to 500 people, that's 5 *billion* edges, and one group membership change rewrites millions of rows. Keep the indirection: documents reference containers and groups, groups reference groups, users reference groups, and you expand *the user side* (small) at query time rather than the document side (huge) at write time.

**🗣 Say this in the room:** "Permission mirroring is one-sided replication of an authorization relation. The invariant I hold is: if we return it, the source grants it. Recall is a quality metric; over-permissioning is an incident. Everything else — how I chunk, cache, expand groups, and set freshness SLOs — falls out of accepting that asymmetry."

### Implement group expansion with nested groups. What's the algorithm, where does it blow up, and how do you keep it fresh?

Group nesting makes membership a reachability query on a directed graph, so expansion is a graph traversal — and the two things that matter operationally are *which direction you traverse* and *when you materialize*.

You almost always want the **user→groups** direction. Given a user, find all groups they belong to transitively; that set is small (typically 10–100) and it's exactly what you need to build a search filter. The document→users direction is the explosive one. So:

```python
def expand_user_groups(user_id: str, edges: Mapping[str, set[str]],
                       max_depth: int = 16) -> set[str]:
    """edges: child_id -> set(parent_group_ids). Returns transitive groups."""
    seen: set[str] = set()
    frontier = {user_id}
    depth = 0
    while frontier and depth < max_depth:
        nxt: set[str] = set()
        for node in frontier:
            for parent in edges.get(node, ()):
                if parent not in seen:          # cycle + revisit guard
                    seen.add(parent)
                    nxt.add(parent)
        frontier, depth = nxt, depth + 1
    if frontier:
        raise DepthExceeded(user_id, max_depth)  # do NOT silently truncate
    return seen
```

Three things in twenty lines that people get wrong. **Cycles**: Active Directory and Okta both permit group A → group B → group A in practice; without the `seen` guard you spin forever. **Depth limits that silently truncate**: if you cap at depth 8 and return what you have, you've silently *under*-permissioned some users, which shows up as "search is broken for the Munich office" and takes a week to find. Raise, alert, and fail closed. **Direction**: writing this document→users is the same code and a catastrophically different cost profile.

For freshness, I cache the expanded set in Redis keyed `acl:groups:{tenant}:{user}` with a short TTL — 60 to 300 seconds — plus **event-driven invalidation** on any group-membership change from SCIM or the source's directory feed. The TTL is the backstop for missed events, not the primary mechanism. And the cache entry stores the `expansion_version` (a monotonic counter bumped on any group-graph write for that tenant) so you can invalidate a whole tenant with one increment instead of scanning keys.

**💰 Math:** 50,000 users, 8,000 groups, average nesting depth 3, average 12 direct memberships expanding to ~35 transitive groups. Materializing user→groups is 50,000 × 35 = 1.75M edges — trivially a Postgres table with a covering index, or 50k Redis sets averaging 35 members ≈ 40 MB. Materializing the document→user cross product for 10M docs at 500 visible users each is 5B rows: at ~24 bytes/row that's 120 GB *and* it must be rewritten on every group change. The factor of ~3,000× between those two numbers is the entire argument for late expansion, and quoting it is worth more than a paragraph of qualitative reasoning.

**⚠ Trap:** forgetting that groups can grant *and deny*. SharePoint and Confluence both support restrictions, and a naive union-of-grants model silently ignores them. Model the ACL as `(allow_set, deny_set)` with deny winning, and make deny non-cacheable-past-invalidation — a stale allow is a leak, a stale deny is merely annoying.

### Talk to me about permission drift. How do you measure it, and what SLO would you commit to a customer for revocation lag?

Permission drift is the divergence between your mirror and the source, and the reason it needs a name is that it is *invisible by construction*: nothing errors, no queue backs up, no latency moves. The only way you find out is by looking for it deliberately.

I measure it with a continuous **shadow audit**: sample N `(user, document)` pairs per tenant per hour — stratified so you get sensitive containers, recently-changed containers, and a uniform tail — and evaluate each pair both ways. Ask your mirror "would we return this?" and ask the source "does this user have read?" (Drive `permissions` check, Graph, `UserRecordAccess` in Salesforce, Confluence's content-restriction endpoint). Then publish two rates:

- **Over-permission rate** = P(mirror allows | source denies). Target: zero. Any nonzero value is a page, not a ticket.
- **Under-permission rate** = P(mirror denies | source allows). Target: low single-digit percent, tracked as a quality metric.

Sample size matters and interviewers like to see it: to detect an over-permission rate of 1-in-10,000 with reasonable power you need on the order of 30,000 samples, so a 1,000-samples-per-hour audit gives you a meaningful signal in about a day, not an hour. Say that rather than "we'd sample some pairs."

For the SLO, I'd separate three latencies and commit to each differently:

**Revocation lag (a grant is removed).** This is the security-critical one. Where the source pushes ACL change events — Drive changes, Graph delta, Slack membership events, SCIM `PATCH active:false` — I'd commit to under 5 minutes p99. Where it doesn't — Jira security schemes, Salesforce sharing rule edits, Confluence space permission changes that don't touch content — the honest number is your reconciliation interval, which might be 6 or 24 hours. **Do not commit to 5 minutes across the board.** The right move in a customer conversation is a per-source table, and the right move in an interview is to volunteer that the table is uneven and explain why.

**Grant lag (a grant is added).** Minutes to hours. Nobody has ever had an incident because a user waited an hour to see a new document.

**Deprovisioning lag (a human leaves).** This is the one that must be near-instant regardless of source, and the way you get it is by not relying on source connectors at all: SCIM deprovision or SSO session revocation flips a *local* kill switch that fails every query from that principal, independent of whether any downstream mirror has caught up. Target: under 60 seconds, enforced at the identity layer.

**🗣 Say this in the room:** "I'd commit to sub-5-minute revocation where the source emits ACL change events, and to the reconciliation interval where it doesn't — and I'd give the customer that table honestly rather than a single number. Separately, offboarding a *person* is enforced at our identity layer in under a minute, because that path must not depend on any connector being healthy."

**⚠ Trap:** the sensitive-container special case. Customers care disproportionately about a handful of containers — the board folder, the comp spreadsheets, the legal matter workspace. Offer a per-container "high-sensitivity" flag that forces just-in-time verification on every access and skips the mirror entirely. It costs a round trip on 0.1% of queries and it converts an unbounded risk into a bounded one. This is a product answer as much as an engineering one, and product judgment is what the senior bar is testing.

### Just-in-time permission checks at answer time versus baked-in index filters — which do you pick and what does it cost you?

Neither, alone. The correct production architecture is **both, in series**, and the interesting content of the answer is why each one can't be dropped.

**Baked-in filters** — the ACL is stored alongside the vector and the search executes `filter: acl_principals OVERLAPS user_groups`. Cost: essentially zero extra latency if the index supports filtered search natively. Benefit: it preserves *recall*, because the ANN search sees only permitted candidates and the top-k you get back is a top-k of the permitted set. Risk: correctness depends entirely on mirror freshness, so it inherits every drift bug above.

**Just-in-time checks** — retrieve candidates, then ask the source (or a Zanzibar-style authorization service) whether this user may read these IDs, then drop the rest. Cost: a round trip, typically 20–150 ms for a batched check against an internal authz service, and considerably more against a SaaS API. Benefit: it is *authoritative* — it captures every rule you didn't mirror. Risk: it does nothing for recall, because you can only prune what the mirror already surfaced.

So: mirror to get candidates efficiently and with good recall, JIT-check to get correctness before anything is shown or sent to the model. The mirror is allowed to be slightly over-permissive because the JIT check is the actual security boundary; that's what lets you keep the mirror simple.

The two variables that decide how much JIT you can afford are **latency budget** and **fan-out**. If you retrieve 200 candidates and rerank to 20, run the JIT check on the 20, not the 200 — one batched call of 20 IDs, after reranking, before context assembly. That is a 10× reduction in check volume for zero loss of correctness, because anything pruned by the reranker was never going to be shown.

**💰 Math:** a 20-ID batched authz check at 30 ms p95 added to a pipeline with a 900 ms TTFT budget is a 3.3% latency tax. If instead you check all 200 candidates pre-rerank against a SaaS API at 5 IDs per request and 120 ms each, that is 40 sequential-ish calls; even at 8-way concurrency that's 5 × 120 = 600 ms, which is two-thirds of your budget gone. Same security property, 20× the cost, purely from where in the pipeline you placed the check.

**⚠ Trap:** doing the JIT check *after* the model has already read the documents. I have seen this ship: retrieve → stuff into context → generate → filter the citations. The model has now been conditioned on content the user cannot see, and it will paraphrase it in the answer body even with the citation removed. **The permission boundary is the context window, not the citation list.** Filtering citations post-generation is theater.

**⚠ Trap:** streaming before the check completes. If you start emitting tokens and then discover a permission failure, you cannot un-send them. Gate the *start* of the stream on the completed check.

### Your vector index supports metadata filtering. Walk me through why naive ACL filtering can destroy recall, and how you fix it.

Because HNSW and IVF are graph and partition structures built over the *whole* corpus, and a filter applied at the wrong point either restricts the traversal in ways the graph can't satisfy, or throws away work already done.

Three regimes, and the interviewer wants to hear that you know which you're in:

**Post-filtering.** Search top-k over everything, then drop what the user can't see. This is correct and catastrophic when the user's access is sparse. If a user can see 0.5% of a 10M-document index and you retrieve top-100, the expected number of survivors is 100 × 0.005 = 0.5. You will routinely return *nothing* while the right answer sits at rank 4,000. Teams "fix" this by raising k to 5,000, which turns a 5 ms search into a 200 ms one and still doesn't guarantee results.

**Pre-filtering (brute force over the allowed set).** Compute the permitted ID set, then do exact search over it. Perfect recall, but the permitted set may be 200,000 documents and exact search over 200k × 1024-dim vectors is ~800 MB of memory traffic per query — tens of milliseconds at best, and it doesn't use the index at all.

**Filtered ANN search (the real answer).** Modern engines push the predicate into the graph traversal: HNSW visits neighbors and skips non-matching nodes while continuing to expand, with an internal escape hatch to brute force when selectivity gets too low. This is what you want, and the parameter you must tune is the search-effort knob (`ef_search` in HNSW terms) — **it needs to scale up as selectivity drops**, because the traversal is discarding most of what it visits. A fixed `ef_search=100` that's fine for an admin who sees everything gives terrible recall for a contractor who sees 2%.

The structural fix that matters more than the tuning: **partition the index by an ACL-correlated key.** In enterprise search, access is not randomly distributed — it clusters by team, project, space, and site. If you shard the index by container or department, the average user's permitted set falls in a handful of shards, and within those shards their selectivity is 60–90% rather than 0.5%. Now filtered ANN is in its comfortable regime. This is exactly the partition-pruning intuition you'd apply to a Postgres table partitioned by tenant, applied one level down.

**📐 Numbers you must know:** filtered-ANN recall degrades sharply below roughly 1–5% selectivity — that's the regime where most engines fall back to brute force or quietly return fewer results than requested. Above ~20% selectivity the filter is nearly free. Know which side of that line your median user sits on, per tenant; it is the single number that predicts whether your enterprise search "just works" or mysteriously misses things.

**⚠ Trap:** assuming a filtered search returns exactly k results. Many engines return fewer, silently, when the filter is restrictive — and your downstream code, which assumes it got k, produces a thin context and a hedging answer. Instrument `returned_k / requested_k` as a per-query metric and alert on the tail. It is the cheapest early-warning signal for permission-related quality collapse that exists.

### A customer reports that a user saw a document they shouldn't have. You're on call. Walk me through the investigation.

First move, before any debugging: **contain and preserve.** Disable the affected tenant's ability to retrieve from that source (a feature flag, not a deploy), snapshot the relevant index shard and the mirror's ACL rows *as they are now* — because your next diagnostic step is a resync that will destroy the evidence — and freeze log retention on that tenant. Enterprise customers judge you on the first hour, and "we resynced and now it's fine" is the worst possible sentence to say to a security team because it means you can never explain what happened.

Then the diagnosis, as a decision procedure:

**Step 1 — Is it actually a leak?** Ask the source, right now, whether that user has read on that document. Roughly a third of these reports are the source saying yes: the user genuinely had access via a group they forgot about or a link-share, and the surprise is about *expectations*, not permissions. That's a product problem, and the fix is surfacing why the user has access in the citation.

**Step 2 — Was it retrieval or generation?** Pull the trace. Did the document appear in the retrieved set, or did the model recite content it saw in a *different* document — a quoted email, a pasted excerpt, a meeting-notes summary? Content leaks between documents constantly and no ACL system can prevent it. If the sensitive text arrived inside a document the user legitimately could read, your permission system worked and the customer has a content-hygiene problem.

**Step 3 — Was it retrieval or cache?** Check the semantic cache, the answer cache, any prefix cache, and any per-user memory store. A cross-user cache hit is one of the most common real causes and it looks exactly like a permission bug. If your cache key doesn't include the requesting principal's expanded group set (or a hash of it), you have this bug and you should assume it.

**Step 4 — Mirror stale or mirror wrong?** Compare the mirror's ACL row for that document against the source's current ACL and against the source's ACL at query time if you can reconstruct it. Stale (there was a revocation your connector didn't process) and wrong (your ACL translation has a logic bug) have completely different blast radii: stale is one document and a lag metric; wrong is *every document with that ACL shape*, which could be a million.

**Step 5 — Blast radius.** This is the step people skip and it's the one the customer will ask about. If it's a translation bug, run the query "how many documents in this tenant have the ACL shape that triggered it" and "how many queries in the last 30 days returned one of those to a user the source would now deny." You need the audit log to answer that, which is why the audit log has to record retrieved document IDs and the principal's expanded group set — not just the question and the answer.

**🔍 Failure taxonomy — over-permissioning, ranked by how often I've actually seen each:** (1) cache key missing the principal; (2) ACL indirection stale after a container move or re-share; (3) group expansion truncated or the wrong direction; (4) deny rules ignored in translation; (5) JIT check placed after context assembly; (6) test/demo tenant data leaking through a shared index because tenant_id was a filter rather than a physical partition; (7) the source's own misconfiguration. Note that five of the top six are *your* bug, not a sync lag — which is why I don't accept "increase sync frequency" as a remediation until step 4 has ruled out step 2 and 3.

### Explain SAML versus OIDC for an enterprise AI product. Which do you support, and what does SSO actually change in your architecture?

Both, because you don't get to choose — the customer's IdP does, and enterprise buyers with an older Okta or ADFS estate will hand you SAML metadata and expect you to deal with it.

Mechanically: **SAML 2.0** is XML assertions delivered by browser POST, signed with the IdP's certificate, and the whole security of it rests on validating that signature, the audience restriction, the recipient, the `NotBefore`/`NotOnOrAfter` window, and replay protection on the assertion ID. **OIDC** is OAuth 2.0 with an identity layer: you get a JWT `id_token` signed with a key from the IdP's JWKS endpoint, validated on `iss`, `aud`, `exp`, `nonce`, and signature. OIDC is strictly nicer to implement, has better library support, and gives you refresh semantics for free. SAML is what half your enterprise pipeline will require anyway.

What SSO actually changes architecturally is more interesting than the protocol details, and this is where a backend engineer can differentiate:

**Identity becomes external and multi-valued.** The IdP's `NameID` or `sub` is your canonical user identity, and it is *not* the same string as the user's Slack ID or Jira account ID. You need a resolution table, and you need it to survive email changes (people get married, companies rename domains). Key on the IdP's immutable identifier, never on email.

**Sessions become revocable by someone else.** SSO means the customer's security team expects that disabling a user in Okta ends their access to you promptly. That's a *push* they will not send you unless you implement SCIM or session-management standards — so at minimum, short-lived sessions with re-validation against the IdP, and ideally SCIM deprovisioning wired to an immediate local kill switch.

**JIT provisioning becomes a policy question.** Do you create a user record on first successful assertion? Most customers want yes. But then group claims in the assertion may drive authorization, and you must decide whether assertion-carried groups are authoritative or advisory. My rule: **assertion groups are authoritative for product-level roles (admin, member, viewer) and never authoritative for content permissions** — content permissions come from the mirrored source graph, because the IdP has no idea who can read a specific Confluence page.

**⚠ Trap:** trusting the email claim for identity linking. If a customer's IdP asserts `alice@corp.com` and you use email as your join key across Slack, Drive and Jira, then an attacker who can control an email claim — or a customer admin who reuses an address for a shared mailbox — collapses two identities into one. Join on IdP subject, store emails as attributes, and treat cross-system linking as a resolution problem with a confidence score and an audit trail.

**⚠ Trap:** implementing SAML yourself. XML signature validation has a long, ugly history of wrapping attacks where a signed assertion is nested inside an unsigned wrapper and a naive parser reads the wrong one. Use a maintained library, keep it patched, and say so in the room — knowing that you *shouldn't* hand-roll it is the signal.

### Walk me through SCIM. What does it give you, and what breaks in your index when a user is deprovisioned?

SCIM (RFC 7643/7644) is a standardized REST API that the customer's IdP calls on *you* to keep users and groups in sync: `POST /Users` on hire, `PATCH /Users/{id}` on attribute change or `active: false` on suspension, `DELETE` on removal, and `POST /Groups` plus membership `PATCH` operations for group changes. It matters because it inverts the direction — you stop polling a directory and start receiving pushes, which is what makes sub-minute deprovisioning possible at all.

Three implementation realities that separate people who've shipped it from people who've read about it:

**Deprovisioning is usually a soft delete.** Okta and Entra typically send `PATCH` with `active: false` rather than `DELETE`. If you only implement `DELETE`, you will believe you handle deprovisioning and you will not. Implement both, and treat `active: false` as fully terminal for access.

**Group membership arrives as PATCH operations, not full replacements.** You get `{"op": "add", "path": "members", "value": [...]}`. These are ordered and lossy under retry — IdPs retry aggressively and don't always guarantee ordering — so your handler must be idempotent (adding an existing member is a no-op, removing an absent one is a no-op) and you need a periodic full reconciliation against `GET /Groups` because you *will* drift.

**SCIM group membership is not source group membership.** The IdP's "Engineering" group and Google Workspace's `engineering@corp.com` group are different objects that usually, but not always, contain the same people. Do not treat them as interchangeable.

Now, what breaks in the index when a user is deprovisioned — this is the actual question:

The user's *own* access must die immediately, and as I said, that's a local kill switch keyed on principal, checked on every request, independent of connectors. But four other things need attention and most implementations miss at least two. **Their delegated OAuth tokens** — if you crawled their mailbox with their token, that token is now for a disabled account and you must stop using it, purge or quarantine the content per the tenant's policy, and not treat the resulting 401s as a connector outage. **Documents they own** — in Drive and Confluence, a deprovisioned user's content is often transferred to a manager or moved to a holding area, which changes its permissions; you need a reconciliation triggered by the deprovision event, not by the weekly sweep. **Dynamic ACL references** — anything whose visibility was defined as "the reporter" or "the assignee" now points at a disabled principal; make sure that resolves to *deny*, not to *unset, therefore allow*. And **their memory/personalization store and their traces** — retention policy applies, and the customer will ask.

**🗣 Say this in the room:** "SCIM gives me a push-based identity feed, which is the only way to hit a sub-minute deprovisioning SLO. I implement `active: false` as terminal, make membership PATCHes idempotent, and run a nightly full reconciliation against GET /Groups because IdP PATCH streams drift. And I fire a deprovision *workflow* — kill switch, token revocation, ownership-transfer reconciliation, memory-store retention — not just a row update."

### OAuth refresh and revocation at scale. You have 40,000 live connections across 300 tenants. What's the design?

Treat tokens as a **leased resource with a background renewer**, not as something you refresh lazily on the request path. The lazy pattern — "call the API, get 401, refresh, retry" — is fine at 10 connections and pathological at 40,000, because failures correlate: a provider blip or a clock skew event expires thousands of tokens at once and you get a refresh stampede against an endpoint that is itself rate-limited.

The design I'd defend:

**Storage.** Refresh tokens are credentials of the highest sensitivity — a stolen Google refresh token is durable read access to a customer's entire Drive. Envelope-encrypt them with a per-tenant data key from a KMS, store ciphertext in Postgres, and make the decrypt path auditable so you can answer "who read this credential and when" during a security review. Never log a token, never put one in a trace, and scrub them from exception payloads — an unhandled `httpx` exception that includes request headers in a Sentry event is a real and common leak.

**Renewal.** A dedicated renewer sweeps tokens expiring within a horizon (say, 20 minutes) and refreshes them, with jitter to spread load, a per-provider concurrency cap, and a distributed lock per connection so two workers never refresh the same token concurrently. That last point is not paranoia: providers that use **refresh-token rotation** invalidate the old refresh token when a new one is issued, so two concurrent refreshes mean one of them writes a token that the provider has already revoked, and you have just permanently broken that connection. Use `SELECT ... FOR UPDATE SKIP LOCKED` on the connection row, or a Redis lock — this is exactly the pattern you'd use for a Celery task that must not double-execute.

**Failure classification.** This is where most implementations are lazy and where the on-call pain lives. `invalid_grant` means the grant is dead — user revoked it, admin removed the app, password changed, token rotation raced — and retrying is pointless and will get you rate-limited. Mark the connection `needs_reauth`, stop all sync for it, and surface it in the customer's admin UI with a re-connect button. A 5xx or a network error is transient — retry with backoff. A 429 is throttling — back off per `Retry-After`. Three buckets, three behaviors; conflating them produces the classic "we retried a dead token 4 million times and got IP-banned" incident.

**Revocation.** Support the customer revoking you at their end (you find out via `invalid_grant`) and revoking at your end via RFC 7009 token revocation when a connection is deleted, a tenant offboards, or a user is deprovisioned. Actually calling the provider's revocation endpoint on tenant deletion is a thing security reviewers check for, and it's about ten lines of code.

**💰 Math:** 40,000 connections with one-hour access tokens is 40,000 refreshes/hour = 11.1 refreshes/second sustained, which is trivial — *if* it's smooth. Without jitter, tokens issued during a Monday-morning onboarding wave all expire in the same minute, and you get 40,000 refreshes in 60 seconds = 667/s against a provider that will happily 429 you at 100/s. The entire cost of fixing this is `expires_at - uniform(0, 15 min)` in the renewer's query. Free, and I've watched it take down a sync fleet.

**⚠ Trap:** assuming refresh tokens don't expire. Google refresh tokens for apps in testing mode expire in days; some providers expire refresh tokens after an idle period (commonly 60–90 days); rotation schemes invalidate on use. **📅 Volatile:** these policies change per provider — verify current behavior rather than trusting a number. What's durable is the design rule: build the `needs_reauth` state, the customer-visible reconnect flow, and the alert on `needs_reauth` count rising, because you *will* need them.

### The same human has six identities across six systems. How do you resolve them, and what happens when you get it wrong?

Identity resolution is a join with no reliable key, and the honest framing is that it's a probabilistic problem you should make as deterministic as possible by pushing the work upstream.

The hierarchy of evidence, best to worst:

**Deterministic from the IdP.** If the customer provisions all their SaaS through Okta or Entra, each system's user record carries a stable external ID that maps back to the IdP subject. Google's directory carries the Workspace ID; Slack's SCIM-provisioned users carry the IdP ID; Jira's `accountId` maps via Atlassian's directory sync. Where this exists, use it and stop. It is the reason I push customers hard toward SCIM: it turns identity resolution from an ML problem into a foreign key.

**Verified email match.** Most systems expose a primary email. It's usually right. It breaks on: shared mailboxes, email changes, aliases (`alice@corp.com` vs `alice.smith@corp.com`), acquired-company domains, and personal-account contamination (someone's Slack is on their gmail).

**Weak signals.** Display name, avatar hash, timezone, behavioral co-occurrence. I would use these only to *suggest* a link for admin confirmation, never to auto-link.

The design consequence is that the identity graph needs a **confidence and provenance model**, not just edges: `(idp_subject, system, external_id, method, confidence, confirmed_by, confirmed_at)`. And it needs an admin UI where a customer can see and correct the mapping, because you will be wrong for 1–3% of a large org and the customer is the only one who can adjudicate.

Now, the consequences of getting it wrong, which is the real question:

**A false merge is a permission leak.** If you link Bob's Jira account to Alice's IdP subject, Alice's group expansion now includes Bob's Jira project roles and she can retrieve his issues. This is the failure mode that matters, and it's why I refuse to auto-link on weak signals. **My rule: only deterministic evidence — IdP-provisioned external ID or verified primary email — creates an authorization-bearing link. Everything else creates a *suggestion*.**

**A false split is a quality bug.** Alice's Slack identity isn't linked to her IdP subject, so her private-channel content never appears in her results. Annoying, invisible, and the user concludes the product doesn't work. Catch it with a coverage metric: per user, how many connected systems have a resolved identity? Users at 3-of-6 are a support ticket waiting to happen.

**⚠ Trap:** case sensitivity and Unicode normalization in email joins. `Alice@Corp.com` and `alice@corp.com` are the same mailbox; `alice@corp.com` and `alicе@corp.com` (with a Cyrillic е) are not, and the second is a real homograph attack surface if any part of your identity input is user-controlled. Normalize with NFKC, lowercase the domain, and be careful about lowercasing local parts — RFC-wise they're case-sensitive, practically every provider treats them as not. Pick a rule, write it down, apply it everywhere.

### External collaborators, guests, and "shared with anyone at the company" content. How does that change the model?

It adds a principal class you probably didn't design for, and it's where over-permissioning most often reaches a *third party* rather than just the wrong employee — which is the version that ends up in a breach notification.

Four principal classes I model explicitly, and the fourth is the one people forget:

`user` (an identified member of the tenant), `group` (mirrored from source or IdP), `public_in_tenant` (everyone in this workspace/domain — a Drive permission of `type: domain`, a Slack public channel, a Confluence space with `confluence-users`), and `external` (guest accounts, B2B collaborators, Slack Connect participants, shared-drive external members, Salesforce community users).

Rules I'd enforce:

**External principals are never members of `public_in_tenant`.** This sounds obvious and is violated constantly, because Slack Connect participants appear in channel member lists and SharePoint guests appear in site groups. If your group expansion naively unions everything, a guest from a customer's vendor inherits "everyone at the company" grants. Type the principal at ingestion and make `public_in_tenant` explicitly exclude `external`.

**External-origin content is labeled and policy-gated.** A document authored by, or shared from, an external org gets an `origin_org` stamp. The tenant policy decides whether it's retrievable, retrievable-but-not-quotable, or excluded. Legal-sector customers (Harvey's market) frequently need the strictest setting because of matter-level confidentiality walls; a Slack-native startup wants the loosest. Make it configuration, not code.

**Guests generally shouldn't use the assistant at all**, or should use a heavily-restricted version. The default I'd ship: external principals get no access to the answer engine unless the tenant explicitly enables it, and when enabled, their retrieval is scoped to explicitly-shared containers only — never to `public_in_tenant`.

**Ethical walls / matter walls are a first-class feature, not an ACL.** In legal and finance, some restrictions are *conflict-of-interest* rules that don't exist in the source system at all: this lawyer must not see anything related to that matter, even content they technically have file-system access to. That's a deny-list evaluated at query time against a tenant-managed policy, sitting above the mirrored ACLs. If you're interviewing at Harvey and you volunteer the phrase "ethical wall" and the observation that it can't be mirrored because it isn't in the source, that's a strong signal.

**⚠ Trap:** counting external users against your identity resolution. A Slack Connect participant has no IdP subject in your customer's directory. If your resolver assumes every identity resolves, it will either error out or — worse — fall back to email matching and link an external person to an internal one who happens to share a display name.

### Would you build a Zanzibar-style authorization service or keep permissions inside the search index? Defend the choice.

I'd build the permission plane as a **separate service with its own storage and its own consistency story**, and have the search index carry only a denormalized, deliberately over-approximate filter key. That's a specific architecture, so let me defend both halves.

Why separate: permissions change on a completely different rhythm and for completely different reasons than content. Content changes when someone edits a doc; permissions change when someone joins a team, and one team change affects a hundred thousand documents. If those live in the same store, a group membership update becomes an index rewrite, and index rewrites are slow, expensive, and eventually-consistent in ways you can't tighten. Separating them means a membership change is one small write in Postgres or a relationship-tuple store, immediately visible to the next query.

Why a Zanzibar-shaped service specifically: it gives you a uniform relation model (`document:d42#viewer@group:eng#member`) that can express *every* source's semantics — inheritance, nesting, deny, dynamic references — in one vocabulary, plus a consistency token so you can say "evaluate this check at or after the point where I applied the revocation." That last property is what turns "we think the revocation landed" into something you can prove. **📄 Paper:** Pang et al. (2019), *Zanzibar: Google's Consistent, Global Authorization System* — introduced relationship-based authorization at Google scale with zookie-based consistency, replacing per-product ACL implementations; SpiceDB and OpenFGA are the open implementations.

Why the index still carries a filter key: the search has to prune *before* the ANN traversal or you're back in the post-filtering recall collapse. So the index stores a coarse key — the set of container/group IDs that could grant access — and the query filters on the intersection with the user's expanded set. It's an over-approximation on purpose. The authorization service then does the precise check on the survivors.

Build versus buy: I'd buy (SpiceDB / OpenFGA / a managed equivalent) unless there's a reason not to, because the hard parts — the check-evaluation engine, caching with correct invalidation, consistency tokens — are exactly the parts that take a team two years to get right and that you can't easily test into correctness. What I'd build is the *translation layer* from each source's model into relation tuples, because that's the part nobody can sell you.

**⚠ Trap:** believing the authorization service removes your need for freshness discipline. It relocates the problem: now the question is how fast a Drive permission change becomes a relation tuple, and that's still your connector's latency. A perfect authz engine fed by a 6-hour-stale mirror gives you 6-hour-stale answers with excellent consistency semantics.

**💰 Math:** a relation-tuple store for a 50,000-user, 10M-document tenant holds roughly: 1.75M user→group edges, ~10M document→container edges, ~200k container→group grants, plus overrides — call it 12M tuples at ~100 bytes ≈ 1.2 GB. That fits in memory on a single node with room to spare, which is why "we can't afford a separate permission service" is almost never a real objection. The cost is a network hop and an operational surface, not storage.
### Every enterprise customer wants their own prompts, their own model, their own retention policy. How do you make tenant configuration a product surface instead of 400 code paths?

The failure mode is well understood and it always starts the same way: a big customer asks for one thing, an engineer adds `if tenant_id == "acme":` behind a flag, it ships on Friday, and eighteen months later there are four hundred of those and nobody can predict what any tenant does. The escape is to stop treating configuration as *code that varies* and start treating it as **data that is validated, versioned, defaulted and evaluated** — the same discipline you'd apply to a database schema.

Four rules, and I'd defend each in review.

**One typed schema for all tenant configuration, with a total default.** A single Pydantic model — `TenantConfig` — with every field having a default, so a tenant with an empty config object is a fully-functional tenant. New fields are added with defaults that preserve existing behavior; that's the backward-compatibility contract. The config is resolved once per request into an immutable object passed down the call stack, never read ad hoc from a global.

```python
class TenantConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: int
    # --- generation
    primary_model: ModelRef = ModelRef.default()
    fallback_model: ModelRef | None = None
    system_prompt_id: str = "std/v14"          # ref into a versioned prompt store
    system_prompt_vars: dict[str, str] = {}    # tenant name, tone, jurisdiction
    max_output_tokens: int = 1024
    # --- retrieval
    retrieval: RetrievalPolicy = RetrievalPolicy()   # k, rerank, hybrid weights
    sources_enabled: frozenset[SourceKind] = frozenset()
    external_origin_policy: Literal["exclude","cite_only","full"] = "exclude"
    # --- governance
    residency: Literal["us","eu","in","au"] = "us"
    retention_days: int = 90
    log_prompts: bool = False
    zdr_required: bool = False
    # --- rollout
    flags: frozenset[str] = frozenset()
```

**Prompts are templates with variables, not free text.** The single biggest source of the 400-code-paths disease is letting each customer have a bespoke system prompt. What they actually want is: their company name, their tone, their jurisdiction, their three domain-specific rules. So ship *one* prompt with slots, plus a bounded `custom_rules: list[str]` capped at, say, ten entries and 1,500 characters. The prompt *id* is versioned and the tenant pins a version, so you can roll a new prompt version tenant by tenant. If a customer truly needs a different prompt structure, that's a signal you've found a new product segment, and the answer is a second named prompt template — not a per-tenant one-off.

**Configuration changes go through the same pipeline as code.** Schema-validated, diffed, reviewed for high-risk fields (residency, retention, model, ZDR), applied with an audit record naming the human, and revertible. Config that can be edited in a database console at 2am is how you get a residency violation.

**Every config field must have an owner and an eval.** If a field changes behavior, there's a test tenant exercising it in CI. Otherwise it's untested production behavior for your largest customer.

**🗣 Say this in the room:** "I'd make tenant configuration a typed, versioned, fully-defaulted schema resolved once per request, with prompts as versioned templates plus a bounded list of custom rules rather than free-text per tenant. The rule I enforce is that a per-tenant branch in code is a bug; the variation must be expressible as data, and if it can't be, that's a product decision to escalate, not a flag to add."

**⚠ Trap:** letting the config schema and the *effective* behavior drift because of flag interactions. With 40 boolean flags you have 2⁴⁰ nominal states and you test maybe eight. Reduce flags to a small number of named **profiles** (`standard`, `regulated`, `air_gapped`, `zero_retention`) that each set a coherent bundle, and allow overrides only from a short allow-list. Testing four profiles plus a few overrides is tractable; testing a flag lattice is not.

### If tenants can pick their own model and prompt, how do you keep evaluation from becoming impossible?

You accept that you can no longer evaluate "the product" and start evaluating **configurations**, and you make the space of configurations small enough to evaluate exhaustively. That reframing is the whole answer.

Concretely, three layers:

**Layer 1 — the golden eval, run against every supported profile.** Not every tenant config, every *profile*. If you support four profiles × three models, that's twelve combinations, and your core eval set (say 400 items covering retrieval quality, faithfulness, refusal correctness, citation accuracy, formatting) runs against all twelve on every prompt or model change. Twelve runs of 400 items is 4,800 generations; at ~1,500 input and 400 output tokens and, say, $3/$15 per million in/out, that's 4,800 × (1,500 × $3 + 400 × $15)/1,000,000 = 4,800 × ($0.0045 + $0.0060) = $50.40 per full sweep. **📅 Volatile:** prices move; the point is that a full profile sweep costs tens of dollars, which is nothing, so "it's too expensive to eval all profiles" is never a real argument.

**Layer 2 — per-tenant eval sets, owned by the tenant.** The highest-leverage thing an FDE does in a deployment is get twenty to fifty real questions with expected answers out of the customer's own people, and turn them into a pinned eval set stored under that tenant. It catches the things a generic eval never will: their acronyms, their document structure, their idea of a correct answer. It also converts vague "the AI is bad" complaints into a regression you can point at. I would make this a formal onboarding deliverable, not a nice-to-have.

**Layer 3 — production-signal evals per tenant.** Thumbs, citation-click-through, answer-abandonment, escalation-to-human rate, and an LLM-judge faithfulness score on a sample of real traffic. These are the only things that scale to 300 tenants, and they're leading indicators: a tenant whose faithfulness sample drops from 0.91 to 0.84 after a config change is a conversation you have *before* they open a ticket.

**⚠ Trap:** letting a tenant edit their system prompt without re-running their eval set. Custom rules interact viciously with your base prompt — a customer adds "always answer concisely" and your citation format instruction silently loses, dropping citation rate from 95% to 60%. **Gate config changes on an eval run.** Save the config, run the tenant's eval set plus the golden set against the new config in the background, and only promote if no metric regresses beyond threshold. This is CI for configuration, and it is the single most senior-sounding thing you can propose in this conversation.

**⚠ Trap:** per-tenant model choice creating an unevaluated matrix. If a tenant pins a model you don't run your golden eval against, you have no idea what they're getting. My rule: **you may only select a model from the set the eval suite covers.** Adding a model to the menu means adding it to the sweep first.

### Give me the tenant-scoping checklist. What are all the places tenant isolation has to hold, and which one gets missed?

I'd rather answer this as an inventory than as a principle, because the principle is obvious and the inventory is where the bugs are. Every one of these is a place I've either seen leak or seen someone almost ship a leak:

**The vector index.** Physically separate namespaces/collections per tenant, or at minimum a partition key — not a metadata filter that a forgotten query path can omit. My rule: **tenant isolation must be a property of the connection or the collection handle, not a `WHERE` clause a developer can forget.** Same reasoning as row-level security in Postgres versus remembering to add `AND tenant_id = ?`.

**The relational store.** RLS enabled, session variable set by middleware, and a test that a query without the variable set returns zero rows rather than everything.

**Object storage.** Prefix per tenant, and IAM policies that make cross-prefix reads impossible, not merely unlikely.

**The semantic cache.** Tenant in the key, *and* the principal's expanded-group hash. See the next question — this is the leak.

**The prefix / prompt cache.** Tenant in the cache-breakpoint content, so no cross-tenant prefix reuse is even representable.

**Memory and personalization stores.** Per user, within per tenant. A shared "learned facts" store across tenants is a breach in a bottle.

**Traces and logs.** Tenant-scoped, and access-controlled so your own support engineers can't browse Tenant A's prompts while debugging Tenant B. If prompts contain customer data — they do — your observability stack is now in scope for the customer's DPA.

**Eval datasets and their outputs.** Tenant eval sets contain real customer questions and often real answers. They are customer data. They get the same retention and residency treatment as production data, which means your eval infrastructure is in your compliance boundary. Almost nobody thinks of this until an auditor asks.

**Metrics.** Cardinality tempts people into dropping the tenant label; then you cannot answer "is this tenant's p95 degraded" and you cannot bill. Keep tenant as a label on the metrics that matter (latency, error rate, tokens, cost, retrieval quality) and aggregate away elsewhere.

**Rate limits, quotas and queues.** Per tenant, or one customer's backfill starves everyone.

**Background jobs.** A crawler task carries `tenant_id` and the credential is resolved from it — never a worker that holds a global admin credential and takes a tenant argument.

**Feature flags and kill switches.** Per tenant, so you can disable a source for one customer during an incident.

**⚠ Trap:** the one that gets missed is **derived artifacts** — the semantic cache, the memory store, the eval set, and the trace store. Everyone remembers the index and the database. Derived artifacts are created later, often by a different engineer, often without a tenant in the key because at the time there was one customer. Write the checklist down, put it in the PR template for anything that introduces a new store, and audit it quarterly.

### Explain exactly how a semantic cache can leak data across users, and how you key it correctly.

The mental model: a semantic cache is a lookup keyed by *meaning* rather than by bytes, and meaning is not a security boundary. Two users asking near-identical questions get the same cache entry — but the correct answer to "what's the Q3 revenue forecast?" is different for a finance VP and a summer intern, because they can retrieve different documents. A cache keyed on the embedding of the question alone will hand the intern the VP's answer, and it will do so *silently, with a cache-hit metric that looks great.*

The correct key is the tuple that fully determines the answer:

```
key = H(
  tenant_id,
  normalized_question_embedding_bucket,   # or exact-text hash for a lexical cache
  principal_authz_fingerprint,            # see below
  retrieval_policy_version,
  prompt_id + prompt_vars_hash,
  model_id + decoding_params,
  corpus_epoch                            # bumped on index alias swap
)
```

The interesting field is `principal_authz_fingerprint`. Using the raw user ID is *safe* but destroys the hit rate — every user gets a private cache and you've built a very expensive memo table. Using nothing is a leak. The right answer is a **stable hash of the user's expanded permission set**: sort the transitive group IDs, hash them, and use that. Now every user in the same permission cohort shares cache entries, which in a real enterprise means engineers share with engineers and the hit rate stays useful, while the intern and the VP are in different cohorts by construction.

Two refinements that matter. The fingerprint must **change when permissions change**, which it does automatically if it's derived from the expanded set — a revocation moves the user to a new cohort and their old entries become unreachable rather than stale-and-wrong. And `corpus_epoch` must be bumped on index changes, or a reindex leaves you serving pre-reindex answers indefinitely; that's the "our RAG went stale after a reindex" bug in one line.

**💰 Math:** suppose a 5,000-user tenant does 40,000 questions/day, and the permission-cohort keying yields a 25% hit rate versus 60% for an unsafe global cache and 4% for per-user keying. An uncached answer of roughly 6k input + 500 output tokens at $3/$15 per Mtok (**📅 Volatile:** frontier per-token prices move; re-derive with current numbers) costs 6,000 × $3/1e6 + 500 × $15/1e6 = $0.018 + $0.0075 = $0.0255 — call it $0.026. The daily spend is then: unsafe-global 40,000 × 0.40 × $0.026 = $416; cohort-keyed 40,000 × 0.75 × $0.026 = $780; per-user 40,000 × 0.96 × $0.026 = $998. So correct cohort keying costs about $364/day more than the unsafe version, and saves about $218/day versus the naive per-user key — $218 × 30 ≈ $6.5k/month for doing the safe thing *intelligently* rather than bluntly. That's the arithmetic that makes the design decision concrete rather than moralistic.

**⚠ Trap:** semantic similarity thresholds on top of a permission-safe key. Even with perfect keying, "what did we decide about the Henderson matter?" and "what did we decide about the Harrison matter?" can land within your embedding-similarity threshold and return the wrong client's answer. That's not a permission bug, it's a *correctness* bug, and in a legal or financial deployment it's worse. My default: semantic caching is off for retrieval-grounded answers over customer corpora, and on only for stable, non-tenant-specific things — classification calls, query rewrites, routing decisions. If a tenant wants it on for answers, they opt in and the threshold is conservative.

### 300 tenants share your infrastructure and one of them just kicked off a 12M-document backfill. How do you stop them from ruining everyone's day?

Three separate mechanisms, because backfill contention, inference contention and provider-quota contention are different problems with different fixes, and answering with only one of them is the tell that someone hasn't run a multi-tenant platform.

**Backfill contention — separate the classes of work.** Backfill and delta must never share a queue. Delta sync is latency-sensitive and small; backfill is throughput-oriented and unbounded. Two queues, two worker pools, and backfill gets a hard concurrency cap per tenant plus a global cap on total backfill workers. Within backfill, schedule with a weighted-fair or deficit-round-robin policy across tenants, not FIFO — FIFO means the 12M-doc tenant's tasks sit in front of every other tenant's for a week. And make backfill *preemptible*: checkpoint every N documents so you can pause a tenant mid-crawl during an incident and resume without losing work.

**Inference contention — per-tenant token budgets, not request budgets.** The unit that matters is tokens per minute, because a request can be 500 tokens or 200,000. A token-bucket per tenant keyed on TPM, with a burst allowance, sitting in your gateway. You already know how to build this in Redis; the only twist is that you must *reserve* on the way in using an estimate and *reconcile* on the way out with actuals, since you don't know the output length in advance. Over-reserve slightly and refund; under-reserving lets a tenant blow through their budget with one long generation.

**Provider-quota contention — this is the one people miss.** Your own capacity is not the binding constraint; your provider's per-org rate limit is. All 300 tenants share it. So the gateway needs a global admission controller in front of the provider with per-tenant fairness, and it needs to degrade gracefully: when the shared quota is saturated, prefer interactive traffic over batch, shed or queue background work (summarization, enrichment, eval runs), and route overflow to a secondary provider or a smaller model if the tenant's config allows it. Batch-tier endpoints exist precisely for the deferrable work — move enrichment and offline eval there.

**📐 Numbers you must know:** the three multi-tenant fairness dials are (1) concurrency cap per tenant on background work, (2) tokens-per-minute bucket per tenant on interactive work, (3) a global admission controller in front of each provider account. If a design has only one of the three, name which failure it doesn't cover.

**⚠ Trap:** enforcing quotas at the API edge only. The 12M-document backfill doesn't come through your API — it's your own scheduler generating the load, against embedding endpoints and the customer's SaaS APIs. The quota system must sit at the *outbound* boundary (a shared client wrapper every caller must use) rather than the inbound one, or your internal jobs bypass every limit you built.

**🗣 Say this in the room:** "Backfill and delta get separate queues with weighted-fair scheduling and preemptible checkpointed tasks; interactive inference gets a per-tenant tokens-per-minute bucket with reserve-and-reconcile; and a global admission controller sits in front of the provider account because the binding constraint is usually the shared provider quota, not our own capacity. Under saturation I shed background work first and degrade to a smaller model before I degrade latency."

### A European customer requires data residency. Walk me through what that actually forces in your architecture.

Residency is not a config flag; it is a **deployment topology decision that partitions your control plane from your data plane**, and the moment you have one EU-resident customer, every future feature has to answer "where does this data live."

Start by enumerating what's actually customer data, because that list is longer than people expect: the raw crawled content, the chunks, the embeddings (an embedding is a lossy but real representation of the source text — treat it as customer data, because a reviewer will), the search index, the prompts sent to the model, the model's outputs, the traces and logs containing both, the semantic cache, the memory store, the eval datasets, the audit log, the backups of all of the above, and the *credentials* for the customer's SaaS connections. Any one of those living in `us-east-1` breaks residency.

What can stay global: your control plane — tenant registry, billing, feature flags, deploy orchestration, aggregate metrics that carry no content. That split is the architecture.

The three parts that actually hurt:

**Model availability by region.** Not every model is offered in every region, and the frontier model your product was tuned around may simply not be available in-region, or may be available only via a different cloud's hosted offering with a different API surface, a different tokenizer, a slightly different behavior, and a different rate limit. **📅 Volatile:** regional availability changes monthly across every provider — verify before you promise. Architecturally, the durable answer is a provider abstraction with per-region model bindings and a per-region eval run, because "same model family, different region" is not the same model in practice and you must prove it with your eval set rather than assume it.

**Cross-region control operations.** Your support tooling, your incident runbooks, your admin console — all of them want to read tenant data, and if your on-call engineer in San Francisco can pull an EU tenant's trace, you have a transfer. Solve it with in-region-only data access, redaction at the boundary, and break-glass procedures that are audited and customer-visible.

**Subprocessors.** Residency isn't only about your servers. If you use a hosted vector DB, an embeddings API, an observability SaaS, an LLM provider, and a reranker API, every one of them is a subprocessor with its own regional footprint, and the customer's DPA will enumerate them. A single US-only observability vendor in your trace path invalidates the whole story.

**💰 Math:** running a full EU stack — its own index, its own workers, its own model endpoints, its own observability — roughly duplicates fixed infrastructure. If your US baseline is $40k/month of fixed cost (index nodes, worker pool floor, observability), the EU region adds most of that again at low utilization, say $28k/month, before a single EU customer's variable cost. That's $336k/year, which means residency needs to be priced into the deal — typically as an enterprise tier — and it means you consolidate: one EU region, not one per country, unless a customer pays for more.

**⚠ Trap:** confusing residency with sovereignty and with zero-retention. A customer may ask for three different things in the same sentence: data *stored* in the EU (residency), data never *accessible* by a US entity (sovereignty — a much harder legal claim involving your corporate structure and cloud provider), and prompts never retained by the model provider (ZDR — a contractual term with your provider, orthogonal to geography). Separate them explicitly in the conversation. Answering "yes we're in eu-west-1" to a sovereignty question is how deals die in legal review three weeks later.

### A customer signs. Size the initial crawl for me — cost, wall clock, and what you tell the AE about go-live.

Let me take a concrete shape and do the arithmetic, because "it depends" is a failing answer and the numbers are what make you credible in a deployment conversation.

**Customer:** 5,000 employees. Google Drive with 6M files, Confluence with 180k pages, Jira with 900k issues, Slack with 4M messages, GitHub with 1,200 repos. Call it, after filtering out non-textual and junk files, **3.5M indexable documents**.

**Content extraction.** Drive is the bottleneck because you must download and parse. Average 400 KB per file for the mixed corpus, so 6M × 400 KB = 2.4 TB of download. At a sustained 200 Mbps of effective throughput across your crawler fleet against the customer's quota (which is the realistic constraint, not your bandwidth), 2.4 TB × 8 = 19.2 Tb / 0.2 Gbps = 96,000 seconds ≈ **27 hours** just to pull bytes. Parsing PDFs and Office documents is CPU-heavy: at 150 ms per document on one core, 3.5M docs = 525,000 core-seconds = 146 core-hours ≈ **5 hours on 32 cores**. Not the bottleneck.

**Chunking and embedding.** 3.5M documents averaging 1,800 tokens gives 6.3B tokens, chunked at 500 tokens with overlap ≈ **14M chunks**. Embedding 6.3B tokens at $0.02 per million (**📅 Volatile:**) = 6,300 × $0.02 = **$126**. That surprises people every time; embedding is not the cost. Throughput: a hosted embeddings API at, say, 5M tokens/minute of provisioned throughput does 6.3B / 5M = 1,260 minutes ≈ **21 hours**. If you self-host on GPUs, a single modern GPU running a small encoder does on the order of 1–3M tokens/minute depending on model and sequence length, so 4 GPUs finishes in roughly the same day. Either way, embedding overlaps with extraction — pipeline them.

**Index build.** 14M chunks at 1,024 dims. Float32 storage: 14M × 1024 × 4 = **57 GB** of raw vectors, plus HNSW graph overhead at ~40% ≈ **80 GB resident**. With int8 scalar quantization that drops to ~14 GB + overhead ≈ 20 GB, which changes your instance class from a ~128 GiB memory-optimized box to a ~32–64 GiB one — roughly a 2–4× cut in instance cost, and the exact multiple depends on what the family charges per GiB. HNSW build at 1–3k inserts/sec/core, parallelized over 16 cores ≈ 14M / 30,000 = **8 minutes to a couple of hours** depending on `M` and `ef_construction`. Build it offline and alias-swap.

**Permission graph.** 3.5M documents × ~1.4 ACL edges after container indirection ≈ 5M tuples, plus 50k user→group and 8k group→group. Minutes.

**The number you give the AE.** With pipelining and a healthy quota: **content live and searchable in 3–5 days, fully caught up in 7–10 days**, and delta sync at steady state within an hour of that. The variance is almost entirely the customer's API quota and how quickly their IT admin actually grants the app registration — which is why the *real* answer includes "the clock starts when we have credentials and elevated quota, and the median gap between contract signature and credentials is two weeks."

**💰 Math summary:** compute for extraction and embedding is a few hundred dollars of one-time cost ($126 embeddings + maybe $200 of CPU/GPU time). Steady-state hosting for that index is the recurring number: an 80 GB float32 index needs ~128 GiB RAM per replica ≈ 2 × r6i.4xlarge-class (128 GiB each) ≈ $1,500/month at on-demand list price, or roughly half that quantized onto 64 GiB nodes — call it ~$700/month. **📅 Volatile:** instance list prices and managed-vector-DB pricing both move, and a managed service costs materially more than raw EC2 — re-derive from a current price sheet rather than quoting this. **The initial crawl is cheap; the resident index and the query traffic are what you price the contract against.** Saying that unprompted is the senior signal in this question.

**⚠ Trap:** sizing on document *count* and forgetting the outliers. One customer's Drive contains a 40,000-page PDF export of a decade of board minutes and 200,000 CAD files. Your p50 is 1,800 tokens and your p99.9 is 900,000. Cap per-document processing, quarantine oversized items to a separate slow lane with an explicit budget, and never let a single document be able to stall a crawl worker for an hour.

### The crawl takes a week and the customer wants a demo on day two. What do you actually do?

You reorder the crawl by expected value and you make the partial state legible instead of hiding it. Both halves matter — the second one is where teams lose trust.

**Reorder the crawl.** Full-corpus completeness is the wrong first objective. Priority order I'd use:

1. **The demo set.** Ask the champion, during kickoff, for the five to fifteen questions they will ask in the demo and the documents that answer them. Crawl those containers first, explicitly. This is not cheating; it's the same thing you'd do for any pilot, and it's what an FDE does on day one.
2. **Recency.** Last 90 days of modifications across all sources. Recency correlates enormously with query relevance in enterprise search — most questions are about current work — and it's cheap because most sources let you filter by modified date.
3. **High-signal containers.** The wiki spaces, the handbook, the runbooks, the top-N most-viewed or most-linked containers if the source exposes analytics. Confluence and SharePoint both do.
4. **Breadth over depth per source.** Get *something* from every connected source early, so the demo shows Slack, Drive and Jira results side by side, rather than 100% of Drive and nothing else. Cross-source synthesis is the product's actual differentiator; demo it.
5. **The long tail.** Everything else, in the background, for a week.

**Make partial state legible.** A per-source indexing status visible in the UI — "Google Drive: 62% indexed, through documents modified before 2024-03" — plus, and this is the important one, **a coverage-aware answer behavior**. When retrieval returns weak results and the tenant is still crawling, the assistant says "I found limited information, and indexing is still in progress for Confluence and Jira" rather than confidently answering from a thin corpus. A confidently wrong answer during a demo kills a deal; an honest "still indexing" does not.

**⚠ Trap:** demoing on a partial index without telling anyone, getting a great demo because the demo set was crawled first, and then having the champion's colleagues try it on day three and get nothing. The trust curve for an enterprise AI product is steep and unforgiving — early users who get a bad answer largely do not come back, and internal adoption is what renews the contract. I would rather gate access to twenty pilot users during the crawl than open it to 5,000 and burn 4,000 of them.

**🗣 Say this in the room:** "I'd crawl in expected-value order — the champion's demo questions, then the last ninety days, then high-signal containers, then breadth across every source, then the tail — and I'd expose indexing coverage in the product so the assistant can say 'still indexing' instead of answering thinly. Adoption is won or lost on the first three answers a user gets, so I'd rather run a gated pilot during the crawl than open it to everyone."

### Onboarding went fine but this tenant says search quality is terrible while every other tenant is happy. Triage it.

The instinct is to reach for retrieval tuning, and that's usually wrong — when quality is bad for *one* tenant and fine for the rest, the cause is almost always coverage, permissions, or corpus shape, not your ranker. My triage order reflects that.

**Is the content actually there?** Run their failing questions against the index with permissions *disabled*, as an admin. If the right document doesn't come back even unfiltered, this is an ingestion problem, not a retrieval problem, and you go look at parse-failure rates, source coverage, and whether a whole container was skipped. Check the parse-failure metric per source per MIME type — a customer whose corpus is 60% scanned PDFs and whose parser silently emits empty strings will have exactly this symptom, and the fix is OCR, not embeddings.

**Is it there but invisible to this user?** Run the same query as the complaining user with permissions on. If it disappears, you have an under-permissioning bug: group expansion missing an edge, identity resolution failing to link their Slack or Confluence account, a container ACL that over-restricts. Check their identity-coverage metric first — how many connected systems resolved an identity for this user.

**Is it there and visible but ranked badly?** Now retrieval is genuinely implicated. The most common tenant-specific cause is **vocabulary mismatch**: the corpus is full of internal acronyms and project codenames that the embedding model has never seen and that tokenize into nonsense. The fixes, in order of leverage: turn up the lexical half of hybrid search (BM25 catches exact codenames that dense retrieval misses entirely), add a tenant-specific synonym/acronym expansion at query time (built from their glossary, or mined from their wiki), and only then consider fine-tuning or swapping the embedding model.

**Is it retrieved but the answer is still bad?** Then it's generation: prompt, context assembly, or reranking. Check whether the right chunk is in the context and the model ignored it — that's a context-ordering or context-length problem — versus the right chunk never made the final k.

**Is the corpus itself the problem?** Enterprise corpora contain contradictory documents: three versions of the expense policy, one current and two from 2019, all equally retrievable. Your system faithfully retrieves and the user says "wrong answer." The fix is not retrieval, it's **freshness signals and authority signals** — boost by recency, boost by container authority (the official handbook space beats a personal folder), and surface document dates in citations so the user can adjudicate.

**🔍 Failure taxonomy — one-tenant quality collapse, in the order I check:** (1) coverage gap (container skipped, connector unhealthy, parse failures); (2) identity resolution gap for the complaining users; (3) permission over-restriction; (4) vocabulary mismatch against a jargon-heavy corpus; (5) duplicate and stale-version pollution; (6) genuinely different question distribution than your eval set assumed; (7) actual ranker regression — last, because if it were the ranker, other tenants would be complaining too.

**⚠ Trap:** accepting "quality is bad" without instances. The first thing I'd ask for is ten specific failing questions with the expected answer and the document that contains it. Without them you're guessing, and with them you have the beginning of a tenant eval set — which is the artifact that prevents the next round of this conversation.

### Per-tenant retention and deletion. A customer asks you to delete everything about one employee. What has to happen?

This is the question that reveals whether someone has actually operated a system with derived data, and the answer is an inventory plus an ordering, because deleting in the wrong order recreates what you deleted.

The inventory of places that employee's data exists: the raw crawled documents they authored or that mention them; the chunks derived from those; the embeddings; the search index entries; the permission graph rows naming them as a principal; their identity mappings; their OAuth tokens; their conversation history with the assistant; their memory/personalization store; the semantic cache entries generated by or for them; the traces of their requests, which contain their questions *and* the document contents retrieved for them; the audit log; your eval datasets if any of their questions were harvested into one; every backup and snapshot of all of the above; and — the one people forget — **anything you sent to a model provider that the provider retains**.

The ordering matters:

1. **Stop the inflow first.** Disable their connections and revoke their tokens, or your connectors will re-ingest what you delete. This is the step that gets skipped and it turns deletion into a game of whack-a-mole.
2. **Delete from the source of truth** — the chunk/document tables — and record a tombstone with the deletion request ID.
3. **Propagate to derived stores** driven by those tombstones, not by a parallel script: index deletes, permission-graph deletes, cache invalidation by tenant+principal prefix, memory store purge.
4. **Traces and logs** by retention policy: either delete the matching records or, if your trace store can't do targeted deletes efficiently, rely on a short TTL and tell the customer the honest number ("traces expire in 30 days; we will not be able to purge them individually before then" is a legitimate position if it's in your DPA).
5. **Backups** — you cannot surgically edit a backup. The industry-standard, defensible answer is that backups are immutable and expire on a stated schedule (say 35 days), deleted data is not restored on recovery because restores replay the tombstone log, and you document this. Claiming you delete from backups is either a lie or an enormously expensive engineering commitment.
6. **Verify and attest.** Produce a deletion report: what was deleted, from which stores, at what time, with counts. Customers in regulated industries need this artifact, not just your word.

On **model providers**: this is why zero-data-retention terms matter. If your provider retains prompts for 30 days for abuse monitoring, then customer content lives there and you must disclose it, and a deletion request cannot reach it. Negotiate ZDR for enterprise tiers and say so on the subprocessor page.

**⚠ Trap:** treating embeddings as not-personal-data. An embedding is a deterministic function of the text and, with the right attack, partially invertible; regulators and security reviewers increasingly treat it as derived personal data. Delete the vectors. Do not argue the philosophy in a DPA negotiation — you will lose and it will cost you weeks.

**⚠ Trap:** deletion that breaks the audit log. Compliance wants both "delete this person's data" and "keep an immutable record of who accessed what." Resolve it by designing the audit log to reference identifiers and content *hashes* rather than content, so it survives deletion of the underlying data while remaining meaningful. Decide this at schema-design time; retrofitting it is brutal.

### You want to roll out a new model version. You have 300 tenants, some regulated. How does that rollout work?

The mental model I'd insist on: **a model change is a dependency upgrade with unbounded, untyped behavioral surface.** You cannot diff it, you cannot unit-test it, and it can regress a behavior nobody wrote down. So the rollout machinery has to be evaluation-gated and reversible per tenant, and it must be a first-class product capability rather than a deploy.

The mechanism:

**Model reference is tenant config, pinned by version.** No tenant ever gets "latest." When a provider deprecates a version you have a scheduled migration, not a surprise. **📅 Volatile:** provider deprecation windows are typically months, not years — track them as a formal dependency with an expiry date, the same way you'd track a TLS certificate.

**Stage 1 — offline eval.** New model against the golden set across every supported profile, plus every tenant eval set you have. This is the cheap gate and it should be automatic. Report per-metric deltas, and treat *any* regression in refusal correctness, citation accuracy, or structured-output validity as blocking regardless of average quality gains, because those are the ones that break integrations silently.

**Stage 2 — shadow.** Mirror a sample of real production traffic to the new model without serving it, and score both with an LLM judge plus deterministic checks. This catches distribution shift your eval set doesn't cover — which is most of it. Shadowing costs you double inference on the sampled slice: at 5% sampling of 2M daily requests, that's 100,000 extra generations/day at $0.026 each = $2,600/day. Run it for three days, spend $7,800, and consider that cheap insurance against a regression across 300 tenants.

**Stage 3 — canary by tenant tier.** Internal tenant → a handful of design partners who've agreed to be early → the general population in waves of 10%, 25%, 50%, 100% → regulated and high-sensitivity tenants **last and only with explicit opt-in**. Regulated customers frequently have change-control obligations that require notice; shipping a model change to a bank without notice is a contract problem, not just a quality risk.

**Stage 4 — automated rollback.** Per-tenant guardrail metrics with automatic revert: refusal rate, faithfulness sample, citation rate, structured-output parse failure rate, p95 latency, cost per resolved task. Rollback is a config write, which is why the model was config in the first place.

**⚠ Trap:** rolling out by percentage of *requests* rather than percentage of *tenants*. Splitting per request means a single user gets different models across turns in the same conversation, which produces visibly inconsistent behavior and makes every complaint unreproducible. **Bucket by tenant, and within a tenant by conversation, never by request.** This is the same stickiness reasoning you'd apply to a session-affinity decision, and interviewers notice when you reach for it.

**🗣 Say this in the room:** "Model version is pinned per tenant in config, so a rollout is a config change with an eval gate in front of it and an automatic revert behind it. Offline eval across profiles, then shadow traffic scored by a judge, then canary by tenant tier with regulated customers last and opt-in. And I bucket by tenant and conversation, never by request, because per-request splits make every quality complaint unreproducible."
### Treat the customer security review as an engineering constraint rather than a sales chore. What does it actually force you to build?

The reframe that makes this useful: **a security questionnaire is a specification written by someone who has never seen your code, and the honest way to pass it is to have already built the things it asks about.** Teams that treat it as paperwork spend two weeks per deal writing careful prose, and then a pen test or an auditor finds the gap anyway. Teams that treat it as a spec build a handful of capabilities once and answer in a day.

What a serious enterprise review — SOC 2 Type II report request, a CAIQ or vendor-specific questionnaire, an architecture call with their security team, and increasingly an AI-specific addendum — actually forces you to have built:

**Tenant isolation you can describe at the storage layer.** "We filter by tenant_id" fails. "Each tenant has a dedicated index namespace and object-storage prefix, and RLS is enforced at the database session level with a test that asserts an unset session variable returns zero rows" passes. The difference is not rhetoric; it's whether isolation is structural or conventional.

**Encryption in transit and at rest, with key management you can name.** Including: are customer OAuth refresh tokens envelope-encrypted with a per-tenant key, and who can decrypt them. Expect a follow-up on BYOK/CMK for regulated buyers — being able to say "customer-managed keys on the enterprise tier" versus "we're looking into it" is worth real money.

**Access control on your own side.** Which employees can read customer content, under what approval, with what logging, and how that's revoked on termination. Break-glass procedures that are audited. This is the question where a lot of AI startups quietly fail, because in practice half the engineering team can read production traces containing customer documents.

**A data-flow diagram that includes every model provider.** Where does customer data go, which third parties see it, is it used for training, is it retained, for how long, in what region. This is the AI-specific part and it's now the most scrutinized section of any questionnaire in this category.

**Logging and audit** to the standard I'll describe in a moment.

**Vulnerability management, dependency scanning, and an incident-response plan with stated notification SLAs.** The notification SLA ends up in the contract — commonly 24 to 72 hours to notify on a confirmed breach — and it is an engineering commitment, because you cannot notify in 24 hours if it takes you a week to determine blast radius. That's the connection people miss: **your audit log design is what makes your contractual notification SLA achievable.**

**🗣 Say this in the room:** "I treat the security questionnaire as a spec I build against before the deal, not prose I write during it. The two items that are genuinely engineering work rather than policy are structural tenant isolation and an audit log rich enough to compute blast radius inside our contractual notification window — and I'd design both before the first enterprise customer, because retrofitting either one is a quarter of work under deadline."

**⚠ Trap:** answering a questionnaire aspirationally. Every "yes" becomes a contractual representation and, in a Type II audit, a control that gets tested over a period. A "no, and here's our roadmap" costs you a follow-up conversation; a "yes" you can't evidence costs you the customer and possibly a breach-of-contract claim.

On **pen-test findings**, which is where this becomes concrete: large customers increasingly reserve a contractual right to test you themselves, or to receive your latest third-party test report. Treat an inbound customer pen test as a scheduled engineering event, not an interruption. Give them a dedicated non-production tenant with seeded synthetic data, a scope document, and a rate-limit exemption — otherwise they will report your WAF as a finding. Expect the findings to cluster in three places for a product of this shape: **prompt injection reachable through indexed content** (a tester plants an instruction in a Confluence page and gets your agent to call a tool), **improper output handling** (your markdown renderer executing what the model emitted, or a generated identifier flowing into a query), and **authorization at the object level** — a document ID or tenant ID in a request path that isn't re-checked server-side. Have a triage rubric ready that maps each finding to severity, a remediation owner and a date, and send the customer a written response per finding. The response document is the deliverable; the fix is table stakes. And where a finding is genuinely unfixable in the current state of the art — no complete defense against indirect prompt injection exists — say that plainly and describe your containment architecture instead, because a vendor who claims injection is solved is one a competent reviewer will stop trusting entirely.

### The customer's legal team sends a DPA and asks for your subprocessor list. What's the engineering reality behind that, and what's specific about the AI supply chain?

The engineering reality is that **your architecture diagram is your subprocessor list**, and most AI products have a longer one than their engineers realize. Walk a typical stack: the cloud provider, the managed Postgres, the object store, the hosted vector database, the embedding API, one or two LLM providers, a reranker API, an OCR or document-parsing service, an observability/tracing SaaS, an error tracker, an email/notification provider, a support tool that ingests conversation transcripts, and possibly an eval or annotation vendor. Each one that touches customer content is a subprocessor that must be disclosed, contractually flowed down (your DPA obligations must be mirrored in your agreement with them), and covered by your incident-notification chain.

What is specific to the AI chain, and what enterprise legal teams now ask by name:

**Training use.** Is customer data used to train or improve the provider's models? The expected answer for enterprise tiers is no, contractually, and you need to point at the provider's terms, not just assert it. Note that consumer and default API tiers of the same provider can have different terms — the distinction between "our enterprise agreement" and "the default terms of service" is exactly what a reviewer is probing.

**Retention.** Most providers retain prompts and completions for some window for abuse monitoring. Zero-data-retention arrangements exist on enterprise agreements and are usually what a regulated buyer requires. If you don't have ZDR, say so and state the window; if you do, be prepared to show it applies to *every* endpoint you call, because ZDR often doesn't cover batch, fine-tuning, or certain safety features.

**Sub-subprocessors.** Your LLM provider runs on someone's cloud, possibly in a region you didn't choose. Enterprise buyers in regulated sectors will ask, and "we don't know" is a bad answer.

**Change notification.** DPAs typically require you to notify customers before adding a subprocessor and give them a window to object. That's an engineering constraint on your ability to swap providers: **you cannot route traffic to a new model provider on Tuesday because the old one had an outage, if that provider isn't already on your disclosed list.** The design consequence is that you pre-disclose your fallback providers and get them approved during onboarding, so failover is contractually legal. I've seen this exact issue block a failover during a real incident.

**⚠ Trap:** the hidden subprocessor in your observability stack. Traces contain prompts; prompts contain customer documents. If you ship traces to a US-hosted SaaS and you told an EU customer their data stays in the EU, you have a problem that no amount of index-region configuration fixes. Audit your telemetry the same way you audit your data plane — that's where the undisclosed transfers live.

**💰 Math:** the procurement cost of getting this wrong is measurable. An enterprise security review that goes cleanly is roughly 2–4 weeks; one that surfaces an undisclosed subprocessor or a retention gap adds a remediation cycle plus re-review — commonly 6–10 weeks. On a $400k ACV deal with a quarterly sales cycle, slipping a quarter is $100k of recognized revenue moved, plus the risk the deal doesn't close at all. That is why I'd argue for building the subprocessor inventory as a maintained artifact in the repo, generated from the actual list of outbound integrations, rather than a Confluence page someone updates when they remember.

### Design the audit log. What does a compliance team actually need to see, and what would you store per request?

The mental model: the audit log answers one question under adversarial conditions — **"prove to me what this person could see, what the system actually showed them, and who authorized it."** That's a different requirement from application logging or from tracing, and if you try to serve all three with the same store you get something that's too expensive to retain and too noisy to search.

So I'd design it as a separate, append-only, tamper-evident stream with a long retention (one to seven years depending on the customer's sector), no free-form fields, and a schema that survives deletion of the underlying content.

Per answered question, I'd record:

```
event_id, ts_utc, tenant_id, session_id, request_id
actor: { idp_subject, display, ip, client, auth_method, session_issued_at }
authz_snapshot: { expanded_group_hash, principal_class, jit_check_result,
                  authz_service_consistency_token }
query: { text_hash, text (if tenant policy permits), rewritten_query_hash }
retrieval: { index_alias, corpus_epoch, retriever_version,
             candidates: [{doc_id, source, container_id, score, acl_decision}],
             served: [doc_id...] }
generation: { model_id, model_version, prompt_id, prompt_version,
              temperature, tool_calls: [{name, args_hash, result_hash}],
              output_hash, output (if policy permits), token_counts }
policy: { guardrails_fired: [...], redactions_applied: n, refusal_reason }
outcome: { delivered: bool, latency_ms, cost_usd }
```

The design decisions worth defending:

**Hashes alongside content, always.** Storing `text_hash` and `output_hash` means the audit record remains meaningful and verifiable after the content itself is deleted under a retention or erasure request. That's how you satisfy "keep an immutable access record" and "delete this person's data" simultaneously.

**Record the *candidates*, not just what was served.** Blast-radius analysis after a permission bug asks "which documents did we surface to whom" — and the near-misses matter, because a doc that was retrieved and then dropped by the reranker still passed through your permission filter and tells you whether the filter was wrong.

**Record the authorization snapshot, not just the user.** `expanded_group_hash` plus the authz service's consistency token lets you reconstruct *why* the system believed the user had access, months later, after the group has changed. Without it, every investigation ends in "we can't tell."

**Record model and prompt versions.** "Which model version answered this question" is a literal question on AI-specific vendor questionnaires now, and in regulated sectors it's how a customer defends their own use of your product. If you can't answer it per request, you fail.

**Tamper evidence.** Append-only storage with object-lock or equivalent, plus a periodic hash chain (each batch's digest includes the previous digest) written to a separate trust boundary. You don't need a blockchain; you need to be able to say "an operator with database access cannot silently alter history," and a chained digest published to a write-once store gets you there.

**⚠ Trap:** conflating the trace store with the audit log. Traces are high-cardinality, high-volume, retained 7–30 days, and sampled. Audit records are one per user-facing action, retained for years, never sampled. Sampling the audit log is the same class of error as sampling your accounting ledger — and I have seen a team discover during an incident that the specific request under investigation wasn't in the 10% sample.

**💰 Math:** at 2M answered questions/day, an audit record of ~4 KB (hashes and IDs, content stored separately under its own policy) is 8 GB/day = 2.9 TB/year. On object storage with lifecycle tiering, that's on the order of $60–$700/month depending on tier and region — trivially affordable, which removes the only argument people make against full, unsampled audit logging. The expensive version is storing full prompt and output text; make that a per-tenant policy field, defaulting off, and priced accordingly.

### A customer requires an air-gapped or VPC deployment. What actually breaks, and how do you architect for it?

Start by separating three genuinely different topologies, because customers use the words interchangeably and the engineering is not the same.

**Single-tenant SaaS in your cloud.** Their own database, index, and workers in your account, isolated at the infrastructure level. Easiest for you, and it satisfies a surprising number of "we need isolation" requests. Nothing breaks; your cost model changes because you now run a floor of infrastructure per customer.

**Customer VPC / BYO-cloud.** Your software runs in *their* cloud account, on their network, against their data, with a control plane you operate. This is what most enterprise "deploy in our environment" requests mean. The hard parts are all operational: you cannot SSH in, you cannot see their logs, you cannot deploy on your schedule, and you now support N versions of your software simultaneously.

**Air-gapped.** No egress at all. This is the one that breaks the product, not just the operations.

What breaks in an air-gapped deployment, concretely:

**No frontier API.** You must ship a model that runs on their hardware. That means an open-weight model, which today means a real capability gap on hard reasoning and long-context tasks — and it means the *entire* product has to be evaluated against that model, not against the one you developed on. Prompts tuned for a frontier model routinely fall apart on a smaller open-weight one: fewer instructions followed, weaker tool-calling, worse structured-output compliance. My rule is that the air-gapped SKU gets its own eval baseline and its own prompt set, and you set customer expectations against *that* baseline, not your SaaS one. **📅 Volatile:** the open-weight capability gap narrows continuously — re-verify the current best open-weight option and its license terms before quoting anything.

**No hosted embeddings, reranker, OCR, or safety classifier.** Every one of those becomes a container you ship and a GPU you size for.

**No telemetry.** You get no traces, no metrics, no error reports. Debugging becomes "ask the customer to run a diagnostic bundle collector and email you a tarball." Build that collector on day one — a CLI that gathers redacted logs, config, version manifest, index stats, and health checks into a single artifact — because without it every support interaction is a week.

**No dynamic updates.** No model updates, no prompt updates, no dependency patches without a scheduled, customer-approved release. Which means your release artifact must be a fully self-contained, hash-pinned bundle: images, model weights, tokenizer, index schema, migration scripts, and a manifest.

**No license checks, no usage metering, no phone-home.** Metering becomes an offline report the customer submits.

**📐 Numbers you must know for GPU sizing in an air-gapped deployment:** weights memory ≈ parameters × bytes-per-parameter. A 70B-parameter model at bf16 (2 bytes) is 140 GB of weights — it does not fit on a single 80 GB accelerator, so you need at least two with tensor parallelism, plus headroom for the KV cache. Quantized to int4 (0.5 bytes) it's 35 GB and fits on one, with real quality cost you must measure rather than assume. This arithmetic — parameters × bytes, then add KV cache — is what lets you answer "what hardware do we need" in a deployment conversation without deferring to someone else, and being able to do it live is a strong differentiator in an FDE loop.

**⚠ Trap:** promising an air-gapped deployment with "the same quality." You cannot, and a customer who discovers the gap after signing is worse than a customer who declines. Be explicit: here's the eval delta between our hosted SKU and the on-prem SKU on your task class, measured, with numbers.

### In a customer-VPC deployment you can't see their logs and can't deploy on your schedule. How do you engineer for that?

The core discipline is that **you are now shipping software, not operating a service**, and every habit that a SaaS team has — deploy daily, read production, hotfix in an hour — is unavailable. The architecture has to compensate.

**Split control plane from data plane, hard.** The data plane runs in their VPC and touches customer content. The control plane runs in yours and holds: version registry, license/entitlement, aggregate non-content telemetry (counts, latencies, error *types* — never payloads), config distribution, and update orchestration. The contract is that the data plane sends *nothing* containing customer content upward, and that constraint has to be enforceable and demonstrable — a reviewer will ask you to prove it, so make the egress allowlist explicit and the telemetry schema a typed, reviewable artifact rather than a log-forwarding rule.

**Ship a release artifact, not a deploy.** A versioned bundle with a manifest, hash-pinned images, and idempotent, forward-only migrations. Support at least N−2 versions in production simultaneously and test upgrade paths across them, because a customer will be on a version from six months ago and will jump three releases at once.

**Make every failure self-describing.** Since you can't inspect anything, the software must explain itself: rich health endpoints (per connector: last successful sync, cursor age, error class, coverage counts), a diagnostic bundle command, and error messages that name the remediation rather than a stack trace. Every error a customer's admin can see should have a stable error code that maps to a documented runbook entry.

**Configuration is declarative and validated up front.** A `validate` command that checks connectivity, credentials, scopes, quotas and resource sizing *before* installation, and fails loudly with specifics. Half of on-prem support load is "we granted the wrong OAuth scope"; a validator turns a two-day email thread into a one-line error.

**Assume you'll be asked to reproduce a bug you can't see.** So invest in a deterministic replay path: given a diagnostic bundle with a redacted trace, can you re-run the pipeline locally against synthetic data of the same shape? If the answer is no, every on-prem bug costs a week.

**💰 Math on why this changes your business, which is worth mentioning:** if a hosted tenant costs you $2k/month of infrastructure and 0.05 support-engineer-months, and a VPC tenant costs $0 of your infrastructure but 0.4 support-engineer-months (upgrade coordination, diagnostics, version-specific bugs), then at a $25k/month fully-loaded engineer, the VPC tenant costs you $10k/month of human time versus $2k + $1.25k = $3.25k. **The VPC SKU has to be priced roughly 3× the hosted one to have the same margin**, and the engineering investments above are exactly what pull that 0.4 down toward 0.15. Framing on-prem support cost as an engineering optimization target, with the arithmetic, is a genuinely senior move in this conversation.

**⚠ Trap:** letting on-prem and SaaS diverge into two codebases. It always starts as one flag and ends as two products, at which point every feature costs double and the on-prem one rots. Enforce one codebase, one artifact, with capabilities *disabled by configuration* in restricted deployments — and run your CI against an air-gapped-profile integration test (no egress, local model, no telemetry) so the restricted path is exercised on every commit, not on every release.

### The customer's region doesn't have the model you built on. What do you do?

Decompose the request first, because "we need it in region X" usually bundles three separable requirements: the data must be stored in region X, the inference must be *executed* in region X, and no data may cross a border even transiently. The third is the expensive one and only some customers actually need it.

If only storage residency is required, you keep your existing model routing and you're done — you just have to be honest in the DPA that inference occurs elsewhere, and many customers accept that with a ZDR term. If inference must be in-region, you have four options and I'd evaluate them in this order:

**The same provider's in-region endpoint.** Best case. Verify not just availability but *parity*: same model version, same context limit, same feature set (structured outputs, tool calling, caching, batch), and same rate limits. Regional endpoints frequently lag on features, and a missing structured-output mode will break your extraction pipeline in a way that no amount of prompt work fixes cleanly.

**A different hosting surface for the same model family** — the model offered through a cloud marketplace in-region. Watch for behavioral drift: different serving stacks can differ in default sampling parameters, tokenizer handling, and safety filtering, so **you re-run your eval set against the regional endpoint and treat it as a different model until proven otherwise.** I'd make that a hard rule; assuming parity is how you ship a silent quality regression to your first EU customer.

**A different provider in-region.** Now you're in a real migration: prompts, tool schemas, structured-output syntax, and token accounting all differ. Budget the eval work, and note the subprocessor-disclosure obligation from earlier — you must have pre-disclosed them.

**Self-hosted open weights in-region.** Full control, no availability question, and you own the serving stack, the GPU capacity planning, and the capability gap. This is the right answer when the regional customer set is large enough to amortize it or when sovereignty requirements rule out every hosted option.

The architectural investment that makes all four cheap is a **provider abstraction with per-region model bindings and a per-binding eval report.** Not a lowest-common-denominator wrapper — those are a trap, because they hide the features you actually need — but a capability-aware layer: each binding declares what it supports (JSON schema mode, tool calling, prefix caching, max context), and your pipeline degrades explicitly and visibly when a capability is absent rather than failing at runtime.

**⚠ Trap:** assuming your token accounting transfers. Different model families tokenize differently; the same document can be 15–25% more tokens on one tokenizer than another. Your context-window budgeting, your chunk-packing logic, and your cost model all shift. If your context assembler hard-codes a token budget derived from one tokenizer, you will silently truncate context on the regional deployment and wonder why answer quality dropped.

**🗣 Say this in the room:** "I'd separate storage residency from inference residency, because most customers need the first and only some need the second. Where inference must be in-region, I treat a regional endpoint as a distinct model — re-run the eval set, check feature parity on structured output and caching, re-derive the token budget — because assuming parity across regions is how you ship a silent quality regression to your most compliance-sensitive customer."

### Migration and export keeps blocking deals. Why, and what do you build?

Because a buyer signing a seven-figure multi-year contract is doing risk analysis, and the risk they care most about is not that your product fails — it's that it succeeds, becomes load-bearing, and then they can't leave. Procurement calls it vendor lock-in and it is frequently a *contractual requirement* in enterprise agreements, sometimes with a specified format and a maximum turnaround. So export is not a feature you add when someone churns; it is a feature you demo during the sales cycle.

What "export" actually means, in ascending difficulty:

**Configuration export.** Their tenant config, prompts, custom rules, source configurations, user and group mappings, in a documented JSON or YAML schema. Easy and it should be self-service.

**Content and permission export.** The crawled documents and the mirrored ACL graph. Note the subtlety: this is *their* data from *their* systems, so the real value isn't the content, it's the derived structure — the identity resolution, the permission graph, the chunk boundaries. Export it in an open format (Parquet plus a documented schema) rather than a proprietary dump.

**Interaction history.** Conversations, questions, answers, citations, feedback. This is the genuinely valuable asset, because it's the customer's own institutional record of what people asked and what the system said, and increasingly it's the training/eval data for whatever they do next. Withholding it is what makes a customer feel trapped, and it will come up.

**Embeddings.** Contested. They're derived from your pipeline choices and only useful with the same model. I'd export them with a manifest naming the model and dimension, because refusing looks worse than the marginal lock-in it buys — and any competent competitor will just re-embed anyway.

**Audit logs.** Often required for the customer's own compliance retention beyond the life of the contract.

The engineering discipline that makes this cheap: **build export against the same schemas you already version for internal use, and run it in CI.** An export job that's written once during a churn event is always broken. Run a nightly export of your internal test tenant, validate the artifact against its schema, and re-import it into a clean instance. That round-trip test is the only thing that keeps export honest, and it also gives you tenant migration between regions and between deployment topologies for free — which is a capability you will need anyway when a customer moves from hosted to VPC.

**⚠ Trap:** the export that technically exists but produces a 400 GB unstructured tarball with no schema, delivered in six weeks. Contract language increasingly specifies "commercially reasonable format" and a turnaround window, and a hostile export is a reputational event in a market where enterprise buyers talk to each other. If you're going to have it, make it good; the marginal engineering is small and it removes an objection from every deal.

**🗣 Say this in the room:** "Export is a sales-blocking requirement, not a churn feature, so I build it early and test it in CI by round-tripping a test tenant nightly. It also gives me tenant migration between regions and deployment topologies for free, which I'll need for residency and VPC customers regardless — so it's one investment that closes deals and unblocks two other roadmap items."

### 🏋 Drill: forty-five minutes, no assistance. Design the full integration surface for a 12,000-person law firm.

**The brief.** Harvey-shaped deployment. Sources: iManage or SharePoint document management (18M documents, matter-organized), Outlook mail (12,000 mailboxes), a matter/billing system exposed via REST, and Teams. Requirements: EU residency for the London office, ethical walls between matters, seven-year audit retention, a partner-level demo in three weeks, and a security review from a firm whose clients are banks.

**Produce, in 45 minutes, on paper:**

1. The connector table: per source, the change-feed mechanism, cursor type, deletion-detection tier, whether you use a service account or delegated tokens, and the ACL dialect you're mirroring. One row per source at minimum, and split a source into several rows wherever the mechanism genuinely differs — SharePoint document libraries versus lists, Teams channel messages versus the files behind them, mail versus calendar.
2. The permission model as a set of relation tuples covering: matter membership, document-level restrictions, ethical walls as a deny layer, and external counsel as a distinct principal class.
3. Crawl sizing arithmetic: bytes to download, documents to parse, tokens to embed, chunks produced, index memory at float32 and at int8, and a wall-clock estimate with the binding constraint named.
4. The tenant config object for this customer, as a typed schema instance.
5. The scoping checklist, marked with which items are at risk given a three-week timeline.
6. Your day-14 demo plan given that the crawl will not be finished.
7. Three answers you'd give in the security review that require engineering you haven't built yet, and what you'd build.

**Pass criteria:** the connector table names a real mechanism per source rather than "poll the API"; ethical walls appear as a query-time deny layer and *not* as mirrored source ACLs, with a sentence explaining why they cannot be mirrored; the crawl arithmetic produces a wall-clock number with the bottleneck identified (it should be download and customer API quota, not embedding); the config object has defaults for every field; and the demo plan gates access rather than opening to 12,000 users. If you cannot do the crawl arithmetic without looking anything up, that is the gap to close first — it is the part of this conversation that most reliably separates candidates.

**🗣 Say this in the room** (the opening you should be able to deliver cold): "Before I design anything I want three numbers and one boundary: how many documents, how many identities, what's the revocation SLO the customer expects — and whether ethical walls exist, because that determines whether the source's ACLs are the whole authorization story or just the floor."

### 🔍 Give me your failure taxonomy for the whole integration surface. If you're on call for this system, what actually pages you and in what order do you look?

Here is the taxonomy I'd write on the runbook's first page, organized by what the *symptom* is, because that's how the page arrives.

**Class 1 — Over-permissioning (page immediately, sev-1).** Symptom: a user reports seeing something they shouldn't; or the shadow audit's over-permission rate goes nonzero. Look, in order: cache key missing the principal fingerprint → stale ACL after a container move or re-share → group expansion truncated or traversed in the wrong direction → deny rules dropped in translation → JIT check placed after context assembly → cross-tenant bleed through a shared index or shared cache. Containment before diagnosis: disable the source for the tenant by flag, snapshot the index shard and ACL rows, freeze log retention.

**Class 2 — Coverage collapse (page, sev-2).** Symptom: document count for a source drops, or user complaints of "search stopped finding things." Cause is usually credential scope loss, a revoked token misclassified as transient, or a 404-driven false tombstone cascade. The guardrail that prevents the worst version: **a coverage-drop threshold that pauses deletion propagation rather than executing it.**

**Class 3 — Freshness regression (ticket, sev-3, escalates on SLO breach).** Symptom: cursor age rising per (tenant, source). Causes: rate-limit throttling, a poison document stalling a worker, a webhook endpoint failing silently, an expired change token forcing a re-crawl nobody scheduled. Alert on *cursor age*, not on job success — a job that succeeds while processing nothing is the failure mode you'll otherwise miss for days.

**Class 4 — Identity drift (ticket).** Symptom: individual users report missing content across a whole source. Check identity-resolution coverage per user before anything else. Root causes: SCIM PATCH drift, email change, an unlinked source account.

**Class 5 — Tenant quality collapse (ticket).** Runs the triage from earlier: coverage → identity → permissions → vocabulary → duplicates/staleness → ranker.

**Class 6 — Noisy neighbor (page if cross-tenant).** Symptom: p95 latency or queue depth up across many tenants. Check backfill worker allocation, per-tenant TPM bucket saturation, and provider-quota admission before you look at your own capacity, because the provider quota is the constraint people check last and it's frequently the answer.

**Class 7 — Compliance-visible failure (page, and involves legal).** Audit log gap, a deletion request that didn't propagate, an undisclosed egress, a residency violation. These are not fixed by engineers alone and the runbook should say so, with the escalation path named.

The metrics that make this taxonomy actionable, and which I'd insist exist before launch: per (tenant, source) cursor age, coverage count and its day-over-day delta, connector error rate by *class* (auth / throttle / transient / poison), over- and under-permission rates from the shadow audit, `returned_k / requested_k` on filtered searches, identity-resolution coverage per user, per-tenant token consumption against bucket, and cache hit rate broken down by whether the key included a principal fingerprint.

**⚠ Trap:** alerting on job completion instead of on *effect*. A sync job that runs every five minutes, succeeds every time, and has been processing zero changes for a week because its cursor is stuck on a swallowed exception will never page you. Every one of the metrics above is an effect metric, deliberately — cursor age, coverage delta, permission rates, returned-k ratio. That distinction is the single most transferable thing from this section, and it's the same lesson as alerting on consumer lag rather than on consumer liveness.

**🗣 Say this in the room:** "I alert on effect, not on execution: cursor age per tenant-source rather than job success, coverage delta rather than document count, over-permission rate from a continuous shadow audit rather than an absence of complaints. And the one guardrail I'd insist on is that a coverage drop pauses deletion propagation instead of executing it — because the worst incident in this system isn't a leak, it's a connector losing access and your pipeline dutifully deleting a third of the customer's corpus."
