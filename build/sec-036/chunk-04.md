### Design the ingestion pipeline end to end. 20 million documents, mixed formats, continuously updated, and it has to be re-runnable.

I will draw it as five stages with a durable artifact between each one, because **the single most important architectural property is that every stage is independently re-runnable without redoing the stages before it.** You will change your chunker four times and your embedding model twice; if those changes force a re-parse of 20M documents you have designed a pipeline that nobody will ever improve.

**Stage 0 — Discovery and change detection.** A crawler or CDC feed per source system (S3, SharePoint, Confluence, Gmail, a Postgres table) emits `(source_uri, external_id, etag_or_mtime, size)`. Compare against a `documents` table in Postgres. Emit work only for new or changed items. **This stage never touches document bytes** — it is cheap and can run every five minutes.

**Stage 1 — Fetch and canonicalize.** Download bytes to object storage under a content-addressed key: `raw/{sha256(bytes)}`. Content-addressing gives you free deduplication across the corpus (the same attachment mailed to forty people is stored once) and makes the fetch stage idempotent by construction. Sniff the real MIME type. Write a `blobs` row.

**Stage 2 — Parse.** Route by MIME type and, for PDFs, by the per-page digital/scanned classifier. Output is the **typed element stream** from the layout-aware chunking design plus a single canonical normalized text, both written to object storage keyed by `{content_hash}/{parser_name}@{parser_version}`. That key is the whole trick: **parse output is a pure function of (bytes, parser version), so it is cacheable forever and a parser upgrade is a new key rather than a destructive overwrite.** This is the most expensive stage — it is the GPU and the paid API calls — and it is the one you most want never to repeat.

**Stage 3 — Chunk and enrich.** Reads the element stream, applies the chunking rules, applies enrichments (section-path prefixing, contextual retrieval, generated captions), writes `chunks` rows to Postgres keyed by `{content_hash}/{parser_ver}/{chunker_config_hash}`. Cheap, CPU-bound, and re-runnable in minutes over the whole corpus — which is exactly why you want it separated from parsing.

**Stage 4 — Embed and index.** Batches chunks, calls the embedding model, writes vectors to the vector index and the tokenized text to the lexical index, under an index alias. Keyed additionally by `embed_model`.

**Stage 5 — Activate.** Atomic alias swap from `docs_v3` to `docs_v4` once the backfill validates. Old index retained for a rollback window.

**The control plane.** Each stage is a Celery/Kafka-style worker pool consuming a queue, with a `document_state` row tracking `(doc_id, stage, status, attempt, last_error, parser_version, chunker_hash, embed_model)`. That row is the source of truth for "what has been done to this document," and it is what makes "re-run stage 3+ for everything parsed with docling@2.13" a single SQL query plus an enqueue. **This is a workflow-orchestration problem you have solved before; the only novel parts are that some stages cost money per item and some stages are nondeterministic.**

**The design decisions I would call out explicitly in the review:**

1. **Content-addressed storage at every stage boundary.** Turns re-runs into cache hits.
2. **Version stamps in the key, not in a mutable column.** You can run two parser versions side by side and diff them, which is how you validate an upgrade.
3. **Parse is separated from chunk.** People fuse these and then cannot iterate on chunking without re-paying for OCR. This is the mistake I look for first when reviewing someone else's pipeline.
4. **Per-page routing for PDFs**, not per-document.
5. **Everything is a queue with a DLQ**, because 20M documents means the long tail of broken files is thousands of documents, not a handful.
6. **Backpressure comes from the slowest downstream stage**, propagated by bounded queues rather than by retries.
7. **A shadow index and an eval gate before the alias swap**, so a bad parser upgrade never reaches users.

**💰 Rough sizing.** 20M documents averaging 12 pages = 240M pages. If 8% need VLM parsing: 19.2M × $0.0053 = **$102k** one-time — which is exactly why the parse cache matters, because if you fuse parse and chunk you pay that again for every chunking experiment. Text: 240M pages × ~500 tokens = 120B tokens; embedding at $0.02/M = **$2,400**; at a premium encoder's $0.13/M = **$15,600**. Chunks: 120B/500 = 240M chunks; at 1024-dim float32 that is 240M × 4KB = **983GB of raw vectors**, which immediately tells you this is a quantized-index or disk-resident problem, not an in-memory HNSW problem, and that constraint should be surfaced on day one rather than discovered at month three.

**🗣 Say this in the room:** "Five stages with a durable, content-addressed artifact between each — discover, fetch, parse, chunk, embed — plus an alias swap to activate. The load-bearing decision is that parse output is keyed by content hash and parser version, so it's cached forever and chunking is re-runnable in minutes. Teams that fuse parse and chunk end up unable to iterate on chunk size because every experiment costs six figures of OCR."

### How do you construct document IDs so the pipeline is idempotent? What's the difference between the document ID and the content hash?

They are two different keys answering two different questions and conflating them is the bug that produces duplicate documents in the index.

**The document ID answers "is this the same document?" — it is an identity key and must be stable across edits.** Derive it deterministically from the source system's stable identifier: `sha256(f"{source_system}:{external_id}")`. A Confluence page ID, an S3 key, a SharePoint item GUID, a database primary key. Never derive it from content, because then editing a document creates a new document and the old one lingers forever. Never derive it from a URL that contains a version, a timestamp, or a query string — I have seen `?utm_source=` produce four copies of the same page.

**The content hash answers "has this changed?" — it is a change-detection key over the *normalized* content.** `sha256(normalized_text)`, where normalization means: after extraction, with page furniture stripped, whitespace collapsed, and — critically — **excluding anything volatile that is not semantic content.** A "Last exported: 2026-08-01 14:22 UTC" footer, a page-render timestamp, a session-scoped download token in a link, an auto-incrementing revision number. If you hash raw bytes, a nightly export re-hashes every document every night and you re-embed a 120-billion-token corpus daily. That mistake costs $2,400 a night.

**Chunk IDs derive from both:** `sha256(f"{doc_id}|{char_start}|{char_end}|{parser_ver}|{chunker_hash}")`. Deterministic, so re-running the chunker produces identical IDs and the upsert is a no-op rather than a duplicate insert.

**The idempotency contract I would write into the design doc:**

```
Re-running ingestion on an unchanged document must produce zero writes
to the vector index and zero embedding API calls.
```

That is testable. Run the pipeline twice on a 1,000-document fixture and assert that the second run's embedding-call counter is zero. **Put that assertion in CI.** It catches the timestamp-in-hash bug, the nondeterministic-parser bug, and the unstable-chunk-ID bug, all of which are otherwise invisible until the bill arrives.

**⚠ Trap — the one that actually happens:** **the document moved.** A file is renamed in SharePoint, or a Confluence page moves between spaces, and if your `external_id` was the path rather than the item GUID, you now have two documents: the new one, and the old one which no longer exists upstream but is still in your index and still retrievable. **A document deleted or moved upstream that is still being quoted to a customer is the highest-severity ingestion bug there is**, and it is caused by an identity key that was not actually an identity. Always prefer an opaque, immutable source identifier over a human-readable path, and run a periodic reconciliation that lists upstream IDs and tombstones anything in your index that no longer appears.

**⚠ Trap two:** the nondeterministic parser. VLM parsing at temperature 0 still varies slightly, so `normalized_text` changes on re-parse, the content hash changes, and change detection fires on documents that did not change. **Hash the aggressively-normalized text** (lowercase, collapse whitespace, strip punctuation runs) for change detection, or cache the parse by input-content hash so you never re-parse an unchanged blob in the first place — which the pipeline above does by construction.

### You upgraded from Docling 2.13 to 2.14 and table extraction changed. Do you reprocess 20 million documents?

Not blindly, and the answer is a decision procedure with a cost gate, because the naive answer costs six figures and the reflexive "no" leaves you with a corpus in two inconsistent states.

**Step 1 — quantify the delta before deciding anything.** Take a stratified sample of 2,000 documents. Parse with both versions into the two content-addressed keys (which cost nothing to keep side by side, by design). Diff the element streams. Compute: what fraction of documents changed at all; of those, what fraction of *chunks* changed; and — the metric that matters — **what fraction of changes touch tables, numbers, or headings** versus whitespace and punctuation. A parser upgrade that changes 40% of documents by a whitespace token is not a quality event.

**Step 2 — measure whether the delta matters to retrieval.** Run your golden set against both indexes on the sampled subset. If Recall@10 and answer correctness are within noise (and you should compute the paired bootstrap interval rather than eyeballing two numbers), the upgrade is not worth a backfill on quality grounds. If tables improved measurably, it is.

**Step 3 — price the backfill.** Only parse is expensive; chunk and embed are cheap relative to it. For our 20M documents / 240M pages, a full re-parse with the 8% VLM share is ~$102k plus weeks of GPU wall clock. **So the real question is never "reprocess or not," it is "reprocess which subset."** Almost always the answer is: **reprocess only documents that contain tables** (you know which, from the element streams you already stored), which might be 15% of the corpus and 15% of the cost — $15k instead of $102k.

**Step 4 — do it as a shadow index with an alias swap, never in place.** Build `docs_v4` alongside `docs_v3`, dual-write new ingestion to both during the backfill, run the golden set against both, and swap the alias only when v4 wins or ties. Keep v3 for a rollback window measured in days. Backfilling in place means a partial backfill leaves you with a corpus where half the tables are parsed one way and half another, retrieval quality is a function of ingestion date, and there is no rollback.

**Step 5 — the drip strategy for cost-sensitive cases.** If the improvement is real but the backfill is unaffordable now, reprocess lazily and by value: re-parse documents on access (a cache-miss-style backfill weighted by query traffic), plus a background trickle ordered by document popularity from the query logs. **The head of your document access distribution is steep — typically a few percent of documents serve most retrievals — so reprocessing the top 5% by access captures most of the user-visible benefit for 5% of the cost.** This is exactly the reasoning you would apply to a cache warm-up and it transfers directly.

**🗣 Say this in the room:** "I wouldn't reprocess on a version bump, I'd measure. Diff both parser versions on a 2,000-document stratified sample, classify the changes as cosmetic versus table/number/heading, and run the golden set against both. If it's a real table improvement I'd reprocess only the table-containing subset — usually 15% of the corpus — as a shadow index with an alias swap, and I'd order the backfill by document access frequency so the top 5% of documents by traffic get fixed in the first day."

**📐 Numbers you must know:** in a corpus with a durable parse cache, the relative cost of the three stages is roughly parse : chunk : embed = 100 : 1 : 3 for a scanned-heavy corpus, and 1 : 1 : 3 for a purely digital one. Knowing which regime you are in tells you instantly whether a chunking experiment is free (it is, always) and whether a parser change is affordable (only in the digital regime).

### 3% of documents fail to parse. What do you do with them, and how does that not become a silent hole in the corpus?

The failure is not that documents fail — with 20M heterogeneous documents, thousands will fail and that is normal. **The failure is that nobody knows which ones, and a user asks about one, and the system says "I don't have information about that" when in fact the document is right there and the pipeline choked on it.** That answer is worse than an error, because it teaches the user the corpus does not contain something it does.

**The mechanism: quarantine, not drop.**

Every stage has three outcomes, not two: success, **retryable failure**, and **permanent failure**. Retryable — a 429, a 503, a timeout, an OOM — goes back on the queue with exponential backoff and a bounded attempt count. Permanent — corrupt file, unsupported format, encrypted PDF, a page that OOMs the parser three times — goes to a `quarantine` table with `(doc_id, source_uri, stage, error_class, error_detail, first_seen, attempts, sample_bytes_key)`. **It never silently disappears and it is never retried forever**, because an infinite retry on a poison document is how one 900MB corrupt PDF consumes an entire worker pool. That is the same poison-message problem you have handled with a Celery DLQ; the difference here is that the payload is expensive to process and the failure is often partial.

**Then the part that matters: quarantine is a work queue with an owner, not a graveyard.** A weekly review that groups quarantined documents by `error_class` and sorts by count. The distribution is always long-tailed and the top three classes are usually 70% of the volume — password-protected PDFs, a specific vendor's malformed export, files over a size limit. Each is a small, tractable fix. Working the top three every week takes a 3% failure rate to under 0.5% in a month.

**And the part that makes it visible to users:** the corpus should know what it does not have. **If a document is in the source system but not in the index, that fact is queryable.** A user asking about "the Q3 vendor agreement" should be answerable with "that document exists but failed to process — here's the link and it's been flagged," not with silence. That requires the `documents` table to hold a row for every discovered document regardless of processing outcome, and the answer path to check it. Most teams never build this and it is a genuine differentiator in a design discussion.

**🔍 Failure taxonomy — the error classes worth having as explicit enum values**, because a generic "parse error" bucket teaches you nothing:
- `ENCRYPTED` / `PASSWORD_PROTECTED` — needs credentials or a policy decision.
- `CORRUPT_CONTAINER` — truncated download vs genuinely bad file; distinguishable by comparing size to the source's reported size.
- `UNSUPPORTED_FORMAT` — count them; if CAD files are 4% of your corpus that is a roadmap item.
- `TOO_LARGE` — a 2,000-page document that blows the per-task memory limit. Fix by page-range sharding, not by raising the limit.
- `EMPTY_EXTRACTION` — parsed fine, produced under N characters. **This is the dangerous one because it looks like success.** Treat sub-threshold extraction as a failure, not a result.
- `PROVIDER_REFUSAL` — a VLM declined the page. Needs a different route, not a retry.
- `TIMEOUT` / `RATE_LIMITED` — retryable, and if they dominate you have a capacity problem, not a document problem.

**⚠ Trap:** counting a document as successfully ingested when it produced zero chunks. Every pipeline I have inherited had this bug. `EMPTY_EXTRACTION` documents pass all the success checks — no exception was raised, the stage completed, the row was written — and contribute nothing to the index. **Assert chunk count > 0 per document and alert on the rate.**

### Give me the alert. 5% of documents are silently failing to parse and nobody notices until a customer asks. What monitoring catches that?

The reason this goes unnoticed is that the standard metrics all look fine: task success rate is 100% (nothing threw), throughput is normal, latency is normal. **You are monitoring the pipeline's health, not the corpus's health, and those are different systems.** So the answer is a set of corpus-level invariants, checked continuously, each with a defined action.

**Tier 1 — per-document invariants, asserted at write time.** These turn silent failures into loud ones.
- `chunk_count > 0`.
- `extracted_chars / (pages × expected_chars_per_page) > 0.3` — a document producing 200 characters from 40 pages is a failure regardless of what the parser returned.
- `chunk_count` is within a factor of ~3 of `estimated_tokens / chunk_size`. Catches both truncation and runaway repetition.
- Figure/table reference balance: `count("Figure \d+" in text)` versus `count(figure elements)`.

**Tier 2 — per-batch distributional alerts.** This is the layer that actually catches the 5%, because a 5% shift never trips a per-document threshold but always moves a distribution.
- **Chunks per page, as a distribution, versus a 7-day baseline.** Alert on a KS-test-style shift or simply on the median moving more than 20%. A parser regression, a format change upstream, or a new document family entering the corpus all show up here first, and they show up the day it happens.
- **Median extracted characters per page**, same treatment.
- **Empty-extraction rate**, absolute threshold and trend.
- **Fraction of pages routed to each parser path.** If the scanned share jumps from 8% to 30% overnight, something upstream changed — an export setting, a new source system — and you want to know before the bill does.
- **Mean perplexity of sampled extracted text under a small LM** — the single best general-purpose garbage detector, catching encoding corruption, column interleaving, and OCR failure with one number.
- **Language distribution.** A sudden spike in "unknown" language is mojibake.

**Tier 3 — coverage reconciliation, run daily.** For each source system: count upstream documents, count documents in the index, and alert on the delta and on its rate of change. **This is the check that directly answers "is anything missing," and it is the one almost nobody builds.** It is a SQL query and a scheduled job.

**Tier 4 — the end-to-end canary.** A fixed set of 50 golden queries whose correct answers depend on 50 specific documents spread across formats and source systems, run every hour against production. If document 37 stops being retrievable, you find out in an hour rather than from a customer. **This catches the entire class of failure that no per-stage metric can see**, including alias-swap mistakes, permission-filter regressions, and index corruption.

**🗣 Say this in the room:** "The reason it's silent is that pipeline health and corpus health are different systems, and everyone instruments the first. I'd add per-document invariants — zero chunks is a failure, characters-per-page below a floor is a failure — then distributional alerts on chunks-per-page and extracted-characters-per-page against a rolling baseline, because a 5% regression never trips a per-document threshold but always moves the median. Then a daily reconciliation of upstream document count against indexed count, and an hourly canary of 50 golden queries. The reconciliation is the one that literally answers 'what's missing' and it's about forty lines of SQL."

**💰 Math on why it is worth building:** a 5% silent parse failure on a 20M-document corpus is 1M documents missing. If your product's value proposition is "ask anything about our documents," a 5% hole produces a wrong-looking "I don't know" on roughly 5% of queries — at 100k queries/day, 5,000 bad experiences a day. The monitoring above is maybe three engineer-days. There is no version of this that does not pay for itself in the first week.

### The embedding provider rate-limits you and the vector store starts rejecting writes. Design the flow control.

This is the part of the section where your backend instincts transfer almost intact, so the interesting answer is about where they *don't*.

**What transfers directly:** bounded queues between stages so backpressure propagates rather than buffering unboundedly; a token-bucket limiter shared across workers (Redis-backed, since the provider's limit is global, not per-process); exponential backoff with full jitter on 429s; a circuit breaker that stops hammering a degraded provider; a concurrency semaphore per downstream. If you built a rate-limited API gateway you have built 80% of this.

**What is different, and this is what I would emphasize:**

**1. The limit is on tokens, not requests.** Provider quotas are typically expressed as both requests-per-minute and *tokens*-per-minute, and the token limit binds first for embedding workloads. So your limiter's cost function is `len(tokens)`, computed before the call, not `1`. A naive request-counting limiter will sail under the RPM limit and get 429'd on TPM all day. **Concretely: you must tokenize before you throttle**, which means the tokenizer runs in the producer, not the client.

```python
class TokenBucket:                        # Redis-backed in production
    def __init__(self, tokens_per_min, burst):
        self.rate = tokens_per_min / 60.0
        self.capacity, self.tokens = burst, burst
        self.ts = time.monotonic()
    async def acquire(self, n):
        while True:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
            self.ts = now
            if self.tokens >= n:
                self.tokens -= n
                return
            await asyncio.sleep((n - self.tokens) / self.rate)
```

**2. Read the response headers and adapt, rather than guessing.** Providers return remaining-quota and reset headers. A limiter that consumes those and adjusts its rate downward on the fly is dramatically better than a statically-configured one, especially with multiple pipelines sharing an account — which is the common case, and the reason your ingestion backfill will happily starve your production query path.

**3. Priority classes are mandatory, and this is the failure I would call out first.** Backfill traffic and live-ingestion traffic and *user-facing query embedding* usually share one provider quota. A 20M-document backfill will consume the entire TPM budget and your product's search will start 429ing. **Separate the quotas — different API keys or projects if the provider supports it — or implement strict priority with the backfill yielding.** Do not rely on "the backfill is low volume"; it is never low volume.

**4. Use the batch/offline tier for backfill.** Roughly half price and, more importantly, a separate quota pool on most providers. Ingestion backfill is the canonical batch workload: latency-insensitive, enormous, and re-runnable.

**5. Backpressure from the vector store is a different shape.** Vector writes are not just I/O; HNSW insertion does graph work proportional to `efConstruction` and is CPU-bound, and bulk insert can trigger index maintenance that degrades query latency on the same node. So the correct response to write rejection is often not "retry slower" but **"stop writing to the serving index at all."** Build the new index offline (or on a separate node), then alias-swap. **Never backfill into a live serving index while it is taking query traffic** — you will degrade p99 for users to serve a batch job, which is exactly the trade you would never make with a database and should not make here either.

**⚠ Trap:** unbounded retries plus a queue with no depth limit produces a retry storm that looks like a provider outage. Cap attempts, cap queue depth, and make the producer block when the queue is full — that is the whole point of a bounded queue and it is routinely defeated by an `asyncio.Queue(maxsize=0)` someone typed without thinking.

### How do you batch embedding calls? Show me the throughput and cost math.

Batching matters more than people expect because the per-request overhead is a large fraction of a small embedding call, and because the provider's limits are usually more generous per-request than per-item.

**The mechanics.** Embedding APIs accept an array of inputs per request — commonly up to ~2,048 items with a per-item token cap around 8,192 **📅 Volatile: array and token limits differ by provider and change; check the current docs before you tune to a number.** You are optimizing under two constraints simultaneously: `len(batch) <= max_items` and `sum(tokens) <= max_tokens_per_request`. So the batcher is a bin-packing loop, not a fixed-size chunker:

```python
async def embed_all(chunks, embed, max_items=256, max_tokens=100_000, conc=8):
    sem = asyncio.Semaphore(conc)
    batches, cur, cur_tok = [], [], 0
    for c in chunks:
        if cur and (len(cur) >= max_items or cur_tok + c.n_tokens > max_tokens):
            batches.append(cur); cur, cur_tok = [], 0
        cur.append(c); cur_tok += c.n_tokens
    if cur:
        batches.append(cur)

    async def run(b):
        async with sem:
            await limiter.acquire(sum(c.n_tokens for c in b))
            return await embed([c.text for c in b])          # retry/backoff inside
    return await asyncio.gather(*(run(b) for b in batches))
```

**Sort by token length before batching.** Grouping similar-length chunks together reduces padding waste on self-hosted encoders substantially — a batch of one 8,000-token chunk and 255 fifty-token chunks pads everything to 8,000 and wastes ~97% of the compute. On a hosted API padding is not your problem, but batch *variance* still is, because a single long item can push you over the per-request token cap and force a retry.

**💰 Throughput math for a 20M-document, 240M-chunk backfill.** At 500 tokens/chunk that is 120B tokens.
- Hosted, at an assumed sustained 5M tokens/minute of provisioned throughput: 120,000M / 5M = **24,000 minutes = 16.7 days.** That number is the reason you negotiate a rate-limit increase *before* the backfill, not during it. At 50M tokens/minute it is **1.7 days.**
- Cost at $0.02/Mtok: 120,000 × $0.02 = **$2,400.** At $0.13/Mtok: **$15,600.**
- Self-hosted alternative: a small encoder on one modern GPU does on the order of 2,000–10,000 short chunks/second depending on model size and sequence length. At 4,000 chunks/s, 240M chunks is 60,000 seconds ≈ **16.7 GPU-hours ≈ $25 at $1.50/hr**, plus engineering time to run it. **Two orders of magnitude cheaper than the hosted premium encoder**, which is why every team doing nine-figure-chunk backfills self-hosts the embedder and why "we call an embedding API" is a fine answer at 10M chunks and a bad one at 1B.

**⚠ Trap:** concurrency without a limiter. Firing 200 concurrent batch requests gets you rate-limited, backed off, and — because everyone retries at the same moment — synchronized into a thundering herd. Full jitter on the backoff, not equal jitter, not fixed.

**⚠ Trap two:** losing the mapping between input order and output order. Every embedding API returns results in input order with an index field; if you `gather` batches and flatten without tracking which chunk was which, you will attach vectors to the wrong chunks. **This produces a system that works — retrieval returns results, nothing errors — and is silently, totally wrong.** Assert `len(vectors) == len(batch)` and carry the chunk id through, never positional trust across an async boundary.

### Price out the full ingestion of 5 million PDF pages. I want the number and how you'd cut it.

Assumptions stated up front, because an unstated assumption is how these estimates become fiction: 5M pages, average 500 tokens of text per page, 8% scanned or structurally hard, 0.3 figures per page worth describing, ~10 chunks per page at 500 tokens... no — at 500 tokens/page and 500-token chunks that is 1 chunk/page, so **5M chunks, 2.5B tokens.**

**Line items:**

| Stage | Arithmetic | Cost |
|---|---|---|
| Storage (raw PDFs, ~1MB avg) | 5TB × $0.023/GB/mo | **$115/mo** |
| Text extraction (92% digital) | 4.6M pages, CPU-only, ~50 pages/s/core, 25,600 core-hours… at 32 cores/instance = 800 instance-hours × $1.20 | **$960** |
| VLM parse (8% hard) | 400k × $0.0053 | **$2,120** |
| OCR cross-check on VLM pages | 400k × $0.0000417 | **$17** |
| Figure descriptions | 1.5M figures × $0.00195 | **$2,925** |
| Contextual retrieval (optimized) | 5M chunks × $0.0045 | **$22,500** |
| Embedding (hosted, $0.13/Mtok) | 2.5B + contextual additions ≈ 3.0B tok × $0.13/M | **$390** |
| Vector storage | 5M × 1024d × 4B = 20.5GB, managed | **~$200–500/mo** |
| **One-time total** | | **≈ $28,900** |
| **Recurring** | | **≈ $400/mo** |

**The immediate observation: contextual retrieval is 78% of the bill.** That is the item to interrogate, not the embedding cost that people instinctively optimize. So the cuts, in order of return:

1. **Gate contextual retrieval** on the dangling-reference detector — skip the ~50% of chunks that are already self-contained. **−$11,250.**
2. **Use the batch tier** for the remaining contextual calls (~50% off). **−$5,600.**
3. **Drop to a cheaper model** for context generation after a 200-chunk quality comparison. Potentially another 5–10× on that line.
4. **Sample the figure descriptions**: classify figures as informational vs decorative with a cheap classifier and describe only the informational ones. If 40% are decorative, **−$1,170.**
5. **Self-host embeddings**: $390 → ~$20 of GPU. Barely worth doing at this scale; decisive at 100× this scale.

**Post-optimization: roughly $8,000–10,000 one-time.** And the sentence that matters: **the optimization did not touch the embedding model or the vector database, which is where every unprepared candidate starts.** The cost is in the LLM calls per chunk, and the fix is to make fewer of them rather than cheaper ones.

**💰 Also price the thing you will actually be asked about: the marginal cost of re-running.** Because parse output is content-addressed and cached, re-running with a new chunker costs only stages 3–5: contextual retrieval (if the chunks changed, which they did) + embedding = ~$10k + $390. **So a chunking experiment on the full corpus is $10k and a chunking experiment on a 1% sample is $100.** That is the number that determines whether your team iterates or guesses, and it is the argument for always running chunking experiments on a stratified sample first.

**⚠ Trap:** quoting only the one-time backfill in a design review. The number leadership needs is the **run rate**: if you ingest 50k new pages/day, that is 1% of the corpus per day, so ~$100/day, **$3,000/month, forever**, plus re-processing on parser upgrades. The one-time number is the smaller half of the decision.

### What do you put on the ingestion dashboard? What's the one metric you'd page on?

I would put four panels up, in this order, because they answer four different questions and mixing them is why most pipeline dashboards are unreadable.

**Panel 1 — Is the pipeline moving?** Documents and pages per hour by stage; queue depth by stage; oldest-message age by queue. **Oldest-message age is the one I actually watch**, because throughput can look healthy while a specific class of document sits in a queue forever. This is straightforwardly your Kafka consumer-lag instinct.

**Panel 2 — Is the corpus healthy?** Chunks per page (median and p10/p90) against a 7-day baseline; extracted characters per page; empty-extraction rate; quarantine rate by error class; parser-route distribution (digital/OCR/VLM shares); sampled-text perplexity. **This is the panel nobody builds and it is the one that catches the silent 5%.**

**Panel 3 — Is the corpus complete?** Upstream document count vs indexed count per source system, and the delta's trend. Time-since-last-successful-sync per source. A source system that silently stopped syncing three weeks ago is an extremely common and extremely embarrassing incident, and this panel is its only detector.

**Panel 4 — What is it costing?** Cumulative spend by stage today and this month; cost per document; tokens consumed against the provider quota. **Cost belongs on the ops dashboard, not in a monthly finance review**, because a runaway retry loop on a VLM parse stage is a $10k-an-hour bug and it should page like an outage.

**The one metric I would page on: staleness of the freshest document per source system.** Formally, `max over sources of (now − most_recent_successfully_indexed_doc_updated_at)`, against a per-source SLO. It is a single number, it composes every upstream failure mode into one signal — crawler down, credentials expired, queue stalled, workers crashed, a poison document blocking a partition — and it maps directly to the user-visible symptom, which is "the assistant doesn't know about last week's policy update." **Throughput can be perfect while this is broken; this cannot be broken while the product is working.**

**Secondary pages:** empty-extraction rate above 2×baseline, quarantine rate above 2×baseline, cost-per-hour above 3×baseline, and the hourly golden-query canary failing.

**🗣 Say this in the room:** "I page on freshness per source — the age of the newest successfully indexed document — because it's one number that composes every upstream failure into the user-visible symptom, and because throughput looking healthy is exactly the state a stalled source hides in. Everything else is a dashboard, not a page: chunks-per-page against baseline, empty-extraction rate, upstream-versus-indexed reconciliation, and spend by stage."

### Now do it for code. How do you build a retrieval index over a 5-million-line monorepo?

Code retrieval is different enough from document retrieval that treating it as "documents that happen to be code" is the primary failure. Three differences drive the design.

**Difference 1: half the queries are not searches.** "Where is `PaymentProcessor` defined?" "Who calls `retry_with_backoff`?" "What implements this interface?" These are graph and index lookups with exact answers, and cosine similarity is the wrong data structure for them. So the system is **three indexes, not one**:

- **A symbol index** — `(symbol_name, kind, file, line, signature, docstring, language, visibility)` — built from a tree-sitter parse of every file. Exact and prefix lookup. This is a Postgres table with a trigram index and it answers "where is X" in a millisecond.
- **A reference/call graph** — edges `(caller_symbol, callee_symbol, file, line)`, plus imports and type references. Answers "who calls X," "what does X depend on," and gives you **in-degree as an authority signal** for ranking.
- **A vector + BM25 index over chunks** for the genuinely semantic queries: "how do we handle idempotency on webhooks."

**Difference 2: the chunk is a declaration, and it needs a synthesized header.** As covered earlier: one chunk per function/class/method, packed if tiny, split at statement boundaries if huge, with a header carrying file path, language, enclosing class, referenced imports, and the enclosing signature. Add the **docstring and the leading comment block** — those are the natural-language surface that semantic queries actually match, and a codebase with good docstrings retrieves dramatically better than one without, which is worth saying because it reframes documentation as a retrieval investment.

**Difference 3: exact literals dominate.** Function names, error strings, config keys, environment variable names. **BM25 does more work than the dense channel in code retrieval than in almost any other domain**, and a code search system without a strong lexical channel — ideally with a code-aware analyzer that splits `snake_case` and `camelCase` into constituent tokens while also keeping the whole identifier — is broken for its most common query. Splitting `getUserById` into `get`, `user`, `by`, `id` *and* keeping `getUserById` is the specific analyzer behavior you want, and it is the same trick as an n-gram index for substring search.

**What to exclude, which is most of the repo.** Lockfiles, `node_modules`/vendored dependencies, generated code (protobuf stubs, ORM migrations, minified assets), test fixtures, and binary blobs. **In a typical monorepo this is 60–80% of the bytes and near 0% of the retrieval value**, and indexing it actively hurts: generated code is highly repetitive, so it floods your top-k with near-identical results. Respect `.gitignore` plus an explicit denylist plus a generated-file heuristic (`@generated` markers, very long lines, extreme line-length uniformity).

**Enrichments that pay:** git metadata per chunk (last modified, author, change frequency — recency and churn are real relevance signals), test-to-implementation links, and the **file's position in the dependency graph** as a static prior.

**⚠ Trap:** indexing every branch or every commit. Index the default branch's current state; keep history in git where it already is. A team that indexed all branches turned a 5M-line repo into a 400M-line index of near-duplicates and got worse results than grep.

### A commit lands every 90 seconds. How do you keep the code index fresh?

The requirement is a freshness SLO — say, the index reflects `main` within 60 seconds of a merge — and the design falls out of one observation: **a commit touches a handful of files, but its blast radius in the index is larger than the files it touched, and getting that blast radius right is the whole problem.**

**The pipeline:**

1. **Trigger on the post-merge webhook**, not on a poll. Payload gives you the commit SHA and the changed paths.
2. **Compute the changed set** with `git diff --name-status <prev_indexed_sha>..<new_sha>`. Handle rename detection (`-M`) explicitly — a renamed file is an update, not a delete-plus-add, and treating it as the latter churns embeddings for unchanged content.
3. **Re-parse only changed files.** Tree-sitter is incremental and fast; 30 files is milliseconds.
4. **Diff at the symbol level, not the file level.** This is the key optimization. A one-line change to one function in a 40-function file should re-embed **one chunk**, not forty. Compare the new symbol table against the stored one by `(symbol_path, body_hash)`; only symbols whose body hash changed get re-chunked and re-embedded. **On a typical commit this reduces embedding work by 20–50×**, which is what makes 90-second cadence affordable.
5. **Update the graph edges** for changed files: delete all outbound edges from changed symbols, re-insert. Inbound edges to a deleted symbol become dangling — reconcile them (mark unresolved) rather than leaving stale pointers.
6. **The blast radius beyond the diff:** if a chunk's header includes the enclosing class signature and that signature changed, every chunk in the class is stale even though its own body did not change. Similarly if you inject imports into headers and the import block changed. **Track this dependency explicitly** — a `header_hash` per chunk that is a function of the file's structural context — and re-embed chunks whose header hash changed. Forgetting this leaves chunks whose synthesized context contradicts the current code, which is worse than no context.
7. **Upsert with deterministic chunk IDs** so the write is idempotent and a replayed webhook is a no-op.
8. **Tombstone deleted symbols immediately.** A deleted function still being returned by code search is the same severity bug as a deleted document still being quoted.

**💰 Math:** at a commit every 90 seconds, ~960 commits/day. If a typical commit changes 8 files and 15 symbols, that is 14,400 chunk re-embeddings/day at ~400 tokens each = 5.8M tokens/day. At $0.02/M = **$0.12/day**. **Freshness is essentially free** once you diff at the symbol level; the naive file-level version at 40 chunks per changed file would be 307k chunks/day, 123M tokens, $2.46/day — still cheap in dollars, but 21× the API calls, 21× the vector-index write pressure, and enough index churn to degrade HNSW quality over time. **The argument for symbol-level diffing is write amplification and index health, not the token bill**, and that is the more sophisticated framing.

**⚠ Trap:** a force-push or a branch reset makes `prev_indexed_sha` unreachable, `git diff` fails or produces a nonsense diff, and the incremental path silently no-ops or corrupts. **Store the indexed SHA, verify it is an ancestor of the new one, and fall back to a full reindex when it is not.** Also handle the merge-commit case where a squash produces a diff spanning weeks of work.

**⚠ Trap two:** ordering. Two commits land 10 seconds apart and their update tasks race; the older one's write lands last and you have indexed a stale version of a file. **Partition the work queue by file path** so all updates to one file are serialized, and stamp every write with the commit timestamp so a stale write can be rejected. This is ordinary concurrency hygiene, but I have seen it missed in three separate code-search systems.

### How do you test an ingestion pipeline? What's in CI so a parser change can't silently regress quality?

This is the question that distinguishes people who have operated one of these from people who have built one, and my answer is a four-layer test pyramid where the top layer is the one that matters and the one everybody skips.

**Layer 1 — Unit tests on the chunker, with fixture documents.** Given this markdown, expect these boundaries. Given a table larger than the budget, expect the header repeated on every fragment. Given `overlap >= size`, expect a raised exception. Given an atom larger than the budget, expect a hard split and an incremented counter. These are fast, deterministic, and catch the boring 60% of regressions.

**Layer 2 — Golden-output snapshot tests on parsing.** A committed set of ~40 documents spanning every format, every source system, and every known pathology (two-column, merged cells, scanned, mixed-language, encrypted-then-decrypted, 500-page, rotated). For each, a committed expected element stream. **Diff on change and require human review of the diff.** These are not assertions that the output is *correct*; they are assertions that it did not change *unnoticed*. A parser bump produces a reviewable diff, which is exactly the artifact you need for the "do we reprocess" decision.

**Layer 3 — Extraction assertions on real values.** For each fixture, assert specific facts: this table's cell (row 4, col 3) equals `4,182`; this document extracts exactly 12 figures; this scanned page's OCR contains the string `Invoice #INV-2024-8871`; this contract's defined-terms list has 23 entries. **These are the tests that catch the class of bug that produces wrong numbers**, and they are the ones people do not write because writing them requires reading the documents. Twenty of them is a day of work and it is the highest-value day in the project.

**Layer 4 — The retrieval eval gate, run on every change to parser, chunker, or enrichment config.** The 200-query golden set from earlier, run end to end against a freshly-built index over a fixed 2,000-document sample corpus. **Report Recall@10, answer correctness, and groundedness, with a paired bootstrap confidence interval against the previous run's per-query scores.** Fail the build on a statistically significant regression; warn on a directional one. This costs a few dollars and ten minutes per run, which is entirely affordable on a change that only lands a few times a month.

**The invariant checks that also belong in CI**, because they are cheap and catch whole families:
- No 5-gram appears in more than 20% of chunks (boilerplate).
- Fewer than 10% of chunks open with a dangling reference.
- Chunk length distribution: p1 above the minimum, p99 below the maximum, no spike at the hard-split boundary.
- Zero chunks with zero tokens; zero documents with zero chunks.
- Re-running ingestion on unchanged fixtures produces zero embedding calls (the idempotency assertion).

**⚠ Trap:** testing only with clean fixtures. Your fixture set must be sampled from production pathologies, not authored. **The rule I enforce: every production ingestion incident adds its triggering document to the fixture set, in the same PR as the fix.** That is how the pyramid grows to cover reality rather than imagination, and it is the same discipline as adding a regression test for every bug — you already do this, and it transfers unchanged.

**🗣 Say this in the room:** "Four layers: unit tests on the chunker, snapshot diffs on parser output so a version bump produces a reviewable diff instead of a silent change, hard assertions on specific extracted values including table cells, and a retrieval eval gate with a paired bootstrap so I can tell a real regression from noise. Plus corpus invariants as assertions — no 5-gram in over 20% of chunks, no document with zero chunks, re-ingestion produces zero embedding calls. And every production incident adds its document to the fixture set in the same PR as the fix."

### Last one — a timed drill. What would you build in three hours to prove you can do this?

Here is the exercise I would set, and it is very close to the real take-homes that AI-product companies send.

**🏋 Drill (3 hours, unaided, no autocomplete):** you are given 500 mixed documents — PDFs (some scanned), a few dozen docx, a handful of xlsx, and an HTML export. Build an ingestion pipeline and a retrieval eval, and produce a one-page recommendation on chunk size.

**What you must deliver:**
1. A pipeline with **separate, independently re-runnable** parse / chunk / embed stages, with the parse output cached by `(content_hash, parser_version)`.
2. Per-page routing for PDFs: text-layer detection, with scanned pages going down a different path (stub the OCR call if you must — the routing logic is what is being graded, not the OCR).
3. Structure-aware chunking for the formats that have structure, with section-path prefixing, and tables never split.
4. A chunk record carrying doc id, section path, page, character span into a stored canonical text, and parser/chunker version stamps.
5. A quarantine table and a `documents` table that has a row for every discovered document, whatever its outcome.
6. A 50-query golden set with **span-level labels** (character offsets into the canonical text, not chunk ids).
7. A sweep over three chunk sizes reporting **Recall@10 and end-to-end correctness separately**, with the number of queries where they disagree.
8. The corpus invariant checks: chunks per document > 0, no 5-gram in more than 20% of chunks, empty-extraction rate.
9. A README with the cost arithmetic for scaling this corpus to 5 million documents.

**Pass criteria — and these are what I would grade in the defense, in this order:**
- **You can state which chunk size you would ship and name the failure mode your data ruled out.** Not "512 scored highest" — "512 won end-to-end while 256 won Recall@10, which told me my failures were context sufficiency rather than retrieval, so I'd ship 256 with parent-document expansion and I'd have tested that if I had another hour."
- **Your parse stage is cached and your chunk stage is not fused into it.** If someone asks "how long to try a different chunk size on the whole corpus," the answer is "about four minutes" and not "about a day."
- **You have a number for what 5M documents costs**, with the arithmetic shown, and you know which line item dominates.
- **You can name three things that would break at 5M that do not break at 500** — and the answers I would accept are: the parse cost dominates and needs per-page routing; the vector index no longer fits in memory at float32 and needs quantization or a disk-resident index; the long tail of parse failures goes from 4 documents you can eyeball to 100,000 you cannot, so quarantine-by-error-class and distributional alerting stop being nice-to-haves.
- **Your quarantine is non-empty and you can describe what is in it.** A pipeline over 500 real mixed documents that reports 100% success is a pipeline that is swallowing failures, and I will go looking for the swallow. Being able to say "eleven documents failed: seven password-protected, three that are actually images with a PDF wrapper, one 900MB scan that OOM'd, and here's what I'd do about each" is a stronger signal than a clean run.

**🗣 Say this in the room, if you are asked to summarize the whole area in one breath:** "Ingestion is where RAG quality is actually determined, and it's mostly an engineering problem rather than an ML one. The four decisions that matter are: route per page rather than per document so you only pay for the expensive parser where it earns its keep; keep parse output content-addressed and cached so chunking stays a four-minute experiment instead of a six-figure one; make the retrieval unit and the context unit different sizes; and monitor the corpus, not just the pipeline, because the failure that kills you is the one where every task succeeded and five percent of your documents aren't in the index."
