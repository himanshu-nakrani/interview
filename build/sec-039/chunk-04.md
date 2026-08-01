### Design the whole production RAG system end to end. I want the boxes, the failure handling, and where the evals live.

I will build it as three planes — ingestion, serving, and evaluation — and the thing I want you to notice is that the evaluation plane is not bolted on at the end; it consumes artifacts the other two planes are designed to emit.

**Ingestion plane.** A connector per source (Confluence, S3, Drive, a Postgres CDC stream) that emits a normalized `SourceObject` with stable coordinates and raw bytes. `doc_id = sha256(source:space:object_id)` — derived from coordinates, never content, so re-ingestion is idempotent and update and delete are expressible. A parse worker writes canonical extracted text plus a `content_version` hash over canonicalized text; failures and content-plausibility assertion trips (chars/page below floor, zero chunks, alphanumeric ratio collapse) go to a **quarantine table with the raw bytes**, and quarantine depth is a dashboard with an alert on rate-of-change, not a log line. A chunker writes immutable chunks keyed `(doc_id, content_version, chunk_index)` carrying `start_char`/`end_char` into canonical text plus display coordinates. Then an **embedding queue** — this is a Celery/SQS-shaped problem you already know: batch 64–256 chunks per request, bounded concurrency to respect provider rate limits, exponential backoff with jitter, and a DLQ. Backpressure matters: the vector store's write throughput, not the embedding API, is usually the bottleneck on a bulk reindex.

Deletion is a first-class path: objects present in the previous sync manifest and absent from the current one get tombstoned, and tombstones drive vector deletion. If you skip this you have built a system that cannot forget, which is both a quality bug and, under GDPR-style deletion requests, a legal one.

**Serving plane.** A query service with these hops, each a traced span: intent/route classification (including the "this is not a retrieval question" branch) → conversational query rewrite against session history → parallel dense + BM25 retrieval with the **ACL predicate applied as a pre-filter resolved at query time** → RRF fusion → cross-encoder rerank 50→8 → near-duplicate suppression → context assembly with stable ordering for prefix-cache stability and explicit `[C1]`-style markers → a retrieval-quality gate that can abstain before generation → streaming generation into a buffer → post-hoc citation verification and NLI groundedness → render.

Two production details that matter. **Order the context for cache stability**: put the stable system prompt and any fixed corpus material first so the provider prefix cache hits, and the volatile retrieved chunks last. And **stream into a buffer for high-stakes surfaces** so you can suppress on a groundedness failure — you cannot un-say a token you already streamed.

**Deploy.** The index is built into a new named collection and promoted by **alias swap**, gated on document-count parity and a golden-set recall run against the new index. Cache namespace version bumps with the alias. Rollback is an alias swap back, which is why the previous index is retained for at least one cycle.

**Evaluation plane.** A golden set in version control with span-level (not chunk-ID) labels. A CI harness that runs retrieval metrics and structural assertions on every PR. A nightly end-to-end run with judges, posted as a trend with confidence intervals. A **feedback loop**: every thumbs-down, every edit-in-place, every escalation-to-human writes a candidate row joined on `request_id` to the full trace, and a weekly triage promotes real failures into the golden set. This is the flywheel; without it your eval set ossifies around last year's failure modes.

**Observability.** Per-hop tracing as described earlier, plus four dashboards: retrieval-score distributions, empty-result rate, chunk-usage skew, and groundedness/citation-verification rates.

**🗣 Say this in the room:** "Three planes. Ingestion is idempotent on coordinate-derived IDs with a real delete path and a quarantine queue. Serving is hybrid-plus-rerank behind a pre-filtered ACL with a retrieval-quality gate before generation and post-hoc citation verification after. Evaluation isn't a separate project — CI gates on deterministic retrieval metrics, nightly runs judged metrics as a trend, and production feedback flows back into the golden set. The alias swap is gated on count parity plus a recall run, because an incomplete index passes every smoke test."

### Make ingestion idempotent and re-ingestion safe. What are the exact transactional boundaries?

The hard requirement: **a re-ingest of an unchanged document must be a no-op, a re-ingest of a changed document must be atomic, and a concurrent re-ingest of the same document must not interleave.** Those are three separate mechanisms.

**No-op on unchanged.** Compare `content_version` against the stored value. Equal → update `last_seen_at` and return. This is the fast path and it is 95%+ of a nightly sync; if your pipeline re-embeds everything on every run, your embedding bill is 20× what it should be and your index churns for nothing.

**Atomicity on changed.** The dangerous window is: old chunks deleted, new chunks not yet written and embedded. During that window queries return nothing for the document. Two options. *Preferred*: write new chunks with the new `content_version`, embed them, wait for the vector store to acknowledge, and only then delete the old `content_version`'s vectors — a small overlap where both versions exist is far better than a gap where neither does, and you filter to the current version at query time with a metadata predicate. *Fallback*, if the store supports it: a transactional upsert with a version predicate. The rule I enforce: **overlap, never gap.** A user seeing a slightly-stale answer for 90 seconds is a non-event; a user seeing "I don't have information about that" for a document that exists is a support ticket and a trust loss.

**Concurrency.** Two workers picking up the same document — from a webhook and a nightly sweep firing within the same minute — will both re-chunk and both write. Take an advisory lock on `doc_id` (a Postgres `pg_advisory_xact_lock(hashtext(doc_id))` is exactly the right tool and costs nothing) or use a per-document serialized queue partition. And make the chunk write **conditional on version**: `WHERE current_version = <the version I read>`, so a losing writer detects the conflict rather than clobbering.

The state machine per document, which is what I would actually draw:

```
discovered -> fetched -> parsed -> chunked -> embedded -> indexed -> live
                  \-> quarantined (parse failure or plausibility assertion)
live -> (source changed) -> fetched ...            [new content_version]
live -> (absent from manifest) -> tombstoned -> vectors_deleted -> purged
```

Every transition is a row update with a timestamp, which gives you a free ops query: documents stuck in a non-terminal state for more than N hours. That single alert catches embedding-queue stalls, rate-limit lockouts, and vector-store write failures, and it is the highest-value alert in the whole ingestion plane.

**⚠ Trap:** treating "the embedding API call succeeded" as "the chunk is queryable." There are two more asynchronous steps — the vector store write and, in many stores, an index build or merge. A chunk can be written and not yet searchable for seconds to minutes. If your ingest pipeline reports success on the API call, your "document is live" signal is a lie, and your post-ingest verification test will flake. Verify by **querying for the chunk you just wrote** before marking live.

### Walk me through the alias-swap deploy for a vector index, step by step.

This is a blue/green with one extra property that is easy to miss: the vector index is *derived data*, so unlike a database migration you can always rebuild it, but unlike a stateless deploy the build takes hours and costs money. That shapes the sequence.

**Step 1 — build into a new named collection.** `chunks_v7`, never mutating `chunks_v6`. Write the build metadata into the collection: embedding model ID and dimension, chunker version, source manifest hash, document and chunk counts, build start/end timestamps, and a git SHA of the ingestion code.

**Step 2 — insert canaries.** A handful of synthetic documents with known content and known-unique tokens. They are how you answer "which build am I actually querying" in one query at 3 a.m., and they cost nothing.

**Step 3 — count parity gate.** Compare `doc_count` and `chunk_count` against the live collection. My rule: a new index more than 2% below the live one does not promote without an explicit human override with a written reason. This is the gate that catches the partial-reindex failure, and it is the single highest-value check in the sequence because a partial index passes every quality smoke test.

**Step 4 — golden-set recall against the new collection, before any traffic.** Point the eval harness at `chunks_v7` directly. Recall@50, recall@5, nDCG@10, versus the recorded numbers for `chunks_v6`. A drop beyond the noise band blocks. Note that this requires span-level golden labels, because if the chunker changed, chunk IDs did not survive.

**Step 5 — shadow traffic.** Mirror a slice of live queries against the new collection without serving the results, and compare retrieved chunk-ID overlap and score distributions against production. A large Jaccard divergence with unchanged golden metrics means your golden set does not cover the real query distribution — useful information either way. This step is optional for small corpora and mandatory for anything user-facing at scale.

**Step 6 — swap the alias, atomically**, and in the same operation **bump the cache namespace version** so every response cache, semantic cache and retrieval cache is logically invalidated. If your store's alias update is not atomic, you route through a config value in your own service and change that instead — the requirement is that no request sees a torn state.

**Step 7 — hold the old collection for at least one full cycle.** Rollback is an alias swap back, which takes seconds. Deleting `v6` the moment `v7` goes live converts a 10-second rollback into a 4-hour rebuild. The storage cost of holding one extra generation is the cheapest insurance in the system.

**Step 8 — post-swap watch.** For 30 minutes: empty-result rate, mean top-1 score, p95 retrieval latency, abstention rate. A jump in empty-result rate or a drop in mean top score is the signature of a query/index embedding-model mismatch and should auto-rollback.

**💰 Math:** holding a second generation of a 20M-chunk, 768-dim index is 20e6 × 768 × 4 bytes = **61 GB** of extra vector storage, plus index overhead — call it 90 GB. On managed vector services in the rough range of $0.30–$1.00 per GB-month for storage-tier data that is $27–$90/month; on a memory-resident tier it is materially more, which is the honest counterargument for very large indexes and the reason some teams keep the old index on a cheaper storage tier rather than hot. 📅 Volatile — price against your vendor. Against a 4-hour rebuild plus an outage, it is still obviously correct.

### What do you monitor in production RAG? Give me the dashboards, not a list of metrics.

Four dashboards, each answering one question, each with a defined alert.

**Dashboard 1 — Retrieval health.** Histogram of top-1 relevance score and of the rank-1-minus-rank-10 gap, both over a rolling window, **overlaid with the same histograms from a baseline week.** The overlay is the point: absolute scores are meaningless, distribution *shifts* are extremely meaningful. A leftward shift of the top-1 distribution means either the query distribution changed (new user cohort, new product surface, a marketing campaign bringing different questions) or the index degraded (embedding model swap, ingest regression, duplicate flood). Also: p50/p95/p99 retrieval latency split by hop (embed, ANN, BM25, fuse, rerank), because the reranker is usually the p99 and it is the first thing to shed under load.

**Dashboard 2 — Coverage and abstention.** Empty-result rate (queries returning zero chunks above the relevance floor), abstention rate, and the **ratio between them** — abstention much higher than empty-result means the model is abstaining despite having material, which is over-abstention leaking into production. Split all three by tenant, because a single tenant whose connector broke will show a 40% empty-result rate while the global number moves by 0.3%.

**Dashboard 3 — Grounding quality.** Citation verification rate, citation coverage (fraction of factual sentences cited), NLI groundedness distribution, and contradiction-flag rate. Contradiction rate gets a hard alert; it is high-precision and it means something specific broke.

**Dashboard 4 — Corpus health.** Documents by state (live / quarantined / stuck / tombstoned), quarantine depth and its derivative, time-since-last-successful-sync **per connector**, zero-chunk document count, and chunk-usage skew.

The alerts I would actually page on, and only these: empty-result rate up more than 3× baseline for 15 minutes; any connector with no successful sync in 2× its expected interval; contradiction-flag rate above threshold; p95 end-to-end above SLO; and documents stuck in a non-terminal ingestion state for over an hour. Everything else is a dashboard you look at during triage, not a page.

**⚠ Trap:** monitoring only aggregate quality. RAG degrades **per-tenant and per-topic**, not globally. A connector that broke for one customer, a document type that a parser update started mangling, a new product area with no docs — all of these are invisible in the global mean and glaring in a per-segment breakdown. Every quality metric in this system should be sliceable by tenant, source system, and query intent. If your metrics pipeline cannot do that, fix the metrics pipeline before you tune the retriever.

### The retrieval-score distribution shifted overnight. What does that tell you and what do you check?

A distribution shift is a *localizing* signal, and its shape tells you which subsystem to open. Read it as three separate patterns.

**Shift down and left, whole distribution, sharp discontinuity in time.** Everything scores worse. Almost always a **component change**: a different embedding model on the query side than the index side, a model version silently updated by your provider, a normalization change (someone removed an L2-normalize and now cosine is computing on unnormalized vectors), or a query-preprocessing change (a new lowercasing or stripping step). Check the query-service embedding model ID against the index metadata first — this is a 30-second check that resolves the most catastrophic cause. A sharp discontinuity means a deploy; correlate with the deploy timeline before anything else.

**Shift down, gradual, over days or weeks.** Usually the **query distribution changed**, not the index. New users, a new entry point, a link from a different surface, a seasonal shift in what people ask. Verify by scoring last week's queries against today's index — if the old queries score the same, the index is fine and your users changed. This is not a bug; it is a signal that your golden set is drifting out of relevance and needs new queries mined from the log.

**Distribution flattens — top-1 falls toward rank-10, gap narrows.** This is the **duplicate flood** signature. A re-export or a dedup bug injected many near-identical chunks; now the top of the ranking is a plateau of equivalent copies and the discriminative signal is gone. Check unique-`doc_id` count within top-10 per query, averaged. If that number dropped from 7 to 2, you have found it. This also shows up as chunk-usage skew.

**Distribution shifts *up*, everything scoring higher.** Counterintuitively often bad. Either the corpus shrank (an ingestion failure deleted a swathe of documents and the survivors are the ones that always scored well) or a filter got more aggressive and you are now searching a smaller, more homogeneous pool. Check document counts. A quality metric moving in the "good" direction with no corresponding change is always worth five minutes.

**🔍 Failure taxonomy — decision procedure:** discontinuous drop → check embedding model IDs on both sides, then the deploy log. Gradual drop → replay last month's queries against today's index; same scores means the users changed. Flattening → count distinct doc_ids in top-10. Rising → count documents. Each of these is a query, not an investigation, and together they cover the overwhelming majority of shift causes in under ten minutes.

### What do empty-result rate and chunk-usage skew actually tell you?

**Empty-result rate** — the fraction of queries where nothing clears your relevance floor — is the closest thing RAG has to a 5xx rate. It is a coverage metric, and its value is almost entirely in its *slicing*.

Globally it is a weak signal (a healthy system with real out-of-scope traffic sits at some non-zero baseline, often 2–8%). Sliced, it is diagnostic: by **tenant** it finds broken connectors and customers whose corpus was never fully ingested; by **query intent** it finds capability gaps ("how do I…" queries returning nothing means you have no procedural documentation indexed); by **language** it finds a multilingual retrieval failure that the English-speaking team will never notice; by **time** it finds ingestion regressions with a sharp edge. And the queries in the empty bucket are your **highest-value content backlog** — a clustered list of what users asked and you could not answer is a better roadmap input than any survey.

**Chunk-usage skew** — the distribution over how often each chunk is retrieved into a final context — is the more subtle one, and I read it as a Gini-like concentration measure plus a long-tail count.

*Extreme concentration*: 1% of chunks serve 60% of retrievals. Two causes, opposite implications. Either your traffic is genuinely head-heavy (fine, and it tells you exactly what to precompute or cache — the top 200 chunks are a cache-warming list and possibly a candidate for a fixed prefix-cached preamble), or a small set of chunks is **over-retrieving spuriously** — usually generic boilerplate, an FAQ index page, or a table of contents that is semantically close to everything. That second case is a quality bug: those chunks are crowding out specific evidence on every query. Look at the top-20 most-retrieved chunks by hand; if any of them is a navigation page or a document header, exclude that content class at ingest.

*Extreme sparsity*: a large fraction of chunks are **never retrieved, ever**, across months of traffic. That is dead weight — storage cost, index cost, and reindex cost for content no user has ever needed. It also raises a real question about whether those documents are unreachable (a metadata filter permanently excludes them, an ACL is wrong, they are in a language your embedder handles poorly) rather than merely unpopular. Distinguish by running your golden queries against the never-retrieved slice specifically.

**💰 Math:** if 55% of a 20M-chunk index has never been retrieved in 6 months, and your index costs, say, $2,400/month at 20M chunks, then $1,320/month is being spent on content that has never served a query. That is not automatically waste — a legal archive exists for the one query a year — but it is a number worth putting in front of whoever owns the budget, alongside the question "is this unreachable or just unpopular," because those have different answers.

**⚠ Trap:** using low retrieval frequency to prune the corpus. Retrieval frequency measures demand *and* reachability, and you cannot tell which without checking. Pruning on frequency alone deletes exactly the documents your retriever was already failing to surface — you would be automating the FP1 failure.

### Overnight, your system started answering "I don't have enough information" to almost everything. Walk me through the debug.

A mass-abstention event is a *good* incident to get, because abstention is the one failure mode that fails loud. It also has a short cause list, and the ordering is by how fast you can check.

**Check 1 — is retrieval returning anything at all?** Look at the empty-result rate and the top-1 score distribution. Two very different shapes here. If retrieval is returning chunks with *normal* scores and the system is still abstaining, skip to check 4. If retrieval is returning zero or scoring far below baseline, you have a retrieval-layer outage and the next three checks find it.

**Check 2 — embedding model mismatch on the query side.** The catastrophic one. A deploy changed the query service's embedding model or its dimension while the index was built with the old one. If dimensions differ you get errors; if they match you get silent garbage — every score collapses toward the noise floor and everything falls below your relevance threshold. Compare the query service's configured model ID against the index metadata. Thirty seconds, and it is the highest-prior cause after any deploy.

**Check 3 — the filter predicate.** A metadata or ACL filter that now matches nothing. Classic causes: a permission-resolution service returning an empty group list on error instead of failing closed *loudly* (so every query filters to zero documents), a `status='current'` predicate against a column an ingestion change stopped populating, or a tenant ID type change. Log the **filter predicate as executed** — this is exactly why that field is in the trace schema — and run the same query with the filter removed. If it returns fine, you have found it. The permission-service-returns-empty case is worth calling out specifically because it looks like a quality bug and is actually a dependency outage.

**Check 4 — the abstention threshold or the sufficiency prompt.** If retrieval looks healthy and the model is still refusing: did someone change the score threshold, or the system prompt, or the model? A model version change is the sneakiest — a new model that is more conservative will raise abstention across the board with no code change on your side. Compare the system-prompt hash and the model ID in traces from before and after the change point. This is why the prompt hash belongs in every trace.

**Check 5 — the index itself.** Did an alias swap point at an empty or partial collection? Query the canary documents. Count documents.

**🔍 Failure taxonomy — decision procedure:** empty-result rate spiked → check query-side embedding model, then the executed filter predicate, then index doc count. Empty-result rate flat but abstention spiked → check model ID, system-prompt hash, and threshold config. Both flat but users complain → it is not mass abstention, it is a per-segment problem and you need the slicing.

**⚠ Trap:** the failure that fails *closed* on permissions. A group-resolution service that returns `[]` on timeout, combined with a pre-filter, produces a system that is perfectly secure and completely useless — and every individual component reports healthy. Permission resolution must distinguish "this user is in no groups" from "I could not determine this user's groups," and the second must raise, not return empty. I check for this specific bug in every RAG review; it is present more often than not.

### Design the CI eval harness and the regression gate. What is the threshold, and what happens when it flakes?

The harness runs at three tiers, and the tiering is the design.

**PR tier — under 3 minutes, zero LLM cost, blocking.** Against a **pinned index snapshot** committed as a fixture (a few thousand chunks, not the production corpus — reproducibility beats realism here). Runs: recall@5/@50, nDCG@10, MRR over 150–300 golden queries with span-level labels; per-hop retention; and structural assertions (every citation resolves, every quoted span matches the source at the claimed offsets, output validates against the schema, abstention fires on the unanswerable slice, token budget respected). Retrieval metrics are deterministic given a pinned index and a pinned embedding model, so **the threshold can be tight**: fail on any drop greater than 0.5 points on recall@50, or greater than 1.5 points on recall@5. Structural assertions are boolean — any failure blocks.

**Nightly tier — non-blocking, full pipeline, judged.** 500 queries end-to-end against a staging index refreshed from production. Faithfulness, answer correctness, citation precision/recall, abstention rates on both slices, and cost/latency percentiles. Posted as a **trend with bootstrap CIs**, and alerting when the trend crosses a band rather than on a single run. Pin the judge model version explicitly and treat a judge upgrade as a re-baselining event.

**Release tier — human.** 50 stratified samples reviewed by a human before any significant release. Nothing replaces this and everybody tries to.

On flakes. First, **measure the noise floor before setting any threshold**: run the identical config three times and record the spread. Your gate must sit outside that spread or it will fire on nothing. For retrieval metrics with a pinned index the spread should be exactly zero — if it is not, you have nondeterminism (an ANN index with a randomized build, unpinned tie-breaking, concurrent evaluation order) and you fix *that* rather than widening the threshold. **Nondeterminism in a gate is a bug in the gate, not a reason to loosen it.** That is the rule I enforce, and it is the difference between a gate that survives and a gate that gets `@skip`-ed in month three.

For judged metrics, never gate on a single run. Gate on a 3-run median if you must gate at all, and prefer alerting a human over blocking a merge.

Two things that make the harness survive contact with a team: **the failure message must name the queries that regressed**, with a link to the trace, not just the delta — a gate that says "recall@5 dropped 2 points" gets ignored, a gate that says "these 7 queries lost their golden chunk, here are the traces" gets fixed. And **the golden set lives in the repo and changes via PR** with a required reviewer, because the easiest way to pass an eval gate is to edit the eval.

**🗣 Say this in the room:** "Three tiers: deterministic retrieval metrics and structural assertions blocking every PR in under three minutes with zero LLM spend; judged end-to-end nightly as a trend with confidence intervals; human review before release. I measure the noise floor before I set any threshold, and if a deterministic gate is flaky I fix the nondeterminism rather than widening the band — otherwise it gets disabled within a month."

### How do you turn a thumbs-down into a golden-set row without drowning in noise?

The flywheel only works if the path from signal to labeled example is short and mostly automatic. Design it as a pipeline, not as a Jira process.

**Capture, richly.** A thumbs-down alone is nearly useless — it conflates "wrong," "incomplete," "too slow," "I don't like the tone," and "I clicked it by accident." Capture the `request_id` (which joins to the full trace: query, rewrite, candidates with scores, reranked set, prompt chunk IDs, answer, citations), plus an optional one-click category (*wrong / missing information / didn't answer my question / wrong source*) whose categories map **directly onto the failure taxonomy** — that mapping is the design decision that makes triage cheap. Capture implicit signals too, which are higher-volume and less biased: did the user immediately rephrase (a strong negative), did they click a citation (weak positive), did they copy the answer (strong positive), did they escalate to a human.

**Triage, mostly automatically.** A nightly job takes each negative-feedback trace and runs the diagnosis ladder programmatically: was any golden chunk retrievable at k=200 (needs a label, so this step is where humans enter), does an NLI sweep find the answer ungrounded, did a post-retrieval hop drop a high-scoring chunk, did the model abstain when material was present. Most traces self-classify into a bucket. Cluster the remainder by query embedding so a human triages *clusters*, not individual events — 400 thumbs-downs a week is often 25 distinct problems.

**Promote, selectively.** Not every complaint becomes a golden row; that path leads to an eval set that is 80% edge cases and no longer represents your traffic. My rule: promote a *cluster representative* when the cluster has ≥3 instances, and label it properly (span-level golden evidence, reference answer). Track the composition of the golden set explicitly — I want to know what fraction is synthetic, mined-from-logs, expert-written, and promoted-from-failures, and I want the last two growing.

**Close the loop visibly.** When a fix ships, re-run the promoted queries and report "these 14 previously-failing queries now pass." This is how the eval set earns its maintenance budget from a skeptical manager, and it is the artifact I would show in an interview.

**⚠ Trap:** feedback selection bias. People thumbs-down confidently-wrong answers and never thumbs-down *incomplete* ones, because incompleteness is invisible to the person who does not know what is missing. So your feedback stream will systematically under-represent FP7, which is one of your most damaging failures. Counter it deliberately: sample random *positive* and *unrated* sessions for periodic human audit, at maybe 50 per week. That audit is the only instrument that finds silent incompleteness, and it is the first thing cut when the team is busy.

### How do you stream an answer when citations are only verified after generation completes?

This is a genuine product/architecture tension and the answer is a design decision, not a trick. You have three options and they trade latency against safety.

**Option A — stream freely, annotate after.** Tokens go to the user immediately; citation verification and NLI groundedness run when the stream ends; the UI then attaches verified citation links and, if verification failed, shows a warning banner or greys the affected sentence. Best TTFT (unchanged), worst safety property: the user has already read an unverified claim, and retracting it after the fact is a bad experience and, in a regulated domain, arguably not a mitigation at all.

**Option B — buffer, verify, then render.** Generate into a server-side buffer, run verification, render or suppress. Safe, and it costs you the entire generation duration in perceived latency: a 400-token answer at ~60 tok/s is **6.7 seconds of blank screen**, versus a TTFT of ~600 ms with streaming. That is a 10× regression in perceived responsiveness and it will show up in your engagement metrics. Only correct for high-stakes, low-frequency surfaces.

**Option C — sentence-level pipelined verification.** This is what I ship when I need both. Stream generation, but buffer at **sentence boundaries** rather than tokens. When a sentence completes, dispatch its verification (quote string-match ≈ microseconds; NLI against 3 candidate premises ≈ 15–40 ms batched on a warm GPU) concurrently while the model continues generating the next sentence. Release the sentence to the client once its check returns. Since NLI latency (~30 ms) is far below sentence generation time (a 25-token sentence at 60 tok/s is ~420 ms), verification is fully hidden behind generation and your added perceived latency is roughly one sentence — **about 400 ms of TTFT, not 6.7 s.**

```
gen ──sent1──┬──sent2──┬──sent3──┬─ ...
             │         │         │
          verify1   verify2   verify3      (30 ms each, concurrent with gen)
             │         │         │
          emit1     emit2     emit3        (~1 sentence behind the model)
```

The one real cost of Option C: if sentence 6 fails verification, sentences 1–5 are already on screen. You cannot un-say them. So the failure handling has to be designed — mark the offending sentence inline, and for hard failures (a contradiction flag) stop the stream and replace the whole answer with an abstention. Stopping mid-stream is jarring but honest, and it is strictly better than the alternative.

A necessary detail: sentence-boundary buffering must handle code blocks, tables, lists and abbreviations, or you will split "e.g." and "v1.2." into sentences and verify fragments. Use a real segmenter with a fenced-block guard, not a regex on periods. This is the kind of unglamorous correctness that separates a demo from a product.

**📐 Numbers you must know:** at ~60 output tokens/sec (a reasonable mid-size frontier model streaming figure — 📅 Volatile, measure yours), a 400-token answer takes **6.7 s** end to end. TTFT on a warm prefix-cached prompt is typically 300–800 ms. Option A costs 0 added latency, Option C costs ~1 sentence (~400 ms), Option B costs the full 6.7 s. Those three numbers are the whole decision.

### Here are three traces from a broken RAG system. You have ten minutes. Localize the fault.

**🏋 Drill — the fault-localization drill, unaided, ten minutes on a clock.** This is the round that grew fastest since 2024 and it is worth rehearsing until the order is automatic. Below are three traces; the pass criterion is naming the correct gate and the correct one-line fix for all three in under ten minutes, without running anything.

**Trace 1.**
```
query: "what is the escalation SLA for a P1 incident on the enterprise plan?"
rewritten: "what is the escalation SLA for a P1 incident on the enterprise plan?"
retrieval: dense k=50 -> 50 hits, top scores [0.71, 0.70, 0.70, 0.69, 0.69, ...0.66]
rerank -> 8; in_prompt chunk_ids = [c_8812, c_4410, c_8813, c_9902, ...]
distinct doc_ids in top-10: 2
answer: "Enterprise P1 incidents are escalated within 4 hours."   (correct SLA is 1 hour)
groundedness: 1.00   citation: c_8812 (verified quote)
```

**Trace 2.**
```
query: "does the new payout policy apply to India?"
rewritten: "does it apply to India?"
retrieval: hybrid k=50, top scores [0.42, 0.41, 0.40, ...]
in_prompt chunk_ids from 8 distinct docs, none from payout-policy-v4.md
answer: "I could not find information about this."
```

**Trace 3.**
```
query: "list every region where same-day payouts are supported"
retrieval: hybrid k=50 -> rerank -> 5
in_prompt: 5 chunks, all from payout-regions.md, ranks 1-5, scores [0.88..0.71]
answer: "Same-day payouts are supported in the US, UK, and Ireland."
groundedness: 1.00   citations: all verified
ground truth: US, UK, Ireland, Netherlands, Germany, Singapore
```

**Trace 1 — gate 2/4, duplicate flood.** The signature is a **flat score distribution** (0.71 down to 0.66 across 50 hits, a 0.05 spread) plus **2 distinct doc_ids in the top 10**, and consecutive chunk IDs `c_8812`/`c_8813`. The index is full of near-identical copies of the same document, so the top-k is one document repeated and the *current* version never surfaces. Groundedness is 1.00 because the answer is faithful to a stale duplicate. Fix: content-version dedup at ingest plus a `status='current'` filter — and note that no amount of prompt work would have touched this.

**Trace 2 — gate 3, query rewriting.** The rewrite *dropped* the subject: "does it apply to India?" has no referent. Whatever the model saw as chat history, the rewriter resolved wrongly or the rewrite ran on the wrong turn. The uniformly low scores (0.42 top) confirm the embedded query matches nothing well. Fix: the conversational rewriter, and add a guard that rejects a rewrite containing an unresolved pronoun or scoring below a floor, falling back to the raw query.

**Trace 3 — FP7, incompleteness.** Everything is green: high scores, single coherent document, perfect groundedness, verified citations. And the answer is 50% complete. This is an **enumeration query answered with top-k**, and `k=5` after reranking structurally cannot return six regions if they span more than five chunks. Fix: intent-route enumeration queries to a metadata-filtered exhaustive retrieval over the whole document, or map-reduce over all chunks of the matched document rather than top-k. The lesson worth stating out loud: **trace 3 is the dangerous one, because every metric you have is passing.**

**🗣 Say this in the room:** "Flat score distribution plus few distinct doc IDs is duplicates. Low absolute scores plus a rewritten query that lost its subject is the rewriter. High scores, one document, perfect groundedness, and a list-type question is incompleteness — and that's the one where all my metrics are green, so I'd want an explicit exhaustiveness check on enumeration intents."

### Build me a retrieval evaluation harness. You have 45 minutes and no frameworks.

**🏋 Drill — 45 minutes, unaided, no LangChain, no RAGAS.** Pass criterion: it runs, it computes paired confidence intervals, and the golden labels are span-based so a chunk-size change does not invalidate the set. Target roughly this shape.

```python
import json, math, random, statistics
from dataclasses import dataclass

@dataclass(frozen=True)
class GoldenItem:
    qid: str
    query: str
    # span-level labels: (doc_id, start_char, end_char) of evidence in canonical text
    spans: tuple
    answer: str | None            # reference answer, may be None for retrieval-only
    answerable: bool

def covers(chunk, span) -> bool:
    doc, s, e = span
    return chunk.doc_id == doc and chunk.start_char <= s and chunk.end_char >= e

def score_query(item, ranked_chunks, k):
    topk = ranked_chunks[:k]
    hit = [any(covers(c, sp) for c in topk) for sp in item.spans]
    recall = sum(hit) / len(item.spans)
    first = next((i for i, c in enumerate(topk, 1)
                  if any(covers(c, sp) for sp in item.spans)), None)
    return {"recall": recall, "rr": 1.0 / first if first else 0.0,
            "full": float(all(hit))}      # all evidence present == sufficiency proxy

def evaluate(golden, retrieve, k=5):
    return {it.qid: score_query(it, retrieve(it.query), k)
            for it in golden if it.answerable}

def paired_bootstrap(a: dict, b: dict, metric: str, n=2000, seed=0):
    """95% CI on mean(b) - mean(a), resampling QUERIES with replacement."""
    rng, qids = random.Random(seed), sorted(set(a) & set(b))
    diffs = [b[q][metric] - a[q][metric] for q in qids]
    boots = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(s) / len(s))
    boots.sort()
    return (statistics.fmean(diffs), boots[int(0.025 * n)], boots[int(0.975 * n)])

def abstention_report(golden, run):
    ans = [it for it in golden if it.answerable]
    una = [it for it in golden if not it.answerable]
    return {
        "over_abstention": sum(run[it.qid]["abstained"] for it in ans) / len(ans),
        "correct_abstention": sum(run[it.qid]["abstained"] for it in una) / len(una),
    }
```

The four things I grade in this drill, in order of how often candidates miss them:

**Span-level labels, not chunk IDs.** Without this the harness is single-use and any chunking ablation silently produces garbage. This is the design decision that matters most and it takes one extra field.

**Paired bootstrap on the difference**, resampling queries, not a t-test on two independent means. If a candidate hands me two point estimates and a "looks better," that is the whole interview signal I need.

**Excluding unanswerable items from recall** and reporting abstention separately. Mixing them produces a metric that means nothing.

**`full` alongside `recall`** — the fraction of queries where *all* evidence spans were retrieved. That is the closest cheap proxy to answer-sufficiency and it is what actually predicts end-to-end correctness on multi-evidence questions, which mean recall smooths away.

If you finish early, add a `slice_by(fn)` that reports every metric split by an arbitrary key — turn index, query length bucket, has-identifier, tenant. Slicing is where a harness stops being a number and starts being a diagnostic.

### You inherit a RAG system with 62% user-rated answer correctness and one week. What is the plan, day by day?

The instinct to resist is starting with the model or the prompt. The plan is measurement-first because with one week you cannot afford one wrong bet.

**Day 1 — build the instrument, not the fix.** Assemble 150 queries: 60 mined from real logs stratified by cluster, 40 synthetic-and-paraphrased from the corpus, 25 written by two domain experts cold, 25 adversarial/unanswerable. Label evidence at the *span* level. This is the whole day and it is non-negotiable — every subsequent decision is an inference from this artifact, and a bad golden set makes the rest of the week noise.

**Day 2 morning — the oracle ablation.** Run production (condition A), oracle context (condition B), oracle-plus-distractors (condition C). Roughly $12 and twenty minutes of compute. This single experiment allocates the remaining four days. Say A=0.62, B=0.90, C=0.74: retrieval is the bottleneck (28 points available), and the generator is fragile to distractors (16-point drop from B to C), which already tells you to favour precision over recall.

**Day 2 afternoon — the retrieval ladder.** recall@5 vs recall@50 vs recall@200 on the golden set, plus per-hop retention. Suppose recall@200 = 0.94, recall@50 = 0.91, recall@5 = 0.58. Ceiling is fine; ranking is broken. Also check distinct doc_ids in top-10 and the empty-result rate for duplicate and coverage problems, and diff the ingested document count against the source system for silent FP1.

**Day 3 — the single highest-leverage change, measured.** Given those numbers: add a cross-encoder reranker over the top-50. Measure paired. Expect recall@5 to move into the 0.78–0.85 range if the pattern holds. Then, if the corpus has identifiers, turn on BM25 and RRF-fuse — a second paired measurement. Two changes, two independent measurements, one variable at a time. Ship whichever clears its CI.

**Day 4 — precision and assembly, given the C-condition result.** Since distractors hurt, sweep `k_final` ∈ {3, 5, 8} at a matched token budget and measure end-to-end, not just retrieval — I expect a smaller `k` to win here, which is the counterintuitive result that the C ablation predicted. Fix context ordering (best chunk adjacent to the question), turn on near-duplicate suppression, and check for a status/version filter.

**Day 5 — the safety floor.** Abstention: a calibrated retrieval-score gate plus an explicit sufficiency instruction, measured as a paired regression on both slices so you can state exactly what it costs in over-abstention. Add citation emission with offset verification. These do not raise correctness; they convert a chunk of the residual wrong answers into honest non-answers, which for most products is a larger perceived-quality win than the same points spent on accuracy.

**Day 6 — lock it in.** Wire the golden set into CI with retrieval metrics and structural assertions as blocking gates. Stand up the four dashboards. Add feedback capture joined to `request_id`. Without this, week two undoes week one.

**Day 7 — write it up with numbers.** Baseline 0.62 → new number, per-change attribution with confidence intervals, cost delta per request, latency delta at p95, and the ranked list of what you did *not* have time to do with the expected value of each. That document is the artifact that gets you the second week, and in an interview it is the answer that separates a senior engineer from a strong mid-level one.

**💰 Math for the write-up:** if the reranker moved end-to-end correctness 0.62 → 0.79 at +180 ms p95 and $0.002/query, then at 200k queries/day the added cost is 200,000 × 0.002 = $400/day = **$12,000/month**. If the product deflects support tickets at, say, $6 of loaded cost each, and 17 points of correctness converts an extra 3% of 200k daily sessions into deflections, that is 6,000 × $6 = $36,000/day of avoided cost. State the assumption explicitly — "at a $6 fully-loaded ticket cost, which I'd want finance to confirm" — because an unchallenged made-up denominator is how these arguments lose credibility. The structure of the argument is what is being graded, not the number.

**🗣 Say this in the room:** "Day one is the golden set, day two is the oracle ablation that tells me whether the next four days go into retrieval or generation, and day seven is a write-up with per-change confidence intervals and a cost delta. I would not touch the prompt before day two, because prompt work against a fifty-eight-percent retrieval ceiling is the most common way teams burn a quarter."
