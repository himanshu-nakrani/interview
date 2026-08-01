### A user tells you the answer your RAG system gave was wrong. Before you touch a single config value, what is your diagnosis order?

The mental model that makes this fast: **a RAG pipeline is a chain of lossy filters, and every filter can only lose information the previous one already passed.** So you never debug it by staring at the output. You walk the chain forward and find the first gate where the required evidence stopped existing. That gate is the bug. Everything downstream of it is a symptom, and everything upstream of it is fine by construction.

Five gates, in this order, each with one question and one metric:

1. **Is it in the corpus at all?** Grep the raw source store — not the index, the source — for the fact. Metric: binary. If the fact never got ingested, no amount of reranking will conjure it.
2. **Is it in the index, correctly chunked?** Search the chunk store by document ID and read the chunks the fact should live in. Metric: does the fact survive intact inside one chunk, or did the splitter cut it in half. This is where parse failures and boundary bugs surface.
3. **Was it retrieved at all?** Run the user's query against the retriever with `k=100` (or 200 — deliberately far past production `k`). Metric: **recall@100 on the golden chunk.** If the right chunk is not in the top 100, you have a *retrieval* failure — embedding mismatch, vocabulary gap, missing lexical channel.
4. **Was it ranked into the window?** If it is at rank 43 with `k=5` in production, you have a *ranking* failure, which is a completely different fix (reranker, hybrid weight, query rewriting) from a retrieval failure. Metric: **rank of the golden chunk, and nDCG@k.**
5. **Was it read, and was the answer faithful to it?** Force the correct chunk into the context (oracle context) and re-ask. Metric: does the model now answer correctly. If yes → the bug is retrieval/ranking, full stop. If no → generation failure: extraction, format, specificity, or an outright faithfulness break.

The reason to run gate 5 as an *oracle* experiment rather than as the last step is that it is the single cheapest disambiguator in the whole taxonomy. One API call splits the problem space in half.

**🗣 Say this in the room:** "In the corpus, in the index, retrieved, ranked, read, faithful — I walk those five gates in order and stop at the first one that fails. The metric changes at each gate: binary presence, then recall@100, then rank and nDCG@k, then an oracle-context re-ask. Most 'the model is dumb' bugs die at gate three."

**⚠ Trap:** the reflex to jump to gate 5 because generation is the visible layer. I have watched teams spend two weeks on prompt engineering for a bug that was a PDF parser silently dropping two-column pages. Gate order is not a preference; it is a cost ordering — gates 1 and 2 cost seconds, gate 5 costs a prompt-engineering sprint.

**💰 Math:** running the full ladder on one complaint is roughly: gate 1 a `SELECT` (0 tokens), gate 2 a `SELECT` (0 tokens), gate 3 one embedding call (~30 tokens at $0.02/Mtok ≈ $0.0000006) plus an ANN query, gate 4 free from the same result set, gate 5 two generation calls at ~4k input / 300 output. At $3/Mtok in and $15/Mtok out that is 2 × (4000 × 3/1e6 + 300 × 15/1e6) = 2 × (0.012 + 0.0045) = **$0.033 per complaint fully diagnosed.** 📅 Volatile: verify prices before your loop. There is no excuse for guessing when diagnosis costs three cents.

### Walk me through the seven failure points people cite for RAG systems, and tell me which ones you have actually hit.

**📄 Paper:** Barnett et al. (2024) — *Seven Failure Points When Engineering a Retrieval Augmented Generation System*. It is a case-study paper across three deployed systems rather than a benchmark, and its contribution is a vocabulary: it lets two engineers argue about *which* failure they have instead of arguing about whether RAG "works." That vocabulary is what interviewers are testing when they ask this.

The seven, in pipeline order, with what each actually means at the tensor/index level:

**FP1 — Missing content.** The answer simply is not in the corpus. The system's job here is to say so; instead it usually confabulates. This is an ingestion or scope failure, not a retrieval one.

**FP2 — Missed the top-ranked documents.** The correct chunk exists and *is* retrievable, but sits below your `k` cutoff. Pure ranking failure. The diagnostic is trivially recall@k for large k versus small k: if recall@100 = 0.95 and recall@5 = 0.55, you have FP2 and the fix is a reranker, not a new embedding model.

**FP3 — Not in context after consolidation.** The chunk made it into the retrieved set, then got dropped by a *later* stage — reranking, dedup, MMR diversification, context compression, or a token-budget truncation. This one is invisible unless you trace per hop, because retrieval metrics look fine and the prompt is missing the evidence anyway.

**FP4 — Not extracted.** The chunk is *in the prompt* and the model does not use it. Causes: it is buried mid-context (lost-in-the-middle), it is surrounded by three distractor chunks that look more on-topic, or the instruction told the model to be concise and it decided this detail was skippable.

**FP5 — Wrong format.** You asked for a table or JSON and got prose, or got a table with the wrong columns. Structural, not knowledge-level.

**FP6 — Incorrect specificity.** The answer is true but at the wrong altitude — you asked "what's the retry timeout for the payments webhook" and got a general paragraph about retry policies. This is very often a *chunking* failure surfacing as a generation symptom: if your chunks are 2,000-token sections, the model has no fine-grained fact to latch onto.

**FP7 — Incompleteness.** Each retrieved chunk is right, the answer covers one of them, and the question needed three. Multi-hop and enumeration questions ("list all regions where we do X") live here.

Which have I actually hit? FP2 and FP3 constantly — they are the bread and butter of a hybrid+rerank pipeline that has never been ablated. FP7 whenever a product ships a "summarize everything about X" surface without a fan-out retrieval strategy. FP1 is the most *expensive* because it is usually a silent ingestion bug rather than a design gap. FP5 mostly vanished once structured-output modes got good.

**⚠ Trap:** treating this as a seven-item list to recite. The interviewer's actual question is "can you map a symptom to a gate." Practise the inverse mapping: "answer was true but too general" → FP6 → suspect chunk size → check whether your chunks are section-level. "Answer covered two of the four regions" → FP7 → suspect top-k and query decomposition, not the model.

### How do you prove quickly that a fact simply is not in the corpus, rather than badly retrieved?

You do not use the vector index for this. That is the whole answer, and candidates get it wrong constantly.

Semantic search is the wrong instrument for a presence question, because it always returns *something* — ANN search over a normalized embedding space has no concept of "no match," it has a concept of "nearest, however far." A cosine similarity of 0.31 and a cosine similarity of 0.89 both come back as `results[0]`. So asking the vector index "is this in the corpus" is asking a question it structurally cannot answer.

What I actually run, in order:

**One: lexical grep over raw source text, not chunks.** A Postgres `to_tsquery` or a plain `ILIKE '%<distinctive token>%'` over the `documents.raw_text` column. Pick the most distinctive noun phrase or identifier in the user's question — an error code, a product SKU, a person's surname. Lexical exact match has no false negatives from semantic drift.

**Two: if lexical misses, grep the *pre-parse* blob.** This distinguishes "we never ingested the document" from "we ingested it and the parser ate it." I keep the raw bytes (or at least the raw extracted text prior to cleaning) precisely so this check is possible. If the string is in the raw text but not in `documents.raw_text`, your cleaning step is the bug. If it is in `documents.raw_text` but not in any chunk, your splitter is the bug.

**Three: check the document *should* be there.** Source-system count versus ingested count, per connector, per day. A Confluence connector that silently stopped paginating at page 10 will pass every health check you have and be missing 80% of the space.

```sql
-- the single most useful query in RAG ops
select d.source_system,
       count(*) filter (where d.parse_status = 'ok')        as parsed,
       count(*) filter (where d.parse_status = 'quarantine') as quarantined,
       count(*) filter (where c.doc_id is null)              as zero_chunk_docs,
       max(d.ingested_at)                                    as last_ingest
from documents d
left join lateral (select 1 from chunks c where c.doc_id = d.id limit 1) c on true
group by 1 order by zero_chunk_docs desc;
```

`zero_chunk_docs` is the column that catches the silent 5%. A document that parsed "successfully" into an empty string produces zero chunks and zero errors.

**⚠ Trap:** believing a low similarity score means absence. Cosine scores are not calibrated across queries — a short query has a different score distribution than a long one, and a domain-shifted query has a different one again. You cannot set a global "no result" threshold on raw cosine and expect it to mean anything. If you want an absence signal from the retriever, calibrate per-query: compare the top score against the mean of scores at ranks 20–50 for that same query. A flat distribution means nothing matched; a sharp drop-off means something did.

### The correct chunk is definitely in the index but did not come back in the top five. Talk me through what you check.

First split the failure, because "did not come back" hides two different bugs with two different fixes. Run the same query at `k=200`. If the golden chunk is at rank 43, this is a **ranking** failure — the retriever can see it, it just ordered badly. If it is not in the top 200 at all, this is a **representation** failure — the retriever cannot see it, and no amount of reranking will help, because a reranker only reorders what retrieval handed it.

For a **ranking** failure, in order of how often they are the cause:

Add a cross-encoder reranker over the top 50–100 and measure. This is the highest-leverage single change in RAG and it exists precisely because bi-encoder recall is good and bi-encoder precision is mediocre. Then check whether hybrid fusion is actually on and weighted sanely — if your query contains an identifier (`ERR_5041`, `v2.14.3`, a surname), a dense-only retriever will rank semantically-similar-but-wrong chunks above the exact match, because embedding models compress rare tokens hardest. Then look at query rewriting: in a multi-turn session, "does it also apply to the enterprise plan?" embedded literally retrieves nothing about your actual subject.

For a **representation** failure:

Check for a vocabulary gap between how users ask and how documents are written. Users say "my card got declined"; the doc says "issuer authorization failure." Dense retrieval is supposed to bridge that and often does not for domain jargon. Fixes are HyDE (embed a hypothetical answer instead of the question), doc-side enrichment (prepend an LLM-generated summary or a list of questions this chunk answers), or fine-tuning the embedding model on your own query-document pairs.

Then check chunk length. A 1,500-token chunk embedded into a single 1,024-dim vector is a heavily lossy average; a specific fact inside it contributes maybe 2% of the vector's direction. This is the mechanical reason small-to-big retrieval works: embed a small precise unit, return a large contextful one.

Then check whether the chunk is *drowned by near-duplicates*. If your corpus has 40 near-identical copies of a boilerplate section, they will occupy your top-k and crowd out the one variant that carries the answer.

**📐 Numbers you must know:** on a decent hybrid pipeline over a domain corpus, expect recall@100 around 0.93–0.97 and recall@5 around 0.55–0.70 *before* reranking, and recall@5 around 0.80–0.88 after a good cross-encoder. Those are the shapes, not guarantees for your corpus — but if your recall@100 is 0.70, stop tuning the reranker. Your ceiling is 0.70 and the reranker can only hurt.

**🗣 Say this in the room:** "I'd first separate retrieval failure from ranking failure by re-running at k equals two hundred. Those have disjoint fixes — a reranker cannot recover a document retrieval never returned, and a new embedding model is a waste of a month if recall at one hundred is already ninety-five percent."

### What is "not in context after consolidation," and why does adding a reranker sometimes cause it?

This is failure point 3, and it is the one that most often survives a code review, because every stage in isolation looks correct.

The mechanism: your pipeline has stages after retrieval that *remove* candidates — a reranker truncating 100 to 8, MMR or near-duplicate suppression dropping redundant chunks, a context compressor summarizing chunks down to fit a budget, or a hard token-budget truncation that lops off whatever did not fit. Any of these can drop the chunk that retrieval correctly surfaced. Your retrieval metrics — recall@100, nDCG@100 — are computed on the retriever's output and look excellent. Your generation is missing the evidence anyway. The two observations are consistent and the gap between them is invisible unless you instrument it.

Reranking causes it specifically because a cross-encoder is optimizing *query-chunk relevance*, and relevance is not the same as *sufficiency*. Consider "what changed between v3 and v4 of the pricing policy?" The v3 chunk and the v4 chunk are both moderately relevant; a chunk that says "the pricing policy governs all enterprise contracts and has been revised twice" is *highly* relevant to the query string and useless as evidence. A cross-encoder trained on MS MARCO-style relevance will happily rank the useless one first.

MMR causes it in the opposite direction: you turn on diversity to fix "five copies of the same paragraph," and the diversity penalty now drops the second-most-relevant chunk because it is textually similar to the first — except that in a "compare A and B" question, the two chunks you need *are* similar. MMR at λ=0.5 is actively harmful for comparison queries and helpful for exploratory ones. There is no single λ; there is a routing decision.

The fix is instrumentation, not a better algorithm. **Log the chunk ID set at every hop.** Retrieved set → reranked set → post-dedup set → post-compression set → what actually landed in the prompt. Then compute a *per-hop retention* metric on your golden set: for each stage, what fraction of golden chunks that entered the stage exited it.

```python
HOPS = ["retrieved", "reranked", "deduped", "compressed", "in_prompt"]

def hop_retention(trace, golden_ids):
    """trace: {hop_name: [chunk_id, ...]}. Returns per-hop survival of golden chunks."""
    out, prev = {}, set(golden_ids)
    for hop in HOPS:
        present = set(trace[hop]) & prev
        out[hop] = len(present) / max(len(prev), 1)   # survival given it entered
        prev = present
    return out
```

Run that over 200 golden queries and the offending stage announces itself: one hop with retention 0.71 while every other hop is 0.98.

**⚠ Trap:** "the reranker improved nDCG@10 so it is helping." nDCG@10 measures ordering quality of the reranked list against *labeled relevance*, which is often annotated as topical relevance. Your system needs *answer sufficiency*. I have seen a reranker raise nDCG@10 by 0.06 and drop end-to-end answer accuracy by 4 points, because it systematically preferred summary-ish chunks over detail-ish ones. Always carry an end-to-end metric alongside the retrieval metric; the retrieval metric is a proxy and proxies drift.

### The right chunk was in the prompt, verbatim, and the model still answered wrong. Diagnose that.

Good — this is failure point 4, and now we are in generation, which means the diagnostic tools change from set operations to controlled prompt experiments. There are five causes and they are separable with cheap ablations.

**Position.** The evidence is at index 6 of 8 chunks in a 12k-token prompt. **📄 Paper:** Liu et al. (2023) — *Lost in the Middle: How Language Models Use Long Contexts* — showed a U-shaped accuracy curve over the position of the gold document: strong at the beginning, strong at the end, materially worse in the middle. Test: move the golden chunk to position 1 and re-run. If accuracy jumps, your context assembly ordering is the bug. Fix: put the top-reranked chunk last (adjacent to the question) or first, and never bury it.

**Distractors.** The other seven chunks are topically adjacent and contradict or dilute. Test: run with the golden chunk *alone*. If it answers correctly with one chunk and wrongly with eight, you have a distractor problem, and the fix is precision — a tighter reranker, a lower `k`, or a filtering pass. This is the empirical case against "just raise k to 20 to be safe." Raising `k` monotonically raises recall and non-monotonically changes accuracy.

**Conflict.** Two retrieved chunks state different things — usually the v1 doc and the v3 doc both live in your index because nobody ever deleted the old one. The model picks one, often the one that looks more authoritative or appears first. Test: check for contradictory chunks in the retrieved set. Fix is metadata and filtering (`status = 'current'`), not prompting.

**Instruction conflict.** Your system prompt says "answer in under 60 words" or "be concise," and the correct answer requires reciting a four-clause condition. The model obeys the style instruction and drops the clause. Test: relax the length constraint and re-run. This one is embarrassing and common.

**Reasoning load.** The chunk contains the raw numbers and the question needs arithmetic or a two-step inference over them. Models fail at this more than people expect when the numbers are embedded in prose. Test: ask the sub-question directly. Fix is either a reasoning-mode model, an explicit "first extract the relevant figures, then compute" scaffold, or a code-execution tool.

**🗣 Say this in the room:** "I'd run three ablations that take about five minutes total: golden chunk alone, golden chunk moved to position one, and length constraint removed. Those three separate distractor interference, positional burial, and instruction conflict, which are the three causes with completely different fixes."

**💰 Math:** those three ablations across a 50-query golden set are 150 calls at ~6k input / 250 output. 150 × (6000 × 3/1e6 + 250 × 15/1e6) = 150 × (0.018 + 0.00375) = **$3.26**. Against an engineer-week of prompt guessing at, say, $4,000 fully loaded, the ablation is free. The rule I enforce in review is that nobody is allowed to change a RAG prompt without an ablation number attached.

### Wrong format, wrong specificity and incompleteness all feel like prompt problems. Are they?

One of the three is. The other two are architecture problems wearing a prompt costume, and conflating them is how teams end up with a 3,000-token system prompt that fixes nothing.

**Wrong format (FP5) genuinely is a prompt/decoding problem** and in 2026 it is close to solved. Use the provider's structured-output mode with a JSON Schema, or constrained decoding locally. If you are still parsing markdown tables out of prose with a regex, you have chosen the failure. The residual risk is that constraining output too hard degrades content quality — a schema demanding `{"answer": str, "confidence": float}` invites a fabricated float. My rule: constrain *structure*, never *epistemics*. Never ask a model for a numeric confidence in a schema slot; it will fill it and it will be meaningless.

**Wrong specificity (FP6) is usually a chunking problem.** If your chunks are 1,500-token document sections, then the only thing the model can see about "retry timeout for the payments webhook" is a paragraph about retry policy generally, and it will faithfully summarize what it has. No prompt fixes an absent detail. The diagnostic is: read the retrieved chunks yourself and ask "does the specific fact appear here at all?" If it does not, this is FP1/FP2 mislabeled. If it does appear but is one sentence inside 1,500 tokens, you have a signal-dilution problem and the fixes are small-to-big retrieval (embed sentences or propositions, return the parent section), or a smaller chunk size with contextual prefixing so the small chunk is still interpretable.

The opposite direction exists too and is rarer: chunks so small the answer needs six of them, so the model gives an over-specific fragment. Specificity failures point at chunk size in both directions, which is why "why 512 tokens" is a defensible question with an ablation as the only acceptable answer.

**Incompleteness (FP7) is a retrieval-strategy problem.** "List every region where we support same-day payouts" is not a top-k question. If the answer is spread across nine documents and `k=5`, you are structurally guaranteed to be incomplete no matter how good the ranking is. The fixes are architectural: query decomposition into sub-questions with a union of retrieved sets, metadata-filtered exhaustive retrieval (filter to `doc_type='payout_policy'` and retrieve *all* matching chunks, not top-k), or a map-reduce pass over the filtered set. The tell that you need this is a query-intent classifier flagging enumeration/aggregation intents — "list all," "how many," "every," "compare across."

**⚠ Trap:** answering an enumeration query with top-k retrieval and a confident-sounding list. The model produces five regions, all correct, and the user believes that is the complete set. This is the single most dangerous RAG failure mode in enterprise settings because the output has no visible defect — it is *silently* incomplete. If you cannot guarantee exhaustiveness, the system must say "here are the ones I found; this may not be exhaustive." I treat that as a hard product requirement, not a nicety.

### Give me your list of ingestion-side failures. Which one costs the most?

Retrieval and generation get all the attention because they are where the demos happen. Ingestion is where the outages happen. Four classes:

**Parse errors, especially silent ones.** A PDF with a text layer parses fine; a scanned PDF parses to an empty string, or worse, to a handful of garbage ligatures. A DOCX with tracked changes parses to the pre-change text. A two-column academic layout parses to interleaved sentence fragments that are individually grammatical and collectively meaningless. The killer is that most of these produce *no exception*. Your pipeline reports 100% success. The alert that catches it is not "did parse throw" but **content-plausibility assertions per document**: characters extracted per page below a floor (I use ~200 for prose), alphanumeric ratio below a floor, chunk count of zero, or a language-detection confidence collapse. Anything that trips goes to a quarantine table with the raw bytes, and quarantine depth is a dashboard, not a log line.

**Stale index.** The source changed and the index did not. Causes: a webhook that fires on create but not on update; a change-detection scheme keyed on `updated_at` from a system that does not touch `updated_at` on content edits; a deletion that removed the row from the source and left the vector behind. The last one is the worst variant — **deletions are the most commonly unimplemented path in every RAG pipeline I have reviewed.** Ingestion is written first, update second, delete never.

**Permission drift.** Documents carry an ACL at ingest time; the ACL changes; your index still has the old one. Now a former contractor's query returns a document they lost access to last quarter. This is the failure that gets a system pulled from production rather than patched, and it deserves its own answer.

**Dedup collisions.** Content-hash dedup that hashes the wrong thing, so two genuinely different documents collide and one is silently discarded — or the more common inverse, a hash that includes a timestamp or a "last exported" footer so every re-export creates a new "document" and your index grows without bound with near-identical copies that then dominate top-k.

Which costs most? Ranked by expected damage: **permission drift** (existential — it is a data breach, not a quality bug), then **silent parse failure** (5% of your corpus is invisible and nobody knows which 5%), then **stale index** (users lose trust in a specific, memorable way: "it told me the old price"), then **dedup collisions** (degrades quality gradually and shows up as unexplained top-k pollution).

**📐 Numbers you must know:** in mixed enterprise corpora, expect **3–8% of documents to fail to yield usable text** with a naive text-extraction pipeline — scanned faxes, image-only slides, password-protected files, spreadsheets whose meaning is entirely in cell formatting. Budget for it explicitly. If your ingest report says 99.9% success on a corpus with any PDFs in it, your success criterion is broken, not your corpus.

### Walk me through exactly how a RAG system leaks a document the user should not see.

There are four mechanisms and only one of them is the obvious one. This matters because the obvious one is the one everybody guards.

**Mechanism 1: no filter at all.** The index is global, retrieval is global, and the only "security" is that the model was told not to reveal certain things. This is not a control. Anyone asking this question in an interview is checking whether you say "prompt-level access control is not access control" out loud. Say it.

**Mechanism 2: post-filtering instead of pre-filtering.** You retrieve top-50 globally, then drop the ones the user cannot see, then pass the survivors. Two bugs. First, quality: a user with narrow permissions gets three chunks instead of ten and their answers are systematically worse in a way you will never diagnose from aggregate metrics. Second, and worse, **the count itself leaks**: if the user can observe that a query returned fewer results, they learn documents exist. The correct design is a *pre-filtered* ANN search — the vector store applies the ACL predicate as part of the search so the candidate pool is already scoped. Every serious vector store supports filtered search; the trap is that some implement it as post-filtering internally and blow through your `k` silently, and you have to read the docs to find out which.

**Mechanism 3: permission drift — the ACL snapshot in the index goes stale.** You wrote `allowed_groups: ["eng", "eng-payments"]` into the vector metadata at ingest time. Six weeks later the document was moved to a restricted space. Your index does not know. The architectural fix I insist on: **do not store the ACL, store the resource identity, and resolve permissions at query time against the source of truth.** Store `source_system`, `space_id`, `object_id`; at query time, expand the user's group memberships (cached in Redis, TTL measured in minutes, invalidated on membership change events) into a filter predicate. You have moved a consistency problem into a cache-invalidation problem, which is a problem your backend instincts already know how to run.

**Mechanism 4: leakage through derived artifacts.** This is the one people miss. Your pipeline generates document summaries, a knowledge graph, community summaries, propositions, or an LLM-generated "context prefix" per chunk. Those derived objects are built from restricted content and are frequently stored in a *different* index with *no* ACL propagation. A GraphRAG community summary that says "the Acme acquisition is projected to close in Q3" is a leak even though the source document is perfectly protected. **Every derived artifact must inherit the intersection of its sources' ACLs** — and if it derives from documents with disjoint ACLs, the intersection may be empty, which means the artifact is unservable and you should not have built it that way.

**⚠ Trap:** believing the LLM enforces anything. The model sees whatever you put in the context window. If a restricted chunk is in the prompt, it is disclosed, regardless of whether the model quotes it — a well-crafted follow-up will extract it, and in many regimes putting it in the prompt is itself the disclosure event.

**🗣 Say this in the room:** "Access control belongs in the retrieval predicate, pre-filter, resolved at query time against the identity provider — never stored as a snapshot in vector metadata and never enforced in the prompt. And every derived artifact — summaries, graph nodes, chunk context prefixes — inherits the ACL of everything it was built from, or you have laundered restricted content into an unrestricted index."

### How does content-hash deduplication silently lose data?

The mental model: **a hash is a claim of identity, and every dedup bug is a mismatch between what you hashed and what you meant by "the same document."**

The lossy direction. You hash normalized text and treat equal hashes as the same document, keeping one. Two failure shapes. **Boilerplate collision:** a corpus of 4,000 contract amendments where 200 of them consist solely of a standard clause block, differing only in the party name that your normalizer stripped as a header. You keep one, and 199 real documents vanish from the index with no error. **Truncation collision:** you hash the first 4 KB for speed, and every document generated from the same template collides on its header.

The bloating direction, which is more common. You hash the raw bytes, and the raw bytes include an export timestamp, a page footer with "printed on 2026-08-01," a PDF `/CreationDate`, or a signed URL in an embedded link. Now re-ingesting the identical document produces a new hash, a new document ID, a new set of chunks. Six months of nightly syncs later you have 180 copies of your onboarding guide. They dominate top-k for any onboarding query because they are all maximally relevant, your effective `k` collapses to 1 unique document, and your context window is full of the same paragraph. This one presents as a *retrieval quality* problem and gets misdiagnosed as one.

The correct construction is two identifiers doing two jobs:

- **A stable document identity**, derived from source coordinates, never from content: `sha256(f"{source_system}:{space}:{object_id}")`. This is the primary key. Re-ingesting the same source object always hits the same row, which is what makes ingestion idempotent and makes deletion and update *possible at all*.
- **A content version hash**, computed over *canonicalized* content — text after normalizing whitespace, stripping headers/footers, and removing volatile fields — used only to answer "did this change since last time," i.e. to skip re-embedding.

```python
def doc_id(source: str, space: str, object_id: str) -> str:
    return hashlib.sha256(f"{source}:{space}:{object_id}".encode()).hexdigest()

def content_version(text: str) -> str:
    canon = re.sub(r"\s+", " ", strip_headers_footers(text)).strip().lower()
    return hashlib.sha256(canon.encode()).hexdigest()
```

Then the ingest decision is a three-way branch: unknown `doc_id` → insert; known `doc_id` with new `content_version` → re-chunk, re-embed, replace chunks transactionally; known `doc_id`, same `content_version` → touch `last_seen_at` and skip. And a fourth path everyone forgets: `doc_id`s present in the previous sync and absent from this one → tombstone and delete their vectors.

**💰 Math:** the bloat case is real money. 180 duplicate copies of a 40-page guide at ~30k tokens each is 5.4M tokens re-embedded per full reindex. At $0.10/Mtok for a good embedding model that is $0.54 per reindex — trivial. But the *storage* is 180 × ~60 chunks × 1,024 dims × 4 bytes = 44 MB of vectors for one document, and multiplied across a corpus it is what turns a $200/month index into a $3,000/month index. The quality cost is larger than either: your top-5 contains one distinct document instead of five, so your effective recall@5 is recall@1.

### Your RAG system started returning stale answers right after a reindex. Debug it.

The phrase "right after a reindex" is the whole clue: this is a deploy problem, not a retrieval problem, and the debugging is closer to a bad database migration than to anything ML.

Six candidate causes, ordered by how often I have seen them:

**One: you are serving the old index.** Your alias or collection pointer did not actually swap, or swapped and got rolled back, or half your replicas swapped. Check: query the store for its build timestamp / a canary document you insert with each build and version-stamp. If the canary comes back with the previous build ID, stop debugging retrieval.

**Two: the reindex was partial and completed "successfully."** A batch job that processes 400k documents, hits a rate limit at document 260k, retries three times, gives up, marks the shard done, and moves on. The new index has 65% of the corpus. Every metric is fine because your metrics are on the queries that worked. Check: **document count and chunk count in the new index versus the old, before the swap, as a hard gate.** My rule: a new index whose document count is more than 2% below the previous index does not get promoted without a human overriding, full stop.

**Three: deletions were never applied.** You rebuilt from a source snapshot that still contains soft-deleted rows, so retired documents came back. Classic when the source system uses a `deleted_at` column and the extractor's query forgot the predicate.

**Four: the embedding model changed and the query path did not.** You reindexed with `embed-v3` and the query service still calls `embed-v2`. Dimensions might even match, so nothing errors — you just get garbage similarity, which presents as "answers got weird" more than "stale," but it belongs on the list because it is catastrophic and silent. Guard: store the embedding model ID and dimension in the index metadata, and have the query service assert equality at startup and refuse to serve on mismatch.

**Five: caching.** A semantic cache or a plain response cache keyed on the query string is serving pre-reindex answers. **Every index build must bump a cache-namespace version** so the entire cache is logically invalidated by the swap. If you do not do this, your cache is a stale-answer machine with a TTL measured in whatever you set it to.

**Six: the reindex actually regressed retrieval quality**, and "stale" is the user's word for "it gave me an old fact that is still in the corpus and ranked it above the new one." Check whether the current-version chunk is retrievable at all, and whether your recency/status filter survived the rebuild. Filters are metadata, and metadata is exactly what a rewritten ingestion path drops.

**🔍 Failure taxonomy — the decision procedure:** query the canary doc → if wrong build ID, it is the swap. Else compare doc/chunk counts old vs new → if short, it is a partial build. Else check whether the *new* fact is retrievable at k=200 → if not, ingestion missed the update. Else check whether the *old* fact should exist → if the old doc is still in the index, it is a deletion/tombstone bug. Else check the query-path embedding model ID → mismatch is catastrophic. Else check cache namespace. Six checks, none of which requires an eval run, all of which are SQL or a metadata read.

**⚠ Trap:** "blue/green means I am safe." Blue/green protects you from *broken* indexes, not *incomplete* ones, because an incomplete index passes every smoke test — it answers the ten queries in your health check perfectly. The gate that catches incompleteness is a count comparison plus a golden-set recall run against the *new* index *before* the alias moves. That is what makes the swap safe, not the swap mechanism itself.

### Is it retrieval or is it generation? Give me the single experiment that decides.

**The oracle-context ablation**, and I want you to internalize it as the cheapest high-information experiment in the discipline.

Take your golden set: N queries, each with a known-correct answer and a known set of golden chunk IDs. Run two conditions.

**Condition A — production:** the real retriever picks the context. Measure end-to-end answer correctness.
**Condition B — oracle:** you inject the golden chunks directly as context, bypassing retrieval entirely. Same prompt, same model, same everything else. Measure end-to-end answer correctness.

Condition B is your **generation ceiling**: the accuracy your system would have if retrieval were perfect. The gap between A and B is exactly the money on the table from improving retrieval. The gap between B and 1.0 is exactly the money on the table from improving generation — prompt, model, format, reasoning.

Interpretation is mechanical:

- **A = 0.62, B = 0.91.** Retrieval is the bottleneck. 29 points are recoverable with better retrieval and you should not touch the prompt. Go run the hybrid/rerank/chunk-size ablation.
- **A = 0.62, B = 0.66.** Retrieval is *not* the bottleneck. Even with perfect context the system is bad. Your problem is generation or your golden answers are wrong or your questions are unanswerable from the corpus. Prompt engineering is now the correct activity, which it almost never is.
- **A = 0.62, B = 0.63, and B is low.** Suspect the golden set itself. Before rewriting anything, hand-inspect 20 items — I find label errors in roughly 10–20% of first-draft golden sets, usually questions whose "correct" answer is not actually derivable from the cited chunks.

A third condition is worth adding once you have the harness: **Condition C — oracle plus distractors.** Golden chunks *plus* the four highest-ranked wrong chunks from production retrieval. C tells you how robust generation is to the noise retrieval actually produces. If B = 0.91 and C = 0.71, your generator is fragile to distractors, and the correct fix is *precision* (fewer, better chunks) rather than recall. That distinction changes your entire roadmap: one path says "raise k," the other says "lower k and add a reranker."

**💰 Math:** for a 200-query golden set, three conditions is 600 generation calls. At ~5k input / 300 output tokens, $3/Mtok in and $15/Mtok out: 600 × (5000 × 3/1e6 + 300 × 15/1e6) = 600 × 0.0195 = **$11.70 per full ablation run.** 📅 Volatile pricing. Twelve dollars to know whether your next month goes into retrieval or generation. There is no defensible reason to guess.

**🗣 Say this in the room:** "Before I optimize anything I run the oracle-context ablation — inject the known-correct chunks and measure. That gives me the generation ceiling, and the gap to production tells me the exact value of retrieval work. Teams that skip it spend quarters on prompt engineering against a sixty-percent retrieval ceiling."

### What has to be in your traces for that five-gate ladder to take ten minutes instead of two days?

If the trace does not contain it, you cannot diagnose it, and you will instead re-run the query by hand and hope it reproduces — which for a system with a nondeterministic generator and a possibly-changed index it often will not. So the trace schema is the actual deliverable here.

Per request, one span tree, with these fields on the spans that matter:

**Request span:** `request_id`, `user_id`, `tenant_id`, `session_id`, raw query text, resolved/rewritten query text, intent classification if you route, model IDs (generator, embedder, reranker) *with versions*, index alias and resolved build ID, total tokens in/out, total cost, end-to-end latency, and the final answer text.

**Retrieval span:** the embedded query's model+dimension, the filter predicate as executed (not as intended — as executed, so ACL and metadata filter bugs are visible), `k` requested, `k` returned, and **the full candidate list as `(chunk_id, score, source)` triples** where source is `dense` / `lexical` / `fused`. Scores, not just IDs. The score distribution is a diagnostic in its own right.

**Each post-processing span** (rerank, dedup, MMR, compression, budget truncation): input chunk ID list, output chunk ID list, and the rule that dropped each removed item. This is what makes failure point 3 visible.

**Prompt-assembly span:** the ordered chunk IDs as they appear in the prompt, their character offsets within the assembled prompt, the token count of each section, and a hash of the system prompt (so you can correlate a quality shift with a prompt deploy — I have caught more regressions this way than with any eval).

**Generation span:** input token count split into cached vs uncached prefix, output tokens, finish reason, and the citation spans the model emitted.

**Feedback span, written later, joined on `request_id`:** thumbs, edit-in-place, escalation to human, downstream conversion.

Two design rules that matter more than the field list. **First: store chunk IDs, not chunk text, in the trace, and keep chunk text immutable and versioned.** If chunks are mutable, your six-week-old trace is a lie. My chunk primary key is `(doc_id, content_version, chunk_index)` so a chunk is never rewritten in place. **Second: sample the heavy fields, log the light ones always.** Full candidate lists with scores at 100% of traffic will dominate your observability bill; I log IDs+scores for the top 20 always, the full 200 at 1–5% sampling, and 100% of any request that got negative feedback or tripped a guardrail — sampled on the *outcome*, which requires deferring the sampling decision to the end of the request rather than the beginning.

**💰 Math:** a trace with 20 candidates × (16-byte id + 4-byte score + overhead) plus prompt metadata is roughly 4–6 KB serialized. At 200k requests/day that is 200,000 × 5 KB = **1 GB/day, ~30 GB/month.** At a typical managed-observability ingest price in the $0.30–$2.00/GB range that is $9–$60/month for the light tier — negligible. The full 200-candidate variant at 100% would be ~40 KB/request → 8 GB/day → 240 GB/month, which is where the bill becomes an argument. Hence sampling. 📅 Volatile: observability pricing varies by an order of magnitude across vendors; price yours.
