### Our top 5 chunks are five copies of the same paragraph from five versions of the same document. Fix it.

This is the diversity-collapse failure, and the fix has two independent halves that people conflate: **detect near-duplicates, and then decide what to do with them.**

**Detection.** You need a similarity notion that is cheap and robust to trivial edits. My ladder:

1. **Exact content hash** at ingestion — SHA-256 of the normalized chunk text (lowercase, collapse whitespace, strip boilerplate). Catches literal republication, which in enterprise corpora is a shocking fraction: the same policy paragraph in the handbook, the onboarding deck and three Confluence pages. Free, do it unconditionally.
2. **MinHash / SimHash over shingles** for near-duplicates with edits. Shingle the chunk into overlapping 5-grams, MinHash to a 128-permutation signature, band into an LSH index. Jaccard ≥ 0.8 on 5-gram shingles is a good "these are the same text with minor edits" threshold. Cost is microseconds per chunk and it runs at ingestion.
3. **Embedding cosine** between candidate chunks at query time. Cheap because you already have all the vectors for your ~50 candidates. Cosine ≥ 0.95 on a normalized modern embedder is roughly "paraphrase or near-copy." Note this catches semantic duplicates that MinHash misses (same fact, different wording) and misses lexical duplicates that changed enough words — the two methods are complementary.

**What to do with them.** Not simply drop. The right move is **collapse and attribute**: keep the highest-reranked member of the duplicate cluster as the context chunk, but carry all cluster members' document ids as citations. This is important for two reasons. First, the user asking "where does this policy come from" wants all five sources, and a UI that shows one is lying by omission. Second, the *number* of independent sources asserting a fact is a real relevance signal in enterprise search — five documents saying the same thing is evidence of authority — and you should surface it rather than destroy it.

**⚠ Trap:** deduplicating at *ingestion* by dropping duplicate chunks entirely. It seems clean and it breaks provenance: the surviving chunk now belongs to whichever document happened to be indexed first, so a user searching within the scope of document B gets a citation to document A, or gets nothing at all if ACLs differ. In a permissioned corpus this is a security bug, not just a UX one — the user may not have access to the document you kept. **Store all copies, cluster them, and collapse at query time inside the user's permission scope.** I enforce this in review.

**💰 Math:** the payoff is direct token spend. If your 5-chunk context is really 1 chunk repeated 5 times at 400 tokens each, you are spending 2,000 input tokens to convey 400 tokens of information — 1,600 tokens of pure waste. At $3/Mtok and 200k queries/day: `1,600 × 200,000 = 3.2e8 tokens/day × $3/1e6 = $960/day = $28,800/month` burned on redundancy, *plus* the answer-quality loss from the four facts that got displaced.

### Derive MMR and implement it. How do you choose lambda?

Maximal Marginal Relevance is the formalization of "a good result set is not the same thing as a set of good results." A reranker scores each document against the query in isolation; MMR is the greedy set-selection algorithm that adds a penalty for redundancy with what you have already chosen.

```
MMR = argmax_{d ∈ R∖S} [ λ · sim(d, q) − (1 − λ) · max_{s ∈ S} sim(d, s) ]
```

`R` is the candidate pool, `S` is what you have selected so far. You build `S` greedily, one document at a time. `λ = 1` is pure relevance (MMR is a no-op). `λ = 0` is pure diversity, which will happily select five irrelevant-but-mutually-different documents. The `max` over `S` rather than a mean matters: you penalize a candidate for being similar to *any* already-selected document, not for being similar to the set on average — a candidate that duplicates one selected doc but differs from four others should still be penalized hard.

```python
import numpy as np

def mmr(query_vec, doc_vecs, rel_scores, k=5, lam=0.7):
    """query_vec: [d]; doc_vecs: [N, d] L2-normalized; rel_scores: [N] reranker scores."""
    # put relevance on the same 0-1 scale as cosine so lambda is interpretable
    r = np.asarray(rel_scores, dtype=float)
    r = (r - r.min()) / ((r.max() - r.min()) or 1.0)
    sim_dd = doc_vecs @ doc_vecs.T                       # [N, N] pairwise cosine
    selected: list[int] = []
    remaining = set(range(len(r)))
    while len(selected) < k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: r[i])     # seed with the top hit
        else:
            best = max(remaining, key=lambda i:
                       lam * r[i] - (1 - lam) * sim_dd[i, selected].max())
        selected.append(best)
        remaining.discard(best)
    return selected
```

**The rescaling line is the one people omit and it is what makes `λ` meaningless without it.** Reranker logits live on an unbounded scale; cosine similarity lives in roughly `[0, 1]` for a normalized embedder. If you subtract one from the other without normalizing, `λ = 0.7` does not mean "70% relevance" — the relevance term dominates by two orders of magnitude and MMR silently degenerates to pure reranking. I have reviewed this bug several times and it always presents as "we added MMR and nothing changed."

**Choosing λ:** start at 0.7 and tune it against an *answer-quality* metric, not a retrieval metric. This is the crucial point. nDCG will monotonically decrease as you lower λ, because nDCG rewards putting relevant documents high and MMR deliberately demotes some relevant documents. **If you tune λ on nDCG you will always pick λ = 1 and conclude MMR is useless.** You must evaluate on end-to-end answer correctness or on a coverage metric — "what fraction of the distinct facts needed to answer this question appear in the context" — which is the metric MMR is actually optimizing.

**📄 Paper:** Carbonell & Goldstein (1998), *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*. The contribution was naming the relevance-versus-redundancy trade-off as a single tunable objective; it replaced the implicit assumption that a ranked list should be sorted by relevance alone.

**⚠ Trap:** running MMR on multi-hop questions with a low λ and congratulating yourself. MMR does not know *which* diversity matters. For "compare our 2024 and 2025 refund policies" you need one chunk from each year, and MMR will happily give you diversity along an axis you don't care about — one chunk from the refund policy and one from the shipping policy. Structured decomposition (retrieve per sub-question, then union) beats MMR whenever the required diversity is known in advance. MMR is for the case where you *don't* know what dimension the answer needs.

### You have 8 reranked chunks and a system prompt. What order do you put them in, and why?

Two forces pull in opposite directions here and you must name both.

**Force one: lost-in-the-middle.** Liu et al. (2023) documented a U-shaped accuracy curve — models retrieve information reliably from the beginning and the end of a long context and measurably worse from the middle. On their multi-document QA setup, moving the relevant document from position 1 to the middle of a 20-document context dropped accuracy substantially, sometimes below what the model achieved with *no* documents at all. **📅 Volatile:** the magnitude of this effect has shrunk considerably in frontier models with better long-context training, and reasoning models handle it better still. Verify against your own model before designing around it — but do not assume it is gone, because it reappears as contexts get longer and it reappears in cheaper models.

**Force two: prefix-cache stability.** Anything you place before a variable region invalidates nothing; anything you place *after* variable content cannot be cached. If your system prompt is 3,000 tokens and you put retrieved chunks in front of it, you have destroyed prefix caching on every query. If you put them after, the system prompt is cached on every request.

The layout I default to, and would draw on a whiteboard:

```
[ system prompt + tool defs + few-shots ]      ← fully static, prefix-cached
[ retrieved chunks, ordered worst → best ]     ← variable
[ the user's question, restated ]              ← last, adjacent to the best chunk
```

Static content first, always, for caching. Then chunks in **ascending relevance order**, so the highest-scored chunk sits immediately before the question — occupying the "end" position of the U-curve, adjacent to the instruction it is meant to answer. This gives you the recency-position advantage without giving up caching, since the chunks are variable either way.

The main alternative is descending order (best first). It is defensible: it puts the best chunk at the "beginning" position, and for very short contexts (3–5 chunks, under ~4k tokens) the U-curve is basically flat and ordering barely matters. **This is genuinely contested and the correct interview answer is to say so and give the decision rule**: A/B it on your model with your context length, default to best-last for long contexts and don't spend time on it for short ones.

**⚠ Trap:** shuffling chunk order per request to "avoid position bias." It defeats prefix caching, makes results non-reproducible, and makes every incident un-debuggable because you cannot replay the exact context. Order deterministically. If you want to measure position sensitivity, do it in an offline eval, not in production.

**💰 Math on the caching half:** a 3,000-token system prompt at $3/Mtok uncached costs `3,000 × $3/1e6 = $0.009` per call. With a 90% cache discount on cached input it is `$0.0009`. At 200k calls/day: uncached `$1,800/day`, cached `$180/day` — **$48,600/month saved by ordering static content first.** That is a bigger number than most model-selection decisions, and it is purely a layout choice.

### How do you add a recency preference without wrecking relevance?

The mental model: **recency is a prior, not evidence.** It should shift the ranking when relevance is close, and it should be unable to promote an irrelevant document no matter how new it is. Any implementation that violates that second property is wrong.

Which rules out the naive version immediately: `final = relevance + w · recency_score`. An additive boost can promote a document with near-zero relevance if it is new enough. I have seen this ship and produce the classic symptom — "our search returns today's meeting notes for every query."

The forms I use instead:

**Multiplicative decay on the relevance score.** `final = relevance × exp(−age_days / τ)`. Because it is multiplicative, an irrelevant document (relevance ≈ 0) stays at ≈ 0 regardless of age. `τ` is the half-life-ish constant: `τ = 90` means a 90-day-old document keeps `e^-1 = 37%` of its score. Choose `τ` from the actual staleness rate of your corpus — for a fast-moving engineering wiki `τ = 60`; for legal contracts, recency should probably not decay at all.

**A hard freshness filter as a separate retrieval leg.** Rather than blending, run a second retrieval restricted to the last 30 days and fuse it in with RRF at a modest weight. This guarantees fresh content gets *considered* without letting it dominate, and it is much easier to ablate than a continuous decay.

**Version-awareness instead of recency.** Often what people want is not "newer" but "current" — the superseded version of a document should be excluded, not down-weighted. That is a metadata filter (`is_current = true`), it is exact, and it does not need tuning. **Ask whether the requirement is really recency or really currency**; nine times out of ten in enterprise search it is currency, and a boolean field beats a decay curve.

**⚠ Trap:** tuning a recency boost against an eval set built six months ago. Every document in that set is old, so the decay term is nearly constant across all candidates and the boost appears to do nothing. You then set `τ` by intuition and ship a parameter no experiment ever validated. **A recency parameter can only be tuned on an eval set with a realistic age distribution, including documents created after the queries were written** — which usually means you must rebuild the eval set, not just reuse it.

### Sales wants documents from our "premium" content set boosted. Product wants recently-updated docs boosted. Legal wants deprecated docs suppressed. How do you handle business-rule boosts?

This is the question that separates people who have shipped search from people who have built a demo, because the technical answer is easy and the organizational answer is what actually matters.

**Architecturally**, all three are the same object: a multiplicative modifier applied after reranking and before final selection. I implement them as a small, ordered, declarative ruleset — not as scattered `if` statements in the retrieval code:

```python
BOOSTS = [
    Rule(name="deprecated_suppress", when=lambda d: d.meta.get("deprecated"), factor=0.05),
    Rule(name="premium_content",     when=lambda d: d.meta.get("tier") == "premium", factor=1.20),
    Rule(name="recency_decay",       factor=lambda d: math.exp(-d.age_days / 90)),
]
```

Multiplicative, declarative, named, and each one **individually toggleable at runtime**. That last property is the whole design: you cannot ablate what you cannot turn off.

**The eval risk is the real content of this question, and I would raise it unprompted:**

**Boosts are unfalsifiable by default.** A relevance change can be measured against labeled relevance. A "premium content" boost cannot, because your labels say what is *relevant*, not what is commercially desirable. So there is no metric that says the boost is wrong, and it accumulates forever. Six quarters later you have fourteen boosts, three of them contradictory, and nobody remembers who asked for the fifth one.

**The governance I enforce:**
1. **Every boost has a named owner and an expiry date.** Unrenewed boosts are auto-disabled. This alone kills most of the problem.
2. **Every boost ships with a measurement of its relevance cost.** Run the eval with and without; report the nDCG delta. "This boost costs 1.8 nDCG points" is the sentence that makes the trade-off a business decision rather than an engineering accident. Sales can absolutely have their boost — they just have to know it costs 1.8 points.
3. **Boost magnitudes are bounded.** I cap them at ~1.5× and ~0.1×. An unbounded boost is a rewrite of the ranking function by someone who is not looking at the ranking function.
4. **Suppression rules are filters, not boosts.** Deprecated content should be *excluded* via a metadata filter, not down-weighted to 0.05. A boost of 0.05 still surfaces the document when nothing else matches, which is exactly when a deprecated doc is most dangerous — the user gets an obsolete answer with no competing signal.
5. **Boosts are logged per result.** The response payload carries which rules fired and their factors. Without this, debugging "why is this document ranked third" is impossible.

**🗣 Say this in the room:** "Business boosts are fine, and I ship them as named, owned, expiring, bounded multiplicative rules with per-result logging. What I insist on is that every boost's relevance cost is measured on the eval set and reported to its requester — 'this boost costs 1.8 nDCG points' turns an unfalsifiable request into a decision someone owns. And suppression is a filter, never a small boost, because a 0.05 multiplier still surfaces a deprecated doc when nothing else matches."

### Design the retrieval layer for an AI coding assistant working in a large monorepo.

I'll assume a 5M-line monorepo, a chat-and-inline-edit product, and a hard TTFT budget of about 800 ms because the user is waiting mid-keystroke.

**The first design decision: code retrieval is much more lexical than prose retrieval, and I would weight accordingly.** Developers query with identifiers — `getUserPermissions`, `ERR_POOL_TIMEOUT`, `useDeferredValue`. Those are exactly the high-IDF literal tokens dense retrieval loses. My starting hybrid weight for code is roughly 60/40 in favor of lexical, versus 30/70 for prose, and I would be prepared to defend that from the query-log distribution.

**The analyzer is where most of the quality is.** A prose tokenizer destroys code. You need a code-aware analyzer that: splits `camelCase` and `snake_case` into subwords *while also keeping the original token* (so `getUserPermissions` indexes as `getUserPermissions`, `get`, `user`, `permissions`); does not stem (stemming `Users` to `user` merges a class name with a variable); preserves `::`, `->`, `.` as meaningful; and keeps punctuation-heavy operator sequences. Getting this wrong is the single largest quality lever in code search and it is invisible in any generic benchmark.

**Four indexes, fused:**
1. **Symbol index** — a structural index of definitions built from tree-sitter parses: `(symbol_name, kind, file, line, signature, docstring)`. Exact-match lookup on symbol name is not really "retrieval," it is a hash lookup, and it should short-circuit the whole pipeline when the query is exactly a symbol that exists. This handles a large fraction of real queries at ~1 ms.
2. **Chunk index (hybrid)** — AST-aware chunks at function/class granularity, with the enclosing file path, imports, and class name prepended as context. Both BM25 with the code analyzer and a code-specialized embedder.
3. **Reference/call-graph edges** — not a retrieval index but an expansion step: once you have a relevant function, pull its callers and callees. This is how you answer "why is this failing" questions that need the call site, not just the definition.
4. **Recent-edit and open-file context**, which is not retrieved at all but injected — the file the user is looking at is overwhelmingly the most relevant context and should never have to win a similarity contest.

**Fusion and reranking:** RRF over symbol/BM25/dense with symbol matches weighted heavily, then a cross-encoder rerank over 50 candidates truncated to 256 tokens (~115 ms per the earlier arithmetic), then dedup — critical in a monorepo where the same utility function exists in four vendored copies.

**The operational requirement that dominates the design: incremental reindex per commit.** A monorepo takes hours to fully index and receives hundreds of commits a day. So the index must be keyed by content hash per file, reindexing only changed files on the post-commit hook, with the symbol index updated transactionally alongside. A retrieval system that is 40 minutes stale in a coding assistant returns code the user just deleted, and that is a trust-destroying failure — worse than returning nothing.

**⚠ Trap:** embedding whole files. A 2,000-line file produces one vector that means nothing; every query matches it weakly and nothing matches it strongly. Chunk at function granularity with the file path and enclosing class prepended, so the chunk is self-describing. Also: index the *docstrings and comments* as a separate field with prose analysis, because natural-language queries ("how do we handle refunds") match comments, not identifiers.

### Design enterprise search over Slack, Google Drive, Jira and Confluence with per-user permissions.

The Glean-shaped problem. I want to lead with the constraint that dominates everything else, because leading with the retrieval architecture is the junior answer.

**Permissions are not a filter you add at the end; they are the primary key of the whole system.** Every result must be authorized for the specific requesting user at the moment of the request. Two failure modes, and they have wildly different severity: showing a document the user cannot see is a **security incident** that can end an enterprise contract; hiding a document they can see is an annoyance. Design asymmetrically.

**The architecture:**

**Ingestion** pulls from each source with a service account that can see everything, and — crucially — pulls the **ACL alongside the content**, normalized into a common representation: a set of principals (user ids, group ids, "anyone in domain", "public link") that grant read. Store the ACL on the chunk, denormalized. Group membership is resolved at query time from a separate, freshly-synced identity store, because group membership changes far more often than document ACLs and you cannot afford to reindex every document when someone leaves a team.

**Retrieval must pre-filter, not post-filter.** This is the technically load-bearing point. If you retrieve top-100 by relevance and then drop unauthorized results, a user with narrow access gets an empty result set for queries where thousands of relevant documents exist — because all 100 belonged to someone else. The recall collapse is severe and it scales with how selective the filter is. You need filtered search that pushes the ACL predicate *into* the index traversal: a filtered-HNSW implementation on the dense side, and a boolean `AND` on an ACL field on the lexical side, which inverted indexes do natively and well.

**Per-source hybrid, then weighted cross-source RRF**, exactly as in the multi-index answer. Each source needs its own analyzer and chunking: Slack messages are 20 tokens and need thread-level aggregation to be meaningful at all; Drive docs are long and need structural chunking; Jira tickets have high-signal structured fields (status, assignee, labels) that belong in metadata filters, not in the embedded text.

**Reranking with a source-aware signal.** A cross-encoder over ~50 fused candidates, plus a small set of authority signals: for Slack, message reactions and thread length; for Drive, the number of distinct viewers and recency of edit; for Confluence, page view counts. **These are the enterprise equivalent of PageRank and they matter a great deal**, because in a corpus where twenty documents answer a query, the one everyone actually reads is the right answer.

**🔍 Failure taxonomy for permissioned enterprise search:**
1. **User sees a document they shouldn't.** Highest severity. Causes: stale ACL after a permission revoke; ACL cached in a query cache without the ACL scope in the cache key; a shared reranker or embedding cache keyed only on content. Mitigation: short ACL TTL, ACL scope in every cache key, and a post-retrieval authorization re-check against the live source for anything you are about to cite.
2. **User sees nothing though relevant docs exist.** Post-filtering instead of pre-filtering. Diagnose by logging candidates-before-filter and after.
3. **Deleted-upstream document still retrievable.** Your connector's deletion webhook failed or the source doesn't emit one. This is the one that gets quoted to a customer. Mitigation: a periodic full reconciliation pass that lists source ids and tombstones anything missing, plus an existence check at citation time.
4. **One source dominates.** Slack has 100× the document count of Confluence, so it floods every candidate pool. Fix with per-source retrieval depth caps and per-source RRF weights derived from your golden set's source distribution, not from corpus size.
5. **Cross-tenant leakage in a multi-tenant deployment.** Shared index, filter-based isolation, and one query path that forgets the tenant predicate. The mitigation is architectural: tenant id in the index namespace, not just in a filter, so a forgotten predicate returns *nothing* rather than *everything*. **Fail closed.**

### Give me the end-to-end cost model for this retrieval pipeline at 1M queries per day.

Let me build it line by line, stating assumptions, because the assumptions are where the interview actually happens.

Assumptions: 1M queries/day, 10M chunks at ~400 tokens, 15-token average query, hybrid retrieval with 100 candidates per retriever, cross-encoder rerank of 50 at 256 tokens, 5 chunks of 400 tokens into the LLM context plus a 2,000-token system prompt, 400 output tokens.

**Query embedding.** `1e6 × 15 tokens = 1.5e7 tokens/day`. At $0.10/Mtok: `15 × $0.10 = $1.50/day = $45/month`. Negligible; stop optimizing it.

**Lexical + ANN retrieval.** This is your own infrastructure, not per-query billing. Sizing: 11.6 QPS average, ~35 QPS peak. A Postgres/OpenSearch node pair plus a vector index of 10M × 1024-dim fp32 = 40 GB, plus HNSW graph overhead at roughly `M × 2 × 4 bytes/vector` — at `M=16` that is `16 × 2 × 4 = 128 bytes × 10e6 = 1.3 GB`. So ~42 GB, comfortably 2–3 memory-optimized nodes for redundancy. Call it **$800–1,500/month** (📅 Volatile — instance pricing).

**Reranking.** From the earlier arithmetic: 7 A10G-class GPUs at ~$1.00/hr = `7 × 730 = $5,110/month`. With a 30% query cache hit rate, ~5 GPUs = **$3,650/month**.

**LLM generation — and this is where the whole model tips.** Input per query: `2,000 (system) + 5 × 400 (chunks) + 50 (query) = 4,050 tokens`. Output: 400 tokens. At $3/Mtok input and $15/Mtok output (📅 Volatile — a mid-tier frontier price point; verify):

- Input: `4,050 × 1e6 = 4.05e9 tokens/day × $3/1e6 = $12,150/day`
- Output: `400 × 1e6 = 4e8 tokens/day × $15/1e6 = $6,000/day`
- **Total: $18,150/day = $544,500/month.**

With prefix caching on the 2,000-token system prompt at a 90% cached-input discount: cached portion drops from `2,000 × $3/1e6 = $0.006` to `$0.0006` per query, saving `$0.0054 × 1e6 = $5,400/day = $162,000/month`.

**The summary table, which is the actual deliverable:**

| Component | $/month | % of total |
|---|---|---|
| Query embedding | 45 | 0.01% |
| Retrieval infra | 1,200 | 0.3% |
| Reranking (self-hosted) | 3,650 | 1.0% |
| LLM generation (cached prefix) | 382,500 | 98.7% |

**🗣 Say this in the room:** "At a million queries a day the retrieval layer is about 1% of spend and the LLM is 99%. Which means every retrieval decision should be optimized for *quality*, not cost — and the highest-leverage cost work is reducing how many tokens retrieval puts into the prompt. Going from 8 chunks to 5 through better reranking and dedup cuts 1,200 input tokens per query, which is `1,200 × 1e6 × $3/1e6 = $3,600/day, $108,000/month` — thirty times the entire reranking bill. A better reranker pays for itself many times over purely through context reduction, before you count any quality gain."

That reframe — **the reranker is a cost-reduction device, not just a quality device** — is the single most valuable thing to say in a retrieval cost discussion, and almost nobody says it.

### Which retrieval metric do you actually report, and why not the others?

The framing that gets this right: **pick the metric that measures the job the stage is doing, and use different metrics for different stages.** Reporting one number for the whole pipeline is how teams end up optimizing the wrong thing.

**For the first stage (retrieval into the candidate pool): recall@k, where k is exactly the reranker's input size.** Nothing else matters. Ordering within the candidate pool is irrelevant because the reranker is about to destroy it. If you retrieve 50 for the reranker, report recall@50 and ignore precision entirely. This is the metric I would put on the dashboard.

**For the final stage (what goes into the LLM): nDCG@k, with k equal to your context slot count.** nDCG is the right choice because it is graded — it distinguishes "perfect answer" from "related background" — and position-discounted, which matches the lost-in-the-middle reality. Its weakness is that it needs graded labels, which are more expensive to produce than binary ones.

**MRR (mean reciprocal rank of the first relevant result)** is right only when there is exactly one correct answer and the user stops at it — a "find the ticket about X" navigational workflow. For RAG, where the LLM synthesizes from several chunks, MRR is actively misleading: it gives full credit for finding one of five needed chunks first.

**Precision@k** matters more than people give it credit for in RAG specifically, because irrelevant chunks in context are not neutral — they cost tokens and they measurably distract the model. But it is usually dominated by nDCG.

**The metric nobody reports and everybody needs: end-to-end answer correctness with retrieval held as the only variable.** All retrieval metrics are proxies. The only question that matters is whether the answer got better. Run the same eval questions through the full pipeline with retrieval config A and config B, judge the answers, and report the delta. Retrieval metrics are for fast iteration; answer metrics are for the ship decision.

**⚠ Trap:** reporting a mean over an eval set that is 90% easy queries. Both configurations get the easy ones right, the mean moves by 0.4 points, and you conclude nothing. **Stratify.** Report the metric separately for identifier queries, natural-language questions, multi-hop questions and long-tail queries. A change that moves the aggregate by 0.4 points but moves identifier queries by 25 points is a large win being hidden by an unrepresentative denominator.

### Search quality dropped noticeably over three weeks. There were no deploys and no config changes. Where do you look?

No-deploy quality regressions are the ones that separate people who have operated a search system from people who have built one, and the answer is that **the corpus is a moving input and your ranking function is a function of it.**

My order of investigation:

**1. Corpus composition shift and IDF drift.** BM25's IDF depends on `N` and `df(t)` across the whole corpus. If a bulk import added 2M auto-generated documents — release notes, log dumps, migrated tickets — every term's document frequency changed. Terms that were rare and discriminative are now common and worthless. **The ranking function you tuned no longer exists.** Diagnostic: snapshot the IDF of your top 200 query terms weekly and diff. A 30% shift in the IDF of your most common query terms is your answer.

**2. `avgdl` shift.** Same mechanism through the `b` term. If the new documents are much shorter or longer than the old ones, `avgdl` moved and every document's length normalization changed. A corpus whose `avgdl` went from 400 to 250 now penalizes all your original 400-token chunks as "long."

**3. Index bloat from deletes.** In HNSW, deletes are tombstones — the graph node stays and continues to be traversed, it is just filtered from results. Accumulate enough and each search wastes traversal budget on dead nodes, so effective `efSearch` drops and recall degrades silently. Symptom: gradual recall decline correlated with delete volume, fixed by a rebuild. Nobody schedules that rebuild until it bites them once.

**4. Embedding staleness on a partially-reindexed corpus.** If someone started a reindex with a new embedder version and it stalled at 60%, you now have two incompatible geometries in one index. Similarity between a v1 query vector and a v2 document vector is meaningless. Diagnostic: histogram the top-1 cosine similarity over time. Mixed-version indexes produce a bimodal distribution — that plot is diagnostic on sight.

**5. Query distribution shift.** Nothing broke; users started asking different questions. A product launch, a new customer segment, a seasonal shift. Your system is as good as it ever was on the old distribution and was never tuned for the new one. Diagnostic: compare the query embedding distribution month-over-month, or just read 50 queries from each period. This is very common and very often misdiagnosed as a regression.

**6. Freshness gap in the ingestion pipeline.** Documents created in the last three weeks never got indexed because a connector's token expired. Users ask about recent things, get old answers, and quality "declines." Diagnostic: the alert you should already have — indexed document count and max `created_at` per source, alerting on staleness, not just on job failure. **A connector that silently returns zero documents does not fail; it succeeds at doing nothing.**

**🗣 Say this in the room:** "No-deploy regressions in search are almost always corpus-side. My first four checks are IDF drift from a bulk import, `avgdl` shift changing length normalization, HNSW tombstone accumulation degrading effective recall, and a stalled reindex leaving two embedding versions in one index. And I'd rule out the boring one first: a connector silently ingesting zero new documents for three weeks."

### When should your retrieval system return nothing at all, and how do you implement that?

Abstention is a feature and most RAG systems don't have it, which is why they hallucinate confidently on out-of-scope questions. **The retrieval layer is the cheapest and most reliable place to detect "we don't have this," far more reliable than asking the LLM to say it doesn't know.**

The mechanism has to work around the fact that raw scores are not calibrated. Three signals, and I use them together:

**1. Calibrated reranker score.** Take your cross-encoder, score a few hundred labeled query-document pairs including known-irrelevant ones, fit a logistic regression from logit to P(relevant), and pick the threshold at your target precision. Now `P(relevant) < 0.3` for the top candidate is a meaningful statement. **This must be redone for every reranker model change** — see the score-drift failure above.

**2. Score gap and shape.** A confident retrieval has a top score well separated from the rest. A "we have nothing" retrieval has a flat distribution of mediocre scores — everything matched a little, nothing matched a lot. The ratio `top1 / mean(top10)` is a cheap, model-agnostic confidence proxy that doesn't require calibration.

**3. Lexical coverage.** What fraction of the query's high-IDF terms appear anywhere in the top candidates? If the user asked about `ERR_TLS_CERT_ALTNAME_INVALID` and that exact string appears in none of the top 20 documents, you almost certainly don't have the answer, regardless of how good the cosine similarities look. **This signal specifically catches the dense-retrieval-confidently-wrong case**, which is the one that produces hallucinations, and it costs nothing to compute.

Then the product decision: what does abstention look like? Not a blank page. The good pattern is "I don't have documentation covering this — here are the three closest things I found, and here's how to file a request," which preserves the user's ability to judge and gives you a training signal about corpus gaps. **Log every abstention as a corpus-gap candidate.** The abstention log is the highest-value input to your content roadmap, and shipping it as a weekly report to whoever owns documentation is a disproportionately effective move.

**💰 Math:** abstention also saves money. If 8% of queries are out-of-scope and you abstain before generation, you skip `0.08 × 1e6 = 80,000` LLM calls/day at roughly `4,050 input + 400 output` tokens ≈ `$0.018` per call: `80,000 × $0.018 = $1,440/day = $43,200/month`, while *improving* quality by not generating a confident wrong answer. Quality and cost pointing the same direction is rare; take it when it happens.

### Give me the drill set for this section.

Four drills, unaided, no autocomplete, timed. These are calibrated so that passing all four means you can hold a retrieval design round at any of the companies in scope.

**🏋 Drill 1 — BM25 from memory (25 minutes).** Write the `BM25` class: index construction with document frequencies, `avgdl`, posting lists, and the scoring function with correct smoothed IDF. Then answer without looking: what does `k1 = 0` do, what does `b = 0` do, and why are scores incomparable across queries? *Pass criterion:* the formula is exactly right including the `(k1 + 1)` numerator and the `(1 − b + b·|D|/avgdl)` normalization, and `search` iterates only the posting-list union rather than all documents.

**🏋 Drill 2 — fusion and diversity (30 minutes).** Implement weighted RRF and MMR from scratch, both with correct edge-case handling: missing documents contribute zero to RRF; MMR rescales relevance onto the cosine scale before combining. Then, on paper: a document is rank 1 in dense only, another is rank 15 in both. Which wins at `k = 60`? Show the arithmetic. *Pass criterion:* `1/61 = 0.0164` vs `2/75 = 0.0267`, second wins, and you can state in one sentence why that is the desired behavior.

**🏋 Drill 3 — the latency budget (15 minutes, whiteboard, no calculator).** You have a 1.2 s TTFT budget. Your reranker is a 110M-parameter cross-encoder on one A10G. How many candidates can you rerank at 512 tokens? At 256? Show `2 × params × tokens`, an effective 25 TFLOP/s, and derive the candidate count. Then name three levers to double it. *Pass criterion:* you get ~0.113 TFLOP/pair at 512 tokens, ~4.5 ms/pair, so ~65 candidates fill a 300 ms rerank budget — and 130 at 256 tokens. Levers: truncation, cascade with a small first reranker, batching with length-sorted padding.

**🏋 Drill 4 — the ablation table (45 minutes, with a laptop).** Take any public QA dataset with relevance labels. Build the six-row ablation table from earlier — dense only, BM25 only, hybrid, +rerank, +MMR, +recency — reporting recall@50, nDCG@10 and p95 latency, with bootstrap 95% confidence intervals on each delta. *Pass criterion:* the table exists, every delta has an interval, and you can point to at least one row whose interval crosses zero and say out loud "this stage does not currently earn its place."

**The meta-drill, and the one that matters most in a real loop:** be able to answer, in under 60 seconds and without notes, "why does your RAG system fail on error codes and what would you do about it." That single question is the highest-frequency retrieval question in applied AI interviews, and the full answer is: dense embeddings are a lossy compression trained on semantic similarity so literal identifiers are discarded; the fix is a lexical channel fused with RRF, not a better embedder; you prove it with a per-slice ablation on identifier queries specifically, because the aggregate mean will hide a 25-point win on the 4% of queries that generate all your support tickets.
