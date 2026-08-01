### You need to combine BM25 results with dense results. Walk me through the two ways to do it and tell me which one you would ship.

There are exactly two families and the choice between them is the most consequential retrieval decision most teams make without noticing they made it.

**Family one: score fusion.** Normalize both score distributions onto a comparable scale, then take a weighted sum: `final = α · norm(dense) + (1 − α) · norm(bm25)`. This preserves *magnitude* information — the fact that the top dense hit scored 0.94 and the second scored 0.61 is a real signal about confidence, and score fusion keeps it.

**Family two: rank fusion.** Throw away the scores entirely and combine positions. Reciprocal Rank Fusion is the canonical form: `RRF(d) = Σ_retrievers 1 / (k + rank_r(d))` with `k = 60`. This discards magnitude and keeps only ordering.

**I ship RRF by default, and I'll tell you exactly why: score fusion requires calibration that you do not have and cannot easily get.** BM25 scores are unbounded, query-length-dependent, and corpus-dependent — a score of 18 means something entirely different for a 2-term query than a 7-term query. Cosine similarities are bounded in `[-1, 1]` but compressed into a narrow band by anisotropy; a typical dense retriever returns top-10 similarities all between 0.72 and 0.81. To weight-sum these you must normalize, and every normalization scheme is a landmine:

- **Min-max over the returned candidate set** is the standard choice and it is treacherous. It forces the top result to 1.0 and the bottom to 0.0 *for every query*, which means a query where BM25 found nothing relevant (top score 3.1) and a query where BM25 nailed it (top score 41.7) both produce a normalized top score of exactly 1.0. **You have deleted the confidence signal you were trying to preserve.** This is the single most common hybrid-search bug I find in code review.
- **Z-score normalization** is better because it preserves relative spread, but it is unstable when the candidate set is small or nearly uniform — the standard deviation of ten near-identical cosines is tiny, so dividing by it amplifies noise into large z-scores.
- **Fixed global scaling** (divide BM25 by some constant) requires you to re-derive the constant every time the corpus changes.

RRF sidesteps all of it. It requires zero calibration, it is invariant to any monotonic transformation of either retriever's scores, and it works the instant you add a third retriever. The cost is real and you should name it: **RRF cannot express "the lexical retriever is extremely confident."** A document that BM25 ranks first with an overwhelming exact-identifier match and a document BM25 ranks first with a weak partial match contribute identically. For identifier-heavy workloads that is a genuine loss, and it is the case where I would move to a tuned score fusion or, better, to a reranker that sees the raw text.

**🗣 Say this in the room:** "I default to RRF because it needs no calibration and stays correct when I add a third retriever, and because every score-normalization scheme I've used has silently destroyed confidence information — min-max in particular forces every query's top hit to 1.0. I move to weighted score fusion only when I have a labeled set to tune α on and I've measured that the magnitude signal is worth it."

### Derive RRF and tell me where the 60 comes from.

RRF says: **the only trustworthy thing a retriever tells you is its ordering, so use the ordering and nothing else.** The formula is deliberately trivial:

```
RRF(d) = Σ_{r ∈ retrievers}  1 / (k + rank_r(d))
```

`rank_r(d)` is 1-indexed; documents missing from a retriever's list contribute zero (equivalently, rank = ∞). The `1/rank` shape is chosen because relevance decays roughly harmonically with rank in practice — the gap between rank 1 and rank 2 matters far more than the gap between rank 50 and rank 51 — and the reciprocal is the simplest function with that property.

The `k` is a **smoothing constant that controls how much the top ranks dominate.** With `k = 0`, rank 1 contributes 1.0 and rank 2 contributes 0.5 — a single retriever's top hit is worth more than another retriever's ranks 2 and 3 combined, so one confident-but-wrong retriever can hijack the fused list. With `k = 60`, rank 1 contributes `1/61 = 0.01639` and rank 2 contributes `1/62 = 0.01613`, a 1.6% difference. The curve is nearly flat across the top ~20, which means **agreement across retrievers dominates position within any single retriever.** That is the behavior you want: a document that both retrievers put in their top 20 beats a document that one retriever put first.

Do the arithmetic explicitly, because interviewers ask: a document at rank 1 in dense only scores `1/61 = 0.0164`. A document at rank 12 in both scores `2/72 = 0.0278`. **Being decent in both beats being first in one**, and that is precisely the property that makes fusion work on decorrelated retrievers.

`k = 60` is not derived from theory — it is the value the original authors reported as working well across their TREC runs, and it stuck as the universal default (Elasticsearch, Qdrant, Weaviate and Milvus all default to 60 or expose it with 60 pre-filled). Treat it as a hyperparameter you *can* tune: lower `k` (10–20) if you trust your retrievers' top ranks strongly and want them to dominate; higher `k` (100+) if your retrievers are noisy and you want broad agreement to win. In practice I have rarely found tuning `k` worth more than 0.5 nDCG points, which is why the default survives.

**📄 Paper:** Cormack, Clarke & Buettcher (2009), *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. The contribution is exactly the surprise in the title — this two-line formula with no learned parameters beat trained learning-to-rank fusion methods. It replaced CombSUM/CombMNZ-style score fusion as the default for combining heterogeneous retrievers.

### Implement RRF and weighted-sum fusion. Show me both, and show me the weighting variant of RRF.

```python
from collections import defaultdict

def rrf(rankings: dict[str, list[str]], k: int = 60,
        weights: dict[str, float] | None = None, top_n: int = 50):
    """rankings: retriever_name -> [doc_id, ...] in descending relevance."""
    weights = weights or {}
    scores = defaultdict(float)
    for name, docs in rankings.items():
        w = weights.get(name, 1.0)
        for rank, doc_id in enumerate(docs, start=1):     # 1-indexed
            scores[doc_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]


def weighted_score_fusion(scored: dict[str, dict[str, float]],
                          weights: dict[str, float], top_n: int = 50):
    """scored: retriever_name -> {doc_id: raw_score}. Min-max per retriever."""
    fused = defaultdict(float)
    for name, doc_scores in scored.items():
        if not doc_scores:
            continue
        vals = list(doc_scores.values())
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0                            # guard: all-equal scores
        for doc_id, s in doc_scores.items():
            fused[doc_id] += weights[name] * (s - lo) / span
    return sorted(fused.items(), key=lambda kv: -kv[1])[:top_n]
```

Three implementation details that separate correct from subtly-wrong:

**Weighted RRF is the right escape hatch.** Note the `weights` argument on `rrf` — multiplying each retriever's reciprocal contribution by a scalar gives you a tunable knob without reintroducing score calibration. `{"dense": 1.0, "bm25": 0.7}` says "trust dense a bit more" while keeping every normalization guarantee. This is what I reach for when plain RRF is 90% right.

**Missing documents contribute zero, not `1/k`.** A document that appears only in the dense list should not get a consolation prize from BM25. The `defaultdict(float)` handles this correctly by construction; an implementation that iterates over the union of doc ids and looks up ranks with a default will get this wrong if the default is anything other than infinity.

**⚠ Trap:** the `span = (hi - lo) or 1.0` guard in the min-max version. When a retriever returns a single result, or when all candidate scores are identical (which happens with dense retrieval more often than you'd think on near-duplicate corpora), you divide by zero. In production this surfaces as a `nan` propagating into the sort, which in Python does not raise — it produces a silently arbitrary ordering. I have debugged this exact incident: search quality degraded for one tenant only, because that tenant's tiny corpus produced single-result candidate lists.

### How many candidates should each retriever return before fusion, and how do you pick that number?

The number that matters is **the recall ceiling of the fused candidate pool**, because everything downstream — fusion, reranking, MMR, the LLM — can only reorder what retrieval handed it. If the correct chunk is not in the union of the two candidate lists, no reranker on earth recovers it. So the question is not "what's a good `k`" but "how deep do I have to go before recall stops improving."

Measure it. Take your labeled set and plot recall@k for each retriever independently, `k ∈ {10, 20, 50, 100, 200, 500}`. You will see a knee. In most RAG corpora the dense curve is steep to about 30 and then flattens; BM25 flattens later because its tail is noisier. The operational rule I use: **retrieve to the point where the marginal recall gain per additional 10 candidates drops below 0.5 percentage points**, which in practice lands at 50–100 per retriever for a typical 1–10M chunk corpus.

Then the second constraint bites: the reranker. A cross-encoder cost is linear in candidates, so the pool size is directly a latency and dollar decision. Retrieving 100 from each retriever gives a fused pool of 100–200 unique documents. Reranking 200 is roughly 2× the cost of reranking 100 for typically under 1 nDCG point of gain. So the shape I default to, and the one I would write on a whiteboard:

```
BM25    top 100  ─┐
Dense   top 100  ─┼─ RRF ─→ top 50 ─→ cross-encoder rerank ─→ top 8 ─→ MMR/dedup ─→ top 5 → LLM
SPLADE  top 100  ─┘        (dedup)
```

**📐 Numbers you must know:** `100 → 50 → 8 → 5` is the default funnel and you should be able to justify every arrow. 100 per retriever because that's past the recall knee. 50 into the reranker because cross-encoder cost is linear and 50 fits a ~200 ms budget on one mid-range GPU. 8 out of the reranker because that's roughly where cross-encoder precision@k starts falling off. 5 into the context because near-duplicate collapse and token budget usually take you from 8 to 5.

**⚠ Trap:** retrieving 100 and then truncating to 10 *before* fusion because "the reranker only needs 10." You have now capped your recall at the intersection of two top-10 lists, which is often 4–6 documents. The whole point of retrieving deep is that fusion promotes documents that neither retriever ranked highly alone. Truncate after fusion, never before.

### Should you always run both retrievers, or can you route the query to just one?

Routing is real and it saves real money, but I want to be precise about when it is worth the complexity, because I have seen a lot of over-engineered routers that were worse than always-both.

The economics first. Always-both costs you one embedding call (typically 5–15 ms and a fraction of a cent) plus one lexical query (10–30 ms), run in parallel, so the *latency* cost of always-both is essentially `max(dense, lexical)` rather than the sum. **The latency argument for routing is therefore weak.** The cost argument is also weak: an embedding call at $0.02–0.13 per million tokens for a 15-token query is roughly `15 × $0.10/1e6 = $0.0000015` — a millionth of a dollar. At 1M queries/day that is $1.50/day. Routing to save that is not an engineering priority.

So route for **quality**, not cost. The cases where running both actively hurts:

**Pure-identifier queries.** Query is `ENG-4471` or a 40-char SHA. The dense retriever will return ten topically-plausible documents, all wrong, and they will occupy half your fused candidate pool, displacing correct lexical results. Here I hard-route to lexical only. The detector is cheap and rule-based: does the query match an identifier regex, is it a single token with high character-class entropy, is it entirely out-of-vocabulary for the corpus' common terms.

**Natural-language conceptual queries with zero rare terms.** "How do I think about pricing for a new tier?" BM25 contributes only noise here; every returned document matched on "pricing" and "tier." Routing to dense-only slightly improves the candidate pool.

The router itself should be a **classifier, not an LLM call**, at least at first. My ladder:

1. Regex/heuristic rules for identifiers, quoted phrases, and code-shaped tokens. Covers most of the value, costs microseconds, and is fully auditable.
2. A tiny logistic regression or gradient-boosted classifier over cheap features: query length, max IDF of any query term, fraction of out-of-vocabulary tokens, presence of digits, presence of camelCase/snake_case, whether the query ends in a question mark. Trains in seconds on 500 labeled queries.
3. An LLM classifier only if 1 and 2 measurably fail — and note the cost: adding a router LLM call adds 200–600 ms to TTFT and a per-query charge, to save you an embedding call worth a millionth of a dollar. That trade is almost never right.

**🗣 Say this in the room:** "I run both retrievers in parallel by default, because the latency cost is max not sum and the dollar cost is negligible. I route only where one retriever actively poisons the candidate pool — pure-identifier queries, where dense returns ten confident wrong answers. And I'd use a regex plus a small classifier, not an LLM, because an LLM router adds 300 ms of TTFT to save a fraction of a cent."

**⚠ Trap:** soft routing that adjusts the fusion weight per query based on a predicted "lexicality score." It sounds elegant and it is almost impossible to evaluate — you now have a continuous function whose errors are invisible in aggregate metrics. Hard routing with a small number of discrete branches is auditable; you can pull the queries that took each branch and inspect them. I enforce this in review.

### We have three indexes: a documentation store, a code index and a support-ticket index. How do you retrieve across all three?

This is the Glean/Cursor-shaped question and it is genuinely harder than two-retriever fusion, because the indexes have **incomparable score distributions, incomparable document lengths, and different prior probabilities of containing the answer.**

The wrong answer is to concatenate all three corpora into one index. You lose per-corpus analyzers (code needs a code tokenizer that preserves `snake_case` and `::`; prose needs stemming), you lose per-corpus chunking strategies, you lose the ability to do per-corpus permission filtering, and your `avgdl` becomes meaningless because a code chunk and a ticket thread differ in length by an order of magnitude — which corrupts BM25's `b` normalization for both.

The architecture I would draw:

**Per-index retrieval, each with its own hybrid.** Each index runs its own BM25 (with its own analyzer) and its own dense retrieval (potentially its own embedding model — a code-specialized embedder for the code index is usually worth it), fused internally with RRF. You now have three ranked lists, each internally calibrated.

**Cross-index fusion with per-index weights.** RRF again, but weighted, because the three sources are *not* equally likely to answer a given query. Weights come from a labeled set: for each query in your golden set, which index contained the answer? If 60% of answers live in docs, 30% in tickets, 10% in code, your priors are not uniform and RRF should reflect that.

**Query-conditional weights are where the real gain is.** A query containing a stack trace should weight the code and ticket indexes up. A "how do I" query should weight docs up. This is the one place I do accept a learned router, because the prior differs so sharply by query type that uniform weighting leaves a lot on the table. Implement it as a classifier over query features producing three weights, and evaluate it as an ablation against uniform.

**Interleaving as a guarantee, not just fusion.** For a UI where the user can see source labels, I sometimes enforce a *quota* — at least one result from each index in the top 10 — rather than pure fusion. This is a product decision, not a relevance decision, and it must be evaluated separately because it will lower your nDCG while raising user satisfaction. Say that distinction out loud; interviewers at product-AI companies specifically listen for it.

**🔍 Failure taxonomy — multi-index fusion:**
1. One index dominates every result set → its score/rank distribution is denser at the top, usually because its documents are much shorter. Check per-index contribution to the final top-10 against the per-index prior in your golden set.
2. One index never appears → its retriever is silently erroring or its candidate list is empty for most queries. Alert on per-index candidate count, not just on end-to-end latency.
3. Results are correct but the citation is from the wrong system → doc-id collisions across indexes. Namespace every id (`docs:1234`, `jira:ENG-4471`) at ingestion. This one causes user-visible wrongness and is trivially preventable.
4. Quality drops after adding a fourth index → RRF's flat top means a fourth mediocre retriever dilutes agreement between the three good ones. Adding retrievers is not free; each one needs an ablation.

### How do you actually tune the hybrid weight? "We set alpha to 0.5" is not going to fly.

Correct, and the reason it doesn't fly is that **α is not a philosophical position about semantics versus keywords — it is a property of your query distribution, and query distributions differ wildly between products.** A code-search product and a customer-support product will land on very different α, and neither one's number transfers.

The procedure, which I would write out step by step in a design round:

**Step 1 — build the labeled set from query logs, not from imagination.** Sample 200–500 real queries stratified by frequency band: take the head (top 50 most frequent), a random sample of the torso, and a random sample of the long tail. If you use only head queries you will tune for the 20% of traffic that a cache should handle anyway. For each query, get relevance judgments — either from click/thumbs data if you have it, or from an LLM judge validated against ~50 human labels to confirm the judge agrees with humans at a rate you can live with. Graded labels (0/1/2/3) beat binary if you can afford them, because nDCG needs graded relevance to be meaningful.

**Step 2 — fix everything downstream and sweep α alone.** `α ∈ {0.0, 0.1, ..., 1.0}`, eleven runs, retrieval only, no reranker, no LLM. Report recall@50 (the pool the reranker will see) and nDCG@10. **Report both, because they can disagree and the disagreement is informative**: α tuned for nDCG@10 optimizes final ordering, α tuned for recall@50 optimizes what the reranker can work with. If you have a reranker, tune α for recall@50 — ordering is the reranker's job.

**Step 3 — check the curve shape, not just the argmax.** If the recall curve is flat between 0.3 and 0.7 and drops at the ends, you have learned something important: both retrievers contribute and the exact weight doesn't matter. Ship 0.5 and stop tuning. If it's sharply peaked, you have a small labeled set and you are overfitting — bootstrap-resample your queries and check that the peak survives. I insist on this because a 300-query eval set gives you a standard error on nDCG@10 of roughly 0.02, so a "win" of 0.01 is noise.

**Step 4 — slice by query type.** Compute the optimal α separately for identifier queries, question queries and short keyword queries. If they differ by more than ~0.2, that is your evidence that routing beats a single global α, and now you have the ablation to justify building the router.

**💰 Math:** the whole sweep is cheap and you should say so. 400 queries × 11 α values = 4,400 retrieval runs, but the retrievals themselves are shared — you retrieve once per query per retriever (800 retrieval calls) and re-fuse 11 times in memory. Embedding 400 queries at ~15 tokens costs `400 × 15 × $0.10/1e6 = $0.0006`. Total compute: under a minute. **The expensive part is the 400 relevance labels, not the sweep**, which is why the labeled set is the asset and the α is a disposable output. Build the set once; re-tune every time you change the embedder, the chunker or the corpus.

### What is SPLADE and why would I use it instead of BM25?

SPLADE is **a learned sparse retriever: a BERT that outputs, for each input, a weight over the entire 30k WordPiece vocabulary — and then you put those weights into an ordinary inverted index.** That last clause is the whole point. You get neural quality with lexical infrastructure: the same posting lists, the same block-max WAND pruning, the same exact-match guarantees on terms that actually appear.

Mechanically: run the text through BERT, and for each output position project through the masked-language-model head to get logits over the vocabulary. Then aggregate across positions with a max (or log-sum-exp) and apply `log(1 + ReLU(x))` to force non-negativity and sublinearity. The result is a `|V|`-dimensional vector that is mostly zero. Sparsity is *trained in* with an explicit FLOPS regularizer on the expected number of non-zeros — without it the model happily outputs a dense 30k vector and you have destroyed the inverted index's entire advantage.

Why this beats BM25: **term expansion.** For the document "myocardial infarction," SPLADE assigns non-zero weight not just to `myocardial` and `infarction` but to `heart`, `attack`, `cardiac`, `coronary` — because the MLM head knows those tokens are predicted by this context. So the query "heart attack" now has genuine lexical overlap with a document that never contains those words. **SPLADE fixes exactly the vocabulary-mismatch failure that is BM25's hard ceiling, while keeping BM25's exact-identifier strength.** It also learns term weights rather than using IDF, which handles the "this word is rare but irrelevant" case that IDF gets wrong.

**📄 Paper:** Formal, Piwowarski & Clinchant (2021), *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*, with SPLADE v2 and the distillation-plus-hard-negatives follow-up (Formal et al., 2022) being the versions actually worth using. The contribution was showing that a learned sparse representation over an existing inverted index could compete with dense bi-encoders — replacing the assumption that neural retrieval required an ANN index.

**⚠ Trap:** SPLADE query expansion inflates the number of posting lists you must merge. A 5-token BM25 query touches 5 posting lists; the same query in SPLADE may activate 40–100 terms, several of which are common words with enormous posting lists. **Query latency can be 3–10× BM25's**, which surprises people who assumed "it's just an inverted index." The mitigations are query-side sparsity regularization (SPLADE-doc variants expand only the document side and leave queries as plain BM25 terms — the "efficient" configuration), and aggressive block-max pruning. If you evaluate SPLADE and only measure nDCG, you will ship a latency regression.

### So would you use SPLADE, BM25+dense, or both? Give me the decision rule.

The honest state of the field: **SPLADE, BM25+dense hybrid, and ColBERT all land in a similar quality band on public benchmarks, and the differentiator in your system will be operational, not accuracy.** I would say that plainly in an interview rather than pretending one obviously wins, because pretending is how you get caught.

My decision rule:

**Use BM25 + dense hybrid** when you are starting, when you already run Elasticsearch or Postgres and a vector store, when your team has no GPU serving story, or when your corpus updates frequently. It is the highest floor with the least machinery, it requires no model inference at index time beyond the embedder you already need, and everything about it is debuggable — you can look at a BM25 score and explain it.

**Use SPLADE** when you have a lexical-heavy domain where vocabulary mismatch is the residual failure, you already have a strong inverted-index operation, and you can absorb a GPU inference step in ingestion. It is especially attractive when you *cannot* run an ANN index — some regulated or on-prem environments — because it gives you semantic matching using infrastructure the customer already approved. It replaces both channels with one, which simplifies fusion.

**Use both SPLADE and dense** only if an ablation shows it. SPLADE and dense fail on more correlated queries than BM25 and dense do (both are neural, both trained on similar data), so the fusion gain is smaller than you would hope.

**⚠ Trap:** treating SPLADE's index-time cost as free. Every document must go through a BERT forward pass at ingestion — the same cost profile as embedding, so roughly doubling your ingestion GPU bill if you also embed. For a 10M-chunk corpus at ~400 tokens per chunk, that is `10e6 × 400 = 4e9` tokens through a 110M-param model, `2 × 110e6 × 4e9 = 8.8e17` FLOPs. On an A10G at an effective ~25 TFLOP/s that is `8.8e17 / 2.5e13 = 35,200 seconds ≈ 9.8 GPU-hours` per full pass, roughly $10–15 of compute (📅 Volatile — verify current instance pricing). That's cheap for a one-time build and *not* cheap if you reindex nightly.

### Our hybrid search scores worse than dense alone on our eval set. Debug it.

Good — this happens constantly and the diagnosis path is short if you know it. I would go in this order.

**First, check that the lexical retriever is actually returning anything.** Log the per-retriever candidate count. The most common cause by far is that BM25 returns an empty or near-empty list for most queries because of an analyzer mismatch — you indexed with a stemming analyzer and query with a keyword analyzer, or the field is `keyword`-typed rather than `text` so the entire document body is one token. If lexical returns nothing, RRF degrades to dense-only and you should score *the same*, not worse. If you score worse, something else is happening.

**Second, check for id mismatch.** If the lexical index returns `doc_id` in a different namespace or format than the vector store (string vs int, with or without a chunk suffix), fusion sees two disjoint sets and awards every document a single-retriever score. This produces a fused list that is neither retriever's list — it interleaves them roughly arbitrarily — and it *is* worse than either alone. I have seen this three times. Assert on set-intersection size in a test.

**Third, check the candidate depths.** If dense returns 100 and BM25 returns 1000, BM25 contributes 900 documents that dense never saw, each with a nonzero RRF score, and they dilute the top of the fused list. Equal depths, or explicitly weighted.

**Fourth, the real relevance answer: your query distribution may genuinely not need lexical.** If your queries are all natural-language questions against prose documentation, BM25's contribution is noise — it matched on common words. Slice your eval by query type. If hybrid loses on question-queries and wins on identifier-queries, and your eval set is 95% question-queries, then hybrid is correct for production and your *eval set is unrepresentative of the queries you're worried about.* This is the diagnosis people miss, and it is the interesting one: the fix is to fix the eval set, not the retriever.

**Fifth, drop the fusion weight and confirm monotonicity.** Sweep α from 0 (pure lexical) to 1 (pure dense). If the curve is not roughly unimodal — if it bounces around — your fusion has a bug, not a tuning problem.

**⚠ Trap:** concluding "hybrid doesn't help" from a single aggregate number. Hybrid's entire value proposition is on the *tail* of the query distribution, and tails are exactly what aggregate means hide. A 500-query eval with 20 identifier queries can show hybrid winning by 0.4 nDCG points overall while it wins by 30 points on those 20 queries — and those 20 queries are the ones that generate support tickets. **Always report the per-slice table, not just the mean.**

### How do you prove that each stage of your retrieval pipeline is actually earning its place?

By running an ablation, one stage at a time, on a frozen eval set — and I mean this literally, as a table you produce and keep in the repo. The alternative is what most teams have: a pipeline where every stage was added because a blog post recommended it and nobody has ever measured any of them.

The table I build:

| Configuration | recall@50 | nDCG@10 | p95 retrieval latency | cost / 1k queries |
|---|---|---|---|---|
| Dense only | — | — | — | — |
| BM25 only | — | — | — | — |
| Hybrid (RRF) | — | — | — | — |
| Hybrid + cross-encoder rerank | — | — | — | — |
| Hybrid + rerank + MMR | — | — | — | — |
| Hybrid + rerank + MMR + recency boost | — | — | — | — |

Rules I enforce on this table:

**Add stages in the order you'd actually build them**, so each row's delta is the marginal contribution *given everything above it*. A reranker's gain on top of dense-only is much larger than its gain on top of a good hybrid, because hybrid already fixed some of what the reranker would have fixed. Reporting the reranker's gain against the wrong baseline overstates it, and I have seen that overstatement used to justify a $60k/year reranking bill.

**Every row carries latency and cost, not just quality.** A stage that adds 1.2 nDCG points and 400 ms of TTFT is a bad trade in a chat product and a fine trade in an async research product. The table is how you have that conversation with a PM instead of arguing from intuition.

**Include a leave-one-out row for anything expensive.** "Full pipeline minus reranker" tells you what the reranker is worth *at the end*, which is different from what it was worth when you added it. Pipelines rot: a stage that earned its place two quarters ago may now be redundant with a better embedder.

**Report confidence, not just point estimates.** Bootstrap over queries, 1,000 resamples, report the 95% interval on the delta. With 300 queries, a 1-point nDCG@10 difference is usually inside the noise band. Saying "the reranker gains 4.2 points, 95% CI [2.1, 6.4]" is what makes the result actionable, and it is the single fastest way to signal that you have done evaluation for real.

**🗣 Say this in the room:** "Every stage in my retrieval pipeline has a row in an ablation table with recall, nDCG, p95 latency and cost per thousand queries, and every row's delta is measured against the pipeline as it actually exists rather than against a naive baseline. If a stage can't show a delta outside the bootstrap confidence interval, I delete it — pipelines accumulate stages that stopped paying rent."

### One more fusion question: when should you deduplicate, and does deduplication happen before or after fusion?

Before fusion, at the identity level; after reranking, at the *content* level. Those are two different operations that get conflated and it matters.

**Identity dedup before fusion** means: the same chunk retrieved by both retrievers must be recognized as one document, or RRF cannot do its job at all — the whole mechanism depends on summing a document's reciprocal ranks across retrievers. If your dense store returns `chunk_8891` and your lexical store returns `doc_442#p3` for the same underlying text, fusion sees two documents. This is an ingestion-contract problem: **one canonical chunk id, minted deterministically at ingestion time (content hash plus source id), used identically by every index.** Enforce it with a test that retrieves the same query from both stores and asserts a nonzero intersection.

**Content dedup after reranking** means: five chunks whose text is 95% identical — because the same paragraph appears in four versions of a policy document plus a FAQ that quotes it — must not consume five of your eight context slots. This is a near-duplicate problem, not an identity problem, and it needs similarity thresholds rather than hash equality. It belongs after reranking because the reranker's score is what you use to decide which of the near-duplicates to keep.

Doing content dedup *before* the reranker is a mistake I would flag in review: you would be discarding candidates based on similarity to each other before you know which one is actually most relevant, so you can drop the good copy and keep the bad one. The cheap version of this argument: dedup is a *selection* operation and selection needs the best available relevance signal, which is the reranker's.
