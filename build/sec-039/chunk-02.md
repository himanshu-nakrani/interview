### Why do you insist on evaluating retrieval separately from generation? Isn't end-to-end accuracy the thing users care about?

Users care about end-to-end. *Engineers* cannot act on end-to-end. That is the whole argument, and it is the same argument as "p99 latency told me the system is slow, the span breakdown told me which service."

The mental model: **end-to-end accuracy is a product of stage-wise success probabilities, and a product tells you nothing about which factor is small.** If retrieval surfaces the right evidence 65% of the time and generation answers correctly given right evidence 90% of the time, end-to-end is roughly 0.585. Move generation to 0.95 — a hard, expensive win — and you get 0.6175, a 3-point gain. Move retrieval to 0.85 — usually much easier — and you get 0.765, an 18-point gain. Without decomposition you cannot see that, and the default human bias is to blame the model, because the model is the part that produced the visible wrong sentence.

Second, decoupled retrieval metrics are **cheap, fast, and deterministic.** Recall@k over 500 queries is a few seconds of ANN queries and zero LLM calls. End-to-end accuracy over 500 queries is 500 generation calls plus 500 judge calls, costs real money, takes minutes, and is *noisy* — the same config re-run gives you a different number because sampling is nondeterministic and your judge is a model. You cannot run a 40-cell ablation grid on a metric like that. You can run it on recall@k before breakfast.

Third, retrieval metrics have a **ceiling interpretation** that end-to-end metrics do not. Recall@k is an upper bound on what any downstream stage can achieve. If recall@10 is 0.70, then no prompt, no model upgrade, no reranker can push end-to-end answer accuracy above 0.70 on questions that need that evidence. Publishing that number kills more bad roadmap items than any other single measurement I know.

The right structure is a metric hierarchy, and I want three tiers in every RAG project:

- **Tier 1, retrieval, no LLM:** recall@k, nDCG@k, MRR, per-hop retention. Runs in CI on every PR, seconds, free.
- **Tier 2, generation given fixed context, LLM but no retrieval variance:** faithfulness, answer correctness against oracle context. Runs nightly.
- **Tier 3, end-to-end with a judge and human spot-checks:** the number you report to the business. Runs weekly and before release.

**🗣 Say this in the room:** "End-to-end accuracy is the number I report; stage-wise metrics are the numbers I act on. Retrieval recall is an upper bound on the whole system, it costs nothing to compute, and it is deterministic — so it is what I gate CI on. The end-to-end number is too noisy and too expensive to run a forty-cell ablation against."

### Define recall@k, nDCG@k and MRR precisely, and tell me which one you would actually optimize for a RAG system.

Assume for each query `q` you have a set of relevant chunk IDs `R_q` (often with graded relevance) and a ranked result list.

**Recall@k** = |retrieved top-k ∩ R_q| / |R_q|, averaged over queries. It answers "what fraction of the evidence did I get into the candidate pool." Ordering-blind within the top k.

**MRR** — Mean Reciprocal Rank = mean over queries of 1/(rank of the first relevant result), 0 if none in the cutoff. It only sees the *first* hit. Perfect for a "one right answer" setting like a lookup or a nav query; wrong for RAG where the answer needs three chunks, because it is indifferent between "one relevant at rank 1, nothing else" and "relevant at ranks 1, 2 and 3."

**nDCG@k** — this is the one worth being able to derive on a whiteboard. With graded relevance `rel_i` at rank `i` (1-indexed):

DCG@k = Σ_{i=1..k} (2^{rel_i} − 1) / log₂(i + 1)

The numerator makes highly-relevant items exponentially more valuable than marginally-relevant ones; the log denominator is the *discount* — a position penalty that decays slowly, encoding "users look further down than a 1/i penalty would suggest." IDCG@k is the same sum computed over the ideal ordering (all relevance grades sorted descending). nDCG@k = DCG@k / IDCG@k, which normalizes to [0,1] so queries with different numbers of relevant documents are comparable and averageable.

Which to optimize? **It depends on the stage, and that is the answer the interviewer wants.**

For the *first-stage retriever*, optimize **recall@k_candidates** where `k_candidates` is what you feed the reranker — typically 50 or 100. Order does not matter here, because a cross-encoder is about to reorder everything. Optimizing nDCG on the first stage is optimizing something you are going to throw away.

For the *reranker*, optimize **nDCG@k_final** where `k_final` is what enters the prompt — typically 5 to 10. Order matters now because of position effects in the context window, and grades matter because a marginally-relevant chunk in the prompt is a distractor.

For the *system*, the honest metric is neither: it is **answer-sufficiency@k** — does the top-k set *jointly* contain enough evidence to answer. That is a set property, not a per-document property, and none of the three classical IR metrics measure it. Recall@k is the closest proxy when your golden labels mark the *full* evidence set rather than one "the" relevant document.

**⚠ Trap:** reporting recall@k with `|R_q| = 1` for every query. If your golden set says each query has exactly one relevant chunk — which is what you get by default when you generate synthetic queries from single chunks — then recall@k collapses to hit-rate@k and you are systematically blind to incompleteness (FP7). Multi-evidence queries are the ones that break in production and the ones your golden set has zero of.

### Implement nDCG@k from scratch. No libraries.

```python
import math
from typing import Sequence, Mapping

def dcg(gains: Sequence[float]) -> float:
    # gains are already in rank order, rank i is 1-indexed
    return sum((2.0 ** g - 1.0) / math.log2(i + 2) for i, g in enumerate(gains))

def ndcg_at_k(ranked_ids: Sequence[str], rels: Mapping[str, float], k: int) -> float:
    """ranked_ids: system output, best first. rels: chunk_id -> graded relevance (0..3)."""
    gains = [rels.get(cid, 0.0) for cid in ranked_ids[:k]]
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_ids, rels, k, thresh=1.0):
    gold = {c for c, r in rels.items() if r >= thresh}
    if not gold:
        return float("nan")          # do NOT return 0; it poisons the mean
    return len(gold & set(ranked_ids[:k])) / len(gold)

def mrr_at_k(ranked_ids, rels, k, thresh=1.0):
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if rels.get(cid, 0.0) >= thresh:
            return 1.0 / i
    return 0.0
```

Three details that separate a correct implementation from a plausible one, and interviewers do check.

`math.log2(i + 2)` with a 0-indexed loop is `log2(rank + 1)` — the discount at rank 1 is log₂2 = 1, i.e. no discount. Off-by-one here silently inflates every score you report.

IDCG must be computed over **all** known relevance grades truncated to k, not over the grades present in your returned list. If your system returned nothing relevant, IDCG is still positive and nDCG is correctly 0. Computing IDCG from the returned gains gives you 0/0 or, worse, 1.0 for a system that returned three equally-bad results.

Queries with no relevant documents must be **excluded**, not scored zero. A golden set where 8% of queries have no labeled answer (because the annotator could not find one) will drag your mean recall down by 8 points and you will chase a phantom regression. Return NaN and use `nanmean`.

**🏋 Drill:** 15 minutes, no references. Write `ndcg_at_k`, `recall_at_k`, `mrr_at_k` and a `bootstrap_ci` that returns a 95% confidence interval on the mean of a per-query metric by resampling queries with replacement, 1,000 iterations. Pass criterion: correct discount indexing, NaN handling, and a CI that widens when you halve the query count. If you cannot write the CI, you cannot honestly claim any A/B result in this discipline.

### Your recall@50 is 0.97 but recall@5 is 0.61. What is your next move, and what is definitively not your next move?

This is the cleanest diagnostic signature in RAG, and it says one thing: **your retriever's recall is excellent and its precision at the top is bad. You have a ranking problem, not a retrieval problem.**

What is definitively *not* the move: swapping the embedding model. A new embedder is a multi-week project — re-embed the corpus, re-tune thresholds, re-run everything — and it improves the metric that is already 0.97. You would be spending the quarter to move recall@50 from 0.97 to 0.98. I would push back hard on this in a design review, and this exact misallocation is extremely common because "upgrade the embedding model" feels like the ML-shaped action.

The move is a **cross-encoder reranker**: retrieve 50, rerank, keep 5–8. A bi-encoder must compress the query and the document into independent vectors that never see each other, so it can only measure "are these in the same region of semantic space." A cross-encoder puts query and document in the *same* forward pass with full attention between them, so it can represent "this document answers this specific question" rather than "this document is about this topic." That is exactly the capability gap your two numbers describe. The price is that it is O(k) model calls and cannot be precomputed — which is why the architecture is always retrieve-wide-then-rerank-narrow rather than rerank-everything.

Secondary moves, in order, if reranking is insufficient:

**Hybrid fusion**, if you are dense-only. A large share of rank-6-to-rank-50 misses are exact-token misses — identifiers, error codes, product names — that BM25 nails at rank 1. RRF at k=60 needs no score calibration, which is why I default to it over weighted-sum normalization.

**Query rewriting**, if the gap is worse on multi-turn traffic. Segment your recall@5 by turn index; if turn 1 is 0.78 and turn 3+ is 0.42, the bug is that you are embedding "does that apply to enterprise too?" literally.

**Chunk-size reduction with small-to-big**, if the gap is worse on specific-fact queries. Diluted embeddings rank poorly at the top even when they are in the pool.

**💰 Math:** a hosted reranker priced around $2 per 1,000 search units, with one search unit covering up to ~100 documents, means reranking 50 candidates costs ~$0.002 per query. At 200k queries/day: 200,000 × $0.002 = **$400/day, ~$12,000/month.** 📅 Volatile — verify current reranker pricing. That is a real number and it is why "just add a reranker" needs a business case: measure the recall@5 lift first, then decide whether $12k/month buys enough. Self-hosting a small open cross-encoder on a single mid-range GPU changes the arithmetic completely — roughly $1–2/hour of GPU, ~$1,100/month, if you can hit the throughput, which for a ~100M-parameter cross-encoder at 50 docs/query and 128–256 token pairs is comfortably feasible at a few hundred QPS with batching. The decision is throughput-dependent, and stating it that way is the senior answer.

### I hand you a corpus and no labels. How do you build a golden query-document set in a week?

Four sources, blended, because each one is biased in a different direction and blending is what makes the set representative rather than convenient.

**Source 1 — real query logs, if they exist. This is the highest-value source and you should fight for it.** Even a shipped keyword search box or a support-ticket subject line corpus gives you the actual distribution of user intent. Take the last 90 days, strip PII, cluster by embedding, and sample *stratified by cluster* rather than by frequency — otherwise 60% of your golden set is "how do I reset my password" and you have no signal on the tail. I typically take 30% head, 40% torso, 30% tail.

**Source 2 — synthetic queries generated from your own chunks.** Cheap, fast, and the standard bootstrap: for each sampled chunk, prompt a model to write the questions this chunk answers. This gives you query-document pairs with the label already attached by construction. It has a specific and severe bias, which gets its own answer below.

**Source 3 — domain experts writing questions cold**, without seeing the corpus. Twenty questions each from five support engineers, five sales engineers, and three actual customers if you can get them. Small N, enormously high value, because these are the only queries in your set that were not generated *from* the corpus and therefore the only ones that can reveal missing content (FP1). Budget: about four hours of expert time total.

**Source 4 — adversarial and negative queries**, written deliberately: questions the corpus genuinely cannot answer, questions with false presuppositions ("what is our SLA for the Bangalore data center" when there is none), questions requiring information from two documents, questions with typos and jargon, and near-miss questions that a naive retriever will answer with the wrong-but-similar document. Target ~15% of the set. **This is the slice that measures abstention**, and without it your system scores well by answering everything.

Then labeling. For each query, retrieve top-50 with a *union of several retrievers* (dense, BM25, and with a reranker) — this is standard IR practice called pooling, and it exists because you cannot afford to label the whole corpus and you must not label only what your current system returns. Label the pooled candidates on a 0–3 graded scale. Two annotators on a 20% overlap sample so you can compute agreement; if Cohen's κ is below ~0.6 your rubric is ambiguous and the labels are noise. Fix the rubric, re-label.

Size: **150–300 queries is the working minimum**, and I will justify that in the next answer. Fewer than 100 and your confidence intervals are wider than any effect you are trying to detect.

**⚠ Trap:** letting the current system define the label pool. If you only label what production retrieval returned, then recall@k is computed against a set your system by construction retrieved, and it will look near-perfect forever. Pooled labeling from multiple retrievers is not academic fussiness; it is the only way the metric can go down when you break something.

### You bootstrapped the golden set with LLM-generated synthetic queries. What bias did you just bake in, and how do you correct for it?

Four biases, and they all push the same direction: **your evaluation gets systematically easier than reality, so your numbers are optimistic and your ablations are underpowered.**

**Bias 1 — lexical leakage.** The generator saw the chunk and writes the question using the chunk's own vocabulary. "According to the escalation matrix, what is the P1 response window?" retrieves trivially because "escalation matrix" and "P1 response window" are literally in the target chunk. Real users write "how fast do you have to call me back when everything's down." Your dense retriever looks superb on synthetic and mediocre in production, and the gap is largest exactly where you need signal: jargon mismatch. Correction: a second LLM pass that *paraphrases the question into naive user language without seeing the chunk*, plus a hard filter dropping any question with a rare-term overlap above a threshold with its source chunk. I compute overlap on terms with corpus document frequency below ~1%.

**Bias 2 — single-chunk answerability.** Generated from one chunk, answerable from one chunk. Your set therefore contains ~zero multi-hop and ~zero enumeration queries, which are the FP7 failures that dominate real complaints. Correction: a separate generation pass that takes *two or three* related chunks and asks for a question requiring all of them, with the golden label being the union. Target 20–25% multi-evidence.

**Bias 3 — no unanswerable queries.** Every synthetic query is answerable by construction. Abstention is unmeasured and your system's over-answering is invisible. Correction: generate questions from chunks, then *delete those chunks from the index* for an unanswerable slice — you get guaranteed-unanswerable queries that are still on-topic and plausible, which is much harder and more realistic than off-topic nonsense.

**Bias 4 — style monoculture.** Every query is a well-formed complete-sentence question with correct spelling. Real query logs are 40% fragments, contain typos, and use internal abbreviations. Correction: a corruption pass — drop stopwords, truncate to a fragment, inject a plausible typo, substitute an internal acronym — applied to a third of the set.

The honest framing for an interview: **synthetic queries are a fine way to get from zero labels to a working harness in a day, and a bad thing to still be relying on in month six.** I treat them as scaffolding, and I track the ratio explicitly as a health metric — "% of golden set derived from real user queries" going up over time, with real feedback-captured failures being promoted into the set continuously.

**🗣 Say this in the room:** "Synthetic queries leak the chunk's vocabulary into the question, so they systematically overstate dense retrieval. I use them to bootstrap, then I paraphrase them into user language, add multi-chunk and unanswerable slices, and replace them with mined real queries as logs accumulate. I'd quote a synthetic-only recall number with an explicit caveat that it's an upper bound."

### How many queries do you need in the golden set before an A/B difference means anything?

This is where backend statistical instincts pay off directly, and it is the question that separates people who run evals from people who run eval theater.

Retrieval metrics are per-query means of bounded values, so the standard error of the mean is `s/√n` with `s` the per-query standard deviation. For a metric like recall@5 with a mean around 0.7, the per-query values are mostly in {0, 0.33, 0.5, 1.0} and `s` typically lands in the 0.35–0.45 range. Take s = 0.40.

- n = 50: SE = 0.40/√50 = **0.057**. A 95% CI on your point estimate is ±0.11. You cannot detect anything smaller than about 11 points.
- n = 200: SE = 0.40/√200 = **0.028**, CI ±0.055.
- n = 500: SE = 0.40/√500 = **0.018**, CI ±0.035.

But the number that matters is not the CI on one arm — it is the CI on the *difference*, and here you get a large gift: **use a paired test.** Both configurations run the same queries. The variance of the paired difference is driven by how often the two systems disagree, not by the overall spread. If system B changes the outcome on only 15% of queries, the per-query difference is 0 for 85% of them and `s_diff` might be 0.25 rather than 0.40 × √2 = 0.57. Paired testing can be **4–5× more sample-efficient** than unpaired, which for a 200-query set is the difference between "we can detect 3 points" and "we can detect 11 points."

So the rule I enforce:

- **Always paired.** Same queries, same seed, same order. Report the mean *difference* and a bootstrap CI on the difference by resampling queries.
- **n ≥ 150** for retrieval metrics, **n ≥ 200** if you want to slice by segment (multi-turn vs single-turn, jargon vs plain), because a slice of 200 is 40 queries and back to useless.
- **Report the CI, always.** "Recall@5 went from 0.712 to 0.741" is not a result. "+0.029, 95% CI [+0.004, +0.055], n=200 paired" is a result. "+0.029, 95% CI [−0.018, +0.076]" is a *non*-result that looks identical if you print only the point estimate, and shipping on it is how you accumulate a pipeline of stages that each "helped."

For LLM-judged metrics add a second variance source: the judge itself. Run the judge at temperature 0, and even then re-run the *identical* config twice and measure the run-to-run delta. That delta is your noise floor; any effect smaller than it is unmeasurable no matter how many queries you add. I have seen noise floors of 2–4 points on judged correctness, which quietly invalidates most reported RAG improvements in blog posts.

**📐 Numbers you must know:** SE = s/√n, so **halving your CI requires 4× the queries.** Going from 200 to 800 queries to detect a 1.5-point improvement is usually the wrong trade — instead reduce variance: use paired tests, use graded rather than binary labels, and stratify. Stratified sampling with proportional allocation reduces variance for free.

### What is BEIR, and how much should a BEIR leaderboard position influence which embedding model you pick?

**📄 Paper:** Thakur, Reimers, Rücklé, Srivastava & Gurevych (2021) — *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. It assembled a suite of retrieval datasets across very different domains and task shapes — fact verification, bio-medical QA, argument retrieval, duplicate-question detection, entity retrieval, news — and evaluated models **zero-shot**: train wherever you like, then evaluate on all of these without in-domain fine-tuning, reporting nDCG@10.

Its headline finding is the one to remember and the one that made it influential: **dense retrievers that dominated in-domain benchmarks generalized poorly, and BM25 was a remarkably strong zero-shot baseline**, beating many neural models on a majority of the datasets. Reranking pipelines (BM25 retrieve + cross-encoder rerank) were the strongest configurations. That result is the empirical foundation for the hybrid-plus-rerank default architecture that the whole industry converged on.

How much should it drive your model choice? **It is a screening filter, not a decision.** Concretely how I use it:

Use aggregate BEIR/MTEB-retrieval numbers to build a shortlist of three to five candidates that are not obviously bad and that fit your constraints — dimensionality (storage and index cost scale linearly), max sequence length (does it handle your chunk size or silently truncate at 512 tokens), multilingual coverage, licence, and whether you can self-host. Then **evaluate the shortlist on your own labeled set**, because the aggregate BEIR ranking has near-zero predictive power for the ordering on *your* domain. The per-dataset variance in BEIR is enormous; a model that is #1 on average can be #9 on the dataset that most resembles your corpus.

The mechanism behind "in-domain still wins": embedding models place documents in a space shaped by their training distribution. Your corpus has jargon, entity names, internal acronyms, and a query style none of them saw. Fine-tuning a bi-encoder on even a few thousand in-domain query-positive pairs — mined from click logs or generated and filtered — routinely beats the best off-the-shelf model on that domain, because it is learning your vocabulary's geometry. That is the honest reason in-domain wins, and it is also why the effect is largest for specialized corpora (legal, medical, internal engineering docs) and smallest for general web-ish text.

**⚠ Trap:** benchmark contamination and leaderboard overfitting. Public retrieval leaderboards have been optimized against for years; some entries are trained with the benchmark's own training splits, and the gap between "trained on the benchmark family" and "genuinely zero-shot" is not always disclosed. Treat a 0.3-point aggregate difference between two models as noise. Treat a 4-point difference on *your* set as signal.

**🗣 Say this in the room:** "BEIR is what taught the field that dense retrieval doesn't generalize zero-shot and BM25 is a strong baseline — that's why hybrid plus rerank is the default. I use it to shortlist, never to decide. The decision comes from a paired run on a hundred and fifty in-domain labeled queries, because per-domain ordering barely correlates with the aggregate."

### You have three candidate embedding models. Design the ablation you would actually run.

The design constraint that makes this hard: an embedding model change is entangled with everything downstream. Change the model and the score distribution changes, so your similarity threshold is now wrong; the dimensionality changes, so your index parameters are wrong; the max sequence length changes, so your chunk size may be wrong. If you swap the model and nothing else, you are measuring "model A with model B's hyperparameters," which is a rigged comparison in an unpredictable direction.

So the protocol:

**Fix everything you can, and re-tune only what is mechanically forced.** Same chunks — identical text, identical boundaries, identical IDs — so the golden labels transfer without relabeling. Same queries. Same `k`. Same reranker or none. What you must re-tune per model: the ANN index build parameters to a matched recall level (an HNSW index tuned to 0.98 recall-vs-exact for one model and 0.90 for another is not a comparison of models), and any absolute score threshold, which must be re-derived per model from its own score distribution.

**Report at multiple k.** Recall@5, recall@10, recall@50, nDCG@10. A model can win at 50 and lose at 5 — that is a real and common pattern, and it means "wins if you add a reranker, loses if you don't." Which fact you need depends on your architecture.

**Run it with and without the reranker.** Half of embedding-model differences evaporate behind a cross-encoder, and if they do, the correct decision is to take the cheaper/smaller/faster model. This is the single most cost-saving finding in a typical ablation and almost nobody runs it.

**Slice the results.** Overall means hide the decision. Slice by: query length, query type (identifier-bearing vs natural-language), turn index, document type, and language. A model that wins by 2 points overall while losing 9 points on identifier queries is a bad choice for a developer-tools product.

**Report cost and latency alongside quality, in the same table.** Dimensions (storage and ANN cost), embedding price per Mtok, embedding throughput for reindex, and query-side embed latency, which lands directly in your TTFT budget.

```
model     dim  recall@5  recall@50  nDCG@10  +rerank r@5  $/Mtok  q-latency  index GB
A         768    0.61      0.94       0.58      0.83        0.02     18 ms      2.9
B        1536    0.68      0.96       0.64      0.85        0.13     31 ms      5.8
C        1024    0.66      0.95       0.62      0.84        0.05     22 ms      3.9
```

Read that table the way I would: B wins on raw retrieval by 7 points over A, but behind a reranker the spread is 2 points, at 6.5× the embedding price and 2× the storage. If you are shipping a reranker — and you should be — **A is the correct choice and B is a $60k/year mistake dressed as a quality win.**

**💰 Math:** the storage delta alone: 20M chunks × 1536 dims × 4 bytes = 123 GB versus 20M × 768 × 4 = 61 GB. On a managed vector service where you pay for memory-resident index capacity, that difference is frequently $1,500–4,000/month. 📅 Volatile — price it against your vendor. Plus reindex cost: 20M chunks × 400 tokens = 8B tokens; at $0.13/Mtok that is $1,040 per full reindex versus $160 at $0.02/Mtok, and you will reindex more often than you expect.

### Chunk size × embedding model × hybrid weight × reranker × top-k is a huge grid. How do you actually run that without burning a month?

You do not run the grid. Running the full Cartesian product is the beginner move, and it is both expensive and — this is the part people miss — **statistically worse**, because a 3×3×5×2×4 = 360-cell grid evaluated on 200 queries will produce a "winner" that is winning on noise. With 360 cells and a per-cell CI of ±0.055, the max-of-360 is inflated by several points purely by selection.

The discipline I use, in four moves:

**Move 1 — order the factors by expected effect size and cost, and go one at a time.** From experience the ordering is roughly: reranker on/off (largest single effect, often +0.10 to +0.20 recall@5) → hybrid on/off (large when your corpus has identifiers) → chunk size (moderate, corpus-dependent) → top-k (moderate, interacts with the generator) → embedding model (small once a reranker is in place) → hybrid weight (smallest; RRF removes the need to tune it at all). Fix everything at a sane default, sweep one factor, keep the winner, move on. This is coordinate descent, it costs O(sum) instead of O(product), and it is correct as long as the factors are not strongly interacting.

**Move 2 — name the interactions you refuse to ignore and grid *only* those.** Two matter in practice: **chunk size × top-k**, because they trade off against a fixed token budget (256-token chunks at k=20 and 1024-token chunks at k=5 are the same 5k tokens and are genuinely different systems), and **reranker × retrieval k**, because a reranker's value depends entirely on how deep a pool you give it. Those are two 3×3 grids, 18 cells, not 360.

**Move 3 — matched budgets, which is the rule that makes the comparison honest.** Every cell must be compared at equal *context tokens* and equal *latency*. Otherwise you are not comparing chunking strategies, you are discovering that more context is better, which you already knew. Concretely: define the budget (say 6,000 context tokens and a 900 ms retrieval p95) and set each cell's `k` to fill the budget. A 256-token chunking scheme gets k=23; a 1,024-token scheme gets k=5. Now the comparison means something.

**Move 4 — stage-gate the metrics.** Run every cell on cheap retrieval metrics first (free, seconds). Take the top 3–5 cells only, and run those on end-to-end judged accuracy. You spend LLM money on five configs instead of 360.

**💰 Math:** naive full grid: 360 cells × 200 queries × 1 generation + 1 judge call ≈ 144,000 calls. At ~$0.02/call that is **$2,880 and, at 2 s/call with concurrency 20, about 4 hours of wall clock** — plus the reindexing, which is the real killer: each chunk-size change requires re-chunking and re-embedding the whole corpus. Coordinate descent with staged metrics: ~25 retrieval-only cells (free, minutes) + 5 end-to-end cells × 200 × 2 calls = 2,000 calls ≈ **$40 and 15 minutes**, with 3 reindexes instead of 3 (chunk size is the only factor that forces one — so pin chunk-size sweeps to one reindex per size and sweep everything else on top of each).

**⚠ Trap:** re-chunking without re-labeling and then reporting recall. If your golden labels are chunk IDs and you changed the chunking, the labels no longer refer to anything. You must define golden labels at the level of **character spans in the source document**, not chunk IDs, and compute "was the golden span covered by any retrieved chunk" at scoring time. Set your golden set up this way from day one or every chunk-size ablation you ever run is invalid. This is the single most common silent methodology error in RAG evaluation and it is worth saying out loud in an interview.

### How do you choose top-k? I want the method, not a number.

Mental model: **top-k is not a retrieval parameter, it is a budget allocation.** You have a context budget in tokens and a latency budget in milliseconds, and `k` decides how you spend both. So the question "what is the right k" is malformed until you have named the budget, and naming the budget is most of the answer.

There are two distinct `k`s and conflating them is the most common error. **`k_retrieve`** is how deep you go before reranking — 50 to 200. Its only job is recall, it is cheap (ANN search is sublinear, BM25 is an index scan), and you tune it by finding the depth at which recall stops improving. If recall@100 = 0.94 and recall@200 = 0.945, retrieving 200 buys 0.005 recall for double the reranker cost, and you stop at 100. **`k_final`** is how many chunks enter the prompt — 3 to 10. Its job is precision under a token budget, and it is the one that trades against generation quality non-monotonically.

The method for `k_final`, run as an ablation at a **matched token budget**:

Sweep `k_final` ∈ {3, 5, 8, 12, 20} and measure three things per cell: recall (monotonically increasing, guaranteed), end-to-end answer correctness (usually peaks and then declines), and cost/latency (linear in tokens). The peak in end-to-end correctness is real and it has a mechanism: each additional chunk adds some probability of containing needed evidence and some probability of being a distractor that pulls attention away from the right chunk. Past the crossover, distractors dominate. Where the crossover sits depends on your reranker quality — a good reranker pushes it right because chunk 8 is still decent; a bad one pushes it left because chunk 4 is already noise.

Then slice it, because a single `k` is usually wrong. Lookup queries ("what is the retry limit") need `k=3`; synthesis and comparison queries need more; enumeration queries need a different retrieval strategy entirely and `k` cannot save them. **A query-intent router choosing between two or three `k` values routinely beats the best single global `k`**, and it is a cheap classifier, not another LLM call.

**💰 Math:** going from k=5 to k=12 at 400 tokens/chunk adds 7 × 400 = 2,800 input tokens per request. At $3/Mtok that is 2,800 × 3/1e6 = $0.0084/request; at 200k requests/day, 200,000 × 0.0084 = **$1,680/day, ~$50,400/month.** If your ablation shows end-to-end correctness at k=12 is *lower* than at k=5 — which it often is — you would be paying fifty thousand dollars a month for worse answers. This is why "just raise k to be safe" is the most expensive four words in RAG.

**⚠ Trap:** measuring the `k` sweep on recall only. Recall is monotone in `k`, so a recall-only sweep always concludes "bigger is better" and you will ship k=20. The sweep must be scored end-to-end, or at minimum on an answer-sufficiency metric with a distractor penalty. This is the single clearest case in RAG where optimizing the proxy metric actively harms the system.

### Walk me through RAGAS. What does each metric actually compute?

**📄 Paper:** Es, James, Espinosa-Anke & Schockaert (2023) — *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. Its contribution is a decomposition: instead of one "is the answer good" score, it splits RAG quality along the axes that map onto your pipeline stages, and computes each with LLM-based judging so you do not need human labels for all of them. The library has evolved well past the paper; treat the paper as the framing and the current docs as the API. 📅 Volatile: metric names and implementations have changed across versions — verify against the version you pin.

The core four, and crucially **which inputs each needs**, because that determines which you can run in production versus only offline:

**Faithfulness** — needs (answer, retrieved context). No ground truth. Decompose the answer into atomic claims, then for each claim ask whether it is inferable from the retrieved context. Score = supported claims / total claims. This is the **hallucination detector**, and because it needs no labels, it is the one metric you can run on live production traffic. That property makes it the most operationally valuable of the four.

**Answer relevancy** — needs (question, answer). Judges whether the answer addresses the question, typically by generating N questions *from* the answer and measuring their embedding similarity to the original question. It penalizes evasive, padded, or off-target answers. Note what it does *not* do: it does not check truth. A confidently wrong answer that directly addresses the question scores high.

**Context precision** — needs (question, retrieved contexts, and a reference). Measures whether the *relevant* items are ranked at the top of the retrieved set. Effectively a rank-aware precision over the context list: signal-to-noise in what you put in the prompt.

**Context recall** — needs (retrieved contexts, ground-truth answer). Decompose the reference answer into claims and ask, for each, whether it can be attributed to the retrieved context. This is "did retrieval bring back everything the correct answer needed," which is exactly the FP1/FP2/FP7 axis, and it is the metric that requires reference answers and therefore real labeling work.

**Noise sensitivity** — measures how often the system produces incorrect claims when irrelevant or distracting documents are present in the retrieved set. It is the direct measurement of the FP4 distractor failure and the empirical counterweight to "just raise k."

The mapping that makes this useful in a room: **context recall and context precision diagnose retrieval; faithfulness and noise sensitivity diagnose generation; answer relevancy diagnoses the prompt and format.** Faithfulness low + context recall high = the generator is inventing despite having the evidence. Faithfulness high + context recall low = the generator is faithfully answering from insufficient evidence, which produces confidently incomplete answers.

**⚠ Trap:** treating faithfulness as correctness. **A perfectly faithful answer to a wrong retrieved document is a wrong answer with a perfect faithfulness score.** Faithfulness measures grounding in the provided context, full stop. If your retrieval surfaced the deprecated v1 policy, faithfulness is 1.0 and the user is misinformed. You need context recall or an end-to-end correctness metric to catch that, and if you report only faithfulness to a stakeholder you are actively misleading them.

### Which of those metrics would you gate CI on, and which would you never gate?

Gating is a different decision from measuring, and the difference is **variance and actionability**. A gate must be low-variance enough that a passing build passes reliably, and specific enough that a red gate tells someone what to fix. Otherwise you have built a flaky test, and flaky tests get disabled within three weeks — which is worse than not having them.

**Gate hard, blocking merge:**

*Retrieval metrics — recall@k, nDCG@k, per-hop retention.* Zero LLM calls, fully deterministic given a pinned index, run in seconds, and directly attributable. A drop in recall@50 is a specific, debuggable fact. This is my primary gate and it catches most real regressions, because most real regressions are ingestion or retrieval changes.

*Structural/contract assertions.* Every citation resolves to an existing chunk ID; every quoted span matches the source text at the claimed offsets; JSON validates against the schema; no answer exceeds the token budget; abstention triggers on the unanswerable slice. These are **boolean and deterministic** — proper tests, not metrics. Any failure blocks.

*Cost and latency budgets.* Mean tokens per request and p95 retrieval latency, with a hard ceiling. A prompt change that adds 3k tokens to every request must not merge silently.

**Gate softly — warn, require an ack, do not block:**

*Faithfulness on a fixed set with a pinned judge model.* It is a judge-model output and therefore has a noise floor; I measure that floor by running the identical config twice and use ~2× the observed run-to-run delta as the alert threshold. Judge model version pinning is mandatory, and a judge upgrade is itself a change that requires re-baselining every historical number.

**Never gate:**

*Answer relevancy* as a merge blocker — too vague, too correlated with style, and it moves when you change the prompt's tone, which is not a regression.

*End-to-end judged correctness on small sets* as a hard gate. With n=100 and a ±0.09 CI, a gate at "must not drop by 3 points" will fire on noise roughly a third of the time. Run it nightly against a larger set, track the trend, and gate on the trend crossing a band — not on a single run.

*Anything whose failure message does not tell an engineer where to look.* That is my actual test for whether something should be a gate.

The structure I ship: **PR gate** = retrieval metrics + structural assertions + budgets, target under 3 minutes, zero LLM cost. **Nightly** = full end-to-end with judges on 500 queries, posted as a trend with CIs. **Pre-release** = human review of 50 sampled outputs, which nothing replaces.

**🗣 Say this in the room:** "I gate merges on deterministic things — retrieval recall against a pinned index, citation-offset validity, token budget — because those fail loudly and point at a line of code. Judged metrics run nightly as a trend with confidence intervals. A judged metric as a blocking gate becomes a flaky test, and a flaky test becomes a disabled test."
