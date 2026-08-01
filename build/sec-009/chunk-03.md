### Derive precision, recall and F1 for me, and then tell me which one you'd optimize for a moderation system.

Everything comes from the 2×2 confusion matrix, and the trick to never confusing the two is to notice what sits in the denominator. **Precision = TP/(TP+FP)** — the denominator is everything you *flagged*, so precision answers "when I raise my hand, how often am I right?" **Recall = TP/(TP+FN)** — the denominator is everything that was *actually positive*, so recall answers "of the things I should have caught, how many did I?" Precision is about the cost you impose on the innocent; recall is about the harm you let through.

They trade off through a single knob, the decision threshold, and that is the whole game. Lower the threshold and you flag more things: recall rises monotonically, precision generally falls. There is no model change involved — the same trained model gives you the entire curve. **F1 is their harmonic mean**, `2PR/(P+R)`, and it is harmonic rather than arithmetic precisely so that a degenerate point cannot game it: a classifier with precision 1.0 and recall 0.01 has arithmetic mean 0.505 but F1 = 2(1.0)(0.01)/1.01 = 0.0198. The harmonic mean is dominated by the smaller term, which is what you want from a summary statistic.

Now the moderation question, and the honest answer is that **F1 is the wrong objective for moderation** and I would say so. F1 asserts that a false positive and a false negative cost the same, and in moderation they emphatically do not — but which way they differ depends entirely on the *action*, and that is the thing to ask about.

The framework I use: decompose by action tier. For an **auto-delete** action, a false positive silences a legitimate user with no recourse — that is a high-cost error, so you want high precision, threshold set to something like precision ≥ 0.95 measured on a held-out set, and you accept whatever recall that buys. For **enqueue-for-human-review**, a false positive costs one reviewer-minute, so you want high recall and you tune to fill exactly the review capacity you have. For **shadow-limit or demote**, you are somewhere in between. So one model, three thresholds, chosen from three different cost structures. If you use `Fβ` at all, use β > 1 (recall-weighted) for the review queue and β < 1 for auto-action, and say why.

**🗣 Say this in the room:** "I wouldn't optimize F1 — it assumes symmetric error costs and moderation isn't symmetric. I'd ask what action fires at each threshold, then set the auto-action threshold by a precision floor and the human-review threshold by the reviewer capacity we can staff. Same model, two operating points, both chosen from cost rather than from argmax F1."

**⚠ Trap:** reporting precision and recall without stating the threshold they were computed at. Those numbers are meaningless without it — they describe a point on a curve, and anyone can pick a flattering point. A comparison of two models at two different, unstated thresholds is not a comparison at all.

### Macro versus micro averaging — when does that choice actually change your conclusion?

The mental model: micro-averaging pools every prediction across all classes into one confusion matrix and computes the metric once. Macro-averaging computes the metric per class and then takes an unweighted mean. Micro therefore weights each *example* equally; macro weights each *class* equally. When your classes are balanced, they are close. When your classes follow a power law — which in every real product they do — they can differ by thirty points and tell opposite stories.

Concretely. Ten support-ticket classes, one of which is 70% of volume. Your model gets 0.95 F1 on the dominant class and 0.30 F1 on the nine tail classes. Micro-F1 (which for single-label multiclass equals accuracy) is dominated by the head: roughly 0.70×0.95 + 0.30×0.30 ≈ 0.76. Macro-F1 is (0.95 + 9×0.30)/10 = **0.365**. Same model, same predictions. Micro says "shipping"; macro says "the tail is broken."

The decision rule I use: **micro when your objective is aggregate throughput and errors on rare classes genuinely matter less** — total tickets correctly routed, total revenue affected. **Macro when the rare classes matter as much as the common ones per-instance** — which is the case for anything safety-related, anything where a rare class is high-severity, and any benchmark where you are trying to demonstrate broad competence rather than exploiting a skewed prior. Weighted-macro (macro weighted by support) is a middle option that mostly reproduces micro and I find it obscures more than it reveals.

**⚠ Trap:** reporting one number without the per-class table. Any single aggregate can hide a class at 0.00 F1 — a class the model has literally never predicted. I require a per-class precision/recall/support table in every classification report I review, and the first thing I look at is the *support* column, because a class with 11 examples in the test set has an F1 whose confidence interval is enormous and should not drive any decision.

**🗣 Say this in the room:** "I report both plus the per-class table. Micro tells me how the system performs on a random ticket; macro tells me whether the tail classes work at all. If they diverge by more than about ten points, that gap is the actual finding and I'd lead with it."

### Someone brings me a moderation classifier with 0.94 ROC-AUC on a dataset with 1% positives. What's your reaction?

My reaction is that the number is nearly uninformative and I would ask for the precision-recall curve before I read anything else. This is one of the sharpest tells in an applied-ML interview, both to give and to receive.

Here is the mechanism. ROC plots true positive rate (= recall) against **false positive rate = FP/(FP+TN)**. That denominator is the negative class, which under 1% prevalence has 99,000 members per 100,000 examples. So a model can produce 990 false positives and still have an FPR of only 0.01 — the curve barely moves. But those 990 false positives are being compared against at most 1,000 true positives, so precision is around 50% at best. **ROC-AUC is insensitive to class imbalance by construction, because both of its axes are normalized within a class**, and that insensitivity is exactly why it flatters imbalanced problems. A random classifier gets 0.5 ROC-AUC regardless of prevalence, which sounds like a virtue and is actually the problem: the metric has been designed to not tell you about the thing you care about.

Precision-recall does not have this property. Precision's denominator (TP+FP) mixes both classes, so the PR curve moves sharply when your false positives start to swamp a small positive class. The baseline for PR-AUC is the prevalence itself — 0.01 here — so a PR-AUC of 0.35 is a 35× lift over random and is genuinely impressive, while a PR-AUC of 0.04 is nearly worthless even though the same model might show 0.94 ROC-AUC.

**📄 Paper:** Davis & Goadrich (2006), "The Relationship Between Precision-Recall and ROC Curves" — establishes that a curve dominating in ROC space dominates in PR space and vice versa, but that PR space is far more discriminative when the negative class is large; this is the canonical citation for "use PR under imbalance."

**💰 Math, so the point lands:** 1M posts/month, 1% positives = 10,000 true violations. Take an operating point with recall 0.90 and FPR 0.03 — that FPR looks tiny and gives a great-looking ROC. False positives = 0.03 × 990,000 = **29,700**. True positives = 9,000. Precision = 9,000/38,700 = **0.233**. So three out of four flags are wrong. At $0.40 per human review, 38,700 flags = **$15,480/month** to catch 9,000 violations, and if this were auto-action instead of review you would be wrongly punishing 29,700 users a month. The 0.94 ROC-AUC told you none of this. The PR curve tells you all of it in one glance.

**🗣 Say this in the room:** "ROC-AUC normalizes false positives by the size of the negative class, so at 1% prevalence it's insensitive to exactly the error that dominates your review cost. I'd want PR-AUC — whose random baseline is the prevalence, 0.01 here — plus precision and recall at the actual deployed threshold. The ROC number isn't wrong, it's just answering a question nobody asked."

### How do you pick the operating threshold, and who actually gets to decide it?

The threshold is a **product decision informed by a cost matrix, not a modeling decision**, and I hold that line firmly because the alternative — an engineer picking `argmax F1` on a validation set — silently encodes an assumption that false positives and false negatives cost the same, which nobody ever agreed to.

The mechanism. Assign a cost to each cell: `C_FP` and `C_FN` (and usually zero for correct predictions, though sometimes a true positive has a handling cost too). Expected cost at threshold `t` is `C_FP·FP(t) + C_FN·FN(t)`, both read off the validation set. Sweep `t` over a few hundred candidates, plot expected cost, take the minimum. That is the whole procedure, and it is ten lines.

```python
import numpy as np
def best_threshold(y_true, scores, c_fp=1.0, c_fn=10.0):
    ts = np.unique(scores)
    costs = [(c_fp * ((scores >= t) & (y_true == 0)).sum() +
              c_fn * ((scores <  t) & (y_true == 1)).sum(), t) for t in ts]
    return min(costs)[1]
```

Three constraints in practice. **Capacity constraints often dominate cost minimization:** if you have 12 reviewers who can process 400 items a day, your threshold is whatever produces 4,800 flags/day, full stop, and the modeling question becomes "maximize recall at fixed volume." Say this explicitly in a design round — it is the constraint that actually binds in every real moderation and fraud system I have seen. **Thresholds need per-segment treatment:** a single global threshold on a model whose score distribution differs by language or by user tenure will produce wildly different precision per segment, which is both an accuracy problem and a fairness problem. **Thresholds drift:** if the score distribution shifts, a fixed threshold changes its operating point silently. I prefer to specify the threshold as a *quantile of the recent score distribution* ("flag the top 0.8% of scores") when volume stability matters more than a fixed precision, and as an absolute value when precision stability matters more. You cannot have both, and picking which one you are defending is the senior move.

**⚠ Trap:** setting a threshold on uncalibrated scores. A gradient-boosted model's raw output is a score, not a probability, and `0.5` means nothing in particular. If a stakeholder is going to reason about "80% confident," you owe them calibration — Platt scaling (fit a one-dimensional logistic regression on held-out scores) or isotonic regression (non-parametric, needs more data, can overfit below a few thousand held-out examples). And re-fit the calibrator whenever you retrain, because calibration does not survive a model change.

### Implement Recall@k and MRR for a retrieval system, and tell me what each one hides.

These are the two metrics that decide whether your RAG system works, and they are ten lines each, so implement them yourself rather than pulling a dependency you cannot audit.

**Recall@k** = the fraction of relevant documents that appear in the top `k`. For the common RAG case where each query has exactly one gold document, this collapses to hit-rate@k — did we retrieve it at all. **MRR** = mean over queries of `1/rank_of_first_relevant`, with 0 for a miss. First result gives 1.0, second gives 0.5, third 0.333, tenth 0.1.

```python
def recall_at_k(retrieved, relevant, k):        # both are sets/lists of doc ids
    top = retrieved[:k]
    return len(set(top) & set(relevant)) / max(len(relevant), 1)

def mrr(all_retrieved, all_relevant, k=10):
    total = 0.0
    for ret, rel in zip(all_retrieved, all_relevant):
        for i, d in enumerate(ret[:k], start=1):
            if d in rel:
                total += 1.0 / i
                break
    return total / len(all_retrieved)
```

What each hides. **Recall@k is rank-blind inside the window**: position 1 and position 10 score identically. That is fine if your reader is an LLM that will attend over all 10 chunks equally — except it will not, because of well-documented positional effects where content in the middle of a long context is used less reliably than content at the edges. So a Recall@10 of 0.90 achieved with the gold chunk usually at position 9 produces a worse system than a Recall@10 of 0.85 with it usually at position 1, and Recall@10 cannot see the difference. **MRR is the mirror image**: it only looks at the *first* relevant document and is completely blind to everything after it. For a question requiring synthesis across three documents, MRR gives full credit for finding one of them.

The rule I enforce: for RAG, **report Recall@k at your actual context budget `k`, plus MRR, plus nDCG@k**, and treat a large gap between Recall@k and MRR as the specific signal that "the reranker is the problem, not the retriever." If Recall@20 is 0.94 and MRR is 0.31, your candidate generation is fine and your ordering is broken — that is a cross-encoder reranker's job, and it is a much cheaper fix than re-embedding your corpus.

**💰 Math:** that diagnostic has real money attached. Re-embedding a 5M-chunk corpus at 400 tokens/chunk is 2B tokens; at $0.02/Mtok that is 2,000 × $0.02 = **$40** of embedding plus a re-index and an alias swap, call it a two-day engineering project. Adding a cross-encoder reranker over the top 50 candidates is a hosted call at roughly $1–2 per 1,000 queries plus ~80–150 ms of latency. **📅 Volatile:** verify reranker pricing. If the metric gap says "ordering," the reranker is the right spend; if Recall@50 itself is low, no reranker will save you and you must fix retrieval.

### Derive nDCG with graded relevance, and tell me where it lies to you.

The mental model: Recall@k and MRR both treat relevance as binary and mostly ignore position. nDCG fixes both — it lets a document be *partially* relevant and it applies a smoothly decaying positional discount — which is why it is the standard metric in search and the right one for any retrieval system whose results a human or a model reads in order.

The derivation, in three steps.

**Gain.** Each document gets a graded relevance label, conventionally 0–3 (irrelevant / marginal / relevant / perfect). The gain is usually `2^rel − 1`, so grades map to 0, 1, 3, 7. The exponential is deliberate: it says a perfect document is worth much more than two marginal ones, which matches how people actually use search results. (A linear-gain variant exists; state which you are using, because the numbers are not comparable.)

**Discount.** Divide by `log2(i + 1)` at rank `i` (1-indexed): rank 1 divides by 1, rank 2 by 1.585, rank 3 by 2, rank 10 by 3.459. Slow decay — position 10 still gets 29% of position 1's credit — which is the right shape for a results page and arguably too generous for a 5-chunk LLM context.

`DCG@k = Σ_{i=1..k} (2^{rel_i} − 1) / log2(i + 1)`

**Normalize.** DCG is unbounded and not comparable across queries — a query with six perfect documents can score far higher than one with a single relevant document, even if both were ranked perfectly. So divide by the **IDCG**: the DCG of the ideal ranking, obtained by sorting that query's judged documents by relevance descending. `nDCG@k = DCG@k / IDCG@k ∈ [0, 1]`, and 1.0 means "you produced the best possible ordering of the documents that were judged."

**📄 Paper:** Järvelin & Kekäläinen (2002) — introduced cumulated gain, discounted cumulated gain and its normalized form as the standard for graded-relevance IR evaluation, replacing binary precision-at-k as the field's default.

```python
import numpy as np
def ndcg_at_k(rels, k=10):                     # rels: graded labels in ranked order
    def dcg(r):
        r = np.asarray(r[:k], dtype=float)
        return float(((2 ** r - 1) / np.log2(np.arange(2, r.size + 2))).sum())
    ideal = dcg(sorted(rels, reverse=True))
    return dcg(rels) / ideal if ideal > 0 else 0.0
```

**⚠ Trap — where nDCG lies:** the normalization is over the *judged* documents only. If your relevance pool is incomplete — you judged the top 10 from your old system and are now evaluating a new system that surfaces a genuinely excellent unjudged document — that document scores as relevance 0, your new system looks worse, and you will reject a real improvement. This is the classic pooling bias in IR evaluation, and in an LLM-era retrieval eval it bites constantly because people build the judgment set from the current system's output. Mitigations: pool candidates from *every* system under comparison before judging, and re-judge whenever you change the retriever substantially. Second lie: nDCG@10 with an IDCG computed over only 2 judged relevant documents saturates at 1.0 trivially — always report the number of judged relevant documents per query alongside the score.

### MAP or nDCG for a RAG retrieval eval — which do you pick and why?

nDCG, in nearly every case, and I will give the decision rule rather than just the answer.

MAP (mean average precision) is built on binary relevance. For each query, average precision is `AP = (1/R) Σ_k P@k · rel_k` — walk down the ranking, and every time you hit a relevant document, record the precision at that position, then average those. It rewards packing all the relevant documents high. Mean it over queries and you have MAP. It is a genuinely good metric with one hard constraint: **relevance must be binary**, and you must know the total number of relevant documents `R` per query.

That constraint is what kills it for RAG. Chunk relevance is not binary. A chunk can contain the full answer, contain half of it, contain context that makes the answer interpretable, mention the topic but not the answer, or be a false lexical match. Forcing that onto {0,1} either inflates your relevant set with chunks that do not actually help the generator, or discards partial-credit information you paid annotators for. Graded relevance is the honest representation, and nDCG is the metric that consumes it.

The decision rule: **binary relevance and you care about finding all of them → MAP. Graded relevance, or you care most about the top few positions → nDCG. Exactly one gold document per query → MRR is sufficient and simpler, and MAP degenerates to MRR anyway in that case.**

But the more important thing to say in a RAG interview is that **all three are proxies for a downstream quantity you can measure directly**: does the generator produce a correct answer given the retrieved context? I run retrieval metrics because they are cheap, deterministic and diagnose *where* the pipeline broke — but the metric that gates a release is end-to-end answer correctness, with retrieval metrics as the explanatory layer underneath. The specific pairing I use is **"context sufficiency"** — a binary judgment of whether the retrieved set contains enough to answer — alongside nDCG. When end-to-end accuracy drops and context sufficiency stayed flat, the generator or the prompt broke. When sufficiency dropped, retrieval broke. That two-signal decomposition is worth more in a design round than the choice between MAP and nDCG.

**⚠ Trap:** optimizing retrieval metrics past the point where they affect the answer. If context sufficiency is already 0.97 at k=8, pushing nDCG@8 from 0.71 to 0.78 will not move end-to-end quality at all, and the time is better spent on the generation half. I have watched a team spend three weeks on reranking for zero product improvement because nobody measured the ceiling first.

### Walk me through BLEU, ROUGE, METEOR and chrF — what do they measure, and why did the field move past them?

These are the generation metrics the field grew up on, and you need them for three reasons: they still appear in papers, they are still correct for narrow tasks, and interviewers use them to check whether you understand *why* n-gram overlap fails.

**BLEU** — **📄 Paper:** Papineni et al. (2002). Modified n-gram precision: for n = 1..4, count how many of the candidate's n-grams appear in the reference, with each reference n-gram consumable only as many times as it appears (this "clipping" is what stops "the the the the" from scoring 1.0). Take the geometric mean of the four precisions, then multiply by a **brevity penalty** `exp(1 − r/c)` when the candidate is shorter than the reference, because precision alone rewards saying almost nothing. It is precision-oriented and corpus-level: BLEU on a single sentence is high-variance and nearly meaningless, which is a frequently-violated property.

**ROUGE** — **📄 Paper:** Lin (2004). The recall-shaped counterpart, built for summarization, where the question is "did you cover the reference content" rather than "was everything you said in the reference." ROUGE-N is n-gram recall; ROUGE-L uses the longest common subsequence, so it rewards in-order overlap without requiring contiguity.

**METEOR** — **📄 Paper:** Banerjee & Lavie (2005). Fixes BLEU's brittlest failure: exact string matching. It aligns unigrams allowing stem matches and synonym matches, computes an F-mean weighted toward recall, and applies a fragmentation penalty when the aligned words are scattered rather than contiguous. It correlated better with human judgment than BLEU at the sentence level, which was the point.

**chrF** — **📄 Paper:** Popović (2015). Character n-gram F-score instead of word n-grams. This matters enormously for morphologically rich languages (Finnish, Turkish, Hindi) where a single word carries what English spreads over four, so word-level overlap is punishingly sparse, and it sidesteps tokenization disagreements entirely.

Why the field moved on: **every one of them measures surface form, and the space of correct answers is enormous.** "The meeting was moved to Thursday" and "They rescheduled it for Thursday" share almost no n-grams and are the same answer. BLEU cannot tell you that. Worse, the failure is asymmetric — a *wrong* paraphrase with high lexical overlap scores well, and a *right* one scores badly, so the metric is actively anti-correlated with quality in exactly the region where you need discrimination.

Where I still use them, unapologetically: **machine translation with multiple references** (BLEU remains the accepted comparability standard, and chrF for non-English targets), **extractive summarization**, **any task where the output is genuinely constrained to near-copy** — SQL string comparison after normalization, structured field extraction — and as a **cheap regression tripwire**: if ROUGE-L against last week's outputs falls off a cliff, something changed, even if a high ROUGE tells you nothing.

**⚠ Trap:** BLEU scores are not comparable across papers unless the tokenization and smoothing match. This is exactly why `sacrebleu` exists — it fixes tokenization and emits a signature string describing the configuration. A BLEU number without that signature is not a number you can compare against anything.

### What did BERTScore fix, and what did it not fix?

**📄 Paper:** Zhang et al. (2020), BERTScore — replaces exact n-gram matching with greedy matching in contextual embedding space, so "moved to Thursday" and "rescheduled for Thursday" can score as similar. That is the fix, and it is a real one.

The mechanism, precisely: run both the candidate and the reference through a contextual encoder, producing one embedding per token. For **recall**, take each *reference* token and find its maximum cosine similarity to any *candidate* token; average. For **precision**, do it the other way — each candidate token to its best reference token. F1 is their harmonic mean. Optionally weight tokens by IDF so that content words count more than function words. Optionally rescale against a baseline computed on random sentence pairs, because raw cosine similarities between contextual embeddings live in a narrow band (roughly 0.8–1.0) and are hard to read — the rescaling spreads them out and is presentation, not substance.

What it did *not* fix, in order of severity:

**It does not detect factual error.** "The revenue was $4.2 million" and "The revenue was $7.8 million" are nearly identical in embedding space — same syntax, same entities, one number differs. BERTScore will rate that pair very highly. For anything where the value of the output is a *fact*, BERTScore is close to useless, and this is the single most important limitation to state.

**It still requires a reference.** For open-ended generation there is no reference, so the metric is inapplicable to most of what an LLM product actually does.

**It inherits the encoder's biases and its cutoff.** Scores depend on which model and which layer you used; comparing BERTScore across papers that used different backbones is meaningless, the same way comparing BLEU across tokenizations is.

**It is insensitive to logical negation and to coherence.** "The patient should not receive the drug" versus "The patient should receive the drug" is a one-token difference in a long sentence, and the score barely moves.

**🗣 Say this in the room:** "BERTScore fixed the paraphrase problem and nothing else. It still needs a reference and it's near-blind to factual substitution — swap a dollar figure and the score moves by a fraction of a point. For anything where correctness is a fact rather than a phrasing, I use an LLM judge with an explicit rubric, or better, an extraction-then-exact-comparison step where the fact is checkable."

The historical arc is worth having straight, because it explains the current state of eval: exact match → n-gram overlap → embedding similarity → model-as-judge. Each step traded determinism and cheapness for semantic sensitivity. LLM-as-judge is where that trade has landed, and it is the first metric in the sequence that needs its *own* evaluation against human labels — which is the whole reason evaluation became a discipline rather than a function call.

### Give me pass@k and its unbiased estimator, and explain why the obvious implementation is wrong.

The mental model: pass@k asks "if I let the model produce `k` independent attempts and a test suite checks each, what is the probability at least one passes?" It is the right metric for code generation because there is a **verifier** — you do not need a reference solution or a human judge, you need the tests to go green — and it reflects the real usage pattern where a developer or an agent gets more than one shot.

The naive implementation is: generate `k` samples, check if any passes, average over problems. That is unbiased but has enormous variance at small `k`, and the number swings wildly between runs, which makes it useless for comparing two models that differ by two points.

The fix, from **📄 Paper:** Chen et al. (2021), the Codex paper (*Evaluating Large Language Models Trained on Code*): generate `n > k` samples per problem, count `c` that pass, and compute the expected pass@k analytically:

```
pass@k = 1 − C(n − c, k) / C(n, k)
```

The reasoning is a one-line combinatorial argument you should be able to reproduce at a whiteboard. `C(n, k)` is the number of ways to choose `k` of the `n` samples. `C(n − c, k)` is the number of ways to choose `k` samples *all of which fail* — you are choosing entirely from the `n − c` failures. Their ratio is the probability that a random subset of size `k` contains no passing sample. One minus that is the probability at least one passes. Average over problems.

```python
import numpy as np
def pass_at_k(n, c, k):
    if n - c < k:                      # fewer than k failures => always at least one pass
        return 1.0
    # numerically stable product form; avoids overflow in binomials at large n
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

The product form is the implementation detail worth knowing: `C(n−c,k)/C(n,k) = Π_{i=n−c+1}^{n} (1 − k/i)`, which avoids computing huge binomial coefficients. Typical practice is `n = 200` with `k ∈ {1, 10, 100}`.

**⚠ Trap:** pass@k with `k > 1` is not a deployment metric unless you have a way to *select* which sample to ship. If a human or an agent reads all `k` candidates and picks, pass@10 is meaningful. If your product returns one completion, the honest metric is pass@1 at your deployment temperature, and quoting a model's headline pass@100 as evidence of production quality is a misuse I would push back on in review. Related: pass@1 depends on temperature, and a model tuned to look good at pass@100 (high temperature, high diversity) can look worse at pass@1. Always state `n`, `k`, and the sampling temperature together.

**💰 Math:** the estimator's whole purpose is variance reduction, and that costs generations. Evaluating a 164-problem benchmark at `n = 200` is 32,800 completions; at ~500 output tokens each that is 16.4M output tokens, so at $15/Mtok = **$246 per full evaluation run**, plus input. If you gate every pull request on that you are spending real money on CI, which is why the standard practice is `n = 200` for a published number and `n = 10` for a fast regression check — accepting the wider interval in exchange for a ~$12 run.

### When is exact match the right metric, and how does it lie?

Exact match is the strictest metric there is: normalize the string, compare, score 1 or 0. Its virtue is that it is completely deterministic, free to compute, and impossible to game — there is no judge to fool and no embedding to exploit. Its vice is that it punishes every acceptable variation.

Exact match is genuinely correct when **the answer space is small and canonical**: multiple-choice letters, yes/no, a classification label from a closed set, a numeric answer after unit normalization, a date after parsing, an enum in a structured-output field. For these, anything softer than exact match is a bug, because it admits partial credit for a wrong answer.

It becomes a liar the moment the answer is a *span of natural language*. Extractive QA is the classic case: gold answer "Paris", prediction "Paris, France" scores 0. Gold "1969", prediction "in 1969" scores 0. This is exactly why the SQuAD-lineage of QA benchmarks reports **both EM and token-level F1** — F1 gives partial credit on overlapping tokens and is the more informative of the two, while EM is the more honest floor. Report both or you are choosing which way to be wrong.

The normalization step is where most of the engineering lives and where most of the bugs are: lowercase, strip articles ("a", "an", "the"), strip punctuation, collapse whitespace, and — the one people forget — normalize numbers ("1,000" vs "1000" vs "1 thousand") and units. Every normalization you add makes the metric more permissive and therefore less comparable to anyone else's number, so pin the normalizer and version it with the eval set.

**⚠ Trap in the LLM era:** exact match against a model that was asked an open question will underreport catastrophically, because models pad. "The answer is Paris." scores 0 against "Paris". The correct fix is not a looser metric — it is **constraining the output format**: structured outputs, a JSON schema with an `answer` field, or a stop sequence, so that EM becomes applicable again. That is the general principle and it is worth stating explicitly: *when your metric is too strict for your output, tighten the output rather than loosening the metric.* Loosening the metric hides errors; tightening the output eliminates a class of them.

**🏋 Drill:** in 25 minutes, unaided, implement `precision/recall/F1` with macro and micro averaging, `recall_at_k`, `mrr`, `ndcg_at_k` with exponential gain, and `pass_at_k` with the stable product form — all in NumPy, no sklearn. Pass criterion: your macro-F1 matches `sklearn.metrics.f1_score(average="macro")` to 1e-9 on a random 5-class array, and your `pass_at_k(n=200, c=3, k=10)` matches a direct `scipy.special.comb` computation. If you cannot write nDCG's discount from memory, that is the one to drill again tomorrow.
