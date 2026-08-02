# PART XIII — Adjacent High-Comp Surfaces

A large share of the highest-paying "AI engineer" openings are recsys/search teams adding LLMs, or cost-pressured teams choosing small models. Ignoring these narrows the target list for no good reason.

## Contents

1. [71. Recsys and Search-Relevance Hybrids, On-Device, Small Models and Model-Choice Economics](#71-recsys-and-search-relevance-hybrids-on-device-small-models-and-model-choice-economics) — 47 questions


---

## 71. Recsys and Search-Relevance Hybrids, On-Device, Small Models and Model-Choice Economics

*Mastering this proves you can interview at Netflix, Uber, Airbnb, Meta ranking, Pinterest and Spotify — where the title says AI Engineer and the team is a ranking team.*

### Walk me through the architecture of a large-scale recommender. Why is it a funnel instead of one model?

The funnel exists for exactly one reason, and it is a reason you already know from database engineering: **you cannot afford to run an expensive scoring function over the whole corpus, so you build a cheap function that is allowed to be wrong in one direction only.** A retrieval stage is an index scan with a recall guarantee; a ranking stage is the expensive predicate you apply to the surviving rows. The same shape, the same reason.

Concretely, a production stack at Netflix/Pinterest/Meta scale has four stages. **Candidate generation (retrieval)** takes a corpus of 10⁷–10⁹ items down to ~1,000, in single-digit milliseconds, usually as a union of several independent sources: an ANN lookup on a two-tower embedding, a co-visitation / item-to-item table, a "trending in your locale" list, a subscription/follow graph, and a small exploration bucket. **Filtering** removes items that are ineligible for policy, availability, region, or already-seen reasons — this stage is boring and it is where most of your production incidents live. **Ranking** scores those ~1,000 with a heavy model that sees full cross-features (user × item × context) and produces calibrated predictions of multiple outcomes — p(click), p(watch>30s), p(complete), p(report) — which get combined into a single utility. **Re-ranking / policy** applies diversity (MMR-style), business rules, freshness boosts, creator-fairness caps, and slate-level constraints, over the top ~50.

The reason each stage exists is a latency budget. Retrieval over 10⁸ items with a model that takes 1 ms per item is 10⁵ seconds. Retrieval over 10⁸ items with a dot product against a precomputed 128-d vector inside an HNSW graph is ~2 ms because HNSW touches maybe 2,000 vectors, not 10⁸. Ranking 1,000 items with a 10 M-parameter DNN batched on one CPU/GPU node is ~15 ms. So total ~30 ms of model time inside a 150 ms page budget. **The funnel is a latency-shaped decomposition, not an accuracy-shaped one.**

**⚠ Trap:** believing the ranking model's job is to be accurate. Its job is to be accurate *on the distribution the retriever hands it*, and that distribution is not the corpus. If you improve the retriever, the ranker's training distribution silently shifts and its calibration degrades — this is the single most common cause of "we shipped a better retriever and engagement went down." The rule I enforce in review is: **retrieval and ranking changes ship in separate experiments, and a retrieval change triggers a ranker retrain before it goes to 100%.**

**🗣 Say this in the room:** "It's a funnel because retrieval must be O(log N) per query and ranking can be O(1000) per query. Retrieval optimizes recall@1000 with a cheap decomposable scorer; ranking optimizes calibrated multi-objective utility with full cross-features. The interesting failure mode is that they're coupled — the ranker is only trained on what the retriever surfaces, so retrieval changes are ranker distribution shifts."

**📐 Numbers you must know:** the shape of the funnel, memorized as orders of magnitude — corpus 10⁸ → retrieval 10³ → ranking 10³ scored → re-rank 10² → slate 10¹. Each arrow is roughly a 10–1000× reduction and roughly a 100× increase in per-item compute budget. If a candidate quotes you "we retrieve 50,000 candidates and rank all of them," ask about their p99; either they have a very small corpus or they have a very slow page.

### Derive the two-tower retrieval model for me. Why can't the towers see each other?

Start from the constraint and the architecture becomes inevitable. **You need to score a user against 10⁸ items in a few milliseconds. The only scoring function that admits a sublinear index is one that factorizes into f(user) · g(item) — a dot product or cosine over precomputed vectors.** Every architectural choice in a two-tower model is downstream of the requirement that the item side be computable *offline, without knowing the user*. That is the whole idea. The towers cannot see each other because if they could, you would have to recompute every item vector for every user, and you are back to a full scan.

Mechanism. The user tower consumes user ID embedding, recent-interaction sequence (usually pooled or attention-pooled), demographics, device, context (time of day, surface), and produces u ∈ ℝ^d, typically d = 64–256. The item tower consumes item ID embedding, content features (category, language, creator, text/image embeddings), and produces v ∈ ℝ^d. Score is s(u,v) = ⟨u, v⟩ / τ, with τ a learned or fixed temperature. Training is a softmax over the item vocabulary — which you cannot compute, so you approximate it with **sampled softmax over in-batch negatives**: for a batch of B (user, positive-item) pairs, treat the other B−1 items in the batch as negatives and minimize cross-entropy over those B logits.

Here is the loss, which you should be able to write from memory:

```python
import torch, torch.nn.functional as F

def two_tower_loss(u, v, log_q, tau=0.05):
    # u: (B, d) user vectors, v: (B, d) item vectors, aligned positives on the diagonal
    u = F.normalize(u, dim=-1)
    v = F.normalize(v, dim=-1)
    logits = (u @ v.t()) / tau            # (B, B); logits[i, i] is the positive
    logits = logits - log_q.unsqueeze(0)  # logQ correction: subtract log sampling prob
    target = torch.arange(u.size(0), device=u.device)
    return F.cross_entropy(logits, target)
```

That `log_q` term is the part people leave out and it is the difference between a working retriever and a popularity oracle — the next question is entirely about it.

**📄 Paper:** Covington, Adams, Sargin (2016) — "Deep Neural Networks for YouTube Recommendations" established the retrieval-plus-ranking split with a neural retriever trained as extreme multiclass classification, replacing hand-tuned co-visitation heuristics as the primary source. **📄 Paper:** Yi et al. (2019), "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" — the in-batch-negative logQ correction for streaming two-tower training.

**⚠ Trap:** adding a "small interaction layer" at the top of a two-tower model because it improves offline metrics. It does improve them, and it also destroys the entire reason the model exists — you can no longer precompute item vectors. If you want interaction, that is what the ranker is for. The legitimate middle ground is a *late-interaction* model (multi-vector, ColBERT-style) which keeps the index but multiplies its size by the number of vectors per item; budget for that explicitly rather than discovering it at index-build time.

### Your two-tower model is trained with in-batch negatives and it recommends nothing but the most popular items. What's wrong and how do you fix it?

This is the canonical two-tower bug and it has a clean mathematical cause: **in-batch negatives are sampled proportional to item popularity, because your training batches are drawn from the interaction log, and popular items appear in the log more often.** The model is therefore trained to discriminate against a popularity-weighted negative distribution, which means the softmax it learns is not p(item | user) but something closer to p(item | user) / p(item)^α — except the gradient pushes in the wrong direction. Popular items appear as negatives so often that the model over-corrects on them in a way that *still* leaves them dominating at inference, because at inference you score against the uniform corpus, not against the batch distribution.

The fix is the **logQ correction**: subtract log q(item) from each logit before the softmax, where q(item) is the probability that item was sampled as an in-batch negative. In a streaming setting you estimate q with a count-min-sketch-style running frequency keyed on item ID with exponential decay — Yi et al. (2019) do exactly this. After correction, the logits approximate the *unbiased* softmax over the corpus.

There are three more levers, and a senior answer names all four:

1. **logQ / sampled-softmax correction** — mandatory, not optional. This is the one that matters most.
2. **Mixed negatives** — union of in-batch negatives with uniformly-sampled negatives from the full corpus. In-batch negatives are "hard-ish and popular"; uniform negatives teach the model the shape of the tail. A typical mix is 8,192 in-batch plus 8,192 uniform.
3. **Hard-negative mining** — mine items the current model ranks highly but the user did not engage with. Powerful and dangerous: too many hard negatives and you train on false negatives (items the user would have liked but never saw), which actively teaches the model to suppress good recommendations.
4. **Temperature** — τ too high flattens the softmax and everything collapses toward the popularity prior; τ too low makes training unstable. τ ≈ 0.05–0.1 on L2-normalized vectors is the usual operating point.

**🔍 Failure taxonomy — "my retriever only returns head items":** (a) Check the logQ term exists at all — 60% of the time this is it. (b) Plot the empirical distribution of retrieved item popularity percentile against the corpus; if retrieval is concentrated above the 95th percentile of popularity while the ground-truth positives are spread out, it is a sampling-bias problem, not a capacity problem. (c) Check whether the item tower has *any* content features — a pure-ID item tower cannot generalize to items with few interactions, so the tail is untrainable by construction. (d) Check the ANN index, not the model: if `efSearch` is too low or the index was built on a stale embedding version, retrieval looks popularity-biased because the graph's entry points are the densest region.

**⚠ Trap:** "false negatives are rare so I'll ignore them." In a corpus of 10⁸ with a user who engages with 100 items, the false-negative rate among uniform negatives is ~10⁻⁶ and genuinely ignorable. Among *mined hard negatives* it can be 5–30%, because the mining criterion is literally "items the model thinks the user would like." Hard-negative mining without a false-negative filter (e.g. dropping any mined negative above a similarity threshold to a known positive) is how teams train a retriever that gets *worse* with more mining rounds.

### How do you actually serve a two-tower model, and what breaks between training the encoder and answering a query?

Serving splits along the same seam as the architecture. The **item tower runs as a batch job** — usually a nightly or hourly Spark/Beam pipeline that encodes every item and writes vectors into an ANN index (HNSW, IVF-PQ, ScaNN, or a managed vector store). The **user tower runs online** at request time, in under ~5 ms, from features fetched out of an online feature store. The query is then an ANN top-k against the index.

Four things break, in descending order of how often I have seen them cause an incident.

**One: version skew between the two towers.** The user tower and item tower are one model; their embedding spaces are only comparable if they came from the same training run. If you redeploy the user tower but the index still holds vectors from last week's item tower, the dot products are meaningless — and critically, *nothing errors*. Recall drops, relevance degrades, and your dashboards show a slow quality slide with no exception in the logs. **The fix is a version tag baked into both the served user-tower artifact and the index metadata, with the serving layer refusing to query an index whose tag does not match.** This is the vector-store analogue of a schema migration: you dual-write two indexes, shift traffic via an alias, and only then drop the old one.

**Two: item-vector staleness.** An item added at 09:00 is not retrievable until the 02:00 batch runs. For a marketplace or a news feed that is a 17-hour cold-start hole. The mitigation is a streaming item encoder for new/changed items writing into a small "fresh" index queried in parallel with the main index, results merged. Budget the second ANN call: two 2 ms lookups in parallel is still 2 ms, in sequence it's 4 ms.

**Three: feature skew on the user tower.** The training pipeline computed `days_since_last_session` from a warehouse table with full history; the serving path computes it from a Redis key that expires after 30 days. Same name, different semantics, and the model silently degrades for exactly the users who matter most (returning-after-a-lapse). This is the general training/serving skew problem and it deserves its own answer.

**Four: ANN recall is not 1.0 and you did not measure it.** HNSW with `efSearch=64` might give you recall@100 of 0.92 against exact search. That 8% is not evenly distributed — it disproportionately drops items in sparse regions of the embedding space, i.e. the tail, i.e. exactly the items your diversity metrics depend on.

**💰 Math:** suppose 10⁸ items × 128 dims × 4 bytes (fp32) = 51.2 GB of raw vectors. That does not fit comfortably in RAM on one box alongside an HNSW graph, which itself costs roughly `M × 2 × 4 bytes` per node per layer-0 link set — at M=32 that is ~256 B/item = 25.6 GB more. Total ~77 GB. Switch to fp16 vectors (25.6 GB) or PQ-compress to 32 bytes/vector (3.2 GB + graph). **The decision rule: if the corpus × dim × 4 exceeds about 60% of a single node's RAM, you are choosing between sharding (adds a fan-out and a merge, ~1–2 ms p99) and quantization (costs recall, measurable). Measure recall@k against a brute-force ground truth on 10,000 sampled queries before and after — if you cannot produce that number, you do not know what your retriever's ceiling is.**

### Take me from logistic regression to a modern CTR ranker. What did each step actually buy?

Every step in this lineage is a different answer to one question: **how do you represent feature interactions when your features are billions of sparse categorical IDs?** That is the whole story of CTR modeling, and if you tell it that way an interviewer will follow you anywhere.

**Logistic regression on one-hot features.** p(click) = σ(w·x). It handles high-cardinality sparsity beautifully and it is perfectly calibrated by construction when trained with log loss. What it cannot do is represent "this user likes this *category* on *mobile* in the *evening*" — an interaction of three features. You get interactions only by hand-crafting crossed features, which is what ad teams did for a decade and which explodes combinatorially and never generalizes to unseen crosses.

**Factorization Machines** (Rendle, 2010) fixed the generalization problem: give every feature value a latent vector, and model the pairwise interaction between features i and j as ⟨v_i, v_j⟩. Now a cross you never saw in training still gets a score, because both sides have learned vectors. **Field-aware FM** (Juan et al., 2016) gave each feature a *different* vector per interacting field, which won several Kaggle CTR competitions and costs F× the parameters.

**GBDT + LR** (He et al., 2014, "Practical Lessons from Predicting Clicks on Ads at Facebook") took a different route: train gradient-boosted trees on dense features, use the leaf indices as new categorical features, feed those into a logistic regression trained online. The trees discover the interactions; the LR keeps the calibration and the online updatability. This still works and I would not be embarrassed to ship it for a small-corpus ranking problem.

**Wide & Deep** (Cheng et al., 2016) said: do both. A wide linear part on crossed features for memorization, a deep MLP on embeddings for generalization, summed before the sigmoid.

**DCN / DCN-v2** (Wang et al., 2017; 2020) replaced the hand-crafted wide part with an explicit **cross network** that computes bounded-degree polynomial interactions in a fixed number of parameters: x_{l+1} = x₀ ⊙ (W_l x_l + b_l) + x_l. Layer l gives you degree-(l+1) interactions. This is the piece I would defend as the best accuracy-per-parameter step in the lineage.

**DLRM** (Naumov et al., 2019) is Meta's production shape: sparse features → embedding tables; dense features → bottom MLP; explicit pairwise dot products between all embedding pairs; concatenate; top MLP. It is architecturally simple and its entire engineering difficulty is that the embedding tables are terabytes.

**📐 Numbers you must know:** embedding table size = (number of distinct IDs) × d × bytes. A user-ID table with 2×10⁹ users at d=64, fp32, is 2e9 × 64 × 4 = **512 GB** — one table, larger than any single GPU's HBM by two orders of magnitude. This one calculation is why industrial recsys uses model-parallel sharded embedding tables (TorchRec-style), why hashing tricks and frequency-based ID pruning exist, and why "just fine-tune an LLM to do ranking" is not a serious proposal at this scale. **The compute is trivial; the memory and the all-to-all communication of embedding lookups is the entire systems problem.**

**⚠ Trap:** carrying a transformer intuition into ranking and assuming bigger = better. CTR models are memory-bound lookup machines with a tiny MLP on top; scaling laws that hold for language do not hold here, and a 10× parameter increase that is all embedding table typically buys you a fraction of a percent AUC while multiplying your serving cost and your training all-reduce volume. The wins in this domain come from **features and freshness**, not depth.

### Your CTR model has great AUC but the predicted probabilities are all wrong. Why, and how do you fix it?

AUC is invariant to any monotone transformation of your scores. It measures ranking, not calibration. So a model that outputs 0.9 for every positive and 0.8 for every negative has AUC 1.0 and is catastrophically miscalibrated. **In a recommender that only picks the top-k, miscalibration is survivable. In anything where the score is multiplied by a real number — ad auctions (bid × pCTR), expected-revenue ranking, budget pacing, a multi-objective utility that sums p(click) + λ·p(purchase) — miscalibration is a correctness bug, and it is invisible to AUC.**

The two dominant causes:

**Negative downsampling.** You train on all positives and a w-fraction of negatives because the raw dataset is 99.8% negatives. That shifts the base rate, so predictions are inflated by exactly a known factor. The correction (He et al., 2014) is
```
p = p' / (p' + (1 - p') / w)
```
where p' is the model's output and w is the negative sampling rate. If you downsampled negatives to w = 0.01 and the model outputs p' = 0.5, then p = 0.5 / (0.5 + 0.5/0.01) = 0.5/50.5 = **0.0099**. Get this formula wrong in an ads interview and the round ends.

**Distribution shift between train and serve.** Trained on last month's traffic, serving today's. Or trained on data logged by the previous ranker (see the position-bias answer). Or a new surface launched and now 30% of impressions come from a context with a different base rate.

The fixes, in the order I apply them: (1) analytically correct for known sampling; (2) fit a **calibration layer** on a held-out recent window — Platt scaling (a 2-parameter logistic on the logit) if you have little data, **isotonic regression** if you have a lot and expect a non-monotone-in-logit distortion; (3) fit calibration *per segment* (per surface, per country, per device) because a single global calibrator will be right on average and wrong everywhere; (4) recalibrate continuously — the calibrator is cheap and should be refit hourly on the freshest labels even when the base model is refreshed weekly.

Measure it with **Expected Calibration Error** (bucket predictions, compare mean prediction to empirical rate per bucket, average weighted by bucket size) and with a plain **calibration plot**, plus the single most useful production check: **sum of predicted probabilities over an hour of traffic vs actual number of clicks in that hour.** If your model predicts 41,000 clicks and you got 33,000, you are 24% hot and every downstream multiplication is wrong by 24%.

**🗣 Say this in the room:** "AUC is invariant to monotone transforms, so it can't see calibration at all. If the score gets multiplied by anything downstream — a bid, a revenue estimate, a multi-objective weight — I need calibrated probabilities, so I'd correct analytically for negative downsampling, fit isotonic regression per segment on a recent holdout, and monitor predicted-clicks-vs-actual-clicks per hour as the production alarm."

**💰 Math:** in an ad system ranked by bid × pCTR, a uniform 20% pCTR inflation is harmless — it cancels in the ranking. A *segment-dependent* 20% inflation is not: if mobile pCTR is 20% hot and desktop is accurate, mobile ads win auctions they should lose, you under-deliver revenue on the true winners, and at $50 M/quarter of ad revenue a 2% efficiency loss from misallocation is **$1 M/quarter** — from a bug that every offline metric on your dashboard reports as fine.

### Explain LambdaMART. Why does a gradient-boosted tree still win ranking competitions?

The mental model is a trick, and once you see the trick the algorithm is obvious. **The metric you care about — NDCG — is flat almost everywhere and discontinuous where it isn't, so it has no useful gradient. LambdaMART's move is to skip defining a loss and instead define the gradient directly, as a pairwise force weighted by how much swapping that pair would change NDCG.** You never write down the objective. You write down the derivative you wish it had.

Mechanism. For a query with documents i and j where i is more relevant than j, RankNet gives a pairwise cost whose gradient magnitude is λ_ij = −σ / (1 + e^{σ(s_i − s_j)}). LambdaRank multiplies that by |ΔNDCG_ij| — the change in NDCG if you swapped positions i and j. Then each document's total lambda is λ_i = Σ_{j: i≻j} λ_ij − Σ_{j: j≻i} λ_ji. Those λ_i are treated as the negative gradients that MART (gradient-boosted regression trees) fits at each boosting round, with a Newton step for the leaf values. That is it: **MART with a hand-specified gradient that encodes "misordering a pair near the top hurts more than misordering a pair at position 40."**

**📄 Paper:** Burges (2010), "From RankNet to LambdaRank to LambdaMART: An Overview" — the definitive write-up of the lineage. LambdaMART won the Yahoo! Learning to Rank Challenge (2010) and its descendants (LightGBM's `lambdarank` objective, XGBoost's `rank:ndcg`) are still the default for feature-based ranking.

Why trees still win: ranking features are **heterogeneous dense tabular numbers** — BM25 score, click-through history, freshness in hours, price, a dozen embedding similarities, counts with heavy tails. Trees are scale-invariant, handle monotone-but-nonlinear relationships without feature engineering, handle missing values natively, need no normalization, train on CPU in minutes, and give you feature importances your PM can read. A neural ranker beats them only when you have (a) a lot of data, (b) raw sequence or text inputs where representation learning matters, or (c) you need to share a representation across tasks. **On a few hundred dense features and a few million labeled query-document pairs, LambdaMART is still the model to beat and I would open a ranking design by proposing it, not apologizing for it.**

**⚠ Trap:** treating listwise vs pairwise as a quality ordering. Listwise (ListNet, ListMLE, softmax-over-list losses) is not automatically better; it is better when your labels are *graded and complete per query* and worse when your labels are sparse binary clicks with heavy position bias, because a listwise loss then confidently learns the logging policy's ordering. The decision rule I use: **graded human judgments → listwise or LambdaMART; sparse implicit feedback → pairwise with propensity weighting; single binary outcome with a downstream multiplication → pointwise with calibration.**

**🏋 Drill (25 min, unaided):** implement `delta_ndcg(labels, i, j)` and the lambda accumulation loop for one query in NumPy — no library. Pass criterion: for a 5-document query with graded labels [3,2,3,0,1], your λ vector sums to approximately zero, the document at rank 1 has the largest |λ| when it is mis-ranked, and swapping ranks 4 and 5 produces a |ΔNDCG| at least 5× smaller than swapping ranks 1 and 2.

### Write down NDCG and then tell me when it lies to you.

DCG at cutoff k is Σ_{i=1}^{k} (2^{rel_i} − 1) / log₂(i + 1), and NDCG@k is that divided by the DCG of the ideal ordering of the same query's judged documents. The two design choices are visible in the formula: the **exponential gain** 2^rel − 1 means a rel=3 document is worth 7 and a rel=1 is worth 1, so relevance grades are strongly nonlinear; the **logarithmic discount** means position 1 is worth 1/log₂2 = 1.0, position 2 is 1/log₂3 = 0.631, position 10 is 1/log₂11 = 0.289. Halving happens slowly — position 10 is still worth 29% of position 1, which is a *much* gentler decay than real user attention.

Now the ways it lies.

**It lies about unjudged documents.** NDCG's denominator is the ideal ordering of the *judged* set. If your new ranker surfaces a genuinely excellent document that nobody labeled, it scores as rel=0 and your improvement registers as a regression. This is the dominant reason "our new retriever looks worse offline" — it is finding things the judgment list does not cover. The fix is **judgment-pool auditing**: any document that appears in the top-10 of any candidate system must be judged, which means the judgment list is a living asset that grows with every experiment.

**It lies about the discount.** Real click-through decays roughly geometrically with position on a search results page — often something like a factor of 0.5–0.7 per position in the first few slots — far steeper than log₂. So NDCG systematically under-weights the top two positions relative to how users behave. If your product is a single-answer surface (a voice assistant, an AI answer box), NDCG@10 is close to meaningless and you should be measuring success@1 or MRR.

**It lies about diversity and slates.** NDCG is a sum over positions with no interaction term. Ten near-identical excellent documents score the same as ten diverse excellent documents. Any product with a feed or a carousel needs an explicit diversity metric (intra-list similarity, category entropy, or α-NDCG which discounts redundant relevance).

**It lies about aggregate movement.** A +2% mean NDCG can be +8% on head queries and −3% on tail, and tail is where your competitive differentiation lives. **Always report NDCG per query segment, never only the mean.** I have killed launches on a positive mean NDCG because the per-segment breakdown showed a navigational-query regression.

**🗣 Say this in the room:** "NDCG@10 with exponential gain and log discount, normalized by the ideal ordering of judged docs. Three caveats I always state: unjudged documents count as zero so a genuinely better retriever can score worse; the log discount is much flatter than real position bias, so for single-answer surfaces I use success@1; and it has no slate-level term, so it cannot see diversity or redundancy."

### What is training/serving skew, and what does a feature store actually do about it?

Training/serving skew is the class of bugs where the value of a feature at training time is not the value the model will see at serving time. It is the most expensive bug class in applied ML because it produces **no error, no exception, and no alert** — only a model that is worse than the offline metrics claim, by an amount nobody can measure until it ships.

It has three distinct sub-species and a senior answer separates them.

**Implementation skew** — the feature is computed twice, once in Spark/SQL for training and once in Python/Java at serving, and the two implementations disagree. `session_count_7d` uses a UTC day boundary in the warehouse and a local-timezone boundary in the service. Or the offline job includes today's partial day and the online one doesn't. This is the plain-old-DRY bug and it is the one a feature store solves cleanly: **define the transformation once, materialize it to both an offline store (warehouse/Parquet) and an online store (Redis/DynamoDB/Cassandra), and let both paths read the same computed values.**

**Temporal skew (label leakage)** — this is the subtle and lethal one. When you build a training row for an event at time T, every feature must be the value that *would have been known at T*, not the value in the table today. `user_lifetime_purchases` joined naively from the current dimension table includes purchases made *after* T, which correlates with the label, which makes your offline AUC beautiful and your online performance flat. Feature stores solve this with **point-in-time-correct joins** ("as-of joins"): the offline store is an append-only log of (entity, feature, value, event_time), and the training join is `ASOF LEFT JOIN ... ON feature.event_time <= label.event_time`. If your feature platform cannot do point-in-time joins, it is a cache, not a feature store.

**Distributional skew** — no bug at all; the world simply moved. Trained on winter traffic, serving in summer. Detected by monitoring, not prevented by architecture.

The other thing a good feature store buys you is **freshness contracts**: each feature declares a max staleness, and the serving path can report which features were served stale. This makes "the model got worse at 3 a.m." a debuggable event instead of a mystery.

**⚠ Trap:** believing a feature store eliminates skew. It eliminates *implementation* skew for features it owns. It does nothing about the request-time features you compute inline in the serving handler (query length, current page, A/B bucket), which is where I have found the majority of real skew bugs. **The rule I enforce: any feature computed in the request path must be logged into the training data verbatim as it was served, not recomputed offline.** Log-and-join beats recompute-and-hope, always. This is the "log the request payload, not the derived value" discipline you already apply to debugging distributed systems.

**🔍 Failure taxonomy — offline-good/online-bad, in triage order:** (1) Compare per-feature distributions between the training set and a sample of live serving requests — KS statistic or just p1/p50/p99 side by side. A feature whose serving p50 differs by more than ~10% from training p50 is your suspect. (2) Check null rates: a feature that is 2% null offline and 40% null online means the online store is missing rows. (3) Run the offline model on *logged serving features* rather than warehouse-recomputed features and see if the offline metric collapses — if it does, you have implementation skew; if it doesn't, look at the label definition. (4) Check for leakage by training a model with only the suspect feature — an AUC of 0.95 from one feature is a leak, not a discovery.

### The new ranker beat the incumbent by 4% NDCG offline and was flat online. Debug it.

I have run this triage a dozen times and I run it in a fixed order, cheapest first, because each step eliminates a whole family of causes.

**Step 0: is the experiment even measuring what you think?** Check the assignment. Is the bucketing sticky per user or per request? Per-request bucketing on a recommender destroys the effect because personalization has memory. Check sample ratio mismatch — if the control got 50.4% of traffic and treatment 49.6%, the randomization is broken and every downstream number is suspect. Check that the treatment model is actually being served (a shocking fraction of "flat" results are a config that never took effect — look at model-version distribution in your serving logs, not in your deploy tool).

**Step 1: is the offline gain real, or is it a judgment artifact?** Re-run the offline eval restricted to queries where *both* systems' top-10 are fully judged. If the 4% collapses to 0.3%, you were measuring unjudged-document penalty on the incumbent, not quality. This is the single most common cause.

**Step 2: is the offline gain on a segment that does not matter?** Break NDCG down by query segment (head/torso/tail), by surface, by device, by session position. A 4% mean gain concentrated on tail queries that make up 3% of traffic is an offline gain of 4% and an online gain of 0.12% — which is exactly "flat." Compute the traffic-weighted expected online lift *before* you launch; if it is under your MDE, the experiment was never going to read out.

**Step 3: was the gain eaten by a downstream stage?** If the ranker improved but the re-ranker's diversity constraint, business rules, or dedup discard the newly-promoted items, the user never sees the change. Log the top-10 before and after re-ranking for both arms and measure how much of the ranking delta survives to the rendered slate. I have seen 4% NDCG turn into 0.4% rendered-slate difference because a freshness rule pinned three slots.

**Step 4: was the metric the wrong metric?** Offline NDCG optimizes graded relevance; the business metric is 7-day retention or paid conversion. Relevance and satisfaction diverge routinely — a more "relevant" feed can be less surprising and reduce session length. Look at the intermediate metrics: did CTR go up while dwell time went down? That is the signature of a model that got better at bait.

**Step 5: statistical power.** Compute the MDE you actually had. With 2 M sessions per arm and a baseline CTR of 5%, the SE per arm is √(0.05×0.95/2e6) = √(2.375e-8) = 1.54e-4, so the two-arm SE is 2.18e-4 and the 80%-power MDE is 2.80 × 2.18e-4 = **6.1e-4 absolute, i.e. a 1.2% relative CTR change.** If your expected effect was 0.5% relative, "flat" is the only result you could have gotten, and the honest conclusion is "underpowered," not "no effect." **This is where I would consider interleaving instead** — it resolves ranking differences with roughly an order of magnitude fewer impressions.

**🗣 Say this in the room:** "Flat online after a strong offline gain is almost always one of four things: judgment-pool bias inflating the offline number, a segment-weighting mismatch so the gain lands where the traffic isn't, downstream re-ranking eating the delta before render, or an underpowered experiment. I'd check them in that order because they cost minutes, hours, hours, and minutes respectively — and I'd compute the traffic-weighted expected lift against the MDE *before* launching next time."

### How do you handle cold start for a brand-new item, a brand-new user, and a brand-new enterprise tenant?

These are three genuinely different problems and conflating them is a tell. The unifying principle: **cold start is a statement about which of your features have no signal, so the solution is always to lean on the features that do — content for new items, context for new users, and priors plus fast feedback for new tenants.**

**New item.** The ID embedding is untrained noise. The fix is a **content-based item tower**: text embedding of the title/description, image embedding, category, creator ID (which is *not* cold if the creator has history), price bucket, language. A two-tower model whose item side is content-dominant will place a brand-new item in roughly the right region of the embedding space on its first impression. Then you need **guaranteed exploration** — a fixed budget of impressions (say, every new item gets 1,000 impressions in a low-risk slot within 24 h) so the ID embedding gets gradient. Without a guaranteed budget, the rich-get-richer dynamic means new items never get shown, never get data, and never escape. Netflix and Pinterest both run explicit new-item exploration budgets for this reason.

**New user.** You have context, not history: device, locale, referrer, time, the acquisition campaign, and — crucially — whatever they do in the first 30 seconds. The architecture that works is a **sequence-based user tower that consumes the in-session interaction sequence**, so the model is useful after one click rather than after one day. Backstop with a popularity-by-locale-and-context prior for the very first request, and run a short onboarding survey only if you can prove it converts (it usually costs more in drop-off than it buys in relevance). Session-based models (GRU4Rec-lineage, or a small transformer over the session sequence) exist precisely for this regime.

**New enterprise tenant** — this is the one that matters for Glean, Notion, Harvey, Sierra, and it is the one candidates never prepare. A new tenant has *zero* interaction data and often forbids you from using other tenants' data to help them. So personalization from behavior is unavailable by contract, not just by data scarcity. What works: (a) **structural signals from their own corpus** — document recency, author seniority from the org chart, link/graph centrality within their workspace, channel membership, folder hierarchy; (b) **a global relevance model trained on cross-tenant data but consuming only tenant-agnostic features** (query-document semantic match, BM25, field matches) so no tenant's content leaks into another's model; (c) **per-tenant lightweight adaptation** — a small reranker or even just learned per-tenant feature weights, trained on that tenant's clicks once they exist, sitting on top of the frozen global model. (d) A deployment ritual: seed with 20–50 tenant-provided golden queries and their expected answers, which doubles as the eval set you will need anyway.

**⚠ Trap:** using cross-tenant behavioral data "anonymized" to bootstrap. In an enterprise contract, "we do not use your data to train models that serve other customers" is usually a hard commitment, and aggregate click data from tenant A influencing tenant B's ranking violates it in spirit and often in letter. The architecturally clean answer is a **shared model on tenant-agnostic features plus per-tenant heads**, and it is worth saying out loud in the room because it demonstrates you have shipped B2B and not just consumer.

**💰 Math:** the new-item exploration budget is a real cost. If you show every new item 1,000 impressions in a slot whose expected engagement is 40% of an optimally-ranked slot, and you onboard 50,000 new items/day, that is 50,000 × 1,000 = 5×10⁷ impressions/day at 60% relative loss = the equivalent of 3×10⁷ "wasted" impressions. Against a daily impression volume of 5×10⁹ that is 0.6% of total engagement spent on exploration. **That number is the price of a functioning catalog and I would defend it in a business review as such — the counterfactual is a corpus where nothing new is ever discoverable.**

### Explain position bias and show me how to correct for it in training.

Position bias is the reason click logs are not relevance labels. **A user clicked position 1 partly because it was relevant and partly because it was position 1.** If you train a ranker on raw clicks, you train it to reproduce the ordering of the ranker that generated the logs — a self-confirming loop that converges to "whatever we did last quarter," with the training data providing steadily less information each cycle.

Model it explicitly. Under the position-based examination model, click(d, k) = examine(k) × relevant(d), with examine(k) = p_k depending only on rank k. So E[click | shown at rank k] = p_k · r_d. Given p_k, an unbiased estimate of relevance is click / p_k — **inverse propensity scoring**. In training, that becomes a per-example weight: each clicked document at rank k contributes with weight 1/p_k, so a click at rank 10 (where p₁₀ might be 0.05) counts 20× as much as a click at rank 1 (p₁ = 1.0). This is exactly importance sampling, and it makes the loss an unbiased estimator of the loss you would have measured under a randomized logging policy.

```python
# unbiased pairwise LTR weight, sketch
# props[k] = examination propensity at rank k, estimated offline
w = clicked.astype(float) / props[shown_rank].clip(min=0.01)   # clip: cap variance
loss = (w * pairwise_hinge(scores_pos, scores_neg)).mean()
```

**📄 Paper:** Joachims, Swaminathan, Schnabel (2017), "Unbiased Learning-to-Rank with Biased Feedback" — established propensity-weighted ERM for ranking and showed it recovers the true ranking from biased click logs, replacing the practice of treating clicks as labels.

How do you get p_k? Three options in ascending order of cost and quality. **RandPair / result randomization:** for a small fraction of traffic, swap the document at rank 1 with the document at rank k; the ratio of click rates gives you p_k / p₁ directly. This is the gold standard and costs you a small, bounded amount of relevance on ~1% of queries. **Intervention harvesting:** exploit natural variation — the same query-document pair appearing at different ranks across A/B arms or over time — to estimate p_k without deliberate randomization. Cheaper, but the variation is not random so it can be confounded. **Joint estimation (EM / "dual learning"):** learn relevance and propensity simultaneously from the click log; elegant, but identifiability is delicate and I would not trust it as the only method.

**⚠ Trap:** clipping propensities without reporting the bias you introduced. The 1/p_k weights have unbounded variance as p_k → 0, so everyone clips at some ε. Clipping trades variance for bias, and the bias is systematically *against* the low-position documents you were trying to learn about. Report the clipping threshold and the effective sample size — Σw)² / Σ(w²) — alongside any propensity-weighted result. If your ESS is 8% of your sample size, your "unbiased" estimate has the statistical power of a dataset 12× smaller than the one you think you have.

**📐 Numbers you must know:** typical measured examination propensities on a ten-blue-links SERP fall roughly as p₁=1.0, p₂≈0.65, p₃≈0.50, p₅≈0.30, p₁₀≈0.10–0.15 — steeper than NDCG's log₂ discount (which gives 1.0, 0.63, 0.50, 0.39, 0.29). **📅 Volatile:** these are layout-dependent and change with every UI redesign, infinite scroll, and mobile-vs-desktop split — measure your own with randomization; never import someone else's curve. The shape of the argument is what you memorize, not the digits.
### Your PM wants to "explore more." Walk me through epsilon-greedy, UCB and Thompson sampling and tell me which one you'd actually ship.

The mental model that makes all three feel inevitable: **you are not choosing between exploring and exploiting; you are choosing how to spend a fixed budget of uncertainty reduction.** An arm you are uncertain about is worth pulling not for its expected reward but for the option value of learning it is good. The three algorithms differ only in how they price that option.

**ε-greedy** picks the argmax with probability 1−ε and a uniform random arm with probability ε. It is uncertainty-blind — it explores an arm you have pulled a million times exactly as often as one you have never pulled. Its regret is linear in T unless you decay ε. It is also the only one you can explain to a PM in one sentence and the only one whose logging propensities are trivially known (ε/K for exploration, 1−ε+ε/K for the greedy arm), which matters enormously for off-policy evaluation later.

**UCB** picks argmax over (μ̂_a + c√(2 ln t / n_a)). The bonus term *is* the uncertainty, so the algorithm explores arms in proportion to how little it knows about them. Regret is O(√(KT ln T)), which is near-optimal. Its practical problem is that it is deterministic — every user in the same state gets the same arm — so you cannot compute a propensity, which quietly rules out most off-policy evaluation.

**Thompson sampling** maintains a posterior per arm, draws one sample from each posterior, and plays the argmax of the draws. For Bernoulli rewards with a Beta(α, β) prior this is four lines of code and the posterior update is `α += reward; β += 1 - reward`. It matches UCB's regret bounds empirically and usually beats it in practice, it is naturally stochastic (so propensities exist, at least approximately), and it handles delayed feedback gracefully because a batch of pending pulls just means the posterior hasn't tightened yet.

```python
import numpy as np
class ThompsonBernoulli:
    def __init__(self, k): self.a = np.ones(k); self.b = np.ones(k)
    def select(self):      return int(np.argmax(np.random.beta(self.a, self.b)))
    def update(self, arm, reward):
        self.a[arm] += reward; self.b[arm] += 1.0 - reward
```

**What I would ship** in a recommender is none of these in their pure form, because a recommender has 10⁸ arms and a context. The realistic answer is a **contextual bandit**: LinUCB (Li et al., 2010) for a linear reward model, or — far more commonly in industry — a neural ranker whose scores are converted to a stochastic policy by **sampling the slate from a softmax over scores with a temperature**, plus a small guaranteed-exploration budget for cold items. That gives you three things at once: exploration proportional to uncertainty-ish (score gaps), known propensities for OPE, and a single knob (temperature) your PM can turn.

**📄 Paper:** Chapelle & Li (2011), "An Empirical Evaluation of Thompson Sampling" — the paper that rehabilitated a 1933 idea by showing it beat UCB on display-advertising data and degraded gracefully under delayed feedback.

**⚠ Trap:** running exploration without logging the propensity. If you explore and do not record the probability with which each shown item was chosen, you have paid the full cost of exploration and cannot cash in any of the benefit — no IPS, no doubly-robust estimator, no counterfactual replay. **The rule I enforce: the impression log schema has a `propensity` column from day one, and it is `NOT NULL`.** Adding it later means every experiment before that date is unusable for off-policy work.

**🗣 Say this in the room:** "For a small arm set, Thompson sampling — it's four lines, it's near-optimal, and unlike UCB it's stochastic so I retain propensities. For a real recommender with a huge item space I'd use a stochastic policy over ranker scores with a temperature knob plus a guaranteed impression budget for new items, and I'd make logging the propensity a hard schema requirement so the exploration data is reusable for off-policy evaluation."

### Explain off-policy evaluation. Derive IPS, tell me why it blows up, and fix it.

Off-policy evaluation answers the question every ranking team asks weekly: **"what would this new policy have earned, using only logs generated by the old policy, without running an A/B test?"** It is the counterfactual-inference tool that lets you kill 90% of candidate policies before they touch traffic, and it is the single most differentiating topic in a Netflix/Uber/Airbnb ranking loop.

**IPS.** You logged (context x, action a, reward r, propensity p = π₀(a|x)). The value of a new policy π is V(π) = E_{x, a∼π}[r]. Importance sampling gives
```
V̂_IPS(π) = (1/n) Σ_i r_i · π(a_i | x_i) / π₀(a_i | x_i)
```
It is unbiased provided **support coverage**: π₀(a|x) > 0 wherever π(a|x) > 0. If your new policy wants to show an item the old policy never showed, no amount of data can tell you what would have happened. That is not a statistical limitation, it is an information-theoretic one, and the honest answer is "this policy is not evaluable offline; it needs live exploration traffic."

**Why it blows up.** The weight w = π/π₀ has variance that scales with how different the policies are. If π is deterministic and π₀ was ε-greedy with ε=0.05 over 100 arms, the weight on an exploration action is 1/(0.05/100) = 2,000. One lucky reward on one such record can move your entire estimate. Concretely: with n = 10⁶ logged events, if 0.05% of them carry weight 2,000 and the rest carry ~0, your **effective sample size** ESS = (Σw)²/Σw² can easily be a few hundred. You have a million rows and the statistical power of five hundred.

**The fixes, in the order I apply them:**

1. **Self-normalized IPS (SNIPS):** divide by Σw instead of n. Slightly biased, dramatically lower variance, and — importantly — invariant to a constant scaling of the weights, so it does not explode when the average weight drifts from 1. This is nearly free and I would call it the default.
2. **Weight clipping / truncation** at some M (say the 99th percentile of w). Trades bias for variance. Always report M and the fraction of mass clipped.
3. **Doubly-robust (DR):** fit a reward model r̂(x, a) — the "direct method" — and use
   `V̂_DR = (1/n) Σ [ r̂(x_i, π) + w_i (r_i − r̂(x_i, a_i)) ]`.
   The importance weight now multiplies the *residual*, not the reward, so if your reward model is decent the weights have far less to amplify. DR is consistent if *either* the propensity model or the reward model is correct — hence "doubly robust." **📄 Paper:** Dudík, Langford, Li (2011), "Doubly Robust Policy Evaluation and Learning."
4. **Report ESS with every estimate.** `ESS = (Σw)² / Σ(w²)`. My review rule: an OPE result with ESS below 10% of n does not get to inform a launch decision; it gets to inform which policies enter a live exploration bucket.

**⚠ Trap:** running OPE on logs from a *deterministic* policy. Your production ranker sorts by score with no randomization, so π₀(a|x) ∈ {0, 1}, so every weight is either 0 or 1/1, and IPS degenerates to "the estimate of any policy that differs from the logging policy is undefined." Teams work around this by *assuming* a propensity (e.g. a softmax over the logged scores) — which is fitting a model of a policy you know was deterministic, and it produces confident nonsense. **You cannot retrofit exploration. If you want off-policy evaluation, you must put stochasticity into production first.** That is the sentence that gets you hired on this topic.

**💰 Math:** an A/B test that needs 2 M sessions per arm at 500 k sessions/day takes 8 days per candidate policy. If your team evaluates 20 candidate rankers per quarter serially, that is 160 days of experiment time against ~90 available days — you are throughput-bound. A working OPE pipeline that correctly kills 15 of 20 before they hit traffic converts that to 5 × 8 = 40 days. **The business case for OPE is not accuracy, it is experiment throughput**, and framing it that way is how you get headcount for it.

### Why do offline metrics keep misleading you in a recommender, specifically?

Because a deployed recommender does not observe the world; it observes **the part of the world it chose to show you**, and then it trains on that. Every offline metric computed on logged data is conditioned on the logging policy, and the logging policy is your previous model. This is not a bias you can correct away with a bigger dataset — a bigger dataset from the same closed loop contains more of the same restriction. It is the defining epistemological problem of the field.

Concretely, four compounding mechanisms:

**Selection / exposure bias.** You only have labels for shown items. A model that would surface a great item you never showed gets no credit; a model that agrees with your old model gets full credit. Offline metrics therefore have a structural bias toward *incumbent similarity*, which is precisely the opposite of what you want when hunting for a step change.

**Feedback loops.** Yesterday's recommendations produce today's training data, which produces tomorrow's recommendations. Popularity amplifies; niche content starves; the item embedding space slowly collapses toward the region your users already occupy. You can measure this: track the **Gini coefficient of impressions over items**, and the **fraction of the catalog receiving ≥1 impression per week**. If catalog coverage is falling month over month, you have a runaway loop regardless of what NDCG says.

**Metric-behavior mismatch.** Offline you measure relevance; online the business measures retention, session depth, subscription renewal. These correlate but not tightly, and the divergences are systematic — more "relevant" often means more predictable, and predictable feeds reduce discovery, which reduces long-run retention. Netflix's long-standing position on this — that short-term engagement and long-term member satisfaction can point in opposite directions — is the standard reference for why ranking teams keep a long-horizon holdout.

**Delayed and multi-outcome rewards.** A click is available in 200 ms, a completed watch in 40 minutes, a subscription renewal in 30 days. Any offline metric you can compute fast is a proxy for the one you care about, and proxies are gameable by the optimizer you are pointing at them.

**🗣 Say this in the room:** "Offline metrics on logged data are conditioned on the logging policy, so they systematically reward agreeing with the incumbent. That's why I treat offline evaluation as a *filter* — it's excellent at killing bad candidates and untrustworthy at ranking good ones — and put the actual decision on interleaving or a live experiment. And I monitor catalog coverage and impression Gini as feedback-loop alarms, because those degrade silently while NDCG looks fine."

**⚠ Trap:** the "offline metric improved, so we're done" review culture. The countermeasure I install is a written rule: **an offline metric may block a launch but may never authorize one.** That asymmetry is the correct one given the bias direction, and stating it that crisply lands well with senior interviewers because it shows you have thought about the *decision procedure*, not just the statistics.

### What is interleaving and why is it so much more sensitive than an A/B test?

An A/B test asks "do users in group A behave differently from users in group B?" and pays for the enormous between-user variance — some users click on everything, most click on nothing, and that variance is in the denominator of your test statistic. **Interleaving asks "within this one user's one result page, which ranker's document did they prefer?" The user is their own control, so between-user variance cancels entirely.** It is the paired-design trick, applied to ranking, and it is the reason a comparison that needs a week of A/B traffic can read out in hours.

**Team-draft interleaving** (the standard construction): treat rankers A and B as two team captains drafting players. Flip a coin for who picks first, then alternate; each captain takes their highest-ranked document not already drafted. Record which team each shown document belongs to. Then credit each click to the owning team, and the test statistic is the per-query preference (team A got more clicks, team B did, or tie), aggregated as a sign test or a per-impression preference score. The coin flip per query removes the positional advantage of going first.

Why it's sensitive:
- Between-user variance is gone (paired).
- Position bias is *balanced by construction* — over many queries each ranker gets the top slot equally often — so you don't have to model p_k.
- Each impression yields a comparison, rather than each *user* yielding an aggregate.

**📄 Paper:** Radlinski, Kurup, Joachims (2008) introduced team-draft interleaving as a click-based evaluation that correlates with explicit relevance judgments where absolute click metrics did not; Chapelle et al. (2012) validated it at web-search scale and reported order-of-magnitude reductions in the data required versus A/B testing. Netflix later published on adapting interleaving to recommendation slates, using it as a fast first stage before full A/B.

The limits, which you must state or you look naive:
- **It measures ranker preference, not product outcome.** It cannot measure retention, revenue, or session length, because the user saw a blended list that neither ranker would have produced. Interleaving is a *screening* stage; the winner still gets an A/B test before launch.
- **It requires the two result sets to be comparable and blendable.** If ranker B changes the *number* of results, the layout, or introduces a different item type, the blend is incoherent.
- **It's biased by novelty in blended lists** — a document that looks odd next to its neighbours can attract clicks for the wrong reason.
- **Slate-level effects vanish.** If B's whole value is a well-diversified slate, blending destroys exactly the property you were testing. This is a real and common failure for recsys as opposed to search.

**💰 Math:** suppose an A/B test needs 2 M sessions per arm for your MDE (from the power calculation earlier: SE = √(0.05×0.95/2e6) = 1.54e-4). At 500 k sessions/day and two arms, that's 8 days. If interleaving resolves the same ranker comparison with roughly 10× fewer impressions — the commonly-reported order of magnitude — you read out in under a day. **Over a quarter, that turns ~11 sequential ranker experiments into ~90.** Experiment throughput is the actual product of a ranking team, and interleaving is the single largest multiplier on it.

**🗣 Say this in the room:** "Interleaving blends two rankers' results into one list with a fair draft, so each user is their own control and position bias cancels by construction. It typically resolves a ranker comparison with about an order of magnitude fewer impressions than A/B. I use it as a screening gate — it tells me which ranker users prefer, it cannot tell me what happens to retention, so the winner still goes to a real A/B before launch."

### Build me a judgment list for a search product. How many labels, what scale, and how do you keep it honest?

A judgment list is a versioned dataset of (query, document, graded relevance) triples plus the rubric that produced them, and I treat it with the same rigor as a database schema: it has an owner, a version, a migration process, and a test. **The single most important property is that it is *pooled* — every document that any candidate system ranks highly must be judged — because otherwise your offline metric punishes systems for finding things your labelers never saw.**

**Sampling the queries.** Never uniform-random from the query log; you will get a list that is 70% head navigational queries and tells you nothing. Stratify: sample by segment (head/torso/tail by frequency decile), by intent class (navigational / informational / transactional / exploratory), by language and locale, and deliberately over-sample the segments where you suspect weakness and where the business cares. A 2,000-query list stratified 400/800/800 across head/torso/tail with segment weights recorded lets you compute *both* the traffic-weighted metric and the per-segment metric from one asset.

**The scale.** A 4-point graded scale is the sweet spot: 0 = irrelevant, 1 = marginal/related, 2 = relevant, 3 = perfect/authoritative answer. Binary loses too much; 5+ points destroys inter-annotator agreement without adding signal. Write a rubric with **worked examples for every adjacent pair** (what makes something a 2 rather than a 3), because the boundaries are where agreement dies.

**Agreement.** Double-judge 15–20% of items and report **Cohen's or Krippendorff's alpha**. Below about 0.6 your rubric is the problem, not your labelers — go rewrite the boundary examples. Adjudicate disagreements with a senior judge and feed every adjudication back into the rubric as a new worked example. Track agreement over time; a drop is the leading indicator of rubric drift or a labeler-pool change.

**LLM judges for relevance labels** — this is now standard and you should have an opinion. My position: **use an LLM to expand the pool, never to define the ground truth.** Concretely, LLM-judge every query-document pair produced by any system (which makes full pooling affordable), then human-judge a stratified sample of ~300 of those and measure the LLM's agreement with humans per segment. If human-LLM alpha is ≥ your human-human alpha, the LLM labels are usable for that segment; if it's materially lower — and it usually is on domain-specific or ambiguous queries — that segment stays human. Re-measure agreement whenever you change the judge model, because a judge upgrade is a silent redefinition of your metric.

**💰 Math:** 2,000 queries × 10 pooled documents = 20,000 judgments. Human at 45 s each and a fully-loaded $30/h expert rate: 20,000 × 45/3600 × $30 = **$7,500** and 250 person-hours. LLM judge at ~1,500 input + 100 output tokens per pair, at $3/Mtok in and $15/Mtok out: 20,000 × (1500/1e6 × 3 + 100/1e6 × 15) = 20,000 × (0.0045 + 0.0015) = **$120**, in about an hour wall-clock. So the honest architecture is: $120 of LLM labels for pool coverage plus $1,100 of human labels on a 3,000-pair calibration sample. That is a 6× cost reduction *and* full pool coverage — the coverage is the bigger win, because it removes the unjudged-document bias that makes offline evals lie.

**⚠ Trap:** letting the judgment list go stale. Documents get deleted, products go out of stock, "best laptop 2024" stops being a good answer. A judgment list without a refresh cadence turns into a metric that rewards serving expired content. Stamp every judgment with a date, expire judgments older than ~6 months for time-sensitive query classes, and run a quarterly re-judgment of a random 10% to measure drift.

### A specific query segment regressed after a reindex. Walk me through the triage.

Regression triage in search is a bisection problem and the discipline is to bisect along the *pipeline*, not along the code. There are five places a relevance change can come from and I check them in this order because each is cheap to rule out and each rules out a whole class.

**1. Is it a corpus change or a scoring change?** Reindexing changes both potentially. Diff document counts per segment and per field: did the analyzer drop documents that failed parsing? A 3% document loss concentrated in one language is invisible in aggregate and catastrophic for that segment. Compare `count(*)` and per-field null rates old index vs new. I have found more relevance regressions here than anywhere else.

**2. Is it the analyzer / tokenizer?** A stemmer version bump, a change in stopword handling, a switch in CJK segmentation, a new ICU normalization — all of these silently change what matches. The test: pick 20 queries from the regressed segment, run `_analyze` (or your engine's equivalent) on both indexes, and diff the token streams. Different tokens means different postings means different BM25.

**3. Is it BM25 statistics?** BM25 depends on corpus-level statistics — average document length and per-term document frequency. If the reindex changed which fields are indexed or added a new document class, `avgdl` moves and *every* score changes even for documents whose content is identical. This is the classic "we added the changelog corpus and product search got worse" bug.

**4. Is it the embedding side?** If it is a hybrid retriever, was the embedding model version identical? Was the vector normalization identical? Did the ANN index build with the same `M` / `efConstruction`, and is `efSearch` the same at query time? A recall drop from an `efSearch` regression looks exactly like a relevance regression and is diagnosed in one minute by comparing recall@100 against brute force on a query sample.

**5. Is it the reranker's input distribution?** Even with an unchanged reranker, if stage-1 now hands it different candidates, output changes. Freeze stage 1 to the old index and run the new reranker (and vice versa) to attribute the delta.

The tooling that makes this fast, and which I would build on day one of owning a search system: **a query-level diff harness.** Given two system versions and a query set, it emits per-query NDCG delta, the top-10 set difference, and the per-stage intermediate outputs. Sort by NDCG delta ascending, read the ten worst queries by hand. In my experience ten queries is enough to name the cause about 80% of the time — the pattern is visible to a human long before it is visible in an aggregate.

**🔍 Failure taxonomy — relevance regression, by signature:** *Uniform small drop across all segments* → scoring-parameter or corpus-statistics change (check `avgdl`, k1/b, field boosts). *Sharp drop in one language or locale* → analyzer/tokenizer. *Drop concentrated in tail/rare queries* → recall problem: candidate generation or ANN parameters. *Drop concentrated in head/navigational queries* → exact-match or field-boost regression; the title/exact-phrase signal got diluted. *Drop only for recent documents* → freshness feature or index-lag. *No metric drop but user complaints* → your judgment list doesn't cover the affected intent; go expand the pool.

**⚠ Trap:** reindexing in place. A reindex that mutates the live index gives you nothing to bisect against and no rollback. **The rule I enforce: build to a new index, validate offline against the frozen query set, then flip an alias.** This is the same blue-green discipline you already apply to database migrations, and search teams routinely fail to apply it because "it's just a search index."

### How do you segment queries, and why does that segmentation change your architecture?

Query segmentation is the act of admitting that "search" is four or five different products wearing one text box, and it changes architecture because **each segment has a different correct answer shape, a different metric, and a different latency budget.**

The two axes I always cut on:

**Frequency.** Head (top ~1% of distinct queries, often 30–50% of volume), torso, and tail (queries seen once ever, typically the majority of *distinct* queries). Head queries can be *cached*, hand-curated, and A/B tested individually — the economics of spending human effort per query only work at the head. Tail queries are where semantic retrieval earns its keep, because there is no click history to lean on and lexical matching fails on paraphrase.

**Intent.** Navigational ("gmail", "acme corp expense policy") — the user has one specific target and success@1 is the only metric; a semantic retriever that returns five thematically related documents instead of the exact one is a *failure* the aggregate NDCG will not show. Informational ("how does our refund window work") — precision@k with good snippets, and the segment where RAG and generated answers genuinely help. Transactional ("buy noise cancelling headphones under 200") — needs structured attribute extraction and filtering, not similarity. Exploratory ("things like X") — diversity matters more than precision.

The architectural consequences are concrete:
- **Navigational** ⇒ keep a strong lexical/exact-match path and an entity/alias dictionary; boost exact title matches hard; never let the vector path outvote an exact match. This is the #1 reason pure-vector search feels worse than the keyword search it replaced.
- **Tail informational** ⇒ hybrid BM25 + dense with RRF fusion, plus a cross-encoder reranker on the top ~50.
- **Transactional** ⇒ an LLM (or a small fine-tuned classifier) that extracts filters into structured predicates, then a normal filtered retrieval. Trying to encode "under 200" into a dense vector is a category error.
- **Head** ⇒ a query-result cache with a TTL and a curated-override table your ops team can edit. This is not a hack; it is the correct engineering answer to a Zipf distribution.

**💰 Math:** if the head 1% of distinct queries is 40% of traffic and you cache their result lists for 10 minutes, you remove 40% of retrieval+rerank compute. At 20,000 QPS with a rerank stage costing 25 ms of GPU time per query, that is 8,000 QPS × 0.025 s = **200 GPU-seconds per second, i.e. ~200 concurrent GPU-equivalents removed.** At roughly $2/h per accelerator-hour that is 200 × $2 × 24 × 30 = **$288,000/month**. **📅 Volatile:** accelerator pricing moves fast; re-derive with your own rate. The structure of the argument — Zipf head × per-query cost — is the durable part.

**⚠ Trap:** reporting one aggregate relevance number over a mixed query mix. Aggregates hide the segment that is on fire. **My rule: every relevance dashboard is faceted by frequency band × intent class, and the launch criterion is "no segment regresses by more than X," not "the mean improved."** State it that way and you will sound like someone who has actually shipped a search product.

### Where do LLMs legitimately belong in a search pipeline, and where are they a bad idea?

I hold a strong and specific line here, because "add an LLM" is the answer that gets candidates rejected in ranking loops. **An LLM belongs anywhere the input is short, the output is short, the work is linguistic, and the latency can be hidden or amortized. It does not belong anywhere it must run once per candidate at retrieval scale.**

**Yes, use an LLM for:**

*Query understanding.* Spelling correction, acronym and jargon expansion, intent classification, entity extraction, decomposition of a compound question into sub-queries, and filter extraction ("laptops under $1,500 with 32GB" → `price < 1500 AND ram_gb >= 32` + the residual free text). Input is ~30 tokens, output is ~50. It's one call per query, and head queries are cacheable so your real call volume is a fraction of QPS.

*Reranking the top ~25–50.* A cross-encoder or an LLM scoring pass over a handful of candidates is where the biggest relevance gains in modern search come from. This is affordable because it is O(50), not O(corpus).

*Answer synthesis and snippets.* The generated answer over retrieved passages.

*Offline enrichment.* Generating document summaries, keywords, hypothetical questions a document answers, and synthetic queries for training a retriever. This is a batch job — use the batch tier at ~50% discount — and it improves retrieval permanently at zero online cost.

*Label generation* for judgment lists, with the human-agreement guardrail from earlier.

**No, don't use an LLM for:**

*Scoring the corpus.* Obvious but worth saying: 10⁸ items × any LLM call is not a system.

*Replacing lexical matching.* Exact identifiers, SKUs, error codes, ticket numbers, and person names must go through an inverted index. An embedding of "ERR_4471" is not meaningfully different from an embedding of "ERR_4417," and your users will notice long before your metrics do.

*Ranking with hard business constraints.* Inventory, eligibility, pricing rules, and regional availability are predicates, not preferences. Enforce them in the retrieval filter; do not put them in a prompt and hope.

**⚠ Trap:** putting the LLM query rewriter on the synchronous path without a fallback. A 400 ms p99 rewrite in front of a 60 ms search stack quadruples your latency and adds a new upstream dependency to your availability math — two nines of provider availability turns your 99.95% search into 99.0%. **Ship it with a hard 150 ms timeout and a fall-through to the raw query, cache aggressively on normalized query text (the head is 40% of traffic), and consider running the rewrite *speculatively in parallel* with the un-rewritten search so the rewrite only ever adds quality, never latency.**

**🗣 Say this in the room:** "LLMs go where the work is linguistic and the fan-out is one: query understanding, top-50 reranking, answer synthesis, and offline enrichment. They do not go where the fan-out is the corpus, and they never replace lexical matching for identifiers and exact names. And anything on the synchronous path ships with a timeout and a degrade-to-baseline path, because I'm adding a dependency with a worse availability profile than my search cluster."

### Explain semantic IDs and generative retrieval. Is this real or is it a research curiosity?

The idea is genuinely elegant, and my honest read is: **real as a research direction with production deployments at large recommenders, not yet the default architecture, and a topic you should be able to discuss fluently because ranking interviewers at Google and Meta love it.**

**The mental model.** A classical retriever stores one vector per item and does nearest-neighbour search — the index is a data structure external to the model. Generative retrieval asks: what if the *model itself* were the index? Give every item a short discrete code — a "semantic ID" like `(42, 7, 113, 5)` — and train a sequence model to emit that code given a user's history. Retrieval becomes decoding, and top-k retrieval becomes beam search over the code space. There is no ANN index at all.

**How the IDs are built.** Take a content embedding of each item (text/image encoder). Train an **RQ-VAE** — residual quantization — which encodes the vector as a first codebook index capturing coarse semantics, then quantizes the residual with a second codebook, and so on for typically 3–4 levels. The resulting tuple is hierarchical: items sharing the first token are semantically close. That hierarchy is the whole point — it gives the decoder a coarse-to-fine search structure and lets a brand-new item inherit meaningful prefix tokens from its content alone, which is a genuine cold-start advantage over a randomly-initialized ID embedding.

**📄 Paper:** Rajput et al. (2023), "Recommender Systems with Generative Retrieval" (TIGER) — introduced RQ-VAE semantic IDs with a seq2seq retriever, replacing the ANN-index-over-learned-embeddings pattern for the retrieval stage. Its intellectual predecessor for search is Tay et al. (2022), "Transformer Memory as a Differentiable Search Index" (DSI), which trained a transformer to map queries directly to document identifiers.

**What it genuinely buys.** Parameter efficiency — you replace a |V| × d embedding table (the 512 GB table from earlier) with a few small codebooks. Cold start — content-derived codes work on day one. Beam search gives you a natural diversity knob. And it composes with the rest of the LLM stack: semantic IDs are just tokens, so a single model can do retrieval, ranking, and explanation.

**What it costs, and why it isn't the default.** Decoding 4 tokens with beam width 100 is much slower than one ANN lookup — you are paying transformer forward passes where you used to pay a graph walk. **Updating the corpus is hard**: adding items to an HNSW index is an insert; adding items to a generative retriever means either retraining the codebook (which invalidates every existing ID) or accepting drift. **Codebook collapse** is a real training pathology. And **filtering is awkward** — you cannot easily constrain beam search to "in stock in Germany" the way you can attach a filter to an ANN query, so you decode and then filter, which wrecks your effective k.

**🗣 Say this in the room:** "Semantic IDs replace the ANN index with the model's own decoder — hierarchical RQ-VAE codes so retrieval is beam search over item tokens. The real wins are parameter efficiency and content-derived cold start. I'd push back on it as a wholesale replacement today because corpus updates require codebook stability and hard filters don't compose with beam search, so I'd run it as an additional candidate source alongside a two-tower retriever rather than instead of one."

### Your PM wants LLM-generated explanations on every recommendation. Cost it and design it.

Start with the arithmetic, because it decides the architecture before any quality question does.

**💰 Math.** Assume 5×10⁹ impressions/day and an explanation of ~250 input tokens (item metadata + a compact user-affinity summary) and 30 output tokens. At $3/Mtok in, $15/Mtok out (a mid-tier frontier price — **📅 Volatile**): per explanation = 250/1e6 × 3 + 30/1e6 × 15 = 0.00075 + 0.00045 = **$0.0012**. Times 5×10⁹ = **$6,000,000 per day.** That is $2.2 B/year to annotate a feed. The design is dead on arrival at that shape, and the value of saying so immediately is enormous — it shows you cost before you build.

So the real design is a hierarchy of cheaper mechanisms, and the LLM appears only at the top:

**Tier 0 — templates from structured reasons (covers ~90%).** Your ranker already knows *why* an item scored well: the dominant feature contributions, the source that generated the candidate, the matched entities. "Because you watched *Arcane*", "Popular with people in your team", "Matches your saved filter: remote, senior". These are free, deterministic, translatable, legally reviewable, and cannot hallucinate. Most of the perceived value of "AI explanations" is delivered here.

**Tier 1 — precomputed LLM explanations for the head.** Item-level and (item × coarse-user-segment)-level explanations generated in batch overnight, cached. If 5% of items receive 60% of impressions, generating 10⁶ item explanations at $0.0012 is **$1,200/day** and covers most of the surface with LLM-quality text. Use the batch API tier at ~50% off → $600/day.

**Tier 2 — on-demand LLM explanation, user-initiated.** A "why am I seeing this?" affordance. If 0.1% of impressions trigger it, that is 5×10⁶ calls × $0.0012 = **$6,000/day** — 1000× cheaper than annotating everything, and it fires exactly where the user wanted an explanation.

**The correctness problem, which is the part interviewers actually probe.** An explanation that is generated *from* the item and *after* the ranking decision is not an explanation; it is a plausible-sounding post-hoc rationalization, and it will confidently say "because you liked sci-fi" about a user who has never watched sci-fi. **The design rule: the LLM must be given the actual ranking reasons as structured input and instructed to verbalize only those, and the output must be validated to reference only entities present in the input.** A cheap entity-containment check (every proper noun in the output must appear in the input) catches most of it. Anything that fails validation falls back to the Tier 0 template.

**⚠ Trap:** explanations that leak. "We're showing you this because your colleague Priya viewed it" is a privacy incident in an enterprise product, and "because you watched [embarrassing title]" is a consumer trust incident on a shared TV profile. **Explanations expose your feature set to the user, and your feature set was never designed to be user-visible.** Every explanation source needs an explicit allow-list review, and cross-user reasons need a k-anonymity floor before they can be named.

### Embedding-based personalization means storing a vector per user. What are the privacy consequences?

A user embedding is a lossy but genuinely informative compression of that user's behaviour, and the legal and security posture that follows is: **treat it as personal data with the same controls as the raw event log it was derived from.** The common engineering intuition — "it's just 128 floats, it's anonymous" — is wrong, and being able to say why is a differentiator in any enterprise or EU-facing loop.

**Why it is not anonymous.** It is *linked* to a user ID, so under GDPR it is personal data by definition, full stop — the interesting question is not "is it PII" but "what can be recovered from it." Empirically: embeddings support **membership inference** (was this item in the user's history?), because the embedding is trained to be close to consumed items; **attribute inference** (a linear probe on user embeddings recovers coarse demographics with well-above-chance accuracy, because those attributes are predictive of behaviour and the model has every incentive to encode them); and **linkage** across services if the same encoder or the same behaviour underlies both. For text embeddings specifically, published inversion work has shown that substantial portions of short input text can be reconstructed from embeddings alone — which should end any argument that "embedding" means "de-identified."

**The controls I would actually implement:**

- **Deletion must propagate.** A GDPR/CCPA erasure request has to remove the user vector from the online store, from every ANN index that contains it, from the training snapshots, and from any cached candidate list. Vector indexes are the hard part: most support "delete" as a tombstone that removes it from results but leaves the data resident until compaction. **Your deletion SLA is therefore bounded by your index rebuild cadence, and you need to know that number and state it in your DPA.** If you rebuild weekly and promise 30-day deletion, you are fine; if you rebuild quarterly, you are not.
- **Retention windows.** Decay or expire user embeddings for inactive users; a vector that has not been updated in 18 months is a liability with no product value.
- **Purpose limitation and separation.** The embedding trained for recommendation must not silently become a feature in a credit, employment, insurance, or pricing decision. That is a regulatory cliff (EU AI Act high-risk classification, US fair-lending law) and the architectural control is separate feature namespaces with lineage that a reviewer can audit.
- **Tenant isolation** in B2B: user vectors are per-tenant data; a shared index with a tenant filter is a single ACL bug away from a cross-tenant disclosure. I strongly prefer physically separate indexes or namespaces per tenant, and I would defend the extra cost in a design review by pricing the incident.
- **On-device as the strong form.** If the user embedding never leaves the device and only the retrieval *query* is sent (or retrieval happens locally over a downloaded candidate set), the whole class of problems shrinks. This is the real driver behind on-device personalization, and it is worth saying that the driver is privacy and residency, not latency.
- **Differential privacy / federated learning** if you must train on-device signals centrally — but be honest in the room that DP at a useful ε costs measurable recommendation quality, and the tradeoff is a product decision, not a checkbox.

**🗣 Say this in the room:** "A user embedding is personal data — it's linked to an ID and it supports membership and attribute inference, so 'it's just floats' isn't a defence. The three controls I insist on are propagating deletion all the way into the ANN index with a stated SLA bounded by rebuild cadence, a retention window that expires dormant vectors, and purpose limitation with auditable lineage so a recommendation embedding can't drift into a credit or employment decision. Where residency is the constraint, keeping the user vector on-device is the architecturally clean answer."

### Design personalized search for a workspace product like Notion or Glean. Ninety seconds, then I'll dig in.

The shape I would put on the whiteboard, and the reason for each box.

**Ingest and permissions.** Connectors pull documents from the workspace and from third-party apps (Drive, Slack, Jira, GitHub). The non-negotiable design constraint is that **permissions are enforced at query time against the source of truth, not baked into the index at ingest**, because access changes constantly and an index that carries stale ACLs will eventually show someone their manager's compensation doc. Concretely: index a document with its ACL identifiers, resolve the querying user's group memberships at request time (cached with a short TTL, e.g. 60 s), and filter *inside* the retrieval call so that k is post-filter. **The pre-filter/post-filter distinction is the interview probe here** — post-filtering an ANN top-100 down to the 3 documents a user can see is how you get an empty result page for anyone with narrow access.

**Index.** Per-tenant namespaces. Hybrid: an inverted index for lexical/exact (people names, project codes, error strings — dominant in workspace search) plus a dense index over chunk embeddings. Structured fields as filters: author, last-modified, doc type, channel, project.

**Retrieval.** Query understanding first (an LLM or a small classifier: is this navigational — "Q3 planning doc" — or a question — "what's our refund policy"). Then parallel BM25 + dense, fused with RRF, ~100 candidates.

**Ranking.** A cross-encoder or LambdaMART over features that are mostly *not* about text: semantic score, BM25 score, recency, whether the user authored or edited it, whether the user's frequent collaborators touched it, org-chart distance to the author, whether it's in a space the user is a member of, click history on this document by this user and by their team. **In workspace search, social and structural signals beat textual signals**, and saying that is the fastest way to show you understand this product class.

**Answering.** For question-intent queries, a RAG answer with inline citations to the exact chunk, plus the ranked document list underneath. Citations are not a nicety here — an uncited answer in an enterprise product is unusable because the user cannot verify it and cannot tell whether it came from a document they're allowed to trust.

**Personalization and cold start.** New tenant: no clicks, so lean entirely on structure (recency, authorship, graph centrality, membership). As clicks accumulate, train a per-tenant reranker — or, more practically, learn per-tenant *feature weights* on top of a frozen global model, which needs orders of magnitude less data than a full model. **Never train a model on tenant A's data that serves tenant B**; the shared global model consumes only tenant-agnostic features.

**Evaluation.** Onboard every tenant with 30–50 golden queries collected during implementation; that is simultaneously the acceptance test, the regression suite, and the thing that makes a renewal conversation go well. Plus click-based online metrics (MRR on clicked results, zero-result rate, reformulation rate) and a quarterly human relevance pass on a stratified sample.

**⚠ Trap:** treating the "did the user find it" metric as CTR. In workspace search the strongest negative signal is **query reformulation within 30 seconds**, and the strongest positive is **click followed by a long dwell with no reformulation**. Optimizing CTR in an enterprise tool teaches the system to produce plausible-looking titles. I would put reformulation rate and zero-result rate on the primary dashboard above CTR.
### Implement top-k over a stream of 200 million scored candidates. What's the data structure and what are the constant factors?

The answer is a **bounded min-heap of size k**, and the reason it is a min-heap rather than a max-heap is the part interviewers watch for: you keep the *smallest* of your current best k at the root so you can reject a new candidate in one comparison. Push until the heap has k elements, then for each subsequent score compare against `heap[0]` and only pay the O(log k) sift when the candidate beats it.

```python
import heapq

def top_k_stream(scored, k):
    """scored: iterable of (score, id). Returns k highest, descending."""
    heap = []                      # min-heap of (score, id)
    for score, _id in scored:
        if len(heap) < k:
            heapq.heappush(heap, (score, _id))
        elif score > heap[0][0]:   # one comparison rejects the common case
            heapq.heapreplace(heap, (score, _id))
    return sorted(heap, reverse=True)
```

**The constant factors, which is the real question.** Total work is O(n log k) worst case but the *expected* number of heap mutations is only O(k log(n/k)) if the stream is in random order — because the probability that the i-th element belongs in the running top-k is k/i, and Σ_{i=k}^{n} k/i ≈ k ln(n/k). For n = 2×10⁸ and k = 1,000 that is 1000 × ln(200,000) = 1000 × 12.2 = **~12,200 heap operations across 200 million items.** Everything else is a single float comparison. This is why the naive-looking heap beats clever alternatives in practice: the branch predicts perfectly and you almost never touch the heap.

Three follow-ups you should pre-empt.

**Ties and determinism.** `(score, id)` tuples break ties by id, which makes the result deterministic across runs and shards — essential if you want reproducible rankings and diffable experiments. Tie-breaking on an unstable value (like a dict iteration order or a float that varies with SIMD reduction order) is a real source of "the same query returns different results" bug reports.

**Sharding.** Each of S shards computes its own local top-k and the coordinator merges S sorted lists of length k with a heap: O(Sk log S). Correctness requires each shard to return k, not k/S — a shard could hold all k winners.

**When a heap is the wrong answer.** If k is a large fraction of n, `numpy.argpartition` (introselect, O(n) expected) beats heaping. If the scores arrive as a NumPy array rather than a Python stream, `np.argpartition(scores, -k)[-k:]` is 50–100× faster than the loop above simply by staying out of the interpreter — and in a ranking service, the scores *do* arrive as an array, so the pure-Python heap is the right whiteboard answer and the wrong production answer. Say both.

**🏋 Drill (10 min, unaided):** write `top_k_stream` from memory, then extend it to a `merge_shards(list_of_heaps, k)` that is correct when shards return fewer than k. Pass criterion: correct on the edge cases n<k, k=0, duplicate scores, and all-equal scores, with no sort of the full input anywhere.

### Build me an inverted index from scratch and then make the query fast. What's WAND?

The mental model: an inverted index is a **column store keyed on term**, and query evaluation is a multi-way merge join over sorted document-ID lists. Everything sophisticated in search-engine internals is an optimization on that join.

The structure: `postings: term -> [(doc_id, term_freq, [positions...]), ...]` sorted by doc_id, plus `doc_len[doc_id]` and corpus stats (`N`, `avgdl`, `df[term]`). Scoring is BM25:

```
score(q, d) = Σ_{t in q}  idf(t) · ( f(t,d) · (k1 + 1) ) / ( f(t,d) + k1 · (1 - b + b · len(d)/avgdl) )
idf(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```
with k1 ≈ 1.2 (term-frequency saturation) and b ≈ 0.75 (length normalization). Two things to internalize: **k1 makes term frequency saturate** — the 20th occurrence of a word adds almost nothing, unlike raw tf-idf — and **b controls how hard you punish long documents**. b=0 means length-agnostic (right for short titles), b=1 means fully normalized.

```python
import math
from collections import defaultdict

def build(docs):                       # docs: {doc_id: [tokens]}
    post = defaultdict(list); dl = {}
    for d, toks in docs.items():
        dl[d] = len(toks); tf = defaultdict(int)
        for t in toks: tf[t] += 1
        for t, f in tf.items(): post[t].append((d, f))
    for t in post: post[t].sort()
    return post, dl, sum(dl.values()) / len(dl), len(dl)

def bm25(post, dl, avgdl, N, query, k1=1.2, b=0.75):
    scores = defaultdict(float)
    for t in set(query):
        pl = post.get(t)
        if not pl: continue
        idf = math.log((N - len(pl) + 0.5) / (len(pl) + 0.5) + 1)
        for d, f in pl:
            denom = f + k1 * (1 - b + b * dl[d] / avgdl)
            scores[d] += idf * f * (k1 + 1) / denom
    return scores
```

**Now make it fast. WAND** (Broder et al., 2003) is the key idea: for each term, precompute its **maximum possible contribution** `ub(t)` over the whole postings list. During a top-k evaluation you maintain a threshold θ = the current k-th best score. Sort the active postings cursors by current doc_id; accumulate upper bounds from the front until the running sum exceeds θ — the document at that cursor is the "pivot." Every document before the pivot *cannot* reach θ no matter what, so you skip all of them with a single `advance-to(pivot_doc)` per cursor instead of scoring them. **Block-max WAND** (Ding & Suel, 2011) refines this by storing per-block maxima inside the postings list rather than one global maximum, which makes the bounds much tighter and the skips much longer.

**📐 Numbers you must know:** on a web-scale corpus, WAND-family pruning typically evaluates only a few percent of the documents that contain the query terms while returning the exact same top-k as full evaluation — it is a *safe* optimization (exact, not approximate) as long as your upper bounds are true upper bounds. That last clause is the trap.

**⚠ Trap:** breaking WAND's safety by adding a score component it doesn't know about. The moment you add a freshness boost, a personalization term, or a document-quality prior *after* BM25, your `ub(t)` is no longer an upper bound on the final score, and WAND will prune documents that would have made the top-k — silently, with no error, producing a recall loss that looks like a relevance regression. The fix is to fold the static prior into the upper bound (`ub(t) + max_static_prior`) which is correct but loosens the bound and costs speed. **This is exactly why "we added a small boost and search got slower/worse" is such a common incident: the boost either broke pruning safety or destroyed the bound's tightness.**

### Design autocomplete for 50 million queries with 30 ms p99. Trie or something else?

A trie is the right mental model and usually the wrong data structure, and knowing the difference is the point of this question.

**Why a trie is the mental model.** Prefix search is a tree walk: descend one node per character, then collect the top-k completions in the subtree. Store at each node the top-k completions of its subtree *precomputed* — that is the single move that makes it fast, because it converts "traverse the whole subtree at query time" into "read k strings at the node you landed on." Now a query is O(|prefix|) pointer hops plus a memcpy.

**Why a plain pointer trie is the wrong structure at 50 M entries.** Pointer-per-node blows memory and destroys locality: 50 M queries × ~25 chars = 1.25×10⁹ nodes in the worst case, and even with sharing you are looking at tens of GB with terrible cache behaviour. The production answers, in order of how often I'd reach for them:

1. **A finite-state transducer (FST)** — a minimized DAG over the sorted key set that shares both prefixes *and* suffixes, mapping key → value. This is what Lucene uses for its term dictionary. Compression of 5–20× over a naive trie is typical, and lookups stay O(|prefix|) with excellent locality because the whole structure is a contiguous byte array you can mmap.
2. **A sorted array + binary search + prefix range.** Sort all completions; a prefix is a contiguous range found by two binary searches. Dead simple, mmap-able, O(log n) with n=5×10⁷ meaning 26 comparisons. If completions per prefix are precomputed and stored alongside, this is often fast enough and it is by far the easiest thing to operate.
3. **A ternary search tree** if you need edit-distance-tolerant traversal.

**The parts that actually determine quality**, and which candidates skip:
- **Ranking within a prefix**, by query frequency with time decay, personalized by the user's own history, and boosted by session context (previous query in this session). The top-k stored per node is *ranked*, not arbitrary.
- **Typo tolerance**: run the exact prefix walk first; only if it yields fewer than k results, fall back to a fuzzy path (Levenshtein automaton intersected with the FST, or a symmetric-delete dictionary). Never pay for fuzzy on the happy path.
- **Freshness**: a static FST cannot absorb "a breaking news query became popular 4 minutes ago." Run a small in-memory recent-query trie updated in near-real-time and merge its results with the FST's. Rebuild the FST daily.
- **Safety filtering**: a blocklist applied at build time, not query time, plus a per-request policy check. Autocomplete is a very high-visibility surface for offensive-suggestion incidents.

**💰 Math:** at 30 M autocomplete requests/hour (users type ~4 characters each with debounce, so a 7.5 M-query/hour product generates ~4× that in keystroke requests) and a 30 ms p99 budget, this must be an in-process memory lookup, not a network call. A 2 GB mmap'd FST loaded into every API pod costs you 2 GB × 200 pods = 400 GB of aggregate RAM. **That is the real design tension: replicate the index into every pod (fast, expensive in RAM) or run a dedicated suggest service (one copy, but you've added a network hop into a 30 ms budget).** I would replicate, because 30 ms p99 with a hop and a GC pause is not a budget I want to defend.

### Explain MinHash and SimHash. Derive the collision probability and tell me which one you'd use for near-duplicate detection.

Both are **locality-sensitive hashes**: hash functions engineered so that the probability of collision equals (or is a known function of) a similarity measure. That is the entire trick, and once you state it that way the algorithms follow.

**MinHash** estimates **Jaccard similarity** J(A,B) = |A∩B| / |A∪B| over sets (typically shingles — overlapping k-grams of words or characters). Take a random permutation π of the universe and define h(A) = min over a ∈ A of π(a). The derivation: the element achieving the minimum over A ∪ B is uniformly random among |A∪B| elements, and h(A) = h(B) exactly when that element lies in A ∩ B. So **P[h(A) = h(B)] = |A∩B|/|A∪B| = J(A,B)** — one line, and you should be able to produce it under pressure. With m independent hashes, the fraction of agreeing positions is an unbiased estimator of J with standard error 1/√m: m=200 gives SE ≈ 0.071, m=1,000 gives 0.032.

**SimHash** (Charikar, 2002) estimates **cosine similarity** over weighted feature vectors. Project onto b random hyperplanes and keep the sign of each projection as a bit. For two vectors with angle θ, a random hyperplane separates them with probability θ/π, so **P[bit agrees] = 1 − θ/π**, and the expected Hamming distance over b bits is b·θ/π. With b=64 and a duplicate threshold of "Hamming distance ≤ 3," you're accepting θ ≤ 3π/64 = 0.147 rad, i.e. cos θ ≥ 0.989.

**LSH banding**, which is what makes either usable at scale. Split the m MinHash values into b bands of r rows each (m = b·r). Two documents are candidates if they agree on *all* r rows of *at least one* band. P(candidate) = 1 − (1 − s^r)^b, which is a sigmoid in s with its inflection near s ≈ (1/b)^(1/r). For m=200, b=20, r=10: threshold ≈ (1/20)^0.1 = e^(−ln20/10) = e^(−0.2996) = **0.741**. At s=0.9 the candidate probability is 1 − (1 − 0.9¹⁰)²⁰ = 1 − (1 − 0.3487)²⁰ = 1 − 0.6513²⁰ = **0.9998**; at s=0.5 it is 1 − (1 − 0.000977)²⁰ = **0.0193**. Tune b and r to place that S-curve where your product needs it.

**Which one.** **SimHash for web-scale near-duplicate crawling** — it's 64 bits per document, so an index of 10⁹ documents is 8 GB, and Manku, Jain, Das Sarma (2007) showed how to do the Hamming-neighbour lookup efficiently with permuted sorted tables. **MinHash + LSH when you need an actual Jaccard estimate and tunable thresholds** — deduplicating a training corpus, clustering listings, finding plagiarism. MinHash gives you a similarity number; SimHash gives you a compact bit signature.

**⚠ Trap:** using cosine similarity of *dense embeddings* for exact-duplicate detection. Embeddings are trained to collapse paraphrases, which is the opposite of what you want here: two different products with near-identical marketing copy embed to nearly the same vector, and a legitimately distinct item gets suppressed as a duplicate. **Dedup on surface form (shingles → MinHash/SimHash), cluster on meaning (embeddings). Using one for the other is the most common LSH mistake I see in review.**

**💰 Math:** deduplicating a 2 TB text corpus before pretraining or before building a RAG index. If 18% is near-duplicate (a typical figure for scraped web text), removing it saves 360 GB of embedding work. At ~4 chars/token that's ~9×10¹⁰ tokens; at an embedding price of $0.02/Mtok (**📅 Volatile**) that is 9e10/1e6 × $0.02 = **$1,800 saved on the embed pass alone** — plus a permanently smaller index, plus the retrieval-quality gain from not returning eight copies of the same document in the top-10, which is the benefit that actually shows up in user metrics.

### Your latency dashboard needs accurate p99s over 10 billion events a day. How do you compute a quantile you can't store?

The naive answer — keep all values, sort, index — needs 10¹⁰ × 8 bytes = **80 GB per metric per day** and is not mergeable across hosts, which is the disqualifying property. **The mental model: you need a sketch that is (a) sublinear in memory, (b) *mergeable*, so per-host sketches combine into a cluster-wide sketch without re-reading data, and (c) accurate where you care — the tail.** Mergeability is the requirement people forget, and it is the one that rules out most naive schemes.

**Naive histogram with fixed buckets.** Mergeable, tiny, and wrong in exactly the place you need it: with linear buckets, your p99 lands in a bucket whose width is a large fraction of the value at the tail. With *exponential* buckets it's actually fine, and that observation is the seed of DDSketch.

**t-digest** (Dunning & Ertl). Maintains a set of centroids (mean, count) with a scale function that forces centroids near q=0 and q=1 to hold very few points while centroids near the median can be fat. Result: high accuracy at the extreme quantiles, low accuracy in the middle — exactly the right trade for latency monitoring. Typically a few hundred centroids (~5 KB) per sketch. It is mergeable and it's what many metrics systems use. Its weakness: the error guarantee is on the *quantile* (rank), not on the *value*, and it is empirical rather than a hard bound.

**DDSketch** (Masson, Rim, Lee, 2019). Bucket index = ⌈log_γ(x)⌉ with γ = (1+α)/(1−α). Because the buckets are logarithmic, the guarantee is a **relative-error bound on the value**: the reported q-quantile is within α relative error of the true one, for any q. That is the guarantee you actually want for latency, because "p99 is 200 ms ± 2%" is meaningful and "p99 is at rank 99% ± 0.5%" is not. At α = 0.02 covering 1 ms to 100 s, the bucket count is ln(10⁵)/ln(1.0408) = 11.5/0.04 ≈ **288 buckets** — a few KB, fully mergeable, with a hard bound. This is my default recommendation.

**⚠ Trap:** averaging percentiles. `mean(p99_host_1, ..., p99_host_N)` is not the cluster p99 and can be off by a large factor when load is skewed — if one host is hot and slow, its p99 is diluted by 49 healthy hosts. This is the most common metrics bug in the industry and the reason mergeable sketches exist: you merge *sketches*, then compute the quantile once. **The rule I enforce in review: percentiles are computed exactly once, at the final aggregation, from merged sketches. Any dashboard that aggregates pre-computed percentiles is lying.**

**Where this shows up in AI systems specifically:** you need mergeable sketches over **token counts per request**, **time-to-first-token**, **inter-token latency**, and **cost per request**, all faceted by model and tenant. TTFT and ITL have wildly different distributions and a shared bucket range that covers both will be wasteful for one of them — configure the sketch range per metric. And note that ITL's p99 is a per-token statistic, so a single 400-token response contributes 400 samples; if you record one ITL sample per *request* you are measuring something else entirely.

### Explain how HNSW actually works, including what happens when you delete a vector.

**The mental model: HNSW is a skip list where the "next pointer" is replaced by "a small set of near neighbours in a metric space."** The top layer is a sparse graph you can cross in a few hops to get roughly into the right region; each layer down is denser and refines the position; layer 0 contains every point. Search is greedy descent, and it works for the same reason a skip list works — logarithmic layers give you logarithmic hops.

**Construction.** Each inserted point is assigned a maximum level ℓ drawn from a geometric distribution, `ℓ = ⌊−ln(U) · mL⌋` with mL = 1/ln(M), so the expected fraction of points on layer 1 is 1/M, on layer 2 is 1/M², and so on. Starting from the global entry point at the top layer, greedily descend to the point's level, then at each layer from ℓ down to 0 run a beam search with beam width `efConstruction` to find candidate neighbours, and connect to at most M of them selected by a **heuristic that prefers diverse directions** rather than the M literally closest — this is the part people miss, and it is what prevents the graph from forming isolated clusters with no long-range links. Layer 0 allows up to 2M connections per node.

**Search.** Enter at the top, greedily walk to the local minimum at each layer, drop down, and at layer 0 run a best-first search with a candidate priority queue and a dynamic result list of size `efSearch`, returning the top k. **`efSearch` is the recall knob and it is a pure runtime parameter** — you can raise it per query without rebuilding, which makes it a great lever for "this query is important, spend more."

**📄 Paper:** Malkov & Yashunin (2016/2018), "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" — added the hierarchy to navigable small-world graphs, giving logarithmic search complexity and displacing tree- and LSH-based ANN as the default in-memory index.

**📐 Numbers you must know:** memory ≈ `n × d × bytes_per_component` (the vectors) **plus** `n × M × 2 × 4 bytes` for layer-0 links (bidirectional, 4-byte IDs) plus a small amount for upper layers. At n = 10⁷, d = 768, fp32, M = 32: vectors = 10⁷ × 768 × 4 = **30.7 GB**; links ≈ 10⁷ × 32 × 2 × 4 = **2.6 GB**. Total ~33 GB. **The graph is usually 5–10% of the cost; the vectors dominate — which is why product quantization (compressing vectors to 32–64 bytes) buys far more than tuning M.** Typical starting parameters: M=16–32, efConstruction=100–200, efSearch=64–128, then measure recall@k against brute force and tune efSearch to hit your target.

**Deletion — the part that bites you in production.** HNSW has no true delete. Removing a node would sever the graph paths that route through it, potentially disconnecting whole regions. So every implementation does **soft delete**: mark the ID as deleted, keep the node as a routing waypoint, and filter it out of results. Consequences you must state:
- **Memory never comes back** until a full rebuild or a compaction pass.
- **Search cost rises** with the deleted fraction, because `efSearch` slots get consumed by tombstoned candidates that are then discarded — at 30% deleted, your effective ef is ~0.7×ef and recall drops correspondingly. The fix is to raise efSearch as the tombstone ratio grows, or rebuild.
- **Right-to-erasure SLAs** are bounded by your rebuild cadence, not by the delete call. If a user's vector is tombstoned but resident, "deleted" is a policy claim, not a physical one — and in a GDPR conversation that distinction matters.
- **Updates are delete + insert**, so a high-churn corpus (a marketplace where prices change hourly and price is a feature in the embedding) degrades continuously. **The rule: track tombstone ratio as a first-class metric and trigger a rebuild at ~20%.**

**⚠ Trap:** filtered search. Applying a filter *after* HNSW returns k gives you far fewer than k results when the filter is selective — ask for 100, filter to tenant X which is 0.1% of the corpus, get 0 results. Real engines handle this with filtered graph traversal (skip non-matching nodes during the walk, which degrades to a scan when the filter is very selective) or with per-partition indexes. **The correct architecture for tenant-scoped search is one index per tenant (or per shard-of-tenants), not one giant index with a filter** — and that also solves the isolation and deletion problems at once.

### Your RAG system started returning stale answers after a reindex. Walk me through it.

I have debugged this exact shape more than once and it is almost always one of five things. The unifying insight: **a vector index is a derived, denormalized, eventually-consistent copy of your source of truth, and every classic cache-coherence failure applies — you just don't have the vocabulary of cache invalidation attached to it, so people don't look there.**

**1. The alias never flipped, or flipped for reads but not writes.** You built `docs_v7`, validated it, pointed the read alias at it — and the ingestion writer is still writing to `docs_v6`. New documents land in a dead index. Symptom: everything older than the reindex is fine, everything newer is missing. Check: write a canary document and query for it end-to-end. Have this as an automated post-deploy check.

**2. Deletes didn't propagate.** The reindex rebuilt from a source snapshot that included documents deleted after the snapshot was taken, so the "stale" answers are documents that no longer exist upstream. Symptom: answers cite documents the user says were deleted weeks ago. Check: sample 100 indexed doc IDs and verify each still exists in the source of truth. **Deletion is the operation ingestion pipelines forget, because a soft-deleted row in Postgres simply stops appearing in the incremental `WHERE updated_at > ?` query — it doesn't produce a delete event.** The fix is a tombstone stream or a periodic full reconciliation.

**3. Chunk IDs changed but the cache didn't.** If you have a semantic cache or a query→answer cache in front of RAG, a reindex does not invalidate it. Users get last week's answers verbatim. **The cache key must include the index version.** This is the one that most often produces literally "stale answers" as the phrase suggests.

**4. Embedding model version skew.** The reindex used a new embedding model but the query path still embeds with the old one. Cosine similarities between two different models' spaces are noise-with-structure — you get results, they're plausible-looking, and they're wrong. Symptom: relevance is uniformly mediocre rather than specifically stale, and recall@k against a brute-force check with the *same* model is fine while end-to-end is bad. **Guard: store the embedding model ID and dimension in the index metadata and have the query path assert equality at startup and refuse to serve on mismatch.** Fail loudly; a search system that silently returns garbage is worse than one that returns 503.

**5. Chunking changed and the metadata didn't.** New chunk boundaries, same stored `last_modified` copied from an old field, so your recency filter or recency boost operates on wrong timestamps and prefers old content.

**🔍 Failure taxonomy — "stale RAG answers," as a decision procedure:** Ask first, *is the correct document in the index at all?* Query the index directly by document ID. If **absent** → ingestion problem (branches 1, 2). If **present but not retrieved** → retrieval problem: check embedding version (4), then recall@k against brute force, then filters and ACLs. If **retrieved but not used** → the reranker dropped it, or context assembly truncated it, or the answer came from cache (3). If **used but the answer is still old** → the generation prompt contains a stale system-prompt fact, or there's a conflicting older chunk that the model preferred; check for duplicate near-identical chunks with different dates and add a recency-aware dedup step.

**⚠ Trap:** reindexing without a **freshness SLI**. You cannot detect staleness you don't measure. The metric I put on every RAG system: **indexing lag = p50 and p99 of (time indexed − time modified at source)**, computed from a continuous canary that modifies a known document every minute and queries for the change. That one number turns a class of silent multi-week failures into a page. It is the same instinct as consumer lag on a Kafka topic, and I would present it that way in the room.

### Cost out a full ranking stack at 20,000 QPS. Where does the money actually go?

Let me build it stage by stage, because the answer is usually counter-intuitive: **the money is in the reranker and the embedding refresh, not in the retrieval index everyone worries about.**

**Assumptions:** 20,000 QPS peak, ~8,000 QPS average (so ~7×10⁸ queries/day), corpus 5×10⁷ items with 768-d fp32 embeddings, top-1000 retrieval, top-50 cross-encoder rerank.

**Stage 1 — ANN retrieval.** 5×10⁷ × 768 × 4 B = 153.6 GB of vectors, plus ~1.3 GB of HNSW links at M=16. That does not fit one commodity node comfortably, so either shard across 4 nodes with 64 GB each, or compress. With fp16 → 76.8 GB; with PQ at 64 bytes/vector → 3.2 GB (and a measurable recall cost you must quantify). Take the fp16-on-4-shards route: 4 × (r6i.4xlarge-class, 128 GB RAM) at roughly $1.0/h each (**📅 Volatile**) × 3 for replication/HA = 12 nodes = $12/h = **$8,640/month.** Retrieval is cheap.

**Stage 2 — feature fetch + light ranker.** 20,000 QPS × ~1,000 candidates = 2×10⁷ item-feature lookups/second. This is the thing people under-cost. You cannot do 2×10⁷ Redis GETs/sec; you must **co-locate item features with the index shard in-process** so the "lookup" is a memory read. If you get this wrong and put it on the network, the design does not work at any price. A small MLP over 1,000 candidates × 20,000 QPS = 2×10⁷ scorings/sec; at ~200 FLOPs/candidate that's 4 TFLOP/s — trivially handled by the same CPU nodes with a batched matmul.

**Stage 3 — cross-encoder rerank of top 50.** This is the bill. A small cross-encoder (say 100 M params) over 50 candidates × ~200 tokens each = 10,000 tokens per query. Forward-pass FLOPs ≈ 2 × params × tokens = 2 × 1e8 × 1e4 = **2×10¹² FLOP per query.** At 20,000 QPS that is 4×10¹⁶ FLOP/s = **40 PFLOP/s**. An accelerator delivering ~300 TFLOP/s bf16 at a realistic 35% utilization gives ~105 TFLOP/s effective, so you need 40,000/105 ≈ **380 accelerators.** At ~$2/h that is $760/h = **$547,000/month.** That single number is why reranking depth is the most contested parameter in any search team.

**Stage 4 — embedding refresh.** If 2% of the corpus changes daily, that's 10⁶ items/day to re-embed. At 300 tokens each = 3×10⁸ tokens/day; at $0.02/Mtok for a hosted embedding model that is **$6/day = $180/month** — negligible. A full reindex of 5×10⁷ items is 1.5×10¹⁰ tokens = **$300 one-time**. Reindexing is cheap; the reason not to do it constantly is operational risk, not money.

**The levers, in order of leverage:**
1. **Cut rerank depth from 50 to 25** → halves $547k to $273k. Measure the NDCG cost; it is often under 0.5%.
2. **Cache the head.** 40% of traffic on 1% of distinct queries → 40% off stages 1–3 → saves ~$220k/month at depth 50.
3. **Distill the cross-encoder** to a 4× smaller model → 4× fewer FLOPs → $137k. This is the highest-leverage engineering project on the list and it is exactly the "small model" work the rest of this section is about.
4. **Quantize the reranker to int8** → typically ~2× throughput on hardware with int8 tensor cores.

**🗣 Say this in the room:** "At this shape the retrieval index is under 2% of the bill and the cross-encoder reranker is over 90% of it — 50 candidates × 200 tokens × 2×params FLOPs is 2 TFLOP per query, 40 PFLOP/s at 20k QPS, about 380 accelerators. So the optimization order is rerank depth, then head caching, then distilling the reranker — and I'd only touch the vector index if recall measurement showed a problem."

### What are time-series foundation models and when would you use one instead of training per series?

**The mental model: a time-series foundation model does for forecasting exactly what a pretrained language model did for text classification — it replaces "fit a model per dataset" with "one pretrained model, zero or few shots, and you only train when you can prove it beats the pretrained baseline."** The bet is that seasonality, trend, level shifts, and intermittency are *shared structure* across millions of unrelated series, so a model pretrained on a huge corpus of time series learns the grammar of temporal patterns and transfers.

**How the main families work, mechanically:**

- **Chronos** (Ansari et al., 2024, Amazon) takes the most literal route: scale each series, quantize the values into a fixed vocabulary of bins, and treat those bins as tokens for an off-the-shelf language-model architecture (T5 family). Forecasting is autoregressive token generation, and sampling many trajectories gives you a probabilistic forecast for free. The striking thing is how little was changed from the LM recipe.
- **TimesFM** (Das et al., 2024, Google) is a decoder-only model over **patches** of time steps rather than individual points, with a longer output patch than input patch so it can produce multi-step forecasts efficiently.
- **Moirai** (Woo et al., 2024, Salesforce) is a masked-encoder architecture designed for **any-variate** input with multiple patch-size projections to handle different sampling frequencies, and a mixture distribution output head for probabilistic forecasts.

All three are trained on large heterogeneous corpora of real and synthetic series and evaluated zero-shot on held-out datasets.

**When I'd reach for one — the decision rule:**
- **Many series, each short.** 200,000 SKUs with 18 months of weekly data each: per-series ARIMA/ETS is 200,000 fits with almost no data per fit. A foundation model amortizes across them. This is the strongest case.
- **Cold start.** A new product, a new customer, a new metric — no history, so nothing to fit. Zero-shot forecasting is the only option that isn't a constant.
- **Operational simplicity at scale.** One model artifact, one deployment, one monitoring story, versus 200,000 model artifacts with individual staleness. The MLOps saving is frequently the real justification and it is fine to say so.
- **You need a strong baseline this week.** Zero-shot gives you a defensible number in an afternoon, against which any bespoke model must justify its maintenance cost.

**When I would not:**
- **A few series with long, rich history and known exogenous drivers** — electricity demand with weather, or ad spend with a promotions calendar. A well-specified model with the right covariates beats a generic model that cannot see the covariates (support for exogenous regressors varies by model and version — **📅 Volatile:** check the specific release).
- **Hard latency floors.** A transformer forward pass is milliseconds; an exponential-smoothing update is nanoseconds. In a high-frequency context that gap is disqualifying.
- **Regulated forecasts needing full explainability** — see the model-risk-governance answer.

**⚠ Trap:** benchmarking a foundation model against a badly-tuned baseline. The literature on forecasting is unusually honest about this: **seasonal naive** (this week equals last week) and a well-implemented ETS/Theta are brutally strong, and a large fraction of published "deep learning beats classical" results have failed to reproduce against properly-tuned classical baselines. My rule before adopting any forecasting model: **it must beat seasonal-naive and a tuned ETS on a rolling-origin backtest of *your* data, on the metric you'll actually be judged on, or it does not ship.** State that in the room and you will sound like someone who has been burned, which is the point.

### How would you evaluate a forecasting model properly? Give me the metric and the split.

The split first, because it is where most people go wrong. **A random train/test split on time-series data is a leak, full stop** — you are training on the future to predict the past. The correct protocol is **rolling-origin backtesting** (a.k.a. walk-forward): pick an origin time t, train/condition on everything ≤ t, forecast horizon h, score, advance t, repeat. Report the distribution of errors across origins, not a single number, because forecast difficulty varies enormously by period and a single split can be lucky. And **the gap between the end of training data and the start of the forecast must equal the operational gap** — if your data pipeline has a 3-day lag in production, backtest with a 3-day lag or your offline number is optimistic in a way you will discover only after launch.

The metric. There are four and each has a failure mode you must name:

- **MAPE** — mean absolute percentage error. Undefined at zero, explodes for near-zero actuals, and is asymmetric: it punishes over-forecasting more than under-forecasting, which systematically biases any model you tune on it toward under-forecasting. **In retail, where zeros are everywhere, MAPE is actively harmful.**
- **sMAPE** — symmetric version; fixes the asymmetry partially, still unstable near zero.
- **MASE** — mean absolute scaled error: MAE divided by the in-sample MAE of a naive (or seasonal-naive) forecast. **Scale-free, defined at zero, and interpretable: MASE < 1 means you beat naive.** This is my default for comparing across many series with different scales.
- **Quantile loss / pinball loss** — the right metric when the decision downstream is asymmetric. Inventory is the canonical case: stocking out costs a lost sale, overstocking costs holding. If understocking costs 4× overstocking, the optimal order quantity is the **0.80 quantile** of the demand distribution (the newsvendor critical ratio = c_u/(c_u+c_o) = 4/5), and you should be evaluating the model's 0.80-quantile forecast with pinball loss, not its mean forecast with MAE.

**🗣 Say this in the room:** "Rolling-origin backtest with the same data lag as production, MASE as the headline because it's scale-free and defined at zero, and pinball loss at the specific quantile the business decision actually uses. And I always report the per-series distribution of the metric, not the mean, because one pathological series can dominate an aggregate and because the business cares which SKUs are bad, not what the average is."

**💰 Math:** for an observability team forecasting per-service traffic to drive autoscaling, the metric is not error, it is **cost of error**. Under-forecast by 10% and you get a cold-start latency incident; over-forecast by 10% on a $400k/month compute footprint and you burn $40k/month. That asymmetry — say incidents are worth 4× the compute — means you forecast the 0.80 quantile, not the mean, and you evaluate with pinball loss at q=0.80. **Reframing "which model is more accurate" as "which quantile does the decision need" is the single most valuable thing an AI engineer brings to a forecasting conversation.**

### What are tabular foundation models, and should I stop using gradient boosting?

No, you should not stop using gradient boosting — but you should know exactly where the exception lives, because "why isn't a transformer the answer here?" is a question big-tech applied-AI interviewers use to check whether you reason from constraints or from hype.

**The mental model.** TabPFN's idea is unusual and worth stating precisely: instead of *fitting* a model to your dataset, it does **in-context learning over the whole dataset**. You pass the training rows *and* the test rows into a single transformer forward pass, and it outputs predictions. There is no gradient descent at inference time. The transformer was pretrained on millions of *synthetic* datasets drawn from a prior over structural causal models, so it has learned an approximation to Bayesian inference over that prior — it effectively performs posterior prediction for whatever function class the prior covers.

**📄 Paper:** Hollmann et al. (2023), "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second" — replaced per-dataset AutoML search with a single pretrained forward pass on small tabular classification tasks. A successor extending the approach to larger tables and regression followed. **📅 Volatile:** the size limits and supported task types have moved with each release; check the current model card before quoting a row or feature cap.

**Where it genuinely wins:** small datasets — the original version targeted roughly a thousand rows and around a hundred features — where gradient boosting overfits and AutoML hyperparameter search costs more compute than the whole problem is worth. In that regime the reported result is competitive-or-better accuracy in a second versus minutes-to-hours of tuning. **That is a real and useful capability**: think a per-customer churn model on 800 rows, an experiment-analysis classifier, a cold-start scoring model for a new tenant.

**Where gradient boosting still wins, and why:**
- **Scale.** In-context learning means the dataset goes in the context window, so cost grows with dataset size in a way that a tree ensemble's does not. Millions of rows is the boosted-trees regime.
- **High-cardinality categoricals.** A user ID with 10⁸ values is an embedding-table problem, not a context-window problem.
- **Latency and determinism.** LightGBM inference is microseconds of branchy CPU work and is bit-for-bit reproducible. A transformer forward pass is milliseconds and, on GPU, is not deterministic across hardware unless you work for it.
- **Explainability.** SHAP on trees is exact and fast (TreeSHAP); attribution on a transformer's in-context prediction is a research question. In regulated settings this alone decides it.
- **Operational maturity.** Monotonic constraints, feature importances, incremental training, ONNX export, and twenty years of production experience.

**🗣 Say this in the room:** "TabPFN-class models do in-context learning over the dataset — no gradient descent at fit time — and they're genuinely the right tool for small tables where AutoML costs more than the problem. Above a few thousand rows, or with high-cardinality IDs, or under a latency or explainability constraint, gradient boosting is still the default and I'd need a specific reason to move. What I'd actually do is run both: TabPFN as an instant baseline on day one, LightGBM as the thing that ships."

**⚠ Trap:** assuming the "foundation model" framing implies the scaling behaviour you know from language. It does not. Tabular problems have no shared vocabulary across datasets — column 3 in your table has nothing to do with column 3 in mine — so transfer works through the *prior over functions*, not through shared semantics. That is a much weaker channel, and it is why tabular has resisted the foundation-model sweep that flattened NLP and vision. Being able to say *why* the transfer is weak, rather than just "it doesn't work as well," is what separates a good answer here.
### When does a fine-tuned 3B model actually beat a prompted frontier model? Give me the decision rule, not a hedge.

The rule I use, and I will defend it: **a fine-tuned small model wins when the task is narrow, the output format is fixed, the input distribution is stable, and the volume is high enough to amortize the pipeline. It loses whenever the task requires reasoning across an open domain, the spec changes weekly, or the volume is low.** Those are four conditions, and I check them in order because failing any one of them ends the conversation.

**Narrow task.** Classification into a fixed label set, extraction into a fixed schema, routing, PII redaction, query rewriting, ranking a short list, style transfer into one house voice, translating an internal DSL. The property they share: **the space of correct outputs is small and enumerable**, so a 3B model has enough capacity to cover it. Contrast with "answer any customer question about our product," where the output space is the language and the small model's capacity ceiling is real.

**Fixed output format.** Fine-tuning is spectacularly good at format adherence — better than prompting a frontier model, in my experience, because the format becomes the model's prior rather than an instruction it might drift from at token 400. If your current failure mode is "the frontier model sometimes emits prose before the JSON," fine-tuning a small model fixes that class of bug outright.

**Stable input distribution.** This is the condition people skip and it is the one that kills projects. A fine-tuned model is a photograph of a distribution. If the upstream product changes what users type — a new surface launches, a new customer segment onboards, the UI adds a field — the photograph is now of somewhere else. This gets its own answer because the detection problem is subtle.

**Volume.** Below roughly 3 million requests/month at current API prices, the maintenance cost exceeds the savings — the arithmetic is in the break-even answer below.

**What "beats" means matters too.** A fine-tuned 3B rarely beats a frontier model on raw task accuracy in an unconstrained comparison. It beats it on the **composite objective**: accuracy-per-dollar, p99 latency, format reliability, and — often decisively — the ability to run inside a VPC or on-device where the frontier model legally cannot go. **📐 Numbers you must know:** a 3B at bf16 serving on a mid-range accelerator delivers on the order of 1,500 output tok/s aggregate with continuous batching; at ~$0.80/GPU-hour that is 1,500 × 3600 = 5.4M tok/h → **$0.148 per million output tokens, versus $15/Mtok for a mid-tier frontier model — a 101× ratio.** That factor of ~100 is the number the spec means and it is worth memorizing with its derivation, because "100× cheaper" said without the throughput and the hourly rate is a slogan.

**🗣 Say this in the room:** "I'd fine-tune a small model when the task is narrow, the schema is fixed, the input distribution is stable, and I'm above roughly three million calls a month — and I'd say explicitly that the win isn't accuracy, it's about a hundred-to-one on output-token cost, tighter p99, better format adherence, and being deployable where a hosted frontier model isn't allowed. Below that volume, or with a spec that's still moving, prompting a frontier model is strictly the better engineering decision and I'd push back on building a training pipeline."

### Walk me through distilling a frontier model into a 3B you'd actually deploy. End to end.

Distillation in the LLM era is usually **sequence-level knowledge distillation**: you do not have logits from a hosted frontier model, so you train the student on the teacher's *outputs* as if they were ground-truth labels. That is the practical form and I'd name it as such. **📄 Paper:** Hinton, Vinyals, Dean (2015) established logit-matching distillation; Kim & Rush (2016), "Sequence-Level Knowledge Distillation," established the generate-and-train-on-outputs variant that everyone actually uses when the teacher is behind an API.

**Step 1 — define the task and the eval *before* generating any data.** Build a golden set of 300–500 human-verified examples first. If you cannot write the eval, you cannot distill, because you will have no way to know whether the student is worse than the teacher and no way to know when it drifts. This is the step teams skip and the reason most distillation projects die quietly.

**Step 2 — mine real inputs.** Sample from production logs, stratified across the segments you care about, deliberately over-sampling rare-but-important cases. Ten thousand real inputs beat a hundred thousand synthetic ones, because the whole value of the student is that it fits *your* input distribution. Where you must synthesize (a new feature with no traffic), generate inputs with a diverse seed-and-persona scheme and then hand-check a sample for realism.

**Step 3 — generate teacher outputs, with the teacher running at its best.** This is the counter-intuitive part: **spend more per teacher call than you would in production.** Use the strongest model, use extended thinking, use a long few-shot prompt, use self-consistency (sample k=5 and keep the majority answer), use a verifier to reject bad outputs. The teacher is a one-time cost amortized over the student's whole lifetime; a 10× more expensive teacher call that raises label quality by 4 points is trivially worth it. Use the batch tier (~50% discount) since latency is irrelevant here.

**Step 4 — filter ruthlessly.** Drop outputs that fail schema validation, fail a programmatic check (does the extracted date exist in the source text?), or disagree across self-consistency samples. **A student trained on 8,000 clean examples beats one trained on 20,000 examples of which 15% are wrong**, because the student has no way to tell which 15%, and errors in a small model's training set become confident errors at inference.

**Step 5 — train.** LoRA (rank 16–64) is almost always sufficient and lets you keep one base model in memory with swappable adapters. Full fine-tune only if LoRA plateaus below your bar. Hold out 10% by *time* (the last two weeks of logged inputs), not randomly, so your validation set tells you something about generalizing forward.

**Step 6 — evaluate against three baselines, not one.** The teacher (your ceiling), the current production system (your bar), and the un-fine-tuned base model with a good prompt (your "was this worth it" check). Report per-segment, and report the *disagreement* rate with the teacher rather than only aggregate accuracy — a student that agrees with the teacher 96% of the time but disagrees mostly on your highest-value segment is not shippable.

**Step 7 — ship behind a shadow deployment**, then a canary with a fallback: if the student's output fails schema validation or its confidence (e.g. sequence log-prob, or an explicit abstain token you trained it to emit) is below threshold, escalate that request to the teacher. **This escalation path is the design that makes distillation safe** — it converts a hard accuracy cliff into a cost/quality dial. If 5% of traffic escalates, you still capture 95% of the savings and your worst-case quality is the teacher's.

**⚠ Trap:** distilling from a teacher whose terms of service prohibit it. Several providers' terms restrict using outputs to train competing models, and several open-weight licenses carry naming or use restrictions on derived models. **📅 Volatile:** these clauses change with every license revision — read the current terms for the exact model version you are calling, and get it in writing for anything you plan to open-source or sell. I have seen this kill a shipped project at legal review, and it is a genuinely good thing to raise unprompted in an interview because it signals you have shipped in an enterprise.

### The upstream task drifted and your distilled model got quietly worse. How do you catch that before your users do?

This is the failure mode that makes distillation operationally expensive, and the reason I said the input distribution has to be stable. **The mental model: your student model is a compressed snapshot of the teacher's behaviour on a particular input distribution. Neither the teacher's behaviour nor that distribution is under your control, and when either moves, the student is wrong in a way that produces valid-looking output.** No exception, no schema violation, no latency change. Just a slow accuracy slide.

There are three distinct drifts and they need three distinct detectors.

**Input drift** — users start typing different things. Detect it *without labels* by monitoring the distribution of inputs: embed a sample of production inputs daily, and track the distance from the training-set distribution (population stability index over embedding cluster assignments, or maximum mean discrepancy, or simply the fraction of inputs whose nearest training neighbour has cosine similarity below a threshold). **The single cheapest useful metric: the "out-of-distribution rate" — the percentage of daily inputs whose nearest neighbour in the training set is below, say, 0.75 cosine.** When that goes from 3% to 18%, you have your alarm, and you have it weeks before quality complaints arrive.

**Label drift** — the correct answer changed even though the input didn't. A new product category, a policy change, a taxonomy revision. This is invisible to input monitoring and requires ongoing human labelling of a small sample. Budget for it explicitly.

**Teacher drift** — the frontier model you distilled from was silently updated, so the "ground truth" your student was trained against no longer matches what the teacher would say today. This one is nasty because your comparison baseline moved. **Pin the teacher to an explicit dated model version and re-run your golden set whenever you change it.**

**The monitoring architecture I would build:**
1. **Continuous shadow sampling.** Route 1–2% of production traffic to the teacher in parallel with the student and log both outputs. This gives you a live, unlabelled agreement metric. Cost: 2% of 20M requests × $0.0096/call (a strong teacher config) = 400,000 × 0.0096 = **$3,840/month.** That is cheap insurance and it is the number to quote.
2. **Alert on agreement rate, faceted by segment.** A 2-point drop in overall agreement is noise; a 12-point drop in one segment is a page.
3. **A frozen golden set re-run nightly** against a pinned teacher version, so you separate "the world moved" from "my model moved."
4. **Human review of a stratified 100-sample per week**, weighted toward low-agreement cases. This is the only detector that catches label drift.
5. **A retraining trigger, written down**: agreement below X, or OOD rate above Y, or golden-set score below Z, opens a ticket automatically.

**⚠ Trap:** monitoring only aggregate accuracy on a static golden set. A static golden set drawn from the original distribution is, by construction, blind to input drift — the model keeps scoring 94% on it while failing on 20% of live traffic that the golden set does not represent. **A golden set must be refreshed from live traffic on a cadence, or it is measuring the past.** I make "% of golden set drawn from the last 90 days" an explicit tracked number.

**💰 Math:** the full ongoing cost of a distilled model is not the GPU bill. It is GPUs (~$2,800/month in the shape below) + shadow teacher traffic ($3,840/month) + human review (100 samples/week × 4 min × $50/h loaded = 100 × 0.067 × 50 × 4.3 weeks = **$1,440/month**) + roughly 0.2 FTE of engineering (**~$6,900/month** at a fully-loaded senior rate). **Total ≈ $15,000/month, of which only 19% is compute.** Anyone who costs a self-hosted model as "just the GPU bill" is off by 5×, and saying so is the fastest way to sound like you have operated one.

### What's actually in the 1-to-8-billion-parameter model landscape, and what should I watch for in the licenses?

I'll give you the durable structure rather than a table that will be wrong by the time you interview. **📅 Volatile — verify every name, size and license before your loop; this space re-shuffles roughly quarterly.**

**The families that matter**, by who maintains them and what they optimize for: Meta's **Llama** line has small variants explicitly aimed at on-device (roughly 1B and 3B scale) alongside the ~8B workhorse; Alibaba's **Qwen** line covers an unusually dense ladder from sub-1B up through 7–8B and is often the strongest open-weight option at a given size, particularly multilingual and on code; Google's **Gemma** line targets the 2–9B range with strong small-model quality and a distinctive license; Microsoft's **Phi** line is the "trained on heavily curated and synthetic data" bet, punching above its parameter count on reasoning-flavoured benchmarks and typically MIT-licensed; **Mistral** ships both Apache-2.0 models and models under a research-only license, and the distinction is per-model, not per-company; and there are small-model efforts aimed squarely at edge (HuggingFace's SmolLM line, IBM's Granite, various Apple on-device models) where the design point is fitting in a phone's memory budget.

**How to choose within the band**, which is the actual question: (1) start from the memory budget, not the benchmark — a 4-bit 3B is ~1.7 GB and fits a phone; a 4-bit 8B is ~4.5 GB and does not fit comfortably alongside an OS and an app; (2) then check whether the family has an **instruction-tuned** variant and whether its chat template is well-documented (a mismatched chat template silently costs you several points and is the most common small-model bug); (3) then check tokenizer efficiency on *your* language and domain — a tokenizer that spends 1.6× the tokens on Hindi or on your code corpus eats your context and your latency; (4) then benchmark on your own 200-case set, because published small-model benchmarks are heavily contaminated and rank ordering does not transfer.

**The license traps, named:**
- **Llama's community license** is not an OSI open-source license. It carries a **monthly-active-user threshold (700 million) above which you must request a separate license from Meta**, plus attribution/naming requirements for derived models and an acceptable-use policy. For almost every company the MAU clause is irrelevant; for a consumer product at scale it is a genuine commercial risk that has to be checked before you build on it.
- **Gemma** ships under Google's own terms with a prohibited-use policy that applies downstream — you must pass the restrictions along to anyone you distribute to.
- **Apache-2.0 models** (much of Qwen, several Mistral releases) are the clean case: use commercially, modify, redistribute, patent grant included. This is what I default to when legal review time is scarce.
- **Research-only licenses** appear on some of the strongest models. Shipping one commercially is a straightforward contract breach, and "we didn't notice" is not a defence.
- **Output-usage clauses** on *hosted* frontier models restricting training competing models — this is the clause that governs whether your distillation project is legal at all.

**🗣 Say this in the room:** "I don't pick a small model from a leaderboard. I start from the deployment memory budget, filter to Apache-2.0 or MIT unless there's a strong reason, verify the chat template and tokenizer efficiency on my own data, and then run my own 200-case eval — because published small-model benchmarks are contaminated enough that the rank ordering doesn't survive contact with a real task. And I check the license before the benchmark, because a model I can't ship has no score."

### Compare the edge inference stacks. llama.cpp, MLX, ONNX Runtime, ExecuTorch, Core ML, TensorRT — when do you reach for which?

There is no single winner; there are four questions that pick for you: **what silicon, what OS, what model, and who maintains this in two years.**

**llama.cpp / GGUF** is the pragmatic default for CPU and mixed CPU/GPU inference across Linux, macOS, Windows, Android, and even browsers via WASM. GGUF is a single-file format carrying weights, quantization scheme, and tokenizer together, which makes distribution trivial — one file, one download, no runtime graph compilation. Its quantization ladder (Q4_K_M, Q5_K_M, Q6_K, and importance-matrix variants) is the de-facto standard vocabulary for open-weight quantization. Reach for it when you want the shortest path from "a model exists on HuggingFace" to "it runs on this laptop," and when portability across heterogeneous consumer hardware matters more than squeezing the last 20%.

**MLX** is Apple's array framework for Apple silicon, designed around **unified memory** — no host-to-device copies, because there is one pool. On a Mac it is frequently the fastest option for local LLM work and it is the natural choice if you are prototyping on Apple hardware or shipping a Mac app. It is not a cross-platform answer.

**Core ML** is the production path for shipping a model inside an iOS/macOS app: it is the only route to the **Apple Neural Engine**, it integrates with the OS's power and thermal management, and Apple's app-review and battery expectations are built around it. The tradeoff is a constrained op set and a conversion step that can fight you. If your deliverable is an App Store binary, this is the destination even if you prototyped in MLX.

**ExecuTorch** is PyTorch's edge runtime: export a PyTorch program to a portable `.pte` artifact and run it on mobile/embedded with backend delegates (CPU, GPU, various NPUs). Its strategic advantage is **staying inside the PyTorch export toolchain** — the same model definition you trained goes to the edge without a hand-maintained conversion. This is the one I'd bet on for a team that trains its own models and ships to both iOS and Android.

**ONNX Runtime** is the interoperability play: one graph format, execution providers for CPU, CUDA, DirectML, CoreML, QNN (Qualcomm), OpenVINO. Reach for it when you must support a matrix of Windows/Android/Linux devices from one artifact, or when your model is not a transformer (ONNX is excellent for the classical-ML and vision models around your LLM — the reranker, the classifier, the embedder).

**TensorRT / TensorRT-LLM on Jetson** is the answer when the edge device is an Nvidia module — a robot, a camera, an industrial box. You get kernel fusion and INT8/FP8 with an ahead-of-time engine build, and the cost is that the engine is compiled for a specific GPU architecture and TensorRT version, so your build pipeline is now hardware-pinned.

**NPU vs GPU on the same device** — the distinction candidates get wrong. An NPU (Apple Neural Engine, Qualcomm Hexagon, Intel/AMD NPUs) is a low-precision matrix accelerator optimized for **performance per watt**, typically with a constrained op set and a preference for static shapes. A mobile GPU is more flexible, faster for dynamic-shape work, and considerably hungrier. **The practical consequence: NPUs are excellent for the prefill phase (big static matmuls) and often awkward for autoregressive decode (dynamic sequence length, KV cache growth), so several production stacks run prefill on the NPU and decode on GPU or CPU.** Being able to say *why* — static shapes and op coverage, not raw throughput — is what separates a real answer here.

**⚠ Trap:** benchmarking these on a plugged-in device with a cold thermal budget and quoting the number. Every edge stack looks good for 30 seconds. The number that matters is sustained throughput after 10 minutes at temperature, on battery, with the app in the foreground and the OS scheduler doing normal things. I have seen a 2.1× difference between first-token-of-first-run and steady state on the same phone.

### What actually limits LLM inference on a phone or a laptop? Give me the constraints in order.

The order matters and it is not what people expect. **Compute is not the binding constraint on-device; memory is, in three separate ways, and then heat.**

**1. Memory capacity — does it fit at all.** A phone gives an app a hard memory allowance far below total RAM (the OS will kill you, not swap you). A 4-bit 3B at roughly 4.5 effective bits/weight is 3×10⁹ × 4.5/8 = **1.69 GB** of weights, plus KV cache, plus activations, plus your app. A 4-bit 8B is 8×10⁹ × 4.5/8 = **4.5 GB** — plausible on a high-end laptop, hostile on a phone. **This single calculation decides your model size before any quality consideration**, and it is why the 1–4B band exists.

**2. Memory bandwidth — how fast can it possibly decode.** Autoregressive decode at batch 1 reads the entire weight matrix once per token, so **tokens/sec ≤ memory_bandwidth / model_bytes**. A recent flagship phone's LPDDR5X gives on the order of 50–70 GB/s. At 1.69 GB of weights: 60/1.69 = **35 tok/s theoretical ceiling**, and real stacks land at perhaps 50–70% of that, so ~18–25 tok/s. That is roughly reading speed, which is why 3B-class on-device assistants feel usable and 8B-class ones feel slow. An M-series Mac with 100–400 GB/s of unified bandwidth changes the arithmetic completely: 400/1.69 = 237 tok/s ceiling. **📅 Volatile:** bandwidth figures move every hardware generation — the formula is the durable part.

**3. Model load time.** Cold-starting a 1.7 GB model means reading 1.7 GB from flash. At ~1.5 GB/s sustained NVMe/UFS read that is **1.1 seconds** in the best case, and much worse if the file is not page-cache resident or if you must decompress or convert. Users perceive this as the app being broken. Mitigations: mmap the weights so pages fault in lazily (llama.cpp does this, and it is why it appears to load instantly), keep the model resident across app sessions if the OS permits, and pre-warm during a splash screen. **Load time is the single most under-modelled on-device constraint** — a beautiful 20 tok/s model that takes 6 seconds to start will lose to a worse model that starts in 200 ms.

**4. Thermal throttling.** Sustained inference saturates memory bandwidth and the matmul units, the SoC heats, and the governor drops clocks. The characteristic curve is full speed for 30–90 seconds, then a step down to 50–70% of peak, sometimes further. **This means your p50 latency measured on the first request is a lie about your p99 in a long conversation.** The engineering response is to design for the throttled number, and to consider deliberately capping throughput below thermal limit so latency is *predictable* rather than fast-then-slow — users tolerate consistent slowness far better than variable speed.

**5. Battery.** A phone battery holds roughly 15–20 Wh. Sustained LLM inference can draw on the order of 5–8 W on the SoC. At 6 W that is 15/6 = **2.5 hours of continuous generation to flat**, and no product wants to own that. Practically this caps on-device inference at bursty, short interactions, and it is a real argument for routing long generations to the cloud even when the model fits locally. **NPUs matter here far more than in throughput terms: doing the same matmuls at 1.5 W instead of 6 W is the difference between a feature and a battery complaint in the App Store review.**

**🗣 Say this in the room:** "On-device is memory-bound, not compute-bound. Capacity decides which model fits — 4-bit 3B is about 1.7 GB. Bandwidth decides the speed ceiling — tokens per second is bandwidth divided by model bytes, so ~60 GB/s over 1.7 GB is a 35 tok/s ceiling and maybe 20 real. Then load time and thermal throttling decide whether it feels good, and battery decides how long a session can be. I design for the throttled steady-state number, not the first-30-seconds number."

### Derive the latency benefit of 4-bit post-training quantization on device. Why is it 40% and not 75%?

This is a great question because the naive arithmetic gives you a wildly wrong answer and the gap between the naive answer and the measured one *is* the mechanism.

**The naive derivation.** Decode at batch 1 is memory-bandwidth-bound: you read every weight once per token. Going from fp16 (16 bits/weight) to 4-bit (with a realistic ~4.5 effective bits/weight once you include per-group scales and zero-points) reduces bytes read by 16/4.5 = **3.56×**. If decode time were purely weight reads, latency would drop by 1 − 1/3.56 = **72%**. That is the "75%" the question is contrasting against.

**Why the measured end-to-end number is closer to ~40%:**

1. **Prefill is compute-bound, not bandwidth-bound.** Processing a 500-token prompt is a set of large matmuls with high arithmetic intensity; quantization does not speed that up proportionally, and on hardware without native 4-bit matmul units you *dequantize to fp16 and multiply anyway*, so the compute is unchanged and you have added dequant work. In a request with a long prompt and a short answer, prefill dominates and the quantization benefit largely vanishes.
2. **Dequantization costs ALU cycles.** Every 4-bit block must be unpacked and scaled before the matmul. That is real work in the inner loop, and on CPU it can eat 15–30% of the bandwidth savings.
3. **The KV cache is usually not quantized** (or is quantized separately to 8-bit). As the conversation grows, KV reads become a larger fraction of per-token bytes and they didn't shrink.
4. **Non-weight tensors stay wide.** Embeddings, layer norms, the LM head (often left at 8-bit or 16-bit for quality), activations, and the softmax all remain at higher precision.
5. **Fixed overheads.** Tokenization, sampling, framework dispatch, and inter-process communication in a mobile app don't scale with weight precision.
6. **You may become compute-bound after quantizing.** That is the healthy outcome — you removed the bottleneck and hit the next one — but it caps the gain.

So: **📐 Numbers you must know — 4-bit PTQ on a typical edge stack delivers roughly a 3.5× reduction in model bytes and a 3–3.5× reduction in *memory footprint*, but only on the order of a 40% reduction in end-to-end task latency for a mixed prefill/decode workload.** The footprint reduction is the reliable, derivable win; the latency reduction is workload-dependent and you should always state the prompt/generation ratio you measured it at. **📅 Volatile:** hardware with native 4-bit or FP4 matmul units changes this materially — re-measure per target device.

**Quality cost.** 4-bit PTQ with a good method (GPTQ-style second-order error compensation, AWQ-style activation-aware scaling, or llama.cpp's k-quants with an importance matrix) typically costs a small perplexity increase and a larger, task-dependent hit on hard reasoning. **📄 Paper:** Frantar et al. (2022), GPTQ — one-shot layer-wise quantization using approximate second-order information, replacing round-to-nearest as the default for 3–4 bit LLM weights. **📄 Paper:** Lin et al. (2023), AWQ — protects the salient weight channels identified from activation magnitudes rather than weight magnitudes.

**⚠ Trap:** validating quantization with perplexity only. Perplexity is an average over a language-modelling corpus and it is remarkably insensitive to exactly the capabilities you care about — instruction following, JSON schema adherence, tool-call argument correctness, and multi-step arithmetic degrade measurably at 4-bit while perplexity moves by a hair. **My rule: quantization is validated on the task eval, not on perplexity, and the acceptance criterion is stated per-task before the quantization runs.** The smaller the model, the sharper this cliff — a 70B tolerates 4-bit far better than a 3B does, because the 3B has less redundancy to spend.

### Design a hybrid on-device / cloud system where privacy is the actual requirement, not a nice-to-have.

**The mental model that keeps this design honest: the router's job is not to pick the better model, it is to enforce a data-flow policy. Quality and cost are the tiebreakers *within* what the policy allows, never across it.** If you present this as "route easy queries locally and hard ones to the cloud," an interviewer at a healthcare, legal, or financial customer will correctly conclude you have never shipped under a data-residency obligation.

**Layer 1 — classify the data, not the query.** Before anything else, the system determines whether this request's content is allowed to leave the device/region. Sources of that determination: an explicit user or admin setting; a tenant's contractual residency terms (EU data stays in the EU; this customer's data never leaves their VPC); a content classifier for regulated categories (PHI, payment data, identifiers); and the *surface* itself (a "confidential mode" toggle). The output is a policy label: `local_only`, `region_pinned`, or `unrestricted`.

**Layer 2 — route within the policy.**
- `local_only` → the on-device model handles it, and if the on-device model cannot, the correct behaviour is **to say so**, not to silently escalate. A graceful "this needs the cloud model; may I send it?" prompt is a product decision, and it must be a decision, not a default.
- `region_pinned` → a model deployed in that region, possibly in the customer's own VPC. Note this rules out most consumer frontier APIs and pushes you toward open-weight models you host, which is one of the strongest real-world drivers for the whole small-model discipline.
- `unrestricted` → now, and only now, do cost/quality routing: local for short/simple, cloud for long/hard.

**Layer 3 — make the split useful rather than binary.** The interesting engineering is in the middle:
- **On-device does the sensitive part, cloud does the general part.** The device extracts entities from a private document and sends only an abstracted query to the cloud; the cloud returns general knowledge; the device recomposes. The private text never leaves.
- **On-device does retrieval, cloud does generation on redacted context** — with the redaction performed locally by a small model and verified by a deterministic checker.
- **Speculative decoding across the boundary is not the answer** — the draft tokens themselves leak content. Say this if asked; it is a trap question.

**Layer 4 — prove it.** A privacy claim you cannot demonstrate is a liability. Ship with an **egress audit log** that records, per request, the policy label, the destination, and a hash of what was sent; make it inspectable by the customer. Add an integration test in CI that asserts no network egress occurs for `local_only` fixtures — an actual socket-level assertion, not a code review. **That test is the artifact that makes the claim real, and mentioning it is what makes this answer land.**

**💰 Math:** the business case usually is not cost. Suppose 40% of a 20 M-request/month workload is `local_only`. Those 8 M requests cost you $0 in API spend (saving 8e6 × $0.00585 = **$46,800/month**) but they cost you the entire on-device engineering program — model selection, quantization, per-platform integration, thermal tuning, an update channel — which is comfortably a 2–3 engineer-year investment. **The justification is almost never the $46,800; it is the enterprise deals or the regulated market you cannot enter otherwise.** Frame it that way in the room: on-device is a market-access decision with a cost side-effect, not a cost decision.

### Do the build-versus-buy break-even properly, with the sensitivity to price deflation.

Here is the full arithmetic. I'll use a concrete, defensible shape and then show which term dominates.

**Workload:** an extraction task, 20 M requests/month, 1,200 input tokens (of which 900 are a stable shared prefix) and 150 output tokens.

**Option A — hosted frontier API.** At $3/Mtok input, $15/Mtok output (**📅 Volatile**): uncached per-request = 1200/1e6 × 3 + 150/1e6 × 15 = 0.0036 + 0.00225 = **$0.00585**. With prompt caching on the 900-token prefix at a 90% discount on cached reads, that prefix drops from 900/1e6 × 3 = $0.0027 to $0.00027, saving $0.00243 → **$0.00342/request**. Monthly: 20e6 × 0.00342 = **$68,400.**

**Option B — distilled 3B, self-hosted.**

*One-time build:*
- Teacher labelling: 20,000 examples × (1,200 in + 400 out with thinking) at strong-model settings ≈ 20,000 × $0.0096 = **$192**; halved on the batch tier → **~$100**.
- Human verification of 2,000 samples at $0.50 = **$1,000**.
- Engineering: 3 weeks of a senior engineer at a fully-loaded ~$8,000/week = **$24,000**; plus 1 week building the eval harness = **$8,000**.
- Training compute: LoRA on 3B, ~10 runs × 4 GPU-hours at $2/h = **$80**.
- **Total one-time ≈ $33,200.**

*Monthly run:* the workload is 20e6 × 150 = 3×10⁹ output tokens and 20e6 × 1,200 = 2.4×10¹⁰ input tokens. At ~1,500 output tok/s and ~15,000 prefill tok/s aggregate on one accelerator: decode time = 3e9/1500 = 2.0×10⁶ s; prefill = 2.4e10/15000 = 1.6×10⁶ s; total = 3.6×10⁶ GPU-seconds = **1,000 GPU-hours/month**. A month is 730 hours, so 1.37 GPUs at 100% utilization; at a realistic 40% average utilization (diurnal peaks) you need ~3.4, round to **4 GPUs for headroom and HA**: 4 × $0.80/h × 730 = **$2,336/month**, call it $2,800 with overhead.
Plus the ongoing terms people forget: shadow-teacher monitoring at 2% = **$3,840/month**; human review = **$1,440/month**; 0.2 FTE maintenance = **$6,900/month**.
**Total monthly ≈ $15,000**, of which compute is 19%.

**Break-even:** monthly saving = $68,400 − $15,000 = **$53,400**. Payback on $33,200 = **0.62 months, about 19 days.** At this volume, building is obviously correct.

**Now the break-even *volume*, which is the real question.** Fixed monthly cost (monitoring + review + FTE) is $12,200 and is volume-insensitive; marginal GPU cost is $2,800/20e6 = $0.00014/request. Set V × 0.00342 = 12,200 + V × 0.00014:
V × 0.00328 = 12,200 → **V = 3.72 million requests/month.** Below that, buying wins.

**💰 Sensitivity to price deflation — the part that decides your strategy.** Token prices have fallen sharply and repeatedly (**📅 Volatile:** the widely-cited figure is on the order of 80% over a recent 12–18 month window for a given capability tier; verify the current curve). Suppose an 80% cut. API per-request → $0.000684. Redo the break-even: V × 0.000684 = 12,200 + V × 0.00014 → V × 0.000544 = 12,200 → **V = 22.4 million requests/month.** The break-even volume moved **6×**.

**The three conclusions I'd state out loud:**
1. **The fixed human cost — the FTE and the review loop — is the term that does not deflate.** Compute deflates; people don't. So the break-even is dominated by maintenance, and any build proposal that omits the FTE is off by 5×.
2. **Model your payback against *future* prices, not today's.** A project with an 18-month payback at today's prices has a negative NPV if prices fall 80% in year one. My rule: **if the payback period exceeds ~9 months at current prices, do not build**, because the price you're arbitraging against will likely be gone before you break even.
3. **The exceptions that ignore all of this arithmetic** are latency floors, data residency, and rate-limit ceilings. If you *cannot* send the data out, cost is not the decision variable and the break-even calculation is not the argument you should be making.

### Give me the model-selection scorecard you'd actually defend in a design review.

I keep this as a scored table because the alternative — an argument about vibes and a benchmark screenshot — loses to whoever talks loudest. **The discipline is that every row has a *measurement*, not an opinion, and that the hard constraints are evaluated as gates before anything is scored.**

**Gates (pass/fail; a failure ends the evaluation regardless of score):**
1. **Data-flow policy** — can this model legally and contractually receive this data, in this region, under this tenant's terms?
2. **License** — can we ship what we're building? (Apache-2.0/MIT clean; Llama-family MAU and naming clauses checked; research-only rejected for commercial.)
3. **Latency ceiling** — measured p95 TTFT and p95 end-to-end on *our* prompt shape, on *our* hardware, under *our* concurrency. Not the vendor's number.
4. **Capability floor** — ≥ X% on our 200-case golden set. If it can't do the job, price is irrelevant.
5. **Rate limits / capacity** — can it absorb our peak? A model we can't get quota for is not an option.

**Scored dimensions (weights are per-product; these are my defaults for an applied-AI feature):**

| Dimension | Weight | How it's measured |
|---|---|---|
| Task quality on our golden set | 30% | Paired eval, per-segment, with CIs |
| Cost per resolved task | 20% | Not $/Mtok — $/task, including retries and escalations |
| p95 latency (TTFT + total) | 15% | Measured under production concurrency |
| Format/tool-call reliability | 10% | % schema-valid, % correct tool args, over 500 runs |
| Operational risk | 10% | Provider availability history, deprecation cadence, quota stability |
| Context capacity, *usable* | 5% | Measured degradation curve, not the advertised limit |
| Prefix-caching / batch economics | 5% | Cache-hit discount and hit rate on our traffic shape |
| Portability / exit cost | 5% | Days to swap; is our prompt/tooling provider-specific? |

**Three things that make this scorecard credible rather than theatrical:**

**"Cost per resolved task," not cost per token.** A cheaper model that needs 2.4 attempts to produce valid output, or that escalates 18% of traffic to a bigger model, is not cheaper. Compute: (base cost × attempts) + (escalation rate × escalation cost). A model at $0.30/Mtok with 2.4 attempts costs more than one at $0.60/Mtok with 1.05 attempts.

**"Usable context," not advertised context.** Measure it: take your real task, extend the context with realistic distractor content, and plot task accuracy against context length. The point where accuracy falls below your bar is your usable limit, and it is routinely a fraction of the advertised number. Put the curve in the review.

**A re-evaluation trigger.** The scorecard has an expiry date. **My rule: re-run it whenever a gate-relevant fact changes (price, license, quota), and on a fixed 90-day cadence regardless.** And the eval harness must be model-agnostic from day one — if switching models requires a week of prompt rework, you have built a switching cost that will keep you on a worse model for a year.

**🗣 Say this in the room:** "I gate first — data-flow policy, license, latency ceiling, capability floor, quota — because those are pass/fail and no score rescues a failure. Then I score on cost per *resolved task* rather than per token, measured quality on our own golden set per segment, p95 under real concurrency, and format reliability. And I keep the eval harness model-agnostic so the switching cost stays low, because at this rate of change the ability to re-run the scorecard in a day is worth more than picking the right model today."

### What's different about deploying AI in a quant or bank setting? Assume I'll push back on hand-waving.

Three things are structurally different — latency floors, determinism, and governance — and each one changes the architecture rather than just adding paperwork.

**Latency floors, and where LLMs actually sit.** The reflex answer is "finance is low-latency so LLMs don't fit," which is half right in a way that matters. A tick-to-order path in a market-making system is measured in **microseconds to low milliseconds**, often with FPGA or kernel-bypass networking; a transformer forward pass is three to six orders of magnitude too slow and will never be in that loop. So LLMs live in the **adjacent** paths where they legitimately belong: extracting structure from filings, earnings-call transcripts and news; summarizing research; generating and reviewing backtest code; compliance surveillance over chat and voice; client reporting; and — increasingly — pre-computing features that a fast model consumes. **The correct framing is: the LLM produces a feature or an artifact offline or near-line; the microsecond path consumes a number.** Saying that crisply immediately separates you from candidates who propose putting a model in the hot path.

**Determinism and reproducibility.** A backtest, a risk number, or a regulatory report must be reproducible, and "temperature 0" is *not* determinism — this is the technical point I'd want to make. Even at temperature 0 you get non-reproducible outputs from: floating-point reduction-order differences across batch sizes and GPU kernels; continuous batching changing which requests are batched together; provider-side model updates behind a stable alias; and any tie-breaking in the sampler. **The controls that actually work:** pin an explicit dated model version and forbid floating aliases in production; log the exact prompt, model version, sampling parameters, and full response for every call, immutably; treat the LLM as a **non-deterministic function whose output is an input to a deterministic pipeline** — i.e. store the extracted JSON as the artifact of record and re-run downstream logic from *that*, never re-calling the model; and where a number must be exactly reproducible, cache the model output keyed by a hash of the input and serve the cache on re-run. **The insight: you don't make the model deterministic, you make the *record* authoritative.**

**Model risk governance.** US banks operate under **SR 11-7 / OCC Bulletin 2011-12**, the supervisory guidance on model risk management, and its three pillars land directly on any LLM you deploy: (1) **sound development with documented conceptual soundness** — you must be able to explain why this model is appropriate for this use, which is hard for a black box and is why constraining the model to a narrow, verifiable task helps enormously; (2) **independent validation with "effective challenge"** — a team that did not build it must be able to test and reject it, which means your eval harness has to be usable by someone who isn't you; (3) **ongoing monitoring** against benchmarks and outcome analysis. Add a **model inventory** entry with owner, purpose, tier, and validation date; plus jurisdiction-specific overlays (FINRA supervision obligations for communications, MiFID II record-keeping in the EU, and the EU AI Act's obligations where a use case lands in a higher-risk tier — **📅 Volatile:** the AI Act's obligations phase in on a published schedule; verify current dates before quoting them).

**🗣 Say this in the room:** "Three deltas. One, LLMs are never in the microsecond path — they produce features and artifacts near-line and a fast deterministic system consumes them. Two, temperature zero isn't determinism, because batching and kernel reduction order and silent model updates all break reproducibility, so I pin dated model versions, log the full request and response immutably, and make the *stored extraction* the artifact of record rather than re-calling the model. Three, under SR 11-7 the model needs documented conceptual soundness, independent validation with effective challenge, and ongoing monitoring — which in practice means my eval harness has to be operable by a validator who didn't build the system."

**⚠ Trap:** proposing an autonomous agent that takes actions in a regulated workflow. In this environment the defensible pattern is **human-in-the-loop with the model producing a reviewable proposal and a full audit trail**, not an agent executing. I would push back on an autonomy design here in a review, and I would frame the pushback as risk-tiering rather than as a blanket prohibition: full autonomy for reversible, low-materiality, internally-scoped actions; proposal-plus-approval for anything client-facing, financially material, or reportable.

### Last one: give me a set of drills that would tell me I've actually mastered this material.

Four drills, timed, unaided, with explicit pass criteria. If you can pass all four you can hold a ranking-team interview at Netflix, Uber, Airbnb, Pinterest or Meta, and a small-model/edge conversation at a product company.

**🏋 Drill 1 — the retrieval-plus-ranking whiteboard (45 min, no notes).** Design a recommender for a 50-million-item catalog at 20,000 QPS. Pass criteria: you name all four funnel stages with per-stage latency budgets that sum inside 150 ms; you write the two-tower loss including the logQ correction and can say why it's there; you compute the embedding-table memory (IDs × d × bytes) and the HNSW memory (vectors + M×2×4 per node) out loud; you name three candidate sources, not one; you state how the ranker's training distribution depends on the retriever and what that implies for shipping order; and you close by describing how you'd know it works — offline metric, interleaving gate, A/B with a guardrail. **Fail if you have not stated a single number.**

**🏋 Drill 2 — the evaluation interrogation (30 min, written).** Given "our new ranker is +4% NDCG@10 offline," write the five-step triage for why it might be flat online, with the arithmetic for the experiment's MDE at 2 M sessions/arm and a 5% baseline. Then write, from memory: the IPS estimator, the effective-sample-size formula, and the reason OPE is undefined on logs from a deterministic policy. Pass criterion: MDE computed correctly to two significant figures (SE = √(0.05×0.95/2e6) = 1.54e-4; two-arm MDE = 2.80 × √2 × 1.54e-4 ≈ 6.1e-4 absolute), and the deterministic-policy point stated as a *support* problem, not a *variance* problem.

**🏋 Drill 3 — search primitives, NumPy and stdlib only (50 min).** Implement, in one file: (a) `top_k_stream` with a bounded min-heap; (b) a BM25 scorer over an inverted index you build from a list of documents, with k1 and b as parameters; (c) MinHash signatures with m=128 and LSH banding with a stated (b, r), including the function that computes the S-curve threshold; (d) RRF fusion of two ranked lists. Pass criteria: BM25 agrees with a reference implementation to 1e-6 on a 20-document toy corpus; your LSH threshold function reproduces (1/b)^(1/r); no library beyond `heapq`, `math`, `hashlib`, `collections`, `numpy`. **Then, unprompted, name the edge case in each — that self-naming step is half the grade in a real round.**

**🏋 Drill 4 — the economics defence (20 min, spoken out loud).** Someone proposes fine-tuning a small model to replace a frontier API call. Deliver, without notes: the per-request API cost with and without prefix caching; the four one-time build cost lines; the four ongoing cost lines with the FTE included; the break-even *volume*; and the sensitivity of that break-even volume to an 80% price cut. Pass criterion: you arrive at a break-even volume in the low millions of requests per month at current prices and roughly 6× higher after deflation, **and you say out loud that the fixed human cost is the term that does not deflate.** That sentence is the whole point of the drill, and it is the sentence that gets remembered in a debrief.

**🗣 Say this in the room, when the interviewer asks what you'd do first on their ranking team:** "I'd find out what the current offline-to-online correlation actually is. Every other decision — how much to trust NDCG, whether to invest in interleaving, whether off-policy evaluation is even possible on our logs — depends on that one number, and most teams have never measured it."
