### Teach me logistic regression from the ground up, and then tell me why it's still the first thing you fit.

Start from what you want: a number between 0 and 1 that you can treat as a probability. You have a feature vector `x ∈ R^d`. The simplest thing that uses all of it is a weighted sum `z = w·x + b`, which lives in `(-∞, ∞)`. So squash it: `p = σ(z) = 1/(1 + e^{-z})`. That is the entire model. Logistic regression is a single linear layer followed by a sigmoid — literally a one-neuron neural network — and every intuition you build here transfers directly to the final layer of every classifier you will ever see, including the LM head of a transformer (which is the same thing with a softmax over 128k classes instead of a sigmoid over 2).

The training objective is negative log-likelihood, and it is worth seeing why it is forced rather than chosen. Under the model, the likelihood of a labeled example is `p^y (1-p)^{1-y}`. Take the log, negate, sum: `L = -Σ [y log p + (1-y) log(1-p)]`. That is binary cross-entropy. Its gradient with respect to the weights is beautifully simple — `∂L/∂w = Σ (p_i - y_i) x_i` — prediction minus truth, scaled by the input. Exactly the same form as the softmax cross-entropy gradient in a transformer, and for the same algebraic reason: the sigmoid's derivative cancels against the `1/p` from the log.

```python
import numpy as np
def fit_logreg(X, y, lr=0.1, l2=1e-3, steps=2000):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y                                  # [N]
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b
```

Twelve lines, no framework. That is the version I expect a senior candidate to write on a whiteboard without hesitation.

Why it is the first model I fit, every time: the coefficients are directly interpretable (`w_j` is the change in log-odds per unit of feature `j`), the loss is convex so there is exactly one optimum and no seed-dependence, it trains on a million rows in seconds, it is naturally well-calibrated when the link function matches the data-generating process, and — the part that matters in interviews — **it establishes the number that every fancier model must beat.** If your transformer fine-tune beats logistic regression on TF-IDF by 0.4 points of F1, you have not built an AI system, you have built an expensive one.

**⚠ Trap:** unregularized logistic regression on perfectly separable data diverges — the weights grow without bound because pushing `z` to infinity keeps decreasing the loss. You see it as coefficients in the hundreds and a convergence warning. It is also the classic *symptom of leakage*: perfect separation usually means a feature encodes the label. When I see huge coefficients, my first hypothesis is not "increase max_iter," it is "which column did I accidentally include?"

### Explain gradient boosting to me — what is being boosted, and why does XGBoost still beat deep learning on tabular data?

The mental model: gradient boosting is gradient descent performed *in function space instead of parameter space*. Ordinary training nudges weights in the direction that reduces loss. Boosting instead asks "what function, added to my current prediction, would reduce the loss fastest?" — computes the negative gradient of the loss with respect to the *current predictions*, and fits a small decision tree to those residuals. Then adds that tree, scaled by a learning rate, to the ensemble. Repeat a few hundred times. The "gradient" in gradient boosting is a gradient with respect to the model's output, not its parameters, and once that clicks the whole family makes sense.

Mechanically, at round `m` you have `F_m(x)`. You compute per-example pseudo-residuals `r_i = -∂L(y_i, F_m(x_i))/∂F_m(x_i)` — for squared error that is literally `y_i - F_m(x_i)`, for logistic loss it is `y_i - p_i`. Fit tree `h_m` to predict `r`. Set `F_{m+1} = F_m + η·h_m` with `η ≈ 0.03–0.1`. XGBoost's contribution on top of this was a second-order expansion (using both gradient and Hessian to choose splits and set leaf values), an explicit regularization term on tree complexity, and a sparsity-aware split finder with a default direction for missing values.

**📄 Paper:** Chen & Guestrin (2016), XGBoost — regularized second-order boosting with a scalable split-finding algorithm; it replaced hand-rolled GBM implementations and won essentially every tabular Kaggle competition for years. **📄 Paper:** Ke et al. (2017), LightGBM — histogram-based binning of features plus leaf-wise (best-first) tree growth, giving order-of-magnitude speedups on wide data. LightGBM is what I reach for by default on anything above a million rows; XGBoost when I want the more conservative depth-wise growth on smaller data.

Why trees still win on tabular data — and this is a real, published finding, not folklore: **📄 Paper:** Grinsztajn et al. (2022), "Why do tree-based models still outperform deep learning on typical tabular data?" The reasons they isolate are that tabular features are often non-smooth step functions of the target (trees represent axis-aligned steps natively, MLPs have to approximate them with smooth compositions), that tabular data contains many uninformative features (trees ignore them by never splitting on them, neural nets must learn to zero them), and that tabular features are not rotation-invariant — the columns mean specific things, and MLPs' rotational invariance is a *mis*match to that structure while trees' axis alignment is a match.

**🗣 Say this in the room:** "Boosting is gradient descent in function space — each tree fits the negative gradient of the loss with respect to the current predictions. On tabular data I start with LightGBM because axis-aligned splits match non-smooth feature-target relationships and it's robust to uninformative columns, and I only move to embeddings-plus-a-head when the signal is genuinely in free text."

**⚠ Trap:** the number of boosting rounds is not a hyperparameter you tune on your test set. Use early stopping on a validation split, and remember that early stopping *is* fitting — the chosen round count is a parameter learned from the validation data, so the validation score is optimistic and must not be reported as a test score. I have seen this inflate reported AUC by 1–2 points, which is exactly the size of the improvements people write blog posts about.

### Implement k-nearest-neighbours, and tell me where you're already using it without calling it that.

k-NN is the model with no training step: memorize the training set, and at inference return the majority label (or mean value) of the `k` closest stored points. The mental model that matters for this guide: **vector search is k-NN.** Every RAG system you will ever build is a k-nearest-neighbour retrieval over embeddings, usually approximate, usually with cosine distance, usually with `k` between 5 and 50. If you understand k-NN's failure modes you already understand most of retrieval's.

```python
import numpy as np
def knn_predict(X_train, y_train, x, k=5, n_classes=2):
    d = ((X_train - x) ** 2).sum(axis=1)          # squared L2, [N]
    idx = np.argpartition(d, k)[:k]                # O(N), not O(N log N)
    return np.bincount(y_train[idx], minlength=n_classes).argmax()
```

`argpartition` rather than `argsort` is the detail I look for — you need the top-k, not a full ordering, and that is `O(N)` instead of `O(N log N)`. An HNSW or IVF index is the same idea taken further: give up the exactness guarantee to get sublinear query time.

Three properties that carry straight into retrieval work. **The curse of dimensionality**: in high dimensions the ratio of the nearest to the farthest neighbour distance approaches 1, so "nearest" becomes weakly meaningful — which is exactly why raw high-dimensional distances are unreliable and why learned embedding spaces (trained so that semantic similarity *is* geometric proximity) work where raw feature spaces do not. **Scale sensitivity**: unnormalized features with different units make distance meaningless, which is why embeddings are L2-normalized and cosine similarity is used instead of raw dot product. **No abstraction**: k-NN cannot extrapolate beyond its stored points, which is precisely why a RAG system cannot answer a question whose answer is not in the corpus, no matter how good the model is.

Where I actually use k-NN as a *model* rather than as retrieval: few-shot example selection (retrieve the k most similar labeled examples and put them in the prompt — this is k-NN with an LLM as the aggregation function), near-duplicate detection in a training corpus, and as an instant baseline classifier over embeddings when I have labels but no time. That last one is genuinely strong: 1-NN over a good embedding space is often within a couple of points of a fitted linear head and requires zero training.

**⚠ Trap:** k-NN's cost is at inference, not training, and it grows linearly with your corpus. This is the opposite of every parametric model's profile, and it is why "just add more documents" degrades your retrieval latency in a way that "just add more training data" does not degrade a classifier's.

### You've got 8,000 production failures and no idea what they have in common. Walk me through clustering them.

This is the highest-value classical-ML technique in an LLM engineer's toolkit, and I would push back on anyone who calls it "not real AI work." Error analysis at scale is a clustering problem, and doing it well is the difference between fixing one bug and fixing a class of bugs.

The pipeline: embed each failure (the user input, or input plus the model's wrong output — I usually do both separately, they cluster differently), then cluster in the embedding space, then have an LLM *name* each cluster from a sample of its members. Note the division of labor: clustering is deterministic and cheap, naming is the open-ended part where the LLM belongs.

**Use HDBSCAN, not k-means.** This is not a style preference. k-means requires you to pick `k` in advance, assumes roughly spherical equal-variance clusters, and — fatally — **assigns every point to a cluster**, so your genuinely idiosyncratic one-off failures get forced into a group and pollute it. HDBSCAN infers the number of clusters from density, handles arbitrary shapes, and explicitly labels low-density points as noise (`-1`). For error analysis, "these 340 failures form a tight cluster and these 900 are unrelated singletons" is exactly the answer you want, and only one of these algorithms can give it to you.

**📄 Paper:** Campello, Moulavi and Sander (2013) — hierarchical density-based clustering that extracts a flat clustering by maximizing cluster stability, removing DBSCAN's single global `eps` parameter.

```python
import umap, hdbscan
# reduce first: HDBSCAN's density estimates degrade badly above ~50 dims
red = umap.UMAP(n_components=15, n_neighbors=15, metric="cosine",
                random_state=0).fit_transform(embs)      # [N, 1024] -> [N, 15]
labels = hdbscan.HDBSCAN(min_cluster_size=25,
                         min_samples=5).fit_predict(red)
```

`min_cluster_size` is the knob that matters and it is a *product* decision, not a statistical one: it is the smallest group of failures you would actually staff work against. If a cluster of 10 wouldn't get fixed, set it to 25.

Then quality-check with **silhouette score** — for each point, `(b - a)/max(a, b)` where `a` is mean intra-cluster distance and `b` is mean distance to the nearest other cluster. Ranges −1 to 1; above ~0.5 is a genuinely well-separated structure, around 0.2 is weak, negative means points are closer to another cluster than their own. Compute it excluding the noise points or it is meaningless.

**⚠ Trap — and this one gets people in interviews:** UMAP and t-SNE are *visualization* tools, and their 2-D output lies to you in specific, well-documented ways. Cluster sizes in the plot do not reflect real cluster sizes. Distances *between* clusters are not meaningful — two blobs on opposite sides of a t-SNE plot may be adjacent in the original space. Apparent density is an artifact of the perplexity/`n_neighbors` setting. And both will happily produce beautiful, convincing clusters from pure Gaussian noise. The rules I enforce: never cluster on 2-D UMAP output (cluster on 10–20 dims, or on the raw embeddings, and use 2-D only to draw the picture); never report a distance read off a t-SNE plot; and always sanity-check by reading 10 actual members of a cluster before you believe it. **📄 Paper:** van der Maaten & Hinton (2008) for t-SNE; McInnes, Healy and Melville (2018) for UMAP, which preserves more global structure than t-SNE but does not preserve it faithfully either.

**🏋 Drill:** take 2,000 logged queries, embed, UMAP to 15 dims, HDBSCAN, and produce a table of the top 10 clusters with size, silhouette, and an LLM-written one-line name, in 30 minutes. Pass criterion: for the largest three clusters, you can read five members each and agree the name is right. If you cannot, your `min_cluster_size` is too small.

### Derive PCA, and tell me where it shows up in a modern LLM stack.

The mental model: PCA finds the directions along which your data varies most, and re-expresses every point in terms of those directions. If most of the variance lives in 40 of your 1,024 dimensions, you can throw away 984 numbers per vector and lose almost nothing — which is a compression argument, and compression is what you care about when you are paying for vector storage and ANN search latency.

The derivation, briefly and correctly. Center the data: `X ∈ R^{N×d}` with column means subtracted (uncentered PCA is a common and wrong shortcut — the first component then just points at the mean). Form the covariance `C = XᵀX / (N-1)`, which is `d×d`, symmetric, positive semi-definite. Its eigenvectors are the principal directions and its eigenvalues are the variance along each. Sort eigenvalues descending, take the top `k` eigenvectors as columns of `W ∈ R^{d×k}`, and project: `Z = XW`. The explained-variance ratio of component `i` is `λ_i / Σλ_j`, and the cumulative curve is what you actually read to choose `k`.

In practice you never form the covariance matrix — you take the SVD of the centered `X` directly (`X = UΣVᵀ`, and `V`'s columns are the eigenvectors of `XᵀX`, with `λ_i = σ_i²/(N-1)`). This is numerically better conditioned and it is what `sklearn.decomposition.PCA` does under the hood. The connection to SVD is the same low-rank-factorization machinery that underlies LoRA, so it is worth having the equivalence at your fingertips.

Where it shows up in an LLM stack, concretely:

- **Embedding dimensionality reduction for vector stores.** 1,024-dim fp32 is 4 KB per vector; 10M vectors is 40 GB. PCA to 256 dims is 1 KB, so 10 GB — a 4× cut in RAM and roughly a 4× cut in the distance-computation cost inside the ANN index, typically for 1–3 points of Recall@10. That is often an excellent trade, and it is a real production decision I have made more than once.
- **The contrast with Matryoshka embeddings.** **📄 Paper:** Kusupati et al. (2022), Matryoshka Representation Learning — train the embedding so that its *prefixes* are themselves good embeddings, so truncating from 1,024 to 256 is a slice, no projection matrix, no fitting, no drift when the corpus changes. Where a model offers Matryoshka dimensions, prefer them to PCA. Where it does not, PCA is the fallback and you must remember to persist and version the projection matrix alongside the index.
- **Whitening and anisotropy correction.** Raw contextual embeddings occupy a narrow cone, so cosine similarities cluster in a tight band; removing the top principal components (which often encode frequency rather than semantics) can measurably improve retrieval separation.
- **Diagnostics.** The eigenvalue spectrum of a set of embeddings tells you about representation collapse — if 95% of variance is in 3 dimensions, your fine-tuned embedding model has collapsed and your retrieval is about to get much worse.

**⚠ Trap:** fitting PCA on the full corpus including your evaluation queries, then reporting retrieval metrics. The projection has seen the eval distribution. Fit on the document corpus only, and treat the matrix as a versioned artifact — if you refit it after adding documents, every existing vector in the index is now in a different basis and your recall silently collapses. That is a genuine, subtle re-indexing hazard and I would call it out in review.

### Random forest or gradient boosting — which do you pick, and when does the answer flip?

Both are ensembles of decision trees; the difference is *what the trees are for*, and that difference drives everything else.

A random forest builds many deep, low-bias, high-variance trees **in parallel and independently**, each on a bootstrap sample of rows and a random subset of features at each split, then averages them. Averaging independent high-variance estimators reduces variance; the decorrelation from feature subsampling is what makes the averaging effective. It is a variance-reduction machine. Gradient boosting builds shallow, high-bias, low-variance trees **sequentially**, each correcting the residual errors of the ensemble so far. It is a bias-reduction machine.

The practical consequences fall out of that. A random forest is nearly impossible to overfit by adding trees — more trees only reduces variance, so `n_estimators` is a compute budget, not a regularization knob. Boosting *will* overfit as you add rounds, which is why early stopping is mandatory. A forest trains embarrassingly in parallel; boosting is inherently sequential across rounds (though split-finding within a round parallelizes). A forest gives you out-of-bag error estimates free, no validation split required. Boosting almost always wins on accuracy when tuned, typically by a couple of points of AUC on structured data.

My decision rule: **boosting by default when accuracy matters and I can afford to tune; random forest when I need a robust answer in ten minutes with no tuning, when I want OOB error without carving out a validation set, or when I'm doing feature triage on a wide messy dataset.** Forests are also the better choice when your label noise is high, because sequential boosting cheerfully spends rounds memorizing mislabeled examples.

**⚠ Trap:** default impurity-based feature importances (`feature_importances_` in sklearn) are biased toward high-cardinality features — a random UUID column will rank high because it offers many possible splits. If a feature importance ranking is going to inform a business decision, use permutation importance on held-out data, or SHAP values. I have watched a team drop a genuinely predictive feature and keep a useless ID hash because they read the default attribute.

**📐 Numbers you must know:** the rough resource profile for 1M rows × 100 features. LightGBM: ~30–90 s to train 500 rounds on 8 CPU cores, model on disk a few MB, single-row inference ~20–80 μs. Random forest with 500 deep trees: several minutes to train, and the *model* can be hundreds of MB to gigabytes because deep trees store every node — which sometimes decides it, since a 2 GB model in a latency-sensitive service is a real operational problem.

### Your positive class is 1.2% of the data and your model reports 99% accuracy. Walk me through what you actually do.

The 99% is the constant predictor. Predicting "negative" for everything on a 1.2%-positive dataset gives 98.8% accuracy, so your model has demonstrated that it beat a rock by 0.2 points. The first move is not modeling, it is **deleting accuracy from the report entirely** and replacing it with precision, recall and PR-AUC. Accuracy under class imbalance is not a weak metric; it is an actively misleading one, and reporting it is a signal I read as inexperience.

Then, in order:

**Fix the metric.** Precision-recall curve and average precision, not ROC. Pick the operating point from the business cost of the two error types, not from `argmax F1`. If a false negative costs 50× a false positive, say that out loud and choose the threshold that reflects it.

**Fix the loss, not the data, first.** `class_weight="balanced"` in sklearn, or `scale_pos_weight = n_neg/n_pos` in XGBoost, re-weights the loss so the rare class contributes proportionally. This is cheap, principled, and preserves the true prior. Do this before you touch resampling.

**Resample only if reweighting is insufficient**, and know what each does. Random undersampling of the majority throws away data — fine when you have 10M negatives, wasteful when you have 50k. Random oversampling of the minority duplicates rows and encourages memorization of those exact points. **📄 Paper:** Chawla et al. (2002), SMOTE — synthesizes new minority points by interpolating between a minority example and its neighbours, which works reasonably in low-dimensional numeric feature spaces and works poorly in high-dimensional embedding space, where the interpolants land off-manifold and mean nothing. I do not use SMOTE on text embeddings and I would question anyone who does.

**⚠ Trap — the one that gets shipped:** resampling before the train/test split, or inside cross-validation folds incorrectly. If you SMOTE the whole dataset and then split, synthetic points interpolated from a training example can land in your test set, and you have leaked. Resampling belongs *inside* the CV loop, applied to the training fold only. `imblearn.pipeline.Pipeline` exists specifically because `sklearn.pipeline.Pipeline` gets this wrong.

**Then fix your calibration**, because resampling destroys it. If you undersample negatives 10:1, your model's output probabilities are now on a distorted prior and `p = 0.5` no longer means 50% likely. Either correct the intercept analytically or fit Platt scaling / isotonic regression on a held-out set drawn from the *original* distribution. Calibrated probabilities are what let you set a threshold from a cost matrix, so if you skipped this you cannot do the step you started with.

**💰 Math:** why the threshold matters more than the model. At 1.2% prevalence and 1M items/month, suppose a model at threshold A gives recall 0.80 / precision 0.25 and at threshold B gives recall 0.60 / precision 0.60. Positives = 12,000. Threshold A: 9,600 caught, and 9,600/0.25 = 38,400 flagged total, so 28,800 false positives. Threshold B: 7,200 caught, 12,000 flagged, 4,800 false positives. If human review costs $0.40 per flag, A costs $15,360/month of review and B costs $4,800 — a $10,560/month difference for 2,400 additional catches, so **$4.40 per additional catch.** That number, not the AUC, is the conversation to have with the business.

### What is feature leakage, how do you catch it, and what's its LLM-era equivalent?

Leakage is any situation where your training data contains information that will not be available — or will not be available *in that form* — at prediction time. The model learns the shortcut, your offline metrics look spectacular, and production accuracy craters. The tell is almost always the same: **a number that is too good.**

The classical taxonomy, which I run as a checklist:

- **Target leakage.** A feature is a downstream consequence of the label. Predicting churn using `cancellation_reason_filled_at`. Predicting fraud using `chargeback_amount`. The feature only exists because the outcome happened.
- **Temporal leakage.** You used a random split on time-ordered data, so the model trained on the future to predict the past. Any dataset with a timestamp needs a time-based split, and the validation window must sit strictly after the training window with a gap the size of your real prediction horizon.
- **Group leakage.** The same entity appears in train and test — the same user, the same document, the same near-duplicate support ticket. The model memorizes the entity. Use `GroupKFold` on the entity ID.
- **Preprocessing leakage.** You fit the scaler, the imputer, the TF-IDF vocabulary or the PCA matrix on the full dataset before splitting. Every one of those has now seen the test set. This is what `sklearn.pipeline.Pipeline` inside `cross_val_score` exists to prevent, and it is why fitting transformers outside a pipeline is a code-review reject for me.

How I catch it: **if a single feature gives near-perfect AUC on its own, treat it as guilty until proven innocent.** Run a per-feature univariate AUC scan as a matter of routine — it takes seconds and catches most target leakage. Then ask, for every feature in the model, the physical question: *at the moment I make this prediction in production, does this value exist yet?* Half the time someone cannot answer, which is itself the answer.

**The LLM-era equivalents**, and interviewers absolutely ask this bridge:

- **Benchmark contamination.** The model saw your eval set during pretraining. This is leakage at civilizational scale and it is why a model scoring 92% on a public benchmark may score 61% on your private restatement of the same task. Mitigation: hold out private evals, generated after the model's cutoff, and never publish them.
- **Few-shot leakage.** Your prompt's few-shot examples were drawn from the eval set. Trivially easy to do by accident when the eval set and the example bank come from the same log dump.
- **Retrieval leakage in RAG evaluation.** Your eval questions were *written from* the documents in your corpus, often verbatim, so retrieval is easy in a way real user queries never are. Your Recall@5 of 0.94 is measuring lexical overlap you created.
- **Judge leakage.** The LLM judge grading outputs is the same model that produced them, and it prefers its own outputs. Well-documented self-preference; use a different model family for judging, or a human-calibrated rubric.

**🗣 Say this in the room:** "The first thing I do with any suspiciously good number is a per-feature univariate AUC scan and a check that every feature physically exists at prediction time. The LLM version of the same discipline is a private eval set built after the model's training cutoff — a public benchmark number tells me about contamination as much as capability."

### Define train/serve skew, and give me its analogue in an LLM feature.

Train/serve skew is when the transformation applied to a feature at training time differs — in code, in data source, or in timing — from the transformation applied at serving time. The model was fit on one distribution and is being asked to predict on another, and nothing crashes.

The three canonical forms. **Code skew:** the training pipeline computes `avg_order_value` in a pandas notebook and the serving path recomputes it in a Java service, and the two disagree on how to handle nulls. **Time skew:** training used a feature aggregated over a full 30-day window, serving computes it over whatever data has landed in the warehouse, which lags by 6 hours, so the serving value is systematically stale. **Source skew:** training read from the analytics warehouse, serving reads from the OLTP replica, and the two have different deduplication semantics. The industry's structural answer is a feature store with a single transformation definition materialized to both an offline and an online store — which, in your vocabulary, is just "don't have two implementations of the same function."

The LLM analogues are pervasive and less well recognized:

- **Prompt/template skew.** You evaluated with one prompt template and production renders a slightly different one — an extra newline, a different system-message ordering, a chat template applied by the client library that your eval harness did not apply. This measurably moves quality, and the fix is that the eval harness must call *the same code path* as production, not a reimplementation of it. The rule I enforce: eval calls your `build_messages()` function, never its own copy.
- **Retrieval skew.** Your evaluation retrieves from a static snapshot; production retrieves from a live index with different chunking, a different embedding model version, or documents added since. Your offline Recall@k is measuring a corpus that no longer exists.
- **Tokenizer/version skew.** The eval ran against `model-x-2025-06`, production is pinned to `model-x-latest`, and the provider rolled it. Same skew, no code change on your side. Pin model versions explicitly; `latest` in a production config is the same class of error as `:latest` on a container image.
- **Truncation skew.** Your eval inputs are 2k tokens; real user inputs are 30k and get truncated by a middleware you forgot about. The model is seeing a different input than you think.

**⚠ Trap:** "we log the prompt" is not sufficient. You must log the *rendered final payload* — messages array, tools array, model ID, temperature, and the prompt-template version hash — on a sample of production requests, and periodically replay a sample of those exact payloads through your eval harness. If replayed production payloads score differently from your eval-set payloads, you have skew, and that diff is the only reliable detector. I would call this a required capability for any LLM feature above toy scale.

### How do you detect drift on a deployed classifier, and what changes when the model is an LLM?

The mental model: you almost never get ground-truth labels in production quickly enough to measure accuracy directly, so drift detection is the discipline of **monitoring proxies that move before your accuracy does.**

Three distinct things get confused under the word "drift." **Covariate shift** — `P(x)` changes, `P(y|x)` does not. New users from a different country write different-looking tickets, but the mapping from ticket to queue is unchanged; often survivable. **Label/prior shift** — `P(y)` changes. Spam campaigns come in waves; your calibration is now wrong even though the model is fine. **Concept drift** — `P(y|x)` changes. The policy was rewritten, so the same input now has a different correct label; this one requires relabeling and retraining and nothing else will fix it.

What I monitor, in order of how early it fires:

1. **Input distribution.** Population Stability Index per feature, or a two-sample KS test against a fixed reference window. PSI heuristics in wide use: below 0.1 stable, 0.1–0.25 moderate shift worth investigating, above 0.25 significant. For embeddings, monitor the mean cosine distance to a reference centroid, or the fraction of inputs whose nearest reference neighbour is beyond a distance threshold.
2. **Prediction distribution.** The share of predictions per class, and the *distribution of confidence scores*. A drop in mean max-probability is often the earliest usable signal, because the model is telling you it is less sure before anyone tells you it is wrong.
3. **Delayed ground truth.** Whatever labels arrive — human review decisions, chargebacks, user corrections — feed a rolling accuracy estimate on the subset you have. Be honest that this subset is biased (you only review what you flagged), and correct for it with a small random-sample audit.
4. **A permanent random-sample audit.** 200 random items a week, labeled by humans regardless of what the model said. This is the only unbiased estimate you will ever have and it is worth its cost.

For an LLM feature the machinery is the same and the signals change. You cannot compute PSI on free text, so you embed inputs and monitor the embedding distribution; you monitor output length distribution, refusal rate, schema-validation failure rate, tool-call error rate, and retrieval score distributions. And you get a genuinely new drift source that classical ML does not have: **the model itself changes underneath you.** A provider silently updating a `-latest` alias is concept drift you did not cause and cannot see in your input monitoring. The defense is a canary eval — a fixed set of 200 prompts with known-good outputs, run hourly, alerting on score deltas. That job costs perhaps 200 × $0.005 = $1 per run, $720/month at hourly cadence, and it is the cheapest insurance in the stack.

**🔍 Failure taxonomy — "quality dropped last Tuesday", in the order I check:**
1. Did the model version change? Check the provider's version alias and your pinned config. (Fastest to check, surprisingly often the answer.)
2. Did the prompt change? Diff the prompt-version hash on production spans across the boundary.
3. Did retrieval change? Compare retrieval score distributions and index document counts before/after; look for a reindex job in the deploy log.
4. Did the *input* change? Embedding-distribution monitor, plus a manual read of 20 recent inputs vs 20 from two weeks ago.
5. Did the world change — a product launch, a policy update, a news event that shifted what users ask? This is real concept drift and needs new labels.
6. Only now consider that the model "got worse" spontaneously. It almost never did.

### A take-home says "build an LLM system to categorize support tickets." How do you use the classical-ML baseline to win it?

The rubric on these take-homes weights evaluation methodology heavily, and the fastest way to demonstrate methodology is a baseline table. Most submissions contain a prompt, a loop, and a claim. Yours should contain a comparison.

Here is what I would build in the four hours, and the ordering is deliberate.

**Hour 1 — the eval set, before any modeling.** Stratified split of the provided data, held out. Define the metric and *justify it in the README*: macro-F1 if the categories matter equally regardless of frequency, micro-F1 if throughput is what matters, and report both plus a per-class table so the tail classes are visible. Add a bootstrap confidence interval — 1,000 resamples of the test set, report the 2.5th and 97.5th percentiles. This one addition puts you above most submissions on its own, because it shows you know a 1.5-point difference on 300 examples is noise.

**Hour 2 — three baselines, cheapest first.** Majority-class predictor (this is your floor, and stating it stops anyone from being impressed by 68% accuracy on a dataset that is 64% one class). TF-IDF + logistic regression, maybe 15 lines. Embeddings + logistic regression. All three in one table with the same CI treatment.

**Hour 3 — the LLM.** Zero-shot with a clean taxonomy in the prompt, then few-shot with retrieved nearest-neighbour examples, then structured output so parsing is not a failure mode. Same eval, same table.

**Hour 4 — the system, and the writeup.** A cascade: classifier when confident, LLM otherwise, with the threshold chosen on the validation split to hit a stated accuracy target, and the resulting escalation rate turned into a cost-per-1M-tickets number with the arithmetic in the README. Plus an error analysis: embed the misclassifications, cluster them, and describe the top three failure modes by name. Plus a short "what I'd do with another week" section that names the highest-value next step rather than listing everything.

**🗣 Say this in the room, at the defense:** "The interesting result was that TF-IDF plus logistic regression got macro-F1 0.78 against the frontier model's 0.83, with a bootstrap CI of roughly ±0.03 on each — so the model is genuinely better, but by about one and a half sigma, at 400× the cost per call. That's why I shipped the cascade: the classifier takes the 84% where its max-probability exceeds 0.7, and the estimated blended cost is $X per million tickets against $Y for routing everything."

**⚠ Trap:** do not present the baseline as a strawman you beat. Present it as the thing you had to beat, with the honest margin, including the case where it *wins*. Reviewers of these assignments have told me repeatedly that the single strongest signal is a candidate who reports a result that cuts against their own design and then explains why they shipped it anyway. That is what judgment looks like on paper.
