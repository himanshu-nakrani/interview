### What does it mean for a model to be calibrated, and why should I care as an applied engineer rather than a researcher?

Mental model: **calibration is the property that lets you build a control flow on top of a probabilistic component.** A model is calibrated if, among all the cases where it says 80%, it is right 80% of the time. That is not a statement about accuracy — a coin-flip predictor that always says "50%" is perfectly calibrated and useless. It is a statement about whether the *number* the model attaches to its answer means anything, and everything you want to build on top of an LLM — cascades, abstention, human escalation, selective automation — is a threshold on that number.

Formally: for a predictor with confidence c, calibration requires P(correct | confidence = c) = c for all c. Two orthogonal properties matter and get conflated: **discrimination** (does the confidence rank correct answers above incorrect ones — measured by AUC) and **calibration** (are the numbers on the right scale — measured by ECE). You need discrimination for a threshold to be *useful* and calibration for the threshold to be *interpretable and stable*. A model with great discrimination but bad calibration is fixable with a monotone rescaling; a model with bad discrimination is not fixable at all.

**💰 Math on why this is money, not theory.** Take a support-automation product at 100k requests/day. A small model resolves 82% correctly at $0.0004/request; a frontier model resolves 94% at $0.006/request. All-frontier costs 100,000 × $0.006 = $600/day = **$18,000/month** at 6.0% error. Now add a calibrated confidence score and escalate the least-confident 25% of requests. If that bottom quartile contains 70% of the small model's errors (which is what a decently-discriminating score buys you), residual errors are 18,000 × 0.30 = 5,400 unescalated, plus 12,600 escalated × 6% the big model also misses ≈ 756, for 6,156 errors/day = **6.2% error**. Cost is 100,000 × $0.0004 + 25,000 × $0.006 = $40 + $150 = $190/day = **$5,700/month**. You bought a 68% cost reduction for 0.2 points of accuracy — and the *entire* saving is created by the confidence score's ability to rank. If the score were uncorrelated with correctness, escalating 25% would capture 25% of errors and the whole design collapses.

**🗣 Say this in the room:** "Calibration is what turns a model output into something I can route on. I care about two separate things — whether the confidence ranks correct above incorrect, which is AUC and which no post-hoc fix can create, and whether the numbers are on the right scale, which is ECE and which temperature scaling can usually fix. The cascade design I just described is worth about two-thirds of the inference bill, and all of that value comes from the ranking."

### Define ECE and implement it. Then tell me its known pathologies, because interviewers ask that second part.

Expected Calibration Error bins predictions by confidence and measures the average gap between confidence and accuracy within each bin, weighted by bin population:

  **ECE = Σ_b (n_b / n) · | acc(b) − conf(b) |**

```python
import numpy as np

def ece(conf, correct, n_bins=15):
    """conf: predicted probability of being right. correct: 0/1 outcome."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return total
```

Now the pathologies, and these are the follow-up:

**It is binning-dependent.** With 10 equal-width bins you get one number; with 50 you get a different, larger one. And equal-*width* bins are a poor choice for LLM confidences, which pile up near 0.9–1.0 — nine of your ten bins end up nearly empty and the whole metric is decided by one bin. Equal-*mass* (quantile) bins are the better default, and you should say which you used whenever you report an ECE.

**It is a biased estimator, and the bias grows with bins.** Within-bin sampling noise never cancels because of the absolute value, so more bins on a fixed sample size inflates ECE. Comparing ECEs computed with different bin counts or different n is meaningless.

**ECE = 0 does not mean the model is useful.** A predictor that outputs the base rate 0.7 on every single example, on a dataset where 70% of answers are correct, has exactly zero ECE and zero discriminative power. **Never report ECE without also reporting AUC or a selective-accuracy curve**; the pair is informative, either alone is not.

**It's a marginal metric.** ECE aggregates over the whole dataset, so a model that is overconfident on one tenant and underconfident on another can show a beautiful global ECE. Always compute it per meaningful slice — per language, per document type, per tenant — because the failures you care about are slice failures.

**It ignores ordering within a bin.** Two very different predictors can share an ECE.

**📄 Paper:** Guo et al. (2017), *On Calibration of Modern Neural Networks* — showed that modern deep networks are substantially more miscalibrated (overconfident) than the shallower networks that preceded them, and that a single-parameter temperature fit on a validation set removes most of the error. It replaced Platt scaling and isotonic regression as the default for neural classifiers by being simpler and, empirically, at least as good.

**⚠ Trap:** reporting a single ECE number as a model-quality claim. In review I ask for the reliability diagram, the bin counts, the bin scheme, and the AUC alongside it. A candidate who volunteers those unprompted has clearly done this on real data.

### How do you read a reliability diagram, and how would you fix a miscalibrated model at serving time?

A reliability diagram plots mean confidence per bin (x) against observed accuracy per bin (y), with the diagonal as perfect calibration. Read it by which side of the diagonal the curve sits on. **Below the diagonal = overconfident** (says 90%, right 70% of the time) — the near-universal condition of modern neural networks and of RLHF'd LLMs in particular. **Above the diagonal = underconfident**, which is rarer and usually the result of over-aggressive label smoothing or an over-corrected temperature. A curve that hugs the diagonal at low confidence and falls away at high confidence — the classic S-shape — means the model is fine when unsure and badly overconfident when sure, which is the worst possible shape for an escalation policy, because it means the cases you auto-approve are exactly the ones where the confidence lies. Always plot bin *counts* underneath as a histogram; a dramatic-looking deviation in a bin holding 11 examples is noise.

Fixes, in the order I'd try them:

**Temperature scaling.** Fit a single scalar T on a held-out calibration set by minimizing NLL of softmax(z/T). One parameter, so it needs only a few hundred labeled examples, and because dividing by a positive constant is monotone, **it cannot change your accuracy or your AUC — it only moves the numbers onto the right scale.** That property is why it's the default: it is a strictly-safe post-hoc fix.

```python
import torch
def fit_temperature(logits, labels, iters=200):
    logT = torch.zeros(1, requires_grad=True)          # optimize in log-space, T > 0
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / logT.exp(), labels)
        loss.backward(); return loss
    opt.step(closure)
    return logT.exp().item()
```

**Platt scaling** fits a logistic (two parameters, a and b, on the logit) — slightly more flexible, still monotone. **Isotonic regression** is nonparametric and can fix non-monotone miscalibration, but it needs thousands of examples, overfits happily, and produces a step function that behaves badly at thresholds you didn't sample. My rule: temperature first, isotonic only with >5k calibration examples and a held-out check.

**The API case, which is what you'll actually face.** With a hosted model you often can't touch logits — and even where logprobs are exposed, the token-level logprob of an answer is not the same thing as P(the answer is correct). So you calibrate a *downstream* score: fit a small logistic regression that maps features → P(correct), where the features are things like the mean logprob of the answer tokens, the self-consistency agreement rate over k samples, the retrieval score of the top passage, the answer length, and whether the model hedged. Train it on a few hundred human-labeled outcomes. **This little logistic model is one of the highest-ROI components you can add to a production LLM system, and almost nobody builds it.** It costs an afternoon and it's what makes the cascade arithmetic above real.

**⚠ Trap:** fitting the temperature (or the logistic) on the same set you evaluate calibration on. You will report a beautiful ECE that does not survive contact with production. Split it: fit on calibration, report on test, and re-fit on a schedule, because calibration drifts as the input distribution drifts.

### Does asking the model "how confident are you, 0–100?" actually work? And what happened to calibration when reasoning models arrived?

Verbalized confidence works better than people expect and worse than you'd want, and the honest answer names both.

What's real about it: it has genuine discriminative signal. Sort your outputs by verbalized confidence and accuracy does go up monotonically — the AUC is well above 0.5, which means it is *usable* as a routing feature. What's broken: the numbers are on the wrong scale (systematically overconfident, typically saying 90% when it's right 70% of the time), and they cluster hard on round numbers — 80, 90, 95, 99 — so you effectively have four or five confidence levels rather than a continuum, which makes fine-grained thresholding impossible. So: use it as a *feature* in a calibrated downstream model, never as a probability you threshold directly.

Alternatives that usually beat it: **self-consistency agreement** (sample k = 5 answers at T = 0.7 and use the plurality fraction as the confidence) is typically the strongest cheap signal, at k× the cost. **Sequence logprob** of the answer span, where available, is decent and free. **Pairwise self-verification** ("here is your answer, find the error") adds signal but also adds a second failure mode.

The post-training effect is the part with a clean documented result. OpenAI's GPT-4 technical report showed the *pre-trained* model was well calibrated on a multiple-choice benchmark and that the post-RLHF model was substantially less so. The mechanism is exactly the reverse-KL / mode-seeking story from earlier: RLHF pushes the policy to commit to a mode, and confidence is a casualty of commitment. **This is a general property of preference-optimized models, not a quirk of one lab.**

**📅 Volatile:** there is a more recent and still-settling line of evidence that extended reasoning makes this *worse* — that models which generate long chains before answering become more confident as the chain lengthens, roughly independently of whether the chain is correct, so "think longer" trades calibration for accuracy. The mechanism is plausible and mechanistically coherent (each self-generated reasoning step conditions the next, so the model accumulates evidence from its own tokens), and it matches what I've seen measuring reasoning models. But verify the current state of this before quoting it as settled — this is one of the faster-moving corners of the literature.

**🗣 Say this in the room:** "Verbalized confidence has real ranking signal but bad scale — it clusters on round numbers and it's systematically overconfident, and RLHF made that worse; the GPT-4 report showed the base model was calibrated and the post-RLHF model wasn't. So I'd never threshold it directly. I'd use it as one feature in a small logistic model alongside self-consistency agreement and retrieval score, fit on a few hundred labeled outcomes, and threshold *that*."

### Your eval says variant B beats variant A, 78% versus 76%, on 500 examples. Do you ship it?

No, and I can show you why in about twenty seconds of arithmetic — which is the point of the question.

**Step 1: the confidence interval on a single number.** For a proportion, SE = √(p(1−p)/n) = √(0.78 × 0.22 / 500) = √0.000343 = 0.0185, so 1.85 percentage points. The 95% interval on B alone is 78% ± 1.96 × 1.85 = **[74.4%, 81.6%]**. A ±3.6-point interval on a 2-point claimed improvement should end the conversation on its own.

**Step 2: the interval on the difference.** For two independent samples, SE_diff = √(p_A(1−p_A)/n + p_B(1−p_B)/n) = √(0.000343 + 0.000365) = √0.000708 = 0.0266, so 2.66 points. The observed difference is 2.0 points — **less than one standard error.** The 95% CI on the difference is 2.0 ± 5.2 = [−3.2, +7.2]. It comfortably contains zero, and it also contains "B is three points *worse*."

So the honest statement is: *this experiment cannot distinguish B from A, and it also cannot rule out B being meaningfully worse.*

**Step 3 — and this is where you gain ground on other candidates: don't stop at "not significant," say what you'd do instead.**

- **Pair the comparison.** If both variants ran on the *same* 500 examples — which they did, this is an eval set, not an A/B test on users — then treating them as independent samples throws away most of your power. The right analysis looks only at the examples where they disagree. More on this below; it typically cuts the required sample size by 5–10×.
- **Look at the disagreements by hand.** 500 examples, maybe 40 disagreements. Read them. Twenty minutes of reading disagreements teaches you more than the p-value does, and it's where you discover that B's "wins" are all on one template and its "losses" are all on the long-tail language.
- **Check whether the delta is even worth the cost.** If B is the same prompt plus a 4k-token few-shot block, you are paying real money for an effect you cannot measure.
- **Slice it.** A 2-point global delta that is +9 on one segment and −4 on another is a genuinely different finding from a uniform +2, and the segmented view often *is* shippable — for that segment.

**⚠ Trap:** the eval-set-of-100 that pervades take-homes and internal dashboards. At n = 100, the 95% CI half-width on a proportion near 0.8 is 1.96 × √(0.8 × 0.2/100) = 7.8 points. **Any improvement smaller than 8 points is invisible at n = 100.** Teams ship prompt changes on 3-point deltas from 100-example evals every day, and it is indistinguishable from shipping noise. Say this number out loud in an interview; it lands.

### Then compute it for me. How many examples do I need to detect a 2-point improvement?

**📐 Numbers you must know — the unpaired case.** For two proportions with α = 0.05 two-sided and 80% power, the per-arm sample size is

  n ≈ 2 · (z_{α/2} + z_β)² · p̄(1 − p̄) / δ²

with z_{0.025} = 1.96, z_{0.20} = 0.84, so (1.96 + 0.84)² = 7.84. Take p̄ = 0.77 and δ = 0.02:

  n = 2 × 7.84 × (0.77 × 0.23) / 0.02² = 2 × 7.84 × 0.1771 / 0.0004 = 2.777 / 0.0004 = **6,943 per arm**, i.e. **~13,900 examples total.**

That is the number people are shocked by, and it should reset how you think about eval sets. Detecting small deltas requires large n, full stop. Some anchors from the same formula, so you can do it in your head: **the sample size scales as 1/δ².** Halving the effect you want to detect quadruples the n. To detect 5 points you need 2 × 7.84 × 0.1771/0.0025 = 1,111 per arm. To detect 10 points, 278 per arm. **To detect 1 point, 27,800 per arm.**

**Now the paired case, which is what you actually have.** Both variants run on identical examples, so most of the variance is *item difficulty*, which is shared and therefore cancels. Only the discordant items — where the two variants disagree — carry information. Suppose 10% of items are discordant and the true effect is that 6% of items flip A-wrong-to-B-right while 4% flip the other way (net +2 points). Conditional on being discordant, B wins 60% of the time, so we're testing p = 0.6 against p = 0.5 on the discordant subset:

  n_discordant = [z_{α/2}√(0.5 × 0.5) + z_β√(0.6 × 0.4)]² / (0.1)²
  = [1.96 × 0.5 + 0.84 × 0.4899]² / 0.01 = [0.98 + 0.4115]² / 0.01 = 1.936 / 0.01 = **194 discordant pairs.**

At a 10% discordance rate, that's **~1,940 total examples** — a **7× reduction** versus the unpaired 13,900, from nothing but analyzing the data correctly.

**🗣 Say this in the room:** "To detect a 2-point delta at 80% power you need roughly 7,000 per arm unpaired. But an eval set is a paired design — same items, both variants — so I'd analyze only the discordant items, and at a typical 10% disagreement rate that drops the requirement to about 2,000 examples total. The single highest-leverage thing most teams could do to their eval methodology is stop running an unpaired analysis on paired data."

**⚠ Trap:** treating "we need 2,000 examples" as a reason not to measure. The alternative is not "measure with 100," it is "measure a 10-point effect with 300, and refuse to claim 2-point effects at all." Size your eval to the effect size you actually intend to act on, and say out loud which effects you are choosing to be blind to.

### Why paired bootstrap rather than a t-test on the eval scores? Implement it.

Three reasons, and the third is the one that matters most in this field.

**One: eval metrics are usually not means of independent identically-distributed scalars.** nDCG@10, pass@k, F1, and any LLM-judge rubric score are not sample means with a clean sampling distribution, so the t-test's assumptions don't apply. The bootstrap doesn't care — it resamples the empirical distribution and reads the answer off the resampled statistic, whatever the statistic is.

**Two: pairing removes item-difficulty variance.** The dominant source of variance in an eval is that some questions are hard for everybody. If you bootstrap the two systems' scores independently, that variance stays in your estimate. If you bootstrap the *per-item differences*, it cancels exactly. In practice this shrinks confidence intervals by a factor of two to four on a typical eval, which is free statistical power.

**Three: it composes with any metric your system actually reports.** You can bootstrap a corpus-level metric like BLEU or a rate like "% of responses passing schema validation" the same way, which a t-test cannot do.

```python
import numpy as np

def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """a, b: per-item scores for the two systems, same items, same order."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = d.mean()
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])            # 95% CI on the difference
    centered = boot - obs                                # simulate the null
    p = float(np.mean(np.abs(centered) >= abs(obs)))     # two-sided bootstrap p-value
    return obs, (lo, hi), p
```

Two implementation notes that separate a correct implementation from a plausible one. **Resample items, not scores** — the unit of resampling must be the unit of independence, so for a multi-turn eval you resample *conversations*, not turns, or your intervals will be far too narrow. And **centre the bootstrap distribution before computing a p-value**; the percentile interval is a CI, but the null hypothesis is "mean difference is zero," which you simulate by subtracting the observed mean. Skipping that step gives you a p-value that is wrong in a direction you won't notice.

**📄 Paper:** Koehn (2004), *Statistical Significance Tests for Machine Translation Evaluation* — established bootstrap resampling as the standard significance procedure for corpus-level NLP metrics, replacing the practice of reporting bare score differences. The methodology transfers directly to LLM evals and almost nobody in the LLM world cites it.

**🏋 Drill:** 15 minutes, no references. Write `paired_bootstrap` from memory. Then generate synthetic data where system B is truly 2 points better on 2,000 items, run it, and confirm the CI excludes zero. Then rerun at 300 items and confirm it doesn't. Pass criterion: both behaviours reproduce and you can state, without looking, why the p-value needs the centering step.

### When is McNemar's test the right tool, and how do you compute it?

McNemar's is the exact right tool for **the most common comparison in this entire field**: two systems, same evaluation items, binary correct/incorrect outcome. That is what nearly every eval is. It is the closed-form version of the pairing argument, and it takes ten seconds to compute by hand, which makes it a good thing to reach for live in an interview.

Build the 2×2 contingency table over items:

|  | B correct | B wrong |
|---|---|---|
| **A correct** | a | b |
| **A wrong** | c | d |

The cells a and d — where both agree — carry *no information about which is better*, and that is McNemar's insight: condition on the discordant pairs only. Under the null hypothesis that the systems are equally good, each discordant item is a fair coin flip between b and c. So:

  **χ² = (|b − c| − 1)² / (b + c)**, with 1 degree of freedom (the −1 is the continuity correction).

When b + c < 25, skip the approximation and run the exact binomial test of b successes out of b + c trials against p = 0.5.

**Worked example on the 500-item case from earlier.** Suppose A scored 76%, B scored 78%, and the table is a = 370, b = 25 (A right, B wrong), c = 35 (A wrong, B right), d = 70. The net is (35 − 25)/500 = +2 points, matching. Now:

  χ² = (|35 − 25| − 1)² / (35 + 25) = 9² / 60 = 81/60 = **1.35**, p ≈ **0.245**.

Not close to significant, consistent with the CI analysis. Note how much more informative the table is than the two headline numbers: you can immediately see that only 60 of 500 items disagreed, which tells you these two variants are 88% identical and that whatever B changed, it changed narrowly. That framing — "the systems agree on 88% of items; the disagreement is 35 vs 25" — is a far better thing to put in front of a PM than "78 vs 76."

**⚠ Trap:** running McNemar on unpaired data, or on items scored by a stochastic judge without accounting for judge noise. If your "correct" label comes from an LLM judge sampled at temperature > 0, some of your b and c cells are judge flips, not system differences. Score the judge at T = 0, or better, measure judge self-agreement on a rescored subset first and treat that as your noise floor — if judge self-disagreement is 8% and your systems differ by 2%, the test is measuring the judge.

### You swept 30 prompt variants and the best one beat baseline by 3 points with p = 0.03. What do you tell the PM?

That we have found approximately nothing, and here is the arithmetic.

**💰 Math on the family-wise error rate.** If all 30 variants were truly identical to baseline, each test has a 5% chance of a false positive at α = 0.05. The probability that *at least one* of 30 independent tests fires is 1 − 0.95³⁰. Compute it: ln(0.95) = −0.05129, × 30 = −1.5388, e^{−1.5388} = 0.2146. So **1 − 0.215 = 78.5% chance of at least one "significant" result from pure noise.** A p = 0.03 in a family of 30 is not evidence; it is the expected outcome of the procedure.

The corrections, and when I use each:

- **Bonferroni:** require p < α/m = 0.05/30 = **0.00167**. Your p = 0.03 fails by a factor of 18. Bonferroni controls the family-wise error rate, is very conservative, and is the right choice when a single false positive is expensive (a safety filter, a pricing change).
- **Benjamini–Hochberg:** sort the m p-values ascending and find the largest k with p_(k) ≤ (k/m)·α; reject everything up to k. This controls the *false discovery rate* — the expected proportion of your rejections that are false — rather than the probability of any false rejection. For an exploratory prompt sweep where you'll validate the survivors anyway, BH is the correct and much less brutal choice. **📄 Paper:** Benjamini & Hochberg (1995) introduced FDR control, replacing family-wise-error methods in settings with many tests where a few false discoveries are tolerable.

But the deeper problem isn't the p-value at all — it's **the winner's curse.** When you select the maximum of 30 noisy estimates, that maximum is biased upward by construction, because a variant is more likely to be selected if its noise happened to be positive. Your 3-point winner has a true effect that is systematically smaller than 3 points, often dramatically so, and applying a multiplicity correction to the p-value does not fix the *effect size* bias at all.

The fix is the one every experienced person converges on: **hold out a confirmation set.** Sweep on set A, take the top 2–3 candidates, then evaluate *only those* on a fresh set B that was never touched during the sweep. The estimate on B is unbiased, and with only 2–3 tests the multiplicity problem nearly vanishes. This is the same discipline as a train/val/test split, and prompt sweeps are exactly a hyperparameter search — treat them that way.

**🗣 Say this in the room:** "With 30 variants at α = 0.05 there's a 78% chance of at least one false positive under the null, so p = 0.03 is what noise looks like. Under Bonferroni I'd need p below 0.0017. But the bigger issue is winner's curse — the max of 30 noisy estimates is biased upward, and correcting the p-value doesn't correct the effect size. I'd take the top three to a held-out confirmation set and report *that* number to the PM, and I'd expect it to come in well under 3 points."

**⚠ Trap:** the invisible version of this, which is a team that runs prompt changes serially over six weeks with no correction at all — twenty implicit comparisons, each shipped on a 2-point delta. That's the same multiple-comparisons problem spread over a quarter, and the aggregate result is a prompt that has been optimized to your eval set's noise. The symptom is an eval score that climbs steadily while user-facing metrics don't move.

### Your eval scores move ±1.5 points between runs even at temperature 0. Walk me through debugging that.

This is a real and common incident, and the value is in having an ordered procedure rather than a list of causes.

**🔍 Failure taxonomy — nondeterministic eval scores, in the order I'd check:**

**1. Establish the noise floor before debugging anything.** Run the identical config three times, unchanged. If it varies ±1.5 points, the variance is in your harness, not in your change. This is step zero and it is the step people skip; they debug a "regression" that is inside their own noise band.

**2. Is the judge stochastic?** If a model grades the outputs, is it running at T = 0 with a pinned model version? An LLM judge at T = 0.7 will re-score the same output differently. Measure judge self-agreement by scoring the same 200 outputs twice and computing the flip rate. If it's 5%, on a 500-item eval that's ±25 items ≈ ±5 points of pure judge noise, which swamps everything else. Fix the judge before you look anywhere else.

**3. Is generation batch-dependent?** As covered earlier, greedy decoding is not deterministic under variable batch composition — floating-point reduction order changes with batch shape. Signature: outputs are *mostly* identical with a handful of complete divergences, and the divergence point is a token where the top-2 logits were nearly tied. Test by running the eval single-threaded at batch size 1 and seeing whether the variance collapses.

**4. Is retrieval nondeterministic?** ANN indexes (HNSW especially) can return different neighbours across builds and across concurrency levels, and ties in the score are broken by insertion order. If your retrieval top-5 differs by one document, downstream generation differs. Pin the index, log the retrieved doc IDs per item, and diff them across runs — this makes the cause visible in one command.

**5. Is the eval set itself changing?** Unpinned dataset version, a database query without an ORDER BY, sampling `n=500` from a larger pool with a fresh seed each run, or dropped items from timeouts. **Timeouts are a sneaky one**: if 8 items time out in run one and 3 in run two, and timed-out items are scored as failures, you've moved a point without anything model-related changing.

**6. Concurrency-dependent truncation.** Under load, more requests hit `max_tokens` or a provider-side truncation, and truncated answers fail. Log finish reasons and assert the distribution is stable across runs.

**The fix set, once you've localized it:** pin every version (model, dataset, index, prompt template) and log the pins in the results artifact; run the judge at T = 0; log per-item outputs so you can diff runs rather than diff aggregates; report **paired** comparisons so shared noise cancels; and publish the harness's own noise floor next to every result. **The rule I enforce in review: no eval result is reported without its repeat-run variance.** A team that doesn't know its noise floor cannot interpret any of its own numbers.

**💰 Math on why this is urgent:** if your noise floor is ±1.5 points and your team ships prompt changes on 2-point improvements, then by the arithmetic above roughly half of your "improvements" are noise. At an engineer-week per prompt iteration and 40 iterations a year, that is 20 engineer-weeks — call it $80–120k of loaded cost — spent producing a random walk. Fixing the harness costs about a week.

### Timed drill — 15 minutes, blank page, no references. Can you produce the three derivations from memory?

**🏋 Drill.** Set a timer for 15 minutes. No notes, no autocomplete, no calculator beyond arithmetic you do by hand. Produce all three:

**(a) The attention scale.** State the assumptions on q and k, derive Var(q·k) = d_k, state the resulting standard deviation for d_k = 128, and explain in one sentence what happens to the softmax and to the gradient without the 1/√d_k. *Pass criterion: you write "variances add over d_k independent unit-variance products, so std = √d_k ≈ 11.3" without hesitating, and you name the softmax Jacobian diag(s) − ssᵀ as the reason the gradient dies.*

**(b) The KV cache formula, applied.** Write bytes/token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes. Then compute it for L = 80, n_kv_heads = 8, head_dim = 128, bf16, and give the total for a 128k-token sequence in GiB. *Pass criterion: 320 KiB/token and 40 GiB, arrived at in under two minutes, and you can immediately say what changes if the model were MHA (×8 → 320 GiB) or if the cache were fp8 (÷2 → 20 GiB).*

**(c) Training compute.** Derive 2N forward and 4N backward from the per-linear-layer matmul counts, state C = 6ND, and compute the H100-days for N = 7e9, D = 1e12 at 40% MFU. *Pass criterion: 4.2e22 FLOPs, ~1,200 H100-days, ~$60k at $2/GPU-hour — and you name at least two things the estimate omits (failed runs, data prep, eval compute, checkpoint storage).*

Grade yourself hard. **A hesitation is a fail**, because in the room the hesitation is what the interviewer records. If you fail any part, the remedy is not to reread this section — it is to write the derivation out longhand three times on separate days. This material is procedural memory, not declarative; you cannot acquire it by recognition.

Extend the drill once you pass it: have someone interrupt you mid-derivation with "why?" at a random step. The real interview stops you; rehearsing an uninterrupted monologue does not prepare you for that.

### Timed drill — 10 minutes. Here's a product spec; give me a cost and latency model on the whiteboard.

**🏋 Drill.** Ten minutes, out loud, on a whiteboard or a shared doc, no calculator. The spec: *an internal document assistant for a 6,000-person company. 40,000 queries per weekday. Each query retrieves 8 chunks of 600 tokens, prepends a 1,500-token system prompt, and produces roughly 400 output tokens. Assume a frontier-tier model at $3 per million input tokens and $15 per million output tokens.* Produce: cost per query, monthly cost, the single highest-leverage optimization with its savings, and a p95 latency estimate with its components.

Here is the shape of a passing answer, so you can grade yourself.

**Tokens per query.** Input = 1,500 (system) + 8 × 600 (chunks) + ~100 (user query) = 1,500 + 4,800 + 100 = **6,400 input tokens**. Output = **400**.

**Cost per query.** Input: 6,400 / 1e6 × $3 = $0.0192. Output: 400 / 1e6 × $15 = $0.0060. **Total $0.0252 per query.**

**Monthly.** 40,000/day × 22 working days = 880,000 queries. 880,000 × $0.0252 = **$22,176/month**. Say ~$22k.

**Highest-leverage optimization.** Input dominates at 76% of cost, and the 1,500-token system prompt is byte-identical on every call — that is what prefix caching is for. At a 90% cache discount on cached input, those 1,500 tokens go from 1,500/1e6 × $3 = $0.0045 to $0.00045, saving $0.004 per query = **$3,560/month**. Better: the chunks are the bigger prize at 4,800 tokens, but they vary per query, so caching doesn't apply — instead, rerank 8 chunks down to 4 and you cut 2,400 input tokens = $0.0072/query = **$6,336/month**, provided an eval shows retrieval quality holds at k = 4. Combined, roughly **$10k/month off a $22k bill**, with the reranking gated on an eval. **📅 Volatile:** cache-discount percentages and per-token prices differ by provider and change often — verify before quoting.

**p95 latency.** Components: retrieval (embed the query ~20 ms + ANN search ~15 ms + fetch chunks ~15 ms ≈ 50 ms), then prefill of 6,400 tokens, then 400 tokens of decode. Prefill at a compute-bound rate of order 10⁴ tokens/s for a frontier-scale model on a served endpoint ≈ 600 ms, so TTFT lands around 650–700 ms. Decode at ~60 tokens/s inter-token rate → 400/60 ≈ **6.7 seconds to completion**. So: **TTFT ~0.7 s, full response ~7.4 s.** Which immediately tells you the product must stream — the perceived latency is TTFT, and a 7-second blocking spinner is a different product than a 0.7-second first token.

**Pass criterion:** you produce cost/query, monthly cost, a named optimization with its arithmetic, and a latency decomposition that separates TTFT from total — inside ten minutes, out loud, without a calculator. The single most common failure is doing the token arithmetic correctly and then forgetting to separate TTFT from end-to-end latency, which is the number the product actually lives or dies on.
