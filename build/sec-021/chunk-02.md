### IPO — what specific failure of DPO does it fix? Write me its loss and explain the fix mechanically.

Start with the failure, because IPO is meaningless without it.

DPO inherits Bradley-Terry, and Bradley-Terry says preferences are probabilistic: `P(y_w ≻ y_l) = σ(Δr)`. But your dataset is *deterministic* — every row says "chosen beat rejected," probability 1. Fitting a σ to a target of exactly 1.0 requires `Δr → +∞`. And `Δr = β·[(log π_w − log π_ref,w) − (log π_l − log π_ref,l)]`, so driving it to infinity means driving `π_θ(y_l|x) → 0`. The KL regularization, which was supposed to prevent exactly this, has no fixed point that stops it: the sigmoid loss keeps rewarding more margin forever. With infinite data the empirical preference frequencies would be soft and this would resolve itself; with the two-or-three samples per prompt you actually have, it does not.

The consequence is that **DPO overfits to deterministic preferences by collapsing the rejected completions to zero probability**, and because rejected completions share most of their token distribution with good completions, this drags real capability down with it.

IPO's fix: throw away the sigmoid and regress the margin to a *finite target*. The loss is

```
L_IPO = E[ ( h_θ(x, y_w, y_l) − 1/(2τ) )² ]

h_θ = log( π_θ(y_w|x)/π_θ(y_l|x) ) − log( π_ref(y_w|x)/π_ref(y_l|x) )
```

which is exactly the same `logits` quantity you computed for DPO. In code, given the four log-probs, it is a two-line change:

```python
logits = (pol_c - pol_r) - (ref_c - ref_r)      # identical to DPO
loss_dpo = -F.logsigmoid(beta * logits)
loss_ipo = (logits - 1.0 / (2.0 * beta)) ** 2   # beta plays the role of tau
```

Now the optimum is a *specific finite margin* of `1/(2τ)`. Push past it and the loss goes back up. The KL constraint becomes real again because the objective has an interior optimum instead of one at infinity. Note the sign of the knob flips: in DPO, small β = weak regularization; in IPO, small τ pushes the target margin `1/(2τ)` *higher*, so small τ = more aggressive. TRL reuses the `beta` field for τ, which is a genuine footgun in configs.

**📄 Paper:** Azar et al. (2023), *A General Theoretical Paradigm to Understand Learning from Human Preferences* — introduces ΨPO as a family that contains RLHF and DPO as special cases, and IPO as the identity-mapping member that avoids the BT assumption entirely.

**⚠ Trap:** describing IPO as "DPO with a different loss shape." The substantive claim is that IPO does not assume a Bradley-Terry latent reward at all — it works directly with preference *probabilities*, which is why it degrades gracefully when preferences are non-transitive or when a pair is a coin flip. The squared loss is a consequence, not the idea.

### So when would you actually pick IPO over DPO in production?

I will be honest that this is contested and that the empirical record is mixed — several large open-weight comparison sweeps have found IPO roughly on par with, or slightly behind, well-tuned DPO on general chat benchmarks. So my decision rule is not "IPO is better," it is "IPO is better under these three conditions."

**Condition 1 — small preference datasets relative to prompt diversity.** IPO's whole advantage is not extrapolating a deterministic label to infinity. With 200k diverse pairs, DPO's overfitting is diluted by data. With 8k pairs from a domain-expert labeling campaign — which is the realistic case at an enterprise applied-AI team — DPO will happily memorize the margin on those 8k and IPO will not.

**Condition 2 — you observe the specific symptom.** Run DPO first, since it is the default everyone understands. If `logps/rejected` is in free-fall, `rewards/margins` blows past 10, and your capability suite regresses while win rate rises, that is the deterministic-preference collapse and IPO is the targeted fix. Do not reach for it prophylactically.

**Condition 3 — many duplicate or near-duplicate prompts with inconsistent labels.** If the same prompt appears with contradictory preferences (which happens the moment you have more than one annotator), DPO's infinite-margin pressure fights itself and produces instability. IPO's finite target handles it as a soft preference, which is what it is.

**🗣 Say this in the room:** "My default is DPO with β = 0.1 because it is the best-understood and every tool supports it. I switch to IPO when I see the deterministic-preference signature — margins running away, rejected log-probs collapsing, and a capability regression that tracks the margin rather than the win rate — or when the dataset is small enough that overfitting a hard label is the dominant risk. I would not claim IPO wins on average; I would claim it wins on that failure mode."

### Explain KTO. Why does it only need binary labels, and what did it give up to get there?

KTO's premise is that requiring *pairs* is a data-collection artifact, not a modelling necessity. What you actually need is a signal about whether a completion is above or below some reference level of goodness — and a thumbs-up in your product already gives you that.

The mental model comes from prospect theory: humans evaluate outcomes relative to a reference point and weight losses more heavily than gains. KTO builds a loss with that shape directly. For each example you have `(x, y, label ∈ {desirable, undesirable})`. Define the same implicit reward as DPO:

```
r̂(x,y) = β [ log π_θ(y|x) − log π_ref(y|x) ]
```

Define a **reference point** `z₀` = an estimate of `KL(π_θ ‖ π_ref)` for this prompt distribution. The trick is that you cannot compute this exactly, so KTO estimates it in-batch by pairing each prompt with a *mismatched* completion from elsewhere in the batch and averaging the log-ratio, clamped at zero — and critically, **you do not backpropagate through `z₀`**. It acts as a slowly-moving baseline, not a trainable quantity.

Then the value of an example is

```
desirable:    v = λ_D · σ(  β (r̂ − z₀) )
undesirable:  v = λ_U · σ( −β (r̂ − z₀) )   [i.e. σ(β(z₀ − r̂))]
L_KTO = E[ λ_y − v(x, y) ]
```

So a desirable example is pushed above the reference point and an undesirable one below it, with separate weights `λ_D`, `λ_U` that let you handle class imbalance. The paper's practical guidance is to set the weights so that `λ_D·n_D / (λ_U·n_U)` sits in roughly the 1 to 4/3 range — i.e. slightly favor the desirable class.

What you gave up: **pairwise data carries more information per label than binary data does.** A pair says "for this exact prompt, A is better than B," which controls for prompt difficulty automatically. A binary label says "this is good," and now the model has to disentangle "good" from "this prompt was easy." That is why KTO needs more raw signals for the same effect, and why the `z₀` baseline exists at all — it is doing the job that the shared prompt did in a pair.

**📄 Paper:** Ethayarajh et al. (2024), *KTO: Model Alignment as Prospect Theoretic Optimization* — replaces pairwise preference with per-example binary desirability, and reports that KTO matches or exceeds DPO at 7B–13B scale even when the pairwise data is discarded down to binary.

**⚠ Trap:** feeding KTO a binarized version of your pairwise data and expecting it to be strictly worse. It often is not — and the reason is instructive. Binarization lets you keep examples where you have a good response but no bad counterpart, and vice versa, which in most real datasets is the majority of your logged signal. More usable data at lower information density frequently beats less data at higher density.

### Do the data-collection economics for me. When does KTO change what a product team can actually do?

This is where KTO stops being a paper and becomes a strategy, and it is exactly the argument I would make in a product-company interview.

**Pairwise labeling cost.** An annotator has to read a prompt and two full responses and make a judgment. For a 300-token pair of responses in a technical domain, budget 90–120 seconds per comparison including context switching. At 90 s, that is 40 comparisons/hour. At a loaded rate of $25/hour for a general annotator (much more for a lawyer, a doctor, or a senior engineer — Harvey-style domains run $80–150/hour), **60,000 pairs = 1,500 hours = $37,500** at the cheap rate, and roughly $150,000 at the expert rate. Turnaround: weeks, plus a pilot round, plus adjudication.

**Binary signal cost.** A thumbs-up/down button in your product. The annotator is the user, the latency is zero, the marginal cost is zero, and the volume is whatever your traffic is. Take a realistic mid-size AI product: 2,000,000 assistant messages/day with a 3% explicit-feedback rate → **60,000 labeled signals per day, at $0 marginal cost.** You get in one day, for free, what the pairwise campaign delivers in three weeks for $37.5k.

**💰 Math, stated as the decision:** the pairwise route costs $37,500 and ~3 weeks for 60k pairs, refreshed quarterly → ~$150k/year plus a labeling-ops owner. The binary route costs one frontend change plus a logging pipeline — call it two engineer-weeks, ~$15k fully loaded, once — and then produces 60k signals/day forever. The break-even is immediate. The reason you would *still* pay for pairwise data is that production feedback is horrifically biased, which is the real cost and I would raise it unprompted.

**🔍 Failure taxonomy — what production thumbs data gets wrong.** (1) *Selection bias*: users click thumbs-down when angry and almost never click thumbs-up when satisfied, so your desirable class is tiny and unrepresentative — this is what `λ_D`/`λ_U` exist for, and it is also why you should log implicit positives (copied the answer, accepted the diff, did not re-ask) as weak desirable signals. (2) *Attribution error*: a thumbs-down on a RAG answer often means retrieval failed, not that generation was bad; training on it teaches the model to change its writing style in response to a retrieval bug. Gate on whether the retrieved context contained the answer before it enters the training set. (3) *Adversarial and joke feedback*, especially in consumer products — dedupe by user, cap per-user contribution, and drop users whose label distribution is degenerate. (4) *Temporal drift*: feedback collected against last quarter's model reflects last quarter's failure modes.

**🗣 Say this in the room:** "KTO's real contribution is not the prospect-theory value function, it is that it lets me train on the feedback signal my product already collects. Pairwise annotation for 60k comparisons is roughly $37k and three weeks; a thumbs button at 2M messages/day with a 3% feedback rate gives me 60k signals daily for free. The catch is that free data is biased data, so I gate it: drop thumbs-downs where retrieval missed, cap per-user contribution, log implicit positives, and keep a small paid pairwise set purely as a clean eval."

### ORPO folds SFT and preference optimization into one stage with no reference model. How, and what's the catch?

The observation behind ORPO is that SFT has a perverse side effect: maximizing the likelihood of good responses also raises the likelihood of *bad* responses that share tokens and style with them, because the model is learning a domain distribution, not a quality boundary. So instead of doing SFT and then correcting it with a second preference stage, ORPO adds a small preference penalty *into* the SFT loss.

```
L_ORPO = L_SFT(chosen)  +  λ · L_OR

L_OR = − log σ( log odds_θ(y_w|x) − log odds_θ(y_l|x) )
```

where the odds are computed from the length-normalized sequence probability: if `p = exp(mean-token-logprob)`, then `odds = p/(1−p)`. In code, given average token log-probs:

```python
# avg_c, avg_r: mean per-token log-prob of chosen / rejected
log_odds_c = avg_c - torch.log1p(-torch.exp(avg_c))
log_odds_r = avg_r - torch.log1p(-torch.exp(avg_r))
l_or  = -F.logsigmoid(log_odds_c - log_odds_r)
loss  = nll_chosen + lam * l_or          # lam typically ~0.1–1.0
```

Two structural consequences. First, **no reference model** — the SFT term anchors the model instead, so you get the 25% of step FLOPs and ~11% of memory back, and you drop a model from your training script. Second, **one stage instead of two**: you go from base checkpoint to aligned model in a single run, which for a small team is a meaningful reduction in pipeline surface area.

Why is the odds ratio used instead of the plain log-ratio? Because the log-ratio penalty is unbounded and would fight the SFT term hard; the odds ratio saturates more gently, so the preference term stays a mild correction rather than dominating. That is the paper's stated motivation and it matches what you see — ORPO runs are notably more stable than a DPO run with the same LR.

**📄 Paper:** Hong, Lee and Thorne (2024), *ORPO: Monolithic Preference Optimization without Reference Model* — folds preference alignment into the SFT loss via an odds-ratio penalty, eliminating both the separate preference stage and the reference model.

**The catch, and it is a real one:** ORPO wants to run **from a base model on SFT-grade data**. Its SFT term needs to be doing real work. If you already have a strong instruct checkpoint and you run ORPO on it with preference data, the NLL term is now re-training the model on data it has already seen, which risks overfitting to your preference set's chosen responses — you are effectively doing another epoch of SFT on 60k examples. The correct use is "I have base weights and a curated instruction set with rejected variants, and I want one run." The incorrect use is "I have Llama-3.1-8B-Instruct and some preference pairs" — there, DPO or SimPO is the right tool.

**⚠ Trap:** λ tuning is more sensitive than people expect and it does not transfer across datasets, because the SFT loss magnitude depends on your data's perplexity while the OR term is bounded by construction. Log both terms separately. If `λ·L_OR` is under ~5% of the total loss, you are doing SFT with extra steps; if it is over ~40%, the NLL anchor has stopped anchoring.

### Write SimPO's reward and loss. Why does the length normalization sit *inside* the reward rather than being a separate penalty?

SimPO's argument is a mismatch argument, and it is the cleanest motivation in this whole family.

At inference you rank and select sequences by their **average** log-probability — beam search normalizes by length, best-of-n reranking by log-prob normalizes by length, and the model's own decoding preferences are shaped by per-token likelihood. But DPO's implicit reward uses the **summed** log-probability with a reference offset. So the quantity you optimize during training is not the quantity that governs generation. SimPO closes the gap by making the training reward be the same shape as the generation criterion.

```
r_SimPO(x, y) = (β / |y|) · log π_θ(y|x)          # β × mean token log-prob

L_SimPO = − log σ( r(x,y_w) − r(x,y_l) − γ )
```

Two changes from DPO. **Length normalization** by `|y|` puts the reward on a per-token scale. **A target margin γ** requires the chosen reward to exceed the rejected reward by at least γ before the loss flattens out, which pushes the model past merely getting the ordering right. And there is no π_ref anywhere — the reward is absolute, not a ratio, so the method is reference-free.

```python
avg_c = seq_logprob(logits_c, labels_c) / n_tokens_c
avg_r = seq_logprob(logits_r, labels_r) / n_tokens_r
loss = -F.logsigmoid(beta * (avg_c - avg_r) - gamma)
```

The reported hyperparameter regime is a much larger β than DPO — roughly 2 to 2.5 — with `γ/β` swept in the low-fractional range (the paper explores roughly 0.3 to 1.4 for γ). That larger β is not a contradiction: β is now multiplying a *per-token* mean log-prob, which is a much smaller number than a summed sequence log-prob, so the scale has to come back up. If you carry β = 0.1 over from DPO into SimPO you will get a run where nothing happens.

**📄 Paper:** Meng, Xia and Chen (2024), *SimPO: Simple Preference Optimization with a Reference-Free Reward* — replaces DPO's reference-normalized summed reward with a length-normalized average log-probability plus a target margin, removing the reference model.

**Why normalization inside the reward beats a separate length penalty:** a separate penalty (as in length-regularized DPO) subtracts something like `α(|y_w| − |y_l|)` from the margin, which corrects the *training signal* but leaves the reward's units still summed — so the model can still buy margin by being long in ways the penalty does not exactly cancel. Normalizing inside makes length dimensionally irrelevant to the reward: a 100-token and a 1000-token response with the same average log-prob get the same reward, full stop.

### Is SimPO actually solving length exploitation, or hiding it? Argue both sides.

Argue the "solving" side first, because it is the stronger one.

Length exploitation in DPO has a specific mechanism: the reward is a summed log-prob difference, and a longer chosen response has more tokens over which to accumulate a positive log-prob delta relative to the reference. So *the objective literally pays you for length*, independent of quality. Any judge-based eval that also prefers longer answers — and every LLM judge does, robustly — then confirms the model "improved." SimPO's normalization removes the mechanism at the source. Reported length-controlled win rates for SimPO-tuned models are strong relative to DPO on the same data, and the average generation length does not blow up. That is a real fix to a real mechanism.

Now the other side. **Length normalization introduces its own bias, in the opposite direction.** Average log-prob is systematically higher for short, safe, high-frequency text. "I cannot help with that." has an outstanding average log-prob. So SimPO's reward has a mild built-in preference for terse, generic responses, and if your preference data does not push back on that, you can get a model that is crisper than it should be — dropping caveats, dropping the explanatory sentence a user needed, truncating a list at three items. In an enterprise assistant that is a *quality* regression that no length metric will catch, because the length went the "good" direction.

The second criticism is subtler and I would raise it to show depth. **Removing the reference removes the only thing tying the model to a known-good starting point.** DPO's reward is a *relative* quantity: it measures how much more likely the policy makes this text than the reference did. SimPO's is absolute. That means SimPO has no built-in notion of "don't drift," and the KL-control story has to be recovered entirely through learning rate, epochs and early stopping. In practice SimPO runs are more sensitive to LR than DPO runs, which is consistent with this.

**🗣 Say this in the room:** "SimPO removes the mechanism by which DPO pays for length, so it fixes the specific pathology. But it substitutes a bias toward high-average-likelihood text, which means terse and generic, and it gives up the reference model that was providing the drift control. My rule is: use SimPO when I have a strong instruct checkpoint, good pairs, and a measured length problem — and I add a terseness check to the eval suite specifically because I know which direction SimPO's residual bias points."

**⚠ Trap:** comparing DPO and SimPO on raw win rate. If DPO inflated length by 60% and your judge prefers long answers, DPO wins the comparison while being the worse model. Any DPO-vs-SimPO comparison must be length-controlled or it is uninterpretable — which is itself a nice illustration of why length-controlled evaluation became standard.

### My preference labels are noisy — annotators flip roughly 10% of the time. Show me cDPO and robust DPO and tell me how to pick ε.

The right frame: DPO's loss is a *cross-entropy against a hard target of 1.0*. If 10% of your labels are wrong, you are asking the model to fit a target of 1.0 on examples where the truth is 0.0, and because the loss is unbounded on the wrong side, those 10% dominate the gradient. Noisy labels in a margin loss are not a small perturbation — they are the loudest thing in your dataset.

**cDPO (conservative DPO)** is label smoothing. Assume the label is flipped with probability ε, so the BT target is `1 − ε` rather than 1:

```
L_cDPO = (1 − ε) · L_DPO(x, y_w, y_l)  +  ε · L_DPO(x, y_l, y_w)
```

In practice this is implemented directly on the logits:

```python
logits = (pol_c - pol_r) - (ref_c - ref_r)
loss = -(1 - eps) * F.logsigmoid(beta * logits) - eps * F.logsigmoid(-beta * logits)
```

The effect is that the loss now has a *finite minimum* at a margin of `log((1−ε)/ε) / β` instead of at infinity. At ε = 0.1 and β = 0.1 that is `log(9)/0.1 ≈ 22` nats of log-ratio difference. So label smoothing quietly buys you the same "bounded optimal margin" property that IPO gets from its squared loss — which is a nice connection to draw out loud. TRL exposes this as `label_smoothing` in `DPOConfig`.

**Robust DPO (rDPO)** goes further: instead of softening the target, it constructs an *unbiased estimator* of the clean-data loss from noisy data, by debiasing:

```
L_rDPO = [ (1 − ε) · L_DPO(y_w, y_l) − ε · L_DPO(y_l, y_w) ] / (1 − 2ε)
```

Note the minus sign — you are subtracting the contribution the flipped labels contribute in expectation, then rescaling. This is unbiased in expectation but higher variance, and it becomes unstable as ε → 0.5 (the denominator vanishes). TRL exposes this family via `loss_type="robust"`. **📅 Volatile:** the exact set of `loss_type` values TRL ships changes between releases — check the current `DPOConfig` docs rather than trusting a list from memory.

**How to pick ε — do not guess it, measure it.** Take 500 pairs, have two independent annotators label them, and compute the disagreement rate `d`. Under a symmetric-noise model where each annotator flips independently with probability ε, the probability two annotators disagree is `2ε(1−ε)`. Solve: `d = 0.18` → `ε(1−ε) = 0.09` → `ε ≈ 0.10`. That is the whole calculation and it takes an afternoon. If you have three annotators, majority-vote adjudication reduces the effective ε to roughly `3ε² − 2ε³` — at ε = 0.10 that is `0.03 − 0.002 = 0.028`, so triple-labeling cuts your noise from 10% to under 3%.

**💰 Math on whether to triple-label:** 60k pairs at 90 s and $25/h is $37.5k single-labeled. Triple-labeling is $112.5k. The alternative is single-labeling and setting ε = 0.1 in the loss, which costs $0. My rule: single-label the bulk with ε set from a measured disagreement study, and spend the extra budget triple-labeling a 3,000-pair *evaluation* set instead — noise in the eval set is far more damaging than noise in the training set, because it caps your ability to detect any improvement at all.

**⚠ Trap:** assuming annotator noise is symmetric. It usually is not — annotators have systematic biases (longer, more formatted, more confident-sounding responses win) that are *correlated* with a real property of the text, so they are not random flips, they are a shifted labeling function. Label smoothing does nothing about that. The fix is bias-aware guidelines, position randomization, and length-stratified spot checks, not a bigger ε.

### My inter-annotator agreement is 68%. What does that number imply for every method we've discussed?

68% raw agreement on a binary comparison is barely above the 50% floor. Convert it to a chance-corrected statistic first, because raw agreement on a two-class task is a misleading number: Cohen's κ = `(p_o − p_e)/(1 − p_e)` = `(0.68 − 0.50)/(0.50)` = **0.36**. That is "fair" agreement at best, and it is the number to say out loud because it is the one a research-adjacent interviewer expects.

Implications, in order of how much they should change your plan:

**Your evaluation ceiling is now the binding constraint, not your method.** If humans agree 68% of the time, then even a perfect model cannot exceed ~68% agreement with your labels. Any win-rate difference between two methods smaller than the noise floor is unmeasurable. Before you tune a single hyperparameter, you need to know whether you can *detect* the improvement you are chasing.

**Symmetric-noise estimate:** `2ε(1−ε) = 0.32` gives `ε ≈ 0.20`. That is a 20% per-label flip rate, which is severe enough that I would not run vanilla DPO at all. cDPO with ε = 0.2, or IPO, or robust DPO — all three cap the achievable margin, which is exactly what you want when the labels cannot support a large margin.

**But first: diagnose whether the disagreement is noise or ambiguity.** Sample 100 disagreements and read them. Three buckets, three different fixes. (1) *Genuine ties* — both responses are fine and annotators are coin-flipping. Fix: add a "tie" option and drop ties from training entirely; they contribute pure noise and cost you real gradient. This alone often takes agreement from 68% to 80%+. (2) *Guideline gaps* — annotators are applying different criteria (one weights factual precision, one weights tone). Fix: rewrite the rubric with worked examples, re-pilot, re-measure. This is labeling ops, not ML. (3) *Task is genuinely subjective* — "which of these two marketing headlines is better." Fix: accept that a single scalar preference is the wrong model, and either split into per-axis ratings (helpfulness, correctness, tone) with separate objectives, or move to a verifiable proxy if one exists.

**🗣 Say this in the room:** "κ of 0.36 means my labels support roughly a 20% flip rate, so my first move is not choosing an algorithm — it is triaging 100 disagreements into ties, guideline gaps, and genuine subjectivity. Ties get a tie option and get dropped. Guideline gaps get a rubric rewrite and a re-pilot. Only what survives is worth training on, and I would run it with label smoothing at the measured ε rather than vanilla DPO, because a margin loss with 20% wrong labels puts most of its gradient on the wrong examples."

### Give me the decision table. DPO vs IPO vs KTO vs ORPO vs SimPO — how do you actually choose?

I choose on **what data I have** and **what checkpoint I'm starting from**, in that order. The algorithmic differences are second-order compared to those two facts.

**Start: base model, and I have SFT-quality instruction data with rejected variants.** → **ORPO.** One stage, no reference model, and the SFT term is doing genuine work because the model has never been instruction-tuned. This is the right answer for "we're building an instruct model from open base weights on a small team." It is the wrong answer for anything starting from an existing instruct checkpoint.

**Start: strong instruct checkpoint, I have clean pairwise data, dataset is large (>50k pairs) and reasonably on-policy.** → **DPO, β = 0.1.** It is the default for a reason: best understood, best tooling, every failure mode documented, and you can hire someone who has debugged it. I do not deviate without a named reason.

**Same, but the dataset is small (<10k) or labels are noisy (κ < 0.5).** → **IPO**, or DPO with `label_smoothing = ε` measured from a disagreement study. Both bound the achievable margin, which is the thing you need when the labels cannot justify an unbounded one.

**I have binary feedback, not pairs — thumbs, accept/reject, resolved/escalated, test-passed/test-failed.** → **KTO.** This is the most common situation at a product company and the one people mis-solve by synthesizing fake pairs. Do not pair a thumbs-up response with an unrelated thumbs-down response from a different prompt; you will teach the model prompt-level differences, not quality. KTO exists precisely so you do not have to.

**I have pairs, a good instruct checkpoint, a measured length problem, and I want to drop the reference model for throughput.** → **SimPO**, with β around 2–2.5 and γ swept, and a terseness check added to the eval suite.

**📐 Numbers you must know** as the memorized starting configs: DPO — β 0.1, LR 5e-7 full-FT (5e-6 to 1e-5 for LoRA), 1 epoch, cosine, warmup ~10%. SimPO — β 2.0–2.5, γ/β swept in the 0.3–1.0 region, same LR regime. ORPO — λ 0.1–1.0, LR closer to SFT territory since the NLL term dominates. KTO — β 0.1, with `λ_D·n_D / (λ_U·n_U)` kept near 1–1.33. **📅 Volatile:** these are recipe conventions from 2024–2025 open-weight runs; re-verify against current library defaults before quoting them as facts.

**⚠ Trap:** treating this as a leaderboard question. There is no stable ranking of these five on general benchmarks; published comparisons disagree, and most differences are inside the noise once you length-control. If an interviewer asks "which is best," the correct answer names the decision inputs and says the ranking is contested — claiming a winner is the tell that you have read benchmark tables rather than run jobs.

### Reference-free methods drop the reference model. What do you lose that nobody talks about?

Three things, and the third is the one that separates a senior answer.

**You lose the KL accounting.** With a reference, `β log(π_θ/π_ref)` *is* a per-example estimate of divergence from the starting point. You can log it, alert on it, and pick a checkpoint on a KL budget rather than on vibes. Reference-free, there is no such quantity in your training loop at all. You can still measure KL post hoc by loading the old checkpoint and scoring a dev set, but nobody does that, so in practice reference-free runs have *no* drift instrumentation and drift is the failure mode.

**You lose the implicit reward model.** Everything I described earlier — best-of-n reranking with `r̂`, active-learning selection of the next round's pairs, cheap in-loop preference labeling for iterative rounds — all of it depends on having `π_ref` at inference time. SimPO's reward `β/|y| · log π(y)` is not the same object: it is an absolute likelihood score, and absolute likelihood is a *bad* quality signal (it ranks generic text highly). If your roadmap includes iterative rounds, that is a strong argument for keeping the reference.

**You lose the safety anchor, and it shows up as regression on things you never trained on.** The reference is what keeps the model tethered to behaviors it learned in SFT that are not represented in your preference data at all: refusal boundaries, tool-call formatting, multilingual competence, the ability to say "I don't know." DPO's reference term applies a restoring force on *every token* of every completion, including tokens governing behaviors your preference pairs never mention. Reference-free objectives have no such force; the only thing preventing regression is that you stopped early. I have seen an ORPO run where the model's structured-output compliance rate fell from 99.4% to 91% and nothing in the preference data mentioned JSON at all.

**💰 The cost you saved, priced honestly:** the reference is ~25% of step FLOPs and ~11% of memory. For an 8B, 60k-pair, 3-epoch run, that is roughly 10 H100-hours total, of which ~2.5 are the reference — about $7.50 at $3/H100-hour. Precomputing reference log-probs recovers most of that for a one-time 8% cost. **So the compute argument for reference-free methods is worth single-digit dollars on a small run, and you are trading it for your only drift instrument.** At 70B and weekly iteration the arithmetic changes and the argument gets real, but at 8B it does not, and I would push back on "we went reference-free to save compute" in a design review.

**🗣 Say this in the room:** "Reference-free is a genuine simplification — one fewer model in the script, ~25% fewer step FLOPs. But the reference is not just a regularizer, it is your instrumentation: it gives you a per-example KL estimate, an implicit reward you can use for best-of-n and for selecting the next round's data, and a restoring force on behaviors your preference data never mentions. At 8B the compute saving is under ten dollars a run. I'd precompute the reference log-probs and keep it."
