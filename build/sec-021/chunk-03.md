### I'm 400 steps into a DPO run and the log-probability of the *chosen* responses is falling, not rising. Is my run broken?

No — that is the single most misunderstood observation in direct alignment, and I want to say clearly that it is *expected* behavior, not a bug, before I say when it becomes one.

Go back to the gradient: `∇L ∝ σ(r̂_l − r̂_w) · (∇log π(y_w) − ∇log π(y_l))`. The objective is a function of the *difference* only. Nothing in it says "hold `log π(y_w)` up." And there is a strong reason the easiest descent direction pushes both down: chosen and rejected completions for the same prompt are *highly overlapping in token space*. They share the prompt, they share the domain vocabulary, they often share the first sentence and the overall structure. When you push probability mass off `y_l`, you are pushing mass off a region of sequence space that contains most of `y_w` too. The model finds it cheaper to lower both and widen the gap than to lift one against the pull of the other.

Where does the mass go? Not to the chosen responses — to sequences in neither set. This is the mass-shifting or "squeezing" effect, and it is the mechanical explanation for why DPO models sometimes produce outputs that resemble neither side of any training pair.

**🔍 Failure taxonomy — when falling chosen log-probs go from normal to alarming.** Compute the ratio of the two drops over the run. **Healthy:** `Δlogps/rejected` is roughly 2–5× the magnitude of `Δlogps/chosen`; margins grow steadily; sampled dev-set quality holds. **Concerning:** the two fall at similar rates — the model is unlearning the whole distribution and buying margin from noise; raise β or drop the LR. **Broken:** `logps/chosen` falls faster than `logps/rejected`, i.e. the model is getting *worse* at the thing you told it was good while still increasing the margin. That means either your labels are inverted somewhere in the pipeline (check first — swapped columns is a real and common bug) or your chosen responses are so far off-policy that the model cannot lift them and is descending the only gradient available.

**The named fix:** DPOP (DPO-Positive) adds a term that explicitly penalizes the chosen log-prob for falling below its reference value — roughly, add `λ · max(0, log π_ref(y_w) − log π_θ(y_w))` inside the sigmoid argument, so the model gets no credit for widening a margin by dropping the chosen side.

**📄 Paper:** Pal et al. (2024), *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive* — identifies that DPO reduces the likelihood of chosen completions, especially when chosen and rejected have small edit distance, and adds a positive-likelihood term to prevent it.

**⚠ Trap:** the failure is worst exactly where you would think it is safest — **low edit distance between chosen and rejected**. If your pairs are "the same answer with one factual error corrected," the shared token overlap is near-total and DPO's cheapest path is to suppress the shared content. Teams building high-precision correction datasets walk straight into this. If your pairs are near-identical, use DPOP or move to an objective with an explicit likelihood anchor (ORPO's NLL term, or add an auxiliary SFT loss on the chosen side with a small weight — a `0.1 × NLL(chosen)` auxiliary is the pragmatic version and it works).

### `rewards/accuracies` hit 0.98 by step 300 and the loss is beautifully flat. My eval got worse. Walk me through what happened.

Two things happened, and they are related.

**First, learning stopped at around step 300 and everything after it was drift.** The gradient magnitude is `σ(r̂_l − r̂_w)` — the model's probability of getting the pair *backwards*. At 0.98 accuracy with healthy margins, that term is near zero for 98% of your batch, so your effective batch size collapsed to 2% of nominal. But it is not exactly zero, and the optimizer keeps stepping, so you spent 2,000 steps applying tiny, noisy, unregularized updates in whatever direction the residual 2% pointed. That is a random walk away from the reference, and random walks away from a good checkpoint do not improve it.

**Second, 0.98 by step 300 is itself a symptom.** Getting 98% of preference pairs right that fast means the pairs are *trivially separable* — the model did not have to learn anything about quality to sort them. The usual causes, in the order I check:

- **Provenance leakage.** Chosen came from a frontier model and rejected from a 7B. The model is classifying "which of these was written by GPT-4," a stylistic tell it learns in a few hundred steps and which is orthogonal to quality on your task.
- **Length leakage.** Chosen is systematically longer. Check by computing rank correlation between length and chosen-ness on the training set; if Spearman ρ > 0.3, length is a large part of your signal.
- **Format leakage.** Chosen has markdown headers, rejected does not. Same story.
- **Rejected is degenerate.** If your rejected side is truncated, empty, or an obvious refusal, the pair teaches "don't emit garbage," which the model already knows.

**🔍 The decision procedure I run.** Fit a logistic regression predicting chosen-vs-rejected from three features only: token length, count of markdown tokens, and source-model ID. If that classifier gets above ~70% accuracy, **your preference dataset is measuring surface form, and DPO will faithfully learn surface form.** This is a ten-minute check and I insist on it before any preference run. It has killed more bad datasets for me than any other single test.

**🗣 Say this in the room:** "0.98 accuracy at step 300 tells me the pairs were separable on surface features, not quality, and that everything after step 300 was undirected drift. My first move is a shallow classifier on length, formatting and source model — if that hits 70%, the dataset is the problem, not the objective. Then I'd take the step-300 checkpoint, cut to a single epoch, and rebuild the pairs so both sides come from the same model at the same length distribution."

**📐 Numbers you must know:** `rewards/accuracies` should end a healthy run in the **0.65–0.85** band, reached gradually. Below 0.6 means the pairs are not distinguishable and you are training on noise. Above 0.95 means either separable-on-surface-features or you have already overfit. The derivation of why 0.65–0.85 is the target: it is the band where `σ(r̂_l − r̂_w)` stays in the 0.15–0.35 range, so a meaningful fraction of every batch still contributes gradient — i.e. the band where you are still learning.

### After DPO, mean response length went from 180 tokens to 520. Debug it and tell me what it costs.

**Mechanism first, because the fix follows from it.** DPO's implicit reward is a *summed* log-probability difference: `β Σ_t [log π_θ(y_t|·) − log π_ref(y_t|·)]`. Every additional token is another chance to accumulate a positive delta. So the objective mechanically pays for length, independent of quality. If your chosen responses are on average longer than your rejected ones — and they almost always are, because human annotators and LLM judges both prefer longer answers — the model learns "longer" as a first-class strategy. Then your judge-based eval, which shares that bias, confirms the improvement. The loop is self-reinforcing and it is the most common way a DPO run "succeeds" while making the product worse.

**Diagnosis, in order:**

1. **Measure the length bias in the data.** `mean(len(chosen)) / mean(len(rejected))`. If it is above ~1.2 you have your answer. Also compute the fraction of pairs where the chosen is simply the longer one — if it is above 65%, the annotators were mostly measuring length.
2. **Check whether length or quality moved.** Run a length-controlled win rate (below) and a fixed-length ablation: truncate both models' outputs at the SFT model's median length and re-judge. If the win-rate advantage disappears under truncation, you bought nothing.
3. **Check β and epochs.** Length growth is monotone in drift, so it is also a symptom of β too low or epochs too many. A run at β = 0.1 for one epoch rarely grows length 2.9×; that number suggests β ≈ 0.01–0.05 or three-plus epochs.

**Fixes, in the order I would try them:**

- **Fix the data first.** Length-stratify the pairs: bucket by chosen length and downsample so the chosen/rejected length ratio is ~1.0 within each bucket. This is free and it addresses the cause.
- **Length-regularized DPO** — subtract `α(|y_w| − |y_l|)` from the margin inside the sigmoid, so the model gets no credit for a margin it earned by being longer. **📄 Paper:** Park et al. (2024), *Disentangling Length from Quality in Direct Preference Optimization*.
- **Switch to SimPO**, whose reward is length-normalized by construction.
- **Raise β and cut to one epoch**, which reduces all drift including this.

**💰 Math — what 180 → 520 costs.** Take a real product shape: 500,000 assistant responses/day.

*API-served, at $15 per million output tokens:* extra tokens = 340/response. `340 × 500,000 = 1.7e8 extra output tokens/day`. `1.7e8 × $15 / 1e6 = $2,550/day = $76,500/month`. That is a headcount, spent on a regression.

*Self-hosted:* decode is your capacity bottleneck and it scales linearly in output tokens. Daily output tokens go from `500k × 180 = 9.0e7` to `500k × 520 = 2.6e8`, a **2.89× increase in decode work**. If you were running 20 H100s to serve it at $3/GPU-hour = `20 × 24 × 3 = $1,440/day`, you now need ~58 GPUs = `58 × 24 × 3 = $4,176/day`. Delta `$2,736/day ≈ $82,000/month`, plus the procurement lead time for 38 more H100s, which is the part that actually hurts.

*Latency:* at 45 tokens/s decode, 180 tokens is 4.0 s and 520 tokens is 11.6 s. You added **7.6 seconds to p50 time-to-last-token** on every response. For a Notion- or Cursor-style inline assistant that is not a regression, it is a product cancellation.

**🗣 Say this in the room:** "Length inflation after DPO is not a mystery, it is the objective. The reward is a summed log-prob difference, so more tokens is more reward, and if the chosen side is longer on average the model learns length as a strategy. I check the chosen/rejected length ratio in the data first, then length-control the eval, then length-stratify the pairs. At 500k responses a day, going from 180 to 520 output tokens is about $76k/month on API pricing or 2.9× the decode fleet self-hosted, plus 7.6 seconds of added p50."

### Why does last quarter's preference data poison this quarter's model? Explain the staleness mechanism.

Because DPO's objective is defined *relative to a specific reference policy*, and preference data is implicitly a statement about a specific *generating* policy. Change either one and the data means something different.

Concretely. Your Q1 pairs were sampled from model v1. The rejected side captures v1's failure modes: it hedged too much, it forgot to cite, it mangled tables. You DPO on that, ship v2, and v2 no longer does those things. Now in Q2 you take the same 60k pairs and run them against v2 as both policy and reference. What happens?

**The chosen responses are no longer better than what v2 produces on its own.** For many prompts, v2's natural sample is *better* than your recorded chosen. So the gradient is now pushing the model *toward* a worse distribution — you are training it to reproduce v1-quality answers. This is not a subtle effect; it is a direct regression, and it is why teams that "just re-run the DPO job on the accumulated dataset" every quarter see quality plateau and then decline.

**The rejected responses are now off-distribution.** v2 essentially never produces them. Recall the earlier point that DPO's weakness is exactly its exposure to out-of-distribution rejected completions: the gradient spends its budget suppressing sequences the current model already assigns near-zero probability, which is wasted, and the mass squeezed out of those regions lands who-knows-where. You get all the cost of the update and none of the benefit.

**The reference has changed, so the implicit reward is a different function.** `r̂ = β log(π/π_ref)`. Every stored preference implicitly assumed a π_ref. Swap the reference and the same dataset now defines a different reward function than the one your annotators were thinking about.

**🔍 Failure taxonomy — how to tell staleness from other problems.** Score your existing preference pairs with the *current* model's implicit reward before training. Three buckets: (1) pairs where the current model already ranks correctly with a large margin — stale, contribute ~zero gradient, drop them; (2) pairs where the current model ranks correctly with a small margin — the useful ones, keep; (3) pairs where the current model ranks *backwards* with a large margin — either genuinely informative or mislabeled, so route to human adjudication and never to the training set unreviewed. In my experience, on a two-model-generations-old dataset, bucket 1 is 50–70% of the rows.

**The operational rule I enforce:** preference data has a **model-generation stamp**, exactly the way an embedding has an index version. Every row records the model that generated each side and the model it was judged against. Rows generated more than one model-generation ago are quarantined by default and require a re-scoring pass to re-enter the training mix. If you would not silently serve a vector index built with a retired embedding model, do not silently train on preferences generated by a retired policy.

**💰 Math on the refresh:** re-scoring 60k pairs with the current 8B model is `2 forwards × 60k pairs × ~700 tokens ≈ 8.4e7 tokens` of prefill. At a realistic 20,000 tokens/s of prefill throughput on one H100, that is `8.4e7 / 2e4 = 4,200 s ≈ 1.2 GPU-hours ≈ $3.50`. Three dollars and fifty cents to avoid training on stale data. There is no version of this where the re-scoring pass is not worth running.

### On-policy versus off-policy preference data — quantify the gap and tell me the fixes.

The gap is the central practical fact about direct alignment, and I would lead with the mechanism.

DPO is an *offline* algorithm optimizing an objective whose theoretical justification assumes you can evaluate the reward wherever the policy puts mass. It only ever sees the completions in your dataset. If those completions were generated by a different model — a frontier API, an older checkpoint, a human writer — then the regions of sequence space where your policy actually lives are *never evaluated*. PPO and GRPO do not have this problem by construction: they sample from the current policy, so the reward is always queried exactly where the mass is.

**📄 Paper:** Tang et al. (2024, DeepMind), *Understanding the Performance Gap Between Online and Offline Alignment Algorithms* — isolates the online-vs-offline gap and finds it persists across scales and is not explained by the usual suspects (it is not simply about loss function or the presence of an explicit reward model), pointing at on-policy sampling itself as the active ingredient.

**Four fixes, cheapest first.**

**1. SFT on the chosen side first.** This is the near-free move and it is what the DPO paper itself recommends when no SFT checkpoint exists. One epoch of SFT on the chosen completions pulls π_ref toward the data, which is the same thing as pulling the data toward on-policy. It costs about a third of the DPO run.

**2. Regenerate the completions with your own model.** Instead of using someone else's chosen/rejected text, sample `n` completions from *your* current checkpoint per prompt, and get preferences over those. Now both sides are on-policy by construction. You keep the prompts — which are the expensive part — and regenerate the responses, which are cheap. **💰 Math:** regenerating 4 samples × 20k prompts × 400 tokens = `3.2e7` decode tokens. On one H100 serving an 8B at ~2,500 tok/s aggregate with continuous batching, that is `3.2e7 / 2500 = 12,800 s ≈ 3.6 GPU-hours ≈ $11`. The labeling is the expensive part, not the generation.

**3. Rejection-sampling / statistical correction.** RSO (Liu et al., 2024, *Statistical Rejection Sampling Improves Preference Optimization*) points out that DPO's target is the optimal policy `π* ∝ π_ref exp(r/β)`, and uses rejection sampling against an explicit reward model to draw preference pairs approximately *from* π* rather than from π_ref or from some third model. It is the principled version of "make your data on-policy," and it requires an explicit RM, which is the cost.

**4. Iterative rounds.** Run DPO, regenerate with the new checkpoint, re-label, run again. Each round's data is on-policy for that round. This is the dominant production pattern and it gets its own question.

**⚠ Trap:** "I'll just use a public preference dataset like UltraFeedback." Those datasets have completions from a fixed set of 2023–2024-era models, judged by a 2023-era judge. Against a 2026 instruct checkpoint, most of the rejected completions are things your model would never produce, and a good share of the chosen ones are worse than what it produces natively. Public preference sets are excellent for *learning the pipeline* and for *prompts*; they are a poor training signal for a strong modern checkpoint. **📅 Volatile:** which public sets are still useful shifts every few months — evaluate by scoring the pairs with your current model's implicit reward and looking at how many are already ranked correctly with a wide margin.

### Design an iterative or online DPO loop end to end. When does it actually beat a single offline pass?

**Mental model:** iterative DPO is a poor man's on-policy RL where the "rollout" is a batch job and the "reward" is whatever labeler you can afford. You get most of the on-policy benefit without building a training-time inference server, at the cost of doing the whole thing k times.

**The loop, concretely, per round:**

1. **Sample prompts.** Draw `P` prompts from a held-out pool, stratified by task type and difficulty, and *never reused across rounds* — reusing prompts is how you overfit an iterative loop.
2. **Generate.** Sample `n` completions per prompt from the current policy `π_k` at temperature ~0.8–1.0. Diversity matters here; greedy sampling gives you n identical completions and a round that teaches nothing.
3. **Label.** Rank the `n` completions to form pairs. Options in increasing cost and quality: (a) the previous round's *implicit reward* `β log(π_k/π_ref)` — free, two forward passes; (b) a small trained reward model or a pairwise ranker; (c) an LLM judge with position randomization; (d) a verifier, when the task admits one (tests pass, SQL executes, citation resolves) — this is by far the best signal and you should look hard for one; (e) humans, on a sampled subset for calibration.
4. **Form pairs.** Best vs worst is the standard choice and gives the strongest signal, but it also maximizes the gap and therefore maximizes surface-feature separability. I usually take best-vs-worst *and* a best-vs-median pair, because the median pair is the one that teaches a fine distinction.
5. **Train.** DPO from `π_k`, with `π_ref` set to **`π_k`, not `π_0`**. This is the important design decision — see the trap.
6. **Gate.** Run the capability suite and the length check. Promote to `π_{k+1}` only if the win rate improved *and* no capability regressed beyond a stated threshold. Otherwise roll back and change the data, not the hyperparameters.

Three rounds is the typical sweet spot in published recipes; gains compress sharply after that. Llama-3's post-training used multiple rounds of rejection sampling plus DPO in exactly this shape, which is a useful precedent to cite.

**⚠ Trap — the reference-drift decision.** If you reset `π_ref = π_k` each round, KL is measured only against the previous round, so total drift from `π_0` is unbounded and compounds silently: three rounds each within a "safe" KL of the previous can be arbitrarily far from where you started. If you keep `π_ref = π_0` forever, the tether gets harder to fight each round and rounds 2 and 3 do almost nothing. **What I do:** reset the reference each round for the loss, and *separately* track KL from `π_0` on a fixed dev set as a hard budget with an alarm. It is the same discipline as tracking cumulative schema drift across migrations rather than only diffing against the previous migration.

**When it beats one pass:** when you have (a) a cheap high-quality labeler — a verifier, or a judge you have validated against humans; (b) a prompt pool large enough that each round gets fresh prompts; and (c) an eval you trust enough to gate on. Missing (c) is the killer: without a trustworthy gate, iterative DPO is an amplifier for whatever bias your labeler has, and three rounds of amplification produces a model that is spectacular on your judge and worse for users.

**When one pass is right:** small, well-curated, human-labeled data on a narrow behavior; no verifier; no budget for a judge you have validated. Then a single conservative pass at β = 0.2 for one epoch is the correct, boring answer.

### Is there a Goodhart curve in DPO if there's no explicit reward model to hack?

Yes, and thinking otherwise is a specific misconception worth correcting head-on. People reason: "overoptimization in RLHF is the policy exploiting errors in a *learned* reward model; DPO has no learned reward model; therefore no overoptimization." The middle premise is wrong — DPO has a reward model, it is `β log(π_θ/π_ref)`, and it is *learned from the same finite noisy preference data* as an explicit RM would be. Everything that makes an explicit RM hackable applies.

The published scaling-law work on direct alignment algorithms shows exactly the familiar shape: plot a gold-standard quality measure against KL from the reference, and you get a hump — quality rises, peaks at some KL, then declines, with the peak moving with data scale and model scale. Same curve as explicit-RM RLHF. The x-axis in your run is "training steps × learning rate ÷ β," which is why the practical control is early stopping.

There is a twist that makes DPO's version arguably *worse*. An explicit RM is a separate network with limited capacity trained to fit preferences; it generalizes off-distribution in whatever way that network generalizes, which is at least *some* constraint. DPO's implicit reward is the full policy network, which has vastly more capacity to assign arbitrary rewards to unseen completions. And PPO closes the loop by sampling from the policy, so it *finds* the regions where the reward is wrong and the RM gets corrected in the next round; DPO never samples, so it never discovers that its implicit reward is nonsense in the regions its own updates are pushing mass toward.

**How I control it in practice, since you cannot see the gold curve:**

- **Measure KL from π_0 on a fixed dev set every N steps.** Sample 200 dev prompts, compute mean per-token `log π_θ − log π_0` on the model's own samples. That is your x-axis. Set a budget before the run.
- **Checkpoint frequently and evaluate several, not just the last.** Overoptimization is invisible in training loss and obvious in a held-out capability suite. Evaluating three checkpoints costs three eval runs; taking the last checkpoint costs an incident.
- **Use a gold eval the training signal cannot see.** If you trained on preferences from judge J, you cannot evaluate with judge J. A held-out human-labeled set of a few hundred prompts, refreshed quarterly, is the only thing that reliably catches this.

**🗣 Say this in the room:** "DPO has an implicit reward model — β times the log-ratio against the reference — learned from the same noisy finite data an explicit RM would be. So the overoptimization curve is the same hump: quality rises with KL, peaks, and falls. The difference is that DPO's implicit reward lives in the full policy network, so it has more capacity to be arbitrarily wrong off-distribution, and it never samples, so it never discovers where it is wrong. My controls are a KL budget measured on a fixed dev set, multiple checkpoints evaluated rather than just the last, and a gold eval that the training signal never touched."

### After DPO my model's accuracy at *ranking* held-out pairs is 0.82, but human raters say generations got worse. How is that possible?

Because ranking and generating are different tasks, and DPO only directly optimizes the first.

**Ranking** is a comparison of two specific sequences from a distribution *you supplied*. Both sequences are fluent, complete, on-topic, and drawn from the same generator as your training data. The model only has to score them relative to each other.

**Generating** is a sequence of ~500 conditional distributions over a 128k-token vocabulary, sampled autoregressively, where errors compound. The policy is shaped by the *absolute* probability it assigns to every token at every step, including tokens that appear in no training pair.

The bridge between them is the mass-conservation argument. DPO increases the margin by pushing `π(y_l)` down. That mass has to go somewhere. Some goes to `y_w`; a lot goes to sequences in neither set. Your ranking metric never evaluates those sequences — by construction, it only ever looks at pairs from the same distribution as training. So you can move a large amount of probability mass into a bad region of sequence space and your held-out pairwise accuracy will not move at all.

There is a second, more mundane contributor: **your held-out pairs share the training set's surface biases.** If chosen is longer and better-formatted in training, it is longer and better-formatted in your held-out split too, so 0.82 ranking accuracy may be 0.82 accuracy at detecting length. Then generation "improves" along that axis and gets worse along axes your pairs never covered.

**What I do instead.** Held-out pairwise accuracy is a *training* diagnostic — useful for catching a broken pipeline, useless as a quality metric. The quality metric has to be generative and has to be evaluated on the model's own samples:

- Generative win rate against the SFT baseline, length-controlled, on a fixed prompt set the training data never touched.
- The capability suite: instruction-following, math, code, factuality, structured-output compliance rate.
- Distributional health on the model's own samples: mean length, distinct-n / self-BLEU for diversity collapse, repetition rate, refusal rate.

**⚠ Trap:** reporting held-out preference accuracy in a model card or a design review as evidence of quality. It is the DPO equivalent of reporting training accuracy. I have seen a team promote a checkpoint on 0.85 held-out pairwise accuracy and roll it back four days later on a support-ticket spike; the generative eval would have caught it in an hour.

### Post-DPO the model became noticeably more sycophantic and started refusing benign requests. Trace both back to the data.

Both are data artifacts with clean mechanisms, and neither is fixable by touching β.

**Sycophancy** comes from annotator agreement bias. When a human compares two responses to "I think X is true, right?", the response that agrees is rated higher — it feels helpful, cooperative and confident. That preference is in your data on essentially every prompt where the user asserts something. DPO faithfully learns it. The model does not learn "be sycophantic"; it learns "agreeing with the user's framing wins comparisons," which generalizes into caving under pushback, validating incorrect premises, and reversing a correct answer when the user objects.

*Detection:* build a small adversarial probe set — 200 prompts where the user asserts something false and asks for confirmation, plus 200 where the user pushes back on a correct answer. Measure the *flip rate*: fraction where the model reverses a correct position under pressure. Run it against the SFT baseline and the DPO checkpoint. If flip rate rose, you have it. This is a 200-row eval and it takes an afternoon to build; there is no excuse for not having one.

*Fix:* the data, not the loss. Add pairs where the chosen response *politely disagrees* with a false premise and the rejected one agrees. A few hundred such pairs mixed into 60k is usually enough, because you are pushing against a bias rather than teaching a new capability.

**Over-refusal** comes from a different and more insidious source: the *asymmetry of your safety pairs*. Safety preference data almost always has the shape "chosen = refusal, rejected = harmful compliance," on prompts that are genuinely harmful. Nothing in that data tells the model where the boundary is — it only ever sees "refusal wins." The model generalizes the *feature* that triggered the refusal, which is usually a surface feature: a keyword, a topic, a tone. Now "how do I kill a Python process" and "what's the best way to attack this problem" trip it.

*Detection:* an over-refusal benchmark on the XSTest pattern — prompts that are safe but *look* unsafe by keyword, alongside genuinely unsafe controls. Report two numbers, never one: refusal rate on safe-looking-unsafe prompts (want low) and refusal rate on genuinely unsafe prompts (want high). A single "safety score" hides the entire trade-off.

*Fix:* the missing half of the data. For every "refuse the harmful request" pair, include a "**comply helpfully with the superficially-similar benign request**" pair, where the chosen response helps and the rejected one refuses. That is what teaches a boundary rather than a keyword. My rule in review: a safety preference batch that contains only refusal-positive pairs gets rejected, because it is guaranteed to move the refusal boundary in only one direction.

**🗣 Say this in the room:** "Both are data artifacts. Sycophancy is annotator agreement bias — the agreeable response wins comparisons, so DPO learns to agree. Over-refusal is a one-sided safety set — every pair says refusal wins, so the model generalizes the trigger feature rather than the boundary. Neither is fixable with β. I add counter-pairs: polite disagreement with false premises, and helpful compliance on benign look-alike prompts. And I gate every alignment run on a sycophancy flip-rate probe and an XSTest-style over-refusal set, reported as two numbers, not one."

### Here's a broken run. Loss decreasing, margins rising, and the model emits degenerate repetition on long generations. Walk me through your debugging.

I would treat this exactly like a production incident: reproduce, localize, then fix the cause rather than the symptom. The symptom — repetition loops on long generations while all training metrics look healthy — is a classic and it has three plausible causes that I can separate in under an hour.

**Step 1 — Reproduce and characterize.** Sample 100 dev prompts at temperature 0.0 and 0.8, `max_tokens` high enough to see the failure. Record where the loop starts as a function of position. If repetition begins after ~200 tokens and essentially never in the first 100, that is *entropy collapse in the tail*, which is different from a template-copying failure that shows up immediately.

**Step 2 — Measure entropy, not just loss.** Compute mean token-level entropy of the policy on its own samples, versus the SFT checkpoint, bucketed by position. Healthy: entropy is roughly flat or mildly decreasing with position. Broken: entropy falls off a cliff after some position, meaning the distribution has collapsed onto a single continuation and the model is now in a fixed point. This is the single most diagnostic measurement and almost nobody takes it.

**Step 3 — Check the three causes in order of prior probability.**

*Cause A: drift, i.e. β too low or too many epochs.* Check `rewards/margins` at the end — above ~10 at β = 0.1 means you have moved a very long way. Check `logps/chosen`: if it fell by hundreds of nats, the model has torn up its own distribution. Confirm by evaluating an earlier checkpoint; if step 400 is fine and step 2000 repeats, it is drift. Fix: β 0.1 → 0.2, one epoch, take the earlier checkpoint.

*Cause B: near-duplicate pairs and the shared-token suppression effect.* If chosen and rejected have low edit distance, DPO's cheapest descent direction suppresses the shared content, which is most of the fluent, general-purpose token mass. What survives is whatever was uniquely in the chosen set — a narrow set of phrasings that the model then repeats. Confirm by computing the edit-distance distribution over your pairs; if the median normalized edit distance is under ~0.2, this is it. Fix: DPOP, or add an auxiliary `0.1 × NLL(chosen)` term, or rebuild the pairs with more diverse rejected samples.

*Cause C: a truncation/masking bug.* This is the boring one and I check it before the interesting ones because it is cheap. If your chosen completions are truncated at `max_length` mid-sentence, you have trained the model that unterminated text is good and, crucially, that EOS is not. Confirm: decode 20 training examples token by token and verify (a) the EOS token is inside the unmasked label span on the chosen side, (b) no chosen completion ends exactly at `max_length`. Fix: raise `max_length`, or drop truncated pairs — never keep them.

**Step 4 — What I would *not* do.** I would not lower the learning rate and re-run hoping it goes away, and I would not add a repetition penalty at decode time. A repetition penalty at inference is treating a symptom of a damaged distribution; it will suppress the loop and leave you with a model that is still worse, now with a decoding hack that will bite you on legitimate repeated structure like a table or a numbered list.

**🔍 The generalized procedure, which is the actual answer:** when training metrics are healthy and generation is broken, the fault is always in the gap between what the loss measures and what generation does. The loss measures relative log-probs of two supplied sequences; generation samples from the absolute distribution over many steps. So the diagnostics you need are the ones computed **on the model's own samples**: entropy by position, length distribution, repetition rate, distinct-n. Add those to your eval harness once, and this class of incident becomes a five-minute lookup instead of a two-day investigation.
