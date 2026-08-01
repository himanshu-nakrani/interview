### Explain entropy collapse — what causes it, how you detect it, and what you do about it.

Entropy collapse is not a pathology you avoid; it is the *equilibrium* of the algorithm, and your job is to slow the approach to it long enough to extract learning.

Mechanism. Policy gradient reinforces sampled trajectories in proportion to their advantage. Trajectories that succeed get their tokens' probabilities raised. Raising a token's probability lowers the entropy of that position's distribution. Lower entropy means the next round of sampling explores less. Less exploration means you rediscover the same successful trajectories, which get reinforced again. It is a positive feedback loop with no restoring force in the objective — nothing in `min(ρA, clip(ρ)A)` rewards being uncertain. The KL-to-reference term is the only thing pushing back, and it pushes toward the *reference's* distribution, not toward high entropy per se, and it is often turned off.

The empirical signature is a steep entropy drop over the first 100–300 steps followed by a long flat tail near zero, with reward plateauing at almost exactly the point where entropy bottoms out. That co-timing is the tell: **when entropy hits the floor, learning stops, because all G samples in a group become the same string, reward variance is 0, and advantages are 0.** Entropy collapse and zero-gradient collapse are the same event viewed from two angles.

**What I put on the dashboard, in priority order:** mean per-token policy entropy; fraction of degenerate groups; distinct-completion rate within a group (exact-dedup the G strings — if it drops below ~3 distinct out of 8, you are done); clip fraction; mean response length; and reward.

**The interventions, roughly in order of how much I trust them:**

1. **Dynamic sampling / difficulty filtering.** Keeps you training on prompts where outcomes still differ, which is the direct antidote.
2. **Clip-higher** (asymmetric ε — next question). Cheap, well-motivated, and the single most effective knob in my experience.
3. **Higher sampling temperature during rollouts.** Free exploration, but it also degrades the on-policy assumption slightly (you are sampling from a tempered policy but computing logprobs under the untempered one unless you are careful — make sure your `old_logprobs` come from the actual sampling distribution).
4. **Entropy bonus in the loss.** The classic RL answer, and I use it last, because in LLM RL it is notoriously touchy: too small and it does nothing, too large and the model degenerates into high-entropy gibberish, and the usable window is narrow and task-dependent.
5. **Targeted regularization on high-covariance tokens.** A 2025 research line observes that entropy loss is driven by a small set of tokens with high covariance between log-prob and advantage, and proposes clipping or KL-penalizing specifically those. **📅 Volatile:** this is recent and the specific method names are moving; describe the mechanism, and verify the current state of the art before citing a technique by name.

**⚠ Trap:** treating falling entropy as a *sign of learning*. It correlates with learning early — the model is becoming confident about correct things — and then it becomes the cause of learning stopping. The useful reading is not the level but the *slope relative to the reward slope*. Entropy falling while reward rises is healthy. Entropy still falling after reward has plateaued means you are burning your remaining exploration budget for nothing, and you should have stopped or intervened 100 steps ago.

### What is "clip-higher" and why does asymmetric clipping specifically help exploration?

The mental model: the standard symmetric clip `[1−ε, 1+ε]` is not symmetric in its *effect*, because probabilities live in `[0,1]` and low-probability tokens have far more headroom above than below.

Consider a token with `π_old = 0.01`. With ε = 0.2, the clip permits `π_θ` up to `0.012` — an absolute increase of 0.002. Now a token with `π_old = 0.9`: the clip permits up to `1.08`, i.e. it does not bind at all in practice; that token can go to 1.0 freely. So the symmetric ratio clip **allows high-probability tokens to be reinforced all the way to saturation while capping low-probability tokens to a tiny absolute gain.** Exploratory tokens — the unusual continuation, the "wait, let me reconsider" — are exactly the low-probability ones, and they are the ones the clip throttles. The clip is structurally anti-exploration.

Clip-higher decouples the two bounds: `clip(ρ, 1−ε_low, 1+ε_high)` with `ε_high > ε_low`, typically something like `ε_low = 0.2, ε_high = 0.28`. Raising only the upper bound gives rare-but-good tokens room to grow while keeping the downward bound tight, so you keep the safety on the destructive direction and remove it on the exploratory one.

**📄 Paper:** Yu et al. (2025), *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* — introduced decoupled clipping ("clip-higher") together with dynamic sampling, token-level policy-gradient loss, and overlong reward shaping, reporting these as the fixes needed to make R1-Zero-style training reproducible; they report the entropy of the policy is maintained substantially longer with clip-higher than with symmetric clipping.

**The implementation is two lines** — the `clip_low`/`clip_high` parameters in the loss function I wrote earlier. There is no excuse not to expose them.

**⚠ Trap:** setting `ε_high` very large (say 0.5) on the theory that more exploration is better. The upper clip is your only protection against a single huge positive-advantage update on a rare token, which can spike that token's probability and destabilize the run. The published values sit around 0.28 for a reason. Treat it as a small, deliberate asymmetry, not as "turn off the clip."

### Walk me through DAPO's full set of changes and why each one exists.

Four changes, and the useful framing is that each one fixes a specific *structural* defect in vanilla GRPO rather than being a tuning trick. I like this paper as an interview topic because reciting the four demonstrates you understand GRPO's failure modes without being asked about them individually.

**1. Clip-higher.** Decoupled `ε_low`/`ε_high`. Fixes the anti-exploration asymmetry of the ratio clip described above. Preserves entropy longer.

**2. Dynamic sampling.** Filter out groups where all samples are correct or all are wrong, and keep generating until the batch is full of groups with non-zero reward variance. Fixes zero-gradient collapse and, importantly, makes the *effective* batch size equal to the *configured* batch size, which restores the meaning of your learning-rate schedule. Costs extra generation; the paper's argument is that convergence in wall-clock is not hurt because the discarded samples produced no gradient anyway.

**3. Token-level policy-gradient loss.** One denominator over all tokens in the batch instead of averaging within each response. Fixes the structural length bias in the loss normalization, so long low-quality responses are penalized in proportion to their length instead of having the penalty diluted.

**4. Overlong reward shaping.** Long-CoT training generates responses that hit the generation length cap. Truncated responses are usually scored 0 by the verifier (no extractable answer), which injects a large penalty for a sample that may have been on a perfectly good trajectory — pure reward noise. DAPO's answer is a soft length penalty: define a length budget and a soft window before the hard cap, apply a graded penalty inside that window, and optionally mask the loss on fully-truncated samples so they contribute nothing rather than contributing a false negative.

Also worth naming: DAPO **removes the KL penalty entirely**, on the argument that long-CoT RL is supposed to move the policy far from the initial model, so a leash to the initial distribution is fighting the objective.

**📐 Numbers you must know:** DAPO reports reaching 50 points on AIME 2024 with a Qwen2.5-32B base using roughly half the training steps of the comparable R1-Zero-style baseline that scored 47. **📅 Volatile:** benchmark numbers move with harness and decoding settings — cite the shape of the result (comparable-or-better accuracy at ~50% of the steps, with an open-sourced system and dataset) rather than defending the exact digits in a room.

**🗣 Say this in the room:** "DAPO's four changes each fix a structural defect rather than tune a knob: clip-higher fixes the clip's anti-exploration asymmetry, dynamic sampling fixes zero-variance groups producing no gradient, token-level loss normalization fixes the length bias baked into the per-sequence denominator, and overlong shaping stops truncated generations from being scored as false negatives. Plus it drops KL, because in long-CoT RL you actively want to leave the reference distribution."

### Describe R1-Zero precisely — what was the setup, what was the result, and what broke?

R1-Zero is the load-bearing experiment of the reasoning-model era and you should be able to state it in four sentences without embellishment.

**Setup.** Start from the DeepSeek-V3 base model — a pretrained base, with **no supervised fine-tuning stage at all**. Apply GRPO directly, with a rule-based reward consisting of an accuracy component (is the final extracted answer correct, checked programmatically for math; do the tests pass, for code) and a format component (did the model put its reasoning inside `<think>` tags and its answer inside `<answer>` tags). No neural reward model. No process reward model. No MCTS.

**Result.** Reasoning performance improved dramatically over the run — the widely-cited figure is AIME 2024 pass@1 rising from roughly 15.6% to roughly 71.0%, and to ~86.7% with majority voting at 64 samples. **📅 Volatile:** verify these before quoting them. More importantly than the number: the model's average response length grew substantially over training *on its own*, with no length reward. It spontaneously started allocating more test-time computation to hard problems, and it started exhibiting reflection and re-verification behaviors — the widely-quoted "aha moment" where a checkpoint mid-training writes something like "wait, let me re-evaluate this step."

**What broke.** Two things, both stated plainly in the report: **poor readability** (the outputs were hard for humans to follow, with inconsistent structure) and **language mixing** (reasoning traces would switch between English and Chinese mid-trace). Neither is surprising given the reward function — nothing in accuracy-plus-format says anything about being readable or monolingual, so those properties drifted freely.

**Why this experiment mattered so much.** Before it, the assumption in the open community was that long chain-of-thought was something you had to *teach*, via SFT on distilled traces or via search-based data generation. R1-Zero demonstrated that a pure outcome-based RL signal on a strong enough base model is sufficient to elicit it. That reframed long CoT from "a data problem" to "a search-and-amplify problem," and it is why every open-weight reasoning effort since has been an RLVR effort.

**📄 Paper:** DeepSeek-AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — introduced both R1-Zero (pure RL from base, rule-based rewards) and R1 (a multi-stage pipeline fixing R1-Zero's readability and language-mixing problems).

**⚠ Trap:** saying "R1-Zero proved RL creates reasoning from scratch." It proved RL elicits it *from a very strong 600B-class base model*. Replications on small bases frequently produce much weaker versions of the effect or none at all, and the effect size varies enormously by base-model family — which is itself evidence for the elicitation reading rather than the creation reading.

### Recite the full DeepSeek-R1 pipeline. All the stages, in order, and why each exists.

Four stages. The structure is the answer, and the shape — SFT, RL, SFT, RL — is worth internalizing because it is now a template.

**Stage 1 — Cold-start SFT.** Collect a small set (thousands of examples) of long chain-of-thought data in a clean, readable format, and SFT the base model on it. Sources included few-shot prompting with long CoT exemplars, prompting a model to produce detailed answers with reflection, gathering readable R1-Zero outputs, and human post-processing. **Purpose:** give the RL a better starting point and, critically, *impose a readable output format up front* so the subsequent RL does not have to discover one and does not drift into language mixing. This is the direct fix for R1-Zero's two pathologies.

**Stage 2 — Reasoning-oriented RL.** GRPO with rule-based accuracy and format rewards on math, code, science and logic prompts — the R1-Zero recipe, but from the cold-start checkpoint. Added here: a **language-consistency reward**, computed as the proportion of target-language words in the CoT, to suppress language mixing. The report notes this slightly degrades benchmark accuracy but improves human preference, which is an honest and instructive tradeoff to be able to cite.

**Stage 3 — Rejection sampling + SFT.** Once stage 2 converges, use that checkpoint to generate a large SFT corpus: sample many completions per prompt, keep the ones that pass, filter for readability (drop mixed-language, drop overlong paragraphs, drop code-block clutter). Roughly 600k reasoning samples. Then **add ~200k non-reasoning samples** — writing, factual QA, self-cognition, translation — reused from the DeepSeek-V3 SFT pipeline. Fine-tune the *base* model on the combined ~800k. **Purpose:** this is the general-capability restoration step. Stage 2 produced a reasoning specialist that had drifted on everything else; stage 3 folds the reasoning gains back into a generalist.

**Stage 4 — RL for all scenarios.** A final RL stage combining rule-based rewards on the verifiable reasoning prompts with preference-model rewards (helpfulness, harmlessness) on the general prompts. **Purpose:** align the generalist for human preference without losing the reasoning, and produce the shipped model.

**Plus: distillation.** The ~800k corpus from stage 3 was used to SFT smaller Qwen and Llama models, producing the R1-distill family. The reported finding — that distilling from R1 outperformed running RL directly on the small models — is one of the most practically important results in the paper for an applied engineer, because it says the cheap path is usually the right path.

**🗣 Say this in the room:** "R1 is SFT → RL → SFT → RL. Cold-start SFT to fix format and readability, reasoning RL with rule-based rewards to build the capability, rejection-sampling SFT over ~800k samples — 600k reasoning plus 200k general — to fold the gains back into a generalist and undo the specialization damage, then a final RL stage mixing rule-based and preference rewards. The reason the third stage exists is the thing people miss: reasoning RL degrades everything you didn't reward, and you need an explicit restoration step."

**⚠ Trap:** describing this as "they did RL." The RL is two of four stages and neither is the largest. The 800k-sample rejection-sampling SFT stage is where most of the general-capability work happened, and it is the stage a startup can actually afford to reproduce.

### The "aha moment" — how much of that is real and how much is narrative?

Real mechanism, overclaimed framing, and the honest answer requires holding both.

**What is real.** Over an RLVR run with no length reward and no reflection reward, response length increases substantially and the frequency of self-correction phrasings — "wait," "let me re-check," "actually, that's wrong" — increases. This is measurable: tokenize the traces, count the markers, plot against step. It is not a cherry-picked sample; it is a distributional shift. And the causal story is clean: trajectories containing a re-check pass the verifier more often, so they get positive advantage, so they get amplified. Nothing mystical.

**What is overclaimed.** Three things.

First, the framing that the model "learned to reflect." It did not learn the behavior; it *reweighted* toward a behavior already in the base model's repertoire, because pretraining data is full of humans catching their own errors. The base model will produce "wait, let me reconsider" at some low rate if you sample it enough. RL made that rate high.

Second, the implication that this is unique to RL. Subsequent work has noted that reflection markers are already present in base models at nontrivial rates and that their frequency alone is a weak proxy for reasoning quality — a model can emit "wait, let me re-check" and then not actually re-check anything. Counting markers is not measuring reasoning; I would push back on any dashboard that treats it as one.

Third, the single dramatic transcript. A quoted "aha" trace is one sample. The distributional plot is the evidence. When you tell this story in an interview, tell it with the plot.

**A genuinely interesting and often-missed corollary:** the length increase is *adaptive*. The model allocates more tokens to harder problems. That is test-time compute scaling emerging from an outcome reward, and it is the thing that actually generalizes — it is the same phenomenon that makes reasoning-effort controls useful at inference time.

**⚠ Trap:** using "emergent" loosely. In an interview at a lab, "emergent" means "a capability that appears discontinuously with scale," and using it to mean "we didn't explicitly train for it" will get you a correction. Say "elicited without explicit supervision," which is both accurate and precise.

### Your reward curve is climbing beautifully and your held-out eval is flat. Debug it.

This is the most common RLVR incident and there is a fixed procedure. Work it in this order because the cheap checks eliminate the most probability mass.

**Step 1 — Confirm the eval is not broken.** Before anything else, re-run the *base* checkpoint through the eval harness. If the base scores differently than it did last week, your harness changed, your prompt template drifted, or your parser broke. I have burned two days on a "regression" that was a chat-template change. Twenty minutes, always first.

**Step 2 — Check for verifier hacking.** Pull 30 of the highest-reward training samples and read them yourself. Not summary statistics — read them. You are looking for: answers that pass the normalizer but are not actually correct (`"0.5000000001"` matching `1/2` under a loose tolerance), code that exploits the test harness rather than solving the problem, outputs that satisfy the format regex with garbage inside. If you find hacking, the reward is real and the capability is not, and the eval is telling you the truth.

**Step 3 — Check for contamination.** Compute n-gram overlap between the training prompt pool and the eval set. Also check the *reverse* direction people forget: did your prompt pool's gold answers leak into the prompts themselves? A prompt template that accidentally includes the answer in a "hint" field will produce a beautiful reward curve and zero transfer.

**Step 4 — Check train/eval distribution match.** Reward is measured on your training prompt distribution. If that pool is 80% algebra and the eval is 40% geometry, improving on the pool need not move the eval. Break the eval down per-category and see whether *anything* moved. Usually something did, and the aggregate was masking it.

**Step 5 — Check whether learning has actually stopped.** Look at entropy, degenerate-group fraction, and distinct-completions-per-group. If entropy is at the floor and 60% of groups are degenerate, your reward is rising because the surviving groups are the easy ones, not because the policy improved. The reward metric is conditioned on a shifting sample — it is not comparable across steps once dynamic filtering kicks in. **Always log reward on a fixed held-out prompt set as well as on the training batch.** This is the single most valuable instrumentation change you can make, and it costs one extra small generation batch per 50 steps.

**Step 6 — Check the decode settings.** Training samples at temperature 1.0; your eval might run at 0.6 or greedy. A policy sharpened by RL behaves very differently across that gap. Evaluate at the temperature you will serve at, and report the sampling config with the number.

**🔍 Failure taxonomy — the four causes, with their distinguishing signature:**
| Cause | Signature |
|---|---|
| Verifier hacking | High-reward samples are visibly wrong on manual read |
| Contamination | Eval moves on the contaminated slice only; clean slice flat |
| Distribution mismatch | Per-category eval shows movement in the trained categories |
| Learning stopped | Entropy floored, degenerate-group fraction high, held-out training reward flat while batch reward rises |

### Your run NaN'd at step 340 and the KL had been climbing for 60 steps. What happened and how do you prevent it?

Climbing KL followed by a NaN is almost always a **policy blow-up**: one or a few updates pushed the policy far from `π_old`, the importance ratio exploded, and `exp(logprobs − old_logprobs)` overflowed or produced an enormous gradient that wrecked the weights.

The proximate causes, in order of frequency in my experience:

**Advantage outliers from near-degenerate groups.** With std normalization and a group at p = 7/8, the single failure gets `A = −2.64`. With a bad reward function you can get worse. A large advantage times a ratio that is already >1 produces a big step, which produces a bigger ratio next inner step. Fix: clamp the std denominator, or drop std normalization, and clip advantages to something like ±5.

**Learning rate too high for the effective batch.** If dynamic sampling silently reduced your effective batch size, your gradient noise went up while your LR stayed constant. Fix: log effective batch size; scale LR with it, or hold effective batch constant by over-generating.

**Numerical issues in the ratio.** `exp()` of a large logprob difference. Fix: compute the ratio in log space and clamp the log-ratio *before* exponentiating — `torch.clamp(logprobs - old_logprobs, -20, 20)` costs nothing and turns an overflow into a merely-bad step.

**A single pathological sample.** A truncated generation, an empty completion (`mask.sum() == 0` → division by zero), a sample where the tokenizer produced something degenerate. Fix: guard `mask.sum().clamp(min=1)`, and drop zero-length completions before the loss.

**Optimizer state corruption after a bad step.** Even if you catch the NaN, Adam's `m` and `v` are now poisoned. Fix: skip the update when `grad_norm` is non-finite or exceeds a threshold — do not just clip it, *skip* it, and log the skip.

**The prevention checklist I put in every RL trainer:**

```
- clamp log-ratio to [-20, 20] before exp
- clip advantages to [-5, 5] after normalization
- global grad-norm clip at 1.0
- skip (do not clip) any step with non-finite grad norm; increment a counter
- alert if skipped-step rate > 1% over 50 steps
- checkpoint every 25 steps; keep the last 5 (recovery is worth the disk)
- log: grad_norm, clip_fraction, kl, entropy, effective_batch, adv_max_abs
```

**⚠ Trap:** responding to a rising KL by lowering the learning rate and restarting. Sometimes right, but the rising KL is a *symptom*, and the most common underlying cause is advantage outliers from degenerate groups, which a lower LR only delays. Diagnose with `adv_max_abs` and the group pass-rate histogram before you touch the LR. **The rule I enforce in review: any RL trainer that does not log `adv_max_abs` and `clip_fraction` is not reviewable.**

### How do length rewards get gamed, and what would you actually implement?

Length is the most gameable dimension in RLVR because it is the one thing the model has unilateral control over, and because both directions of pressure produce failure.

**If you reward length** (or if your loss normalization implicitly does, as sequence-level normalization does), the model pads. Not with useful reasoning — with restatements of the problem, enumerated "let me consider case 1... case 2..." scaffolding that goes nowhere, repeated verification of trivially-true steps, and eventually near-verbatim repetition. The reward goes up, tokens go up, accuracy is flat, and your serving cost per request triples.

**If you penalize length**, the model truncates its reasoning and accuracy drops — often sharply on the hard tail, because hard problems are precisely the ones that need the tokens. A naive linear penalty also creates a perverse incentive to *not attempt* hard problems, since a short wrong answer scores better than a long wrong answer.

**What I actually implement** — a budget with a soft shoulder, not a gradient:

```python
def length_shaped_reward(correct: bool, n_tok: int,
                         soft: int = 12_000, hard: int = 16_000) -> float:
    if n_tok >= hard:
        return 0.0                       # truncated: no extractable answer anyway
    base = 1.0 if correct else 0.0
    if n_tok <= soft:
        return base                      # inside budget: length is free
    # soft window: linear ramp of the penalty from 0 to -1 across [soft, hard)
    frac = (n_tok - soft) / (hard - soft)
    return base - frac
```

Key properties: **length is free inside the budget** (no pressure to pad, no pressure to truncate, which is the point), the penalty only engages near the cap where truncation is imminent, and the penalty applies to correct and incorrect answers alike so it does not distort the correctness signal's sign.

The separate decision is what to do with samples that hit the hard cap. Two defensible options: score them 0 (as above), or **mask them out of the loss entirely** so they contribute no gradient. Masking is cleaner when truncation is frequent, because a truncated trajectory that was on a good path is a false negative and you are injecting noise. I mask when the truncation rate exceeds ~5% and score-as-zero below that.

**💰 Math:** length inflation is a serving-cost bug, not just a training curiosity. If RL grows mean response length from 1,800 to 5,400 tokens for a flat accuracy, and you serve 500k requests/day at $15/Mtok output, you went from `500,000 × 1,800 / 1e6 × $15 = $13,500/day` to `$40,500/day` — **$810k/month of pure waste**. That is why I put mean-response-length on the same dashboard as reward, with an alert, and why "did accuracy per token improve" is the metric I actually report to a product team.

### Talk about format rewards and language-consistency rewards. What do they cost you?

Both are examples of the general principle: **every non-correctness term you add to the reward is a tax on capability that you pay in exchange for a property you need.** Be explicit about which side of that trade you are on.

**Format rewards** exist so you can parse the answer. `<think>...</think><answer>...</answer>` or a boxed final answer. Implement as a gate (fail → 0) rather than an additive bonus, so the model cannot collect reward for well-formatted nonsense. The cost is essentially zero once the model has learned it — within 50–100 steps the format compliance rate is >99% and the term is inert. The risk is if you make the format elaborate: every additional structural requirement is another thing that can fail, and a 6-field XML schema will cost you real accuracy because tokens spent on structure are tokens not spent on reasoning.

**Language-consistency rewards** exist because outcome-only RL will happily mix languages mid-trace if that is a higher-probability path through the base model's distribution. R1's implementation computes the proportion of target-language words in the CoT and adds it to the reward. The DeepSeek report is unusually honest here: this **slightly degrades reasoning benchmark performance** but improves human preference. That is exactly the right way to describe an alignment tax, and it is a great thing to cite when an interviewer asks how you would think about a tradeoff with no clean answer.

The general shape of the trade, stated as a rule I use:

- If the property is needed for the *system to function* (parseable answer, valid JSON, tool call in the right schema), make it a **hard gate**. Non-negotiable, no partial credit, cost accepted.
- If the property is needed for *human acceptability* (language consistency, readability, tone), prefer to handle it in **SFT, before the RL** — which is precisely why R1 has a cold-start stage. A format learned in SFT is a prior the RL inherits for free; a format learned via reward is a term competing with correctness for the whole run.
- If the property is *nice to have*, leave it out and measure it as an eval instead. You can always fix it in a post-RL SFT pass.

**🗣 Say this in the room:** "My default is that anything the system needs to parse becomes a hard gate in the reward, and anything about style or readability gets handled in the cold-start SFT rather than as a reward term. R1 is the canonical example: they added a language-consistency reward, and they report it costs a little benchmark accuracy for a gain in human preference. Reward terms compete with correctness for the entire run; SFT-imposed formats are free priors."

### What does the difficulty distribution of your prompt pool do to the run, and how would you build a curriculum?

The pool's difficulty distribution determines what fraction of your compute produces gradient, and because the policy improves, that distribution is a moving target you must actively steer.

**The static picture.** With group size G and per-prompt pass probability p, useful gradient requires reward variance in the group, which requires p away from 0 and 1. The expected number of non-degenerate groups per batch is `N · (1 − p^G − (1−p)^G)` — so a pool centred at p=0.5 gives you ~99% useful groups at G=8, and a pool centred at p=0.9 gives you 57%.

**The dynamic picture.** Training moves p up. A pool that started centred at 0.5 drifts toward 0.8 over a few hundred steps, and your useful-gradient fraction slides from 99% to 83% to 57%. Without intervention, the run decays on its own even if nothing is wrong.

**The curriculum I build**, which is deliberately simple because elaborate curricula are hard to debug:

1. **Pre-score once with the base model**, n=16 per prompt. Bucket: dead (p=0 or p=1), hard (0 < p ≤ 0.3), medium (0.3 < p ≤ 0.7), easy (0.7 < p < 1).
2. **Drop dead prompts** from the training mix permanently. Keep them in a "graveyard" file — some become hard prompts again after the policy changes, and re-scoring the graveyard every few hundred steps occasionally resurrects useful material.
3. **Sample the batch as a mixture**, roughly 20% easy / 50% medium / 30% hard at the start, shifting toward hard over the run.
4. **Re-score continuously for free.** Every rollout you generate is a difficulty measurement. Maintain an exponential moving average of pass rate per prompt from the live rollouts and re-bucket on the fly. No extra compute.
5. **Retire prompts** whose EMA pass rate exceeds 0.9 for 3 consecutive appearances.

**⚠ Trap:** building an aggressive curriculum that trains *only* on prompts at the frontier. You get faster initial progress and then a model that has catastrophically forgotten the easy cases, because nothing has reinforced them in 400 steps. Keep a floor of ~15–20% easy prompts in every batch as a retention mix. This is the same instinct as keeping a replay buffer of old tasks in continual learning, and the failure mode if you skip it is exactly the same: a model that is better at the hard tail and worse at the median, which is usually a net product regression.

**📐 Numbers you must know:** the arithmetic that justifies all of this — at G=8, a prompt at p=0.5 contributes useful gradient 99.2% of the time; at p=0.9, 57%; at p=0.97, 22%. Difficulty filtering is not a refinement, it is a 4× swing in gradient-per-dollar.
