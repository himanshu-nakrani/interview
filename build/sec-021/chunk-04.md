### You're building the preference dataset for an enterprise legal assistant from scratch. Take me from zero to a training set.

I would fight the instinct to start collecting, because the first artifact is not a training set.

**Step 0 — build the eval set first, and never let it touch training.** 300–500 prompts sampled from real traffic, stratified by task type, with the *reference* answers written or adjudicated by domain experts and triple-labeled. This is the most expensive per-row data you will buy and the only data whose noise you cannot recover from: if your eval is noisy, you cannot detect any improvement, and every downstream decision is a coin flip. At $120/hour for a lawyer and ~10 minutes per adjudicated item, 400 items is `400 × (10/60) × 120 = $8,000`. That is the cheapest $8k in the project.

**Step 1 — get the prompts right, they are the expensive half.** Preference pairs are cheap to generate and expensive to *prompt well*. Sources, in order of value: (a) real production traffic, deduplicated and PII-scrubbed, which is the only source with the true difficulty distribution; (b) the tail of your support tickets and escalations, which is where the model is failing; (c) expert-written prompts covering task types you know are coming but have no traffic for yet. I stratify by task type (contract clause extraction, redline explanation, precedent summary, drafting), by jurisdiction, and by difficulty, and I hold the strata fixed across rounds so I can attribute a regression to a slice.

**Step 2 — generate both sides from your own model.** This is the on-policy requirement. For each prompt, sample `n = 4` completions from the current serving checkpoint at temperature 0.9. Do not take chosen from a frontier API and rejected from your model — that dataset teaches "which model wrote this," and a shallow classifier will confirm it.

**Step 3 — label, in a cascade.** Full expert labeling of 60k pairs at $120/hour is `60,000 × (90/3600) × 120 = $180,000`, which nobody approves. So: (a) auto-filter with a verifier wherever one exists — a cited statute must resolve, a quoted clause must appear verbatim in the source document, a date must parse; these are deterministic checks and they resolve a surprising share of pairs for free; (b) LLM-judge the rest with position randomization and a rubric derived from the expert eval set; (c) route only the *disagreements* and the low-confidence judgments to experts. **💰 Math:** if the verifier resolves 25% and the judge confidently resolves 60% of the remainder, experts see `60,000 × 0.75 × 0.40 = 18,000` pairs → `18,000 × 0.025 h × $120 = $54,000`. Still large, but a third of the naive cost, and you can cut further by capping expert review at the pairs the current model's implicit reward gets *backwards*.

**Step 4 — validate the judge against the experts before you trust it.** On the 400-item eval set, measure judge-vs-expert agreement. If it is below ~75%, the judge rubric is wrong and everything built on it is noise. This gate is non-negotiable and it is what separates a real pipeline from a demo.

**Step 5 — clean.** Dedup, contamination-check against the eval set, length-stratify, drop ties, drop truncated completions, drop pairs the current model already ranks correctly with a wide margin. Details in the next question.

**Step 6 — stamp and version.** Every row carries: prompt source, generating model version, labeler (verifier / judge model+version / expert ID), label timestamp, and confidence. Preference data is not a static asset; it is a cache with a TTL tied to model generation.

**⚠ Trap:** collecting preference data before you have a stable definition of "better." In a legal assistant, "better" is genuinely multi-dimensional — factual precision, appropriate hedging, citation completeness, tone for the audience. Collapsing that to a single pairwise preference produces annotator disagreement that looks like noise and is actually a specification failure. Either write a rubric with an explicit priority ordering ("factual precision dominates; among factually equal answers, prefer the one with complete citations; among those, prefer the shorter"), or collect per-axis ratings and train per-axis. A single scalar is a modelling choice, not a neutral one.

### Be specific about dedup, difficulty balance and contamination control on preference pairs.

Four passes, and I run them in this order because each one changes what the next one sees.

**1. Prompt-level dedup.** Exact match first, then normalized (lowercase, whitespace, strip boilerplate), then near-duplicate via MinHash-LSH on character shingles at a Jaccard threshold around 0.8, then semantic near-dup via embeddings with a cosine threshold you *tune by inspecting the boundary* rather than picking 0.95 off a blog post. Production traffic is brutally duplicated — a real assistant's top 100 prompt templates can be 30–50% of volume — and if you skip this, DPO spends most of its gradient on your most common query and overfits it. **The rule I enforce: cap the number of pairs per near-duplicate prompt cluster at something like 3–5**, so no single template can dominate.

**2. Response-level dedup and degeneracy filtering within a pair.** Drop pairs where chosen and rejected are near-identical (normalized edit distance below ~0.05) — they carry no signal and, per the shared-token suppression mechanism, they actively damage the model. Drop pairs where either side is truncated at `max_length`, empty, or a bare refusal (unless refusal is what you are training). Drop pairs where the rejected side is degenerate garbage, because "don't emit garbage" is not a lesson your instruct checkpoint needs and it inflates `rewards/accuracies` while teaching nothing.

**3. Difficulty balance, measured not guessed.** Score every pair with the current model's implicit reward `β log(π/π_ref)` and bucket by margin. Wide-positive margin = already solved, contributes ~zero gradient (recall the σ weighting), keep only a small fraction as an anchor. Near-zero margin = the frontier, this is where the learning is, oversample it. Wide-*negative* margin = the model is confidently wrong, which is either the highest-value signal or a mislabeled row — send to human adjudication, never straight to training. A dataset that is 70% already-solved pairs is a dataset where 70% of your GPU-hours produce no gradient.

**4. Contamination against every eval you will report.** N-gram overlap (13-gram is the conventional threshold) between your preference prompts and: your held-out eval set, your capability suite (MMLU, GSM8K, HumanEval, IFEval), and any public benchmark you plan to quote. Then embedding near-dup at a low threshold as a second net, because paraphrased contamination survives n-gram checks. This matters more for preference data than people expect, because a large fraction of public preference sets were assembled from instruction corpora that themselves drew from benchmark-adjacent sources — there is a documented history of widely-used binarized preference sets containing prompts that overlap eval suites, and of binarization bugs where the "chosen" response was selected by the wrong score field and later corrected in cleaned re-releases. Treat any public preference set as untrusted input: run your own contamination scan and your own sanity read of 100 random rows before it enters a run.

**⚠ Trap:** running dedup *after* forming pairs rather than on the prompt pool. If you generate 4 completions each for two near-duplicate prompts and then dedup pairs, you keep both clusters because the pairs differ. Dedup at the prompt level, then generate.

**🏋 Drill:** thirty minutes on any public preference dataset. Compute and report: (a) mean chosen/rejected length ratio, (b) fraction of pairs where chosen is simply longer, (c) accuracy of a logistic regression predicting chosen from `[length, markdown-token count]` alone, (d) the top-10 near-duplicate prompt clusters by size. Pass criterion: you can state in one sentence what the dataset is actually measuring. On most public sets the honest answer is "length and formatting."

### You've sampled 8 completions per prompt and have a scoring function. How do you form the pairs?

The naive answer is best-versus-worst, and it is *a* right answer, but which pairs you form is a real design decision with three axes.

**Axis 1 — margin width.** Best-vs-worst maximizes the score gap, which maximizes learning signal per pair and also maximizes *surface separability*: the worst of 8 samples is often malformed, and a pair whose distinguishing feature is "one of these is broken" teaches nothing your model does not know. Best-vs-median produces a much finer distinction — the exact distinction you care about at the frontier of the model's ability — but with a lower signal-to-noise ratio per pair, because your scorer is least reliable when the two candidates are close. **What I ship: both.** One best-vs-worst pair for gross signal and one best-vs-median for the fine boundary, per prompt, with the fine pair weighted the same. If I can only afford one, and my scorer is a validated verifier rather than a judge, I take best-vs-median; if my scorer is a noisy judge, I take best-vs-worst because it is the pair the noisy scorer can actually order correctly.

**Axis 2 — how many pairs per prompt.** From 8 samples you can form 28 pairs. Do not. They are massively correlated — the same 8 sequences appearing in 28 rows — and using all of them silently upweights that prompt 28× relative to a prompt where you sampled twice. Cap at 1–2 pairs per prompt and spend the remaining budget on more prompts. Prompt diversity beats pair count at essentially every budget I have seen.

**Axis 3 — length control at pair-formation time.** This is the cheapest place to kill length bias and almost nobody does it. When choosing the rejected sample, prefer the one whose length is *closest to the chosen*, among the low-scoring candidates. You are deliberately discarding the pair that would have been easiest to learn from surface features, and you are constructing a dataset where length carries no information. It costs you nothing but a sort.

**On ties:** if your scorer says two completions are within noise, do not force a pair. A forced pair on a genuine tie is a label with ε ≈ 0.5, which is pure gradient noise, and it is worse than dropping the row because it consumes batch capacity. Measure your scorer's noise floor (score the same pair twice with position swapped; the disagreement rate is your floor) and require the score gap to exceed it.

**💰 Math on the generation budget.** 20,000 prompts × 8 samples × 400 tokens = `6.4e7` decode tokens. On one H100 serving an 8B with continuous batching at ~2,500 tok/s aggregate, that is `6.4e7 / 2500 = 25,600 s ≈ 7.1 GPU-hours ≈ $21` at $3/GPU-hour. Judging with an LLM at roughly 1,200 input + 150 output tokens per comparison, at $3/Mtok in and $15/Mtok out, is `(1200×3 + 150×15)/1e6 = (3600 + 2250)/1e6 = $0.00585` per judgment; two comparisons per prompt, each run twice with positions swapped, is `20,000 × 2 × 2 = 80,000` judge calls → `80,000 × $0.00585 = $468`. **So the full generate-and-label loop for a 20k-prompt, 40k-pair on-policy dataset is under $500 and about a day.** That number is the argument for regenerating your data every round instead of reusing last quarter's — say it in the room, because most people assume on-policy data is prohibitively expensive and it is not.

### How do you evaluate an aligned model? I'll tell you now that "win rate went up" is a failed answer.

Right, because win rate measures one thing — whether a judge prefers your outputs on a prompt distribution — and alignment can improve that while destroying capabilities the prompt distribution does not exercise. The alignment tax is real, it is usually invisible on a chat eval, and reporting a single number is how it ships.

I report a **four-panel** result and I would draw the four panels on the whiteboard.

**Panel 1 — Preference quality.** Generative win rate against the previous production checkpoint (not against the SFT baseline forever — against what you are replacing), on a held-out prompt set the training data never touched, judged with position randomization, **length-controlled**, with confidence intervals. Report both raw and length-controlled; a large gap between them is itself the finding.

**Panel 2 — Capability regression.** A fixed suite run at every checkpoint, chosen to cover what your product actually needs: instruction-following (IFEval-style constraint satisfaction), reasoning and math (GSM8K or equivalent), code (HumanEval or a repo-relevant harness), factuality/hallucination on a domain set, multilingual if you serve it, and long-context retrieval if you use it. The acceptance criterion is stated *before* the run — for example "no metric may drop more than 1.5 points absolute, and the sum of drops may not exceed 3 points." A regression budget you write down in advance is the difference between engineering and rationalization.

**Panel 3 — Behavioral safety and calibration, as pairs of numbers.** Refusal rate on genuinely unsafe prompts (want high) *and* refusal rate on safe-but-unsafe-looking prompts, XSTest-style (want low). Sycophancy flip rate under user pushback. Structured-output / format compliance rate, which is the one that silently breaks agent pipelines and never appears on a chat leaderboard.

**Panel 4 — Distributional health, computed on the model's own samples.** Mean and p90 output length versus baseline. Diversity (distinct-n or self-BLEU across samples) to catch mode collapse. Repetition rate. Mean per-token entropy by position. These are the metrics that catch the failures the other three panels are blind to, and they cost one generation pass.

**🗣 Say this in the room:** "I report four panels and I set the acceptance thresholds before the run. Win rate, length-controlled and with a confidence interval. A fixed capability suite with a written regression budget. Safety as two numbers — unsafe-refusal and over-refusal — never one. And distributional health on the model's own samples: length, diversity, repetition, entropy by position. A DPO checkpoint that wins on panel one and regresses two points of instruction-following does not ship, and the reason I know that is that panel two is a gate, not a report."

**⚠ Trap:** evaluating with the same judge model that generated your preference labels. That is training on the test set with extra steps: you have optimized the policy against judge J's biases, and then you measure with judge J. Use a different judge family for evaluation than for labeling, and anchor both to a small human-labeled set.

### What is a length-controlled win rate actually doing, and what does it not fix?

**Mechanism.** LLM judges have a robust, large and well-documented preference for longer responses. So a raw win rate confounds "the model got better" with "the model got longer," and since DPO's objective mechanically encourages length, the confound is not small — it can account for most of an apparent gain. Length-controlled evaluation fits a statistical model of the judge's preference as a function of both the quality signal and the length difference between the two responses, then reports the win rate *with the length term held fixed* — the win rate you would have observed if both models produced the same length. It is a regression adjustment, exactly the shape of controlling for a covariate in any A/B analysis.

**📄 Paper:** Dubois et al. (2024), *Length-Controlled AlpacaEval* — introduces the regression-based debiasing of automatic win rates and reports substantially improved correlation with human Chatbot Arena rankings compared to the raw metric, while being far cheaper to run.

**What it fixes:** the specific confound where a model wins because it wrote more. It makes DPO-vs-SimPO and DPO-vs-baseline comparisons interpretable at all, and it removes the most exploitable axis for a model that has learned to game a judge.

**What it does not fix, and this is the part to volunteer:**

- **Other judge biases.** Position bias (mitigate with randomization and by requiring consistency under swap), self-preference (judges rate their own family's outputs higher — use a different judge family than your generator), formatting bias (markdown, headers, bullet lists), and confidence/assertiveness bias. Length is simply the bias somebody built a correction for; it is not the only one.
- **The judge's ceiling.** If the judge cannot tell whether a legal citation is real, length control does not help — you have a metric that is blind to the failure mode you most care about. Verifiers beat judges wherever a verifier exists.
- **Prompt distribution.** A length-controlled win rate on a general chat set says nothing about your production traffic. This is the largest error I see: teams tune on a public chat benchmark and ship to a product whose prompt distribution looks nothing like it.
- **The alignment tax.** It is a preference metric. It will happily report a big win for a model that lost two points of instruction-following.

**⚠ Trap:** using length control as a substitute for fixing the data. If your pairs have a 1.6× chosen/rejected length ratio, the model is *learning* length as a strategy and length control merely stops you from seeing the reward for it. The model still generates 520 tokens in production and you still pay for them. Fix the data; use length control to verify you fixed it.

### I ran 500 comparisons and got a 55% win rate. Ship it?

No, and I would want to show the arithmetic rather than assert it.

Treat each comparison as a Bernoulli trial (ties counted as 0.5, or excluded with the count adjusted). The standard error of the estimate is `√(p(1−p)/n) = √(0.55 × 0.45 / 500) = √(0.0004950) = 0.0223`. The 95% confidence interval is `0.55 ± 1.96 × 0.0223 = 0.55 ± 0.0437`, i.e. **[50.6%, 59.4%]**. It barely excludes 50%, and that is *before* accounting for judge noise and prompt-set selection, both of which are real and neither of which is in that interval. A single-digit win-rate lead at n = 500 is not a result; it is a coin that landed slightly to one side.

**Sample size, done properly.** To detect a 5-point lift (50% → 55%) at α = 0.05 two-sided with 80% power: `n ≈ (z_{α/2} + z_β)² · p(1−p) / δ² = (1.96 + 0.84)² × 0.25 / 0.05² = 7.84 × 0.25 / 0.0025 = 784`. So ~**800 comparisons** for a 5-point claim. For a 2-point claim: `7.84 × 0.25 / 0.0004 = 4,900` — about **5,000 comparisons**. Internalize that ratio: halving the detectable effect quadruples the sample.

**📐 Numbers you must know:** ~800 comparisons for a 5-point win-rate claim, ~5,000 for a 2-point claim, at 95%/80%. Derived from `n = 7.84·p(1−p)/δ²` with p ≈ 0.5. This single formula settles most "is our eval big enough" arguments, and it is worth having the two anchor numbers memorized.

**Variance reduction, which is cheaper than more samples.** Use the *same* prompts for both models and analyze paired — a paired bootstrap over prompts, resampling prompts rather than comparisons, which also correctly propagates the fact that prompts (not judgments) are your independent unit. Run each comparison twice with the positions swapped and count only the *consistent* judgments; the position-inconsistency rate is simultaneously a free measure of your judge's reliability and a filter that removes its noisiest decisions. Stratify by task type and report per-slice, because an aggregate 55% that is 70% on one slice and 45% on another is a routing decision, not a model decision.

**💰 Math — this is affordable, so there is no excuse.** 800 comparisons, position-swapped (1,600 judgments), at ~1,200 input and ~150 output tokens each, priced at $3/Mtok in and $15/Mtok out: `1,600 × (1200×3 + 150×15)/1e6 = 1,600 × 0.00585 = $9.36`. Generating the 1,600 responses to compare is the larger cost and is still trivial on your own hardware. **A statistically defensible eval costs under ten dollars of judge tokens.** When a team tells me they only ran 200 comparisons because eval is expensive, they have not done this arithmetic.

**⚠ Trap:** running the eval, seeing 55% at n = 500, adding 300 more comparisons, checking again, and stopping when it crosses significance. That is optional stopping and it inflates your false positive rate substantially. Fix `n` before you look, or use a sequential test designed for it.

### Give me your decision framework: when is DPO enough, and when do you actually need RL?

I answer this with preconditions, because the honest answer is that RL is a capability *and* a liability and most teams asking the question do not meet the preconditions.

**DPO-family is enough when all of these hold:**

1. **You are shaping style, format, tone, refusal boundaries, or a preference ordering over already-achievable outputs.** DPO reweights mass the model can already produce. It is a rotation of the output distribution, not an expansion of it.
2. **You have — or can cheaply produce — pairwise or binary labels on completions from your own current model.** Say 10k–100k pairs.
3. **You do not have a cheap, reliable, automatic verifier.** If you had one, you would be in RL territory.
4. **You need this shipped in weeks with one or two engineers and no dedicated training-infra investment.**
5. **The horizon is one turn.** The thing you are improving is a single response, not a multi-step trajectory.

**You genuinely need RL (RLVR/GRPO-style) when:**

1. **You have an automatic verifier that is cheap, deterministic and hard to game.** Tests pass, code compiles, SQL returns the right rows, the theorem checks, the extracted field matches ground truth, the tool call succeeds. This is the single dominant precondition — without it you are back to a learned reward model and most of RL's advantage evaporates.
2. **You need capability at the frontier of what the model can do, discovered by search.** RL samples, gets scored, and reinforces what worked. That lets it find behaviors that appear in no dataset you own. DPO cannot; it can only reweight what you showed it.
3. **The task is multi-step and the credit assignment matters** — an agent trajectory where the reward arrives at the end.
4. **You can absorb the infrastructure.** Rollout throughput, weight sync between trainer and sampler, sandboxed execution for the verifier, environment reset semantics, flakiness. This is a team and a quarter, not an engineer and a sprint.
5. **You have a large, non-contaminated prompt pool.** RL burns prompts far faster than DPO.

**And the middle path I recommend most often:** **rejection-sampling fine-tuning.** Sample n, keep what the verifier accepts, SFT on it. It captures a large share of RL's benefit — it is on-policy, it uses a verifier, it expands the distribution toward what works — on your existing SFT stack, with none of the RL orchestration. If a team tells me they want to do RL, my first question is whether they have run best-of-n with their verifier and fine-tuned on the winners, because that is the cheaper experiment that tells you whether the verifier is any good, which is the thing that determines whether RL will work at all.

**🗣 Say this in the room:** "The precondition that decides it is whether I have a cheap automatic verifier. If I do, RL — or at minimum rejection-sampling fine-tuning, which I would run first because it tests the verifier for a fraction of the cost. If I don't, DPO or KTO, because a learned preference signal in an offline objective is where the cost/benefit lands. And I'd add that DPO reweights what the model can already do while RL can discover new behavior through search — so if the ask is 'be more concise and cite properly,' that's DPO. If the ask is 'solve problems it currently fails,' DPO will not get there no matter how much preference data I buy."

### Price DPO against PPO and GRPO for me — GPU-hours and engineer-weeks, not adjectives.

Take an 8B policy, 60k prompts, and show the arithmetic. Use `C ≈ 6ND` FLOPs per token for forward+backward and `2ND` for a forward pass.

**DPO.** Per pair: policy fwd+bwd on chosen and rejected, plus reference fwd on both. Average 600 tokens per side → 1,200 policy tokens and 1,200 reference tokens per pair. Policy: `6 × 8e9 × 1,200 = 5.76e13` FLOPs/pair. Reference: `2 × 8e9 × 1,200 = 1.92e13`. Total `7.68e13`/pair × 60,000 = **`4.6e18` FLOPs per epoch.** An H100 at ~989 TFLOP/s bf16 peak running at a realistic 40% MFU gives `3.96e14` FLOP/s → `4.6e18 / 3.96e14 = 11,600 s ≈ 3.2 H100-hours per epoch`. One epoch is the standard recipe, so ~3–4 GPU-hours; budget 40 GPU-hours for the real project including failed runs and evals → **~$120 at $3/GPU-hour**. Memory: policy training state ~128 GB + reference 16 GB ≈ 144 GB → 2–4 H100s with FSDP. **Engineer-weeks: 1–2**, and almost all of it is data.

**PPO.** Four models: policy (trainable, ~128 GB of state), reference (16 GB), reward model (16 GB), value model (trainable, another ~128 GB if it is a full model). **~290 GB** → roughly double the GPUs before you have generated a single token. And you must *first* train the reward model, which is its own dataset, its own run, and its own eval. The dominant compute is rollout generation, which is autoregressive decode and therefore bandwidth-bound: generating 60,000 prompts × 512 tokens = `3.07e7` tokens per epoch at a realistic 2,500 tok/s per GPU is `12,300 s ≈ 3.4 GPU-hours` *per epoch of rollouts*, and PPO needs many epochs over many rollout batches, plus the RM forward pass on every sample, plus training two networks instead of one. In practice budget **5–20× the DPO compute** depending on rollouts per prompt and PPO epochs, and note that the wall-clock penalty is worse than the FLOP penalty because decode utilizes the GPU poorly. **Engineer-weeks: 6–12**, including the reward-model pipeline, rollout throughput work, and the hyperparameter sensitivity that is PPO's real cost.

**GRPO.** Drops the value model — advantages come from normalizing rewards within a group of `G` sampled completions per prompt — so memory is policy + reference + verifier, closer to DPO's footprint than PPO's. But compute goes the other way: you generate `G` (typically 8–16) completions per prompt instead of one. At G = 8 that is `8 × 3.07e7 = 2.46e8` rollout tokens per epoch → `2.46e8 / 2500 = 98,000 s ≈ 27 GPU-hours per epoch` of generation alone, roughly 8× DPO's total. **Engineer-weeks: 4–8, and the distribution is different** — the engineering is not in the RL algorithm, it is in the verifier and the sandbox. Building a deterministic, fast, hard-to-game grader with proper answer normalization *is* the project.

**💰 The summary I would put on the whiteboard:** DPO ≈ 4 GPU-hours and 1–2 engineer-weeks; GRPO ≈ 30+ GPU-hours per epoch and 4–8 engineer-weeks mostly spent on the verifier; PPO ≈ 5–20× DPO compute, 2× the memory, and 6–12 engineer-weeks including the reward model. **📅 Volatile:** GPU pricing and per-GPU throughput move constantly; the *ratios* are the durable part, the dollar figures are not. Re-derive with current numbers before you quote them.

**🗣 Say this in the room:** "The compute difference is real but it is not the reason PPO lost operationally — four GPU-hours versus forty is not what kills a project. What kills it is that PPO is four models, a rollout server, weight synchronization, and a hyperparameter surface where a bad KL coefficient produces a plausible-looking run that is quietly collapsing. DPO is a supervised job on the stack you already have. That is the whole story, and it is why the field defaulted to direct alignment for preference-shaped problems and reserved RL for verifiable ones."

### Design the preference-data flywheel for a Cursor-style code assistant. This is the system, not the algorithm.

**The framing I lead with: in a coding product you have something almost nobody else has — implicit preference signal at enormous volume, plus an actual verifier.** That changes the design completely, and the algorithm choice falls out of it rather than driving it.

**Signals, ranked by strength.**
- *Strong positive:* the suggestion was accepted **and survived** — still present, largely unmodified, five minutes later or at the next commit. Immediate acceptance is a weak signal; survival is a strong one, because developers accept and then delete constantly.
- *Strong positive, verifiable:* the resulting file compiles / typechecks / the touched tests pass. This is a genuine verifier and it should gate everything.
- *Strong negative:* accepted then reverted within 30 seconds, or undone, or immediately rewritten in the same region.
- *Weak negative:* shown and ignored — heavily confounded by the developer not looking, so I would use it only for downsampling, never as a hard label.
- *Explicit:* thumbs, "regenerate," chat follow-ups like "that's wrong." Low volume, high precision.

**Volume, and why this argues for KTO.** 100,000 daily active developers × 200 completions shown/day = `2e7` events/day. At a 30% acceptance rate that is `6e6` accepted and `1.4e7` not — orders of magnitude more binary signal than anyone could ever produce as pairwise comparisons. These are *per-completion binary labels*, not pairs, which is exactly KTO's data shape. The alternative — synthesizing pairs by matching an accepted completion with a rejected one from a *different* context — teaches context differences, not quality. Do not do it. Where you do want pairs, form them properly: for the same context prefix, sample n completions from the current model offline and rank them with the verifier.

**Pipeline.** Client emits an event with a context hash, model version, suggestion, and outcome, never raw code by default. A per-tenant policy gate decides whether that tenant's data can be used for training at all — this is the constraint that dominates the design in an enterprise code product, and I would raise it before anyone asks. For opted-in tenants, an offline job reconstructs `(context, completion, label)`, runs secret scanning and license filtering, deduplicates aggressively by repository and by context near-duplicate (a monorepo's boilerplate will otherwise be 40% of your dataset), stamps model version and timestamp, and lands it in a versioned dataset with a TTL.

**Training.** KTO on the binary signal, from the current serving checkpoint, with `λ_D`/`λ_U` set from the actual class ratio. Verifier-passing completions get promoted to a higher-confidence tier. Rejection-sampling fine-tuning on verifier-passing samples runs in parallel as the cheaper capability lever. Weekly or biweekly cadence, with every round's data regenerated against the current checkpoint.

**Evaluation, which is the hard part.** Offline: a held-out set of repositories the model never trained on, with completion acceptance predicted by the verifier, plus a capability suite so a coding-tuned model does not lose its ability to explain code in English. Online: a real A/B on acceptance rate, survival rate, and — the metric that actually matters commercially — **time-to-merged-PR**, which is the only one that cannot be gamed by producing more, shorter, easier-to-accept suggestions.

**🔍 Failure taxonomy for this flywheel.** (1) *Acceptance-rate gaming*: the model learns to emit shorter, more obvious completions that get accepted more and help less. Detect via survival rate and characters-accepted-per-session, not acceptance rate. (2) *Popularity collapse*: the flywheel amplifies whatever the model already does well because those get accepted, and starves the tail — detect by tracking per-language and per-framework acceptance rates as separate slices, and by holding a fixed prompt set constant across rounds. (3) *Contamination*: your eval repositories leak into training via forks and vendored copies; dedup on content hash, not repo name. (4) *Staleness*: signals collected against v1 describe v1's failures, so quarantine anything older than one model generation. (5) *Privacy and IP*: a training set assembled from customer code is a legal artifact — per-tenant opt-in, deletion propagation into dataset versions, and the ability to retrain without a given tenant are requirements, not features.

### Before I let you burn a GPU-week on a DPO run, how do you convince me the implementation is correct?

Eight checks, all of which run in under twenty minutes on a laptop or one GPU, and I would refuse to launch without them. This is exactly the discipline of not deploying a migration without running it against a snapshot first.

**1. Step-zero identity.** With `π_θ = π_ref` at initialization, `logits = 0`, so the loss must be exactly `−log σ(0) = log 2 = 0.6931`, `rewards/chosen = 0.0`, `rewards/rejected = 0.0`, and `rewards/accuracies = 0.5`. Any deviation means the policy and reference are not the same model, or dropout is on in the reference, or your masking differs between the two paths. This single check catches most integration bugs.

**2. Label-swap sanity.** Swap the chosen and rejected columns and run 50 steps. `rewards/margins` must go the other way. If your metrics look substantially the same either way, the labels are not actually wired into the loss — which sounds impossible and is a bug I have seen twice.

**3. Overfit eight pairs.** Take 8 pairs, no shuffling, high LR, 200 steps. Loss must approach zero and accuracy must hit 1.0. If it cannot overfit 8 examples, the gradient path is broken and no amount of data will fix it.

**4. Frozen-reference assertion.** Assert in code, not in your head: `all(not p.requires_grad for p in ref.parameters())` and `ref.training is False`. Then assert the reference's parameter checksum is unchanged after 100 steps. A reference model that is accidentally training makes the loss trivially satisfiable and the run meaningless.

**5. Cross-check against the library.** Compute your hand-rolled loss and TRL's on the identical batch and assert `torch.allclose` to 1e-5. If you are using the library directly, do the reverse — hand-compute one batch and check the library agrees, because it validates your understanding of its masking conventions.

**6. Read the masks.** Decode 20 training examples token by token and print `(token, label)` pairs. Verify: prompt tokens are `-100`, padding is `-100`, the completion is unmasked, and the EOS token is *inside* the unmasked span on both sides. Truncated completions are a training-set poison and this is where you see them.

**7. Zero-LR null run.** One step at `lr=0`; assert every policy parameter is bitwise unchanged and the logged metrics are identical. This separates "the model changed" from "my metrics are nondeterministic," which matters when you are later trying to attribute a regression.

**8. One-hour smoke run with the full eval.** 500 steps on a 2k-pair subset, then run the entire four-panel evaluation. The point is not the result; the point is that the *eval harness* works end to end before you have a checkpoint you care about. Discovering your capability suite has a broken tokenizer path after a week-long run is a real and stupid way to lose a week.

**🏋 Drill:** ninety minutes, unaided. Implement DPO from scratch on a small model (a 0.5B–1B instruct checkpoint is plenty), on 2,000 pairs from any public preference set, and produce: the eight checks above all green, a training curve with all six metrics, and a length-controlled win rate against the starting checkpoint with a bootstrap confidence interval. **Pass criterion:** the step-zero loss is `0.6931`, the label-swap check flips the margin, and you can state — with numbers — whether your model improved or merely got longer. If you can do this once, you can answer essentially any DPO question in a loop, because every one of them is a question about something in this pipeline.

**🗣 Say this in the room** if you are asked what you would do first on a new alignment project: "Before any training, I build the eval and I verify the loss. Step-zero loss must be exactly log 2 with zero rewards on both sides; the label-swap must flip the margin; the run must be able to overfit eight pairs. Those three take twenty minutes and they catch the bugs that otherwise look like a bad result and get debugged as a hyperparameter problem for a week."
