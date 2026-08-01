### Your fine-tuned model scores 94% on your internal eval and 71% in production. How do you check for contamination?

Contamination against your *own* eval set is the version of this problem that will actually bite you, and it's more insidious than benchmark contamination because nobody is watching for it. The mechanism is mundane: your training data and your eval data came from overlapping sources — the same document corpus, the same log window, the same synthetic generator with the same seeds — so eval examples or their near-twins are in the training set, and your eval measures memorization.

The detection procedure, run as a mandatory pipeline stage:

**1. Exact-match on normalized prompts.** Lowercase, collapse whitespace, strip punctuation, hash. Any hit is an immediate hard fail. You'd think this never fires. It fires constantly, usually because someone built the eval set by sampling from the same dump.

**2. N-gram overlap.** The convention established by large pretraining work is a 13-gram check: an eval example is flagged if any 13-gram from it appears in the training set. The exact `n` is a knob — smaller n is more sensitive and more false-positive-prone, and for short instructions n=13 may be longer than the whole prompt, so I use n=8 for instructions and n=13 for long-form. Implement with a hash set over training n-grams; it's one pass and it's fast.

**3. Near-duplicate via MinHash.** Jaccard ≥ 0.8 between an eval prompt and any training prompt. Catches substitutions and reorderings that defeat n-grams.

**4. Semantic near-neighbour.** Embed eval and training prompts, index the training side, and for each eval item look at the top-5 neighbours above cosine 0.92. This catches paraphrase. It also produces false positives on genuinely similar-but-distinct questions, so this tier is *review*, not *auto-fail* — I dump the flagged pairs and read them.

**5. The answer-side check, which people forget.** Contamination can live in the response rather than the prompt. If a training response contains the eval's gold answer verbatim for an unusual answer string, that's leakage even if the prompts differ.

**6. The behavioral test.** Take 50 eval prompts and truncate each to its first 40%, then have the model greedily complete. If it reconstructs the rest of the prompt, it has seen it. This is the test that requires no access to the training data at all, which makes it the one you use to audit a vendor's model or a public benchmark claim.

**When contamination is found: always remove from training, never from eval.** Removing from eval makes the number go up and destroys the measurement, and once you've done it you can never compare to previous baselines. Then re-run the baseline and report the corrected number alongside the contaminated one, so the delta is legible.

**🔍 Failure taxonomy — internal eval good, production bad.** Work the list in order, cheapest first: (1) contamination — run the six checks above; (2) distribution mismatch — is your eval drawn from a different time window and tenant mix than production? plot the embedding distributions side by side; (3) prompt-construction skew — is production's system prompt, retrieval context or template different from your harness? this is the most common cause and the cheapest to check; (4) sampling params — greedy in eval vs temperature in prod; (5) the eval metric doesn't measure what users care about — exact match while users care about usefulness; (6) the eval is stale relative to a product change. In my experience (3) and (2) account for most of these tickets, and (1) is the one people never check because it's embarrassing.

**⚠ Trap:** believing a public benchmark number for a model you didn't train. Every widely-published benchmark is in the pretraining corpus of every recent model to some degree. This is why frontier evaluation moved toward held-out, freshly-authored, and dynamically-generated benchmarks, and why "we scored X on Y" is worth almost nothing without a contamination analysis. If you cite a benchmark in an interview, say what its contamination status is — that single sentence marks you as someone who evaluates rather than someone who reads leaderboards.

### What is model collapse and how worried should I be about training on synthetic data?

**📄 Paper:** Shumailov et al. — "The Curse of Recursion: Training on Generated Data Makes Models Forget" (2023), later published in Nature (2024) as work on models collapsing when trained on recursively generated data. The finding: if you train a model on its predecessor's outputs, and repeat that for generations, the models progressively lose the tails of the distribution and converge toward a low-variance mode. Rare events disappear first, then variance shrinks, then the distribution degenerates.

The mechanism is worth stating precisely because the popular version of this result is overstated. Three compounding error sources: (a) **statistical approximation error** — you sample a finite number of outputs, so rare events are under-sampled and vanish from the next generation's data; (b) **functional expressivity error** — the model class can't represent the true distribution exactly; (c) **functional approximation error** — the optimizer doesn't find the best fit anyway. Iterate the loop and the tails get chopped at every step. It's the same shape as generation loss in repeated lossy compression, or repeated JPEG re-encoding.

**How worried you should actually be, and this is where I'd push back on the doom framing:** collapse as demonstrated requires *fully replacing* real data with generated data, generation after generation, with no filtering and no fresh real data injected. That is not what anyone does in practice. The published follow-up work on accumulating data — keeping all previous real data and *adding* synthetic rather than replacing — shows the collapse is substantially avoided. And every serious synthetic pipeline includes a verification step, which is exactly a mechanism for re-injecting external information (the verifier's ground truth) into the loop.

So the operational rules I actually enforce, which is the useful form of this answer:

1. **Accumulate, never replace.** Real human data stays in the mix at every iteration. My floor is a meaningful fraction — I'd want at least 20–30% non-model-generated content in an SFT set, and the exact number is a judgment call I'd want to defend with an eval.
2. **Verify, because verification injects external signal.** Synthetic data checked against execution, a database, or a document corpus is not purely recursive — the verifier is a channel from reality. Unverified model output filtered only by another model's opinion is the dangerous case.
3. **Ground in real documents.** Seed-grounded generation is structurally resistant to collapse for the same reason: the content comes from your corpus, not from the model's prior.
4. **Monitor diversity across iterations, not just quality.** Collapse shows up as diversity loss *before* it shows up as quality loss. Track mean pairwise embedding similarity and output entropy generation over generation; a monotone rise is your early warning.
5. **Watch for the production version of this loop**, which is the one that will actually get you: your model's outputs go into your logs, your logs become next quarter's training data, and you've built a recursive self-training system by accident. Nobody designed it; it emerges from the pipeline. Prefer human-edited turns, cap the unedited-model-output fraction, and keep a standing human-authored injection.

**🗣 Say this in the room:** "Collapse is a real result about *recursively replacing* real data with generated data. In practice the mitigations are structural: accumulate rather than replace, ground generation in real documents, and verify against something external so the loop isn't closed. The version I actually worry about is accidental — production logs containing our own model's output becoming next quarter's training set — and I guard that with per-turn provenance and a cap on unedited model output."

### Explain catastrophic forgetting mechanically, and tell me how you'd measure it on a fine-tune.

Mechanically it's the most ordinary thing in the world: gradient descent on objective B moves weights toward B's optimum, and nothing in the loss function knows or cares about A. The pretrained weights sit at a point that is good for a vast distribution; your SFT gradient points toward a point that is good for 14,000 examples. Every step trades a little of the first for the second. There is no separate storage for "general capability" that fine-tuning politely avoids — it's the same parameters.

**📄** The phenomenon was named by McCloskey and Cohen (1989) in connectionist networks and it has never been solved, only managed. The reason it's *catastrophic* rather than gradual in classical settings is that the network has no constraint keeping it near the old solution and no data reminding it of the old task.

Why it looks less catastrophic in LLM SFT than the classical literature suggests: the update is tiny relative to pretraining (the 3-parts-per-million calculation from earlier), the LR is two orders of magnitude below pretraining LR, and modern SFT sets are broad enough to be partial replay by accident. But "less catastrophic" is not "absent," and the pattern is very consistent: **narrow, repetitive, high-epoch fine-tunes forget the most.** A 3-epoch run on 2,000 examples of a single format is the worst case. A 1-epoch run on 200,000 diverse examples barely forgets at all.

**Measurement, which is the actionable half.** You need a **held-out general-capability suite** that has nothing to do with your task, fixed before the run, run at every checkpoint. Mine has five slices:

1. **Knowledge** — an MMLU-style multiple-choice battery. Cheap, high variance on small samples, so use ≥1,000 items.
2. **Reasoning** — GSM8K-style grade-school math, or a held-out slice of it. Sensitive to forgetting and cheap to grade exactly.
3. **Code** — HumanEval-style pass@1 with actual execution. This one moves first on non-code fine-tunes and it's the canary.
4. **Instruction following** — IFEval-style verifiable constraints ("respond in exactly three bullets, no word longer than ten letters"). Programmatically gradable, and it catches the specific damage where a format fine-tune destroys general format-following.
5. **Safety / refusal behavior** — your refusal set plus your over-refusal set. Fine-tuning on benign task data is documented to weaken safety behavior, so this is not optional if you're shipping.

Plot all five against your task metric on the same x-axis of checkpoints. What you're looking for is the **knee** — the checkpoint where task metric has mostly saturated and capability metrics start dropping steeply. That's your stopping point, and it's usually earlier than the best-task-metric checkpoint.

**Set the gate before the run.** "Ship requires task metric +5pt and no capability slice below −2pt absolute." Writing it down in advance is what stops the retroactive negotiation where everyone decides −6pt on code was fine actually.

**📐 Numbers you must know:** the shape I've seen repeatedly on a narrow 3-epoch LoRA of an 8B — task metric +8 to +15 points, general knowledge −1 to −3, code −3 to −8, instruction-following −2 to −5, and a measurable weakening of refusal behavior. Full fine-tuning at the same data and epochs roughly doubles the capability drops. Treat these as the priors you go in with, not as measurements.

**⚠ Trap:** measuring forgetting only on knowledge benchmarks. MMLU-style scores are the *most* robust to SFT — they're multiple-choice, so the model only has to rank options, and a lot of format damage doesn't show up. Code and instruction-following degrade far earlier. A team reporting "MMLU only dropped 0.8 points, no forgetting" has measured the least sensitive thing available.

### How does replay work, and what exactly do you mix in?

Replay is the direct fix: interleave examples from the old distribution so the gradient has a term pulling back toward it. Conceptually it converts a sequential problem into a joint one — instead of "learn B after A," you're doing "learn B and a bit of A simultaneously," and joint training doesn't forget.

**What to mix, in descending order of what I'd actually use:**

1. **General instruction data from a broad public SFT set.** The pragmatic default. A few tens of thousands of diverse instruction–response pairs covering reasoning, code, writing, chat. It's not the model's original pretraining distribution, but it's a decent proxy for "generally useful assistant behavior" and it's freely available. Check licences.
2. **Self-generated replay from the base model.** The cleaner idea: before fine-tuning, sample the *base model's own* responses to a set of diverse prompts, and train on those alongside your task data. This is exactly a distillation-from-yourself term, and it directly penalizes drift from the base behavior in a data-driven way. It costs one generation pass and it's better matched than public data because it's on-policy for the model you're actually modifying.
3. **Pretraining-style raw text**, if you have access to something resembling the original corpus. Rare for open-weight models since the corpora usually aren't released. Works if you're doing continued pretraining anyway.
4. **Your previous fine-tune's data**, when you're doing iterative rounds. Round 3 should include samples from rounds 1 and 2, or you'll forget round 1's behavior while learning round 3's. This is the version people forget until their second release regresses the first release's feature.

**The ratio.** Published continual-learning work and my own experience both land in the same place: replay fractions in the **5–30%** range recover most of the lost capability, with diminishing returns above that and a real cost to task performance at high fractions (you're spending gradient budget on the old task). I start at **20%** and tune it as an explicit knee-finding exercise: run 0%, 10%, 20%, 40% and plot task metric against capability metric. You get a Pareto frontier and you choose a point on it. That plot is the artifact I'd bring to a design review, and producing it is a two-hour job on an 8B.

**Implementation detail that matters:** mix at the **example level, shuffled**, not as alternating phases. Training on task data for an epoch then replay data for an epoch gives you forgetting of the task data. Interleaving means every optimizer step sees a mixture, which is what makes it a joint objective rather than a sequential one.

**Second detail:** the replay examples' loss mask follows the same rules as everything else. And if your replay set uses a different chat template convention than your task set — very likely if it's a public dataset — normalize it through *your* template first. Otherwise you're training the model to handle two formats and reinforcing neither.

**💰 Math:** replay's cost is straightforwardly proportional. A 14,000-example task set at 1,450 tokens with 20% replay means adding 3,500 replay examples; if replay examples average 900 tokens, that's 3.15M extra tokens per epoch against 20.3M, a **15.5% increase in training compute**. On the earlier $7.50 run, that's $1.16. The cost of replay is essentially zero and the cost of not doing it is a capability regression that reaches users. This is the easiest trade in the entire section and I'd argue for it every time.

### Is LoRA genuinely a mitigation for forgetting, or is that just a story people tell?

It's genuine, and the mechanism is clean enough to state precisely: LoRA constrains the update to a low-rank subspace and freezes everything else, so the reachable set of final weights is much smaller and much closer to the base. You cannot destroy what you cannot modify. Add the zero-initialized `B` and you start exactly at the base model's function, so the entire trajectory is a controlled departure from it.

**📄 Paper:** Biderman et al. (2024), "LoRA Learns Less and Forgets Less" — the systematic version of this claim. Comparing LoRA against full fine-tuning on code and math, they found LoRA underperforms on target-domain learning while retaining substantially more base-model capability outside it, and behaves like a regularizer that preserves output diversity. The title is the summary and it's the honest framing: it's a trade, not a free win.

Three qualifications, because "LoRA doesn't forget" as an unqualified claim is wrong:

1. **It's a matter of degree, not a guarantee.** A high-rank LoRA on all linear layers, trained for 5 epochs at 3e-4 on a narrow set, will absolutely damage general capability. The protection comes from the *constraint being tight*, so it weakens exactly as you loosen the constraint to get more task performance. Rank, LR, epochs and module coverage are all dials that trade forgetting for fit.
2. **It does not protect safety behavior reliably.** Published work on fine-tuning breaking alignment has found that small amounts of fine-tuning — including parameter-efficient fine-tuning, and including on benign data — can meaningfully weaken safety training. Do not skip the safety slice of your capability suite because you used LoRA.
3. **Merged vs unmerged matters for the claim.** If you merge and then fine-tune again on the merged model, you've lost the constraint. Iterative merge-then-LoRA rounds accumulate drift like full fine-tuning does.

The stronger combined recipe, and what I'd actually propose: **LoRA plus replay plus an early-stopping gate on the capability suite.** They attack different parts of the problem — LoRA restricts the reachable set, replay supplies counter-gradient, the gate stops you before the knee. Each alone gets you most of the way; all three together let me ship without a capability review, which is the real deliverable.

**🗣 Say this in the room:** "LoRA forgets less because the update is confined to a low-rank subspace with a zero-init that starts at the base function — it's a hard constraint on how far you can move, not a preference. The trade is documented: less target-domain gain, more capability retention. I combine it with 20% replay and a hard gate on a general-capability suite, and I never assume it protects safety behavior, which is documented to degrade even under parameter-efficient fine-tuning on benign data."

### Someone on the team suggests EWC. Explain it, then tell me whether you'd use it.

**📄 Paper:** Kirkpatrick et al. (2017), "Overcoming Catastrophic Forgetting in Neural Networks" (PNAS) — **Elastic Weight Consolidation**. The idea is elegant and worth being able to derive.

The intuition: not all parameters matter equally for the old task. Some are load-bearing; moving them destroys prior capability. Others are nearly free. If you knew which was which, you'd anchor the important ones and let the rest move. EWC estimates importance via the **diagonal of the Fisher information matrix** — for parameter θᵢ, `Fᵢ = E[(∂ log p(y|x, θ)/∂θᵢ)²]` over data from the old task. High Fisher means the log-likelihood is sensitive to that parameter, i.e. it's important.

Then you add a quadratic penalty to the new task's loss:

```
L(θ) = L_new(θ) + (λ/2) · Σᵢ Fᵢ · (θᵢ − θ*ᵢ)²
```

where `θ*` is the old-task solution. It's a per-parameter-weighted L2 pull toward the pretrained weights — a spring whose stiffness is the Fisher value. In the Bayesian reading, it's a Laplace approximation to the old task's posterior used as the prior for the new task, which is the framing to give if the interviewer wants depth.

**Would I use it? Almost never, and I'd say so directly.** The reasons are practical:

1. **You need data from the old task to estimate the Fisher.** For a pretrained LLM that means the pretraining corpus, which you generally don't have. You can estimate on a proxy corpus, but then you're estimating importance for the wrong distribution, and at that point you could have just done replay with the same proxy corpus — which is simpler and works better.
2. **The memory cost is a full extra copy of the parameters for `F` plus another for `θ*`.** For an 8B in fp32 that's 32 GB × 2 = 64 GB of additional state on top of an already-tight budget. Replay costs a few percent of extra tokens.
3. **The diagonal Fisher approximation is crude** on a model with enormous parameter correlation, and λ is a finicky hyperparameter with no good default.
4. **Replay and LoRA dominate it empirically for this use case** and are vastly simpler to operate.

So my answer in a room is: **know it, name it as the classical reference, explain the Fisher-weighting idea, and then explain why the field routed around it for LLMs.** That's a stronger answer than either "I'd use EWC" (naive) or "never heard of it" (a gap). The general lesson — regularization-based continual learning lost to replay-based continual learning in practice, because replay is trivially implementable and scales — is a real piece of judgment about the field.

**⚠ Trap:** confusing EWC with plain L2-to-init (sometimes called L2-SP), which applies uniform stiffness. Uniform L2 toward the pretrained weights is a much weaker idea that fights *all* learning equally, and it's what people accidentally implement when they add "weight decay toward the initialization." The Fisher weighting is the whole point of EWC; without it you just have a badly-tuned regularizer.

### You're standing up a labeling program from zero. Walk me through the first four weeks.

This is a Forward-Deployed / Applied AI question and the answer separates people who have run one from people who have specified one. Labeling ops is a process-design problem, not a tooling problem.

**Week 1 — write the guidelines and the taxonomy, then break them.** The guidelines document is the actual product. It contains: the task definition in one paragraph; the label taxonomy with a positive and negative example for each category; **the edge cases, which is 70% of the document by length**; the tie-breaking rules; and a "when in doubt, do X" default. Then — and this is the step people skip — *you* label 50 examples yourself against your own guidelines. You will find your guidelines are ambiguous on 15 of them. Fix, repeat. If the person who wrote the guidelines hasn't labeled with them, they're wrong.

**Week 2 — pilot round with 3–5 annotators on 100 overlapping items.** Everyone labels the *same* 100. You are not collecting data; you are measuring the guidelines. Compute inter-annotator agreement (next question), then hold a **disagreement review** where you walk through every disagreement live with the annotators. Two-thirds of disagreements are guideline gaps, one-third are annotator error. Rewrite the guidelines from the transcript. Expect the guidelines to double in length. Then run a *second* pilot on 100 fresh items and confirm agreement improved; if it didn't, the task itself is underspecified and you need to decompose it into simpler sub-questions.

**Week 3 — qualification and calibration.** Build a **gold set** of 50–100 items with adjudicated correct answers. Annotators must pass it (say ≥85% agreement with gold) to work on real data. Then salt gold items into the live queue at a low, unannounced rate — I use 3–5% — as a continuous quality monitor. Per-annotator gold accuracy over a rolling window is your quality dashboard, and it's what lets you catch drift, fatigue, and the annotator who discovered that clicking the modal category pays the same.

**Week 4 — production labeling with adjudication.** Route real items. Multi-label the items that need it: I use double-labeling on 100% for high-stakes small sets, and 10–20% overlap for large sets with the overlap used purely for ongoing agreement measurement. **Adjudication** handles conflicts — a senior annotator or the task owner resolves, and critically, *every adjudication decision goes back into the guidelines as a new edge case*. The guidelines document should be growing throughout the project. If it stopped growing, either the task is genuinely solved or nobody's escalating.

**Running items throughout:** a weekly office hour where annotators ask questions (their questions are your best source of guideline bugs); a changelog on the guidelines with a version number stamped onto every label, so when you change a rule you know which labels predate it; and a re-labeling budget for when a rule change invalidates prior work.

**⚠ Trap:** treating annotator disagreement as annotator failure. High disagreement is usually a signal that the *task* is ill-posed — you asked "is this response good?" instead of "does this response answer the question asked, yes or no?" Decomposing a subjective judgment into several objective sub-judgments is the single most effective intervention on a low-agreement task, and it's what I'd propose first.

**🗣 Say this in the room:** "The guidelines document is the deliverable, and it's built by pilot rounds: everyone labels the same hundred items, we measure agreement, we review every disagreement live, and the guidelines roughly double. Every adjudication after that goes back in as a new edge case. If agreement stays low after two pilots, the task is underspecified and I decompose it rather than pushing harder on the annotators."

### How do you measure inter-annotator agreement, and what number is good enough?

Raw percent agreement is the wrong instrument and you should say so immediately, because it doesn't correct for chance. If 90% of your items are the majority class, two annotators who both always guess the majority class agree 81% of the time while providing zero information.

**Cohen's κ** corrects for chance agreement between exactly two annotators on categorical labels: `κ = (p_o − p_e) / (1 − p_e)`, where `p_o` is observed agreement and `p_e` is the agreement expected from the annotators' marginal distributions. κ = 0 means chance-level, 1 means perfect. **Fleiss' κ** generalizes to more than two raters when each item is rated by the same *number* of raters, though not necessarily the same raters.

**Krippendorff's α** is the one I actually reach for, and knowing *why* is the differentiator. It handles: any number of annotators, **missing data** (annotators who didn't rate every item — which is the normal case in a real labeling queue), and **any measurement level** — nominal, ordinal, interval, ratio — via a pluggable distance function. That last property matters enormously for the ratings tasks common in LLM work: on a 1–5 quality scale, a 4-vs-5 disagreement should count far less than a 1-vs-5 disagreement, and κ treats them identically because it's nominal. Krippendorff with an ordinal or interval distance function gets this right. The structure is `α = 1 − D_o/D_e`, observed disagreement over expected disagreement.

**What number is good enough.** The commonly-cited convention from content-analysis practice is α ≥ 0.80 for confident conclusions and 0.67 as a floor for tentative ones. Landis and Koch's oft-quoted κ bands (0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect) are widely used and equally widely criticized as arbitrary — I'd cite them and immediately note they're heuristics, not statistics.

**My actual operating position:** the acceptable number is task-dependent and the right framing is *what agreement do you need for the downstream decision*. For a binary factuality label feeding a hard training filter, I want α ≥ 0.8 — below that my filter is noise. For a 5-point helpfulness rating used to rank candidate responses in aggregate, 0.6 is workable because the noise averages out across many items. **And the honest thing to say: for genuinely subjective preference tasks, human-human agreement is often only 60–75%, and that is the ceiling on any model you train from those labels.** Reporting a model at 85% "accuracy" against labels whose own agreement is 70% means your metric is measuring something other than correctness.

**⚠ Trap:** κ is deflated by class imbalance — the "kappa paradox." With 95% of items in one class you can have 94% observed agreement and κ near 0.2, and people conclude the annotators are bad when actually the *prevalence* is the problem. When you see high percent agreement with low κ, check the marginals before firing anyone. Report both numbers, always.

**🗣 Say this in the room:** "Percent agreement, then Krippendorff's α with a distance function matched to the measurement level — ordinal for rating scales, so a 4-vs-5 disagreement is penalized less than 1-vs-5. Krippendorff over Cohen's κ because real queues have partial overlap and missing ratings. I target α ≥ 0.8 for labels driving a hard filter and I treat human-human agreement as the ceiling on any model trained from them, which is why I report it next to model accuracy."

### What are gold questions and active learning doing in a labeling pipeline?

Two different jobs that people conflate: gold questions control *quality*, active learning controls *what you spend labeling budget on*.

**Gold questions** are items with a known, adjudicated-correct answer, injected into the live queue indistinguishably from real work. Three functions:

1. **Qualification.** A candidate annotator does 50 golds; pass at a threshold or don't get access.
2. **Continuous monitoring.** 3–5% of live items are gold. A rolling per-annotator accuracy is your quality signal, and it catches the things a one-time qualification cannot: fatigue late in a shift, drift after a guideline change, and the annotator who was replaced by a less careful colleague on the same account.
3. **Weighting.** Per-annotator accuracy becomes a weight in aggregation, so a 0.95-accuracy annotator's label outranks a 0.7 one when they disagree.

The design constraint is that golds must be **indistinguishable and representative.** If your golds are all easy, they measure attention rather than skill, and everyone passes. If they're all pathological edge cases, everyone fails and you've measured nothing. Sample golds from the actual difficulty distribution, and refresh them — annotators memorize a static gold set surprisingly fast, especially when the pool is shared.

**Active learning** answers "which of my 2 million unlabeled items should the humans see?" Labeling is the dominant cost in the program, so choosing well is worth real money. The strategies, in the order I'd try them:

- **Uncertainty sampling.** Label the items where the current model is least confident — highest entropy over the output, or lowest margin between the top two options, or highest disagreement across k sampled generations. Cheap and effective. The k-sample disagreement version is my default for generative tasks since there's no clean "confidence" scalar.
- **Diversity / core-set sampling.** Pick items that are far from what you've already labeled in embedding space, so you buy coverage rather than redundancy. Essential as a *combination* with uncertainty, because pure uncertainty sampling clusters — the model is uncertain about 500 near-identical items and you label all 500.
- **Error-driven sampling.** Cluster the model's known production failures and label from those clusters preferentially. Most directly tied to the metric you're trying to move, and my first choice once a model is in production.
- **Query-by-committee.** Multiple models (or checkpoints) disagree → label it. More expensive, and mostly redundant with the k-sample version above.

**The composition I run** is a mixture, not a pure strategy: roughly 50% uncertainty/error-driven, 30% diversity-driven, and **20% uniform random — which is non-negotiable.** The random slice is the only thing that gives you an unbiased estimate of your true production error rate. A dataset labeled purely by active learning is systematically enriched for hard cases, so any metric computed on it is pessimistic and, worse, incomparable across rounds because the selection distribution changed.

**⚠ Trap:** active learning biasing your eval set. Never build the eval set with active learning. It must be uniformly (or stratified-by-production-traffic) sampled or it does not estimate production performance. I've seen a team celebrate a model that "solved the hard cases" while overall production quality was flat, because both their training *and* eval sets had been actively selected toward the same hard cluster.

### How do you work with a labeling vendor, and what does a label actually cost?

The vendor decision is a sourcing decision with three archetypes, and I'd name them by what they're actually good at rather than by brand.

**Managed large-scale platforms** (Scale AI is the canonical one) — you bring a task spec, they bring workforce, tooling, project management and QA. Right for high volume, well-specified tasks, and for the case where you need a contract, a SOC 2 report and an SLA. Slower to spin up, more expensive per unit, and there's a real management overhead on your side regardless of what the sales deck says.

**Quality-focused / expert marketplaces** (Surge AI positioned this way; Mercor in the expert-sourcing direction) — smaller, more curated workforces, better for tasks needing genuine domain expertise or careful judgment. This is what you want for a Harvey-style legal task or a clinical one, where a general crowd worker cannot produce a usable label at any price.

**In-house / hybrid** — your own domain experts, or a small dedicated contractor team you train directly. Highest quality, best feedback loop, doesn't scale, and it's what I'd use for the first 1,000 examples of anything *regardless* of the eventual plan, because that's how you learn what the guidelines need to say. **📅 Volatile:** the vendor landscape consolidates and repositions constantly; verify who does what before citing anyone in an interview.

**The economics.** Order-of-magnitude figures I've seen quoted, offered as a mental model rather than a quote:

- Simple categorical labeling by a general crowd: **cents to low single-digit dollars** per item.
- Careful preference comparison or quality rating on a model response: **$1–$10** per item.
- Writing or heavily editing a high-quality instruction–response pair: **$5–$50**, driven mostly by response length.
- Domain-expert work (attorney, physician, competitive programmer): priced hourly, **$50–$200+/hour**, which for a 20-minute item is $17–$67 each.

**💰 Math — the number that decides the project.** 10,000 expert-written examples at 20 minutes each = 3,333 expert-hours. At $80/hour blended that's **$266,000**, and at 5 experts working half-time it's about 32 weeks of calendar. Compare: 10,000 model-drafted, expert-*edited* examples at 6 minutes each = 1,000 hours = **$80,000** and 10 weeks, plus maybe $500 of generation cost. That 3.3× on both money and calendar is why draft-then-edit is my default and why I'd challenge any plan that starts with writing from scratch. Then compare both against the fine-tuning compute from earlier — **$7.50.** The compute is 0.003% of the program cost. Say that number out loud in a design review; it reframes the entire conversation about where to be careful.

**Vendor management mechanics that matter:** run a **paid pilot with two vendors on the same 500 items** before committing, and compare on agreement-with-your-gold, not on price; require the *same* annotators across the engagement (turnover destroys consistency and vendors will silently rotate); own the guidelines document yourself rather than letting the vendor own it; demand per-annotator IDs in the delivered data so you can compute per-annotator quality and exclude bad ones; and negotiate a re-labeling clause for items that fail your gold check. Also insist on a data-handling addendum covering where the data goes and whether annotators can see PII — a labeling vendor is a data processor and it belongs in your DPA.

**⚠ Trap:** budgeting for labels and forgetting the internal cost. For every dollar of vendor spend, expect meaningful internal engineering and domain-expert time on guidelines, pilots, adjudication, and QA. A program with a $200k vendor budget and no allocated internal owner produces $200k of unusable labels. The single best predictor of a labeling program's success is whether one named person owns the guidelines document.

### Design it end to end. A legal-research product, GPT-class model behind it, and the complaint is "it makes things up on niche jurisdictions." Give me the program.

I'll structure this as I would in the room: diagnose before prescribing, then the plan, then the gates.

**Step 0 — establish that fine-tuning is the right instrument, and be willing to conclude it isn't.** "Makes things up on niche jurisdictions" is, on its face, a *knowledge* problem, and I opened this section by saying SFT doesn't install knowledge. So the first move is decomposition: sample 200 failures and categorize. My prior on what we'd find: (a) retrieval didn't surface the relevant authority at all — a retrieval problem; (b) retrieval surfaced it and the model ignored it or contradicted it — a grounding-behavior problem; (c) the corpus genuinely lacks the jurisdiction — a data-coverage problem; (d) the model answered confidently when it should have abstained — an abstention problem. Only (b) and (d) are SFT-shaped. If the split is 70% (a) and (c), I'd say so and redirect the budget to the index, and I'd expect that to be the right call more often than not.

Assume the diagnosis lands at 45% (b), 30% (d), 25% (a). Then:

**Step 1 — the eval set, first, before any data pipeline.** Slices with targets: grounded-answer accuracy with correct citation ≥ 92%; **false-answer rate on unanswerable-from-context ≤ 1%** (the ship-blocking metric — a fabricated citation in a legal product is a professional-liability event); false-abstention rate ≤ 15% (we'll accept annoyance to buy safety); contradiction-handling on conflicting-authority cases; and a general-capability suite with a hard −2pt gate. 600 examples, authored by two attorneys against a written spec, with an adjudication pass, from a *later* time window than any training data. Freeze, hash, check in. Baseline the current prompted system on it — that number is the thing everything is measured against.

**Step 2 — data construction, seed-grounded from the actual corpus.** For each of ~8,000 sampled authorities, generate the cross-product of {task type} × {persona}: factual lookup, multi-hop across two authorities, comparison across jurisdictions, **answer-absent (built by ablation — remove the answering document, keep topically-similar distractors)**, and contradictory-authority. Personas from real user roles: litigator, transactional associate, paralegal, in-house counsel. Target composition 50/20/20/10 across the four grounding buckets from the abstention discussion.

**Step 3 — verification ladder.** Structural (citation format parses, cited doc IDs exist in the offered context — that check alone kills fabricated citations at the data level). Then entailment verification: every factual claim in the response must be supported by a span in the provided context, checked by a second model, and unsupported claims fail the example. Then attorney spot-check: 500 uniform for a ±2pp error-rate estimate, plus stratified sampling from the boundary band and every generation bucket. Expect the first batch to have a 10–20% error rate and expect the review to rewrite the generation prompts twice.

**Step 4 — the human spine.** 1,500 attorney-edited examples on the hardest slices — contradictory authority, partial-coverage, and the jurisdictions in the complaint. Draft-then-edit at ~6 minutes each = 150 hours ≈ $18k at expert rates. This is the part I would not cut.

**Step 5 — train.** LoRA r=32 α=64 on all linear layers, 2 epochs, LR 2e-4, cosine, warmup 0.03, 20% replay from a general instruction set plus base-model self-generated responses, packing with the varlen leak test passed, system prompts sampled across production/paraphrase/minimal. Checkpoint every half epoch, full eval sweep on all of them.

**Step 6 — gates and shipping.** Select on false-answer rate subject to the capability floor, not on loss. Then a shadow deployment against live traffic with attorney review of a sample, because your eval is 600 items and production is not. Then a canary at 5%, with false-answer rate and false-abstention rate as the rollback triggers.

**💰 The whole program:** generation ~$4k, verification ~$2k of model calls, attorney time ~$40k across eval authoring, spot-checks and the spine, engineering ~8 weeks, compute **~$50**. Total call it $250k fully loaded. The justification is Step 0's diagnosis plus the incident math: at 50,000 queries/month with 12% unanswerable and an 8% false-answer rate, that's 480 fabrication events/month, and in this domain a single one reaching a filing is a malpractice conversation. Taking that to 1% removes 420 events/month. **That is the business case, and I'd lead the design review with it rather than with the architecture.**

**⚠ Trap:** the thing I'd fight hardest in this design review is the suggestion to "fine-tune on all our case law so the model knows the jurisdictions." It will produce a model that writes fluent, plausible, correctly-formatted legal prose with fabricated citations — strictly more dangerous than what we have now, because the fabrications will be *better disguised*. Knowledge goes in the index. Behavior goes in the weights. I'd put that sentence in the doc.

### Give me the take-home: fine-tune a small model and prove it beat the prompted baseline. What does a passing submission look like?

**🏋 Drill — the canonical applied-AI take-home.** Ten hours. Pick a narrow, verifiable task (structured extraction from a document type, SQL generation against a fixed schema, or classification-plus-justification). Fine-tune a model in the 1B–8B range. Prove it beats a prompted baseline. Here's what I'd grade, in the order I'd look.

**1. The eval harness, and it should be the first thing in the README.** A reviewer opens the repo and should immediately find: what the metric is, why that metric, how it's computed, and how to reproduce it with one command. If the eval appears at the bottom under "Results," you've already signalled the wrong ordering. Points for: a frozen, versioned eval set; a temporal or otherwise principled split; a stated confidence interval on the headline number; and per-slice breakdowns rather than a single scalar.

**2. Baselines, plural, and honestly constructed.** The prompted baseline must be *good* — a real system prompt with few-shot examples, structured output enforced, the strongest reasonable model you could use. A submission that beats a lazily-prompted baseline has proven nothing, and an experienced reviewer will spot it in thirty seconds. I want to see at least: zero-shot prompted, few-shot prompted, and the fine-tune. Bonus for a "prompted with the same examples used in training" baseline, which is the honest control for "did fine-tuning add anything over just showing the model the data?"

**3. Statistical honesty on the delta.** 87% vs 84% on a 200-item eval is not a result — the standard error on a proportion at n=200 and p≈0.85 is √(0.85 × 0.15 / 200) = 0.025, so the 95% CI is roughly ±5 points and your 3-point delta is inside the noise. Show the interval, or use a paired test (McNemar's, since the same items are evaluated by both systems, which is much more powerful than comparing independent proportions). **A candidate who reports a confidence interval or a paired test on their delta immediately reads as senior.** One who reports "94.2%" to four significant figures on 200 items reads as junior.

**4. A capability-regression check.** Even on a take-home. A small held-out general suite, run on base and fine-tuned, reported. It shows you know fine-tuning has a cost.

**5. The cost and latency table.** Tokens per request, cost per 1,000 requests for baseline vs fine-tune (self-hosted amortized or per-token), p50/p95 latency, and the break-even volume where the fine-tune becomes cheaper. This is the beat almost every submission misses and it's the one that maps directly to the job.

**6. A failure analysis.** 20 remaining errors, categorized, with a named next step for each category. "The model still fails on nested tables; the fix is 400 more examples of that shape, which I'd source by X" is worth more than another point of accuracy.

**7. Reproducibility.** Pinned versions, a seed, the exact config file, the data-construction script, and a `make eval` that works. The config file *is* the artifact.

**What sinks a submission:** selecting the checkpoint on eval loss; no contamination check between train and eval; a train/test split done randomly on data with obvious grouping structure; a beaten-down baseline; a headline number with no interval; and — most commonly — spending nine of the ten hours on training infrastructure and one on evaluation. The ratio should be inverted.

**🗣 Say this in the room, if they ask you to summarize the submission in a minute:** "I built the eval first and baselined a well-prompted model on it at 81% exact match with a ±3.4 point interval. The LoRA fine-tune of an 8B reached 91%, and a paired McNemar test on the 600-item eval gives p < 0.001, so the delta is real. It cost $6 of compute and 40 minutes. General-capability suite moved −1.1 points, inside my pre-registered −2 gate. Break-even against the prompted baseline is about 90k requests/month; below that I'd ship the prompt."
