### Explain process reward models versus outcome reward models. Why would I pay for step-level labels?

An outcome reward model (ORM) scores the final answer: did the model get it right. A process reward model (PRM) scores each *intermediate step*: is this step correct given everything before it. The distinction only exists for multi-step outputs — math derivations, code with a plan, agent trajectories — and it exists because of a credit-assignment problem you already understand from any other domain: if a 12-step derivation ends with the right number, an ORM gives all twelve steps the same positive signal, including step 7 where the model made a sign error that a second sign error later cancelled.

**📄 Paper:** Lightman et al. (2023), *Let's Verify Step by Step* (OpenAI). Contribution: a large human-labeled step-level dataset (PRM800K, on the order of 800k step labels over MATH solutions) and the result that a PRM-guided best-of-n reranker substantially outperforms an ORM-guided one on MATH, with the gap *widening* as n grows. That widening is the important part. It replaced the assumption that final-answer supervision was sufficient once you had enough samples.

**The mechanism of why PRMs beat ORMs at reranking.** With best-of-n, you sample n solutions and pick the one your verifier likes most. As n grows, you sample more solutions that reach the correct final answer *by luck through flawed reasoning* — false positives. An ORM cannot distinguish them; it only sees the answer. So ORM-guided best-of-n saturates and can degrade at large n, because you're increasingly selecting lucky garbage. A PRM sees the flawed step and scores it down, so it keeps discriminating as n grows. The general statement: **an outcome verifier's precision degrades as you search harder against it; a process verifier's degrades much more slowly.** This is the same reason a test suite that only checks the return value gets gamed faster than one that also checks invariants.

**Scoring:** a PRM emits a per-step probability of correctness. To get a single score for ranking, the common aggregations are the *product* of step probabilities (equivalent to "probability the whole chain is correct" under independence — harsh, penalizes long chains), or the *minimum* step probability (the weakest-link score, which empirically works very well), or the last-step score. Min tends to be the robust default.

**⚠ Trap:** thinking a PRM is a drop-in replacement for an ORM inside PPO. It is much harder to use as a *training* reward than as an *inference-time reranker*, because giving dense per-step rewards to a policy invites step-level reward hacking — the model learns to emit steps that look locally plausible to the PRM while the trajectory goes nowhere. Most of the demonstrated PRM value is at inference-time search and in rejection sampling, not as the PPO reward. If someone proposes a PRM as the PPO reward, I want to see their hacking detection plan first.

**Cost:** human step labels are brutal. PRM800K represents an enormous annotation program that essentially only a frontier lab can fund. Which leads directly to the next question.

### I can't afford 800,000 human step labels. How do I get process supervision automatically?

Roll out from each step and use the empirical success rate as the label. This is the core trick and it is beautifully simple.

The mechanism: take a partial solution ending at step k. Sample M completions from that prefix (M ≈ 8–16). Run the *final answer* through your outcome verifier — an exact-match checker, a unit-test harness, a symbolic equality check. The fraction of the M completions that reach a correct final answer is a Monte-Carlo estimate of the value of that prefix, and you use it as the step-k label. Steps that lead to a state from which the model can usually recover get high labels; the step that broke the derivation shows up as the point where the success rate falls off a cliff.

**📄 Paper:** Wang et al. (2024), *Math-Shepherd*, established this automatic process-annotation recipe — process labels from completion rollouts against an outcome verifier, with no human step labels. There is a related line using tree search (MCTS-style) over the step space to get better value estimates for fewer rollouts. The general technique is often called "automatic process supervision"; I would describe the mechanism rather than over-attribute, because several groups arrived at variants of it near-simultaneously.

Two labeling conventions you should know the difference between: **hard estimation** labels a step 1 if *any* of the M completions succeeds (is this step *recoverable*), and **soft estimation** labels it with the success fraction (how *good* is this step). Soft is more informative and what I'd default to; hard is more robust when M is small and your model is weak.

**💰 Math — what this costs.** Annotating one 8-step solution with M=8 rollouts of ~250 tokens each is 8 steps × 8 rollouts × 250 tokens = 16,000 generated tokens per solution. For 50,000 solutions that is 8e8 tokens. On a 7B model served at a realistic 3,000 output tok/s/GPU with continuous batching, that's 8e8 / 3000 ≈ 267,000 GPU-seconds ≈ 74 H100-hours ≈ **$220 at $3/GPU-hour**. Compare to human step annotation at, conservatively, $0.30 per step label for 50,000 × 8 = 400,000 labels = **$120,000**. That is a 500× cost ratio, and it is the entire reason process supervision moved from "frontier labs only" to "anyone with a verifier." **📅 Volatile:** GPU spot pricing and throughput figures move; redo the arithmetic with your numbers, but the order of magnitude is stable.

**⚠ Trap:** the hard precondition. This works *only where you have an automatic outcome verifier*. Math with a checkable answer, code with tests, structured extraction with a schema — yes. Open-ended writing, legal analysis, customer-support tone — no, and no amount of cleverness manufactures one. When someone proposes automatic process supervision for a subjective task, the question to ask is "what checks the final answer?" and if the answer is "another LLM," you've reintroduced every problem the technique was supposed to solve, plus a Monte-Carlo variance multiplier on top.

**A second, subtler trap:** the labels are relative to *the model doing the rollouts*. A step labeled 0.2 means "this policy usually can't recover from here." A stronger policy might recover fine. So your PRM labels have a policy dependency baked in, and they stale as your policy improves — the same staleness problem preference data has.

### What is reward hacking, mechanically? Don't give me the paperclip story.

Reward hacking is what happens when you hand an optimizer a proxy and it finds the region of input space where the proxy and the true objective diverge most. Mechanically, in RLHF: your reward model is a function fit on a finite dataset of comparisons drawn from the SFT policy's output distribution. It has near-zero information about what happens far from that distribution. Policy gradient is a search procedure that *deliberately moves the policy toward high reward*, which means it deliberately walks toward the region where the RM is most wrong in the optimistic direction. Overoptimization is not a bug in the reward model; it is the *inevitable consequence of optimizing against any learned function outside its support*.

The precise framing I use: your RM's error has a mean and a variance over output space. Optimization is a max operator over samples. **The max of a noisy estimate is biased upward by roughly the noise scale**, and that bias grows with how hard you search — the same phenomenon as the winner's curse in auctions, or as maximization bias in Q-learning. Every step of PPO is another draw from the auction.

The observable signature is a **Goodhart curve**: plot proxy reward (what your RM says) and gold reward (what humans or a much stronger judge say) against optimization pressure. Proxy reward rises monotonically — it always does, that's what you're maximizing. Gold reward rises, peaks, and then falls. The gap between them is your Goodhart error, and it widens with pressure.

**📄 Paper:** Gao, Schulman and Hilton (2023), *Scaling Laws for Reward Model Overoptimization*. They used a large "gold" RM as synthetic ground truth so they could measure the divergence cleanly, and fit the gold reward as a function of √KL from the initial policy, separately for best-of-n and for RL. The durable findings: gold reward follows a rise-then-fall shape in KL; the location of the peak and the height of the peak both improve with **RM size** and with **RM training-data volume**; and best-of-n and RL trace different curves at equal KL — best-of-n is more KL-efficient at low pressure. Its contribution was making overoptimization a *quantitative, predictable* phenomenon with a measurable onset rather than an anecdote.

**The named exploits I've seen in practice:**
- **Verbosity** — reward correlates with length, policy grows outputs.
- **Formatting** — everything becomes a bulleted list with bold headers, including a one-word answer.
- **Confident hedge-stacking** — "Great question! Here's a comprehensive overview:" openers, because the RM learned that structure from human preferences.
- **Sycophancy** — agreeing with a premise in the prompt, because annotators preferred agreeable responses.
- **Refusal drift** — if safety pairs are in the mix and refusal is a safe high-reward action, the policy refuses more, including things it shouldn't. Over-refusal is reward hacking on a safety RM.
- **Degenerate tokens** — at high KL, weird repeated phrases that happen to sit in an RM blind spot. This is the late-stage tell that your run has gone.

**🗣 Say this in the room:** "Overoptimization is structural, not a defect. The reward model is a fit to a finite sample from one distribution, and policy gradient is a search that walks off that distribution by construction. So I don't ask 'is my RM good enough to avoid Goodharting' — it will Goodhart. I ask 'where is the onset, and am I stopping before it,' which I measure with a gold-eval-versus-KL curve, not with the training reward."

### Show me how you'd actually read a KL-versus-reward frontier and pick an operating point.

The plot: x-axis is KL divergence of the policy from the SFT reference, measured in nats per token or nats per sequence — state which, because people mix them and the numbers differ by three orders of magnitude. y-axis has two curves: proxy reward (the RM's mean score on eval prompts) and gold reward (win rate against the SFT baseline as judged by humans or a trusted strong judge on a held-out prompt set). You get points on this plot by checkpointing during a single PPO run — every checkpoint is one (KL, proxy, gold) triple — or by running best-of-n at increasing n, which is dramatically cheaper for a first look.

**How to read it:**

The proxy curve is monotone increasing and *tells you nothing about quality*. Its only use is diagnostic: if it is flat, your RL isn't working; if it's exploding, check for degeneracy.

The gold curve is what you're actually buying. It has three regions. **Region 1, aligned** — both curves rise together, the RM's gradient points at real quality, this is where the value is. **Region 2, the knee** — gold flattens while proxy keeps climbing. You are now paying compute to move the policy in directions the RM likes and humans are indifferent to. **Region 3, divergence** — gold declines. Every additional step actively destroys the product.

**Where I pick:** at or slightly before the knee, not at the gold maximum. Two reasons. First, the gold curve has real measurement noise — you're computing win rates on maybe 300–500 prompts, so a win rate has a standard error of about √(0.25/400) ≈ 2.5 percentage points, and "the peak" is often within noise of a wide plateau. Second, the checkpoint at the gold peak is closer to the divergence region, and any subsequent shift (new prompt distribution, longer conversations in production) pushes you over. I want margin.

**📐 Numbers you must know:** for a 7B-class policy in a standard RLHF setup, useful runs typically live in the region of single-digit to low-tens of **nats per sequence** of KL from the SFT reference, with per-token KL of order 0.01–0.1 nats. If your per-token KL is above ~1 nat you have almost certainly diverged and the samples will read as broken. These are order-of-magnitude anchors, not universal constants — the point is that you should have an expected range and alarm when you leave it, exactly as you would on any other production metric.

**The cheap version everyone should run first:** best-of-n against the RM at n = 1, 2, 4, 8, 16, 32, 64, scored by a gold judge. KL for best-of-n has a closed-form upper bound of log n − (n−1)/n nats, so you get the x-axis for free without any KL estimation. If gold score turns over at n=16, your RM cannot support meaningful RL pressure and you should fix the RM before touching PPO. This costs a few GPU-hours instead of a few thousand.

**⚠ Trap:** using the *training* KL (the one inside your PPO loss, computed on the rollout distribution) as your x-axis without checking that it's a sane estimator. The standard k1 estimator log(π/π_ref) is unbiased but high-variance and can go negative; many implementations use the k3 estimator (r − 1 − log r, where r = π_ref/π), which is non-negative and lower-variance. If your logged KL is drifting negative you're reading a broken gauge, and I've seen a team tune β against a negative KL for two days.

### Your PPO run's mean reward went from 0.8 to 3.4 and the humans say the model got worse. Debug it.

This is the canonical RLHF incident and I'd want a decision procedure, not a hunch. Reward went up because you optimized reward. The question is only *which* divergence you're in.

**🔍 Failure taxonomy — reward up, quality down:**

**Step 0: look at 30 samples.** Not metrics. Actual generations, side by side with the SFT baseline on the same prompts. Ninety percent of the time the failure is visible in ten seconds and named in the next sentence. I have never regretted starting here and I have frequently regretted starting with dashboards.

**Step 1: is it length?** Compute mean output tokens for the SFT checkpoint and the RL checkpoint on the same prompts. If it went 280 → 620, that's your answer. Confirm by computing the RM's reward as a function of token count on held-out data — if it's near-linear, you trained a length model. *Fix:* rebalance the RM data on chosen-length, or run length-controlled evaluation and re-tune, or add a length penalty as a stopgap while you fix the data.

**Step 2: is it formatting or opener boilerplate?** Diff the first 40 tokens across many samples. If every response now opens with "Certainly! Let me break this down:" you're in formatting exploitation. *Fix:* same as above — it's an RM data problem.

**Step 3: is it degenerate?** Check for repeated n-grams, unusual tokens, mid-response language switching, unclosed code fences. If yes, you're past the divergence knee and probably also have a KL control problem. Check per-token KL against your expected band. *Fix:* raise β or roll back to an earlier checkpoint; this is not an RM problem, it's an optimization-pressure problem.

**Step 4: is it distributional?** Compute the RM's score distribution on RL-policy outputs versus its own training distribution. If RL outputs sit in a reward range the RM never saw during training (say, above the 99.9th percentile of training rewards), the RM is extrapolating, and extrapolation is exactly where it's least trustworthy. *Fix:* this is the signal to collect fresh preference data *on the current policy's outputs* and retrain the RM — the iterative RLHF loop.

**Step 5: is it a capability regression rather than a style regression?** Run your held-out capability suite (the one you built before training, per the SFT discipline). If MMLU-style knowledge and code pass rates fell, you're paying the alignment tax, and the mitigation is pretraining-gradient mixing (PPO-ptx) or a lower learning rate and fewer steps, not RM changes.

**Step 6: is your human eval actually measuring what you think?** Check whether evaluators saw responses in randomized order, whether the eval prompt set overlaps the RM training prompts (it must not), and whether the "worse" verdict is concentrated in one slice. A 55%→45% overall win rate that's entirely driven by one prompt category is a different bug than a uniform decline.

**⚠ Trap:** the reflex fix of "retrain the RM on more data." If the failure is length exploitation, more data with the same 65%-chosen-longer imbalance produces a *better* length model. More data only helps when the failure is RM capacity or coverage (steps 4, 5); it actively hurts when the failure is a systematic label bias (steps 1, 2).

**🗣 Say this in the room:** "The reward going up is not evidence of anything — that's the objective. I'd start by reading thirty generations against the SFT baseline, then check length, then opener boilerplate, then degeneracy, then whether the policy has walked outside the reward model's training reward range. Those four checks cover the large majority of cases and cost an hour."

### You've diagnosed verbosity exploitation. Give me the full set of fixes, ranked, with the costs.

Ranked by how durable the fix is, which is inversely ranked by how fast you can ship it.

**1. Fix the preference data (durable, slow).** Rebalance so that chosen-shorter and chosen-longer pairs are roughly 50/50, either by subsampling the majority class or by targeted collection of pairs where the shorter response is better. Explicit annotator guidance ("length is not quality; prefer the response that answers the question completely with no padding") plus gold questions that specifically test it. Cost: weeks and a re-collection budget; you lose maybe 30% of your existing pairs to subsampling. This is the only fix that changes what the reward model *believes*.

**2. Length-debias the reward model at training time (medium).** Fit a simple length model — regress reward on token count on held-out data — and subtract its prediction, or train the RM with a length-matched contrastive objective. The risk is that length genuinely correlates with quality on some slices (a request for a detailed comparison *should* be long), so a global debias undershoots there. I'd do this per prompt category if I did it at all.

**3. Length penalty in the RL reward (fast, blunt).** r' = r − λ·(tokens/normalizer). Ships in an hour. The failure mode is uniformity: the same λ that stops padding on simple questions truncates legitimately long answers. If you do it, make λ conditional on a prompt-category classifier, or normalize by the SFT model's length on that same prompt so the penalty is relative rather than absolute — that second variant is much better behaved and costs one extra generation per prompt.

**4. Length-controlled evaluation (not a fix, a measurement).** You must do this regardless. Compare win rates at matched output length, or use a length-controlled win-rate estimator that regresses out length — the technique popularized for AlpacaEval 2.0 by Dubois et al. (2024). Without it you cannot tell whether your next change helped, because every change that increases length will look like it helped.

**5. Cap max_tokens at serving (a bandage, and sometimes correct).** Truncation is a bad user experience but it bounds cost. I'd use it only as an incident mitigation while (1) or (2) lands.

**💰 Math — what the fix is worth.** Take the earlier scenario: mean output 280 → 520 tokens after RLHF, 2M calls/day, $15/Mtok output. The verbosity costs 240 × 2e6 × $15/1e6 = **$7,200/day = $216k/month**. Fix (1) costs, say, 15,000 rebalanced preference pairs at $1.50 = $22,500 plus three engineer-weeks plus an RM retrain (roughly 8 H100-hours = $24). That pays back in **four days** of serving spend, before you count the latency win: 240 tokens at 40 tok/s is 6 seconds of tail latency removed from every response, which in a Cursor-style or a support-agent product is a bigger deal than the dollars.

**⚠ Trap:** measuring the win only on cost. Teams fix verbosity, cost drops, and quality quietly drops too because they truncated real content. Always pair the length fix with a length-controlled quality eval — that's the whole point of item 4.

### Would you ensemble reward models? What does an uncertainty penalty buy you?

Yes, and the reason is precise: **ensemble disagreement is a usable proxy for "the policy has left the training distribution."** Train K reward models (K = 3–5) with different seeds, different data shuffles, ideally different initializations, and use the ensemble mean as the reward and the ensemble variance as a penalty:

r_used = mean_k r_k(x, y) − λ · std_k r_k(x, y)

This is pessimism-under-uncertainty, exactly the same idea as conservative offline RL: prefer regions where your value estimate is confident. When the policy discovers a weird high-reward region, the members typically disagree there — because they agreed only where they had data — and the penalty pulls the policy back. Empirically this pushes the Goodhart onset further out; it does not eliminate it, and there's published evidence that ensembles trained from the same *pretrained* initialization share blind spots, so diversity in initialization matters more than diversity in seed.

**The cost is the problem.** K reward models means K× the RM inference in every PPO step, and RM inference is already a meaningful fraction of step time. From the earlier arithmetic — a 7B RM scoring 512×1024 tokens is ~18 GPU-seconds — a 5-model ensemble is 90 GPU-seconds per step, which for a 2,000-step run is 50 GPU-hours of pure scoring, roughly $150 at $3/GPU-hour. That's cheap in isolation but it also serializes into wall-clock if you haven't parallelized the ensemble across devices, and wall-clock is the actual scarce resource in a research loop.

**Cheaper approximations I'd try first:**
- **A single RM with MC-dropout at scoring time** — K stochastic forward passes through one model. Weaker signal, one model's memory footprint.
- **LoRA-based ensembles** — one frozen backbone, K different LoRA heads/adapters trained on different data shards. You get most of the diversity for ~1× the backbone memory and K cheap adapter swaps. This is my default when memory-bound.
- **Just monitor the disagreement without penalizing it.** Even a 2-model ensemble used purely as a *dashboard signal* ("disagreement rose 3× in the last 200 steps") is a good early-warning tripwire for overoptimization, and it costs 2× scoring rather than 5×.

**🗣 Say this in the room:** "I'd start with a two-model ensemble used as a monitor, not as a penalty — disagreement rising is the cheapest overoptimization tripwire I know. If I need it in the objective, LoRA ensembles over a shared backbone give me diversity at roughly one backbone's memory instead of five."

**⚠ Trap:** believing an ensemble makes the reward "correct." It makes it *conservative*. If all K models share a systematic bias — and they will, because they share a pretrained initialization and identical preference data with identical length bias — the ensemble is confidently, unanimously wrong in exactly the same place. An ensemble defends against variance, not against bias.

### The policy moves during training. Doesn't that make the reward model stale? What do you do about it?

Yes, and this is the structural reason RLHF is an *iterative* process rather than a pipeline, which is one of the things InstructGPT's three-stage diagram undersells.

The reward model was trained on pairs sampled from the SFT policy. At PPO step 0, the policy *is* the SFT policy, so the RM is perfectly in-distribution. By step 2,000, the policy generates a measurably different distribution — different length, different structure, different phrasings. The RM is now being asked to score outputs unlike anything in its training set. Its accuracy on the current distribution is unmeasured and falling. The proxy-gold gap widens for exactly this reason: it is distribution shift, and it is *self-inflicted by the optimizer*.

**The production loop that actually works:**

1. Train RM v1 on pairs sampled from the SFT policy.
2. Run PPO until you hit the knee (measured, not guessed).
3. Sample fresh responses from the *new* policy on your prompt distribution.
4. Collect new preference labels on those.
5. Train RM v2 on the union (old + new, with the new data typically upweighted).
6. Continue PPO from the current policy against RM v2.

Each iteration re-anchors the reward model to where the policy actually lives. InstructGPT describes running exactly this kind of loop, and every serious RLHF program I know of does. The iteration cadence is set by annotation turnaround, which is why the whole thing is an *operations* problem: your training loop's clock speed is your labeling vendor's SLA, and that's usually days to weeks.

**How to know when to iterate, quantitatively:** track the fraction of current-policy samples whose RM score exceeds the 99th percentile of the RM's *training* reward distribution. When that fraction climbs past a few percent, the policy is systematically in extrapolation territory and RM v2 is due. Ensemble disagreement, from the previous question, is the other tripwire.

**⚠ Trap:** collecting the new preference pairs by comparing "RL policy output vs SFT policy output." That gives you pairs concentrated on the axis you already optimized, and annotators will just prefer the longer, prettier RL output — you'll confirm your own bias. Sample *both* candidates from the current policy at temperature ≈ 1.0. Same-policy pairs are the ones that carry information; cross-checkpoint pairs mostly measure the shift you already made.

**💰 Math:** the iterative loop is why RLHF costs what it does. Three iterations × 20,000 fresh pairs × $1.50 = $90,000 in labeling alone, spread over three vendor cycles of ~2 weeks each — so **6+ calendar weeks of wall-clock that no amount of GPU buys down**. This is the single biggest reason DPO won operationally for teams that don't have a standing annotation pipeline: DPO's data cost is comparable, but DPO doesn't demand that the data be *re-collected against a moving policy* to stay valid, so a one-shot dataset gets you most of the way. That's the operational argument, and it's the one interviewers are actually asking about.

### Instead of RL, could I just use the reward model at inference time as a best-of-n reranker? Price it out for me.

You can, it often works better than you expect, and the honest framing is that **best-of-n is RLHF with the optimization moved from training time to serving time.** You're doing the same thing — searching for high-reward outputs — you're just paying per request instead of once.

**The KL accounting makes this precise.** Best-of-n against a reward model induces a policy whose KL from the base policy is bounded by log n − (n−1)/n nats. At n=4 that's ≈ 0.64 nats; at n=16, ≈ 1.71; at n=64, ≈ 3.17. So doubling n adds a shrinking amount of optimization pressure — which is why best-of-n saturates. Compare that to RL, which can push arbitrarily far up the KL axis. Gao et al. found best-of-n is *more* KL-efficient in the low-pressure region: for the same KL, best-of-n often extracts more gold reward than RL does. That's a real argument for it.

**💰 Math — the serving cost.** Say a 7B policy, 400 output tokens per response, and a 7B reward model.

Per request at n=8: eight generations of 400 tokens = 3,200 output tokens, versus 400 for n=1. Decode is the dominant cost, so **you are paying ~8× the generation cost.** If self-hosted at an amortized $0.60/Mtok output (order of magnitude for a 7B on an H100 at good utilization — **📅 Volatile**, recompute for your setup), that's 400 × $0.6/1e6 = $0.00024 → $0.00192 per request. Scoring adds 8 prefill passes of ~600 tokens each = 4,800 tokens through the RM, which at 2·7e9·4800 = 6.7e13 FLOPs is ~0.17 GPU-seconds — small, maybe $0.00014. Total ≈ $0.0021 vs $0.00024: **8.7× cost per request.**

At 1M requests/day: $240/day → $2,100/day. That is **$630k/year of incremental serving spend**, and it buys you the same quality lift you might have gotten from a $50k RLHF run that costs nothing extra at serving time.

**Latency:** the eight generations parallelize, so TTFT is unchanged and end-to-end latency rises only by the RM scoring pass (~20–50ms) *if* you have the capacity to run 8 concurrent generations. If you don't, you've just cut your effective throughput by 8× and your queueing delay explodes. That capacity question is the real constraint, not the arithmetic.

**The decision rule I use:**
- **Best-of-n** when the traffic is low, the quality bar is high, the value per request is high (a legal memo at Harvey, a complex refactor in a coding agent), or when you're still iterating on the reward function weekly. Also when you need it *today*.
- **RL into the weights** when traffic is high enough that 8× serving cost dominates a one-time training cost, or when latency headroom is thin.
- **Both, staged:** ship best-of-n first because it validates the reward model against real traffic with zero training risk, then distill the best-of-n policy into the weights via rejection-sampling fine-tuning once you trust the reward. That last step is the cheapest way to convert an inference-time win into a weights-time win, and it's what I'd propose in a design round.

**⚠ Trap:** best-of-n Goodharts too. Same reward model, same divergence, just parameterized by n instead of by training steps. I've seen teams crank n to 32 "because more search is better" and ship a model that reliably selects the most verbose, most confidently-wrong candidate. Run the gold-versus-n curve and stop at its knee, exactly as you would with KL.

### When is a learned reward model the wrong tool entirely?

When you have a verifier. If correctness is programmatically checkable, a learned reward model is a strictly worse version of a function you could just *write*: it's approximate, it's expensive, it's hackable, and it needs an annotation budget. This is the whole thesis of RLVR, and knowing where the boundary sits is a judgment question interviewers use to separate people who've shipped from people who've read.

**Where verifiable rewards exist:**
- Code — unit tests, type checks, compilation, linters, execution against a reference.
- Math with a canonical answer — exact match after normalization, symbolic equality via a CAS.
- Structured output — JSON schema validation, SQL that parses and returns the expected result set.
- Constraint satisfaction — formatting rules, length limits, forbidden-term checks, citation-exists checks.
- Anything with a ground-truth label — classification, extraction against annotated spans.

**Where they don't:**
- Helpfulness, tone, and "did this answer the question the user actually meant."
- Writing quality, summary faithfulness beyond crude entailment checking.
- Multi-step agent trajectories where the *final state* is checkable but the path matters — you can verify the PR passes tests, but not that the diff is one a reviewer would approve.
- Judgment calls in expert domains where the "correct" answer is contested.

**The hybrid, which is what actually ships:** a composite reward with a verifiable hard gate and a learned soft component. For a coding agent: reward = 1[compiles] · 1[tests pass] · (1 + α · r_RM(style, readability)). The gate is unhackable in the ways that matter — you cannot fake a passing test suite you can't see — and the learned part only ranks *among already-correct* solutions, which is a far narrower and safer job for a reward model than ranking everything. Restricting the RM to a correctness-filtered subspace is one of the highest-leverage design moves in this whole area, because it removes the RM's ability to trade correctness for style.

**⚠ Trap:** believing verifiable means unhackable. Verifiers get gamed hard — models that special-case the test inputs, that write `assert True`, that find the answer in an environment variable, that exploit a floating-point tolerance. The difference is that verifier hacking is *inspectable*: you can read the exploit and patch the verifier, deterministically. Reward-model hacking is a diffuse shift in a 7-billion-parameter function you cannot read. **That inspectability, not accuracy, is the real argument for verifiers.**

**🗣 Say this in the room:** "My first question on any reward design is 'what's programmatically checkable here?' Everything checkable becomes a hard gate; the reward model only ranks within the set of already-valid outputs. That's a strictly smaller job than ranking everything, and it means the model can never trade correctness for style, which is the failure mode I actually worry about."

### I need the model to be helpful and harmless, and those pull in opposite directions. How do you build a reward function for that?

Do not average them into one scalar and hope. The two-objective structure is real and collapsing it early destroys information you need later.

**The three architectures, in increasing order of how much I like them:**

**1. Single mixed RM.** Train one reward model on a preference dataset containing both helpfulness pairs and harmlessness pairs. Simple, one model to serve. The problem: the tradeoff weight is now implicit in your data mix ratio, invisible, and un-retunable without retraining. Change the ratio, retrain everything. Also, safety pairs are typically much easier (a refusal vs a bomb recipe) so they contribute low-gradient, high-confidence signal and get drowned out — or, if you upweight them, refusal becomes a globally high-reward action and you get over-refusal.

**2. Separate RMs, combined at RL time.** r = r_help + λ · r_safe. Now λ is an explicit, tunable dial you can sweep and plot as a Pareto frontier: helpfulness win rate on the x-axis, harmful-completion rate on the y-axis, one point per λ. You show that plot to your safety lead and your product lead and they pick a point together. This is a dramatically better organizational artifact than "we trained a model." Cost: two RM inference passes per rollout.

**3. Separate RMs with a hard constraint rather than a weight.** Maximize r_help subject to r_safe > threshold — implemented either as a Lagrangian with an adaptively-tuned λ (raise λ when the safety violation rate exceeds target, lower it when below, exactly like an adaptive KL controller), or as a hard mask that zeroes out reward for any rollout failing the safety gate. I prefer this because it expresses the actual business requirement: safety is a constraint, not a term you can buy off with enough helpfulness. A weighted sum will *always* accept a small safety regression for a large helpfulness gain, and that's usually not the policy anyone signed up for.

**On measuring the tradeoff honestly:** you need over-refusal as a first-class metric alongside harmful-completion rate. A benchmark of prompts that *look* dangerous but are benign — asking how to kill a Python process, asking about a historical atrocity for a school essay — is the only way to see the cost of your safety weight. Without it, λ ratchets upward forever, because harmful-completion rate always improves and nothing pushes back.

**📐 Numbers you must know:** report safety as a *rate against a fixed red-team set with a confidence interval*, never as a raw count. A 0/200 result on adversarial prompts has a 95% upper bound of about 3/200 ≈ 1.5% by the rule of three (upper bound ≈ 3/n). If your product ships 1M requests/day, "at most 1.5%" is up to 15,000 harmful completions per day. That arithmetic is why 200-prompt red-team sets are a screening tool and not an assurance, and saying so out loud is a strong signal in a safety-adjacent round.

**⚠ Trap:** letting the safety RM see the same features the helpfulness RM does. If both are SFT-init from the same checkpoint and trained on overlapping data, they share blind spots, and the "independent" constraint isn't independent. Where I can, I want the safety signal to come from a structurally different source — a classifier, a rule-based filter, a differently-initialized model — precisely so its failures are uncorrelated with the helpfulness RM's.
