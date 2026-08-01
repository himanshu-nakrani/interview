### Start me at the beginning — what is RLVR, and why did the field pivot to it away from learned reward models?

The one-sentence mental model: RLVR is what happens when you replace a *learned* approximation of "was this good?" with a *program* that returns the right answer. Everything else — the stability, the ability to train for thousands of steps without the reward saturating, the disappearance of half the reward-hacking failure modes — falls out of that one substitution.

Think about it as a backend engineer. A reward model is a cache of human judgement: you paid annotators, you fit a Bradley-Terry model to their pairwise preferences, and now you have a 7B-parameter function that returns a scalar. Like any cache, it is stale the moment the thing it approximates moves — and in RL the policy *deliberately* moves off the distribution the RM was fit on. That is the whole overoptimization story: the policy climbs the RM's score while the true quality plateaus and then falls, because you are optimizing the error term. In practice PPO-with-an-RM runs get maybe a few hundred steps before you have to stop, and the stopping criterion is "the KL got big and the samples got weird," which is not a criterion.

A verifier is not a cache. It is ground truth, evaluated at request time. `is_correct(model_answer, gold_answer)` for a math problem. `pytest` exit code for a code problem. `lean --check` for a proof. It does not drift when the policy drifts. It cannot be flattered by verbosity, confident tone, markdown headers, or sycophancy — the four things every RM in existence secretly loves. So you can crank the optimization pressure far higher and run far longer without the reward signal degrading.

**📄 Paper:** The term *RLVR* was named in AI2's Tülu 3 report (Lambert et al., 2024), which trained on verifiable-answer instruction data with a binary correctness reward; DeepSeek-R1 (DeepSeek-AI, 2025) then showed the same recipe at scale produces emergent long chain-of-thought.

**⚠ Trap:** saying "RLVR eliminates reward hacking." It eliminates *reward-model* hacking and replaces it with *verifier* hacking, which is a different and often nastier bug because the verifier is code you wrote and therefore you trust it. A model that discovers `pytest` passes when it writes `import sys; sys.exit(0)` in `conftest.py` has hacked your reward perfectly, and your reward curve will look beautiful. More on this later — it is the largest single block of this section for a reason.

**🗣 Say this in the room:** "RLVR swaps a learned reward model for a deterministic checker. The reason that matters operationally isn't purity — it's that a verifier doesn't go out of distribution when the policy does, so you get to run thousands of RL steps instead of a few hundred before the signal rots. The cost you pay is that you now own a verifier service, and verifier bugs become reward bugs."

### Where do verifiable rewards genuinely exist, and where do teams pretend they exist?

The honest taxonomy is a spectrum from "a program decides" to "a person decides," and the interview tell is whether you can name the boundary rather than gesture at it.

**Tier 1 — fully verifiable, cheap, deterministic.** Competition math with a canonical final answer (checked by symbolic equivalence, not string equality). Code with a hidden test suite. Formal proofs in Lean or Coq. Constraint-satisfaction puzzles. Unit conversion, SQL that must return a specific result set, JSON that must validate against a schema. Structured extraction where you have gold fields. This is where RLVR was born and where it works with almost no cleverness.

**Tier 2 — verifiable but expensive or noisy.** Agentic software engineering: SWE-bench-style "did the repo's tests pass after your patch." It is verifiable, but each rollout costs a container, a dependency install, and 30–300 seconds. Terminal/tool tasks where you check final environment state. Web navigation where you check whether the target record actually changed in the backend. Long-horizon tasks where the check is real but sparse and flaky.

**Tier 3 — partially verifiable via decomposition.** Not the whole output, but *checkable properties* of it. A summary must cite only spans present in the source (checkable). A generated SQL query must parse, must not full-scan, must return non-empty (checkable) — even if "is it the query the user wanted" is not. A legal drafting task where you can check that every cited case exists in the corpus. This tier is where most applied AI product work actually lives, and the skill is decomposing a fuzzy task into a bundle of hard constraints plus one soft residual.

**Tier 4 — not verifiable.** "Is this the right tone for our brand." "Is this summary the *useful* one." "Did this support conversation leave the customer happy." Open-ended creative writing. Anything where two competent humans disagree 30% of the time. Here you are back to preference data and judges, and pretending otherwise is how you ship a model that is technically correct and commercially useless.

**⚠ Trap:** the reflex to force Tier 4 into Tier 1 by writing a rubric and having an LLM grade against it, then calling it "verifiable." An LLM judge is a reward model with a prompt instead of a trained head. It has the same overoptimization dynamics and generally *worse* calibration. Calling it a verifier does not change its physics. It is a legitimate tool — but you must budget for hacking, and you must hold out a human-labelled slice to detect it.

**🗣 Say this in the room:** "The question I ask first is 'what fraction of this task can a program decide?' If it's above about 80%, I run RLVR on the checkable part and use a judge only for the residual. If it's below 30%, RLVR is the wrong tool and I'd reach for preference optimization or, honestly, better prompting and retrieval first."

### Design the reward function for a math RLVR run. Be specific about exactly what your function returns.

The mental model that keeps you out of trouble: the reward function is a *specification of the task*, and every term you add is a clause in a contract the model will read adversarially, like a lawyer. Add fewer clauses than you think you need.

For competition-style math I return a scalar in `[0, 1]` composed of at most two terms:

```python
def reward(prompt, completion, gold) -> float:
    # 1. Format gate — hard, binary, no partial credit.
    m = re.search(r"<answer>(.*?)</answer>", completion, re.S)
    if m is None:
        return 0.0
    pred = m.group(1).strip()

    # 2. Correctness — symbolic equivalence, not string equality.
    if math_equal(pred, gold):        # normalize + sympy simplify + numeric fallback
        return 1.0
    return 0.0
```

That is it. Binary correctness behind a format gate. Note what I did *not* do:

- No partial credit for "close" answers. Numeric proximity on a math answer is meaningless — `1/2` and `1/3` are equally wrong, and rewarding closeness teaches the model to guess plausible-looking magnitudes.
- No length bonus. Ever. That is a direct invitation to pad.
- No "reasoning quality" score from a judge on top. If you add a 0.2-weight judge term to a 1.0-weight correctness term, the model will find the judge's blind spot long before it finds a better proof strategy, because the judge term is dense and the correctness term is sparse.
- No reward for using a particular method. You are not qualified to know the best method; that is what the RL is for.

The format gate deserves a note. In R1-Zero-style training the format reward exists to make the answer *extractable*, not to make the output pretty. I implement it as a gate (fail → 0) rather than an additive bonus, because an additive format bonus creates a local optimum where the model emits perfect tags around garbage and collects free reward, which is exactly the zero-signal group problem discussed later. Once the model reliably emits the tags — usually within 50–100 steps — the gate is effectively inert and you have a clean binary correctness signal.

**⚠ Trap:** string-comparing the answer. `math_equal` must handle `\frac{1}{2}` vs `0.5` vs `1/2`, `\dfrac` vs `\frac`, `\left(` wrappers, trailing `\,`, `\text{ cm}` units, `π` vs `\pi`, `2\sqrt{3}` vs `\sqrt{12}`, and `(3, 4)` vs `(3,4)`. I have seen a run where 18% of correct answers were scored wrong because of LaTeX whitespace, which silently caps the achievable reward and — worse — teaches the model to prefer whichever surface form happens to pass. Your normalizer is part of your reward function whether you think of it that way or not.

**📐 Numbers you must know:** a well-built math verifier should agree with careful human grading on ≥99% of a 500-sample audit set. Below ~97% and the label noise starts competing with the learning signal: at a 60% pass rate, a 3% false-negative rate on correct answers means 3% of your positive gradient is being deleted and 3% is being applied in the wrong direction — roughly a 6% corruption of the signal on the exact samples that matter most.

### Why not just do rejection-sampling fine-tuning on verifier-passing samples? What does RL actually buy you over that?

This is the highest-value question in the section because the honest answer is "less than people think, and you should try the simple thing first."

Rejection-sampling SFT — sample k completions, keep the ones the verifier passes, fine-tune on them — is the same signal delivered through a simpler pipe. It is what STaR and ReST-style methods do, and it is what DeepSeek-R1 itself used as one of its four stages. Mechanically it is a crude policy-gradient step: you are upweighting high-reward trajectories, just with a hard 0/1 filter instead of a continuous advantage, and with no penalty on the failures.

What RL adds, concretely:

**Negative gradient.** Rejection sampling only tells the model what to do more of. Policy gradient also pushes *down* on the trajectories that failed. When the failure mode is a specific seductive-but-wrong pattern — a common algebraic slip, calling the wrong tool, a plausible-looking off-by-one — the down-weighting is the whole game. Rejection SFT can never unlearn it; it can only dilute it.

**Relative credit within a group.** GRPO's advantage is `(r_i − mean(r))/std(r)` within a group of samples from the *same prompt*. That normalizes away prompt difficulty. Rejection SFT weights every passing sample equally, so easy prompts — which produce the most passing samples — dominate your fine-tuning batch. You end up training hardest on what the model already does best.

**On-policy freshness.** RL resamples from the current policy every step. Rejection SFT samples once, then does multiple epochs on a now-stale dataset, so by epoch 3 you are doing off-policy imitation of a model you no longer are.

**Continued improvement past the first plateau.** In practice rejection SFT gives you a large, fast, cheap jump — often most of the total gain — and then flattens because the sampling distribution has collapsed onto what already worked. RL keeps grinding.

**💰 Math:** a rejection-SFT pass over 50k prompts at k=8, 2k tokens each, is 800M generated tokens. At ~8k tok/s aggregate on an 8×H100 node that is `800e6 / 8000 = 100,000 s ≈ 28 wall-clock hours`, i.e. 222 GPU-hours, ≈ **$445** at $2/GPU-hr — plus a 2-hour SFT job. Call it under $1,000 and one engineer-day, done in a day and a half. A 500-step GRPO run on the same prompt pool at group size 8 generates roughly 6.1 *billion* tokens — about 8× the decode, but spread across 500 sequential steps, so it is weeks of wall-clock rather than a day, plus a dedicated engineer for the debugging. **The rule I enforce in review is: do rejection-sampling SFT first, always, and only escalate to RL when the rejection-SFT curve has visibly flattened and you have an eval sensitive enough to see the next 3 points.**

**🗣 Say this in the room:** "Rejection-sampling fine-tuning is 80% of the benefit for 2% of the operational cost, and it is the correct first move. RL earns its keep through the negative gradient, difficulty-normalized credit, and staying on-policy — which matters most once you're past the easy gains and you're fighting specific failure modes rather than general incompetence."

### Derive the policy gradient for me from scratch. I'll stop you if you hand-wave.

Start from what we want. We want to maximize the expected reward of trajectories sampled from our policy:

```
J(θ) = E_{y ~ π_θ(·|x)} [ R(x, y) ]
```

The problem is that θ appears in the *sampling distribution*, not in the thing being averaged, so you cannot just differentiate the integrand. Write it out:

```
∇_θ J = ∇_θ Σ_y π_θ(y|x) R(x,y) = Σ_y R(x,y) ∇_θ π_θ(y|x)
```

That gradient is not an expectation — you cannot estimate it by sampling, because `∇π` is not a probability distribution. The trick is the log-derivative identity `∇π = π ∇log π`:

```
∇_θ J = Σ_y π_θ(y|x) R(x,y) ∇_θ log π_θ(y|x) = E_{y ~ π_θ} [ R(x,y) ∇_θ log π_θ(y|x) ]
```

Now it *is* an expectation over samples from the policy, so Monte Carlo works: sample G completions, compute `(1/G) Σ_i R_i ∇log π_θ(y_i|x)`. That is REINFORCE. Intuitively: take the gradient that makes this trajectory more likely, and scale it by how good the trajectory was.

For an autoregressive LM, `log π_θ(y|x) = Σ_t log π_θ(y_t | x, y_<t)`, so the estimator is a sum of per-token log-prob gradients, each scaled by the *same* sequence-level reward. This is the sentence to say out loud, because it is the origin of the credit-assignment problem: **a token emitted at position 3 and a token emitted at position 3000 receive identical credit for the outcome.**

Two more moves get you to something usable. First, **baselines**: for any function `b(x)` that does not depend on `y`, `E[b(x) ∇log π(y|x)] = b(x) ∇_θ Σ_y π(y|x) = b(x) ∇_θ 1 = 0`. So you can subtract any such baseline for free — it does not bias the estimator, and a good one cuts variance enormously. PPO learns `b(x)` as a value network; GRPO uses the group mean, which is unbiased-by-construction and free. Second, **off-policy correction**: if you take multiple gradient steps on one batch of rollouts, the data is no longer from `π_θ`, so you reweight by the importance ratio `π_θ/π_old` and clip it to stop the ratio from exploding. That is PPO's surrogate objective, and GRPO inherits it verbatim.

**⚠ Trap:** thinking the baseline must be *accurate* to be valid. It must only be independent of the sampled `y`. An inaccurate baseline gives you a high-variance but still unbiased estimator; a baseline that peeks at `y` (for example, normalizing by statistics computed from the sampled group *including* the sample itself) introduces a small bias. That bias is why leave-one-out baselines exist, and why the `std` division in vanilla GRPO is contested.

### If RL only reinforces trajectories the model already produces, where does new behavior come from?

This is the conceptual question that separates people who have run these jobs from people who have read about them, and the answer is: **new behavior comes from recombination under sampling temperature, not from invention.**

The mechanism. At temperature 1 with a 100k-token vocabulary and a 4,000-token completion, the policy is sampling from an astronomically large space of trajectories. Any given behavior the model exhibits at, say, 2% frequency — "let me check this by substituting back" — appears in roughly 1 of every 50 samples. With group size 8 across 1,024 prompts you draw 8,192 samples per step, so that behavior appears ~160 times per step. If, conditional on that behavior appearing, the pass rate is 65% instead of the baseline 50%, then trajectories containing it get a systematically positive advantage and the policy shifts probability mass toward it. Repeat 500 times and a 2% behavior becomes a 60% behavior.

So RLVR is a **search-and-amplify** loop over the base model's existing behavioral repertoire, run at enormous sample volume. The "emergent" long chain-of-thought in R1-Zero is not the model inventing self-verification from nothing; it is the model discovering, empirically, that trajectories in which it backtracks and re-checks have higher pass rates, and that discovery being burned into the weights. The base model already knew how to say "wait, let me reconsider" — pretraining is full of humans doing that. RL found out it pays.

Two consequences you should state, because they are the ones interviewers probe.

First, **the base model is the ceiling and the search prior**. If a behavior has probability ~0 in the base model, no amount of RLVR will find it, because you never sample it and therefore never get gradient toward it. This is why "which base model" is the single highest-leverage decision in an RLVR project, and why identical recipes give wildly different results on Qwen bases vs Llama bases at the same size.

Second, **exploration is controlled by entropy, and entropy monotonically decreases under this loop**. Every step you sharpen toward what worked, which shrinks the search space for the next step. Left alone, the run self-terminates: entropy → 0, all G samples in a group become identical, advantages become 0, gradient becomes 0. That is not a bug you fix at the end; it is the central dynamic you manage from step 1.

**🗣 Say this in the room:** "RLVR doesn't teach; it selects. It runs a very wide sampling search over behaviors the base model already has non-trivial probability of producing, and amplifies the ones that correlate with passing the verifier. That framing predicts the two things you actually observe: base-model choice dominates the outcome, and entropy collapse is the default failure mode rather than an unlucky one."

### pass@1 versus pass@k — which do you train on, which do you report, and where do they diverge?

The distinction is the whole reason the "does RL create capability" argument exists, so treat it as load-bearing rather than as a metrics footnote.

`pass@1` is the probability a single sample is correct — for a fixed prompt, it is just the expected reward. `pass@k` is the probability that *at least one* of k samples is correct. The standard unbiased estimator, given n samples of which c are correct, is `1 − C(n−c, k)/C(n, k)`; you always compute it with a large n (say 64) and evaluate at several k, never by literally drawing k samples once.

Here is the divergence, and it is not subtle. RLVR reliably raises pass@1 by a lot. At large k — k = 128, 256 — RLVR-trained models often match or *underperform* their own base model. The mechanism is exactly the entropy story: RL concentrates mass on the modes that work, which raises the chance that your one sample lands on a good mode, and lowers the chance that 256 samples cover a wide enough region of solution space to stumble onto the hard problem's unusual solution. You have traded coverage for precision.

**📄 Paper:** Yue et al. (2025), *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* — reported that across math and code benchmarks the pass@k curves of RLVR models cross below their base models at large k, arguing RLVR sharpens the sampling distribution rather than expanding the reachable solution set. **📅 Volatile:** this is a live argument; work on prolonged RL with entropy control (NVIDIA's ProRL line, 2025) reports boundary expansion under long enough training with sufficient exploration pressure. Verify the current state before you assert either side as settled.

What I actually do:

- **Train on pass@1** (binary per-sample reward). Training on a pass@k objective directly is possible but couples samples within a group in ways that complicate the advantage.
- **Report both**, at k ∈ {1, 4, 16, 64}, base vs trained, on the same eval. The k=1 line is your product metric. The k∈{16,64} lines are your *diversity health check* — if they degrade sharply, you have overcooked the run and you have lost the ability to do best-of-n or agentic retry at inference, which for an applied team is often worth more than the pass@1 point you gained.
- **Watch the crossover point.** If base beats trained above k=16, and your product does best-of-8 at inference anyway, your RL run has net *hurt* the product. I have seen this shipped.

**⚠ Trap:** evaluating a reasoning model at temperature 0 and reporting that as pass@1. Greedy decode on a long-CoT model is not representative and is unusually prone to repetition loops; the standard is temperature ~0.6 with top-p ~0.95, averaged over multiple seeds (often reported as avg@n or maj@n). A single-seed temperature-0 number is not a measurement, it is an anecdote — and interviewers at labs will call it out.

### What does an RLVR run actually do to the model's output distribution? Be concrete.

Four things happen, and you should be able to name all four with the observable that tracks each.

**Entropy falls.** Per-token policy entropy — average `−Σ_v p_v log p_v` over generated positions — drops steeply in the first few hundred steps and then asymptotes toward a small value. This is the master variable. I plot it on every run, and I treat a fall below roughly 20–30% of the starting value as a red flag that exploration is dead. It is not a metric you glance at post-hoc; it is the thing you steer.

**Response length changes, direction depending on the task.** On competition math with no length penalty, length typically *grows* — often 2–4× over a run — because longer trajectories with backtracking and verification genuinely pass more often. On tasks where the answer is short and the verifier is strict, length can collapse to near-minimal. Both are real; neither is inherently wrong. What is wrong is length growing while pass rate is flat, which means you are paying for tokens that do nothing.

**Style homogenizes.** The model converges on one solution template — same opening phrase, same section markers, same self-check ritual. Sample 20 completions from a step-500 checkpoint and they will look like variations on one document. This is what "mode collapse" means concretely, and it is why an RLVR'd math model often feels worse at general chat: the distribution over *how to write* narrowed along with the distribution over *what to conclude*.

**Off-task capability drifts.** Instruction-following, formatting compliance, multilingual behavior, and refusal calibration all wander, because nothing in your reward function constrains them. R1-Zero's headline pathologies were exactly this — language mixing mid-answer and poor readability — which is why DeepSeek added a cold-start SFT stage and a language-consistency reward in R1 proper.

**🔍 Failure taxonomy — read these four traces in this order when a run looks off:**
1. Entropy near zero and group rewards identical → exploration is dead; you are computing zero gradients. Raise sampling temperature, raise clip-higher, add dynamic sampling, or restart from an earlier checkpoint with more entropy headroom.
2. Reward rising, held-out eval flat or falling → verifier is being hacked, or your training prompts leaked into eval. Audit high-reward samples by hand — 30 of them, personally, not via a script.
3. Reward flat from step 0, entropy normal → prompts are too hard (everything fails) or too easy (everything passes). Check the per-prompt pass-rate histogram; you want mass in the 0.2–0.8 band.
4. Reward rising, length rising 3×, eval up 1 point → length exploitation. Add an overlong penalty or a token budget and re-measure.

**📐 Numbers you must know:** the healthy training-prompt difficulty band is a per-prompt pass rate of roughly 0.2–0.8 at your group size. At pass rate 0 or 1, a group of G samples has zero reward variance, hence zero advantage, hence zero gradient — that prompt contributed compute and nothing else. At G=8 and a per-prompt pass rate p, the probability the whole group has identical outcomes is `p^8 + (1−p)^8`: at p=0.5 that is 0.008 (harmless), at p=0.9 it is 0.43, and at p=0.95 it is 0.66. Two-thirds of your batch wasted, purely from prompt difficulty.

### How do you select and maintain the training prompt set for an RLVR run?

The prompt set is your curriculum and it is a live object, not a static file. I treat it the way I'd treat a sharded dataset with a hot-set: sourced, filtered, difficulty-labelled, deduped, contamination-checked, and re-scored as the policy improves.

**Sourcing.** For math: competition archives, synthetic problem generators with programmatically-known answers, and problems back-translated from solutions. For code: repository-mined function+test pairs, competitive programming archives, and synthesized tasks where you generate the tests first and the reference solution second. The non-negotiable requirement is that every prompt ships with a machine-checkable gold artifact, and that artifact must be verified *by running the verifier against the reference solution* before the prompt enters the pool. Any prompt whose own gold answer fails your verifier is a verifier bug or a bad label; either way it poisons training.

**Difficulty labelling.** Before training, sample the *base* model n=16 times per prompt and record the pass rate. Bucket into impossible (0), hard (0.06–0.25), medium (0.25–0.75), easy (0.75–0.94), trivial (1.0). Drop trivial and impossible from the training mix entirely — they generate zero-variance groups and therefore zero gradient. Keep a small tail of hard prompts for headroom.

**Re-scoring during the run.** Difficulty is relative to the current policy, and the current policy is improving. A prompt that was 0.3 at step 0 is 0.9 at step 400 and is now dead weight. I re-score the pool every ~100 steps using the rollouts I am already generating — it is free, they're already in the buffer — and demote prompts whose running pass rate exceeds ~0.9. This is the cheap version of curriculum learning and it materially improves effective gradient per GPU-hour.

**Deduplication and contamination.** Exact-dedup on normalized problem text, then near-dedup by embedding with a high threshold. Then check every eval benchmark you intend to report against the training pool — n-gram overlap plus embedding nearest-neighbour, manually reviewing the top matches. Competition math corpora are riddled with near-duplicates of AIME and MATH problems. **If you report an AIME number after training on a pool you did not decontaminate, you have reported nothing, and at a frontier lab that is a credibility-ending mistake rather than an oops.**

**💰 Math:** difficulty filtering is the single cheapest optimization in RLVR. If 40% of your pool is trivial or impossible, 40% of your rollout compute produces zero gradient. On a run costing $18 per step in rollout (8×H100 at $2/GPU-hr, ~68 min of decode per step at 32.7M tokens and ~8k tok/s aggregate), 500 steps is $9,000, and you have thrown away $3,600 of it. One afternoon of pre-scoring recovers that.

### Your PM wants a "quality" reward for customer-support responses with no ground truth. Talk me through your options.

I push back on the framing first, then give them a path — and the pushback is the senior move, not obstruction.

**The pushback.** "Quality" is not one thing, and RL will optimize the specific thing you write down, not the thing you meant. So the first deliverable is not a reward function; it is a decomposition of what makes a support response good into (a) hard constraints a program can check, (b) properties a judge can check with acceptable agreement against humans, and (c) an irreducible residual we will not optimize and will instead monitor.

**Tier (a) — programmatic, and there is far more here than people expect.** Did the response cite only KB articles that exist? (lookup) Did every factual claim have a supporting retrieved span? (NLI/span-attribution check, itself imperfect but checkable) Did it call the refund tool with an amount matching the order record? (compare to DB) Did it stay under the length limit, avoid promising anything in the forbidden-promises list, escalate when the intent classifier says escalate, and produce valid JSON for the ticket-update payload? Each of these is a deterministic check, and bundled together they cover an enormous fraction of what "bad response" means in practice.

**Tier (b) — judge with guardrails.** For helpfulness and tone, use an LLM judge, but: score *pairwise against a reference response* rather than absolutely (absolute scores drift and compress), use a rubric with concrete anchors, randomize position, and — this is the part people skip — hold out 300 human-labelled examples and measure judge-human agreement before you let the judge touch a gradient. If Cohen's κ against your humans is below ~0.6, the judge is not ready and you are about to optimize noise.

**Tier (c) — outcome signals from production, if you have them.** Did the customer reply "that didn't help"? Was the ticket reopened within 72 hours? Did it escalate to a human? These are the closest thing to real reward that exists, they arrive with days of delay, and they are confounded to hell. I would use them as an *offline eval* and a monitoring signal, not as an online RL reward, unless the volume is enormous.

**My actual recommendation in this scenario, nine times out of ten:** do not run RL. Run the tier-(a) checks as a *guardrail and eval suite*, fix the prompt and the retrieval against it, and if there is still a gap, collect preference pairs on the residual and run DPO. RLVR earns its complexity when you have a dense, cheap, trustworthy verifier and a task where the model's ceiling is genuinely limited by search rather than by knowledge. Customer support is usually limited by retrieval quality and policy specification, and no amount of RL fixes a knowledge base that is wrong.

**🗣 Say this in the room:** "Before I write a reward I'd split the task into what a program can check, what a judge can check with measured human agreement, and what neither can. In support, the programmatic tier — citation validity, tool-argument correctness, policy-violation detection, schema compliance — covers most of the failure mass. I'd ship that as an eval first, because you cannot optimize a metric you haven't built, and half the time building it reveals the problem is retrieval, not the policy."

### For a code-generation model, would you reach for DPO or RLVR? Defend the choice.

RLVR, and it is not close — *provided* I have a test-execution harness. The reason is that code is the domain with the strongest verifier and the weakest preference signal, and those two facts both point the same way.

The preference signal is weak because human annotators comparing two code completions are slow, expensive, and frequently wrong. They prefer readable code over correct code; they cannot execute either in their head; and inter-annotator agreement on "which of these two implementations is better" for anything nontrivial is poor. So a DPO dataset for code is expensive to build and noisy once built. Meanwhile `pytest` costs 200ms and is never wrong about whether the tests pass.

The verifier is strong because correctness is decidable against a test suite. That gives you a dense supply of training signal at near-zero marginal cost — you can generate 8 samples per prompt, run 8 test suites in parallel, and get 8 exact labels, forever, for as many prompts as you have tests. No annotation budget appears anywhere in that loop. That is the structural asymmetry.

The one situation where I would use DPO on code: when the thing you want is *not* correctness. Style adherence to a house convention, preferring a particular library, comment density, matching a codebase's idioms — none of that is verifiable, all of it is a legitimate preference target, and DPO on a few thousand pairs will get you there in an afternoon. In a real pipeline these compose: RLVR for correctness, then a light DPO or SFT pass for style, then re-check correctness on a held-out suite to confirm the style pass did not regress it.

**⚠ Trap:** "we'll just use the verifier to *build* DPO pairs — chosen = passes, rejected = fails — and run DPO, no RL infra required." This is a real and reasonable technique and it works. But understand what you gave up: DPO's implicit reward is a *relative* comparison within a pair, so a pair of (barely-passing, barely-failing) gets the same gradient as (elegant-correct, catastrophically-wrong). And it is off-policy after the first epoch. It is the cheap 70% option — I would call it out explicitly as such rather than pretending it is equivalent, because an interviewer who has run both will know.

**💰 Math:** the annotation asymmetry is the argument. 20,000 human-labelled code preference pairs at 8 minutes each and $30/hr fully-loaded is 2,667 hours ≈ $80,000 and roughly 3 calendar months of vendor coordination. the verifier side of a 500-step run at a 512-prompt batch and G=8 is `512 × 8 = 4,096` executions per step at ~0.5 CPU-seconds each = `0.57` CPU-hours per step; at $0.04/vCPU-hr that is **$0.023 per step, about $12 for the entire run**, and it arrives in days rather than months. The annotation budget and the verifier budget are not in the same universe.

### What role does the KL penalty play in RLVR, and would you ever drop it?

Mental model: the KL term to a frozen reference is a leash. In classic RLHF it is load-bearing because the reward model is only trustworthy near the reference distribution — wander too far and you are optimizing the RM's extrapolation error. In RLVR the verifier is trustworthy *everywhere*, so the leash's original justification largely evaporates, and the question becomes empirical rather than principled.

What the KL term still buys you in RLVR:

- **Preserving off-task behavior.** Your reward says nothing about instruction-following, safety refusals, multilingual competence, or tone. The KL is the only thing preventing those from drifting, because it penalizes moving away from the reference everywhere, including on behaviors you never rewarded.
- **Numerical stability.** A high-β run is a slower, tamer run. It reduces the chance of a step that blows the policy up.

What it costs you:

- **Memory and compute**: a full extra forward pass over every rollout token through a frozen copy of the model. For a 7B in bf16 that is 14 GB of resident memory plus roughly a 15–25% throughput tax on the training step.
- **A hard ceiling on improvement.** The KL term actively fights the thing you are trying to do. On long runs where you want large behavioral change — and R1-Zero-style emergent long CoT *is* a large behavioral change — a meaningful β caps how far you can get.

**The current practice, and it is genuinely contested, so say so:** several strong 2025 reasoning-RL recipes drop the KL penalty entirely (DAPO explicitly removes it, on the argument that during long-CoT RL the policy is *supposed* to move far from the initial model). Others keep a small β. My decision rule: **drop KL when the checkpoint you are training is a specialist you will serve behind a router and whose only job is the verified task, and keep it when the checkpoint must remain a general-purpose assistant.** If I drop it, I replace the safety it provided with an explicit capability-regression suite — instruction-following, safety refusals, general knowledge, multilingual — run every 50 steps, and I treat a regression there as a stop condition. You do not get to remove a guardrail and not replace the monitoring.

**⚠ Trap:** reporting "KL" without saying which estimator and which reference. The naive `log(π/π_ref)` sample estimate is unbiased for the KL but extremely high-variance and can go negative; the commonly-used low-variance non-negative estimator is `r − log r − 1` where `r = π_ref/π` (the "k3" estimator from Schulman's approximating-KL note). Two teams comparing "our KL was 12" with different estimators, different reference checkpoints, and one summing over tokens while the other averages, are comparing nothing at all. Fix the estimator and the reduction, and write them down in the run config.
