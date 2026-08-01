### Compare PPO, DPO and GRPO for me — but in GPU-hours and engineer-weeks, not in math.

This is the question the section exists for, so let me be concrete. Assume a 7B policy, a 20,000-pair preference dataset, and H100s at ~$3/GPU-hour (**📅 Volatile** — recompute with your rates; the ratios are what's durable).

**PPO.**
*Memory:* policy 112 GB + value 112 GB + ref 14 + RM 14 ≈ 252 GB of state → 8× H100 minimum, realistically 8–16.
*Wall-clock:* generation dominates. Say 512 prompts × 512 output tokens per iteration = 262k generated tokens; at a realistic 2,500 tok/s aggregate for a 7B under continuous batching on the rollout GPUs, that's ~105 s of generation per iteration, plus RM/ref/value scoring (~30 s) plus the optimization phase (~25 s) ≈ **160 s/iteration**. A 2,000-iteration run is 320,000 s ≈ **89 hours**. On 8 GPUs: 8 × 89 × $3 = **$2,136 per run.**
*Engineer-weeks:* this is the real number. Standing up a PPO stack that doesn't diverge — rollout/train weight sync, log-prob consistency, critic warmup, KL controller, the dashboard from the previous question — is **4–8 engineer-weeks for a team that hasn't done it before**, and you should budget 3–6 failed runs before the first good one. Call it **6 weeks and ~$10k of GPU**, most of it burned on failures.

**DPO.**
*Memory:* policy 112 GB + ref 14 GB = 126 GB, and you can precompute reference log-probs offline to drop to 112 GB. With LoRA you're under 40 GB and it fits on one H100.
*Wall-clock:* it's supervised training. 20,000 pairs × 2 sequences × ~700 tokens = 28M tokens per epoch, 3 epochs = 84M tokens. Training FLOPs ≈ 6·N·T = 6 × 7e9 × 8.4e7 = 3.5e18. At 8 GPUs × 400 TFLOP/s × 40% MFU = 1.28e15 FLOP/s effective, that's ~2,750 s ≈ **46 minutes**. Cost: 8 × 0.77 × $3 = **$18.** With LoRA on one GPU, ~4 hours = **$12**.
*Engineer-weeks:* **1–2**, most of it data curation, because the training loop is `DPOTrainer(...)` and it works.

**GRPO.**
*Memory:* policy 112 + ref 14 + RM/verifier 14 = 140 GB → 4× H100 comfortably. No critic.
*Wall-clock:* still generation-bound and actually generates *more* than PPO per prompt (G responses per prompt, typically G=4–16). 128 prompts × 8 responses × 512 tokens = 524k tokens per iteration, ~210 s of generation, but no value forward/backward: total ≈ **250 s/iteration**. 1,000 iterations ≈ 69 hours on 4–8 GPUs = 8 × 69 × $3 ≈ **$1,656.**
*Engineer-weeks:* **2–4.** Meaningfully simpler than PPO because there's no critic to tune, no GAE, no explained-variance debugging — you deleted the component that caused most of PPO's instability. But it's still an RL loop with generation in it, so you still own the rollout infrastructure.

**💰 The summary table in numbers:**

| | GPU cost/run | Wall-clock/run | Min GPUs | Eng-weeks to first good run | Failed runs expected |
|---|---|---|---|---|---|
| DPO | ~$20 | <1 h | 1–8 | 1–2 | 0–1 |
| GRPO | ~$1,700 | ~70 h | 4 | 2–4 | 1–3 |
| PPO | ~$2,100 | ~90 h | 8 | 4–8 | 3–6 |

**The number that decides it is not the GPU cost — it's the engineer-weeks and the failed-run count.** PPO's GPU bill is 100× DPO's, but $2,100 is nothing at these companies. Six engineer-weeks of a senior AI engineer is $30–60k of fully-loaded cost and, more importantly, six weeks of calendar time you cannot buy back. That is the operational argument, and stating it in that form is what the interviewer is listening for.

**⚠ Trap:** presenting this as "DPO is better." It isn't better; it's *cheaper per unit of iteration*, and it does something slightly different (offline, off-policy, on a fixed dataset). The right framing is that DPO moved the bottleneck from *engineering* to *data*, and for most teams data was already the bottleneck, so DPO's constraint was free.

### So — why did PPO actually lose? Give me the real reason, not "DPO is simpler."

"Simpler" is the conclusion, not the mechanism. Four specific things lost it, and I'd name them in this order.

**1. It has two coupled learning systems, and one of them is invisible.** The policy and the critic train simultaneously, each on data produced by the other. When the critic is bad, advantages are noise; when advantages are noise, the policy wanders; when the policy wanders, the critic's targets shift. *And none of this appears in the reward curve.* You need explained variance, clip fraction, KL and entropy on a dashboard to even see it. DPO has one loss and one model. The number of ways a DPO run can be subtly wrong is maybe five; for PPO it's fifty.

**2. It requires an online generation loop, which is a distributed-systems problem, not an ML problem.** You need an inference engine and a training engine holding the same weights in different parallelism layouts, synchronized every iteration. That's a whole subsystem — weight broadcast, KV cache invalidation, handling stragglers, keeping the trainer fed while the sampler runs. DPO is a dataloader and a loss function. **This is the part backend engineers underestimate least and ML engineers underestimate most**, and it's why the frameworks that made RL practical (OpenRLHF, veRL) are fundamentally *orchestration* projects.

**3. The reward model is a second model with its own failure modes, and it goes stale.** PPO needs a trained RM plus the iterative re-collection loop to keep it valid against a moving policy — 6+ calendar weeks per iteration cycle, gated on an annotation vendor. DPO's key structural insight is that the *implicit* reward β·log(π/π_ref) never needs to be trained or maintained. You deleted an entire model, an entire training job, and an entire ongoing data-ops obligation.

**4. Hyperparameter fragility with an expensive feedback loop.** PPO has ~4 knobs that matter, they interact, and each evaluation costs 90 hours. DPO has essentially β and learning rate, and each evaluation costs 45 minutes. **The ratio of experiments-per-week is 100:1**, and in a field where nobody knows the right answer in advance, iteration speed *is* capability.

**📄 Paper:** Rafailov et al. (2023), *Direct Preference Optimization*. Its contribution was showing that the KL-regularized RLHF objective has a closed-form optimum, and that you can invert that form to express the reward in terms of the policy and reference — eliminating the reward model and the RL loop entirely, replacing them with a classification-style loss on preference pairs. It replaced the RM-plus-PPO stage of InstructGPT for the large majority of practitioners within about a year.

**🗣 Say this in the room:** "PPO lost on operations, not on math. It couples two learning systems whose interaction is invisible in the reward curve, it requires an online generation loop that's a distributed-systems project in its own right, it needs a reward model that goes stale against a moving policy, and every hyperparameter experiment costs ninety hours instead of forty-five minutes. DPO didn't beat it on quality at the frontier — it beat it on experiments per week, and at the frontier the labs kept RL anyway, they just moved to variants that dropped the critic."

**⚠ Trap:** concluding that RL itself lost. It didn't — the *value-model-based, reward-model-driven* variant did. Reasoning models are trained with RL. Agentic post-training is RL. GRPO and its descendants are RL. What actually happened is that the field kept RL's online, on-policy structure and deleted its two most expensive components: the critic (replaced by a group baseline) and, wherever possible, the learned reward model (replaced by a verifier).

### When would you still reach for PPO in 2026?

There are real cases, and being able to name them is what stops you sounding like you've only read blog posts.

**When you need a dense, shaped, per-token reward.** GRPO assigns one advantage to an entire trajectory. If your reward genuinely varies within a response — a safety classifier that fires on span 3, a factuality checker per claim, a per-step process reward — you need machinery that can attribute at the token level, and GAE plus a critic is that machinery. This is the strongest technical case for PPO and it's underappreciated.

**When rollouts are extremely expensive and you must extract maximum signal per sample.** Long-horizon agent trajectories where a single rollout is a 15-minute sandboxed run costing real money. GRPO needs G responses *per prompt* to form a baseline — that's a G× multiplier on your most expensive resource. A critic learns a baseline from *all* your data, amortized. If your rollout is a full SWE-bench-style attempt, paying 8× for a group baseline is a serious cost, and a learned critic can be the right call.

**When you have a strong, well-calibrated reward model already in production** and the marginal cost is just running it. If someone else already paid the RM cost, PPO's marginal disadvantage shrinks a lot.

**When you need fine-grained KL control per token.** The explicit per-token KL penalty inside the reward is a control surface that DPO's implicit β doesn't give you in the same way — you can make the penalty state-dependent, tighter in some regions than others.

**Where I'd honestly *not* use it:** first alignment pass on a new model; any team without a standing RL infrastructure; anything on a deadline shorter than a quarter; anything where a verifiable reward exists (use RLVR); anything where a static preference dataset is sufficient (use DPO). That's most cases, which is why the honest default is "not PPO, and here's the specific condition that would change my mind."

**🗣 Say this in the room:** "PPO earns its keep when the reward is genuinely dense — token-level or step-level — because the critic plus GAE is the only mechanism in this family that does per-token credit assignment. For a sequence-level scalar reward, the critic is buying you precision it can't actually deliver, and GRPO's group baseline is both cheaper and empirically better. So my question is always: is my reward sequence-level or token-level? That single answer routes the decision."

### Design the distributed system for a PPO run on a 70B policy. Where do the GPUs go?

This is the systems question and it's where a backend engineer should shine, because it's a resource-scheduling and consistency problem dressed in ML clothes.

**The core tension:** the rollout phase wants an inference layout (tensor parallelism sized for decode latency, paged KV cache, continuous batching, weights in bf16, no optimizer state). The training phase wants a training layout (FSDP/ZeRO-3 sharding of weights + gradients + optimizer state across all ranks). The same parameters need to exist in both forms, and one has to be refreshed from the other every iteration.

**Two architectures:**

**Colocated (hybrid engine).** Rollout and training share the same GPUs, time-multiplexed. Before generation, you offload optimizer state to CPU/NVMe and materialize inference weights; after generation you reverse. *Pro:* maximum GPU utilization when you're memory-rich, no idle fleet, weight sync is a local memory operation rather than a network transfer. *Con:* the offload/reload is real wall-clock, and you cannot overlap generation with training at all — the phases strictly alternate. This is what you do when GPUs are scarce.

**Disaggregated.** A dedicated rollout pool running vLLM/SGLang and a dedicated training pool running FSDP, connected by a weight-sync path. *Pro:* each pool is tuned for its job; you can overlap by having the sampler produce iteration N+1's rollouts while the trainer consumes N (accepting one step of staleness); you can scale the pools independently, which matters because the right ratio is workload-dependent. *Con:* weight synchronization every iteration over the network, and idle time whenever the ratio is wrong.

**Weight sync is the crux.** For a 70B model in bf16 that's 140 GB to move every iteration. Over a 400 Gb/s interconnect that's 140e9 × 8 / 400e9 = **2.8 seconds** if you saturate it; over 25 GbE it's 45 seconds and your run is dead. The techniques that make it viable: NCCL broadcast from trainer ranks directly into the inference engine's weight buffers (rather than gather-to-host-then-scatter); sending only the sharded slice each inference rank needs; CUDA IPC / shared-memory handoff when the two engines are on the same node; and overlapping the transfer with the tail of the optimization phase. **📅 Volatile:** which of these each framework implements has moved fast; verify against the version you're pinning.

**The ratio question.** If generation is 70% of iteration time and optimization is 30%, a disaggregated split of roughly 7:3 GPUs keeps both busy — but only if you allow one step of staleness so they can overlap. Without overlap, disaggregation is strictly worse than colocation, because one pool is always idle. **That's the design insight worth stating: disaggregation only pays if you accept asynchrony, and asynchrony is off-policy, which changes your algorithm.** You cannot make this a purely infrastructural decision.

**📐 Numbers you must know for a 70B PPO run:** policy 70e9 × 16 = 1,120 GB; critic another 1,120 GB; ref + RM at 70B frozen = 280 GB. **~2.5 TB of state**, which is 32 H100s at 80 GB just to hold it, before activations or KV cache. Realistically 48–64 GPUs. At $3/GPU-hour that's $144–192/hour, so a 4-day run is **$14k–18k**. This arithmetic is exactly why 70B PPO is a frontier-lab activity and why everyone else does LoRA-PPO, GRPO, or DPO.

**⚠ Trap:** the "just use a smaller critic" instinct. It's tempting — the critic only outputs a scalar. But a critic much smaller than the policy cannot represent the value function of the policy's state space well, explained variance stays low, and you've traded memory for the exact failure mode that makes PPO unstable. If you're going to shrink the critic to save memory, you should probably just delete it and use GRPO. That's not a rhetorical flourish; it's genuinely the reasoning chain that produced GRPO.

### Which framework would you actually use, and what would you build yourself?

**Do not write your own PPO loop for production.** I have watched teams do it and the failure is always the same: the loop is 300 lines and correct-looking, and then six weeks go into the weight-sync path, the log-prob consistency bug, and the sharded checkpoint format. Write it once as a *learning exercise* on a 0.5B model — you should be able to — and then use a framework.

**TRL (HuggingFace).** The default entry point. `PPOTrainer`, `DPOTrainer`, `GRPOTrainer`, `RewardTrainer` under one API, tight integration with `peft` for LoRA and with the HF ecosystem for datasets and tokenizers. Best for: reward-model training, DPO, single-node or small-multi-node experimentation, anything up to ~8B with LoRA. Weakness historically has been large-scale PPO orchestration, and its PPO implementation has been substantially reworked across versions. **📅 Volatile:** TRL's APIs move fast and tutorials go stale within months — pin the version and read *that version's* docs, not a blog post.

**OpenRLHF.** Built for scale: Ray for orchestration, vLLM for the rollout engine, DeepSpeed ZeRO-3 for training, with the four models placed on separate resource groups. Best for: full-parameter PPO/GRPO at 7B–70B when you actually need the distributed story. The Ray-based placement is the feature — you declare how many GPUs each of the four roles gets and it wires the rest.

**veRL.** ByteDance's framework, built around a hybrid programming model that separates the *control flow* of the RL algorithm from the *dataflow* of the distributed execution, so you can write the algorithm once and remap the parallelism. Best for: teams doing algorithm research at scale who need to change the RL algorithm without rewriting the distributed layer. It has become a common substrate for RLVR/GRPO work.

**NeMo-Aligner (NVIDIA)** if you're already in the NeMo/Megatron ecosystem, where the tensor/pipeline parallelism story is mature.

**My decision rule:** TRL until you outgrow a single node or need real PPO/GRPO throughput; OpenRLHF or veRL after that; and never your own, unless the framework's abstraction is genuinely blocking the algorithm you're inventing. The thing to say in a room is *why* — "the framework's value isn't the loss function, it's the rollout/training weight synchronization and the resource placement, which is where all the engineering time actually goes."

**What I would still build myself:** the reward function (always — it's your product's spec, expressed as code), the verifier and its sandbox, the eval harness and gold-eval loop, the dashboard from the failure-taxonomy question, and the data pipeline. Those are where your differentiation lives. The RL algorithm is a commodity.

### Should the preference labels come from humans or from a model? Price both.

RLAIF — using a strong LLM to produce the preference labels that a reward model trains on — is the default starting point in 2026 for anything that isn't safety-critical or expert-domain, and the reason is a cost ratio that isn't close.

**💰 The arithmetic for 50,000 preference pairs:**

*Human, general-purpose crowd:* ~$1.00–1.50 per pairwise judgment at a vendor, plus overlap for agreement (10% triple-labeled adds ~20%), plus adjudication of disagreements, plus guideline development and pilot rounds. Call it **$75,000–90,000** and **4–8 weeks** of calendar time including the pilot.

*Human, expert domain (attorneys, senior engineers, clinicians):* $10–25 per judgment. 50,000 pairs = **$500k–$1.25M**. Nobody does 50,000 expert pairs; they do 3,000–5,000 and treat them as gold.

*AI judge, frontier model:* each judgment is ~1,200 input tokens (prompt + two responses + rubric) and ~200 output tokens with a brief rationale. At $3/Mtok in and $15/Mtok out: (1200 × 3 + 200 × 15)/1e6 = $0.0036 + $0.003 = **$0.0066 per judgment**. 50,000 pairs = **$330** and about **6 hours** with reasonable concurrency. **📅 Volatile:** verify current per-token pricing; the ratio is the point.

That is roughly a **250× cost ratio against crowd labels and 2,000×+ against expert labels**, with a 20× improvement in turnaround. There is no version of this where you don't at least *start* with AI labels.

**Where AI labels are not enough, and this is the judgment part:**

- **Expert domains where the judge doesn't have the knowledge.** A frontier model cannot reliably tell you which of two contract-review memos an M&A partner would sign. Model judges are confidently wrong in exactly the places your domain expertise is the product.
- **Anything where the judge shares the policy's blind spots.** If you're aligning a model and judging with a sibling model, correlated errors go straight into the reward model and then into the policy. Cross-family judging helps but doesn't eliminate it.
- **Safety and legal.** You want a human in the accountability chain, full stop.
- **Calibrating the judge itself.** This is non-negotiable: collect **500–2,000 human labels on a stratified sample** and measure the judge's agreement with humans, sliced by prompt category. If the judge agrees with humans at 78% while humans agree with each other at 76%, you are at the human ceiling and you should scale the judge freely. If it's 58%, the judge is not fit for this distribution and cheap labels are worse than no labels.

**The architecture I recommend:** AI judge for volume, human labels for (a) calibrating the judge, (b) the hard/ambiguous slice the judge is uncertain about — route by judge self-consistency across two orderings — and (c) the held-out gold eval set, which must be human and must never be used for training. That's a $330 + ~$3,000 program instead of a $90,000 one, and it's *better*, because the human budget goes to the items where human judgment actually differs from the model's.

**📄** Constitutional AI (Bai et al., 2022) is the canonical reference for AI-generated preference labels — a critique-and-revise SFT stage followed by RL against an AI-generated preference model guided by a written set of principles. Its contribution was showing that a written constitution plus model self-critique can substitute for a large human harmlessness-labeling program.

**⚠ Trap:** running the AI-judge pipeline without order randomization. Model judges have strong position bias — often 10–20 points of preference for whichever response appears first, and it varies by model. Randomize order, and for the pairs that matter, judge both orders and keep only the consistent verdicts. Skipping that means your reward model learns your judge's positional artifact.

### Your PPO run has been going for 400 steps and the reward is completely flat. Walk me through your checklist.

Flat reward is a *better* problem than diverging reward because it usually has a mechanical cause. My order:

**1. Is the reward model actually discriminating?** Take 200 rollouts, score them, and look at the distribution. If the std of rewards within a batch is near zero, PPO cannot learn anything — whitened advantages become noise divided by noise. Causes: the RM is broken (score two obviously-different responses by hand and confirm the ordering is right); or the policy's outputs are already homogeneous (check the diversity metrics). This is the first check because it's the most common.

**2. Are the advantages reaching the loss?** Print the mean absolute advantage after whitening, restricted to unmasked tokens. If it's ~0, either your `action_mask` is zeroing everything (a common off-by-one — the mask must align with the *shifted* logits) or your whitening is dividing by a huge std.

**3. Is the gradient nonzero?** Log grad norm. If it's ~1e-8, something upstream is detached. The classic: computing `logprob_old` and `logprob_new` from the same forward pass, so ρ ≡ 1 identically, `min(A, A) = A`, and — because A is a constant with respect to θ in that formulation — the gradient vanishes. Or accidentally running the policy under `torch.no_grad()` in the optimization phase.

**4. Is the KL penalty eating the entire reward?** Compute mean(β·KL_per_sequence) and compare it to mean(r_RM). If the KL term is 5× the reward term, the optimal policy is "don't move," and your run is correctly finding it. Lower β. I've seen this exact thing when someone swapped in an RM with rewards centered near zero and a small variance against a β tuned for an RM with ±10 range.

**5. Is the learning rate effectively zero?** Check the scheduler. A cosine schedule with a warmup longer than your total steps is a real and stupid bug. Print the actual LR at step 400.

**6. Is the clip fraction zero?** If nothing is ever clipped, the policy is barely moving. Combined with a healthy grad norm, that suggests the LR is too small. Combined with a zero grad norm, see item 3.

**7. Is the reward being applied to the right token?** If you're adding r_RM at a padded position instead of the last real token, the reward lands on a masked position and contributes nothing. Assert that the index you're writing to equals `attention_mask.sum(1) - 1`.

**8. Are the rollouts even different from each other?** Print 10 generations. If they're identical, your generation is running greedily (temperature 0, or `do_sample=False` — an easy default to inherit), and with identical rollouts there is no variance to learn from. This one takes ten seconds to check and I check it early despite listing it here.

**⚠ Trap:** concluding "PPO doesn't work for our task" after one flat run. In my experience the ordering of root causes is roughly: masking/indexing bugs (35%), β or reward-scale mismatch (25%), greedy or truncated sampling in rollouts (15%), reward model not discriminating on this distribution (15%), everything else (10%). It is almost always plumbing. Verify the plumbing with a synthetic reward first — set r = −(number of tokens) and confirm responses shorten within 50 steps. If that doesn't work, your loop is broken and no amount of RM work will help.

### Design the whole preference-alignment program for a vertical AI product — say a customer-support agent for an enterprise. Budget, timeline, evals.

Let me take Sierra-style: an AI support agent for a mid-market SaaS customer, where the business metric is *resolution rate without human escalation* and the failure modes are wrong policy answers, over-promising refunds, and tone.

**Step 0 — build the eval before anything else.** 300 real conversations sampled stratified across intents, each with a human-adjudicated verdict on: resolved correctly (binary), policy-compliant (binary), tone acceptable (binary), escalated appropriately (binary). This is the gold set. It is human-labeled, it never enters training, and it is the only number anyone gets to quote. Cost: 300 × ~15 min of a support lead's time ≈ 75 hours ≈ **$5k**. Two weeks. **If a design answer doesn't open with this, it fails.**

**Step 1 — decide what's verifiable.** A surprising amount is: did the agent cite an existing KB article (checkable), did it stay inside the refund policy limits (checkable with a rules engine), did it emit a valid tool call with a real order ID (checkable), did it avoid forbidden phrases (checkable). Those become **hard gates**, not reward-model territory. The learned reward only ranks among responses that already pass. This is the single highest-leverage architectural decision in the whole program.

**Step 2 — preference data.** Sample 4 responses per conversation from the current SFT policy at temperature 1.0 across 5,000 held-out conversations. Filter through the hard gates. Rank the survivors with a frontier-model judge against a written rubric derived from the support team's actual escalation criteria. That's ~5,000 × C(4,2) = 30,000 pairs at ~$0.0066 = **$200**. Calibrate the judge against 500 human-labeled pairs from the support leads: **$2,500** and one week. If judge-human agreement is at the human-human ceiling, proceed; if not, fix the rubric and re-calibrate before spending anything else.

**Step 3 — train.** DPO first, not PPO, and I would say this explicitly and defend it. β sweep at 0.05/0.1/0.3, three runs, ~1 hour each on 8×H100 = **$72**. Evaluate each against the gold set plus a capability-regression suite (does it still do the boring things — order lookups, address changes — correctly).

**Step 4 — measure honestly.** Report: gold-set resolution rate with a confidence interval; policy-compliance violation rate against the red-team set; **length-controlled** win rate vs. the SFT baseline; and the capability-regression deltas. Never a bare win rate. With 300 gold items, a resolution rate of 72% has a standard error of √(0.72×0.28/300) ≈ 2.6 points, so a 3-point improvement is *not* significant and I would say so rather than ship it.

**Step 5 — the escalation decision.** Only if DPO plateaus below the business target *and* I can point to a specific failure the offline data can't fix — typically multi-turn recovery, where the right response depends on the model's own earlier turns and no static dataset covers that distribution — do I propose online RL. And then it's GRPO against the verifiable gates plus the calibrated judge, not PPO, because the reward here is sequence-level and I have no use for a critic.

**💰 Total program:** $5k eval + $2.7k label calibration + $200 judge labels + $72 training + ~$1k of iteration ≈ **$9k of direct cost and 6 weeks of calendar**, dominated by human eval construction and judge calibration, not by GPUs. That cost shape — humans and calendar, not compute — is the true one for applied AI companies, and getting it right in a design round matters more than any algorithm detail.

**⚠ Trap:** proposing RLHF-with-PPO in this interview. It's the reflex answer and it's wrong here: the team has no RL infrastructure, the reward is sequence-level, a static preference dataset covers the distribution, and the business needs an answer in six weeks. Reaching for PPO would be a resume-driven design decision, and interviewers at applied companies are explicitly screening for that.

### You're in a meeting with the VP of Product and the general counsel. Explain RLHF in ninety seconds.

**🗣 Say this in the room** (this is the verbatim script — practice it until it's ninety seconds):

"The model starts out as a very good autocomplete. It's read most of the internet and it can continue any text plausibly — but 'plausible continuation' isn't the same as 'a helpful answer to my question.' So we do two more steps.

First, we show it a few thousand examples of what a good answer looks like — written by our own support leads. That teaches it the *shape* of the job: answer the question, cite the policy, don't ramble. That's the cheap step and it gets us most of the way.

Second — and this is the part that's actually hard — we can't write a good answer for every situation, but our experts *can* look at two answers and say which one they'd send to a customer. So we collect thousands of those comparisons. Then we train a second model whose only job is to predict which answer our experts would pick. That's the scoring model. Finally, we let the main model practice: it writes answers, the scoring model grades them, and we nudge the main model toward the answers that score well.

The thing to know as a risk matter: the scoring model is an approximation of our experts' judgment, and if we push the main model too hard toward high scores, it starts gaming the scorer instead of getting genuinely better — it writes longer, more confident-sounding answers that our scorer likes and our customers don't. So we deliberately stop short, and we check against a set of three hundred real conversations that our support leads graded by hand and that the training process has never seen. That held-out set — not the training score — is the number I'd put in front of you."

**Why this script works:** it uses one analogy per concept, it names the failure mode before anyone asks (which is how you earn credibility with a GC), it ends on the measurement rather than the method, and it never says "reward model," "KL divergence," or "policy gradient." The last paragraph is the one that matters — a non-technical stakeholder needs to know that the training metric and the truth are different things, because that's the fact that determines whether they trust your dashboards.

**⚠ Trap:** the "it's like training a dog with treats" analogy. It's the reflex one and it's actively harmful in a stakeholder meeting, because it implies the model *wants* things, which sends the conversation straight into sentience and away from the risk you actually need them to understand. The comparison-based framing above keeps them focused on "this is an approximation of your team's judgment, and approximations have error."

### What is the alignment tax, and how do you measure whether you're paying it?

The alignment tax is the measurable degradation on general capabilities caused by preference optimization. It's not a metaphor; InstructGPT reported it directly — RLHF'd models regressed on several academic NLP benchmarks relative to the base model, and that observation is why the paper introduces PPO-ptx.

**The mechanism** is straightforward once you see it. Preference optimization moves probability mass toward a narrow region of output space — helpful, formatted, hedged assistant responses. That mass has to come from somewhere. It comes from the model's ability to produce everything else: raw code completion, terse factual recall, unusual formats, non-English, the long tail of styles the base model could produce. You're not deleting knowledge; you're making it harder to elicit. That distinction matters because it means the tax is often recoverable with prompting, and often *not* recoverable at all for behaviors the alignment data actively punished.

**PPO-ptx** — the mitigation — mixes pretraining gradients into the PPO update: the loss becomes the PPO objective plus a coefficient times the next-token loss on a batch of pretraining data. It's the same idea as replay in continual learning, and it works for the same reason: you're explicitly re-asserting the old objective while you optimize the new one. The equivalent in a DPO pipeline is mixing SFT loss into the DPO objective, which several later methods do by construction.

**How to measure it, which is the actual question:**

Build a **held-out capability suite before you start** — this is the same discipline as building the eval before the training set. Mine has four parts:
1. **Knowledge/reasoning**: an MMLU-style multiple-choice slice, scored with log-likelihood over choices, not generation (generation-scored MCQ conflates format compliance with knowledge and will show a fake improvement).
2. **Code**: a pass@1 harness on a fixed problem set with real execution.
3. **Format flexibility**: prompts requiring unusual output formats — raw CSV with no commentary, a single word, a JSON blob with no markdown fence. Aligned models are notoriously bad at "no preamble," and this slice catches it immediately.
4. **Long-tail behaviors** specific to your product: whatever your users do that isn't a chat turn.

Run it on the SFT checkpoint and every RL checkpoint. **Report the Pareto frontier**: preference win rate on the x-axis, capability-suite delta on the y-axis, one point per checkpoint. Then the tradeoff is a business decision made with data instead of an accident discovered in production.

**📐 What "acceptable" looks like:** I hold the line at **≤2 percentage points** of regression on the capability suite for a meaningful preference win. Beyond that I want an explicit sign-off, because a 5-point code-pass@1 regression in a coding product is not worth a 10-point win rate on subjective helpfulness — the win rate is measuring something users notice less than a broken completion.

**⚠ Trap:** measuring the tax only on benchmarks the alignment data resembles. If your capability suite is all chat-formatted multiple choice and your alignment data is chat, you will measure zero tax and ship a model that can no longer emit bare CSV. The format-flexibility slice exists precisely to catch the regression that in-distribution benchmarks are blind to, and it's the one I've seen catch real incidents.

### Give me a drill set. What should I be able to do unaided before I walk into an RLHF round?

Six drills, in order of increasing scope. All timed, no autocomplete, no docs — several of these companies ban AI tools in live rounds.

**🏋 Drill 1 — Bradley-Terry from scratch (10 minutes).** On paper: write the BT probability model, derive the negative log-likelihood, and state in one sentence what quantity is *not* identified by it. Then write the 6-line PyTorch reward-model loss including the accuracy metric. *Pass:* you write `-F.logsigmoid(r_w - r_l).mean()` without hesitation and you say "only within-prompt differences are identified" unprompted.

**🏋 Drill 2 — the memory budget (5 minutes).** Given "13B policy, full fine-tune, bf16 mixed precision with AdamW, PPO," compute total memory for all four models and state the minimum GPU count on 80 GB cards. *Pass:* 16 bytes/param for trainables, 2 for frozen, 13e9 × 16 × 2 (policy + critic) + 13e9 × 2 × 2 (ref + RM) = 416 + 52 = **468 GB**, therefore 8 GPUs minimum and realistically 16. You should produce this in under two minutes.

**🏋 Drill 3 — the clipped surrogate on a whiteboard (10 minutes).** Write L^CLIP. Then answer, without looking: when Â > 0 and ρ = 1.5 with ε = 0.2, which branch does `min` select and what is the gradient? When Â < 0 and ρ = 0.5? *Pass:* you get the asymmetry right — clipped when promoting too hard, unclipped when demoting — and you can say why in one sentence.

**🏋 Drill 4 — GAE implementation (15 minutes).** Write the backward-recursion GAE function with masking, from memory. Then answer: what does λ=0 give you, what does λ=1 give you, and why is γ=1 correct for LLMs? *Pass:* correct recursion, correct handling of the bootstrap value past the end, and you compute 0.99^500 ≈ 0.0066 to justify γ=1.

**🏋 Drill 5 — the incident (20 minutes, spoken).** Someone hands you: "PPO run, step 1,400, mean reward up 4.2×, human win rate down from 61% to 48%, KL at 14 nats/sequence, entropy down 30%, mean length 280 → 610." Deliver a diagnosis and a plan out loud, in five minutes, without notes. *Pass:* you name length exploitation as the leading hypothesis, cite the length number as your evidence, propose reading 30 samples first, and distinguish the *data-level* fix (rebalance chosen-length) from the *stopgap* (length penalty or earlier checkpoint) rather than conflating them.

**🏋 Drill 6 — the operational comparison (10 minutes, spoken).** "Should we use PPO, DPO or GRPO?" Answer in three minutes with GPU-hours, engineer-weeks, memory, and one named precondition that would flip your recommendation. *Pass:* you produce actual digits (four models vs two, ~250 GB vs ~126 GB for 7B, ~90 h vs <1 h, 6 eng-weeks vs 2), and your flip condition is specific — "if the reward is genuinely token-level or step-level, the critic earns its keep and I'd go PPO."

**The meta-drill:** for every one of the six, be able to state what you'd *measure* to know it worked. That's the reflex the whole guide is training, and it's the one that separates a hire from a debrief argument.
