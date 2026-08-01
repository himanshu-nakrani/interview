### Draw me the PPO setup for RLHF. How many models are in memory, what does each do, and what does it cost?

Four models, and the fact that it is four is the entire operational story of why PPO lost.

**Policy (π_θ)** — the model you are training. Trainable, so it carries gradients and optimizer state. It also has to *generate*, which means it runs autoregressive decode with a KV cache.

**Reference (π_ref)** — a frozen copy of the SFT checkpoint. Its only job is to supply log-probs so you can compute the per-token KL penalty. Forward-only, no gradients, no optimizer state.

**Reward model (r_φ)** — frozen, forward-only, scores completed sequences. Typically initialized from SFT and often the same size as the policy.

**Value model / critic (V_ψ)** — predicts the expected return from each token position, used to compute advantages via GAE. **Trainable**, so it carries its own gradients and optimizer state. In most implementations it is a full transformer the size of the policy with a scalar head — this is the expensive one that GRPO later deletes.

**📐 Numbers you must know — the memory arithmetic for a 7B policy, full fine-tune, bf16 mixed precision with AdamW.** Per trainable parameter: 2 bytes bf16 weights + 4 bytes fp32 master weights + 4 bytes Adam m + 4 bytes Adam v + 2 bytes bf16 gradient ≈ **16 bytes/param**. Frozen models are just 2 bytes/param.

- Policy: 7e9 × 16 = **112 GB**
- Value model (7B, trainable): 7e9 × 16 = **112 GB**
- Reference (7B, frozen): 7e9 × 2 = **14 GB**
- Reward model (7B, frozen): 7e9 × 2 = **14 GB**
- **Subtotal: 252 GB of weights and optimizer state**, before a single activation.

Add activations for training (with gradient checkpointing, maybe 10–30 GB depending on batch and sequence length) and the KV cache for generation (for 512 concurrent sequences at 1,024 tokens on a 7B model with GQA, order of 10–20 GB). Call it **~290–300 GB**. On 80 GB H100s that is a minimum of 4 GPUs with ZeRO-3 sharding, realistically 8 to have room to breathe and to get generation throughput.

Contrast: **DPO** needs policy (112 GB) + frozen reference (14 GB) = **126 GB**, and you can drop the reference entirely by precomputing its log-probs offline, which gets you to 112 GB — a *single node's worth*, sometimes a single GPU with LoRA. **GRPO** drops the value model: policy (112) + ref (14) + reward (14) = 140 GB.

**💰 Math:** PPO needs roughly **2× the accelerator memory of GRPO and 2.3× of DPO** for the same policy size. On a cloud at ~$3/H100-hour, an 8-GPU PPO node is $24/hour versus a 4-GPU DPO setup at $12/hour, and PPO runs longer in wall-clock because generation is in the critical path. A 3-day PPO run is 8 × 72 × $3 = **$1,728**; the equivalent DPO run on the same data is often 4 × 8 × $3 = **$96**. That is an 18× gap on infrastructure alone, before you count the engineer-weeks. **📅 Volatile:** GPU pricing moves; the ratio is the durable part.

**⚠ Trap:** saying "four models" and stopping. The interviewer wants to hear that only *two* are trainable and that the memory cost is dominated by optimizer state on those two, not by the frozen forward-only pair. Someone who says "four models so four times the memory" has memorized a slide. It's 16 bytes/param versus 2 — the frozen models are nearly free by comparison.

### Walk me through one full PPO iteration, end to end. What happens in what order, and where do the buffers live?

An iteration has two distinct phases with completely different compute profiles, and that mismatch is the source of every PPO systems headache.

**Phase 1 — rollout (generation-bound, memory-bandwidth-bound):**

1. Sample a batch of prompts from the prompt dataset — say 512.
2. Generate a completion for each with the *current* policy, at temperature ~1.0 (you need entropy; greedy decoding gives you no exploration and no gradient signal diversity). This is autoregressive decode, so it's memory-bandwidth-bound and it is the wall-clock bottleneck — commonly 60–80% of iteration time.
3. Store, per token: `token_id`, `logprob_old` (the policy's log-prob at generation time — **you must capture this during generation**, not recompute it later), and the position/mask.
4. Forward the full sequences through the **reference** model to get `logprob_ref`.
5. Forward through the **reward model** to get a scalar `r` per sequence.
6. Forward through the **value model** to get `V_t` for every token position.

**Phase 2 — optimization (compute-bound, standard training):**

7. Build the per-token reward: `reward_t = −β·(logprob_old_t − logprob_ref_t)` for every token, plus the RM scalar `r` added *only at the final token*. (More on this in a moment — it's the sequence-vs-token attribution question.)
8. Compute advantages with GAE over the token sequence, then whiten them.
9. For E epochs (typically 1–4) over this rollout batch, in minibatches: forward the policy, get `logprob_new`, compute the ratio ρ = exp(logprob_new − logprob_old), apply the clipped surrogate loss, add the value loss and an entropy bonus, backward, step.
10. Discard the buffer. Go to step 1.

**The buffer.** This is the "experience buffer" or "rollout buffer," and unlike classical RL it is **on-policy and ephemeral** — you fill it, you consume it in a handful of epochs, you throw it away. It is not a replay buffer in the DQN sense; keeping data across many policy updates makes the importance ratios blow up and PPO's clipping stops being a valid approximation. Concretely, for 512 sequences × 1,024 tokens you're storing roughly 512 × 1024 × (4 bytes token_id + 4 logprob_old + 4 logprob_ref + 4 value + 4 advantage + 4 return) ≈ 512 × 1024 × 24 bytes ≈ **12 MB**. Tiny. The buffer is never the memory problem; the four models are.

**The systems consequence.** Phase 1 wants an inference engine — continuous batching, paged KV cache, tensor parallelism tuned for decode. Phase 2 wants a training framework — ZeRO/FSDP sharding, gradient accumulation, activation checkpointing. These want *different* parallelism layouts for the *same weights*. So every serious PPO implementation has to solve weight synchronization between a training copy and an inference copy, and that's a distributed-systems problem, not an ML problem. It's why modern stacks (veRL, OpenRLHF, TRL's newer PPO paths) integrate vLLM or SGLang for the rollout phase and then push updated weights across — via NCCL broadcast, or shared-memory handoff on colocated GPUs, every iteration.

**⚠ Trap:** recomputing `logprob_old` with a forward pass after generation instead of capturing it during generation. It *looks* equivalent. It is not, because the generation engine and the training engine use different kernels, different attention implementations, and sometimes different precision — so you get log-probs that differ by 1e-3 or worse. Those differences enter as ρ = exp(Δ) ≈ 1 ± 0.001 noise on every token, which is survivable, until a numerics mismatch makes it 1 ± 0.05 and your clipping fraction goes to 40% and your run silently underperforms. This "rollout–training log-prob mismatch" is a well-known and actively-discussed source of instability in RL-for-LLM stacks, and **logging the mean absolute difference between the generation log-probs and a recomputed training-forward log-prob is the single highest-value diagnostic you can add.** If it's not ~1e-5, something in your stack disagrees with itself.

### Write the PPO clipped surrogate objective and explain what each piece is protecting you from.

The objective, per token t:

L^CLIP = E_t[ min( ρ_t · Â_t , clip(ρ_t, 1−ε, 1+ε) · Â_t ) ]

where ρ_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t) is the importance ratio, Â_t is the advantage estimate, and ε ≈ 0.2. You *maximize* this, so in code it's a negated loss.

**The mental model.** Vanilla policy gradient says "increase the log-prob of actions with positive advantage." The problem is step size: one large gradient step can move the policy far enough that the advantage estimates — computed under the *old* policy — are no longer valid, and you fall off a cliff you can't climb back up, because your new rollouts come from the broken policy. TRPO solved this with a hard KL trust region and a constrained optimization solve, which is correct and miserable to implement. PPO's insight is that you can get most of the trust-region benefit with a **first-order objective that simply refuses to reward moving too far**.

**Term by term:**

`ρ_t · Â_t` is the importance-weighted policy gradient. Because you take multiple gradient epochs over one rollout batch, π_θ has drifted from π_θ_old, and ρ corrects for the fact that your samples came from the old distribution.

`clip(ρ, 1−ε, 1+ε)` bounds that correction. The `min` is the clever bit and it is asymmetric on purpose:

- When **Â > 0** (good action, push probability up): if ρ > 1+ε, the clipped branch is (1+ε)·Â, which is smaller, so `min` selects it. The objective flattens — **the gradient goes to zero** and there's no incentive to push this action's probability higher. You've captured the improvement and stopped.
- When **Â < 0** (bad action, push probability down): if ρ < 1−ε, the clipped branch is (1−ε)·Â, which is *larger* (less negative), so `min` selects the unclipped ρ·Â. **The gradient is not cut off.** You can always keep pushing a bad action's probability down.

That asymmetry is the part people get wrong, and it's a great thing to volunteer: **clipping limits how aggressively you promote, but never limits how aggressively you demote.** The intuition is safety — undoing a bad action is always allowed; over-committing to a good one is not.

**What clipping is NOT.** It is not a KL constraint. It bounds the per-token probability *ratio*, which does not bound the sequence-level KL, and empirically the policy can drift substantially in KL while the clip fraction looks healthy. That's exactly why RLHF PPO carries a *separate explicit KL penalty against the reference model* in the reward. Two different mechanisms, two different jobs: clipping stabilizes the optimizer, the KL penalty anchors the policy to the SFT distribution. Conflating them is one of the most common misconceptions in this area and I'd correct it directly if a candidate said "the clip keeps it close to the reference."

**📐 The number to monitor:** clip fraction — the share of tokens where clipping was active. Healthy is roughly **5–20%**. Near 0% means your updates are tiny (raise LR or epochs). Above ~40% means the policy is trying to move much further than the trust region allows every step; lower the learning rate or reduce epochs per rollout, because you're mostly training against a wall.

### What is the value model predicting, and derive GAE for me.

The value model predicts the expected return from a given state — in LLM terms, given the prompt plus the tokens generated so far, what total future reward do we expect this trajectory to earn. Its purpose is **variance reduction**. The raw policy gradient uses the return R as the weight; subtracting a baseline V(s) gives A = R − V, which has the same expectation (because the baseline doesn't depend on the action) but far lower variance. Without it, every token in a good sequence gets a big positive push and every token in a bad one a big negative push, which is an extremely noisy signal — the "did we win the game" credit assignment problem.

**GAE.** The single-step TD residual is

δ_t = r_t + γ·V(s_{t+1}) − V(s_t)

which measures "was this token better than the critic expected." You could use δ_t alone as the advantage — low variance, but heavily biased by whatever errors V has. Or you could use the full Monte-Carlo return minus V — unbiased, enormous variance. GAE interpolates with an exponentially-weighted sum of the residuals:

Â_t = Σ_{l≥0} (γλ)^l · δ_{t+l}

λ=0 recovers the pure TD estimate (low variance, high bias); λ=1 recovers Monte Carlo minus baseline (unbiased, high variance). **📄 Paper:** Schulman et al. (2015/2016), *High-Dimensional Continuous Control Using Generalized Advantage Estimation* — its contribution was making this bias/variance knob explicit and continuous.

In LLM RLHF the conventions are specific and worth stating: **γ = 1.0** and **λ ∈ [0.95, 1.0]**. γ=1 because there is no reason to discount future tokens within a single response — a response isn't an infinite-horizon control problem, it's a finite episode of a few hundred steps, and discounting at γ=0.99 over 500 tokens would attenuate the terminal reward by 0.99^500 ≈ 0.0066, effectively deleting your reward signal. If you see γ=0.99 in an LLM PPO config, that's usually a copy-paste from a robotics repo and it is a bug.

The backward recursion, which is how you actually compute it:

```python
def gae(rewards, values, mask, gamma=1.0, lam=0.95):
    # rewards, values, mask: (B, T). values must include a bootstrap 0 past the end.
    T = rewards.size(1)
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t + 1 < T else torch.zeros_like(last)
        delta = rewards[:, t] + gamma * next_v * mask[:, t] - values[:, t]
        last = delta + gamma * lam * last * mask[:, t]
        adv[:, t] = last
    returns = adv + values          # regression target for the value head
    return adv, returns
```

**⚠ Trap:** the mask. Padding tokens and post-EOS tokens must not contribute, and if you forget to zero `last` at episode boundaries you bleed advantage backward across sequences in a packed batch. The symptom is bizarre: the policy starts optimizing tokens near the *end* of one sequence based on the reward of the *next* one, and the training curve looks fine while sample quality quietly rots.

**⚠ Trap, the bigger one:** the value model is trained from scratch (or from an SFT init with a fresh scalar head) *simultaneously* with the policy, on data generated by the policy it is trying to evaluate. At step 0 it predicts nothing useful, so your advantages are pure noise and your first several hundred updates are effectively random. This is why implementations do value-head warmup, or freeze the policy for the first N steps, or accept that early training is throwaway. It is also a large part of the answer to "why is PPO fragile" — you are training two coupled networks in a feedback loop, and the critic's failure mode (divergence, collapse to a constant) is silent in the reward curve.

### Why do you whiten advantages, and what can go wrong when you do?

Whitening — subtracting the batch mean and dividing by the batch standard deviation of the advantages — is doing two separate jobs, and conflating them causes real bugs.

**Job 1: removing the per-prompt reward offset.** Recall that Bradley-Terry identifies rewards only up to a per-prompt constant, so "hard" prompts might sit at reward −3 and "easy" ones at +4. Without centering, every token of every response to an easy prompt gets a positive advantage regardless of whether it was a good response, and the policy learns "produce whatever you produce on easy prompts." Centering kills that.

**Job 2: making the gradient scale invariant to reward magnitude.** The RM's output scale is arbitrary. If you swap in a new RM whose rewards are 5× larger, your effective learning rate becomes 5× larger unless you normalize. Dividing by the std makes your PPO hyperparameters portable across reward models, which in practice is the difference between "tuned once" and "retuned every time anyone touches the RM."

**Where it goes wrong:**

**Batch-level whitening with a small batch destroys signal.** If your batch has 8 prompts and 7 of them got genuinely good responses and 1 didn't, whitening rescales that into "1 is bad, 7 are mildly good" with the std computed from 8 points. Small-batch whitening is high-variance normalization and it can flip the sign of legitimately-informative advantages. Rule: whiten over a batch of at least a few hundred sequences, and never over a microbatch.

**Whitening after masking, or before — pick one and be right.** You must compute the mean and std over *unmasked tokens only*. If pad tokens contribute zeros to the mean, your normalization is wrong by the padding fraction, which varies per batch. This is a genuinely common bug and it produces a slow, unexplained degradation rather than a crash.

**Whitening destroys the "everything was bad" signal.** If every response in the batch is terrible, whitening produces a distribution centered at zero — some responses look good *relative* to the others. The policy is now being taught to prefer the least-bad garbage. In classical RL that's fine because the value function carries the absolute level; in short RLHF runs with a weak critic it's a real pathology. Some implementations whiten only the mean (center) and not the scale, or whiten the *rewards* rather than the advantages, for exactly this reason. **The version I default to: center and scale the advantages at the batch level, and separately monitor the raw un-whitened mean reward** so I can still see the absolute level.

**⚠ Trap:** whitening the returns used as the value-head regression target. Advantages get whitened; **returns do not**. If you whiten returns, the value head is now regressing to a moving, batch-dependent target and it will never converge — the classic symptom is value loss that oscillates without trending down while explained variance stays near zero.

**📐 Monitor this:** *explained variance* of the value head, 1 − Var(returns − values)/Var(returns). Healthy after warmup is roughly 0.3–0.8 for LLM RLHF. Near 0 means the critic is useless and your advantages are essentially raw returns; negative means the critic is actively worse than predicting the mean, which means something is broken (usually the target, usually whitening).

### Explain the KL penalty. Where exactly is it applied and what is β actually trading off?

The KL penalty is applied **per token, inside the reward**, not as a separate loss term on the objective. That placement matters and it's a good discriminator question.

Concretely, for each generated token t:

r_t = −β · ( log π_θ(a_t|s_t) − log π_ref(a_t|s_t) )

and then the reward model's scalar score is added *once*, at the final token:

r_T += r_RM(x, y)

So the per-token reward stream is mostly a small KL penalty, with a single large spike at the end. Those go into GAE, which propagates the terminal reward backward while the KL terms provide dense per-token shaping. Putting KL in the reward rather than in the loss means the credit-assignment machinery (advantages, the critic) handles it consistently with everything else — the critic learns to predict the KL cost of a trajectory too.

**What β trades:** β is the exchange rate between "how much reward model score is one nat of divergence from SFT worth." Low β → the policy is free to move far, extracts more proxy reward, Goodharts faster, and degrades in ways the RM can't see (fluency, diversity, factuality — all the things the SFT model was good at and the RM never scored). High β → the policy barely moves, you get a small quality lift and a wasted run.

The deeper framing, and the one that connects to DPO: the KL-regularized objective

max_π E[r(x,y)] − β·KL(π ‖ π_ref)

has a known closed-form optimum, π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β). **This is the exact identity DPO exploits**, inverting it to express r in terms of π and π_ref and eliminating the RL loop entirely. So β is not an incidental regularization coefficient — it is the temperature parameter of the target distribution, and it appears with the same meaning in DPO. Saying that in a room demonstrates you understand these as one family rather than two techniques.

**📐 Typical values:** β in the range **0.01–0.1** for LLM RLHF, with 0.02–0.05 a common starting band. But β is not portable across reward models, because reward scale is arbitrary — β=0.05 against an RM with unit-variance rewards is a completely different constraint than β=0.05 against an RM with rewards spanning ±20. **The rule I enforce: normalize your reward to roughly unit variance on a reference batch, then β becomes comparable across runs.** Without that, every β you've ever tuned is a magic number tied to one checkpoint.

**Adaptive KL control.** Rather than fixing β, set a *target* KL (say 6 nats per sequence) and adjust β with a proportional controller: if measured KL exceeds the target, raise β; if below, lower it. **📄** This adaptive-KL scheme comes from Ziegler et al. (2019), *Fine-Tuning Language Models from Human Preferences*, which predates InstructGPT and established most of the RLHF-for-LMs plumbing. I like adaptive control because it turns an unpredictable quantity (KL, which depends on RM scale and task) into a specified one, and it's the same instinct as any feedback controller you'd write for a backend system. The failure mode is oscillation if your gain is too high, so damp it.

**⚠ Trap:** thinking the KL penalty prevents reward hacking. It does not; it *rate-limits* it. The policy can still find the hack, it just has to pay KL to get there, and if the hack is worth more reward than the KL costs, it goes. The KL penalty buys you time to notice, which is why the metric that matters is gold-eval-versus-KL, not KL alone.

### Implement a minimal PPO update step. PyTorch, from memory.

Here is the core, with the pieces that matter and none that don't.

```python
import torch, torch.nn.functional as F

def ppo_update(policy, value, batch, opt, clip=0.2, vf_coef=0.5,
               ent_coef=0.0, vf_clip=0.2):
    """
    batch keys (all (B, T) unless noted):
      input_ids, attn_mask, action_mask   # action_mask=1 on generated tokens only
      logprob_old                          # captured DURING generation
      advantages, returns                  # from GAE; advantages already whitened
      values_old                           # critic output at rollout time
    """
    logits = policy(batch["input_ids"], attention_mask=batch["attn_mask"]).logits
    logits = logits[:, :-1]                              # predict token t+1 from t
    targets = batch["input_ids"][:, 1:]
    logprob_new = torch.log_softmax(logits.float(), -1) \
                       .gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    m   = batch["action_mask"][:, 1:].float()
    adv = batch["advantages"][:, 1:]
    ratio = torch.exp(logprob_new - batch["logprob_old"][:, 1:])

    # --- clipped policy surrogate (negated: we minimize) ---
    s1 = ratio * adv
    s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    pg_loss = -(torch.min(s1, s2) * m).sum() / m.sum()

    # --- clipped value loss (pessimistic, mirrors the policy clip) ---
    v_new  = value(batch["input_ids"], attention_mask=batch["attn_mask"])[:, :-1]
    v_old  = batch["values_old"][:, :-1]
    ret    = batch["returns"][:, :-1]
    v_clip = v_old + (v_new - v_old).clamp(-vf_clip, vf_clip)
    v_loss = 0.5 * (torch.max((v_new - ret) ** 2, (v_clip - ret) ** 2) * m).sum() / m.sum()

    # --- entropy bonus (usually 0 for LLMs; the KL penalty does this job) ---
    ent = -(torch.softmax(logits.float(), -1) * torch.log_softmax(logits.float(), -1)) \
              .sum(-1)
    ent = (ent * m).sum() / m.sum()

    loss = pg_loss + vf_coef * v_loss - ent_coef * ent
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step(); opt.zero_grad()

    with torch.no_grad():
        clipfrac = (((ratio - 1).abs() > clip).float() * m).sum() / m.sum()
        approx_kl = (((ratio - 1) - torch.log(ratio)) * m).sum() / m.sum()  # k3, ≥0
    return dict(pg=pg_loss.item(), vf=v_loss.item(), ent=ent.item(),
                clipfrac=clipfrac.item(), approx_kl=approx_kl.item())
```

The details I would defend in review:

**`action_mask`, not `attn_mask`, for the loss.** Prompt tokens were not actions; they get no policy gradient. This is the same masking discipline as SFT and the same class of bug.

**The value loss is clipped too**, and the `max` makes it *pessimistic* — you take the larger of the clipped and unclipped squared errors, which prevents a single update from moving the critic wildly. It mirrors the policy clip in spirit but the sign logic is opposite (max, not min) because value loss is minimized while the surrogate is maximized.

**`approx_kl` uses the k3 estimator**, (ρ−1) − log ρ, which is non-negative and much lower-variance than the naive log-ratio. This is the number I watch every step; if it exceeds a few times your target you stop the run.

**Entropy coefficient defaults to 0** in LLM RLHF. In classical RL an entropy bonus is your exploration mechanism; here the KL-to-reference penalty already prevents distribution collapse, and adding entropy on top over a 128k-token vocabulary tends to inject gibberish. Reach for it only if you see measured entropy collapsing and the KL penalty isn't holding.

**🏋 Drill:** write this function from memory in 25 minutes with no reference. Pass criterion: correct off-by-one on the logits/targets shift, `action_mask` applied to every reduction, `min` on the policy surrogate and `max` on the value loss, and `logprob_old` treated as an input rather than recomputed. Then run it against a tiny model with a fake reward (`r = −len(response)`) and confirm responses get shorter within 50 steps. If they don't, your sign is wrong somewhere and that is the whole point of the drill.

### Why can PPO reuse a rollout for multiple epochs at all? Isn't policy gradient on-policy?

Vanilla policy gradient is strictly on-policy: the gradient estimator is only valid for data drawn from the current policy. The moment you take one gradient step, your buffered data is stale.

PPO buys reuse with importance sampling. The ratio ρ = π_θ(a|s)/π_θ_old(a|s) reweights the stale samples to correct for the distribution mismatch, giving you an unbiased estimator of the current policy's gradient — *in principle*. In practice importance sampling has catastrophic variance when the distributions separate: the ratio is a product-like quantity that can be enormous when π_θ_old assigned a token low probability and π_θ now assigns it high probability. The clip is exactly the variance control: it caps how much any single stale sample can contribute.

So the honest characterization is **PPO is "near-on-policy."** It tolerates a small amount of staleness — a few gradient epochs' worth — and degrades badly beyond that. This is fundamentally different from off-policy methods like Q-learning that are designed for a replay buffer full of ancient data.

**How many epochs?** For LLM RLHF, **1–4**, and I default to 1 or 2. Two considerations push it down relative to classical RL: (a) the batches are large (hundreds of sequences × hundreds of tokens is a lot of gradient signal per rollout, so you don't need the extra passes), and (b) LLM policies move fast in log-prob space, so ρ separates quickly. If your clip fraction climbs across epochs within a single rollout — epoch 1 at 8%, epoch 4 at 35% — that's the direct measurement that you're over-reusing, and it's a metric worth logging per-epoch rather than per-step.

**The systems reason this matters more than the theory.** Generation is 60–80% of your wall-clock. Every extra epoch over the same rollout is nearly free compute-wise (phase 2 is fast) and reduces the number of expensive generation phases you need. So there's real pressure to crank epochs, and the clip fraction is the metric that tells you when you've cranked too far. That's the tradeoff to state in a room: **epochs are how you amortize your generation cost, and clip fraction is your budget meter.**

**⚠ Trap:** the asynchronous variant. Modern RL-for-LLM stacks run generation and training concurrently on separate GPU pools for throughput, which means the rollouts you train on came from a policy that is now N steps stale. That's a *deliberate* off-policyness, and it's a different regime with different failure modes — you need larger clip ranges or explicit off-policy corrections, and the staleness bound becomes a first-class tuning parameter. If someone says "we made PPO async and it just worked," I'd ask what their measured staleness was and what happened to clip fraction.

### The reward model gives me one number for a 600-token response. How does that become a per-token learning signal?

This is the credit assignment problem and it is the deep reason PPO exists here rather than something simpler.

**The mechanism:** the RM's scalar is placed at the terminal token as r_T. All other tokens get only the KL penalty term. GAE then propagates that terminal reward backward through the value function: Â_t = Σ (γλ)^l δ_{t+l}, and with γ=1 the terminal reward contributes to *every* token's advantage, attenuated by λ^(T−t). At λ=0.95 and a 600-token response, the terminal reward's influence on token 1 is 0.95^599 ≈ 10^-14 — effectively zero. So with λ<1, early tokens learn essentially nothing from the terminal reward directly; they learn from the *critic's* prediction, which is why the critic's quality determines whether early tokens get useful gradient at all. With λ=1, GAE reduces to Monte-Carlo return minus baseline and every token sees the full terminal reward equally — high variance, but no attenuation. That's the real reason LLM RLHF often uses λ near 1.

**The honest limitation:** for a long response, the reward model is telling you "this whole thing was worth 2.3," and the algorithm has no idea which of the 600 tokens deserved credit. If the response was excellent except for one fabricated citation at token 412, the signal is a slightly-lower scalar spread diffusely over all 600 positions. The gradient nudges everything a bit, including all the tokens that were fine. **The learning is real but the signal-to-noise per token is terrible**, and that's why RLHF needs a lot of rollouts to make progress. It's the same reason training an agent on "did the episode succeed" is harder than training on per-step feedback.

**The alternatives and what they cost:**

- **Process rewards** — per-step scores from a PRM, which gives genuinely dense signal. Trades the credit-assignment problem for a step-level reward-hacking problem.
- **Token-level reward from a fine-grained RM** — train an RM that emits a score per span (factuality span, relevance span). This was explored in the fine-grained-RLHF line of work; it demonstrably helps, and it costs you span-level annotation, which is much more expensive than pairwise.
- **Group-relative advantages (GRPO)** — do not attempt per-token attribution at all. Sample G responses to the same prompt, and assign every token of response i the same advantage: (r_i − mean(r))/std(r). This is *cruder* credit assignment than GAE, not finer. It works because the variance reduction from comparing G responses to the *same prompt* is worth more than the fake precision of a poorly-trained critic. That is a genuinely surprising empirical result and worth being able to state.

**🗣 Say this in the room:** "The scalar goes on the last token and GAE spreads it backward. With λ near 1 every token effectively sees the terminal reward, which is high variance but avoids exponential attenuation over a 600-token response. The uncomfortable truth is that per-token credit assignment from a sequence-level reward is mostly fictional — GRPO's result is that deleting the critic and using a group baseline works better in practice, which tells you the critic wasn't buying much precision in the first place."

### Tell me about entropy collapse and mode collapse in RLHF. How do you detect them before they've ruined the run?

They're related but distinct, and both are silent in the reward curve — which is exactly what makes them dangerous.

**Entropy collapse** is a distributional statement: the policy's per-token predictive entropy falls, meaning it becomes increasingly deterministic. Mechanically, policy gradient pushes probability mass toward high-advantage actions, and nothing in the objective rewards keeping alternatives alive. Left unchecked the model becomes near-greedy: temperature sampling stops producing variety, all your rollouts within a batch become near-identical, advantage variance collapses toward zero, and learning stalls. In extreme cases it degenerates into repetition loops.

**Mode collapse** is a behavioral statement: the model produces one *style* for everything. Every answer opens the same way, has three bullets, and closes with a hedge. This can happen without dramatic entropy loss — the model is still uncertain about wording, it just always picks the same *structure*. This is the "RLHF made the model boring" complaint, and it's the source of the well-known reduction in output diversity after alignment.

**Detection — the metrics I put on the dashboard from step 0:**

1. **Mean per-token entropy of the policy.** Log it every step. Healthy runs decline gently. A sharp knee downward is your alarm. This is the single most predictive early-warning signal in RLHF and most teams don't log it.
2. **Within-batch response diversity.** Sample G=4 responses to the same prompt at temperature 1.0 on a fixed probe set every N steps; compute pairwise distinct-n-gram overlap or embedding cosine similarity. When self-similarity climbs, you're collapsing. This directly measures the thing you care about.
3. **Advantage standard deviation before whitening.** If the raw spread of rewards within a batch is shrinking, the policy is producing homogeneous outputs and the gradient is dying. This is the same signal that shows up in GRPO as the zero-gradient problem when all group members earn identical reward.
4. **Opener n-gram histogram.** Cheap and shockingly effective: take the first 10 tokens of 500 sampled responses and count uniques. Going from 380 uniques to 40 is mode collapse, visible in one number.

**Mitigations, ranked:**
- **Raise β** on the KL penalty. The reference model has high entropy; anchoring to it is the primary defense and usually sufficient.
- **Stop earlier.** Entropy collapse is monotone in optimization pressure; the checkpoint before the knee is usually the one you wanted anyway.
- **Entropy bonus** (`ent_coef` > 0). Works, but over a 100k+ vocab it's a blunt instrument that can inject low-probability junk. I use it last.
- **Diversity-aware sampling in rollouts** — higher temperature, or explicitly sampling from multiple prompts per batch rather than many completions of few prompts.

**⚠ Trap:** measuring diversity only at temperature 0. Your production serving might use temperature 0.7, but if you evaluate diversity greedily you see one sample and learn nothing. Worse, teams sometimes compensate for collapse by *raising serving temperature*, which restores surface variety while the underlying distribution is still collapsed — you get variation in word choice over a single rigid structure. That's cosmetics, not a fix.

**💰 The product cost:** for Notion or a writing product, mode collapse is a direct retention problem — users notice that every output has the same skeleton within a week. For a coding agent it's worse and subtler: collapsed policies stop exploring alternative implementations, so your best-of-n reranking stops working because all n candidates are the same candidate. If you spent 8× on serving for best-of-8 and your policy collapsed, you are paying 8× for 1× of diversity — the entire $630k/year from the earlier arithmetic, wasted.

### Give me the failure taxonomy for a PPO run that goes off the rails. What's on your dashboard and what does each pattern mean?

**🔍 Failure taxonomy — PPO instability, as a decision procedure.** The dashboard, in the order I read it:

**1. KL (k3 estimator) vs. its target.**
- *KL explosion* — approx_kl rises superlinearly, samples become degenerate. Causes, in order of likelihood: learning rate too high; β too low; the reward model has a blind spot the policy found; log-prob mismatch between generation and training engines making ρ garbage. *Action:* kill the run, roll back to the last checkpoint under target, halve LR, and check the log-prob mismatch diagnostic before restarting.
- *KL near zero and flat* — the policy isn't moving. β too high, LR too low, or the advantages are all ~0 (check advantage std before whitening; if it's tiny, either the RM is not discriminating or your rollouts are all identical).

**2. Clip fraction.** Healthy 5–20%. Above 40% means the policy wants to move much further than the trust region permits every step — the optimizer is fighting the clip. Lower LR or fewer epochs per rollout. Below 1% means updates are trivially small.

**3. Value-head explained variance.** Healthy 0.3–0.8 after warmup.
- *Near 0 or negative* — the critic is useless, so advantages are essentially raw returns with no variance reduction. Causes: value LR wrong (the critic often needs a *different*, usually higher, LR than the policy); returns being whitened by mistake; the critic head initialized too large; or genuinely too little warmup. *This is the failure that most often masquerades as "PPO is unstable."*
- *Value loss diverging while policy loss looks fine* — classic value-head divergence. The critic chases a target that moves because the policy moves, the policy uses the bad critic, feedback loop. Fix with value clipping (already in the code above), a lower value LR, or more critic warmup steps before unfreezing the policy.

**4. Mean reward, raw and un-whitened.** Rising is necessary but never sufficient — this is the metric that lies. Read it *next to* gold eval, never alone.

**5. Policy entropy.** Gentle decline is fine. A knee is your collapse alarm.

**6. Mean response length.** If it's climbing steadily, verbosity exploitation is underway. I put a hard alert on length exceeding 1.5× the SFT baseline.

**7. Gradient norm.** Spikes correlate with the batch containing a degenerate rollout. Global grad-norm clipping at 1.0 is standard and non-negotiable.

**8. The generation/training log-prob mismatch.** Mean |Δlogprob| between what the inference engine reported and what a training forward pass computes on the same tokens. Should be ~1e-5 in the same precision. If it's 1e-2, your ratios are noise and every other metric on this list is untrustworthy.

**9. Sample dumps.** Twenty generations, every 100 steps, logged to somewhere a human will actually look. Every incident I have seen was visible here first. Automate it; do not rely on remembering to check.

**🗣 Say this in the room:** "The reward curve is the metric that lies, because it's the objective. My alarm set is KL against a target, clip fraction in the 5–20% band, value explained variance above ~0.3, policy entropy without a knee, and mean length under 1.5× the SFT baseline — plus twenty sample dumps every hundred steps. If those five are green and reward is climbing, I believe the run. If reward is climbing and any of them is red, I believe the metric, not the reward."

### Which PPO hyperparameters actually matter for LLM RLHF, and what values would you start from?

The honest framing first: PPO for LLMs has roughly a dozen knobs, most of the published defaults come from continuous-control robotics, and about four of them determine whether your run works. Knowing which four is the practical skill.

**The four that matter:**

**β (KL coefficient), 0.01–0.1, start ~0.05.** The single most important knob. It sets your position on the KL/reward frontier. Tune it by running short jobs at 0.01 / 0.03 / 0.1 and plotting gold eval against measured KL — one number does not generalize across reward models, so this is a per-RM sweep. If you can only afford one sweep, sweep this.

**Policy learning rate, ~1e-6 to 5e-6 for full fine-tuning.** Notably *lower* than SFT's 1e-5 — often by 3–10×. The reason: PPO's gradient is much noisier (it's a policy-gradient estimator, not a supervised loss) and a bad step is unrecoverable because subsequent rollouts come from the damaged policy. SFT can absorb a bad step; PPO compounds it. If your run diverges and you change one thing, change this.

**Rollout batch size (prompts per iteration), 256–1024.** Bigger batches mean lower-variance advantage estimates and more stable whitening. This is the knob that most directly buys stability, and it costs generation throughput linearly. If PPO feels unstable and you have GPUs, spend them here before you spend them on more steps.

**PPO epochs per rollout, 1–4, start at 1–2.** Governed by clip fraction as described. This is your generation-cost amortization dial.

**The ones with safe defaults you should not fiddle with:** clip ε = 0.2, γ = 1.0, λ = 0.95, grad-norm clip = 1.0, value clip = 0.2, value coefficient = 0.5, entropy coefficient = 0.0.

**The one people forget:** the **value-model learning rate**, which often wants to be higher than the policy's (a common choice is ~1e-5 against a policy at 2e-6) because the critic starts from nothing and must catch up. Coupled with a warmup period where the policy is frozen. If your explained variance never gets off the floor, this is your first suspect.

**And the generation settings**, which are hyperparameters even though nobody lists them: temperature 1.0 (not 0.7 — you need the entropy for exploration and for valid importance ratios), top_p = 1.0, top_k disabled. **⚠ Trap:** sampling rollouts with top_p=0.9 and then computing log-probs over the *full* distribution. Your `logprob_old` no longer describes the distribution you actually sampled from, the importance ratios are systematically wrong, and the bias is invisible in every metric except sample quality. Either sample from the full distribution, or correctly account for the truncation in the log-probs. I have seen this bug survive months in a production RLHF stack.

**📐 The rule of thumb I'd give a new team:** if PPO is unstable, in order — (1) halve the policy LR, (2) double the rollout batch, (3) raise β, (4) drop to 1 epoch. Those four, in that order, fix the large majority of unstable runs, and each one costs you throughput rather than correctness.
