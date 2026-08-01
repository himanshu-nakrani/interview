### Write the GRPO objective on the board and tell me what every term is doing.

GRPO is PPO with the value network deleted and replaced by a sample statistic. That is the entire idea, and if you lead with it the rest of the derivation is bookkeeping.

For a prompt `q`, sample a group of `G` completions `{o_1..o_G}` from the old policy `π_old`. Score each with the verifier to get `r_1..r_G`. The advantage assigned to *every token* of completion `i` is the group-normalized reward:

```
A_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)
```

Then run a PPO-style clipped surrogate over tokens, with the KL term added to the loss rather than folded into the reward:

```
J(θ) = E_q, {o_i}~π_old [
    (1/G) Σ_i  (1/|o_i|) Σ_t  min( ρ_{i,t} · A_i ,  clip(ρ_{i,t}, 1−ε, 1+ε) · A_i )
    −  β · D_KL[ π_θ ‖ π_ref ]
]

where  ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t})
```

Term by term:

- **`ρ_{i,t}`, the importance ratio.** Present because you take multiple gradient steps (`μ` inner epochs) on one batch of rollouts, so after the first step the data is off-policy. On the very first inner step `ρ ≡ 1` exactly. If you use `μ = 1` — which many production recipes do — the ratio is 1 everywhere on the first and only step, the clip never activates, and GRPO degenerates to a normalized REINFORCE. Say that out loud; it shows you understand why the machinery is there.
- **`min(ρA, clip(ρ)A)`, the clipped surrogate.** A trust region enforced without computing a trust region. When `A > 0` it caps how much you can raise the probability in one update (`1+ε`); when `A < 0` it caps how much you can crush it (`1−ε`). The `min` makes the objective a pessimistic lower bound, so the clip only ever restricts improvement, never permits a bigger step.
- **`A_i`, the group-relative advantage.** The baseline is the group mean, which by the baseline theorem is a valid variance reduction: it does not depend on which sample you are scoring... except it does, slightly, because sample `i` is inside the mean. That is a `1/G` bias, and it is why leave-one-out variants exist.
- **`β · D_KL`.** Applied as a loss term over the policy's distribution, not as a per-token reward subtraction the way classic PPO-RLHF does it. Practically this means the KL gradient flows directly rather than through the advantage, which is cleaner.
- **The two `1/` normalizers.** `1/G` averages over the group; `1/|o_i|` averages over tokens within a completion. That second one is the contested one and I will come back to it.

**📄 Paper:** Shao et al. (2024), *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* — introduced GRPO, replacing PPO's learned value network with a group-mean baseline computed from multiple samples of the same prompt.

**🗣 Say this in the room:** "GRPO's insight is that PPO's critic exists only to produce a per-state baseline for variance reduction, and for a single-turn LLM task with one terminal reward, you can get that baseline for free by sampling the same prompt G times and using the group mean. You delete a model the size of the policy from your training job, and you swap a learned, biased, hard-to-tune estimate for an unbiased empirical one — paying for it in extra decode."

### Why is deleting the value model defensible? What exactly are you giving up?

The value model in PPO estimates `V(s_t)` — expected return from the current partial sequence — and serves two purposes: it is the baseline in `A_t = r_t + γV(s_{t+1}) − V(s_t)`, and via GAE it distributes credit across timesteps. GRPO keeps the first purpose and abandons the second.

**Why the first is fine.** For a single-turn task with a single terminal reward, the true value function of the prompt is `E_{o~π}[r(q,o)]` — exactly what the group mean estimates. With G=8 samples that estimate has standard error `σ/√8 ≈ 0.35σ`, which for a binary reward with p=0.5 is about 0.18. That is noisy, but it is *unbiased*, and it is measuring the actual current policy. A learned value head on a 7B policy is a regression problem on a rapidly-shifting target distribution; it starts randomly initialized, spends the early run being wrong, and is a documented source of instability (value loss divergence is one of the classic PPO-for-LLM debugging headaches). Trading a biased learned estimator for an unbiased noisy one is a good trade when you can afford the samples.

**Why abandoning the second hurts, and where.** GAE gives you per-token advantages: token 3 and token 3000 can receive different credit. GRPO gives every token in a completion the *same* advantage `A_i`. For a 4,000-token math solution where the answer hinges on one algebraic step, GRPO's gradient rewards the 3,999 tokens of scaffolding exactly as much as the one that mattered. This is fine in practice for math and code — the signal averages out over enough samples — because the correlation between "solution was right" and "these tokens were good" is strong enough at the trajectory level.

It stops being fine for **long-horizon multi-turn agent tasks**. A 40-step agent trajectory that fails at step 38 gets the same negative advantage on steps 1–37, which were correct. You are actively punishing correct behavior. This is the single biggest known limitation of vanilla GRPO for agentic RL and it is a live 2026 research area — I cover the verifier-gated step-credit approaches later in this section.

**⚠ Trap:** claiming GRPO removes the critic "to save memory." Memory is the headline benefit but not the deepest one. The deeper one is that you delete an entire second training problem — a network with its own optimizer, its own learning rate, its own loss curve, and its own way to silently diverge and corrupt every advantage in the run. Halving the number of things that can be misconfigured is worth more to a small team than the GB.

### Give me the memory and orchestration comparison of PPO versus GRPO, with the arithmetic.

Use a 7B policy in bf16 mixed precision with Adam. Per-parameter training memory: 2 bytes bf16 weights + 2 bytes bf16 gradients + 4 bytes fp32 master weights + 4 bytes Adam `m` + 4 bytes Adam `v` = **16 bytes/param**. Inference-only copies are 2 bytes/param.

**PPO-RLHF, four models:**

| Model | Role | Bytes/param | 7B total |
|---|---|---|---|
| Policy | trained | 16 | 112 GB |
| Value | trained | 16 | 112 GB |
| Reference | frozen fwd | 2 | 14 GB |
| Reward | frozen fwd | 2 | 14 GB |
| | | | **252 GB** |

**GRPO with a program verifier:**

| Model | Role | Bytes/param | 7B total |
|---|---|---|---|
| Policy | trained | 16 | 112 GB |
| Reference | frozen fwd | 2 | 14 GB |
| Verifier | CPU sandbox | — | 0 GB |
| | | | **126 GB** |

Drop the KL term as DAPO does and you are at 112 GB — a **2.25× reduction** versus PPO, before activations and before the KV cache for rollouts.

**📐 Numbers you must know:** 16 bytes/param for bf16+Adam training, 2 bytes/param for bf16 inference. Memorize these two and you can size any RL job in your head. A 7B policy alone needs 112 GB, which does not fit on one 80 GB H100 — so even single-model GRPO on a 7B requires FSDP/ZeRO sharding across at least 2 cards, and realistically 4–8 once you add activations and the rollout KV cache.

The orchestration story matters more than the memory. PPO requires you to keep two trainable models in step, tune a value learning rate and a value-loss coefficient, tune GAE's λ and γ, and debug value divergence. GRPO's config surface is group size, ε, β, learning rate, and the sampling params. **In engineer-weeks — which is the currency your manager cares about — I budget roughly 4–8 weeks to get a PPO-RLHF pipeline stable for a team that has not run one, and roughly 1–2 weeks for GRPO on top of an existing vLLM + TRL stack.** That gap, not the GB, is why GRPO won for open-weight reasoning work.

**💰 Math:** the cost is not free — you pay in decode. PPO with a value head does one rollout per prompt; GRPO does G. At G=8 you have 8× the generation. If generation is 70% of step wall-clock (typical for long-CoT), GRPO's step is roughly `0.7×8 + 0.3 = 5.9×` the rollout-side cost of a G=1 method. On 8×H100 at $2/GPU-hr, a step that took 8 minutes now takes ~47 minutes: $2.13 → $12.53 per step. GRPO is not cheaper per step; it is cheaper per *engineer* and per *unit of stability*, and it is cheaper in resident memory. Be precise about which axis you mean.

### Implement GRPO's advantage computation and loss. Forty lines, from memory.

```python
import torch, torch.nn.functional as F

def group_advantages(rewards: torch.Tensor, G: int, eps: float = 1e-4,
                     scale_by_std: bool = True) -> torch.Tensor:
    """rewards: (B*G,) verifier scores. Returns (B*G,) per-sequence advantages."""
    r = rewards.view(-1, G)                      # (B, G)
    adv = r - r.mean(dim=1, keepdim=True)        # group-mean baseline
    if scale_by_std:
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)                          # (B*G,)


def grpo_loss(logits, old_logprobs, ref_logprobs, actions, mask, adv,
              clip_low=0.2, clip_high=0.2, beta=0.0):
    """
    logits:       (N, T, V)  current policy logits over completion tokens
    old_logprobs: (N, T)     logprobs under the sampling policy (pi_old)
    ref_logprobs: (N, T)     logprobs under the frozen reference
    actions:      (N, T)     sampled token ids
    mask:         (N, T)     1 for completion tokens, 0 for prompt/padding
    adv:          (N,)       per-sequence advantage from group_advantages
    """
    logprobs = torch.gather(
        F.log_softmax(logits.float(), dim=-1), 2, actions.unsqueeze(-1)
    ).squeeze(-1)                                             # (N, T)

    ratio = torch.exp(logprobs - old_logprobs)                # (N, T)
    A = adv.unsqueeze(1)                                      # (N, 1) broadcast
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * A
    pg = -torch.min(unclipped, clipped)                       # (N, T)

    if beta > 0.0:                                            # k3 KL estimator
        lr = ref_logprobs - logprobs
        kl = torch.exp(lr) - lr - 1.0                         # >= 0, low variance
        pg = pg + beta * kl

    # token-level normalization: one denominator over the whole batch
    return (pg * mask).sum() / mask.sum().clamp(min=1.0)
```

Four things to point at while you write it, because they are what the interviewer is checking:

**`ratio` is computed in log space then exponentiated.** Never compute a ratio of probabilities directly; the logprobs are order −5 to −15 and the probabilities underflow.

**`old_logprobs` must come from the sampler, not be recomputed.** If you generate with vLLM and recompute logprobs with the HF forward pass, you get small numerical differences — different kernels, different attention implementations, different batching — and `ratio` will not be exactly 1 on the first inner step. That silent mismatch is one of the most common real bugs in GRPO implementations; it manifests as a clip fraction that is nonzero at step 0 when it mathematically must be zero. **Assert `ratio ≈ 1` on the first inner epoch in your tests.**

**The final normalizer is `mask.sum()` over the whole batch, not per-sequence.** That is the DAPO-style token-level loss. The original GRPO formulation averages within each sequence first (`1/|o_i|`), which weights short sequences more per-token; the next question is about why that matters.

**`beta=0.0` by default.** I write the KL as opt-in, because for reasoning runs I usually turn it off, and a default of 0 forces the config to be explicit.

**🏋 Drill:** 25 minutes, no references. Write `group_advantages` and `grpo_loss` from scratch. Pass criterion: your `ratio` is exactly 1.0 on the first inner step given `old_logprobs` from the same forward pass; your advantage is exactly 0 for a group with identical rewards; and you can state, without looking, which shape each tensor has at each line.

### Every sample in a group gets the same verifier score. Walk me through exactly what happens.

This is the question I would ask to separate people who have read the GRPO paper from people who have watched a GRPO run stall, and the answer is short and mechanical.

`A_i = (r_i − mean(r)) / std(r)`. If all `r_i` are equal, the numerator is exactly 0 for every sample. The advantage vector is all zeros. The policy-gradient term `−min(ρA, clip(ρ)A)` is identically zero for every token in every sample of that group. **That prompt contributed exactly zero gradient**, after you spent G full autoregressive generations and G verifier invocations on it. The `std` in the denominator is also 0, which is why every implementation has that `+ 1e-4`; without it you get NaN, and with it you get 0/1e-4 = 0, which is the right answer but arrived at nervously.

Now the part that makes it a production problem rather than a curiosity. It happens on **both ends**: a prompt so easy that all 8 samples pass, and a prompt so hard that all 8 fail. Both are dead. And the easy end grows over the run — the whole point of training is to make prompts easy — so the fraction of your batch producing zero gradient *increases monotonically* unless you intervene. I have seen runs where by step 600, 70% of sampled groups were degenerate and the effective batch size had silently collapsed to 30% of what the config said. Loss looked fine. Reward looked great (of course it did — the surviving groups were the easy ones). Learning had essentially stopped.

**📐 Numbers you must know:** for a per-prompt pass probability `p` and group size `G`, `P(degenerate group) = p^G + (1−p)^G`. At G=8: p=0.5 → 0.008; p=0.8 → 0.17; p=0.9 → 0.43; p=0.95 → 0.66; p=0.99 → 0.92. At G=16, p=0.9 → 0.19. Doubling the group size roughly halves the degenerate fraction at high p, at double the decode cost.

**The fixes, in the order I apply them:**

1. **Dynamic sampling (DAPO's fix).** Over-generate prompts per step, discard degenerate groups, and keep sampling until you have filled the target number of *non-degenerate* groups. This makes your effective batch size constant and honest. It costs extra generation — but you were spending that generation on zero gradient anyway, so you are paying for something instead of nothing.
2. **Difficulty-aware prompt scheduling.** Re-score prompts from the rollouts you already have and demote anything whose running pass rate leaves the 0.15–0.85 band.
3. **Raise G on the hard tail.** Adaptive group size: 8 for medium prompts, 16–32 for prompts near p=0 where you need more shots to find any success at all.
4. **Instrument it.** `frac_degenerate_groups` goes on the dashboard next to reward and entropy. If you have one metric to add to a GRPO run beyond reward, it is this one.

**🗣 Say this in the room:** "Zero-variance groups are GRPO's structural failure mode: if all G samples get the same verifier score, the group-mean baseline makes every advantage exactly zero and that prompt produces no gradient at all. It's not rare — it's the default outcome for easy prompts, and easy prompts are what you manufacture by training. I track the degenerate-group fraction as a first-class metric and use dynamic sampling to hold effective batch size constant."

### How do you choose the group size G, and what is the tradeoff?

G controls the quality of your baseline and therefore the variance of your gradient, and it also controls how much of your run is wasted on degenerate groups. Both push it up. Cost pushes it down, linearly and hard.

**The variance argument.** The group mean estimates the prompt's true value with standard error `σ/√G`. Going from G=4 to G=8 cuts baseline noise by 29%; G=8 to G=16 by another 29%; G=16 to G=64 by half again. Diminishing returns set in fast, and you are doubling decode each time. The published GRPO/DAPO/R1-family recipes cluster at **G = 8 to 16 for math and code**, and that is where I start.

**The degenerate-group argument** pushes up when your prompts are easy, as computed above. If you are training a strong model on a pool where the median pass rate is 0.85, G=8 wastes 27% of groups and G=16 wastes 8%.

**The cost argument.** Generation cost is exactly linear in G, and generation is 60–80% of step time for long-CoT. Going from G=8 to G=32 roughly triples total step cost.

**My decision procedure:**

- Start at **G=8**. It is the sweet spot in every published recipe and the burden of proof is on deviating.
- Compute the degenerate fraction after 50 steps. If it is above ~15%, first fix the *prompt pool difficulty*, not G. Raising G to paper over an easy pool is buying variance reduction you did not need at 2× the price.
- Raise G to 16 only when your prompt pool is already well-centred and you still see high degenerate fractions — which happens on genuinely hard sets where p is near 0.
- For agentic tasks where a rollout costs dollars rather than cents (container spin-up, real API calls, 40-step trajectories), G=4 with dynamic sampling and aggressive difficulty filtering, and accept the noisier baseline. The economics dominate the statistics.

**💰 Math:** an agentic RL rollout that runs a 20-step tool trajectory with a container, ~15k tokens of context accumulation, and 3 real API calls costs roughly $0.04 in inference plus ~$0.01 in compute — call it $0.05. At G=8 across a 256-prompt batch that is 2,048 rollouts = $102 per step. A 400-step run is **$41,000 in rollouts alone**, before a single gradient. At G=4 it is $20,500. That single config line is a $20k decision, which is why "what G" is a real engineering question and not a hyperparameter you copy.

### Should you divide the advantage by the group's standard deviation? Argue both sides.

This looks like a normalization detail and it is actually a bias question, so treat it seriously.

**The case for dividing (vanilla GRPO).** It puts advantages on a consistent scale regardless of the reward function's magnitude, which makes the learning rate transferable across tasks and reward designs. It also stabilizes the optimizer: without it, a task with rewards in `[0, 10]` needs a 10× smaller LR than one in `[0, 1]`.

**The case against.** For a binary reward with group pass rate `p`, the group std is `√(p(1−p))`. Dividing by it means groups with `p` near 0 or 1 get their advantages *amplified*. At p=0.5, std=0.5 so a correct sample gets advantage `0.5/0.5 = 1.0`. At p=0.875 (7 of 8 correct), std ≈ 0.331, and the single *failing* sample gets advantage `(0 − 0.875)/0.331 = −2.64`. At p=0.125, the single succeeding sample gets `+2.64`.

So the std division systematically up-weights the extreme, nearly-degenerate groups — the ones carrying the least reliable signal — and down-weights the balanced groups that carry the most. That is backwards. It is a **difficulty bias**: your gradient is dominated by lucky-success and unlucky-failure outliers on prompts the model has essentially decided.

**📄 Paper:** Liu et al. (2025), *Understanding R1-Zero-Like Training: A Critical Perspective* (the Dr. GRPO work) — identified both the `std` division and the per-response `1/|o_i|` length normalization as introducing optimization bias, and proposed removing both, showing comparable or better results with less length inflation.

**What I do.** Default to **no std division** (subtract the mean only), with the learning rate tuned once for a `[0,1]` binary reward. If I am mixing reward functions with different scales in one run, I normalize the *reward definitions* to a common range rather than reintroducing per-group std. If I keep the std division, I clamp it with a floor (`std.clamp(min=0.1)`) so a near-degenerate group cannot produce a ±3σ advantage.

**⚠ Trap:** deciding this by reading the loss curve. Both variants train and both curves look fine. The difference shows up as length inflation and as a gradient budget quietly spent on prompts that were already decided. Diagnose it by histogramming the advantage magnitudes against the group pass rate — if your biggest-magnitude advantages sit at p<0.2 and p>0.8, you have the bias.

### Explain the length-normalization problem in the GRPO loss and how you'd fix it.

The original GRPO objective averages token losses *within* each response before averaging across the group: `(1/G) Σ_i (1/|o_i|) Σ_t (...)`. That means every response contributes equal total weight regardless of length, so each token in a 200-token response carries 10× the gradient weight of each token in a 2,000-token response.

Why that is bad, mechanically. Consider a long response that fails. Its negative advantage is spread across 2,000 tokens, so each individual bad token gets a tiny push down. Now consider a short response that fails: the same negative advantage is concentrated into 200 tokens, so each gets a big push down. The optimizer therefore learns much more strongly from short failures than from long failures. Symmetrically, the *penalty* for long garbage is diluted. The net effect is a systematic pressure toward longer responses, especially longer *bad* responses — a low-grade, structural length bias that lives in the loss normalization rather than in the reward.

**The fix** is token-level loss normalization: one denominator over all tokens in the batch.

```
J = ( Σ_i Σ_t  min(ρ_{i,t}A_i, clip(ρ_{i,t})A_i) ) / ( Σ_i |o_i| )
```

Now every token in the batch carries equal weight, long failures get penalized in proportion to their length, and the structural pull toward verbosity is gone. This is DAPO's "token-level policy gradient loss" and it is one of the cheapest correctness wins available — it is a one-line change to your loss reduction.

**⚠ Trap:** confusing this with an explicit length penalty in the reward. They are independent. Token-level normalization removes a *bias in the loss*; a length penalty adds a *term to the reward*. You often want both, but if you add a length penalty while keeping sequence-level normalization, you are adding a hand-tuned correction on top of an unnecessary bias, and you will spend a week tuning the penalty coefficient to cancel something you could have deleted.

**📐 Numbers you must know:** in a batch where responses range 200–4,000 tokens with a mean around 1,200, sequence-level normalization gives the shortest responses roughly 6× the per-token gradient weight of the longest. That is not a subtle effect — it is a 6× weighting difference sitting silently in your loss reduction. Print the per-token weight histogram once and you will never write sequence-level normalization again.

### Which KL estimator do you use, and where in the objective does the penalty go?

Two independent choices that people conflate.

**Estimator.** You cannot compute `D_KL[π_θ ‖ π_ref]` exactly over a 128k-token vocabulary for every position cheaply, so you estimate from samples. Three options, given `r = π_ref(y)/π_θ(y)` on the sampled token:

- **k1 = `−log r` = `log π_θ − log π_ref`.** Unbiased for the KL under samples from `π_θ`. High variance, and individual samples can be negative even though the true KL is non-negative, which makes the metric hard to read.
- **k2 = `½ (log r)²`.** Low variance, always non-negative, slightly biased.
- **k3 = `r − log r − 1`.** Non-negative for all `r > 0`, unbiased, and much lower variance than k1. This is the standard choice in modern implementations and it is what I use. (It comes from Schulman's widely-cited note on approximating KL divergence; it is a control-variate applied to k1.)

**Placement.** Classic PPO-RLHF folds the KL into the *reward*: `r_total = r_RM − β·log(π_θ/π_ref)` per token, then computes advantages on that. GRPO adds it directly to the *loss* as `+ β·KL`. The practical difference: reward-shaping routes the KL signal through the advantage estimator (and through GAE, and through the value function's target), so it gets mixed into the credit-assignment machinery. Adding it to the loss gives a direct, clean gradient. I prefer the loss form; it is easier to reason about and easier to ablate.

**⚠ Trap:** using the k3 *estimator* for the loss but reporting the k1 *value* on your dashboard, or vice versa, and then comparing to a paper's number. Also: whether you sum KL over tokens or average it changes the number by ~3 orders of magnitude on a 2,000-token completion. Pin the estimator, the reduction, and the reference checkpoint in the run config, and log all three with the metric.

**🗣 Say this in the room:** "I use the k3 estimator — `r − log r − 1` with `r = π_ref/π_θ` — because it's unbiased and non-negative, so the logged value is actually readable as a divergence. And I put the penalty in the loss rather than shaping the per-token reward, so the KL gradient doesn't get laundered through the advantage estimator."

### If you only do one gradient step per batch of rollouts, why does GRPO have an importance ratio and a clip at all?

Because GRPO is written as PPO's objective and inherits its machinery, and because `μ = 1` is a *choice*, not the definition.

With `μ = 1` inner epoch, `π_θ = π_old` exactly when you compute the loss, so `ρ = 1` for every token, `clip(1, 1−ε, 1+ε) = 1`, and the objective reduces to `Σ A_i · log π_θ` — plain policy gradient with a group baseline. The clip is inert. Some teams run exactly this and call it GRPO, which is fair, and it is why you sometimes see GRPO described as "REINFORCE with a group baseline."

You want `μ > 1` when generation dominates step time, which it does for long-CoT. If rollout is 70% of your step and you can extract 4 gradient steps from one rollout batch instead of 1, your gradient-steps-per-GPU-hour goes up nearly 3×. That is a large win and it is why the ratio and the clip exist: they let you reuse expensive rollouts without the policy walking away from the distribution that generated them.

The subtlety is that `μ > 1` also means **your data is stale by construction**, and the clip is the only thing bounding how stale. Watch the clip fraction: the proportion of tokens where the clip actually binds. Near 0% means you could take more inner steps for free. Above ~15–20% means most of your update is being truncated, you are wasting compute on clipped gradients, and you should reduce `μ` or the learning rate.

**⚠ Trap:** the "first inner step must have `ρ = 1`" invariant is a load-bearing test, and it breaks constantly in real systems for a boring reason: you generate with vLLM/SGLang and compute logprobs with the training framework's forward pass. Different attention kernels, different batching, bf16 non-associativity, and possibly a different tokenizer path give you `ρ = 1.0 ± 0.02` instead of exactly 1. That looks harmless. It is not — it means your ratio contains pure noise, and with a tight `ε` the clip starts binding on noise rather than on real policy movement. Either recompute `old_logprobs` with the *training* forward pass on step 0, or explicitly track and bound the sampler/trainer logprob divergence as a monitored metric. This mismatch is one of the top three real-world GRPO bugs.

### Compare GRPO, RLOO, REINFORCE with a baseline, and PPO. When would you pick each?

They are four points on one axis: how you get the baseline that reduces variance in the policy gradient.

**REINFORCE with a moving-average baseline.** Baseline is a running mean of rewards across all prompts. Cheapest, one sample per prompt. Fails when prompts vary in difficulty — which they always do — because a hard prompt's success and an easy prompt's success get the same credit relative to a global average. Fine for toy setups, not for a real prompt pool.

**RLOO (REINFORCE Leave-One-Out).** Sample G per prompt; the baseline for sample `i` is the mean of the *other* G−1 samples: `A_i = r_i − (1/(G−1)) Σ_{j≠i} r_j`. This removes the `1/G` bias that GRPO's inclusive mean has. No std division, no clipping, no inner epochs. It is arguably the cleaner formulation and it is a genuinely strong baseline; the AI2 and Cohere lines of work made a good case for it. Note that RLOO's advantage is a fixed rescaling of GRPO's mean-centred advantage by `G/(G−1)`, so the two are closer than the different names suggest.

**GRPO.** Group-mean baseline, optional std division, PPO-style clipping so you can take multiple inner steps. Pick it when you want to amortize expensive rollouts over several gradient steps, or simply because the tooling ecosystem (TRL, veRL, OpenRLHF) has the most mature GRPO paths and you want to spend your weeks on the verifier rather than on the trainer.

**PPO with a learned critic.** Pick it when you have **dense or intermediate rewards** and genuinely need per-token credit assignment: multi-turn agents with per-step rewards, tasks with shaped intermediate signals, or anything where GAE's temporal credit distribution earns its cost. Also when your rollouts are so expensive that generating G samples per prompt is unaffordable and you would rather pay for a critic than for 8× decode. That inversion is real for agentic tasks with $0.05 rollouts.

**🗣 Say this in the room:** "For single-turn verifiable tasks I default to GRPO or RLOO — the terminal-reward structure means a critic buys you almost nothing and costs you a second trainable model. I'd go back to PPO with GAE when rewards are dense enough that per-token credit assignment is real information rather than noise, or when a rollout is expensive enough that paying for 8 samples per prompt costs more than paying for a critic."

**⚠ Trap:** presenting GRPO as strictly better than PPO. It is better *for the regime it was designed for* — single terminal reward, cheap sampling, one turn. Frontier labs still run critic-based methods for multi-turn and agentic settings precisely because the credit-assignment problem GRPO ignores is the dominant problem there. Saying "GRPO replaced PPO" is a 2024 talking point and it will read as not having tracked the field.
