### Before we get to the algorithm — why is speculative decoding even possible? What's the slack it exploits?

The slack is that **autoregressive decode leaves almost all of the GPU's arithmetic idle**, and speculative decoding is the trick that converts that idle arithmetic into tokens.

Here is the argument, and it is the single most important thing in this section. To generate one token at batch size 1, you must read every weight of the model out of HBM into the compute units, and then you do roughly two FLOPs per weight. So the ratio of arithmetic to memory traffic — the **arithmetic intensity** — is about 2 FLOPs per parameter-byte-ish, which for FP16 is ~1 FLOP per byte. An H100 does ~989 TFLOP/s dense FP16 against ~3.35 TB/s of HBM bandwidth, giving a machine balance of roughly **295 FLOPs per byte**. You are running at 1 against a machine that wants 295. Decode is memory-bandwidth-bound by a factor of a few hundred.

Concretely: a 70B model in FP16 is 140 GB of weights. At 3.35 TB/s (and no card holds 140 GB, so this is across at least two, but the ratio is what matters), one forward pass costs 140 GB / 3.35 TB/s ≈ **42 ms of pure memory traffic**, and only 2 × 70e9 = 1.4×10¹¹ FLOPs of arithmetic — which at 989 TFLOP/s would take **0.14 ms**. You spend 42 ms moving weights to do 0.14 ms of math.

So here is the observation: **the cost of a forward pass is nearly independent of how many tokens you push through it**, up to the point where you saturate compute. Verifying 5 candidate tokens in one forward pass costs essentially the same 42 ms as generating 1. The weights get read once either way.

Speculative decoding is exactly this arbitrage. Something cheap proposes K tokens; the expensive model checks all K in **one** forward pass, in parallel, using the same weight read it was going to pay for anyway. If the proposals are usually right, you emit several tokens per weight-read instead of one.

**🗣 Say this in the room:** "Decode is memory-bandwidth-bound — arithmetic intensity around 1 FLOP/byte against a machine balance near 300 — so a forward pass costs the same whether you push one token through it or eight. Speculative decoding spends that free compute: a cheap drafter proposes K tokens and the big model verifies all K in a single pass. It's not an approximation; it's using capacity you were already paying for."

**⚠ Trap:** candidates say "it's faster because the small model does most of the work." That's wrong and it's the tell. The big model still runs on every step and still processes every token that gets emitted. What changed is the *number of sequential big-model forward passes*, not the amount of big-model work per token. If you describe it as offloading work to the draft model, an interviewer who knows the area will stop trusting the rest of your answer.

### Walk me through one speculative step in tensor shapes. What actually happens?

Take a target model `M_p` and a draft `M_q`, and let the current context be `T` tokens with a warm KV cache in both models.

**Phase 1 — draft, K sequential small forward passes.** Run `M_q` autoregressively for `K` steps (say K=5). Each step is one forward pass of the small model, so this phase is `K` sequential passes but each is cheap. You come out with `K` proposed token IDs `x̃₁..x̃₅` and, critically, the **full probability distributions** `q(·|prefix), q(·|prefix,x̃₁), …` at each of those positions — shape `[K, vocab]`, e.g. `[5, 128256]`. You need those distributions, not just the argmax; the acceptance test uses them.

**Phase 2 — verify, one large forward pass.** Feed the target model the sequence `[context_tail, x̃₁, x̃₂, x̃₃, x̃₄, x̃₅]`. Because attention is causal, one pass over those 5 positions yields the target's next-token distribution *at each of the 5 positions simultaneously* — `p(·|prefix)`, `p(·|prefix,x̃₁)`, …, `p(·|prefix,x̃₁..x̃₅)`. That's shape `[6, vocab]`: five conditioned on the accepted-so-far prefix, plus one bonus distribution conditioned on all five having been accepted. This is the whole trick: **the target model tells you what it would have said at every prefix position, in one pass**, because that's what a causal forward pass over a sequence already computes.

**Phase 3 — accept/reject, sequentially over the K positions.** For `i = 1..K`, sample `r ~ U(0,1)` and accept `x̃ᵢ` if `r < p(x̃ᵢ)/q(x̃ᵢ)` (always accept if `p ≥ q`). On the first rejection at position `j`, discard `x̃ⱼ..x̃_K` and sample a replacement token from the **normalized residual** `norm(max(0, p − q))` at position `j`. If all K are accepted, sample a free bonus token from the K+1-th distribution `p(·|prefix,x̃₁..x̃_K)`.

**Phase 4 — KV bookkeeping, and this is where implementations get bugs.** The target's KV cache now contains entries for all 5 speculated positions, but only `n` were accepted. You must **truncate the KV cache back to `T + n`** (or `T + n + 1` counting the bonus/replacement token) in both models. In a paged engine this is decrementing the sequence length and freeing the tail blocks; in a naive PyTorch implementation it is slicing the cache tensors. Forget this and the model silently attends to tokens that were never emitted — which produces plausible-looking but wrong output that no unit test catches.

**📐 Numbers you must know:** per iteration you emit **between 1 and K+1** tokens. Never zero — even a total rejection at position 1 emits the residual-sampled replacement token, which is a real, correctly-distributed token. That guarantee is why speculative decoding cannot be worse than 1 token per big-model pass on the emit side; the only way it loses is on the *cost* side, via the draft overhead.

### Derive the rejection-sampling rule and prove to me the output distribution is exactly preserved.

This is the part that makes speculative decoding a *lossless* optimization rather than an approximation, and being able to derive it in three lines is a strong signal.

Setup: at some position, the draft proposes `x ~ q(·)` and the target's true distribution is `p(·)`. We want a procedure whose output is distributed exactly as `p`.

**The rule.** Sample `x ~ q`. Accept with probability `min(1, p(x)/q(x))`. If rejected, sample the output from the residual distribution
```
p'(y) = max(0, p(y) − q(y)) / Σ_z max(0, p(z) − q(z))
```

**The proof.** Write `P(output = x)` as (probability we proposed x and accepted it) + (probability we rejected anything and then drew x from the residual):

1. Accept path: `q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))`.
2. Rejection probability: `β = Σ_z min(q(z), p(z))` is the total accept probability, so `P(reject) = 1 − β`.
3. The residual normalizer is `Σ_z max(0, p(z) − q(z)) = Σ_z [p(z) − min(p(z), q(z))] = 1 − β`. So the reject path contributes `(1 − β) · max(0, p(x) − q(x)) / (1 − β) = max(0, p(x) − q(x))`.

Sum: `min(p(x), q(x)) + max(0, p(x) − q(x))`. If `p(x) ≥ q(x)` that's `q(x) + p(x) − q(x) = p(x)`. If `p(x) < q(x)` that's `p(x) + 0 = p(x)`. Either way **exactly `p(x)`**. ∎

Two corollaries you should volunteer.

**The acceptance rate has a closed form and a beautiful interpretation.** `α = E_{x~q}[min(1, p(x)/q(x))] = Σ_x min(p(x), q(x)) = 1 − TV(p, q)`, where TV is total variation distance. So **acceptance rate is one minus how far apart the two models' distributions are.** Everything about draft-model selection, distillation, and out-of-distribution collapse follows from this single identity: to raise α, reduce TV(draft, target) *on your traffic*.

**The residual is why rejection isn't wasted.** On rejection you don't fall back to running the target again — you already have `p` at that position from the verification pass, so you sample the replacement for free. That's what guarantees ≥1 token per iteration.

**📄 Paper:** Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers via Speculative Decoding* (ICML), and independently Chen et al. (2023), *Accelerating Large Language Model Decoding with Speculative Sampling* (DeepMind). Both established the draft-then-verify scheme with the modified-rejection-sampling correction and the exactness proof; they replaced the assumption that you must trade quality for decode latency.

**⚠ Trap:** "it's lossless in expectation" or "the outputs are close enough." No — it is **exactly** the target distribution, sample by sample, for any `q`, including a terrible `q`. A bad draft model makes it *slow*, never *wrong*. This matters practically: it means you can ship speculative decoding without re-running your eval suite for quality, which is the argument that gets it approved. (The caveat: exactness holds for the sampling procedure. Floating-point nondeterminism from different kernel paths and batch shapes means outputs won't be bitwise identical to non-speculative decoding — that's a reproducibility caveat, not a distributional one, and you should say so.)

### Implement a toy speculative decoding loop. Forty lines, from memory, with correct rejection sampling.

```python
import torch

@torch.no_grad()
def speculative_generate(target, draft, prompt_ids, n_new=128, K=4, temp=1.0):
    """Lossless speculative decoding. target/draft: callables ids -> logits [B,T,V]."""
    ids = prompt_ids                                   # [1, T]
    def probs(model, seq, n_last):                     # last n_last distributions
        logits = model(seq)[:, -n_last:, :] / temp
        return torch.softmax(logits, dim=-1)           # [1, n_last, V]

    while ids.shape[1] < prompt_ids.shape[1] + n_new:
        # ---- Phase 1: draft K tokens, keeping the distributions ----
        draft_ids, q_list = ids, []
        for _ in range(K):
            q = probs(draft, draft_ids, 1)[:, -1, :]   # [1, V]
            tok = torch.multinomial(q, 1)              # [1, 1]
            q_list.append(q)
            draft_ids = torch.cat([draft_ids, tok], dim=1)

        cand = draft_ids[:, ids.shape[1]:]             # [1, K] proposed tokens
        # ---- Phase 2: ONE target pass verifies all K (+1 bonus) ----
        p_all = probs(target, draft_ids, K + 1)        # [1, K+1, V]

        # ---- Phase 3: sequential accept / reject ----
        n_acc = 0
        for i in range(K):
            x = cand[0, i]
            p_i, q_i = p_all[0, i, x], q_list[i][0, x]
            if torch.rand(()) < torch.clamp(p_i / q_i, max=1.0):
                n_acc += 1
            else:
                resid = torch.clamp(p_all[0, i] - q_list[i][0], min=0)
                resid = resid / resid.sum()
                fix = torch.multinomial(resid, 1).view(1, 1)
                ids = torch.cat([ids, cand[:, :n_acc], fix], dim=1)
                break
        else:
            bonus = torch.multinomial(p_all[0, K], 1).view(1, 1)
            ids = torch.cat([ids, cand, bonus], dim=1)  # all K accepted + free token
    return ids
```

**🏋 Drill:** write this from a blank editor in **25 minutes, no autocomplete, no reference**. Pass criteria, all four: (1) the residual is `clamp(p − q, min=0)` renormalized, not `p` and not `p − q`; (2) the loop emits at least one token on every iteration including total rejection; (3) the all-accepted branch samples the bonus token from `p_all[K]`, the distribution conditioned on all K drafts; (4) you can state in one sentence why the output distribution is exactly the target's. Then, second pass: add a KV cache to both models and correctly truncate it to the accepted length. That second pass is what separates a toy from something you could put behind a serving engine, and it is where every real bug lives.

**⚠ Trap:** the `for…else` is doing real work here — `else` fires only when the loop completes without `break`, i.e. all K accepted. Candidates who restructure it with a flag usually get the bonus-token case wrong and silently emit K instead of K+1, throwing away the free token that is a meaningful share of the speedup. Also note the sequential accept loop: you cannot vectorize it, because rejection at position `i` invalidates positions `i+1..K` — the check is inherently a prefix scan.

### Derive the expected speedup as a function of acceptance rate and K. What does the curve look like?

Two quantities, and you should be able to write both from memory.

**Expected tokens per iteration.** Model acceptance at each position as i.i.d. Bernoulli(α) — an approximation, since acceptance is correlated along a sequence, but the standard one. You accept a geometric run: `P(exactly j accepted) = α^j(1−α)` for `j < K`, and `α^K` for all K. Emitting `j+1` tokens (accepted run plus the residual or bonus token):

```
E[tokens/iter] = Σ_{j=0}^{K} α^j = (1 − α^(K+1)) / (1 − α)
```

**Expected wall-clock speedup.** Let `c` = cost of one draft forward pass ÷ cost of one target forward pass. One iteration costs `K·c + 1` target-pass-equivalents. So

```
speedup = (1 − α^(K+1)) / ((1 − α)(K·c + 1))
```

**📐 Numbers you must know** — memorize this table, it is the fastest way to reason about whether speculation is worth it (c = 0.05, i.e. a draft ~5% the cost of the target):

| α | K=2 | K=4 | K=6 | K=10 | hard ceiling `1/(1−α)` |
|---|---|---|---|---|---|
| 0.5 | 1.59× | 1.61× | 1.53× | 1.33× | 2.0× |
| 0.7 | 1.99× | 2.31× | 2.35× | 2.18× | 3.3× |
| 0.8 | 2.22× | 2.80× | 3.04× | 3.05× | 5.0× |
| 0.9 | 2.46× | 3.41× | 4.01× | 4.57× | 10.0× |

Three readings of that table, and they are the whole engineering story.

**One: α dominates.** Going from α=0.5 to α=0.9 at K=4 takes you from 1.61× to 3.41×. Doubling K at fixed α buys far less. **Acceptance rate is the entire economics**; K is a second-order tuning knob. That is why the literature since 2023 has been an arms race on drafters (Medusa, EAGLE, Hydra) rather than on K.

**Two: the curve in K is non-monotonic and peaks.** At α=0.5 the optimum is around K=3–4 and K=10 is *worse* than K=2. The reason is structural: the numerator saturates at `1/(1−α)` — you cannot accept more than the geometric run allows — while the denominator grows linearly forever in `K·c`. You are paying for drafts you will almost certainly reject. **Take ∂/∂K and you get the optimal K; in practice sweep it empirically per workload.**

**Three: there is a hard ceiling of `1/(1−α)`.** Expected tokens per iteration is bounded above by the infinite geometric sum `1/(1−α)` no matter how large K gets, and the `K·c + 1` denominator only makes things worse, so `speedup ≤ 1/(1−α)` and equality requires `c → 0`. **At α=0.8 you cannot exceed 5× however clever your draft tree is.** Anyone quoting a 10× speculative speedup is either claiming α > 0.9, or using a zero-cost drafter on an unusually copy-heavy workload, or measuring something other than end-to-end latency.

**🗣 Say this in the room:** "Expected tokens per iteration is the geometric sum `(1 − α^(K+1))/(1 − α)`, and speedup divides that by `K·c + 1` for draft cost `c`. The reading that matters is that α dominates and K is second-order — and that speedup in K is non-monotonic, because the numerator saturates at `1/(1−α)` while the denominator grows linearly forever. At α=0.8 the ceiling is 5× no matter how clever the drafter is."

**⚠ Trap:** this formula assumes batch size 1 and a decode-bound target. At high batch the target's forward pass is no longer free-at-the-margin — verification of K tokens × B sequences pushes you into compute-bound territory and `K·c + 1` becomes an underestimate of the cost. See the batch-size question later; it is the single most common senior-level gotcha in this topic.

### How do you pick K in production, and what does the wrong K cost you?

Empirically, per workload, by sweeping — but you should walk in with priors, and you should know which direction the errors go.

**Priors by workload,** because α varies enormously by domain:
- **Code completion / code edit** — highly predictable continuations, α often 0.8–0.9 with a decent drafter. Push K to 6–8.
- **Structured/JSON output under a grammar** — extremely predictable, α can exceed 0.9; K of 8+ pays, and n-gram drafting alone often suffices.
- **Open-ended chat / creative** — α more like 0.5–0.7. K of 3–5.
- **Multilingual, math, or long-tail domains** — α frequently collapses below 0.5 if the drafter wasn't trained on them. K of 2–3, or turn speculation off for that route.

The asymmetry that matters: **overshooting K is worse than undershooting.** Undershooting leaves speedup on the table linearly; overshooting adds `c` per extra draft step *unconditionally* while the marginal accepted token arrives with probability `α^K`, which decays geometrically. At α=0.6 and K=10, the tenth draft token is accepted with probability 0.6¹⁰ ≈ 0.6%, and you paid full draft cost for it on every single iteration. **When in doubt, set K low.**

The adaptive answer, which is what modern engines actually do and what I'd propose: **make K a function of observed acceptance.** Track a running estimate of α (an EWMA over the last few hundred iterations, ideally per-request as well as globally) and adjust:

```python
# adaptive K: raise it when the drafter is being trusted, cut it fast when it isn't
alpha_ewma = 0.9 * alpha_ewma + 0.1 * (accepted_this_iter / K_current)
if alpha_ewma > 0.75 and K_current < K_MAX: K_current += 1
elif alpha_ewma < 0.45 and K_current > 1:   K_current -= 1
```

Asymmetric adjustment (fast down, slow up) is correct here for the same reason it is in congestion control: the cost of being too aggressive is paid every iteration, while the cost of being too conservative is only foregone upside. This also gives you graceful degradation when a request wanders out of distribution mid-generation — the model starts writing Japanese, α tanks, K drops to 1, and you're back to plain decoding instead of paying 8 wasted draft passes per token.

**💰 Math:** a 70B on 2×H100 at 40 tok/s baseline decode. At α=0.8, c=0.05, K=6 gives 3.04×, so ~122 tok/s. Serving 500,000 requests/day at 600 output tokens each: baseline needs 500,000 × 600 / 40 = 7.5M GPU-seconds/day of decode = 2,083 GPU-hours = **$5,208/day** at $2.50/GPU-hr. At 3.04× it's **$1,713/day** — a saving of ~$105,000/month. Push K to 12 and the formula gives 2.95×, i.e. $1,765/day: you handed back **$52/day** and added six wasted draft passes per iteration for it. Small in isolation, but it is pure loss, and the same mistuning at α=0.5 costs you 17% of the speedup. Sweep the knob.

### What actually determines the acceptance rate, and how would you measure it in production?

Since `α = 1 − TV(p, q)` on the realized distribution, α is determined by **how closely the drafter's next-token distribution tracks the target's, on your traffic**. That decomposes into four factors you can actually act on:

1. **Draft/target capability gap.** A 68M drafter against a 70B target has a large TV; a 7B drafter against a 70B has a small one. But cost scales with capability, so this trades directly against `c`.
2. **Distribution alignment, which is separate from capability.** A drafter *distilled from the target* has much lower TV than an equally-sized drafter trained independently, because distillation explicitly minimizes divergence from `p`. This is the highest-leverage lever and the reason draft distillation is standard practice.
3. **Domain match.** α is a property of the *realized* traffic. A drafter trained on English web text against Japanese legal prompts has enormous TV. This is the OOD-collapse failure mode.
4. **Sampling temperature.** At temperature 0 (greedy), acceptance reduces to "does the drafter's argmax match the target's argmax" — often high on predictable text. At high temperature both distributions flatten; `min(p,q)` summed over a flat distribution is actually *higher* than over two peaked distributions that disagree, so α can rise with temperature. The direction is workload-dependent — measure, don't assume, and be honest in the room that this one is not a simple monotone relationship.

**Measurement.** Instrument the loop directly; do not infer α from wall-clock speedup, because speedup confounds α with batch effects, scheduler behavior, and `c`. Emit per iteration:

- `drafted_tokens` (= K), `accepted_tokens` (0..K), `emitted_tokens` (1..K+1).
- **`acceptance_rate = Σ accepted / Σ drafted`**, aggregated per route, per model pair, and per **language/domain tag** if you have one.
- **`tokens_per_target_forward = Σ emitted / Σ target_passes`** — this is the metric that actually maps to throughput, and it's the one I'd put on the dashboard. It equals `E[tokens/iter]` and is directly comparable to the theoretical `(1−α^(K+1))/(1−α)`.
- The **distribution** of accepted-run length, not just its mean. A bimodal histogram — many 0s and many Ks — means you have two traffic populations with different α, and the right response is per-route configuration, not a global K.

**🗣 Say this in the room:** "Acceptance rate is one minus the total variation distance between the draft and target next-token distributions on the realized traffic — that's exact, not a heuristic. So the levers are draft capability, distributional alignment via distillation from the target, and domain match. I instrument accepted/drafted per route and tokens-per-target-forward-pass, and I look at the histogram of run lengths, because a bimodal histogram means I should be configuring K per route rather than globally."

### How do you choose a draft model? Talk me through the 68M-parameter case and what distillation buys you.

The selection problem is a two-variable optimization: maximize `α` while minimizing `c`, subject to the drafter fitting in memory alongside the target without cutting your KV budget.

**The tiny-drafter regime.** The canonical example is a ~68M-parameter Llama-architecture model drafting for a 7B–70B Llama target — the `llama-68m` class of checkpoint that became the community default for speculative decoding experiments. Against a 70B, `c ≈ 68M/70B ≈ 0.001` — the draft is essentially free, so even modest α pays. Plug into the formula at α = 0.6, K = 5, c = 0.001: `(1 − 0.6⁶)/(0.4 × 1.005) = 0.953/0.402 = 2.37×`. A model 1000× smaller than the target, agreeing with it only 60% of the time, still more than doubles throughput. **That's the headline: c so small that the draft cost basically vanishes from the denominator, so the entire question becomes α.**

Practical constraints that narrow the field:
- **Same tokenizer, mandatory.** Different vocabularies mean you cannot compare `p(x)` and `q(x)` at all; you'd need a cross-tokenizer mapping, which breaks the exactness proof. In practice this restricts you to drafters from the same model family, which is why "which drafter for Qwen?" has a much better answer than "which drafter for our fine-tuned proprietary model?"
- **Draft KV cache costs memory too.** A 68M drafter's KV is negligible; a 7B drafter's KV at long context is not, and it comes out of the same pool as your target's batch capacity.
- **Draft latency is on the critical path K times per iteration**, including kernel launch overhead. A tiny model can be *launch-bound* rather than compute-bound, so `c` measured in wall clock is worse than the parameter ratio suggests. CUDA graphs on the draft loop matter here.

**Distillation is the highest-leverage lever available.** Train the drafter on the *target's output distributions* — sequence-level knowledge distillation on the target's own generations, ideally on your production traffic distribution — rather than on generic pretraining data. You are directly minimizing the divergence that α is defined by. Reported gains in the literature are substantial (α improvements of tens of points translating to roughly +0.5× to +1× end-to-end), and the intuition is exact: distillation minimizes KL(p‖q), and α = 1 − TV(p,q), and Pinsker's inequality bounds TV by √(KL/2). **Lower the KL and you provably raise the acceptance floor.**

**⚠ Trap:** picking the drafter by benchmark quality. A drafter that scores better on MMLU is not necessarily a better drafter — you want *agreement with the target*, not *correctness*. A drafter that is smarter than the target in some domain will disagree with it and get rejected. The selection metric is α on your traffic, full stop, and I'd reject a design doc that justifies a drafter choice with any benchmark other than measured acceptance rate.

### What if I don't want to run a separate draft model at all? Walk me through self-speculation and Medusa.

Two families here, and the motivation for both is operational: a separate draft model is another checkpoint to version, another set of weights in HBM, another thing to distill when the target changes, and an extra tokenizer-compatibility constraint. Self-drafting deletes all of that.

**Self-speculation / layer skipping.** Use a *subset of the target's own layers* as the drafter: run the first `L/2` layers, project to vocab with the shared LM head, and treat that as `q`. Zero extra weights, zero extra memory, guaranteed tokenizer match. The catch is that the KV cache for skipped layers must be handled coherently — the draft pass populates only the early layers' KV, and the verify pass must fill the rest — which is real engine complexity. Acceptance rates are typically decent but below a distilled dedicated drafter, because early-exit representations are not calibrated as next-token distributions.

**Medusa** takes a different route: bolt **multiple extra decoding heads** onto the target's final hidden state, where head `k` predicts the token at position `t+k+1`. One forward pass of the target produces the hidden state, and the heads read it in parallel to emit several candidate continuations at once — so **the drafting itself costs one head-forward, not K sequential model passes.** Because each head is uncertain, Medusa takes the top few candidates per head and forms their **Cartesian product as a token tree**, then verifies the whole tree in a single target pass using **tree attention**: a specially-constructed causal mask over the flattened tree such that each candidate node attends only to its own ancestors, never to sibling branches. That mask is the key implementation object — it lets you verify dozens of candidate continuations for the price of one forward pass over a slightly longer sequence.

**📄 Paper:** Cai et al. (2024), *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* Introduced multi-head parallel drafting with tree attention and typical-acceptance verification, replacing the need for a separate draft model. Medusa-1 trains the heads with the backbone frozen (cheap, preserves the base model exactly); Medusa-2 trains heads and backbone jointly for higher acceptance at the cost of touching the base model.

**⚠ Trap:** Medusa's default configuration uses a *typical acceptance* criterion rather than the exact rejection-sampling rule — it accepts candidates whose target probability exceeds a threshold, which is **not distribution-preserving.** It trades a small amount of exactness for higher acceptance. If your compliance or eval story depends on "speculative decoding is provably lossless," you must check which acceptance criterion your engine is configured with, because the guarantee is not automatic. This is a genuinely good thing to raise unprompted in an interview; it shows you've read past the abstract.

### Explain the EAGLE line. What did EAGLE-1 change, and what did 2 and 3 add?

EAGLE is the most important development in this area since the original 2023 papers, and the core insight is one sentence: **draft in feature space, not token space.**

**EAGLE-1.** Medusa's heads predict tokens directly from the final hidden state, which is hard because the mapping from one hidden state to *several future tokens* is highly uncertain. EAGLE instead trains a small autoregressive head that predicts the target model's **second-to-top-layer feature vector** for the next position, and — this is the crucial second ingredient — **feeds in the actually-sampled token** from the previous step to resolve the ambiguity. Feature sequences are smoother and more predictable than token sequences (they're continuous and lower-entropy), so a small head tracks them well; and conditioning on the sampled token removes the fundamental uncertainty that makes multi-step token prediction hard. The predicted feature goes through the target's own frozen LM head to produce `q`, so the drafter inherits the target's output geometry for free.

**📄 Paper:** Li et al. (2024), *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* Moved drafting from token-level to feature-level autoregression with sampled-token conditioning; substantially higher acceptance than Medusa-style independent heads at comparable drafter cost.

**EAGLE-2.** The draft *tree* in EAGLE-1 (and Medusa) is **static** — a fixed shape, same branching factor at every position, regardless of context. EAGLE-2's observation is that the drafter's own confidence is a good proxy for acceptance probability, so it builds a **context-aware dynamic draft tree**: expand where the drafter is confident, prune where it isn't, and re-rank candidates before verification so the fixed verification budget is spent on the branches most likely to be accepted. Same drafter, better token allocation.

**EAGLE-3.** The reported limitation of EAGLE-1/2 is that the feature-prediction objective *constrains* the drafter — forcing it to reproduce a specific intermediate representation caps how much it benefits from more training data. EAGLE-3 drops the feature-prediction loss in favour of direct token prediction, and fuses features from **multiple layers** of the target rather than one, with a training procedure that simulates the multi-step drafting conditions it will face at inference. The result reported is better scaling with training data and higher acceptance.

**📅 Volatile:** the reported speedup figures across this line (roughly 3× for EAGLE, ~4× for EAGLE-2, higher for EAGLE-3, all measured at **batch size 1** in the papers' settings) are the numbers that get quoted and are the numbers that mislead. They are batch-1 latency results on specific model/dataset pairs. **Do not quote them as production throughput figures** — at production batch sizes the picture changes qualitatively, as we'll get to. If asked, give the mechanism and say "the papers report X at batch 1; I'd expect substantially less at our serving batch and I'd measure it."

### Cover the rest of the family for me — Hydra, SpecInfer, Lookahead. What distinct idea does each contribute?

**Hydra** fixes a specific defect in Medusa: Medusa's heads are **conditionally independent** given the hidden state — head 3 predicts token `t+3` without knowing what head 2 predicted for `t+2`. That's a bad factorization of a sequence, and it caps acceptance because the heads can propose mutually incoherent continuations. Hydra makes the draft heads **sequentially dependent**: each head's input includes the previous head's sampled token, restoring the autoregressive structure inside the draft while keeping the cheap multi-head form.

**📄 Paper:** Ankner et al. (2024), *Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding.* Replaced Medusa's independent heads with sequentially-conditioned ones; higher acceptance at essentially the same drafting cost.

**SpecInfer** contributes the systems-level generalization: **tree-based speculation and verification as a serving primitive.** Instead of one linear draft sequence, merge proposals from *multiple* small draft models (or multiple sampled continuations) into a single **token tree**, and verify the entire tree against the target in one forward pass with a tree-structured attention mask. The insight is that if drafters disagree, you shouldn't pick one — you should verify both branches simultaneously, because the marginal cost of extra tokens in a memory-bound verification pass is nearly zero. Tree verification is now the standard shape; Medusa, EAGLE-2, and modern engine implementations all use some form of it.

**📄 Paper:** Miao et al. (ASPLOS 2024), *SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification.* Established multi-drafter token-tree speculation with a tree-attention verification kernel, replacing single-sequence draft-then-verify.

**Lookahead decoding (Jacobi-style)** removes the drafter entirely and reframes decoding as **solving a nonlinear system by fixed-point iteration.** Autoregressive decoding is a triangular system `x_i = f(x_{<i})`; Jacobi iteration guesses all positions at once and refines them in parallel until they stop changing, which provably converges to the same output. Lookahead decoding makes this practical by maintaining a pool of n-grams generated as a byproduct of the Jacobi trajectory and using them as verification candidates each step. **No draft model, no training, no extra weights** — it trades those for extra FLOPs per step, which is exactly the resource decode has spare.

**📄 Paper:** Fu et al. (2024), *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding* (ICML). Reframed speculation as Jacobi iteration plus an n-gram cache, eliminating the draft model.

**The taxonomy to hold in your head, because interviewers ask you to organize the space rather than list it:** every one of these methods is answering one question — *where does `q` come from?* A separate model (Leviathan/Chen, SpecInfer), the target's own early layers (self-speculation), extra heads on the target (Medusa, Hydra, EAGLE), or no model at all (Lookahead, n-gram/prompt lookup). And every one of them uses the *same* verification step and, when done correctly, the *same* rejection sampling rule. **Draft is where the innovation is; verify is invariant.**

### Tell me about prompt-lookup / n-gram decoding. Why does it work so well for RAG and code editing specifically?

This is the cheapest trick in the entire section and the one I'd ship first, before any draft model.

**The mechanism:** the drafter is a **string search**. Take the last `n` tokens generated (n typically 2–4), search backwards through the *prompt itself* for a previous occurrence of that n-gram, and if you find one, propose the tokens that followed it there as your K draft tokens. No model, no weights, no training, no GPU. It is `str.find` in the token buffer, running on the CPU in microseconds, so `c ≈ 0` and the speedup formula collapses to just `(1 − α^(K+1))/(1 − α)`.

**Why it wins on specific workloads.** The method only works when output copies substantially from input — and there is a whole class of very high-value workloads where output is *mostly* a copy:

- **RAG and grounded summarization.** The model quotes retrieved passages, names, numbers, and entity strings verbatim from context. Every quoted span is a long accepted run.
- **Code editing and refactoring.** The model is emitting a modified version of code that is right there in the prompt. Rewriting a function to add error handling means 85% of the emitted tokens are byte-identical to the input, and the n-gram lookup nails every unchanged line.
- **Structured extraction into a schema** whose field names appear in the prompt.
- **Diff/patch generation, translation with heavy terminology reuse, and document-grounded QA.**

Because accepted runs on copied spans are *long* — you can accept 10+ tokens in a row when the model is transcribing — the effective tokens-per-iteration on these workloads can far exceed what a small draft model achieves, and it costs nothing.

**📄 Paper:** there isn't one, and you should say so rather than invent a citation. Prompt-lookup decoding was popularized by Apoorv Saxena's 2023 implementation and was absorbed into the major serving engines as an n-gram speculation mode. Its intellectual ancestor is the n-gram pool inside Lookahead decoding.

**The decision rule I'd give a team at Cursor, Glean, or Perplexity:** turn on n-gram speculation **first**, measure the acceptance rate on your real traffic, and only invest in a draft model if the n-gram acceptance is poor. It has zero memory cost, zero training cost, zero additional failure modes, and it degrades to plain decoding when there's no match (a lookup miss simply means no speculation that step).

**🗣 Say this in the room:** "The drafter is a string search over the prompt — match the last few generated tokens against the input and propose whatever followed them there. `c` is effectively zero because it runs on CPU, so the speedup is purely the acceptance term. On RAG, code edits, and structured extraction, output is largely a copy of input, so accepted runs are long and it beats a small draft model at zero cost. I'd turn it on before considering any draft model at all."

**⚠ Trap:** it is *anti-correlated* with the workloads a draft model helps most. Open-ended creative generation, novel reasoning, and answers that synthesize rather than quote have essentially zero n-gram hits, and you'll measure α near 0 and conclude speculation "doesn't work." It doesn't work *for that route*. The right architecture is **route-dependent speculation**: n-gram on the copy-heavy routes, a draft model or EAGLE head on the generative routes, and nothing at all on routes where neither pays.

**💰 Math:** a code-edit endpoint emitting 900 tokens, of which ~80% are copied from the prompt. With n-gram drafting at K=8 and a measured α of 0.85 on the copied spans, `E[tokens/iter] = (1 − 0.85⁹)/0.15 = (1 − 0.232)/0.15 = 5.12` tokens per target forward pass, with `c ≈ 0`. Baseline 900 target passes at 25 ms each = 22.5 s; speculative 900/5.12 = 176 passes = **4.4 s**. That is a 5× latency cut on the exact metric a coding product is judged on, for a feature flag and a CPU-side string search.
