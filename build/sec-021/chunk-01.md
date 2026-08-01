### Before any math — tell me what problem DPO actually solved, in operational terms, not theoretical ones.

DPO did not solve a modelling problem. It solved an *orchestration* problem, and if you frame it as a math result in an interview you will sound like someone who read the abstract.

Here is the mental model. PPO-based RLHF is a distributed system with four models in the loop: the policy you are updating, a frozen reference for the KL penalty, a reward model, and a value/critic head. Every optimization step requires you to *generate* — sample completions from the current policy, score them with the RM, compute advantages, then do a clipped policy update. Generation is autoregressive decode, which is memory-bandwidth-bound and slow, so you end up building an inference server inside your training loop just to keep the GPUs fed. That is why RLHF teams have a dedicated infra engineer: the hard part is rollout throughput, weight synchronization between the trainer and the sampler, and the fact that the whole thing has about six knobs that can silently produce garbage.

DPO replaces all of that with a supervised loss. You take a static dataset of `(prompt, chosen, rejected)` triples, run four forward passes (policy and reference, on chosen and rejected), compute a scalar loss, and backprop. No sampling, no reward model, no value head, no rollout buffer. It is architecturally indistinguishable from SFT with a slightly funnier loss function — which means it runs on the training stack you already have, in hours instead of days, and a single engineer can own it.

**🗣 Say this in the room:** "DPO's contribution is that it shows the KL-constrained RLHF objective has a closed-form optimal policy, and if you invert that relation you can express the reward *in terms of the policy*. Substitute that into the Bradley-Terry likelihood and the reward model disappears — you are left with a classification loss over preference pairs. Operationally that turns a four-model online RL system into a two-model offline supervised job, and that is why essentially every open-weight instruct model between 2024 and 2026 shipped with a DPO-family stage rather than PPO."

**📄 Paper:** Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model* — showed the RLHF reward model and the optimal policy are the same object under a change of variables, replacing the RM-training + PPO pipeline with one supervised loss.

**⚠ Trap:** claiming "DPO is equivalent to RLHF." It is equivalent *at the exact optimum, with unlimited data, an unconstrained reward class, and the same reference policy*. In practice you optimize on a finite offline dataset, which means the KL constraint is only meaningfully enforced on the support of that dataset. Off the support, the implicit reward is unconstrained and the model can put mass wherever it likes. That gap is the source of nearly every documented DPO pathology, and naming it is how you signal you have actually run these jobs.

### Derive the DPO objective from the KL-constrained RLHF objective. Whiteboard, take your time.

This is the single most-asked derivation in post-training interviews, so I rehearse it as five beats and I say the beats out loud as I write.

**Beat 1 — write the RLHF objective.** For a prompt distribution `D` and a reward `r`:

```
max_π  E_{x~D} E_{y~π(·|x)} [ r(x,y) ]  −  β · D_KL( π(·|x) ‖ π_ref(·|x) )
```

β is the strength of the tether to the reference model. Everything downstream is a consequence of β.

**Beat 2 — fold the KL into the expectation and flip to a minimization.** Since `D_KL = E_{y~π}[log(π/π_ref)]`, the whole objective is a single expectation under π:

```
max_π  E_x E_{y~π} [ r(x,y) − β log( π(y|x) / π_ref(y|x) ) ]
```

Divide by −β and flip:

```
min_π  E_x E_{y~π} [ log( π(y|x) / π_ref(y|x) ) − (1/β) r(x,y) ]
```

**Beat 3 — complete the KL.** This is the step people fumble. The bracket is *almost* a KL divergence against something; you just have to name that something. Define

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β ),   Z(x) = Σ_y π_ref(y|x) exp( r(x,y)/β )
```

Z(x) is the partition function — a sum over every possible completion, so it is utterly intractable, and that intractability is exactly what everyone before DPO thought forced you into RL. Now rewrite the bracket:

```
log(π/π_ref) − (1/β) r  =  log( π / (π_ref · exp(r/β)) )
                        =  log( π / (Z(x) · π*) )
                        =  log( π / π* ) − log Z(x)
```

So the objective becomes

```
min_π  E_x [ D_KL( π(·|x) ‖ π*(·|x) ) − log Z(x) ]
```

`Z(x)` does not depend on π, so it is a constant with respect to the optimization. KL is non-negative and zero only when the arguments are equal. **Therefore the optimal policy is π = π\*, exactly, with no approximation.** That is the whole trick: the KL-constrained RLHF problem has a closed-form solution, it is just one you cannot sample from because of Z.

**Beat 4 — invert.** Take logs of the π* expression and solve for r:

```
r(x,y) = β log( π*(y|x) / π_ref(y|x) )  +  β log Z(x)
```

This says *any* reward function induces an optimal policy, and conversely any policy defines a reward — up to a term `β log Z(x)` that depends only on the prompt, not on the completion.

**Beat 5 — substitute into Bradley-Terry, watch Z die.** The preference model is

```
P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) )
```

Substitute the inverted reward. The `β log Z(x)` terms are identical for both completions of the same prompt, so they cancel:

```
P(y_w ≻ y_l | x) = σ( β log(π(y_w|x)/π_ref(y_w|x)) − β log(π(y_l|x)/π_ref(y_l|x)) )
```

Maximum likelihood over your preference dataset gives the DPO loss:

```
L_DPO = − E_{(x,y_w,y_l)~D} [ log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

**🗣 Say this in the room** as the one-sentence summary while you underline the last line: "The intractable partition function only ever appears as a per-prompt constant, and Bradley-Terry only ever looks at reward *differences* within a prompt — so it cancels, and what's left is a binary classification loss on preference pairs where the classifier is the language model itself."

**⚠ Trap:** saying "Z(x) cancels because it's a normalizing constant." That is not sufficient and a sharp interviewer will push. Z(x) cancels because it is a function of `x` alone and BT compares two completions *of the same prompt*. If you were comparing across prompts, it would not cancel. Say the second version.

**⚠ Trap:** forgetting to state that BT is an assumption. The derivation is exact given (a) the KL-constrained objective, (b) an unconstrained reward class, and (c) the Bradley-Terry model of human preference. (c) is the shakiest: BT assumes preferences are generated by a latent scalar utility with logistic noise and are transitive. Real annotators are non-transitive and their noise is not logistic. IPO exists precisely because of what happens when you take BT literally.

### Why does DPO not need to learn a reward model at all? Be precise about what is being reparameterized.

The reparameterization is: *the space of reward functions and the space of policies are the same space, under a bijection induced by β and π_ref.*

Concretely, given π_ref and β, the map `r ↦ π*` is `π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β)`, and the inverse map is `π ↦ β log(π/π_ref)`. It is not injective on rewards in the strict sense — two rewards differing by any function `f(x)` of the prompt alone induce the *identical* optimal policy, because the difference gets absorbed into Z(x). So the right statement is that the map is a bijection between *equivalence classes* of rewards (modulo prompt-only shifts) and policies.

That equivalence class matters, and it is the source of a nice interview follow-up. Bradley-Terry is also invariant to prompt-only shifts, so the reward class DPO can express is exactly the reward class that preference data can identify. Nothing is lost. Under the DPO parameterization, the language model *is* the reward model — you just have to read it out as `r̂(x,y) = β log(π_θ(y|x)/π_ref(y|x))`.

**⚠ Trap:** thinking "DPO has no reward model" means "DPO does not overoptimize a reward." It has an implicit reward model and it overoptimizes it just fine. The published scaling-law work on direct alignment algorithms shows the same hump-shaped reward-vs-KL curve familiar from explicit RM overoptimization — quality rises, peaks, and then degrades as KL from the reference grows, even though there is no separate RM to hack. Goodhart does not care whether your proxy is a separate network or a log-ratio.

**📄 Paper:** Rafailov, Chittepu, Park et al. (2024), *Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms* — documents that DPO-family methods exhibit the same overoptimization curve as explicit-RM RLHF, which many people assumed DPO had escaped.

### Implement the DPO loss from scratch. No TRL, no imports beyond torch.

Two functions, and both should be writable from memory in under ten minutes.

```python
import torch
import torch.nn.functional as F

IGNORE = -100

def seq_logprob(logits, labels):
    """logits: (B, T, V) over the full sequence. labels: (B, T) with IGNORE on
    the prompt and padding. Returns (B,) summed log-prob of the completion."""
    logits = logits[:, :-1, :]                 # position t predicts token t+1
    labels = labels[:, 1:]
    mask = labels != IGNORE
    safe = labels.masked_fill(~mask, 0)        # gather needs a valid index
    logp = logits.log_softmax(dim=-1)
    per_tok = torch.gather(logp, 2, safe.unsqueeze(2)).squeeze(2)
    return (per_tok * mask).sum(dim=-1)

def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1):
    """All four args are (B,) summed completion log-probs."""
    logits = (pol_c - pol_r) - (ref_c - ref_r)     # = (r̂_w − r̂_l) / beta
    loss = -F.logsigmoid(beta * logits)
    chosen_rw   = beta * (pol_c - ref_c).detach()  # implicit rewards, for logging
    rejected_rw = beta * (pol_r - ref_r).detach()
    return loss.mean(), chosen_rw, rejected_rw
```

Four things I check in review on any hand-rolled version:

1. **The shift.** HuggingFace causal LMs do not shift for you when you compute logits yourself; `logits[:, t]` predicts `labels[:, t+1]`. Off-by-one here produces a loss that trains and converges and is subtly wrong.
2. **Sum, not mean.** Vanilla DPO uses the *summed* completion log-probability. Switching to a per-token mean is not a stylistic choice — it changes the objective into something closer to SimPO and changes the length dynamics completely. If you want length normalization, make that an explicit, argued decision.
3. **Masking.** Prompt tokens and padding must both be `IGNORE`. A common bug is masking padding but not the prompt, which makes the chosen/rejected log-probs dominated by the shared prompt — and since the prompt is identical for both, the difference is *nearly* right but the gradients flow into prompt tokens, which is wrong and slows convergence.
4. **`.detach()` on the logged rewards** so your monitoring does not silently keep a graph alive and OOM you at step 4,000.

**🏋 Drill:** twenty-five minutes, no references. Write `seq_logprob` and `dpo_loss`, then write a test that constructs a two-token vocabulary, a batch of 4, and asserts (a) loss is `−log σ(0) = log 2 ≈ 0.693` when policy equals reference, (b) loss decreases monotonically as you increase `pol_c`, and (c) `rewards/accuracies` — the fraction of the batch with `chosen_rw > rejected_rw` — is 0.5 at init. Pass criterion: all three assertions green on the first run after you stop typing.

### Write out the gradient of the DPO loss and tell me what each factor is doing.

Differentiating `L = −log σ(β·[(logπ_c − logπ_ref,c) − (logπ_r − logπ_ref,r)])` and using `d/dz log σ(z) = σ(−z)`:

```
∇_θ L = − β · E[ σ( r̂(x,y_l) − r̂(x,y_w) ) · ( ∇_θ log π_θ(y_w|x) − ∇_θ log π_θ(y_l|x) ) ]
```

Read it in two parts.

The **direction** is `∇log π(y_w) − ∇log π(y_l)`: push up the likelihood of the chosen completion, push down the likelihood of the rejected one. That is a contrastive update, and it is the same shape as any noise-contrastive objective.

The **magnitude** is `σ(r̂_l − r̂_w)` — the probability that the *implicit reward model currently gets this pair backwards*. When the model already ranks the pair correctly with a wide margin, `r̂_l − r̂_w` is very negative, σ of it is near zero, and the example contributes almost no gradient. When the model has it wrong, σ is near one and the example dominates. DPO is automatically hard-example-weighted; you do not need a curriculum.

Three consequences I would want a candidate to draw unprompted:

**It only constrains the difference.** Nothing in the gradient says "keep `log π(y_w)` high in absolute terms." Raising the margin by shoving `log π(y_l)` down is exactly as rewarded as raising `log π(y_w)`. That is the mechanical root of the "both likelihoods fall" pathology.

**β appears twice with opposite effects.** It scales the gradient linearly (bigger β, bigger steps) *and* it sits inside the σ, so bigger β saturates the weighting faster (bigger β, fewer examples contribute). The net effect is that raising β both tightens the KL tether and shortens the effective training signal. This is why β and learning rate cannot be tuned independently.

**Once accuracy saturates, learning stops.** If `rewards/accuracies` hits 0.98, then 98% of your batch has σ ≈ 0 and your effective batch size has collapsed to 2% of nominal. The loss curve will look beautifully flat and nothing is happening. That is your signal to stop, not your signal that it worked.

### What is β physically, and walk me through what you'd see at β = 0.01 versus β = 0.5.

β is the inverse temperature on the KL tether — the price, measured in nats of reward, that the optimizer must pay per nat of divergence from the reference policy. Small β means the reference is cheap to abandon; large β means it is expensive.

The standard operating point is **β = 0.1**, and I would say that out loud as the default and then justify departures. The Zephyr-7B recipe — the one most open-weight DPO runs are descended from — used β = 0.1 with a learning rate of 5e-7 for full fine-tuning, cosine schedule, small warmup, and 1–3 epochs. Those three numbers travel together; a candidate who says "β = 0.1, lr 5e-7" has clearly run one.

At **β = 0.01**: the tether is nearly gone. The model drifts hard from the reference, `rewards/margins` grows without bound, and you get the full catalogue of pathologies quickly — length inflation, loss of formatting learned in SFT, degenerate repetition in the tail of long generations, and a collapse in output diversity. Your win rate on a judge-based benchmark may still *go up* for the first few hundred steps, which is why this is dangerous: the eval that motivated the run rewards the drift.

At **β = 0.5**: the tether dominates. The implicit rewards move very little, `rewards/margins` creeps up to maybe 0.3 instead of 3, the model stays close to the SFT checkpoint, and your capability suite shows almost no regression — because almost nothing changed. This is the correct choice when the SFT checkpoint is already good and you are making a narrow behavioral correction (say, tightening citation formatting) rather than a broad quality push.

**📐 Numbers you must know:** β = 0.1 default; the useful range is roughly 0.01–0.5; and `rewards/margins` at the end of a healthy run typically lands in the low single digits of nats. Derivation of why the range is what it is: the implicit reward is `β · Δ log-prob`, and a meaningful behavioral change corresponds to a summed log-prob shift of order 10 nats over a 300-token completion (≈ 0.03 nats/token). At β = 0.1 that is a reward of ~1 nat, which puts σ comfortably in its responsive region. At β = 0.01 you would need a 100-nat log-prob shift to move the loss at all, which is a catastrophic distribution change. The math is telling you the safe range.

**⚠ Trap:** tuning β on training loss. Lower β always produces lower training loss, because it makes the sigmoid easier to saturate. β must be selected on a held-out evaluation that includes a *capability regression* suite, not on the loss curve and not on win rate alone.

**🔍 Failure taxonomy — reading a run by its β symptoms.** Length grew >40% and MMLU dropped >2 points → β too low, or too many epochs; raise β to 0.2 and cut to one epoch. `rewards/margins` flat near zero after a full epoch → β too high, or your pairs are not actually distinguishable; check pair quality before you touch β. Margins growing but win rate flat → you are optimizing the implicit reward without improving the policy, i.e. overoptimization; stop and take an earlier checkpoint.

### The implicit reward β·log(π/π_ref) — what can you actually do with it at inference time?

This is my favorite "does this person get it" question, because the answer is a genuinely useful production trick that most people who have run DPO have never thought of.

After DPO, you have a reward model for free. `r̂(x,y) = β [ log π_θ(y|x) − log π_ref(y|x) ]` is, by construction, the reward whose KL-constrained optimum is your trained policy. It is a scalar you can compute for any `(x, y)` with two forward passes and no extra training.

Three uses:

**Best-of-n reranking.** Sample n completions from π_θ, score each with `r̂`, return the argmax. This is the cheapest quality lever available after alignment and it needs no new model. The cost is n decodes plus 2n scoring forwards — but scoring is prefill, not decode, so it is compute-bound and cheap. **💰 Math:** for a 300-token completion, decode at 50 tok/s is 6 s of wall clock per sample; scoring 300 tokens of prefill on the same 8B model at, say, 20k tok/s prefill throughput is 15 ms per pass, 30 ms for both models. So best-of-8 costs 8 decodes (parallelizable across the batch, so ~1 decode of latency if you have the capacity) plus 8 × 30 ms = 240 ms of scoring. You are buying a meaningful quality lift for a quarter-second of added latency and 8× the output tokens.

**Data filtering for the next round.** Score your candidate preference pairs with `r̂` from the previous round's model. Pairs where the current model already has a large positive margin teach it nothing (recall the gradient weighting σ(r̂_l − r̂_w) ≈ 0). Pairs where it is confidently *wrong* are the highest-value training signal — and also the most likely to be mislabeled, so those are the ones you send to human adjudication. This is straightforward active learning and it is the highest-ROI thing you can do to a preference dataset.

**Online-DPO scoring.** In an iterative loop, `r̂` from round k can rank the round-k+1 samples, giving you a preference labeler that costs two forward passes instead of a judge-model API call.

**⚠ Trap:** using `r̂` as a *calibrated* quality score across different prompts. It is not. Remember the derivation: the reward is only identified up to `β log Z(x)`, a prompt-dependent constant that cancelled in Bradley-Terry. So `r̂` is meaningful for comparing completions *of the same prompt* and meaningless for comparing across prompts. If you build a dashboard that averages `r̂` over a mixed traffic sample and alerts on it, you have built a noise generator. I have seen this ship.

### What does the reference model cost you, and what are my options for not paying it?

Take an 8B policy, bf16, full fine-tune with AdamW and mixed precision. Weights 8e9 × 2 = 16 GB. Gradients bf16 = 16 GB. AdamW first and second moments in fp32 = 8e9 × 8 = 64 GB. fp32 master weights = 32 GB. That is ~128 GB before a single activation, so you are already on FSDP/ZeRO across multiple H100s. The reference model adds 16 GB of forward-only weights — about 11% more memory. That part is annoying but survivable.

The compute cost is worse and people miss it. Per optimizer step you run: policy forward+backward on chosen and rejected (≈ 6ND FLOPs per token, twice), plus reference forward on chosen and rejected (≈ 2ND per token, twice). So the reference is `2×2 / (2×6 + 2×2) = 4/16 = 25%` of your step FLOPs. A quarter of your GPU time is spent computing numbers that never change.

Four ways out, in the order I reach for them:

**1. Precompute reference log-probs.** One pass over the dataset, store two float32 scalars per pair. This is what `precompute_ref_log_probs=True` does in TRL. **💰 Math:** the precompute costs `2×2ND` per pair once; a 3-epoch run costs `3 × 16ND` per pair. So you pay 4/48 ≈ 8% one time to remove 25% from every step — net saving ~20% of the run, and you free the 16 GB. For a 10-GPU-hour job at $3/H100-hour that is $6, which sounds trivial until you are doing weekly iterations on a 70B and the same ratio saves you $400 a week. Storage is nothing: 60k pairs × 8 bytes = 480 KB.

**2. LoRA with adapter-disable.** If the policy is `base + LoRA`, then the reference *is* the base — just run the forward with adapters disabled. Zero extra weights, zero extra memory, though you still pay the extra forward pass. This is the default I recommend for anyone doing DPO under 8 GPUs, and TRL will do it automatically when it detects a PEFT model and you pass `ref_model=None`.

**3. Reference-free objectives.** ORPO, SimPO and CPO drop the reference entirely by construction. That is not free — see the question later in this section on what you lose.

**4. Quantize or offload the reference.** It is forward-only, so int8 is usually acceptable and it can live on a separate device or even be served over RPC. I reach for this last because it adds a moving part for a modest win.

**⚠ Trap:** precomputing reference log-probs while dropout is active, or with a different tokenization/masking path than the training loop uses. The reference log-probs must be computed in eval mode with *exactly* the same label masking. If your precompute path pads to a different length or masks the EOS differently, you get a constant offset in `ref_c − ref_r` that quietly biases the entire run. The tell is `rewards/chosen` and `rewards/rejected` both sitting at a large nonzero value at step 0, when both should be exactly 0.0.

**⚠ Trap:** precomputing when you are doing iterative or online DPO. The moment the reference changes between rounds, or you generate new completions mid-run, the cache is stale and silently wrong. Precompute is for a single offline pass with a fixed reference.

### Why is an SFT warm start effectively mandatory before DPO?

Because π_ref is not a regularizer you chose for convenience — it is the *base measure* of the entire objective, and the derivation only means anything if the preference data lives on π_ref's support.

Look at the optimal policy again: `π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β)`. If `π_ref(y_w|x)` is tiny — because the chosen completion came from GPT-4 and your reference is a raw base model that has never emitted a well-formatted assistant turn — then reaching π* requires an enormous multiplicative change, which means an enormous KL, which means the tether is doing no work and you are effectively running unconstrained maximum likelihood on a contrastive objective. Empirically that produces exactly what you would predict: the model learns the *surface form* of the preference data while its behavior off-distribution degenerates.

There is a second, simpler reason. On a base model, `log π(y_w) − log π(y_l)` is dominated by which completion happens to look more like generic web text, not by which is better. Your preference signal is buried under a much larger formatting signal, and DPO spends its gradient budget learning "assistant turns look like this" — which is SFT, done badly and with half the data.

The DPO paper itself is explicit about this: they initialize π_ref to the SFT model, and they note that when an SFT model is unavailable you should first fine-tune on the *chosen* completions to construct π_ref. That is the recipe. If someone hands you preference data and no SFT checkpoint, your first move is a one-epoch SFT on the chosen side, and that becomes both your init and your reference.

**🗣 Say this in the room:** "π_ref is the base measure of the objective, not just a regularizer. DPO can only reweight mass that the reference already assigns; it cannot create it. So the reference has to already be able to produce something in the neighborhood of the chosen responses. If I have no SFT checkpoint, I make one by fine-tuning on the chosen completions for an epoch — that is what the DPO paper recommends and it also makes the preference data on-policy-ish, which fixes a second problem at the same time."

**⚠ Trap:** using a *different* model as the reference than the one you initialize from. `π_θ` at step 0 must equal `π_ref`, otherwise `rewards/chosen` and `rewards/rejected` are nonzero at initialization and your KL accounting is meaningless from the first step. I have reviewed PRs where someone initialized from an instruct checkpoint and referenced the base — the run trains, the metrics look plausible, and the model is being pushed toward a target nobody intended.

### Explain why DPO's KL guarantee is much weaker in practice than the derivation suggests.

The derivation gives you a guarantee about the *global* optimum over the space of all policies with the reward known everywhere. You have neither.

Three gaps, in increasing order of how much damage they do:

**Gap 1 — finite data.** The loss is an empirical average over your `N` pairs. The KL is only pinned where you have data. For a prompt `x` not in your set, and for completions `y` not in your set, `β log(π_θ/π_ref)` is whatever the network's inductive bias makes it. There is nothing in the loss that says "behave sensibly here."

**Gap 2 — the objective rewards mass movement, not mass placement.** The loss increases the margin `log π(y_w) − log π(y_l)`. Probability mass is conserved: if you push `π(y_l)` down, that mass has to go *somewhere*, and there is no term telling it to go to `y_w`. In practice a large share of it flows to sequences that appear in neither the chosen nor rejected set. This is sometimes called the squeezing or mass-shifting effect, and it is why DPO models sometimes produce outputs that look nothing like either side of the training pairs.

**Gap 3 — the reward class is unconstrained off-support.** An explicit reward model at least has to fit a single network to all your data, so it generalizes *somehow* off-distribution. DPO's implicit reward is the policy itself, and the policy is a 8-billion-parameter function with enormous capacity to assign arbitrary values to unseen completions. This is the sharpest form of the argument for why PPO can still beat DPO: PPO *samples from the current policy*, so it evaluates the reward exactly where the policy is putting mass, which closes the off-support hole by construction.

**📄 Paper:** Xu et al. (2024), *Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study* — argues DPO's weakness is exactly its exposure to out-of-distribution responses relative to the training pairs, and reports settings where a well-tuned PPO beats DPO. Worth citing when an interviewer asks "so is DPO just strictly better?"

**🗣 Say this in the room:** "The equivalence holds at the optimum with full support. In practice DPO is offline and the KL is only enforced where you have data, so the implicit reward is unconstrained off-distribution and the model can move mass to completions that appear in neither side of a pair. That is the structural reason on-policy data and iterative rounds help so much, and it is the honest answer to why PPO-family methods have not disappeared."

### I'm watching a DPO run. What are you logging, and what does healthy look like on each curve?

Six curves, and I want a candidate to name them and their healthy shape without prompting. These are also the exact metric names TRL emits, which is a small credibility signal.

**`rewards/chosen`** — mean of `β(log π_c − log π_ref,c)`. Starts at exactly 0.0. Healthy: drifts slightly positive or stays near zero, sometimes goes mildly negative. Alarming: strongly negative and falling. See the pathology question later — a falling chosen reward is *normal*, a fast-falling one is not.

**`rewards/rejected`** — same for the rejected side. Starts at 0.0, should go clearly negative. This is where most of the margin comes from and that is expected.

**`rewards/margins`** — `chosen − rejected`. Healthy: monotonic rise, decelerating, ending somewhere in the 0.5–5 range for β = 0.1. If it blows past 10 you have drifted far from the reference and I would take an earlier checkpoint.

**`rewards/accuracies`** — fraction of the batch where `chosen_rw > rejected_rw`. Starts at ~0.5. Healthy: rises to 0.65–0.85 over a run. **If it exceeds ~0.95 you are done learning** — the gradient weighting σ(r̂_l − r̂_w) has collapsed to zero on nearly every example and further steps are just amplifying whatever the model already believed. High accuracy is not a success metric here; it is a stopping criterion.

**`logps/chosen` and `logps/rejected`** — the raw summed log-probs. Both will fall. Watch the *ratio of the falls*: if rejected falls 3× faster than chosen, the run is doing what it should. If they fall together at a similar rate, the model is unlearning the whole distribution and you should raise β or cut the LR.

**Sequence length of a fixed sampled dev set, per checkpoint.** Not a loss metric — you have to generate. I insist on this because length exploitation is invisible in every other curve. Sample 200 fixed dev prompts every 200 steps, log mean completion length, and alert on >25% growth from the SFT baseline.

**🏋 Drill:** ten minutes with a real DPO run's TensorBoard or W&B (any public one will do). Without reading the config, estimate β from the ratio of `rewards/margins` to the gap in `logps`. Pass criterion: you are within 2× of the true β. This forces you to internalize that the rewards are just β times a log-prob difference.

**⚠ Trap:** treating a smoothly falling training loss as evidence the run is working. DPO training loss falls monotonically for essentially any hyperparameters, including catastrophically bad ones, because the objective is a margin loss and margins are always increasable. The loss curve tells you the optimizer works. It tells you nothing about the model.
