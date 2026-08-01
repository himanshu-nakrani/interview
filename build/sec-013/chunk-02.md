### Motivate Mixture-of-Experts for me from what we just covered. And why do we sparsify the FFN and not attention?

The motivation is a scaling-law argument with a hardware constraint attached. Loss falls predictably with parameters and with training tokens, but *serving* cost is set by FLOPs per token, which for a dense model is rigidly `≈ 2 · N`. MoE breaks that rigidity: it decouples **total parameters** (which buy quality) from **activated parameters** (which cost FLOPs). You keep growing the model's capacity to store knowledge while holding the per-token arithmetic roughly fixed. That is the entire pitch, and it is a good one — a well-trained MoE genuinely gets quality closer to a dense model of its *total* size than of its *active* size.

Why the FFN and not attention? Three reasons, in order of force.

First, **the FFN is where the parameters are** — 80% of the block, as we derived. Sparsifying attention's projections would be sparsifying the small side of the ledger.

Second, **the FFN is position-independent.** It runs on `[N, d]` with no cross-token dependency, so you can permute tokens freely, group them by expert, run `E` independent GEMMs, and unpermute. Attention is intrinsically a cross-token operation; you cannot send token 5 to a different device than token 6 without splitting the sequence, which is a completely different (and much harder) parallelism problem.

Third, **the FFN is already contextually sparse**, so routing is learning to predict a structure that exists. Attention heads are not sparse in the same way — a head that fires on 3% of tokens is still doing something essential on those tokens and you cannot cluster heads into "usually irrelevant" groups nearly as cleanly. There *is* research on sparse/routed attention heads; none of it has displaced dense attention at the frontier, whereas MoE FFNs are now the dominant frontier architecture.

**🗣 Say this in the room:** "MoE decouples the parameter count that buys quality from the FLOP count that pays for it. You apply it to the FFN because that's 80% of the weights, because it's position-independent so tokens can be permuted and grouped into dense per-expert GEMMs, and because the FFN is already contextually sparse — the router is learning a structure that's already there. The bill comes due in HBM capacity and in all-to-all communication, not in FLOPs."

### Walk me through a forward pass through an MoE layer — top-2 of 8 — with actual shapes at every step.

Start with `x: [B, T, d_model]`, say `[4, 512, 4096]`. First thing every implementation does is flatten the token axis, because the layer does not care about sequence structure at all:

**1. Flatten.** `x → [N, d]` with `N = B·T = 2048`, `d = 4096`.

**2. Route.** A single small linear layer, the router (a.k.a. gate): `W_r: [d, E] = [4096, 8]`. `logits = x @ W_r → [N, E] = [2048, 8]`. Note the router is *tiny* — 32,768 parameters against 1.4 billion of experts in this layer. It is also, per unit of parameter, by far the most fragile thing in the model.

**3. Select top-k.** `topk(logits, k=2)` gives `topk_idx: [N, 2]` (int64, values in `[0, 8)`) and `topk_logit: [N, 2]`. Then `weights = softmax(topk_logit, dim=-1) → [N, 2]`, summing to 1 per token. Mixtral softmaxes *after* selection, over just the two chosen logits. Switch Transformer, with k=1, softmaxes over all `E` and takes the chosen probability as a scalar multiplier. DeepSeek-V3 uses a sigmoid affinity and then normalizes the selected scores. These are not interchangeable — they change gradient flow to the router, and swapping them when porting a checkpoint is a silent-corruption bug.

**4. Permute / dispatch.** Build a flat list of `N·k = 4096` (token, expert) assignments and sort by expert id. You now have contiguous runs of tokens per expert, plus an offset array — exactly a `cu_seqlens`-style ragged layout. `sorted_tokens: [4096, d]`, `expert_offsets: [9]`.

**5. Expert compute.** For each expert `e`, take its slice `[n_e, d]` and run that expert's SwiGLU FFN with its own `[d, d_ff]` weights. Modern kernels do this as a single **grouped GEMM** (one kernel launch, ragged batch dims) rather than a Python loop, because 8 small launches at decode are pure overhead. Output: `[4096, d]`.

**6. Unpermute and combine.** Scatter back to token order, multiply each of the `k` results by its routing weight, and sum: `out[i] = Σ_j weights[i,j] · expert_out[i,j]`. Result `[N, d]`, reshaped to `[B, T, d]` and added to the residual stream.

Note what did *not* happen: no token exchanged information with any other token. The router made 2048 independent decisions. The layer is still perfectly parallel across positions — which is exactly why it survives causal masking untouched.

**⚠ Trap:** thinking the permute/unpermute is bookkeeping you can ignore. At decode with a small batch, permutation, the ragged GEMM's poor tile utilization, and (under expert parallelism) two all-to-all collectives per layer can *dominate* the actual expert math. In a 61-layer model that is 122 collectives per token. The expert FLOPs are the cheap part; the data movement is the product.

### Implement top-k token-choice routing from scratch. Assume I want to see the dispatch and combine, not just the argmax.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8, k=2):
        super().__init__()
        self.E, self.k = n_experts, k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        # one SwiGLU expert per slot; real impls hold fused [E, d, 2*d_ff] tensors
        self.w13 = nn.Parameter(torch.empty(n_experts, d_model, 2 * d_ff).normal_(std=0.02))
        self.w2  = nn.Parameter(torch.empty(n_experts, d_ff, d_model).normal_(std=0.02))

    def forward(self, x):                                  # x: [B, T, d]
        B, T, d = x.shape
        xf = x.reshape(-1, d)                              # [N, d]
        logits = self.router(xf)                           # [N, E]
        topv, topi = logits.topk(self.k, dim=-1)           # [N, k], [N, k]
        gates = F.softmax(topv, dim=-1, dtype=torch.float32).to(x.dtype)

        flat_expert = topi.reshape(-1)                     # [N*k]
        order = flat_expert.argsort()                      # group by expert
        token_of = order // self.k                         # source token per slot
        counts = torch.bincount(flat_expert, minlength=self.E)
        offs = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

        out = torch.zeros_like(xf)
        for e in range(self.E):
            lo, hi = int(offs[e]), int(offs[e + 1])
            if lo == hi:                                   # expert got no tokens
                continue
            idx = token_of[lo:hi]                          # [n_e]
            h = xf[idx] @ self.w13[e]                      # [n_e, 2*d_ff]
            g, u = h.chunk(2, dim=-1)
            y = (F.silu(g) * u) @ self.w2[e]               # [n_e, d]
            w = gates.reshape(-1)[order[lo:hi]].unsqueeze(-1)
            out.index_add_(0, idx, y * w)                  # scatter-accumulate
        return out.reshape(B, T, d)
```

Three details that are load-bearing and that I look for when someone writes this. **`index_add_` not `index_put_`** — a token appears `k` times and its contributions must accumulate, not overwrite; `index_put_` here is a silent 50%-of-the-signal bug. **`softmax` in fp32** — the router's logits in bf16 have ~3 decimal digits of mantissa, and top-2 gates that should be 0.501/0.499 can quantize to a tie; every serious implementation upcasts the router. **The `lo == hi` guard** — an unused expert is normal early in training and a zero-row GEMM will throw or produce garbage in some backends.

**⚠ Trap:** the Python `for e in range(E)` loop. It is correct and it is the right thing to write on a whiteboard, but it is *not* what ships. With 256 experts and a decode batch of 32 you would launch 256 kernels to do a few thousand FLOPs each; launch overhead alone is ~5 µs × 256 = 1.3 ms per layer. Production uses grouped/batched GEMM (`torch._grouped_mm`-style APIs, CUTLASS grouped GEMM, MegaBlocks block-sparse kernels, or vLLM's fused MoE kernel). Say that out loud after you write the loop — writing the loop and *knowing* it's a teaching version is the senior answer.

### Give me the MoE lineage. What did GShard and Switch Transformer each actually contribute?

**📄 Paper:** Shazeer et al. (2017), "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" — the modern origin. Sparse gating inside an LSTM stack, noisy top-k routing, and the load-balancing loss as an explicit auxiliary objective. Everything after this is refinement.

**📄 Paper:** Lepikhin et al. (2020), "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" — put MoE inside a transformer at 600B parameters and, crucially, contributed the *systems* vocabulary the field still uses: **top-2 routing**, **expert capacity** with a **capacity factor**, **token dropping** when capacity overflows, and expert parallelism with all-to-all dispatch, plus the sharding annotations to express it. When someone says "capacity factor," they are speaking GShard.

**📄 Paper:** Fedus, Zoph & Shazeer (2021), "Switch Transformers" — simplified routing to **top-1**, which the field had assumed was too few to train stably, and showed it works with the right recipe. Contributions worth naming individually: top-1 halves the dispatch communication and the expert FLOPs; selective precision (route in fp32 even when the model is bf16) fixed a real instability; a smaller initialization scale for expert weights; expert dropout; and a demonstration that MoE distills into dense models with a good fraction of the gain retained.

**📄 Paper:** Zoph et al. (2022), "ST-MoE: Designing Stable and Transferable Sparse Expert Models" — the "we actually made it stable" paper. Contributed the **router z-loss**, and much of the practical guidance on fine-tuning sparse models without collapse.

**📄 Paper:** Du et al. (2022), "GLaM" — 1.2T total parameters, top-2, and the first widely-cited demonstration that an MoE could beat GPT-3-class quality at a fraction of the training energy.

Then the open-weight era: **Mixtral 8x7B** (Jiang et al., 2024) made a strong MoE downloadable and forced everyone to actually serve one; **DeepSeekMoE** (Dai et al., 2024) introduced fine-grained experts plus shared-expert isolation; **DeepSeek-V3** (2024) scaled that to 671B total / 37B active with auxiliary-loss-free balancing.

**🗣 Say this in the room:** "Shazeer 2017 invented sparse gating and the balancing loss. GShard 2020 gave us the serving vocabulary — capacity factor, token dropping, all-to-all expert parallelism. Switch 2021 proved top-1 works and fixed the precision issues. ST-MoE 2022 added the router z-loss and made fine-tuning survivable. DeepSeekMoE 2024 changed the granularity: many small experts plus an always-on shared one."

### What is capacity factor, what happens to a dropped token, and what does that drop actually cost?

Capacity factor exists because a GPU wants static shapes and routing produces dynamic ones. If you are going to allocate a fixed buffer per expert before you know the routing decisions, you must decide *in advance* how many tokens each expert may accept. That number is the capacity:

```
capacity_per_expert = ceil( capacity_factor · (N · k) / E )
```

With `N = 4096` tokens in the local batch, `k = 2`, `E = 8`, and `CF = 1.25`: `1.25 · 8192 / 8 = 1280` slots per expert against a perfectly-balanced expectation of 1024. That 25% headroom absorbs mild imbalance. Tokens arriving at a full expert are **dropped** — not errored, not retried, dropped.

What "dropped" means mechanically: the expert produces no output for that token, so its contribution to the combine step is zero. Under top-1, the token's FFN output for that layer is the zero vector, and because of the residual connection the layer becomes a **no-op** — the hidden state passes through untransformed. Under top-2 with one expert dropping, the token still gets the surviving expert's output, usually with the gate weights renormalized, so the damage is partial. This is why top-2 is more robust to imbalance than top-1 at equal capacity factor: the failure degrades instead of erasing.

**💰 Math:** the cost is a *quality* cost, and it is nonlinear in a way that matters. At `CF = 1.0` with realistic early-training imbalance, drop rates of 5–15% of token-slots are common; at `CF = 1.25` they fall to low single digits; at `CF = 2.0` they approach zero — but you have doubled the dispatch buffer and the all-to-all payload, so you have doubled the communication volume of every MoE layer to protect a few percent of tokens. That is the trade, stated honestly: capacity factor is a knob that buys quality with bandwidth. My default for training is 1.25 and I raise it only when the drop-rate metric says to.

**⚠ Trap:** believing drops are uniformly distributed and therefore harmless noise. They are not. Drops concentrate exactly where routing is most confident and most concentrated — a rare token type that all fires the same specialist expert will overflow that expert and get dropped *as a class*. So your drop rate can read 2% overall while a specific domain (say, non-Latin script, or a specific code language) is being dropped at 30%. **The instrumentation rule I enforce: never report drop rate as a scalar. Report it per-expert and, if you can, joined against a token-class label.**

**⚠ Trap (serving edition):** capacity factor and dropping are largely a *training-time* concept. At inference most serving stacks run **dropless** — MegaBlocks-style block-sparse kernels (Gale et al., 2023) or grouped GEMMs with ragged shapes — because at inference you cannot tell a user "we dropped your token." If an interviewer asks about capacity factor in a serving context, the correct answer includes "modern inference is dropless, and the dynamic-shape cost moved into kernel complexity and CUDA-graph friendliness instead."

### Compare expert-choice routing with token-choice routing. What are their fairness properties, and why hasn't expert-choice taken over?

Token-choice — the default — is *tokens pick experts*: each token takes its top-k experts. Expert-choice inverts it: **📄 Paper:** Zhou et al. (2022), "Mixture-of-Experts with Expert Choice Routing" — each expert picks its top-`c` tokens from the batch, where `c` is its capacity.

The fairness properties are exactly dual, and this is the cleanest way to hold it:

**Token-choice guarantees every token gets exactly `k` experts** (fairness to tokens) **but guarantees nothing about expert load** (unfairness to experts). Hence the entire apparatus of auxiliary losses, capacity factors and dropping — all of it exists to patch the load side.

**Expert-choice guarantees every expert gets exactly `c` tokens** (perfect load balance by construction — no auxiliary loss, no capacity factor, no dropping) **but guarantees nothing about tokens** (unfairness to tokens). A token that no expert ranks highly gets *zero* experts and passes through on the residual alone; a popular token might be picked by six experts. Zhou et al. argued this is a feature — variable compute per token, more where it is needed — and reported meaningfully faster convergence.

So why is nearly every deployed model token-choice? **Because expert-choice is not causal.** To let expert `e` pick its top-`c` tokens, you must see the whole set of candidate tokens at once. During training on a packed batch that is fine. During autoregressive decoding you have one new token and no future tokens to compete with it — the routing decision for position `t` would depend on positions `t+1...`, which is information leakage in training and simply unavailable at inference. You get a train/serve mismatch that is structural, not tunable. Workarounds exist (restrict the selection pool to the current sequence prefix, or use expert-choice only in the encoder of an encoder-decoder), and they narrow the applicability to exactly the settings where autoregressive generation is not the product.

**🗣 Say this in the room:** "They're duals. Token-choice is fair to tokens and unfair to experts, so you bolt on balancing losses and capacity factors. Expert-choice is fair to experts by construction and unfair to tokens, so it needs neither — but it's non-causal, because an expert's choice depends on the whole batch. That kills it for autoregressive decoding, which is why every deployed LLM MoE is token-choice."

### Write me the auxiliary load-balancing loss and explain what it's trading off.

The Switch/GShard formulation, per MoE layer, over a batch of `N` tokens and `E` experts:

```
f_i = (1/N) · Σ_t  1[ expert i is in top-k(token t) ]     # dispatch fraction, non-differentiable
P_i = (1/N) · Σ_t  softmax(router_logits(t))_i            # mean routing probability, differentiable
L_aux = α · E · Σ_{i=1}^{E} f_i · P_i
```

The construction is cleverer than it looks. `f_i` is a hard count and carries no gradient, so the loss reaches the router *only through* `P_i`. What it says is: "if expert `i` is currently overloaded (`f_i` large), push down the probability mass you assign to it." It is a differentiable surrogate for `Σ f_i²`, which is what you actually want minimized. At perfect balance `f_i = P_i = 1/E`, so `Σ f_i·P_i = E · (1/E²) = 1/E` and `L_aux = α`. That constant floor is a useful sanity check — if your logged aux loss is sitting at `α`, routing is uniform; if it is meaningfully above `α`, it is collapsing. Switch used `α = 0.01`.

The trade-off it makes, and this is the part interviewers push on: **the auxiliary loss is a gradient that does not care about your task.** It actively fights specialization. If the true optimal routing for your data is skewed — and it usually is, because token frequencies are Zipfian — the balancing loss drags routing toward a uniformity the data does not support, and you pay for it in loss. Set `α` too high and you get well-balanced, undifferentiated experts and worse perplexity. Set it too low and you get expert collapse, dropped tokens, and stragglers. There is no principled way to pick `α`; you tune it, and the tuning is annoying because the symptom of a bad value appears thousands of steps later.

That unsatisfying trade is exactly what DeepSeek-V3's auxiliary-loss-free scheme was designed to eliminate.

**⚠ Trap:** computing the aux loss over the *local* micro-batch on each data-parallel rank. If each rank balances its own few hundred tokens independently, you are imposing a much stronger uniformity constraint than the global batch requires, and you suppress genuine specialization — a rank that happens to hold a batch of Python code *should* route to code experts. Several implementations compute it over the global batch (an all-reduce of `f_i` and `P_i` counts) for exactly this reason, and Qwen3's reported recipe emphasizes global-batch balancing. This is a real, subtle, expensive bug.

### What is the router z-loss, and what would you monitor to know routing is healthy during training?

**📄 Paper:** Zoph et al. (2022), ST-MoE. The z-loss penalizes the magnitude of the router's log-partition function:

```
L_z = (1/N) · Σ_t ( logsumexp_j( router_logits(t)_j ) )²
```

typically with a coefficient around `1e-3`. The motivation is bluntly numerical rather than statistical: the router is a tiny linear layer whose output feeds an exponential. Nothing constrains its logits, they drift upward over training, and large logits in low precision produce roundoff in the softmax and eventually overflow. The z-loss is a soft constraint pulling `logsumexp` toward zero, which keeps logits small and centered. A pleasant side effect is regularization — smaller logits mean a softer, higher-entropy routing distribution, which discourages premature collapse.

What I actually put on a dashboard for a training run with MoE layers, and what each signal means when it moves:

**Per-expert dispatch fraction `f_i`, as a distribution, per layer.** The single most informative plot. Healthy: a spread within roughly 2× of uniform after warmup. Sick: a few experts at 5–10× uniform and a tail at near-zero. Track `max(f)/min(f)` as a scalar and `Σ f_i²·E` (the coefficient of variation squared, 1.0 at uniform) as the alert metric.

**Routing entropy, and be careful to track two of them.** *Per-token* entropy `H = -Σ_i p_i log p_i` should *fall* over training — that is the router becoming decisive, which is what you want; it starts near `log E` (2.08 nats for E=8, 5.55 for E=256) and settles well below. *Batch-marginal* entropy, computed on the averaged distribution `P̄`, should stay *high*, near `log E` — that is load balance. Confusing these two is common and gives you exactly the wrong alarm. Collapse looks like both falling together.

**Token drop rate, per expert.** Discussed above; never a scalar.

**Router logit magnitude / `logsumexp`.** If z-loss is doing its job this stays bounded. A rising trend is your early warning of a numerics problem before you see NaNs.

**🔍 Failure taxonomy — routing pathologies, as a decision procedure:**
1. *Marginal entropy falling toward zero and `max(f)/min(f)` exploding* → **expert collapse**. Almost always: aux-loss coefficient too small, or the router initialized too large, or you removed router noise. Fix: raise `α`, re-add jitter, check router init scale.
2. *Per-token entropy stuck near `log E` after thousands of steps* → **router not learning**. The MoE is behaving like an average of experts, and you are paying MoE costs for dense behavior. Fix: check that the router's gradient path is intact (a common bug is detaching gates), check learning rate on the router, check that `α` is not so high it dominates.
3. *Both entropies fine, but drop rate high on 2–3 experts* → **genuine data skew**, not a bug. Fix: raise capacity factor, or go dropless, or add fine-grained experts.
4. *Loss spikes correlated with router `logsumexp` spikes* → **numerics**. Fix: add/raise z-loss, force fp32 routing.

### Mixtral is called "8x7B." Why isn't it 56 billion parameters? Give me the real number and where it comes from.

Because only the FFN is replicated eight times. Everything else — the attention projections, the embeddings, the norms, the output head — exists exactly once and is shared by all experts. "8x7B" is a marketing name, not an arithmetic statement, and catching that is a small but reliable competence signal.

Mixtral 8x7B: 32 layers, `d_model = 4096`, `d_ff = 14336`, 32 query heads / 8 KV heads with `d_head = 128`, vocab 32000, 8 experts, top-2.

```
attention per layer:  Q 4096·4096  = 16.78 M
                      O 4096·4096  = 16.78 M
                      K 4096·1024  =  4.19 M
                      V 4096·1024  =  4.19 M      → 41.94 M
experts per layer:    8 · 3 · 4096 · 14336        = 1409.3 M
router per layer:     4096 · 8                    = 0.03 M
                                          layer total ≈ 1451.3 M
× 32 layers                                        = 46.44 B
+ embeddings 32000·4096 × 2 (in and out)           =  0.26 B
                                             TOTAL ≈ 46.7 B
```

And the activated count, which is what an inference FLOP estimate uses: attention 41.94 M + **two** experts 352.3 M = 394.2 M per layer, × 32 = 12.6 B, plus the embedding lookup and output head ≈ **12.9 B active**.

So: **46.7 B total, 12.9 B active.** Ratio 3.6×.

**💰 Math — and this is the whole thesis of the section in one comparison.** Mixtral's active count (12.9 B) is close to Llama-3-8B's total (8.0 B), so a naive reading says "similar serving cost." Now put both on H100s in bf16. Llama-3-8B needs 16 GB of weights and fits on one 80 GB card with 64 GB left for KV; at 128 KiB/token of GQA cache that is ~500k tokens of KV, or 15 concurrent users at 32k context on **one GPU**. Mixtral needs 93.4 GB of weights and **does not fit on one H100 at all**. On two H100s (160 GB) you have 66 GB left for KV → ~503k tokens → about the same 15 users at 32k context, on **two GPUs**. Same FLOPs per token, same effective concurrency, exactly double the hardware bill. At an on-demand rate of roughly $2.50/GPU-hour (**📅 Volatile:** GPU spot and on-demand pricing moves constantly — verify before your loop), that is $3,600/month versus $1,800/month for the same served capacity.

**🗣 Say this in the room:** "8×7B is 46.7 B parameters, not 56 B, because only the FFN is replicated — attention and embeddings are shared. And 12.9 B active does not mean it serves like a 13 B model: you still have to hold all 46.7 B in HBM. Active parameters predict FLOPs; total parameters predict your GPU count. The GPU count is what shows up on the invoice."

### DeepSeek-V3 does three unusual things: fine-grained experts, a shared always-on expert, and load balancing with no auxiliary loss. Take them one at a time.

**Fine-grained experts.** Instead of 8 fat experts with top-2, use many thin ones with a larger `k` — DeepSeek-V3 runs 256 routed experts per MoE layer with top-8, and each expert's intermediate width is small (2048 against a `d_model` of 7168). **📄 Paper:** Dai et al. (2024), "DeepSeekMoE." The argument is combinatorial: with 8-choose-2 you have 28 possible expert combinations per token; with 256-choose-8 you have on the order of 10¹⁴. Specialization is no longer "which of eight topics is this," it is a *combination code*. Empirically it improves quality at matched active parameters. The cost is systems cost — more, smaller GEMMs (worse tensor-core efficiency), a much wider all-to-all fan-out, and a routing decision with far more ways to go wrong.

**A shared always-on expert.** One expert (or a few) that *every* token passes through unconditionally, in addition to its top-k routed experts. The reasoning: in a pure routed design, every expert must independently relearn the generic, high-frequency transformations that all tokens need — grammar, common-word handling, basic syntax. That is redundant capacity, duplicated 256 times. Factor it out into a shared expert and the routed experts are freed to be genuinely specialized. There is also a stability benefit that I think is underrated: the shared expert guarantees every token receives *some* FFN transformation, so a routing failure degrades quality rather than turning the layer into a no-op. At serving time the shared expert is a fully dense component — it is read on every token regardless of batch composition, so it never benefits from sparsity, but it also never suffers from imbalance.

**Auxiliary-loss-free load balancing.** **📄 Paper:** Wang et al. (2024), "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts." Instead of adding a balancing term to the loss — which, as established, fights the task objective — maintain a per-expert bias `b_i` that is added to the affinity scores **only for the purposes of top-k selection**, and *not* to the gating weights used to combine expert outputs. After each step, nudge `b_i` down for over-subscribed experts and up for under-subscribed ones by a small fixed rate `γ`. It is a control loop, not a loss term.

Why that separation is the elegant part: the bias steers *which* experts get chosen (fixing load) while leaving the *magnitude* of each expert's contribution determined purely by the learned affinity (preserving the task gradient). No interference term is ever added to the gradient of the language-modeling loss. DeepSeek-V3 additionally retains a very small sequence-level balancing loss as a guard against extreme within-sequence skew, which is worth mentioning because it shows the technique is a strong default rather than a total replacement.

**⚠ Trap:** describing the bias as being added to the gate weights. If you add it to the combination weights you have injected an arbitrary, non-learned scalar into the model's output and you will damage quality. Selection-only is the entire trick. Interviewers who know this paper listen specifically for that distinction.

### Where does Qwen3's MoE fit, and what's the state of the open-weight MoE landscape?

**📅 Volatile:** model names, parameter counts and configs in this answer move every few months — verify against the current model cards and `config.json` before your loop. The *mechanisms* are durable; the numbers are not.

Qwen3 shipped MoE variants using the naming convention `Qwen3-<total>-A<active>` — e.g. Qwen3-235B-A22B and Qwen3-30B-A3B — which I think is the right convention and wish everyone used it, because it forces both numbers into the same breath. Architecturally the family sits closer to DeepSeek's fine-grained direction than to Mixtral's: on the order of 128 routed experts with top-8 selection, and — this is the notable departure — **no shared expert**, with load balance handled via a global-batch balancing loss rather than per-micro-batch. The global-batch detail is the interesting one: computing the balancing statistic across the whole data-parallel global batch instead of per-rank lets a rank holding domain-homogeneous data route in a domain-specific way without being penalized, which preserves specialization that per-rank balancing destroys.

The shape of the open-weight landscape as a set of design points, which is what you should actually be able to reason about:

- **Coarse-grained, low-k** (Mixtral 8x7B / 8x22B style, 8 experts top-2): simple, ~3–4× total-to-active ratio, easy to serve on 2–8 GPUs, expert parallelism is optional. The pragmatic choice if you must self-host.
- **Fine-grained, high-k, shared expert** (DeepSeek-V3 style, 256+1 experts top-8): best quality-per-active-parameter, ~18× total-to-active ratio, and a genuinely hard serving problem requiring wide expert parallelism.
- **Fine-grained, high-k, no shared expert** (Qwen3-MoE style): between the two on serving complexity, with the balancing work moved into the training recipe.
- **Small MoE for edge/local** (the ~30B-total / ~3B-active class): the ratio is the point — you want the memory footprint of a mid-size model with the *decode speed* of a small one, on a machine where you own all the memory and there is exactly one user. This is the one regime where an MoE is unambiguously right at batch 1, because unified-memory laptops have plenty of RAM and terrible bandwidth.

**⚠ Trap:** treating "shared expert or not" as settled. It is not. DeepSeek and Qwen made opposite calls at similar scales and both produced strong models. My decision rule if asked to choose: include a shared expert when your `k` is small relative to `E` (so a routing mistake is catastrophic for that token) and when you want inference-time robustness; skip it if you are willing to invest in the training recipe to get balance right, since the shared expert is dense compute you pay on every single token.
