### Start at the beginning — what is LoRA actually doing to the model, and why should a low-rank update be enough?

Start from the shape of the problem. Fine-tuning a 70B model means producing a 70-billion-parameter delta `ΔW`. But the *thing you are teaching it* — respond in our house JSON schema, adopt this legal register, prefer this tool-call ordering — is not 70 billion parameters of new information. It is a narrow behavioural adjustment sitting on top of a model that already knows the language, the domain, and the format. The empirical bet LoRA makes is that the useful `ΔW` for a downstream task has low **intrinsic rank**: it lives in a small subspace, and you can parameterize that subspace with two thin matrices instead of one fat one.

Concretely, for a frozen weight matrix `W ∈ R^{d_out × d_in}`, LoRA replaces the update with a factored one:

```
h = W x + (α/r) · B (A x)
A ∈ R^{r × d_in}   initialized ~ N(0, σ²) (Kaiming, in practice)
B ∈ R^{d_out × r}  initialized to ZERO
r ≪ min(d_in, d_out), typically 8–64
```

`W` never receives a gradient. Only `A` and `B` do. The number of trainable parameters drops from `d_out · d_in` to `r · (d_in + d_out)`. For a 4096×4096 projection at `r=16` that is 16.7M → 131k, a 128× reduction, and it compounds across every targeted layer.

The backend analogue I find useful: `W` is a compiled binary you are not allowed to recompile, and LoRA is an LD_PRELOAD shim. You do not modify the original; you intercept and add a small correction at specific call sites, and at deploy time you can either keep the shim live or statically link it in (merge). The shim is small because the behaviour change is small — not because the model is small.

The reason this is *the* PEFT method rather than one of many is the merge property. `B A` has the same shape as `W`, so `W' = W + (α/r)BA` is a legal weight matrix. Adapters that add new *layers* (Houlsby bottlenecks, prefix tuning) change the computation graph and therefore cost inference latency forever. LoRA does not have to.

**📄 Paper:** Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* — introduced the frozen-base + low-rank-factored-delta formulation with zero-init on `B`, replacing bottleneck adapters as the default PEFT method precisely because the update is mergeable and therefore inference-free.

**🗣 Say this in the room:** "LoRA freezes `W` and learns a rank-`r` factored correction `(α/r)·BA`. The trainable parameter count drops by two to three orders of magnitude, but the real win in practice is memory: I no longer store Adam moments for 70 billion parameters. And because `BA` is the same shape as `W`, I can fold it back in and serve with zero added latency."

### Why is B initialized to zero, and what would go wrong if you initialized both matrices randomly?

Because the adapter must be a **no-op at step 0**. With `B = 0`, the product `BA = 0`, so `h = Wx` exactly — the model at initialization is bit-for-bit the base model. You start from a known-good checkpoint and move away from it under gradient descent, rather than starting from a randomly perturbed model and having to climb back.

If you initialized both `A` and `B` randomly, `BA` would be a random rank-`r` matrix injected into every targeted projection. The scale is not small: with Kaiming init on both, the product has entries on the order of the product of two random matrices' scales, summed over `r` inner terms. Injecting that into 7 attention projections × 80 layers means the model's forward pass at step 0 produces garbage. Your first hundred optimizer steps are then spent undoing your own initialization — which is not just wasted compute, it is *destructive*, because the large early gradients pass through the frozen base too (as activations), and you land somewhere worse than where you started.

The asymmetry matters and is a good follow-up question. Why zero `B` and not zero `A`? Either choice makes the product zero, but if you zero `A`, the gradient with respect to `B` is `∂L/∂h · (Ax)ᵀ = 0`, so `B` never moves — and `A`'s gradient is `Bᵀ · ∂L/∂h · xᵀ`, which is nonzero only because `B` is random. It works, but it is the worse-conditioned choice: the *output-side* matrix is the one whose random values then get amplified through the rest of the network. Zeroing `B` means the first gradients flow into `A` through a zero `B`... which is also zero. Let me be precise, because this catches people:

```
∂L/∂A = (α/r) · Bᵀ · (∂L/∂h) · xᵀ     → zero when B = 0
∂L/∂B = (α/r) · (∂L/∂h) · (Ax)ᵀ       → NONZERO, since A is random
```

So at step 1, only `B` moves. Once `B ≠ 0`, `A` starts receiving gradient too. The dynamics bootstrap correctly. With the reverse convention (`A = 0`, `B` random) you get the mirror image, and it also works — some implementations do exactly that. What does *not* work is both random.

**⚠ Trap:** people say "LoRA starts as the identity." It does not start as the identity function; it starts as *the base model*, which is a different and stronger statement. The distinction matters when you are debugging: if your step-0 eval does not exactly reproduce base-model eval numbers, you have a bug — a nonzero `B`, an adapter applied to the wrong module, a dropout layer active inside the adapter path, or a dtype mismatch. That step-0 equality check is the first thing I run on any new LoRA integration, and it is the cheapest bug-catcher in the whole pipeline.

### Walk me through implementing LoRA from scratch in PyTorch — no PEFT library.

Here is the whole thing. It is short enough to write on a whiteboard, and interviewers at Meta, Databricks and Cursor do ask for it.

```python
import torch, torch.nn as nn, math

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False          # freeze W and bias
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if not self.merged:
            out = out + self.drop(x) @ self.A.T @ self.B.T * self.scaling
        return out

    @torch.no_grad()
    def merge_(self):
        # cast to base dtype ONLY at the end; accumulate in fp32
        delta = (self.B.float() @ self.A.float()) * self.scaling
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        self.merged = True

def apply_lora(model, targets=("q_proj","k_proj","v_proj","o_proj"), **kw):
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(mod, child_name, LoRALinear(child, **kw))
    return model
```

Three details that separate a correct implementation from one that trains but underperforms.

**Dropout goes on the input `x`, not on the adapter output.** LoRA dropout is regularizing the *low-rank path only*; applying it to `x` before `A` means you are dropping input features into the adapter. Some implementations put it after `A`; the PEFT convention is before. Either regularizes, but be consistent, and note the base path is never dropped.

**Order the matmuls `(x @ Aᵀ) @ Bᵀ`, never materialize `BA`.** Written as `x @ (B@A).T` you allocate a `d_out × d_in` intermediate — the exact 16.7M-element tensor you were trying to avoid — on every forward. Cost of getting this wrong at batch 8, seq 2048, hidden 4096: the correct order costs `2·8·2048·4096·16 + 2·8·2048·16·4096 ≈ 4.3 GFLOP` per projection; the wrong order costs an extra `2·4096·4096·16 = 0.54 GFLOP` of matrix construction *plus* a 33 MB bf16 allocation per layer per step, which is what actually kills you via allocator churn.

**Merge in fp32.** `B @ A` in bf16 accumulates rounding across `r` terms and then you add it to a bf16 weight whose magnitude is much larger — classic catastrophic cancellation of the small update. Compute in fp32, cast once.

**🏋 Drill:** 15 minutes, no references. Write `LoRALinear` and `apply_lora`, wire it onto a small HF causal LM, and assert three things: (1) `sum(p.numel() for p in model.parameters() if p.requires_grad)` matches your hand-computed `r·(d_in+d_out)·n_targets·n_layers`, (2) logits at step 0 are `allclose` to the base model's, (3) after `merge_()`, unmerged and merged logits are `allclose` to 1e-2 in bf16. **Pass criterion:** all three green and the parameter count is exact, not approximate.

### What does alpha actually control, and why do people tell you to set alpha = 2r?

Alpha is a scaling constant, and the honest one-line answer is that `α/r` exists so that **changing `r` does not implicitly change your learning rate.**

Think about what happens without it. The adapter output is a sum of `r` rank-one terms. Double `r` and, with the same per-entry initialization scale, the magnitude of `BA` roughly grows — so the effective step size on the *function* grows even though the optimizer's `lr` is unchanged. Every time you re-tuned rank you would also have to re-tune LR. Dividing by `r` normalizes this: `α` becomes the knob for "how strongly does the adapter speak," and `r` becomes the knob for "how much capacity does it have," approximately decoupled.

So what is `α` numerically? It multiplies the adapter's contribution. `α = r` gives scaling 1.0. `α = 2r` gives 2.0 — a common default because in practice a slightly amplified adapter path converges faster on instruction-style data. `α = 16, r = 8` (scaling 2.0) and `α = 32, r = 16` (scaling 2.0) are the same effective amplification; that is why so many public configs use those pairs. Some practitioners set `α = r` and simply raise LR instead. Both are defensible; what is not defensible is treating `α` as an independent magic number.

**⚠ Trap — the one I see in review most often:** `α` and `lr` are *not* independent, and tuning both simultaneously wastes a sweep. Increasing `α` by 2× and increasing `lr` by 2× produce very similar early dynamics, because the gradient with respect to `B` carries a factor of `α/r`, so the effective update to the composite `BA` scales roughly as `lr · α²/r²` in the early phase (both matrices' updates carry a scaling factor). The rule I enforce: **fix `α = 2r`, sweep only `lr`.** You get 90% of the achievable quality with a quarter of the compute. If you must sweep something else, sweep target modules, which matters far more than `α`.

**⚠ Trap two:** `α` is *not* a regularizer. People reason "smaller `α` = smaller update = less overfitting." It is not — with enough steps the optimizer simply grows `B` to compensate, because nothing constrains `‖B‖`. If you want regularization, use LoRA dropout, weight decay on the adapter, or fewer epochs.

### How do I choose the rank? Give me the actual decision procedure, not "it depends."

The single most useful empirical result here, and the one that reframes the question: **for a fixed trainable-parameter budget, spending it on more target modules beats spending it on higher rank.** A rank-8 adapter on all seven linear projections (`q,k,v,o,gate,up,down`) reliably outperforms a rank-64 adapter on `q,v` only, at comparable or lower parameter count. This is because the attention projections and the MLP do different jobs — restricting adaptation to attention constrains *what kind* of change you can make, and no amount of rank recovers that.

So my procedure is:

1. **Target all linear layers first.** `q,k,v,o,gate,up,down`. This is the default in modern recipes and it is the right default.
2. **Start at `r = 16`, `α = 32`.** Sweep LR over `{5e-5, 1e-4, 2e-4, 5e-4}` — LoRA tolerates and usually wants LRs 10–100× higher than full fine-tuning, because you are moving a tiny, well-conditioned parameter set.
3. **Only raise rank if training loss plateaus above where you need it.** Under-capacity looks like: train loss stops falling while still far from your target, and raising LR makes it unstable rather than better. That is the signal for `r = 64` or `128`.
4. **If train loss falls fine but eval degrades, you have a data problem, not a rank problem.** Raising rank here makes it worse.

The tasks that genuinely want high rank are the ones with real distribution shift — teaching a new language, a new modality projection, a genuinely new output format the base has never produced. Style, tone, format-following and tool-call shaping essentially never need `r > 32`.

**📐 Numbers you must know — LoRA parameter count, derived.** Llama-3-8B: `d_model = 4096`, 32 layers, GQA with 8 KV heads × 128 head-dim, so `k_proj`/`v_proj` are `4096 → 1024`; MLP intermediate is 14336.

At `r = 16`, per layer:
- `q_proj`: `16 · (4096 + 4096) = 131,072`
- `k_proj`: `16 · (4096 + 1024) = 81,920`
- `v_proj`: `81,920`
- `o_proj`: `131,072`
- attention subtotal: `425,984`
- `gate_proj`: `16 · (4096 + 14336) = 294,912`; `up_proj`: `294,912`; `down_proj`: `16 · (14336 + 4096) = 294,912`
- MLP subtotal: `884,736`

Attention-only: `425,984 × 32 = 13.6M` params → **0.17%** of 8B. All-linear: `1,310,720 × 32 = 41.9M` → **0.52%** of 8B. In bf16 the all-linear adapter is `41.9M × 2 = 84 MB` on disk. Memorize that shape: *a rank-16 all-linear adapter on an 8B model is under 100 MB.* It is the number that makes multi-tenant serving obviously correct.

### Which modules do you target, and does it ever make sense to include the embeddings or the LM head?

The default I ship is **all seven linear projections**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. If I have to cut, I cut in this order: drop `k_proj` first (with GQA it is the smallest and, empirically, the least load-bearing), then `gate_proj`. I would not go down to the classic `q,v`-only configuration from the original paper unless memory is genuinely binding — that config was chosen in a 2021 GPT-3 context and has been superseded by the "target everything" result.

Some specifics worth knowing:

**Embeddings and `lm_head`.** Normally excluded, and correctly so. They are enormous (`128,256 × 4096 = 525M` params for Llama-3's embedding — 6.5% of the whole model), and a low-rank update to a lookup table is a strange object: it applies the *same* rank-`r` correction structure to every token. There is one legitimate case: **you added new tokens.** New special tokens for tool syntax, a new domain vocabulary, a new chat template sentinel. Those rows are randomly initialized and must be trained or the model emits noise for them. In that case the right move is usually not LoRA on the embedding — it is to make the new embedding rows fully trainable (`modules_to_save=["embed_tokens","lm_head"]` in PEFT's vocabulary), accepting the memory cost, because you need real capacity there, not a low-rank correction.

**MoE models.** Targeting every expert's `gate/up/down` multiplies your adapter count by the number of experts — a 64-expert model turns a 42MB adapter into something in the gigabytes, and worse, each expert sees only the tokens routed to it, so per-expert adapters are data-starved. My default on MoE is: adapt the attention projections and the *router* is left alone (perturbing routing during a small fine-tune destabilizes load balancing badly), and only adapt shared/dense FFN paths if the architecture has them. This is an area where practice is still moving; say so rather than asserting a rule.

**Normalization layers.** LayerNorm/RMSNorm gains are tiny (`d_model` params each) and sometimes worth making fully trainable — that is essentially BitFit-style tuning bolted onto LoRA. It costs nothing memory-wise. I do not consider it a default, but it is a cheap thing to try when a LoRA run is stubbornly under-fitting.

**🗣 Say this in the room:** "The 2021 paper's `q,v`-only configuration is not the modern default. The result that changed practice is that at fixed parameter budget, breadth beats depth — rank 8 on all seven projections beats rank 64 on two. So I target all linear layers, start at rank 16 alpha 32, and only touch rank if the *training* loss plateaus."

### What is rsLoRA fixing? Why does raising rank sometimes stop helping — or hurt?

This is a nice one because the answer is a gradient-scaling argument, not a capacity argument.

The observed phenomenon: you raise `r` from 16 to 64 to 256 expecting monotonically more capacity, and instead quality flattens and then degrades, or training gets unstable and needs a lower LR. The naive reading is "the task doesn't need that much rank." The better reading is that the standard `α/r` scaling **over-corrects** at high rank, shrinking the adapter's effective gradient signal so that larger ranks train more slowly and less stably per step.

The argument is a variance one. The adapter output is a sum of `r` terms. If those terms were perfectly correlated, their sum would scale like `r`, and dividing by `r` would be right. But they are approximately independent at initialization, so the sum's *magnitude* scales like `√r`, not `r`. Dividing by `r` therefore suppresses the adapter by an extra factor of `√r` as rank grows. rsLoRA replaces the scaling with:

```
γ = α / √r        (rank-stabilized)   instead of   α / r
```

With this, the adapter's effective contribution stays roughly rank-invariant, and raising rank behaves like adding capacity rather than like silently lowering your learning rate.

**📄 Paper:** Kalajdzievski (2023), *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* — showed that the `α/r` divisor causes gradient collapse at high rank and that `α/√r` restores the expected rank-vs-performance scaling.

**When it matters, practically:** below `r ≈ 32`, the difference between `α/r` and `α/√r` is a constant factor you would have absorbed into your LR anyway, so rsLoRA is a no-op you cannot detect. Above `r ≈ 64` and especially at `r ≥ 128`, it is the difference between "higher rank helped" and "higher rank did nothing." It is a one-flag change in PEFT (`use_rslora=True`) and it costs zero compute, so my rule is: **turn it on whenever `r ≥ 64`, and leave the LR sweep alone otherwise.**

**⚠ Trap:** enabling rsLoRA changes the effective adapter magnitude for a given `(α, r)`, so a config you tuned under `α/r` will behave differently — usually a larger effective update, meaning your previously-good LR may now be too hot. If you flip the flag mid-project and quality drops, that is the reason; re-sweep LR, do not conclude rsLoRA is bad.

### Explain DoRA. What does decomposing into magnitude and direction actually buy you?

Start with the observation that motivates it. If you look at how weights change under full fine-tuning versus under LoRA, they change differently: full fine-tuning tends to make *large magnitude* changes with relatively *small directional* changes, or vice versa, in a pattern LoRA struggles to reproduce because a single low-rank additive term couples the two. LoRA can only move `W` along a rank-`r` subspace, and changing a column's length and changing its direction are entangled in that one update.

DoRA decomposes each pretrained weight matrix into a per-column magnitude vector and a directional matrix:

```
W = m · (V / ‖V‖_c)          # ‖·‖_c = column-wise L2 norm, m ∈ R^{1 × d_out} (one scalar per output column)
```

Then it trains `m` **fully** (it is only `d_out` parameters — trivially cheap) and applies LoRA to the *direction* only:

```
W' = m · (W + (α/r)·BA) / ‖W + (α/r)·BA‖_c
```

So the magnitude gets a free, unconstrained, full-rank-in-its-own-space update, and the low-rank budget is spent entirely on direction. That is the whole idea: stop making the low-rank subspace pay for scale changes it is bad at expressing.

**📄 Paper:** Liu et al. (2024), *DoRA: Weight-Decomposed Low-Rank Adaptation* — reported that DoRA closes part of the LoRA-vs-full-fine-tuning gap, most visibly at *low* rank (r = 4–8), where the extra magnitude freedom matters most.

**The costs, which is where the judgment is.** DoRA is not free at training time: computing `‖W + (α/r)BA‖_c` requires materializing that sum, which is exactly the `d_out × d_in` intermediate that plain LoRA avoids. Implementations use tricks to reduce this, but expect a real training-throughput hit — plan for something in the tens of percent, and measure rather than trusting a number. At *inference*, DoRA still merges cleanly into a single weight matrix, so serving cost is unchanged. That is the important part.

**My decision rule:** if you are at `r ≥ 16` and quality is fine, DoRA is not worth the throughput hit. If you are memory-constrained into `r = 4–8` and you are leaving quality on the table, DoRA is the highest-value upgrade available and I would try it before raising rank. It is `use_dora=True` in PEFT — a one-line experiment, so the cost of finding out is an afternoon.

### What is LoRA+ and why do A and B want different learning rates?

Because they are not symmetric objects, even though the formula looks symmetric.

Look at the dimensions. `A` maps `d_in → r`; `B` maps `r → d_out`. In a wide model, `d_in` and `d_out` are thousands while `r` is tens. Under standard initialization the two matrices sit at very different scales, and — this is the load-bearing part — the *feature-learning* analysis of infinite-width networks says that for the composite `BA` to update efficiently, the two factors need learning rates that differ by a factor scaling with the model width, not the same rate. With a single shared LR, one of the two factors is effectively frozen relative to the other and the composite update is suboptimal. `B` is the under-served one (it starts at zero and must travel farthest).

LoRA+ therefore sets `lr_B = λ · lr_A` with `λ ≫ 1`; the paper's practical recommendation is `λ` around 16, with the caveat that it interacts with your base LR.

**📄 Paper:** Hayou, Ghosh & Yu (2024), *LoRA+: Efficient Low Rank Adaptation of Large Models* — argued from width-scaling analysis that a single LR for both LoRA factors is inefficient, and that assigning `B` a substantially larger LR improves both convergence speed and final quality at no memory cost.

Implementing it is trivial — it is just parameter groups:

```python
opt = torch.optim.AdamW([
    {"params": [p for n,p in model.named_parameters() if n.endswith("A") and p.requires_grad],
     "lr": base_lr},
    {"params": [p for n,p in model.named_parameters() if n.endswith("B") and p.requires_grad],
     "lr": base_lr * 16},
])
```

**⚠ Trap:** LoRA+ is often reported as "converges faster." Faster convergence in a fixed-epoch regime is a *quality* improvement; in a converged regime it may be nothing. Before you claim it helped, check whether your baseline was actually converged. Half the reported PEFT wins in the wild are "method X reached in 1 epoch what the baseline reached in 3," which matters if you are compute-bound and does not matter if you are not. Say which regime you are in.

### PiSSA, OLoRA, VeRA — what do these initialization variants change, and do any of them matter to me?

They all attack the same thing from different angles: the standard random-`A`/zero-`B` init means your adapter starts with **no knowledge of `W` at all**, and has to discover a useful subspace from scratch.

**PiSSA** takes the SVD of the pretrained `W = UΣVᵀ` and initializes `A` and `B` from the **top-`r` principal components**, then freezes the residual (`W` minus those components) as the base. The reasoning: the principal directions carry most of `W`'s energy, so putting your trainable rank exactly there means you are adapting the directions that matter most, and you start from a decomposition of the real weights rather than from noise. Reported effect is faster convergence and better final loss than vanilla LoRA at equal rank.

**📄 Paper:** Meng et al. (2024), *PiSSA: Principal Singular Values and Singular Vectors Adaptation* — initializes the adapter from the leading singular subspace of `W` and trains that, freezing the residual.

**OLoRA** initializes the LoRA factors using an orthonormal basis derived from a QR decomposition of the pretrained weights, so the adapter starts in a well-conditioned, orthonormal frame rather than a random one. The claim is faster and more stable convergence. I would describe the mechanism at this level of detail and not overclaim magnitudes — the evidence base is thinner than LoRA's or QLoRA's.

**VeRA** goes the other direction entirely: freeze `A` and `B` as **shared random matrices across all layers** and train only two small scaling vectors per layer (`d` applied inside, `b` applied on the output). Trainable parameters collapse by roughly another order of magnitude — you are storing per-layer vectors, not per-layer matrices, and the random matrices are reproducible from a seed. This makes VeRA interesting for the extreme multi-tenant case: thousands of adapters where even 84 MB each is too much. It gives up capacity to get there.

**📄 Paper:** Kopiczko, Blankevoort & Asano (2024), *VeRA: Vector-based Random Matrix Adaptation* — freezes a single pair of random low-rank matrices shared across layers and learns only per-layer scaling vectors.

**My honest ranking for a product engineer:** none of these is a first move. The order of magnitude of gains available from *data quality* and from *targeting all linear layers* dwarfs the difference between init schemes. PiSSA is the one I would actually reach for, and only in the specific case where I am compute-limited to very few steps and need the fastest possible convergence. VeRA I would reach for only if a multi-tenant storage calculation says I need it — and I would do that calculation first.

**🗣 Say this in the room** if asked which PEFT variant you use: "Vanilla LoRA at rank 16 on all linear layers, alpha 32, rsLoRA on if I go above rank 64. I know DoRA, LoRA+, PiSSA and VeRA and what each fixes, but I treat them as second-order — I have never seen one of them rescue a run that a better dataset wouldn't have rescued more."

### There's a well-known result titled "LoRA learns less and forgets less." Explain it, and tell me where LoRA genuinely underperforms full fine-tuning.

The result is exactly what the title says, and both halves are important because they are two sides of one mechanism: **LoRA is a constrained optimizer, and the constraint acts as a strong regularizer.**

*Learns less*: on tasks with substantial distribution shift from pretraining — the study's headline domains were continued pretraining on code and on math — LoRA does not reach full fine-tuning's target-domain performance, even at generous ranks. The low-rank subspace simply cannot express the size of update those tasks require.

*Forgets less*: the same constraint means LoRA-tuned models retain far more of their original general capability. Full fine-tuning on a narrow corpus degrades the source distribution measurably — the model gets worse at things you never intended to touch. LoRA's degradation is much smaller.

**📄 Paper:** Biderman et al. (2024), *LoRA Learns Less and Forgets Less* — a systematic comparison across code and math domains at both instruction-tuning and continued-pretraining scale, establishing the regularization framing rather than the "LoRA ≈ full FT" folk claim.

**Where LoRA genuinely underperforms, as a checklist I use:**

1. **Large distribution shift / continued pretraining.** Tens of billions of new-domain tokens. LoRA is the wrong tool; you want full or at least high-rank tuning.
2. **A new language**, especially one with poor tokenizer coverage. You are fighting the tokenizer and the embedding table simultaneously, and LoRA touches neither well.
3. **A new modality.** Bolting a vision or audio encoder onto an LLM requires training a real projection module; there is nothing low-rank about learning a cross-modal mapping from scratch.
4. **Genuinely new factual knowledge.** This is not a LoRA limitation specifically — fine-tuning of *any* kind is an unreliable knowledge-injection mechanism, and LoRA's smaller capacity just makes it fail sooner and more confidently. Use retrieval.
5. **Very small base models.** At 1B and below, full fine-tuning is cheap enough that LoRA's memory argument evaporates, and the quality gap goes the other way.

**Where LoRA matches or wins:** instruction tuning, style and register, output-format compliance, tool-call shaping, domain terminology, safety/refusal behaviour, and — importantly — any setting where you *care* about not regressing general capability. In multi-tenant products, "forgets less" is not a consolation prize; it is the requirement. A tenant adapter that makes the model excellent at their ticket taxonomy and mediocre at English is a support escalation.

**🗣 Say this in the room:** "LoRA is a regularized fine-tune. The trade is real and it is directional: less target-domain gain, much less catastrophic forgetting. For instruction tuning, format and style — which is most of what a product team actually needs — that trade is free. For continued pretraining into a new domain or a new language, it is a bad trade and I'd argue for full fine-tuning or a different base model."

### How many epochs, what learning rate, what schedule — give me a LoRA recipe you'd actually defend in review.

Here is the recipe I start every run from, and the reasoning for each number. Treat it as a prior to be updated by your eval, not a law.

**Learning rate: `1e-4` to `2e-4`** with cosine decay to ~10% of peak and a warmup of 3–5% of total steps. This is 10–100× higher than full fine-tuning's `1e-5`–`2e-5`, and the reason is structural: you are optimizing 42M well-conditioned parameters that started at zero, not 8B parameters sitting at a delicate pretrained optimum. If your loss is noisy and spiky, halve it; if loss falls smoothly but slowly, double it.

**Epochs: 1–3.** This is where most people break their run. LoRA on a few thousand examples overfits fast — you will see train loss keep falling while eval loss turns up somewhere in epoch 2. **My rule in review: if you ran more than 3 epochs, you need to show me an eval curve, not just a final number.** For datasets under ~1,000 examples, 3–5 epochs is defensible; for 50k+, 1 epoch is often right.

**Batch: as large as fits, via gradient accumulation.** Effective batch 64–128 sequences is a good target. LoRA gradients are noisy because you are estimating an update to a small subspace; larger batches help more than they do in full fine-tuning.

**LoRA dropout: 0.05–0.1** for small datasets, 0 for large ones. **Weight decay: 0** on adapter parameters as a default — decay pulls `B` back toward zero, which is toward the base model, which is a strange implicit regularizer to stack on top of an already-regularized method.

**Sequence packing and loss masking.** Mask the loss to the completion tokens only unless you deliberately want the model to learn the prompt distribution. Pack short examples to fill the context and set attention boundaries properly — if you pack without a block-diagonal mask, examples attend across each other and you have silently corrupted your data. This bug does not crash, does not show up in loss curves clearly, and is genuinely common.

**⚠ Trap — the chat-template mismatch.** Train with one chat template and serve with another and your adapter is trained on tokens the model never sees at inference. The symptom is maddening: eval-in-the-training-script looks great, production is mediocre, nothing errors. The check is mechanical and takes two minutes: `tokenizer.apply_chat_template` on one training example, print the exact decoded string, then print the exact string your serving stack sends. Byte-for-byte, including the trailing generation prompt and BOS handling. I make this a required artifact in any fine-tuning PR.

**💰 Math — what a starter run actually costs.** 8B model, QLoRA, 20k examples averaging 1,200 tokens, 2 epochs = 48M training tokens. On one H100, a QLoRA 8B run with gradient checkpointing sustains order-of 4–8k tokens/sec depending on sequence length and kernel stack — call it 6k. `48e6 / 6e3 = 8,000 s ≈ 2.2 hours`. At roughly $3/hr for an on-demand H100 (**📅 Volatile:** GPU spot and on-demand pricing moves quarterly — verify before quoting), that is **$6.60 of compute**. The lesson to carry into the room: the GPU bill for a LoRA run is a rounding error. Your real costs are data curation and the eval harness, and that is exactly the argument for not reaching for fine-tuning until you have a stable eval — the eval is the expensive part either way.
