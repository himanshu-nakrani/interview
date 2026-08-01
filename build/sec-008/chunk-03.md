### Why did the field move from fp16 to bf16 for LLM training? Be precise about the bit layout.

Mental model: fp16 and bf16 are the same size and are *not* a precision-versus-precision trade — they are a trade of mantissa bits for exponent bits, i.e. precision for **dynamic range**. Deep learning turned out to need range far more than it needs precision, because gradients span many orders of magnitude and a value that underflows to zero is infinitely wrong while a value that is 0.4% off is fine.

The layouts:

| | sign | exponent | mantissa | max finite | smallest normal |
|---|---|---|---|---|---|
| fp32 | 1 | 8 | 23 | ~3.4e38 | ~1.2e-38 |
| fp16 | 1 | 5 | 10 | 65,504 | ~6.1e-5 |
| bf16 | 1 | 8 | 7 | ~3.4e38 | ~1.2e-38 |

bf16 is literally fp32 with 16 mantissa bits chopped off — the exponent field is identical, so conversion is a truncation and the *range* is the same as fp32. That single property is why bf16 killed fp16 for training.

What it buys, concretely. In fp16, gradient values below ~6e-5 flush to zero, and late-training gradients in a large transformer routinely live at 1e-6 to 1e-8. So fp16 training requires **loss scaling**: multiply the loss by a large factor `S` (e.g. 2^16) before backward so gradients land in representable range, then divide by `S` before the optimizer step. And because the right `S` changes over the run, you need a *dynamic* scaler that detects `inf`/`NaN` in the gradients, skips that step, and halves `S` — a control loop that skips real training steps and adds a synchronization point. bf16 has fp32's range, so gradients never underflow, so loss scaling is unnecessary and the whole GradScaler apparatus disappears.

The cost: 7 mantissa bits gives roughly 2–3 decimal digits of precision, so bf16 accumulation error is much larger than fp16's. This is fine because you never accumulate in bf16 — matmuls accumulate in fp32 inside the tensor core, and everything sensitive is upcast.

**📐 Numbers you must know:** **fp16 max = 65,504; `exp()` overflows at 11.09.** **bf16 has 8 exponent bits, same range as fp32, ~3 decimal digits of precision.** These two facts answer roughly six different interview questions.

**⚠ Trap:** "bf16 is lower precision so my results will be less accurate." The forward pass of a well-normalized transformer is remarkably insensitive to bf16 rounding. What is sensitive is *accumulation over many terms* — the softmax denominator, LayerNorm's mean and variance, the loss reduction over a batch, and above all the optimizer's weight update, where you are adding a 1e-7-scale increment to a 1e-2-scale weight. Add those in bf16 and the increment is entirely below the ULP of the weight: the update is silently a no-op and the model stops learning. That is the reason for fp32 master weights, and it is the crispest way to explain them.

### Where in a training step is fp32 mandatory, and why exactly?

Answer this as a list with a reason attached to each, because that is what separates "I read the mixed-precision docs" from "I understand the numerics."

**1. Optimizer state and the master weights.** The update `Δθ = lr · m̂/√v̂` is typically 1e-6 to 1e-8 in magnitude while `θ` is 1e-2. bf16 has ~8 bits of mantissa; the ratio 1e-8/1e-2 = 1e-6 needs ~20 bits to even register. In bf16, `θ + Δθ == θ` exactly, and training silently stalls. Hence the fp32 master copy: the authoritative weights live in fp32, updates accumulate there, and a bf16 cast is produced for the forward pass.

**2. Loss reduction and cross-entropy.** Summing `B·T ≈ 4M` per-token losses in bf16 loses catastrophic precision once the running sum grows large relative to the increment. And the softmax over a 128k vocabulary computes `logsumexp` over 128k terms. Both go in fp32. This is why every serious implementation calls `.float()` on the logits before CE, and why fused/chunked CE kernels exist — a `[8, 8192, 128000]` fp32 logit tensor is `8 × 8192 × 128000 × 4 = 33.5 GB`, which is why you chunk over the sequence dimension rather than materializing it.

**3. Normalization statistics.** LayerNorm/RMSNorm computes a mean and variance over `d_model` elements. In bf16 the variance of a vector with a few large outliers is badly estimated, and then you divide by its square root — error amplification exactly where the residual stream's scale is set. PyTorch's `LayerNorm` upcasts internally; if you hand-roll RMSNorm you must do it yourself.

**4. Softmax accumulation inside attention.** FlashAttention keeps the running max and running sum in fp32 registers even when Q/K/V are bf16, for the same log-sum-exp reason.

**5. Gradient all-reduce for very large world sizes** — summing across 1,024 ranks in bf16 accumulates meaningful error; fp32 reduction (or a hierarchical reduction) is the safe default, and it is a real bandwidth cost you pay knowingly.

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        dt = x.dtype
        x = x.float()                                       # fp32 statistics
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt) * self.weight)                     # cast back, then scale
```

**⚠ Trap:** putting `eps` outside the square root (`rsqrt(mean) + eps`) instead of inside. Inside, it is a variance floor that prevents division by zero on an all-zero vector. Outside, an all-zero input gives `rsqrt(0) = inf` and you get NaN on the first padded row. Every hand-rolled RMSNorm I have reviewed with a NaN bug had this.

### Walk me through what `torch.autocast` actually does, and what it deliberately does not touch.

Mental model: autocast is not "run the model in bf16." It is a per-operation dispatch policy — a lookup table that says, for each ATen op, whether to cast its inputs down to the autocast dtype, keep them in fp32, or promote to the widest input type. Understanding it as a policy table rather than a global mode is what lets you predict its behaviour.

The three lists, in spirit:
- **Cast to bf16/fp16**: the matmul-shaped ops — `linear`, `matmul`, `bmm`, `conv*`, `einsum`. These are the ops that hit tensor cores and where the entire speedup lives.
- **Keep in fp32**: reductions and numerically-sensitive ops — `softmax`, `log_softmax`, `layer_norm`, `sum`, `norm`, `cross_entropy`, `pow`, `exp`, and the loss functions. This is why the fp32 guarantees above are mostly automatic if you use standard modules.
- **Promote to widest input**: elementwise ops with mixed-dtype inputs — `add`, `cat`, `dot` — so you do not silently downcast an fp32 tensor by adding a bf16 one to it.

What autocast does **not** do, and this is the part that gets missed:
- It does **not** change your parameter dtypes. Your weights stay fp32; autocast inserts a cast at each op. That is why plain autocast on an fp32 model uses *more* memory than you might expect — you are holding fp32 weights and materializing bf16 copies. Actual memory savings come from holding bf16 parameters (FSDP `MixedPrecision`, `model.to(bfloat16)`) which is a different mechanism.
- It does **not** apply to the backward pass by dispatch — the backward of an op runs in whatever dtype the forward chose, recorded on the autograd graph. So you do not wrap `loss.backward()` in the autocast context; you wrap only the forward.
- It does **not** cast custom CUDA ops or anything going through a `torch.autograd.Function` unless you decorate with `torch.amp.custom_fwd/custom_bwd`.

```python
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
for batch in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss          # forward only inside the context
    scaler.scale(loss).backward()           # backward OUTSIDE
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
```

With bf16 the GradScaler is a no-op (`enabled=False`) and you can drop it entirely; I keep it in the template so the same code runs on fp16-only hardware.

**⚠ Trap:** casting the model with `model.half()` *and* using autocast, then wondering why numerics are worse than pure autocast. `model.half()` destroys your fp32 master weights — the parameters themselves are now fp16 and the optimizer is updating fp16 tensors, which is the "update below the ULP" failure. If you want bf16 parameters, get them through FSDP's mixed-precision policy which maintains a separate fp32 shard for the optimizer, not through a blanket `.half()`.

### Explain the autograd graph well enough that I believe you've debugged one. Cover `retain_graph`, `detach`, and in-place errors.

Mental model: the autograd graph is a DAG of `Function` nodes built *during the forward pass*, where each node holds references to the tensors it needs to compute its own backward. `backward()` is a topological traversal that, by default, frees those saved tensors as it goes — because they are the dominant memory cost and holding them after use would be a leak. Almost every autograd error you will hit is a consequence of that freeing, or of someone mutating a tensor that a node had saved.

**`retain_graph=True`** tells backward not to free the saved tensors, so you can call backward through the same graph again. The legitimate uses are narrow: multiple losses that share a subgraph and cannot be summed, or higher-order gradients (where you actually want `create_graph=True`, which is a different and stronger flag). The illegitimate use — the one I see constantly — is as a fix for "RuntimeError: Trying to backward through the graph a second time." That error usually means you accumulated a tensor across loop iterations without detaching, and `retain_graph=True` "fixes" it by growing the graph unboundedly until you OOM after 40 steps.

**`detach()`** returns a tensor sharing storage but with `requires_grad=False` and no `grad_fn` — it severs the edge. Use it for: the reference-model log-probs in DPO, the target in a distillation loss when you do not want gradient to the teacher, anything you log or store across steps (`total_loss += loss.detach()` — without the detach you retain the entire graph of every step of the epoch), and the target of a straight-through estimator.

**In-place errors** — "one of the variables needed for gradient computation has been modified by an inplace operation" — mean a node saved a tensor for backward and someone later mutated it, so the saved value is now wrong. Autograd detects this with a version counter on each tensor's storage; it is not a heuristic, it is exact. The usual culprits: `x += y` inside a block where `x` is needed for a backward; `relu_(...)` or any trailing-underscore op on a tensor that feeds a matmul; slice assignment into an activation. The debugging move is `torch.autograd.set_detect_anomaly(True)`, which makes autograd record the forward-time stack for each node so the error points at the *creating* line rather than at `backward()`.

**⚠ Trap:** `zero_grad()` versus `zero_grad(set_to_none=True)`. The latter is now the default in modern PyTorch and it is what you want: it releases the gradient tensors instead of filling them with zeros, saving memory and avoiding a pointless kernel launch per parameter. But it changes semantics subtly — `p.grad` is `None` rather than a zero tensor — so any custom code that reads `p.grad` unconditionally (a gradient-norm logger, a custom optimizer) must handle `None`. This breaks quietly: your grad-norm dashboard starts reporting the norm over only the parameters that happened to get gradients.

### Write me a custom `autograd.Function`. Pick a case where you actually need one.

The case worth showing is a **straight-through estimator**, because it is the canonical example of "the forward and backward are deliberately not each other's transpose" — which is the only reason to write a custom Function at all. If your op is composed of differentiable primitives, autograd already handles it and a custom Function is strictly worse.

The setup: you want to quantize activations (or round, or take a hard threshold) in the forward pass, but the derivative of a step function is zero almost everywhere, so gradients would die. STE says: do the discrete thing forward, pretend it was the identity backward.

```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x, scale)          # only tensors; use ctx.attr for scalars
        return torch.round(x / scale) * scale

    @staticmethod
    def backward(ctx, grad_out):
        x, scale = ctx.saved_tensors
        # identity gradient, but zero it where the input was outside the clip range
        mask = (x.abs() <= scale * 127).to(grad_out.dtype)
        return grad_out * mask, None             # one grad per forward input
```

The contract points that get graded: `forward` and `backward` are `@staticmethod`; you save tensors with `ctx.save_for_backward` (which participates in the version-counter checks) and non-tensors as plain attributes on `ctx`; `backward` returns exactly as many values as `forward` took inputs, with `None` for inputs that do not need gradients; and you call it as `RoundSTE.apply(x, scale)`, never `RoundSTE()(x, scale)`.

Two more things you must add for production use. First, `@torch.amp.custom_fwd` / `@torch.amp.custom_bwd` decorators, because autocast does not know your Function's dtype policy and will otherwise hand you whatever dtype the caller had. Second, `gradcheck`:

```python
x = torch.randn(20, dtype=torch.double, requires_grad=True)
torch.autograd.gradcheck(lambda a: SomeFn.apply(a), (x,), eps=1e-6, atol=1e-4)
```

`gradcheck` compares your analytic backward against finite differences and it must be run in **float64** — in fp32 the finite-difference noise swamps the signal and you get spurious failures. For an STE specifically gradcheck will fail by design (the backward is intentionally wrong), so you gradcheck the *differentiable* Functions and unit-test the STE's masking behaviour separately.

**⚠ Trap:** writing a custom Function to "save memory" by not saving activations, and getting a wrong gradient that trains anyway. A wrong-but-correlated gradient still descends the loss — often to a visibly worse plateau. If you write a custom backward and do not gradcheck it, you have shipped an untested numerical kernel into the middle of a training run. That is the review comment I write every time.

### `torch.compile` gave me a 5% speedup instead of the 30% I expected. How do you diagnose it?

Mental model: `torch.compile` wins by fusing many small kernels into few, eliminating launch overhead and intermediate memory traffic. Every **graph break** splits the region into two separately-compiled graphs with an eager segment between them, and fusion cannot cross that boundary. A model with 40 graph breaks is a model that got almost none of the benefit. So the diagnosis is: find the breaks, then decide whether to fix or accept each.

The first command, always:

```bash
TORCH_LOGS="graph_breaks,recompiles" python train.py
```

or in-process, `torch._dynamo.explain(model)(sample_input)`, which returns the break count and reasons. Setting `torch.compile(model, fullgraph=True)` turns every break into a hard error, which is the fastest way to enumerate them during development.

**The common break causes, in the order I find them:**
- **Data-dependent control flow** — `if x.max() > threshold:` or `if loss.item() > ...`. Any Python branch on a tensor value forces a sync and a break. Fix with `torch.where` or by moving the branch outside the compiled region.
- **`.item()`, `.cpu()`, `.numpy()`, printing a tensor** — all force a graph break plus a device sync. Logging code inside a compiled forward is the single most common accidental cause.
- **Unsupported Python** — some library calls, some `try/except`, generators, and anything Dynamo cannot trace.
- **Custom `autograd.Function`** without the right registration — traced opaquely, breaking fusion around it.
- **Dynamic shapes.** This is the subtle one: variable sequence lengths cause **recompiles**, not breaks. Dynamo specializes on shapes; a new sequence length triggers a fresh compile. Hit the recompile limit (default 8) and it falls back to eager permanently, so your throughput silently *degrades* over the run. The fixes are `dynamic=True`, or — far better for training — bucketing/padding sequence lengths to a small set of values so you compile a handful of variants and reuse them.

**💰 Math:** for a 7B training step where the eager step time is 420 ms, a realistic well-compiled step is ~330 ms — a 21% saving. On an 8×H100 node at roughly $2.50/GPU-hour on-demand (**📅 Volatile:** rates move), a 30-day run costs `8 × 24 × 30 × 2.50 = $14,400`. A 21% step-time reduction saves ~$3,000 on that run, or equivalently gets you 21% more tokens for the same money. That is why the two hours of graph-break hunting is worth it, and it is also why it is *not* worth it for a 2-hour LoRA job.

**⚠ Trap:** measuring compile speedup including the compile itself. The first step after `torch.compile` can take 60–120 seconds. Always warm up for several steps and measure steady state, and always compare against an eager baseline measured the same way with `torch.cuda.synchronize()` around the timing — asynchronous CUDA execution makes naive wall-clock timing report the queue-submission time, not the work.

### Explain gradient checkpointing. Derive the √n memory result and tell me what selective recompute changed.

Mental model: the backward pass needs the forward activations. You can either store them (memory) or recompute them (compute). Gradient checkpointing is that dial, and the surprising part is that the optimal setting is not "store all" or "store none" but a specific intermediate that gives you `O(√n)` memory for one extra forward pass.

The derivation. Take a network of `n` sequential layers. Store activations at every `k`-th layer — call these checkpoints; there are `n/k` of them. During backward, when you reach a segment you recompute its `k` layers' activations from the checkpoint at its start, holding at most `k` activations at once. Peak memory is therefore `n/k + k` activation-units. Minimize over `k`: `d/dk (n/k + k) = −n/k² + 1 = 0` gives `k = √n`, and peak memory `2√n`. **📄 Paper:** Chen et al. (2016), "Training Deep Nets with Sublinear Memory Cost."

The compute cost: each layer's forward is executed twice (once in the original forward, once in recompute). Since backward is roughly 2× the cost of forward, a full step is `forward + backward = 1 + 2 = 3` units; adding a second forward makes it 4, i.e. **~33% more compute** — which is where the "about 30%" figure everyone quotes comes from.

In practice nobody uses `√n` segments — the standard is `k = 1`, checkpoint every transformer block, because a transformer block's *internal* activations (attention scores, the 4×-wide FFN intermediate) dwarf its input, so checkpointing at block boundaries captures nearly all the savings with trivial bookkeeping.

```python
from torch.utils.checkpoint import checkpoint
def forward(self, x):
    for blk in self.blocks:
        x = checkpoint(blk, x, use_reentrant=False)  # use_reentrant=False is the modern path
    return x
```

**Selective recompute** is the refinement that matters now. **📄 Paper:** Korthikanti et al. (2022), "Reducing Activation Recomputation in Large Transformer Models." The observation: not all activations cost the same to store or to recompute. The attention softmax output is `O(T²)` per head — enormous to store, cheap to recompute. The FFN intermediate is `O(T·4d)` — large but a matmul to recompute, which is expensive. So instead of an all-or-nothing block checkpoint, recompute only the *memory-heavy, compute-cheap* pieces and store the rest. That gets you most of full checkpointing's memory saving at a fraction of the 33% compute penalty — the paper reports being able to eliminate the bulk of activation memory for only a few percent of overhead when combined with sequence parallelism.

**⚠ Trap:** using checkpointing with a model that has dropout or any RNG-dependent op, without RNG state handling. The recomputed forward must draw the *same* random numbers as the original or your gradient is computed against a different network than the one that produced the loss. PyTorch's `checkpoint` handles this by saving and restoring RNG state (`preserve_rng_state=True`, the default), but if you write your own recompute logic or use `use_reentrant=True` with custom RNG, you can get a silently wrong gradient. Modern LLM pretraining mostly has dropout at 0.0, which is why this bites people on fine-tuning runs rather than pretraining.

### Do the activation-memory arithmetic for one transformer layer. I want to see where the bytes go.

Mental model: activation memory scales with `batch × sequence × width × layers`, and unlike weights it scales with your *traffic shape*, not just your model. This is why a config that trains fine at 2k context OOMs at 8k, and it is the number you must be able to produce before someone asks "can we extend context?"

Take one pre-LN transformer block, batch `B`, sequence `T`, model width `d`, `h` heads, FFN width `4d`, all activations stored in bf16 (2 bytes). Counting the tensors the backward actually needs:

- Block input (for the residual): `B·T·d`
- Post-norm input to attention: `B·T·d`
- Q, K, V projections: `3·B·T·d`
- Attention output before the out-projection: `B·T·d`
- Attention probabilities: `B·h·T·T` — **this is the quadratic term**
- Post-norm input to the FFN: `B·T·d`
- FFN intermediate (post-up-projection, pre-activation): `B·T·4d`
- FFN activation output: `B·T·4d` (with SwiGLU there are two of these and a gate, so it is more)

Summing the linear-in-`T` terms: roughly `(1+1+3+1+1+4+4)·B·T·d = 15·B·T·d` elements, call it ~16 for round numbers, plus the quadratic `B·h·T²`.

**💰 Math — Llama-3-8B shape at 8k context, batch 1:** `d = 4096`, `h = 32`, `T = 8192`, `L = 32` layers.
- Linear term per layer: `16 × 1 × 8192 × 4096 × 2 bytes = 1.07e9 = 1.07 GB`. Times 32 layers = **34 GB.**
- Quadratic term per layer *if attention probabilities are materialized*: `1 × 32 × 8192² × 2 = 4.29e9 = 4.29 GB`. Times 32 layers = **137 GB.** Alone, on one 80 GB card, at batch 1.

That second number is the entire argument for FlashAttention in one line: it never materializes the `T×T` matrix, so the quadratic activation term drops out completely and you are left with the ~34 GB linear part. Add gradient checkpointing at block granularity and the stored-per-block term collapses to just the block inputs — `32 × 8192 × 4096 × 2 = 2.1 GB` for the whole model — at the cost of ~33% more compute.

**📐 Numbers you must know:** **activations without FlashAttention and without checkpointing are dominated by `B·h·T²` per layer** — it is the term that makes long context impossible naively. **With FlashAttention, activations are ≈ `16·B·T·d·L` bytes in bf16**, which for the 8B/8k/batch-1 case is 34 GB. Both numbers are worth carrying.

**⚠ Trap:** believing FlashAttention reduces *KV-cache* memory. It does not — it reduces *attention activation* memory during training and prefill by never writing the score matrix to HBM. KV cache is a serving-time structure and is entirely unaffected. Conflating the two is a fast way to lose credibility in a serving round.

### Derive the backward pass through attention. Where does FlashAttention's recompute fit?

Mental model: attention is `softmax(QKᵀ/√d)V`, a composition of three ops, so its backward is three chain-rule steps — and the only interesting one is the softmax Jacobian, which has the same "subtract the weighted mean" structure as the cross-entropy derivative and for the same reason.

Forward, with `S = QKᵀ/√d`, `P = softmax(S)` (row-wise), `O = PV`:

Given `dO`, step backwards.
1. `dV = Pᵀ · dO` — straightforward matmul transpose rule.
2. `dP = dO · Vᵀ`.
3. Through the row-wise softmax: for a row, `dS = P ⊙ (dP − rowsum(dP ⊙ P))`. The `rowsum(dP ⊙ P)` term is the projection that keeps the gradient tangent to the simplex — the direct analogue of `p − onehot(y)` summing to zero. Note it requires `P`, not `S`.
4. `dQ = dS · K / √d` and `dK = dSᵀ · Q / √d`.

So a naive backward needs `P` — the `[B, h, T, T]` probability matrix — which is exactly the tensor FlashAttention refuses to store. Its resolution: **recompute `P` tile by tile in the backward pass.** For each block of queries, reload the corresponding `K` and `V` tiles, recompute the block's scores, re-apply softmax using the *saved* per-row log-sum-exp statistic `L` (a single `[B, h, T]` vector, cheap to store) so no second pass over the row is needed, and immediately consume the tile to accumulate `dQ`, `dK`, `dV`.

That is the whole trick and it is worth naming precisely: **FlashAttention is gradient checkpointing applied at the granularity of the attention kernel, made exact and cheap by saving the log-sum-exp normalizer.** It is IO-aware — the win comes from keeping tiles in SRAM rather than round-tripping HBM — not from doing fewer FLOPs. It does *more* FLOPs in the backward and is still much faster, which is the cleanest possible demonstration that attention was memory-bound.

**📄 Paper:** Dao et al. (2022) for FlashAttention, Dao (2023) for FlashAttention-2 which rebalanced the work partitioning to reduce non-matmul FLOPs and improve occupancy.

**⚠ Trap:** thinking the softmax backward is `P ⊙ (1 − P) ⊙ dP` — the elementwise sigmoid-style derivative. Softmax's Jacobian is not diagonal; the off-diagonal terms are what produce the `rowsum` subtraction. Getting this wrong in a from-scratch implementation gives you a gradient that is correlated with the true one, so the model still trains, just worse — the most expensive kind of bug.

**🏋 Drill:** write the softmax backward in 5 lines from memory and verify with `torch.autograd.gradcheck` in float64 against a hand-written `autograd.Function`. Pass criterion: gradcheck passes at `atol=1e-6` on the first attempt.

### My loss just went to NaN at step 14,000 of a 50,000-step run. Walk me through what you do, in order.

This is a triage question and the grading is on *order and instrumentation*, not on guessing the cause. Here is the ladder I actually run.

**Step 0 — do not restart from scratch.** Before anything, confirm you have a checkpoint before the spike and that you can reproduce the failing step deterministically: same checkpoint, same data-loader state, same seed. If you cannot replay the exact batch, fix that first, because every subsequent step depends on it.

**Step 1 — locate it in time.** Was it a spike-then-NaN or an instant NaN? Plot loss, grad-norm-pre-clip, and LR on one axis. A grad-norm spike one or two steps *before* the NaN means the gradient blew up first and the weights are now poisoned. An instant NaN with a healthy grad norm the step before means it is the forward pass, usually data.

**Step 2 — locate it in space.** Attach forward and backward hooks that check for non-finite values and report the first module that produces one:

```python
def nan_hook(name):
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t) and not torch.isfinite(t).all():
            raise RuntimeError(f"non-finite output in {name}")
    return hook

for name, mod in model.named_modules():
    mod.register_forward_hook(nan_hook(name))
```

This tells you "layer 27's attention" rather than "the loss." For gradients, `register_full_backward_hook` with the same check, or `torch.autograd.set_detect_anomaly(True)` for a one-off run — it is ~3× slower so never leave it on.

**Step 3 — inspect the batch.** Decode the offending batch and look at it with your eyes. In my experience the majority of single-step NaNs at a stable point in training are data: a document of repeated characters, an enormous base64 blob, a broken Unicode sequence that tokenizes to thousands of byte-fallback tokens, or a sequence where every label is masked (all-`-100` rows make `cross_entropy` return NaN because it divides by zero valid tokens).

**Step 4 — check the usual numerics.** All-masked attention rows (a padding mask that masks every position produces `softmax(all −inf) = NaN`); `rsqrt` of a zero variance in a hand-rolled norm; a `log` of a zero probability in a custom loss; fp16 overflow in the logits; a GradScaler that has driven its scale to the floor.

**Step 5 — decide the remediation.** In order of increasing intervention: skip the batch and continue; rewind to the last good checkpoint and skip forward past that data shard; rewind and lower `β₂` from 0.999 to 0.95 for the affected region; rewind with z-loss/QK-norm enabled; lower the LR. See the loss-spike ladder for the reasoning behind that ordering.

**🗣 Say this in the room:** "First I make the failure replayable from a checkpoint. Then I bisect in space with non-finite hooks to find the first module producing NaN, and in time by looking at whether grad norm spiked before the loss did. Most single-step NaNs at a stable point in a run turn out to be one pathological document, so I decode the batch before I touch any hyperparameters."

### The loss looks completely healthy but the model is garbage at eval. What's your differential diagnosis?

This is the failure mode I care most about in a candidate, because a descending loss curve is enormously reassuring and it is reassuring about the wrong thing: it tells you the model is fitting *whatever objective you actually wrote*, which may not be the one you meant.

**🔍 Failure taxonomy — silently-wrong training, in the order I check:**

1. **Train/eval formatting mismatch.** The model was trained on a chat template and evaluated with raw concatenation, or vice versa; or the BOS token is added at training and not at eval. This is the single most common cause and it is nearly free to check: print the exact token IDs of one training example and one eval example side by side and diff them. Not the strings — the IDs.

2. **Loss on the wrong tokens.** Prompt tokens supervised, or assistant EOS not supervised (the model never learns to stop, so generation runs to the token limit and eval scores it as failure). See the masking taxonomy earlier in this section.

3. **Double shift or no shift.** Loss descends beautifully to a plateau ~1.5 nats above where it should be. The tell is comparing your plateau against the anchors: a 7B SFT settling at 2.4 instead of 0.8 is not "hard data."

4. **Eval leakage in the other direction — you are evaluating on training data.** Loss is great, eval is great, production is terrible. Check n-gram overlap between train and eval sets; for anything scraped, assume contamination until proven otherwise.

5. **Generation config mismatch.** Trained a model that produces good log-probs, evaluated it with `temperature=1.0, top_p=1.0` when your production config is `temperature=0`. Or the reverse: greedy eval on a model tuned for sampled diversity. Loss is completely insensitive to this; eval is not.

6. **The tokenizer changed.** Someone added special tokens and resized the embedding, and the new rows were initialized randomly (or worse, the resize dropped the tied output head). Loss barely moves because the new tokens are rare; behaviour around them is nonsense.

7. **Catastrophic forgetting.** Loss on your fine-tuning set is excellent because the model has overfit to it; general capability has collapsed. The instrument is a small held-out *general* benchmark run at every checkpoint, not just your task loss.

**⚠ Trap:** trusting `eval_loss` as your only eval. Eval loss is teacher-forced — at every position the model is given the *ground-truth* prefix. Generation is autoregressive and compounds its own errors. A model can have excellent teacher-forced loss and fall apart in free generation, which is exactly the exposure-bias gap. The rule I enforce: **every training run must have at least one generative eval that runs the actual inference path**, even if it is only 50 examples and an exact-match score. Loss curves are for debugging training; generations are for deciding whether to ship.

**🗣 Say this in the room:** "A healthy loss curve only tells me the model is fitting the objective I wrote. My first three checks are always: diff the token IDs of a training example against an eval example, decode the positions where labels aren't `-100`, and confirm the EOS token is supervised. Those three catch most of these."

### How do you make a training run reproducible, and where does exact reproducibility stop being achievable?

Mental model: reproducibility in training has three tiers, and conflating them wastes days. Tier one is *bitwise* determinism — the same run produces byte-identical weights. Tier two is *statistical* reproducibility — the loss curve lands in the same place within noise. Tier three is *provenance* — you can reconstruct exactly what produced a given checkpoint. Tier three is mandatory. Tier two is what you should actually target. Tier one is expensive and often not worth it.

For tier one you need: fixed seeds for Python, NumPy and Torch (`torch.manual_seed`, plus per-worker seeds in the DataLoader); `torch.use_deterministic_algorithms(True)`; `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic cuBLAS; `cudnn.benchmark = False` (autotuning picks different algorithms based on timing, so it is nondeterministic by construction); a fixed data order with a seeded sampler that is checkpointed alongside the weights; and no atomics-based kernels. The cost is real — deterministic algorithm selection can be 10–30% slower — and even then, changing the number of GPUs changes the reduction order in the all-reduce, so multi-GPU bitwise determinism does not survive a topology change.

The fundamental limit: floating-point addition is not associative. `(a + b) + c ≠ a + (b + c)` in fp32, and any parallel reduction is free to choose an order. NCCL's ring all-reduce, a `scatter_add`, an fp16 atomicAdd in a fused kernel — all of these give run-to-run variation in the last bits, which the exponential-ish dynamics of a long training run then amplify into visibly different loss curves after a few thousand steps. This is not a bug you can fix; it is the hardware.

So the discipline I actually enforce is tier three plus tier two. Tier three: every checkpoint carries a manifest with the git SHA (and a dirty-tree flag), the full resolved config, the dataset version hash, library versions, the world size, and the RNG + data-loader state. Tier two: run your baseline three times with different seeds and report the seed-to-seed standard deviation of your headline metric *before* you evaluate any experiment. If your metric moves 0.8 points across seeds, a 0.5-point "improvement" is not an improvement, and you have just saved yourself from shipping noise.

**⚠ Trap:** debugging a "nondeterminism bug" that is actually seed variance. Someone reports "the same config gave a different result" and the team spends a week. The first question is always "what is the seed-to-seed spread on this metric?" — and it is astonishing how often nobody has measured it. This is the same instinct as demanding error bars on a benchmark; carry it over from backend performance work, where you already refuse to accept a single p99 measurement as evidence.
