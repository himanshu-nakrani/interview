### Start me at the beginning. What does the feed-forward block in a transformer layer actually compute, and why is it there at all when attention already mixes information?

Attention and the FFN do two orthogonal jobs, and the cleanest way to hold it is this: **attention mixes across the token axis, the FFN mixes across the feature axis.** Attention is the only place in the whole stack where position `t` can read anything from position `t-500`. But attention itself is almost embarrassingly linear — it computes a convex combination of value vectors. Softmax is the only nonlinearity, and it acts on the *weights*, not on the content. If you stacked attention blocks with no FFN, you would have a very expensive way of averaging vectors together. The FFN is where the model actually *computes* something about the token it is currently holding: it is a per-position, position-independent, two-layer MLP applied identically to every token in the sequence.

Mechanically, for a hidden state `x` of shape `[B, T, d_model]`, the classic FFN is `down(act(up(x)))` where `up: d_model → d_ff`, `down: d_ff → d_model`, and `d_ff` is conventionally `4 · d_model`. No token talks to any other token inside it. You can literally reshape `[B, T, d]` to `[B·T, d]`, run the FFN, and reshape back — which is exactly what every serving kernel does, and exactly why the FFN is the piece you can shard by *expert* later without touching the sequence dimension at all.

The interpretability story that makes this feel less arbitrary: **📄 Paper:** Geva et al. (2021), "Transformer Feed-Forward Layers Are Key-Value Memories" — showed that the `up` projection's rows behave like *keys* that fire on recognizable input patterns, and the corresponding `down` columns behave like *values* that write a specific update into the residual stream. So the FFN is a large, learned, content-addressable lookup table: `up` decides which of the `d_ff` memories match this token's current state, the activation gates them, and `down` sums the matched values back into the residual stream. That framing is what makes Mixture-of-Experts feel inevitable rather than exotic — if the FFN is a lookup table with `d_ff` slots, and any given token only lights up a small fraction of the slots, then you are paying to read the entire table on every token for no reason.

**🗣 Say this in the room:** "Attention routes information between positions; the FFN transforms information within a position. It's a per-token MLP with an intermediate width of roughly 4× the model dimension, and it holds about two-thirds of the parameters. Because it's position-independent, it's the natural unit to sparsify — that's the whole basis for MoE."

### How much of a transformer's parameter budget is the FFN? Don't quote me a number, derive it.

Count matrices per layer, in units of `d_model²`, and the answer falls out.

Classic multi-head attention has four square projections: Q, K, V, and the output projection O. Each is `d_model × d_model`, so attention is `4 · d²`. The classic FFN has `up` of shape `d × 4d` and `down` of shape `4d × d`, so it is `4d² + 4d² = 8d²`. Total per layer: `12d²`. FFN share: `8/12 = 2/3` exactly. That is where the folk number comes from, and it is a *derivation*, not a memorized constant — which is why an interviewer asks for it this way.

Now update it for a modern model, because the classic ratio is stale in the direction that *strengthens* the point. Grouped-query attention shrinks K and V, and SwiGLU adds a third FFN matrix. Take Llama-3-8B: `d_model = 4096`, 32 query heads, 8 KV heads, `d_head = 128`, `d_ff = 14336`, 32 layers.

```
Q:  4096 × 4096          = 16.78 M
O:  4096 × 4096          = 16.78 M
K:  4096 × (8·128=1024)  =  4.19 M
V:  4096 × 1024          =  4.19 M
attention total          = 41.94 M

gate: 4096 × 14336       = 58.72 M
up:   4096 × 14336       = 58.72 M
down: 14336 × 4096       = 58.72 M
FFN total                = 176.2 M
```

FFN share of the block is `176.2 / 218.1 = 80.8%`. Sanity-check the whole model: `218.1 M × 32 layers = 6.98 B`, plus untied input and output embeddings at `128256 × 4096 = 525 M` each, giving `6.98 + 1.05 = 8.03 B`. That is Llama-3-8B, to three digits. The arithmetic closing exactly is how you know you have the config right.

**📐 Numbers you must know:** FFN is **2/3 of parameters under classic MHA + 4× FFN**, and **~80% under modern GQA + SwiGLU**. GQA moved the ratio *up*, because it deleted attention parameters without touching the FFN. Llama-3-70B works out the same way: 704.6 M FFN vs 151 M attention per layer, 82.4%.

**⚠ Trap:** people say "the FFN is two-thirds of the model" and then conclude "so attention is a third of the compute." Not at long context. Parameter count governs the *projection* FLOPs, which are linear in sequence length. The `QKᵀ` and `attn·V` matmuls have no parameters at all and grow as `T²`. At `T = 128k` the attention score computation dwarfs everything the parameter count would predict. Keep "parameters" and "FLOPs" as two separate ledgers; conflating them is one of the two or three most common ways candidates fail this section.

### Why is `d_ff` four times `d_model`? Is there anything principled about the 4, or is it folklore?

It is mostly folklore that got empirically confirmed and then frozen. Vaswani et al. (2017) used `d_model = 512, d_ff = 2048` for the base model with no ablation defending the 4 specifically, and the field inherited it. The honest answer is: the ratio is a width-vs-depth allocation knob, the loss surface is quite flat in it over roughly 2×–8×, and 4 sits comfortably in the flat region while being a power of two that tiles nicely onto tensor cores.

There are two real forces behind it. First, expressivity: the FFN needs an intermediate space wide enough to hold many more "memories" than the residual stream has dimensions, so features can live in superposition — `d_ff > d_model` is the point, and a 4× overcomplete basis is a lot of room. Second, hardware: a `[B·T, d] × [d, 4d]` GEMM is large enough to saturate tensor cores, whereas a very narrow FFN would leave the machine memory-bound on small matmuls.

What actually shifted the number in practice was SwiGLU, which introduced a third matrix and forced a compensating shrink to `8/3 · d_model ≈ 2.67×`. Then implementers rounded that up to hardware-friendly multiples and, in several families, *past* the parameter-neutral point because they wanted the extra capacity. Llama-2-7B: `d = 4096`, `8/3 × 4096 = 10922.7`, rounded to `11008` (a multiple of 256) — parameter-neutral, as Shazeer intended. Llama-3-8B: `d_ff = 14336 = 3.5 × d_model` — that is a deliberate *increase* in FFN capacity beyond parameter neutrality, paid for by the parameter savings GQA delivered elsewhere. Llama-3-70B keeps the same 3.5× ratio at `8192 → 28672`.

**⚠ Trap:** reading `intermediate_size: 14336` in a config and saying "that's the 8/3 rule." It isn't — 8/3 of 4096 is 10922. The 8/3 rule is about *holding parameter count constant when you add the gate matrix*, and modern configs frequently break it on purpose. If you are asked to reason about a config, compute the ratio rather than pattern-matching it.

### Walk me through GELU, SiLU and ReLU. What's actually different, and does the choice matter?

All three answer the same question — "how do I decide, per unit, how much of this pre-activation to pass through?" — and they differ only in how sharply they decide near zero.

ReLU is `max(0, x)`: a hard gate, exactly zero below the threshold, exactly identity above. Its virtue is that it produces genuine sparsity — in a ReLU FFN a large majority of intermediate units are exactly zero for any given token, which is exploitable at inference time and which is, philosophically, the ancestor of MoE. Its vice is the dead-unit problem and a non-differentiable kink.

GELU is `x · Φ(x)`, where `Φ` is the standard normal CDF. The intuition is stochastic-regularizer-made-deterministic: instead of gating on `x > 0`, gate on the *probability* that a standard normal draw falls below `x`. It is smooth everywhere and slightly non-monotonic near `x ≈ -0.75`, where it dips below zero — that small negative lobe turns out to help optimization. **📄 Paper:** Hendrycks & Gimpel (2016), "Gaussian Error Linear Units (GELUs)." Everyone ships one of two forms — the exact `x·Φ(x)` via `erf`, or the tanh approximation `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`. They differ by roughly 1e-3 in the worst region.

SiLU (a.k.a. Swish) is `x · σ(x)`. Same shape as GELU — smooth, small negative lobe — but cheaper, because a sigmoid is one exponential and no `erf`. Numerically, `x·σ(1.702x) ≈ GELU(x)` to within about 0.02 over the relevant range, which tells you these two are the *same function* to any accuracy the model cares about.

Does the choice matter? At the level of "GELU vs SiLU," no — I have never seen a credible, controlled, statistically-honest gap between them at scale, and I would push back on any design doc that claims one. At the level of "gated vs ungated," yes, measurably — that is the SwiGLU question, and it is a different question. The one place the choice genuinely bites is **numerics and portability**: exact-GELU and tanh-GELU are *not* the same function, and a checkpoint trained with one and served with the other will show a small, diffuse, maddening quality drift with no single failing test.

**⚠ Trap:** the exact-vs-approximate GELU mismatch. HuggingFace configs distinguish `gelu`, `gelu_new`, `gelu_pytorch_tanh`; ONNX exporters and some inference engines silently pick one. I have watched a team burn a week on "the quantized model is slightly worse" that was entirely an activation-variant mismatch, not quantization at all. The rule I enforce in review: pin the activation string from the training config into the serving config, and add a numerical parity test on one batch of logits with a tolerance tight enough to catch it (`max |Δlogit| < 1e-2` in bf16).

### What is a GLU, and write out exactly what SwiGLU computes.

A gated linear unit splits the up-projection into two parallel paths: one path is passed through a nonlinearity and used as a **multiplicative gate**, the other is passed through untouched as the **content**. The elementwise product is the FFN's intermediate representation. The mental model: an ungated FFN decides *how much* of a fixed direction to write; a gated FFN decides *how much* and *which*, because the gate is itself input-dependent and multiplies feature-by-feature. Multiplicative interactions are something a stack of linear-plus-pointwise-nonlinearity layers approximates only awkwardly, so you are handing the model a primitive it otherwise has to build.

**📄 Paper:** Dauphin et al. (2017), "Language Modeling with Gated Convolutional Networks" introduced the GLU; Shazeer (2020), "GLU Variants Improve Transformer" swapped the sigmoid gate for Swish/GELU/ReLU variants inside a transformer FFN and measured them.

Concretely, with `W_gate, W_up: d_model → d_ff` and `W_down: d_ff → d_model`:

```
SwiGLU(x) = W_down · ( SiLU(x @ W_gate) ⊙ (x @ W_up) )
GeGLU(x)  = W_down · ( GELU(x @ W_gate) ⊙ (x @ W_up) )
ReGLU(x)  = W_down · ( ReLU(x @ W_gate) ⊙ (x @ W_up) )
```

Three matrices instead of two. Llama, Mistral, Qwen and DeepSeek all use SwiGLU; Gemma uses GeGLU. Naming is a genuine minefield: HuggingFace calls the three `gate_proj`, `up_proj`, `down_proj`; some checkpoints fuse `gate` and `up` into a single `[d, 2·d_ff]` matrix for one bigger GEMM (the fused form is faster — one launch, better tensor-core occupancy); and llama.cpp uses `ffn_gate`, `ffn_up`, `ffn_down`.

**⚠ Trap:** when you fuse `gate` and `up` into one weight, the halves' order is a convention, not a law. Getting it backwards gives you `SiLU(up) ⊙ gate` instead of `SiLU(gate) ⊙ up`. The model does not crash. It does not NaN. It produces fluent, confident, subtly worse text — because both halves are trained linear maps of the same input and the network partially compensates. This is the canonical "passes review, breaks quietly" bug in this section, and the only defense is a golden-logits parity test against the reference implementation on a fixed prompt.

### SwiGLU uses three matrices where the classic FFN used two. How do you keep the parameter count constant, and where does 8/3 come from?

Set the two expressions equal and solve — this is a ten-second derivation and interviewers ask it precisely because it separates people who read the paper from people who memorized "8/3."

Classic FFN parameters: `d · d_ff + d_ff · d = 2 · d · d_ff`, and with `d_ff = 4d` that is `8d²`.
SwiGLU FFN parameters: `3 · d · d_ff'`.

Setting `3 · d · d_ff' = 8d²` gives `d_ff' = (8/3) · d ≈ 2.667 · d`. Equivalently: multiply the classic `4d` by `2/3`, because you went from 2 matrices to 3. That is exactly the compensation Shazeer applied so his ablations compared architectures at matched parameter count and matched FLOPs rather than accidentally comparing a bigger model to a smaller one.

FLOPs check, because parameter neutrality and FLOP neutrality coincide here and you should be able to say so. Per token, a `[d] × [d, n]` matmul is `2·d·n` FLOPs. Classic: `2·d·4d + 2·4d·d = 16d²`. SwiGLU: three matmuls of `d × (8/3)d`, so `3 · 2 · d · (8/3)d = 16d²`. Identical. So SwiGLU is a free lunch in the accounting sense — the extra quality is not bought with extra FLOPs.

**⚠ Trap:** the *memory bandwidth* is not neutral even though parameters and FLOPs are. At decode with small batch you are bandwidth-bound, and you now issue three weight reads and one extra elementwise kernel instead of two reads. If your kernels are unfused, SwiGLU is measurably slower per token than a classic FFN of identical parameter count. This is why every serious engine fuses `gate` and `up` into one GEMM and fuses the SiLU-and-multiply into a single elementwise kernel. If you profile a naive PyTorch SwiGLU at batch 1 and conclude "SwiGLU is slow," you have measured your kernel launcher, not the architecture.

**🗣 Say this in the room:** "Three matrices instead of two, so you scale the intermediate width by 2/3 — `4d` becomes `8/3·d` — and parameters and FLOPs both come out identical to the classic FFN. That's how Shazeer's ablation was a fair comparison. Modern configs often break neutrality deliberately: Llama-3 runs `3.5·d`, spending the parameters GQA freed up."

### What did Shazeer's "GLU Variants" paper actually establish, and what's your honest read on why SwiGLU won?

**📄 Paper:** Shazeer (2020), "GLU Variants Improve Transformer" — took the T5 encoder-decoder setup, replaced the ReLU FFN with GLU variants at matched parameter count, and reported consistent but small improvements in pretraining perplexity and downstream GLUE/SuperGLUE scores, with GEGLU and SwiGLU at the top. The paper's own closing line is famously and honestly deflationary — it offers no explanation for why the variants work and attributes the success to "divine benevolence." That is not a joke you should skip in an interview; it is the correct epistemic status. Nobody has a mechanistic account of the gain that survives scrutiny.

My honest read on why SwiGLU is now near-universal:

The measured gain is real but small — on the order of a fraction of a perplexity point at matched parameters, which at the frontier translates to something like a low-single-digit-percent effective compute saving. It is free (no FLOP cost, as derived above), it is stable, and it introduced no new hyperparameters. When a change is free and non-negative, the field adopts it and never revisits — that is a *path-dependence* argument, not a mechanism argument.

The mechanistic hypotheses I would actually voice: the multiplicative gate gives the FFN a data-dependent, per-feature attenuation that an ungated MLP can only synthesize; and the gate provides a second, multiplicative gradient path that appears to improve conditioning early in training. I would explicitly label both as hypotheses.

**🗣 Say this in the room:** "SwiGLU buys a small, consistent, free improvement — matched parameters, matched FLOPs, no new hyperparameters. The paper itself declines to explain why, which I think is the right level of confidence. I'd characterize the adoption as 'costless and non-negative, therefore universal,' not as a deeply understood result."

**⚠ Trap:** claiming SwiGLU gives some large percentage improvement. If you assert a big number you will be asked for the source, and there isn't one. Being calibrated about the size of a well-known effect is a stronger signal than enthusiasm about it.

### Implement a SwiGLU FFN from scratch in PyTorch, and tell me its per-token FLOPs and its parameter count.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """Llama-style FFN. Fused gate+up for one GEMM instead of two."""
    def __init__(self, d_model: int, d_ff: int | None = None, bias: bool = False):
        if d_ff is None:                       # parameter-neutral 8/3 rule,
            d_ff = int(8 * d_model / 3)        # rounded up to a multiple of 256
            d_ff = ((d_ff + 255) // 256) * 256
        super().__init__()
        self.d_ff = d_ff
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=bias)  # fused
        self.down    = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):                       # x: [..., d_model]
        gate, up = self.gate_up(x).chunk(2, dim=-1)   # each [..., d_ff]
        return self.down(F.silu(gate) * up)

    def flops_per_token(self):
        d_in = self.gate_up.in_features
        return 2 * d_in * (2 * self.d_ff) + 2 * self.d_ff * d_in   # = 6·d·d_ff

if __name__ == "__main__":
    ffn = SwiGLUFFN(4096)                      # -> d_ff = 11008
    x = torch.randn(2, 7, 4096)
    assert ffn(x).shape == (2, 7, 4096)
    p = sum(t.numel() for t in ffn.parameters())
    print(ffn.d_ff, f"{p/1e6:.1f}M params", f"{ffn.flops_per_token()/1e9:.2f} GFLOP/token")
    # 11008  135.3M params  0.27 GFLOP/token
```

Check the arithmetic by hand: `3 · 4096 · 11008 = 135.3 M` parameters, and `6 · 4096 · 11008 = 270.5 MFLOP` per token — which is exactly `2 × parameters`, the universal rule that a forward pass costs about 2 FLOPs per parameter per token (one multiply, one add). That identity is your fastest sanity check on any FLOP estimate you produce under pressure, and it holds for every dense matmul in the model.

**⚠ Trap:** `chunk(2, dim=-1)` on a fused weight assumes the gate half comes first. When you load a checkpoint that stored them the other way, or that stored them unfused, you must map explicitly. If you write this in an interview, say the assumption out loud — "I'm assuming gate-major ordering in the fused weight" — because that one sentence is the difference between someone who has written this and someone who has read it.

### Where does the FFN sit on the roofline? Is it compute-bound or memory-bound, and does the answer change between prefill and decode?

The FFN is the same matmul in both phases; what changes is how many tokens share each weight read. That single sentence is the whole answer, and it generalizes to every weight matrix in the model.

Arithmetic intensity for a weight matrix of `P` parameters processing `B` tokens: you move `P · bytes_per_param` bytes once and do `2 · P · B` FLOPs, so intensity is `2B / bytes_per_param` FLOP per byte. In bf16 that is `B` FLOP/byte. It depends *only on the token count*, not on the matrix shape — which is why the answer to "compute or memory bound?" is always "what's the batch?"

An H100 SXM does roughly 495 TFLOP/s dense bf16 against 3.35 TB/s of HBM. The ridge point is `495e12 / 3.35e12 ≈ 148 FLOP/byte`. So:

- **Prefill** of a 4k-token prompt: `B = 4096` tokens flow through each FFN weight in one pass. Intensity ≈ 4096 FLOP/byte, ~28× past the ridge. Solidly compute-bound. Prefill is where you burn tensor-core FLOPs and where MFU numbers of 40–50% are achievable.
- **Decode at batch 1:** `B = 1`. Intensity ≈ 1 FLOP/byte, ~148× *below* the ridge. You are reading the entire weight matrix from HBM to do one vector-matrix product. Utterly memory-bound; tensor cores are idle ~99% of the time.
- **Decode at batch 256:** intensity ≈ 256 FLOP/byte, just past the ridge. This is why continuous batching exists and why "batch until you hit the ridge point" is the single most valuable serving lever.

**📐 Numbers you must know:** the H100 bf16 ridge point is **~148 FLOP/byte**; at fp8 (≈990 TFLOP/s dense, 1 byte/param) it is `990/3.35 ≈ 295` FLOP/byte but the bytes-per-param halves too, so the *token* threshold works out similar — roughly **a few hundred tokens in flight before you leave the memory-bound regime.** Derive it, do not memorize it: `ridge_tokens ≈ peak_FLOPs / (bandwidth · FLOPs_per_byte_per_token)`.

**💰 Math:** at batch 1 on Llama-3-8B in bf16 (16 GB of weights), the theoretical floor per decode step is `16e9 / 3.35e12 = 4.8 ms`, i.e. ~210 tok/s, and *no amount of FLOPs helps.* At batch 64 you read the same 16 GB and produce 64 tokens: still 4.8 ms, now 13,300 tok/s. Same hardware, 64× the throughput, near-identical per-token latency. That is the entire economic argument for a shared serving tier over per-tenant dedicated instances.

**⚠ Trap:** believing that the FFN being "two-thirds of the parameters" makes it two-thirds of decode latency in a way you can optimize away. At batch 1 you are bound by *total bytes read*, so the FFN is two-thirds of your latency simply because it is two-thirds of your bytes. The fix is never "make the FFN faster," it is "read fewer FFN bytes per token" — which is quantization, or sparsity, or MoE.

### Given the FFN is 80% of the weights, what are the levers to make it cheaper, and how do they rank?

Ranked by how often I actually reach for them:

**1. Quantization.** Cut `bytes_per_param` from 2 to 1 (fp8) or 0.5 (int4-ish). Directly halves or quarters the bytes read, which at decode is a direct latency win. It touches nothing about the model's structure, it is orthogonal to everything else, and modern weight-only int4 with per-group scales costs very little quality on the FFN specifically — the FFN is markedly more forgiving than attention's KV or the embedding table. This is the first lever, always.

**2. Sparsity / MoE.** Stop reading FFN weights you would have multiplied by ~zero anyway. Two flavors: *learned, structural* sparsity (MoE — you decide at train time that the FFN is 8 or 256 separate blocks and route), and *contextual* sparsity at inference time (**📄 Paper:** Liu et al. (2023), "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time" — predict which FFN units will be near-zero for this token and skip loading them). MoE is the one that won at scale, for a reason worth stating: contextual sparsity needs a predictor and gives you irregular memory access, whereas MoE gives you dense, contiguous, tile-friendly GEMMs on a *subset* of blocks.

**3. Distillation to a smaller `d_model`.** Shrinking `d` shrinks the FFN quadratically (`3·d·(8/3)d = 8d²`). The most effective lever per unit of quality loss, and the most expensive to execute, because it requires a training run.

**4. Kernel work.** Fusing gate+up into one GEMM, fusing SiLU-and-multiply, using the right tile sizes. Real but bounded: single-digit to low-double-digit percent, and you get it for free from vLLM/SGLang/TensorRT-LLM.

**5. Layer skipping / early exit.** I list it to note that I rarely ship it. Quality is unpredictable per-request, the KV cache bookkeeping for skipped layers is a genuine mess, and it fights continuous batching because different sequences in a batch want different depths.

**⚠ Trap:** reaching for MoE as a *cost* lever on an existing deployment. MoE reduces FLOPs per token, and you are not FLOP-bound at decode — you are bound by bytes and by HBM capacity, both of which MoE makes *worse* in absolute terms. That inversion is the thesis of the rest of this section, and getting it backwards is the single most common failure in an MoE serving interview.

### If the FFN is a key-value memory with `d_ff` slots, how much of it actually fires per token? Why does that matter?

This is the empirical observation that makes MoE more than an efficiency hack — it is a claim that dense FFNs are *already* sparse and are simply paying to prove it every token.

In ReLU-activated FFNs the measurement is unambiguous, because the sparsity is exact: for a typical token, the overwhelming majority of the `d_ff` intermediate units are exactly zero. Reported figures for older ReLU models sit around 90–97% zeros per token, meaning you loaded and multiplied by roughly the entire `up` matrix in order to discard nearly all of it. With smooth activations like SiLU and GELU the zeros become near-zeros rather than exact zeros, so "sparsity" becomes a thresholding question rather than a counting question — the magnitude distribution stays heavy-tailed, but you no longer get sparsity for free, which is precisely why some research pushed back toward ReLU-family activations for inference efficiency.

Why it matters, in three consequences:

*It licenses MoE.* If only a small fraction of memories fire per token, and if which ones fire is predictable from the token's hidden state, then you can pre-cluster the memories into `E` groups, learn a tiny classifier that predicts the right group, and read only that group. That is exactly an MoE layer: the router is the predictor and the experts are the clusters. MoE is coarse-grained, learned-at-train-time contextual sparsity, made regular so that GEMMs stay dense.

*It sets a ceiling on pruning.* Because the *union* over tokens of active units is close to all of them, you cannot statically prune the FFN much without losing rare-but-important behaviors. Static magnitude pruning of an FFN reliably degrades exactly the long tail you care about in production — rare entities, unusual formats, low-resource languages — while leaving your generic benchmark scores nearly intact. I have seen a pruned model ship on a green eval dashboard and then destroy quality on the 4% of traffic that was non-English.

*It explains why per-expert calibration data is scarce.* When you later quantize an MoE, each expert only sees a fraction of your calibration tokens. Hold that thought — it becomes a real production problem.

**🗣 Say this in the room:** "Dense FFNs are already contextually sparse — for any given token most intermediate units contribute nothing. MoE is that sparsity made *structural and predictable* so the hardware can exploit it: instead of loading everything and discarding 90%, you learn a router that tells you which block to load. The catch is that the union of active units across tokens is nearly everything, so you still have to *store* it all."
