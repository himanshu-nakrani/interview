### Before we touch any math — what problem is attention actually solving that a stack of MLPs or an RNN doesn't?

The mental model is content-addressed lookup. An MLP is position-addressed: weight `W[i][j]` connects slot `i` to slot `j`, so what a layer does to a token is a fixed function of *where* that token sits, never of what it says. An RNN is bandwidth-limited: everything token 900 knows about token 3 has to have survived 897 sequential overwrites of a fixed-size hidden state. Attention is the third option — every token emits a query, every token advertises a key, and the routing is decided at runtime by content similarity. It is a differentiable hash lookup where instead of one bucket you get a convex combination of all buckets, weighted by how well each key matches.

The backend bridge that makes this click: think of the sequence as a small in-memory table with `T` rows. Each row publishes a key (its index entry) and a value (its payload). A query is a fuzzy `SELECT value FROM table ORDER BY similarity(key, query)` — except instead of taking the top-1 row you take a softmax-weighted average of *all* rows. Nothing about that operation cares whether the matching row is 2 positions back or 90,000, which is exactly the property RNNs lack. The path length between any two tokens is O(1), not O(T), and that is the single sentence that explains why transformers won.

The cost of that property is the whole rest of this section: to let every token look at every token you must compute `T × T` similarities, which is quadratic in compute *and*, if you are naive about it, quadratic in memory.

**🗣 Say this in the room:** "Attention is content-based routing. RNNs give you an O(T) path between distant tokens through a fixed-width bottleneck; convolutions give you a fixed receptive field; attention gives you an O(1) path with a data-dependent weighting, at the price of O(T²) similarity computations. Everything in modern architecture research is negotiating that price."

**⚠ Trap:** describing attention as "the model learns which words are important." That's a summary of the output, not the mechanism, and it will get you a follow-up you can't answer. The mechanism is that the *weights of the averaging* are computed from the data at inference time, not stored in parameters. That's the whole novelty. Say "data-dependent" or "input-dependent routing," not "important."

### Derive scaled dot-product attention for me. Start from the lookup analogy and end at the formula.

Start with an exact dictionary. You have `T` key–value pairs `(k_j, v_j)` and one query `q`. A hard lookup returns `v_j` where `k_j == q`. Write that as a selection vector: `s_j = 1[k_j == q]`, then output `= Σ_j s_j v_j`. That's already a matrix-vector product; the only problem is that equality is not differentiable and returns nothing when there's no exact match.

Soften it in two steps. First, replace equality with a similarity score. The dot product `q · k_j` is the cheapest similarity there is — one fused multiply-add per dimension, it's a matmul, GPUs do nothing else as well — and it's high when the vectors point the same way and large in magnitude. Second, replace the hard indicator with a softmax over those scores, which gives you a proper probability distribution over the `T` rows: non-negative, sums to one, differentiable everywhere, and it degenerates to the hard lookup as the scores spread out. So:

```
a_j = softmax_j(q · k_j)        # attention weights, a ∈ Δ^{T-1}
out = Σ_j a_j v_j               # convex combination of values
```

Now batch it. Stack queries into `Q ∈ R^{T_q × d_k}`, keys into `K ∈ R^{T_kv × d_k}`, values into `V ∈ R^{T_kv × d_v}`. All the pairwise dot products are one matmul: `QK^T ∈ R^{T_q × T_kv}`, where entry `(i, j)` is how much query `i` wants value `j`. Softmax along the last axis (over keys — this is the axis people get wrong), multiply by `V`:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The `√d_k` is the one term the lookup analogy doesn't hand you; it comes from a variance argument I'll do next. And notice the separation of concerns that the K/V split buys you: the key is what a position advertises about itself for matching purposes, the value is what it actually contributes if selected. Those are different jobs, so they get different learned projections. If you tied `K = V` you'd force "how findable am I" and "what do I contribute" into the same vector, and models trained that way are measurably worse.

**⚠ Trap:** softmaxing over the wrong axis. `softmax(scores, dim=-1)` normalizes each query's distribution over keys — correct. `dim=-2` normalizes each key's distribution over queries, which is a different, mostly meaningless operation that still trains, still produces plausible loss curves, and still runs. I have seen this survive code review twice. The unit test is one line: `assert torch.allclose(attn.sum(-1), torch.ones(...))`.

### Why √d_k? I want the derivation, not "it stabilizes training."

Because the dot product's variance grows linearly with dimension, and softmax is exponentially sensitive to the scale of its inputs. Those two facts together mean an unscaled dot product saturates the softmax at large `d_k`, and a saturated softmax has ~zero gradient.

Do the algebra. Assume the components of `q` and `k` are independent, zero-mean, unit-variance — which is roughly what your initialization scheme buys you at step 0. Then `q · k = Σ_{i=1}^{d_k} q_i k_i`. Each term has mean `E[q_i k_i] = E[q_i]E[k_i] = 0` and variance `E[q_i² k_i²] = E[q_i²]E[k_i²] = 1`. The terms are independent, so variances add:

```
Var(q · k) = d_k        ⇒        std(q · k) = √d_k
```

So at `d_k = 128` the raw logits have standard deviation ≈ 11.3, meaning typical spreads between the largest and smallest logit in a row are on the order of 40–50. Feed that to a softmax and one entry gets essentially all the mass: `e^{45} / (e^{45} + e^{0} + …)` is 1 to within floating-point noise. Dividing by `√d_k` renormalizes the logits back to unit variance regardless of head dimension, so the softmax starts in its high-entropy regime and can actually learn where to look.

Now the gradient argument, which is the part interviewers actually push on. For `p = softmax(z)`, the Jacobian is `∂p_i/∂z_j = p_i(δ_ij − p_j)`. If one `p_i → 1` and the rest → 0, every entry of that Jacobian → 0. The layer is not just confidently wrong, it is *ungradiented* — no signal flows back into `W_Q` or `W_K` to fix the confident mistake. That's the failure mode: not instability, paralysis.

**📐 Numbers you must know:** `std(q·k) = √d_k` under unit-variance init. `d_head = 64` → std 8; `d_head = 128` → std 11.3 (the modern default: Llama, Qwen, Mistral all use 128). Memorize the derivation, not the constant — "variances of independent terms add, there are `d_k` terms, each with variance 1, so std is `√d_k`" is a 12-second answer that reads as understanding.

**⚠ Trap:** claiming the scale is `1/√d_model`. It is `1/√d_head`. With `d_model = 4096` and 32 heads you'd be off by a factor of `√32 ≈ 5.7`. And a related one: some implementations fold the scale into the query projection's initialization instead of applying it at runtime. That's mathematically equivalent at step 0 and *not* equivalent after training, since weight decay and the optimizer act on the folded version differently. `F.scaled_dot_product_attention` takes a `scale=` argument defaulting to `1/√E` where `E` is the last dim of the query — don't override it unless you know why.

### The original attention mechanism used a small MLP to score query-key pairs, not a dot product. Why did the dot product win?

Because the dot product is a matmul and the MLP scorer is not, and on the hardware that exists that is the entire argument.

The original formulation — additive or "Bahdanau" attention, from the 2014 neural-machine-translation work that introduced attention to seq2seq — scores a query-key pair as `v^T tanh(W_q q + W_k k)`. It's more expressive per pair: a learned nonlinearity can represent similarity functions a bilinear form cannot. Multiplicative or "Luong" attention used `q^T W k`, an intermediate. Vaswani et al. (2017) took the pure dot product and added the `1/√d_k` scale, and noted in the paper that additive attention performs comparably at small `d_k` but that dot-product attention is much faster in practice because it is implemented as a highly optimized matrix multiplication.

Quantify the gap and it stops being a close call. For `T` queries against `T` keys of width `d`, the dot-product score matrix is one GEMM: `2T²d` FLOPs, executed on tensor cores at ~989 TFLOP/s bf16 on an H100, with data reuse — each key tile is loaded once from HBM and reused across a whole tile of queries. Additive attention must, for each of the `T²` pairs, form `W_q q + W_k k` (a `d`-vector), apply `tanh` elementwise, and dot with `v`. The projections `W_q q` and `W_k k` can be hoisted out (they're per-token, not per-pair), but the *addition, the tanh, and the final dot* cannot — they are `T² × d` elementwise operations with no reuse, so you're generating and consuming `T² × d` intermediate values. At `T=2048, d=128` that's `5.4e8` intermediates versus a `4.2e6`-element score matrix: **128× more memory traffic**, running on elementwise ALU throughput rather than tensor cores. In an era where attention is already memory-bound, that's not a 2× penalty, it's disqualifying.

The deeper point, and the one worth making in a room: **the architecture that won is the one that maps onto a GEMM.** The same logic explains why we use a linear `W_O` rather than an MLP to combine heads, why the FFN is two big matmuls rather than something cleverer, and why every "more expressive attention" proposal since has had to justify itself against a fused GEMM baseline that gets ~3× faster every hardware generation for free. Hardware sympathy is a first-class architectural constraint, not an implementation detail.

**⚠ Trap:** claiming dot-product attention is *better* than additive attention on quality grounds. It isn't obviously — the original paper reports comparable performance at small `d_k`, and additive attention is arguably more robust at large `d_k` precisely because it doesn't suffer the variance-growth problem that forced the `1/√d_k` fix. It won on throughput. Saying "it won because it's a matmul" is the correct and more impressive answer.

### Where do Q, K, and V actually come from? Walk me through the shapes for a real batch.

They are three learned linear projections of the same residual stream. That's the whole answer to "where," and the interesting part is the shape bookkeeping, which is what a from-scratch round is really testing.

Input to the block is `x` of shape `[B, T, d_model]` — batch, sequence, model width. Three weight matrices `W_Q, W_K, W_V`, each `[d_model, d_model]` in vanilla MHA, produce `q, k, v` each `[B, T, d_model]`. Now split the width into heads: `d_model = n_heads · d_head`, so reshape to `[B, T, n_heads, d_head]` and transpose to `[B, n_heads, T, d_head]`. That transpose is not cosmetic — it puts `(B, n_heads)` in the leading batch dimensions so the following matmuls are batched over `B·H` independent `[T, d_head]` problems, which is exactly the layout cuBLAS's batched GEMM wants.

Then `q @ k.transpose(-2, -1)` gives `[B, H, T, T]`, softmax over the last axis, `@ v` gives `[B, H, T, d_head]`, transpose back to `[B, T, H, d_head]`, reshape (contiguous!) to `[B, T, d_model]`, and finally through `W_O ∈ R^{d_model × d_model}` back to `[B, T, d_model]`. Same shape in, same shape out — that's what makes the block stackable and what makes the residual `x + attn(ln(x))` type-check.

Concretely for Llama-3-8B: `d_model = 4096`, `n_heads = 32`, `d_head = 128`, 32 layers. So a batch of 4 sequences of 2048 tokens carries `q` of shape `[4, 32, 2048, 128]` = 33.5M elements = 67 MB in bf16, and the score matrix is `[4, 32, 2048, 2048]` = 537M elements = **1.07 GB per layer**, which is the number that motivates FlashAttention.

**⚠ Trap:** forgetting `.contiguous()` before the final reshape (or using `.view` where you need `.reshape`). After `transpose(1, 2)` the tensor's strides are non-monotonic, and `.view` will throw — or worse, in some code paths a later kernel silently reads it with the wrong stride assumption. The rule I enforce: after any transpose that precedes a reshape, either call `.contiguous()` or use `.reshape`, and never `.view`.

**🗣 Say this in the room:** "`[B, T, d_model]` in, three projections, split the width into `n_heads × d_head`, transpose heads into the batch dimension so you get `[B, H, T, d_head]`, do the `T×T` thing, then merge heads back and hit it with `W_O`. Shape-preserving block, which is why you can stack 80 of them."

### Write me single-head scaled dot-product attention. From memory, in PyTorch, and tell me what each line costs.

```python
import torch, math, torch.nn.functional as F

def sdpa(q, k, v, mask=None):
    # q: [B, H, Tq, D]   k, v: [B, H, Tkv, D]   mask: broadcastable to [B, H, Tq, Tkv], True = keep
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))   # [B, H, Tq, Tkv]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)                        # over keys
    return attn @ v                                             # [B, H, Tq, D]
```

Line by line on cost, for `B=1, H=32, T=2048, D=128`. The first matmul is `2·B·H·T²·D = 2·32·2048²·128 ≈ 3.4e10` FLOPs and *materializes* a `[1,32,2048,2048]` tensor: 134M elements, 268 MB in bf16. The `masked_fill` allocates another one of those unless you fuse. The softmax reads and writes it again — that's 3 more passes over 268 MB, and at ~3 TB/s of HBM bandwidth on an H100 that's `3 × 0.268 GB / 3000 GB/s ≈ 0.27 ms` of pure memory traffic for an operation whose math takes maybe 0.02 ms. That ratio — 10× more time moving the score matrix than computing it — is precisely the observation FlashAttention monetizes.

In production you do not write this. You write:

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # dispatches to a fused kernel
```

which never materializes `[B,H,T,T]` in HBM at all. But you must be able to write the naive version unaided in under four minutes, because every from-scratch round asks for it and because you cannot reason about the fused version without knowing what it fused.

**⚠ Trap:** `masked_fill(mask, -inf)` versus `masked_fill(~mask, -inf)`. Half the codebases you'll read use `True = attend`, half use `True = masked out` (HuggingFace's `attention_mask` is `1 = real token`, but the *additive* masks it builds internally are `0 = keep, -inf = drop`). Getting the polarity backwards produces a model that attends only to padding, which trains to a flat, high loss that looks like "bad hyperparameters." Always name the variable for its semantics — `keep_mask` or `pad_mask`, never `mask`.

### What does the softmax denominator actually do here, and what breaks if you remove it or replace it?

The denominator enforces that attention is an *averaging* operation rather than an accumulation, and that single property is what keeps the residual stream's variance from exploding as `T` grows. Strip it and the output norm scales with the number of tokens attended to; a 100k-token context would produce activations ~1000× larger than a 100-token one, from the same weights. Nothing downstream — LayerNorm statistics, the FFN's operating range, the unembedding's logit scale — survives that.

Three consequences worth naming precisely. First, competition: because the weights sum to 1, attending more to token A necessarily means attending less to token B. That's what makes attention a routing decision rather than a set of independent gates, and it's what makes head specialization emerge. Second, shift invariance: `softmax(z + c) = softmax(z)`, which is why implementations subtract the row max before exponentiating and why masking with `-inf` works cleanly. Third — and this is the one that leads somewhere interesting — the model has **no way to attend to nothing**. Every row must spend its full unit of probability mass somewhere, even when the honest answer is "nothing here is relevant to me." That constraint is the direct cause of attention sinks, which I'll come back to.

The named alternatives and their honest status. *Softmax-1* (adding a constant 1 to the denominator so a row can output near-zero, popularized by a widely-circulated 2023 blog post under the title "Attention Is Off By One") gives the model an explicit no-op and empirically reduces the massive outlier activations that hurt quantization; it has not become the default. *Sigmoid attention* — per-element sigmoid with no normalization — has been studied and can be made to work with a length-dependent bias correction, but it needs that correction precisely because you removed the averaging. *Linear attention* replaces `exp(q·k)` with a kernel feature map `φ(q)·φ(k)` and keeps a denominator `φ(q)·Σφ(k)`; that keeps normalization and buys O(T) but reliably loses exact-recall quality. *ReLU attention with a `1/T` scale* has been shown to be competitive in vision transformers.

**🗣 Say this in the room:** "The denominator makes attention a convex combination, which does three things: it bounds the output norm independently of sequence length, it makes heads compete so they specialize, and it forces every query to spend all its mass — which is why models invent a sink token to dump unwanted mass on. Any softmax replacement has to answer all three, and the ones that skip the third one produce huge outlier activations."

**⚠ Trap:** claiming softmax exists "to make the weights sum to one so they're probabilities." That's a description, not a reason — you could normalize by an L1 sum and also get weights summing to one. The *exponential* is doing separate work: it makes the mapping from score gaps to weight ratios multiplicative and scale-free, so a fixed logit gap always means a fixed odds ratio no matter the absolute scores. That's what lets one temperature-like scale (`1/√d_k`) work across the whole model.

### Explain multi-head attention properly. Why more than one head, and what is the output projection actually doing?

One head can only compute one weighted average per position, so it must pick a single reason to route. Real language needs several simultaneously: the current token wants the previous token's identity, *and* the subject of the sentence, *and* the matching open bracket, *and* the last occurrence of this same token. Multi-head attention runs `n_heads` independent lookups in parallel over disjoint low-dimensional slices of the residual stream and then sums them back in.

The mechanism people mis-state: it is not "concatenate then project." Algebraically it is a *sum of per-head contributions*. Partition `W_O ∈ R^{d_model × d_model}` row-wise into `n_heads` blocks `W_O^{(h)} ∈ R^{d_head × d_model}`. Then

```
MHA(x) = Σ_h  head_h(x) · W_O^{(h)}          where head_h(x) ∈ R^{T × d_head}
```

Concatenation-then-one-matmul is just the efficient way to compute that sum. The reason to insist on the sum form: it tells you each head writes *independently and additively* into the residual stream, which is what makes ablating a single head meaningful, what makes head-level interpretability possible at all, and what justifies the circuits framing (`W_V W_O^{(h)}` is head `h`'s "OV circuit" — what it writes — and `W_Q W_K^{(h)T}` is its "QK circuit" — where it reads from). **📄 Paper:** Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits* — recast a head as two low-rank bilinear forms (QK and OV) acting on a shared residual stream, replacing the "concatenate features" intuition with something you can actually do algebra on.

Why the projections are low-rank matters too. Each head's `W_Q^{(h)}` maps `d_model=4096 → d_head=128`, so the QK circuit `W_Q W_K^T ∈ R^{4096×4096}` has rank at most 128. Attention heads are *forced* to be low-rank readers of the residual stream, which is a capacity constraint, not an accident: it's why 32 heads of width 128 behave differently from 1 head of width 4096, despite identical parameter counts.

**⚠ Trap:** "more heads is more capacity." At fixed `d_model`, more heads means *narrower* heads — the parameter count of the projections is identical whether you use 8 heads of 512 or 64 heads of 64. What changes is the rank of each head's read/write circuit and how many distinct routing patterns you can run at once. There is a real trade: too-narrow heads (`d_head = 32`) lose the ability to express fine-grained matching; too-few heads lose parallel routing. The industry converged on `d_head ∈ {64, 128}` for exactly this reason, and `128` won out partly because it maps cleanly onto tensor-core tile sizes.

### Is multi-head attention more expensive than single-head at the same total width? Do the FLOPs.

No — it's the same, to within the reshapes, and being able to show that in ten seconds is a good signal.

Take `d_model = D`, `n_heads = H`, `d_head = D/H`. Per head, the score matmul is `Q^{(h)} K^{(h)T}`: `[T, D/H] × [D/H, T]` = `2·T²·(D/H)` FLOPs. There are `H` heads, so the total is `2·T²·D` — the `H` cancels. Identical for the `attn @ V` product: `2·T²·D`. The projections are `4 × 2·T·D²` (Q, K, V, O) regardless of head count. So:

```
FLOPs per attention layer ≈ 8·T·D²  (projections)  +  4·T²·D  (the T×T part)
```

The score *memory* is also head-count invariant in total: `B·H·T²` elements is `B·H·T²`, and `H` doesn't cancel there — but at fixed `H` it's fixed; comparing `H=8` to `H=64` at the same `d_model` does change score memory by 8×, and that's the one asymmetry. That's a good detail to volunteer.

**💰 Math — the crossing point, which is the real question behind this one.** Attention's quadratic term overtakes the projections when `4T²D > 8TD²`, i.e. `T > 2D`. For Llama-3-8B (`D = 4096`) that's `T > 8192`. So at 2k context the `T×T` machinery is a *minority* of the block's FLOPs — roughly `4·2048²·4096 / (8·2048·4096²) = 1/4` — and people who say "attention is quadratic so it dominates" at 2k are wrong. At 128k context, the ratio is `4·131072²·4096` vs `8·131072·4096²` = **16×** in favour of attention. That single ratio, `T / 2D`, is the entire economic argument for sliding windows, linear attention, and SSM hybrids: it only bites past `T ≈ 2·d_model`.

**📐 Numbers you must know:** attention FLOPs cross FFN+projection FLOPs at `T ≈ 2·d_model`. Llama-3-8B: 8k. Llama-3-70B (`d_model = 8192`): 16k. Derive it live from `4T²D` vs `8TD²`; do not memorize the crossing points.

### Give me the compute and memory complexity of attention. Be precise about where the T² memory actually lives.

Compute is `O(T² · d)` per head per layer and that is irreducible for exact softmax attention — every one of the `T²` query-key pairs needs a `d`-dimensional dot product, and no algorithm avoids computing them if you want exact softmax over all pairs. FlashAttention does not change this. Sparse and linear attention change it by not computing all pairs.

Memory is where the nuance is, and it's the distinction that separates people who've read the FlashAttention paper from people who've read a tweet about it. There are two different memories:

1. **The score matrix** `[B, H, T, T]`. This is `O(T²)` and it is *optional*. A naive implementation writes it to HBM, reads it back for the softmax, writes the normalized version, reads it again for the `@V`. FlashAttention tiles the computation so a block of the score matrix is produced in SRAM, consumed immediately by an online-softmax accumulator, and never written to HBM. Peak HBM for the scores drops to `O(T)` (just the running row max and sum). **📄 Paper:** Rabe & Staats (2021) showed the `O(√T)`/`O(1)` memory formulation is possible with online softmax; Dao et al. (2022), *FlashAttention*, made it fast by being explicitly IO-aware about the SRAM↔HBM hierarchy, replacing the materialize-and-reduce implementation everyone was using.
2. **The KV cache** `2 · n_layers · B · n_kv_heads · T · d_head` elements. This is `O(T)` *linear*, it's mandatory during autoregressive decode, and it is the memory that actually kills you in production. Confusing (1) and (2) is the single most common attention-memory error in interviews.

**💰 Math — why (1) matters even though it's optional.** For `B=1, H=32, T=8192`, bf16: `32 × 8192² × 2 bytes = 4.29e9` = **4.3 GB** for one layer's score matrix. Materialize that on 32 layers with any activation checkpointing granularity and you are OOM on an 80 GB card at a context length people consider small. At `T = 131072` a *single head's* score matrix is `131072² × 2 = 34.4 GB`. So the honest statement is: exact attention at 128k is only tractable *because* of FlashAttention, not despite the quadratic.

**⚠ Trap:** saying "FlashAttention makes attention linear." It makes attention's *memory* linear and its wall-clock 2–4× faster by cutting HBM traffic; the FLOPs are still `O(T²d)` and it does not extend context for free. If an interviewer hears "linear," they will assume you've confused it with linear attention, which is a completely different (and lossy) thing.

### Explain causal masking. And be specific about why we set masked scores to −inf before the softmax rather than zeroing the weights after it.

Causal masking is what makes a decoder-only transformer trainable on all `T` positions at once instead of `T` separate forward passes. Position `t` must predict token `t+1`, so it must not see anything at index `> t`. The mask is lower-triangular: `keep[i, j] = (j <= i)`.

Now the actual question. Zeroing after the softmax is *wrong*, not merely inelegant, and the reason is the denominator. Softmax normalizes over *all* `T` entries including the future ones. If you then zero the future entries, the surviving weights no longer sum to 1 — they sum to whatever fraction of the mass the past happened to receive. Position 0 in a 2048-token sequence would keep roughly 1/2048 of its mass and output a vector 2048× too small; position 2000 would be nearly fine. You have introduced a position-dependent output scale that the model must waste capacity compensating for, and worse, the *relative* weights among past tokens are still contaminated because the future logits contributed to the denominator that shrank all of them.

Masking with `-inf` before the softmax is exact: `exp(-inf) = 0` contributes nothing to the numerator *and* nothing to the denominator, so the row is a proper distribution over the visible keys only. Mathematically it's `softmax` restricted to a sub-simplex, which is what you meant.

There is a third wrong answer that shows up in real code: "zero after softmax, then renormalize." That gives numerically the same result as `-inf` masking in exact arithmetic — but it costs an extra pass over the `[B,H,T,T]` tensor, it can't be fused into a FlashAttention kernel (the kernel needs the mask *inside* the online softmax), and if all entries in a row are masked you divide 0 by 0 instead of getting a clean, diagnosable NaN from `-inf`.

**🗣 Say this in the room:** "Pre-softmax `-inf` and post-softmax zeroing are not equivalent, because the softmax denominator sums over the masked entries too. Zeroing after leaves each row summing to less than one, by a position-dependent amount — token 0 would come out 1/T scale. You mask the logits, not the weights."

**⚠ Trap:** in HuggingFace-style code you'll see an *additive* mask (`0.0` for keep, large negative for drop) added to the scores rather than a boolean `masked_fill`. Both are correct; mixing them is not. If you add an additive `-inf` mask to scores that were already `masked_fill`-ed you can get `-inf + -inf = -inf` (fine) but `-inf * 0` or `-inf + inf` (NaN) in some kernel paths. Pick one convention per codebase.

### Show me how the mask shapes and broadcasting work when you compose a padding mask with a causal mask.

Two masks with different ranks that must broadcast into the same `[B, H, T_q, T_kv]` score shape. Getting the unsqueezes right is a graded whiteboard skill.

The causal mask depends only on positions: `[T_q, T_kv]`, or `[1, 1, T_q, T_kv]` once you unsqueeze for batch and head. Same for every sequence in the batch, same for every head — that's why frameworks cache one triangular buffer and reuse it.

The padding mask depends only on the batch element and which *key* is real: `[B, T_kv]` → `[B, 1, 1, T_kv]`. Batch dim broadcasts over nothing (each sequence has its own padding), head dim broadcasts (all heads see the same padding), query dim broadcasts (every query ignores the same pad keys).

Compose with a logical AND, since both are "keep" masks:

```python
def build_mask(pad_keep, T_q=None):
    # pad_keep: [B, T_kv] bool, True where the key is a real token
    B, T_kv = pad_keep.shape
    T_q = T_q or T_kv
    causal = torch.ones(T_q, T_kv, dtype=torch.bool, device=pad_keep.device).tril(
        diagonal=T_kv - T_q)                              # offset handles cached decode
    keep = pad_keep[:, None, None, :] & causal[None, None]  # -> [B, 1, T_q, T_kv]
    return keep
```

Two details that earn points. The `diagonal=T_kv - T_q` offset: during cached decode you have `T_q = 1` new query but `T_kv = past + 1` keys, and the single new query is allowed to see *everything* — a plain `tril` would let it see only key 0. Getting this offset wrong is the most common cache bug in hand-rolled decode loops. Second, the leading dim stays `1` rather than being expanded to `H`; expanding to `[B, H, T, T]` materializes a `32 × 2048 × 2048` boolean per batch element (134 MB at `B=1,H=32,T=2048` for bool) that broadcasting would have given you for free.

**⚠ Trap:** applying the padding mask to *queries* as well as keys. It feels symmetric and it is not necessary — pad query rows produce garbage outputs, but those outputs are multiplied by zero in the loss (via `ignore_index`) and never read. Masking pad queries is how you get the all-masked-row NaN, since a pad query with all-pad keys has an entirely `-inf` row. The rule: mask keys always, queries never.

**🗣 Say this in the room:** "Causal is `[1, 1, T_q, T_kv]`, padding is `[B, 1, 1, T_kv]`, AND them, let broadcasting expand to `[B, H, T_q, T_kv]` lazily. And the causal triangle needs a `T_kv − T_q` diagonal offset the moment you have a KV cache, otherwise your single decode query can only see the first token."

### Self-attention versus cross-attention — what actually changes, and where does cross-attention still survive in 2026?

Mechanically, one line changes: where `K` and `V` come from. Self-attention projects `Q, K, V` all from the same `x`. Cross-attention projects `Q` from the decoder stream and `K, V` from a *different* tensor — an encoder output, an image embedding, a retrieved memory. Shapes become `Q: [B, H, T_q, D]`, `K/V: [B, H, T_ctx, D]`, scores `[B, H, T_q, T_ctx]`, and `T_q ≠ T_ctx` is now normal. Causal masking usually disappears on the cross path (the decoder may attend to all of the encoder output; there's no leakage since the encoder input isn't the thing you're predicting), while padding masking on `T_ctx` stays.

The serving property that matters: **cross-attention K/V depend only on the context, not on what you've generated so far.** So you compute them once at the start and reuse them for every decode step — they are a static cache, not a growing one. That's a real efficiency argument, and it's why encoder-decoder never fully died.

Where it survives, weighted toward what you'll actually be asked about:

- **Speech and translation.** Whisper is a genuine encoder-decoder: the audio encoder runs once over the mel spectrogram, the text decoder cross-attends to it every step. T5 and its descendants likewise.
- **VLM connectors.** Two dominant designs. Flamingo-style (**📄 Paper:** Alayrac et al., 2022) interleaves *gated* cross-attention layers into a frozen LLM, so image tokens never enter the text sequence and never consume context; the gate initializes at zero so the pretrained LLM is unchanged at step 0. LLaVA-style (**📄 Paper:** Liu et al., 2023) instead projects vision-encoder patches through an MLP straight into the text embedding space and lets ordinary self-attention handle it. LLaVA-style won on simplicity and on being trainable with far less data; Flamingo-style wins when you cannot afford to spend context on images (a 1024×1024 image can be 1–2k tokens of context you're not getting back).
- **Retrieval-augmented architectures** that attend to retrieved chunks in a separate stream rather than concatenating them into the prompt.
- **Diffusion image models**, where the text conditioning enters U-Net/DiT blocks purely through cross-attention.

**⚠ Trap:** saying "decoder-only models killed cross-attention." They killed the *encoder-decoder text* architecture for general LM, largely because decoder-only scales and pretrains more simply on raw text. But every production VLM and every text-conditioned image model in the world is doing cross-attention or a cheap substitute for it, and "how do image tokens reach the LLM" is a standard question at any company shipping multimodal — which now includes Notion, Figma, Perplexity, and Glean.

**🗣 Say this in the room:** "Cross-attention is the same kernel with `K` and `V` sourced from a different tensor, so `T_q ≠ T_kv` and the cross KV cache is static across decode steps rather than growing. It survives wherever the conditioning signal isn't naturally a prefix of the token stream — audio encoders, gated VLM connectors, diffusion text conditioning."
