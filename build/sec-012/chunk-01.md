### Before we talk about RoPE — convince me a transformer needs positional information at all. Why isn't it already in there?

Because attention is a set operation, not a sequence operation. Strip away the positional encoding and permute the input tokens: every query still dots against every key, the softmax still normalizes over the same multiset of scores, and the output at each slot is the same value it was before, just moved. Formally, self-attention is *permutation-equivariant* — permute the input rows by a permutation matrix P and the output rows come out permuted by exactly the same P. The FFN is applied position-wise so it is equivariant too. LayerNorm is equivariant. Residuals are equivariant. So the entire block is equivariant, and "the dog bit the man" and "the man bit the dog" produce the same bag of output vectors in a different order.

The backend analogue that makes this click: attention is a `GROUP BY` with a similarity-weighted aggregate, and `GROUP BY` does not care what order the rows arrived in. If you want ordering to matter you have to put the order *into the rows*, exactly the way you would add a sequence number column because your aggregate cannot see insertion order.

That leaves exactly three places to inject it. You can add it to the token embeddings before layer 0 (absolute encodings — sinusoidal, learned). You can add it to the attention logits as a bias term (T5 relative buckets, ALiBi). Or you can rotate the query and key vectors themselves so that the dot product becomes a function of relative offset (RoPE). Those are not stylistic choices; they land in different places in the compute graph and therefore have completely different consequences for extrapolation, KV caching, and kernel design. Every argument in this section falls out of *where in the graph* the position lives.

**⚠ Trap:** claiming causal masking already encodes position. It does not. The mask tells token *t* which tokens it may look at — a fixed set — but it says nothing about the *order* of the tokens inside that set. Permute positions 0..t-1 arbitrarily and the causal mask is satisfied identically. (There is a subtle and genuinely interesting caveat here, which is the NoPE result — a causal decoder can *learn* to derive position from the shrinking size of the visible set. But that is an emergent property of training, not something the mask hands you.)

**🗣 Say this in the room:** "Self-attention is permutation-equivariant, so without an explicit position signal the model sees a bag of tokens. The design question is not *whether* to add position but *where in the graph* — into the embeddings, into the logits as a bias, or into Q and K as a rotation — because that choice is what determines whether the model extrapolates past its training length."

### Walk me through sinusoidal positional encoding as the original paper defined it, and tell me what property its designers were actually after.

The construction: for position `pos` and embedding dimension index `i`, you build a vector of length `d_model` where even components are `sin(pos / 10000^(2i/d_model))` and odd components are `cos(pos / 10000^(2i/d_model))`. You add that vector to the token embedding at layer 0 and that is it — no parameters, no training, nothing downstream knows position exists.

The mental model is a binary odometer with continuous wheels. Component pair `i` rotates at angular frequency `1/10000^(2i/d)`. At `i = 0` the wheel turns a full revolution every `2π ≈ 6.28` positions — it distinguishes adjacent tokens sharply but wraps constantly. At the far end, with `d_model = 512`, the slowest wheel has wavelength `2π · 10000 ≈ 62,832` positions — it barely moves across the whole sequence and effectively encodes "roughly where in the document am I." Read all the wheels together and you get a unique fingerprint per position with resolution at every scale, which is precisely what a fixed-width binary counter gives you with 0/1 wheels instead of sinusoidal ones.

The property the authors were after was linear relative addressability: for any fixed offset `k`, `PE(pos+k)` is a *linear* function of `PE(pos)` — specifically a 2×2 rotation by angle `k·ω_i` applied independently to each sin/cos pair. That means a linear attention projection can in principle learn a "look 3 tokens back" operator. This is the direct ancestor of RoPE: RoPE takes the same rotation idea but applies it multiplicatively to Q and K instead of additively to the embeddings, which converts "in principle learnable" into "true by construction."

**📄 Paper:** Vaswani et al. (2017), *Attention Is All You Need* — introduced sinusoidal encodings alongside the transformer, and explicitly reported that learned absolute embeddings performed about the same, which is why the field spent five years treating position as a solved detail.

**⚠ Trap:** repeating the paper's claim that sinusoidal encodings "may allow the model to extrapolate to longer sequences than those encountered during training." That hypothesis did not survive contact with reality. Because the encoding is *added* to the residual stream and then processed by attention weights that were only ever trained on the position vectors seen during training, out-of-distribution position vectors produce out-of-distribution attention patterns. Measured extrapolation for sinusoidal absolute encodings is poor — this is one of the main empirical motivations for ALiBi. Say the hypothesis was stated and then say it was falsified; that distinction is what a senior answer sounds like.

### What's the case for learned absolute position embeddings, and what exactly breaks when you go past the trained length?

The case is brutally simple: it is an `nn.Embedding(max_position_embeddings, d_model)` lookup added to the token embedding, it costs `max_pos × d_model` parameters, and it is free at inference. GPT-2 and BERT both used it. When your maximum sequence length is a hard product constraint (512 for BERT, 1024 for GPT-2) and every training example is shorter than that, a learned table is strictly more expressive than a fixed sinusoid — the model can allocate whatever geometry it wants to position 37 versus position 500.

What breaks past the trained length is not graceful degradation; it is an `IndexError`, and that is the *good* case. Position 1025 has no row in a 1024-row table. There is nothing to extrapolate because there is no function — just a lookup. If you resize the table you get randomly-initialized rows that are uncorrelated with the learned geometry, and generation past the old limit turns into fluent-sounding noise within a few dozen tokens.

There is a second, quieter failure that matters more in practice: even *within* the trained range, learned absolute embeddings are only as good as the position histogram of your training data. If your pretraining corpus packed documents to a fixed 1024 and 90% of your documents were under 300 tokens, positions 700–1023 saw far fewer gradient updates and are systematically undertrained. You will observe quality falling off past ~700 with no error, no warning, and no configuration knob. I have seen teams chase this as a "retrieval bug" for a week.

**🔍 Failure taxonomy — you inherited a model with learned absolute PEs and need more context.** (1) Check `max_position_embeddings` against the checkpoint's actual `wpe`/`position_embeddings` tensor shape — they disagree more often than you would like. (2) Plot the L2 norm of each position embedding row; undertrained tail rows have visibly smaller norms and look like initialization noise. (3) If you must extend, do *not* random-init new rows — interpolate the existing table (upsample it linearly in position space) and then continue-pretrain. (4) The honest answer in 2026 is usually: don't. Swap to a RoPE-based checkpoint. Extending learned absolute position embeddings is throwing continued-pretraining budget at a problem the architecture already solved.

### Explain T5's relative position buckets. Why did that design not become the standard despite being genuinely relative?

T5 removed positional information from the embeddings entirely and instead added a learned scalar bias directly to the pre-softmax attention logits, indexed by the *relative* distance `key_pos − query_pos`. But it does not learn one scalar per distance — that would be unbounded. It bins distances into a fixed number of buckets (32 in the T5 config) that are linear-spaced for small distances and logarithmically-spaced beyond a threshold (128), so distances 0, 1, 2, ... 7 get their own buckets while distances 200 and 250 share one. Each bucket has one learned scalar per head, and the bias is shared across all layers (computed once in the first layer and reused).

The design is elegant for two reasons. It is genuinely translation-invariant — the same relative offset gets the same bias anywhere in the sequence — and the log-bucketing means it degrades gracefully rather than crashing past the training length, since long distances all collapse into the final bucket.

It did not win for reasons that are entirely about kernels and serving. The bias is a `[n_heads, T, T]` tensor that must be materialized and added to the attention logits. That is exactly the tensor FlashAttention exists to *never materialize*. Supporting an arbitrary additive bias inside a tiled, IO-aware attention kernel means either loading a bias tile per block (extra HBM traffic, killing the whole point) or writing a specialized kernel per bias form. RoPE, by contrast, is a pointwise transformation of Q and K *before* the kernel runs — the kernel sees ordinary Q and K and needs no modification at all. Position encodings that touch the logits lost to position encodings that touch the inputs, and the reason is FlashAttention.

**🗣 Say this in the room:** "T5 relative buckets are correct and translation-invariant, but they require an additive `[H, T, T]` logit bias, which is incompatible with the whole premise of FlashAttention. RoPE won because it is a pointwise rotation applied to Q and K outside the kernel, so the attention kernel stays a plain GEMM-softmax-GEMM and you get the memory-IO win for free."

### What is NoPE, and what does it tell us about what decoder-only models learn on their own?

NoPE is the observation that a *causal* decoder-only transformer trained with no positional encoding whatsoever does not collapse — it learns position anyway, and on some length-generalization benchmarks it extrapolates better than models with explicit encodings.

The mechanism is that causal masking breaks permutation equivariance in a way bidirectional attention does not. Token at position 0 can attend to exactly one token. Token at position 5 can attend to six. The *cardinality of the visible set* is a monotone function of absolute position, and the softmax denominator plus the number of nonzero attention weights carry that cardinality signal into the residual stream. The first layer can therefore compute something monotone in position, later layers can sharpen it, and induction-style circuits can build relative offsets from it. Position is not given; it is *derivable*, and gradient descent derives it.

Why does this matter for you in an interview? Two reasons. First, it is the clean disproof of "the mask encodes position" — the mask does not encode position, but it makes position *learnable*, and those are different claims. Second, it reframes what an explicit positional encoding is actually buying: not the ability to know position, but a strong inductive bias that makes short-range relative structure cheap to represent, so that pretraining does not have to spend capacity rediscovering it. That is why NoPE is interesting research and not a production architecture — it works, it generalizes in length, and it is worse at the same parameter count on ordinary language modeling.

**⚠ Trap:** generalizing NoPE to encoder models or to embedding models. Bidirectional attention has no causal mask, so there is no cardinality signal, so NoPE genuinely is a bag of words there. If someone proposes dropping positional encodings from a bidirectional retrieval encoder, that is a hard no.

### Derive RoPE for me. Start from what you want to be true and get to the rotation.

Start from the property you want and let the construction fall out. In attention, position only ever enters through the score `q_m · k_n`. So the *only* thing you actually need is: the score should depend on `m` and `n` only through `m − n`. Write that as a requirement — find functions `f_q(x, m)` and `f_k(x, n)` such that `⟨f_q(x_q, m), f_k(x_k, n)⟩ = g(x_q, x_k, m − n)` for some `g`.

Now recall the one operation whose inner products depend only on angle differences: rotation. In 2D, if `R(θ)` is a rotation matrix, then `⟨R(α)u, R(β)v⟩ = u^T R(α)^T R(β) v = u^T R(β − α) v`, because rotations are orthogonal (`R(α)^T = R(−α)`) and compose additively. The `α` and `β` cancel and only `β − α` survives. That is exactly the property, handed to you for free by the group structure of SO(2).

So: split the `d_head`-dimensional query and key vectors into `d_head/2` two-dimensional subspaces. Rotate subspace `i` of the vector at position `m` by angle `m · θ_i`, where `θ_i = base^(−2i/d_head)`. Then

```
⟨RoPE(q, m), RoPE(k, n)⟩ = Σ_i ⟨R(m·θ_i) q_i, R(n·θ_i) k_i⟩ = Σ_i q_i^T R((n − m)·θ_i) k_i
```

which depends on `m` and `n` only through `n − m`. Relative by construction — not learned, not approximated, an algebraic identity.

Two design consequences fall straight out. First, RoPE has **zero parameters** and does not touch the residual stream; it is applied to Q and K only, immediately before the attention score, and never to V. Second, because `θ_i` decays geometrically with `i`, the subspaces span a spectrum of wavelengths: high-index-frequency pairs rotate fast (fine positional resolution, wraps quickly) and low-frequency pairs rotate slowly (coarse "where in the document" signal that never wraps within the trained length). That spectrum is the entire substrate on which every context-extension method operates — PI, NTK, YaRN and LongRoPE are all different ways of saying "which of these wheels do I slow down."

**📄 Paper:** Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* — replaced additive absolute encodings with a multiplicative rotation on Q and K, giving exact relative dependence without materializing any bias tensor. It is now the default in essentially every open-weight decoder-only family.

### Implement RoPE from memory. I want the frequency construction, the cache-aware position offset, and I want you to tell me about the layout convention.

```python
import torch

def rope_inv_freq(d_head: int, base: float = 10000.0, device="cpu"):
    # one frequency per 2D subspace -> d_head/2 of them
    i = torch.arange(0, d_head, 2, device=device, dtype=torch.float32)
    return 1.0 / (base ** (i / d_head))                       # [d_head/2]

def build_cos_sin(positions: torch.Tensor, d_head: int, base: float = 10000.0):
    """positions: [T] int64 absolute positions. Returns cos, sin of shape [T, d_head]."""
    inv = rope_inv_freq(d_head, base, positions.device)        # [d/2]
    ang = positions.float()[:, None] * inv[None, :]            # [T, d/2]
    ang = torch.cat([ang, ang], dim=-1)                        # [T, d]  half-split layout
    return ang.cos(), ang.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, d_head]; cos/sin: [T, d_head] -> broadcast over B, H."""
    return x * cos[None, None] + rotate_half(x) * sin[None, None]

# decode step: positions must be the ABSOLUTE positions, not 0..T-1
past_len = k_cache.shape[2]
pos = torch.arange(past_len, past_len + q.shape[2], device=q.device)
cos, sin = build_cos_sin(pos, d_head, base=500000.0)
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

Three things to say out loud while you write it. **V is never rotated** — only Q and K, because position only needs to affect the score, and rotating V would corrupt the content you are aggregating. **Positions are absolute offsets into the sequence, not indices into the current chunk** — during decode with a KV cache your query is at position `past_len`, and if you pass `0` there (which is what `torch.arange(q.shape[2])` gives you at `q.shape[2] == 1`) every generated token believes it is at position 0. The generation stays grammatical for a while and then degenerates into loops, because relative offsets to all cached keys are wrong by a growing amount. That is the single most common RoPE bug in hand-rolled inference code and it produces no exception. **Cos/sin should be precomputed once** and sliced, not rebuilt per step; it is cheap but it is on the critical path of every decode token.

**⚠ Trap — the pairing convention.** There are two incompatible ways to form the 2D subspaces. The *interleaved* convention pairs `(x[0], x[1]), (x[2], x[3]), ...` — this is what the original RoFormer and Meta's reference Llama code do, expressed as complex multiplication. The *half-split* convention pairs `(x[0], x[d/2]), (x[1], x[d/2+1]), ...` — this is `rotate_half`, what GPT-NeoX and Hugging Face's Llama implementation do, because it vectorizes into a single `cat` with no strided gather. **The two are equivalent only if you permute the Q and K projection weight matrices accordingly** — which is exactly what Hugging Face's Llama conversion script does with its `permute` helper. If you write your own loader and skip that permutation, the model loads with no error, produces plausible-looking logits, and is quietly wrong. Perplexity roughly doubles and short prompts still look fine. Name this trap; it is a real-world weight-conversion incident that has bitten multiple teams.

**🏋 Drill:** 20 minutes, no references, no autocomplete. Write `build_cos_sin`, `rotate_half`, `apply_rope`, and a `decode_step` that takes a KV cache and appends one token with correct absolute positions. Pass criterion: `apply_rope(q, *build_cos_sin(pos_m, d))` dotted with `apply_rope(k, *build_cos_sin(pos_n, d))` must equal the same dot product computed at positions `pos_m + c, pos_n + c` for any integer shift `c`, to within 1e-4 in fp32. If your shift-invariance test does not pass, your pairing convention is inconsistent between Q and K.

### What does the RoPE base θ actually control, and why did Llama-3 move from 10,000 to 500,000?

The base sets the *wavelength spectrum* of the rotation wheels, and the wavelength of the slowest wheel is the model's positional dynamic range. Frequency `θ_i = base^(−2i/d_head)`, so wavelength `λ_i = 2π · base^(2i/d_head)`. The fastest wheel (`i = 0`) always has wavelength `2π ≈ 6.3` tokens regardless of base. The slowest wheel (`i = d_head/2 − 1`) has wavelength approximately `2π · base`.

Run the numbers for `d_head = 128`. At base 10,000, the slowest wavelength is `2π · 10000^(63/64) = 2π · e^(0.984 · 9.2103) = 2π · 8,630 ≈ 54,200 tokens`. At base 500,000 it is `2π · 500000^(63/64) = 2π · e^(0.984 · 13.1224) = 2π · 405,600 ≈ 2,549,000 tokens`. So Llama-3's base change stretched the longest wavelength by ~47×, from about 54k tokens to about 2.5M.

Why that matters: a dimension whose wavelength is shorter than your context length *wraps* — position 100 and position 100 + λ are rotationally indistinguishable in that subspace, so the model cannot use it to disambiguate long-range order. At base 10,000 and 8k context, no dimension wraps (54,200 > 8,192), so the encoding is fine — but only about the last few dimensions have wavelengths comfortably longer than the context, meaning very few "coarse" wheels are available to represent document-scale position. Raising the base rebalances the spectrum toward long wavelengths, giving the model far more headroom for coarse positional structure at 128k, at the cost of slightly coarser fine-grained resolution in the mid-band.

**📐 Numbers you must know:** longest RoPE wavelength ≈ `2π · base`. Base 10,000 → ~63k tokens; base 500,000 → ~3.1M tokens (the `2π·base^(63/64)` correction pulls these to ~54k and ~2.5M for `d_head=128`). The memorizable rule: **you want your context length to sit well inside the longest wavelength, ideally by a factor of 10 or more**, or the low-frequency dimensions start aliasing.

**⚠ Trap:** treating "raise the base" as a free context extension you can apply post-hoc to a trained checkpoint. Changing the base changes *every* rotation angle at *every* position, including positions the model already handled fine. The model's attention heads were trained against a specific angular geometry; rewriting it wholesale is a distribution shift on every head. Base scaling ("ABF", adjusted base frequency) works as a context-extension technique **only with continued pretraining afterward** — Meta raised the base *during* training, not after. Doing it zero-shot on a Llama-2 checkpoint measurably degrades short-context quality, which is the exact failure mode that motivated the NTK-aware and YaRN "don't touch the high frequencies" designs.

### Where in the compute graph is RoPE applied relative to the KV cache, and what does that imply for prefix caching?

RoPE is applied to K *before* it is written to the cache in every mainstream implementation. The cached tensor is post-rotation `K_rot`, not raw `K`. This is not arbitrary — if you cached raw K you would have to re-rotate the entire cache on every decode step, turning an O(1) pointwise op into O(T) work per token. Caching post-rotation makes the cache immutable: key at position 42 is rotated by `42·θ_i` once, forever.

The implication that interviewers actually probe: **a cached KV block is bound to the absolute position it was computed at.** Prefix caching in vLLM or SGLang keys blocks on the hash of the token IDs *and their position prefix*, precisely because block content is position-dependent. You cannot take a cached block computed at positions 0–15 and reuse it at positions 512–527 — the rotations are wrong. This is why prefix caching only ever gives you a *prefix* hit: the shared span must start at the same position in both requests. If your system prompt is identical but one tenant prepends a 40-token per-user preamble, the cache hit rate for the shared system prompt is zero, not partial.

That single fact is worth internalizing because it is the design constraint behind "put the stable stuff first." Ordering your prompt as `[static system] [static tool schemas] [retrieved docs] [conversation] [user turn]` is not stylistic; it is the only ordering under which RoPE-bound cache blocks are reusable across requests.

**💰 Math:** take a 12,000-token system-prompt-plus-tool-schema block at $3.00 per million input tokens, with a 90% discount on cached input (**📅 Volatile — see §5:** cache discount and price both move; verify before your loop). Uncached: 12,000 × $3.00/1e6 = **$0.036 per call**. Cached: **$0.0036**. At 200,000 calls/day the difference is 200,000 × $0.0324 = **$6,480/day**, or about **$194,000/month**. Now put a 40-token per-user preamble in front of it and every one of those hits becomes a miss. That is a six-figure monthly line item destroyed by prompt ordering, and it is caused by RoPE binding keys to absolute positions.

**⚠ Trap:** the "attention sink" variants (StreamingLLM and its descendants) *do* re-index positions when evicting, and some long-context serving tricks re-rotate cached keys. If you assert "the cache is always immutable" as a universal law you will be corrected. The correct framing is: the cache is immutable *given a fixed position assignment*, and any scheme that changes a token's assigned position must recompute or re-rotate its keys.

### Explain ALiBi. How does it get extrapolation, and what does it do to your KV cache?

ALiBi throws out positional encodings entirely and instead adds a *static, non-learned* linear penalty to the attention logits: score becomes `q·k/√d − m_h · |i − j|`, where `i − j` is the distance from query to key and `m_h` is a per-head slope. The slopes form a geometric sequence — for `n` heads, head `h` gets slope `2^(−8h/n)` — so with 8 heads you get slopes from 1/2 down to 1/256. Steep-slope heads see only a few tokens back; shallow-slope heads see far. The mechanism is a *recency prior with a per-head length scale*, hard-coded.

The extrapolation story is genuine and mechanically clear: the penalty is a linear function of distance, defined for every integer distance, with no wraparound, no wavelength, and no trained lookup. Feed it distance 50,000 when you trained on 1,024 and it returns a perfectly well-defined large negative number. The paper's result — train on 1,024, evaluate at 2,048+ with *lower* perplexity than a sinusoidal model trained at 2,048 — is real.

But look at what extrapolation *means* here. A large negative bias at large distance means the model effectively cannot attend far. ALiBi extrapolates by having each head degenerate into a soft sliding window whose width is set by its slope. That is fine for perplexity, which is dominated by local statistics, and it is bad for long-range retrieval, which is exactly what people buy long context for. This is the honest reason ALiBi did not become the standard for 128k models: it is excellent at *not breaking* and poor at *actually using* distant tokens.

For serving, ALiBi has one real advantage and one real cost. Advantage: keys are position-free, so a cached K vector is genuinely position-independent and can be reused at any offset — prefix caching is not position-bound the way it is under RoPE. Cost: the bias is a per-head function of `(i, j)`, so like T5 it wants to touch the attention logits. FlashAttention added explicit ALiBi support because the bias is cheap to compute on the fly from `(i, j)` inside a tile — you never materialize it — but it is a kernel feature that had to be built, and any exotic kernel you want to use must support it.

**📄 Paper:** Press, Smith & Lewis (2022), *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* — replaced positional embeddings with a fixed per-head linear distance penalty and demonstrated length extrapolation without any position vectors.

### You're designing a new open-weight model targeting 256k context. RoPE, ALiBi, or something else — pick one and defend it.

RoPE with a large base, trained with an explicit long-context curriculum, and I would not consider ALiBi seriously. Here is the argument I would make in the room.

The decision hinges on what you want the model to *do* at 256k, not on what keeps perplexity low at 256k. Those are different objectives and ALiBi optimizes the wrong one. ALiBi's slopes impose a monotone recency prior per head; a token 200,000 positions back is penalized by `m_h × 200,000`, which for even the shallowest head (slope 1/256 at 8 heads) is a logit penalty of ~780. Nothing survives that. So an ALiBi model at 256k is, functionally, a sliding-window model with a soft edge — it will have decent perplexity and it will fail every needle-in-a-haystack and multi-hop retrieval eval you run. If your product is "summarize this 400-page contract and cite the clause on page 312," ALiBi cannot do it.

RoPE has the opposite profile: no built-in recency prior at all, so long-range attention is *possible*, but the model must learn it and the encoding must be scaled to the target length rather than extrapolated. That is a training-cost problem, and training-cost problems are solvable with money. Recency-prior problems are architectural and are not.

The secondary arguments seal it. RoPE requires no kernel support — it is a pointwise op on Q and K before the attention call, so every kernel in the ecosystem (FlashAttention, FlashInfer, Triton hand-rolls, vLLM's paged kernels, the Metal and CUDA graph paths) works unmodified. ALiBi requires bias support in every kernel you ever want to use, which is a permanent tax on your ability to adopt new attention implementations. And the entire context-extension toolchain — PI, NTK-aware, YaRN, LongRoPE — exists for RoPE and does not exist for ALiBi, so choosing ALiBi means opting out of five years of community tooling.

The one thing I would add on top of RoPE: an interleaved local:global attention pattern rather than every layer being global, because at 256k the KV cache, not the positional encoding, is the binding constraint.

**🗣 Say this in the room:** "RoPE, base scaled to the target length, with a staged length curriculum. ALiBi extrapolates by suppressing distant attention, which is exactly the capability a 256k model is being bought for — it wins on perplexity and loses on retrieval. And RoPE is a pointwise op on Q and K, so I keep kernel compatibility with the entire FlashAttention ecosystem."

### A colleague changed `max_position_embeddings` in `config.json` from 8192 to 131072 on a Llama-2 checkpoint and says long context now works. Tell me what actually happens.

Nothing good, and — critically — nothing that raises an exception. `max_position_embeddings` on a RoPE model is not a capability declaration; it is mostly a bookkeeping field that tells the tokenizer/generation stack how long a sequence it is willing to build and, in some implementations, how big a cos/sin table to precompute. RoPE itself is a closed-form function of position. There is no table with 8,192 rows to overflow. Position 100,000 has a perfectly well-defined rotation. So the code runs, the model generates, and the output past roughly the trained length is fluent garbage.

The mechanism of the failure is out-of-distribution rotation angles. During training on 4k–8k sequences, the low-frequency dimensions of Llama-2 (base 10,000) only ever swept a small arc — the slowest wheel has wavelength ~54k tokens, so at 8k it rotated through `8192/54200 ≈ 15%` of a revolution. The attention heads learned to interpret angular differences within that 15% arc. At position 100,000 that wheel has gone around nearly twice and is presenting angles the model has literally never seen, in a regime where the *mapping from angle to relative distance is ambiguous* because of wraparound. Meanwhile the high-frequency dimensions were already wrapping constantly and are fine.

The symptoms are diagnostic and worth memorizing: coherent for the first few thousand tokens, then attention entropy collapses or the model starts ignoring the prompt, repetition loops, and the perplexity curve versus position shows a knee right around the original training length followed by a steep climb. If you plot per-position NLL you will see the knee at ~4k–8k precisely.

**⚠ Trap:** the reviewer version of this. This change is a **one-line diff in a JSON file** with no test that fails. Nothing in CI catches it. The unit tests pass, the smoke test prompt is 200 tokens and works fine, and the regression shows up as "users say it hallucinates on long documents" three weeks later. The rule I enforce in review: **any change to `max_position_embeddings`, `rope_theta`, or `rope_scaling` requires an accompanying per-position perplexity curve out to the new claimed length, on real documents, attached to the PR.** No curve, no merge. It is a config file; treat it as a model change.

**🗣 Say this in the room:** "`max_position_embeddings` doesn't gate RoPE — RoPE is closed-form, so it will happily give you position 131072. What you get is out-of-distribution rotation angles in the low-frequency dimensions, which is fluent nonsense past the trained length with no error. Extending context needs `rope_scaling` plus continued pretraining, and I'd want a per-position NLL curve as the acceptance test."
