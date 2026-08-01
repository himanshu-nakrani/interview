### Explain multi-head latent attention from first principles. And tell me why it isn't just "LoRA on the KV projection."

Mental model: MQA and GQA shrink the cache by **removing heads**. MLA shrinks the cache by **changing the basis you cache in**. The observation is that across the `n_heads · d_head`-dimensional concatenation of all key and value vectors at a position, the information is highly redundant — the true rank is far below the ambient dimension. So instead of caching K and V, cache a single low-rank *latent* vector per token per layer, and reconstruct per-head K and V from it on demand. Every head keeps its own full-rank key and value subspace; you just don't store them.

Mechanism, using DeepSeek-V3's published config so the numbers are concrete (61 layers, `d_model` 7168, 128 heads, `d_head` 128, `kv_lora_rank` 512, `qk_rope_head_dim` 64):

```
c_kv[t] = W_DKV @ h[t]                  # [7168] -> [512]      <-- THIS is cached
k_C[t,i] = W_UK[i] @ c_kv[t]            # [512] -> [128] per head i
v_C[t,i] = W_UV[i] @ c_kv[t]            # [512] -> [128] per head i
k_R[t]   = RoPE(W_KR @ h[t])            # [7168] -> [64], shared across heads, also cached
key[t,i] = concat(k_C[t,i], k_R[t])     # [192] per head
```

Queries are compressed too (`q_lora_rank` 1536 in V3), but query compression is a *parameter* saving, not a cache saving, since queries are never cached.

Now, why it is not LoRA. LoRA factorizes a weight matrix `W ≈ W₀ + BA` to reduce the number of *trainable parameters* during adaptation; the forward pass still produces the same full-width activation and you still cache full-width K and V. MLA factorizes the *activation path* and then **caches the bottleneck**. The low-rank structure is trained from scratch as part of the architecture, and the payoff is measured in cache bytes, not trainable params. A second structural difference: MLA's up-projections `W_UK` and `W_UV` are per-head, so each head reads a different linear readout of the same shared latent — the heads retain independent key subspaces, which is exactly what MQA destroys. That is the crux of MLA's quality claim: it gets GQA-beating cache sizes *without* forcing heads to share a key space.

**📄 Paper:** DeepSeek-AI (2024), *DeepSeek-V2*, introduced MLA — low-rank joint compression of keys and values into a cached latent with a decoupled RoPE component — replacing the "reduce head count" family of cache reductions with a "reduce rank" one. Carried forward unchanged into DeepSeek-V3.

**🗣 Say this in the room:** "MLA caches a 512-dim latent per token per layer instead of caching K and V, and reconstructs per-head K and V from it with per-head up-projections. Unlike MQA, every head keeps its own key subspace — you're compressing the rank, not deleting heads."

### Why does MLA need a separate "decoupled" RoPE dimension? What breaks if you just apply RoPE to the decompressed keys?

This is the question that separates people who read the DeepSeek paper from people who read a summary of it, and the answer is a commutativity argument.

MLA's entire decode-time efficiency depends on **never materializing the per-head keys**. The attention logit is `q_i · k_i = (W_UQ[i] c_q)ᵀ (W_UK[i] c_kv) = c_qᵀ (W_UQ[i]ᵀ W_UK[i]) c_kv`. The bracketed product is a fixed matrix you can precompute once at load time. So at decode you dot the query latent against the cached KV latent directly — the 512-dim latent *is* the key, in an absorbed basis. Beautiful.

Now insert RoPE. RoPE applies a position-dependent rotation `R_t` to the key at position t: `k_i = R_t W_UK[i] c_kv[t]`. The logit becomes `c_qᵀ W_UQ[i]ᵀ R_sᵀ R_t W_UK[i] c_kv[t]`, and `R_sᵀR_t = R_{t−s}` depends on the *query's* position s as well as the key's. That means the matrix sandwiched between the two latents is `W_UQ[i]ᵀ R_{t−s} W_UK[i]` — **a different matrix for every (query, key) position pair**. You cannot precompute it. You are forced to decompress `c_kv[t]` into full per-head keys for every cached token, at every decode step, for every head. That is strictly worse than GQA: you pay MLA's decompression FLOPs *and* end up with MHA-sized intermediates.

DeepSeek's fix is to split the key into two parts. A **compressed, position-free** part `k_C` of dimension 128 per head, derived from the latent, which absorbs cleanly. And a **decoupled RoPE** part `k_R` of dimension 64, produced by a separate small projection directly from the hidden state, rotated by RoPE, **shared across all heads**, and cached as-is. The per-head key is the concatenation, dimension 192. The logit splits into two terms: the absorbed latent term (no RoPE, precomputable matrix) plus a small 64-dim rotary term computed the ordinary way. Positional information lives entirely in the cheap 64 dims; content lives in the compressible 128 dims.

So the cached state per token per layer is `512 (latent) + 64 (rope key) = 576 elements` — not `2 × 576`, because the latent is *joint* for K and V and the rope key has no value counterpart.

**⚠ Trap:** describing MLA's cache as `2 · n_layers · d_c · bytes` by reflex, applying the MHA formula's factor of 2. There is no factor of 2 in MLA. K and V share one latent. If you write `2 × 61 × 512` you will overstate DeepSeek-V3's cache by roughly 1.8× and the interviewer will know exactly which mistake you made.

**🗣 Say this in the room:** "RoPE's rotation depends on the relative position, so it sits between the query up-projection and the key up-projection and blocks matrix absorption. DeepSeek decouples it: a shared 64-dim rotary key carries position, and the 512-dim latent carries content and stays absorbable. 576 elements cached per token per layer, total."

### What does "matrix absorption" mean in MLA, and why do you only do it at decode?

Absorption is the algebraic identity I sketched above, promoted to an implementation strategy: because `W_UQ[i]ᵀ W_UK[i]` is a product of two fixed matrices, fold them into one at load time and attend in latent space. Likewise on the output side: `out = W_O · concat_i(attn_i · W_UV[i] c_kv)`, and since `W_UV[i]` and the relevant slice of `W_O` are both fixed, precompute `W_O[i] W_UV[i]` and apply attention weights directly to the *latent* values, then project once. Net effect at decode: you never build a `[T, n_heads, 128]` key tensor or value tensor at all. You read `[T, 576]` from cache and do two small matmuls.

Why only at decode: the two regimes have opposite arithmetic. At **prefill**, T is large (say 4096) and you are compute-bound. Absorbed form requires a `[T, 512] × [512, 512]`-ish contraction per head — the absorbed matrix is `d_c × d_c`-shaped per head, so you do 128 heads × 512 × 512 work per query token, which is *more* FLOPs than decompressing once into `[T, 128, 128]` keys and running a standard FlashAttention over them. Decompression amortizes across all T queries. So at prefill you materialize; the extra memory is transient activation memory, not cache, and FlashAttention handles it.

At **decode**, T_query = 1. There is nothing to amortize the decompression over — you would decompress `T_kv` cached tokens to serve a single query. Absorbed form reads 576 elements per cached token instead of reconstructing 128 heads × 192 dims = 24,576 elements. That is a 42× reduction in the bytes flowing through the attention kernel. Since decode is bandwidth-bound, that ratio translates almost directly into throughput.

So a correct MLA implementation is **two code paths sharing one set of weights**: materializing form for prefill and chunked prefill, absorbed form for decode. This is not a nicety — it is why naive MLA implementations were initially *slower* than GQA, and why DeepSeek shipping FlashMLA (an open-source Hopper decode kernel for paged MLA) mattered so much for adoption. An architecture that needs a bespoke kernel to beat the baseline has a real adoption tax.

**⚠ Trap:** benchmarking MLA with a single generic implementation and concluding the architecture is bad. If your MLA is slower than GQA at decode, the first thing to check is whether you are running the absorbed path or accidentally decompressing. This is the MLA equivalent of the `repeat_kv` mistake in GQA.

### Compute DeepSeek-V3's KV cache per token and compare it to a GQA model of similar activated size.

DeepSeek-V3: 61 layers, cached state per token per layer = 512 (KV latent) + 64 (decoupled RoPE key) = **576 elements**. In bf16: 576 × 2 = 1152 bytes per layer. × 61 layers = **70,272 bytes ≈ 68.6 KiB per token**.

Now three comparisons at the same layer count and head geometry (61 layers, 128 heads, `d_head` 128):

- **MHA equivalent:** 2 × 61 × 128 × 128 × 2 = 3,997,696 B = **3.81 MiB/token**. MLA is **56.9× smaller**.
- **GQA-8 equivalent:** 2 × 61 × 8 × 128 × 2 = 249,856 B = **244 KiB/token**. MLA is **3.56× smaller**.
- **Restated as an equivalent head count:** MLA's 576 elements per layer equals `2 · n_kv · 128` at `n_kv = 2.25`. **MLA-576 is "GQA-2.25" on cache, with better-than-GQA-8 quality.** That framing is the cleanest way to explain the win.

At context lengths this gets serious. At 128k tokens: 70,272 × 131,072 = 9.21 GB = **8.58 GiB per sequence**. The GQA-8 equivalent would be 249,856 × 131,072 = 32.75 GB = **30.5 GiB per sequence**, and MHA would be 524 GB — more than an 8×H100 node, for one user.

**💰 Math:** concurrency on an 8×H200 node (8 × 141 GB = 1128 GB). DeepSeek-V3 weights at fp8 ≈ 671e9 × 1 = 671 GB. Usable KV pool ≈ 1128 × 0.9 − 671 − ~25 = **~319 GB**. At 32k context, MLA cache per sequence = 70,272 × 32,768 = 2.30 GB, giving **319/2.30 ≈ 138 concurrent sequences**. Under a GQA-8 attention block with the same layer count, per-sequence cache would be 249,856 × 32,768 = 8.19 GB → **39 sequences**. Same hardware, same weights, **3.5× the concurrent users**. If your node costs ~$30/hr, cost per session-hour drops from $0.77 to $0.22. (**📅 Volatile:** GPU pricing and the exact V3 config — verify against the published `config.json` before quoting.)

**🗣 Say this in the room:** "DeepSeek-V3 caches 576 elements per token per layer — 512 latent plus 64 rotary — which is 68.6 KiB per token across 61 layers. That's 3.6× smaller than a GQA-8 block of the same geometry and 57× smaller than MHA. In cache terms MLA-576 behaves like GQA with 2.25 KV heads."

### MLA beats GQA on cache at equal or better quality. What does it cost you? Nothing is free.

Four costs, in the order I would raise them in a design review.

**Compute at prefill.** MLA's per-token attention FLOPs are higher than GQA's, because you pay the down-projection, and in materializing form you pay the up-projections too. In DeepSeek-V3, the query path alone carries a `7168 → 1536 → 128·192` chain that GQA does not have. On a prefill-heavy workload — think document ingestion, RAG with 30k-token contexts and short answers — you have traded a resource you had (HBM) for a resource you were already short of (FLOPs). MLA is optimized for the decode-heavy, long-conversation regime.

**Kernel complexity and ecosystem lag.** GQA is a two-line change to FlashAttention. MLA needs a bespoke paged decode kernel operating on a latent layout, plus a separate prefill path, plus correct handling of the concatenated 192-dim key with only the last 64 dims rotated. Every serving framework had to implement this from scratch; support arrived months after the models did, and remains less mature than the GQA path. If you are a platform team betting on a model family, "will my engine of choice have a good kernel for this in six months" is a real risk, not a theoretical one.

**Weight memory.** MLA adds projection matrices — `W_DKV`, `W_UK`, `W_UV`, `W_DQ`, `W_UQ`, `W_KR`, `W_QR`. In a dense model you would notice; in a 671B MoE where the FFN dominates, it disappears into the noise. **MLA is cheapest to adopt in exactly the regime DeepSeek was in.**

**Quantization headroom.** This is the one people miss. GQA's cache is raw K and V activations with substantial redundancy — that redundancy is what fp8 and int4 KV quantization exploit. MLA's cache is a learned low-rank latent that has already had the redundancy squeezed out. My prior — and I would state it as a prior, not a measured fact — is that MLA tolerates aggressive KV quantization *worse* than GQA, because you are quantizing an already-compressed representation. So the gap between "MLA at bf16" and "GQA-8 at fp8" is narrower than the raw 3.6× suggests: GQA-8 at fp8 is 122 KiB/token versus MLA at bf16's 68.6 KiB/token, only a 1.8× gap.

**⚠ Trap:** comparing architectures at different numeric precisions and calling it an architecture result. Always normalize: quote KV bytes per token *at a stated dtype*, for both sides.

### If MLA dominates GQA, why haven't Llama, Qwen and Mistral adopted it? Is this settled?

It is not settled, and I would say so plainly rather than pretend there is consensus. Here is the honest decision structure.

The case that MLA is genuinely better: it achieves smaller cache than GQA at comparable or better reported quality, it composes well with MoE (where the FFN dominates parameters so the extra attention weights are free), and it scales especially well to long context where the cache dominates HBM. DeepSeek shipped it in two flagship models and then open-sourced the decode kernel, which is a strong signal of production conviction rather than paper conviction.

The case for staying on GQA: **switching costs are enormous and they are not the model builder's costs, they are the ecosystem's.** GQA is a config-file change that every inference engine, every quantization toolkit, every fine-tuning library, every mobile runtime and every custom kernel already supports. MLA requires new kernels in each of them. If you are Meta shipping open weights whose value proposition is "runs everywhere on day one," an architecture that runs well in two engines is a strategic mistake even if it is technically superior. There is also a genuine open question about MLA's behavior under KV quantization and under aggressive sparse-attention schemes, both of which are active areas — betting the flagship on it carries research risk.

Third factor: the frontier is simultaneously pushing on **sparse and hybrid attention** — sliding-window/global interleaving, native sparse attention, SSM hybrids — which attack the same problem from a different direction and are largely orthogonal to whether the dense part is GQA or MLA. A team might reasonably conclude that the bigger win is `O(T)` cache growth becoming `O(1)` for most layers, and that spending architecture-risk budget there beats spending it on a 3.6× constant factor.

**🗣 Say this in the room:** "MLA is technically ahead of GQA on the cache-versus-quality frontier, and DeepSeek has shipped it twice. It hasn't spread because it needs bespoke kernels and GQA needs none — that's an ecosystem cost, not a technical rebuttal. My decision rule: if I control the serving stack end to end and my workload is long-context decode-heavy, MLA. If I'm shipping open weights that need to run in ten engines on day one, GQA."

### Sliding-window attention — what does it actually do to the KV cache, and what does it not do?

The mental model: sliding window changes the cache from a **log** to a **ring buffer**. Every variant so far reduced bytes *per token*; sliding window bounds the *number of tokens* you keep. Those are multiplicative, and they fail in different ways.

Mechanism: with window W, the attention mask at position t allows attending to positions `[t−W+1, t]` only. Any key/value older than W positions is provably never read again by that layer, so it can be freed. The cache becomes a fixed-size rolling buffer of W slots per layer per sequence, with position `t` writing to slot `t mod W`. Cache memory becomes **independent of sequence length**: `2 · n_layers · n_kv · d_head · bytes · min(T, W)`.

Concretely for Mistral-7B v0.1 (32 layers, 8 KV heads, `d_head` 128, W = 4096): cache per token is 128 KiB, so a full window is 128 KiB × 4096 = **512 MiB**, flat, whether the user is at 4k or 128k tokens. Without the window, 32k tokens would be 4 GiB and 128k would be 16 GiB. (**📅 Volatile:** later Mistral-7B point releases removed the sliding window — read the `sliding_window` field in `config.json` rather than assuming.)

Now the three things sliding window does **not** do, which is where the interview actually goes:

1. **It does not reduce prefill compute below O(T·W).** You still process all T tokens; you just mask more of the score matrix. It reduces prefill from `O(T²)` to `O(T·W)`, which is a real win at long T, but it is not free.
2. **It does not give you unlimited effective context.** Information from beyond the window reaches later positions only by propagating up through layers — each layer moves information at most W positions forward, so with L layers the theoretical receptive field is `L·W`. In practice that indirect path is lossy, and exact recall of a specific fact from 40k tokens ago through a 4k window degrades badly. This is the failure mode: the model *seems* to have long context and quietly cannot retrieve from it.
3. **It does not compose trivially with prefix caching.** More on that shortly, but the short version: a cached prefix under SWA is only reusable if the window alignment matches, and evicted-by-window blocks are gone whether or not the prefix cache wanted them.

**⚠ Trap:** evicting position 0. Decoder-only transformers develop **attention sinks** — token 0 absorbs enormous probability mass because softmax must sum to one and heads need somewhere to dump attention when nothing is relevant. A naive rolling buffer that drops token 0 causes the softmax denominator to redistribute, logits to blow up, and perplexity to spike sharply — a very recognizable failure where output stays fluent for a while and then degenerates. StreamingLLM's fix is to pin the first few tokens permanently and slide the rest.

**📄 Paper:** Xiao et al. (2023), *Efficient Streaming Language Models with Attention Sinks* — identified that retaining a handful of initial tokens alongside a sliding window preserves quality indefinitely, replacing naive window eviction which collapses.

### Gemma uses interleaved local and global layers. Do the arithmetic on the cache savings at 128k.

The design: rather than making every layer sliding-window (which caps effective context) or every layer global (which caps concurrency), interleave them. Most layers are local with a small window; a minority are global with full attention. The global layers provide the long-range path; the local layers provide cheap depth. Gemma-2 used a 1:1 alternation with a 4096-token window; Gemma-3 moved to roughly **5 local : 1 global with a 1024-token window** (**📅 Volatile:** verify the ratio and window in the released config before quoting — these have changed between generations).

The arithmetic, generalized. Let `r` = fraction of layers that are global, `W` = local window, `T` = context. Effective cached tokens averaged over layers:

```
tokens_eff = r · T + (1 − r) · min(T, W)
```

At r = 1/6, W = 1024, T = 131,072:

```
tokens_eff = (1/6)(131,072) + (5/6)(1,024)
           = 21,845.3 + 853.3
           = 22,698.7
reduction  = 131,072 / 22,698.7 = 5.77×
```

So a 5.8× cache reduction at 128k, on top of whatever GQA gives you per token. Note how the ratio behaves: at short context the saving vanishes (at T = 1024 everything is within the window, so `tokens_eff = T`), and asymptotically as T → ∞ the saving converges to `1/r` = 6×. **The interleaving buys you a bounded constant factor, and you get essentially all of it once T ≫ W/r.** That asymptote is the number to remember: the maximum possible saving from an interleaved pattern is exactly the reciprocal of the global-layer fraction.

**💰 Math:** suppose a 27B-class model with 46 layers, 16 KV heads, `d_head` 128, bf16. All-global cache per token = 2 × 46 × 16 × 128 × 2 = 376,832 B = 368 KiB. At 128k that is 46 GiB per sequence — unservable. With 1:5 interleaving the effective figure is 46 GiB / 5.77 = **7.97 GiB**, which fits eight concurrent users into 64 GiB. That is the difference between "we support 128k" being true and being marketing.

**⚠ Trap:** assuming the global layers are cheap because they are few. At 128k, the global layers hold `(1/6) × 131,072 = 21,845` tokens' worth of cache, which is 96% of the total cache in the interleaved model. **Your entire memory profile is set by the minority of layers.** If someone proposes going from 1:5 to 1:3 for quality, that is a 1.6× cache increase, not a small tweak.

### Explain cross-layer KV sharing. Where would you place the shared layers, and what does it compose with?

The idea: adjacent transformer layers compute keys and values that are highly correlated, because the residual stream changes slowly layer to layer. So let layer `l+1` reuse the K and V that layer `l` already computed and cached, rather than computing and caching its own. The cache shrinks by the sharing factor along the **layer** axis, which is completely orthogonal to shrinking along the head axis (GQA), the rank axis (MLA) and the time axis (sliding window). These multiply.

**📄 Paper:** Brandon et al. (2024), *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention* — showed that sharing KV between groups of adjacent layers (CLA-2 shares across pairs) gives a further ~2× cache reduction on top of MQA/GQA at a better accuracy-per-byte trade-off than simply reducing head count further. It replaced "shrink heads more" as the next lever after GQA.

A related but more aggressive design is the decoder-decoder split: a first stack computes a **single** global KV cache, and all layers of a second stack cross-attend to that one cache. That makes cache size independent of the second stack's depth — an `O(1)`-in-layers cache rather than `O(L)`. It is a bigger architectural commitment and changes how you shard, but the memory result is dramatic.

Where to place sharing. My rule, and this is empirical folklore more than settled science, so I would flag it as such: **share in the middle, not at the ends.** Early layers are doing token- and syntax-level work where K/V change fast between layers; final layers are doing output-shaping work that is disproportionately important to the last token. The middle third of the stack is where representations are most stable across depth and where sharing costs least. Any team doing this should ablate it rather than trust the folklore — measure per-layer K/V cosine similarity between adjacent layers on your own data and share the pairs with the highest similarity.

The composition story is the reason to care. Character.AI publicly described stacking **MQA + interleaved local/global attention + cross-layer KV sharing** and reported roughly a 20× KV cache reduction versus a naive baseline, with a very high cache hit rate on top. That is the template: none of these techniques is a 20× win alone, but three orthogonal axes at 3–4× each multiply out. When an interviewer asks "how would you get an order of magnitude," the answer is never one technique — it is picking one lever per axis.

**⚠ Trap:** stacking two techniques on the *same* axis and expecting multiplication. Sliding window and StreamingLLM both act on the time axis; GQA and MLA both act on the width axis. Combining GQA and MLA is not 3.6 × 8 = 29×; MLA already replaces the head-count reduction. Know which axis each lever acts on.

### Give me the taxonomy: KV compression versus KV eviction versus KV quantization. When do you reach for which?

Four distinct families, and confusing them is the most common vocabulary failure in this area.

**Architectural reduction (train-time).** MQA, GQA, MLA, cross-layer sharing, sliding window. You change the model so it *produces* less cache. Requires training or uptraining. Lossless with respect to the model you trained — the model was never going to produce more. Largest and safest wins; zero runtime risk. Unavailable if you did not train the model.

**Quantization (serving-time, lossy per element).** Store the same number of K/V elements in fewer bits: bf16 → fp8 (2×) → int4 (4×). No architecture change, works on any checkpoint, supported by a flag in vLLM and TensorRT-LLM. Loses precision uniformly. The failure mode is a graceful-looking degradation that concentrates in long-context retrieval.

**Eviction / sparsification (serving-time, lossy per token).** Keep full precision but drop *some tokens'* KV entries. StreamingLLM keeps sinks plus a recent window. H2O keeps "heavy hitters" — tokens that historically received high accumulated attention — plus a recent window. SnapKV-style methods score tokens by observed attention from the tail of the prompt and prune before generation. Failure mode is *not* graceful: it is catastrophic on the specific query whose evidence you evicted, and invisible on every other query. That asymmetry is why I am conservative here.

**📄 Paper:** Zhang et al. (2023), *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* — showed that a small set of tokens accounts for most accumulated attention mass, and that keeping those plus a recent window preserves quality at a fraction of the cache. It replaced uniform-recency eviction.

**Offload and reuse (serving-time, lossless).** PagedAttention to eliminate fragmentation, prefix caching to avoid recomputing shared prefixes, CPU/NVMe offload to trade bandwidth for capacity, and disaggregated prefill/decode. These change *where* the cache lives and *how often you recompute it*, not what is in it. Always do these first: they are free in quality terms.

**🔍 Failure taxonomy — my decision procedure, in order:**
1. Are you leaving lossless wins on the table? Paged allocation, prefix caching, chunked prefill, correct block size. **Do these before anything lossy. Always.**
2. Can you pick a different checkpoint? A GQA-8 or MLA model at the same quality tier beats any post-hoc trick on your existing MHA model.
3. Can you afford ~5% of pretraining compute to uptrain to GQA? If yes and you own the weights, this dominates everything below.
4. fp8 KV quantization, gated on a long-context retrieval eval at your P95 context length — not on MMLU.
5. Sliding window or eviction, only if your workload is genuinely streaming (chat where old turns don't matter) and you can measure that it doesn't. **Never on a document-QA or code-context workload.**
6. int4 KV or aggressive eviction. Treat as an emergency lever with a named owner and a rollback plan.

### How do fp8 and int4 KV quantization interact with the attention variant? Which variant tolerates it best?

The unifying principle: **quantization exploits redundancy, and each architectural variant has already spent some of that redundancy.** The more the architecture compressed, the less room quantization has.

**MHA** tolerates KV quantization best. 64 independent key heads carry substantial mutual redundancy; per-element noise on any one head is partially averaged out across heads in the final attention output. This is a slightly perverse result — the most wasteful architecture is the most quantizable — but it is consistent with what people report.

**GQA** is the practical sweet spot and where nearly all production fp8 KV runs live. 8 KV heads still leave room. fp8 e4m3 (4 exponent bits, 3 mantissa) with per-head or per-token scaling is broadly reported as near-lossless on general benchmarks, and it exactly halves the cache: Llama-3-70B goes from 320 KiB/token to **160 KiB/token**, doubling concurrency at fixed HBM.

**MQA** is the most fragile. One shared key head means quantization noise on that head hits *every* query head identically — no averaging, fully correlated error. If you are running MQA, be conservative about KV precision.

**MLA** is the interesting case and I want to be honest that it is less settled. Mechanically, the latent has been trained to be a minimal sufficient representation, so per-element noise is more damaging than noise on a redundant raw activation. Additionally, the decoupled RoPE dimensions are structurally different from the content dimensions and want a different scale — quantizing the concatenated 576-vector with one shared scale is asking for trouble. My rule: **quantize the 512-dim latent and the 64-dim rotary key with separate scales, and validate at your real context length before enabling.**

The technique detail that matters regardless of variant: **keys and values want different quantization granularity.** Key activations exhibit strong per-*channel* outliers — specific coordinates that are large for essentially every token — so keys should be quantized per-channel. Value activations do not have that structure and quantize better per-token. KIVI (Liu et al., 2024) is the clean statement of this: asymmetric 2-bit quantization, per-channel for K, per-token for V, tuning-free. If you implement KV quantization with a single per-tensor scale you will get much worse results than the literature and conclude, wrongly, that the technique doesn't work.

**⚠ Trap:** the attention sink wrecks per-tensor scales. Token 0's key and value activations are frequently an order of magnitude larger than everything else. With a per-tensor scale, that one token sets the scale and crushes the resolution of all 131,071 other tokens. Either exclude sinks from the quantized pool and keep them in bf16, or use per-token scaling. This is a real, reproducible failure, and it is why naive int4 KV implementations look catastrophically bad.

### Same byte budget: GQA-8 at bf16, or MHA-64 at int4. Which do you ship?

Set up the comparison honestly. On a 80-layer, 128-`d_head` model: GQA-8 at bf16 is 2 × 80 × 8 × 128 × 2 = 327,680 B/token. MHA-64 at int4 (0.5 bytes) is 2 × 80 × 64 × 128 × 0.5 = 1,310,720 B/token — that is 4× *more*, so the premise doesn't hold at those settings. To equalize you would need MHA at int2, or GQA-16 at int4 versus GQA-8 at bf16 (2 × 80 × 16 × 128 × 0.5 = 163,840, actually 2× *less*). Let me take the honest equal-bytes pair: **GQA-8 at bf16 (327,680 B) versus GQA-16 at fp8 (2 × 80 × 16 × 128 × 1 = 327,680 B)**. Identical bytes, and now it is a real question.

I ship the **bf16 GQA-8**, and here is the reasoning.

The quality loss from halving KV heads (16 → 8) is a *trained* loss: the model learned to work with 8 heads and its weights compensate. The quality loss from fp8 is an *untrained* loss: noise injected at inference into a representation the model was never trained to be robust against. Trained compressions are almost always cheaper than post-hoc ones at equal compression ratio, because the optimizer got to route around them. This is the same reason quantization-aware training beats post-training quantization.

Second reason: risk profile. A head-count choice is fixed at training time, evaluated once, and never surprises you. A quantization choice interacts with every input distribution — outlier tokens, unusual languages, long contexts, code with repeated tokens — and its failures are input-dependent and therefore latent. I would rather take a known 1% hit than an unknown 0–8% hit that correlates with my hardest queries.

Third: **they compose.** If I have GQA-8 at bf16 and I need more room, fp8 on top gives me 160 KiB/token. Starting from GQA-16 at fp8, my next move is int4, which is a much scarier step. Choosing the trained compression first preserves the runtime lever for later.

**🗣 Say this in the room:** "At equal bytes I take the trained compression over the post-hoc one — the optimizer got to compensate for GQA and did not get to compensate for quantization noise. And it leaves me the quantization lever in reserve for when traffic doubles."

**⚠ Trap:** comparing KV configurations without stating the dtype. "We use GQA-8" is an incomplete specification of your cache footprint. The only complete statement is bytes per token at a stated precision — 320 KiB/token bf16, 160 KiB/token fp8 — and I would push back on any capacity plan written without it.
