### Introduce the KV cache from scratch. What exactly are you caching, and why is caching it correct?

The mental model, in your vocabulary: the KV cache is a per-request memoization table whose eviction policy you do not control and whose size grows linearly with the conversation. It is not an optimization bolted on afterwards — without it, generating token `n` requires recomputing the entire forward pass over all `n-1` previous tokens, making generation `O(T²)` *per token* and `O(T³)` for the sequence. With it, each decode step is `O(T)`. The KV cache is the reason autoregressive generation is tractable at all.

What makes it correct is causality plus the fact that K and V depend only on their own position. Under a causal mask, the key and value vectors at position `j` are computed as `k_j = RoPE_j(RMSNorm(x_j) W_K)` and `v_j = RMSNorm(x_j) W_V`, and `x_j` at every layer is a function of positions `≤ j` only. Appending token `n` cannot change `x_j` for `j < n`. Therefore `k_j` and `v_j` are *immutable* once computed. They are pure functions of the prefix, which is the precondition for caching anything.

What you do **not** cache: queries (position `n`'s query is needed only at step `n`), attention weights (they change every step because the query changes), FFN activations (recomputed from the new token only), or the logits. It's exactly K and V, per layer, per head.

So the decode step becomes: take the single new token, compute `q, k, v` for it (three `[1, d] × [d, d]` matvecs), append `k, v` to the cache, attend the single query against `t+1` cached keys, and run the FFN on one token. Total FLOPs per token ≈ `2N` where `N` is parameter count, plus `4 · n_layers · n_kv_heads · d_head · t` for the attention against the cache.

**⚠ Trap:** "the KV cache stores the previous tokens." It stores the *projected* K and V per layer — the token IDs and embeddings are not what's cached, and there are `2 × n_layers` tensors per sequence, not one. Candidates who say "we cache the previous tokens" get asked "how big is it" and cannot answer.

**🗣 Say this in the room:** "Under causal masking, `k_j` and `v_j` are pure functions of the prefix up to `j`, so appending a token can never invalidate them. That immutability is the entire justification for the cache — and it's also why any change that makes them position-dependent in a new way, like altering the RoPE base, invalidates every cached entry you have."

### Give me the exact cache tensor shapes, and tell me what changes on each decode step.

Per layer you hold two tensors:

```
K_cache: [B, n_kv_heads, T_cached, d_head]
V_cache: [B, n_kv_heads, T_cached, d_head]
```

and there are `n_layers` such pairs, so `2 · n_layers` tensors total. `n_kv_heads` equals `n_heads` in vanilla MHA and is smaller under GQA/MQA.

**Prefill.** Input `[B, T_p]` tokens. You compute `q, k, v` for all `T_p` positions at once: `q: [B, H, T_p, d]`, `k/v: [B, n_kv, T_p, d]`. Attention is `[B, H, T_p, T_p]` with a full causal mask. You write all `T_p` K/V entries into the cache. `T_cached` becomes `T_p`.

**Decode step `i`.** Input `[B, 1]` — a single token per sequence. `q: [B, H, 1, d]`, new `k/v: [B, n_kv, 1, d]`. Append along dim 2, so `T_cached → T_p + i + 1`. Scores are `[B, H, 1, T_cached]` — a matrix-*vector* product per head, not a matmul. **No causal mask is needed** on this row, because the cache by construction contains only positions `≤` the current one. Output `[B, H, 1, d]` → concat → `W_O` → `[B, 1, d_model]`.

The shape change that trips people: `T_q` goes from `T_p` to `1`, but `T_kv` keeps growing. The mask, if you build one, must be `[B, 1, T_q, T_kv]` with the diagonal offset `T_kv − T_q` (see the mask question in chunk 1). During prefill `T_kv − T_q = 0` (plain `tril`); during decode `T_kv − T_q = T_cached − 1` (everything visible).

The allocation question you should raise unprompted: naive `torch.cat` on every step reallocates and copies the whole cache each token — `O(T²)` memory traffic over a generation. Real implementations either preallocate to `max_len` and write into a slice (HuggingFace's `StaticCache`, and what any CUDA-graph-captured decode path requires), or allocate in fixed-size blocks (vLLM's paged approach). `DynamicCache` with concatenation is fine for a notebook and wrong for a server.

**💰 Math — the cost of naive concatenation.** Generating 2,000 tokens for one sequence of a 32-layer, 8-KV-head, `d_head=128` model in bf16: per-token cache is `2 × 32 × 8 × 128 × 2 = 128 KB`. Concatenating at step `t` copies `t × 128 KB`. Summed over 2,000 steps: `128 KB × (2000 × 2001 / 2) ≈ 256 GB` of pointless HBM traffic. At ~3 TB/s that's ~85 ms of pure copying per sequence — on a request whose total budget might be 20 s, so ~0.4%, small but it also fragments the allocator and blocks CUDA graph capture. Preallocation is not premature optimization here.

### Explain `cache_position` and the off-by-one that silently corrupts long generations.

`cache_position` answers one question: *for the tokens I am passing in right now, what are their absolute indices in the full sequence?* It is a 1-D integer tensor with one entry per input token. During prefill of a 500-token prompt it is `[0, 1, ..., 499]`. On the first decode step it is `[500]`, then `[501]`, and so on. **📅 Volatile:** the exact keyword is a Transformers-library API detail — verify against the version you're on — but every serving engine has this concept under some name.

It matters because **two** different things consume it and they are not the same thing:

1. **The positional encoding.** RoPE rotates `q` and `k` by an angle proportional to the token's absolute position. If you pass the wrong position, the rotation is wrong, and since RoPE encodes relative position through the *difference* of rotations, every relative distance involving that token is wrong.
2. **The cache write index.** With a preallocated static cache, this is the slot you write `k, v` into. Write to the wrong slot and you either overwrite a valid entry or leave a hole of uninitialized memory that later gets attended to.

Now the off-by-one, which is the specific failure this question is about. The naive decode loop does something like `position = past_key_values.get_seq_length()`. Consider what that returns *when you call it*. If you query it before appending the current token's K/V, it returns `T_cached` = the number of tokens already cached = the correct index for the new token (0-indexed). If you query it after, it returns `T_cached + 1` and you are one too far. Both look reasonable in code review.

Why it's *silent*: an off-by-one in position affects RoPE by one rotation step. At `θ_base = 10000`, the lowest-frequency dimension has a wavelength of tens of thousands of tokens, so one step is a microscopic rotation there; the highest-frequency dimension has wavelength ~2π, where one step is a large rotation but the model is fairly robust to high-frequency noise. So at 50 tokens of generation, outputs are indistinguishable from correct. Your unit tests pass. Your smoke test passes. The damage accumulates: every generated token is placed one position off relative to the prompt, so by 4,000 tokens the model's internal geometry of "how far back is the instruction" is consistently distorted, and you get the classic report — "it's great for short answers but drifts and loses the format on long ones."

**🔍 The regression test that actually catches it.** Not an equality test on short generation. Do this: take a prompt, generate 512 tokens greedily with the cache. Then take the concatenated `prompt + generated` sequence and run a **single cache-free forward pass**, teacher-forced. Assert that the argmax at each position of the cache-free run matches the token the cached loop actually emitted, for all 512 positions. Cached and uncached must agree exactly under greedy decoding; if they diverge at position `k`, you have a cache bug and `k` tells you where. This test costs one extra forward pass and I make it mandatory on any hand-rolled decode loop.

**⚠ Trap:** assuming `cache_position` and `position_ids` are always equal. They are for a plain sequence. They diverge whenever you left-pad (position ids must start at 0 for the *real* tokens, while cache positions index the padded buffer), whenever you do prefix caching with a reused block, and in some speculative-decoding paths where you feed multiple candidate tokens whose positions branch. Any code that computes one from the other by assumption will break on the first of those you enable.

### Write me multi-head attention with a working KV cache. From memory, no references.

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x):                      # [B, T, D] -> [B, H, T, dh]
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def forward(self, x, cache=None, keep_mask=None):
        # x: [B, T_q, D];  cache: dict with 'k','v' of [B, H, T_past, dh] or None
        B, T_q, _ = x.shape
        q, k, v = self._split(self.wq(x)), self._split(self.wk(x)), self._split(self.wv(x))

        if cache is not None and cache.get("k") is not None:
            k = torch.cat([cache["k"], k], dim=2)      # [B, H, T_kv, dh]
            v = torch.cat([cache["v"], v], dim=2)
        if cache is not None:
            cache["k"], cache["v"] = k, v

        T_kv = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)    # [B, H, T_q, T_kv]
        causal = torch.ones(T_q, T_kv, dtype=torch.bool, device=x.device).tril(
            diagonal=T_kv - T_q)                                    # THE offset
        keep = causal[None, None]
        if keep_mask is not None:                                   # [B, T_kv] True=real
            keep = keep & keep_mask[:, None, None, :]
        scores = scores.masked_fill(~keep, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v                     # [B, H, T_q, dh]
        return self.wo(out.transpose(1, 2).reshape(B, T_q, -1))
```

Forty lines including the class scaffolding, and it handles prefill (`cache={}`, `T_q = T_p`, offset 0 → plain `tril`) and decode (`T_q = 1`, offset `T_kv - 1` → the row is all-True) with the same code path. That single-expression unification is the thing to point at while you write it; interviewers are watching for whether you special-case decode, which is where bugs live.

Three things I'd say aloud while writing. RoPE is omitted for brevity but goes between the split and the cache append, applied to `q` and `k` only, with `q` rotated at positions `[T_past, T_past + T_q)` and the *new* `k` at the same positions. In production this uses `F.scaled_dot_product_attention(q, k, v, attn_mask=keep)` or `is_causal=True` during prefill, which avoids materializing `[B,H,T_q,T_kv]`. And `torch.cat` is the notebook version; a server preallocates.

**🏋 Drill:** write the above unaided in 25 minutes on a blank file, then verify with the cached-vs-uncached greedy equivalence test from the previous question over 128 generated tokens. Pass criterion: exact token-for-token match, first attempt, no debugger.

### Good — three follow-ups on what you just wrote. What's the cache at 128k context, what changes under GQA, and what changes at batch size 1?

**Cache at 128k.** Build it from the code: two tensors per layer, `[B, H, T, d_head]`, so bytes = `2 (K and V) × n_layers × n_heads × d_head × T × bytes_per_elem`. Take a 32-layer model with `H = 32`, `d_head = 128`, bf16, `B = 1`, `T = 131072`:

```
per token = 2 × 32 × 32 × 128 × 2 = 524,288 bytes = 0.5 MB
total     = 0.5 MB × 131,072 = 65,536 MB = 64 GB
```

Sixty-four gigabytes for **one** sequence, on a card that has 80 GB total and is already holding 16 GB of weights. So a vanilla-MHA 8B model cannot serve a single 128k request on an H100 — the cache alone exceeds the free memory. That number, computed live, is the entire motivation for everything in the next section.

**Under GQA.** The code changes in exactly one place: `wk` and `wv` project to `n_kv_heads × d_head` instead of `d_model`, so `k` and `v` come out `[B, n_kv, T, dh]` while `q` stays `[B, H, T, dh]`. Before the score matmul you must expand the KV heads to match the query heads — `k.repeat_interleave(H // n_kv, dim=1)`, or better `k[:, :, None].expand(...)` which doesn't copy, or best, pass `enable_gqa=True` to `F.scaled_dot_product_attention` and let the kernel handle it. The cache formula's `n_heads` becomes `n_kv_heads`: at `n_kv = 8` instead of 32, the 64 GB above becomes **16 GB**. FLOPs are unchanged (you still do `H` query heads' worth of work); only the bytes moved and stored change. That distinction — memory yes, FLOPs no — is the graded part.

**At batch size 1.** The character of the computation changes completely. During decode with `T_q = 1`, every matmul in the model is a matrix-*vector* product: arithmetic intensity is ~2 FLOPs per parameter byte read, so you are memory-bandwidth-bound, not compute-bound. Concretely, for an 8B model in bf16 you must read 16 GB of weights per token; on an H100 at ~3.35 TB/s that's a hard floor of `16 / 3350 ≈ 4.8 ms/token ≈ 210 tokens/s`, and no amount of extra FLOPs capacity helps. Batching is how you amortize that weight read across many sequences — at batch 32 you read the same 16 GB once and produce 32 tokens, so per-token cost drops ~32× until the KV cache reads (which do *not* amortize, being per-sequence) become the new bottleneck.

**📐 Numbers you must know:** `KV bytes/token = 2 × n_layers × n_kv_heads × d_head × dtype_bytes`. Derive it from the code, never recite it. For a 32-layer / 8-KV-head / 128-dim / bf16 model it is 128 KB/token, which is the single most useful constant in LLM serving.

### Why does batched decode require left padding? Be specific about what goes wrong with right padding.

Because decode always appends at the *end* of the buffer, and with right padding the end of the buffer is padding, not the last real token.

Walk it concretely. Batch of two, right-padded to length 8:

```
seq A: [t0 t1 t2 t3 t4 t5 t6 t7]   (8 real tokens)
seq B: [u0 u1 u2 P  P  P  P  P ]   (3 real tokens, 5 pads)
```

Generation appends at index 8 for both. For A that's correct — the new token follows `t7`. For B, the new token follows five pad tokens. Three separate things now break. The model's *input* to the next step is the previously generated token, which for B was itself generated from a context ending in padding, so it's already garbage. The *position* index is 8 for B when its real length is 3, so RoPE places the new token five slots too far away from its actual context. And the KV cache for B contains five entries of pad-token K/V sitting between the real prefix and the new token, which the causal mask will happily let the new query attend to unless the padding mask is threaded correctly through every decode step — which most hand-rolled loops don't do.

Left padding fixes all three at once:

```
seq A: [t0 t1 t2 t3 t4 t5 t6 t7]
seq B: [P  P  P  P  P  u0 u1 u2]
```

Now index 8 immediately follows the last real token for *both* sequences. Append works uniformly. The only remaining requirement is that `position_ids` start at 0 at the first *real* token (so B's real tokens get positions 0,1,2, not 5,6,7) and that the padding mask keeps the leading pads masked out of attention forever. Both are standard and both are things you must pass explicitly — `generate()` will warn you if you feed a right-padded batch without a mask, but if you're writing the loop yourself nothing warns you.

**⚠ Trap:** the mirror-image error. **Training** wants *right* padding, because the loss is computed at every position and label shifting assumes real tokens start at index 0; left padding during SFT puts pad tokens at the start where they interact badly with the BOS/sink and with any code that assumes `input_ids[:, 0]` is BOS. So the rule is: **right-pad for training, left-pad for batched generation**, and if your tokenizer object is shared between the two paths you must flip `tokenizer.padding_side` explicitly. A single shared tokenizer with the wrong padding side is one of the most common sources of "my fine-tune is great on eval loss and terrible in the demo."

**🗣 Say this in the room:** "Decode appends at the buffer's end, so the end must be the last real token — that's left padding. Right-padding a generation batch makes short sequences generate from a context of pad tokens, with positions offset by the pad count. And it inverts for training, where right padding is correct, so the tokenizer's `padding_side` has to be set per code path, not globally."

### Even with left padding, position IDs and the mask have to be right. Show me how.

Left padding solves *where you append*. It does not by itself solve *what position number each token gets* or *what the query is allowed to see*. Those are two more tensors and both have to be derived from the padding mask.

**Position IDs.** RoPE must place the first real token at position 0, otherwise two identical prompts in the same batch get different geometry purely because one was padded more. The standard derivation is a masked cumulative sum:

```python
# pad_keep: [B, T] bool, True where real
position_ids = (pad_keep.cumsum(dim=-1) - 1).clamp(min=0)   # pads get 0, reals get 0,1,2,...
position_ids = position_ids.masked_fill(~pad_keep, 1)       # value is arbitrary for pads
```

Pads get an arbitrary position because their outputs are masked out of every real query's attention, so the value is unobservable — but it must be a *valid index*, not −1, or your RoPE cache lookup goes out of bounds.

**Attention mask.** The padding mask applies to keys forever, including across decode steps: as the cache grows, the mask has to grow with it, with `True` appended for each newly generated token. The bug I see most often is building the padding mask once from the prompt and never extending it, so after 50 generated tokens the mask shape no longer matches `T_kv` and either broadcasts wrong or throws — and if you were using an additive float mask it may broadcast *silently*.

**Per-sequence lengths for the cache.** When sequences in a batch finish at different times, you either keep generating for finished sequences and throw away the output (simple, wastes compute proportional to the length variance) or you compact the batch (complex, and it's exactly what continuous batching in a real engine does for you). At a batch of 32 with lengths varying 100–2000, naive padded batching wastes roughly `1 − mean/max ≈ 1 − 600/2000 = 70%` of decode compute. That number is the whole argument for continuous batching, and it's worth volunteering.

**⚠ Trap:** computing `position_ids` as `torch.arange(T)` broadcast across the batch. It's correct for unpadded batches, it's correct for right-padded batches, and it is silently wrong for left-padded batches — which is exactly the configuration you use for generation. Left-padded `arange` positions shift every sequence's content by its pad count, so a heavily-padded short prompt is evaluated as if it started 1,500 tokens into the context. Symptom: quality depends on what *else* is in the batch. That's a horrifying bug to debug from the outside and a one-line fix from the inside.

### What's the difference between the attention call in prefill and in decode? I want shapes, masks, and hardware behaviour.

They are the same mathematical operation and two completely different computational regimes, and treating them as one thing is how you end up with a serving stack that has good throughput and terrible TTFT, or vice versa.

| | Prefill | Decode |
|---|---|---|
| `T_q` | `T_p` (hundreds to 100k) | 1 |
| `T_kv` | `T_p` | `T_p + i` |
| Mask | full causal triangle | none needed (offset makes it all-visible) |
| Score shape | `[B, H, T_p, T_p]` | `[B, H, 1, T_kv]` |
| Core op | matmul (GEMM) | matrix-vector (GEMV) |
| Bound by | compute (FLOPs) | memory bandwidth |
| Arithmetic intensity | ~`T_p` FLOPs/byte | ~2 FLOPs/byte |
| Metric it drives | TTFT | ITL / TPOT |
| Parallelism | across `T_p` positions | across batch only |

The consequence: prefill saturates tensor cores and scales roughly linearly with prompt tokens; decode leaves the tensor cores ~95% idle and is limited by how fast you can stream weights plus KV out of HBM. This is why the two get scheduled differently, why chunked prefill exists (slice a long prefill into pieces and interleave decode steps so one long prompt doesn't stall every other user's token stream), and why a single "tokens/sec" number for a serving system is meaningless without saying which phase.

**💰 Math — one 8B-class model, H100, bf16.** Prefill of 8,000 tokens: `2 × 8e9 × 8000 = 1.28e14` FLOPs; at a realistic 400 TFLOP/s achieved, that's `1.28e14 / 4e14 = 0.32 s` of TTFT from the model math alone. Decode: 16 GB of weights read per step at 3.35 TB/s = 4.8 ms/token floor, so 500 output tokens = 2.4 s. The prompt is 16× longer than the output and takes 13% of the time. That asymmetry — prefill is FLOP-cheap per token, decode is bandwidth-expensive per token — is the single most important operational fact about serving, and it inverts everything a backend engineer's instinct says about "the big input is the expensive part."

**🗣 Say this in the room:** "Prefill is a GEMM over the whole prompt and is compute-bound; decode is a GEMV over one token and is bandwidth-bound. Same math, opposite hardware regimes, different SLOs — prefill owns TTFT, decode owns inter-token latency — which is why you schedule them separately and why chunked prefill exists."

### You're packing multiple documents into one training sequence. How do you stop them attending to each other, and does it matter?

It matters, and the answer to "how" is a block-diagonal mask composed with the causal mask.

Why pack at all: with a 4,096-token training context and documents averaging 600 tokens, padding each document to 4,096 wastes `1 − 600/4096 ≈ 85%` of your compute. Packing concatenates documents end to end until the window is full, so utilization goes to ~100%. On a 1T-token run that is not an optimization, it's a 6× difference in cost — the difference between a $2M run and a $12M one.

The naive version (just concatenate, use a plain causal mask) lets document 2's tokens attend to document 1's. People do ship this, and it does train, because the model learns to ignore the irrelevant prefix — but it teaches a bad habit (that unrelated preceding text is context) and it corrupts position information (document 2's tokens sit at positions 600–1200 with RoPE distances computed against document 1's content). The measured effect on general LM loss is small; the effect on long-context and instruction-following behaviour is not, and the fix is nearly free.

The correct mask, expressed as a predicate:

```
keep(q, kv) = (doc_id[q] == doc_id[kv]) and (kv <= q)
```

Two ways to implement it. With PyTorch's FlexAttention (2.5+) you write exactly that predicate and it compiles into a block-sparse kernel:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def doc_causal(b, h, q_idx, kv_idx):
    return (doc_id[b, q_idx] == doc_id[b, kv_idx]) & (q_idx >= kv_idx)

block_mask = create_block_mask(doc_causal, B, None, T, T)   # None = shared across heads
out = flex_attention(q, k, v, block_mask=block_mask)
```

The block-sparse compilation matters: with 8 documents packed into 4,096 tokens, the block-diagonal structure means you skip ~7/8 of the score-matrix tiles, so packing with the correct mask can actually be *faster* than packing with a full causal mask. The alternative is the varlen path, which I'll describe next — it's the same idea expressed as offsets rather than a predicate, and it's what FlashAttention exposes.

You must also **reset position IDs per document**, otherwise document 2 begins at RoPE position 600. Same masked-cumsum trick as with padding, keyed on `doc_id` boundaries instead of the pad mask.

**⚠ Trap:** packing with the correct mask but forgetting the position reset (or vice versa). Each alone produces a subtly wrong training signal that shows up as degraded long-context behaviour months later, and neither produces an error. My review checklist for any packing implementation is exactly three assertions: no cross-document attention (verify by constructing a batch where doc 2 is a repeat of doc 1 and confirming the loss on doc 2 does *not* drop), positions restart at 0 per document, and the loss mask drops the boundary/EOS tokens you intend to drop.

### Explain `cu_seqlens` and varlen attention. Why is that layout better than padding plus a mask?

`cu_seqlens` is the cumulative-sum-of-lengths layout: instead of a rectangular `[B, T_max, ...]` tensor with padding, you concatenate all sequences into one flat `[total_tokens, n_heads, d_head]` tensor and carry an `int32` array of length `B+1` giving the start offset of each sequence. For lengths `[3, 5, 2]`, `cu_seqlens = [0, 3, 8, 10]`. Sequence `i` occupies rows `cu_seqlens[i] : cu_seqlens[i+1]`. If you've ever built a CSR sparse matrix or a Postgres array-offset column, it's the identical idea.

The API, which is worth knowing by shape:

```python
from flash_attn import flash_attn_varlen_func
# q, k, v: [total_tokens, n_heads, d_head]  (no batch dim!)
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,     # int32 [B+1], on device
    max_seqlen_q, max_seqlen_k,     # python ints, for kernel launch shape
    causal=True,
)
```

**📅 Volatile:** argument order and extra kwargs shift between flash-attn releases; verify against the installed version.

Why it beats pad-plus-mask, in three concrete ways. **Compute:** padding to `T_max` costs `O(B · T_max²)` attention work; varlen costs `O(Σ T_i²)`. With lengths uniformly distributed in `[128, 4096]`, `E[T²]/T_max² ≈ (4096²/3)/4096² ≈ 1/3`, so you're doing ~3× the necessary attention work with padding — and that ratio gets worse as length variance grows. **Memory:** no padded K/V is allocated, so at that same distribution you hold roughly half the activation memory. **Kernel simplicity:** the kernel launches one CTA block per (sequence, head, query-tile) and reads `cu_seqlens` to bound its loop, so masking is a loop bound rather than a `[B, 1, T, T]` tensor you allocate, write, and read.

The subtlety worth volunteering: **varlen and document packing are the same mechanism used for two different purposes.** Packing several documents into one training sequence with a block-diagonal mask and treating them as separate varlen sequences produce identical attention patterns. The difference is bookkeeping — packing keeps one logical sequence with per-document position resets, varlen keeps `B` logical sequences. Choose varlen when the sequences are genuinely independent (a serving batch, an SFT batch); choose packing when you want one fixed-shape tensor for a compiled graph.

**⚠ Trap:** `max_seqlen` being wrong. It's used to size the kernel's internal loop bounds; passing a value smaller than the actual longest sequence silently truncates attention for that sequence rather than erroring, in some versions. Compute it as `(cu_seqlens[1:] - cu_seqlens[:-1]).max().item()` and eat the device sync, or track it on the host as you build the batch.

### Varlen kernel or a FlexAttention block mask — which do you reach for, and why?

The decision rule I use: **varlen when the structure is "independent sequences of different lengths," FlexAttention when the structure is a nontrivial *pattern* within a sequence.**

Varlen's strengths are that it's the most mature path (it's what every production engine uses for batching), it eliminates padded memory entirely, and it has no compilation step. Its weakness is that it expresses exactly one kind of structure — block-diagonal — via offsets. If your mask is "causal, plus a sliding window of 4096, plus 4 pinned sink tokens, plus document boundaries," varlen cannot express it and you're back to hand-writing a kernel.

FlexAttention's strength is that you write the predicate and it generates a fused block-sparse kernel, including the backward pass, and it exploits the sparsity — `create_block_mask` computes which 128×128 tiles are entirely masked and the kernel skips them. So a sliding window of 4,096 in a 128k context skips ~97% of tiles and you get near-linear cost from a five-line predicate. Its weaknesses are real: it depends on `torch.compile`, so you pay compilation time and inherit graph-break debugging; the block granularity means masks with fine-grained structure that doesn't align to tiles get less speedup than the mask sparsity suggests; and dynamic shapes cause recompiles unless you're careful. **📅 Volatile:** FlexAttention's maturity and feature coverage have been moving fast — check the current state before committing a production path to it.

My actual decision procedure:

**🔍 Choosing an attention path:**
1. Plain causal, uniform lengths → `F.scaled_dot_product_attention(..., is_causal=True)`. Done. Don't reach further.
2. Independent sequences, variable lengths, standard causal → varlen. This is 90% of serving and most of training.
3. Custom pattern (sliding window + sinks, document packing with resets, prefix-LM bidirectional prefix, custom relative bias) → FlexAttention, and validate against a naive reference implementation on small shapes before trusting it.
4. Pattern the block structure can't exploit and you need the last 20% → a hand-written Triton kernel, and you should be very sure the 20% is worth the maintenance.
5. Prototyping, correctness-first → the naive materialized version, always kept in the repo as the reference oracle for (3) and (4).

Keeping (5) is the part people skip and shouldn't. Every custom attention path I've shipped had a `torch.allclose` test against the naive implementation at `T=64` with a random mask, and it caught real bugs in every single case.

### Embedding models use bidirectional attention while generative models use causal. Why, and what do you do when you only have a causal model?

Because the tasks have different information requirements. A generative model must not see the future — that's the training objective. An embedding model has no such constraint: it's producing one fixed vector for a complete, already-known text, so restricting token `i` to positions `≤ i` throws away information for free. In a causal encoder, token 0's representation is computed from *one token*, and the first half of the document is systematically under-contextualized. Bidirectional attention (drop the mask entirely, `keep[i,j] = True` for all real `i,j`) lets every token see the whole text, which is why BERT-lineage encoders punched far above their parameter count on retrieval and classification.

But decoder-only LLMs got vastly better at semantics than any 110M-parameter encoder, so the field spent 2023–2024 figuring out how to extract embeddings from them. Three approaches, in increasing invasiveness:

**Last-token pooling.** Append an `[EOS]`-like token and take its final hidden state. Under causal attention, that position is the only one that has seen everything, so it's the least-bad pooling choice. Mean-pooling a causal model's hidden states is *worse* than last-token pooling, which surprises people coming from BERT — because in a causal model the early positions genuinely have less information, so averaging them in dilutes. E5-Mistral-style recipes (Wang et al., 2023) took this route with strong results.

**Echo embeddings.** Feed the input twice — "Rewrite: {x}. Rewritten: {x}" — and pool over the second copy. Every token of the second copy has, through causal attention, seen the entire first copy, so you get bidirectional information without changing the architecture. Cost: 2× the tokens, so 2× prefill FLOPs and 4× the attention term. **📄 Paper:** Springer et al. (2024) introduced this as a training-free fix for causal-model embedding.

**Convert to bidirectional and continue training.** Drop the causal mask and adapt with a masked-prediction objective plus unsupervised contrastive training. **📄 Paper:** BehnamGhader et al. (2024), *LLM2Vec*, showed the three-step recipe (enable bidirectional attention, adapt with masked next-token prediction, then contrastive learning) converts a decoder-only LLM into a strong text encoder. This is the highest-quality path and it requires a training run.

**⚠ Trap:** flipping the mask off at inference and expecting it to work. A model trained causally has never seen a token attend forward; the activation statistics are out of distribution and the embeddings degrade rather than improve. Bidirectionality requires adaptation training — this is the whole reason LLM2Vec is a paper and not a config flag.

**💰 Math — why this choice is a serving decision, not just a quality one.** Embedding 100M documents averaging 512 tokens: with a 7B decoder-based embedder at ~2 FLOPs/param/token, that's `2 × 7e9 × 512 × 1e8 = 7.2e20` FLOPs; at 400 TFLOP/s achieved on an H100, `7.2e20 / 4e14 = 1.8e6` GPU-seconds = 500 GPU-hours ≈ $1,250 at $2.50/hr. With a 110M-param encoder: `2 × 1.1e8 × 512 × 1e8 = 1.13e19` → 7.8 GPU-hours ≈ $20. A 60× cost difference on the *indexing* side, and it recurs every time you re-embed. That's why the honest answer to "which embedder" is almost never "the biggest one" — it's "the smallest one that clears your Recall@k bar on *your* corpus."

### How do image tokens actually reach an LLM? Compare the connector designs and tell me which you'd ship.

Two families, and the trade is context budget versus training simplicity.

**Projector / prefix style (LLaVA-lineage).** A vision encoder (a ViT, typically a CLIP or SigLIP-family one) produces patch embeddings — for a 336×336 image at patch 14, that's `24 × 24 = 576` patches. An MLP projects them from vision-encoder width into `d_model`, and they are inserted into the token sequence like any other tokens. From the LLM's perspective there is no such thing as an image; there are 576 embeddings it self-attends over. **📄 Paper:** Liu et al. (2023), *Visual Instruction Tuning* (LLaVA), showed a two-layer MLP connector plus instruction data was enough to get strong multimodal ability, replacing far more complex cross-attention adapters.

**Gated cross-attention style (Flamingo-lineage).** Image features stay outside the text sequence. New cross-attention layers are interleaved into a *frozen* LLM; text tokens query the image features. The gate is initialized so the new layers contribute exactly zero at step 0, meaning the model is provably identical to the base LLM before training starts — an elegant property when you can't afford to damage a frontier text model. **📄 Paper:** Alayrac et al. (2022), *Flamingo*.

The trade, quantitatively. Prefix style spends context: 576 tokens per image, and high-resolution schemes that tile an image into 4–9 crops spend 2,300–5,200 tokens for **one** image. On a 128k budget, twenty screenshots consume a third of your context and, more importantly, the KV cache grows accordingly — at 128 KB/token that's `5200 × 128 KB = 665 MB` of KV for a single high-res image. Cross-attention style spends *parameters and a static cache* instead: the image K/V are computed once and reused across every decode step, and they never enter the text KV cache at all.

**Which I'd ship:** prefix/projector style, by default, for almost any product. It requires no architecture surgery, it works with every off-the-shelf serving engine (the image tokens are just tokens, so continuous batching, prefix caching, and paged attention all work unmodified), and the data requirements are far lower. I'd only reach for gated cross-attention if I were (a) adapting a model I must not perturb, or (b) in a regime where images per request is high enough that context is the binding constraint — a document-understanding product ingesting 50-page scanned PDFs is the real case.

**⚠ Trap:** budgeting an image as "about a thousand tokens" without checking the tiling scheme. High-resolution modes multiply patch count by the number of crops plus a downsampled global view, and providers price them accordingly. Before you quote a cost for a vision feature, tokenize a representative image through the actual provider's counter — the difference between 576 and 5,200 tokens per image is a 9× swing in both cost and prefill latency, and I have watched a feature's unit economics get discovered in production because someone estimated instead of measuring.
