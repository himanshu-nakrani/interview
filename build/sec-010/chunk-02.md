### Trace one token's residual stream through a full attention block for me. Every tensor, in order.

The framing that makes this easy to hold in your head: the residual stream is a shared bus of width `d_model`, and every block reads a projection of it, computes something, and *adds* the result back. Nothing overwrites. The stream at layer `L` is literally `x_0 + Σ_{l<L} (attn_l + ffn_l)`, a running sum of contributions. That's why you can ablate one head, why LayerNorm is applied to the *read* and not the bus, and why "the model stores information in the residual stream" is a literal claim about a tensor rather than a metaphor.

Concretely, for one token at position `t` in Llama-3-8B (`d_model=4096`, `H=32`, `d_head=128`, RMSNorm, pre-norm):

1. `x_t ∈ R^4096` arrives on the bus.
2. `h = RMSNorm(x_t)` — scale-normalize, elementwise learned gain, no mean subtraction, no bias. `[4096]`.
3. `q = h W_Q` → `[4096]` reshaped `[32, 128]`. Likewise `k = h W_K`, `v = h W_V` (with GQA these have fewer heads; that's §11's territory).
4. RoPE rotates `q` and `k` in-place, per head, in 2D subspaces indexed by position `t`. `v` is *not* rotated — a detail people get wrong.
5. `k, v` get appended to the cache at slot `t`, giving `K ∈ [32, t+1, 128]`, `V` likewise.
6. Per head `h`: `s = q_h · K_h^T / √128` → `[t+1]` scores; mask (nothing to mask if `t` is the newest and the cache holds only the past); `a = softmax(s)` → `[t+1]`; `o_h = a @ V_h` → `[128]`.
7. Concatenate the 32 heads → `[4096]`, multiply by `W_O` → `[4096]`.
8. `x_t ← x_t + o` — the write back to the bus.
9. Then the FFN reads `RMSNorm(x_t)`, does `SwiGLU` up to ~14336 and back down, and adds again.

**⚠ Trap:** applying RoPE to `V`. Values carry content, not position-matching information; rotating them corrupts what the head writes and, subtly, breaks the relative-position property (which relies on the *product* `q·k` picking up the rotation difference). Every from-scratch implementation I've reviewed that had "long context is broken" as a symptom had either this or the cache-position bug.

**🗣 Say this in the room:** "Pre-norm read, three projections, RoPE on Q and K only, append K/V to the cache, per-head softmax over the cache, concat, `W_O`, add back to the residual. The residual stream is never overwritten — every block is a read-compute-add, which is exactly why per-head ablation is a coherent experiment."

### What is an attention head actually writing into the residual stream, and how would you reason about that algebraically?

Split the head into two independent bilinear circuits and the whole thing becomes tractable. **📄 Paper:** Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*.

The **QK circuit** is `W_Q W_K^T ∈ R^{d_model × d_model}`, rank ≤ `d_head`. It never touches the output; it only decides *where* each position reads from. The score between positions `i` and `j` is `x_i (W_Q W_K^T) x_j^T / √d_head` — a bilinear form on the residual stream. When you ask "what does this head attend to," you are asking about the top singular directions of this matrix.

The **OV circuit** is `W_V W_O^{(h)} ∈ R^{d_model × d_model}`, also rank ≤ `d_head`. It decides *what gets written* given that you read from somewhere: the head's contribution is `Σ_j a_ij · (x_j W_V W_O)`. Crucially it's independent of `i` — the same linear map applies to whatever you attend to.

So a head is: "a low-rank bilinear scorer decides a distribution over source positions; a low-rank linear map transports content from those positions into the destination's residual stream." Two consequences you can use in an interview. First, a "copy head" is one whose OV circuit approximates the identity in the token-embedding subspace — it moves a token's identity verbatim, which you can check by looking at `W_E W_V W_O W_U` (embedding → OV → unembedding) and seeing whether it's diagonal-dominant. Second, because both circuits are rank ≤ 128 out of 4096, a single head can only read from and write to a small subspace, which is exactly why heads specialize rather than each learning a blurry average of everything.

**⚠ Trap:** treating the attention *weights* as the head's function. Two heads with identical attention patterns can do completely opposite things if their OV circuits differ in sign — one copies the attended token, the other suppresses it. There is a documented class of "copy suppression" heads in GPT-2-small that attend *to* a token specifically in order to reduce its logit. Attention pattern tells you the read; only the OV circuit tells you the write.

### Where exactly does fp16 blow up in attention, and why does bf16 not have the same problem?

fp16 has a 5-bit exponent: max finite value **65504**, smallest normal ≈ 6.1e-5. bf16 has fp32's 8-bit exponent (max ≈ 3.4e38) and only 8 bits of mantissa. So bf16 trades precision for range, and attention is a range problem, not a precision problem. That's the entire answer, and every specific failure follows from it.

The three places it bites:

**1. Raw dot products before scaling.** If your kernel computes `Q @ K^T` and *then* divides by `√d_head`, the intermediate lives at `√d_head ×` the final scale. That's usually fine — until you hit massive activations. Real trained LLMs develop a handful of residual-stream dimensions with magnitudes in the hundreds or thousands (**📄 Paper:** Sun et al., 2024, *Massive Activations in Large Language Models*, documented outliers orders of magnitude above the typical feature and tied them to attention-sink behaviour). Two vectors each with a coordinate of magnitude 300 contribute `9e4` from that one dimension alone — already over 65504 in fp16. Result: `inf` scores, then `inf - inf = NaN` in the max-subtraction, then a NaN that propagates through the whole batch. The fix in every serious kernel is to fold the scale into `Q` *before* the matmul: `q = q * (1/√d_head)`, so the accumulator never sees the unscaled magnitude.

**2. The exponentials.** `exp(z - max(z))` is bounded by 1 by construction, so this is safe *if* you subtract the max. Skip the max subtraction and `exp(20)` = 4.85e8 already overflows fp16.

**3. Accumulating the softmax denominator.** Summing `T` terms each ≤ 1 in fp16 with T = 128k: fp16 has ~3 decimal digits, so once the running sum exceeds ~2048 further additions of small terms are silently dropped. Every kernel accumulates the running sum and running max in **fp32**, regardless of the storage dtype of Q/K/V.

**📐 Numbers you must know:** fp16 max = 65504 (5-bit exponent, 10-bit mantissa); bf16 max ≈ 3.39e38 with 8-bit mantissa — same exponent as fp32, ~3 decimal digits of precision vs fp16's ~3.3. Rule: bf16 for training and for anything with a dynamic range you don't control; fp16 only where you have a loss scaler or a known-bounded range. Both need fp32 accumulation in softmax, LayerNorm statistics, and loss reduction.

**⚠ Trap:** "bf16 is more accurate than fp16." It is strictly *less* precise (8 mantissa bits vs 11 effective). It's preferred because the failures of insufficient range are catastrophic and silent, while the failures of insufficient precision are gradual and mostly absorbed by SGD noise. Say it that way and you sound like you've debugged a training run.

### `-1e9` or `-inf` as the mask value? Somebody's PR uses `-1e9`. What do you say in review?

I ask which dtype the scores are in, and then I usually ask them to change it — but the interesting part is that both choices have a real failure mode and they fail *differently*.

`-inf` is the mathematically correct value: `exp(-inf) = 0` exactly, the masked key contributes nothing to numerator or denominator, and the row is a clean distribution over visible keys. Its failure mode is the fully-masked row: `max = -inf`, `scores - max = -inf - (-inf) = NaN`, and NaN propagates through the entire batch — one bad row poisons everything downstream, including the gradients of unrelated sequences.

`-1e9` is the "defensive" choice, and it fails **silently**, which is worse. In fp16, `-1e9` is not representable (max 65504) so it *becomes* `-inf` anyway — the defensiveness evaporates exactly where you thought you were buying safety. In fp32/bf16 it does avoid the NaN, but consider a fully-masked row: the max is `-1e9`, every score minus max is `0`, `exp(0) = 1`, and you get a **perfectly uniform attention distribution over the padding tokens**. No NaN, no error, no alert. Your model quietly attends to garbage for those rows and the loss curve looks slightly worse than it should for reasons nobody can find.

There's a subtler one too. `-1e9` added to a score that is itself large-and-negative doesn't reach `-inf`, so with additive masks stacked (padding + causal + a custom bias) you can accumulate `-3e9`, which is fine — but if any code path *multiplies* a mask by a scaling factor, or if a later kernel assumes masked entries are exactly `-inf` to skip blocks, `-1e9` defeats the optimization. FlashAttention-style kernels skip entire tiles when the block mask says so; a soft `-1e9` looks like a real (very negative) score and the tile gets computed.

**🗣 Say this in the room:** "Use `-inf`, or better, use a boolean mask and let the fused kernel handle it. `-1e9` doesn't protect you in fp16 because it saturates to `-inf` anyway, and in fp32 it converts a loud NaN into a silent uniform-attention-over-padding bug. The right fix for the all-masked row is to never produce one — mask keys, not queries, and drop fully-padded sequences upstream."

**⚠ Trap:** the "fix" of clamping the output with `torch.nan_to_num` after the attention. It hides the symptom, produces zeros where the model expected a real vector, and the gradient through `nan_to_num` is not what you want. If you see this in a codebase, the bug is upstream.

### A row of the attention matrix is fully masked. Walk me through exactly what happens and how you defend against it.

Step by step in fp32 with `-inf` masking. Scores row is `[-inf, -inf, ..., -inf]`. Numerically-stable softmax computes `m = max(row) = -inf`, then `row - m` = `-inf - (-inf)` = **NaN** for every entry. `exp(NaN) = NaN`. Sum = NaN. Division = NaN. The output vector for that query is all-NaN, it gets added to the residual stream, and from there it contaminates the rest of the layer stack for that position. In the backward pass, NaN gradients flow into `W_Q, W_K, W_V, W_O` — which are *shared across the batch* — so one padding row destroys the whole optimizer step. The classic symptom is "loss goes NaN at step 4,300 and never recovers," and the classic red herring is blaming the learning rate.

How you get a fully-masked row, in the order I check:

**🔍 Failure taxonomy — all-masked rows:**
1. **A fully-padded sequence in the batch.** Empty string, a document that tokenized to zero tokens, or a filtered-out example that a collator turned into all-pad. Fix upstream in the data loader: assert `pad_keep.any(dim=-1).all()`.
2. **Masking pad queries as well as pad keys.** A pad query row has every key masked by construction. This is the #1 cause. Fix: don't mask queries.
3. **Sliding-window masks with a window smaller than the left-padding.** Query at position `t` sees `[t-W, t]`; if all of those are padding, the row is empty. Common when you bolt a window onto a left-padded batch.
4. **Cross-attention to an empty context.** A VLM request with zero image tokens, a RAG request where retrieval returned nothing, an encoder output that got fully padded.
5. **Chunked prefill where a chunk contains only future-masked positions** because someone got the diagonal offset wrong.

The two defenses I actually ship. First, a data-layer invariant: every sequence has at least one real token, asserted in the collator, cheap and catches (1) and (4). Second, if you genuinely cannot guarantee it, don't fix it in the softmax — fix it in the mask, by forcing `keep[..., 0] = True` for any otherwise-empty row so it attends to a single dummy position, and then zero that query's output explicitly. That produces a defined, inspectable zero rather than a NaN.

**💰 Math — what this costs.** A NaN at step 4,300 of a run that checkpoints every 1,000 steps costs you 3,300 steps of recompute. On 64×H100 at ~$2.50/GPU-hour, if those 3,300 steps took 4 hours, that's `64 × 4 × $2.50 = $640` of GPU time — trivial. The real cost is the 1.5 days of a senior engineer bisecting the data loader, and the fact that you only find it after it happens twice. A three-line assertion in the collator is one of the highest-ROI things in a training stack.

### Explain the online softmax trick and why every fast attention kernel is built on it.

The naive softmax needs three passes over a row: find the max, exponentiate and sum, then divide. Three passes means the whole row must be readable three times, which means it must live in memory — and for attention that row is `T` long and there are `B·H·T` of them. Online softmax fuses this into **one** pass by maintaining a running max and a running sum that are corrected whenever the max changes. That single algorithmic change is what makes it possible to never write the score matrix to HBM. **📄 Paper:** Milakov & Gimelshein (2018) introduced the one-pass "online" softmax; FlashAttention (Dao et al., 2022) extended the same correction trick through the `@V` accumulation so the entire attention output is computed in one streaming pass.

The recurrence, which you should be able to write:

```python
# streaming over blocks of keys; m = running max, l = running sum, o = running output
m, l, o = -inf, 0.0, zeros(d)
for (k_blk, v_blk) in blocks:                 # k_blk: [Bk, d], v_blk: [Bk, d]
    s = (q @ k_blk.T) / sqrt(d)               # [Bk]
    m_new = max(m, s.max())
    alpha = exp(m - m_new)                    # correction factor for everything so far
    p = exp(s - m_new)                        # [Bk]
    l = l * alpha + p.sum()
    o = o * alpha + p @ v_blk                 # rescale the accumulated output too
    m = m_new
out = o / l
```

The insight is that `exp(s - m_old) = exp(s - m_new) · exp(m_new - m_old)`, so when the max increases you can retroactively rescale *both* the running denominator and the running numerator by a single scalar `alpha` — you never need to revisit the old blocks. Everything runs in fp32 registers/SRAM; only `o` and `l` (size `d` and 1 per query) ever touch memory.

What this unlocks beyond speed: it's the same mechanism behind **splitting attention across devices** (Ring Attention passes partial `(m, l, o)` triples around a ring and combines them), behind **chunked prefill** (process the prompt in slices, carry the accumulator), and behind combining a cached prefix's partial attention with a new suffix's. Once you see attention as an associative reduction over key blocks with a `(max, sum, weighted-sum)` monoid, all of those become obvious rather than clever.

**⚠ Trap:** thinking FlashAttention is an approximation. It is numerically *exact* — bit-for-bit identical to the naive version up to floating-point reassociation, and often *more* accurate because it accumulates in fp32. Candidates conflate it with sparse or linear attention constantly. The one caveat worth volunteering: because the reduction order differs, FlashAttention output is not bitwise-identical to naive attention or to itself across different tile configurations, which matters for reproducibility debates.

### What is attention entropy collapse, how would you detect it mid-run, and what fixes it?

Attention entropy is `H = -Σ_j a_ij log a_ij` for a query row, averaged over rows, heads, and layers. High entropy means diffuse averaging; low entropy means the head has become a hard argmax. Collapse is when entropy crashes toward zero early in training and *stays* there — heads lock onto a single key and the softmax Jacobian `p_i(δ_ij - p_j)` goes to zero, so those heads stop learning. It shows up as a loss plateau or a loss spike, and it correlates with the logit magnitudes growing without bound, because nothing in the architecture bounds `q·k`.

The causal chain: `‖W_Q x‖` and `‖W_K x‖` grow during training (weight decay fights this weakly, and the residual stream's norm grows layer by layer anyway) → dot products grow → post-`√d` logits still grow → softmax saturates → entropy → 0 → gradient → 0 → the head is frozen at whatever half-learned pattern it had. **📄 Paper:** Zhai et al. (2023), *Stabilizing Transformer Training by Preventing Attention Entropy Collapse*, identified the entropy metric as the leading indicator of transformer training instability and proposed σReparam (spectral reparameterization of the weights) to bound it.

**🔍 Detection, in order of what I put on the dashboard:**
1. **Mean attention entropy per layer**, logged every N steps on a fixed probe batch. Cheap: you need the softmax output, which fused kernels don't return — so compute it on a tiny probe batch with the naive path, not on the training batch.
2. **Max absolute attention logit per layer.** Even cheaper and correlates tightly. If you see it climbing past ~50 you are in trouble; past 100 you are collapsed.
3. **Gradient norm of `W_Q`/`W_K` relative to `W_V`/`W_O`.** Collapse shows as QK gradients dying while OV gradients continue.

**The fixes, in the order I'd try them:**
- **QK-norm** — apply RMSNorm (or L2 normalization) to `q` and `k` per head before the dot product. This bounds `q·k` by `d_head · scale²` structurally, so no amount of weight growth can saturate the softmax. It costs two extra norms per layer (~1–2% throughput) and it is the fix I reach for first; it was used in ViT-22B and has been adopted in several recent LLM recipes. **📅 Volatile:** which specific 2026 open-weight models ship QK-norm changes fast — verify against the actual config before your loop.
- **Attention logit softcapping** — `logits = C · tanh(logits / C)`, which bounds them smoothly. Gemma-2 used this with a cap around 50 for attention logits and around 30 for the final output logits. **📅 Volatile:** cap values are per-model config, check `config.attn_logit_softcapping`. The cost: `tanh` is incompatible with the fastest FlashAttention paths, which is exactly why the industry drifted toward QK-norm instead.
- **z-loss** on the final logits, lower `β₂` in Adam, and gradient clipping, all of which help the general instability but do not target attention specifically.

**🗣 Say this in the room:** "Entropy collapse is a gradient-death problem, not an explosion problem. Nothing bounds `q·k`, so as the weights grow the softmax saturates and the QK circuit stops receiving gradient. I log mean attention entropy and max attention logit per layer on a probe batch; if logits are climbing past 50 I add QK-norm, which bounds the dot product structurally for about 1–2% throughput."

### Why does token 0 hoard probability mass in a trained model? Give me the mechanism, not the observation.

Because the softmax denominator forces every query to spend exactly 1.0 of probability mass, and there is frequently nothing a head wants. A head that is looking for, say, "the last open bracket" has no work to do on a token in the middle of prose — but it cannot output zero. It must dump its mass somewhere harmless. Token 0 (the BOS token, or just the first token) is the ideal dump site because it is **visible to every query under a causal mask** — it is the one key guaranteed to be in every row's visible set — and because it carries no content the model needs, so writing its value into the residual stream is a cheap no-op if the OV circuit learns to map it near zero.

So attention sinks are an *emergent workaround for a missing no-op*, not a bug and not a feature the designers put there. **📄 Paper:** Xiao et al. (2023), *Efficient Streaming Language Models with Attention Sinks*, named the phenomenon and showed that in trained LLMs a large fraction of attention mass across most heads and layers lands on the first few tokens regardless of semantic content.

Two corroborating observations worth having ready. First, the sink token typically develops **massive activations** — residual-stream coordinates orders of magnitude larger than typical, which is the mechanism by which it becomes a strong, easy-to-hit key (Sun et al., 2024, tied massive activations to sink behaviour). Second, the same phenomenon appears in vision transformers, where the model repurposes uninformative background patches as scratch space; **📄 Paper:** Darcet et al. (2023), *Vision Transformers Need Registers*, fixed it by adding explicit learnable "register" tokens that exist purely to absorb this mass — direct evidence that the model wants a no-op and will manufacture one if you don't provide it.

The design implication people miss: a model trained *with* a consistent BOS token has a stable, dedicated sink. A model trained without one recruits whatever token happens to be first, which means the sink identity shifts with your prompt. That's why "should I prepend BOS at inference" is not a stylistic question — for some checkpoints, dropping BOS moves the sink to your first real token and measurably degrades output.

**⚠ Trap:** interpreting an attention heatmap that's 60% on token 0 as "the model thinks BOS is important." It's the opposite: that head has decided the current query has no relevant source and is idling. This is the most-misread visualization in the field.

### StreamingLLM exploits sinks. Explain what it does, and tell me exactly what breaks if you evict token 0.

The setup: you want infinite-length streaming generation with bounded memory, so you keep a sliding window of the most recent `W` KV entries and evict the rest. The naive version catastrophically fails — perplexity explodes by orders of magnitude the moment the window slides past the start. The StreamingLLM finding is that the culprit is not losing the *content* of the early tokens; it's losing the *sink*.

The fix is almost insultingly small: keep the first ~4 tokens' KV pinned forever, plus the rolling window of the last `W`. Cache size becomes `4 + W` instead of `T`, constant in generation length, and perplexity stays flat over millions of tokens. **📄 Paper:** Xiao et al. (2023) — replaced "window attention degrades gracefully" (it doesn't) with "window attention plus a handful of pinned sink tokens is stable."

What breaks without token 0, mechanically: every head that was idling on the sink must now redistribute its full unit of mass across the *visible window*, which is all recent, all semantically loaded content. That mass is not neutral — it flows through each head's OV circuit and writes real vectors into the residual stream. So the residual stream now receives a large spurious contribution at every layer, in a direction determined by whatever the window happens to contain. The distribution shifts off-manifold, and the model, which was never trained in that regime, produces garbage. The failure is abrupt, not gradual: it happens on the step the sink leaves the window.

The second thing that breaks, and this one catches people: **positions**. StreamingLLM assigns positions by *position within the cache*, not by absolute position in the stream. If you keep absolute positions, then after a million tokens your window's queries carry RoPE positions around 1,000,000 while the model was trained to at most 8,192 — extrapolation failure on top of eviction failure. Getting the re-indexing right is half the implementation.

**⚠ Trap:** believing StreamingLLM gives you long-context *understanding*. It gives you long-context *fluency*. Anything outside the window is genuinely gone — you cannot answer a question about token 500,000 when your window is 4,096 wide. It's the right tool for an endless chat that only needs recent coherence, and the wrong tool for document QA. I've seen this proposed as a "free 1M context" and it is not; conflating the two is a fast way to lose a serving-design round.

**💰 Math:** for a 7B-class model with 32 layers, 8 KV heads (GQA), `d_head=128`, bf16, the per-token KV cost is `2 × 32 × 8 × 128 × 2 = 131,072` bytes = **128 KB/token**. Unbounded streaming to 1M tokens would need 128 GB — impossible on one card. StreamingLLM with `W = 4096` plus 4 sinks needs `4100 × 128 KB ≈ 525 MB`, constant forever. That's the pitch, in one line of arithmetic.

### Where does dropout go in an attention block, and why has it nearly vanished from modern pretraining?

There are three canonical dropout sites in the original architecture: on the attention *weights* (after softmax, before `@V`), on the attention output (after `W_O`, before the residual add), and inside the FFN. Attention dropout is the interesting one — it randomly zeroes entries of the post-softmax distribution, which means the surviving weights no longer sum to 1 (PyTorch's inverted dropout rescales by `1/(1-p)` in expectation, but per-sample the row is no longer normalized). Effectively it's stochastic edge-dropping on the attention graph.

It has nearly vanished from large-scale pretraining for a simple reason: **dropout is a regularizer for the overfitting regime, and frontier pretraining is not in that regime.** When you train a 70B model on 15T tokens, you see each token roughly once. There is no overfitting to prevent; there is only underfitting relative to the data. Adding dropout in that setting costs you effective capacity and slows convergence for zero generalization benefit. The GPT-3-era configs already used small or zero dropout; modern Llama/Qwen/Mistral-class pretraining configs typically set it to 0.

Where it *does* survive, and this is the judgment part:
- **Fine-tuning on small datasets.** A 10k-example SFT run on a 7B model absolutely can overfit; dropout 0.05–0.1 plus early stopping is reasonable. LoRA configs commonly ship `lora_dropout=0.05` for exactly this.
- **Encoder models and embedding models.** BERT-lineage and many embedding trainers keep dropout, partly for regularization and partly because in contrastive training the dropout noise *is* the augmentation (two forward passes of the same sentence with different dropout masks give you a positive pair — that's the core trick in SimCSE-style unsupervised contrastive embedding training).
- **Small models on small corpora**, e.g. an academic 100M-param model on a 1B-token corpus, where you genuinely will do multiple epochs.

**⚠ Trap:** leaving attention dropout on and then wondering why your fused attention kernel isn't being used, or why inference results differ from training-mode evaluation. Two concrete gotchas: `F.scaled_dot_product_attention(..., dropout_p=0.1)` must receive `dropout_p=0.0` at inference — it does not read `model.eval()` — so a hand-rolled decode loop that forwards a config value straight into the kernel will apply dropout during generation and produce nondeterministic garbage. And several fast attention paths simply do not support nonzero dropout, so you silently fall back to a slower kernel. The rule: gate it on `self.training` explicitly at the call site.

### Explain induction heads. What are they, how were they identified, and why do people claim they explain in-context learning?

An induction head implements pattern completion over the context: having seen `... A B ... A`, it predicts `B`. Not because `B` is likely in general, but because *in this context* `A` was followed by `B`. That is the minimal mechanism for learning from the prompt rather than from the weights — which is the definition of in-context learning.

It takes **two layers**, and the two-layer circuit is the part to be able to draw. In layer 0, a **previous-token head** attends from each position to the position immediately before it and writes the previous token's identity into the current position's residual stream. So after layer 0, position `j` holds a composite: "I am token `B`, and I was preceded by token `A`." In layer 1, the **induction head** at the final position (currently `A`) forms a query from "I am token `A`" and matches it against keys built from the *previous-token* information written in layer 0 — so it finds positions whose predecessor was `A`. That's position `j`. Its OV circuit then copies `j`'s token identity (`B`) into the output, boosting `B`'s logit. Read as circuits: it's a QK-composition (layer 1's key reads what layer 0's OV wrote) plus a copying OV circuit.

**📄 Paper:** Elhage et al. (2021) identified the two-layer prefix-matching-plus-copying circuit in attention-only toy models; Olsson et al. (2022), *In-Context Learning and Induction Heads*, showed the formation of induction heads coincides with a visible **phase change** in the training loss curve — a small bump then a sharp drop — and that the model's in-context learning ability appears at the same moment across model scales. That co-occurrence is the evidence behind "induction heads explain ICL."

Be honest about the strength of the claim, because a good interviewer will push. What's well-established: the circuit exists, it's found reliably across models, and its formation coincides with the ICL phase change. What's a stronger claim than the evidence supports: that induction heads are *the* mechanism for all in-context learning including few-shot task learning on complex tasks. The literal copy-the-next-token behaviour is clearly a component; whether sophisticated few-shot behaviour is "induction heads all the way up" or something more is genuinely open. Say "the strongest mechanistic story we have, established for literal pattern completion, plausibly a building block for the rest" and you'll be exactly right.

**🗣 Say this in the room:** "Induction is a two-layer circuit: a previous-token head in an early layer writes 'my predecessor was X' into each position, and a later head queries on the current token, matches those predecessor annotations, and copies the successor. It's the smallest mechanism that does prompt-conditioned prediction, and its emergence lines up with a phase change in the loss curve and the onset of in-context learning."

### Name three other head types you'd expect to find, and tell me what practical use head-level analysis has.

Beyond induction and previous-token heads, the ones with documented, reproducible behaviour: **copy heads**, whose OV circuit approximates the identity in token space so they transport a token's identity forward verbatim (the "give me back the name that appeared earlier" primitive); **name-mover heads**, characterized in the indirect-object-identification circuit in GPT-2-small (**📄 Paper:** Wang et al., 2022, *Interpretability in the Wild*) which attend to a name in the context and write it to the output; **copy-suppression heads**, which attend to a token specifically to *lower* its logit and thereby prevent naive repetition; and **positional/syntactic heads** that attend to fixed relative offsets or to delimiters, brackets, and separators.

Now the honest part, because this is what separates a research-literacy answer from a research-cosplay answer: **head-level analysis rarely changes what I ship.** Its practical uses are narrower than the literature's enthusiasm suggests:

1. **Debugging a broken long-context model.** If the sink heads have shifted or the retrieval heads (the small set of heads that empirically do the needle-fetching) are attending to the wrong span, you have a positional-encoding bug, not a data bug. This actually localizes problems.
2. **Deciding what you can prune or share.** Cross-layer KV sharing and head pruning schemes are informed by which heads are doing redundant work. Practically relevant to serving cost.
3. **Communicating with research partners.** At a big-tech applied-AI org you'll sit next to people doing this work; being able to say "is this a QK-composition failure or an OV problem" makes you a useful collaborator rather than a consumer.
4. **Not over-trusting attention maps in a product**, which is the next question.

**⚠ Trap:** offering interpretability as your debugging strategy in a production round. If someone says "our RAG answers went stale after a reindex" and you propose looking at attention heads, you have failed the round. Head analysis is for model-internal problems in models you control; almost every production LLM incident is a data, retrieval, prompt, or serving-configuration problem. Know the tool and know its blast radius.

### A product manager wants to ship attention weights as an "explanation" feature. What's your position?

I push back, and I can give the literature, the mechanism, and the alternative.

The mechanism argument first, because it's the strongest. An attention weight tells you where a head *read from*. It tells you nothing about what the head *wrote*, because that's the OV circuit — and as noted, copy-suppression heads attend hard to a token precisely to suppress it. It also tells you nothing about the 30-odd other layers of processing, nothing about what the FFNs did with the result, and nothing about counterfactual dependence, which is what a user means by "explanation": *if this source hadn't been there, would the answer change?* Attention weight is correlational within one head; the user's question is causal across the whole model.

The literature: **📄 Paper:** Jain & Wallace (2019), *Attention is not Explanation*, showed that on classification tasks you can find alternative attention distributions that produce the same prediction, and that attention weights correlate poorly with gradient-based feature importance. **📄 Paper:** Wiegreffe & Pinter (2019), *Attention is not not Explanation*, pushed back that this depends on what you mean by explanation and that adversarially-found distributions aren't what the model actually learned. The honest summary is: attention is *not* a faithful explanation in the counterfactual sense, and the field never resolved it into consensus. In a decoder-only LLM with 80 layers × 64 heads it's substantially worse than in the single-attention-layer classifiers those papers studied.

What I'd ship instead, in increasing order of cost:
- **Citations from the retrieval layer**, not the model — "this answer used chunks 3, 7, 11" is true by construction because those are the chunks you put in the context, and you can verify the claim with an entailment check.
- **Inline span-level attribution** produced by asking the model to emit citation markers under structured output, then *verifying* each cited span actually supports the claim with a separate cheap model call. This is what Perplexity-class products do, and the verification step is what makes it honest.
- **Leave-one-out ablation** for high-stakes cases: re-run with a chunk removed and see if the answer changes. Genuinely causal, costs one extra generation per chunk, so reserve it for audit paths.

**🗣 Say this in the room:** "Attention weights are a read-side signal from one head out of thousands, and they say nothing about what that head wrote or whether the answer would change without that token — so they're not faithful explanations. For a product, I'd derive citations from the retrieval layer where the provenance is true by construction, have the model emit span-level citations under structured output, and verify a sample of them with an entailment check. That's auditable; an attention heatmap is decoration."
