# PART I — Foundations Interviews Actually Probe

Only the math that gets asked, plus the single most reliable senior-vs-mid judgment tell: knowing when not to use an LLM at all.

## Contents

1. [7. Math for LLM Engineers: Linear Algebra, Probability, Information Theory, Statistics](#7-math-for-llm-engineers-linear-algebra-probability-information-theory-statistics) — 50 questions
2. [8. Training Math: Losses, Optimizers, Backprop, Numerics, Normalization, Stability](#8-training-math-losses-optimizers-backprop-numerics-normalization-stability) — 51 questions
3. [9. Classical ML, Pre-Transformer NLP, and the "Is an LLM Even Right Here?" Gate](#9-classical-ml-pre-transformer-nlp-and-the-is-an-llm-even-right-here-gate) — 44 questions


---

## 7. Math for LLM Engineers: Linear Algebra, Probability, Information Theory, Statistics

*Mastering this proves you can derive rather than recite when an interviewer stops you mid-answer and asks "why √d_k?"*

### Be honest with me — how much math do I actually need for an AI Engineer loop, and what can I safely not know?

The mental model: **you are not being tested on mathematics, you are being tested on whether you can derive a quantity you have forgotten.** Nobody at Cursor or Meta is going to ask you to prove the spectral theorem. What they will do is stop you mid-answer and say "why √d_k?" or "how big is that KV cache?" and watch whether you reach for a memory or reach for a derivation. The candidate who says "because the paper says so" and the candidate who says "because the dot product of two d-dimensional unit-variance vectors has variance d, so the logits grow like √d and saturate the softmax" get graded a full level apart, and the second one took eleven extra seconds.

Concretely, the surface that gets probed in these loops is small and I can enumerate it: tensor shapes through a transformer block; the √d_k derivation; rank and low-rank factorization (because LoRA); rotation matrices (because RoPE); FLOP counting (2N, 6N, C = 6ND); the memory equation and the KV-cache formula; softmax-as-Boltzmann and what temperature does; entropy, cross-entropy, KL and reverse-KL; perplexity and its tokenizer dependence; three sampling tricks (Gumbel-max, rejection, importance); calibration; and enough frequentist statistics to say "that eval delta is noise" with a number attached. That is roughly one week of study, and it is load-bearing for five of the other sections in this guide.

What you can skip without guilt: measure theory, proofs of convergence, the full derivation of the EM algorithm, anything about kernels and RKHS, Bayesian nonparametrics, the entire manifold-learning literature. If a topic doesn't cash out in a shape, a byte count, a dollar, or a confidence interval, it will not appear in an applied loop.

**⚠ Trap:** over-studying probability theory and under-studying arithmetic. The single most-failed math moment in these interviews is not a proof — it is a candidate who cannot estimate, out loud, how many GB of KV cache 64 concurrent 32k-token sessions need. That is multiplication, and people freeze on it because they never rehearsed it out loud.

**🗣 Say this in the room** (when asked how comfortable you are with the math): "I'm fluent in the arithmetic that governs a serving deployment — shapes, FLOPs, memory, cost — and I can derive the attention scale and the KV-cache formula on a whiteboard. I'm not going to pretend to be a research mathematician; if you need someone to prove a convergence bound I'd flag that as outside my range."

### Take a batch of 8 sequences of 2,048 tokens through one transformer block and write every tensor shape on the whiteboard.

This is the single most common opening warm-up in a frontier-lab or big-tech applied round, and the thing being measured is not whether you know the architecture — it is whether you can hold four indices in your head without stalling. Rehearse it until it is muscle memory. I'll use the Llama-3-8B-ish config so the numbers are concrete: `d_model = 4096`, `n_heads = 32`, `head_dim = 128`, `n_kv_heads = 8` (grouped-query), `d_ff = 14336`, `L = 32` layers, `vocab = 128256`.

Start: token ids `[B, T] = [8, 2048]`, integers. Embedding lookup — a gather from a `[vocab, d] = [128256, 4096]` table — gives `x: [B, T, d] = [8, 2048, 4096]`.

Inside the block, pre-norm first: RMSNorm is elementwise over the last axis, so shape is unchanged, `[8, 2048, 4096]`.

Projections. `W_q: [d, n_heads·head_dim] = [4096, 4096]`, so `q: [8, 2048, 4096]`, reshaped to `[B, T, H, d_h] = [8, 2048, 32, 128]` and transposed to `[B, H, T, d_h] = [8, 32, 2048, 128]`. **That last shape — `[B, H, T, d_h]` — is the one to be able to write from memory; it is the canonical attention layout.** With GQA, `W_k` and `W_v` are `[4096, 8·128] = [4096, 1024]`, so `k, v: [8, 8, 2048, 128]`. The 8 KV heads are broadcast (repeated 4×) against the 32 query heads.

Scores: `q @ k^T` contracts `d_h`, producing `[B, H, T, T] = [8, 32, 2048, 2048]`. This is the tensor that eats your memory. In bf16 that is 8 × 32 × 2048 × 2048 × 2 bytes = **2.15 GB for one layer's scores**, which is exactly why FlashAttention exists — it never materializes this. Scale by 1/√128, add the causal mask, softmax over the last axis (still `[8,32,2048,2048]`), then `attn @ v` contracts the key axis and gives `[B, H, T, d_h] = [8, 32, 2048, 128]`. Transpose and merge heads back to `[8, 2048, 4096]`, then the output projection `W_o: [4096, 4096]` keeps it there.

Residual add, second RMSNorm, then the MLP. SwiGLU has three matrices: `W_gate, W_up: [4096, 14336]` and `W_down: [14336, 4096]`. So `[8,2048,4096] → [8,2048,14336] → [8,2048,4096]`. Residual add. Out.

After all 32 blocks, a final norm and the LM head `[4096, 128256]` produce logits `[B, T, V] = [8, 2048, 128256]` — in bf16, 8 × 2048 × 128256 × 2 = **4.2 GB of logits**, which is why every serious training stack fuses the LM head with the cross-entropy loss and chunks over T rather than materializing that tensor.

**⚠ Trap:** saying `[B, T, H, d_h]` when you mean `[B, H, T, d_h]`. They differ by a transpose, and the transpose is what makes the per-head matmul a clean batched GEMM. Interviewers notice, because getting it backwards is the tell that you've read about attention but never debugged a shape error in it.

**📐 Numbers you must know:** attention scores are `[B, H, T, T]` — quadratic in T, linear in H, and *independent of d_h*. Every "why is long context expensive" answer starts from that shape.

### Why is Q·Kᵀ a similarity? Walk me through what's actually happening geometrically.

The mental model: **attention is a soft, differentiable dictionary lookup, and the dot product is the lookup key comparison.** Every token emits a query vector — "here is what I'm looking for" — and every token emits a key vector — "here is what I offer." The dot product q·k is large when those two point in the same direction in R^{d_h}, and the softmax turns the vector of similarities into a probability distribution over which values to blend. Nothing about attention requires that this comparison be a dot product; the dot product is chosen because it is the cheapest bilinear form that a GPU can evaluate as a GEMM.

Mechanically, q·k = ‖q‖‖k‖cos θ. So it is cosine similarity, *unnormalized* — magnitude is not divided out. That's deliberate and it matters: the norm of a key is a learnable "how much do I want to be attended to at all" knob, and the norm of a query is a learnable inverse-temperature — a large-norm query produces a peaked distribution, a small-norm query produces a diffuse one. The model gets to control the sharpness of its own lookup per token, per head. If you normalized q and k to unit length you would take that away, which is why cosine attention is a real but distinctly different design (and one reason QK-norm, which does exactly that normalization, changes training dynamics rather than being a free stability win).

The projections are the other half of the story. q = xW_q and k = xW_k, both from the same residual stream x. So the score is x_i W_q W_kᵀ x_jᵀ — a bilinear form x_i M x_jᵀ with M = W_q W_kᵀ, of rank at most d_h = 128 even though it acts on a 4096-dimensional space. **Attention does not compare tokens in the residual stream; it compares them after projecting into a low-rank subspace that this particular head has learned to care about.** That's why heads specialize: one head's M might be near-rank-1 and essentially implement "attend to the previous token," another's might implement "attend to the most recent open bracket."

**🗣 Say this in the room:** "Q·Kᵀ is an unnormalized cosine similarity in the head's 128-dimensional subspace. The projections mean each head compares tokens through its own learned low-rank lens, and leaving it unnormalized lets the model control the sharpness of the softmax per query."

**⚠ Trap:** calling it "cosine similarity" flat out. An interviewer who is paying attention will ask "where's the normalization?" and you need the answer above ready — the missing normalization is a feature, not an oversight.

### Derive the √d_k. Where does that number come from, exactly?

This is the canonical stop-you-mid-answer question in this field. Here is the full derivation, which takes about ninety seconds to say out loud.

Model the query and key components as independent random variables with mean 0 and variance 1 — which is roughly what they look like at initialization, when W_q and W_k are drawn from a scaled Gaussian and the residual stream is normalized. Then the score is

  s = q · k = Σ_{i=1}^{d_k} q_i k_i.

Each product q_i k_i has E[q_i k_i] = E[q_i]E[k_i] = 0 by independence, and Var(q_i k_i) = E[q_i²k_i²] − 0 = E[q_i²]E[k_i²] = 1 · 1 = 1. The terms are independent, so variances add:

  Var(s) = d_k, and therefore std(s) = √d_k.

For d_k = 128, the raw logits have standard deviation about 11.3. Feed a vector of logits with spread ±11 into a softmax and you get a distribution that is effectively one-hot: the gap between the top logit and the runner-up is typically several units, and e^{5} ≈ 148, so one token takes essentially all the mass. Divide by √d_k and Var(s) returns to 1 — logits with unit spread, a softmax that is smooth and has gradient everywhere.

That is the whole argument: **the scale is a variance-normalization that keeps the softmax off its saturated ends at initialization.** The original transformer paper states exactly this in a footnote, and it is one of the few places in deep learning where the folk explanation and the real one coincide.

**📄 Paper:** Vaswani et al. (2017), *Attention Is All You Need* — introduced scaled dot-product attention; the 1/√d_k factor is justified in the paper by precisely this variance-growth argument, replacing additive (Bahdanau-style) attention, which needed no scaling because it never formed a large dot product in the first place.

**⚠ Trap:** saying "it normalizes the vectors to unit length." It does not — it divides the *scalar score* by a constant, and the constant is the standard deviation of the score at init, not the norm of anything. A related trap is saying "so the logits are unit variance during training." They are not; after training, learned norms and QK circuits drift far from the init assumption, which is exactly why models like Gemma-2 add logit softcapping and why QK-norm was introduced as a training-stability fix. The √d_k argument is an *initialization* argument.

**🗣 Say this in the room:** "Dot products of d-dimensional unit-variance vectors have variance d, so scores at init have standard deviation √d_k — about 11 for a 128-dim head. That saturates the softmax and kills the gradient. Dividing by √d_k restores unit variance. It's an init-time argument, and it's why models that drift away from it later need softcapping or QK-norm."

### Follow-up: show me concretely what goes wrong to the gradients if I drop the scale.

The chain is: large logits → saturated softmax → near-zero Jacobian → no gradient reaches W_q and W_k → those matrices never learn → attention is frozen at whatever random pattern it initialized with.

Make it precise. The softmax Jacobian is J = diag(s) − s sᵀ, where s is the output probability vector. Its diagonal entries are s_i(1 − s_i). If the distribution has saturated so that s_max = 0.999, that entry is 0.999 × 0.001 ≈ 10⁻³, and every other entry is smaller still — for a token with s_i = 10⁻⁶, the diagonal is ~10⁻⁶. Every gradient flowing back through this softmax into the score matrix is multiplied by numbers of that magnitude. In bf16, which has roughly 8 bits of mantissa, a gradient scaled by 10⁻⁶ against neighboring terms of order 1 simply disappears into rounding.

The empirical signature, if you run the experiment (and it is worth running once — it's a ten-minute experiment on a two-layer toy model): unscaled attention trains, but the attention entropy collapses within a few hundred steps to near zero, the loss plateaus well above the scaled run's, and the gradient norm on W_q is one to two orders of magnitude below the gradient norm on the MLP weights. That last diagnostic — **per-parameter-group gradient norms diverging by orders of magnitude** — is the general debugging move, not just for this bug.

The subtler point, and the one that gets you credit: the problem gets *worse with larger heads*. At d_k = 64 the unscaled std is 8; at d_k = 128 it's 11.3; at d_k = 256 it's 16. So an unscaled architecture becomes untrainable as you scale head dimension, which is exactly the regime transformers were designed to enter. The scale isn't a nicety, it's what makes the architecture scale-invariant in head width.

**🔍 Failure taxonomy — "my attention isn't learning":** (1) check attention entropy per head per layer; if it's near 0 from step ~100, the logits are saturated. (2) Check logit magnitude directly — histogram `scores` before softmax; std should be O(1), not O(10). (3) If std is large and you *are* dividing by √d_k, the culprit is downstream drift, not init — the fix is QK-norm or softcapping, not a bigger divisor. (4) If entropy is near log(T) — maximal — for every head at step 10k, the opposite failure: your queries have collapsed to near-zero norm, usually from over-aggressive weight decay on W_q.

### What's the Jacobian of the softmax, and why do interviewers keep asking for it?

They ask because it is the one matrix derivative that shows up in every part of this stack — the attention softmax, the output distribution, the policy in RLHF, the teacher/student distributions in distillation — and because it takes exactly one line if you understand it and three minutes of flailing if you don't.

For s = softmax(z) with s_i = e^{z_i}/Σ_k e^{z_k}, differentiate. When i = j, the quotient rule gives ∂s_i/∂z_i = s_i − s_i² = s_i(1 − s_i). When i ≠ j, only the denominator depends on z_j, giving ∂s_i/∂z_j = −s_i s_j. Combine both cases with a Kronecker delta:

  ∂s_i/∂z_j = s_i(δ_ij − s_j),  i.e.  **J = diag(s) − s sᵀ.**

Three properties worth stating unprompted. First, J is symmetric and positive semi-definite. Second, its rows sum to zero — J·1 = s − s(1ᵀs) = s − s = 0 — which is the formal statement that softmax is invariant to adding a constant to all logits. That invariance is what licenses the log-sum-exp trick: subtract max(z) before exponentiating and nothing changes except that you stop overflowing. Third, J is rank-deficient (rank d−1), so there is always a direction in logit space with zero gradient.

The payoff everyone actually cares about: compose it with cross-entropy loss L = −log s_y. Then ∂L/∂z = s − y, where y is the one-hot target. All the messy Jacobian structure cancels, and **the gradient of the loss with respect to the logits is just "predicted distribution minus target distribution."** This is why every framework fuses softmax and cross-entropy into one op — not just for numerical stability, but because the fused backward is a single subtraction instead of a matrix-vector product against a d×d Jacobian.

```python
# verify the identity in 6 lines
import torch
z = torch.randn(5, requires_grad=True)
s = z.softmax(0)
J_auto = torch.autograd.functional.jacobian(lambda t: t.softmax(0), z)
J_hand = torch.diag(s) - torch.outer(s, s)
assert torch.allclose(J_auto, J_hand.detach(), atol=1e-6)
```

**⚠ Trap:** materializing the Jacobian in your own custom layer. For a 128k vocabulary that matrix is 128256² × 4 bytes = 65 TB. Backprop never forms a Jacobian; it forms vector-Jacobian products. If your answer to "how does backprop work" involves building J, you've described something that cannot run.

### Write multi-head attention with einsum, and tell me what einsum buys you over reshape and transpose.

What einsum buys you is that **the shape contract is written down in the source code instead of living in your head**. Every `.transpose(1,2).reshape(B, T, -1)` chain is an unwritten assumption about memory layout that a reviewer has to simulate mentally; `'bqhd,bkhd->bhqk'` states the contraction and the output layout in eleven characters. In review I will take an einsum over an equivalent permute chain almost every time, and the exception is when I need a specific fused kernel that only accepts a particular layout.

```python
import torch, math

def mha_einsum(x, Wq, Wk, Wv, Wo, H):
    """x: [B,T,D]; Wq/Wk/Wv/Wo: [D,D]; returns [B,T,D]."""
    B, T, D = x.shape
    dh = D // H
    q = torch.einsum('btd,de->bte', x, Wq).view(B, T, H, dh)
    k = torch.einsum('btd,de->bte', x, Wk).view(B, T, H, dh)
    v = torch.einsum('btd,de->bte', x, Wv).view(B, T, H, dh)

    scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(dh)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
    scores = scores.masked_fill(causal, float('-inf'))
    attn = scores.softmax(-1)

    out = torch.einsum('bhqk,bkhd->bqhd', attn, v).reshape(B, T, D)
    return torch.einsum('btd,de->bte', out, Wo)
```

Read the two attention lines as sentences. `'bqhd,bkhd->bhqk'`: for each batch b and head h, contract over d (the only index missing from the output) between query position q and key position k. `'bhqk,bkhd->bqhd'`: contract over k, the key axis, restoring the per-position head layout. The reshape at the end is the only place layout knowledge is required, and it is correct precisely because the output index order `bqhd` puts h and d adjacent and last.

Two things to know about einsum's cost model. It does **not** magically pick an optimal contraction order for more than two operands — `torch.einsum` with three or more tensors uses a heuristic, and for anything performance-critical you should either use `opt_einsum` or split it into explicit pairwise contractions. And an einsum whose output has more elements than either input (an outer product) will silently allocate; `'bqhd,bkhd->bhqk'` is exactly that, allocating the `[B,H,T,T]` tensor. In production you replace this entire block with `torch.nn.functional.scaled_dot_product_attention`, which dispatches to a FlashAttention kernel and never materializes it.

**🏋 Drill:** 12 minutes, no references, no autocomplete. Write `mha_einsum` from a blank file, then write a second version using only `view`/`transpose`/`matmul`, then assert the two agree to 1e-5 on random inputs with B=2, T=16, D=64, H=4. Pass criterion: both run and agree on the first attempt after your own review pass. If the assert fails, the bug is almost certainly a transpose before the reshape — which is the whole point of the drill.

### Explain rank and low-rank decomposition to me — then tell me why it's the entire basis of LoRA.

Mental model: **rank is the number of independent directions a matrix actually uses, and almost every large weight matrix in a trained network uses far fewer than its dimensions allow.** A `[4096, 4096]` matrix has 16.8M parameters but if its rank is 64, all of its action is captured by a 4096×64 and a 64×4096 pair — 524k parameters, a 32× compression, with *zero* approximation error. Low-rank methods are the bet that the interesting part of a weight matrix, or of a weight *update*, lives in a small subspace.

Formally, rank r means the matrix can be written W = AB with A ∈ R^{m×r}, B ∈ R^{r×n}. Storage drops from mn to r(m+n), and the matmul cost drops the same way: x(AB) computed as (xA)B costs 2·T·r·(m+n) FLOPs instead of 2·T·mn. The break-even is r < mn/(m+n); for square m = n = 4096 that's r < 2048, so any rank below half the dimension is a win.

LoRA's specific claim is sharper than "weights are low rank." It is that **the *update* is low rank even when the weights are not.** You freeze W₀ and learn W = W₀ + BA with A ∈ R^{r×d}, B ∈ R^{d×r}, r typically 8–64, A initialized Gaussian and B initialized to zero so the adapter starts as an exact no-op. The scaling α/r is applied to the product so that changing r doesn't require retuning the learning rate.

```python
class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.A), self.B) * self.scale
```

**💰 Math on why this is the default:** full fine-tuning a 7B model in mixed precision needs bf16 weights (14 GB) + bf16 grads (14 GB) + fp32 master weights (28 GB) + Adam's two fp32 moments (56 GB) = **112 GB of state** before a single activation — it does not fit on an 80 GB H100. LoRA at r = 16 over all linear layers is roughly 40M trainable parameters. Gradients and optimizer state now cost 40e6 × 16 bytes = 0.64 GB, and the frozen base is 14 GB in bf16 (or ~4 GB at 4-bit under QLoRA). You went from "needs 2 nodes" to "fits on one card with room for a real batch," which is the difference between a $40/hr experiment and a $2/hr one.

**📄 Paper:** Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* — froze the base weights and trained a rank-r additive update, replacing adapter-layer approaches that inserted extra sequential modules and therefore added inference latency. LoRA's adapters can be merged into W₀ after training, so inference latency is identical to the base model.

**⚠ Trap:** believing LoRA is always as good as full fine-tuning. It is close for style, format and task adaptation, and it measurably lags when you are injecting substantial new knowledge or doing long continued-pretraining runs — the rank constraint is a real capacity constraint. The honest framing in an interview is "LoRA first, and I'd escalate to full fine-tuning only with an eval showing the rank ceiling is binding."

### Do the SVD for me. Given a matrix, how would you actually pick the rank r, and what does the spectrum tell you?

Mental model: **SVD is the statement that every linear map is a rotation, then an axis-wise stretch, then another rotation.** W = UΣVᵀ, where U and V are orthogonal and Σ is diagonal with non-negative entries σ₁ ≥ σ₂ ≥ … sorted descending. The columns of V are the input directions the map cares about, the columns of U are where they land, and σ_i is how much each is amplified. Rank is the count of nonzero σ.

The reason SVD is *the* tool for choosing r rather than one option among many is the Eckart–Young theorem: truncating to the top-r singular triplets gives the best possible rank-r approximation in both Frobenius and spectral norm, and the leftover error in Frobenius norm is exactly √(Σ_{i>r} σ_i²). So rank selection is not a hyperparameter search — it is reading off a cumulative sum.

```python
import torch
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
energy = (S**2).cumsum(0) / (S**2).sum()
r = int((energy < 0.95).sum()) + 1       # smallest r capturing 95% of squared mass
W_r = (U[:, :r] * S[:r]) @ Vh[:r]        # the optimal rank-r approximation
print(r, (W - W_r).norm() / W.norm())    # relative Frobenius error
```

How I read a spectrum in practice. A **sharp elbow** — σ dropping two orders of magnitude by index 50 — means genuine low-rank structure and low-rank compression will be nearly free. A **slow power-law decay**, which is what most trained transformer weight matrices actually look like, means there is no clean cutoff and any truncation is a real quality trade. A **flat spectrum** means the matrix is effectively full rank and low-rank methods will hurt. There is also a diagnostic use: the *stable rank* ‖W‖_F²/‖W‖₂² = Σσ_i²/σ₁² is a continuous, differentiable proxy for rank that you can log during training to watch representation collapse — if it falls toward 1, your layer is collapsing to rank 1 and something upstream is broken.

Two practical cautions. First, SVD on a `[4096, 14336]` matrix costs O(mn·min(m,n)) ≈ 2.4e11 FLOPs — seconds on a GPU, but do not put it in a training step. For large matrices use randomized SVD (`torch.svd_lowrank`), which gets the top-r triplets in O(mnr). Second, **the spectrum of W is not the spectrum of the update ΔW**, and LoRA's premise is about the latter. A common and genuinely interesting empirical exercise: full-fine-tune a small model, compute ΔW = W_ft − W₀, and look at *its* spectrum. It typically decays much faster than W's — which is the actual empirical justification for LoRA.

### DeepSeek's multi-head latent attention — convince me it's the same linear-algebra trick as LoRA, just applied somewhere else.

It is, and seeing that equivalence is a strong senior signal because it shows you factor ideas rather than memorize architectures.

The problem MLA attacks is not parameter count, it is **KV-cache bytes**. Standard multi-head attention caches, per token per layer, one K vector and one V vector of size n_heads × head_dim. GQA reduces this by sharing K/V across query-head groups — a coarse, discrete reduction (32 heads → 8 KV heads is a 4× cut, and you cannot ask for 3.7×). MLA instead says: rather than caching K and V, cache a **single low-rank latent vector c of dimension d_c**, and reconstruct K and V from it at use time via learned up-projections. That is exactly the LoRA factorization — a wide matrix expressed as a thin bottleneck plus two projections — moved from "the weight update" to "the cached activation."

Concretely, per token per layer you compute c = x·W_down with c ∈ R^{d_c}, and cache only c. When you need attention you form K = c·W_K_up and V = c·W_V_up. Because W_down and the up-projections are all fixed weights, the up-projection into the key path can be algebraically absorbed into W_q — you fold W_K_up into the query projection once, offline, and then score directly against the cached latents. So you pay no extra runtime matmul on the key side for the decompression.

The consequence is that KV bytes per token per layer drop from 2 × n_kv_heads × head_dim to d_c, and d_c is a free continuous knob. The reason a lab reaches for MLA over just pushing GQA further is quality: past a certain point, cutting KV heads costs measurable capability because you're removing independent key subspaces, whereas the latent bottleneck keeps all heads distinct and only constrains the rank of the shared representation.

**📄 Paper:** DeepSeek-AI (2024), *DeepSeek-V2* — introduced Multi-head Latent Attention, compressing the KV cache into a low-rank latent that is cached in place of K and V; it replaces the "just use fewer KV heads" (MQA/GQA) approach with a continuous rank knob. Note that MLA composes awkwardly with RoPE — position-rotated components can't be absorbed the same way — which is why the design carries a small separate rope-carrying dimension alongside the latent.

**🗣 Say this in the room:** "MLA is LoRA applied to the KV cache instead of to the weights. You cache a low-rank latent per token and reconstruct K and V from it, and because the up-projection is a fixed matrix you can fold it into the query projection so decode costs nothing extra. GQA gives you discrete 2×/4×/8× cuts; MLA gives you a continuous one."

### What's an orthogonal matrix and where does orthogonality actually earn its keep in this field?

An orthogonal matrix Q satisfies QᵀQ = I: its columns are orthonormal. The one property everything else follows from is that **it preserves norms and angles** — ‖Qx‖ = ‖x‖ for all x, because ‖Qx‖² = xᵀQᵀQx = xᵀx. Geometrically it is a rotation, possibly composed with a reflection. Its inverse is its transpose, so it costs nothing to undo, and all its singular values are exactly 1.

That last fact is the whole reason it matters for deep networks. When you backpropagate through a linear layer, the gradient is multiplied by Wᵀ. If W's singular values are mostly less than 1, gradients shrink geometrically with depth; if greater than 1, they explode. An orthogonal W multiplies gradient norms by exactly 1 at every layer — the depth-invariant sweet spot. That's the motivation for orthogonal initialization, which mattered enormously for deep RNNs and matters less in transformers only because residual connections and normalization already provide a norm-preserving path.

Where it earns its keep today, concretely:

- **RoPE** is literally a block-diagonal rotation matrix applied to q and k. It works precisely because rotation preserves the norm — RoPE injects positional information without changing how "loud" a token is, which additive positional embeddings cannot claim.
- **Muon**, the optimizer that got a lot of attention in 2025 speedrun results, orthogonalizes the momentum update (via a few Newton–Schulz iterations) before applying it, on the argument that a gradient update whose singular values are wildly unequal wastes capacity on a few directions. **📅 Volatile:** optimizer fashion moves fast; verify what the current state of practice is before quoting it as standard.
- **Randomized SVD and QR-based projections** rely on orthonormal bases for numerical stability — Gram–Schmidt is numerically horrible in float, which is why every library uses Householder QR.
- **Attention heads want near-orthogonal subspaces.** A useful diagnostic on a trained model is the pairwise cosine between head output subspaces; heads that have collapsed onto each other are redundant capacity you're paying for.

**⚠ Trap:** claiming "orthogonal init fixes vanishing gradients in transformers." Residual streams already give an identity path, and pre-LN already controls variance; orthogonal init is a marginal effect there. Overclaiming it signals that your knowledge is from a 2015 RNN paper rather than from a transformer you've actually trained.

### Derive RoPE for me. Why is a rotation the right way to encode position at all?

The mental model: **absolute position embeddings tell each token where it is; attention cares about where tokens are *relative to each other*. RoPE gets relative position for free by encoding absolute position as a rotation angle, because rotations compose by subtracting angles.**

The construction. Take a head dimension d_h and split it into d_h/2 two-dimensional pairs. For position m and pair index i, define an angle θ_i = base^{−2i/d_h} (base is typically 10,000) and rotate that pair by mθ_i:

  [x_{2i}, x_{2i+1}] ↦ [x_{2i}cos(mθ_i) − x_{2i+1}sin(mθ_i), x_{2i}sin(mθ_i) + x_{2i+1}cos(mθ_i)].

Stack those 2×2 rotations into a block-diagonal orthogonal matrix R_m ∈ R^{d_h×d_h}, and apply it to **q and k only, never to v**. Each pair rotates at its own frequency: pair 0 rotates once per token (fast, encodes fine local order), the last pair rotates with period ~2π·10000 tokens (slow, encodes coarse global position). It's a positional clock with hands of many different speeds — the same idea as sinusoidal embeddings, but applied multiplicatively to the query/key vectors rather than added to the residual stream.

```python
def rope(x, base=10000.0):
    """x: [B, H, T, dh] with even dh. Rotate-half convention (Llama/HF)."""
    B, H, T, dh = x.shape
    half = dh // 2
    inv_freq = base ** (-torch.arange(half, device=x.device).float() / half)
    ang = torch.arange(T, device=x.device).float()[:, None] * inv_freq[None, :]
    cos, sin = ang.cos(), ang.sin()                    # [T, half]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

Why rotation rather than addition. Because R_m is orthogonal, ‖R_m q‖ = ‖q‖ — the positional encoding cannot change the magnitude of a query or key, so it cannot make a token systematically louder or quieter just because of where it sits. Additive positional embeddings *do* change norms, and they also consume residual-stream capacity that every downstream layer must route around. And because it is applied at every layer inside the attention computation rather than once at the embedding, RoPE's positional signal cannot be "forgotten" by deep layers.

**📄 Paper:** Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* — encoded absolute position as a rotation of q and k such that attention scores depend only on relative offset; it replaced learned-absolute and additive-sinusoidal embeddings and is now the default in essentially every open-weight family.

**⚠ Trap:** applying RoPE to V. Rotating values would rotate the *content* being aggregated, and since the output is a weighted sum over many positions, you'd be summing vectors in inconsistent frames. RoPE belongs to the score computation only. A second trap: applying RoPE once at the embedding layer. It must be applied inside every attention layer, to that layer's freshly projected q and k.

### Prove it. Show me that RoPE actually gives relative-position dependence.

One line, and then the consequences.

The attention score between query at position m and key at position n is

  (R_m q)ᵀ (R_n k) = qᵀ R_mᵀ R_n k = qᵀ R_{n−m} k,

using two properties of the block-diagonal rotation: R_mᵀ = R_{−m} (a rotation's inverse is its transpose, i.e. rotation by the negative angle) and R_a R_b = R_{a+b} (rotations by the same axis compose additively in angle). So the score depends on m and n **only through their difference n − m**. Absolute positions were used to build the encoding, and they cancel exactly in the bilinear form. That is the entire theorem, and being able to write those three symbols on a whiteboard is the answer they're looking for.

Now the consequences, which is where the follow-ups go.

**Why length extrapolation fails.** The score is qᵀR_{n−m}k, a function of relative offset. During training the model only ever sees offsets up to the training context — say 8,192. At inference with offset 40,000, the fast-rotating dimensions have wrapped around many times into angle regions the model never saw with meaningful data, and the effective attention pattern becomes garbage. This is not a bug in RoPE; it is the model having no data at those offsets.

**How the fixes work.** *Position interpolation* rescales positions by a factor L_train/L_target so offset 40,000 maps back into the trained angle range — cheap, and it costs some fine-grained local resolution because you've squeezed all the fast dimensions. *NTK-aware scaling* / *YaRN* recognize that not all frequency bands need the same treatment: high-frequency dimensions (which encode local order and already have many full periods inside the training window) should be left nearly alone, low-frequency ones (which never complete a period) should be interpolated. That per-band treatment is why YaRN-style scaling extends context with less short-context quality loss than naive interpolation. In practice you still need a short fine-tune at the new length; scaling alone gets you most but not all of the way.

**Raising the base.** Many long-context models simply train with base = 500,000 or 1,000,000 instead of 10,000, which lengthens every period so the slowest dimensions still haven't wrapped at 128k. **📅 Volatile:** the specific base value per model family changes with each release — check the config, don't quote from memory.

**🗣 Say this in the room:** "R_mᵀR_n = R_{n−m}, so the absolute positions cancel and the score is a function of relative offset only. That's the whole property. It also explains why extrapolation fails — the model has simply never seen those offsets — and why the fixes are all about remapping unseen offsets back into the trained angular range rather than about RoPE being wrong."
### Where does "2N FLOPs per token" come from? Derive it rather than quoting it.

Mental model: **every parameter in a dense model is a multiply-accumulate that gets used exactly once per token, and a multiply-accumulate is two floating-point operations.** That's the whole derivation. Two FLOPs per parameter per token, so 2N per token forward.

Make it concrete on one linear layer. y = xW with x ∈ R^{d_in}, W ∈ R^{d_in×d_out}. Computing y requires d_in × d_out multiplications and about the same number of additions: 2·d_in·d_out FLOPs, and d_in·d_out is exactly the parameter count of W. Every matmul in the network has this property, so summing over all layers gives 2N FLOPs per token, where N is the total parameter count. For a batch of B sequences of length T you have B·T tokens, so the forward pass costs 2N·B·T.

Three things are being swept under the rug, and you should name them before the interviewer does. First, **elementwise operations are ignored** — softmax, GELU/SiLU, RMSNorm, the residual adds. They're O(B·T·d) rather than O(B·T·d²), so they're a fraction of a percent of the FLOPs. They are emphatically *not* a fraction of a percent of the runtime, because they're memory-bound and the matmuls are not; this is the entire reason kernel fusion exists. Second, **the attention score computation is not counted**, because its cost depends on T and not on N — I'll come back to when that matters. Third, the **embedding lookup is a gather, not a matmul**, so the embedding table's parameters cost ~0 FLOPs at the input side (they do cost 2·d·V at the output side, via the LM head, which is why the head is often the single most expensive matmul in a small model).

**📐 Numbers you must know:** forward = 2N per token, backward ≈ 4N per token, so training = 6N per token. Total training compute C ≈ 6ND for N parameters and D tokens. Inference prefill = 2N per token; inference decode = 2N per *generated* token. These five numbers let you estimate anything in this field on a napkin, and interviewers will ask you to.

**⚠ Trap:** applying 2N to a mixture-of-experts model using total parameter count. MoE decouples parameters from compute — that's the entire point of the architecture. A model with 400B total parameters and 30B active per token costs 2 × 30e9 = 6e10 FLOPs per token, not 8e11. When you cite a FLOP number for an MoE, say "active parameters" explicitly. And note the flip side: the *memory* footprint follows total parameters, so an MoE is cheap in compute and expensive in HBM — the opposite trade from what people assume.

### And why 6N for training? Where do the other 4N come from?

Backprop through a linear layer requires two matmuls where the forward required one, and each is the same size as the forward. That's the whole answer, but you should be able to name both matmuls.

For y = xW, given the incoming gradient ∂L/∂y you must compute two things:

- **∂L/∂x = (∂L/∂y)Wᵀ** — needed to keep propagating backward to the previous layer. Contracts over d_out, costs 2·d_in·d_out per token.
- **∂L/∂W = xᵀ(∂L/∂y)** — needed to actually update the weights. Contracts over the batch-and-time axis, costs 2·d_in·d_out per token.

Two matmuls, each the same FLOP cost as the forward's one. Forward 2N + backward 4N = **6N per token**, and multiplying by D tokens gives Kaplan's C ≈ 6ND. The optimizer step itself is elementwise — Adam is a handful of ops per parameter, so ~10N FLOPs per *step*, not per token, and at a batch of a million tokens that's five orders of magnitude below the matmuls. Ignore it in FLOP accounting; do not ignore it in memory accounting.

Two refinements worth having ready.

**Gradient checkpointing changes the constant.** If you discard activations in the forward and recompute them during the backward, you pay an extra forward pass: 8N per token instead of 6N, a 33% compute increase. In practice full recompute lands closer to +30–40% wall-clock, and *selective* recompute — recomputing only the cheap-to-recompute, expensive-to-store tensors like the attention softmax output, while keeping the matmul outputs — lands closer to +5–10% for most of the memory saving. Selective recompute is what modern training stacks actually do.

**The first layer's ∂L/∂x is wasted work.** You compute the gradient with respect to the input embeddings at layer 0 and then throw it away (unless you're doing input-gradient-based attribution). It's one layer out of 32; nobody optimizes it.

**🗣 Say this in the room:** "Forward is 2N per token because every parameter is one multiply-accumulate. Backward is 4N because each linear layer needs two gradient matmuls — one for the input gradient and one for the weight gradient — each the same size as the forward. Six total, so C ≈ 6ND. Gradient checkpointing pushes it to 8N with full recompute, or about 6.5N with selective."

### Estimate the GPU-days to pretrain a 7B model on 1 trillion tokens. Show me the arithmetic.

**💰 Math, step by step:**

1. **Total compute.** C = 6ND = 6 × 7×10⁹ × 1×10¹² = **4.2 × 10²² FLOPs**.
2. **Per-GPU throughput.** An H100 SXM does roughly 990 TFLOP/s dense bf16 (ignore the sparsity-doubled marketing number; you will not get it). Call it 9.9 × 10¹⁴ FLOP/s peak.
3. **Apply MFU.** Model FLOPs Utilization for a well-tuned dense pretraining run at this scale is typically 35–50%. Take **40%**: 9.9 × 10¹⁴ × 0.40 = 3.96 × 10¹⁴ FLOP/s achieved.
4. **Divide.** 4.2 × 10²² / 3.96 × 10¹⁴ = **1.06 × 10⁸ GPU-seconds** = 29,500 GPU-hours = **≈ 1,230 H100-days**.
5. **Wall-clock.** On a 256-GPU cluster: 1,230 / 256 ≈ **4.8 days**. On 1,024 GPUs: ~1.2 days, minus the scaling efficiency you lose to communication — assume 85–90% at that width, so call it 1.4 days.
6. **Dollars.** At a market rate of roughly $2/H100-hour on a reserved cluster: 29,500 × $2 = **≈ $59,000** for the compute alone. **📅 Volatile:** H100 hourly rates have moved a lot and continue to; verify the current number before quoting it.

Then add the things the naive estimate omits, because the interviewer's follow-up is always "and what did you leave out?" Data preparation and tokenization is a large CPU job. Failed runs — the honest multiplier on a first-time-through pretraining effort is 1.5–3× total spend, because you will restart from a loss spike, discover a data bug at 200B tokens, and re-tune the LR schedule. Checkpoint storage: a 7B checkpoint with optimizer state is ~112 GB; saving every 1,000 steps for 250,000 steps is not something you keep all of. Evaluation compute during the run. And the salaries, which dominate everything above at this scale.

**🗣 Say this in the room:** "6ND gives 4.2e22 FLOPs. An H100 at 40% MFU delivers about 4e14 FLOP/s, so that's roughly 1.1e8 GPU-seconds — about 1,200 H100-days, five days on 256 cards, and around $60k of compute at $2 a card-hour. I'd budget 2× that for failed runs, and the real cost is engineer-months, not GPU-hours."

**⚠ Trap:** quoting MFU above 60% for a dense pretraining run. That number gets reported for carefully-tuned specific configurations and it is not what you get. Quoting 70% signals you have read a blog post rather than watched a training dashboard. Conversely, if you're at 15% MFU, that is a real bug — usually a small micro-batch, a bad sequence-parallel config, or a data loader stalling the GPU, and the first thing I'd do is profile for GPU idle time before touching the model.

### When does the 2N approximation break down? At what context length does attention start to matter?

The 2N estimate counts weight FLOPs. Attention scores use no weights, so they're invisible to it — but they scale with T while weight FLOPs don't, so there is a crossover, and being able to find it is what separates "I memorized 6ND" from "I understand it."

Derive the attention term. For one query token attending over T keys, in one head: QKᵀ is T dot products of length d_h = 2·T·d_h FLOPs, and the AV weighted sum is the same, 2·T·d_h. Total 4·T·d_h per head; over H heads that's 4·T·d_model (since H·d_h = d_model); over L layers, **4·L·T·d_model FLOPs per token** in the forward pass. (Under causal masking, averaged over the whole sequence, the effective T is T/2 — I'll keep the conservative full-T version.)

Now the weight term. A standard transformer layer has ≈ 4d² attention parameters and ≈ 8d² MLP parameters, so N ≈ 12·L·d², and the weight FLOPs per token are 2N = 24·L·d². Take the ratio:

  attention / weights = (4·L·T·d) / (24·L·d²) = **T / (6·d_model)**.

That's the rule I keep in my head. **Attention FLOPs equal weight FLOPs when T ≈ 6 × d_model.**

Plug in numbers. For d_model = 4096 (a 7–8B model), crossover is at T ≈ 24,600 tokens. At T = 2,048, attention is 2048/24576 = **8% of FLOPs** — safely ignorable, which is exactly why "2N" became the standard approximation in the 2020–2022 era when 2k context was normal. At T = 32,768 it is 133% of the weight FLOPs — attention is now the *majority* of the compute. At 128k it is 5.3× the weight FLOPs, and your model's parameter count has become nearly irrelevant to prefill cost.

Two consequences that make this worth deriving rather than memorizing. First, **the crossover scales with d_model**, so bigger models tolerate longer context before attention dominates: a d = 8192 model doesn't cross over until ~49k tokens. Second, this is a *prefill* story. During decode with a KV cache, you attend over T keys but only for one new query, so the attention FLOPs per generated token are 4·L·T·d_h·H... which is the same expression, but now compared against 2N for a single token. At 128k context decode, attention is again 5× the weight compute — except decode is memory-bound, so what actually kills you is reading the cache, not the FLOPs. Different bottleneck, same T-dependence.

**⚠ Trap:** the FLOP crossover is *not* the same as the point where attention starts hurting you in production. Memory hits first. The `[B,H,T,T]` score tensor is quadratic in bytes as well as FLOPs, which is what motivated FlashAttention — it computes attention in tiles and never writes the T×T matrix to HBM, turning attention from O(T²) memory to O(T). **📄 Paper:** Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — an exact, tiled, recomputation-based attention kernel that removed the quadratic HBM traffic; it replaced both naive attention and the approximate-attention literature that had been trying to solve the wrong bottleneck.

### Give me the four-term memory equation for training, and size a full fine-tune of a 7B model.

**Mental model: model weights are the small part.** Every backend engineer's first instinct is "7B parameters × 2 bytes = 14 GB, fits on any card." The training-time footprint is eight times that, and the four terms are why.

The equation, per parameter, for standard mixed-precision AdamW:

1. **Weights** — bf16 copy used for forward/backward: **2 bytes/param**
2. **Gradients** — bf16 (sometimes fp32): **2 bytes/param**
3. **Optimizer state** — Adam's first and second moments in fp32: **8 bytes/param**; plus the fp32 master copy of the weights that the optimizer actually updates: **4 bytes/param**
4. **Activations** — everything saved in the forward for use in the backward: depends on batch × sequence × depth, *not* on parameter count

Terms 1–3 sum to **16 bytes per parameter**, and 18 if your framework also keeps fp32 gradients. For 7B:

- weights: 7e9 × 2 = **14 GB**
- gradients: 7e9 × 2 = **14 GB**
- fp32 master weights: 7e9 × 4 = **28 GB**
- Adam m and v: 7e9 × 8 = **56 GB**
- **subtotal: 112 GB of static state**, before a single activation.

An 80 GB H100 cannot hold that. This is the number to have memorized, because it instantly explains why LoRA/QLoRA became the default for anything under a research budget, and why ZeRO exists.

Activations, term 4, with gradient checkpointing at layer granularity: you store only each layer's input, so B·T·d·L·2 bytes. For B = 4, T = 4096, d = 4096, L = 32: 4 × 4096 × 4096 × 32 × 2 = **4.3 GB**. Without checkpointing you store many intermediates per layer — the norm outputs, the qkv projections, the attention output, the MLP's intermediate at d_ff = 3.5d — and the figure is an order of magnitude higher, comfortably 40–80 GB at this shape, which is why nobody trains long-sequence models without checkpointing.

**📐 Numbers you must know:** 16 bytes/param for AdamW mixed precision (2 + 2 + 4 + 8). 2 bytes/param for bf16 inference. 4 bytes/param at int32-free fp32 inference. 1 byte/param at int8, 0.5 at int4. Memorize the 16, because the follow-up question in every training round is "so how do you make it fit?"

**⚠ Trap:** forgetting the fp32 master weights. Candidates confidently recite "2 + 2 + 8 = 12 bytes" and get 84 GB, then conclude a 7B fine-tune fits on an 80 GB card — and it doesn't. Mixed precision keeps an fp32 master copy precisely because a bf16 weight has ~8 mantissa bits, and an update of relative size 10⁻⁴ against a weight of order 1 rounds to zero. That master copy is the difference between "fits" and "OOM," and knowing it is a strong tell that you've actually run a training job.

### So it doesn't fit on one 80 GB card. What do you actually change, in what order?

I'd work down this ladder, cheapest intervention first, and I would state the ladder out loud in an interview because the ordering *is* the answer.

**1. Don't full fine-tune.** For 90% of applied use cases, LoRA is the right call, and it's not a compromise — it's the correct default. Freeze the base (14 GB bf16), train r = 16 adapters over all linear layers (~40M params). Optimizer state and gradients now cost 40e6 × 16 = **0.64 GB** instead of 98 GB. Total: ~15 GB + activations, comfortably on one card with a real batch size. QLoRA quantizes the frozen base to NF4, taking it to ~4 GB and letting you fine-tune a 7B on a 24 GB consumer card.

**2. Gradient checkpointing.** As above: ~30% more compute (or ~5–10% with selective recompute) for roughly an order of magnitude less activation memory. **📄 Paper:** Chen et al. (2016), *Training Deep Nets with Sublinear Memory Cost* — showed you can train an n-layer net in O(√n) activation memory by checkpointing every √n layers and recomputing, at the cost of one extra forward pass.

**3. Cheaper optimizer state.** 8-bit Adam (bitsandbytes) cuts the moments from 8 bytes to 2, saving 42 GB on a 7B. Adafactor factorizes the second moment into row and column statistics, cutting it to O(d) per matrix instead of O(d²). Both cost some quality; 8-bit Adam costs very little in my experience and is underused.

**4. Shard across GPUs — ZeRO / FSDP.** **📄 Paper:** Rajbhandari et al. (2020), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* — partitions optimizer state (stage 1), then gradients (stage 2), then parameters themselves (stage 3) across data-parallel ranks instead of replicating them. On 8 GPUs, ZeRO-2 puts the 84 GB of gradient+optimizer state at 10.5 GB per card; ZeRO-3/FSDP shards the weights too, at the cost of all-gathering them per layer during forward and backward, which is bandwidth you need NVLink to afford.

**5. Reduce the batch, then recover it with gradient accumulation.** Activations scale linearly with micro-batch. Micro-batch 1 with 32 accumulation steps has the same effective batch and the same convergence as micro-batch 32, at a fraction of the activation memory and worse GPU utilization. This is the lever you pull last because it's the one that costs throughput most directly.

**⚠ Trap:** reaching for model parallelism (tensor or pipeline) before exhausting this list. Tensor parallelism introduces an all-reduce per layer and only makes sense inside a node with NVLink; pipeline parallelism introduces bubbles and a microbatch scheduling problem. For a 7B, if you're reaching for TP you have skipped four cheaper options. For a 70B+ you have no choice, and that's a different conversation.

### Derive the KV cache size formula and compute it for a 70B-class model at 128k context.

Mental model, and this is the bridge that lands with a backend engineer: **a KV cache is a per-request memo table whose eviction policy you do not control and whose size you cannot bound at admission time.** It exists because at decode step t, the keys and values for tokens 1…t−1 are byte-identical to what you computed at step t−1 — recomputing them turns an O(T) generation into O(T²). But unlike a Redis cache, it is not shared across requests, it grows monotonically for the life of the request, and it lives in the same HBM as your weights.

The formula, derived by counting. Per token, per layer, you store one K vector and one V vector, each of size n_kv_heads × head_dim, in your KV dtype:

  **bytes/token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element**

The leading 2 is K and V. Note what is *absent*: batch size (it's per token, so multiply later) and n_query_heads (with GQA, only the KV heads count).

Llama-3-70B-class config: L = 80, n_kv_heads = 8, head_dim = 128, bf16 (2 bytes).

  2 × 80 × 8 × 128 × 2 = 327,680 bytes = **320 KiB per token**.

At 128k context (131,072 tokens): 131,072 × 327,680 = 4.295 × 10¹⁰ bytes = **40 GiB for a single sequence**. The model weights are 140 GB in bf16, so on an 8-way tensor-parallel node with 8×80 = 640 GB total, weights take 140 GB and *one* 128k-context request takes another 40 GB. You can serve about twelve of them concurrently before HBM is gone, and that is with zero headroom for activations or fragmentation.

Now the comparison that makes the point about GQA. Llama-2-7B is MHA: L = 32, 32 KV heads, head_dim 128 → 2 × 32 × 32 × 128 × 2 = **512 KiB per token**. Llama-3-8B is GQA-8: 2 × 32 × 8 × 128 × 2 = **128 KiB per token**. **A 7B model with MHA has a 1.6× larger KV cache per token than a 70B model with GQA.** State that in an interview and watch the room reorder its priors: KV cache size is governed by architecture, not by parameter count.

**📐 Numbers you must know:** 128 KiB/token for an 8B GQA-8 model; 320 KiB/token for a 70B GQA-8 model; ×8 if the model is MHA; ÷2 if you quantize the cache to fp8. Multiply by context length, then by concurrency. This is the single most-asked arithmetic in an inference-serving round.

**💰 Math on why this is the capacity planner:** an 8B model on one H100 (80 GB) has 16 GB of weights, leaving ~60 GB for cache after activations and overhead. At 128 KiB/token that's 60e9/131072 ≈ **458,000 tokens of total cache**. Sixty-four concurrent users at 8k context each = 512k tokens — you're already over. At 4k each, 256k tokens, you fit with room. Your maximum concurrency is a division problem, and the number that comes out of it is what your autoscaler should be tracking instead of CPU.

### Follow-up: why does GQA cut the cache but not the compute? Show me.

Because grouped-query attention shrinks the K and V *projections*, and the projections were never where the FLOPs were.

Walk the two costs separately. With MHA at d = 4096, H = 32, d_h = 128, the four projections W_q, W_k, W_v, W_o are each `[4096, 4096]`, so attention holds 4d² = 67M parameters per layer. With GQA-8, W_k and W_v become `[4096, 1024]` — a quarter the size — so attention holds 2d² + 2(d²/4) = 2.5d² = 42M per layer. That is a real 37% cut in *attention* parameters, but the MLP at SwiGLU with d_ff = 14336 holds 3 × 4096 × 14336 = 176M per layer, so total per-layer parameters fall from 243M to 218M: about **10% fewer parameters and 10% fewer weight FLOPs**. Nice, not transformative.

Now the attention-score computation. You still have 32 query heads. Each still attends over all T keys. The 8 cached KV heads are *broadcast* — each is reused by 4 query heads — so the score matmul is still `[B, 32, T, 128] × [B, 32, T, 128]ᵀ`, exactly the same shape and exactly the same FLOP count as MHA. **The arithmetic is unchanged; only the operand storage shrank.** That is the entire trick: you trade a 4× reduction in bytes for zero reduction in flops, and since decode is memory-bandwidth-bound, bytes are what you were paying for.

The KV cache falls exactly 4×, from 512 KiB/token to 128 KiB/token at this config, and that is the number that shows up in your serving capacity.

**📄 Paper:** Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Query Checkpoints* — interpolated between MHA and Shazeer's (2019) multi-query attention by sharing K/V across groups of query heads, and showed you can uptrain an existing MHA checkpoint into GQA cheaply. MQA (1 KV head) was faster but lost measurable quality; GQA recovered nearly all of it at 8 heads.

**⚠ Trap:** claiming GQA "makes attention 4× faster." It does not touch attention FLOPs. It makes *decode throughput* faster, indirectly, by shrinking the bytes you read from HBM per step and by letting you hold more concurrent sequences — which raises batch size, which raises arithmetic intensity. The causal chain runs through memory, and stating it as a compute win is the tell that you memorized the outcome.

### Explain arithmetic intensity, and use it to tell me why decode is memory-bound and prefill is not.

Mental model: **a GPU is a machine with two separate budgets — FLOPs per second and bytes per second — and every kernel spends them in a fixed ratio determined by its algorithm. If your ratio is below the machine's ratio, the FLOP units sit idle waiting on memory, and no amount of a faster GPU helps.** This is the roofline model, and it is the single most useful mental tool in inference engineering.

Arithmetic intensity = FLOPs performed / bytes moved from HBM. The machine's balance point for an H100 SXM is 9.9×10¹⁴ FLOP/s ÷ 3.35×10¹² bytes/s ≈ **295 FLOPs per byte**. Below that intensity you are memory-bound; above it, compute-bound.

**Decode, batch size 1.** To generate one token you read every weight once — 2N bytes at bf16 — and perform 2N FLOPs. Intensity = 2N/2N = **1 FLOP per byte**. You are 295× below the machine balance. The tensor cores are idle 99.7% of the time and you are, precisely, a memory-copy engine that occasionally multiplies.

**Decode, batch size B.** The weights are read once and reused across all B sequences in the batch, so bytes stay ≈ 2N while FLOPs become 2N·B. Intensity ≈ **B FLOPs per byte**. So the batch size at which decode becomes compute-bound is *approximately the machine balance point*: **B ≈ 300 on an H100**. (Approximately, because KV-cache reads scale with B and don't amortize, which pushes the real crossover higher, especially at long context.) This is why continuous batching is not a nice-to-have — it is the only lever that moves you off the memory roofline.

**Prefill.** You process T tokens at once through the same weights: bytes ≈ 2N, FLOPs ≈ 2N·T. Intensity ≈ T. At T = 2,048 you are at 2,048 FLOPs/byte, far above 295, solidly compute-bound. Prefill and decode are *different workloads on the same weights*, and that asymmetry is why serving stacks separate them — chunked prefill, or full prefill/decode disaggregation onto different hardware pools.

**🗣 Say this in the room:** "Decode at batch 1 has an arithmetic intensity of about 1 FLOP per byte; an H100 needs about 295 to saturate its tensor cores. So single-stream decode uses well under 1% of the GPU's math throughput and is purely a bandwidth problem. Batching is what fixes it — intensity rises roughly linearly with batch size — which is why continuous batching and KV-cache capacity, not FLOPs, are the things I'd size a deployment around."

**⚠ Trap:** concluding "so buy a GPU with more FLOPs." If you are at intensity 1, doubling FLOP throughput changes your decode latency by zero. The levers that actually work are all bandwidth-or-bytes levers: higher-bandwidth memory (H200's 4.8 TB/s over H100's 3.35), fewer bytes per weight (int8/fp8/int4 quantization halves or quarters the read), fewer bytes per token of cache (GQA, MLA, fp8 KV), or more work per byte read (bigger batch, speculative decoding — which is the trick of verifying k draft tokens in one weight-read).

### Estimate single-stream decode throughput for a 7B model in bf16 on one H100. First principles, no benchmarks.

**💰 Math:**

1. **Bytes that must move per generated token.** At batch 1 you read every weight once: 7×10⁹ params × 2 bytes = **14 GB per token**.
2. **Time to move them.** H100 SXM HBM3 bandwidth is 3.35 TB/s = 3,350 GB/s. 14 / 3,350 = **4.18 ms per token**.
3. **Theoretical ceiling.** 1 / 0.00418 = **239 tokens/second**.
4. **Apply achieved-bandwidth efficiency.** Real kernels hit 70–85% of peak HBM bandwidth. At 78%: 239 × 0.78 ≈ **186 tok/s**.
5. **Add the KV cache read.** At 4k context with a GQA-8 7B-class model (128 KiB/token), the cache is 4,096 × 131,072 = 0.54 GB — about 3.8% on top of the 14 GB, so ~180 tok/s. At 128k context the cache is 16.8 GB, *larger than the weights*, and per-token traffic goes from 14 GB to 30.8 GB: throughput drops to roughly 186 × (14/30.8) ≈ **85 tok/s**. Same model, same GPU, 2.2× slower purely from cache traffic.
6. **Sanity check against reality.** Published single-stream numbers for 7–8B models on an H100 with a good engine land in the 100–180 tok/s range. The estimate is in the right place, which is what a first-principles estimate is for.

Two follow-ups worth pre-empting. **"How would you double it?"** Quantize the weights to fp8 or int8 — 7 GB instead of 14, so ~2× on the weight-read term, which is the dominant term at short context. That is the single highest-leverage change to single-stream latency, and it is why every latency-sensitive deployment is quantized. **"And if I need 10× ?"** You can't get it from one stream on one card; you change the problem — speculative decoding (verify k drafted tokens per weight-read, 2–3× realistic), or accept that per-stream latency is bandwidth-bound and optimize throughput per dollar via batching instead.

**⚠ Trap:** conflating tokens/second per stream with tokens/second per GPU. At batch 64 the *per-stream* rate might fall to 60 tok/s while the *aggregate* rises to 3,800 tok/s. Those are different SLOs owned by different stakeholders: the user experiences inter-token latency, finance experiences aggregate throughput per dollar. Every serving design question is a negotiation between those two, and an answer that doesn't name both is incomplete.

### Backprop is a chain of Jacobians. Explain why we never actually build one.

The mental model: **autograd computes vector-Jacobian products, never Jacobians.** Given a composite function L = f_n ∘ … ∘ f_1 (x), the chain rule says the gradient is a product of Jacobian matrices J_n J_{n−1} … J_1. Written that way it looks like you need those matrices. You don't, because you only ever need the product against a *vector* — and reverse-mode differentiation evaluates the product right-to-left starting from the scalar loss, so at every step you're computing vᵀJ, which is a vector, not a matrix.

The size argument makes it visceral. Consider the LM head: input `[T, 4096]`, output logits `[T, 128256]`. Its full Jacobian with respect to the input has 4096 × 128256 entries per token — 5.25 × 10⁸ per token, 4 bytes each, **2.1 GB per token**. For a 2,048-token sequence you'd need 4.3 TB to represent one layer's derivative. Instead, the vector-Jacobian product for y = xW is just `grad_x = grad_y @ W.T` — one matmul, no materialization, and the "Jacobian" only ever exists as the algorithm that applies it.

That's the abstraction PyTorch encodes: every op registers a `backward` that consumes an upstream gradient and returns downstream gradients. When you write `torch.autograd.Function`, the method you implement is exactly a VJP.

```python
class ScaledDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, scale):
        ctx.save_for_backward(q, k); ctx.scale = scale
        return (q @ k.transpose(-1, -2)) * scale

    @staticmethod
    def backward(ctx, g):                 # g is the upstream vector, not a Jacobian
        q, k = ctx.saved_tensors; s = ctx.scale
        return (g @ k) * s, (g.transpose(-1, -2) @ q) * s, None
```

The direction matters and is worth naming. **Reverse mode** (backprop) costs one pass per *output* and is cheap when outputs are few — a scalar loss — and inputs are many, which is exactly the training setting. **Forward mode** (JVPs) costs one pass per *input* and is cheap in the opposite regime. `torch.func.jvp` and `jacrev`/`jacfwd` expose both; you reach for forward mode when differentiating with respect to a handful of scalars, e.g. hyperparameter sensitivity.

**⚠ Trap:** using `torch.autograd.functional.jacobian` on anything but a toy. It works by running the backward pass once per output element. On a `[T, V]` output that's 262 million backward passes. People do this in notebooks, watch it hang, and conclude "autograd is slow." Autograd is not slow; asking for a Jacobian is.

### Quick matrix-calculus check. Y = XW. Give me dL/dW and dL/dX, with shapes.

This is a two-minute filter question, and the thing being tested is whether you check shapes reflexively rather than recalling a formula.

Setup: X is `[B, T, d_in]`, W is `[d_in, d_out]`, Y = XW is `[B, T, d_out]`. Let G = ∂L/∂Y, which has the same shape as Y: `[B, T, d_out]`.

**∂L/∂X = G Wᵀ.** Shape check: `[B,T,d_out] @ [d_out,d_in]` → `[B,T,d_in]`. Matches X. ✓

**∂L/∂W = Xᵀ G, summed over the batch and time axes.** Shape check: you need `[d_in, d_out]`, and you have X as `[B,T,d_in]` and G as `[B,T,d_out]`, so contract over both B and T: `torch.einsum('bti,bto->io', X, G)`. Matches W. ✓

The rule that makes this derivable rather than memorized, and the one I'd give a junior: **the gradient of a scalar loss with respect to any tensor has exactly that tensor's shape.** So there is only one way to arrange the available operands to produce the required shape, and if there's only one arrangement, you don't need to remember it — you derive it in five seconds by shape-matching. The only genuine choices are which axes to contract and which to transpose, and the shape constraint fixes both.

Two follow-ups that show up. **"Why is the W gradient summed over the batch?"** Because W is shared across all B·T token positions — every position contributed to the loss through the same weights, and the chain rule sums contributions over all paths. This is also why the weight-gradient matmul reduces over the biggest axis and is therefore the memory-traffic-heavy half of the backward. **"What if there's a bias?"** ∂L/∂b = G summed over B and T, giving `[d_out]` — same logic, same reduction.

**🏋 Drill:** 6 minutes, blank page, no autocomplete. Write the forward and backward for a two-layer MLP with GELU by hand — `h = gelu(xW1 + b1); y = hW2 + b2` — and produce all six gradients with their shapes. Then verify against autograd with `torch.autograd.gradcheck` on float64 inputs of shape `[4, 8]`. Pass criterion: gradcheck returns True on your first submission. Most people get W2 and b2 right and fumble the GELU derivative chain into W1 — that's the part worth rehearsing.
### Someone says "softmax is just a Boltzmann distribution." Unpack that, and tell me what temperature physically is.

Mental model: **the softmax is the maximum-entropy distribution consistent with a set of energies, and the logits are negative energies.** In statistical mechanics, a system in thermal equilibrium occupies state i with probability p_i ∝ exp(−E_i / kT). Set E_i = −z_i and absorb k, and you get p_i ∝ exp(z_i/T), which is the softmax with temperature. That is not an analogy — it is the same functional form derived from the same variational principle (maximize entropy subject to a fixed expected energy), and knowing that is why the temperature parameter behaves the way it does rather than being a magic knob.

The physics gives you the intuition for free. **Temperature is inverse inverse-coldness**: write β = 1/T and p_i ∝ exp(βz_i). Large β (low temperature) means the system settles into its lowest-energy state — the argmax. Small β (high temperature) means thermal noise swamps the energy differences and every state becomes equiprobable. The entropy of the distribution is monotonically increasing in T. In sampling terms:

- **T → 0⁺:** the distribution converges to a point mass on argmax(z). Greedy decoding.
- **T = 1:** the model's calibrated distribution, i.e. the one it was trained to produce by minimizing cross-entropy.
- **T → ∞:** uniform over the vocabulary. Every logit difference is divided into irrelevance.

The mechanism at the tensor level is trivially cheap: divide the logit vector by T before the softmax. One elementwise op on a `[V]` vector, ~128k FLOPs, utterly free compared to the forward pass that produced the logits.

There is one non-obvious consequence worth stating unprompted, because it is what makes temperature different from top-p. **Temperature is a monotone transformation — it never changes the ranking of tokens, only the gaps.** So T can never make an impossible token likely relative to another; it can only redistribute mass along the existing order. Top-k and top-p, by contrast, are *truncations* — they set tails to exactly zero, changing the support. That's why the two compose usefully: top-p removes the garbage tail, temperature reshapes what's left.

**⚠ Trap:** treating temperature as a "creativity" dial in a design doc without saying what it does to the failure distribution. Raising T raises entropy, and in a structured-output or tool-calling path the extra entropy lands on schema violations and hallucinated argument names. My rule in review: any code path that parses the model's output into a typed object runs at T = 0 unless there is an eval showing otherwise, and any path whose output a human reads may run hotter.

### What actually happens at temperature 0, and why do people say "temperature 0 is a lie"?

At T = 0 the expression exp(z_i/T) is a division by zero, so no provider literally computes it — the implementation branches to `argmax` and skips sampling entirely. That part is fine. The lie is in what people *infer* from it: that T = 0 gives you a deterministic, reproducible function from prompt to output. It does not, and this catches backend engineers hard because determinism is something we're used to being able to demand.

The sources of nondeterminism, in the order I'd check them:

**Floating-point non-associativity under variable batching.** A GPU reduction sums partial products in an order determined by how the kernel splits work across thread blocks, and that split depends on the shape of the batch. Your request landing in a batch of 3 versus a batch of 47 produces a *different summation order*, and (a+b)+c ≠ a+(b+c) in floating point. The logits differ in the last few bits. Usually irrelevant — but when the top two tokens are within that epsilon, argmax flips, and because generation is autoregressive, one flipped token at position 40 produces a completely different completion from position 41 onward. **This is the dominant cause, and it is invisible in single-request testing** because you only see it under production batching.

**Mixture-of-experts routing.** In implementations with per-expert capacity limits, which tokens get routed where depends on the other tokens in the batch. Your token can be dropped from its preferred expert because someone else's request filled it. Batch composition becomes an input to your output.

**Atomics and non-deterministic kernels.** Some reduction kernels use atomic adds whose ordering is not fixed run to run.

**Infrastructure drift.** A different GPU SKU, a different engine version, a different tensor-parallel degree, or a silently-updated model behind a stable API name — all change the numerics. **📅 Volatile:** provider-side model updates behind an unversioned alias are a real and recurring cause of "my prompt stopped working"; pin versioned model IDs where the provider offers them.

**🗣 Say this in the room:** "Temperature 0 is greedy decoding, not determinism. Floating-point reductions aren't associative, and the reduction order depends on batch shape, so the same prompt in a batch of 3 and a batch of 50 can produce different logits in the last bits. When the top two tokens are close, argmax flips and autoregression amplifies it into a totally different answer. So I never write a test that asserts exact output equality — I assert on a property, a schema, or a scored threshold."

**⚠ Trap:** golden-output snapshot tests in CI against a hosted model. They pass on your machine, then flake at 2% in CI, and the team's response is to add retries — which hides a real regression the day the provider does update the model. Test properties and evals, not strings. This is the single most common testing mistake a strong backend engineer makes on their first LLM system.

### Define entropy and cross-entropy in the units the training loop actually prints. Connect them to the loss I see on the dashboard.

Entropy H(p) = −Σ_i p_i log p_i is the average surprise of a distribution — the expected number of nats (if log is natural) or bits (if log₂) needed to encode a sample from p under an optimal code. Cross-entropy H(p, q) = −Σ_i p_i log q_i is the average cost of encoding samples from p using a code built for q. It is always ≥ H(p), and the gap is exactly the KL divergence: **H(p,q) = H(p) + D_KL(p‖q)**. That identity is the whole reason "minimizing cross-entropy" and "matching the data distribution" are the same sentence.

In a language model, p is the empirical one-hot target — the actual next token — so H(p) = 0 and cross-entropy *equals* the KL divergence from the data to the model. The per-token loss collapses to −log q(y_true), and the reported training loss is the mean of that over the batch:

  loss = −(1/N) Σ_{t} log q_θ(y_t | y_{<t}),  **in nats**, because PyTorch's `cross_entropy` uses natural log.

That gives you the conversion table you need for reading a dashboard:

- **loss in nats × 1.4427 = loss in bits** (1/ln 2).
- **perplexity = exp(loss_in_nats)**.
- Loss 2.0 nats = 2.885 bits/token = perplexity 7.39.
- Loss 1.6 nats = 2.31 bits/token = perplexity 4.95.
- **A uniform model over a 128k vocabulary has loss ln(128256) = 11.76 nats.** That's your step-0 sanity check: if your loss doesn't start within a hair of ln(V), your initialization or your label alignment is broken. This one check has saved me more debugging hours than any other.

Two subtleties that come up. **Masking** — padding positions and prompt tokens in an instruction-tuning setup must be excluded via `ignore_index`, and getting that wrong silently rescales your loss by the fraction of masked tokens, so your loss curve looks fine and your model learns to predict padding. **Reduction** — `mean` over a batch with variable-length sequences weights short sequences more per token than a global token-mean does; for gradient-accumulation correctness you want to sum the token losses and divide by the global token count, not average the per-microbatch means.

**⚠ Trap:** comparing loss numbers across runs with different tokenizers or different sequence packing. Loss is per-token, and "a token" is not a fixed unit. More on this below — it is the single most common information-theory error in this field.

### Forward KL versus reverse KL. Which one does pretraining use, which one does RLHF use, and what's the behavioral difference?

Mental model: **forward KL is a coward and reverse KL is a specialist.** Forward KL punishes you for assigning low probability to anything the target does; reverse KL punishes you for assigning probability to anything the target doesn't. One makes you hedge, the other makes you commit.

The asymmetry falls straight out of which distribution is doing the weighting.

**Forward KL, D(p ‖ q) = Σ p log(p/q).** The expectation is under p, the target. Wherever p has mass and q is near zero, log(p/q) → ∞ and the loss explodes. So q is *forced to cover every mode of p* — it is **zero-avoiding / mode-covering**. Where p has no mass, q is unpenalized for putting some there. Result: a broad, hedging q that spreads mass over the union of the target's modes and smears across the valleys between them.

**Reverse KL, D(q ‖ p) = Σ q log(q/p).** The expectation is under q, your own model. Wherever q has mass and p is near zero, you pay heavily — so q is **zero-forcing / mode-seeking**. But wherever p has mass and q has none, the integrand is 0·log 0 = 0 and there is no penalty at all. Result: q collapses onto one high-probability mode of p and confidently ignores the rest.

Where each lives:

- **Pretraining and SFT use forward KL.** Maximum-likelihood on next-token prediction is exactly minimizing D(p_data ‖ q_θ). This is why base models hedge — they are trained to cover, not to commit, and it's the information-theoretic root of "the model gives you a wishy-washy answer that mentions every possibility."
- **RLHF's regularizer uses reverse KL.** The PPO objective is maximize E_{q_θ}[r(x)] − β·D(q_θ ‖ π_ref), with the expectation under the *learned* policy — mode-seeking with respect to the reference. That is intentional: you want the aligned policy to commit to the high-reward mode, not to keep covering everything the base model would have said.
- **Distillation defaults to forward KL** (match the teacher's full distribution) and there is a real line of work that switches to reverse KL specifically to stop students from hedging over teacher modes they don't have the capacity to represent. When a small student must approximate a big teacher, forward KL makes it smear and reverse KL makes it pick — and for generation quality, picking usually wins.
- **DPO's implicit reward is a KL artifact.** The closed-form optimum of "maximize reward subject to a reverse-KL leash" is π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β). Rearranged, r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + const. That log-ratio *is* the reward, which is what lets DPO train directly on preference pairs with no separate reward model.

**🗣 Say this in the room:** "Forward KL takes the expectation under the data, so it's zero-avoiding — the model must cover every mode, which is why MLE-trained base models hedge. Reverse KL takes the expectation under the model, so it's zero-forcing — the model commits to one mode and ignores the rest. That's why RLHF uses reverse KL as its leash: you want the aligned policy to pick the high-reward mode, not to keep covering everything."

**⚠ Trap:** calling KL a distance. It is not symmetric and does not satisfy the triangle inequality. Saying "the KL distance between the models" in a room with a research engineer is a small but real credibility cost, and the asymmetry is exactly the thing that makes it useful here.

### Where does the KL term actually sit in an RLHF objective, and what breaks if I set β too low or too high?

The objective is: maximize over θ, E_{x∼D, y∼π_θ(·|x)} [ r_φ(x, y) ] − β · D_KL(π_θ(·|x) ‖ π_ref(·|x)). The reward model r_φ scores completions; π_ref is the frozen SFT checkpoint; β is the leash length. In implementation, the KL is not computed exactly over the vocabulary at every position (that would be a `[T, 128k]` reduction per sample, and it's usually estimated on the sampled tokens instead) and it is commonly folded into the per-token reward as a shaping term rather than kept as a separate loss.

**β too low** — the leash is long, and you get **reward hacking**. The policy discovers regions of output space where the reward model is wrong, because the reward model was trained on completions from π_ref's distribution and has no idea what to do off-distribution. The classic signatures: outputs get longer and longer (length is correlated with human preference in most preference datasets, so the RM learned "longer = better"), the model develops verbal tics that happen to score well, sycophancy increases, and eventually the text degrades into something that scores 9/10 on the RM and is unreadable. Your monitoring signal is **the gap between RM score and held-out human/judge score widening over training steps** — RM score climbing while your independent eval falls is the definitive diagnosis.

**β too high** — the leash is short, and nothing happens. The policy stays glued to π_ref, KL stays near zero, the reward barely moves, and you've spent a lot of GPU-hours producing your SFT model back. Less obviously, an overly tight leash can *look* like it's working for a while and then plateau early, which people misdiagnose as "the reward model isn't good enough."

The practical loop: **treat KL as a controlled variable, not a hyperparameter.** Plot KL(π_θ ‖ π_ref) against training step and against reward. What you want is a monotone reward increase at bounded KL — a "reward per nat of KL" efficiency curve. Many production setups use an adaptive controller that adjusts β to hold KL near a target rather than fixing β. And regardless of β, keep a held-out eval that the reward model never saw, evaluated by something other than the reward model, as the actual stopping criterion.

**⚠ Trap:** believing GRPO/DPO-style methods removed this problem. They moved it. DPO's β is the same leash — set it too low and you get degenerate outputs that maximize the implicit reward margin; the failure just looks different (probability mass collapsing off *both* chosen and rejected responses is a documented DPO pathology). Any method with a reference-model regularizer has a β, and a β always has these two failure modes at its ends.

### What's Jensen–Shannon divergence, and why does it barely show up in LLM training?

JSD is the symmetrized, bounded cousin of KL: with m = (p + q)/2, JSD(p, q) = ½D(p‖m) + ½D(q‖m). Two properties make it attractive on paper. It is symmetric, so there's no forward/reverse choice to argue about. And it is bounded — by log 2 with natural log, or exactly 1 bit with log₂ — so it never explodes when the supports disagree, unlike KL, which goes infinite the moment p has mass where q has none. Its square root, √JSD, is a genuine metric satisfying the triangle inequality.

It barely shows up in LLM training for a simple reason: **the boundedness that makes it safe also makes its gradients weak exactly where you need them strong.** If your model assigns near-zero probability to the true token, forward KL screams; JSD shrugs, because the mixture m still has half the target's mass there. For maximum-likelihood training you *want* the unbounded penalty — that's the learning signal. JSD's boundedness was the reason it appeared in the GAN literature (where an unbounded objective destabilizes the discriminator), and LLM training doesn't have that problem.

Where you do see it in an applied LLM stack is **outside the training loop, as a measurement**:

- **Distribution drift monitoring.** Compute the token or embedding-cluster distribution of this week's production traffic against last month's; JSD gives you a bounded, symmetric, interpretable-in-bits number you can alert on. KL would go infinite the first time a new token appears.
- **Diversity metrics** across generations — JSD between the n-gram distributions of two decoding configurations.
- **Ensemble/routing agreement** — how far apart are two models' output distributions on the same prompt, as a routing or escalation signal.

**⚠ Trap:** alerting on JSD without a baseline distribution of JSD. A JSD of 0.05 bits means nothing in isolation — you need the week-over-week JSD distribution during a known-healthy period to know whether 0.05 is Tuesday or an incident. This is the same discipline as any drift metric; the failure mode is a threshold picked by vibes.

### Mutual information — give me a real use for it in an LLM system.

I(X;Y) = H(X) − H(X|Y) is "how many bits does knowing Y save you when encoding X," and equivalently D_KL(p(x,y) ‖ p(x)p(y)) — the KL between the joint and the product of marginals, so it's exactly zero iff X and Y are independent. It's symmetric, non-negative, and measures *any* dependence, not just linear correlation, which is what distinguishes it from correlation coefficients.

Four places it genuinely earns its keep in this stack:

**Contrastive embedding training.** The InfoNCE objective — the loss behind essentially every modern text embedding model — is a lower bound on the mutual information between the two views (query and positive passage). Maximizing InfoNCE over a batch of size K is maximizing a bound that saturates at log K, which is the formal reason **larger batch sizes produce better embedding models**: the bound you can achieve is capped by log(batch size), so a batch of 64 caps you at 6 bits of MI regardless of how good your encoder is. **📄 Paper:** van den Oord et al. (2018), *Representation Learning with Contrastive Predictive Coding* — introduced InfoNCE and proved the log-K mutual-information bound; it replaced ad-hoc triplet losses as the standard contrastive objective.

**Pointwise mutual information in retrieval.** PMI(term, doc) = log[p(term,doc)/(p(term)p(doc))] is, structurally, what IDF weighting approximates. When you're debugging why BM25 beats your dense retriever on a corpus, the answer is usually that the corpus has high-PMI rare terms — part numbers, error codes, proper nouns — that the embedding model has smeared into a generic subspace.

**Eval-set construction and contamination checks.** MI between a candidate feature (query length, language, tenant) and the outcome (task success) tells you which slice actually explains your failures, which is how you decide what to stratify your eval on rather than guessing.

**Deduplication and diversity in data curation.** Selecting a training or few-shot subset that maximizes information about the target task while minimizing redundancy among the selected items is, formally, an MI-maximization problem, and submodular selection algorithms operationalize it.

**⚠ Trap:** estimating MI from a few thousand samples in a high-dimensional continuous space and believing the number. MI estimation is notoriously biased upward with limited samples, and the neural MI estimators have known variance pathologies. Use it as a *ranking* signal between candidate features on the same sample size, not as an absolute quantity you report to a stakeholder.

### Define perplexity, derive it from cross-entropy, and tell me what perplexity 8 actually means.

Perplexity is the exponentiated mean negative log-likelihood:

  **PPL = exp( −(1/N) Σ_{t=1}^{N} log p(x_t | x_{<t}) ) = exp(mean NLL in nats) = 2^(mean NLL in bits)**.

The derivation is one line: the model's per-token cross-entropy loss *is* the mean NLL, so PPL is just `exp(loss)`. Which is why, on any PyTorch training run, `math.exp(loss.item())` is your perplexity and you never need a separate metric.

The interpretation is the part worth being crisp about. **Perplexity is the effective branching factor** — the size of the uniform distribution that would leave you equally uncertain. PPL = 8 means that, averaged over the corpus, the model is as uncertain as if it were choosing uniformly among 8 equally-likely tokens at each position. A model with PPL = 1 predicts the text perfectly. A model that hasn't learned anything on a 128k vocabulary has PPL = 128,256 (loss ln V = 11.76 nats).

Some anchors for calibration. A well-trained modern model on general English web text lands somewhere in the single digits to low teens on a held-out sample of its own distribution — but that range is nearly meaningless without knowing the tokenizer and the eval corpus, which is the next question. On a *very* predictable corpus (repetitive logs, structured JSON), perplexity in the low single digits is normal. On a corpus of dense technical prose or a language underrepresented in training, double digits.

Two mechanical cautions. **Perplexity is corpus-dependent to a degree people underestimate** — the same model on Wikipedia versus on a private codebase can differ by 3× — so PPL numbers are only comparable within a fixed eval corpus. And **stride matters**: evaluating a long document by chopping it into disjoint 2,048-token windows means every window's first tokens are predicted with no context, inflating perplexity. The correct procedure is a sliding window with a stride, scoring only the tokens that had full context. Getting the stride wrong is a common way to "discover" that your fine-tune made the model worse.

### I have two models. Model A has perplexity 6.2 and Model B has 7.9. Which one is better?

**You cannot tell, and if the two models have different tokenizers, the question is not just unanswerable but malformed.** This is the flagship information-theory trap in this field and it is asked deliberately.

Here's why. Perplexity is per *token*, and a token is not a unit of language — it is a unit of a particular BPE vocabulary. A tokenizer that produces longer tokens has fewer, harder predictions; a tokenizer that produces shorter tokens has more, easier ones. The *total* information content of the text is fixed; how you slice it into prediction steps is not.

**💰 Worked demonstration.** Suppose two models are *exactly equally good* — both compress the eval corpus to 0.75 bits per byte. Model A's tokenizer averages 3.5 bytes per token; Model B's averages 4.5.

- Model A's per-token loss = 0.75 bits/byte × 3.5 bytes/token = 2.625 bits → PPL = 2^2.625 = **6.17**.
- Model B's per-token loss = 0.75 × 4.5 = 3.375 bits → PPL = 2^3.375 = **10.37**.

Identical modeling quality; Model B's perplexity is 68% higher. Under the naive reading you'd ship A and be wrong. In fact if A had *better* PPL than B on the same corpus with those tokenizers, A could still be the worse model — you'd need to convert to a common unit to know.

So what do I actually ask before answering?

1. **Same tokenizer?** If not, convert to bits-per-byte and compare that. If yes, proceed.
2. **Same eval corpus, byte-for-byte, with the same preprocessing?** Different whitespace normalization alone moves PPL by percent.
3. **Same stride and context length?** A 2,048-window evaluation and an 8,192-sliding-window evaluation of the same model give different numbers.
4. **Is the eval corpus in either model's training data?** Contamination makes perplexity arbitrarily low and is the easiest metric in the world to game.
5. **And the real question: does perplexity predict what I care about?** Below a certain quality floor it correlates well with downstream capability; among frontier-adjacent models it is a weak predictor of instruction-following, tool use, or reasoning. An RLHF'd chat model typically has *worse* perplexity on raw web text than its own base model, because alignment moved it off the pretraining distribution. It is not worse.

**🗣 Say this in the room:** "I'd refuse to answer until I know whether they share a tokenizer. Perplexity is per token and tokens aren't a fixed unit — two identically-good models can differ 60% in perplexity purely from average token length. I'd convert both to bits-per-byte, which is tokenizer-invariant, and even then I'd only use it as a pretraining health metric, not as a decision metric for a product model."

**⚠ Trap:** the corollary trap, which is subtler and gets senior candidates: using perplexity to compare a base model against its own RLHF'd or instruction-tuned descendant. Same tokenizer, same corpus, and the comparison is still invalid, because post-training deliberately shifts the output distribution away from the raw-text distribution the perplexity is measured against. Perplexity going up after post-training is expected, not a regression.

### Then give me bits-per-byte. Derive it and show me the conversion from a training loss.

Bits-per-byte is the tokenizer-invariant version of perplexity: instead of "how surprised am I per token," it asks "how many bits do I need per byte of the original UTF-8 text." Since bytes are a property of the text and not of your vocabulary, two models with different tokenizers become directly comparable. It is, literally, a compression rate — a model with 0.75 BPB compresses the corpus to 0.75/8 = 9.4% of its original size, which is a legitimate and rather beautiful way to think about what a language model is.

The derivation. Total negative log-likelihood over the corpus, in nats, is Σ_t NLL_t. Convert to bits by dividing by ln 2. Divide by the number of *bytes* in the raw text:

  **BPB = (Σ_t NLL_t) / (ln 2 × n_bytes) = mean_NLL_nats / (ln 2 × bytes_per_token)**

where bytes_per_token = n_bytes / n_tokens for that tokenizer on that corpus.

Worked both directions:

- Training loss 2.0 nats/token, tokenizer averaging 4.0 bytes/token → BPB = 2.0 / (0.6931 × 4.0) = 2.0 / 2.7726 = **0.721 bits/byte**.
- Same model, and you want the perplexity a different tokenizer would report at 3.2 bytes/token → per-token loss = 0.721 × 0.6931 × 3.2 = 1.599 nats → **PPL = e^1.599 = 4.95**. Same model, same quality, PPL drops from 7.39 to 4.95 by changing nothing but the tokenizer.

```python
import math
def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    return total_nll_nats / (math.log(2) * n_bytes)

# accumulate over the eval set:
#   total_nll_nats += F.cross_entropy(logits, targets, reduction="sum").item()
#   n_bytes        += len(raw_text_chunk.encode("utf-8"))
```

The implementation detail people get wrong: **n_bytes must be the byte length of the raw text, not of the decoded tokens after any normalization.** If your tokenizer strips or adds whitespace, or your loader lowercases, you're dividing by the wrong denominator and your "invariant" metric isn't. Compute the byte count from the original file, before any preprocessing touches it.

**📐 Numbers you must know:** BPB and PPL relate as BPB = log₂(PPL) / bytes_per_token. Typical BPE tokenizers for English average roughly 3.5–4.5 bytes per token; code tokenizers on code are lower; non-Latin scripts under an English-centric tokenizer can drop toward 1–2 bytes/token, which is simultaneously a perplexity artifact and a real cost problem — the same paragraph costs 3–4× more tokens in Hindi than in English on many tokenizers, which is a billing and context-budget issue, not just an academic one.

### Derive the Gumbel-max trick and tell me why anyone bothers.

The claim: if g_1…g_V are i.i.d. Gumbel(0,1) noise, then **argmax_i (z_i + g_i) is distributed exactly as Categorical(softmax(z))**. You sample from a softmax without ever computing the softmax.

Generating the noise is one line: if U ~ Uniform(0,1), then G = −log(−log U) is Gumbel(0,1). So:

```python
def gumbel_sample(logits, temperature=1.0):
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-20)))
    return ((logits / temperature) + g).argmax(-1)
```

Sketch of the proof, which is what they want if they ask you to derive it. The Gumbel CDF is F(x) = exp(−e^{−x}). For argmax to be i, you need z_i + g_i > z_j + g_j for all j ≠ i. Condition on the value of z_i + g_i = m: each other term must be below m, contributing ∏_{j≠i} exp(−e^{z_j − m}). Multiply by the density of z_i + g_i at m, integrate over m, and the exponentials collect into exp(z_i)/Σ_j exp(z_j). The Gumbel is precisely the max-stable distribution that makes this integral collapse — that's why this particular noise and no other.

Why it matters:

- **Reparameterization.** The argmax is not differentiable, but replacing it with a softmax over (z + g)/τ gives the Gumbel-Softmax / Concrete relaxation — a differentiable, temperature-controlled approximation to a categorical sample. That is how you backprop through a discrete choice, which is how differentiable MoE routing and various discrete-latent models are trained. **📄 Paper:** Jang, Gu & Poole (2017) and, concurrently, Maddison, Mnih & Teh (2017) introduced the Gumbel-Softmax / Concrete distribution, replacing high-variance REINFORCE estimators for categorical latents with a low-variance biased one.
- **Sampling without replacement.** Take the **top-k** of z + g rather than the top-1, and you get an exact sample of k distinct items from the categorical without replacement. That's the basis of stochastic beam search, and it's a genuinely useful trick for diverse candidate generation in a best-of-n pipeline.
- **Decoupling randomness from the distribution.** Because the noise is drawn independently of the logits, you can fix the Gumbel noise per position with a seed and get reproducible sampling that is *still* a correct sample. That's operationally useful for debugging a sampler and for A/B tests where you want the same random draw across two model variants.

**⚠ Trap:** believing Gumbel-max is faster than ordinary softmax sampling. It isn't, meaningfully — both are O(V) and neither is your bottleneck. It also *worsens* numerical behavior if you're careless: the double log needs a clamp on the uniform draw or you'll get infinities. Its value is structural (differentiability, without-replacement sampling, reproducibility), not performance.

### Write rejection sampling, prove it's correct, and then show me that speculative decoding is exactly this.

**Plain rejection sampling.** You want samples from p but can only sample from q. If you know a constant M with p(x) ≤ M·q(x) everywhere, then: draw x ~ q, draw u ~ Uniform(0,1), accept x if u ≤ p(x)/(M q(x)), else redraw. Accepted samples are exactly distributed as p, and the acceptance probability is 1/M. Correctness: P(accept and X = x) = q(x)·p(x)/(M q(x)) = p(x)/M, so conditioning on acceptance normalizes to p(x). The whole scheme is a way to convert "I can evaluate p up to a constant" into "I can sample from p," and its cost is the 1/M acceptance rate, which is why it's useless in high dimensions where M is astronomical.

**Speculative decoding is a modified rejection sampler**, and this is the part that makes the algorithm *correct* rather than merely fast — which is the point interviewers are probing. A small draft model q proposes k tokens; the large target model p scores all k in a single forward pass (cheap, because verifying k tokens is one batched prefill, not k decode steps). For each drafted token x in order:

- Accept with probability **min(1, p(x)/q(x))**.
- On the first rejection, discard the remaining drafts and **sample the replacement token from the normalized residual distribution, proportional to max(0, p(x) − q(x))**.

The residual step is the non-obvious part, and it's what makes the output distribution exactly p. The proof is short enough to say out loud: the probability of emitting token x is

  P(x) = q(x)·min(1, p(x)/q(x)) + P(reject)·[p(x) − q(x)]₊ / Σ_y [p(y) − q(y)]₊.

The first term is min(q(x), p(x)). The rejection probability is 1 − Σ_y min(p(y), q(y)) = Σ_y [p(y) − q(y)]₊, which cancels the denominator of the second term exactly. So P(x) = min(p(x), q(x)) + [p(x) − q(x)]₊ = p(x). **The output is drawn from the target model's distribution, exactly — not approximately.** Speculative decoding is lossless, and that's the claim that lets you deploy it without re-running your evals.

**📄 Paper:** Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers via Speculative Decoding*, and concurrently Chen et al. (2023), *Accelerating Large Language Model Decoding with Speculative Sampling* — both established the modified-rejection scheme that guarantees the target distribution while amortizing the target model's forward pass over multiple tokens. It replaced the assumption that autoregressive decoding must be strictly sequential in target-model forward passes.

**💰 Math on the speedup.** With per-token acceptance rate α and k drafted tokens per verification round, the expected number of tokens emitted per round is (1 − α^{k+1})/(1 − α). At α = 0.8 and k = 4: (1 − 0.8⁵)/(0.2) = (1 − 0.328)/0.2 = **3.36 tokens per target forward pass**. If the draft model costs 10% of the target per token, the cost per round is 1 target pass + 4 draft passes = 1 + 0.4 = 1.4 target-equivalents, so throughput improves by 3.36/1.4 = **2.4×**. Push α down to 0.6 and it falls to (1−0.078)/0.4 = 2.31 tokens per round, 2.31/1.4 = 1.65×. **Acceptance rate is the whole ballgame**, and it's what you should measure first when speculative decoding underdelivers.

**⚠ Trap:** assuming the speedup is free. It is free in *quality* (exactly lossless) but not in *memory* — you're holding a second model and a second KV cache in HBM — and not in *throughput at high batch*, because when you're already compute-bound from large batches, the extra draft FLOPs cost you. Speculative decoding is a latency optimization for low-to-medium batch regimes, and turning it on under heavy load can make aggregate throughput worse.

### Importance sampling — where does it show up in RLHF, and what's its failure mode?

The identity is E_{x∼p}[f(x)] = E_{x∼q}[f(x)·p(x)/q(x)]. You wanted an expectation under p, you only have samples from q, so you reweight by the likelihood ratio. That's it — a correction for evaluating one distribution's expectation using another's samples.

In RLHF this is the entire reason PPO is allowed to reuse rollout data. You generate trajectories from the current policy π_old, then take several gradient steps. After the first step, θ has moved, so your data is off-policy with respect to π_θ. Importance sampling corrects it: the policy-gradient term becomes r_t(θ)·Â_t where **r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)** and Â_t is the advantage. GRPO uses the same ratio with a group-normalized advantage in place of a learned value function.

**The failure mode is variance, and it is severe.** The importance weight is a ratio of probabilities, and ratios of small numbers are wild. If π_θ has drifted so that a sampled token now has 20× the probability it had under π_old, that one token contributes a weight of 20 to the gradient and dominates the batch. In a sequence of hundreds of tokens, per-sequence products of ratios explode or vanish exponentially in length. The formal diagnostic is **effective sample size**, ESS = (Σw_i)²/Σw_i², which tells you how many of your N samples are actually contributing; when ESS collapses to a handful, your gradient estimate is a few lucky trajectories with enormous weights and training destabilizes.

The mitigations, and each is worth naming:

- **Clipping.** PPO's clipped surrogate, min(r_t Â_t, clip(r_t, 1−ε, 1+ε)Â_t) with ε ≈ 0.2, hard-bounds the weight's influence. It's a biased estimator, deliberately — bounded bias in exchange for bounded variance is the right trade here.
- **Few epochs per rollout batch.** The further θ drifts from θ_old, the worse the ratios. One to four epochs is standard; ten is asking for it.
- **The KL leash** discussed above does double duty — it also keeps ratios near 1.
- **Token-level rather than sequence-level ratios**, so you never multiply hundreds of ratios together.

**🔍 Failure taxonomy — RL run destabilizes mid-training:** (1) Plot the distribution of importance ratios per batch; if the tail exceeds ~3–5, the policy has drifted too far — reduce epochs per batch or raise β. (2) Plot the fraction of tokens hitting the clip boundary; above ~20% you are mostly training on a clipped, biased signal. (3) Plot ESS; a collapse precedes the loss blowup by many steps and is your best early warning. (4) **Check for a sampling/training numerics mismatch** — if you generate rollouts with an inference engine and compute log-probs in a training framework, kernel and precision differences make π_old subtly wrong, so your ratios are wrong even at step 0. This is a real, common, and extremely confusing production bug in RLHF pipelines; the diagnostic is to recompute log-probs of the generated tokens in *both* stacks and compare them directly before you debug anything else.
### What does it mean for a model to be calibrated, and why should I care as an applied engineer rather than a researcher?

Mental model: **calibration is the property that lets you build a control flow on top of a probabilistic component.** A model is calibrated if, among all the cases where it says 80%, it is right 80% of the time. That is not a statement about accuracy — a coin-flip predictor that always says "50%" is perfectly calibrated and useless. It is a statement about whether the *number* the model attaches to its answer means anything, and everything you want to build on top of an LLM — cascades, abstention, human escalation, selective automation — is a threshold on that number.

Formally: for a predictor with confidence c, calibration requires P(correct | confidence = c) = c for all c. Two orthogonal properties matter and get conflated: **discrimination** (does the confidence rank correct answers above incorrect ones — measured by AUC) and **calibration** (are the numbers on the right scale — measured by ECE). You need discrimination for a threshold to be *useful* and calibration for the threshold to be *interpretable and stable*. A model with great discrimination but bad calibration is fixable with a monotone rescaling; a model with bad discrimination is not fixable at all.

**💰 Math on why this is money, not theory.** Take a support-automation product at 100k requests/day. A small model resolves 82% correctly at $0.0004/request; a frontier model resolves 94% at $0.006/request. All-frontier costs 100,000 × $0.006 = $600/day = **$18,000/month** at 6.0% error. Now add a calibrated confidence score and escalate the least-confident 25% of requests. If that bottom quartile contains 70% of the small model's errors (which is what a decently-discriminating score buys you), residual errors are 18,000 × 0.30 = 5,400 unescalated, plus 12,600 escalated × 6% the big model also misses ≈ 756, for 6,156 errors/day = **6.2% error**. Cost is 100,000 × $0.0004 + 25,000 × $0.006 = $40 + $150 = $190/day = **$5,700/month**. You bought a 68% cost reduction for 0.2 points of accuracy — and the *entire* saving is created by the confidence score's ability to rank. If the score were uncorrelated with correctness, escalating 25% would capture 25% of errors and the whole design collapses.

**🗣 Say this in the room:** "Calibration is what turns a model output into something I can route on. I care about two separate things — whether the confidence ranks correct above incorrect, which is AUC and which no post-hoc fix can create, and whether the numbers are on the right scale, which is ECE and which temperature scaling can usually fix. The cascade design I just described is worth about two-thirds of the inference bill, and all of that value comes from the ranking."

### Define ECE and implement it. Then tell me its known pathologies, because interviewers ask that second part.

Expected Calibration Error bins predictions by confidence and measures the average gap between confidence and accuracy within each bin, weighted by bin population:

  **ECE = Σ_b (n_b / n) · | acc(b) − conf(b) |**

```python
import numpy as np

def ece(conf, correct, n_bins=15):
    """conf: predicted probability of being right. correct: 0/1 outcome."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return total
```

Now the pathologies, and these are the follow-up:

**It is binning-dependent.** With 10 equal-width bins you get one number; with 50 you get a different, larger one. And equal-*width* bins are a poor choice for LLM confidences, which pile up near 0.9–1.0 — nine of your ten bins end up nearly empty and the whole metric is decided by one bin. Equal-*mass* (quantile) bins are the better default, and you should say which you used whenever you report an ECE.

**It is a biased estimator, and the bias grows with bins.** Within-bin sampling noise never cancels because of the absolute value, so more bins on a fixed sample size inflates ECE. Comparing ECEs computed with different bin counts or different n is meaningless.

**ECE = 0 does not mean the model is useful.** A predictor that outputs the base rate 0.7 on every single example, on a dataset where 70% of answers are correct, has exactly zero ECE and zero discriminative power. **Never report ECE without also reporting AUC or a selective-accuracy curve**; the pair is informative, either alone is not.

**It's a marginal metric.** ECE aggregates over the whole dataset, so a model that is overconfident on one tenant and underconfident on another can show a beautiful global ECE. Always compute it per meaningful slice — per language, per document type, per tenant — because the failures you care about are slice failures.

**It ignores ordering within a bin.** Two very different predictors can share an ECE.

**📄 Paper:** Guo et al. (2017), *On Calibration of Modern Neural Networks* — showed that modern deep networks are substantially more miscalibrated (overconfident) than the shallower networks that preceded them, and that a single-parameter temperature fit on a validation set removes most of the error. It replaced Platt scaling and isotonic regression as the default for neural classifiers by being simpler and, empirically, at least as good.

**⚠ Trap:** reporting a single ECE number as a model-quality claim. In review I ask for the reliability diagram, the bin counts, the bin scheme, and the AUC alongside it. A candidate who volunteers those unprompted has clearly done this on real data.

### How do you read a reliability diagram, and how would you fix a miscalibrated model at serving time?

A reliability diagram plots mean confidence per bin (x) against observed accuracy per bin (y), with the diagonal as perfect calibration. Read it by which side of the diagonal the curve sits on. **Below the diagonal = overconfident** (says 90%, right 70% of the time) — the near-universal condition of modern neural networks and of RLHF'd LLMs in particular. **Above the diagonal = underconfident**, which is rarer and usually the result of over-aggressive label smoothing or an over-corrected temperature. A curve that hugs the diagonal at low confidence and falls away at high confidence — the classic S-shape — means the model is fine when unsure and badly overconfident when sure, which is the worst possible shape for an escalation policy, because it means the cases you auto-approve are exactly the ones where the confidence lies. Always plot bin *counts* underneath as a histogram; a dramatic-looking deviation in a bin holding 11 examples is noise.

Fixes, in the order I'd try them:

**Temperature scaling.** Fit a single scalar T on a held-out calibration set by minimizing NLL of softmax(z/T). One parameter, so it needs only a few hundred labeled examples, and because dividing by a positive constant is monotone, **it cannot change your accuracy or your AUC — it only moves the numbers onto the right scale.** That property is why it's the default: it is a strictly-safe post-hoc fix.

```python
import torch
def fit_temperature(logits, labels, iters=200):
    logT = torch.zeros(1, requires_grad=True)          # optimize in log-space, T > 0
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / logT.exp(), labels)
        loss.backward(); return loss
    opt.step(closure)
    return logT.exp().item()
```

**Platt scaling** fits a logistic (two parameters, a and b, on the logit) — slightly more flexible, still monotone. **Isotonic regression** is nonparametric and can fix non-monotone miscalibration, but it needs thousands of examples, overfits happily, and produces a step function that behaves badly at thresholds you didn't sample. My rule: temperature first, isotonic only with >5k calibration examples and a held-out check.

**The API case, which is what you'll actually face.** With a hosted model you often can't touch logits — and even where logprobs are exposed, the token-level logprob of an answer is not the same thing as P(the answer is correct). So you calibrate a *downstream* score: fit a small logistic regression that maps features → P(correct), where the features are things like the mean logprob of the answer tokens, the self-consistency agreement rate over k samples, the retrieval score of the top passage, the answer length, and whether the model hedged. Train it on a few hundred human-labeled outcomes. **This little logistic model is one of the highest-ROI components you can add to a production LLM system, and almost nobody builds it.** It costs an afternoon and it's what makes the cascade arithmetic above real.

**⚠ Trap:** fitting the temperature (or the logistic) on the same set you evaluate calibration on. You will report a beautiful ECE that does not survive contact with production. Split it: fit on calibration, report on test, and re-fit on a schedule, because calibration drifts as the input distribution drifts.

### Does asking the model "how confident are you, 0–100?" actually work? And what happened to calibration when reasoning models arrived?

Verbalized confidence works better than people expect and worse than you'd want, and the honest answer names both.

What's real about it: it has genuine discriminative signal. Sort your outputs by verbalized confidence and accuracy does go up monotonically — the AUC is well above 0.5, which means it is *usable* as a routing feature. What's broken: the numbers are on the wrong scale (systematically overconfident, typically saying 90% when it's right 70% of the time), and they cluster hard on round numbers — 80, 90, 95, 99 — so you effectively have four or five confidence levels rather than a continuum, which makes fine-grained thresholding impossible. So: use it as a *feature* in a calibrated downstream model, never as a probability you threshold directly.

Alternatives that usually beat it: **self-consistency agreement** (sample k = 5 answers at T = 0.7 and use the plurality fraction as the confidence) is typically the strongest cheap signal, at k× the cost. **Sequence logprob** of the answer span, where available, is decent and free. **Pairwise self-verification** ("here is your answer, find the error") adds signal but also adds a second failure mode.

The post-training effect is the part with a clean documented result. OpenAI's GPT-4 technical report showed the *pre-trained* model was well calibrated on a multiple-choice benchmark and that the post-RLHF model was substantially less so. The mechanism is exactly the reverse-KL / mode-seeking story from earlier: RLHF pushes the policy to commit to a mode, and confidence is a casualty of commitment. **This is a general property of preference-optimized models, not a quirk of one lab.**

**📅 Volatile:** there is a more recent and still-settling line of evidence that extended reasoning makes this *worse* — that models which generate long chains before answering become more confident as the chain lengthens, roughly independently of whether the chain is correct, so "think longer" trades calibration for accuracy. The mechanism is plausible and mechanistically coherent (each self-generated reasoning step conditions the next, so the model accumulates evidence from its own tokens), and it matches what I've seen measuring reasoning models. But verify the current state of this before quoting it as settled — this is one of the faster-moving corners of the literature.

**🗣 Say this in the room:** "Verbalized confidence has real ranking signal but bad scale — it clusters on round numbers and it's systematically overconfident, and RLHF made that worse; the GPT-4 report showed the base model was calibrated and the post-RLHF model wasn't. So I'd never threshold it directly. I'd use it as one feature in a small logistic model alongside self-consistency agreement and retrieval score, fit on a few hundred labeled outcomes, and threshold *that*."

### Your eval says variant B beats variant A, 78% versus 76%, on 500 examples. Do you ship it?

No, and I can show you why in about twenty seconds of arithmetic — which is the point of the question.

**Step 1: the confidence interval on a single number.** For a proportion, SE = √(p(1−p)/n) = √(0.78 × 0.22 / 500) = √0.000343 = 0.0185, so 1.85 percentage points. The 95% interval on B alone is 78% ± 1.96 × 1.85 = **[74.4%, 81.6%]**. A ±3.6-point interval on a 2-point claimed improvement should end the conversation on its own.

**Step 2: the interval on the difference.** For two independent samples, SE_diff = √(p_A(1−p_A)/n + p_B(1−p_B)/n) = √(0.000343 + 0.000365) = √0.000708 = 0.0266, so 2.66 points. The observed difference is 2.0 points — **less than one standard error.** The 95% CI on the difference is 2.0 ± 5.2 = [−3.2, +7.2]. It comfortably contains zero, and it also contains "B is three points *worse*."

So the honest statement is: *this experiment cannot distinguish B from A, and it also cannot rule out B being meaningfully worse.*

**Step 3 — and this is where you gain ground on other candidates: don't stop at "not significant," say what you'd do instead.**

- **Pair the comparison.** If both variants ran on the *same* 500 examples — which they did, this is an eval set, not an A/B test on users — then treating them as independent samples throws away most of your power. The right analysis looks only at the examples where they disagree. More on this below; it typically cuts the required sample size by 5–10×.
- **Look at the disagreements by hand.** 500 examples, maybe 40 disagreements. Read them. Twenty minutes of reading disagreements teaches you more than the p-value does, and it's where you discover that B's "wins" are all on one template and its "losses" are all on the long-tail language.
- **Check whether the delta is even worth the cost.** If B is the same prompt plus a 4k-token few-shot block, you are paying real money for an effect you cannot measure.
- **Slice it.** A 2-point global delta that is +9 on one segment and −4 on another is a genuinely different finding from a uniform +2, and the segmented view often *is* shippable — for that segment.

**⚠ Trap:** the eval-set-of-100 that pervades take-homes and internal dashboards. At n = 100, the 95% CI half-width on a proportion near 0.8 is 1.96 × √(0.8 × 0.2/100) = 7.8 points. **Any improvement smaller than 8 points is invisible at n = 100.** Teams ship prompt changes on 3-point deltas from 100-example evals every day, and it is indistinguishable from shipping noise. Say this number out loud in an interview; it lands.

### Then compute it for me. How many examples do I need to detect a 2-point improvement?

**📐 Numbers you must know — the unpaired case.** For two proportions with α = 0.05 two-sided and 80% power, the per-arm sample size is

  n ≈ 2 · (z_{α/2} + z_β)² · p̄(1 − p̄) / δ²

with z_{0.025} = 1.96, z_{0.20} = 0.84, so (1.96 + 0.84)² = 7.84. Take p̄ = 0.77 and δ = 0.02:

  n = 2 × 7.84 × (0.77 × 0.23) / 0.02² = 2 × 7.84 × 0.1771 / 0.0004 = 2.777 / 0.0004 = **6,943 per arm**, i.e. **~13,900 examples total.**

That is the number people are shocked by, and it should reset how you think about eval sets. Detecting small deltas requires large n, full stop. Some anchors from the same formula, so you can do it in your head: **the sample size scales as 1/δ².** Halving the effect you want to detect quadruples the n. To detect 5 points you need 2 × 7.84 × 0.1771/0.0025 = 1,111 per arm. To detect 10 points, 278 per arm. **To detect 1 point, 27,800 per arm.**

**Now the paired case, which is what you actually have.** Both variants run on identical examples, so most of the variance is *item difficulty*, which is shared and therefore cancels. Only the discordant items — where the two variants disagree — carry information. Suppose 10% of items are discordant and the true effect is that 6% of items flip A-wrong-to-B-right while 4% flip the other way (net +2 points). Conditional on being discordant, B wins 60% of the time, so we're testing p = 0.6 against p = 0.5 on the discordant subset:

  n_discordant = [z_{α/2}√(0.5 × 0.5) + z_β√(0.6 × 0.4)]² / (0.1)²
  = [1.96 × 0.5 + 0.84 × 0.4899]² / 0.01 = [0.98 + 0.4115]² / 0.01 = 1.936 / 0.01 = **194 discordant pairs.**

At a 10% discordance rate, that's **~1,940 total examples** — a **7× reduction** versus the unpaired 13,900, from nothing but analyzing the data correctly.

**🗣 Say this in the room:** "To detect a 2-point delta at 80% power you need roughly 7,000 per arm unpaired. But an eval set is a paired design — same items, both variants — so I'd analyze only the discordant items, and at a typical 10% disagreement rate that drops the requirement to about 2,000 examples total. The single highest-leverage thing most teams could do to their eval methodology is stop running an unpaired analysis on paired data."

**⚠ Trap:** treating "we need 2,000 examples" as a reason not to measure. The alternative is not "measure with 100," it is "measure a 10-point effect with 300, and refuse to claim 2-point effects at all." Size your eval to the effect size you actually intend to act on, and say out loud which effects you are choosing to be blind to.

### Why paired bootstrap rather than a t-test on the eval scores? Implement it.

Three reasons, and the third is the one that matters most in this field.

**One: eval metrics are usually not means of independent identically-distributed scalars.** nDCG@10, pass@k, F1, and any LLM-judge rubric score are not sample means with a clean sampling distribution, so the t-test's assumptions don't apply. The bootstrap doesn't care — it resamples the empirical distribution and reads the answer off the resampled statistic, whatever the statistic is.

**Two: pairing removes item-difficulty variance.** The dominant source of variance in an eval is that some questions are hard for everybody. If you bootstrap the two systems' scores independently, that variance stays in your estimate. If you bootstrap the *per-item differences*, it cancels exactly. In practice this shrinks confidence intervals by a factor of two to four on a typical eval, which is free statistical power.

**Three: it composes with any metric your system actually reports.** You can bootstrap a corpus-level metric like BLEU or a rate like "% of responses passing schema validation" the same way, which a t-test cannot do.

```python
import numpy as np

def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """a, b: per-item scores for the two systems, same items, same order."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = d.mean()
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])            # 95% CI on the difference
    centered = boot - obs                                # simulate the null
    p = float(np.mean(np.abs(centered) >= abs(obs)))     # two-sided bootstrap p-value
    return obs, (lo, hi), p
```

Two implementation notes that separate a correct implementation from a plausible one. **Resample items, not scores** — the unit of resampling must be the unit of independence, so for a multi-turn eval you resample *conversations*, not turns, or your intervals will be far too narrow. And **centre the bootstrap distribution before computing a p-value**; the percentile interval is a CI, but the null hypothesis is "mean difference is zero," which you simulate by subtracting the observed mean. Skipping that step gives you a p-value that is wrong in a direction you won't notice.

**📄 Paper:** Koehn (2004), *Statistical Significance Tests for Machine Translation Evaluation* — established bootstrap resampling as the standard significance procedure for corpus-level NLP metrics, replacing the practice of reporting bare score differences. The methodology transfers directly to LLM evals and almost nobody in the LLM world cites it.

**🏋 Drill:** 15 minutes, no references. Write `paired_bootstrap` from memory. Then generate synthetic data where system B is truly 2 points better on 2,000 items, run it, and confirm the CI excludes zero. Then rerun at 300 items and confirm it doesn't. Pass criterion: both behaviours reproduce and you can state, without looking, why the p-value needs the centering step.

### When is McNemar's test the right tool, and how do you compute it?

McNemar's is the exact right tool for **the most common comparison in this entire field**: two systems, same evaluation items, binary correct/incorrect outcome. That is what nearly every eval is. It is the closed-form version of the pairing argument, and it takes ten seconds to compute by hand, which makes it a good thing to reach for live in an interview.

Build the 2×2 contingency table over items:

|  | B correct | B wrong |
|---|---|---|
| **A correct** | a | b |
| **A wrong** | c | d |

The cells a and d — where both agree — carry *no information about which is better*, and that is McNemar's insight: condition on the discordant pairs only. Under the null hypothesis that the systems are equally good, each discordant item is a fair coin flip between b and c. So:

  **χ² = (|b − c| − 1)² / (b + c)**, with 1 degree of freedom (the −1 is the continuity correction).

When b + c < 25, skip the approximation and run the exact binomial test of b successes out of b + c trials against p = 0.5.

**Worked example on the 500-item case from earlier.** Suppose A scored 76%, B scored 78%, and the table is a = 370, b = 25 (A right, B wrong), c = 35 (A wrong, B right), d = 70. The net is (35 − 25)/500 = +2 points, matching. Now:

  χ² = (|35 − 25| − 1)² / (35 + 25) = 9² / 60 = 81/60 = **1.35**, p ≈ **0.245**.

Not close to significant, consistent with the CI analysis. Note how much more informative the table is than the two headline numbers: you can immediately see that only 60 of 500 items disagreed, which tells you these two variants are 88% identical and that whatever B changed, it changed narrowly. That framing — "the systems agree on 88% of items; the disagreement is 35 vs 25" — is a far better thing to put in front of a PM than "78 vs 76."

**⚠ Trap:** running McNemar on unpaired data, or on items scored by a stochastic judge without accounting for judge noise. If your "correct" label comes from an LLM judge sampled at temperature > 0, some of your b and c cells are judge flips, not system differences. Score the judge at T = 0, or better, measure judge self-agreement on a rescored subset first and treat that as your noise floor — if judge self-disagreement is 8% and your systems differ by 2%, the test is measuring the judge.

### You swept 30 prompt variants and the best one beat baseline by 3 points with p = 0.03. What do you tell the PM?

That we have found approximately nothing, and here is the arithmetic.

**💰 Math on the family-wise error rate.** If all 30 variants were truly identical to baseline, each test has a 5% chance of a false positive at α = 0.05. The probability that *at least one* of 30 independent tests fires is 1 − 0.95³⁰. Compute it: ln(0.95) = −0.05129, × 30 = −1.5388, e^{−1.5388} = 0.2146. So **1 − 0.215 = 78.5% chance of at least one "significant" result from pure noise.** A p = 0.03 in a family of 30 is not evidence; it is the expected outcome of the procedure.

The corrections, and when I use each:

- **Bonferroni:** require p < α/m = 0.05/30 = **0.00167**. Your p = 0.03 fails by a factor of 18. Bonferroni controls the family-wise error rate, is very conservative, and is the right choice when a single false positive is expensive (a safety filter, a pricing change).
- **Benjamini–Hochberg:** sort the m p-values ascending and find the largest k with p_(k) ≤ (k/m)·α; reject everything up to k. This controls the *false discovery rate* — the expected proportion of your rejections that are false — rather than the probability of any false rejection. For an exploratory prompt sweep where you'll validate the survivors anyway, BH is the correct and much less brutal choice. **📄 Paper:** Benjamini & Hochberg (1995) introduced FDR control, replacing family-wise-error methods in settings with many tests where a few false discoveries are tolerable.

But the deeper problem isn't the p-value at all — it's **the winner's curse.** When you select the maximum of 30 noisy estimates, that maximum is biased upward by construction, because a variant is more likely to be selected if its noise happened to be positive. Your 3-point winner has a true effect that is systematically smaller than 3 points, often dramatically so, and applying a multiplicity correction to the p-value does not fix the *effect size* bias at all.

The fix is the one every experienced person converges on: **hold out a confirmation set.** Sweep on set A, take the top 2–3 candidates, then evaluate *only those* on a fresh set B that was never touched during the sweep. The estimate on B is unbiased, and with only 2–3 tests the multiplicity problem nearly vanishes. This is the same discipline as a train/val/test split, and prompt sweeps are exactly a hyperparameter search — treat them that way.

**🗣 Say this in the room:** "With 30 variants at α = 0.05 there's a 78% chance of at least one false positive under the null, so p = 0.03 is what noise looks like. Under Bonferroni I'd need p below 0.0017. But the bigger issue is winner's curse — the max of 30 noisy estimates is biased upward, and correcting the p-value doesn't correct the effect size. I'd take the top three to a held-out confirmation set and report *that* number to the PM, and I'd expect it to come in well under 3 points."

**⚠ Trap:** the invisible version of this, which is a team that runs prompt changes serially over six weeks with no correction at all — twenty implicit comparisons, each shipped on a 2-point delta. That's the same multiple-comparisons problem spread over a quarter, and the aggregate result is a prompt that has been optimized to your eval set's noise. The symptom is an eval score that climbs steadily while user-facing metrics don't move.

### Your eval scores move ±1.5 points between runs even at temperature 0. Walk me through debugging that.

This is a real and common incident, and the value is in having an ordered procedure rather than a list of causes.

**🔍 Failure taxonomy — nondeterministic eval scores, in the order I'd check:**

**1. Establish the noise floor before debugging anything.** Run the identical config three times, unchanged. If it varies ±1.5 points, the variance is in your harness, not in your change. This is step zero and it is the step people skip; they debug a "regression" that is inside their own noise band.

**2. Is the judge stochastic?** If a model grades the outputs, is it running at T = 0 with a pinned model version? An LLM judge at T = 0.7 will re-score the same output differently. Measure judge self-agreement by scoring the same 200 outputs twice and computing the flip rate. If it's 5%, on a 500-item eval that's ±25 items ≈ ±5 points of pure judge noise, which swamps everything else. Fix the judge before you look anywhere else.

**3. Is generation batch-dependent?** As covered earlier, greedy decoding is not deterministic under variable batch composition — floating-point reduction order changes with batch shape. Signature: outputs are *mostly* identical with a handful of complete divergences, and the divergence point is a token where the top-2 logits were nearly tied. Test by running the eval single-threaded at batch size 1 and seeing whether the variance collapses.

**4. Is retrieval nondeterministic?** ANN indexes (HNSW especially) can return different neighbours across builds and across concurrency levels, and ties in the score are broken by insertion order. If your retrieval top-5 differs by one document, downstream generation differs. Pin the index, log the retrieved doc IDs per item, and diff them across runs — this makes the cause visible in one command.

**5. Is the eval set itself changing?** Unpinned dataset version, a database query without an ORDER BY, sampling `n=500` from a larger pool with a fresh seed each run, or dropped items from timeouts. **Timeouts are a sneaky one**: if 8 items time out in run one and 3 in run two, and timed-out items are scored as failures, you've moved a point without anything model-related changing.

**6. Concurrency-dependent truncation.** Under load, more requests hit `max_tokens` or a provider-side truncation, and truncated answers fail. Log finish reasons and assert the distribution is stable across runs.

**The fix set, once you've localized it:** pin every version (model, dataset, index, prompt template) and log the pins in the results artifact; run the judge at T = 0; log per-item outputs so you can diff runs rather than diff aggregates; report **paired** comparisons so shared noise cancels; and publish the harness's own noise floor next to every result. **The rule I enforce in review: no eval result is reported without its repeat-run variance.** A team that doesn't know its noise floor cannot interpret any of its own numbers.

**💰 Math on why this is urgent:** if your noise floor is ±1.5 points and your team ships prompt changes on 2-point improvements, then by the arithmetic above roughly half of your "improvements" are noise. At an engineer-week per prompt iteration and 40 iterations a year, that is 20 engineer-weeks — call it $80–120k of loaded cost — spent producing a random walk. Fixing the harness costs about a week.

### Timed drill — 15 minutes, blank page, no references. Can you produce the three derivations from memory?

**🏋 Drill.** Set a timer for 15 minutes. No notes, no autocomplete, no calculator beyond arithmetic you do by hand. Produce all three:

**(a) The attention scale.** State the assumptions on q and k, derive Var(q·k) = d_k, state the resulting standard deviation for d_k = 128, and explain in one sentence what happens to the softmax and to the gradient without the 1/√d_k. *Pass criterion: you write "variances add over d_k independent unit-variance products, so std = √d_k ≈ 11.3" without hesitating, and you name the softmax Jacobian diag(s) − ssᵀ as the reason the gradient dies.*

**(b) The KV cache formula, applied.** Write bytes/token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes. Then compute it for L = 80, n_kv_heads = 8, head_dim = 128, bf16, and give the total for a 128k-token sequence in GiB. *Pass criterion: 320 KiB/token and 40 GiB, arrived at in under two minutes, and you can immediately say what changes if the model were MHA (×8 → 320 GiB) or if the cache were fp8 (÷2 → 20 GiB).*

**(c) Training compute.** Derive 2N forward and 4N backward from the per-linear-layer matmul counts, state C = 6ND, and compute the H100-days for N = 7e9, D = 1e12 at 40% MFU. *Pass criterion: 4.2e22 FLOPs, ~1,200 H100-days, ~$60k at $2/GPU-hour — and you name at least two things the estimate omits (failed runs, data prep, eval compute, checkpoint storage).*

Grade yourself hard. **A hesitation is a fail**, because in the room the hesitation is what the interviewer records. If you fail any part, the remedy is not to reread this section — it is to write the derivation out longhand three times on separate days. This material is procedural memory, not declarative; you cannot acquire it by recognition.

Extend the drill once you pass it: have someone interrupt you mid-derivation with "why?" at a random step. The real interview stops you; rehearsing an uninterrupted monologue does not prepare you for that.

### Timed drill — 10 minutes. Here's a product spec; give me a cost and latency model on the whiteboard.

**🏋 Drill.** Ten minutes, out loud, on a whiteboard or a shared doc, no calculator. The spec: *an internal document assistant for a 6,000-person company. 40,000 queries per weekday. Each query retrieves 8 chunks of 600 tokens, prepends a 1,500-token system prompt, and produces roughly 400 output tokens. Assume a frontier-tier model at $3 per million input tokens and $15 per million output tokens.* Produce: cost per query, monthly cost, the single highest-leverage optimization with its savings, and a p95 latency estimate with its components.

Here is the shape of a passing answer, so you can grade yourself.

**Tokens per query.** Input = 1,500 (system) + 8 × 600 (chunks) + ~100 (user query) = 1,500 + 4,800 + 100 = **6,400 input tokens**. Output = **400**.

**Cost per query.** Input: 6,400 / 1e6 × $3 = $0.0192. Output: 400 / 1e6 × $15 = $0.0060. **Total $0.0252 per query.**

**Monthly.** 40,000/day × 22 working days = 880,000 queries. 880,000 × $0.0252 = **$22,176/month**. Say ~$22k.

**Highest-leverage optimization.** Input dominates at 76% of cost, and the 1,500-token system prompt is byte-identical on every call — that is what prefix caching is for. At a 90% cache discount on cached input, those 1,500 tokens go from 1,500/1e6 × $3 = $0.0045 to $0.00045, saving $0.004 per query = **$3,560/month**. Better: the chunks are the bigger prize at 4,800 tokens, but they vary per query, so caching doesn't apply — instead, rerank 8 chunks down to 4 and you cut 2,400 input tokens = $0.0072/query = **$6,336/month**, provided an eval shows retrieval quality holds at k = 4. Combined, roughly **$10k/month off a $22k bill**, with the reranking gated on an eval. **📅 Volatile:** cache-discount percentages and per-token prices differ by provider and change often — verify before quoting.

**p95 latency.** Components: retrieval (embed the query ~20 ms + ANN search ~15 ms + fetch chunks ~15 ms ≈ 50 ms), then prefill of 6,400 tokens, then 400 tokens of decode. Prefill at a compute-bound rate of order 10⁴ tokens/s for a frontier-scale model on a served endpoint ≈ 600 ms, so TTFT lands around 650–700 ms. Decode at ~60 tokens/s inter-token rate → 400/60 ≈ **6.7 seconds to completion**. So: **TTFT ~0.7 s, full response ~7.4 s.** Which immediately tells you the product must stream — the perceived latency is TTFT, and a 7-second blocking spinner is a different product than a 0.7-second first token.

**Pass criterion:** you produce cost/query, monthly cost, a named optimization with its arithmetic, and a latency decomposition that separates TTFT from total — inside ten minutes, out loud, without a calculator. The single most common failure is doing the token arithmetic correctly and then forgetting to separate TTFT from end-to-end latency, which is the number the product actually lives or dies on.


---

## 8. Training Math: Losses, Optimizers, Backprop, Numerics, Normalization, Stability

*Mastering this proves you can hold a technical conversation with a research engineer about a training run without being escorted out of it.*

### Write me the next-token cross-entropy loss for a causal LM from scratch, and tell me exactly where the off-by-one lives.

The mental model: a causal LM is a batch of `B × T` independent classification problems that happen to share weights. Position `t` produces a probability distribution over the vocabulary, and the correct answer for that classification problem is the token that actually appeared at position `t+1`. The entire "language modelling objective" is average multi-class cross-entropy over those `B × T` problems. Everything else — the shifting, the masking, the reduction — is bookkeeping around that one sentence.

The mechanism. The model consumes `input_ids` of shape `[B, T]` and emits `logits` of shape `[B, T, V]`. Logit row `logits[b, t, :]` is the model's prediction *for the token at position t+1*, because the causal mask let position `t` see positions `0..t` inclusive and nothing after. So the label for `logits[b, t]` is `input_ids[b, t+1]`. That is the off-by-one, and it is not in the model — it is in the loss call. You drop the last logit (it predicts a token you do not have) and the first label (nothing predicts it).

```python
import torch, torch.nn.functional as F

def causal_lm_loss(logits, input_ids, attention_mask=None, ignore_index=-100):
    # logits: [B, T, V] float32/bf16, input_ids: [B, T] long
    shift_logits = logits[:, :-1, :].contiguous()          # [B, T-1, V]
    shift_labels = input_ids[:, 1:].clone()                 # [B, T-1]
    if attention_mask is not None:
        shift_labels[attention_mask[:, 1:] == 0] = ignore_index
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
```

Three details that are graded. First, `.float()` on the logits before the softmax — cross-entropy accumulates a log-sum-exp over `V ≈ 128k` terms and you want that accumulation in fp32 even when the forward pass ran in bf16. Second, `ignore_index=-100` is PyTorch's sentinel: those positions contribute neither loss nor gradient, and critically they are excluded from the denominator of the mean. Third, `-100` must be written into the *labels*, not the inputs; you still feed the pad tokens forward, you just refuse to score them.

**⚠ Trap:** HuggingFace `*ForCausalLM` models shift internally when you pass `labels=`. If you also shift before passing, you have shifted twice and the model is now trained to predict the token *two* ahead. The loss will still descend — it is a learnable task — it will just plateau around 1.5–2.5 nats higher than it should, and nothing will crash. The rule I enforce in review: either pass `labels=input_ids` unshifted and let the model shift, or compute the loss yourself from `outputs.logits`, and never both. Write a unit test that overfits eight tokens to loss < 0.01 in 200 steps; a double shift makes that test impossible to pass.

**🗣 Say this in the room:** "Logits at position t predict the token at t+1, so the loss shifts logits left by one and labels right by one. Padding and prompt tokens get `ignore_index = -100` so they are dropped from both the numerator and the denominator of the mean, and I upcast logits to fp32 before the softmax because the log-sum-exp runs over the full vocabulary."

### On an SFT run, which tokens do you actually compute loss on, and what happens if you get it wrong?

Only the assistant's tokens — and this decision is worth more eval points than any hyperparameter in the run. The mental model: SFT is teaching a conditional distribution p(response | prompt). Putting loss on the prompt tokens teaches the model to *generate prompts*, which is a different distribution you do not want and which you will never sample from at inference.

Mechanically you build a per-token label tensor that is a copy of `input_ids` with `-100` written over every position belonging to the system prompt, the user turn, and the chat-template scaffolding, leaving only the assistant's content tokens (and usually the assistant's end-of-turn token, which is how the model learns to stop) scored. For multi-turn conversations you do this per turn: turns 1..n-1 of the assistant are also supervised in most recipes, since they are still assistant behaviour.

**🔍 Failure taxonomy — masking bugs, in the order I check them:**
1. *Loss on everything.* Symptom: training loss is suspiciously low from step 0 (prompts are highly predictable boilerplate) and the model starts hallucinating user turns mid-response. Check: print the decoded tokens where `labels != -100` for one example, by hand, on the first batch. This costs 30 seconds and catches most of these.
2. *Loss on nothing.* Symptom: loss is exactly 0.0 or NaN, or grad norm is exactly 0. Cause: the template rendering changed and your span-finding string match no longer hits. `cross_entropy` with an all-`-100` row returns NaN because the denominator is zero.
3. *EOS not supervised.* Symptom: perfect-looking eval loss, but generation never stops and runs to `max_new_tokens` every time. This one gets shipped constantly.
4. *Off-by-one at the span boundary.* Symptom: the model reliably drops or duplicates the first token of every response.

**💰 Math:** the cost of getting this wrong is not compute, it is a wasted run. A 7B full-finetune on 100k examples averaging 1,200 tokens is 1.2e8 tokens; at roughly 6·N·D FLOPs that is 6 × 7e9 × 1.2e8 ≈ 5.0e18 FLOPs. On 8×H100 at a realistic 400 TFLOP/s each — 3.2e15 FLOP/s aggregate — that is 5.0e18 / 3.2e15 ≈ 1,570 s of pure math, call it ~1 hour wall clock with overheads, so ~$25–40 of GPU time. Cheap. The expensive part is the three days of eval confusion before someone decodes the label tensor.

**🏋 Drill:** given a tokenized multi-turn chat, write the label-masking function unaided in 10 minutes and then assert that `tokenizer.decode(input_ids[labels != -100])` equals exactly the concatenated assistant turns plus their end-of-turn tokens. Pass criterion: the assertion passes on a 3-turn example on the first run.

### Derive the gradient of cross-entropy with respect to the logits. I want the algebra, not the result quoted.

The mental model first: the softmax-plus-cross-entropy pair is deliberately constructed so that its composite derivative is the single cleanest expression in deep learning — predicted probability minus one-hot truth. The nastiness of the softmax Jacobian cancels exactly against the `1/p` from the log. That cancellation is why nobody implements softmax and NLL as separate layers in a real framework.

Let `z ∈ R^V` be the logits, `p = softmax(z)`, and let `y` be the index of the true class. The loss is `L = -log p_y = -z_y + log Σ_j exp(z_j)`.

Differentiate term by term with respect to `z_i`. The first term contributes `-1` when `i = y` and `0` otherwise, i.e. `-[i = y]`. The second term: `∂/∂z_i log Σ_j exp(z_j) = exp(z_i) / Σ_j exp(z_j) = p_i`. Adding them:

```
∂L/∂z_i = p_i − [i == y]
```

so `dL/dz = p − onehot(y)`. Note that this vector sums to zero, since `Σ p_i = 1` and the one-hot sums to 1 — the gradient lives in the subspace orthogonal to the all-ones direction, which is the algebraic statement that softmax is shift-invariant and the overall logit offset is unidentifiable. That fact is exactly what z-loss later exploits.

Two consequences worth stating out loud. The gradient magnitude is bounded by 1 per logit regardless of how wrong the model is — cross-entropy on logits is a well-conditioned objective, unlike, say, MSE on probabilities, which saturates. And the *true class* always receives a negative gradient (`p_y − 1 < 0`, push its logit up) while every other class receives a positive one proportional to its current probability (push down, hardest on the confident wrong answer).

```python
def softmax_xent_backward(logits, target_idx):
    # logits: [N, V], target_idx: [N]
    p = torch.softmax(logits.float(), dim=-1)
    grad = p.clone()
    grad[torch.arange(len(target_idx)), target_idx] -= 1.0
    return grad / len(target_idx)     # mean reduction
```

**⚠ Trap:** dividing by `N` where `N` is the batch size rather than the number of *unmasked* tokens. With `ignore_index` positions in the batch, PyTorch's mean reduction divides by the count of scored tokens; if you hand-roll the backward and divide by the full `B·T`, your effective learning rate silently scales with the padding ratio of each batch. Batches full of short sequences then get systematically smaller updates than batches of long ones — a data-order-dependent LR schedule you did not intend.

### Why do we compute cross-entropy through log-sum-exp instead of literally softmaxing and taking a log?

Because `exp` overflows and a naive implementation produces `inf` and then `NaN` on perfectly healthy logits. The mental model: log-sum-exp is not a numerical nicety, it is the only way to evaluate this function over the range of logits an LLM actually produces.

Concretely: `logsumexp(z) = log Σ_j exp(z_j)`. In fp16 the maximum representable value is 65,504, and `exp(z)` overflows at `z ≈ 11.09`. LLM logits routinely reach 20–30 in magnitude, so a literal `exp` in fp16 gives `inf`, `inf/inf = NaN`, and the run dies. Even in fp32 (`exp` overflows around `z ≈ 88.7`) a poorly-conditioned run late in training can get there.

The fix is the shift identity. Because softmax is invariant to adding a constant to all logits:

```
logsumexp(z) = m + log Σ_j exp(z_j − m),   where m = max_j z_j
```

Now the largest argument to `exp` is exactly 0, so the largest term is exactly 1 and cannot overflow; the small terms underflow gracefully to 0, which is the correct answer to within fp precision. Cross-entropy becomes `L = logsumexp(z) − z_y`, computed without ever forming a probability.

```python
def stable_xent(z, y):                 # z: [V], y: int
    m = z.max()
    lse = m + torch.log(torch.exp(z - m).sum())
    return lse - z[y]
```

**📐 Numbers you must know:** fp16 is 1 sign / 5 exponent / 10 mantissa bits → max 65,504, so `exp` overflows above ln(65504) ≈ **11.09**. bf16 is 1/8/7 → max ≈ 3.39e38, same exponent range as fp32, so `exp` overflows above ln(3.39e38) ≈ **88.7**. This single difference — exponent bits, not mantissa bits — is why the entire field switched from fp16 to bf16 for LLM training and threw away loss scaling.

**⚠ Trap:** the "I don't need this, my framework handles it" reflex. It does — `F.cross_entropy` fuses this, and so do the chunked/fused CE kernels used to avoid materializing `[B, T, V]` in fp32. But you will hand-roll a log-prob computation the first time you implement DPO, importance-weighted RL, or speculative-decoding rejection sampling, and at that moment you need `log_softmax` and not `log(softmax(...))`. Reviewing PRs, `torch.log(torch.softmax(x))` is an automatic change-request.

### What is z-loss, why does it exist, and what would make you turn it on?

The mental model: softmax is invariant to adding a constant to every logit, so the *absolute scale* of the logits is completely unconstrained by cross-entropy — it is a free direction the optimizer can wander along forever without any loss penalty. Nothing stops it drifting to ±30, ±60, ±200. Then one day an fp16 exponential overflows, or an attention softmax saturates, and you get a loss spike from a direction the objective never supervised. Z-loss puts a weak spring on that free direction.

Mechanically, let `Z = Σ_j exp(z_j)` be the softmax partition function, so `log Z = logsumexp(z)` — a quantity you already computed. Add

```
L_total = L_CE + λ · (log Z)²,     λ ≈ 1e-4
```

This penalizes `log Z` drifting away from 0 in either direction, which anchors the overall logit magnitude without touching the *differences* between logits, which is all the softmax actually consumes. Cost: essentially zero, since `logsumexp` is already materialized in the CE kernel. The gradient contribution is `2λ·log Z · p_i` on each logit — proportional to the current probability, so it shrinks the whole logit vector toward a fixed scale.

```python
def ce_with_zloss(logits, labels, z_coef=1e-4, ignore_index=-100):
    logits = logits.float()
    lse = torch.logsumexp(logits, dim=-1)                    # [N]
    valid = labels != ignore_index
    ce = F.cross_entropy(logits, labels, ignore_index=ignore_index)
    z = (lse[valid] ** 2).mean()
    return ce + z_coef * z, ce.detach(), z.detach()
```

**📄 Paper:** Chowdhery et al. (2022), the PaLM report, popularized the auxiliary z-loss at `1e-4` explicitly as a training-stability measure for large-scale runs. Zoph et al. (2022), ST-MoE, applied the same idea to the *router* logits, where it matters even more because the router is a tiny softmax whose logits directly select experts.

**⚠ Trap:** believing z-loss fixes loss spikes. It does not fix them; it removes one specific *cause* of them (unbounded logit growth in the output head or the router). If your spike is coming from a bad data shard, a diverging Adam second moment, or an attention-logit blowup, z-loss will do nothing and you will have burned a week. The honest framing: z-loss is cheap insurance you turn on by default at scale, in the same spirit as gradient clipping, not a diagnostic tool.

**🗣 Say this in the room:** "Cross-entropy only constrains logit *differences*, so the mean logit level is an unsupervised free parameter that drifts. Z-loss penalizes `log Z` squared with a coefficient around 1e-4, pinning that free direction so the numerics stay in range. It costs nothing because `logsumexp` is already computed inside the cross-entropy."

### Gemma 2 uses logit softcapping and several recent models use QK-norm. What problem are both solving, and what does each cost?

Both attack the same failure: a softmax whose inputs grow without bound becomes a near-one-hot function, its gradient collapses toward zero, and en route to that it can overflow in reduced precision. Softcapping and QK-norm are two different ways of putting a ceiling on softmax inputs — one by squashing, one by normalizing.

**Logit softcapping** applies a smooth bounded map before the softmax:

```
z ← cap · tanh(z / cap)
```

`tanh` is odd, roughly identity near 0, and asymptotes to ±1, so this leaves small logits untouched and compresses large ones into `(−cap, +cap)`. Gemma 2 reported applying it in two places with different caps — a larger cap on attention logits and a tighter one on the final output logits. **📅 Volatile:** the exact cap values (in the region of 50 for attention and 30 for final logits) are per-model config, so read `config.json` rather than quoting from memory in a room.

**QK-norm** takes a different route: apply an RMSNorm (or LayerNorm) to the query and key vectors *before* the `QKᵀ` dot product, per head. Since both operands then have controlled norm, the dot product's magnitude is bounded by roughly `d_head` times the learned norm gains rather than by whatever the projections happen to produce. It costs two extra normalizations per attention block — a few percent of step time — and buys you attention logits that cannot explode.

**📄 Paper:** Dehghani et al. (2023), scaling ViT to 22B, introduced QK-normalization specifically to fix attention-logit divergence at scale; it has since been adopted broadly in open text models.

**⚠ Trap:** softcapping is not free at inference. `tanh` on the attention logits is incompatible with the standard FlashAttention fast path unless the kernel explicitly supports it, which is exactly why early Gemma 2 serving support was a mess — engines had to either add softcapping to their fused kernel or fall back to a slower path. When I evaluate an architecture change for a product deployment, "does this survive contact with the serving kernel?" is a first-class question, not an afterthought. QK-norm is the friendlier choice here because it is a pointwise op *outside* the fused attention kernel and needs no kernel changes at all.

**🗣 Say this in the room:** "Both bound the softmax input. Softcapping squashes the logits with a scaled tanh, which is architecture-invasive because the fused attention kernel has to know about it. QK-norm normalizes Q and K before the dot product, which lives outside the kernel and is therefore the cheaper option operationally. That's why newer models tend to pick QK-norm."

### Explain label smoothing, and then tell me why you would refuse to use it in a distillation run.

Label smoothing replaces the one-hot target with a mixture: mass `1 − ε` on the true class and `ε/(V−1)` spread across the rest, typically `ε = 0.1`. The mental model: hard targets ask the model to drive `p_y → 1`, which requires the true logit to run away to `+∞` relative to all others. Smoothing sets a finite target confidence, so the logit gap converges to a finite value instead of growing forever. That is why it improves calibration and, historically, top-1 accuracy in image classification and NMT.

Mechanically the gradient becomes `p − (1−ε)·onehot(y) − ε/(V−1)·ones`, so every non-target class gets a small constant *upward* pull opposing the usual push-down. The equilibrium is a fixed logit margin rather than an unbounded one.

Now the distillation problem. **📄 Paper:** Müller, Kornblith and Hinton (2019), "When Does Label Smoothing Help?", showed that smoothing tightens the penultimate-layer representations into tight equidistant clusters per class, which *erases the relative similarity structure among the wrong classes*. And that structure is precisely the signal distillation transfers: the teacher's claim that "this is a 3, but it's 8× more like an 8 than like a 7" is the dark knowledge. Train the teacher with label smoothing and the teacher's own soft distribution over the non-target classes has been flattened by construction, so the student learns less. Their empirical result is that a label-smoothed teacher yields a *worse* student than a hard-target teacher even when the smoothed teacher itself has higher accuracy.

For LLMs specifically I go further: I would not use label smoothing in language-model pretraining or SFT at all, and I would push back on a PR that added it. Over a 128k-token vocabulary, `ε/(V−1)` puts a floor of ~7.8e-7 probability on every token including complete nonsense, which is a systematic bias toward a higher-entropy output distribution. That degrades exactly the behaviours we then try to elicit — deterministic structured output, exact string reproduction, code. If you want the calibration benefit, temperature-scale at eval time or use z-loss for the numerics; both are targeted and reversible where smoothing is baked into the weights.

**🗣 Say this in the room:** "Label smoothing caps the target confidence, which helps calibration but collapses the inter-class similarity structure in the penultimate layer — and that structure is the whole payload of distillation. Müller et al. 2019 showed a smoothed teacher produces a worse student despite being a better classifier. For LLM training I'd skip it entirely and get the calibration back with temperature scaling at eval."

### When would focal loss show up in an LLM pipeline, and when is reaching for it a mistake?

Focal loss down-weights examples the model already gets right, so training capacity concentrates on the hard tail. **📄 Paper:** Lin et al. (2017), RetinaNet — the setting was dense object detection with roughly 100,000 background boxes per foreground box, an extreme imbalance that swamped the gradient signal with easy negatives.

The mechanism is one multiplier on the standard CE term: `L = −(1 − p_y)^γ · log p_y` with `γ ≈ 2`. When the model is already confident and correct (`p_y = 0.95`), the modulating factor is `(0.05)² = 0.0025`, cutting that example's contribution 400×. When it is badly wrong (`p_y = 0.1`), the factor is `0.81` — essentially unchanged. The result is that the loss surface is dominated by the examples that are still wrong.

Where this legitimately shows up in a GenAI system is in the *classifiers around the model*, not the model. Concretely: a safety/moderation head where the positive class is 0.3% of traffic; a routing classifier deciding cheap-model vs frontier-model, where "needs escalation" is rare; a retrieval reranker trained pointwise on click data with a 1% positive rate; a hallucination detector. All of these are ordinary imbalanced binary classification problems where focal loss (or its simpler cousin, class weighting) is a reasonable first lever.

**⚠ Trap:** applying focal loss to next-token prediction. It looks superficially appealing — "most tokens are easy, focus on the hard ones" — and it is wrong for two reasons. First, the easy tokens are not noise you want to suppress; fluency, grammar and formatting *are* the product, and de-weighting them degrades them. Second, the hard tail of next-token prediction is heavily populated by genuinely unpredictable tokens: the specific proper noun, the arbitrary variable name, the coin-flip stylistic choice. Focal loss up-weights exactly the irreducible-entropy positions, so you are spending capacity memorizing noise. This is the LLM analogue of tuning a system on its most-retried requests when those retries are hitting a third-party outage.

The honest ordering I use for imbalance: (1) fix the threshold, not the loss — most "imbalance problems" are actually "someone reported accuracy at 0.5 threshold" problems; (2) reweight classes; (3) resample; (4) focal loss; and always report PR-AUC rather than ROC-AUC when positives are under a few percent.

### Implement InfoNCE for training a text embedding model, and tell me what the temperature and the batch size are actually doing.

Mental model: a contrastive loss is cross-entropy where the "classes" are the other items in your batch. You are not predicting a label; you are running a `B`-way multiple-choice quiz — "which of these B documents goes with this query?" — and the answer is always the one on the diagonal. This reframing is the whole trick, because it means you get `B−1` negatives for free from data you already loaded.

Mechanism. Encode queries to `Q ∈ R^{B×d}` and positives to `P ∈ R^{B×d}`, L2-normalize both so dot products are cosine similarities in `[−1, 1]`, form the `B×B` similarity matrix, divide by temperature `τ`, and apply cross-entropy against the labels `[0, 1, ..., B−1]`.

```python
def info_nce(q, p, tau=0.05):
    q = F.normalize(q, dim=-1)          # [B, d]
    p = F.normalize(p, dim=-1)          # [B, d]
    logits = (q @ p.T) / tau            # [B, B]
    labels = torch.arange(len(q), device=q.device)
    loss_q = F.cross_entropy(logits, labels)        # query -> doc
    loss_p = F.cross_entropy(logits.T, labels)      # doc -> query (symmetric)
    return 0.5 * (loss_q + loss_p)
```

What `τ` does: after normalization, similarities are confined to `[−1, 1]`, which as logits is far too flat for softmax to produce any gradient signal — the max achievable logit gap is 2. Dividing by `τ = 0.05` rescales that to a usable range of `[−20, 20]`. Lower `τ` sharpens the softmax and puts almost all the gradient on the single hardest negative; higher `τ` spreads it. The practical range is 0.01–0.1, and `τ` interacts strongly with batch size: a large batch with a high temperature learns almost nothing because no negative is ever penalized hard.

What `B` does: it is your negative count. `B = 8` means the model only has to beat 7 random documents, which any bag-of-words model can do, so the loss goes to ~0 and stops teaching. `B = 4096` means it must beat 4,095, which forces genuinely discriminative representations. This is why contrastive training famously needs enormous batches, and why the standard trick is `all_gather` of the embeddings across every data-parallel rank before forming the similarity matrix — 8 GPUs at local batch 512 gives you an effective 4,096-way problem for the cost of one all-gather of `[512, 768]` floats.

**📄 Paper:** van den Oord et al. (2018), Contrastive Predictive Coding, is the canonical source of the InfoNCE name and the bound-on-mutual-information framing.

**💰 Math:** the loss value itself is a diagnostic. A randomly-initialized model on a `B`-way problem should sit at `ln(B)`: at `B = 4096` that is `ln(4096) = 8.32`. If your first-step loss is not near 8.3, your labels or your gather are wrong. If your loss reaches 0.02 by epoch 1, your negatives are too easy and you need hard-negative mining, not more epochs.

**⚠ Trap:** forgetting that in-batch negatives assume batch members are mutually non-relevant. Deduplicate and shuffle so that near-duplicate documents do not land in the same batch — otherwise you are actively training the model to push apart two things that *should* be close, which is a false-negative gradient and it is worse than no gradient.

### Compare InfoNCE, triplet loss, and multiple-negatives-ranking. Which do you actually pick for a production retrieval model?

Triplet loss is the oldest of the three: `max(0, d(a,p) − d(a,n) + margin)`. It considers exactly one negative per anchor and enforces a *margin* in distance space rather than a softmax over candidates. Its problem is sampling — with random negatives, the margin is satisfied almost immediately and the loss goes to exactly 0 for most triplets, contributing no gradient at all. So triplet needs hard-negative mining to work, and hard-negative mining with a margin loss is notoriously unstable: mine too hard and you collapse the embedding space, since your "hard negatives" are often mislabeled positives.

MultipleNegativesRankingLoss (the sentence-transformers name) is, mathematically, InfoNCE with in-batch negatives — the softmax-over-batch formulation above, usually with an optional column of explicitly-supplied hard negatives appended to the similarity matrix. The reason it dominates in practice is that it uses `B−1` negatives per anchor instead of 1, and the softmax automatically weights the hardest negatives most, giving you implicit hard-negative mining without an explicit mining pipeline.

My decision rule, stated as a rule: **default to MNRL/InfoNCE with in-batch negatives plus 1–4 explicitly mined hard negatives per query, with cross-device gathering.** Use triplet only when your data is inherently triplet-shaped (e.g. you have human "A is closer to Q than B" judgments and no natural positives). Never start with triplet.

The hard-negative pipeline that actually matters more than the loss choice: take your current retriever (or BM25 for round zero), retrieve top-50 for each training query, drop anything the labeling says is relevant, and sample negatives from ranks ~10–50 rather than ranks 1–5. Ranks 1–5 are where your false negatives live — documents that are actually relevant but unlabeled — and training on them teaches the model that correct answers are wrong. **📄 Paper:** the "denoised" hard-negative idea and the value of a cross-encoder to filter mined negatives is standard in the dense-retrieval literature following ANCE and RocketQA-style pipelines.

**⚠ Trap:** evaluating an embedding model on the loss. Contrastive loss on your training distribution correlates poorly with Recall@10 on your production query distribution, because your batch size defines the difficulty of the training task and your corpus size defines the difficulty of the real one. A model with a 4,096-way training task is being asked a far easier question than one retrieving from 40 million chunks. Always evaluate with a real index at real corpus size — I would push back hard on any embedding-training PR whose only reported metric is train loss.

### Derive the Bradley-Terry reward model loss and write the training step.

The mental model: a reward model does not learn "how good is this response" on an absolute scale, because humans cannot label that consistently. It learns a *latent score* whose differences reproduce observed pairwise preferences. Bradley-Terry is the 1952 statistical model that turns "A beat B" observations into scalar strengths, and it is the same math as Elo.

Bradley-Terry says the probability that response A is preferred to response B is a logistic function of the score difference:

```
P(A ≻ B) = exp(r_A) / (exp(r_A) + exp(r_B)) = σ(r_A − r_B)
```

Take the negative log-likelihood over your preference dataset of (prompt `x`, chosen `y_w`, rejected `y_l`) triples:

```
L = −E[ log σ( r_θ(x, y_w) − r_θ(x, y_l) ) ]
```

That is the entire loss. The reward model is the base LM with the LM head replaced by a scalar head reading the hidden state of the final token; `r_θ(x, y)` is that scalar.

```python
def bt_loss(reward_model, batch, margin=None):
    # batch has chosen_ids/mask and rejected_ids/mask, same prompt prefix
    r_w = reward_model(batch["chosen_ids"], batch["chosen_mask"])     # [B]
    r_l = reward_model(batch["rejected_ids"], batch["rejected_mask"]) # [B]
    diff = r_w - r_l
    if margin is not None:                # margin variant, e.g. Llama-2 style
        diff = diff - margin              # margin from rating-gap metadata
    return -F.logsigmoid(diff).mean(), (diff > 0).float().mean()      # loss, acc
```

The **margin variant** exists because not all preferences are equally strong. If your annotation UI collects "slightly better / better / much better," you can subtract a per-example margin `m` so that a "much better" pair must clear a larger gap to stop contributing gradient. The Llama-2 paper used this form. Without it, a barely-preferred pair and an overwhelmingly-preferred pair exert identical pressure, which wastes capacity on coin flips.

Three properties to state unprompted. First, the loss is invariant to adding a constant to all rewards — only differences are identified — which is why reward scores are meaningless across two separately-trained RMs and why you must whiten/normalize rewards before feeding them into PPO. Second, pairwise accuracy (`diff > 0`) is the metric you report, and a well-trained RM lands somewhere in the 65–75% range on held-out human preferences, because human annotators only agree with each other around 70–80% of the time. Third, the RM must be evaluated *out of distribution* — on completions from the policy you intend to optimize — because that is where it will be queried.

**⚠ Trap:** reporting RM eval accuracy near 90% and treating it as success. That almost always means your chosen/rejected pairs are separable by a surface feature: length, markdown formatting, or which model generated them. Run the length-only baseline — a "reward model" that just returns the token count — and report its accuracy alongside. If length alone gets 65% and your RM gets 72%, your RM contributes 7 points of real signal, and RLHF against it will produce a policy that mostly learns to write longer.

### Show me how DPO's implicit reward drops out of the RLHF objective. Why is that derivation the whole point of the method?

The mental model: DPO's contribution is not a new loss; it is the observation that the KL-constrained RLHF objective has a *closed-form* optimal policy, and that this closed form can be inverted to express the reward in terms of the policy. Once you have reward-as-a-function-of-policy, you substitute it into the Bradley-Terry likelihood and the reward model disappears entirely. The policy becomes its own reward model.

The RLHF objective is: maximize expected reward minus a KL leash to the reference (SFT) policy,

```
max_π  E_{x, y~π}[ r(x,y) ] − β · KL( π(·|x) ‖ π_ref(·|x) )
```

This is a standard KL-regularized bandit problem and its exact solution is a Boltzmann tilt of the reference policy:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

You can verify this by writing the objective as a negative KL to that target distribution plus a constant. Now take logs and solve for `r`:

```
r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

That is the implicit reward: **β times the log-ratio of policy to reference**, plus a prompt-dependent constant. Substitute into Bradley-Terry, `P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l))`, and the `β log Z(x)` term cancels because both completions share the same prompt `x`. What survives:

```
L_DPO = −E[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

**📄 Paper:** Rafailov et al. (2023), "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." What it replaced: the three-stage SFT → reward model → PPO pipeline, collapsing it into a single supervised-style training loop with no sampling, no value network, and no reward model in memory.

Implementation-wise this is four forward passes per step conceptually — policy on chosen and rejected, reference on chosen and rejected — but the reference log-probs are constant, so you precompute them once over the dataset and cache them, halving the compute and freeing the reference model's weights from GPU memory.

**⚠ Trap:** believing that because the reward model vanished, the reward-hacking problem vanished. It did not — it moved. DPO's gradient pushes up `log π(y_w)` and pushes down `log π(y_l)`, and there is nothing constraining the *total* probability mass: the well-documented failure is that both chosen and rejected log-probs decrease over training while their gap widens, meaning the model is fleeing to unlabeled regions of the output space. The instrument you must log is not the loss — it is `mean log π_θ(y_w)` and `mean log π_θ(y_l)` as separate curves, plus the implicit-reward margin and the reward accuracy. If `log π(y_w)` is falling, stop the run.

**🗣 Say this in the room:** "The KL-constrained RLHF optimum is `π_ref · exp(r/β)` normalized. Invert it and the reward equals β times the log policy-to-reference ratio plus a prompt constant. Put that into Bradley-Terry, the constant cancels between the two completions, and you're left with a pure supervised loss on log-ratios. That's DPO — the reward model was always implicit in the policy."

### Explain the MoE auxiliary load-balancing loss, and then explain why DeepSeek argued you should drop it.

Mental model: an MoE router is a learned scheduler, and like every scheduler it has a degenerate optimum — send everything to the one expert that is currently best, which makes that expert better, which attracts more traffic. Routing collapse is a positive-feedback loop, and the auxiliary loss is a hand-written damper on it. The problem with a damper on the loss is that it is a *gradient* that fights the *quality* gradient, and you are hard-coding a trade-off you cannot inspect.

The Switch/GShard-style auxiliary loss, per layer:

```
L_aux = α · N · Σ_{i=1..N} f_i · P_i
```

where `N` is expert count, `f_i` is the fraction of tokens in the batch actually dispatched to expert `i`, and `P_i` is the mean router probability assigned to expert `i` over the batch. `f` is non-differentiable (it comes from a top-k argmax) and `P` carries the gradient, so the product yields a gradient that pushes probability mass away from over-subscribed experts. It is minimized when both are uniform at `1/N`, giving `L_aux = α·N·N·(1/N²) = α`. **📄 Paper:** Fedus, Zoph and Shazeer (2021), the Switch Transformer, with `α = 0.01` as the standard coefficient.

**Router z-loss** is the companion: `λ_z · mean( (logsumexp(router_logits))² )`, same idea as output z-loss but on the tiny router softmax, where exploding logits are especially damaging because they make the top-1 selection brittle and the gradient near-zero. **📄 Paper:** Zoph et al. (2022), ST-MoE.

DeepSeek's argument, in their auxiliary-loss-free load balancing work and shipped in DeepSeek-V3: the auxiliary loss is a gradient that is *not* the language-modelling gradient, so it degrades quality in exchange for balance, and you cannot tune that exchange rate well. Replace it with a non-gradient control loop. Maintain a per-expert bias `b_i` that is added to the router's affinity scores *for the purposes of top-k selection only* (the gating weight used to scale the expert output stays un-biased, so no gradient flows through `b`). After each step, adjust: if expert `i` was over-loaded, `b_i -= γ`; if under-loaded, `b_i += γ`. That is a proportional controller on queue depth — precisely the shape of a load balancer you have written before — and it steers routing without perturbing the loss surface at all.

**🗣 Say this in the room:** "The auxiliary loss balances experts by adding a competing gradient, so you're trading quality for balance at a rate you set with a magic constant. DeepSeek's auxiliary-loss-free approach replaces it with a per-expert routing bias updated by a feedback controller — balance is enforced in the selection step, not in the gradient. It's the difference between penalizing a hot shard in the loss function and just adjusting its weight in the load balancer."

**⚠ Trap:** measuring balance only in aggregate over a whole epoch. Expert utilization is near-uniform on average while being catastrophically skewed *per batch* and *per domain* — code tokens all hitting three experts, for instance. The metric that matters for throughput is the max-to-mean tokens-per-expert ratio within a single dispatch, because the slowest expert sets the step time in an expert-parallel deployment. Log the per-step maximum, not the running mean.
### Lay out the pretraining objective families for me — causal LM, masked LM, prefix-LM, span corruption — and tell me which survived and why.

Mental model: every pretraining objective is a choice about *which tokens are visible when predicting which other tokens*. That is it. The architecture (encoder, decoder, encoder-decoder) is downstream of the mask pattern, not the other way around. Once you see it as a visibility matrix, the family tree is obvious.

**Causal LM** (GPT family): predict token `t+1` from tokens `0..t`. Lower-triangular visibility. Every position produces a training signal, so you get `T` predictions per sequence of `T` tokens — 100% token efficiency. That efficiency is the reason it won: at fixed compute, you extract more gradient signal per token read from disk than any competitor.

**Masked LM** (BERT): corrupt ~15% of tokens with `[MASK]` and predict them from full bidirectional context. Bidirectional context is strictly more information per prediction, which is why encoder models still beat decoders per-parameter at classification and embedding. But you only score 15% of positions, so token efficiency is ~6.7× worse, and there is a train/serve mismatch because `[MASK]` never appears at inference. MLM did not lose on quality; it lost on the scaling economics and on the fact that it cannot generate.

**Prefix-LM** (UniLM, PaLM's mixture): bidirectional attention over a prefix, causal over the suffix. Visibility is a block matrix: full attention inside the prefix, causal after. This is genuinely useful when the input is fully known ahead of time — a document you are summarizing gains from bidirectional encoding — and it is the honest ancestor of "the prompt could be encoded bidirectionally." It lost mostly on ecosystem: KV caching, continuous batching, and every serving engine assume a single causal mask.

**Span corruption** (T5): mask contiguous spans, replace each with a sentinel token, and have the decoder emit the sentinel-plus-content sequence. It makes MLM generative and denser than single-token masking. **📄 Paper:** Raffel et al. (2020), T5, which systematized this whole comparison and is still the best single empirical study of objective choice.

The verdict I would give in a room: causal LM won on token efficiency, on the fact that it is the only objective whose training-time and inference-time computation are identical, and on the emergent finding that a big enough causal LM does classification and embedding well enough via prompting. Bidirectional objectives survive exactly where generation is not required — embedding and reranking models — and that niche is not shrinking.

**⚠ Trap:** claiming causal LM won because it is "more natural" or "how humans read." It won on economics. Per unit of FLOPs and per token of scraped data, it produces more supervision than anything else, and the field was data- and compute-bound, not idea-bound.

### Explain fill-in-the-middle. Why does every serious code model train on it, and what's the classic serving bug?

Mental model: pure left-to-right prediction is the wrong shape for the actual job of a code assistant. When a developer's cursor sits in the middle of a function, the model has a prefix *and* a suffix, and a causal LM trained only left-to-right has no way to condition on the code below the cursor. FIM is a data transformation — not an architecture change — that teaches a causal model to use both.

The mechanism is a preprocessing trick applied to a fraction of the pretraining documents. Split a document at two random points into `prefix`, `middle`, `suffix`. Then re-serialize with sentinel tokens so that a purely causal model, reading left to right, sees the suffix *before* it has to produce the middle:

```
PSM order:  <PRE> prefix <SUF> suffix <MID> middle <EOT>
SPM order:  <PRE> <SUF> suffix <MID> prefix middle <EOT>
```

Training loss is ordinary next-token CE over the whole rearranged sequence. At inference you construct the same layout with an empty middle and let the model generate until `<EOT>`. **📄 Paper:** Bavarian et al. (2022), "Efficient Training of Language Models to Fill in the Middle," which established the transformation and — the load-bearing result — showed that at a moderate FIM rate the model gains infilling ability essentially *free*, with no measurable degradation of left-to-right capability. That "FIM-for-free" property is why it is universal in code models rather than a specialty variant.

SPM ("suffix-prefix-middle") exists for a caching reason that will feel familiar: in an editor, the prefix above the cursor changes on every keystroke while the suffix below is stable. Putting the suffix first means the stable part is the shared prefix of the token sequence, so it stays in the prefix cache across keystrokes.

**⚠ Trap — the classic serving bug:** using the wrong sentinel strings or the wrong order at inference. The model was trained on specific special token IDs; if your serving path builds the FIM prompt as plain text (`"<fim_prefix>"` as characters, tokenized into five ordinary subword tokens) instead of injecting the reserved token IDs, the model has literally never seen this input and degrades into vaguely-plausible continuation. It will not error. It will not look broken in a smoke test. It will just be quietly 20 points worse on acceptance rate, and the metric that catches it is offline exact-match on a held-out infill set, not eyeballing.

**💰 Math:** this matters commercially. A code-completion product at 5M completion requests/day with a 25% acceptance rate that drops to 20% because of a sentinel mismatch loses 250k accepted completions/day. If you also pay for the generation — say 300 tokens in, 40 out, at $0.20/$0.60 per Mtok for a small self-hosted-equivalent rate — the tokens cost `5e6 × (300×2e-7 + 40×6e-7) = 5e6 × 8.4e-5 = $420/day` regardless of acceptance, so the entire loss is on the value side, not the cost side. That asymmetry is why code-assistant teams instrument acceptance rate per prompt-construction path, not just latency.

### What loss values should I actually expect to see? Give me the sanity anchors.

This is one of the highest-leverage things to have memorized, because it converts "the loss looks fine" into a falsifiable statement. Mental model: cross-entropy in nats is the log of an effective branching factor. `exp(loss)` is the perplexity — roughly, how many equally-likely tokens the model is choosing among. Every anchor below is just a branching factor you can reason about.

**📐 Numbers you must know:**
- **Step 0 of pretraining from random init** = `ln(V)`. For a 128k vocabulary that is `ln(131072) = 11.78`. For a 32k vocab, `ln(32000) = 10.37`. If your first loss is not within a few percent of `ln(V)`, your initialization or your label alignment is wrong, and you should stop before spending money. This is the single cheapest bug detector in training.
- **A well-trained modern LLM on general English web text**: roughly 1.8–2.3 nats/token, i.e. perplexity ~6–10. **📅 Volatile:** this depends heavily on tokenizer and data mix — a model with a bigger vocabulary packs more characters per token and therefore has *higher* loss per token for the same underlying quality. Never compare loss across tokenizers; convert to bits-per-byte if you must.
- **Code**: lower, ~0.7–1.2 nats, because code is far more predictable than prose (boilerplate, keywords, closing brackets).
- **SFT on instruction data**: typically starts near 1.2–1.8 and settles to 0.6–1.0 over an epoch or two. Below ~0.3 you are memorizing; check for train/eval overlap.
- **Contrastive embedding training**, step 0: `ln(B)` for a `B`-way in-batch problem — `ln(4096) = 8.32`.
- **A `B`-way router or classifier**, step 0: `ln(n_classes)`.

The diagnostic procedure: at step 0, compute the theoretical random-init loss by hand and compare. At step 100, the loss should have dropped fast (the model learns the unigram frequency distribution almost immediately — that alone gets you from `ln(V)` to roughly the entropy of the token unigram distribution, around 6–7 nats). If it has not moved by step 100, your learning rate is too low or your gradients are not flowing. If it went to NaN, see the loss-spike ladder.

**🏋 Drill:** given a config (`vocab_size`, `n_layers`, `d_model`) and a screenshot of a loss curve, state in 60 seconds whether the curve is plausible and, if not, which of the four canonical bugs it indicates (double shift, all-masked labels, wrong vocab size in the head, LR too low). Pass criterion: correct call on 8 of 10 synthetic curves.

### Derive Adam from SGD. I want to know what each of the two moments is buying you.

Start from plain SGD: `θ ← θ − η·g`. Its defect is that a single global `η` must serve every parameter, but gradient magnitudes across a transformer differ by orders of magnitude — embedding rows for rare tokens get gradient roughly `1/frequency` of common ones, LayerNorm gains get tiny gradients, attention output projections get large ones. Any `η` that moves the small-gradient parameters will blow up the large-gradient ones.

**Momentum** fixes a different problem: gradient noise. `m ← β₁·m + (1−β₁)·g` is an exponential moving average with an effective window of `1/(1−β₁)` steps — at `β₁ = 0.9` that is a 10-step average. It attenuates the stochastic component of the minibatch gradient while preserving the consistent component, which is a low-pass filter, exactly the same device as an EWMA on a noisy metric before you alert on it.

**The second moment** is the per-parameter scaling. `v ← β₂·v + (1−β₂)·g²` estimates the uncentered variance of each coordinate's gradient over a window of `1/(1−β₂)` steps — at `β₂ = 0.999`, a 1,000-step average. Then divide:

```
θ ← θ − η · m̂ / (√v̂ + ε)
```

Now the update magnitude is roughly `η · sign-like`, because `m/√v` is dimensionless — a coordinate with consistently large gradients and a coordinate with consistently tiny gradients both move about `η` per step. Adam is, to first order, a smoothed sign-SGD with a per-coordinate trust region. That is the entire value proposition, and it is why Adam is non-negotiable for transformers while SGD+momentum remains competitive for CNNs (whose gradient scales are far more uniform).

**Bias correction**: `m` and `v` are initialized to zero, so early estimates are biased toward zero by a factor of `(1 − β^t)`. Divide them out: `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)`. Without this, the first steps take a wildly wrong effective LR — and note it takes about 1/(1−β₂) = 1,000 steps for `v` to be well-estimated at all, which is one of the two real reasons LR warmup exists.

```python
def adam_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    p.add_(mhat / (vhat.sqrt() + eps), alpha=-lr)
```

**📄 Paper:** Kingma and Ba (2014). **⚠ Trap:** placing `eps` inside the square root (`√(v̂ + ε)`) instead of outside. PyTorch puts it outside; some papers put it inside; the two differ meaningfully when `v` is near zero, which is exactly the regime for rarely-updated embedding rows. It is a real behaviour difference, not a typo, and it is the kind of thing that makes a from-scratch reimplementation not match the reference.

### Adam versus AdamW — what's actually different, and why did every LLM recipe switch?

The mental model: L2 regularization and weight decay are the same thing under SGD and *different things* under any adaptive optimizer. Adam's classic implementation folds the L2 penalty into the gradient, where it then gets divided by `√v` along with everything else. That means the effective decay applied to each parameter is inversely proportional to its gradient magnitude — parameters with large, noisy gradients get almost no regularization, and parameters that are barely being updated get decayed hard. That is precisely backwards from what you want.

The fix is one line of code and it is entirely about *where* the decay term goes:

```python
# Adam with L2 (wrong for adaptive methods):
g = g + wd * p                    # decay enters the moment estimates
...update with m/sqrt(v)...

# AdamW (decoupled):
p.mul_(1 - lr * wd)               # decay applied directly to the weights
...then the ordinary Adam update from the un-penalized gradient...
```

**📄 Paper:** Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (2017, ICLR 2019). What it replaced: the default Adam-with-L2 that had made Adam look worse than SGD+momentum on generalization for years — a substantial part of that gap turned out to be this bug.

Two consequences that get asked as follow-ups. First, in AdamW the decay is coupled to the *learning rate* (`p *= 1 − lr·wd`), so decaying the LR on a cosine schedule also decays your regularization strength — the two hyperparameters are not independent, and if you change your schedule you have implicitly changed your regularization. Second, decoupling means you can reason about weight decay as a simple exponential shrinkage with a known half-life: at `lr = 3e-4` and `wd = 0.1`, each step multiplies weights by `1 − 3e-5`, so an unupdated weight halves in `ln(2)/3e-5 ≈ 23,100` steps.

**⚠ Trap:** applying weight decay to everything. The convention that every serious recipe follows is to exclude LayerNorm/RMSNorm gains, all biases, and usually the embedding and output-head weights from decay, applying it only to the 2D matrices. Decaying a norm gain toward zero is actively destructive — it shrinks the residual stream's scale, which the network then has to compensate for elsewhere. The parameter-group split looks like:

```python
decay, no_decay = [], []
for n, p in model.named_parameters():
    (no_decay if p.ndim < 2 or "norm" in n or "embed" in n else decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)
```

The `p.ndim < 2` heuristic is the one I actually use — it catches every bias and every norm gain without string matching, which is fragile across model families.

### Do the memory arithmetic for me. How much GPU memory does a full fine-tune of a 7B model in mixed precision actually take, before activations?

Mental model: parameters are the *smallest* of the four terms. The four-term equation is weights + gradients + optimizer state + activations, and for AdamW mixed precision the optimizer state alone is four times the size of the bf16 weights.

Walk the standard mixed-precision setup, per parameter:
- bf16 weights (used for the forward/backward): **2 bytes**
- bf16 gradients: **2 bytes**
- fp32 master weights (the authoritative copy the optimizer updates): **4 bytes**
- fp32 Adam first moment `m`: **4 bytes**
- fp32 Adam second moment `v`: **4 bytes**

Total: **16 bytes per parameter.** For 7B parameters: `7e9 × 16 = 1.12e11 bytes = 112 GB`. That does not fit on an 80 GB H100. It does not fit on an H200's 141 GB once you add activations and fragmentation. This is the number that explains, in one line, why everyone uses ZeRO/FSDP sharding or LoRA.

**💰 Math, extended:** shard those 112 GB across 8 GPUs with FSDP/ZeRO-3 and you get 14 GB/GPU of persistent state, leaving ~60 GB per H100 for activations and communication buffers — comfortable. With ZeRO-2 (optimizer + gradients sharded, weights replicated) you hold 2 GB/param-copy replicated plus 14/8... concretely: replicated bf16 weights `7e9×2 = 14 GB` plus sharded `(2+4+4+4)=14 bytes/param ÷ 8 = 12.25 GB`, so ~26 GB/GPU. With LoRA at rank 16 on the attention projections you might train ~20M parameters: optimizer state becomes `2e7 × 12 = 240 MB`, and the frozen base weights are 14 GB in bf16 — the whole thing fits on a single 24 GB card with a short context. That is a 4.7× memory reduction from one architectural choice.

**📐 Numbers you must know:** **16 bytes/param for AdamW mixed-precision full fine-tuning; 8 of those bytes are the two fp32 Adam moments.** Memorize both halves, because the follow-up question is always "so what does 8-bit Adam buy you?" — answer: it quantizes `m` and `v` to 1 byte each, taking 16 → 10 bytes/param, a 37.5% cut in persistent state for essentially no quality loss on fine-tuning-scale runs.

**⚠ Trap:** quoting the "2 bytes per parameter" inference number in a training context. Inference of a 7B model in bf16 is 14 GB of weights plus KV cache; training the same model is 112 GB before a single activation. Conflating these is the single most common way a backend engineer gets caught out on a capacity question. And the follow-up trap: gradient accumulation does **not** reduce this — accumulation reduces *activation* memory per micro-batch, and leaves all four persistent terms untouched.

### Walk me through Adafactor, Lion, and 8-bit Adam. When would you actually reach for each?

All three are answers to the same question — "Adam's `m` and `v` cost 8 bytes/param, can I pay less?" — and they trade different things.

**Adafactor** (Shazeer and Stern, 2018) attacks `v`. For a 2D weight matrix of shape `[n, m]`, instead of storing the full `n×m` second-moment matrix, it stores a rank-1 factorization: row sums (`n` values) and column sums (`m` values), reconstructing `v ≈ (r ⊗ c)/sum(r)`. Memory for `v` drops from `O(nm)` to `O(n+m)`. Optionally it drops momentum entirely (`β₁ = 0`), which removes the other 4 bytes. It also introduces update clipping and relative step sizes. It was the workhorse of the T5 era and remains reasonable when memory is the binding constraint and you can tolerate more hyperparameter fiddling. It is generally regarded as slightly worse than AdamW at equal budget for LLM pretraining.

**Lion** (Chen et al., 2023, "Symbolic Discovery of Optimization Algorithms") attacks both. It keeps only momentum and takes the *sign* of an interpolated momentum as the update direction:

```
update = sign(β₁·m + (1−β₁)·g);   m ← β₂·m + (1−β₂)·g;   θ ← θ − lr·(update + wd·θ)
```

One state tensor instead of two: 4 bytes/param instead of 8. Because the update is a pure sign, every parameter moves by exactly `lr` in magnitude, so the LR must be roughly 3–10× smaller than AdamW's and weight decay correspondingly larger. It works well in practice, particularly at large batch sizes, but it is more sensitive to hyperparameters and less battle-tested for very long runs.

**8-bit Adam** (bitsandbytes, from the block-wise quantization work of Dettmers et al.) keeps Adam's exact algorithm and quantizes the two state tensors to 8 bits using block-wise dynamic quantization — each block of 2048 values gets its own scale, so a single outlier cannot destroy the block. State goes 8 → 2 bytes/param.

My decision rule: **for fine-tuning, 8-bit AdamW is the default** — it changes nothing about the optimization dynamics you understand and it is a drop-in `bnb.optim.AdamW8bit`. Reach for Adafactor only in memory-desperate situations with very wide matrices. Reach for Lion only if you have budget to re-tune LR and WD and are running large-batch pretraining. And for a pretraining run at any real scale, the memory question is usually answered by sharding (ZeRO/FSDP) rather than by changing the optimizer, because sharding costs you bandwidth rather than optimization quality.

**⚠ Trap:** swapping optimizers and keeping the learning rate. Lion at AdamW's LR diverges immediately. This sounds obvious and it is nevertheless the most common failed experiment I see — someone reports "Lion is worse" after a single run at the inherited LR, which is not evidence about Lion.

### What are Shampoo, SOAP and Muon actually doing differently, and should I care as an applied engineer?

Mental model: Adam treats the parameter tensor as a bag of independent scalars and gives each its own scalar learning rate. That throws away the fact that a weight matrix has *structure* — the coordinates are not independent, and the gradient's covariance across rows and columns is highly non-isotropic. This family of optimizers exploits that structure by preconditioning with matrix-valued (rather than diagonal) statistics.

**Shampoo** (Gupta, Koren, Singer, 2018) maintains, for a weight matrix `W ∈ R^{n×m}`, two preconditioner matrices: `L ≈ Σ G Gᵀ` (`n×n`) and `R ≈ Σ Gᵀ G` (`m×m`), and applies `L^{-1/4} · G · R^{-1/4}` as the update. This is a Kronecker-factored approximation to full-matrix AdaGrad — full-matrix would need an `(nm)×(nm)` preconditioner, which is absurd, so it factors it into row-space and column-space pieces. The cost is periodic inverse-quarter-roots of `n×n` matrices, which you amortize by recomputing every ~50–100 steps.

**SOAP** (Vyas et al., 2024) makes the connection explicit: it shows Shampoo is equivalent to running Adafactor in Shampoo's eigenbasis, and then just runs *Adam* in that eigenbasis instead, which is both simpler and empirically better. The eigenbasis is refreshed infrequently, so the per-step cost is close to Adam's.

**Muon** (Keller Jordan, 2024) is the one that actually broke through into large open training runs. It applies momentum, then *orthogonalizes* the resulting update matrix via a few Newton–Schulz iterations — approximating `U Vᵀ` from the update's SVD, which is the nearest semi-orthogonal matrix. The intuition is that a rank-deficient update wastes a step; orthogonalizing spreads the update energy across all singular directions equally. It applies only to 2D parameters; embeddings, the output head, and all 1D parameters are still trained with AdamW. It has been used in large-scale open model training, including by Moonshot for Kimi K2 with an added QK-clipping mechanism to control attention-logit growth. **📅 Volatile:** this area is moving fast — verify current adoption before claiming any particular frontier model uses it.

Should you care as an applied engineer? Honest answer, and I would say it exactly this way: **not for your day job, but yes for the interview.** You will not swap optimizers on a fine-tuning run — the wins are in the 1.3–2× tokens-to-target-loss range for *pretraining*, and fine-tuning is not where that matters. But "do you follow what's happening in optimization" is a legible research-literacy signal, and the three-sentence version above is what separates a candidate who reads from one who does not.

**🗣 Say this in the room:** "Adam is diagonal preconditioning — one scalar LR per coordinate. Shampoo, SOAP and Muon are all attempts to use the matrix structure of the gradient instead. Shampoo uses Kronecker-factored second moments, SOAP runs Adam inside Shampoo's eigenbasis, and Muon just orthogonalizes the momentum with Newton–Schulz so no singular direction is wasted. The reported wins are in pretraining tokens-to-target-loss, not in fine-tuning."

### Why does learning-rate warmup exist? Give me the mechanism, not the folklore.

Mental model: at step 1 the optimizer has no idea what the gradient distribution looks like, and both of Adam's estimators are maximally wrong at exactly the moment the loss surface is steepest. Warmup is not superstition; it is waiting for the second-moment estimate to become a valid estimate.

Two concrete mechanisms, and a good answer names both.

**Mechanism one — the variance of the adaptive step.** Adam's update is `m̂/(√v̂ + ε)`. At small `t`, `v̂` is estimated from a handful of samples, so its variance is enormous, and `1/√v̂` for a small sampled `v` produces an enormous step. `β₂ = 0.999` implies an effective averaging window of 1,000 steps; the estimate is not trustworthy until you are a good fraction of the way into that window. This is the argument formalized in the RAdam line of work, which proposed rectifying the adaptive term instead of warming up the LR — and notably, that work's own framing is that warmup is a heuristic fix for the same underlying problem.

**Mechanism two — the residual stream and the loss landscape at init.** At initialization the model is at a high-curvature point in a badly-conditioned region. A full-size step there can push the network into a state that is hard to recover from — attention entropy collapses, logits explode, and the effective learning rate over the first few hundred steps determines the basin you end up in. This is the empirical observation behind the fact that removing warmup from a Post-LN transformer causes divergence, while a Pre-LN transformer is much more tolerant of no warmup.

Practical shape: linear ramp from 0 (or a tiny value) to peak LR over a warmup period. **📐 Numbers you must know:** the conventions worth having are roughly **2,000 steps or ~0.5–1% of total steps for pretraining**, and **3–5% of total steps for fine-tuning** (where runs are short and a 2,000-step warmup would be the entire run). If you use a fixed step count, remember it interacts with your batch size — 2,000 steps at 4M tokens/step is 8B tokens of warmup, which for a small run is absurd.

**⚠ Trap:** treating warmup length as free. Too-short warmup produces an early loss spike that the run may or may not recover from; too-long warmup wastes tokens at a low LR and measurably costs final loss on a fixed budget. It is also one of the few hyperparameters whose failure signature is unmistakable — plot grad-norm against step, and a run with insufficient warmup shows a grad-norm spike in the first few hundred steps that clipping then flattens into a plateau.

### Cosine schedule versus WSD/trapezoid — which do you pick and why does it matter for anything other than the final loss?

Mental model: a cosine schedule bakes your total training length into the schedule shape. That is not a small inconvenience — it means you cannot stop early, cannot extend, and cannot branch. WSD decouples "how long do I train" from "how do I anneal," and that decoupling is worth more operationally than the small final-loss difference between them.

**Cosine**: after warmup, anneal `lr` from peak to some floor (commonly 10% of peak) following `0.5·(1 + cos(π·t/T))`. It has been the default since GPT-3-era recipes and it works well. Its defining property is that `T` — the total step count — is a parameter of the curve. Stop at `0.6·T` and you get a model that was trained at a still-high LR and is measurably worse than a model that ran a full cosine over `0.6·T` steps. Your intermediate checkpoints are not usable models.

**WSD (warmup–stable–decay), also called trapezoid**: warm up, then hold LR *constant* for the bulk of training, then decay sharply over the last ~10–20% of steps. **📄 Paper:** the MiniCPM report (Hu et al., 2024) is the most-cited systematic study, showing WSD matches or beats cosine at equal budget. The operational payoff is large:

1. **You can decide the run length after starting it.** Keep training in the stable phase; when you decide to stop, branch off and run a decay. This is how continued-pretraining recipes work in practice.
2. **You get multiple final models from one run** by branching decays at different points, which makes scaling-law data collection dramatically cheaper — you no longer need a separate full run per token budget.
3. **The stable-phase checkpoint is the right thing to continue from** for domain adaptation, because it has not been annealed into a sharp minimum.

The decay phase is where a disproportionate amount of the loss improvement happens, and it is also where high-quality data is typically upweighted (the "annealing data mix"). That combination — LR decay plus a data-quality shift — is a standard modern recipe.

**🗣 Say this in the room:** "Cosine hard-codes the total step count into the curve, so intermediate checkpoints are undertrained and you can't extend a run. WSD holds LR constant and puts a short sharp decay at the end, which matches cosine's final loss but lets you branch a decay from any point. For anything where the budget might change — which is every real project — I'd pick WSD."

**⚠ Trap:** setting the cosine's `T` to a number of steps you then don't reach because the job was preempted or the data ran out. You now have a model annealed to, say, 40% of a cosine, which is strictly worse than a properly-scheduled shorter run and is not fixable without retraining the tail. On a preemptible cluster this alone justifies WSD.

### Explain gradient clipping. What does the clipping rate tell you that the loss doesn't?

Mental model: clipping is a rate limiter on the update, and like every rate limiter its *trigger frequency* is a better health signal than its effect. The loss curve tells you where you are; the fraction of steps that clip tells you how turbulent the ride is.

Mechanism, global-norm clipping (the only variant that should be used for transformers): compute the L2 norm of the *concatenation of all gradients*, `‖g‖ = √(Σ_p ‖g_p‖²)`. If `‖g‖ > c`, scale every gradient by `c/‖g‖`. Crucially this preserves the *direction* of the update — it shortens the step without rotating it. Per-parameter clipping does rotate it, which distorts the update in ways that interact badly with Adam, and I would flag it in review.

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# returns the PRE-clip norm — log this every step, it is your best stability metric
```

`clip_grad_norm_` returns the norm *before* clipping, which is the number you want on a dashboard. Typical value: `c = 1.0`, occasionally 0.5 for unstable runs.

**🔍 Failure taxonomy — reading the grad-norm curve:**
- *Grad norm stable at ~0.3, clipping fires on <1% of steps.* Healthy. Clipping is doing its job as insurance.
- *Clipping fires on >20% of steps.* Your `c` is below the natural scale of your gradients; you have converted AdamW into a strange normalized-gradient method and your effective LR is now data-dependent. Either raise `c` or lower LR — but understand you are no longer running the algorithm you think you are.
- *Grad norm slowly rising over thousands of steps.* Logit growth or residual-stream variance growth. Add z-loss / QK-norm; check for a norm layer whose gain is drifting up.
- *Isolated grad-norm spikes of 10–100× with no loss impact.* Almost always a specific data shard — a repeated token, a corrupted document, a page of base64. Log the batch indices at spike time and go read the data.
- *Grad norm collapses to ~0 while loss is flat.* Dead run: either all labels are masked, or you have hit an fp16 underflow with a GradScaler that has driven the scale into the floor, or a `.detach()` crept into the path.

**⚠ Trap:** clipping *after* the optimizer's `.step()`, or before `unscale_` when using an fp16 GradScaler. In the AMP path the gradients are still multiplied by the loss scale when the backward finishes; clipping them at that point clips against a meaningless threshold that changes every time the scaler adjusts. The correct order is `scaler.unscale_(opt)` → `clip_grad_norm_` → `scaler.step(opt)` → `scaler.update()`. This is a silent correctness bug: nothing raises, the run just behaves unpredictably.

### Explain gradient accumulation and effective batch size — and then tell me about the normalization bug that bit half the open-source fine-tuning ecosystem.

Mental model: gradient accumulation trades wall-clock for memory by simulating a large batch as a sequence of small ones. Because gradients are additive, `∇L(A ∪ B) ∝ ∇L(A) + ∇L(B)` — you run several micro-batches, let the gradients pile up in `p.grad`, and step once. Effective batch = `micro_batch × accum_steps × data_parallel_world_size`.

```python
for i, batch in enumerate(loader):
    with model.no_sync() if (i + 1) % accum else contextlib.nullcontext():
        loss = model(**batch).loss / accum      # scale so the sum is a mean
        loss.backward()
    if (i + 1) % accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()
```

Two details. `no_sync()` on non-final micro-steps suppresses DDP's all-reduce, which otherwise fires on *every* backward and multiplies your communication cost by `accum`. And `sched.step()` belongs with `opt.step()`, not in the inner loop — otherwise your LR schedule runs `accum` times too fast, which is a shockingly common bug that manifests as "my LR hit the floor a quarter of the way through the run."

**The normalization bug.** Divide by `accum` and you have implicitly assumed every micro-batch contains the *same number of loss-bearing tokens*. It does not — sequences have different lengths and different amounts of `ignore_index` masking. Each micro-batch's loss is a mean over its own token count `n_i`, so summing `Σ (L_i / accum)` gives you an unweighted average of per-micro-batch means, not the mean over all tokens. Micro-batches with few scored tokens get the same weight as ones with many. The correct reduction is:

```
L_total = (Σ_i sum_of_token_losses_i) / (Σ_i n_i)
```

which requires computing the loss with `reduction="sum"` per micro-batch, tracking `n_i`, and dividing once at the end — and, in a data-parallel setting, all-reducing the token count too. This was a real, widely-discussed bug in mainstream fine-tuning libraries (surfacing in 2024 in the HuggingFace Trainer's gradient-accumulation path and in several downstream trainers), and its effect is that runs with `accum > 1` produced systematically different — and worse — results than the mathematically equivalent large-batch run, with the discrepancy scaling with how variable your sequence lengths are.

**⚠ Trap:** the reason this survived so long is that it does not look like a bug. The loss curve is smooth, training completes, the model is usable — it is just a few percent worse, and nobody A/B tests `accum=1` against `accum=8` at matched effective batch because "they're obviously equivalent." The lesson generalizes: **any time you claim two configurations are mathematically equivalent, write the test that asserts it.** That is a backend instinct you already have; carry it over.

### How do you set batch size and learning rate together? Talk me through the scaling rules and where they stop working.

Mental model: batch size buys you gradient *quality* — the variance of the minibatch gradient estimate falls as `1/B`. Learning rate spends that quality on step size. The scaling rules are all statements about how much extra step size a given variance reduction buys you, and every one of them stops working past a point called the critical batch size.

**The two classic rules.** Linear scaling (`lr ∝ B`) comes from SGD analysis: doubling `B` halves the gradient variance, so you can double the step. It was validated at scale for SGD+momentum on ImageNet (Goyal et al., 2017, the "1 hour ImageNet" work, which also popularized the warmup-with-linear-scaling combination). Square-root scaling (`lr ∝ √B`) is the rule usually preferred for Adam, on the argument that Adam's update is already normalized by `√v` and the noise enters differently. In practice for LLM training, people scale sub-linearly and then tune, and the honest answer in an interview is: **these rules are starting points for a sweep, not laws.** Anyone who states one as settled fact is overclaiming.

**Critical batch size** is the concept that actually matters and is the one worth naming. **📄 Paper:** McCandlish et al. (2018), "An Empirical Model of Large-Batch Training," introduced the gradient-noise-scale estimate. Below the critical batch size, doubling `B` roughly halves the number of steps to a target loss — you get near-perfect parallel efficiency. Above it, doubling `B` barely reduces step count at all, so you are burning twice the FLOPs for nothing. The critical batch size grows as training proceeds (the gradient gets noisier relative to its magnitude as you approach the optimum) which is why batch-size ramp-up schedules exist in large pretraining runs.

**💰 Math — why you care:** suppose your critical batch size is 2M tokens and you run at 8M because you had the GPUs. You are paying roughly 4× the FLOPs per unit of progress in the regime where returns have flattened. On a run that would have cost $500k at the efficient batch size, that is $1.5M of pure waste. Conversely, running at 256k tokens/batch when 2M is efficient means 8× more optimizer steps, each with a full gradient sync — you become communication-bound and your MFU collapses. The batch-size decision is a cost decision, and framing it that way is the senior signal.

**🗣 Say this in the room:** "I'd start from a target effective batch in tokens, not sequences, because that's the unit the gradient noise actually scales with. Then set LR by scaling from a known-good pair — roughly square-root for Adam — and verify with a short LR sweep at small scale. The thing I'd actually be watching is whether we're past critical batch size: if halving the batch doesn't increase steps-to-target-loss proportionally, we're wasting compute."

### What are μP and μTransfer, and what problem do they solve that a hyperparameter sweep doesn't?

Mental model: under standard initialization and parametrization, the optimal learning rate *shifts* as you widen a model — a 256-wide model and a 4096-wide model have different best LRs, so hyperparameters tuned on a cheap proxy do not transfer to the expensive run. μP is a re-parametrization designed so that the optimal hyperparameters become *width-invariant*, which means you can tune on a small model and transfer the settings to a big one for free. That is the whole product.

Mechanically, μP (Maximal Update Parametrization) prescribes width-dependent scalings for three things: initialization variance, the learning rate, and per-layer output multipliers. The design criterion is that as width `n → ∞`, the *change in each layer's activations per optimizer step* stays `Θ(1)` — neither vanishing (the model stops learning as it widens) nor exploding. Under standard parametrization it does not: the per-step activation change scales with width, so the optimal LR must shrink to compensate, and that is exactly why LR does not transfer.

The three-part recipe in rough terms: input/embedding layers, hidden layers, and output layers get *different* treatment. Hidden-layer weights get init variance `∝ 1/fan_in` and Adam learning rate `∝ 1/fan_in`; the output layer gets an explicit `1/fan_in` multiplier on its logits; embeddings are treated like input layers with width-independent LR. **📄 Paper:** Yang and Hu (2021) on the theory (Tensor Programs series), and Yang et al. (2022) on μTransfer, which demonstrated tuning on a small proxy and transferring to a 6.7B model.

The practical workflow: build a proxy model at, say, `d_model = 256` with the same depth and data, sweep LR (and optionally batch size, init scale, warmup) there for a few hundred GPU-hours, then transfer the winner to the target width. The reported result is that the LR-vs-loss curve's minimum sits at the same LR across widths, which is the visual you should have in your head — a family of U-curves whose minima are vertically aligned.

**💰 Math:** the value is straightforward. A 20-point LR sweep at `d_model=256` might cost 200 GPU-hours; the same sweep at the target scale would cost 20 × the full run. If the full run is 50,000 GPU-hours, the sweep would be a million GPU-hours — obviously impossible, which is why pre-μP the answer was "guess, and accept you're 20% off optimal." Getting within a factor of 1.3 of optimal LR is worth several percent of final loss, which at these budgets is millions of dollars of equivalent compute.

**⚠ Trap:** μP transfers across **width**, and depth transfer is a separate and much less settled problem. If you widen and deepen simultaneously and the LR does not transfer, μP is not broken — you were using it outside its guarantee. Also: μP requires changing your init, your per-layer LRs, *and* your output multiplier consistently. Implementing one of the three (a common shortcut: just the LR scaling) gives you none of the guarantee and a model that trains differently from both baselines.
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
### LayerNorm versus RMSNorm — write both, and tell me what Llama gave up by dropping mean-centering.

Mental model: normalization in a transformer exists to keep the residual stream's *scale* controlled so that downstream matmuls see inputs in a predictable range. LayerNorm does that with a full standardization — subtract the mean, divide by the standard deviation, then apply a learned gain and bias. RMSNorm keeps only the scaling half. The empirical finding, which is the whole story, is that the re-centering contributes essentially nothing to quality while costing you a pass over the data and a second reduction.

```python
# LayerNorm: 2 reductions (mean, then variance about the mean), 2 params
y = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + eps)
y = y * gain + bias

# RMSNorm: 1 reduction, 1 param, no bias
y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gain
```

**📄 Paper:** Zhang and Sennrich (2019) introduced RMSNorm and made exactly this argument — that LayerNorm's benefit comes from re-scaling invariance, not re-centering invariance. Llama adopted it, and Qwen, Mistral, Gemma and essentially every open model since have followed.

What you actually save. One reduction instead of two along `d_model`, no subtraction, and one fewer parameter tensor. Because normalization is a *memory-bound* elementwise op — you read `B·T·d` elements and write `B·T·d` elements while doing almost no math — the saving is real in wall clock, not just FLOPs. The commonly cited figure is on the order of **7–15% of step time** for the norm layers in an LLM, translating to a few percent of total training time. **📅 Volatile:** the exact figure depends entirely on whether your norms are fused; a fused Triton RMSNorm and a fused LayerNorm are much closer than the unfused versions, so quote the mechanism (one reduction vs two, memory-bound) rather than the percentage.

What you give up: the guarantee that the activation mean is zero. In practice modern transformers do not need it, partly because the residual stream is not centered anyway and partly because the learned gain plus the subsequent linear layer's own bias-free structure absorbs any offset. Note the related trend — most recent LLMs also dropped **biases** from their linear layers entirely, for the same "it doesn't help and it costs" reason plus a mild stability benefit.

**⚠ Trap:** implementing RMSNorm and computing the statistics in bf16. `x.pow(2).mean(-1)` over `d_model = 4096` elements in bf16, with a residual stream that has a few large-magnitude outlier features, gives a materially wrong RMS. Upcast to fp32 for the statistic, compute `rsqrt` in fp32, cast back before applying the gain — as in the snippet in the numerics discussion. And put `eps` **inside** the square root as a variance floor; outside, an all-zero row gives `inf`.

**🗣 Say this in the room:** "RMSNorm keeps the re-scaling and drops the re-centering. Zhang and Sennrich showed re-centering wasn't doing the work, so you save a full reduction over `d_model` on an op that's purely memory-bound, plus the bias parameter. That's why every model since Llama uses it."

### Pre-LN, Post-LN, sandwich — where exactly do you put the normalization and what changes?

Mental model: the residual stream is a highway, and the question is whether the normalization sits *on* the highway (Post-LN) or on the *on-ramp to each block* (Pre-LN). Post-LN normalizes the sum, so the identity path is repeatedly rescaled and the gradient must pass through a norm at every layer on its way back. Pre-LN leaves the identity path completely clean, so the gradient has a straight-through path from the loss to layer 0.

```
Post-LN (original Transformer):   x ← LN( x + Attn(x) );  x ← LN( x + FFN(x) )
Pre-LN  (GPT-2 onward):           x ← x + Attn(LN(x));    x ← x + FFN(LN(x))
Sandwich:                         x ← x + LN_post( Attn( LN_pre(x) ) )
```

**📄 Paper:** Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture," gave the gradient analysis: with Post-LN, the expected gradient norm at initialization grows with depth, which is why the original transformer *required* warmup to train at all; with Pre-LN the gradients are well-behaved at init and warmup becomes optional (though still helpful). That result is the reason Pre-LN became the default for every large model.

The catch with Pre-LN, and this is the part a good answer includes: because nothing normalizes the residual stream itself, its variance **grows monotonically with depth**. Each block adds its output to the stream, and those additions accumulate. By layer 60 of an 80-layer model the residual magnitude can be an order of magnitude larger than at layer 2. Two consequences follow. First, later blocks contribute proportionally less, because `LN(x)` divides out the now-large stream — deep Pre-LN models have measurably "wasted" late layers, and this is one hypothesized reason layer-pruning works as well as it does on them. Second, the large dynamic range across depth is exactly the setting where bf16 activations and quantization start to hurt.

**Sandwich / peri-LN** is the response: normalize both before *and* after the sublayer, so each block's *contribution* to the stream is bounded even though the stream itself is not re-normalized. Gemma 2 uses a pre- and post-norm arrangement around each sublayer. A 2025 line of work analyzing residual-stream variance growth argues this "peri-layer" placement gives Pre-LN's trainability with Post-LN's controlled activation growth. **📅 Volatile:** this is an active area; describe the mechanism (bound the block's contribution without normalizing the highway) rather than asserting a settled winner.

**⚠ Trap:** a final norm before the output head is not optional in a Pre-LN model. Since nothing normalizes the residual stream, the last block's output has that accumulated large variance, and feeding it directly to the LM head produces enormous logits. Every Pre-LN model has a `final_norm` after the last block for exactly this reason; omit it in a from-scratch implementation and you will see logits in the hundreds and an immediate divergence.

**🗣 Say this in the room:** "Post-LN normalizes the residual sum, which puts a norm in the gradient path at every layer and makes the model require warmup to train at all. Pre-LN moves the norm inside the branch, leaving a clean identity path, at the cost of residual-stream variance growing with depth. Sandwich normalization is the compromise: normalize the block's output so its contribution is bounded, but leave the highway alone."

### How would you initialize a transformer from scratch? Give me the actual numbers and justify the residual scaling.

Mental model: initialization has one job — make the forward activations and the backward gradients have roughly unit scale at every layer *before any training happens*. If activations grow by 1.2× per layer, an 80-layer model has a 1.2^80 ≈ 1.6e6 amplification and the first forward pass overflows. Every init scheme is a variance-preservation argument, and the residual scaling is a variance-preservation argument specifically about the residual stream.

**Xavier/Glorot** (Glorot and Bengio, 2010): `Var(W) = 2/(fan_in + fan_out)`, derived by requiring variance preservation in both the forward *and* backward direction simultaneously, for a linear/tanh network. **Kaiming/He** (He et al., 2015): `Var(W) = 2/fan_in`, which adds the factor of 2 to compensate for ReLU zeroing half the activations. For a transformer with GELU/SwiGLU the Kaiming form is the right family.

In practice, GPT-2-lineage models use a simpler recipe that is worth being able to state verbatim: **`N(0, 0.02)` for all linear and embedding weights, zeros for biases, ones for norm gains** — plus one correction.

**The residual scaling.** Consider the residual stream. Each of the `2·L` residual branches (attention out-projection and FFN down-projection, per layer) adds its output to the stream. If each addition has variance `σ²` and they are roughly independent, the stream's variance after `2L` additions is `2L·σ²` — growing linearly with depth, so its standard deviation grows as `√(2L)`. To keep the stream at unit scale at initialization, divide the initial std of every *residual output projection* by `√(2L)`:

```python
for name, p in model.named_parameters():
    if name.endswith("attn.o_proj.weight") or name.endswith("mlp.down_proj.weight"):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
```

The GPT-2 paper states this as scaling residual-layer weights by `1/√N` with `N` the number of residual layers; the `2·n_layers` count is the standard implementation because a transformer block has two residual additions. Note this applies **only** to the projections that write *into* the residual stream, not to Q/K/V or the FFN up-projection.

**Embeddings** deserve their own thought. Many recipes initialize the token embedding at a *smaller* std than the rest, or normalize it, because at init the embedding output goes straight into the first norm and its scale sets the whole stream. If your input and output embeddings are tied, remember the same tensor is doing double duty as an embedding lookup and as a logit projection — an init good for one may be poor for the other, which is part of why large models increasingly untie them.

**⚠ Trap:** initializing and then never checking. The five-minute test that catches every init bug: run one forward pass on random data with hooks that print the per-layer activation RMS. It should be roughly flat across depth, not growing geometrically. Then run one backward and print per-layer gradient norms — also roughly flat. If activations grow 1.5× per layer you have found your future NaN before you spent a dollar on it, and this test costs less than the coffee you drink while writing it.

### What is attention entropy collapse and how does it show up in a training run?

Mental model: attention is a softmax, and a softmax's output distribution has an entropy. Early in training that distribution is nearly uniform over the context (high entropy). As the model learns, heads sharpen. Collapse is when a head sharpens *all the way* — it puts essentially all its mass on one position for every query — and at that point the softmax is saturated, its gradient with respect to the logits is nearly zero, and the head stops learning. It is a dead unit, and enough of them make the loss stall or spike.

The mechanism that produces it: attention logits are `q·k/√d`, and nothing bounds `‖q‖` or `‖k‖`. The weight matrices `W_Q` and `W_K` can grow, the logit range grows with them, the softmax sharpens, and there is a positive feedback loop — a sharper head gets a cleaner gradient signal for the direction it already picked, which grows the weights further. Left alone this diverges. **📄 Paper:** Zhai et al. (2023), "Stabilizing Transformer Training by Preventing Attention Entropy Collapse," named this failure and proposed a spectral reparametrization of the attention weight matrices (σReparam) to bound the logits.

The instrumentation is what makes this an engineering answer rather than a trivia answer. Log, per layer and per head, the mean attention entropy `−Σ p log p` over a fixed probe batch, every few hundred steps. A healthy run shows entropy falling from near `ln(T)` and settling at a spread of values across heads — some heads are sharp (induction, previous-token), some stay diffuse. The pathology is entropy going to ~0 across *many* heads at once, and it typically precedes the loss spike by hundreds of steps, which makes it a genuine leading indicator rather than a post-mortem.

The fixes are the ones already in your kit: **QK-norm** (normalize Q and K before the dot product, bounding logit magnitude directly), **logit softcapping**, spectral normalization of `W_Q`/`W_K`, or simply lowering the LR. QK-norm is the one that has actually been adopted at scale because it is cheap and does not touch the fused attention kernel.

**⚠ Trap:** confusing entropy collapse with **attention sinks**. A sink is a head that dumps mass on token 0 (or a few fixed early tokens) as a learned no-op — "I have nothing to attend to right now." That is healthy and universal, and its entropy is low by design. Collapse is when the *task-relevant* heads saturate and stop moving. Distinguish them by looking at whether the low-entropy mass is on a fixed absolute position (sink) or on a query-dependent position (possibly fine) or is frozen and unresponsive to the input (collapse). If you report "my attention entropy is low, that's collapse" without checking which, you will be corrected in the room.

### Give me your loss-spike debug ladder. I want an ordered procedure, not a list of things that can go wrong.

**🔍 Failure taxonomy — the loss-spike ladder.** Run these in order; each rung is more expensive than the one above it, and you stop at the first one that works.

**Rung 0 — classify the spike.** Plot loss, pre-clip grad norm, and (if you have it) attention entropy on a common time axis, around the spike. Three signatures, three different treatments:
- *Loss spikes, grad norm spikes at the same step, both recover within ~50 steps.* Benign. This is a hard batch. Do nothing; if it happens every few thousand steps that is normal for a large run.
- *Loss spikes and does not recover; grad norm stays elevated.* The optimizer state is now poisoned — the second moment absorbed a huge gradient and the effective LR is wrong for thousands of steps. You must intervene.
- *Grad norm was creeping up for thousands of steps before the spike.* This is not a data problem. It is logit or residual-variance growth, and skipping the batch will not help — it will spike again in 500 steps.

**Rung 1 — skip the batch.** Add a guard: if the pre-clip gradient norm exceeds `k×` the running median (I use `k = 5` on a 1,000-step median), zero the gradients and skip the optimizer step. This costs one wasted micro-batch and rescues the run from single pathological documents. It is the cheapest possible intervention and it should be in your trainer by default, not added reactively.

```python
gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
median.update(gn.item())
if gn.item() > 5 * median.value or not math.isfinite(gn.item()):
    opt.zero_grad(set_to_none=True); skipped += 1; continue
opt.step()
```

**Rung 2 — lower `β₂`.** If the run recovers from spikes only slowly, the culprit is Adam's second moment: at `β₂ = 0.999` a single enormous gradient inflates `v` and it takes ~1,000 steps to decay out, during which the effective LR for those parameters is suppressed. Dropping to `β₂ = 0.95` shortens that memory to ~20 steps, so a spike is forgotten in tens of steps instead of thousands. This is a well-known large-run stabilization and it costs a little optimization quality in exchange for much faster recovery. The companion knob is `eps`: *raising* it from 1e-8 toward 1e-6 damps the adaptive scaling for coordinates whose `v` has collapsed toward zero, which caps how large the `1/√v̂` amplification can get. Both are one-line config changes and both are reversible, which is why they sit this high on the ladder.

**Rung 3 — add z-loss (and router z-loss if MoE).** If your logit magnitudes or `logsumexp` are trending upward across the run, pin them. `λ = 1e-4` on the output head, and on the router if you have one.

**Rung 4 — add QK-norm.** If attention entropy is collapsing or attention logits are trending up, bound them at the source. Note this is an *architecture* change: you are adding parameters mid-run, so you must handle checkpoint loading, and the model you end up with is not the model you started training. Acceptable in a research run, painful in a production one — which is why you decide this before you start, not at step 40k.

**Rung 5 — rewind and skip the data.** Restore the last checkpoint before the spike, advance the data loader past the offending shard (this is why deterministic, checkpointed data ordering matters), and resume. This is the standard practice on very large runs and it is why the open reports of frontier pretraining describe manual intervention as routine rather than exceptional.

**Rung 6 — lower the peak LR and restart from an earlier checkpoint.** The most expensive rung. Only after the others, and only when the spikes are recurrent rather than isolated.

**🗣 Say this in the room:** "First I classify the spike from grad-norm behaviour: an isolated co-spike that recovers is a bad batch and I skip it with a running-median guard; a slow grad-norm creep before the spike is logit growth and skipping won't help. Then the ladder is skip-batch, lower β₂ to 0.95 so a spike is forgotten in tens of steps instead of a thousand, add z-loss, add QK-norm, and only then rewind past the data shard."

### Grad norm has been climbing steadily for 8,000 steps in a fine-tune, loss is still descending. Do you intervene?

Yes, and the interesting part is *why* the loss looking fine is not reassurance. Mental model: a monotonic grad-norm trend means some scale in the network is growing without a restoring force, and gradient clipping is masking the consequence. Clipping preserves direction while shrinking magnitude, so as the raw norm grows past your clip threshold, your effective step size is being silently reduced — you are no longer running AdamW at your configured LR, you are running normalized-gradient descent at whatever rate `c/‖g‖` implies. The loss keeps descending because the direction is still right. The run is drifting away from the algorithm you designed.

The diagnosis, in order. First, check the **clip rate**: what fraction of steps are actually clipping? If it went from 2% to 40% over those 8,000 steps, that confirms the effective-LR story. Second, decompose the norm **per parameter group** — log `‖g‖` separately for embeddings, attention projections, FFN, norm gains, and the LM head. A global number tells you nothing; a group-level number usually points straight at the culprit. In my experience on fine-tunes the two usual answers are the LM head (logits growing, fix with z-loss) and the norm gains (a norm's learned scale drifting up, which then amplifies everything downstream).

Third, check `logsumexp` of the logits and the max absolute logit on a fixed probe batch. If `log Z` has gone from 8 to 25, you have your answer and z-loss is the fix. Fourth, check the **weight** norms, not just the gradient norms — if you excluded norm gains and embeddings from weight decay (which you should have) then nothing is restraining their growth at all, and a gain that has drifted from 1.0 to 3.5 is a real and fixable finding.

The intervention, in the order I would apply it on a fine-tune specifically (where restarts are cheap, unlike pretraining): re-check that weight decay is applied to the 2D matrices; add z-loss at 1e-4; if the LR is at the top of its plausible band, halve it; and if the grad-norm growth is localized to a specific layer group, consider freezing or lowering LR on that group.

**💰 Math:** the cost of not intervening. A 7B SFT run on 8×H100 for 6 hours is `8 × 6 × $2.50 = $120` (**📅 Volatile:** on-demand rates). Cheap enough that "just restart with z-loss" is obviously correct — the expensive resource is the *engineer-week* spent later trying to explain why the fine-tuned model quantizes badly. And it will quantize badly: a model with a wide activation dynamic range and outlier features is exactly the model where int8/int4 post-training quantization falls apart, so a stability problem you tolerated in training becomes a serving problem you cannot fix without retraining.

**⚠ Trap:** raising the clip threshold to make the clip-rate metric look healthy again. That is treating the thermometer. Clipping is not the problem; it is the only thing currently keeping the run alive.

### Design the training telemetry for a fine-tuning platform used by a hundred internal teams. What do you put on the dashboard and what do you alert on?

I would frame this exactly as you would frame observability for a multi-tenant service, because it is one — the jobs are heterogeneous, the users are not experts, and the failure modes are silent. The design goal is that a team with no ML background can tell whether their run is healthy without asking me.

**Per-step metrics (high frequency, cheap):** training loss; pre-clip global grad norm; clip rate as a rolling percentage; current LR; tokens/second and MFU; step time broken into forward, backward, optimizer, and data-wait; GPU memory allocated versus reserved (the gap is fragmentation); the count of skipped steps.

**Per-N-steps metrics (a probe batch, fixed across the whole run so numbers are comparable):** `logsumexp` mean and max-absolute-logit; per-parameter-group gradient norms and weight norms; attention entropy per layer; for MoE, per-expert token counts with max-to-mean ratio; for DPO/preference runs, `mean log π(chosen)` and `mean log π(rejected)` as separate series plus the implicit reward margin and accuracy.

**Per-checkpoint metrics:** a generative eval on the real inference path (not teacher-forced), a small general-capability suite to detect forgetting, and eval loss on a held-out slice.

**The four automated gates that stop a run** — this is where the platform earns its keep, because these are the failures that waste the most money:
1. **Step-0 loss check.** Assert the first loss is within 5% of `ln(vocab_size)` for pretraining, or within a configured band for fine-tuning. Fail the job immediately if not. This catches label misalignment, wrong vocab size, and broken masking before the second minute.
2. **Label-mask assertion.** On the first batch, decode `input_ids[labels != -100]` and log it as text into the run artifacts. Do not gate on it — just make it impossible to *not* see. Half the escalations I have handled would have been self-served by this one line of output.
3. **Non-finite guard.** Any non-finite loss or grad norm skips the step, increments a counter, and pages the owner if the counter exceeds 0.1% of steps.
4. **Throughput regression gate.** If tokens/sec drops more than 20% below the first-500-step baseline, alert. This catches `torch.compile` recompile-limit fallbacks, a data loader that started hitting cold storage, and a straggler rank — all of which are invisible in the loss curve and all of which cost real money.

**💰 Math for the platform pitch:** suppose 100 teams run an average of 20 jobs a month, average 4 GPU-hours each, at $2.50/GPU-hour → `100 × 20 × 4 × 2.50 = $20,000/month` of fine-tuning spend. Internal audits of platforms like this routinely find 20–40% of jobs are dead on arrival — wrong masking, wrong template, diverged in the first 100 steps — and nobody noticed until the eval came back. The step-0 gate alone reclaims a large share of that, so a week of platform engineering pays for itself in under two months and, more importantly, converts a class of silent failures into loud ones.

**🗣 Say this in the room:** "The metric that catches the most bugs for the least effort is asserting that step-0 loss equals `ln(vocab)`. After that it's grad-norm-pre-clip and clip rate as the stability pair, `logsumexp` as the leading indicator of logit drift, and tokens/sec against a first-500-step baseline to catch silent throughput regressions that the loss curve can never show you."

### Give me a concrete recipe: 7B model, SFT on 100k instruction examples, 8×H100. What do you set and why?

I would state the recipe with the reasoning attached to each number, because "AdamW, 2e-5, cosine" is a memorized answer and the reasoning is what is being tested.

**Precision and memory.** bf16 with fp32 master weights, FSDP full-shard. Persistent state is 16 bytes/param × 7e9 = 112 GB, sharded over 8 GPUs = 14 GB/rank, leaving ~60 GB for activations. With activation checkpointing at block granularity and FlashAttention, I can fit a per-device micro-batch of a few sequences at 4k context comfortably.

**Effective batch size.** Target ~1–2M tokens per optimizer step? No — that is a *pretraining* target. For SFT on 100k examples, a much smaller effective batch is right: **128–256 sequences**, roughly 0.5–1M tokens if sequences average 4k. Reasoning: 100k examples at effective batch 128 gives `100000/128 ≈ 780` steps per epoch. Fewer than ~500 steps and the LR schedule has no room to do anything; more than a few thousand and you are over-fitting a small dataset. So: micro-batch 4 per device × 8 devices × accumulation 4 = 128 sequences.

**Optimizer.** AdamW, `β₁ = 0.9`, `β₂ = 0.95` (not 0.999 — short runs benefit from the shorter second-moment memory and faster spike recovery), `eps = 1e-8`, `weight_decay = 0.0–0.1` applied only to `p.ndim >= 2`. For a 2-epoch SFT I usually run `wd = 0.0`; the run is too short for regularization to matter and it is one fewer interaction with the LR schedule.

**Learning rate.** `1e-5` to `2e-5` peak for a full fine-tune of a 7B. The heuristic worth stating: **SFT LR is one to two orders of magnitude below pretraining LR** (which for a 7B is ~3e-4), because you are adapting a converged model, not training one. For LoRA the LR is 10–20× higher (1e-4 to 3e-4) because the adapter is randomly initialized and the update is low-rank.

**Schedule.** Warmup over 3% of total steps (≈47 steps of 1,560 for 2 epochs), then cosine decay to 10% of peak. For SFT specifically I do not use WSD — the run length is known and fixed, so cosine's main drawback does not apply.

**Clipping.** Global norm 1.0, with a skip-if-5×-median guard.

**Packing.** Pack multiple examples per sequence to 4k with correct cross-attention masking (`cu_seqlens` / a block-diagonal mask), because unpacked instruction data is mostly padding. **⚠ Trap:** packing *without* the document mask lets example B attend to example A. This trains cross-contamination, does not error, and typically shows up as the model referencing content that was not in its prompt. If your kernel does not support varlen masking, do not pack.

**💰 Math:** `100k examples × ~1,500 tokens avg = 1.5e8 tokens`. Two epochs = 3e8 tokens. Training FLOPs ≈ `6 × 7e9 × 3e8 = 1.26e19`. At 8×H100 and a realistic 40% MFU on bf16 (`8 × 989e12 × 0.4 ≈ 3.2e15` FLOP/s), that is `1.26e19 / 3.2e15 ≈ 3,940 s ≈ 1.1 hours` of math; call it ~2 hours wall clock with checkpointing overhead and data stalls. At $2.50/GPU-hour that is `8 × 2 × 2.50 = $40`. **📅 Volatile:** GPU pricing moves; the structure of the calculation is the durable part. The point to make out loud: at $40 a run, the correct strategy is to run three seeds and report the variance, not to agonize over one configuration.

### You're asked to continue pretraining a base model on 20B tokens of proprietary domain data. What breaks, and what's your LR schedule?

Mental model: continued pretraining is a re-entry problem. The model arrived at its current weights via a specific LR trajectory that ended in an annealed, low-LR state. If you now jump the LR back to a high value, you knock it out of that basin — the loss spikes, general capability drops, and it re-converges somewhere worse than where it started. If you set the LR too low, the model does not actually learn the domain. The whole design is about re-entering at the right energy.

**What breaks, in order of how often I see it:**

1. **Catastrophic forgetting.** Train on pure domain data and general capability collapses. The standard mitigation is a **replay mix**: 5–30% of the original-distribution data blended in throughout. There is no universal ratio; the honest answer is that you pick it by running a small sweep and measuring both domain loss and a general benchmark, and that the right ratio depends on how far your domain is from the pretraining distribution.
2. **The initial loss spike.** Almost universal on the first few hundred steps as the model adapts to the new distribution. It is usually benign and recovers, but it is much worse without warmup — re-warm over 1–2% of steps even though this is a "continued" run.
3. **Tokenizer mismatch.** If your domain has vocabulary the tokenizer fragments badly (a genomics corpus, a proprietary code language, a non-Latin script), your effective tokens-per-document explodes and the model spends capacity on tokenization artifacts. Measure bytes-per-token on your corpus versus on general text before you start. Extending the tokenizer is possible but means new randomly-initialized embedding rows, which need their own warmup treatment.
4. **Data ordering effects.** Continued pretraining is short enough that the model retains a recency bias toward the last data it saw. Shuffle globally; do not train domain-by-domain sequentially.

**The schedule I would use:** WSD, explicitly. Re-warm from ~0 to a peak of roughly **10% of the original pretraining peak LR** (so ~3e-5 if the base was trained at 3e-4) over 1–2% of steps; hold constant for ~80% of the run; decay to ~10% of that peak over the final 10–20%, and **upweight your highest-quality domain data during the decay phase**. This is the standard annealing recipe and the reason WSD is the right family here is precisely the branching property — 20B tokens is a big enough commitment that you will want to evaluate at multiple points, and with WSD you can branch a short decay from any stable-phase checkpoint to get a usable model without committing to a total length up front.

**💰 Math:** 20B tokens on a 7B model is `6 × 7e9 × 2e10 = 8.4e20` FLOPs. At 64×H100 and 40% MFU (`64 × 989e12 × 0.4 = 2.53e16` FLOP/s) that is `8.4e20 / 2.53e16 ≈ 33,200 s ≈ 9.2 hours` of math — call it ~12 hours wall clock. At $2.50/GPU-hour, `64 × 12 × 2.50 = $1,920`. **📅 Volatile.** That is cheap enough that the replay-ratio sweep at 1/10 scale (2B tokens, ~$200 per arm) is obviously worth running before committing, and framing it that way — "I'd spend 10% of the budget de-risking the mix" — is the answer that lands.

**⚠ Trap:** evaluating a continued-pretraining run only on domain perplexity. Domain perplexity will improve monotonically as the model memorizes your corpus, including as it forgets everything else. You must run a general suite at every checkpoint, and you must decide *in advance* how much general regression you are willing to accept, because the trade-off is real and someone will otherwise negotiate it after the fact.

### 🏋 Implement AdamW and softmax cross-entropy with its backward, from scratch, in 25 minutes. What am I grading?

**The drill.** No autograd, no `F.cross_entropy`, no optimizer library. Given `logits: [N, V]` and `targets: [N]`, implement: (1) the forward loss with numerically-stable log-sum-exp and `ignore_index` support, (2) the analytic gradient with respect to the logits, and (3) an AdamW step including bias correction and decoupled decay. Then verify (2) against `torch.autograd` and (3) against `torch.optim.AdamW` on a toy problem.

```python
def xent_fwd_bwd(logits, targets, ignore_index=-100):
    z = logits.float()
    valid = targets != ignore_index
    n = valid.sum().clamp(min=1)
    m = z.max(dim=-1, keepdim=True).values
    lse = m.squeeze(-1) + torch.log(torch.exp(z - m).sum(-1))     # [N]
    tgt = targets.clamp(min=0)
    nll = lse - z.gather(1, tgt[:, None]).squeeze(1)               # [N]
    loss = (nll * valid).sum() / n
    p = torch.exp(z - lse[:, None])                                # softmax, stable
    grad = p
    grad.scatter_add_(1, tgt[:, None], -torch.ones_like(tgt[:, None], dtype=z.dtype))
    grad = grad * valid[:, None] / n
    return loss, grad

def adamw_step(p, g, m, v, t, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    p.mul_(1 - lr * wd)                          # decoupled decay, BEFORE the update
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    step = (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
    p.add_(step, alpha=-lr)
```

**Pass criteria, which is what I am actually grading:**
1. The gradient matches `torch.autograd.grad` on random `[64, 1000]` logits to `atol=1e-6` in float64, **including** rows with `ignore_index`.
2. The mean is divided by the count of *valid* tokens, not `N`. This is the single most common failure and I check it first.
3. `logsumexp` is computed with the max subtracted. If you wrote `torch.log(torch.exp(z).sum(-1))` you fail regardless of whether the test happens to pass on well-conditioned random data — I will hand you logits with a value of 100 in them.
4. Bias correction is present and applied to both moments.
5. Weight decay is applied to the parameter, not added to the gradient. If you wrote `g = g + wd * p` you implemented Adam-with-L2 and you should be able to explain, unprompted, why that differs.
6. The AdamW step matches `torch.optim.AdamW` on 20 steps of a quadratic to `atol=1e-6`.

**Time budget:** 8 minutes for the loss and gradient, 7 for AdamW, 10 for the verification harness. If you cannot write the verification harness you have not finished — an unverified numerical kernel is not a deliverable, and saying that out loud is worth points.

### 🏋 Two-minute memory drill: I give you a config, you tell me if it fits.

**The drill.** For each configuration below, state within two minutes whether it fits, and if not, what the single highest-leverage change is. Do it on paper. The pass criterion is a correct fit/no-fit call on all five plus a correct primary remedy on all five.

Use these three formulas, which you should be able to write without hesitation:
- **Persistent state (full FT, AdamW mixed precision)** = `16 bytes × params`
- **Activations with FlashAttention, no checkpointing** ≈ `16 × B × T × d × L` bytes in bf16
- **KV cache per token** = `2 × n_layers × n_kv_heads × d_head × bytes_per_elem`

**Case 1.** Full fine-tune, 8B params, 1×H100 80 GB. → `8e9 × 16 = 128 GB` of persistent state alone. No fit, not close. Remedy: LoRA (frozen base is `8e9 × 2 = 16 GB`, adapter optimizer state is negligible) or shard across nodes. Note that gradient checkpointing does *not* help here — it touches activations, and activations are not the problem.

**Case 2.** Full fine-tune, 8B, 8×H100 with FSDP full-shard, 4k context, micro-batch 2. → Persistent: `128/8 = 16 GB`/rank. Activations, no checkpointing: `16 × 2 × 4096 × 4096 × 32 × 2 bytes`... compute it: `16 × 2 × 4096 × 4096 = 5.37e8` elements per layer, × 32 layers × 2 bytes = **34.4 GB**. Total ~50 GB/rank. Fits on 80 GB with headroom. Remedy if tight: block-level checkpointing drops activations to ~2 GB for +33% compute.

**Case 3.** Same as case 2 but 32k context. → Activations scale linearly with `T`: `34.4 × 8 = 275 GB`/rank. No fit. Remedy: activation checkpointing first (it is the only thing that touches the term that grew), then sequence/context parallelism if still short.

**Case 4.** Serving Llama-3-70B-shaped (80 layers, 8 KV heads, `d_head` 128) in bf16, 2×H100. → Weights `70e9 × 2 = 140 GB`, leaving 20 GB across the pair. KV per token: `2 × 80 × 8 × 128 × 2 = 327,680 bytes = 320 KiB`. At 8k context that is 2.5 GiB per sequence, so ~8 concurrent sequences. Marginal. Remedy: fp8 or int8 weight quantization halves the weight footprint and roughly quadruples the concurrency.

**Case 5.** Continued pretraining, 70B, 64×H100, ZeRO-3. → `70e9 × 16 = 1,120 GB / 64 = 17.5 GB`/rank of persistent state. Fits comfortably; the binding constraint here is not memory but interconnect bandwidth for the parameter all-gathers, and the right follow-up question is about your NVLink/InfiniBand topology, not your memory.

**⚠ Trap:** in every case, check *which term grew* before choosing a remedy. Gradient checkpointing fixes activations. Sharding fixes persistent state. Quantization fixes weights. Applying the wrong one is the most common wasted afternoon in this whole discipline, and being able to say "checkpointing won't help you here, your problem is optimizer state" in five seconds is a strong seniority signal.

### Last one: you're in a room with a research engineer discussing their pretraining run. What do you ask, and what do you avoid saying?

This is the question the whole section serves, so let me answer it as advice rather than as mechanism.

**What to ask, in rough order of how much respect it earns:**
- "What's your grad-norm-to-clip-threshold ratio, and what fraction of steps clip?" — this immediately signals you have watched a real run rather than read about one.
- "Are you on cosine or WSD, and if cosine, is the total step count locked?" — invites the interesting conversation about branching and budget flexibility.
- "What's your `β₂`, and did you lower it after a spike?" — the `0.999 → 0.95` move is an in-group signal.
- "What does your loss-spike playbook look like — do you skip batches automatically or intervene manually?"
- "What's your MFU, and where does the gap to peak go?" — moves the conversation to systems, which is where your backend background is an asset rather than a gap.
- "How do you decide the data mix for the decay phase?"

**What to avoid saying:**
- Anything that implies training is just a bigger version of fine-tuning. It is not; the failure modes are different, the budgets make experimentation impossible, and the person you are talking to has spent months on problems that do not exist at fine-tuning scale.
- Confidently asserting a specific optimizer or architecture is "what everyone uses now." This field moves faster than any of us can track and the person across the table probably ran the ablation. Say "last I checked, X was the common choice — has that moved?" and you convert a potential correction into a conversation.
- Quoting a benchmark number without its date and setup.
- Suggesting a fix without asking what they already tried. The ladder in this section is the *obvious* ladder; assume they have been up it.

**🗣 Say this in the room:** "I haven't run pretraining at scale — my depth is in the serving and post-training layer. What I can do is reason about the numerics and the memory arithmetic, and I know the stability ladder well enough to be useful in a debugging conversation. Where I'd want to learn from you is what actually goes wrong above the scale where the textbook fixes stop working."

That last line is the honest positioning for an applied AI engineer, and it is much stronger than pretending. The archetype of role you are targeting does not need you to have trained a 70B model. It needs you to be someone a research engineer can talk to without translating, and someone who will never ship a fine-tune with an unmasked prompt, an unverified gradient, or an uninstrumented loss curve.


---

## 9. Classical ML, Pre-Transformer NLP, and the "Is an LLM Even Right Here?" Gate

*Mastering this proves the maturity signal interviewers explicitly look for: knowing when not to apply an LLM, and saying so.*

### Before you design anything, what's the checklist you run to decide whether this feature needs an LLM at all?

The mental model: an LLM is a probabilistic, non-idempotent, rate-limited, per-call-billed remote dependency with no schema guarantee and a p99 measured in seconds. You already have an instinct for what that kind of dependency costs you — you would never put one on a hot path without a very specific reason. The entire gate is just forcing yourself to name that reason before you write the prompt, because once a prompt exists nobody ever deletes it.

The checklist I actually run, in order, and I run it out loud in design review:

1. **Is the output space finite and known?** If the answer is one of N labels, one of N routes, or a number, you are describing a classifier or a query, not a generator. The LLM is only justified if N is large, open, or changes weekly.
2. **Is the input format stable?** Fixed-layout documents, well-formed JSON, a known CSV schema — a parser is exact, costs microseconds, and fails loudly. LLMs are for the long tail of format, not the head.
3. **Does a deterministic system already know the answer?** If the fact lives in Postgres, the correct architecture is SQL. An LLM that reads rows and re-derives an aggregate is a slow, expensive, occasionally-wrong replacement for `SUM()`.
4. **Is there a correctness authority other than human taste?** If a solver, a type checker, a test suite, a regex or a business rule can *verify* the answer, the LLM's job is to propose, not to decide.
5. **What is the cost of being wrong 3% of the time, and who eats it?** For an autocomplete suggestion, nothing. For a compliance decision, an enforcement action.
6. **Does the head of the distribution repeat?** If 40% of queries are 200 distinct strings, precompute them and the LLM only sees the tail.
7. **Do I have labels?** With 5,000 labeled examples, a supervised model is usually both cheaper and better. With 40 examples and an open output space, the LLM is genuinely the only thing that works.
8. **What is my latency budget and my QPS?** At 2,000 QPS with a 50 ms budget, the conversation is over — no frontier API meets that, and you are now discussing a distilled small model or a classical model.

**🗣 Say this in the room:** "My gate is: known finite output space, stable input format, an existing source of truth, and an external verifier. If three of those four are true, I ship a deterministic system and use the model only where the format is genuinely open — and I'd rather spend the model budget on the 8% tail than on the 92% that a parser handles exactly."

**⚠ Trap:** the failure mode this checklist prevents is not "we used an LLM unnecessarily." It is *entanglement*. Once the model is on the path, the schema becomes suggestive rather than enforced, retries become non-idempotent, your p99 becomes the provider's p99, and every downstream consumer starts defensive-parsing. Removing it six months later is a rewrite. That is why the gate runs before the prompt exists and not after.

### A PM wants an LLM to pull invoice number, date and total out of our vendor PDFs. The layout is identical every time. What do you tell them?

I tell them no, and then I tell them where the LLM *does* go, because "no" alone reads as incapacity.

The mental model: extraction from a fixed layout is a parsing problem, and parsers have a property no model has — they fail loudly on unexpected input instead of confidently inventing a plausible value. If a vendor changes their template, a regex returns `None` and your pipeline raises. A model returns an invoice number that looks exactly like an invoice number and is wrong. The first failure costs you an alert; the second costs you a wrong payment and a reconciliation project.

The mechanism I ship: text layer via `pdfplumber` or `pypdf` (or `pdftotext -layout` if the layout is column-sensitive), then anchored regex or positional extraction, then a validator — invoice number matches the vendor's known format, date parses and is within 400 days, total is a decimal that equals the sum of line items. Three layers, all deterministic.

```python
import re, datetime
INV = re.compile(r"Invoice\s*(?:No\.?|#)\s*[:\-]?\s*([A-Z]{2,4}-\d{6,8})", re.I)
TOT = re.compile(r"Total\s+Due\s*[:\-]?\s*\$?([\d,]+\.\d{2})", re.I)

def extract(text):
    inv, tot = INV.search(text), TOT.search(text)
    if not (inv and tot):
        raise ExtractionMiss(text[:400])       # loud, routed to the LLM fallback
    return {"invoice_no": inv.group(1),
            "total": Decimal(tot.group(1).replace(",", ""))}
```

Where the model earns its keep is the `ExtractionMiss` branch. Onboard a new vendor, or a template changes: the miss rate for that vendor spikes, the failures route to a vision-capable model with a strict JSON schema, a human confirms the first ten, and the confirmed extractions become the new anchors. That is the pattern I would defend in any design review — **deterministic head, model tail, and the miss rate is the metric that tells you when to re-cut the boundary.**

**💰 Math:** 200k invoices/month. Regex path: ~2 ms CPU each, effectively free — call it one small worker, $30/month. Model path at a frontier vision model, ~1,800 input tokens and 120 output tokens per invoice: at $3/Mtok in and $15/Mtok out that is 1,800 × 3e-6 + 120 × 15e-6 = $0.0054 + $0.0018 = **$0.0072/invoice**, so 200k × $0.0072 = **$1,440/month** if you route everything. With a 96% regex hit rate you route 8,000 invoices and pay **$58/month**, a 25× reduction — and more importantly your p50 goes from ~900 ms to ~3 ms. **📅 Volatile:** verify per-token prices before your loop.

**⚠ Trap:** "the model handles layout variation for free" is the seductive and wrong claim. It handles variation *silently*, which is the problem. With regex you can measure your miss rate exactly and it is a first-class metric on a dashboard. With a model your error rate is unmeasurable without a labeled sample, and nobody builds that labeled sample until after the incident.

### We currently route support tickets into one of twelve queues by calling a frontier model on every ticket. Talk me out of it — or into keeping it.

Twelve fixed labels is the textbook definition of a supervised classification problem, and you almost certainly have the training data already: every ticket ever routed, with its final queue after human correction, is a labeled example. If you have been running the LLM router for three months at even 5k tickets/day you are sitting on ~450k labeled rows. That is an embarrassment of data for a 12-class problem.

The mechanism, in ascending cost order. First, embed the ticket text with a small embedding model and fit multinomial logistic regression on top of the embeddings. That is genuinely ~15 lines and typically lands within a point or two of the LLM on a 12-class routing task, because routing is mostly topical and embeddings encode topic extremely well. Second, if the classes have strong lexical signal (product names, error codes), a TF-IDF + linear SVM baseline is often *better* than embeddings and costs nothing. Third, if you have per-queue exemplars rather than labels, embed the exemplars, take cosine similarity, and threshold — the nearest-centroid approach — which needs no training at all and gives you an abstention band for free.

```python
# embed once at ingest; you are probably already embedding for search
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
clf.fit(X_train_emb, y_train)                 # X: [N, 1024] float32
probs = clf.predict_proba(x_emb)              # [12]
top, conf = probs.argmax(), probs.max()
route = QUEUES[top] if conf >= 0.62 else "escalate_to_llm"
```

That last line is the whole design. You do not replace the LLM; you demote it to the abstention branch. Calibrate the threshold on held-out data to hit whatever routing accuracy the support org demands, and the fraction of traffic above threshold is your cost saving.

**💰 Math:** 5,000 tickets/day = 150k/month. LLM path at ~900 input tokens, 15 output tokens: 900 × 3e-6 + 15 × 15e-6 = $0.0027 + $0.000225 ≈ **$0.0029/ticket** → **$437/month**, p50 ~700 ms. Classifier path: embedding at ~$0.02/Mtok is 900 × 2e-8 = $0.000018/ticket → **$2.70/month**, and the logistic regression forward pass is a 1024×12 matmul, roughly 25 μs. If 85% clears the threshold, you pay $2.70 + 0.15 × $437 = **$68/month**. The dollars are small at this volume; the interesting number is latency — routing becomes synchronous and sub-millisecond instead of a 700 ms async job, which changes the product.

**🗣 Say this in the room:** "Twelve fixed labels with three months of human-corrected routing history is a supervised problem, not a generation problem. I'd fit a linear head on embeddings, calibrate an abstention threshold, and keep the LLM strictly for the low-confidence tail. The win isn't the $370 a month — it's that routing becomes deterministic, testable, and 25 μs."

**⚠ Trap:** do not let anyone tell you the classifier "can't handle a new queue." Adding a 13th queue means refitting a logistic regression on 450k rows, which takes about 40 seconds. The real constraint is *labels for the new class*, and the honest answer is: run the embedding-threshold or LLM path for the new class for two weeks, harvest labels, then fold it in. That is a boring, solved MLOps loop, and saying so is a seniority signal.

### Our "ask your data" feature works by dumping rows into the context window and asking the model for the answer. What would you ship instead?

The mental model that ends this discussion: a language model asked to compute `SUM(revenue) WHERE region='EMEA' AND quarter='Q3'` is performing arithmetic in a representation that was never designed for arithmetic, over a sample of rows that fit in the context, with no guarantee it saw all of them. Postgres does this exactly, over all rows, in milliseconds, and has done for thirty years. The model's job is **translation**, not computation.

The correct architecture has three deterministic components and one probabilistic one:

- **Schema-grounded generation.** The model sees the table DDL, column descriptions, a handful of gold query examples, and the question. It emits SQL — never an answer.
- **A validation gate.** Parse the generated SQL with `sqlglot`, assert it is a single `SELECT`, assert every referenced table and column exists in the schema catalog, assert the tenant predicate is present, reject anything with DDL/DML tokens. This is a whitelist, not a blacklist — blacklisting SQL injection patterns in generated SQL is how you get owned.
- **Execution under a leash.** Read-only role, `statement_timeout`, `LIMIT`, per-tenant row-level security. The database, not the prompt, enforces access.
- **Rendering.** The result set goes back through the model only to phrase it in English, with the numbers passed through verbatim, or better: rendered by a template so the model cannot perturb a digit.

Then there is the part everybody skips. For a real analytics product, the *head of the question distribution is tiny*. "What was revenue last quarter", "top 10 customers by ARR", "churn this month" — maybe 200 canonical questions cover 60% of traffic. Those become **precomputed metrics with hand-written, reviewed SQL**, matched by embedding similarity with a high threshold. Text-to-SQL only runs on the tail.

**🔍 Failure taxonomy — text-to-SQL in production, in the order these bite:**
1. *Semantic-not-syntactic errors.* The SQL runs, returns a number, and the number is wrong because the model joined on the wrong key or ignored a soft-delete flag. Undetectable without gold-query evals. This is 80% of your real error mass.
2. *Ambiguous business terms.* "Revenue" means bookings to sales and recognized revenue to finance. No model resolves this; a semantic layer does.
3. *Silent partial filters.* The tenant predicate gets dropped and you leak cross-tenant data. Mitigated only by RLS at the database, never by prompt instructions.
4. *Cost bombs.* An unbounded join across two fact tables. Mitigated by `statement_timeout` and a cost check via `EXPLAIN` before execution.
5. *Stale schema in the prompt.* A column is renamed; generation silently degrades. The schema block must be generated from the live catalog at request time, not pasted into a prompt file.

**🗣 Say this in the room:** "The model writes the query; the database computes the answer. I gate the generated SQL through a parser whitelist and run it as a read-only role with RLS and a statement timeout, and I evaluate on gold question-SQL pairs where I compare result sets, not query strings — because two different queries can both be right and string match will tell you nothing."

### Legal wants an LLM to decide whether a transaction is allowed under our sanctions policy. Push back.

This is the highest-value "no" in the whole section, and the way you say it matters as much as the content.

The mental model: a compliance gate is a *decision with an audit trail*. What a regulator asks for is not accuracy, it is **reproducibility and explanation** — the same input must produce the same decision today and in the deposition three years from now, and you must be able to name the rule that produced it. A model with temperature 0 is still not reproducible across a provider's silent version bump, and "the model said so" is not an explanation. That is a legal-defensibility argument, not an ML argument, and it is the one that wins the room.

So the architecture inverts. Deterministic rules make the decision; the model does the unstructured work that feeds them and the unstructured work that follows them:

- **Before the gate:** the model normalizes messy input — extracts the counterparty name, address and beneficial-owner text from a free-form document into a strict schema. Fuzzy name matching against a sanctions list is itself a classical problem (Jaro-Winkler, phonetic keys, an ML scorer trained on adjudicated matches), not an LLM problem.
- **The gate itself:** a rules engine over the normalized record. Versioned, unit-tested, with every rule traceable to a clause in the policy document. `BLOCK if counterparty in SDN_LIST or jurisdiction in EMBARGOED or (amount > threshold and kyc_tier < 2)`.
- **After the gate:** the model *drafts* the analyst's narrative — "flagged because the counterparty matched an SDN entry at 0.91 similarity" — for a human to review and sign. Drafting is a real, valuable, well-scoped LLM job.

**⚠ Trap:** "we'll use the LLM but log its reasoning for audit." Chain-of-thought text is a post-hoc narrative, not a causal trace of the computation, and treating it as an audit record is the most dangerous mistake in this entire domain. It is well documented that models produce explanations that do not correspond to the features actually driving the output. If your audit artifact is model-generated prose, you have an audit artifact that can be confidently, fluently wrong, and it will be read in court as if it were a log line.

**🗣 Say this in the room:** "I'd put the model on both sides of the gate and never inside it. It normalizes messy input into a schema and it drafts the analyst's rationale, but the block/allow decision comes from a versioned rules engine, because the regulator's requirement is reproducibility and a traceable rule — and generated reasoning text is a narrative, not a trace."

The general form of this pattern is worth naming, because it generalizes far past compliance: **the LLM handles the unstructured boundary; a deterministic core handles the decision.** Refunds above a threshold, medical dosing, access control, pricing, credit limits — same shape every time.

### Half our chat traffic is the same forty questions. How do you exploit that, and where does it go wrong?

The mental model: query distributions in every product I have measured are brutally Zipfian, and a Zipfian head is a caching problem — which you already know how to solve. The only new thing is that the cache key is fuzzy, and that fuzziness is where the danger lives.

Three tiers, and I would build them in this order.

**Tier 1 — curated answers for the true head.** Pull the top 200 queries by volume from a month of logs, cluster near-duplicates, and have a human write or approve the answer for each. These are not cached model outputs; they are content. Served from Postgres or Redis in single-digit milliseconds, versioned, reviewable, and correct by construction. For a docs assistant this routinely covers 30–50% of traffic.

**Tier 2 — exact-match cache.** Normalize (lowercase, strip punctuation, collapse whitespace) and hash. Zero risk of a wrong hit. Catches maybe another 5–10%.

**Tier 3 — semantic cache.** Embed the query, ANN-search the cache, and return the stored answer if cosine similarity exceeds a threshold. This is the one that gets people fired.

**⚠ Trap:** the semantic cache returns a *semantically close, factually wrong* answer. "How do I cancel my Pro plan?" and "How do I cancel my Team plan?" sit at cosine ~0.94 with most embedding models, and they have different answers. A threshold tuned on a similarity histogram rather than on labeled pairs will absolutely serve one for the other. Three rules I enforce: (a) the threshold is chosen on a hand-labeled set of near-miss pairs, not on the score distribution; (b) anything the query distinguishes by a *named entity* — plan name, product, version, region — is excluded from semantic caching entirely, or the entity is part of the cache key; (c) personalized or account-scoped answers are never semantically cached, only exact-cached within a user scope.

**💰 Math:** 1M chats/month, RAG-flavored, ~6k input tokens and 400 output tokens each. Uncached: 6,000 × 3e-6 + 400 × 15e-6 = $0.018 + $0.006 = **$0.024/chat** → **$24,000/month**. Now layer it: 35% curated (free), 8% exact-hit (free), 20% semantic-hit (embedding only, ~$0.0001) — that leaves 37% hitting the model, so 370,000 × $0.024 = **$8,880**, plus ~$100 of embedding, ≈ **$9,000/month**. A **62% reduction**, and the 43% served from tiers 1–2 comes back in ~8 ms instead of ~4 s.

And note what else you get: the curated tier is your regression suite. Those 200 questions with approved answers are exactly the eval set you were going to have to build anyway.

### Give me the full comparison: a 200-line scikit-learn model versus a frontier API, at one million requests a month. Cost, latency, accuracy — all of it.

Let me fix a concrete task so the numbers mean something: binary content classification — does this user-generated post violate policy — at 1M posts/month (~23 QPS average, ~70 QPS peak), with 1.2% positives and 50k human-labeled examples in hand.

**Cost.** The sklearn path is a gradient-boosted tree over TF-IDF plus a few hand features, or logistic regression over embeddings. Say embeddings: 1M × 300 tokens × $0.02/Mtok = 300M tokens × 2e-8 = **$6/month** for embedding, plus inference on two `c6i.large` instances for redundancy at ~$62/month each = **$130/month all-in**. The frontier path at ~450 input tokens (post + a compact policy prompt) and 8 output tokens: 450 × 3e-6 + 8 × 15e-6 = $0.00135 + $0.00012 = **$0.00147/post** → **$1,470/month**. With prompt caching on a 2,000-token policy preamble at a 90% cached-input discount, the cached portion costs 2,000 × 3e-7 = $0.0006 instead of $0.006 — but note the preamble was not in my 450 tokens, so realistically the honest comparison is **$1,470–$2,000/month vs $130/month, roughly 11–15×**. Using a small hosted model at ~$0.15/Mtok input instead: 450 × 1.5e-7 = $0.0000675 → **$68/month**, which is genuinely competitive. **📅 Volatile:** all four prices need re-verification.

**Latency.** GBDT over TF-IDF: feature hashing ~200 μs, tree ensemble forward ~50 μs, so **p99 well under 5 ms** end-to-end including network. Logistic-regression-over-embeddings adds an embedding call, ~30–60 ms p50, ~150 ms p99 — the embedding hop dominates entirely. Frontier API with 8 output tokens: TTFT ~250–500 ms, plus 8 tokens at ~25 ms/token ≈ 200 ms, so **p50 ~600 ms, p99 1.5–3 s**, and the p99 is not yours to control. That is a 100–600× latency ratio, and it decides whether moderation can be synchronous (block before publish) or must be asynchronous (publish then retract) — a *product* difference, not an infra one.

**Accuracy.** Here is where I refuse to be dogmatic. With 50k labels on a well-specified policy, a tuned GBDT will typically beat a zero-shot frontier model on the *head* of the policy — the common, lexically-signposted violations — often by a wide margin on PR-AUC, because it has learned your actual label distribution including your annotators' idiosyncrasies. The frontier model wins decisively on: novel phrasings it has never seen, multi-hop reasoning ("is this a threat given the referenced event?"), code-switched and low-resource languages, and *any policy that changed last Tuesday*, where the classifier has zero labels and the model needs a prompt edit. It also gives you a rationale string for the appeals queue, which the tree cannot.

**What I would actually ship**, and this is the answer: a cascade. GBDT scores everything at 5 ms. Confident-clean (say `p < 0.02`, ~88% of traffic) auto-passes. Confident-violation (`p > 0.85`, ~1.0%) auto-actions. The uncertain band — ~11% — goes to the frontier model, whose output both makes the decision and becomes a training label. Cost: $130 + 0.11 × $1,470 = **$292/month**. Latency: p50 5 ms, p89 5 ms, and only the ambiguous 11% pay the second. Accuracy: better than either alone, because the model is spending its capability on exactly the examples where capability matters.

**📐 Numbers you must know:** the cascade savings factor is just `1 / escalation_rate` on the LLM line. At 11% escalation you pay 9× less than routing everything; at 30% you pay 3.3× less. This means **the entire economics of a cascade live in the abstention threshold**, and tuning that threshold is a one-afternoon job with a labeled set. It is the highest-ROI hour in applied AI engineering and almost nobody does it deliberately.

**🗣 Say this in the room:** "I'd build both and cascade them. The classical model handles the 89% where it's confident at 5 ms and $130 a month; the frontier model handles the ambiguous 11% and its outputs become labels that shrink that band over time. That gets me sub-10ms p50, an order of magnitude less spend than routing everything, and a system that gets cheaper as it runs."

### How do you say "I would not use an LLM here" in an interview without sounding like you can't build with LLMs?

This is a real interview skill and it is graded, so let me be explicit about the mechanics.

The failure mode is answering with a refusal: "I wouldn't use an LLM for that." It reads as either dogma or inability, and the interviewer cannot distinguish the two. The fix is structural: **never lead with the negative, and never end without giving the LLM a job.**

The three-move pattern I use:

**Move 1 — name what the LLM is uniquely good at, so they know you know.** "The thing a frontier model gives me that nothing else does is handling unbounded input format and open output spaces without labeled data."

**Move 2 — show that this problem doesn't have that shape, with a specific property.** Not "it's simple" — a *property*: the output space is 12 fixed labels; the input is a fixed template; the answer is already in Postgres; there's a verifier. This is the move that signals seniority, because it shows you evaluated rather than pattern-matched.

**Move 3 — put the LLM somewhere real anyway.** Bootstrapping labels, the abstention branch, generating the eval set, drafting the human-reviewed rationale, handling the tail. There is almost always a genuine job for it, and offering one proves the "no" was analysis and not allergy.

Worked example, delivered end to end:

**🗣 Say this in the room:** "A model is what I reach for when the input format is unbounded or the output space is open and I don't have labels. Here neither is true — twelve fixed queues and three months of human-corrected routing history — so I'd fit a linear classifier on embeddings and calibrate an abstention threshold. I'd still use the model in two places: it labels the bootstrap set for a new queue before I have data, and it takes the low-confidence tail. That's a 25-microsecond p50 on 85% of traffic and the model spending its budget where it actually adds information."

Two calibration notes. First, **read the room's incentive**. If you are interviewing at a company whose product *is* an LLM product, the gate question is usually testing whether you know the boundary, not whether you will refuse — so lead with the LLM's genuine strengths and be crisp about the carve-out. Second, **if they push back, hold the line with a number, not with conviction.** "At 70 QPS peak with a 50 ms budget, no API round trip fits — that's the constraint, not a preference" ends the argument. Restating your opinion louder does not.

### Fine — flip it. What are the positive criteria that make an LLM genuinely the right call?

I want to be equally rigorous in the other direction, because an engineer who only knows how to say no is just as useless as one who says yes to everything.

An LLM is the right tool when **the input format is unbounded, the output space is open, and there is no cheap labeled dataset** — and it becomes the *only* tool when two or more of those hold simultaneously.

Concretely, the cases where I reach for a model without hesitation:

- **Open-ended generation with no single correct answer.** Drafting, summarizing, rewriting, explaining. There is no `SUM()` for "write a release note from this diff."
- **Zero-shot on a task that will be defined next week.** The killer property of a frontier model is that the spec lives in a prompt, editable in a deploy, with no retraining. When policy changes weekly, that is worth an enormous amount of latency and money.
- **Semantic understanding over long, messy, heterogeneous text.** Reading a 40-page contract and answering "does this have an auto-renewal clause" is not a regex problem and never was.
- **Long-tail coverage where per-class labels will never exist.** 4,000 intents with a power-law distribution: you will have labels for 30. The tail is the model's.
- **Compositional instruction-following.** "Extract these seven fields, but if the doc is an amendment, follow the parent contract's numbering." Encoding that as rules is possible and horrible; the maintenance cost dominates.
- **Anything requiring natural-language output for a human to read.** Even in a fully deterministic system, the last mile is often prose.
- **Code generation and transformation**, where the crucial property is that a compiler and a test suite verify the output — the LLM proposes, the toolchain disposes.

**⚠ Trap:** "it needs reasoning, so it needs an LLM." Most tasks people describe as reasoning are lookup plus arithmetic, both of which have exact solutions. The honest test for whether a task needs a model is: *write down the decision procedure*. If you can write it down completely, implement it — you just did the hard part. If you cannot write it down because the cases are unbounded, that is the genuine signal for an LLM, and it is a much narrower signal than it feels like.

### Someone says "only about 20% of a production agent system is the LLM." What's the other 80%, and why should I care that you know that?

The claim is a direct descendant of a well-known systems observation: **📄 Paper:** Sculley et al. (2015), "Hidden Technical Debt in Machine Learning Systems" — the famous diagram showing the ML code as a small box surrounded by configuration, data collection, serving infrastructure, monitoring and process management. The LLM era rediscovered it. The model call is a function invocation; the system around it is the product.

The other 80%, enumerated because vagueness here is a tell:

- **Retrieval and context assembly** — chunking, indexing, hybrid search, reranking, dedup, the token budget allocator that decides what gets dropped when the context overflows. This alone is often 30% of the code and 60% of the quality.
- **Tool layer** — schemas, argument validation, timeouts, retries with backoff, idempotency for side-effecting tools, permission scoping per user, and result truncation so a 40k-token API response doesn't blow the window.
- **Control flow and state** — the loop, the step budget, termination conditions, checkpointing so a 40-step trajectory can resume, and human-in-the-loop interrupts. This is a durable-workflow problem and you already know how to build it.
- **Output contracts** — schema-constrained decoding or validate-and-repair, and what you do on the third failure.
- **Evaluation** — the offline suite, the regression gate in CI, the LLM-judge with its own calibration set, online metrics, and the annotation pipeline that feeds all of it.
- **Observability** — per-step traces, token and cost attribution per tenant and per feature, prompt version stamped on every span.
- **Safety and policy** — input/output filters, prompt-injection defenses, PII redaction, rate limiting keyed on tokens rather than requests.
- **Cost and capacity control** — per-tenant token budgets, model routing, caching tiers, fallback when the provider degrades.

**🗣 Say this in the room:** "The model is a stateless function call with a bad p99 and no schema guarantee. Everything that makes it a product — retrieval, tools, the control loop, output contracts, evals, cost attribution — is ordinary distributed-systems engineering, which is exactly why my backend background transfers. I've watched teams spend a quarter tuning prompts when the actual defect was that their chunker split tables across boundaries."

Why interviewers care: this framing is the single fastest way to distinguish someone who has shipped from someone who has done tutorials. Tutorial-shaped answers are all prompt. Shipped-shaped answers spend most of their words on the surrounding machinery — and that machinery is where a senior backend engineer is already strong, which is the strategic reason to lead with it.

### I've got 400 labeled examples and a classification task. Few-shot a frontier model, or train something?

Both, in a specific order, and the order is the answer.

The mental model: 400 examples is squarely in the zone where the right move is not "pick one" but "use the model to escape the data-poverty trap." Labels are the scarce resource; compute is not.

The ladder I run:

**Step 1 — spend 100 of them on an eval set, immediately, before anything else.** Stratified by class, held out, never trained on, never looked at during prompt iteration except through an aggregate number. If you have 400 labels and you spend zero on evaluation, every subsequent decision is guesswork. This is the step people skip and it is the one that is graded.

**Step 2 — few-shot the model as the baseline.** 8–16 examples in the prompt, measure on the eval set. This is your reference point and it takes an hour. It also tells you whether the task is even well-defined: if a frontier model with 16 examples gets 61% on a 5-class problem, your *label definitions* are ambiguous, and no amount of modeling fixes that. Go re-adjudicate the taxonomy.

**Step 3 — fit a classical model on the 300 training examples over embeddings.** With 300 examples and a 1024-dim embedding, logistic regression with strong L2 is the right estimator — it will not be the best possible model but it will not overfit catastrophically either. Compare on the same eval set. My honest prior: below ~500 examples per class the LLM usually wins; above ~2,000 per class the linear head usually wins on the head classes and loses on the tail. Between those, it is genuinely task-dependent and you must measure. Anyone who tells you the crossover without knowing the task is guessing.

**Step 4 — if the LLM wins, use it to make more labels.** Run it over 20k unlabeled examples, keep only high-confidence predictions, hand-audit a 200-row sample of those to estimate pseudo-label precision, then train the classical model on the union. This is distillation-by-labeling and it is the highest-leverage move in the whole ladder. Then re-run step 3.

**⚠ Trap:** fitting on 400 examples and reporting cross-validated accuracy after you tuned the regularization strength on the same folds. You have now selected a hyperparameter on your evaluation data and your reported number is optimistic by several points — at n=400 the standard error on an accuracy estimate is already about `sqrt(0.85 × 0.15 / 100) ≈ 3.6%` on a 100-example test set, so a 2-point "improvement" is noise. Report a confidence interval or do not report a comparison.

**🏋 Drill:** take any public 5-class text dataset, subsample to 400 rows, and in 45 minutes produce a table with four rows — zero-shot, 8-shot, logistic-regression-on-embeddings, and LLM-pseudo-labeled-then-fit — each with accuracy, macro-F1 and a bootstrap 95% CI on a fixed 100-row eval split. Pass criterion: you can state which differences are real and which are inside the interval, without hedging.
### Teach me logistic regression from the ground up, and then tell me why it's still the first thing you fit.

Start from what you want: a number between 0 and 1 that you can treat as a probability. You have a feature vector `x ∈ R^d`. The simplest thing that uses all of it is a weighted sum `z = w·x + b`, which lives in `(-∞, ∞)`. So squash it: `p = σ(z) = 1/(1 + e^{-z})`. That is the entire model. Logistic regression is a single linear layer followed by a sigmoid — literally a one-neuron neural network — and every intuition you build here transfers directly to the final layer of every classifier you will ever see, including the LM head of a transformer (which is the same thing with a softmax over 128k classes instead of a sigmoid over 2).

The training objective is negative log-likelihood, and it is worth seeing why it is forced rather than chosen. Under the model, the likelihood of a labeled example is `p^y (1-p)^{1-y}`. Take the log, negate, sum: `L = -Σ [y log p + (1-y) log(1-p)]`. That is binary cross-entropy. Its gradient with respect to the weights is beautifully simple — `∂L/∂w = Σ (p_i - y_i) x_i` — prediction minus truth, scaled by the input. Exactly the same form as the softmax cross-entropy gradient in a transformer, and for the same algebraic reason: the sigmoid's derivative cancels against the `1/p` from the log.

```python
import numpy as np
def fit_logreg(X, y, lr=0.1, l2=1e-3, steps=2000):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y                                  # [N]
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b
```

Twelve lines, no framework. That is the version I expect a senior candidate to write on a whiteboard without hesitation.

Why it is the first model I fit, every time: the coefficients are directly interpretable (`w_j` is the change in log-odds per unit of feature `j`), the loss is convex so there is exactly one optimum and no seed-dependence, it trains on a million rows in seconds, it is naturally well-calibrated when the link function matches the data-generating process, and — the part that matters in interviews — **it establishes the number that every fancier model must beat.** If your transformer fine-tune beats logistic regression on TF-IDF by 0.4 points of F1, you have not built an AI system, you have built an expensive one.

**⚠ Trap:** unregularized logistic regression on perfectly separable data diverges — the weights grow without bound because pushing `z` to infinity keeps decreasing the loss. You see it as coefficients in the hundreds and a convergence warning. It is also the classic *symptom of leakage*: perfect separation usually means a feature encodes the label. When I see huge coefficients, my first hypothesis is not "increase max_iter," it is "which column did I accidentally include?"

### Explain gradient boosting to me — what is being boosted, and why does XGBoost still beat deep learning on tabular data?

The mental model: gradient boosting is gradient descent performed *in function space instead of parameter space*. Ordinary training nudges weights in the direction that reduces loss. Boosting instead asks "what function, added to my current prediction, would reduce the loss fastest?" — computes the negative gradient of the loss with respect to the *current predictions*, and fits a small decision tree to those residuals. Then adds that tree, scaled by a learning rate, to the ensemble. Repeat a few hundred times. The "gradient" in gradient boosting is a gradient with respect to the model's output, not its parameters, and once that clicks the whole family makes sense.

Mechanically, at round `m` you have `F_m(x)`. You compute per-example pseudo-residuals `r_i = -∂L(y_i, F_m(x_i))/∂F_m(x_i)` — for squared error that is literally `y_i - F_m(x_i)`, for logistic loss it is `y_i - p_i`. Fit tree `h_m` to predict `r`. Set `F_{m+1} = F_m + η·h_m` with `η ≈ 0.03–0.1`. XGBoost's contribution on top of this was a second-order expansion (using both gradient and Hessian to choose splits and set leaf values), an explicit regularization term on tree complexity, and a sparsity-aware split finder with a default direction for missing values.

**📄 Paper:** Chen & Guestrin (2016), XGBoost — regularized second-order boosting with a scalable split-finding algorithm; it replaced hand-rolled GBM implementations and won essentially every tabular Kaggle competition for years. **📄 Paper:** Ke et al. (2017), LightGBM — histogram-based binning of features plus leaf-wise (best-first) tree growth, giving order-of-magnitude speedups on wide data. LightGBM is what I reach for by default on anything above a million rows; XGBoost when I want the more conservative depth-wise growth on smaller data.

Why trees still win on tabular data — and this is a real, published finding, not folklore: **📄 Paper:** Grinsztajn et al. (2022), "Why do tree-based models still outperform deep learning on typical tabular data?" The reasons they isolate are that tabular features are often non-smooth step functions of the target (trees represent axis-aligned steps natively, MLPs have to approximate them with smooth compositions), that tabular data contains many uninformative features (trees ignore them by never splitting on them, neural nets must learn to zero them), and that tabular features are not rotation-invariant — the columns mean specific things, and MLPs' rotational invariance is a *mis*match to that structure while trees' axis alignment is a match.

**🗣 Say this in the room:** "Boosting is gradient descent in function space — each tree fits the negative gradient of the loss with respect to the current predictions. On tabular data I start with LightGBM because axis-aligned splits match non-smooth feature-target relationships and it's robust to uninformative columns, and I only move to embeddings-plus-a-head when the signal is genuinely in free text."

**⚠ Trap:** the number of boosting rounds is not a hyperparameter you tune on your test set. Use early stopping on a validation split, and remember that early stopping *is* fitting — the chosen round count is a parameter learned from the validation data, so the validation score is optimistic and must not be reported as a test score. I have seen this inflate reported AUC by 1–2 points, which is exactly the size of the improvements people write blog posts about.

### Implement k-nearest-neighbours, and tell me where you're already using it without calling it that.

k-NN is the model with no training step: memorize the training set, and at inference return the majority label (or mean value) of the `k` closest stored points. The mental model that matters for this guide: **vector search is k-NN.** Every RAG system you will ever build is a k-nearest-neighbour retrieval over embeddings, usually approximate, usually with cosine distance, usually with `k` between 5 and 50. If you understand k-NN's failure modes you already understand most of retrieval's.

```python
import numpy as np
def knn_predict(X_train, y_train, x, k=5, n_classes=2):
    d = ((X_train - x) ** 2).sum(axis=1)          # squared L2, [N]
    idx = np.argpartition(d, k)[:k]                # O(N), not O(N log N)
    return np.bincount(y_train[idx], minlength=n_classes).argmax()
```

`argpartition` rather than `argsort` is the detail I look for — you need the top-k, not a full ordering, and that is `O(N)` instead of `O(N log N)`. An HNSW or IVF index is the same idea taken further: give up the exactness guarantee to get sublinear query time.

Three properties that carry straight into retrieval work. **The curse of dimensionality**: in high dimensions the ratio of the nearest to the farthest neighbour distance approaches 1, so "nearest" becomes weakly meaningful — which is exactly why raw high-dimensional distances are unreliable and why learned embedding spaces (trained so that semantic similarity *is* geometric proximity) work where raw feature spaces do not. **Scale sensitivity**: unnormalized features with different units make distance meaningless, which is why embeddings are L2-normalized and cosine similarity is used instead of raw dot product. **No abstraction**: k-NN cannot extrapolate beyond its stored points, which is precisely why a RAG system cannot answer a question whose answer is not in the corpus, no matter how good the model is.

Where I actually use k-NN as a *model* rather than as retrieval: few-shot example selection (retrieve the k most similar labeled examples and put them in the prompt — this is k-NN with an LLM as the aggregation function), near-duplicate detection in a training corpus, and as an instant baseline classifier over embeddings when I have labels but no time. That last one is genuinely strong: 1-NN over a good embedding space is often within a couple of points of a fitted linear head and requires zero training.

**⚠ Trap:** k-NN's cost is at inference, not training, and it grows linearly with your corpus. This is the opposite of every parametric model's profile, and it is why "just add more documents" degrades your retrieval latency in a way that "just add more training data" does not degrade a classifier's.

### You've got 8,000 production failures and no idea what they have in common. Walk me through clustering them.

This is the highest-value classical-ML technique in an LLM engineer's toolkit, and I would push back on anyone who calls it "not real AI work." Error analysis at scale is a clustering problem, and doing it well is the difference between fixing one bug and fixing a class of bugs.

The pipeline: embed each failure (the user input, or input plus the model's wrong output — I usually do both separately, they cluster differently), then cluster in the embedding space, then have an LLM *name* each cluster from a sample of its members. Note the division of labor: clustering is deterministic and cheap, naming is the open-ended part where the LLM belongs.

**Use HDBSCAN, not k-means.** This is not a style preference. k-means requires you to pick `k` in advance, assumes roughly spherical equal-variance clusters, and — fatally — **assigns every point to a cluster**, so your genuinely idiosyncratic one-off failures get forced into a group and pollute it. HDBSCAN infers the number of clusters from density, handles arbitrary shapes, and explicitly labels low-density points as noise (`-1`). For error analysis, "these 340 failures form a tight cluster and these 900 are unrelated singletons" is exactly the answer you want, and only one of these algorithms can give it to you.

**📄 Paper:** Campello, Moulavi and Sander (2013) — hierarchical density-based clustering that extracts a flat clustering by maximizing cluster stability, removing DBSCAN's single global `eps` parameter.

```python
import umap, hdbscan
# reduce first: HDBSCAN's density estimates degrade badly above ~50 dims
red = umap.UMAP(n_components=15, n_neighbors=15, metric="cosine",
                random_state=0).fit_transform(embs)      # [N, 1024] -> [N, 15]
labels = hdbscan.HDBSCAN(min_cluster_size=25,
                         min_samples=5).fit_predict(red)
```

`min_cluster_size` is the knob that matters and it is a *product* decision, not a statistical one: it is the smallest group of failures you would actually staff work against. If a cluster of 10 wouldn't get fixed, set it to 25.

Then quality-check with **silhouette score** — for each point, `(b - a)/max(a, b)` where `a` is mean intra-cluster distance and `b` is mean distance to the nearest other cluster. Ranges −1 to 1; above ~0.5 is a genuinely well-separated structure, around 0.2 is weak, negative means points are closer to another cluster than their own. Compute it excluding the noise points or it is meaningless.

**⚠ Trap — and this one gets people in interviews:** UMAP and t-SNE are *visualization* tools, and their 2-D output lies to you in specific, well-documented ways. Cluster sizes in the plot do not reflect real cluster sizes. Distances *between* clusters are not meaningful — two blobs on opposite sides of a t-SNE plot may be adjacent in the original space. Apparent density is an artifact of the perplexity/`n_neighbors` setting. And both will happily produce beautiful, convincing clusters from pure Gaussian noise. The rules I enforce: never cluster on 2-D UMAP output (cluster on 10–20 dims, or on the raw embeddings, and use 2-D only to draw the picture); never report a distance read off a t-SNE plot; and always sanity-check by reading 10 actual members of a cluster before you believe it. **📄 Paper:** van der Maaten & Hinton (2008) for t-SNE; McInnes, Healy and Melville (2018) for UMAP, which preserves more global structure than t-SNE but does not preserve it faithfully either.

**🏋 Drill:** take 2,000 logged queries, embed, UMAP to 15 dims, HDBSCAN, and produce a table of the top 10 clusters with size, silhouette, and an LLM-written one-line name, in 30 minutes. Pass criterion: for the largest three clusters, you can read five members each and agree the name is right. If you cannot, your `min_cluster_size` is too small.

### Derive PCA, and tell me where it shows up in a modern LLM stack.

The mental model: PCA finds the directions along which your data varies most, and re-expresses every point in terms of those directions. If most of the variance lives in 40 of your 1,024 dimensions, you can throw away 984 numbers per vector and lose almost nothing — which is a compression argument, and compression is what you care about when you are paying for vector storage and ANN search latency.

The derivation, briefly and correctly. Center the data: `X ∈ R^{N×d}` with column means subtracted (uncentered PCA is a common and wrong shortcut — the first component then just points at the mean). Form the covariance `C = XᵀX / (N-1)`, which is `d×d`, symmetric, positive semi-definite. Its eigenvectors are the principal directions and its eigenvalues are the variance along each. Sort eigenvalues descending, take the top `k` eigenvectors as columns of `W ∈ R^{d×k}`, and project: `Z = XW`. The explained-variance ratio of component `i` is `λ_i / Σλ_j`, and the cumulative curve is what you actually read to choose `k`.

In practice you never form the covariance matrix — you take the SVD of the centered `X` directly (`X = UΣVᵀ`, and `V`'s columns are the eigenvectors of `XᵀX`, with `λ_i = σ_i²/(N-1)`). This is numerically better conditioned and it is what `sklearn.decomposition.PCA` does under the hood. The connection to SVD is the same low-rank-factorization machinery that underlies LoRA, so it is worth having the equivalence at your fingertips.

Where it shows up in an LLM stack, concretely:

- **Embedding dimensionality reduction for vector stores.** 1,024-dim fp32 is 4 KB per vector; 10M vectors is 40 GB. PCA to 256 dims is 1 KB, so 10 GB — a 4× cut in RAM and roughly a 4× cut in the distance-computation cost inside the ANN index, typically for 1–3 points of Recall@10. That is often an excellent trade, and it is a real production decision I have made more than once.
- **The contrast with Matryoshka embeddings.** **📄 Paper:** Kusupati et al. (2022), Matryoshka Representation Learning — train the embedding so that its *prefixes* are themselves good embeddings, so truncating from 1,024 to 256 is a slice, no projection matrix, no fitting, no drift when the corpus changes. Where a model offers Matryoshka dimensions, prefer them to PCA. Where it does not, PCA is the fallback and you must remember to persist and version the projection matrix alongside the index.
- **Whitening and anisotropy correction.** Raw contextual embeddings occupy a narrow cone, so cosine similarities cluster in a tight band; removing the top principal components (which often encode frequency rather than semantics) can measurably improve retrieval separation.
- **Diagnostics.** The eigenvalue spectrum of a set of embeddings tells you about representation collapse — if 95% of variance is in 3 dimensions, your fine-tuned embedding model has collapsed and your retrieval is about to get much worse.

**⚠ Trap:** fitting PCA on the full corpus including your evaluation queries, then reporting retrieval metrics. The projection has seen the eval distribution. Fit on the document corpus only, and treat the matrix as a versioned artifact — if you refit it after adding documents, every existing vector in the index is now in a different basis and your recall silently collapses. That is a genuine, subtle re-indexing hazard and I would call it out in review.

### Random forest or gradient boosting — which do you pick, and when does the answer flip?

Both are ensembles of decision trees; the difference is *what the trees are for*, and that difference drives everything else.

A random forest builds many deep, low-bias, high-variance trees **in parallel and independently**, each on a bootstrap sample of rows and a random subset of features at each split, then averages them. Averaging independent high-variance estimators reduces variance; the decorrelation from feature subsampling is what makes the averaging effective. It is a variance-reduction machine. Gradient boosting builds shallow, high-bias, low-variance trees **sequentially**, each correcting the residual errors of the ensemble so far. It is a bias-reduction machine.

The practical consequences fall out of that. A random forest is nearly impossible to overfit by adding trees — more trees only reduces variance, so `n_estimators` is a compute budget, not a regularization knob. Boosting *will* overfit as you add rounds, which is why early stopping is mandatory. A forest trains embarrassingly in parallel; boosting is inherently sequential across rounds (though split-finding within a round parallelizes). A forest gives you out-of-bag error estimates free, no validation split required. Boosting almost always wins on accuracy when tuned, typically by a couple of points of AUC on structured data.

My decision rule: **boosting by default when accuracy matters and I can afford to tune; random forest when I need a robust answer in ten minutes with no tuning, when I want OOB error without carving out a validation set, or when I'm doing feature triage on a wide messy dataset.** Forests are also the better choice when your label noise is high, because sequential boosting cheerfully spends rounds memorizing mislabeled examples.

**⚠ Trap:** default impurity-based feature importances (`feature_importances_` in sklearn) are biased toward high-cardinality features — a random UUID column will rank high because it offers many possible splits. If a feature importance ranking is going to inform a business decision, use permutation importance on held-out data, or SHAP values. I have watched a team drop a genuinely predictive feature and keep a useless ID hash because they read the default attribute.

**📐 Numbers you must know:** the rough resource profile for 1M rows × 100 features. LightGBM: ~30–90 s to train 500 rounds on 8 CPU cores, model on disk a few MB, single-row inference ~20–80 μs. Random forest with 500 deep trees: several minutes to train, and the *model* can be hundreds of MB to gigabytes because deep trees store every node — which sometimes decides it, since a 2 GB model in a latency-sensitive service is a real operational problem.

### Your positive class is 1.2% of the data and your model reports 99% accuracy. Walk me through what you actually do.

The 99% is the constant predictor. Predicting "negative" for everything on a 1.2%-positive dataset gives 98.8% accuracy, so your model has demonstrated that it beat a rock by 0.2 points. The first move is not modeling, it is **deleting accuracy from the report entirely** and replacing it with precision, recall and PR-AUC. Accuracy under class imbalance is not a weak metric; it is an actively misleading one, and reporting it is a signal I read as inexperience.

Then, in order:

**Fix the metric.** Precision-recall curve and average precision, not ROC. Pick the operating point from the business cost of the two error types, not from `argmax F1`. If a false negative costs 50× a false positive, say that out loud and choose the threshold that reflects it.

**Fix the loss, not the data, first.** `class_weight="balanced"` in sklearn, or `scale_pos_weight = n_neg/n_pos` in XGBoost, re-weights the loss so the rare class contributes proportionally. This is cheap, principled, and preserves the true prior. Do this before you touch resampling.

**Resample only if reweighting is insufficient**, and know what each does. Random undersampling of the majority throws away data — fine when you have 10M negatives, wasteful when you have 50k. Random oversampling of the minority duplicates rows and encourages memorization of those exact points. **📄 Paper:** Chawla et al. (2002), SMOTE — synthesizes new minority points by interpolating between a minority example and its neighbours, which works reasonably in low-dimensional numeric feature spaces and works poorly in high-dimensional embedding space, where the interpolants land off-manifold and mean nothing. I do not use SMOTE on text embeddings and I would question anyone who does.

**⚠ Trap — the one that gets shipped:** resampling before the train/test split, or inside cross-validation folds incorrectly. If you SMOTE the whole dataset and then split, synthetic points interpolated from a training example can land in your test set, and you have leaked. Resampling belongs *inside* the CV loop, applied to the training fold only. `imblearn.pipeline.Pipeline` exists specifically because `sklearn.pipeline.Pipeline` gets this wrong.

**Then fix your calibration**, because resampling destroys it. If you undersample negatives 10:1, your model's output probabilities are now on a distorted prior and `p = 0.5` no longer means 50% likely. Either correct the intercept analytically or fit Platt scaling / isotonic regression on a held-out set drawn from the *original* distribution. Calibrated probabilities are what let you set a threshold from a cost matrix, so if you skipped this you cannot do the step you started with.

**💰 Math:** why the threshold matters more than the model. At 1.2% prevalence and 1M items/month, suppose a model at threshold A gives recall 0.80 / precision 0.25 and at threshold B gives recall 0.60 / precision 0.60. Positives = 12,000. Threshold A: 9,600 caught, and 9,600/0.25 = 38,400 flagged total, so 28,800 false positives. Threshold B: 7,200 caught, 12,000 flagged, 4,800 false positives. If human review costs $0.40 per flag, A costs $15,360/month of review and B costs $4,800 — a $10,560/month difference for 2,400 additional catches, so **$4.40 per additional catch.** That number, not the AUC, is the conversation to have with the business.

### What is feature leakage, how do you catch it, and what's its LLM-era equivalent?

Leakage is any situation where your training data contains information that will not be available — or will not be available *in that form* — at prediction time. The model learns the shortcut, your offline metrics look spectacular, and production accuracy craters. The tell is almost always the same: **a number that is too good.**

The classical taxonomy, which I run as a checklist:

- **Target leakage.** A feature is a downstream consequence of the label. Predicting churn using `cancellation_reason_filled_at`. Predicting fraud using `chargeback_amount`. The feature only exists because the outcome happened.
- **Temporal leakage.** You used a random split on time-ordered data, so the model trained on the future to predict the past. Any dataset with a timestamp needs a time-based split, and the validation window must sit strictly after the training window with a gap the size of your real prediction horizon.
- **Group leakage.** The same entity appears in train and test — the same user, the same document, the same near-duplicate support ticket. The model memorizes the entity. Use `GroupKFold` on the entity ID.
- **Preprocessing leakage.** You fit the scaler, the imputer, the TF-IDF vocabulary or the PCA matrix on the full dataset before splitting. Every one of those has now seen the test set. This is what `sklearn.pipeline.Pipeline` inside `cross_val_score` exists to prevent, and it is why fitting transformers outside a pipeline is a code-review reject for me.

How I catch it: **if a single feature gives near-perfect AUC on its own, treat it as guilty until proven innocent.** Run a per-feature univariate AUC scan as a matter of routine — it takes seconds and catches most target leakage. Then ask, for every feature in the model, the physical question: *at the moment I make this prediction in production, does this value exist yet?* Half the time someone cannot answer, which is itself the answer.

**The LLM-era equivalents**, and interviewers absolutely ask this bridge:

- **Benchmark contamination.** The model saw your eval set during pretraining. This is leakage at civilizational scale and it is why a model scoring 92% on a public benchmark may score 61% on your private restatement of the same task. Mitigation: hold out private evals, generated after the model's cutoff, and never publish them.
- **Few-shot leakage.** Your prompt's few-shot examples were drawn from the eval set. Trivially easy to do by accident when the eval set and the example bank come from the same log dump.
- **Retrieval leakage in RAG evaluation.** Your eval questions were *written from* the documents in your corpus, often verbatim, so retrieval is easy in a way real user queries never are. Your Recall@5 of 0.94 is measuring lexical overlap you created.
- **Judge leakage.** The LLM judge grading outputs is the same model that produced them, and it prefers its own outputs. Well-documented self-preference; use a different model family for judging, or a human-calibrated rubric.

**🗣 Say this in the room:** "The first thing I do with any suspiciously good number is a per-feature univariate AUC scan and a check that every feature physically exists at prediction time. The LLM version of the same discipline is a private eval set built after the model's training cutoff — a public benchmark number tells me about contamination as much as capability."

### Define train/serve skew, and give me its analogue in an LLM feature.

Train/serve skew is when the transformation applied to a feature at training time differs — in code, in data source, or in timing — from the transformation applied at serving time. The model was fit on one distribution and is being asked to predict on another, and nothing crashes.

The three canonical forms. **Code skew:** the training pipeline computes `avg_order_value` in a pandas notebook and the serving path recomputes it in a Java service, and the two disagree on how to handle nulls. **Time skew:** training used a feature aggregated over a full 30-day window, serving computes it over whatever data has landed in the warehouse, which lags by 6 hours, so the serving value is systematically stale. **Source skew:** training read from the analytics warehouse, serving reads from the OLTP replica, and the two have different deduplication semantics. The industry's structural answer is a feature store with a single transformation definition materialized to both an offline and an online store — which, in your vocabulary, is just "don't have two implementations of the same function."

The LLM analogues are pervasive and less well recognized:

- **Prompt/template skew.** You evaluated with one prompt template and production renders a slightly different one — an extra newline, a different system-message ordering, a chat template applied by the client library that your eval harness did not apply. This measurably moves quality, and the fix is that the eval harness must call *the same code path* as production, not a reimplementation of it. The rule I enforce: eval calls your `build_messages()` function, never its own copy.
- **Retrieval skew.** Your evaluation retrieves from a static snapshot; production retrieves from a live index with different chunking, a different embedding model version, or documents added since. Your offline Recall@k is measuring a corpus that no longer exists.
- **Tokenizer/version skew.** The eval ran against `model-x-2025-06`, production is pinned to `model-x-latest`, and the provider rolled it. Same skew, no code change on your side. Pin model versions explicitly; `latest` in a production config is the same class of error as `:latest` on a container image.
- **Truncation skew.** Your eval inputs are 2k tokens; real user inputs are 30k and get truncated by a middleware you forgot about. The model is seeing a different input than you think.

**⚠ Trap:** "we log the prompt" is not sufficient. You must log the *rendered final payload* — messages array, tools array, model ID, temperature, and the prompt-template version hash — on a sample of production requests, and periodically replay a sample of those exact payloads through your eval harness. If replayed production payloads score differently from your eval-set payloads, you have skew, and that diff is the only reliable detector. I would call this a required capability for any LLM feature above toy scale.

### How do you detect drift on a deployed classifier, and what changes when the model is an LLM?

The mental model: you almost never get ground-truth labels in production quickly enough to measure accuracy directly, so drift detection is the discipline of **monitoring proxies that move before your accuracy does.**

Three distinct things get confused under the word "drift." **Covariate shift** — `P(x)` changes, `P(y|x)` does not. New users from a different country write different-looking tickets, but the mapping from ticket to queue is unchanged; often survivable. **Label/prior shift** — `P(y)` changes. Spam campaigns come in waves; your calibration is now wrong even though the model is fine. **Concept drift** — `P(y|x)` changes. The policy was rewritten, so the same input now has a different correct label; this one requires relabeling and retraining and nothing else will fix it.

What I monitor, in order of how early it fires:

1. **Input distribution.** Population Stability Index per feature, or a two-sample KS test against a fixed reference window. PSI heuristics in wide use: below 0.1 stable, 0.1–0.25 moderate shift worth investigating, above 0.25 significant. For embeddings, monitor the mean cosine distance to a reference centroid, or the fraction of inputs whose nearest reference neighbour is beyond a distance threshold.
2. **Prediction distribution.** The share of predictions per class, and the *distribution of confidence scores*. A drop in mean max-probability is often the earliest usable signal, because the model is telling you it is less sure before anyone tells you it is wrong.
3. **Delayed ground truth.** Whatever labels arrive — human review decisions, chargebacks, user corrections — feed a rolling accuracy estimate on the subset you have. Be honest that this subset is biased (you only review what you flagged), and correct for it with a small random-sample audit.
4. **A permanent random-sample audit.** 200 random items a week, labeled by humans regardless of what the model said. This is the only unbiased estimate you will ever have and it is worth its cost.

For an LLM feature the machinery is the same and the signals change. You cannot compute PSI on free text, so you embed inputs and monitor the embedding distribution; you monitor output length distribution, refusal rate, schema-validation failure rate, tool-call error rate, and retrieval score distributions. And you get a genuinely new drift source that classical ML does not have: **the model itself changes underneath you.** A provider silently updating a `-latest` alias is concept drift you did not cause and cannot see in your input monitoring. The defense is a canary eval — a fixed set of 200 prompts with known-good outputs, run hourly, alerting on score deltas. That job costs perhaps 200 × $0.005 = $1 per run, $720/month at hourly cadence, and it is the cheapest insurance in the stack.

**🔍 Failure taxonomy — "quality dropped last Tuesday", in the order I check:**
1. Did the model version change? Check the provider's version alias and your pinned config. (Fastest to check, surprisingly often the answer.)
2. Did the prompt change? Diff the prompt-version hash on production spans across the boundary.
3. Did retrieval change? Compare retrieval score distributions and index document counts before/after; look for a reindex job in the deploy log.
4. Did the *input* change? Embedding-distribution monitor, plus a manual read of 20 recent inputs vs 20 from two weeks ago.
5. Did the world change — a product launch, a policy update, a news event that shifted what users ask? This is real concept drift and needs new labels.
6. Only now consider that the model "got worse" spontaneously. It almost never did.

### A take-home says "build an LLM system to categorize support tickets." How do you use the classical-ML baseline to win it?

The rubric on these take-homes weights evaluation methodology heavily, and the fastest way to demonstrate methodology is a baseline table. Most submissions contain a prompt, a loop, and a claim. Yours should contain a comparison.

Here is what I would build in the four hours, and the ordering is deliberate.

**Hour 1 — the eval set, before any modeling.** Stratified split of the provided data, held out. Define the metric and *justify it in the README*: macro-F1 if the categories matter equally regardless of frequency, micro-F1 if throughput is what matters, and report both plus a per-class table so the tail classes are visible. Add a bootstrap confidence interval — 1,000 resamples of the test set, report the 2.5th and 97.5th percentiles. This one addition puts you above most submissions on its own, because it shows you know a 1.5-point difference on 300 examples is noise.

**Hour 2 — three baselines, cheapest first.** Majority-class predictor (this is your floor, and stating it stops anyone from being impressed by 68% accuracy on a dataset that is 64% one class). TF-IDF + logistic regression, maybe 15 lines. Embeddings + logistic regression. All three in one table with the same CI treatment.

**Hour 3 — the LLM.** Zero-shot with a clean taxonomy in the prompt, then few-shot with retrieved nearest-neighbour examples, then structured output so parsing is not a failure mode. Same eval, same table.

**Hour 4 — the system, and the writeup.** A cascade: classifier when confident, LLM otherwise, with the threshold chosen on the validation split to hit a stated accuracy target, and the resulting escalation rate turned into a cost-per-1M-tickets number with the arithmetic in the README. Plus an error analysis: embed the misclassifications, cluster them, and describe the top three failure modes by name. Plus a short "what I'd do with another week" section that names the highest-value next step rather than listing everything.

**🗣 Say this in the room, at the defense:** "The interesting result was that TF-IDF plus logistic regression got macro-F1 0.78 against the frontier model's 0.83, with a bootstrap CI of roughly ±0.03 on each — so the model is genuinely better, but by about one and a half sigma, at 400× the cost per call. That's why I shipped the cascade: the classifier takes the 84% where its max-probability exceeds 0.7, and the estimated blended cost is $X per million tickets against $Y for routing everything."

**⚠ Trap:** do not present the baseline as a strawman you beat. Present it as the thing you had to beat, with the honest margin, including the case where it *wins*. Reviewers of these assignments have told me repeatedly that the single strongest signal is a candidate who reports a result that cuts against their own design and then explains why they shipped it anyway. That is what judgment looks like on paper.
### Derive precision, recall and F1 for me, and then tell me which one you'd optimize for a moderation system.

Everything comes from the 2×2 confusion matrix, and the trick to never confusing the two is to notice what sits in the denominator. **Precision = TP/(TP+FP)** — the denominator is everything you *flagged*, so precision answers "when I raise my hand, how often am I right?" **Recall = TP/(TP+FN)** — the denominator is everything that was *actually positive*, so recall answers "of the things I should have caught, how many did I?" Precision is about the cost you impose on the innocent; recall is about the harm you let through.

They trade off through a single knob, the decision threshold, and that is the whole game. Lower the threshold and you flag more things: recall rises monotonically, precision generally falls. There is no model change involved — the same trained model gives you the entire curve. **F1 is their harmonic mean**, `2PR/(P+R)`, and it is harmonic rather than arithmetic precisely so that a degenerate point cannot game it: a classifier with precision 1.0 and recall 0.01 has arithmetic mean 0.505 but F1 = 2(1.0)(0.01)/1.01 = 0.0198. The harmonic mean is dominated by the smaller term, which is what you want from a summary statistic.

Now the moderation question, and the honest answer is that **F1 is the wrong objective for moderation** and I would say so. F1 asserts that a false positive and a false negative cost the same, and in moderation they emphatically do not — but which way they differ depends entirely on the *action*, and that is the thing to ask about.

The framework I use: decompose by action tier. For an **auto-delete** action, a false positive silences a legitimate user with no recourse — that is a high-cost error, so you want high precision, threshold set to something like precision ≥ 0.95 measured on a held-out set, and you accept whatever recall that buys. For **enqueue-for-human-review**, a false positive costs one reviewer-minute, so you want high recall and you tune to fill exactly the review capacity you have. For **shadow-limit or demote**, you are somewhere in between. So one model, three thresholds, chosen from three different cost structures. If you use `Fβ` at all, use β > 1 (recall-weighted) for the review queue and β < 1 for auto-action, and say why.

**🗣 Say this in the room:** "I wouldn't optimize F1 — it assumes symmetric error costs and moderation isn't symmetric. I'd ask what action fires at each threshold, then set the auto-action threshold by a precision floor and the human-review threshold by the reviewer capacity we can staff. Same model, two operating points, both chosen from cost rather than from argmax F1."

**⚠ Trap:** reporting precision and recall without stating the threshold they were computed at. Those numbers are meaningless without it — they describe a point on a curve, and anyone can pick a flattering point. A comparison of two models at two different, unstated thresholds is not a comparison at all.

### Macro versus micro averaging — when does that choice actually change your conclusion?

The mental model: micro-averaging pools every prediction across all classes into one confusion matrix and computes the metric once. Macro-averaging computes the metric per class and then takes an unweighted mean. Micro therefore weights each *example* equally; macro weights each *class* equally. When your classes are balanced, they are close. When your classes follow a power law — which in every real product they do — they can differ by thirty points and tell opposite stories.

Concretely. Ten support-ticket classes, one of which is 70% of volume. Your model gets 0.95 F1 on the dominant class and 0.30 F1 on the nine tail classes. Micro-F1 (which for single-label multiclass equals accuracy) is dominated by the head: roughly 0.70×0.95 + 0.30×0.30 ≈ 0.76. Macro-F1 is (0.95 + 9×0.30)/10 = **0.365**. Same model, same predictions. Micro says "shipping"; macro says "the tail is broken."

The decision rule I use: **micro when your objective is aggregate throughput and errors on rare classes genuinely matter less** — total tickets correctly routed, total revenue affected. **Macro when the rare classes matter as much as the common ones per-instance** — which is the case for anything safety-related, anything where a rare class is high-severity, and any benchmark where you are trying to demonstrate broad competence rather than exploiting a skewed prior. Weighted-macro (macro weighted by support) is a middle option that mostly reproduces micro and I find it obscures more than it reveals.

**⚠ Trap:** reporting one number without the per-class table. Any single aggregate can hide a class at 0.00 F1 — a class the model has literally never predicted. I require a per-class precision/recall/support table in every classification report I review, and the first thing I look at is the *support* column, because a class with 11 examples in the test set has an F1 whose confidence interval is enormous and should not drive any decision.

**🗣 Say this in the room:** "I report both plus the per-class table. Micro tells me how the system performs on a random ticket; macro tells me whether the tail classes work at all. If they diverge by more than about ten points, that gap is the actual finding and I'd lead with it."

### Someone brings me a moderation classifier with 0.94 ROC-AUC on a dataset with 1% positives. What's your reaction?

My reaction is that the number is nearly uninformative and I would ask for the precision-recall curve before I read anything else. This is one of the sharpest tells in an applied-ML interview, both to give and to receive.

Here is the mechanism. ROC plots true positive rate (= recall) against **false positive rate = FP/(FP+TN)**. That denominator is the negative class, which under 1% prevalence has 99,000 members per 100,000 examples. So a model can produce 990 false positives and still have an FPR of only 0.01 — the curve barely moves. But those 990 false positives are being compared against at most 1,000 true positives, so precision is around 50% at best. **ROC-AUC is insensitive to class imbalance by construction, because both of its axes are normalized within a class**, and that insensitivity is exactly why it flatters imbalanced problems. A random classifier gets 0.5 ROC-AUC regardless of prevalence, which sounds like a virtue and is actually the problem: the metric has been designed to not tell you about the thing you care about.

Precision-recall does not have this property. Precision's denominator (TP+FP) mixes both classes, so the PR curve moves sharply when your false positives start to swamp a small positive class. The baseline for PR-AUC is the prevalence itself — 0.01 here — so a PR-AUC of 0.35 is a 35× lift over random and is genuinely impressive, while a PR-AUC of 0.04 is nearly worthless even though the same model might show 0.94 ROC-AUC.

**📄 Paper:** Davis & Goadrich (2006), "The Relationship Between Precision-Recall and ROC Curves" — establishes that a curve dominating in ROC space dominates in PR space and vice versa, but that PR space is far more discriminative when the negative class is large; this is the canonical citation for "use PR under imbalance."

**💰 Math, so the point lands:** 1M posts/month, 1% positives = 10,000 true violations. Take an operating point with recall 0.90 and FPR 0.03 — that FPR looks tiny and gives a great-looking ROC. False positives = 0.03 × 990,000 = **29,700**. True positives = 9,000. Precision = 9,000/38,700 = **0.233**. So three out of four flags are wrong. At $0.40 per human review, 38,700 flags = **$15,480/month** to catch 9,000 violations, and if this were auto-action instead of review you would be wrongly punishing 29,700 users a month. The 0.94 ROC-AUC told you none of this. The PR curve tells you all of it in one glance.

**🗣 Say this in the room:** "ROC-AUC normalizes false positives by the size of the negative class, so at 1% prevalence it's insensitive to exactly the error that dominates your review cost. I'd want PR-AUC — whose random baseline is the prevalence, 0.01 here — plus precision and recall at the actual deployed threshold. The ROC number isn't wrong, it's just answering a question nobody asked."

### How do you pick the operating threshold, and who actually gets to decide it?

The threshold is a **product decision informed by a cost matrix, not a modeling decision**, and I hold that line firmly because the alternative — an engineer picking `argmax F1` on a validation set — silently encodes an assumption that false positives and false negatives cost the same, which nobody ever agreed to.

The mechanism. Assign a cost to each cell: `C_FP` and `C_FN` (and usually zero for correct predictions, though sometimes a true positive has a handling cost too). Expected cost at threshold `t` is `C_FP·FP(t) + C_FN·FN(t)`, both read off the validation set. Sweep `t` over a few hundred candidates, plot expected cost, take the minimum. That is the whole procedure, and it is ten lines.

```python
import numpy as np
def best_threshold(y_true, scores, c_fp=1.0, c_fn=10.0):
    ts = np.unique(scores)
    costs = [(c_fp * ((scores >= t) & (y_true == 0)).sum() +
              c_fn * ((scores <  t) & (y_true == 1)).sum(), t) for t in ts]
    return min(costs)[1]
```

Three constraints in practice. **Capacity constraints often dominate cost minimization:** if you have 12 reviewers who can process 400 items a day, your threshold is whatever produces 4,800 flags/day, full stop, and the modeling question becomes "maximize recall at fixed volume." Say this explicitly in a design round — it is the constraint that actually binds in every real moderation and fraud system I have seen. **Thresholds need per-segment treatment:** a single global threshold on a model whose score distribution differs by language or by user tenure will produce wildly different precision per segment, which is both an accuracy problem and a fairness problem. **Thresholds drift:** if the score distribution shifts, a fixed threshold changes its operating point silently. I prefer to specify the threshold as a *quantile of the recent score distribution* ("flag the top 0.8% of scores") when volume stability matters more than a fixed precision, and as an absolute value when precision stability matters more. You cannot have both, and picking which one you are defending is the senior move.

**⚠ Trap:** setting a threshold on uncalibrated scores. A gradient-boosted model's raw output is a score, not a probability, and `0.5` means nothing in particular. If a stakeholder is going to reason about "80% confident," you owe them calibration — Platt scaling (fit a one-dimensional logistic regression on held-out scores) or isotonic regression (non-parametric, needs more data, can overfit below a few thousand held-out examples). And re-fit the calibrator whenever you retrain, because calibration does not survive a model change.

### Implement Recall@k and MRR for a retrieval system, and tell me what each one hides.

These are the two metrics that decide whether your RAG system works, and they are ten lines each, so implement them yourself rather than pulling a dependency you cannot audit.

**Recall@k** = the fraction of relevant documents that appear in the top `k`. For the common RAG case where each query has exactly one gold document, this collapses to hit-rate@k — did we retrieve it at all. **MRR** = mean over queries of `1/rank_of_first_relevant`, with 0 for a miss. First result gives 1.0, second gives 0.5, third 0.333, tenth 0.1.

```python
def recall_at_k(retrieved, relevant, k):        # both are sets/lists of doc ids
    top = retrieved[:k]
    return len(set(top) & set(relevant)) / max(len(relevant), 1)

def mrr(all_retrieved, all_relevant, k=10):
    total = 0.0
    for ret, rel in zip(all_retrieved, all_relevant):
        for i, d in enumerate(ret[:k], start=1):
            if d in rel:
                total += 1.0 / i
                break
    return total / len(all_retrieved)
```

What each hides. **Recall@k is rank-blind inside the window**: position 1 and position 10 score identically. That is fine if your reader is an LLM that will attend over all 10 chunks equally — except it will not, because of well-documented positional effects where content in the middle of a long context is used less reliably than content at the edges. So a Recall@10 of 0.90 achieved with the gold chunk usually at position 9 produces a worse system than a Recall@10 of 0.85 with it usually at position 1, and Recall@10 cannot see the difference. **MRR is the mirror image**: it only looks at the *first* relevant document and is completely blind to everything after it. For a question requiring synthesis across three documents, MRR gives full credit for finding one of them.

The rule I enforce: for RAG, **report Recall@k at your actual context budget `k`, plus MRR, plus nDCG@k**, and treat a large gap between Recall@k and MRR as the specific signal that "the reranker is the problem, not the retriever." If Recall@20 is 0.94 and MRR is 0.31, your candidate generation is fine and your ordering is broken — that is a cross-encoder reranker's job, and it is a much cheaper fix than re-embedding your corpus.

**💰 Math:** that diagnostic has real money attached. Re-embedding a 5M-chunk corpus at 400 tokens/chunk is 2B tokens; at $0.02/Mtok that is 2,000 × $0.02 = **$40** of embedding plus a re-index and an alias swap, call it a two-day engineering project. Adding a cross-encoder reranker over the top 50 candidates is a hosted call at roughly $1–2 per 1,000 queries plus ~80–150 ms of latency. **📅 Volatile:** verify reranker pricing. If the metric gap says "ordering," the reranker is the right spend; if Recall@50 itself is low, no reranker will save you and you must fix retrieval.

### Derive nDCG with graded relevance, and tell me where it lies to you.

The mental model: Recall@k and MRR both treat relevance as binary and mostly ignore position. nDCG fixes both — it lets a document be *partially* relevant and it applies a smoothly decaying positional discount — which is why it is the standard metric in search and the right one for any retrieval system whose results a human or a model reads in order.

The derivation, in three steps.

**Gain.** Each document gets a graded relevance label, conventionally 0–3 (irrelevant / marginal / relevant / perfect). The gain is usually `2^rel − 1`, so grades map to 0, 1, 3, 7. The exponential is deliberate: it says a perfect document is worth much more than two marginal ones, which matches how people actually use search results. (A linear-gain variant exists; state which you are using, because the numbers are not comparable.)

**Discount.** Divide by `log2(i + 1)` at rank `i` (1-indexed): rank 1 divides by 1, rank 2 by 1.585, rank 3 by 2, rank 10 by 3.459. Slow decay — position 10 still gets 29% of position 1's credit — which is the right shape for a results page and arguably too generous for a 5-chunk LLM context.

`DCG@k = Σ_{i=1..k} (2^{rel_i} − 1) / log2(i + 1)`

**Normalize.** DCG is unbounded and not comparable across queries — a query with six perfect documents can score far higher than one with a single relevant document, even if both were ranked perfectly. So divide by the **IDCG**: the DCG of the ideal ranking, obtained by sorting that query's judged documents by relevance descending. `nDCG@k = DCG@k / IDCG@k ∈ [0, 1]`, and 1.0 means "you produced the best possible ordering of the documents that were judged."

**📄 Paper:** Järvelin & Kekäläinen (2002) — introduced cumulated gain, discounted cumulated gain and its normalized form as the standard for graded-relevance IR evaluation, replacing binary precision-at-k as the field's default.

```python
import numpy as np
def ndcg_at_k(rels, k=10):                     # rels: graded labels in ranked order
    def dcg(r):
        r = np.asarray(r[:k], dtype=float)
        return float(((2 ** r - 1) / np.log2(np.arange(2, r.size + 2))).sum())
    ideal = dcg(sorted(rels, reverse=True))
    return dcg(rels) / ideal if ideal > 0 else 0.0
```

**⚠ Trap — where nDCG lies:** the normalization is over the *judged* documents only. If your relevance pool is incomplete — you judged the top 10 from your old system and are now evaluating a new system that surfaces a genuinely excellent unjudged document — that document scores as relevance 0, your new system looks worse, and you will reject a real improvement. This is the classic pooling bias in IR evaluation, and in an LLM-era retrieval eval it bites constantly because people build the judgment set from the current system's output. Mitigations: pool candidates from *every* system under comparison before judging, and re-judge whenever you change the retriever substantially. Second lie: nDCG@10 with an IDCG computed over only 2 judged relevant documents saturates at 1.0 trivially — always report the number of judged relevant documents per query alongside the score.

### MAP or nDCG for a RAG retrieval eval — which do you pick and why?

nDCG, in nearly every case, and I will give the decision rule rather than just the answer.

MAP (mean average precision) is built on binary relevance. For each query, average precision is `AP = (1/R) Σ_k P@k · rel_k` — walk down the ranking, and every time you hit a relevant document, record the precision at that position, then average those. It rewards packing all the relevant documents high. Mean it over queries and you have MAP. It is a genuinely good metric with one hard constraint: **relevance must be binary**, and you must know the total number of relevant documents `R` per query.

That constraint is what kills it for RAG. Chunk relevance is not binary. A chunk can contain the full answer, contain half of it, contain context that makes the answer interpretable, mention the topic but not the answer, or be a false lexical match. Forcing that onto {0,1} either inflates your relevant set with chunks that do not actually help the generator, or discards partial-credit information you paid annotators for. Graded relevance is the honest representation, and nDCG is the metric that consumes it.

The decision rule: **binary relevance and you care about finding all of them → MAP. Graded relevance, or you care most about the top few positions → nDCG. Exactly one gold document per query → MRR is sufficient and simpler, and MAP degenerates to MRR anyway in that case.**

But the more important thing to say in a RAG interview is that **all three are proxies for a downstream quantity you can measure directly**: does the generator produce a correct answer given the retrieved context? I run retrieval metrics because they are cheap, deterministic and diagnose *where* the pipeline broke — but the metric that gates a release is end-to-end answer correctness, with retrieval metrics as the explanatory layer underneath. The specific pairing I use is **"context sufficiency"** — a binary judgment of whether the retrieved set contains enough to answer — alongside nDCG. When end-to-end accuracy drops and context sufficiency stayed flat, the generator or the prompt broke. When sufficiency dropped, retrieval broke. That two-signal decomposition is worth more in a design round than the choice between MAP and nDCG.

**⚠ Trap:** optimizing retrieval metrics past the point where they affect the answer. If context sufficiency is already 0.97 at k=8, pushing nDCG@8 from 0.71 to 0.78 will not move end-to-end quality at all, and the time is better spent on the generation half. I have watched a team spend three weeks on reranking for zero product improvement because nobody measured the ceiling first.

### Walk me through BLEU, ROUGE, METEOR and chrF — what do they measure, and why did the field move past them?

These are the generation metrics the field grew up on, and you need them for three reasons: they still appear in papers, they are still correct for narrow tasks, and interviewers use them to check whether you understand *why* n-gram overlap fails.

**BLEU** — **📄 Paper:** Papineni et al. (2002). Modified n-gram precision: for n = 1..4, count how many of the candidate's n-grams appear in the reference, with each reference n-gram consumable only as many times as it appears (this "clipping" is what stops "the the the the" from scoring 1.0). Take the geometric mean of the four precisions, then multiply by a **brevity penalty** `exp(1 − r/c)` when the candidate is shorter than the reference, because precision alone rewards saying almost nothing. It is precision-oriented and corpus-level: BLEU on a single sentence is high-variance and nearly meaningless, which is a frequently-violated property.

**ROUGE** — **📄 Paper:** Lin (2004). The recall-shaped counterpart, built for summarization, where the question is "did you cover the reference content" rather than "was everything you said in the reference." ROUGE-N is n-gram recall; ROUGE-L uses the longest common subsequence, so it rewards in-order overlap without requiring contiguity.

**METEOR** — **📄 Paper:** Banerjee & Lavie (2005). Fixes BLEU's brittlest failure: exact string matching. It aligns unigrams allowing stem matches and synonym matches, computes an F-mean weighted toward recall, and applies a fragmentation penalty when the aligned words are scattered rather than contiguous. It correlated better with human judgment than BLEU at the sentence level, which was the point.

**chrF** — **📄 Paper:** Popović (2015). Character n-gram F-score instead of word n-grams. This matters enormously for morphologically rich languages (Finnish, Turkish, Hindi) where a single word carries what English spreads over four, so word-level overlap is punishingly sparse, and it sidesteps tokenization disagreements entirely.

Why the field moved on: **every one of them measures surface form, and the space of correct answers is enormous.** "The meeting was moved to Thursday" and "They rescheduled it for Thursday" share almost no n-grams and are the same answer. BLEU cannot tell you that. Worse, the failure is asymmetric — a *wrong* paraphrase with high lexical overlap scores well, and a *right* one scores badly, so the metric is actively anti-correlated with quality in exactly the region where you need discrimination.

Where I still use them, unapologetically: **machine translation with multiple references** (BLEU remains the accepted comparability standard, and chrF for non-English targets), **extractive summarization**, **any task where the output is genuinely constrained to near-copy** — SQL string comparison after normalization, structured field extraction — and as a **cheap regression tripwire**: if ROUGE-L against last week's outputs falls off a cliff, something changed, even if a high ROUGE tells you nothing.

**⚠ Trap:** BLEU scores are not comparable across papers unless the tokenization and smoothing match. This is exactly why `sacrebleu` exists — it fixes tokenization and emits a signature string describing the configuration. A BLEU number without that signature is not a number you can compare against anything.

### What did BERTScore fix, and what did it not fix?

**📄 Paper:** Zhang et al. (2020), BERTScore — replaces exact n-gram matching with greedy matching in contextual embedding space, so "moved to Thursday" and "rescheduled for Thursday" can score as similar. That is the fix, and it is a real one.

The mechanism, precisely: run both the candidate and the reference through a contextual encoder, producing one embedding per token. For **recall**, take each *reference* token and find its maximum cosine similarity to any *candidate* token; average. For **precision**, do it the other way — each candidate token to its best reference token. F1 is their harmonic mean. Optionally weight tokens by IDF so that content words count more than function words. Optionally rescale against a baseline computed on random sentence pairs, because raw cosine similarities between contextual embeddings live in a narrow band (roughly 0.8–1.0) and are hard to read — the rescaling spreads them out and is presentation, not substance.

What it did *not* fix, in order of severity:

**It does not detect factual error.** "The revenue was $4.2 million" and "The revenue was $7.8 million" are nearly identical in embedding space — same syntax, same entities, one number differs. BERTScore will rate that pair very highly. For anything where the value of the output is a *fact*, BERTScore is close to useless, and this is the single most important limitation to state.

**It still requires a reference.** For open-ended generation there is no reference, so the metric is inapplicable to most of what an LLM product actually does.

**It inherits the encoder's biases and its cutoff.** Scores depend on which model and which layer you used; comparing BERTScore across papers that used different backbones is meaningless, the same way comparing BLEU across tokenizations is.

**It is insensitive to logical negation and to coherence.** "The patient should not receive the drug" versus "The patient should receive the drug" is a one-token difference in a long sentence, and the score barely moves.

**🗣 Say this in the room:** "BERTScore fixed the paraphrase problem and nothing else. It still needs a reference and it's near-blind to factual substitution — swap a dollar figure and the score moves by a fraction of a point. For anything where correctness is a fact rather than a phrasing, I use an LLM judge with an explicit rubric, or better, an extraction-then-exact-comparison step where the fact is checkable."

The historical arc is worth having straight, because it explains the current state of eval: exact match → n-gram overlap → embedding similarity → model-as-judge. Each step traded determinism and cheapness for semantic sensitivity. LLM-as-judge is where that trade has landed, and it is the first metric in the sequence that needs its *own* evaluation against human labels — which is the whole reason evaluation became a discipline rather than a function call.

### Give me pass@k and its unbiased estimator, and explain why the obvious implementation is wrong.

The mental model: pass@k asks "if I let the model produce `k` independent attempts and a test suite checks each, what is the probability at least one passes?" It is the right metric for code generation because there is a **verifier** — you do not need a reference solution or a human judge, you need the tests to go green — and it reflects the real usage pattern where a developer or an agent gets more than one shot.

The naive implementation is: generate `k` samples, check if any passes, average over problems. That is unbiased but has enormous variance at small `k`, and the number swings wildly between runs, which makes it useless for comparing two models that differ by two points.

The fix, from **📄 Paper:** Chen et al. (2021), the Codex paper (*Evaluating Large Language Models Trained on Code*): generate `n > k` samples per problem, count `c` that pass, and compute the expected pass@k analytically:

```
pass@k = 1 − C(n − c, k) / C(n, k)
```

The reasoning is a one-line combinatorial argument you should be able to reproduce at a whiteboard. `C(n, k)` is the number of ways to choose `k` of the `n` samples. `C(n − c, k)` is the number of ways to choose `k` samples *all of which fail* — you are choosing entirely from the `n − c` failures. Their ratio is the probability that a random subset of size `k` contains no passing sample. One minus that is the probability at least one passes. Average over problems.

```python
import numpy as np
def pass_at_k(n, c, k):
    if n - c < k:                      # fewer than k failures => always at least one pass
        return 1.0
    # numerically stable product form; avoids overflow in binomials at large n
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

The product form is the implementation detail worth knowing: `C(n−c,k)/C(n,k) = Π_{i=n−c+1}^{n} (1 − k/i)`, which avoids computing huge binomial coefficients. Typical practice is `n = 200` with `k ∈ {1, 10, 100}`.

**⚠ Trap:** pass@k with `k > 1` is not a deployment metric unless you have a way to *select* which sample to ship. If a human or an agent reads all `k` candidates and picks, pass@10 is meaningful. If your product returns one completion, the honest metric is pass@1 at your deployment temperature, and quoting a model's headline pass@100 as evidence of production quality is a misuse I would push back on in review. Related: pass@1 depends on temperature, and a model tuned to look good at pass@100 (high temperature, high diversity) can look worse at pass@1. Always state `n`, `k`, and the sampling temperature together.

**💰 Math:** the estimator's whole purpose is variance reduction, and that costs generations. Evaluating a 164-problem benchmark at `n = 200` is 32,800 completions; at ~500 output tokens each that is 16.4M output tokens, so at $15/Mtok = **$246 per full evaluation run**, plus input. If you gate every pull request on that you are spending real money on CI, which is why the standard practice is `n = 200` for a published number and `n = 10` for a fast regression check — accepting the wider interval in exchange for a ~$12 run.

### When is exact match the right metric, and how does it lie?

Exact match is the strictest metric there is: normalize the string, compare, score 1 or 0. Its virtue is that it is completely deterministic, free to compute, and impossible to game — there is no judge to fool and no embedding to exploit. Its vice is that it punishes every acceptable variation.

Exact match is genuinely correct when **the answer space is small and canonical**: multiple-choice letters, yes/no, a classification label from a closed set, a numeric answer after unit normalization, a date after parsing, an enum in a structured-output field. For these, anything softer than exact match is a bug, because it admits partial credit for a wrong answer.

It becomes a liar the moment the answer is a *span of natural language*. Extractive QA is the classic case: gold answer "Paris", prediction "Paris, France" scores 0. Gold "1969", prediction "in 1969" scores 0. This is exactly why the SQuAD-lineage of QA benchmarks reports **both EM and token-level F1** — F1 gives partial credit on overlapping tokens and is the more informative of the two, while EM is the more honest floor. Report both or you are choosing which way to be wrong.

The normalization step is where most of the engineering lives and where most of the bugs are: lowercase, strip articles ("a", "an", "the"), strip punctuation, collapse whitespace, and — the one people forget — normalize numbers ("1,000" vs "1000" vs "1 thousand") and units. Every normalization you add makes the metric more permissive and therefore less comparable to anyone else's number, so pin the normalizer and version it with the eval set.

**⚠ Trap in the LLM era:** exact match against a model that was asked an open question will underreport catastrophically, because models pad. "The answer is Paris." scores 0 against "Paris". The correct fix is not a looser metric — it is **constraining the output format**: structured outputs, a JSON schema with an `answer` field, or a stop sequence, so that EM becomes applicable again. That is the general principle and it is worth stating explicitly: *when your metric is too strict for your output, tighten the output rather than loosening the metric.* Loosening the metric hides errors; tightening the output eliminates a class of them.

**🏋 Drill:** in 25 minutes, unaided, implement `precision/recall/F1` with macro and micro averaging, `recall_at_k`, `mrr`, `ndcg_at_k` with exponential gain, and `pass_at_k` with the stable product form — all in NumPy, no sklearn. Pass criterion: your macro-F1 matches `sklearn.metrics.f1_score(average="macro")` to 1e-9 on a random 5-class array, and your `pass_at_k(n=200, c=3, k=10)` matches a direct `scipy.special.comb` computation. If you cannot write nDCG's discount from memory, that is the one to drill again tomorrow.
### Teach me RNNs from scratch, and show me exactly where the gradient dies.

The mental model: an RNN is a `for` loop with a hidden state that is both the accumulator and the model's entire memory. At each timestep you fold the next token into a fixed-size vector, and that vector is the *only* thing carrying information forward. Everything the model knows about token 1 when it reaches token 500 has to have survived 499 successive squashing multiplications inside a `d`-dimensional vector. Stated that way, the failure is obviously structural rather than a tuning problem.

The mechanism is two lines:

```python
h = torch.zeros(B, d)
for t in range(T):
    h = torch.tanh(x[:, t] @ W_xh + h @ W_hh + b)   # [B, d]
    y[:, t] = h @ W_hy                               # [B, V]
```

Training is backpropagation through time: unroll the loop into a `T`-layer network with **shared weights** and backpropagate. That weight sharing is the whole problem. The gradient of the loss at step `T` with respect to the hidden state at step `t` is a product of Jacobians:

`∂h_T/∂h_t = Π_{k=t+1..T} diag(tanh'(z_k)) · W_hhᵀ`

Look at what that product does. `tanh'(z) = 1 − tanh²(z) ≤ 1`, and equals 1 only exactly at zero — for any activated unit it is strictly less than 1, often 0.1 or smaller once the unit saturates. And `W_hhᵀ` appears `T − t` times. So the magnitude of the gradient scales roughly as `(σ_max(W_hh) · γ)^{T−t}` where `γ < 1` is the typical `tanh` derivative. If that product is below 1, the gradient decays **geometrically** in the distance: at a decay factor of 0.9 per step, 50 steps back you have `0.9^50 ≈ 0.005`, and at 100 steps `0.9^100 ≈ 2.7e-5`. The gradient has not disappeared for numerical reasons; it has been multiplied into irrelevance relative to the short-range gradients that dominate the update. If instead the product exceeds 1, you get the mirror problem — exploding gradients, which show up as a NaN loss and are *easy* to fix with gradient clipping.

**📄 Paper:** Bengio, Simard and Frasconi (1994) established the fundamental difficulty: with gradient descent, learning long-term dependencies is in tension with stability of the recurrence. **📄 Paper:** Pascanu, Mikolov and Bengio (2013) gave the norm-based analysis and popularized gradient-norm clipping as the standard mitigation for the exploding half.

**⚠ Trap:** people say "vanishing gradients mean the model can't learn long dependencies." Sharpen it: the model *can* represent them, and the gradient is not zero — it is *exponentially smaller than the short-range gradient*, so the optimizer spends its entire budget on local structure and the long-range signal is drowned out. This distinction matters because it explains why the fix is architectural (create an additive path with derivative ≈ 1) rather than optimizational (a better optimizer does not rescue a geometrically decaying signal).

**📐 Numbers you must know:** the second, independent problem is that the recurrence is *sequential in T*. Training on a 2,048-token sequence requires 2,048 dependent steps that cannot be parallelized on a GPU, each a small `[B, d] × [d, d]` matmul that leaves the hardware almost entirely idle. A transformer computes all 2,048 positions in a handful of large matmuls. On modern accelerators that is a one-to-two-order-of-magnitude throughput difference on the same FLOP count, and it is the reason transformers won even before you argue about quality.

### What does the LSTM cell state actually do, mechanically, and how is a GRU different?

The mental model: the LSTM's answer to the vanishing gradient is to build a highway. Instead of transforming the memory at every step with a matrix multiply and a squashing nonlinearity, it carries a cell state `c_t` that is updated **additively**, gated by learned multiplicative gates. Because the update is addition rather than matrix multiplication, the gradient path along the cell state has a Jacobian that is a diagonal of forget-gate values rather than a repeated `W_hh` — and if the forget gate sits near 1, the gradient passes through essentially unattenuated. Hochreiter and Schmidhuber called this the constant error carousel, and the phrase is exactly right: the cell state is a conveyor belt, and the gates decide what gets put on and taken off.

The equations, which are worth being able to write:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)        # forget: what to erase from c
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)        # input:  how much new to write
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)     # candidate content
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t            # the additive highway
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        # output: what to expose
h_t = o_t ⊙ tanh(c_t)
```

Note the crucial line: `∂c_t/∂c_{t-1} = f_t`, elementwise. No weight matrix in that path. If `f_t ≈ 1` for a dimension, that dimension's information — and its gradient — travels arbitrarily far. That is the entire trick, and it is the same trick as a residual connection in a transformer: create a path whose local derivative is approximately the identity.

**📄 Paper:** Hochreiter & Schmidhuber (1997) introduced the LSTM with input and output gates; the forget gate was added by Gers, Schmidhuber and Cummins (2000), which is a detail worth knowing because the forget gate is the part everyone now considers essential. A practical consequence: initialize the forget-gate bias positive (commonly 1.0) so the cell defaults to remembering rather than erasing.

**📄 Paper:** Cho et al. (2014) introduced the GRU as a simplification — two gates instead of three, and no separate cell state. An update gate `z_t` interpolates directly between the old hidden state and a candidate (`h_t = (1−z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t`), and a reset gate controls how much of the previous state feeds the candidate. Roughly 25% fewer parameters per unit and correspondingly faster; empirically comparable on most tasks, with LSTMs sometimes edging ahead on the very longest dependencies.

**⚠ Trap:** LSTMs did not *solve* long-range dependencies, they extended the usable range. The gradient still attenuates whenever the forget gate is below 1, and the practical horizon is typically hundreds of tokens, not tens of thousands. Also — and this is the part that killed them commercially — the LSTM does nothing at all about the sequential-in-T training problem. It fixed the gradient, not the parallelism, and the parallelism is what the hardware cared about.

### Walk me through seq2seq with attention. I want to see where attention came from and why it was invented.

This is the most important piece of history in the section, because attention was not invented as a general-purpose architecture — it was invented as a bug fix for a specific, visible failure, and understanding the bug makes the mechanism inevitable rather than magical.

**📄 Paper:** Sutskever, Vinyals and Le (2014), and independently Cho et al. (2014), established sequence-to-sequence: an encoder RNN reads the source sentence and its final hidden state — one fixed-size vector, typically 512 or 1,000 dimensions — becomes the "thought vector" that a decoder RNN generates from. Elegant, and it worked well enough to be exciting.

The bug was immediate and measurable: translation quality **degraded sharply with source length.** Of course it did. You are compressing an arbitrarily long sentence into one fixed vector, so the information-theoretic bottleneck is right there in the architecture. A 40-word sentence has to fit in the same 1,000 numbers as a 6-word one. Sutskever's team even resorted to reversing the source sentence so the beginning of the source ended up closer to the start of decoding — a hack that helped, which is itself diagnostic of the problem.

**📄 Paper:** Bahdanau, Cho and Bengio (2015) removed the bottleneck. Instead of one vector, keep *all* the encoder hidden states `h_1..h_S`. At each decoding step `i`, compute a relevance score between the decoder's previous state `s_{i−1}` and every encoder state, softmax those scores into weights, and form a **context vector** as the weighted sum: `c_i = Σ_j α_ij h_j`. The decoder now reads a fresh, query-dependent summary of the source at every output step. Their scoring function was additive — a small feedforward net, `e_ij = vᵀ tanh(W_s s_{i−1} + W_h h_j)` — which is why it is called additive or "Bahdanau" attention. The paper's title, *Neural Machine Translation by Jointly Learning to Align and Translate*, names the insight exactly: the alignment between source and target words, which statistical MT had modeled explicitly with alignment tables, is now learned end to end as a differentiable soft lookup.

**📄 Paper:** Luong, Pham and Manning (2015) simplified the scoring function to multiplicative forms — `s ᵀh` (dot) and `sᵀ W h` (general) — and introduced the global-versus-local distinction. The dot-product variant is the direct ancestor of scaled dot-product attention: take Vaswani et al. (2017), drop the RNN entirely, apply the same mechanism from the sequence to itself rather than from decoder to encoder, add the `1/√d_k` scale and multiple heads, and you have the transformer. The line from Bahdanau to modern attention is genuinely straight.

**🗣 Say this in the room:** "Attention was invented to fix the fixed-size-bottleneck failure in seq2seq translation — quality fell off with source length because you were compressing an arbitrary sentence into one vector. Bahdanau kept all the encoder states and let the decoder compute a fresh weighted summary per output step, which is a differentiable soft dictionary lookup. Transformers are what you get when you notice the RNN was the part you could delete."

**⚠ Trap:** do not describe attention weights as an explanation of the model's reasoning. Even in 2015 the alignment maps were interpreted more confidently than the evidence supported, and the subsequent literature is genuinely contested on whether attention weights constitute explanation. The safe and defensible framing is that they are a *routing* mechanism whose weights sometimes correlate with human-legible alignment.

### What are teacher forcing and exposure bias, and do they still matter now that everyone trains transformers?

Teacher forcing is how you train an autoregressive model efficiently: at every position you feed the **ground-truth** previous token rather than the model's own prediction. This makes all `T` positions independent given the input, which is what allows the whole sequence to be trained in one parallel forward pass with a causal mask. Without it you would have to sample step by step during training, which is both sequential and high-variance. So teacher forcing is not optional — it is the thing that makes modern pretraining tractable.

Exposure bias is its cost. At training time the model has only ever conditioned on perfect prefixes drawn from the data distribution. At inference it conditions on **its own outputs**, which include its own mistakes, and those prefixes are off the training distribution. One error nudges the context into a region the model saw less of during training, which raises the probability of the next error, which pushes further off-distribution. The classic manifestation in RNN days was degeneration: repetition loops, drift, and a translation that starts fine and dissolves by the end. The term and the framing come from the sequence-level training literature of that era; **📄 Paper:** Bengio et al. (2015) proposed scheduled sampling — probabilistically feeding the model's own prediction during training, annealing from full teacher forcing toward full self-conditioning — as a direct mitigation.

Does it still matter? This is genuinely contested, and I would say so rather than pretend. The strong version of the exposure-bias story is much weaker for large transformers than it was for small RNNs, for three reasons: the models are trained on vastly more data so the "off-distribution" region is far better covered; the architecture has no compounding recurrent state to drift; and modern post-training explicitly conditions on model-generated text — RLHF and any on-policy RL method sample from the model and score those samples, which is precisely training on the model's own distribution, so it attacks exposure bias directly whether or not that is how it is advertised.

What survives, and what I would say in a room: the failure is real but it now shows up as **degenerate repetition loops under greedy decoding**, as **long-context drift** in very long generations, and as **error compounding in multi-step agent trajectories** — where a wrong tool call at step 3 poisons every subsequent step, which is exposure bias at the trajectory level rather than the token level. The practical mitigations are not scheduled sampling; they are sampling with temperature/top-p, repetition penalties, and — for agents — explicit state validation and recovery paths so a bad step can be detected and retried rather than conditioned on.

**⚠ Trap:** "we train on our own model's outputs to fix exposure bias" without a filter is how you get model collapse. On-policy training helps only when the samples are *scored* — by a reward model, a verifier, or a filter. Unfiltered self-training amplifies whatever bias the model already has.

### word2vec and GloVe versus contextual embeddings — what actually changed, and why does it matter for RAG?

The mental model: static embeddings assign one vector per *word type*; contextual embeddings assign one vector per *token occurrence*. That single distinction explains everything that follows.

**📄 Paper:** Mikolov et al. (2013), word2vec — train a shallow network to predict a word from its context (CBOW) or the context from the word (skip-gram), and throw away everything but the input embedding matrix. The companion contribution, negative sampling, replaced the full softmax over the vocabulary with a handful of sampled negatives per update, which is what made it fast enough to train on billions of words. The famous result — `king − man + woman ≈ queen` — was a genuine surprise: linear structure in the embedding space emerged from a purely predictive objective. **📄 Paper:** Pennington, Socher and Manning (2014), GloVe — instead of local windows, factorize the global word-word co-occurrence matrix so that the dot product of two word vectors approximates the log of their co-occurrence count. Different route, similar geometry, and it made explicit that these methods are matrix factorizations of co-occurrence statistics.

The limitation is fatal and simple: **"bank" gets one vector.** River bank, savings bank, and to bank a plane are averaged into a single point, so the representation of every polysemous word is a blend of its senses weighted by corpus frequency. There is also no way to represent a word that was not in the training vocabulary, and no representation at all of word order or syntax.

**📄 Paper:** Peters et al. (2018), ELMo — run a deep bidirectional LSTM language model over the sentence and use its internal states as the word representations, so the vector for "bank" depends on the sentence it appears in. This was the phase change; the "pretrain a language model, use its representations" recipe dates from here. BERT then replaced the LSTM with a transformer encoder and the bidirectional-LM objective with masked language modeling.

Why this matters for RAG, concretely and in a way interviewers probe: **a modern retrieval embedding is not a bag of word vectors, and treating it as one leads to wrong design decisions.** Because the encoder is contextual, chunk boundaries change the vector — a sentence embedded alone and the same sentence embedded inside its paragraph produce different vectors, which is exactly why chunking strategy is a first-class quality lever rather than a plumbing detail. Because the encoder is contextual, a pronoun-heavy chunk ("it costs $40 per seat") embeds poorly with no antecedent in scope, which is the mechanical argument for contextual chunk enrichment — prepending the document title and section heading to each chunk before embedding. And because sentence-level retrieval quality requires a *training objective* that makes cosine similarity meaningful — **📄 Paper:** Reimers & Gurevych (2019), Sentence-BERT, showed that mean-pooling raw BERT outputs gives poor sentence similarity and that a siamese contrastive objective fixes it — you should never build retrieval on a raw masked-LM encoder's pooled output. That is a mistake I still see, and it costs you double-digit points of Recall@10.

Static embeddings are not obsolete, incidentally. They are ~100× cheaper to compute and useful as a lexical-signal feature in a hybrid retrieval scorer, for fast vocabulary-level analysis, and as a cheap baseline you can run over 50M documents on a laptop.

### Give me the family tree — ELMo, BERT, RoBERTa, T5, BART. Which of these still matter in a 2026 stack?

Three architectural branches came out of the transformer, and the family tree is really about which branch each model sits on.

**Encoder-only** — bidirectional attention, every token sees every other token, trained with masked language modeling. **📄 Paper:** Devlin et al. (2019), BERT — mask 15% of tokens and predict them, plus a next-sentence-prediction objective; produced representations that, fine-tuned, immediately beat the state of the art across the GLUE tasks. **📄 Paper:** Liu et al. (2019), RoBERTa — same architecture, better recipe: drop NSP (it was not helping), use dynamic masking, larger batches, more data, longer training. RoBERTa's real contribution was demonstrating that BERT was *undertrained*, which is a lesson the field then applied at every subsequent scale.

**Encoder-decoder** — a bidirectional encoder plus a causal decoder with cross-attention. **📄 Paper:** Raffel et al. (2020), T5 — reframe every NLP task as text-to-text ("translate English to German: ..." → German text), pretrain with span corruption (mask contiguous spans, generate them), on the C4 corpus. **📄 Paper:** Lewis et al. (2020), BART — a denoising autoencoder that corrupts text with several noise functions (token masking, token deletion, text infilling, sentence permutation, document rotation) and trains the decoder to reconstruct the original; strong on summarization in particular.

**Decoder-only** — causal attention, next-token prediction. The GPT line, and the branch that won for generation.

What still matters in a 2026 stack, honestly:

- **Encoder-only models are alive and important**, just not as chatbots. Every retrieval embedding model and every cross-encoder reranker you deploy is an encoder — bidirectional attention is *strictly better* for producing a single representation of a fixed text, because a causal encoder's early tokens cannot see the later ones. When someone asks why your embedding model is bidirectional but your generator is causal, that is the answer. Encoders are also still the right tool for classification and token tagging when latency matters: a small encoder classifier runs in single-digit milliseconds on CPU.
- **Encoder-decoder survives in narrower niches** — translation, some speech and multimodal connectors — but the decoder-only branch absorbed most of what T5 was used for, because a single causal stack with a long context does conditional generation perfectly well and is simpler to serve.
- **BERT-as-a-fine-tuned-classifier is still, frequently, the right answer** to the classification problems in this section. A fine-tuned small encoder at 5 ms and effectively zero marginal cost, beating a frontier API on your specific labeled task, is the classical-ML punchline of Part I dressed in transformer clothing.

**⚠ Trap:** "BERT is obsolete" is a claim that gets people rejected. It is obsolete as a *generator* — it was never a generator. It and its descendants are the backbone of the retrieval half of every RAG system in production.

### Why did transformers win? Give me the structural reasons, not "they scale better."

Two structural properties, and both are about *shape*, not quality. I want to be precise because "they scale better" is a conclusion, not a reason.

**Reason one: parallelism over the sequence dimension.** An RNN's computation at position `t` depends on position `t−1`, so training a length-`T` sequence takes `T` sequential steps regardless of how many cores you have. Self-attention computes all positions simultaneously — `QKᵀ` is one `[B, H, T, d_h] × [B, H, d_h, T]` batched matmul — so the number of *sequential* operations is `O(1)` in `T` (it is `O(L)` in depth, but depth is small and fixed). That converts training from a latency-bound problem into a throughput-bound one, which is precisely the shape modern accelerators are built for. This is the reason transformers won, and it is a hardware argument.

**Reason two: `O(1)` path length between any two positions.** In an RNN, information from position 3 reaches position 500 by passing through 497 transformations, and the gradient must survive the same trip — that is the vanishing-gradient mechanism restated as a graph property. In self-attention, position 500 attends directly to position 3 in a single hop. The maximum path length between any pair of positions is 1, so the gradient path is constant-length regardless of distance. Long-range dependency modeling stops being an optimization problem and becomes an addressing problem.

The cost of both is explicit and worth stating so the answer is balanced: self-attention is `O(T²·d)` in compute and `O(T²)` in attention memory per head in the naive formulation, against the RNN's `O(T·d²)` and `O(T)`. Transformers traded an asymptotically worse dependence on sequence length for a dramatically better *constant factor on real hardware* and a vastly better gradient path — and then the field spent the following years reclaiming the quadratic term with FlashAttention-style tiling, sparse and sliding-window patterns, and KV-cache engineering.

**📐 Numbers you must know:** at `T = 1,024` and `d = 1,024`, `T²·d = 1.07e9` versus `T·d² = 1.07e9` — exactly equal, which is the useful crossover to remember. The quadratic term only dominates above `T ≈ d`. That is why quadratic attention was a non-issue at the original 512-token context and became the central engineering problem at 128k, where `T²·d` is `128000² × 1024 ≈ 1.7e13` per layer per head-group and `T·d²` would be `1.3e11` — a factor of 128 apart. Being able to produce this crossover on demand is a strong signal.

**🗣 Say this in the room:** "Two reasons, both structural. Attention has O(1) sequential operations across the sequence where an RNN has O(T), so training saturates the hardware instead of serializing. And the path between any two positions is one hop instead of T, so long-range gradients don't decay geometrically. The price is O(T²) attention, which was irrelevant at 512 tokens and became the whole engineering problem at 128k."

### Design me a shift-scheduling feature. The user describes what they need in English and we produce the schedule.

This is the canonical neurosymbolic case and I would answer it by drawing a hard line: **the LLM translates, the solver decides.** An LLM asked to directly emit a schedule for 40 nurses across 21 days with coverage minima, skill requirements, consecutive-shift limits and time-off requests will produce something that looks like a schedule and violates constraints in ways nobody notices until someone shows up to an unstaffed shift. A constraint solver either produces a schedule satisfying every stated constraint or proves that none exists. Those are categorically different products.

The architecture, in four stages:

**1. Natural language → structured specification.** The LLM's job is to emit a typed constraint object, not a schedule. A Pydantic model with an explicit closed vocabulary of constraint types: `MinCoveragePerShift`, `MaxConsecutiveDays`, `RequiredSkillOnShift`, `TimeOffRequest`, `MinRestHours`, `SoftPreference(weight)`. Structured outputs / constrained decoding so the shape is guaranteed. Anything the user says that does not map onto a supported constraint type must produce an explicit `unsupported` entry rather than a silent drop — this is the single most important design decision in the whole system.

**2. Human confirmation of the specification.** Render the parsed constraints back in plain English and have the user confirm. This is the step that makes the whole design defensible: the model's fallible output is reviewed *before* it drives anything, by the person who owns the requirement. It costs one UI screen and eliminates the entire class of "the model misunderstood and we didn't find out for three weeks."

**3. Solve.** CP-SAT (OR-Tools) is my default for scheduling and rostering — the problem is naturally expressed with boolean assignment variables `x[employee, day, shift]` and combinatorial constraints, which is exactly CP-SAT's strength. Hard constraints go in as constraints; soft preferences become weighted terms in the objective. MIP (Gurobi/CBC via PuLP) is the alternative when the problem is more naturally linear with continuous quantities — cost minimization, blending, network flow. Either way, set a time limit and accept the best incumbent solution rather than waiting for proven optimality.

**4. Explain, using the LLM again.** If the model is infeasible, the solver can give you an irreducible infeasible subset of constraints; the LLM turns that into "you asked for at least 3 nurses on every night shift, but only 4 of your staff are night-certified and you've approved time off for 2 of them." That is the highest-value use of the model in the whole flow and it is pure translation.

**⚠ Trap:** letting the LLM "fix" an infeasible model by relaxing constraints on its own. It will silently drop the one the user cared most about. Infeasibility is a conversation with the user, mediated by the model, never resolved by it.

**💰 Math:** the economics are what make this an easy sell. A 40-employee, 21-day, 3-shift roster is 40 × 21 × 3 = 2,520 boolean variables — trivial for CP-SAT, typically solved to a good incumbent in under a second on one core, so effectively free and fully deterministic given a fixed seed and time limit. The LLM cost is two calls per session: parse (~1,500 in / ~600 out) and explain (~800 in / ~300 out), so roughly (2,300 × 3e-6) + (900 × 15e-6) = $0.0069 + $0.0135 = **$0.02 per scheduling session.** Against an LLM-generates-the-schedule design, which needs several thousand output tokens, multiple repair rounds, and still gives you no guarantee: call it 6,000 output tokens across retries = $0.09, four times the cost for an unverified answer.

**🗣 Say this in the room:** "The model writes the specification; the solver produces the answer. Anything the model emits is a typed constraint object the user confirms in plain English before we solve — so the failure mode becomes a visible misparse rather than an invisible constraint violation. And when it's infeasible, the solver gives me the conflicting subset and the model explains it, which is the part users actually value."

### Where else does the "LLM writes the spec, a deterministic engine produces the answer" pattern apply?

This is one general pattern with several well-established instances, and the way to recognize it is: **whenever there exists an engine that is exactly correct on formal input, the LLM's job is to be the parser into that formalism, and the engine's job is to be the answer.** The LLM contributes coverage over messy natural language; the engine contributes correctness. Neither can do the other's job.

The instances worth naming:

**Arithmetic and symbolic math.** A model doing multi-digit arithmetic in-context is doing pattern completion over a representation with no place-value structure. Give it a calculator or a CAS (SymPy) as a tool and the arithmetic becomes exact. Extended: **📄 Paper:** Gao et al. (2023), PAL (Program-Aided Language Models) and, concurrently, Chen et al. (2022) Program-of-Thoughts — the model emits a Python program as its reasoning chain and an interpreter executes it, so the reasoning steps are natural language but the *computation* is delegated. The reported gains on math word problems were large, and the reason is mechanical: you have moved the error-prone part out of the model.

**Classical planning.** Encode the domain and goal in PDDL and hand it to a classical planner such as Fast Downward (**📄 Paper:** Helmert, 2006), which returns a plan that is correct by construction and often optimal. The LLM's contribution is translating a fuzzy goal statement and world description into PDDL predicates and actions. This is a documented line of work; the pattern's value is that a planner never emits a plan with an unsatisfied precondition, which an LLM does routinely on anything past six steps.

**Program synthesis with verification.** The model proposes code; a test suite, type checker, or property-based test verifies it; failures feed back as a repair loop. The key property is that the verifier is *sound* — passing tests is real evidence, not a model's opinion of itself — which is why code is the domain where agents work best. Sample `n` candidates, keep those that pass, and you have converted an unreliable generator into a reliable system, which is exactly what pass@k measures.

**SMT/SAT for logical constraints.** Puzzle-like or configuration problems (does this access-policy set have a conflict? is this configuration satisfiable?) go to Z3. The LLM writes the assertions.

**Query languages as the formalism.** Text-to-SQL, text-to-Cypher, text-to-PromQL are the same pattern wearing enterprise clothes: the database is the engine, the query is the spec.

**Theorem proving and geometry.** **📄 Paper:** Trinh et al. (2024), AlphaGeometry (Nature) — a language model proposes auxiliary constructions while a symbolic deduction engine does the actual deriving; the combination reached a level near International Mathematical Olympiad gold medalists on geometry problems that neither component approaches alone. It is the cleanest published demonstration of the division of labor: the neural half handles the unbounded search over "what should I try," the symbolic half guarantees every step is valid.

**⚠ Trap:** the pattern only pays off when the *specification is easier to verify than the answer is to produce.* That is true for code (run the tests), for constraint problems (check the constraints), for SQL (inspect the query). It is false for open-ended summarization or for a legal argument, where the "spec" is as fuzzy as the output — and forcing a formalism there produces a brittle system that fails on everything outside its ontology. Knowing which side of that line a problem sits on is the judgment being tested.

**🗣 Say this in the room:** "Wherever there's an engine that's exactly right on formal input — a solver, a planner, an interpreter, a database — I use the model as the translator into that formalism and never as the answer. The precondition is that verifying the spec is cheaper than producing the answer; if it isn't, the formalism becomes a cage and I'd argue against it."

### Design the full cascade: classical model in front, LLM behind. Thresholds, evaluation, monitoring — all of it.

The mental model: a cascade is a cache with a quality dimension. Cheap tier answers when it is confident, expensive tier answers when it is not, and the escalation rate is the single number that determines your entire cost curve. You already know how to reason about hit rates; this is that, with the twist that a "miss" is defined by a calibrated confidence rather than by key absence.

**The components.** Tier 0: exact-match cache and curated answers for the head of the distribution. Tier 1: a fine-tuned small model or classical classifier, sub-10 ms. Tier 2: the frontier model. Optionally Tier 3: human review for the tail of the tail.

**Setting the threshold.** This is the part people do by feel and should not. Take a labeled validation set of a few thousand examples. For each candidate threshold `t`, compute two curves: the escalation rate `e(t)` (fraction of examples with confidence below `t`) and the **accuracy of Tier 1 on the examples it keeps**, `a(t)`. Then blended accuracy is `a(t)·(1−e(t)) + a_llm·e(t)` and blended cost is `c1 + e(t)·c2`. Now you have a two-column table and can either fix a quality floor and minimize cost, or fix a budget and maximize quality. **State which of those two you are optimizing, because the business almost always has one and not the other.**

The subtlety that separates a good answer from a great one: **the model's raw max-probability is not a calibrated confidence**, so `p > 0.7` does not mean 70% accurate. Fit temperature scaling or isotonic regression on a held-out set first, then the threshold means something and you can specify it as "escalate anything below 92% calibrated confidence" — a statement a product manager can actually reason about.

**Evaluating a cascade properly.** Three numbers, always reported together: end-to-end blended quality, escalation rate, and blended cost per 1k requests. Plus one more that people forget — **the accuracy of the escalation decision itself**, measured as: of the examples Tier 1 kept and got wrong, what fraction should have escalated? That is your *silent failure rate* and it is the metric that determines whether the cascade is safe. A cascade with 92% blended accuracy where the 8% of errors are all confidently-wrong Tier-1 answers is much worse than one where they are Tier-2 errors, because the former is undetectable at runtime.

**Monitoring.** Escalation rate is your primary operational signal and it is beautifully diagnostic. A gradual rise means input drift — the world is moving away from Tier 1's training data. A step change means a deploy: someone changed the threshold, the calibrator, or the model. A fall in escalation rate with flat quality is good; a fall with dropping quality means your calibrator is broken and Tier 1 is confidently wrong more often. I alert on escalation rate at ±20% of a rolling 7-day baseline, and I page on it, because it is the earliest and cheapest signal in the system.

**The virtuous loop.** Every escalated example gets a Tier-2 answer, and those answers are training data for Tier 1 — filtered by confidence and audited on a sample. Retrain monthly on the accumulated escalations and the escalation rate falls, which means the cost falls. **This is the thing to say last, because it reframes the cascade from a cost hack into a system that improves itself.**

**💰 Math, end to end:** 3M requests/month. All-frontier: 3M × $0.0021 = **$6,300/month**, p50 700 ms. Cascade with a 12% escalation rate: Tier 1 hosting at two instances ≈ $150 + embeddings ≈ $50 + 360,000 × $0.0021 = $756, total **$956/month**, an 85% reduction, with p50 at 8 ms and p88 at 8 ms. After three months of retraining on escalations, escalation falls to 7%: 3M × 0.07 × $0.0021 = $441 + $200 = **$641/month**. The marginal saving from that retraining is $315/month, which by itself does not justify an engineer's time — so **be honest that the retraining loop is justified by the quality gain and the latency, not the dollars, at this volume.** At 300M requests/month the same arithmetic gives $630k versus $64k and the conversation is entirely different. Scale decides which argument you make, and saying that is the senior move.

### Last one. You have four hours, a dataset, and a brief that says "use AI to solve this." Walk me through your first thirty minutes.

I want to end on the meta-skill, because everything in this section converges on it: the first thirty minutes determine whether you produce a system with a defensible boundary or a prompt with a demo.

**Minutes 0–8: characterize the data before touching a model.** Row count. Class distribution — print it, because a 78%-majority class silently sets your floor. Text length distribution, so you know your token costs and whether truncation will bite. Duplicate and near-duplicate rate. Missing values. Label quality: read twenty examples by hand, and specifically read five where you disagree with the label. If you find the taxonomy is ambiguous, *that is the headline finding of your submission* and it outranks anything you build.

**Minutes 8–15: run the gate, out loud, in the README.** Is the output space finite? Is the input format stable? Does a deterministic system already have the answer? Is there a verifier? Do I have labels? Write the answers down as a short section titled "why this design." Whatever you build after this, the reviewer now knows you chose rather than defaulted, and that is the largest single delta in how these are graded.

**Minutes 15–22: build the eval harness before the model.** Stratified train/validation/test split with a fixed seed. The metric, chosen and justified — macro-F1 with a per-class table, or nDCG@k with the judged-relevant count, or exact match on a constrained output. A bootstrap confidence interval function, ten lines. A single `evaluate(predict_fn) -> dict` entry point so every subsequent approach plugs into the same measurement. Now every experiment costs you three minutes instead of thirty.

**Minutes 22–30: the trivial baselines.** Majority class. TF-IDF plus logistic regression, or BM25 if it is retrieval. Get a number on the board. You will now spend the remaining three and a half hours knowing exactly what you have to beat, and — the part that wins these — you will have a first row in a comparison table that makes every later number interpretable.

Then build the sophisticated thing. Then cascade it against the baseline. Then write the honest paragraph about where it fails.

**🏋 Drill:** set a 30-minute timer on any public labeled text dataset you have never used. Pass criterion: at the buzzer you have a printed class distribution, a written gate answer, a working `evaluate()` with bootstrap CIs, and two baseline rows in a markdown table. Do this three times on three different datasets before your loop. The point is not the baselines — it is that the first thirty minutes stop being a decision and become a reflex, which is what frees your attention for the interesting part while an interviewer is watching.

**🗣 Say this in the room, when they ask how you'd start:** "Before I pick an approach I'd spend twenty minutes on the data and the eval harness — class distribution, twenty examples read by hand, a fixed split with bootstrap intervals — and then get a TF-IDF-plus-logistic-regression number on the board. Everything after that is a comparison against a known baseline instead of a claim, and if the baseline turns out to be within noise of the fancy version, that's the most useful thing I can tell you."
