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
