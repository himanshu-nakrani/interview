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
