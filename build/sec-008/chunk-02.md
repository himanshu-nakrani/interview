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
