### Switch me to pruning. Teach it from the ground up — what does it mean to prune an LLM, and why do you keep saying unstructured pruning is usually worthless?

Pruning is setting weights to zero permanently. That's it. The whole field is about *which* weights and *what shape* the resulting zeros take, and the second question matters far more than the first, which is the part people get wrong.

**Unstructured pruning** zeroes individual weights wherever they happen to be. You can typically remove 50% of an LLM's weights this way with a small quality cost. The problem is that a matrix with 50% of its entries zero, scattered randomly, is **not faster to multiply on a GPU**. Dense GEMM kernels read contiguous tiles; a zero contributes a multiply-by-zero that costs exactly as much as any other multiply. To exploit the sparsity you need a sparse format (CSR and friends), which stores indices alongside values, has irregular memory access, cannot use tensor cores, and on the ~50% sparsity regime is reliably *slower* than the dense kernel it replaced. Sparse formats start winning somewhere north of 90–95% sparsity, and LLMs fall apart well before that.

You don't even get the memory saving cleanly. A CSR representation of a 50%-sparse fp16 matrix stores half the values (1 byte/weight amortised) plus an index per stored value (2 bytes at 16-bit indices), so you're at 3 bytes per surviving weight — *worse* than the 2 bytes/weight of the dense fp16 matrix you started with. Unstructured 50% sparsity makes your model bigger and slower. State that plainly in an interview; it is the fastest way to show you've actually deployed something.

**Structured pruning** removes whole units — an entire attention head, an entire FFN intermediate channel, an entire layer, an entire expert. The result is a **smaller dense model**: fewer rows and columns, same kernels, same tensor cores, real speedup proportional to the removal, real memory saving with no index overhead, works on every engine, composes with quantization. The cost is that you can remove far less before quality collapses — 20–30% of parameters structurally is aggressive where 50% unstructured is routine — and you almost always need a recovery training phase.

**Semi-structured (2:4)** is the compromise: unstructured-ish flexibility within a hardware-mandated pattern, with silicon support. It's the only "sparse" thing with a credible hardware story.

**⚠ Trap:** quoting "we pruned 50% of weights" as an efficiency win without saying what the wall-clock and memory numbers were. In review I ask exactly one question of any pruning proposal — *what shape are the zeros, and which kernel exploits that shape?* — and it kills most unstructured proposals in thirty seconds.

**🗣 Say this in the room:** "Unstructured pruning is an accuracy result, not a systems result — random zeros don't speed up a dense GEMM and a sparse format at 50% density costs more bytes than dense fp16. The only sparsity I'd deploy is 2:4 semi-structured, which has silicon support, or actual structured removal of heads, channels and layers, which just gives me a smaller dense model."

### Implement a one-shot pruning method from scratch, and explain why Wanda works despite being so simple.

Wanda's mental model: importance is not `|w|`, it is `|w|` times *how big the thing it multiplies is*. A weight of 0.01 sitting on an input channel whose activations run at 200 contributes more to the output than a weight of 0.5 on a channel that idles at 0.1. Magnitude pruning ignores the input distribution entirely, which is why it fails on LLMs — the outlier-channel structure means activation scales vary by two orders of magnitude across channels, so `|w|` is a terrible proxy for contribution.

**📄 Paper:** Sun, Zhao, Zhou, Xiao, Kolter et al. (2023/2024), "A Simple and Effective Pruning Approach for Large Language Models" (Wanda) — showed that the metric `|W_ij| · ‖X_j‖₂`, compared *within each output row*, matches SparseGPT-class quality at 50% sparsity with **no weight update and no second-order computation**, i.e. one forward pass over calibration data.

Two details carry all the weight. First, the metric multiplies the weight by the **L2 norm of the corresponding input activation across the calibration batch** — that's the "and activations" in the name. Second, the **comparison group is per output row**, not the whole tensor. If you rank globally, entire rows get wiped out because some output neurons naturally have smaller weights, and you've silently done structured pruning by accident. Per-row ranking guarantees every output neuron keeps exactly 50% of its inputs.

```python
import torch

@torch.no_grad()
def wanda_prune_linear(layer: torch.nn.Linear, calib_inputs, sparsity=0.5, n_m=None):
    """calib_inputs: (n_samples, seq, in_features) activations entering this layer."""
    X = calib_inputs.reshape(-1, layer.in_features).float()
    act_norm = X.pow(2).sum(0).sqrt()                 # (in_features,)  == ||X_j||_2
    W = layer.weight.data                             # (out_features, in_features)
    metric = W.abs() * act_norm.unsqueeze(0)          # importance, broadcast per column

    mask = torch.ones_like(W, dtype=torch.bool)
    if n_m:                                           # semi-structured, e.g. (2, 4)
        n, m = n_m
        met = metric.view(W.shape[0], -1, m)
        drop = met.argsort(dim=-1)[..., : m - n]      # keep the n largest of every m
        mask.view(W.shape[0], -1, m).scatter_(-1, drop, False)
    else:                                             # unstructured, per output row
        k = int(W.shape[1] * sparsity)
        drop = metric.argsort(dim=-1)[:, :k]
        mask.scatter_(1, drop, False)

    layer.weight.data = W * mask
    return mask
```

Note that the same code produces 2:4 semi-structured masks by changing the comparison group to every contiguous block of 4 along the input dimension — which is precisely the constraint Ampere's sparse tensor cores require, and it's why Wanda is the cheapest way to produce a hardware-compatible sparse checkpoint.

**⚠ Trap:** computing `act_norm` from a calibration set that doesn't match production is the same poisoning failure as with GPTQ/AWQ, and it's sharper here because pruning is *irreversible* — the weight is gone, there's no scale you can retune. And running the calibration forward pass with the *already-pruned* earlier layers versus the dense earlier layers gives materially different norms; sequential (pruned-input) calibration is the correct choice and the lazy implementation is the common one.

### Explain SparseGPT and its relationship to GPTQ.

They are the same algorithm with a different constraint set, by the same lead author, and noticing that is a strong signal in an interview.

**📄 Paper:** Frantar & Alistarh (2023), "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot" — showed that 100B+ models can be pruned to 50–60% unstructured sparsity in a single pass, in a few GPU-hours, with minimal perplexity loss and no retraining, which no prior method could do at that scale.

The shared machinery: both minimise layer-wise reconstruction error `‖WX − ŴX‖²` using the layer Hessian `H = 2XXᵀ`, both process columns in a fixed order with a Cholesky factorisation of `H⁻¹` shared across rows, and both compensate the error of each decision by updating the not-yet-processed weights. The difference is only what the per-weight decision *is*: GPTQ decides "which grid point does this weight snap to?"; SparseGPT decides "is this weight zero or does it stay?" — and the same second-order formula gives you both the optimal selection criterion (`w_i² / [H⁻¹]_{ii}`, the loss increase from zeroing `w_i`) and the optimal compensating update to the survivors.

The extra engineering SparseGPT needed: the classical OBS formulation requires a *different* Hessian inverse per row, because each row makes different pruning decisions and therefore has a different remaining-weight set. That's `d_row` Cholesky factorisations, which is fatal at scale. SparseGPT's fix is to fix a shared column *order* and let rows differ only in their mask within that order, so one factorisation serves all rows — and to interleave mask selection with the weight updates in blocks, so the mask for block `k+1` is chosen after block `k`'s error has been propagated.

It also handles the n:m constraint directly: within each block of `m` columns, keep the `n` with the lowest loss-increase, which is why SparseGPT produces good 2:4 masks natively rather than as an afterthought.

Choosing between them in practice: Wanda is essentially free (one forward pass, no weight updates, minutes) and lands close to SparseGPT at 50%. SparseGPT costs GPU-hours and a Hessian inverse per layer but pulls ahead at higher sparsity and in the n:m regime where the constraint is tight. My rule: **try Wanda first because it's an afternoon; escalate to SparseGPT only if the Wanda mask fails your eval and you've already confirmed the sparsity target is non-negotiable.**

### 2:4 structured sparsity — how does the hardware actually exploit it, and what speedup do you really get?

The mental model: Nvidia decided that fully-general sparsity is unschedulable in silicon, so they picked the most constrained pattern that still leaves useful freedom — **exactly two of every four contiguous elements along the reduction dimension must be zero** — and built a tensor core that skips the zeros deterministically.

Mechanically, the compressed representation stores the two surviving values per group of four, plus a **2-bit index per surviving value** saying which of the four positions it came from. So a 2:4 sparse fp16 matrix stores 2 values (4 bytes) + 4 bits of metadata per 4 original weights (8 bytes dense) = 4.5 bytes vs 8 — a **1.78× memory reduction**, not 2×. The Sparse Tensor Core, when fed the compressed operand plus metadata, selects the matching elements of the dense operand on the fly and does half the MACs in the same number of cycles, giving a **2× theoretical math throughput**. Available on Ampere and later; this is why an A100's spec sheet quotes 312 dense / 624 sparse TFLOPS in bf16 and an H100 quotes ~990 / ~1,979.

Now the honest part, which is what an interviewer is fishing for. The 2× is a **GEMM-kernel ceiling on the compute-bound path only**, and LLM inference frequently isn't there:

- **Batch-1 decode is bandwidth-bound**, so the relevant number is 1.78× on weight bytes, not 2× on math — and even that is diluted because attention, norms, and the KV path are untouched. Expect something in the 1.2–1.4× range end-to-end, less if the engine's sparse path is less mature than its dense path.
- **Prefill and large-batch decode are compute-bound**, where the 2× applies to the GEMM but the GEMM is maybe 70–85% of the time, so Amdahl caps you around 1.5–1.7×.
- **Composing with 4-bit weights** — the memory win largely evaporates. 2:4 on top of int4 gives you 2 nibbles + metadata per 4 weights; the metadata is a larger fraction of a smaller value, and support for sparse-int4 kernels is much thinner than for sparse-fp16.

And the quality cost is real: 50% sparsity is not free, and Nvidia's own recommended recipe (their ASP tooling) is **train dense → prune 2:4 → retrain with the mask fixed**, i.e. it assumes you can afford a recovery training run. One-shot Wanda/SparseGPT 2:4 without retraining costs meaningfully more quality than their unstructured 50% counterparts, because the per-group constraint forces you to drop weights you'd rather keep.

**📅 Volatile:** engine support for 2:4 in LLM serving stacks is patchy and changes release to release; a "supported" flag frequently means "the checkpoint loads," not "there's a fast kernel." Benchmark, don't assume.

**🗣 Say this in the room:** "2:4 is two-of-four zeros along K, with 2 bits of index per surviving value, so it's 1.78× on memory and 2× on the sparse tensor core's math ceiling. In practice I'd budget 1.2–1.4× end-to-end on decode and maybe 1.5× on prefill, against a real quality cost that usually needs a retraining pass to recover. I'd reach for 4-bit weight-only before I'd reach for 2:4 — better ratio, better tooling, less quality risk — and I'd only combine them if I had a validated sparse-int4 kernel."

### Width pruning versus depth pruning — which do you cut, and why does it matter for latency?

They remove the same number of parameters and produce completely different latency and quality profiles, so the choice is a real engineering decision rather than a preference.

**Width pruning** shrinks tensors: fewer attention heads, a smaller FFN intermediate dimension, sometimes fewer hidden channels. The layer count is unchanged. **Depth pruning** removes whole transformer blocks.

The latency argument favours depth, and it's about the critical path. A transformer's forward pass is a strictly sequential chain of `L` blocks; each block has kernel-launch overhead, several normalisation and elementwise ops that don't scale with width, and a synchronisation point. Removing 25% of the layers removes 25% of *everything*, including the fixed per-layer overhead and the sequential dependency chain. Removing 25% of the width shrinks the GEMMs but leaves the same number of launches, the same number of norms, and the same sequential depth — and narrower GEMMs achieve *lower* hardware utilisation, so a 25% width cut yields less than 25% time. Depth pruning also reduces the KV cache linearly (KV bytes are proportional to `n_layers`), which width pruning only does if you cut KV heads specifically.

The quality argument favours width. Depth is where sequential computation happens — reasoning, multi-step composition, anything that needs to build a representation across many transformations. Empirically, dropping layers degrades chain-of-thought and math far more than trimming FFN width does, and the damage is very unevenly distributed across depth (more on that next). Width pruning removes redundant capacity, which transformers have in abundance, especially in the FFN, whose intermediate dimension is typically 3.5–4× hidden and demonstrably over-provisioned.

The recipe that works, which is the Minitron-class answer: **prune width aggressively, prune depth modestly, and always distill afterwards.** In practice that means cutting FFN intermediate hardest, attention heads moderately, embedding/hidden channels least, and dropping only the layers you can prove are redundant.

**⚠ Trap:** cutting width without preserving the constraints your kernels need. Attention head count must stay divisible by your tensor-parallel degree; with GQA, the KV head count must also divide TP; hidden dimensions want to stay multiples of 128 for tensor-core tiling and for group-128 quantization. I've seen a "prune to 5,504 FFN intermediate" produce a model that ran *slower* than the unpruned one on TP=4 because of padding. Round every pruned dimension to a multiple of 128 and re-check TP divisibility before you spend a dollar on recovery training.

### If you were going to drop layers, how would you pick which ones?

Empirically and with a stated metric, never by intuition — and the empirical answer is consistently the same shape across model families, which is worth knowing going in.

The standard diagnostic is **block-level redundancy measured by representational similarity**: for each candidate block (or contiguous group of `n` blocks), compare the hidden state entering it with the hidden state leaving it, over a calibration corpus. Cosine similarity, or equivalently the angular distance between input and output residual-stream vectors, tells you how much that block actually changes the representation. A block whose output is nearly parallel to its input is doing very little; removing it and letting the residual stream pass through is a small perturbation. Rank blocks by that distance, drop the least-impactful contiguous span. This family of "which layers are redundant" analyses converges on a robust finding: **the middle-to-late-middle layers are the most redundant, and the first few and the last one or two are the most load-bearing.**

That asymmetry has a clean explanation. Early layers do detokenisation and local syntactic assembly — genuinely hard, genuinely non-redundant work on raw embeddings. Final layers do the projection into vocabulary space and any last-step calibration of the output distribution; removing them destroys the head's expectations about its input. The middle is where the model does many small, incremental, partly-redundant refinements to the residual stream, and it can survive losing some of them.

The alternatives worth naming: a **greedy leave-one-out** sweep — remove each layer individually, measure perplexity or your task metric, drop the `k` cheapest — is more accurate than the similarity heuristic and costs `L` evals, which for an 80-layer model at a few minutes per eval is an afternoon and is usually worth it. And **learned/gated depth**, where you train a gate per block and regularise toward sparsity, is more accurate still but needs a training run.

**⚠ Trap:** the layer-importance ranking is **task-dependent and, critically, length-dependent**. Layers that look redundant on short 2k perplexity are frequently the ones doing long-range retrieval work at 32k. Rank on a calibration mix that includes your production length distribution, and re-validate the pruned model on a long-context eval specifically — otherwise you ship a model that reads beautifully and can no longer find a fact on page 40.

**🏋 Drill (30 minutes with a GPU, unaided):** take any 1–3B open-weight model. Compute per-block input/output cosine similarity over 64 calibration sequences, rank the blocks, then measure WikiText perplexity after dropping the 4 least-impactful blocks versus 4 random blocks versus the last 4 blocks. Pass criterion: you can state the three perplexities and explain the ordering, and you should observe that dropping the *last* blocks is catastrophically worse than dropping middle ones.

### After you prune, what do you do? Walk me through the recovery step.

You distill, and this is the part that separates a pruning result from a pruning *product*. One-shot pruning without recovery is a demo; the standard industrial recipe is **prune → distill → (optionally) quantize**, and the distillation is not optional at any meaningful sparsity.

The mental model: pruning removes capacity the model was using, so the surviving weights are now in the wrong place — they were optimised as part of a larger system. Recovery training moves them to a good configuration for the new architecture. And you have a uniquely good teacher available: **the unpruned model**, which can generate soft targets on any amount of unlabelled text.

**📄 Reference recipe:** Nvidia's Minitron work (2024) on compressing language models via pruning and knowledge distillation established the modern standard — estimate per-component importance with activation-based criteria, structurally prune width (attention heads, FFN intermediate, embedding channels) and depth, then retrain the student with a distillation loss against the original model. The headline systems result is the token budget: they report needing on the order of **tens of billions of tokens rather than trillions — roughly 40× less training compute than pretraining a model of that size from scratch** — to recover to competitive quality.

The loss: forward KL between the teacher's and student's next-token distributions over the full vocabulary, usually blended with the standard cross-entropy on the ground-truth token, and often with an intermediate-representation matching term (hidden states or attention maps, with a projection when dimensions differ). Logit distillation carries far more signal per token than hard labels — a full 128k-way distribution per position instead of one index — which is exactly why the token budget collapses by ~40×.

**💰 Math — is this actually worth it?** Pretraining an 8B from scratch at Chinchilla-ish ratios is roughly 1.5e23 FLOPs (6 × 8e9 params × 3e12 tokens). Recovering a pruned 8B (from a 15B, say) with 100B distillation tokens is 6 × 8e9 × 1e11 ≈ 4.8e21 FLOPs — about **31× cheaper**. On H100s at, say, 400 TFLOPS achieved and $3/GPU-hour: 4.8e21 / 4e14 = 1.2e7 GPU-seconds = 3,333 GPU-hours ≈ **$10k**, against roughly $310k to pretrain. **📅 Volatile:** GPU pricing and achieved-FLOPS assumptions move; redo the arithmetic with current numbers, but the ~30× ratio is the durable part.

**⚠ Trap:** distilling on a corpus that doesn't include the *behaviours* you need. If the teacher is instruction-tuned and you distill on raw web text, you recover perplexity and lose instruction-following, tool-calling and chat formatting — the pruned model will look fine on benchmarks and be unusable in the product. Distill on a mixture that includes your instruction data and, critically, your tool-call and structured-output formats, with the chat template applied.

### How would you prune attention heads, and how does that interact with GQA?

Head pruning is the most tractable structured axis in attention, because a head is a genuinely independent computational unit: heads write into disjoint slices of the concatenated output which is then mixed by the output projection, so removing head `h` means deleting its Q/K/V projection rows and the corresponding columns of `W_O`. The result is a valid, smaller, dense model.

**Importance criteria**, in increasing cost and accuracy: (1) weight-norm based — the Frobenius norm of the head's slice of `W_O`, cheap and mediocre; (2) activation-based — the mean magnitude of the head's output contribution over calibration data, which is the Wanda/Minitron-style criterion and is what I'd use by default; (3) a first-order Taylor / gradient-times-activation estimate of the loss change from masking the head, which is more accurate and needs backward passes; (4) actual leave-one-out ablation, gold-standard and `n_heads × n_layers` evals, feasible only on small models.

The GQA interaction is where the real constraints bite, and it's the part interviewers use to check whether you've touched a real model. Under grouped-query attention, several query heads **share** one KV head. So:

- Pruning a **query head** is cheap and local: drop its Q rows and its `W_O` columns. The group it belonged to just gets smaller.
- Pruning a **KV head** requires removing the *entire group* of query heads that read it, or reassigning those query heads to a different KV head — which changes what they attend to and is a much larger perturbation. But KV heads are the ones that matter for the cache: KV bytes scale with `n_kv_heads`, not `n_q_heads`. So the memory-relevant pruning is the expensive one.
- Any pruning must preserve **divisibility by the tensor-parallel degree**. With TP=8 and 8 KV heads, you cannot drop a single KV head without breaking the sharding — you'd go to 4 KV heads (halving KV cache) or nothing.

A related and often better move than pruning heads: **converting MHA to GQA post-hoc** by mean-pooling groups of key and value projections and then distilling to recover. This is the "GQA-ification" trick and it attacks the KV cache directly — going from 64 KV heads to 8 on a 70B cuts KV bytes 8×, which is a far bigger production win than shaving a few query heads. It needs a recovery distillation run, and it's how several model families acquired GQA variants of already-trained checkpoints.

**⚠ Trap:** pruning heads uniformly across layers. Head redundancy is extremely uneven — some layers have several heads doing near-identical work while others have every head specialised. A uniform 25% cut destroys the specialised layers to save the redundant ones. Allocate the pruning budget per-layer proportional to measured redundancy, which is what the importance-ranking step is for.

### An MoE model has an expert that barely gets routed to. Would you delete it? How, and what breaks?

My answer is "maybe, and only after I've measured utilisation on traffic that actually represents production" — because expert utilisation is the most seductively misleading metric in this whole area.

The mechanism first. In a mixture-of-experts FFN, a router produces logits over `E` experts per token and a top-`k` selection routes each token to `k` experts. Removing expert `e` means: delete its up/gate/down matrices, delete row `e` from the router weight matrix, and let the softmax renormalise over the remaining `E−1`. That's structurally clean — MoE is the one architecture where an entire large module can be excised without touching the rest of the graph. Memory saving is direct: on a model where experts are, say, 85% of parameters, dropping 8 of 64 experts removes about 11% of the model.

How to measure utilisation properly: run a large, *representative* corpus and accumulate two things per expert — **token count** (how many tokens selected it) and **routing mass** (the sum of its gate weights, which captures that a token can select an expert with weight 0.51 or 0.02). Rank by routing mass, not count. And measure at the *per-layer* level, since routing distributions differ dramatically across depth.

Now the trap, and it is the whole question. **Experts specialise, and the specialisation is often along an axis your calibration corpus doesn't span.** An expert that receives 0.3% of tokens on your English chat corpus may be the expert that handles code, or Chinese, or mathematical notation. Delete it and you get a model that is fine on everything you measured and catastrophically worse on a slice you didn't. This is the same class of failure as calibration poisoning but with sharper edges, because the deletion is total and irreversible. The measurement corpus must span every language, domain and format in production, and I would weight it by *importance*, not by frequency — a 2%-of-traffic legal-document workload can be 40% of revenue.

The safer alternatives to deletion: **expert merging** (average the weights of two similar experts and merge their router rows, which preserves some of the capability), and **expert offloading** (keep all experts but hold cold ones in CPU memory and page them in, trading latency for capacity — viable because MoE activates only `k` experts per token so the hot working set is small). Offloading is usually the right answer for a capacity problem, deletion for a genuine cost problem where you've validated the slice coverage.

**🗣 Say this in the room:** "I'd measure routing mass per expert per layer over a corpus that spans every language, domain and output format in production — not just token counts, and not on a generic English sample. Experts specialise, so a low-utilisation expert is frequently the code expert or the multilingual expert. If the numbers hold up I'd merge before I'd delete, and I'd gate on a per-slice eval, not an aggregate."

### Early exit sounds great in a paper. Why does it almost never survive contact with a production serving stack?

Because it optimises the wrong resource under the batching regime that actually runs in production, and because it breaks the KV cache invariant. Both are fatal; the second is the one people don't see coming.

The idea: attach a classifier or a confidence estimate to intermediate layers; if the model is already confident about the next token at layer 12 of 32, skip layers 13–32. Easy tokens (function words, closing brackets, the second half of a memorised phrase) genuinely are decided early, so you'd think you'd save a lot of compute. **📄 Paper:** Schuster et al. (2022), "Confident Adaptive Language Modeling" (CALM) — formalised per-token early exit for encoder-decoder LMs with calibrated confidence thresholds *and* a state-propagation mechanism, which exists precisely because of the problem I'm about to describe.

**Problem 1 — the KV cache.** If token `t` exits at layer 12, layers 13–32 never computed keys and values for position `t`. But token `t+1` will run to layer 32, and at layer 20 its attention needs the KV of *every* previous position including `t`. That KV does not exist. Your options are all bad: recompute the missing layers for `t` (destroying the saving), propagate the layer-12 hidden state upward as a cheap surrogate for the missing states (CALM's approach — an approximation that accumulates error over a long generation), or forbid early exit entirely once any later token needs deep attention. This is not an implementation detail; it is a structural conflict between early exit and autoregressive attention.

**Problem 2 — continuous batching.** Production serving runs 32–256 sequences through the model in lockstep, layer by layer, because that's what makes the GEMMs large enough to be efficient. If one sequence in the batch wants to exit at layer 12, the GPU still has to run layers 13–32 for the other 63 sequences. You save nothing in wall-clock; you only save the FLOPs of one row of a GEMM, and GEMM cost is dominated by the tiles, not the rows. To get a real win you'd need to *compact* the batch mid-forward — remove the exited rows and continue with a smaller GEMM — which means ragged, dynamically-shaped kernels, a scatter/gather per layer, and a scheduler that can handle sequences at different depths. The engineering cost is enormous and the payoff shrinks as batch size grows, because with 64 sequences the probability that *all* of them want to exit at layer 12 is essentially zero, so you always run to full depth anyway.

The regime where early exit does pay: **batch size 1, on-device**, where there is no batch to hold you back and the KV problem is manageable because generations are short. That's a real but narrow niche.

**⚠ Trap:** benchmarking early exit at batch 1 and projecting the savings to a batched fleet. The saving does not just shrink with batch size — it goes to approximately zero, because the expected max exit depth over `B` sequences converges to the full depth very quickly. Compute that expectation before you fund the project.

### So what's the version of adaptive compute that does work in production?

**Cascades** — and the key structural difference is that a cascade puts the routing decision *between* model invocations rather than inside one forward pass, so each stage is an ordinary service that batches normally.

The design: a cheap model (small, quantized, distilled, or just a cheaper API tier) handles the request. A decision function decides whether to accept its answer or escalate to the expensive model. Because the stages are separate deployments, each one runs its own continuous batching at full efficiency, each scales independently, and nothing about the transformer's internals has to change. **📄 Paper:** Chen, Zaharia, Zou (2023), "FrugalGPT" — formalised LLM cascades with a learned scoring function and reported large cost reductions at matched or better accuracy, which made cascading a mainstream production pattern rather than a research curiosity.

The decision function is the whole game, and there are three families. **Self-confidence** — the small model's own sequence log-probability or a calibrated verbalised confidence; cheap, but LLM confidence is poorly calibrated and this leaks. **A trained router/scorer** — a small classifier over the query (and optionally the draft answer) predicting whether the big model would do better; this is the FrugalGPT shape and it works well when you have logged data from both models. **A verifier** — for tasks with checkable output (code that must compile, JSON that must validate against a schema, SQL that must parse, a math answer you can check), run the check and escalate on failure. Verifier-based cascades are the strongest because the escalation signal is *ground truth*, not a guess, and I push for them whenever the task has any checkable structure.

**💰 Math — the cascade economics.** Suppose the small model costs $0.20/Mtok blended and the large costs $6/Mtok, and an average request is 2,000 in + 300 out ≈ 2.3k tokens. Large-only: 2.3e3 × 6/1e6 = **$0.0138/request**. Cascade with a 70% acceptance rate: every request pays the small model (2.3e3 × 0.2/1e6 = $0.00046) and 30% additionally pay the large ($0.0138), giving 0.00046 + 0.3 × 0.0138 = **$0.00460/request** — a 3.0× reduction. At 2M requests/day that's $27,600/day → $9,200/day, about **$550k/month saved**. Now the honest counterweight: the escalated 30% pay the small model's latency *plus* the large model's, so their p95 gets worse. If your SLO is on p95 rather than p50, you may need to run both in parallel and cancel — which recovers the latency and loses about half the cost saving. **📅 Volatile:** those per-token prices; rerun with current rates.

**⚠ Trap:** measuring cascade quality only in aggregate. A cascade that accepts 70% of requests has, by construction, made 70% of your traffic *worse* than the large model would have been, in exchange for the 30% it escalates. If your acceptance threshold is tuned on average quality, you will silently degrade a specific hard slice. Tune the threshold per-slice and report per-slice.

### Explain nested or elastic models — the Matryoshka idea. What problem does a single-checkpoint multi-size deployment solve?

The operational problem is real and boring: you want a 2B for mobile, a 4B for the cheap API tier, and an 8B for the premium tier. Training three models costs three pretraining runs, three eval campaigns, three quantization pipelines, three sets of weights in the registry, three sets of GPU images. Nested models let you train **one** checkpoint whose prefixes are themselves valid smaller models, so you extract sizes rather than train them.

The idea originated in embeddings. **📄 Paper:** Kusupati et al. (2022), "Matryoshka Representation Learning" — trains an embedding so that the first `d` dimensions of the vector are themselves a good embedding for many `d`, by applying the training loss at multiple nested truncation points simultaneously. The practical consequence is enormous for retrieval: you index at 1,536 dimensions but do the first-stage ANN search at 256 dims (6× less memory and distance-computation cost) and rerank the top candidates at full width, with no separate model. This is why several modern embedding APIs let you request a shorter vector — that's MRL, and it is the single most immediately-useful compression idea in this whole section for a RAG system.

Applied to transformers, the analogue is **MatFormer** (Devvrit et al., 2023), "MatFormer: Nested Transformer for Elastic Inference": the FFN of each block is trained with nested sub-networks — the first `d_ff/8`, `d_ff/4`, `d_ff/2` and full intermediate dimensions are each trained to be a working FFN, using a randomly-sampled granularity per training step. At inference you pick a granularity per layer, which means you can extract a whole combinatorial family of models from one checkpoint, and mix granularities across layers to hit an exact latency target.

There's also a quantization analogue worth naming — training a single checkpoint whose high-order bits are themselves a valid lower-bit model, so int8, int4 and int2 versions come from one artifact by truncation. **📅 Volatile:** this line of work is recent and moving; describe the mechanism, don't over-claim specific results.

**⚠ Trap:** expecting a nested model's extracted 2B to match a dedicated 2B trained with the same total compute. It generally does not — the nested objective is a compromise across granularities, and the smallest slices pay for it most. You are trading a point or two of quality for operational simplicity and for the ability to choose a size at *request* time rather than deploy time. That's frequently a good trade for a product with tiered pricing; it's a bad trade if you ship exactly one size.

### If I'm going to apply several of these — quantize, prune, distill — what order do I do them in, and why?

The order I recommend, and the reasoning, because the reasoning is the answer:

**1. Architecture / structured pruning first.** Removing heads, FFN channels and layers changes the *shape* of the tensors. Every downstream step — calibration, quantization scales, sparsity masks — is computed against a specific shape and must be redone if the shape changes. Prune first so everything after it is computed once.

**2. Distillation / recovery training second.** The pruned model is in a bad configuration; recover it while it is still in high precision, where gradients are well-behaved and you have the full optimisation toolkit. Trying to recover a *quantized* model requires QAT machinery you probably don't want.

**3. Sparsity (2:4) here, if at all** — during or immediately before the recovery training, with the mask fixed, so the retraining adapts the surviving weights to the mask. One-shot 2:4 after everything else is materially worse.

**4. Quantization last.** It's the cheapest step (hours, not days), it's the most reversible (rerun with different settings), and it's the one most tightly coupled to your serving engine and GPU generation — which is the thing most likely to change under you. Making quantization the final, re-runnable step means an engine upgrade costs you a quantization job, not a training run.

**5. Evaluate after every step, not at the end.** If you prune, distill and quantize and the result is 6 points down, you have no idea which step cost what, and each re-run of step 2 is days. Gate each stage.

The interactions to know about, because an interviewer will probe them:

- **Pruning then quantizing is harmful in a specific way**: pruning concentrates the model's function into fewer weights, which *increases* their relative importance and reduces the redundancy quantization was exploiting. A model that tolerates 4-bit at full width may not tolerate it at 70% width. Budget for this — if you're doing both, consider 4-bit + mild pruning rather than 4-bit + aggressive pruning.
- **Distillation after quantization** is possible (that's essentially quantization-aware fine-tuning) but you're now optimising through a straight-through estimator, which is slower to converge and less stable.
- **Sparsity plus 4-bit** is the combination with the thinnest kernel support. Verify before you plan around it.

**🗣 Say this in the room:** "Prune, distill, then quantize — shape changes first because everything downstream is computed against the shape, recovery training while you're still in high precision, quantization last because it's cheap, reversible and coupled to the serving engine. And I gate on eval between every stage, because a 6-point drop measured only at the end tells me nothing about which step to fix."
