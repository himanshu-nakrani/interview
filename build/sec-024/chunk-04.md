### Averaging the weights of two different fine-tuned models sounds like it should produce garbage. Why does it work?

It should produce garbage, and for two independently-trained models it does. The thing that makes merging work is a precondition, and the precondition is the answer: **both models must have been fine-tuned from the same pretrained checkpoint.**

The geometry is this. A randomly-initialized network trained twice lands in two entirely different loss basins, related by permutation symmetries of the hidden units — averaging them averages unit 7 of one model with unit 7 of the other, which are unrelated features, and the result is noise. But fine-tuning does not travel far. Starting from a shared pretrained `θ_pre`, two fine-tunes on different tasks stay inside the same basin: the linear path between them does not cross a loss barrier. This is **linear mode connectivity**, and the empirical finding that fine-tuning from a shared initialization preserves it is what licenses everything in this part of the section.

Given that, the simplest merge is the arithmetic mean:

```
θ_merged = (1/n) Σ θ_i        # "uniform soup"
```

**📄 Paper:** Wortsman et al. (2022), *Model Soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time* — showed that averaging many fine-tunes of the same base (varying only hyperparameters) beats picking the single best on a validation set, and introduced greedy soup, which adds models one at a time only if they improve held-out accuracy. **📄 Paper:** Neyshabur, Sedghi & Zhang (2020), *What is being transferred in transfer learning?* — established that models fine-tuned from a shared pretrained checkpoint remain in the same loss basin, which is the mechanism soups rely on.

Why does averaging *help* rather than merely not hurt? Two framings, both worth having. **Ensembling in weight space**: an ensemble of `n` models averages their *predictions* and reduces variance, at `n×` inference cost. A soup averages their *weights* and approximates that variance reduction at `1×` inference cost — which is a remarkable deal when it holds. **Flat-minimum bias**: the average of several points in a basin sits closer to the basin's centre, and flatter minima generalize better; merging is a cheap way to move toward the middle.

**⚠ Trap:** "merging is free extra quality." It is not. Merging is *interference management*, and every merge trades some per-task peak for generality. A model merged from a code fine-tune and a math fine-tune is typically worse at code than the code fine-tune and worse at math than the math fine-tune, while being better than either at the union. If your production traffic is 95% one domain, merging is the wrong move and routing to the specialist is right. The honest framing is: **merging buys you one deployment instead of `n`, and you pay for it in peak per-domain quality.**

### Explain task vectors. What does subtracting one actually do?

The task-arithmetic framing is the most conceptually productive idea in merging, and it is one line:

```
τ_task = θ_finetuned − θ_pretrained
```

A **task vector** is the displacement in weight space that fine-tuning produced. Ilharco et al.'s observation is that these vectors behave, to a surprising degree, like vectors: you can scale them, add them, and subtract them, and the resulting model behaves as the arithmetic suggests.

**Addition** — combining capabilities:
```
θ_multi = θ_pre + λ₁·τ_code + λ₂·τ_math       # λ typically 0.3–1.0 each
```
Note that with `λ = 1/n` for all tasks this is exactly the uniform soup. Task arithmetic generalizes it by letting you weight tasks unequally and by making the pretrained anchor explicit.

**Negation** — removing a behaviour:
```
θ_clean = θ_pre − λ·τ_toxic
```
This is the striking result. Fine-tune a model *toward* a behaviour you do not want — toxicity, a specific style, a domain — then subtract that direction from the base. The paper reports meaningful reduction in the targeted behaviour with limited damage to general capability, which is a genuinely different mechanism from refusal training: you are removing a *direction*, not adding a *filter*.

**📄 Paper:** Ilharco et al. (2022), *Editing Models with Task Arithmetic* — introduced task vectors as first-class objects supporting addition (multi-task), negation (forgetting), and analogy composition, giving model editing an algebra.

**The `λ` question, which is where the judgment is.** `λ` is a real hyperparameter and there is no theory for it. Too small and the task contribution is invisible; too large and you degrade general capability, at the extreme producing repetition loops and gibberish. My starting point for two-task addition is `λ = 0.5` each, and for negation `λ ∈ [0.5, 1.0]` with an aggressive guardrail eval. **Always sweep `λ` against a per-domain suite plus a general suite — it is a two-dimensional trade and you must see both axes.**

**Where I would and would not use negation.** *Would:* stripping a stylistic tic, reducing a measurable behaviour you can fine-tune toward and evaluate against, ablating a capability for a restricted deployment. *Would not:* as a **GDPR/right-to-erasure mechanism**. Negation reduces a behaviour in expectation; it is not a proof that information was removed, and a regulator asking "is this customer's data gone?" does not accept "the loss on it went up." For genuine deletion the architecture must keep the tenant's contribution as a *separable artifact* — an unmerged adapter you can delete, or a retrainable base — which is the real reason I argued earlier against merging tenant adapters into shared weights. **This is a strong senior signal in an enterprise design round: recognize that unlearning is a research area, not a compliance control.**

### Why would I use SLERP instead of a linear average? Show me the problem it fixes.

Because in high dimensions, linear interpolation **shrinks the norm**, and norm shrinkage in a transformer is not a cosmetic problem — it changes the scale of activations flowing into every downstream normalization and softmax.

Here is the arithmetic that makes it concrete. Take two vectors `a` and `b` of equal norm `‖a‖ = ‖b‖ = 1`. In high dimensions, independently-trained directions are close to orthogonal (`a·b ≈ 0`). Then:

```
‖(a + b)/2‖ = (1/2)·√(‖a‖² + 2a·b + ‖b‖²) = (1/2)·√2 ≈ 0.707
```

You lost **29% of the norm** simply by averaging. For nearly-parallel vectors the loss is negligible; for near-orthogonal ones it is severe. Since different fine-tunes push weights in genuinely different directions, the merged weight matrices come out systematically smaller than either parent's.

**SLERP** — spherical linear interpolation, originally from quaternion animation (Shoemake, 1985) — interpolates along the great-circle arc on the hypersphere instead of along the chord, preserving norm:

```
Ω = arccos( (a·b) / (‖a‖‖b‖) )
slerp(t; a, b) = [ sin((1−t)Ω)/sin Ω ]·a  +  [ sin(tΩ)/sin Ω ]·b
```

At `t = 0.5` with `Ω = 90°`: `sin(45°)/sin(90°) = 0.707` for both coefficients, so `‖result‖ = √(0.707² + 0.707²) = 1.0`. Norm preserved exactly. Note that SLERP degenerates gracefully: as `Ω → 0` the coefficients approach `(1−t)` and `t`, so for nearly-parallel weights SLERP and LERP agree — which is why SLERP is never *worse*, just sometimes unnecessary.

**Practical limits.** SLERP is defined for **exactly two** models. Merging three or more means chaining pairwise merges, and the result depends on the order — which is a genuine wart, and the reason the multi-model methods (TIES, DARE, task arithmetic) are formulated differently. Also, SLERP is usually applied **per-tensor**, not to the flattened model, and mergekit lets you set a different `t` per layer — a common recipe is a gradient from `t=0` at the bottom to `t=1` at the top, biasing early layers toward one parent and late layers toward the other on the theory that early layers hold general features and late layers hold task-specific behaviour. That theory is folklore, not established science; try it, measure it, do not assert it.

**⚠ Trap:** applying SLERP to *task vectors* rather than to *weights*. SLERP's whole justification is norm preservation of the thing being interpolated; task vectors already encode "distance from base," and spherically interpolating them means something different. Mergekit's `slerp` method operates on model weights. If you are working in task-vector space, use `task_arithmetic`, `ties` or `dare_ties`.

### Walk me through TIES-merging. What is the "interference" in the name?

Interference is the specific failure of naive averaging, and TIES names two distinct kinds.

**Redundancy interference.** A task vector is mostly noise. Most parameters changed by a tiny amount that contributes nothing to the task; a small fraction changed substantially and carries the behaviour. When you average `n` task vectors, the signal from each is divided by `n` while the accumulated noise from all `n` stays — signal-to-noise degrades linearly in the number of models merged.

**Sign interference.** For a given parameter, model A wants `+0.03` and model B wants `−0.028`. Averaging gives `+0.001` — you have not compromised between two capabilities, you have **cancelled both**. This is the mechanism behind "I merged four good models and got one mediocre one."

TIES fixes both in three steps, which is where the acronym comes from — **T**rim, **E**lect **S**ign, disjoint merge:

1. **Trim.** For each task vector independently, keep only the top-`k`% of parameters by magnitude (typically `k = 20`) and zero the rest. This is the noise floor removal, and the empirical finding is that discarding 80% of the delta costs almost nothing — most of it was never doing anything.
2. **Elect sign.** For each parameter position, sum the *magnitudes* of the positive contributions and the negative contributions across all trimmed task vectors, and elect whichever direction has more total mass. This produces a single agreed-upon sign vector `γ`.
3. **Disjoint merge.** For each parameter, average **only** the task vectors whose sign agrees with the elected sign, ignoring the dissenters entirely. Do not average dissenters in at zero weight — exclude them from the denominator too, so a parameter that only two of five models care about is the mean of those two, not of five.

Then `θ_merged = θ_pre + λ · τ_merged`, with `λ` swept as usual.

**📄 Paper:** Yadav et al. (2023), *TIES-Merging: Resolving Interference When Merging Models* — identified redundancy and sign conflict as the two mechanisms degrading naive averaging and fixed each with trimming and sign election.

**My rule:** TIES is the default for merging **three or more** models. For exactly two closely-related models SLERP is simpler and often equal. TIES's `k` (density) is the knob that matters more than `λ` — start at 20% and go up only if the merge underperforms every constituent, which is the signature of over-trimming.

### And DARE — dropping 90% of the delta and rescaling sounds like it should destroy the model. Why doesn't it?

Because the delta is enormously redundant, and because rescaling preserves the *expected* update. The dramatic version of the claim is what makes it memorable: you can randomly discard 90% or even 99% of a fine-tune's parameter changes and, after rescaling, recover essentially the fine-tuned model's behaviour.

The operation on a task vector `τ`, with drop rate `p`:

```
m ~ Bernoulli(1 − p)        elementwise mask
τ_DARE = (m ⊙ τ) / (1 − p)
```

The rescale is the load-bearing part, and it is exactly the inverted-dropout trick: `E[τ_DARE] = (1−p)·τ/(1−p) = τ`. Without the `1/(1−p)` factor the merged delta is systematically `(1−p)` times too small and the fine-tuning simply does not take effect.

Why it works: fine-tuning deltas are individually tiny and collectively redundant, so any random 10% subset, appropriately amplified, spans nearly the same functional direction as the whole. This is the same intuition as random projection preserving distances — you do not need every coordinate to preserve the geometry.

**📄 Paper:** Yu et al. (2024), *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch* — introduced DARE (Drop And REscale), showing extreme sparsification of delta parameters with minimal degradation and using it as a preprocessing step before merging.

**How it is actually used.** DARE is rarely the merge method; it is the **preprocessing step**. `dare_ties` in mergekit is: DARE-sparsify every task vector, then run TIES on the sparsified vectors. The reason this composes so well is that sign interference is a *collision* problem — two models fighting over the same parameter — and randomly sparsifying to 10% density makes collisions roughly 100× rarer. Two dense vectors conflict everywhere; two 10%-dense vectors overlap on ~1% of positions. DARE does not fix sign conflict directly; it makes sign conflict *rare*, which is often better.

**⚠ Trap:** DARE's guarantees are about the *expectation*, and expectation is not the same as the realized draw. A single random mask is one sample, and at high drop rates the variance is not negligible — some seeds produce a noticeably worse merge than others. If your merge quality moves when you change the seed, that is not a mystery, it is DARE variance, and the response is to lower `p` or to average over a few masks rather than to keep re-rolling until you like the eval number. **Re-rolling seeds until the benchmark improves is benchmark hacking, and it is how a lot of 2024-era open-model leaderboard results were produced.**

**On the caveat the paper itself raises:** DARE's tolerance for extreme sparsification depends on the deltas being small. Task vectors from *heavy* fine-tuning or continued pretraining have large-magnitude deltas, and there the technique degrades. Check the norm of your task vectors relative to the base weights before assuming 99% drop is safe.

### Model Breadcrumbs, Fisher merging, RegMean — when would I reach past TIES for one of these?

**Model Breadcrumbs** takes TIES's trimming idea and adds a second cut. TIES removes the small-magnitude tail as noise. Breadcrumbs removes **both** the small-magnitude tail *and* the largest-magnitude outliers, keeping a middle band. The reasoning is that the extreme outliers are frequently task-specific overfitting artifacts or numerical anomalies rather than transferable capability, and they dominate any magnitude-weighted merge. **📄 Paper:** Davari & Belilovsky (2024), *Model Breadcrumbs: Scaling Multi-Task Model Merging with Sparse Masks* — two-sided masking of task vectors, reported to be more robust than one-sided trimming as the number of merged tasks grows. I would reach for it when merging many (5+) task vectors and TIES is underperforming.

**Fisher-weighted averaging** replaces the uniform mean with an importance-weighted one. For each model, estimate the diagonal of the Fisher information — a per-parameter measure of how much the model's output distribution depends on that parameter, computable from squared gradients on a sample of that model's own data. Then weight each model's contribution per-parameter by its Fisher value:

```
θ_merged[j] = Σ_i F_i[j]·θ_i[j]  /  Σ_i F_i[j]
```

The intuition is precise and appealing: a parameter that matters enormously to model A and not at all to model B should be taken from A, not averaged. **📄 Paper:** Matena & Raffel (2022), *Merging Models with Fisher-Weighted Averaging* — replaced uniform weight averaging with a per-parameter posterior-precision weighting.

**RegMean** solves a different and, to a systems person, more satisfying problem: it treats merging as a **least-squares regression per linear layer**. For each linear layer, find merged weights `W*` minimizing the sum over models of `‖W* x_i − W_i x_i‖²`, where `x_i` are activations from model `i`'s own data. There is a closed-form solution requiring only the Gram matrices `X_iᵀX_i` of the input activations, which are cheap to collect and small. The merged layer is optimized to *reproduce each model's outputs on that model's inputs* — which is the thing you actually want, rather than a proxy for it. **📄 Paper:** Jin et al. (2023), *Dataless Knowledge Fusion by Merging Weights of Language Models* — closed-form per-layer regression merging using input Gram matrices.

**My decision rule, honestly stated:** TIES and DARE-TIES require nothing but the checkpoints, and that is why they dominate practice. Fisher and RegMean need **data and a forward pass per model**, which puts them in a different operational category — you now have a data dependency in your merge pipeline. Reach for them when (a) you have the data anyway, (b) the models are more dissimilar than usual so uniform weighting is clearly wrong, and (c) you have already tried TIES and it disappointed. In a product setting I have almost never gotten past step (c).

### What is evolutionary model merging and is mergekit something I'd actually run?

**Evolutionary merging** treats "which recipe?" as a black-box optimization problem rather than a thing you reason about. The merge configuration — per-layer interpolation coefficients, per-model weights, density parameters, and even which layers come from which model — is a vector of hyperparameters. Define a fitness function (your eval suite), and run an evolutionary strategy such as CMA-ES over that space, evaluating each candidate merge.

**📄 Paper:** Akiba et al. (Sakana AI, 2024), *Evolutionary Optimization of Model Merging Recipes* — searched both *parameter space* (per-layer merge coefficients) and *data-flow space* (which layers from which model, in what order), and produced capable Japanese-language math and vision-language models by merging existing open models without any gradient-based training.

The result that matters conceptually: they composed capabilities that **no constituent model had** — Japanese fluency from one lineage plus math ability from another — without a single gradient step. That is a real and slightly unsettling demonstration of how much capability is sitting latent in the open-weight ecosystem.

The cost is that each fitness evaluation is a full merge plus a full eval run. If a merge takes 5 minutes on CPU (out-of-core, disk-bound) and your eval takes 20 minutes on one GPU, a 200-candidate search is `200 × 25 min = 83 hours` of wall clock, parallelizable across GPUs. At $3/hr per H100 with 8-way parallelism: `200 × 0.33 hr × $3 = $200` of GPU time over ~10 hours. **That is genuinely cheap for the amount of capability being searched over** — which is the honest argument for the technique. What makes it dangerous is that a 200-candidate search against a fixed eval will overfit that eval. Hold out a suite the search never sees, and report on it.

**mergekit** is the tool, and yes, I would run it — it is the de-facto standard, it is a YAML config plus a CLI, and critically it is **out-of-core**: it streams tensors from disk, so you can merge two 70B models on a machine with modest RAM and enough disk. **📄 Paper:** Goddard et al. (2024), *Arcee's MergeKit: A Toolkit for Merging Large Language Models*.

```yaml
# dare_ties merge of three same-lineage fine-tunes
merge_method: dare_ties
base_model: meta-llama/Meta-Llama-3-8B
models:
  - model: org/llama3-8b-code-sft
    parameters: {weight: 0.4, density: 0.5}
  - model: org/llama3-8b-math-sft
    parameters: {weight: 0.3, density: 0.5}
  - model: org/llama3-8b-chat-sft
    parameters: {weight: 0.3, density: 0.5}
parameters:
  int8_mask: true
dtype: bfloat16
```

**📅 Volatile:** mergekit's method names and parameter schema change between releases — check the version's docs rather than copying a config from a blog post.

### Why can't I merge Llama-3-8B with Mistral-7B? Be specific about what breaks.

Two separate constraints, and it is worth separating them because only one is fundamental.

**The mechanical constraint: shapes.** Merging is elementwise arithmetic on named tensors. Llama-3-8B has `d_model = 4096`, 32 layers, FFN 14,336, vocabulary 128,256, 8 KV heads. Mistral-7B has `d_model = 4096`, 32 layers, FFN 14,336, vocabulary 32,000, 8 KV heads. Several tensors happen to match — and that near-match is exactly what tempts people. But the embedding and `lm_head` are `128256×4096` versus `32000×4096`. There is no elementwise sum of those. The merge tool errors, which is the good case.

**The fundamental constraint: shared ancestry.** Suppose you resolved the vocabulary problem. It still would not work, and here is why: **the meaning of a coordinate in weight space is arbitrary, and set by the pretraining run.** Neuron 1,742 in layer 12 of Llama-3 encodes some feature; neuron 1,742 in layer 12 of Mistral encodes an unrelated one. Both networks are invariant to permuting hidden units (with the corresponding permutation of the next layer's inputs), so there are `d!` equally-valid labelings and two independent pretraining runs pick different ones. Averaging aligns coordinate 1,742 with coordinate 1,742 — two unrelated features — and the result is not a compromise between two models, it is noise. There is no low-loss linear path between them; you would be interpolating straight through a loss barrier.

**The rule, stated as a test I would apply in review:** merging is valid only between models that share a **common ancestor checkpoint** and have not diverged past the point of linear mode connectivity. Two fine-tunes of Llama-3-8B: yes. Llama-3-8B and Llama-3.1-8B: risky — the continued pretraining between them may have moved them into different basins; test with a small merge and a general eval before believing it. Llama-3-8B and a Llama-3-8B that was continued-pretrained on 500B tokens of Chinese: probably no. Llama and Mistral: never.

The research direction that attacks the permutation problem is **weight-space alignment** — search for a permutation of one model's units that best matches the other before merging (the "Git Re-Basin" line of work). It has shown that permutation-aligned independently-trained networks can sometimes be merged. **📅 Volatile / contested:** I would describe this as an active research direction with results mostly at small vision-model scale, not as something to ship on an 8B LLM. Saying that accurately is better than either dismissing it or overclaiming it.

**🗣 Say this in the room:** "Merging requires a shared pretrained ancestor. It is not a shape constraint — even with matching shapes, two independent pretraining runs assign completely different meanings to the same coordinates, because the network is permutation-invariant in its hidden units. Averaging then aligns unrelated features. Same-lineage fine-tunes stay in one basin, which is the only reason any of this works."

### What is a frankenmerge, and why do they degrade?

A frankenmerge — mergekit calls the method `passthrough` — does not average anything. It **stacks layers**, building a new model by concatenating layer ranges from one or more models. Take layers 0–23 of model A, layers 8–31 of model B, stack them, and you have a deeper model with more parameters than either. This is how the 2023–24 open-model scene produced things like 120B models from pairs of 70Bs, and how depth-upscaling recipes built larger models from smaller ones by duplicating layer blocks.

Why it degrades, precisely: **every layer is trained to consume the specific distribution its predecessor produces.** A transformer layer's input is the residual stream at a particular depth, with a particular scale, a particular directional structure, and particular features already written into it. Layer 20 has learned to expect what layer 19 emits. Splice layer 8 of another model in front of it and layer 20 receives a residual stream from a different depth of a differently-trained network — wrong scale, wrong feature composition, wrong stage of processing. The layer still computes *something*, and because of residual connections the model does not collapse entirely, but the output degrades in characteristic ways: increased repetition, weakened long-range coherence, degraded instruction-following, and occasional language drift.

**The thing that makes it work when it works: continued pretraining.** Depth-upscaling approaches that succeeded duplicated layers and then **continued pretraining the stacked model** on a substantial token budget, giving the spliced layers a chance to re-adapt to their new inputs. That is not merging; that is using a merge as an initialization for training. The distinction is the whole answer.

**⚠ Trap — the benchmark story.** A large number of frankenmerges in 2023–24 posted excellent leaderboard numbers and were noticeably worse to actually use. The mechanisms: multiple-choice benchmarks like MMLU probe knowledge that survives splicing (the facts are still in the weights) while degrading generation quality that those benchmarks never measure; and heavy contamination in the open-model ecosystem meant merges of contaminated ancestors inherited the contamination. **If someone shows me a frankenmerge with a great benchmark table, the first thing I ask for is long-form generation samples and a head-to-head preference eval, not more benchmarks.**

**My position:** frankenmerging is a research toy and an initialization strategy, not a production technique. If I need a bigger model, I use a bigger model. If I need to combine capabilities, I use weight-space merging of same-lineage fine-tunes, or I route. I would push back hard on any proposal to ship a passthrough merge without continued pretraining and a generation-quality eval.

### How do you merge several LoRA adapters, and what's the gotcha?

The math first, because it is the gotcha. Adapter `i` contributes the weight delta `Δ_i = (α_i/r_i)·B_i A_i`. A weighted merge of `n` adapters is:

```
Δ_merged = Σ w_i · (α_i/r_i) · B_i A_i           ✅ sum the PRODUCTS
```

**⚠ Trap — the one people get wrong:** summing the *factors* instead of the products.

```
Δ_wrong = ( Σ w_i B_i ) ( Σ w_i A_i )            ❌
        = Σ_i w_i² B_i A_i  +  Σ_{i≠j} w_i w_j B_i A_j
```

Those cross terms `B_i A_j` — the up-projection of adapter `i` composed with the down-projection of adapter `j` — are meaningless products of unrelated learned subspaces. They are not a small error; they are half the terms. This is not a hypothetical: it is the natural thing to write if you think of `A` and `B` as independent parameters to average, and it produces a merge that trains-fine-then-behaves-strangely.

Given that, PEFT's `add_weighted_adapter` offers a few real combination strategies and each has a distinct cost:

- **`cat` (concatenation).** Stack the `A` matrices vertically and the `B` matrices horizontally. The resulting rank is `Σ r_i` and the result is **mathematically exact** — the concatenated product is exactly the sum of products. The cost is rank growth: merging five rank-16 adapters gives rank 80, so 5× the adapter parameters, 5× the adapter memory in a serving cache, and 5× the adapter FLOPs. For an 8B model that is `84 MB → 420 MB` per merged adapter, which matters in a multi-tenant cache.
- **`linear` / `svd`.** Sum the deltas correctly, then re-factor back down to a target rank via truncated SVD. This holds rank constant, which is what you want for serving, but the SVD truncation is lossy — you are projecting the sum of `n` rank-`r` matrices (rank up to `nr`) back into rank `r`. Expect real degradation when the adapters are dissimilar, and essentially none when they are near-parallel.
- **`ties` / `dare_ties`** applied in adapter space. Same interference logic as full-model merging, operating on the deltas.

**When would I actually do this?** Two cases. **Serving consolidation:** three adapters that are always applied together should be one adapter, saving cache slots and reducing batch adapter diversity — use `cat` if rank budget allows, `svd` if not. **Capability composition:** a domain adapter plus a format adapter. Here I am much more sceptical, and my honest recommendation is usually different: **if you want a model that does both things, train one adapter on the union of both datasets.** Joint training is strictly better than post-hoc merging when you have the data and a couple of GPU-hours, because the optimizer resolves the interference for you instead of you approximating it with a hyperparameter. Merging adapters is what you do when you *cannot* retrain — the data is gone, the tenants are separate, or the adapters came from different teams.

### How do you evaluate a merge? Assume the interviewer thinks "MMLU went up" is an answer.

"MMLU went up" is exactly the answer I would push back on, and the pushback is the substance of this question. A merge has a specific failure mode — it degrades things you were not looking at — so the eval has to be structured to catch degradation, not just to confirm improvement.

**The four-panel structure I require before any merge ships:**

**1. Per-domain suites, one per constituent, each with a baseline.** For every model that went into the merge, evaluate the merge on *that model's* domain, and compare against three references: the merge, the specialist, and the base. You are checking a specific claim — that the merge retained enough of each specialist. My gate: **the merge must stay within a stated tolerance of each specialist** (I use 3 points absolute as a starting negotiation), and it must beat the base on every constituent domain. If it beats base on only three of four domains, one constituent was crowded out and you should reweight, not ship.

**2. A general-capability guardrail suite that no merge weight was tuned against.** Broad knowledge, instruction-following, basic reasoning, and — this one is regularly forgotten — **format and tool-calling compliance**, because a merge that damages JSON-schema adherence breaks an agent pipeline silently while looking fine on knowledge benchmarks. This suite is *held out from the `λ`/density sweep*. If you tune merge coefficients against your guardrail, it stops being a guardrail.

**3. Generation-quality checks that multiple-choice benchmarks structurally cannot catch.** Merges degrade in a recognizable pattern and you should test for it directly: repetition rate on long generations (n-gram repeat fraction over 1,000-token outputs), unintended language switching (script/language-ID on a monolingual prompt set), perplexity on a general held-out corpus versus base, and a head-to-head preference eval on real prompts. **The frankenmerge lesson is that knowledge benchmarks and generation quality decouple**, and per-domain multiple-choice suites will happily tell you a broken model is fine.

**4. Safety and refusal calibration.** Merging can move refusal behaviour in either direction — a merge with a heavily safety-tuned constituent may over-refuse, and a merge diluting it may under-refuse. Run both an over-refusal suite and a harmful-request suite. This is not optional in an enterprise product.

**On the sweep discipline.** You will sweep `λ`, density, and per-layer coefficients. That is a search over a benchmark, and it overfits like any search. Split your evaluation data: a **development** set the sweep sees, and a **held-out** set it never sees, reported once at the end. Report the held-out number. If the dev and held-out numbers diverge by more than noise, your search overfit and the merge is worse than it looks.

**🗣 Say this in the room:** "I evaluate a merge on four panels: one suite per constituent domain with the specialist and the base as references, a general guardrail suite the merge coefficients were never tuned against, generation-quality checks — repetition, language drift, perplexity, format compliance — because multiple-choice benchmarks miss exactly the thing merges break, and a safety panel in both directions. The gate is: beats base on every constituent domain, within tolerance of each specialist, no regression on the held-out guardrail."

### You have eight domain-specific LoRAs and users whose queries span domains. Design the system.

Three architectures are on the table and the right answer depends on numbers I would ask for first: **what fraction of requests are single-domain, how separable are the domains at request time, and what is the latency budget?**

**Architecture A — route to a specialist.** Classify the request, load the matching adapter, serve. *Pros:* every request gets peak per-domain quality; adapters stay independently updatable and independently deletable; a new domain is a new adapter, not a retrain. *Cons:* you need a classifier, and its errors are your quality ceiling — a 5% misroute rate means 5% of traffic gets the wrong specialist, which is often worse than the base model. Cross-domain queries have no correct route. *Cost:* the classifier adds a hop; a small model or an embedding-similarity route costs single-digit milliseconds, a frontier-model router costs 200–500 ms and would blow most latency budgets.

**Architecture B — merge into one adapter.** TIES or DARE-TIES over the eight task vectors, one deployment, no routing. *Pros:* simplest serving path, zero routing latency, handles cross-domain queries natively because it is one model. *Cons:* you pay per-domain peak quality; updating one domain means re-running the merge and re-validating all eight; you lose per-tenant deletability.

**Architecture C — multi-adapter serving with per-request selection.** One base, all eight adapters resident, engine-level selection. This is Architecture A's quality with a much cheaper deployment story — you are not running eight fleets, you are running one with eight adapters at 84 MB each (`8 × 84 MB = 672 MB`, nothing). The routing problem remains.

**What I would actually build**, and the reasoning I would give:

Start from the traffic distribution. If **>80% of requests are cleanly single-domain**, build Architecture C with a cheap embedding-based router: embed the query, nearest-centroid over per-domain centroids computed from your training data, with a confidence threshold. Above threshold, route to the specialist; below threshold, fall back to the **merged** adapter, which you also keep resident as a ninth adapter. That hybrid handles the ambiguous tail correctly instead of forcing a bad route, and it costs one extra 84 MB adapter.

If **cross-domain queries are common**, the routing framing is wrong and I would argue for a different move entirely: **retrain one adapter on the union of the eight datasets.** Joint training resolves interference through the optimizer rather than through merge hyperparameters, and it is strictly better than merging when you still have the data. Eight datasets of 5k examples each is 40k examples — a 2-hour QLoRA run at roughly `$6` of H100 time. I would not spend a week tuning merge coefficients to approximate something a $6 training run does properly. **The rule: merge when you cannot retrain; retrain when you can.**

**How I'd know it works.** Per-domain suites for all eight, plus a deliberately-constructed **cross-domain** set (queries that genuinely span two domains), plus router accuracy measured independently — because router errors and model errors need separate attribution. Ship the router with a shadow mode first: route in the log, serve from the merged/joint model, and compare what the router *would* have done against outcomes before you let it control traffic.

### Give me the failure taxonomy. A LoRA fine-tuning project comes to you having produced a worse model — how do you triage?

**🔍 Failure taxonomy — a decision procedure**, ordered by how often each is the real cause and by how cheap it is to rule out.

**Stage 1 — is the adapter even being applied? (2 minutes.)** Run a prompt where base and adapter demonstrably differ. If outputs are identical, the adapter is not loaded, is loaded onto the wrong modules, or the merge silently no-opped. Also verify `print_trainable_parameters()` from the training run matches your hand-computed count — a wrong `target_modules` string (`"q_proj"` versus `"self_attn.q_proj"` conventions differ across integrations) silently adapts nothing or adapts less than you think.

**Stage 2 — is it a plumbing bug rather than a model problem? (30 minutes.)** Chat-template byte-diff between training and serving. Sampling parameters on both paths. Truncation of long inputs. QLoRA-trained-merged-into-fp16. These four account for the majority of "the fine-tune made it worse" reports and none of them is fixed by training.

**Stage 3 — did training actually work? (look at the curves.)** Flat loss with `grad_norm ≈ 0` means the gradient-checkpointing `requires_grad` bug or a frozen adapter. Loss spikes to `nan` means LR too high or an fp16 overflow (use bf16). Loss falls smoothly but eval degrades from epoch 1 means the data teaches something you did not intend. Loss falls and eval improves then degrades means overfitting — cut epochs.

**Stage 4 — is it the data? (this is usually the answer.)** Sample 50 training examples at random and read them, as raw decoded token sequences with the loss mask visualized. Look for: truncated completions (teaching the model to stop mid-sentence), the loss unmasked on prompts (teaching it to generate user turns), missing EOS (teaching it never to stop), format inconsistency across examples, and near-duplicates inflating apparent dataset size. **The rule I enforce: nobody reports a fine-tuning result without having read fifty raw examples from the tokenized dataset.** It takes twenty minutes and it finds problems that a week of hyperparameter search will not.

**Stage 5 — is it catastrophic forgetting on the axis you did not measure?** The adapter is better on the target task and worse on general instruction-following, format compliance, or refusal calibration, and your eval only covered the target task. Run a general guardrail suite. Fix by lowering LR, lowering epochs, lowering rank, or mixing 10–30% general instruction data into the fine-tuning set — the last of which is the most reliable and the most commonly skipped.

**Stage 6 — is the method wrong for the goal?** You are trying to inject facts (use retrieval), teach a new language (LoRA under-fits, needs full fine-tuning or a different base), or fix something a better prompt fixes. This is the stage where the correct output is "we should not be fine-tuning," and being willing to say that is the senior signal.

**🗣 Say this in the room:** "My triage order is: is the adapter applied, is the tokenization identical between train and serve, did the loss curve behave, have I read fifty raw training examples, did I measure general capability regression, and finally — should this have been retrieval instead of a fine-tune. Steps one through four are free and they resolve most of these. I do not touch a hyperparameter until they are all clean."

### Last thing — I've got two minutes. Give me your PEFT decision framework, top to bottom.

Here is the compressed version I would actually deliver, and I have rehearsed it in this order deliberately.

**First, should we be fine-tuning at all?** Fine-tuning is late on the escalation ladder. Before it: better context, better retrieval, better tool design, structured-output constraints, model choice. My preconditions for a fine-tune are more than roughly a thousand quality labeled examples, a stable eval that has been reproduced twice, and a *measured* plateau on the prompt-engineering path. If I am teaching facts rather than form, I go to retrieval and stop.

**Second, PEFT or full?** Write the memory equation: `16 bytes/param` static for mixed-precision AdamW. At 3B and below, full fine-tuning fits on one 80 GB card and is simply better — take it unless you need multi-tenant adapters. At 8B and above, or when you need many variants, LoRA. If the model does not fit in bf16, QLoRA — NF4 base, bf16 adapters, paged optimizer, gradient checkpointing — and 70B lands near 50 GB on one card at 2k context.

**Third, the configuration.** Rank 16, alpha 32, all seven linear projections, LR 1e-4 with cosine decay, 2 epochs, dropout 0.05. Target breadth over rank — that is the empirical result. rsLoRA above rank 64. I do not sweep alpha; I sweep LR and target modules.

**Fourth, serving.** One or two variants at high volume: merge, zero overhead. Many variants or per-tenant customization: hot-swap on a multiplexed base — one 8B base plus fifty 84 MB adapters is 20 GB and roughly 25× cheaper than fifty deployments. Adapter overhead is about 0.5% of decode bandwidth per *distinct* adapter in a batch, so cap and schedule for adapter locality. Never merge a QLoRA adapter into fp16 weights it was not trained against.

**Fifth, when to merge models.** Only from a shared ancestor. TIES or DARE-TIES for three or more, SLERP for two. And the rule that saves the most time: **if I still have the data, I retrain jointly instead of merging** — the optimizer resolves interference better than any merge coefficient I will find.

**Finally, how I know it worked.** Per-domain suites with base and specialist as references, a held-out general guardrail the sweep never touched, generation-quality checks for repetition and format compliance, and a canary prompt in the health check that proves the adapter is actually applied in production.

**🏋 Drill — the capstone, 3 hours, unaided, no autocomplete.** Take an open 7–8B base. (1) Write `LoRALinear` from scratch and verify step-0 logit equality against the base. (2) QLoRA fine-tune it on 2,000 examples of a task you define, with a held-out eval built *before* you train. (3) Produce the memory budget you predicted beforehand and the peak you actually measured, and explain any gap over 15%. (4) Merge the adapter, verify merged and unmerged outputs agree, then serve both merged and hot-swapped and measure decode throughput for each. (5) Report one number for target-task improvement and one for general-capability regression. **Pass criterion:** the predicted and measured memory agree within 15%, merged and unmerged evals agree within noise, and you can state — with arithmetic, not adjectives — what the fine-tune cost and what it bought.
