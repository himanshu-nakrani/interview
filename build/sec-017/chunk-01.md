### Before we talk about features or probes — explain the residual stream to me. Why does every interpretability paper start there?

The residual stream is the model's only shared memory, and the single most useful reframe is that **a transformer is not a pipeline of layers that transform a representation; it is a sequence of independent modules reading from and writing to one append-only bus.** Every attention head and every MLP reads the *entire* stream, computes something, and adds its result back. Nothing is ever overwritten. In backend terms: it is not a chain of RPC calls where each service receives its predecessor's output, it is 64 workers all subscribed to the same topic, each publishing a delta that everyone downstream also sees.

Mechanically, per layer: `x_{l+1} = x_l + Attn(LN(x_l)) + MLP(LN(x_l + Attn(LN(x_l))))` for a pre-norm model. The `+` is the whole story. It means the final residual vector at the last position is a *literal sum* of the token embedding, the positional contribution, and one term per attention head and one per MLP. For Llama-3-8B with `d_model = 4096` and 32 layers, that final vector is a sum of roughly `1 + 32 + 32 = 65` additive terms (more if you decompose attention per-head: 32 layers × 32 heads = 1024 head terms). Because the unembedding is also linear, you can push that sum through the LM head and get a per-component decomposition of the logits. That technique has a name — **direct logit attribution** — and it exists only because the stream is additive.

The consequence that makes interpretability tractable: composition between layers happens *through* the stream, so "layer 5 head 3 talks to layer 9 head 7" is a well-defined claim you can test by deleting the message. LayerNorm is the only thing that breaks strict linearity, and since it is approximately a per-position scalar rescale, the decomposition stays approximately valid if you freeze the LN scale when you do the math.

**⚠ Trap:** thinking the stream has a fixed semantic basis, i.e. "dimension 812 means past tense." The basis is arbitrary — it is whatever the initialization and optimizer landed on. Directions are meaningful; individual coordinates usually are not. Every method in this section exists to find the meaningful directions.

**📐 Numbers you must know:** the stream is a *bandwidth-limited* channel. Llama-3-8B carries `d_model = 4096` floats per position, and the model plainly wants to represent far more than 4096 distinct things about a token. That mismatch is not a bug you can engineer away; it is the direct cause of superposition, and it is why "one neuron, one concept" was never going to be true.

**🗣 Say this in the room:** "The residual stream is a shared additive bus with `d_model` slots that every layer reads in full and writes deltas to. That additivity is what makes attribution possible at all — the final logits are a sum over components, so I can ask which component contributed the answer."

### Walk me through the logit lens. What is it measuring, and where does it lie to you?

The logit lens is the cheapest interpretability tool that exists and it is the one I reach for first in a real debugging session. The intuition: since every layer writes into the same stream that the LM head eventually reads, you can apply the LM head *early* — at layer 12 instead of layer 32 — and ask "if the model had to answer right now, what would it say?" You get a per-layer trajectory of the model's evolving guess. When a model confidently emits the wrong token, the logit lens tells you at which depth the wrong token became the argmax, which converts "the model is wrong" into "the decision was made in layers 14–18," which is a localizable claim.

```python
@torch.no_grad()
def logit_lens(model, tokenizer, prompt, k=5):
    out = model(**tokenizer(prompt, return_tensors="pt").to(model.device),
                output_hidden_states=True)
    head, norm = model.lm_head, model.model.norm   # final RMSNorm before the head
    for layer, h in enumerate(out.hidden_states):  # hidden_states[0] = embeddings
        logits = head(norm(h[0, -1]))              # [V]
        top = logits.softmax(-1).topk(k)
        print(layer, [(tokenizer.decode(i), round(p.item(), 3))
                      for p, i in zip(top.values, top.indices)])
```

The one detail people get wrong: you must apply the *final* norm before the head, not the layer-local one, because the head was trained to consume normalized vectors. Skip it and every layer looks like noise.

Where it lies: the logit lens assumes intermediate layers write in the same basis the final unembedding reads. That holds decently for GPT-2-style models and degrades badly on others — you get long stretches of layers where the top prediction is garbage or is stuck on the input token, not because the model is confused but because the readout is mismatched. The **tuned lens** fixes this by learning a per-layer affine map `A_l h_l + b_l` trained to minimize KL against the model's own final distribution, so each layer gets a calibrated translator into the output basis.

**📄 Paper:** the logit lens originates as a 2020 LessWrong post by nostalgebraist, not a paper — say that plainly, it signals you actually read the source. **Belrose et al. (2023), "Eliciting Latent Predictions from Transformers with the Tuned Lens"** replaced it with the learned-affine-probe version and showed the tuned lens is both more predictive and less prone to the representational-drift artifact.

**⚠ Trap:** reading the logit lens as "the model believes X at layer 12." It shows what the *unembedding* would output given the partial stream — a projection, not a belief. A direction can be present and simply not yet rotated into readout position. Treat it as a hypothesis generator that you then confirm with a causal intervention, never as evidence on its own.

### What is an induction head, and why do people claim it explains in-context learning?

An induction head is the smallest genuinely mechanistic result in the field, and it is worth being able to draw on a whiteboard. The behavior it implements is: given a sequence containing `... A B ... A`, predict `B`. It is a learned, attention-implemented pattern-completion primitive — "last time I saw this token, what came next?"

It requires two layers to compose, and the composition is the beautiful part. In layer `L`, a **previous-token head** attends from each position to the position immediately before it and copies that token's identity into the residual stream. So position `i` now carries a tag: "the token before me was X." In some later layer `L' > L`, the **induction head** forms its query from the *current* token `A`, and forms its keys from that previous-token tag — so it matches "current token A" against "positions whose predecessor was A." That lands attention on the position right after the earlier `A`, which holds `B`, and the head's OV circuit copies `B` into the output direction. Query-key composition through the residual stream is what makes it work; it is a two-hop program running over a shared bus.

**📄 Paper:** **Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits"** introduced the QK/OV decomposition and found induction heads in two-layer attention-only models. **Olsson et al. (2022), "In-Context Learning and Induction Heads"** is the load-bearing one: induction heads form during a narrow window of training that coincides with a visible bump in the loss curve, and that same window is when in-context learning ability appears. Ablating induction heads sharply degrades in-context learning. It is correlational-plus-ablational evidence, not proof, and I would characterize it that way in a room.

Why you care as an applied engineer: this is the mechanism underneath few-shot prompting, format mimicry, and the reason a model will faithfully continue a made-up pattern you established three turns ago. It is also the mechanism behind a class of prompt injection — an attacker who establishes a strong repeated pattern in retrieved content gets copying for free, because the copying circuit does not care whether the pattern came from the system prompt or a scraped web page.

**🗣 Say this in the room:** "Induction heads are a two-layer circuit — a previous-token head tags each position with its predecessor, and a later head uses that tag to attend to the continuation of the earlier occurrence and copy it. Their formation coincides with the emergence of in-context learning, which is the strongest mechanistic story we have for why few-shot prompting works at all."

### Explain superposition. Why does everyone insist that neurons are not features?

Because the model has more things to say than it has dimensions to say them in, and it solves that the way any compression scheme would.

Start with the counting argument. A model wants to represent, conservatively, hundreds of thousands of distinct concepts: "this is a Python identifier," "this is a legal citation," "the subject is plural," "we are inside a JSON string." It has `d_model = 4096` orthogonal directions available. If you insist on one concept per orthogonal direction you get 4096 concepts, full stop. But if you allow *almost*-orthogonal directions, the capacity explodes — in high dimensions you can pack exponentially many vectors with pairwise cosine similarity below a small ε. The cost is interference: any two co-active features bleed into each other.

The reason the model gets away with it is **sparsity**. Almost every feature is off for almost every token. If only ~50 of your 100,000 concepts are active at once, the expected interference is tiny, and the ReLU/GELU nonlinearity in the MLP acts as a threshold that cleans up the small cross-talk. So superposition is a rational compression strategy under sparse activation, not pathology.

The consequence is **polysemanticity**: an individual neuron fires on an unrelated-looking grab bag — academic citations, HTTP headers, and Korean text — because it is a coordinate in a basis where features were packed at oblique angles, not a feature detector. This is why "we found the DNA neuron" results from the 2010s-2020s mostly did not replicate as causal claims.

**📄 Paper:** **Elhage et al. (2022), "Toy Models of Superposition"** is the reference. They train tiny autoencoders on synthetic features with controllable sparsity and show the phase transition directly: at low sparsity the model learns an orthogonal basis and neurons are monosemantic; as sparsity increases it starts packing features into geometric configurations (pentagons, digons) and neurons go polysemantic. That paper is the intellectual justification for every sparse autoencoder that came after it.

**⚠ Trap:** the max-activating-examples fallacy. You take neuron 1423, run 10M tokens through, print the top 20 activating snippets, they all mention DNA, and you write "neuron 1423 is the DNA neuron" in a doc. You have measured the top 0.0002% of the activation distribution. The other 99.9998% may be doing something else entirely, and you have shown zero causal evidence that the model *uses* this neuron for DNA. The fix is a two-sided check: sample across the whole activation range, and ablate the neuron to see if DNA-related predictions actually move.

**🗣 Say this in the room:** "Features are directions, not coordinates. The model packs far more features than dimensions using near-orthogonality, which works because features are sparse — and that's exactly why individual neurons look polysemantic and why we need a learned dictionary to recover the basis."

### Implement a sparse autoencoder for me and tell me what you'd measure to know it worked.

The mental model: an SAE is a **learned overcomplete dictionary that undoes superposition**. You take activations `x ∈ R^d` from one site (residual stream at layer `l`, or MLP output), and learn to write `x ≈ Σ_i a_i(x) · d_i` where the dictionary has far more atoms than `d` and only a handful of `a_i` are nonzero per token. It is compressed sensing applied to a model's internals: the model compressed many sparse features into `d` dimensions, and you are decompressing.

```python
class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=32):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) / d_model**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(self.W_enc.data.T.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):                       # x: [B, d_model]
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        v, i = pre.topk(self.k, dim=-1)         # hard sparsity, no L1 needed
        acts = torch.zeros_like(pre).scatter_(-1, i, F.relu(v))
        return acts @ self.W_dec + self.b_dec, acts

def loss_fn(sae, x):
    xhat, acts = sae(x)
    return ((xhat - x) ** 2).sum(-1).mean()     # TopK: reconstruction only
```

Two variants matter. The original **ReLU + L1** formulation (`loss = MSE + λ·‖a‖₁`) needs unit-normalized decoder rows, otherwise the model cheats by shrinking activations and growing decoder norms; it also suffers systematic activation shrinkage because L1 penalizes magnitude as well as count. **TopK** SAEs sidestep both by making sparsity a hard constraint, and give you `L0 = k` exactly, which is enormously convenient for comparing runs. JumpReLU and Gated SAEs are the other main line, decoupling the "is it on" decision from the "how much" magnitude.

**📄 Paper:** **Bricken et al. (2023), "Towards Monosemanticity"** and **Cunningham et al. (2023)** established that sparse dictionary learning recovers interpretable features from polysemantic activations, replacing neuron-level analysis. **Gao et al. (2024), "Scaling and Evaluating Sparse Autoencoders"** introduced the TopK formulation and the systematic scaling study. **Templeton et al. (2024), "Scaling Monosemanticity"** scaled it to a production model (Claude 3 Sonnet) with dictionaries up to tens of millions of features.

What I measure, in this order. **L0** — mean active features per token; the usable band is roughly 20–100, and anything under ~10 is losing information while anything over a few hundred is not really sparse. **Fraction of variance unexplained** on held-out activations. And the metric that actually decides whether the SAE is honest: **loss recovered** — splice `x̂` back into the model in place of `x`, run the rest of the forward pass, and measure the cross-entropy increase relative to the clean model, normalized against the CE you get from zero-ablating that site. If you recover 85% of the loss, then 15% of what the model was doing at that site is not in your dictionary at all.

**📐 Numbers you must know:** a 16× dictionary on `d_model = 4096` is `d_sae = 65,536` features; encoder plus decoder is `2 × 4096 × 65,536 ≈ 537M` parameters, which at bf16 is `537e6 × 2 = 1.07 GB` per hooked site — larger than most people guess, and the reason you hook one site rather than all 32 layers. The usable sparsity band is `L0 ≈ 20–100` and the honesty metric is loss-recovered above ~85%; below that you are analyzing a dictionary that is missing a sixth of the computation.

**⚠ Trap:** **dead features.** In a 65k-wide dictionary you will routinely find 10–50% of features that never fire after the first few thousand steps — they are burning parameters and quietly capping your reconstruction. You need an explicit mitigation: neuron resampling (reinitialize dead atoms toward high-residual examples) or an auxiliary loss that asks the top-k *dead* features to reconstruct the residual error. Reporting dictionary width without reporting the alive-feature count is the single most common way SAE results get overstated.

### What does it actually mean to say you "found a feature"? How do you know it's real and not an artifact of your dictionary?

This is the question that separates people who have run SAEs from people who have read about them, and I would answer it with an operational definition rather than a philosophical one. A feature is real to me when it passes three tests, and the third is non-negotiable.

**Test one, specificity.** Sample activations across the *entire* range, not just the top. A real feature has a coherent interpretation at the 20th percentile of its nonzero activations, not only at the 99.9th. Auto-interp helps here: have an LLM write an explanation from a sample of activating contexts, then have a second LLM predict activations on held-out text from the explanation alone, and score the correlation. This is a useful screen and a terrible ground truth — auto-interp scores are inflated by features that correlate with easy surface cues like "is this token a proper noun."

**Test two, sensitivity.** Does it fire on *all* the instances of the concept, or only some? A feature that fires on 30% of Golden Gate Bridge mentions has been split.

**Test three, causality.** Clamp the feature's activation high (or ablate it) via its decoder direction during a forward pass and check whether the output moves in the direction the interpretation predicts. If clamping "legal citation formatting" high does not make the model start emitting legal citations, you have found a correlate, not a mechanism.

Now the artifacts you have to name. **Feature splitting:** as you widen the dictionary, one feature reliably fractures into several finer ones — "birds" becomes "raptors," "waterfowl," "birds as metaphor." This means the feature set is a function of your hyperparameters, not a discovered natural kind, and it is the strongest argument that SAE features are a *useful coordinate system* rather than the model's true ontology. **Feature absorption:** a general feature stops firing on cases covered by a more specific feature, so "starts with the letter E" mysteriously fails on "elephant" because a dedicated elephant feature absorbed it — which wrecks any attempt to read the feature set as a clean taxonomy. **Dead and ultra-low-frequency features** pad your width without contributing.

**⚠ Trap:** the interpretability illusion. **Bolukbasi et al. (2021)** showed for BERT that plausible-looking neuron interpretations derived from max-activating examples do not survive testing on other corpora. The same trap applies at the feature level: a dictionary trained on web text will produce features that look crisp on web text and fall apart on your enterprise contracts corpus. If you are going to use SAE features for production monitoring, train or at least validate the dictionary on *your* distribution.

**🗣 Say this in the room:** "Operationally a feature is a direction whose activation I can predict from an interpretation, and whose ablation or amplification causally changes the output in the predicted way. Without the causal leg it's just a correlate — and given feature splitting, I'd describe any specific dictionary as a useful basis rather than the model's ground-truth ontology."

### Transcoders and crosscoders — what problem do those solve that a plain SAE doesn't?

Plain SAEs have a structural limitation for circuit work: they decompose an activation *at a point*, but circuits are about *paths*, and every MLP sits in the middle of a path as an opaque nonlinearity. If you have an SAE on the layer-8 residual and another on the layer-9 residual, you cannot cleanly say how a layer-8 feature caused a layer-9 feature, because the MLP in between is not linear and does not decompose.

A **transcoder** attacks exactly that. Instead of learning `x ≈ Dec(Enc(x))` at one site, you learn a sparse dictionary that maps *MLP input to MLP output*: `MLP_out ≈ Dec(Enc(MLP_in))` with a sparse hidden code. You are replacing the MLP with an interpretable sparse approximation. Now the path from an input feature to an output feature is a single sparse linear hop you can trace and attribute through, and you can do circuit analysis that passes *through* MLPs rather than stopping at them.

**📄 Paper:** **Dunefsky et al. (2024), "Transcoders Find Interpretable LLM Feature Circuits"** established this as an alternative to SAEs specifically for circuit tracing. Anthropic's 2025 circuit-tracing work extends it to **cross-layer transcoders**, where a feature's output is written into several downstream layers at once — which matters because real models spread a single computation across adjacent layers rather than confining it to one.

A **crosscoder** generalizes in the other direction: one dictionary that reads activations from multiple layers simultaneously (or from multiple *models* simultaneously), producing features shared across those sites. The two applications I care about are (a) collapsing the redundancy where the "same" feature appears in a dozen consecutive layers' SAEs, and (b) **model diffing** — train a crosscoder jointly on a base model and its fine-tune, and the features that are present in one and absent in the other are a direct, structured answer to "what did my fine-tune actually change?"

That second use is the one that shows up in an applied interview, because it is the only interpretability technique with a direct answer to a question engineers ask every week. You fine-tuned Llama on 40k support tickets, and evals moved in a way you cannot explain. Weight diffs tell you *where* parameters moved; a crosscoder tells you *what representations* appeared or vanished.

**⚠ Trap:** treating "feature exists in fine-tune but not base" as proof the fine-tune created it. It may exist in the base at lower magnitude and simply fall below your sparsity threshold, or the crosscoder may have allocated capacity asymmetrically. The control is to check the base model's activation on the crosscoder's shared decoder direction directly, rather than relying on the feature being selected.

**📅 Volatile:** this whole subfield is moving fast enough that the relative standing of SAEs, transcoders, crosscoders and their successors will likely have shifted by the time you interview. Learn the *problem each one solves* — point decomposition, path decomposition, cross-site decomposition — and verify the current state of the art before your loop.

### What is the linear representation hypothesis, and what's the evidence against it?

The hypothesis: high-level concepts are represented as **directions** in activation space, and the model's computation reads them by projection. If true, then a concept has a vector, concepts compose additively, and you can measure a concept with a linear probe and control it by adding a vector. Almost every practical technique in this section — probes, steering vectors, difference-of-means directions, SAE decoder columns, the refusal direction — is downstream of this hypothesis being approximately true.

The evidence for it is substantial and mostly *causal*, which is what makes it credible rather than numerological. Difference-of-means vectors computed from two sets of prompts, when added to the residual stream, change behavior in the predicted direction. Ablating a single direction removes a behavior across many prompts. Linear probes recover sentiment, truth-ish judgments, board state in game-playing models, and spatial and temporal attributes with high accuracy while nonlinear probes add surprisingly little. SAEs, which *assume* linearity in their decoder, produce features that survive causal tests.

The evidence against it, which you should volunteer because volunteering it is what makes you sound like a practitioner: some features are demonstrably **not** one-dimensional. Cyclic quantities — days of the week, months, clock positions — appear to be represented on circles or higher-dimensional manifolds rather than lines, which is exactly what you would expect for a quantity where "distance" wraps around. **Engels et al. (2024), "Not All Language Model Features Are Linear"** is the reference for multi-dimensional features. Separately, the hypothesis is basis- and normalization-sensitive: "a direction" only means something relative to a choice of where you measure and whether you have accounted for LayerNorm's rescaling, and sloppiness there produces directions that look real and do not transfer.

**📄 Paper:** **Park et al. (2023), "The Linear Representation Hypothesis and the Geometry of Large Language Models"** gives the formal treatment, including the observation that the right inner product for measuring these directions is not the naive Euclidean one — the causal geometry differs from the representational geometry.

My working stance, and I would say it this way in a room: linearity is an excellent engineering approximation that buys enormously cheap control surfaces, and it is not a law. When a steering vector behaves non-monotonically in its coefficient, or a probe that hits 0.97 AUROC in-distribution collapses to 0.6 out of distribution, the linear assumption is where I look first.

### Derive a steering vector for me and tell me when you'd ship one instead of a prompt or a fine-tune.

The mental model: if a concept is a direction, then adding a scalar multiple of that direction to the residual stream at inference time is a **runtime configuration knob on the model's internal state** — the cheapest control surface that exists, orders of magnitude cheaper than a fine-tune and more reliable than a politely-worded instruction.

The standard construction is contrastive difference-of-means. Build paired prompts that differ only in the attribute you want, run both, take activations at one layer, average each set, subtract.

```python
@torch.no_grad()
def steering_vector(model, tok, pos_prompts, neg_prompts, layer, pos_idx=-1):
    def mean_act(prompts):
        acc = 0
        for p in prompts:
            out = model(**tok(p, return_tensors="pt").to(model.device),
                        output_hidden_states=True)
            acc = acc + out.hidden_states[layer][0, pos_idx]
        return acc / len(prompts)
    return mean_act(pos_prompts) - mean_act(neg_prompts)   # [d_model]

def make_hook(v, coeff):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, :, :] = h + coeff * v                 # add to every position
        return out
    return hook

handle = model.model.layers[layer].register_forward_hook(make_hook(v, 2.0))
```

Three implementation details carry all the risk. **Layer choice:** mid-depth, roughly 40–70% through the stack, is where behavioral abstractions live; steering at layer 1 mostly perturbs token identity and steering at the last layer mostly perturbs style. Sweep it, do not guess. **Coefficient:** the behavior/capability trade-off is not monotone — you get a band where the trait appears and coherence is intact, and above it the model degenerates into obsessive repetition. **Position scope:** adding at all positions (prompt included) versus only at generated positions is a real product decision, and I will come back to why it interacts with your prefix cache.

**📄 Paper:** **Turner et al. (2023)** introduced activation addition (ActAdd) as inference-time steering; **Rimsky et al. (2024), "Steering Llama 2 via Contrastive Activation Addition"** systematized the difference-of-means construction over paired multiple-choice data and evaluated it across behaviors. The famous public demo, **Golden Gate Claude (Anthropic, 2024)**, is a different mechanism worth distinguishing: it clamped a specific *SAE feature* to a high value rather than adding a difference-of-means vector, which is why the effect was so semantically precise — and why the model would insist it *was* the bridge, since the feature was participating in self-description too.

When I would ship one: the behavior is a continuous dial (verbosity, formality, hedging, refusal-strictness), you serve your own open weights, and you want it adjustable per request without maintaining N fine-tuned checkpoints. When I would not: you are on a hosted API (no hook surface exists), the behavior is knowledge rather than style, or the required coefficient is high enough to measurably move your capability evals.

**⚠ Trap:** validating a steering vector only on the steered behavior. You will always be able to find a coefficient that makes the model more concise. The question nobody asks until the incident is what it did to everything else. The rule I enforce in review: any steering change ships with a held-out capability suite (a code eval, a retrieval-QA eval, an instruction-following eval) run at the exact coefficient you plan to deploy, plus the perplexity delta on a neutral corpus. If perplexity on neutral text rises more than a few percent, the coefficient is too high regardless of how good the demo looks.

### Design a clean activation-patching experiment for me. What's the counterfactual, what's the metric, and which direction do you patch?

Activation patching is the field's causal workhorse and it is conceptually just a controlled experiment: run two forward passes, copy one internal value from one run into the other, and measure how much of the behavior transfers. It converts correlational "this head attends to the right token" claims into causal "deleting this head's message changes the answer" claims.

A clean experiment has four parts, and getting any one wrong invalidates the result.

**One, the counterfactual pair.** Two prompts that are minimally different and differ in exactly the variable of interest, with the same token count and ideally the same token positions. For factual recall: `"The Eiffel Tower is in the city of"` versus `"The Colosseum is in the city of"`. The original causal-tracing work corrupted the subject with Gaussian noise on the embeddings instead, and I would push back on that in review — noised embeddings are off-distribution, so the model's response to them tells you partly about its behavior on garbage input rather than about the circuit. Symmetric prompt pairs are strictly better when you can construct them.

**Two, the metric.** Use a **logit difference** — `logit(correct) − logit(counterfactual_answer)` — not probability, not accuracy, not loss. Logit difference is linear in the residual stream, so it composes with direct logit attribution; probability is squashed by softmax so large upstream effects look small near saturation; accuracy is a threshold that throws away all your resolution. Then normalize: report the fraction of the clean-to-corrupt gap that the patch restores, so 0 means no effect and 1 means full restoration.

**Three, the direction, and these answer different questions.** *Denoising* (patch a clean activation into the corrupted run) finds components **sufficient** to restore the behavior. *Noising* (patch a corrupted activation into the clean run) finds components **necessary** for it. They routinely disagree, and the disagreement is informative rather than an error: a component can be sufficient on its own and unnecessary because a backup component would cover for it.

**Four, the granularity.** Patch at increasing resolution — whole residual stream at layer `l` and position `p` first (a `layers × positions` heatmap is the standard first plot), then attention output vs MLP output, then individual heads, then head-and-position, then specific subspaces.

**⚠ Trap:** **self-repair / the hydra effect.** Ablate an important attention head and the effect is often far smaller than you expect, because downstream "backup" heads increase their contribution to compensate. This means single-component ablation systematically *understates* importance, and a naive sweep will tell you nothing is important. The mitigation is to ablate sets jointly, and to always report both directions.

**⚠ Trap (the second one):** the subspace illusion. Patching along a learned subspace can produce the behavioral change you predicted through a path that has nothing to do with your hypothesis — the intervention activates dormant pathways rather than manipulating the feature you thought you found. **Makelov et al. (2023)** documented this directly for subspace patching. The control is to check that your intervention has the predicted effect on *intermediate* quantities, not just on the final logit.

**🏋 Drill:** on GPT-2 small, build a 10-pair counterfactual dataset for a simple factual-recall task, produce the `layers × positions` denoising heatmap using normalized logit difference, and identify the two sites with the largest restoration. Pass criterion: 45 minutes, unaided, and you can state in one sentence why you used logit difference rather than probability.

### Patching every head at every position doesn't scale. What do you do instead?

You linearize. The mental model: patching asks "what happens if I set activation `a` to a different value?" and for small changes the answer is well approximated by a first-order Taylor expansion — which means one backward pass gives you the estimated effect of patching *every* activation simultaneously, instead of one forward pass per activation.

Concretely, **attribution patching** estimates the patching effect on metric `M` as `(a_clean − a_corrupt) · ∇_a M`, with the gradient taken at one of the two runs. Two forward passes and one backward pass gets you a score for every node in the graph. You then rank, keep the top candidates, and *verify those with real patching*. It is a candidate generator, not a replacement.

**💰 Math, and this is the number that justifies the technique:** GPT-2 small has 12 layers × 12 heads = 144 attention heads. A head-by-position sweep over a 30-token prompt is `144 × 30 = 4,320` forward passes per prompt pair; over a 100-pair dataset that is 432,000 forwards. At ~8 ms each on an A100 that is roughly `432,000 × 0.008 s ≈ 3,456 s ≈ 58 minutes` — annoying but survivable. Now scale to Llama-3-70B: 80 layers × 64 heads = 5,120 heads, a 500-token prompt, and a forward pass costing ~250 ms. That is `5,120 × 500 = 2.56M` forwards per pair, `2.56M × 0.25 s ≈ 178 hours` for a *single* prompt pair. Attribution patching turns that into three passes — under a second — at the cost of approximation error. There is no version of exhaustive patching that works at frontier scale; this is why the approximation exists.

**📄 Paper:** attribution patching was introduced by Neel Nanda in a 2023 write-up rather than a formal paper; **Syed et al. (2023)** developed it into edge attribution patching for automated circuit discovery and showed it matches or beats the earlier search-based approach, **ACDC (Conmy et al., 2023)**, at a fraction of the compute. **Marks et al. (2024), "Sparse Feature Circuits"** applies the same linearization over SAE features rather than raw components, which is what gives you circuits described in human-readable features instead of head indices.

**⚠ Trap:** trusting the linear approximation where the effect is largest. The Taylor expansion is worst exactly where the function is most nonlinear — a saturated attention softmax, or a component whose ablation moves the logit difference by several nats. So attribution patching is *least* accurate on the components you most care about. The discipline: use it to prune 5,000 candidates to 50, then run true patching on those 50 and report only the verified numbers. I have seen a circuit paper's headline claim evaporate because nobody re-verified the top edges.

### Walk me through the IOI circuit. Why is it the canonical example?

Indirect Object Identification is the field's reference result because it is the first end-to-end, human-legible circuit for a nontrivial behavior in a real (if small) language model, and because the *surprises* in it generalized.

The task: given `"When John and Mary went to the store, John gave a drink to"`, the model should output `" Mary"`. Solving it requires identifying the duplicated name (John appears twice), suppressing it, and copying the other name.

**What GPT-2 small actually does, as decomposed by Wang et al. (2022):** early **duplicate-token heads** detect that `John` at the second position matches `John` earlier and write "this name is a duplicate" into the stream; **previous-token** and **induction** heads provide the positional machinery for that matching. Their output feeds **S-inhibition heads**, whose job is to write a signal that *suppresses attention to the duplicated name*. That suppression signal modifies the queries of the final group: **name-mover heads**, which attend from the final position to the name tokens and copy whatever they attend to into the output logits via their OV circuit. Because S-inhibition has knocked down the duplicate, the name movers land on `Mary` and copy it.

Two findings from that paper matter more than the circuit itself. **Negative name-mover heads** exist — heads that attend to the correct answer and write it *down*, actively reducing the correct logit. That is deeply counterintuitive and is now understood as part of a broader calibration/hedging phenomenon. And **backup name-mover heads**: ablate the primary name movers, and heads that were doing nothing relevant step in and perform the copying instead. That is the self-repair result in its original habitat, and it is the reason I will not accept a single-head ablation as evidence of unimportance.

**📄 Paper:** **Wang et al. (2022), "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small"** — 26 heads across seven functional classes, identified with path patching and validated by ablation. It replaced hand-wavy attention-map storytelling with a falsifiable, intervention-validated account.

**⚠ Trap:** citing IOI as evidence that we can explain model behavior in general. It is one narrow syntactic behavior in a 124M-parameter model, it took a team months, and later work showed the circuit is not as clean or as complete as the original presentation suggested — the "circuit" explains most but not all of the behavior, and components outside it contribute. Present it as a proof of concept for the *methodology*, not as a template for explaining GPT-4.

**🗣 Say this in the room:** "IOI is the field's canonical worked example — duplicate detection feeding inhibition heads that gate name-mover heads that copy the non-duplicated name. The durable lessons for me are the methodology (path patching plus ablation validation) and the discovery of backup heads, which means I treat any single-component ablation result as a lower bound on importance."
