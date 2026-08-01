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
