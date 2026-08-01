### Explain the rotation-based methods — QuaRot and SpinQuant. Why would multiplying by a random orthogonal matrix make a model easier to quantize?

This is my favourite result in the whole area because it is both surprising and, once you see it, obvious. The intuition: outliers are a *basis-dependent* phenomenon. A vector whose energy is concentrated in three of 4,096 coordinates is a terrible thing to quantize, but the vector's *norm* is unchanged by rotation, and after a well-chosen rotation that same energy is spread evenly across all 4,096 coordinates. Same information, same norm, no outliers. The max-to-mean ratio collapses from 80× toward √(something small), because a random rotation of a sparse vector produces something close to isotropic — this is the "incoherence" idea borrowed from compressed sensing.

The engineering trick that makes it free: transformers are full of places where you can insert `Q` and `Qᵀ` and have them cancel. For a linear layer, `XW = (XQ)(QᵀW)`. If you rotate the residual stream by `Q` and rotate every weight matrix that reads from it by `Qᵀ` and every matrix that writes to it by `Q`, the network computes exactly the same function. The rotations on weights are absorbed offline into the weights themselves — zero runtime cost. Only the rotations that must happen at runtime (inside the attention block, around the KV cache, before the down-projection) cost anything, and those use **Hadamard matrices**, whose matrix-vector product is `O(n log n)` via the fast Walsh–Hadamard transform instead of `O(n²)`.

**📄 Paper:** Ashkboos, Mohtashami, Croci, Li, Jaggi, Alistarh, Hoefler et al. (2024), "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" — used randomized Hadamard rotations to make weights, activations *and* the KV cache simultaneously 4-bit quantizable end-to-end, which weight-only methods could not do.

SpinQuant (Meta, 2024) is the natural follow-up: rather than a *random* Hadamard, **learn** the rotation. The rotation matrices are optimised on the orthogonal (Stiefel) manifold against a small calibration objective, so `Q` is chosen to minimise the actual quantization error rather than being merely incoherent on average. It reports consistently better W4A4 results than random rotations, at the cost of a short optimisation run.

Why this matters strategically: rotations are the reason **W4A4 stopped being a fantasy**. SmoothQuant migrates outliers between operands; rotations *destroy* them. LayerNorm has to be RMSNorm-style for the fold to be exact (the mean-subtraction in classic LayerNorm doesn't commute with a general rotation cleanly, which is why implementations fuse the LN affine into the adjacent linear first), and RoPE interacts with the head-dimension rotation, so the head-internal Hadamards are applied on the K/V after RoPE rather than before.

**⚠ Trap:** assuming rotation is free because "orthogonal matrices preserve norms." The online Hadamards inside the attention block are real FLOPs and real kernel launches, and on short sequences at small batch they can eat a meaningful fraction of the win. Also, a rotated checkpoint is not interchangeable — the model file now encodes a specific rotation, and any tool that assumes canonical weight layout (a LoRA adapter trained on the unrotated model, a layer-wise interpretability probe, a merge script) will be silently wrong.

### What is HQQ and when would you pick it over GPTQ or AWQ?

HQQ (Half-Quadratic Quantization) is the **calibration-free** option, and its value proposition is operational rather than statistical. GPTQ and AWQ both need a calibration corpus, a forward pass over it, and — for GPTQ — a Hessian inverse per layer. HQQ needs neither: it treats quantization as a per-group optimisation problem over just the weights themselves.

The mechanism: rather than setting scale and zero-point from min/max (which is dominated by outliers) or minimising plain L2 error (which is also dominated by outliers, since squared error rewards fitting the tail), HQQ minimises a **sparsity-promoting** error norm — an Lp objective with p < 1 — on the residual `W − dequant(quant(W))`. Under that objective the optimiser is happy to let a few weights be badly wrong if it means the bulk is well represented, which is exactly the right prior for weight tensors with a few outliers. The "half-quadratic" part is the solver: split the problem with an auxiliary variable so it alternates between a closed-form shrinkage step on the residual and a closed-form update of the zero-point, both cheap, and converge in a handful of iterations. No gradients, no data.

The consequence you care about: quantizing a 70B takes **minutes rather than hours**, with no calibration data to procure, license, or accidentally leak. That matters more than it sounds. Three real scenarios where I would reach for HQQ:

1. **You have many model variants.** A platform serving 40 customer-fine-tuned checkpoints cannot run a GPTQ job per checkpoint per release; calibration-free turns a two-hour step into a two-minute one.
2. **You have no legal calibration data.** A healthcare or legal deployment where the only in-distribution text is data you may not copy onto a quantization box.
3. **The model is unusual.** A heavily fine-tuned or domain-shifted checkpoint where a generic C4/WikiText calibration set is actively misleading — a calibration-free method can't be poisoned by a wrong calibration set because there isn't one.

The trade: on standard benchmarks with a *good* calibration set, GPTQ and AWQ generally edge it out at 4 bits, because they are using information HQQ refuses to look at. HQQ's own answer to that is an optional short LoRA-style fine-tune afterwards to recover the gap.

**🗣 Say this in the room:** "HQQ is calibration-free — it solves a sparsity-promoting Lp objective per weight group rather than fitting activation statistics. Slightly behind a well-calibrated AWQ on benchmarks, but it runs in minutes and can't be poisoned by a bad calibration set, so it's what I'd standardise on for a fleet of many fine-tuned checkpoints."

### Decode GGUF quant names for me — what is Q4_K_M, and what does an importance matrix add?

GGUF is llama.cpp's single-file container format, and its "k-quant" family is a distinct design point from GPTQ/AWQ: it is optimised for **CPU and Apple Silicon inference with mixed bit-widths per tensor**, not for GPU tensor cores.

The k-quant structure: weights are organised into **super-blocks of 256**, subdivided into 16 sub-blocks of 16 weights. Each sub-block gets its own scale (and, for the asymmetric variants, its own min), and those 16 scales are *themselves* quantized to a small number of bits with a per-super-block fp16 scale. So it is a two-level hierarchical scaling scheme — the same double-quantization idea as QLoRA, but at finer granularity and with per-sub-block adaptivity. That hierarchy is why k-quants beat the old flat `Q4_0` at the same nominal bit-width.

The naming: `Q<bits>_K_<size>`. `Q4_K` means 4-bit k-quant. The `_S` / `_M` / `_L` suffix is **not** a different bit-width for all tensors — it is a *mixing policy*. The heuristic llama.cpp encodes is that some tensors are far more sensitive than others, specifically the attention `V` projections, the FFN down-projections, and the token embedding / output head. So `Q4_K_M` ("medium") quantizes most tensors at 4 bits but promotes several sensitive ones to 6 bits; `Q4_K_S` ("small") uses 4 bits more uniformly; `Q5_K_M` raises the base. The effective bits-per-weight of `Q4_K_M` is around 4.8–4.9, not 4.0 — which is why it is noticeably better than a flat 4-bit scheme and why comparing it to "GPTQ 4-bit" as though they were the same budget is unfair to GPTQ.

The **importance matrix** (`imatrix`) is llama.cpp's version of the activation-awareness that AWQ and GPTQ get from calibration. You run a calibration corpus through the fp16 model and accumulate, per weight column, the mean squared activation that column multiplies. That gives a per-column importance weight. The quantizer then minimises **importance-weighted** error rather than plain error when choosing scales — mathematically the same instinct as GPTQ's diagonal-Hessian term, implemented as a diagonal reweighting rather than a full inverse. On low-bit quants (Q2_K, Q3_K, IQ-series) the imatrix is the difference between usable and broken; at Q5/Q6 it barely matters.

**⚠ Trap:** imatrix corpora are usually generic English (wiki text, a code sample, some multilingual scraps). If you quantize a Japanese-tuned or a domain-specific model against a generic imatrix, the importance weights are wrong for your traffic — columns that matter for your language get treated as unimportant and get crushed. Build the imatrix from a sample of *your* traffic. This is the same failure as calibration-set poisoning and it hits GGUF users hardest because the community imatrix files are so easy to reuse.

**📅 Volatile:** the k-quant and IQ-quant family evolves fast in llama.cpp; specific bits-per-weight and mixing policies change between releases. Verify against the current `llama-quantize` output table before quoting a number.

### Our AWQ-quantized model passed perplexity checks but is measurably worse on our production traffic. Walk me through how a calibration set silently poisons a quantization.

This is the most under-taught failure mode in the field, and I would open by naming the mechanism rather than reaching for tools.

Both GPTQ and AWQ build their entire objective from calibration activations. GPTQ's Hessian is literally `2XXᵀ` where `X` is calibration activations — the covariance structure of *that data* defines which weight directions get error compensation. AWQ's saliency is `mean|X_j|` over calibration — which channels are "important" is defined by that data. So the quantizer is answering the question "which weights matter for *this* distribution?" If your calibration distribution is C4 English web text and your production distribution is legal contracts in three languages plus JSON tool calls, the quantizer optimised the wrong function. Nothing errors. The checkpoint loads. Perplexity on WikiText looks great — of course it does, WikiText is in-distribution for the calibration set.

**🔍 Failure taxonomy — calibration poisoning, as a decision procedure:**

1. **Wrong domain.** Symptom: benchmark scores hold, task metrics drop. Test: build a second calibration set sampled from production traffic, re-quantize, re-run *your* eval. If the gap closes, that was it. This is the single highest-yield experiment and it takes an hour.
2. **Wrong length.** Calibration sequences are usually 2,048 tokens. If production runs 32k, the activation statistics at position 20,000 — where RoPE has rotated far, where attention-sink behaviour is fully developed, where the value norms have grown — were never observed. Symptom: quality that degrades monotonically with context length. Test: quantize with 8k or 16k calibration sequences and re-measure at long context.
3. **Wrong language / script.** Non-Latin tokenisation produces different token-frequency and different embedding-row activation profiles. Symptom: English fine, CJK or Devanagari degraded. Test: 30% non-English calibration mixture.
4. **Too few samples.** Under ~128 sequences the Hessian is rank-deficient and the dampening term dominates, so GPTQ degenerates toward round-to-nearest. Symptom: quantized model is about as bad as naive RTN. Test: compare against an actual RTN baseline — if GPTQ isn't clearly beating RTN, your calibration is not doing its job.
5. **Contaminated / degenerate samples.** Calibration text with long runs of repeated tokens, boilerplate headers, or a single dominant document skews the covariance. Symptom: erratic per-layer error. Test: plot per-layer reconstruction error; a healthy run is smooth across depth.

**📐 Numbers you must know:** the working recipe is **128–512 sequences of 2,048+ tokens**, sampled to match production in *domain, language mix, length distribution and format* (if 40% of your traffic is chat-templated with a tool schema, 40% of calibration should be too). Below 128 you are under-determined; above ~512 the returns are flat and the runtime is not. Include the chat template — an instruction-tuned model's activation statistics on raw web text are not the statistics it sees in production.

**🗣 Say this in the room:** "Calibration data defines the quantizer's objective — GPTQ's Hessian is literally the calibration covariance and AWQ's saliency is calibration activation magnitude. A generic English 2k-token calibration set optimises for a distribution you don't serve, and it fails silently because the standard benchmarks are also in that distribution. I sample 256 sequences from production traffic, with the real chat template and the real length mix, and I always keep an RTN baseline to prove the calibration is contributing at all."

### FP8 on Hopper — E4M3 versus E5M2. Why two formats, and where does each go?

Because a single 8-bit float cannot serve both the forward and the backward pass, and the reason is the range/precision split.

**E4M3** is 1 sign, 4 exponent, 3 mantissa. In the OCP/Nvidia definition it sacrifices infinities and most NaN encodings to buy one extra binade, giving a max magnitude of **448** and a min normal around 2⁻⁶. Three mantissa bits means ~2 decimal digits of precision. This is the format for **weights and activations in the forward pass**: those values are bounded (post-LayerNorm activations, normalised weights), so you don't need much range, and you want every mantissa bit you can get.

**E5M2** is 1 sign, 5 exponent, 2 mantissa, with IEEE-style inf/NaN, max magnitude **57,344**, and denormals reaching down to ~2⁻¹⁶. Five exponent bits is the same as fp16, so E5M2's dynamic range equals fp16's. This is the format for **gradients**, which span an enormous dynamic range across layers and across training — precision barely matters for a gradient (you're going to average it over a batch and multiply by 1e−4), range matters enormously, and an underflow to zero is a silently dead weight.

The critical piece nobody mentions until pushed: **fp8 requires per-tensor scaling factors** to place your data inside the format's window. 448 is not a large number; an activation tensor with values around 3,000 saturates entirely. Nvidia's Transformer Engine maintains a running `amax` history per tensor and computes a scale so the tensor's max lands near the format's max, applying the scale before the cast and the inverse in the accumulation. That's the delayed-scaling scheme. Get the scale management wrong and you get either silent clamping or silent flush-to-zero, both of which look like "training is a bit worse than bf16" rather than like a crash.

For **inference**, fp8 has become the pragmatic default on Hopper-class hardware and later, and the reasons are worth stating precisely: the accumulation still happens in fp32 inside the tensor core, so error doesn't compound across the reduction; the per-element exponent means fp8 tolerates the activation outlier problem far better than int8 at the same bit count (an outlier just uses a bigger exponent instead of blowing out the shared scale); and native hardware support means no dequantization step in the kernel. W8A8 in fp8 typically lands within a fraction of a percent of bf16 on most evals, which is a much easier sell than int8's tuning burden.

**📐 Numbers you must know:** E4M3 max = 448, E5M2 max = 57,344, both 1 byte. H100 SXM: ~990 dense bf16 TFLOPS, ~1,980 fp8 TFLOPS, 80 GB HBM3 at 3.35 TB/s. H200: same compute, 141 GB at 4.8 TB/s. Those six numbers carry most of a serving-sizing round.

**⚠ Trap:** "fp8 is int8 but better, so just switch." fp8 gives you the same 2× tensor-core throughput as int8 on Hopper — it does *not* give you more. Its advantage is accuracy-per-engineering-hour, not peak FLOPS. And on Ampere (A100) there is **no fp8 tensor core at all**; an fp8 checkpoint on an A100 will either refuse to load or run through an emulation path that is slower than bf16. Format support is hardware-generational and it is the first thing to check.

### MXFP4 and NVFP4 on Blackwell — explain the microscaling idea and why it's the current throughput frontier.

The mental model: 4-bit *floating point* alone is useless — E2M1 gives you eight magnitudes and a dynamic range of about 12×, so any tensor with more spread than that saturates. Microscaling fixes this by pairing a very small block of fp4 elements with a **shared exponent**, recreating dynamic range at block granularity. It is the same insight as group-wise int4 scaling, but with the scale expressed as a power of two (or a tiny float) so applying it is a bit-manipulation, not a multiply — which is what lets the hardware do it inside the tensor core at full rate.

**MXFP4** is the OCP Microscaling standard: blocks of **32** elements, each element E2M1 (4 bits), sharing one **E8M0** scale — a pure 8-bit power-of-two exponent, no mantissa. Effective width = 4 + 8/32 = **4.25 bits per element**.

**NVFP4** is Nvidia's Blackwell-native variant: blocks of **16** elements, each E2M1, sharing one **E4M3** (fp8) scale, plus a second-level per-tensor fp32 scale. Effective width = 4 + 8/16 = **4.5 bits per element**. Two design differences from MXFP4 and both matter: the smaller block (16 vs 32) localises outlier damage twice as tightly, and the scale having a *mantissa* means the block can be normalised to the format's range precisely rather than to the nearest power of two — which alone is worth up to a half-bit of effective precision. In exchange NVFP4 carries 0.25 more bits of metadata and is Nvidia-specific.

Why this is the frontier rather than a curiosity: unlike int4 W4A16, these are formats the **tensor cores execute natively for both operands**. Blackwell's fp4 tensor cores roughly double fp8 throughput. So for the first time you get the 4× memory reduction versus fp16 *and* the math-unit speedup, instead of choosing. That combination is what makes 4-bit viable for prefill and for large-batch serving, not just for batch-1 decode.

**💰 Math:** take a 70B on a hypothetical fp16 baseline of 140 GB of weights versus 70e9 × 4.5/8 = **39.4 GB** at NVFP4 — a 3.55× memory reduction, which is the "4× less memory than FP16" claim once you count the KV cache in fp8 alongside it. On bandwidth-bound decode that is a 3.55× token-rate multiplier from the same argument as before. On compute-bound prefill, fp4 tensor cores roughly double fp8, which is roughly double bf16, so ~4× on the GEMM ceiling. The honest caveat: achieved throughput is well below ceiling because attention, normalisation, sampling, and the KV path do not run in fp4.

**📅 Volatile:** Blackwell part numbers, per-SKU fp4 TFLOPS, and which serving engines have production-quality NVFP4 kernels all move quarterly. Quote the *format design* — block sizes 32/16, scale types E8M0/E4M3, effective 4.25/4.5 bpw — which is stable, and verify the throughput figures before your loop.

**⚠ Trap:** treating MXFP4 and NVFP4 as interchangeable because both are "fp4." They are different on-disk layouts with different block sizes and different scale types. A checkpoint in one is not readable by a kernel expecting the other, and the quality gap between them at the same nominal 4 bits is not noise — the finer block and the mantissa'd scale are real. Ask which one before you commit a conversion pipeline.

### Give me the decision rule between W4A16, W8A8 and W4A4 for a real service.

Start from the bottleneck, not from the bit count. Each configuration removes a different constraint, and picking one without knowing which constraint binds is how teams spend a sprint for a 2% win.

**W4A16** removes *weight bandwidth*. Choose it when: you are decode-dominated (long outputs, short prompts), concurrency per replica is low (batch 1–16), or the model does not fit in memory at 8 bits. Canonical fits: local/on-device inference, a coding-assistant autocomplete serving one user per GPU slot, a 70B you're trying to squeeze onto a single 80 GB card. What it does not do: help prefill, help large-batch throughput, or reduce KV pressure.

**W8A8 (fp8 on Hopper+)** removes *compute*. Choose it when: prefill dominates (RAG, long-document analysis, big system prompts), or you serve at high concurrency where the GEMMs are already compute-bound. Canonical fits: an enterprise RAG service at Glean/Harvey scale with 8k–32k contexts, a batch summarisation pipeline, any high-QPS API. Quality risk is lowest of the three because fp8's per-element exponent tolerates outliers and per-token dynamic scaling is cheap. This is my default for a Hopper-class serving fleet.

**W4A4** removes *both* — and is where the honesty has to come in. It requires rotation-based outlier removal (QuaRot/SpinQuant class) to be viable at all, it typically costs several points more than W4A16 on reasoning-heavy evals, and until Blackwell's native fp4 path it needed hand-tuned kernels that only existed for a handful of shapes. Choose it when you are *simultaneously* memory-capacity-bound and compute-bound and you have the eval budget to prove the quality is acceptable for your specific task — which in practice means high-volume, latency-critical, relatively simple tasks (classification, routing, extraction, on-device assistants), not agentic reasoning.

The compound option people forget: **W4A16 weights + fp8 KV cache**. These are orthogonal, and for a long-context service the KV is often the binding constraint, not the weights.

**🗣 Say this in the room:** "W4A16 is a bandwidth fix, W8A8 is a compute fix, W4A4 is both and costs you quality. I'd profile first — if prefill FLOPs dominate the request, fp8 W8A8; if it's decode at low batch or the model doesn't fit, W4A16; and I'd quantize the KV cache separately from either, because the KV is usually the thing actually capping my concurrency."

### Make the case for FP8 KV cache. Show me the arithmetic and tell me what breaks.

The KV cache is the per-request state whose eviction policy you do not control, and it grows linearly with context and linearly with concurrency. It is the reason your throughput ceiling is not what your FLOPs suggest.

The formula, which you should be able to write without hesitation:

```
bytes/token = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_element
```

**💰 Math — Llama-3-70B** (80 layers, 8 KV heads after GQA, head_dim 128):
- fp16: 2 × 80 × 8 × 128 × 2 = **163,840 bytes = 160 KB per token**.
- fp8: **80 KB per token**.

At 32k context that is 5.24 GB per sequence in fp16, 2.62 GB in fp8. Now size a serving node: 2×H100-80GB = 160 GB total, minus 4-bit weights at 37.2 GB, minus ~8 GB of activations/workspace/CUDA context, leaves ~115 GB for KV. In fp16 that's 115 / 5.24 = **21 concurrent 32k sequences**. In fp8 it's **43**. You just doubled the concurrency of the node without touching the model quality knob that anybody evaluates.

The second win is bandwidth, and it is the one people forget. Attention re-reads the **entire** KV cache for every generated token, so KV bytes are decode traffic, not just decode storage. Per token at 32k context, one sequence's KV read is 5.24 GB in fp16 against 37.2 GB of 4-bit weights — so the KV is already 14% of your decode traffic for a *single* sequence, and with a batch of 8 such sequences the KV read (42 GB) **exceeds** the weight read, because weights are read once for the whole batch while KV is read per sequence. That crossover is the number to remember: past roughly 10k tokens at any real batch size, KV traffic dominates your decode step, and halving it is worth more than another bit off the weights.

Why fp8 specifically rather than int8: keys and values have outliers too, particularly in the key tensor along the head dimension where RoPE and attention-sink behaviour concentrate magnitude. fp8's per-element exponent absorbs that; int8's shared scale does not, which is why int8 KV cache historically needed per-head or per-channel scales and finicky handling while fp8 KV works almost out of the box. E4M3 is the right variant — bounded values, want the mantissa.

**⚠ Trap:** quantizing K and V with the same policy. The **key** tensor is the sensitive one: keys go through a dot product with the query and then a softmax, and softmax is exponentially sensitive to errors in the logits — a small key error becomes a large attention-weight error, and the effect compounds across 80 layers. Values are averaged with the attention weights, so their errors average out. The empirically-supported practice is to be more conservative with K than with V; several engines expose separate `kv_cache_dtype` handling and some support per-channel key scales specifically for this reason. If you must go to int4 KV, go int4 on V and keep K at 8 bits.

**🔍 Failure taxonomy — quantized KV in production:** (1) short-context evals pass, long-context retrieval fails — because the softmax error compounds with the number of keys being compared, so needle-in-a-haystack at 64k breaks while 4k is clean; (2) multi-turn conversations degrade over turns as the quantized prefix accumulates; (3) prefix-cache hits now serve *quantized* KV to a request that would have had fp16, so quality becomes a function of cache-hit status and your A/B looks noisy for no visible reason.

### How much quality do you actually lose to 4-bit weight-only quantization, and where does the loss concentrate?

The headline is small and the distribution of the damage is what gets people fired.

**📐 Numbers you must know:** a well-executed 4-bit weight-only PTQ (GPTQ or AWQ, group 128, act-order or equivalent, good calibration) on a modern instruction-tuned model typically costs **on the order of 1–2% relative on standard aggregate benchmarks** — call it a point or two on MMLU-class multiple-choice, and a small fraction of a perplexity point on WikiText. That is the number to quote, with the immediate caveat that it is an *average over an easy benchmark distribution*, and averages hide everything that matters here.

Where the loss actually concentrates, in rough order of severity:

**Long context.** Degradation grows with sequence length, for three compounding reasons: calibration was done at 2k so the quantizer never saw the activation statistics at position 30,000; attention over more keys means more accumulated error in the softmax; and any positional-extrapolation behaviour the model has is a delicate, low-margin computation. A model that loses 1.5 points at 2k can lose 8+ points on a 32k retrieval task.

**Math and multi-step reasoning.** Chain-of-thought is a sequential process where every token conditions the next. A quantization error that changes one token from `7` to `9` doesn't degrade gracefully — the whole trace is wrong. Error is *not* averaged away in autoregressive generation; it is amplified. This is why GSM8K/MATH-style evals reliably show the largest drops, often 3–5× the aggregate benchmark drop.

**Multilingual, especially non-Latin scripts.** Low-resource languages occupy lower-margin regions of the model's representation and are almost never in the calibration set. This is the section's thesis case: the regression that only appears at 32k context in Japanese.

**Rare formats and structured output.** JSON schema adherence, tool-call argument formatting, and code syntax are low-entropy, high-precision behaviours where a single wrong token invalidates the output. Aggregate benchmarks never test this; your product does on every request.

**Safety and refusal behaviour.** Alignment is a comparatively small perturbation on top of a pretrained model, and quantization is a perturbation of similar magnitude. Refusal rates and jailbreak resistance genuinely shift under compression, in both directions.

**🗣 Say this in the room:** "The honest number is 1–2% on aggregate benchmarks at 4-bit weight-only, and that number is a lie about the failure profile. The loss concentrates on long context, on chain-of-thought math where errors amplify rather than average, on non-English, and on strict structured output. So the aggregate number is what I report and the sliced eval is what I gate the deploy on."

### We shipped an AWQ 4-bit model. English is fine; Japanese output degraded noticeably. Debug it.

Good — this is the exact regression the section is built around, so let me be concrete about the procedure rather than listing hypotheses.

**Step 0: confirm it's the quantization.** Run the fp16 checkpoint on the same Japanese prompts through the same serving stack. If fp16 also degraded, it's not quantization — it's the tokenizer config, the chat template, a Unicode normalisation step, or a sampling parameter, and you've just saved a week. I insist on this step because roughly a third of "the quantization broke it" reports are not quantization.

**Step 1: quantify with a paired eval.** You need a Japanese slice with the same task shape as your English slice. Report per-slice, not aggregate — the whole point is that the aggregate hid this.

**Step 2: check calibration composition.** Open the calibration set. If it is 100% English (the default `pileval` / C4 sample almost always is), you have the answer. AWQ's saliency is `mean|X_j|` over calibration. The input channels that light up for Japanese tokens — different embedding rows, different subword statistics, and in a BPE tokenizer Japanese text produces many more tokens per character with different frequency structure — were never observed to be salient, so AWQ chose scaling factors that protect the English-activated channels and let the Japanese-activated ones absorb the error. Fix: re-quantize with a calibration mixture that matches your traffic's language distribution (I'd use at least 25–30% of the affected language even if production traffic is less, because you want the covariance to *see* those channels).

**Step 3: if the calibration is already mixed, look at the tokenizer-adjacent tensors.** The embedding matrix and the LM head are where language-specific behaviour lives most densely, and they are the largest tensors in the model. Many pipelines quantize them; many keep them in fp16 or at higher precision precisely because they're sensitive. Check what your pipeline did. Promoting `lm_head` and `embed_tokens` to 8-bit costs a few GB on a 70B (vocab 128k × hidden 8192 × 2 = 2.1 GB each in fp16) and often fixes multilingual regressions outright.

**Step 4: layer-wise diagnosis.** Run both checkpoints on the same Japanese batch, capture hidden states per layer, and compute cosine similarity fp16-vs-quantized per layer. A healthy quantization is ≥0.99 everywhere with a slow drift. A bad one shows a knee — two or three layers where similarity falls to 0.9 or below, and those are your culprits. Promote just those layers' weights to a higher bit-width (mixed-precision by layer is cheap: a handful of layers at 8 bits on an 80-layer model is a ~5% size increase).

**Step 5: if it still fails, escalate the method.** Group size 128 → 64 for the affected layers; or move to a rotation-based scheme, which is basis-independent and therefore much less sensitive to which channels a particular language activates.

**⚠ Trap:** "just use more calibration data." Volume is not the issue — *composition* is. Going from 128 to 2,048 English sequences changes nothing about a Japanese regression. I've watched a team burn three days on that.

**🗣 Say this in the room:** "First I'd rule out the tokenizer by running fp16 on the same inputs. Then I'd look at calibration composition, because AWQ's saliency is defined by calibration activation magnitudes and an English-only calibration set literally defines Japanese-activated channels as unimportant. Then I'd check whether embeddings and the LM head were quantized, and do a per-layer cosine-similarity diff to localise it. The permanent fix is a language-matched calibration mixture plus a non-English slice in the gating eval."

### Same model, different symptom: it's fine at 2k context and clearly worse at 32k. What's your hypothesis ordering?

Long-context degradation under quantization has four distinct causes and they need different fixes, so ordering matters.

**Hypothesis 1 — the KV cache, not the weights.** If you quantized the KV cache, this is your first suspect and it's the most likely one. Attention error compounds with the number of keys: the softmax over 32,000 logits amplifies small key perturbations far more than a softmax over 2,000, and every generated token re-reads the entire cache. Test in five minutes: keep the quantized weights, set the KV cache back to fp16, re-measure. If the regression vanishes, you're done — and the fix is either fp8 KV instead of int8/int4 KV, or asymmetric treatment (keep K at higher precision than V).

**Hypothesis 2 — calibration sequence length.** Your calibration was 2,048-token sequences. The model's activation distribution at position 30,000 is genuinely different: RoPE has rotated through many cycles, attention-sink tokens have accumulated enormous magnitudes, and value-state norms drift with position. GPTQ's Hessian and AWQ's saliency were both computed on statistics that don't hold there. Test: re-quantize with 8k–16k calibration sequences (fewer of them, to keep the token budget similar) and re-measure at 32k.

**Hypothesis 3 — the model's long-context ability was marginal to begin with.** Many models advertise a context window they only weakly support. Quantization eats whatever margin existed. Test: run the *fp16* model on the same 32k eval. If fp16 is also mediocre at 32k, quantization exposed a pre-existing weakness rather than creating one, and the fix is a different model, not a different quantizer.

**Hypothesis 4 — position-sensitive components got crushed.** RoPE frequency handling, and in particular any attention scaling or long-context adaptation (YaRN-style scaling factors, per-layer attention temperature), can be sensitive. Also check whether your engine applies RoPE before or after KV quantization — quantizing *post*-RoPE keys is standard and correct, but a mismatch between the writer and the reader path produces exactly this symptom.

The eval that catches this before deploy, which is the actual lesson: **a length-stratified needle-in-a-haystack or multi-hop retrieval eval at 2k / 8k / 32k / your max**, run on both fp16 and quantized, reported as a curve rather than a number. If your quantization gate is a single MMLU score at 2k, you will ship this bug repeatedly.

**🏋 Drill (20 minutes, unaided):** given a 70B with 80 layers, 8 KV heads, head_dim 128, on 2×H100-80GB with 4-bit weights, produce a table of max concurrent sequences at contexts {4k, 32k, 128k} for KV in {fp16, fp8, int4}. Pass criterion: correct 160/80/40 KB-per-token figures, correct available-KV-memory figure (~115 GB), and all nine cells within 5% — and you should notice that 128k fp16 gives you 5 sequences, which is a business problem, not an engineering one.

### On-device and voice — pitch me 4-bit there, with the latency arithmetic.

On-device is where quantization stops being an optimisation and becomes the enabling condition, because the constraints are absolute: a phone has 8–16 GB of unified memory shared with the OS and every other app, memory bandwidth of roughly 50–120 GB/s (versus 3,350 on an H100), and a thermal budget measured in watts. And in a voice assistant the latency target is set by human conversational turn-taking — you have roughly 200–300 ms of silence before the interaction feels broken, and ASR plus TTS have already eaten most of it.

**💰 Math — the ~40% latency reduction, derived rather than quoted.** Take an 8B model on a device with ~100 GB/s effective bandwidth. Decode is bandwidth-bound, so time per token ≈ weight bytes / bandwidth. At fp16: 16 GB — it doesn't fit, so start at int8: 8 GB / 100 GB/s = 80 ms/token, i.e. 12.5 tok/s. At 4-bit group-128 (4.25 bpw): 8e9 × 4.25/8 = 4.25 GB → 42.5 ms/token, 23.5 tok/s. For a 40-token spoken response that is 3.2 s versus 1.7 s of generation — and the **time-to-first-token** improves similarly because prefill of a short voice prompt is also dominated by streaming the weights in. A ~47% reduction in that example; the commonly-cited **~40% end-to-end latency reduction from 4-bit PTQ** is that bandwidth ratio diluted by the parts that don't scale (audio front-end, tokenisation, sampling, TTS). That is how you should present the number: derive the bandwidth ratio, then discount it for fixed overhead.

The second, less-discussed win is **energy**. DRAM access dominates the energy budget of on-device inference — moving a byte from off-chip memory costs orders of magnitude more energy than a MAC. Halving weight bytes roughly halves the dominant term, which shows up as battery life and as sustained throughput before thermal throttling kicks in. A phone that thermally throttles after 90 seconds of fp16 inference may run indefinitely at 4-bit.

Practical choices here: the format must match the accelerator. Apple Silicon → MLX or Core ML with its own palettisation schemes; Qualcomm/MediaTek NPUs → int8 or int4 in the vendor's runtime, frequently with per-tensor-only support, which is exactly the case where QAT earns its cost. llama.cpp/GGUF is the portable fallback and runs everywhere at some efficiency cost.

**⚠ Trap:** benchmarking on-device quantization on a plugged-in dev phone in a cool room, at batch 1, with the model already in page cache. Production is a thermally-throttled device with 300 MB of free memory, cold-starting from flash. **Model load time from storage frequently dominates first-request latency** — a 4.25 GB file off a phone's storage at ~1 GB/s is 4+ seconds before a single token. Measure cold start separately and design for a warm resident model or memory-mapped weights.

### A format your kernel doesn't support is worthless. Walk me through how you actually check engine and hardware support before committing.

Nobody teaches this and it costs more sprints than every accuracy question in this section combined: **the quantization method is a joint choice with the serving engine and the GPU generation, and choosing them in the wrong order wastes a sprint.** I have seen a team spend two weeks producing beautiful GPTQ checkpoints for a format their engine loaded via a generic fallback path that was slower than fp16.

The check, in the order I run it:

**1. Does the target GPU have the math unit?** fp8 tensor cores: Hopper (H100/H200) and later, plus Ada (L40S/4090). Not Ampere. fp4 tensor cores: Blackwell. 2:4 sparse tensor cores: Ampere and later. int8 tensor cores: Turing and later. If the unit isn't there, the format runs through emulation or dequant-to-fp16 and you get the memory saving only — which may still be the right call, but you must know which win you're buying.

**2. Does the engine have a *fast* kernel for this exact combination, or just a loader?** The combination is (weight format × activation format × group size × act-order × head/hidden shapes × batch regime). Engines ship optimised kernels for a subset and a slow generic path for the rest. The tell is a benchmark: quantize, then measure tokens/sec against fp16 at your real batch size. If 4-bit isn't beating fp16 by close to the bits ratio at batch 1, you are on the slow path.

**3. Does it compose with everything else you need?** This is where things actually break. The questions I ask explicitly: does the quantized path support **tensor parallelism** at my TP degree (group size must divide the per-rank shard of the K dimension — TP=8 on a 4096 hidden dim gives 512 per rank, fine for group 128; some quant + TP combinations are simply unimplemented). Does it support **LoRA adapters on top of a quantized base**? Does it support **prefix caching** with quantized KV? Does **chunked prefill** work? Does **structured/constrained decoding** work? Each of these has, at some point, been mutually exclusive with some quantization format in some engine.

**4. Is the checkpoint format canonical?** GPTQ, AWQ, GGUF, and the compressed-tensors style used by several serving stacks all have different on-disk layouts and metadata. A conversion is usually possible but is a step in your release pipeline that can silently change quality.

**The rule I enforce in review:** before anyone quantizes anything, produce a one-page matrix — rows are candidate formats, columns are {target GPU has the unit, engine has a fast kernel, works with TP=N, works with LoRA, works with prefix caching, works with structured output, quality delta on our eval} — and fill the last column *last*, because a format that fails any of the first six is not a candidate no matter how good its perplexity is.

**📅 Volatile:** every cell of that matrix changes across engine releases; vLLM, SGLang and TensorRT-LLM add and deprecate quantization paths continuously. Build the matrix from the version you will deploy, not from memory or from a blog post.

**🗣 Say this in the room:** "I pick the engine and the GPU generation first, then the quantization format from what that combination has a fast kernel for, and I validate with a tokens-per-second benchmark at production batch size before I care about perplexity. A format with great accuracy and a generic fallback kernel is a regression, not an optimisation."
