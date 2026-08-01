### Before we talk about any specific method — why does quantization work at all? Why can you throw away 12 of 16 bits of a weight and still have a working model?

Because a trained neural network is a massively over-parameterised, redundantly-encoded function, and the *precision* of any individual weight carries almost none of the information. The information lives in the aggregate: a matrix-vector product sums 4,096 or 8,192 terms, and independent rounding errors on those terms partially cancel — the error on the sum grows like √n while the signal grows like n. Quantization is a lossy compression of the parameters where the loss budget is set by how much perturbation the network's output can absorb before the argmax over the vocabulary changes. Empirically, that budget is very large for weights and very small for activations, and understanding *why those two differ* is most of this section.

Here is the backend bridge that makes it click. You already accept that a float column in Postgres can often be a `numeric(6,2)` or even a small int with a scale factor, because the downstream consumer only needs three significant figures. Quantization is exactly that decision, applied per-tensor, with the twist that you cannot look at the consumer's requirements analytically — you have to measure them empirically on an eval. And the second twist: unlike your Postgres column, the *storage format is also the compute format*, so choosing int4 changes which silicon multiplier runs.

The distributional fact underneath it all: trained transformer weights within a layer are approximately zero-mean and roughly Gaussian/Laplacian, with no long tail. A 4,096-wide row of a projection matrix has values clustered in a narrow band, so a uniform 16-level grid stretched across that band represents them with a signal-to-quantization-noise ratio that is entirely tolerable. Activations are not like this at all — a handful of channels run 20–100× larger than the rest — which is why weight-only 4-bit is routine and activation 4-bit is a research problem.

**⚠ Trap:** candidates say "quantization works because neural nets are robust to noise." That is half-true and it will get you pushed on. The precise statement is that networks are robust to *small, roughly-independent, zero-mean* perturbations of weights, and catastrophically non-robust to *large, systematic* ones. Every real quantization method is an engineering effort to make the error look like the first kind and not the second — that is literally what GPTQ's error compensation, AWQ's channel scaling and QuaRot's rotations all do.

**🗣 Say this in the room:** "Weights are redundant and narrowly distributed, so a coarse grid costs almost nothing; activations have massive per-channel outliers, so the same grid is catastrophic. That asymmetry is why the entire field splits into weight-only methods and outlier-handling methods, and it's the first thing I'd establish before picking a scheme."

### Walk me through the numeric formats — fp32, fp16, bf16, int8, int4 — bit layouts and what each one actually buys you.

A floating-point format is a sign bit, an exponent field, and a mantissa field. The exponent sets **dynamic range** (how big and how small a number you can represent at all); the mantissa sets **precision** (how finely spaced representable numbers are within a decade). Every format choice in ML is a trade between those two, and confusing them is the single most common source of "why is my loss NaN."

fp32 is 1+8+23: range roughly 1e−38 to 3e38, about 7 decimal digits. fp16 is 1+5+10: max ~65,504, min normal ~6e−5, about 3 decimal digits. bf16 is 1+8+7: **the same exponent field as fp32**, so the same range, but only ~2 decimal digits of precision. That single design decision is why bf16 won training — it is a truncation of fp32, conversion is a shift, and you never overflow where fp32 wouldn't. fp16 has more mantissa but its narrow exponent is why fp16 training needs loss scaling and bf16 does not.

int8 and int4 are not floating point at all — they are fixed-point integers that only become numbers when paired with a scale (and optionally a zero-point). int8 gives you 256 levels, int4 gives you 16. There is no exponent, so there is no per-element dynamic range; all the dynamic range comes from *how many elements share a scale*, which is precisely the per-tensor/per-channel/per-group question.

fp8 comes in two variants because one format cannot serve both roles: **E4M3** (1+4+3, max 448 in the OCP/Nvidia definition, no infinities) is for forward-pass weights and activations, where you want precision and values are bounded; **E5M2** (1+5+2, max 57,344, IEEE-like with inf/NaN) is for gradients, which span a huge dynamic range and need the exponent more than the mantissa. fp4 as E2M1 has essentially no usable dynamic range on its own — 1+2+1 gives you the magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} and their negatives — which is exactly why the fp4 formats that matter (MXFP4, NVFP4) always pair it with a fine-grained *block* scale.

**📐 Numbers you must know:** bytes per parameter — fp32 4, bf16/fp16 2, fp8 1, int4 0.5. A 70B model is therefore 280 / 140 / 70 / 35 GB of raw weights. Add ~6% for a 4-bit group-128 scheme's scale/zero overhead (4 bits + 32 bits per 128 weights = 4.25 bits/weight), giving 70e9 × 4.25 / 8 = **37.2 GB**. Memorize those five numbers; you will use them in every serving-design round.

**⚠ Trap:** "bf16 is more accurate than fp16." It is strictly *less* precise — 7 mantissa bits vs 10. It is more *robust*, because its range matches fp32. If an interviewer hears you say bf16 is more accurate, they will ask you the bit layout, and the answer will be visibly reconstructed rather than known.

### Derive the quantization mapping for me and implement quantize/dequantize from scratch. Symmetric or asymmetric — which and why?

The mental model: quantization is a lossy affine change of basis on a tensor. You choose an integer grid, you choose an affine map from real numbers onto that grid, and dequantization applies the inverse. Everything else — GPTQ, AWQ, k-quants — is about choosing *which* real values to map well, not about changing this equation.

Asymmetric (affine) quantization: given a group of values with min `a` and max `b`, and `n` bits, set `scale = (b − a) / (2^n − 1)` and `zero = round(−a / scale)`. Then `q = clamp(round(x/scale) + zero, 0, 2^n − 1)` and `x̂ = scale · (q − zero)`. Symmetric quantization drops the zero-point: `scale = max|x| / (2^(n−1) − 1)`, `q = clamp(round(x/scale), −2^(n−1), 2^(n−1)−1)`, `x̂ = scale · q`.

```python
import torch

def quantize_group(w, bits=4, symmetric=False):
    # w: (..., group_size) — quantize along the last dim
    qmax = 2**bits - 1
    if symmetric:
        s = w.abs().amax(-1, keepdim=True) / (2**(bits-1) - 1)
        s = s.clamp(min=1e-8)
        q = (w / s).round().clamp(-2**(bits-1), 2**(bits-1) - 1)
        return q, s, None
    lo, hi = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    s = ((hi - lo) / qmax).clamp(min=1e-8)
    z = (-lo / s).round()
    q = (w / s + z).round().clamp(0, qmax)
    return q, s, z

def dequantize(q, s, z=None):
    return s * (q - z) if z is not None else s * q

W = torch.randn(4096, 4096)
G = 128
q, s, z = quantize_group(W.view(-1, G), bits=4)
err = (dequantize(q, s, z).view_as(W) - W)
print(err.pow(2).mean().sqrt().item() / W.pow(2).mean().sqrt().item())  # ~1-2% relative RMSE
```

Which to pick: **symmetric for weights, asymmetric where the distribution is skewed.** Weights are near zero-mean so a symmetric grid wastes almost nothing and the kernel is cheaper — with symmetric int8 weights the GEMM is a pure integer dot product plus one scale multiply at the end. Asymmetric costs you a cross-term: `(q_x − z_x)(q_w − z_w)` expands into four products, and the correction terms have to be folded into the epilogue. Activations after ReLU/GELU-like nonlinearities are one-sided and genuinely want a zero-point. In practice modern 4-bit weight schemes (GPTQ, AWQ, GGUF k-quants) use asymmetric per-group because at 4 bits you have only 16 levels and cannot afford to waste half of them on a side of the distribution that's emptier.

**⚠ Trap:** `scale.clamp(min=1e-8)` is not decoration. A group of all-zero weights (common in a pruned or a padded tensor) gives scale 0, then `w/s` is NaN, and the NaN silently propagates into a checkpoint that loads fine and produces garbage. I have seen this exact bug ship. Always clamp, and always assert `torch.isfinite(q).all()` in the quantization pipeline.

### Per-tensor, per-channel, per-group with group size 128 — explain the trade and tell me why 128 is the number everyone uses.

Group size is the knob that trades metadata overhead against the damage done by outliers, and it's the single most under-discussed parameter in quantization.

One scale for an entire 4096×4096 weight matrix (per-tensor) means one outlier weight anywhere in 16.7M values stretches the grid for everything else. If the largest magnitude is 20× the typical one, you have effectively spent 4.3 of your bits representing the empty space between the bulk and the outlier — at 4 bits, catastrophic. Per-channel (one scale per output row) localises that damage to a single row. Per-group takes it further: split each row into contiguous chunks of `G` elements, each with its own scale and zero-point. Now an outlier ruins 128 weights, not 4,096, and not 16.7M.

The cost is metadata. At group 128 with fp16 scale and 4-bit-packed zero-point you carry roughly 16+4 = 20 extra bits per 128 weights ≈ 0.16 bits/weight; with the common fp16 scale + fp16 zero it is 32/128 = 0.25 bits/weight. So the effective bit-width of "4-bit group-128" is **4.25 bpw**, a 6% overhead. Drop to group 32 and it becomes 4 + 32/32 = 5 bpw — a 25% overhead, which eats most of the reason you went to 4 bits. Go up to group 256 and overhead falls to 4.125 bpw but quality degrades measurably on the harder models.

128 wins for two reasons, one statistical and one about silicon. Statistically it is near the knee: the perplexity-versus-group-size curve is steep from per-tensor down to ~128 and nearly flat below it. Mechanically, 128 divides evenly into every hidden dimension you will meet (4096, 5120, 8192, 11008, 14336), it matches the K-dimension tiling of int4 GEMM kernels so the dequant-scale lookup happens once per k-tile rather than mid-tile, and Tensor Core MMA fragments along K are 16/32 elements so 128 is a clean multiple. A group size that isn't a divisor of the reduction dimension forces a slow path or a pad, and you lose the speedup you quantized for.

**⚠ Trap:** the group axis must run along the **reduction (input/K) dimension**, not the output dimension. Group along the wrong axis and the kernel has to apply a different scale per output element mid-accumulation, which is not expressible in a tensor-core epilogue. This is the reason `desc_act` / act-order GPTQ checkpoints were historically slower — reordering columns by activation importance breaks the contiguity of the group along K and forces gather-style scale loads.

**🏋 Drill (10 minutes, no references):** Compute effective bits-per-weight for: 4-bit group-128 with fp16 scale + fp16 zero; 4-bit group-32 with fp16 scale only (symmetric); MXFP4 (fp4 elements, one E8M0 scale per 32); NVFP4 (fp4 elements, one E4M3 scale per 16). Then compute total weight bytes for a 70B model in each. Pass criterion: 4.25, 4.5, 4.25, 4.5 bpw and 37.2 / 39.4 / 37.2 / 39.4 GB, all within 5%.

### PTQ versus QAT — draw the line for me. When would you actually pay for quantization-aware training?

Post-training quantization takes a trained checkpoint and compresses it, using at most a few hundred calibration samples and a forward pass. Quantization-aware training inserts fake-quant (quantize-then-dequantize) ops into the graph and continues training, so the weights *learn* to be robust to the rounding. The gradient can't flow through `round()`, so QAT uses a straight-through estimator: forward applies the quantization, backward pretends it was the identity within the clipping range.

The mental model: PTQ asks "given this function, what's the closest low-precision function?" QAT asks "what's the best low-precision function, period?" The second is strictly more powerful and roughly 10,000× more expensive.

For LLMs at 8-bit and at 4-bit weight-only, **PTQ is the correct default and I would push back on anyone proposing QAT without evidence.** GPTQ or AWQ on 128–512 calibration sequences runs in 10 minutes to a few hours on one GPU and lands within ~1% of the fp16 baseline on standard benchmarks. QAT on a 70B needs a training cluster, a data mixture you probably no longer have, and a week of someone's life, to buy back maybe 0.3%.

QAT earns its keep in exactly three situations. First, **sub-4-bit** — 2-bit and 3-bit weights, where PTQ error is no longer a small perturbation and the network genuinely needs to reorganise. Second, **W4A4 or aggressive activation quantization**, where activation outliers must be trained away rather than shifted around. Third, **on-device targets with a fixed hardware format** — a phone NPU that only does int8 per-tensor, or a voice model that must hit a hard 4-bit budget, where you ship one artifact to millions of devices and a week of training amortises trivially.

There is a middle path worth naming in the room: **QAT-lite / quantization-aware fine-tuning**, where you run PTQ first and then do a short LoRA or full fine-tune with fake-quant enabled on a few hundred million tokens. It recovers most of the QAT gain for ~1% of the cost, and it's how most open-weight "QAT" releases are actually produced.

**🗣 Say this in the room:** "PTQ for anything at 4-bit weight-only or above — it's an afternoon and it lands within a point. I'd only fund QAT for sub-4-bit, for 4-bit activations, or for a fixed on-device format where one artifact ships to millions of devices. And even then I'd try quantization-aware *fine-tuning* on a small token budget first."

### There are three independent axes of quantization. Name them, and explain why the notation W4A16 versus W8A8 actually matters.

The axes are **weights**, **activations**, and the **KV cache**, and treating them as one knob is the most common conceptual error I see. They are independent because they buy different things and break in different ways.

Quantizing **weights** shrinks the static footprint and — critically — the bytes you stream from HBM per decode step. Decode at small batch is memory-bandwidth-bound, so weight bits map almost linearly to decode tokens/sec. Weight quantization is *easy* because weights are static, well-behaved, and you can spend an hour optimising them offline.

Quantizing **activations** is what lets you use the low-precision *math* units. An int8 or fp8 GEMM only runs if both operands are in that format. So activation quantization buys prefill throughput and compute-bound large-batch throughput — it does almost nothing for batch-1 decode. It is *hard* because activations are computed at runtime, are input-dependent, and contain systematic outlier channels.

Quantizing the **KV cache** shrinks the per-token state, which is what actually limits your concurrency and your max context. It buys you batch size and context length, not per-token compute.

The `WxAy` notation names the first two. **W4A16** means 4-bit weights, 16-bit activations: the kernel loads int4 weights, dequantizes them into registers, and does the multiply in fp16/bf16. You get the bandwidth win, you get *no* math-unit win, and you pay a small dequant cost. **W8A8** means both in int8/fp8: you get the actual tensor-core int8 or fp8 path, roughly 2× the fp16 math throughput on Hopper-class silicon, but only half the memory saving of 4-bit. **W4A4** wants both, and is where the outlier problem becomes existential.

The decision rule I use: **decode-dominated, low-concurrency, memory-capacity-tight → W4A16. Prefill-heavy or high-concurrency batch serving → W8A8 (fp8 on Hopper). Long context or high concurrency → quantize the KV regardless of what you did to the weights.** Those are three different bottlenecks and quantization is not one lever.

**⚠ Trap:** shipping W4A16 to fix a throughput problem in a high-batch serving fleet and being surprised it did nothing. At batch 64+, the GEMM is compute-bound; you've cut the weight bytes but you were never bandwidth-limited, and now you've *added* a dequantization step in the inner loop. I have watched a team lose 8% throughput this way. W4A16 is a batch-1-to-8 optimisation.

### Here's the part that confuses people: if a W4A16 kernel dequantizes weights back to fp16 before multiplying, where does the speedup come from?

From HBM traffic, and the roofline makes it inevitable. In autoregressive decode at batch 1, every weight in the model is read from HBM exactly once and used for exactly one multiply-accumulate per output element. The arithmetic intensity is about 1 FLOP per 2 bytes — a dot product of a 1×K vector with a K×N matrix does 2KN FLOPs while reading 2KN bytes of fp16 weights. Compare that to the hardware's ridge point: an H100 SXM does roughly 990 bf16 TFLOPS against 3.35 TB/s of HBM3, so its balance point is 990e12 / 3.35e12 ≈ **295 FLOPs per byte**. Batch-1 decode sits at ~1. You are two and a half orders of magnitude away from the compute roof. The GPU is idle, waiting on memory, essentially the entire time.

So the only thing that matters at batch 1 is how many bytes you move. Cut the weights from 2 bytes to 0.53 bytes and you move 3.8× fewer bytes; the dequant happens in registers/shared memory on functional units that were sitting idle anyway, and it costs you close to nothing.

**💰 Math:** Llama-3-70B on one H100 (3.35 TB/s). fp16 weights = 140 GB — that doesn't even fit in 80 GB, but ignore capacity for the bound: 3350 GB/s ÷ 140 GB = **23.9 tok/s** theoretical ceiling, and real kernels hit maybe 70–80% of that. At 4-bit group-128 (37.2 GB): 3350 ÷ 37.2 = **90 tok/s** ceiling. That is the 3.8× and it is why every on-device and low-concurrency deployment is 4-bit. Now do the same on an A100-80GB at 1.94 TB/s: fp16 is 13.9 tok/s, 4-bit is 52 tok/s. The ratio is identical because the ratio is just bits.

**⚠ Trap:** believing this speedup persists as you raise batch size. It does not. At batch B the weights are still read once but now do B times the work, so arithmetic intensity scales with B. Around B ≈ 128–256 on an H100 you cross the ridge and become compute-bound, at which point W4A16's dequant overhead makes it *slower* than fp16 or fp8. Serving engines know this: several will fall back to a different kernel or even to fp16 above a batch threshold. If you benchmark W4A16 at batch 1 and deploy at batch 128, your production numbers will not match your benchmark, and the direction of the miss will surprise you.

**🗣 Say this in the room:** "W4A16 doesn't speed up the math, it speeds up the memory. Batch-1 decode has arithmetic intensity around 1 FLOP/byte against an H100 ridge point near 295, so it's bandwidth-bound by 300×. Fewer weight bits is fewer bytes is more tokens per second, roughly linearly — until batch size pushes you over the ridge, where the dequant becomes pure overhead."

### Why are activations so much harder to quantize than weights? Be specific about the mechanism.

Because transformer activations contain **systematic, persistent, massive per-channel outliers**, and they emerge as a function of model scale.

The empirical finding, first characterised carefully in Dettmers et al. (2022), is that beyond roughly 6.7B parameters a small number of hidden dimensions — often on the order of 0.1% of channels, sometimes literally a handful out of 4,096 — carry activation magnitudes 20× to 100× larger than every other channel. Crucially they are the *same* channels across tokens and across sequences: it is not random noise, it is a structural property the model learned. These outlier features are load-bearing; ablating them destroys the model. Later work connected them to the attention mechanism's need for a "no-op" — attention sinks and massive-activation dimensions that let a head attend nowhere by dumping probability mass, which requires a very large value in a specific dimension.

Now put that through the quantization equation. A per-tensor int8 scale is set by max|x|. If one channel is 80× the rest, then `scale = 80·typical/127`, and every ordinary activation lands in the first two integer levels. You have thrown away 6 of your 8 bits on the 99.9% of the tensor that carries the semantics. Weights have no such structure — a weight matrix's max-to-median ratio is typically under 10×, and the distribution is unimodal.

The second reason activations are harder: **you cannot solve them offline.** A weight is a fixed number; you can run an optimiser against it for an hour. An activation is computed at runtime from an input you have not seen. Your only tools are (a) static ranges from a calibration set — which will be wrong on out-of-distribution input, (b) dynamic per-token scales computed on the fly — which costs a reduction over the hidden dim on every layer, or (c) restructuring the model so outliers stop existing, which is what SmoothQuant and QuaRot do.

**🔍 Failure taxonomy — how activation quantization breaks:**
1. *Overflow on a rare input.* Static ranges calibrated on English prose, a user pastes base64 or CJK text, an activation exceeds the calibrated max, clamps, and the output degenerates. Detect: log per-layer clamp-rate as a metric; a nonzero clamp rate on a healthy model is a bug.
2. *Uniform mush.* Ranges are correct but per-tensor, so the bulk quantizes to 2 bits of effective resolution. Detect: perplexity looks only slightly worse but instruction-following and JSON validity collapse.
3. *Layer-specific blowup.* The outliers concentrate in a few layers (often the down-projection input after the MLP). Detect: per-layer cosine similarity between fp16 and quantized activations; you will see two or three layers at 0.7 while everything else is 0.999.

### Explain LLM.int8() — what it does, and why it's more of a landmark than a production default now.

LLM.int8() is the method that named the outlier problem and then routed around it rather than solving it. **📄 Paper:** Dettmers, Lewis, Belkada, Zettlemoyer (2022), "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" — introduced mixed-precision decomposition and, more importantly, established that emergent outlier features are what makes naive int8 fail beyond ~6.7B parameters.

The mechanism is a decomposition. For each matmul `XW`, identify the columns of `X` (i.e. hidden dimensions) whose absolute magnitude exceeds a threshold — the paper uses 6.0. Split the multiplication into two: the outlier columns and their corresponding weight rows are multiplied in **fp16**, while the remaining ~99.9% of columns are multiplied in **int8** with per-row (per-token) and per-column (per-output-channel) scales. Sum the two partial results. Because the outlier set is tiny, the fp16 part is a skinny GEMM and the memory saving is close to the full 2×.

Vector-wise scaling is the other half of the contribution: instead of one scale per tensor, one scale per row of X and one per column of W, so the dequantization is an outer product of two scale vectors applied in the epilogue. That alone removes a lot of the per-tensor damage.

Why it's no longer the default: the decomposition is *slow*. You pay a runtime pass to find outliers, a scatter/gather to split the tensor, two GEMMs of awkward shapes, and a merge. In `bitsandbytes` the practical result was often **slower than fp16** for inference on large models, sometimes substantially, despite halving memory. It solved "the model doesn't fit" but not "the model is slow." SmoothQuant and later rotation-based methods make the whole tensor quantizable so you get one clean int8 GEMM, and fp8 on Hopper sidesteps the problem with per-element exponents. LLM.int8() lives on today mostly as `load_in_8bit=True` for research convenience — fitting a model on a smaller card for experimentation — not as a serving format.

**🗣 Say this in the room:** "LLM.int8 is the paper that discovered emergent outlier features. It handles them by pulling ~0.1% of channels out into an fp16 GEMM and int8-ing the rest. It gets you the memory but historically it's slower than fp16 because of the decomposition overhead, so I'd use it to fit a model for experimentation, not to serve one."

### Explain NF4 and what QLoRA actually contributed. Why is a "NormalFloat" better than int4 for weights?

NF4 starts from the observation I made earlier: pretrained weights within a block are approximately zero-centred Gaussian. If your data is Gaussian, a *uniform* 16-level grid is provably not the best allocation of levels — you want more levels packed where the density is high, near zero, and fewer out in the tails. NF4 is exactly that: a fixed, non-uniform 16-level codebook whose levels are placed at the quantiles of a standard normal distribution, so each of the 16 bins receives approximately equal probability mass under a Gaussian. Then within each block you normalise by the block's absmax so the data is unit-scaled and the codebook applies.

**📄 Paper:** Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023), "QLoRA: Efficient Finetuning of Quantized LLMs" — introduced NF4, double quantization, and paged optimizers, and demonstrated fine-tuning a 65B model on a single 48 GB GPU with fp16-comparable quality.

Two supporting tricks matter. **Double quantization**: with block size 64 and an fp32 absmax per block, the scales themselves cost 32/64 = 0.5 bits per weight, which is 12.5% overhead on a 4-bit format. So QLoRA quantizes the scales too — groups of 256 scales get quantized to 8-bit with their own fp32 scale — bringing the overhead down to roughly 8/64 + 32/(64·256) ≈ 0.127 bits per weight. That is a real ~0.37 bits/weight saved, which on a 65B model is about 3 GB. **Paged optimizers** use unified memory so optimizer states page to CPU on a gradient-checkpointing spike instead of OOM-ing.

The architectural point of QLoRA, and the one people miss: the base weights are frozen in NF4 and **never updated**; the LoRA adapters are bf16 and take all the gradient. The forward pass dequantizes NF4 to bf16 on the fly per tile. So you are not training a quantized model — you are training a small full-precision correction on top of a quantized, frozen backbone. That is why it works so well and why the resulting quality tracks full fine-tuning closely.

**⚠ Trap:** NF4 is a *fine-tuning* format, not a serving format. Its kernels are optimised for the QLoRA training loop, not for high-throughput batched inference, and it typically loses to a proper GPTQ/AWQ/Marlin int4 kernel on serving throughput by a wide margin. The rule I enforce: fine-tune with QLoRA if you must, then **merge the adapter into fp16 and re-quantize with your serving method** before deploying. Shipping the NF4 checkpoint to vLLM because "it's already 4-bit" is a real and repeated mistake.

### Walk me through GPTQ. What is the Hessian doing there, and why does the order of quantization matter?

Mental model: naive round-to-nearest quantizes each weight independently and accepts the error. GPTQ quantizes weights **one at a time and then updates all the not-yet-quantized weights in the same row to compensate for the error just introduced.** It is Gauss-Seidel elimination applied to the layer's reconstruction problem. That is the whole idea; everything else is making it fast enough to run on a 175B model.

Formally, for a linear layer with weight `W` and calibration inputs `X`, you want to minimise `‖WX − ŴX‖²` over quantized `Ŵ`. Expand it and the curvature of that objective with respect to `W` is `H = 2XXᵀ` — the second-order information, a `K×K` matrix over the input dimension. This is the layer-wise Hessian, and it is *not* the Hessian of the training loss; it is the Hessian of the local reconstruction error, which is what makes it cheap and exact.

The classical result says: if you quantize weight `i` and incur error `δ_i = w_i − quant(w_i)`, the optimal compensating update to the remaining weights is `δ · H⁻¹_{:,i} / [H⁻¹]_{ii}`, and the resulting increase in loss is `δ_i² / [H⁻¹]_{ii}`. That lineage runs Optimal Brain Damage (LeCun, 1990) → Optimal Brain Surgeon (Hassibi & Stork, 1993) → OBQ (Frantar & Alistarh, 2022) → GPTQ.

**📄 Paper:** Frantar, Ashkboos, Hoefler, Alistarh (2023), "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" — made second-order PTQ tractable at 175B scale, replacing OBQ's per-row greedy ordering with three engineering changes: (1) quantize all rows in the *same fixed column order* so one Cholesky factorisation of `H⁻¹` is shared across all rows; (2) process in lazy blocks of 128 columns, applying updates to the far block only at block boundaries, which turns a memory-bound elementwise update into a GEMM; (3) add dampening `H += λ·mean(diag(H))·I` for numerical stability. Result: 175B in ~4 GPU-hours instead of weeks.

On ordering: the plain algorithm goes left to right through columns. The `act-order` / `desc_act` variant sorts columns by `diag(H)` descending — quantize the highest-activation-energy columns first, while the full error budget of all remaining columns is available to compensate. This measurably helps quality, especially at group sizes ≥128, and it historically cost throughput because the permutation breaks group contiguity along K. Modern kernels (Marlin-family) handle this by materialising the permutation into the packed weights.

**⚠ Trap:** GPTQ minimises **layer-wise activation reconstruction error on your calibration set**. It does not minimise end-to-end loss, it does not know about your task, and it will happily overfit to the calibration distribution. Errors also compound layer to layer — GPTQ quantizes layer `n` using the *quantized* outputs of layers `1..n−1` if you implement it correctly (sequential calibration), and using fp16 activations if you implement it lazily, and the lazy version is worse in a way that only shows up deep in the network.

### Now AWQ. It's a different philosophy from GPTQ — articulate the difference and the mechanism.

AWQ's thesis is: **weights are not equally important, and importance is determined by the activations they multiply, not by the weight magnitudes themselves.** A weight in a channel that receives large activations contributes proportionally more to the output; protect those. GPTQ says "let me spread the error optimally"; AWQ says "let me make the important weights not have error in the first place."

**📄 Paper:** Lin, Tang, Tang, Yang, Dang, Han et al. (2023/2024), "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" — showed that protecting ~1% of salient weight channels, selected by *activation* magnitude, recovers most of the quantization loss without any backpropagation or reconstruction, and that you can do it with pure scaling rather than mixed precision.

Mechanism, in two moves. First, run calibration data through the layer and compute the average magnitude per input channel of `X`. The channels with the largest values are salient. Second — and this is the clever part — rather than keeping those weight channels in fp16 (which creates ragged mixed-precision kernels, exactly LLM.int8()'s performance problem), **scale them up before quantizing and scale the activations down by the same factor.** For a diagonal `s`, `(X · diag(s)⁻¹)(diag(s) · W)` is mathematically identical to `XW`, but `diag(s)·W` puts the salient weight channels higher in their quantization group's range, so they land on grid points with lower relative error. The `diag(s)⁻¹` on the activation side is folded into the *preceding* layer's weights (into the LayerNorm's affine parameters or the previous linear's output columns), so it costs zero at runtime.

The scale is chosen by a grid search: `s = mean|X|^α` per channel, sweeping `α ∈ [0,1]` and picking the α minimising output MSE for that layer. α=0 is no scaling, α=1 is full activation-proportional scaling.

Practically: AWQ requires no backprop and no Hessian inverse, so it is faster to run than GPTQ and less prone to numerical blowups; it's generally regarded as slightly more robust to calibration-set choice because it uses only per-channel activation *magnitudes* rather than a full covariance; and it tends to generalise better to instruction-tuned models. GPTQ with act-order and group 128 is often marginally better on raw perplexity. **The honest answer in an interview is that the gap between well-tuned GPTQ and well-tuned AWQ at W4A16 is small enough that engine kernel support should drive the choice, not the paper.**

**🗣 Say this in the room:** "GPTQ is second-order error compensation — quantize a column, push the error into the columns you haven't done yet, using the layer Hessian XXᵀ. AWQ is activation-aware channel scaling — find the ~1% of input channels with the biggest activations, scale those weight channels up before quantizing and fold the inverse into the previous layer, so the important weights get better grid resolution for free. At 4-bit group-128 they land within noise of each other; I'd pick on kernel support in my serving engine."

### SmoothQuant — what problem does it solve that GPTQ and AWQ don't?

SmoothQuant targets **W8A8**, and that is the distinction. GPTQ and AWQ are weight-only methods: they leave activations in fp16, so they help decode bandwidth but never light up the int8 math units. SmoothQuant's goal is to make the *activations* quantizable so you get a real int8 GEMM.

**📄 Paper:** Xiao, Lin, Seznec, Wu, Demouth, Han (2023), "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" — introduced offline migration of activation quantization difficulty into the weights, enabling accurate W8A8 on models up to 530B without retraining.

The observation: activations are hard to quantize (outlier channels) and weights are easy (flat distribution). Those are complementary, and the mathematical identity `Y = XW = (X·diag(s)⁻¹)·(diag(s)·W)` lets you move difficulty between them. Choose a per-input-channel smoothing factor

```
s_j = max|X_j|^α / max|W_j|^(1−α)
```

with migration strength α, typically 0.5. At α=0.5 the resulting activation range and weight range are balanced — you have flattened the activation outliers by dividing them down, at the cost of stretching a few weight channels up, and the weights had range to spare. `diag(s)⁻¹` is folded into the preceding LayerNorm's scale parameter or previous linear layer offline, so there is no runtime cost. Now both operands are per-tensor or per-token int8-quantizable and you run a genuine int8 tensor-core GEMM.

Note the structural similarity to AWQ: both use a diagonal reparameterisation folded into the previous op. The difference in intent is the tell — AWQ picks `s` to protect salient *weights* under weight-only quantization; SmoothQuant picks `s` to balance the *ranges* of both operands so both can be int8.

**⚠ Trap:** α is not a universal constant. The paper uses 0.5 as a default but explicitly notes that models with more extreme outliers (they cite the larger OPT models) need higher α to push more difficulty into the weights. Ship α=0.5 on a model whose outlier profile you never measured and you can land in a regime where you've now broken the weights *and* not fixed the activations. The right practice is a small sweep, α ∈ {0.5, 0.65, 0.8}, scored on your eval — 30 minutes of GPU time that repeatedly saves a bad deploy.

**💰 Math:** the payoff for getting to W8A8 on Hopper. An H100 does roughly 990 dense bf16 TFLOPS and roughly 1,980 fp8/int8 TFLOPS — a 2× math ceiling. In a prefill-heavy workload (say a RAG service with 8k-token contexts and 200-token answers, so ~97% of FLOPs are prefill), moving from bf16 to fp8 compute takes prefill of a 70B model from 2 × 70e9 × 8192 = 1.15 PFLOP per request at ~700 TFLOPS achieved (1.64 s) to ~1,300 TFLOPS achieved (0.88 s). That is ~0.76 s off TTFT per request. At 200k requests/day and $3/GPU-hour, the freed GPU-seconds are 200,000 × 0.76 = 152,000 s = 42 GPU-hours/day = **$126/day, ~$3.8k/month** — before counting the latency win, which is usually the thing the product actually cares about.
