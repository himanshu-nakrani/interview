### Derive the full fine-tuning memory equation for me. Term by term.

This is the equation the whole section exists to make automatic, so I want to be able to write it without thinking. Under standard bf16 mixed-precision training with AdamW, for a model of `P` parameters:

| Term | Bytes per parameter | Why |
|---|---|---|
| Model weights (bf16) | 2 | what the forward pass reads |
| Gradients (bf16) | 2 | one per trainable weight |
| FP32 master weights | 4 | bf16 has ~8 bits of mantissa; accumulating tiny updates into bf16 silently rounds them to zero |
| Adam `exp_avg` (m, fp32) | 4 | first moment |
| Adam `exp_avg_sq` (v, fp32) | 4 | second moment |
| **Total static** | **16 bytes/param** | |

Then on top of the static `16P`:

```
M_total = 16·P  +  A(batch, seq, model)  +  C_cuda  +  F_frag
```

`A` is activation memory retained for the backward pass, `C_cuda` is the CUDA context plus cuBLAS/NCCL workspaces (budget 1–2 GB per process, it is not negligible), and `F_frag` is allocator fragmentation — I budget 10% headroom because PyTorch's caching allocator will happily hold 6 GB of unusable free blocks and then OOM.

**Worked instances, the ones to have memorized:**

- **70B:** `16 × 70e9 = 1.12e12 bytes = 1,120 GB` of static state alone. That is 14 × 80 GB cards before a single activation. In practice a 70B full fine-tune is a 16–32 GPU job with FSDP/ZeRO-3 sharding the optimizer states.
- **8B:** `16 × 8e9 = 128 GB`. Does **not** fit on one 80 GB H100. This surprises people, and it is the number that kills the "just full fine-tune the 8B" suggestion in most rooms.
- **3B:** `16 × 3e9 = 48 GB` + activations ≈ 55–65 GB. Fits on one 80 GB card. This is the boundary.
- **1B:** `16 GB` + activations. Comfortable on a 24 GB consumer card with checkpointing.

**Variants worth naming, because interviewers probe them.** 8-bit Adam (bitsandbytes `AdamW8bit`) stores `m` and `v` as int8 with block-wise scales: `2 + 2 + 4 + 1 + 1 = 10 bytes/param`, dropping 8B from 128 GB to 80 GB — still not enough, which is the point. Pure-bf16 optimizer with no fp32 master gets you to `8 bytes/param` but risks stalled updates on small gradients. ZeRO-3 / FSDP shards weights, grads and optimizer states across `N` ranks, so per-GPU static becomes `16P/N` plus a full-precision working copy of whatever layer is currently gathered.

**🗣 Say this in the room** when someone asks "can I fine-tune a 70B on one 80GB card?": "Full fine-tuning needs 16 bytes per parameter — bf16 weights, bf16 grads, fp32 master, and two fp32 Adam moments — so 70B is 1.12 terabytes of static state. Not on one card, not on eight. bf16 LoRA doesn't fix it either, because the frozen weights alone are 140 GB. QLoRA does: NF4 base is 35 GB, the rank-16 adapter and its optimizer state is about 3 GB, and with gradient checkpointing at 2k context you land near 50 GB. So: yes, but only with 4-bit."

### If LoRA freezes the weights, which of those terms actually go away — and which stubbornly do not?

The clean way to see it: LoRA deletes the **optimizer-side** terms and barely touches the **activation-side** terms. That asymmetry is the single most misunderstood thing about PEFT memory.

What goes away. Gradients, fp32 master weights and both Adam moments are allocated *per trainable parameter*. With LoRA that set shrinks from `P` to `P_lora ≈ 0.005·P`. For an 8B model at rank 16 on all linear layers, `P_lora = 41.9M`, so:

```
adapter params   (bf16): 41.9e6 × 2  =  84 MB
adapter grads    (bf16): 41.9e6 × 2  =  84 MB
fp32 master            : 41.9e6 × 4  = 168 MB
Adam m + v      (fp32) : 41.9e6 × 8  = 335 MB
----------------------------------------------
optimizer-side total                 ≈ 0.67 GB
```

versus `14 × 8e9 = 112 GB` for full fine-tuning. That is the win, and it is a factor of about 170.

The frozen base weights still cost `2P` = 16 GB in bf16, because you still have to run the forward pass. This is why LoRA alone does not put a 70B on one card: `2 × 70e9 = 140 GB`.

What does **not** go away: activations. You are training adapters in *layer 0*, so gradients must flow all the way back through every layer, which means the backward pass needs the same retained activations a full fine-tune needs — every softmax output, every SiLU input, every residual branch point. There is a modest saving (a frozen `nn.Linear` does not need to retain its input `x`, because you never compute `∂L/∂W`; you only need `W` to compute `∂L/∂x`), but the nonlinearities and normalizations dominate and they are unaffected.

**⚠ Trap — "LoRA fixes my OOM."** If you were OOMing on activations because you raised sequence length or batch size, LoRA changes nothing and you will OOM at the same place. The diagnostic is trivial: if your OOM happens at a stable point every step regardless of batch, it is static state and LoRA/quantization helps; if it moves with batch size or sequence length, it is activations and you need gradient checkpointing, a shorter context, or a fused cross-entropy. I have watched engineers spend a day switching to QLoRA to fix an activation OOM.

**⚠ Trap two — the logits tensor.** At vocabulary 128,256 and sequence 4,096 with batch 4, the output logits are `4 × 4096 × 128256 = 2.1e9` elements. In bf16 that is 4.2 GB, and the standard cross-entropy path upcasts to fp32 (8.4 GB) and retains a similar-sized buffer for the backward — you can be 20+ GB into your budget before touching a transformer layer. This scales with *vocabulary*, which people never think about, and it is why large-vocab models feel disproportionately expensive to fine-tune. The fix is a fused/chunked cross-entropy that never materializes the full logits (Liger Kernel and Unsloth both ship one); it is frequently the single largest memory win available in a fine-tuning stack.

### Explain gradient checkpointing and how it interacts with a LoRA run.

Gradient checkpointing is the classic time-for-space trade, and if you have ever chosen between caching a computed result and recomputing it on demand, you already have the model: instead of retaining every intermediate activation for the backward pass, you retain only *segment boundaries* and recompute the interior during backward.

For a transformer, the natural segment is one decoder block. You keep the block's input hidden state and throw away everything inside. Activation memory drops from "everything each block produced" to:

```
A_checkpointed ≈ batch · seq · d_model · dtype_bytes · n_layers  +  (peak activations of ONE block)
```

**Llama-3-8B, batch 4, seq 2048, bf16:** `4 × 2048 × 4096 × 2 = 67 MB` per boundary × 32 layers = **2.1 GB**, plus one block's recompute peak. Without checkpointing the same configuration retains roughly an order of magnitude more. The cost is one extra forward pass through the recomputed segments during backward — canonically **~30% more step time** (a forward is roughly half the cost of a backward, so adding one forward to a forward+backward is ≈ +33%).

Three interactions with LoRA that are worth having at your fingertips:

**1. The `requires_grad` bug that silently kills LoRA runs.** PyTorch's checkpointing wrapper only builds a graph through a segment if that segment's *inputs* require grad. In a LoRA setup the embedding layer is frozen, so the hidden state entering block 0 has `requires_grad=False`, the checkpointed segments produce no graph, and **your adapters receive no gradient at all**. Symptom: loss is flat, `grad_norm` logs as 0 or `nan`, nothing errors. The fix is `model.enable_input_require_grads()` (which registers a forward hook making the embedding output require grad) or `use_reentrant=False` on the checkpoint call. `peft.prepare_model_for_kbit_training()` does this for you — which is exactly why you should call it rather than hand-rolling.

**2. `use_cache` must be off.** The KV cache is an inference structure; leaving `config.use_cache=True` during training allocates cache tensors you never read and conflicts with checkpointing. HF warns and disables it, but if you built a custom loop, set it yourself.

**3. Checkpointing composes with QLoRA to create *spiky* memory.** Every recomputed block must dequantize its NF4 weights into bf16 to run the forward again, so the peak is bursty: a transient `d_model × d_ffn × 2` bf16 buffer per projection, allocated and freed at high frequency. This is precisely the workload that fragments the caching allocator, and it is the reason QLoRA's paged optimizer exists.

**📐 Numbers you must know:** gradient checkpointing costs about **+30% step time** and typically buys **5–10× activation memory reduction** at long sequence lengths. My default is on for anything above 3B or above 2k context, off for small models where you are throughput-bound and memory-rich.

### Walk me through QLoRA. What are its four ingredients and what does each one solve?

QLoRA is best understood as four independent memory optimizations that happen to compose, and the interviewer usually wants all four named.

**📄 Paper:** Dettmers, Pagnoni, Holtzman & Zettlemoyer (2023), *QLoRA: Efficient Finetuning of Quantized LLMs* — showed that a 4-bit NF4-quantized frozen base with bf16 LoRA adapters could fine-tune a 65B model on a single 48 GB GPU while reporting parity with 16-bit fine-tuning on their evaluations.

**1. NF4 — 4-bit NormalFloat quantization of the frozen base.** This is the big one: `2 bytes/param → 0.5 bytes/param`, a 4× reduction on the term that dominates. Details in the next question.

**2. Double quantization.** The quantization constants themselves are quantized. Small in isolation, ~3 GB on a 65B model, which is not small when 3 GB is the difference between fitting and not.

**3. Paged optimizers.** Optimizer state allocated in NVIDIA unified memory, so it can page out to host RAM under pressure instead of throwing OOM. The direct analogue is OS demand paging with the GPU as physical memory and host DRAM as swap — same mechanism, same trade (you eat PCIe latency instead of dying).

**4. bf16 compute with LoRA adapters in bf16.** The base is *stored* in NF4 but never *computed* in NF4: each matmul dequantizes the needed block to bf16 on the fly, does the GEMM in bf16, and discards the dequantized copy. The adapters are always bf16 and always trained normally.

That fourth point is the one candidates get wrong most often, so state it explicitly: **QLoRA is 4-bit storage, 16-bit compute.** There is no 4-bit arithmetic. The dequantization is fused into the kernel, and it costs throughput — a QLoRA step is meaningfully slower than a bf16 LoRA step on the same hardware (budget 20–40%, and measure) because you are paying dequantization on every weight read. You are trading time for the ability to fit at all.

**🗣 Say this in the room:** "QLoRA is NF4 4-bit storage for the frozen base, double-quantized scaling constants, paged optimizer state, and bf16 compute with bf16 LoRA adapters. The base is never computed in 4 bits — each block is dequantized to bf16 inside the kernel. That is why it fits a 70B on one 80 GB card and why it is slower per step than bf16 LoRA."

### What exactly is NF4, and why not just use INT4?

INT4 quantizes a range into 16 *evenly spaced* levels. That is optimal if your values are uniformly distributed. Neural network weights are not uniformly distributed — within a block they are approximately zero-centred and roughly Gaussian. Uniform levels therefore waste most of their codebook on the sparse tails while the dense region around zero, where nearly all the weights actually live, gets coarse resolution.

NF4 — 4-bit NormalFloat — picks its 16 levels to be **quantiles of a standard normal distribution**, so each bin carries approximately equal probability mass. Dense region near zero gets closely-spaced levels; the tails get widely-spaced ones. Under the assumption that the weights are normally distributed, this is the information-theoretically optimal 4-bit code, which is the paper's actual claim. It is also symmetric and includes an exact zero, which matters because exact zero appears frequently and you do not want to introduce error into it.

Mechanically it is **block-wise**:

```
for each contiguous block of 64 weights:
    c = absmax(block)                 # one fp32 scale per block
    q = quantize_to_NF4(block / c)    # normalize into [-1, 1], map to nearest of 16 quantiles
store: q (4 bits each) + c (fp32)
dequantize: w_hat = nf4_lookup(q) * c
```

Block size 64 is the choice that matters. Larger blocks mean fewer scale constants (less overhead) but one outlier weight in the block inflates `absmax` and crushes the resolution of all 63 of its neighbours. Smaller blocks are more robust but the scale overhead grows. 64 is the empirical sweet spot.

**📐 The overhead arithmetic:** one fp32 scale per 64 weights is `32 bits / 64 weights = 0.5 bits/weight` — a 12.5% overhead on top of a 4-bit payload, taking you from a nominal 4.0 to an effective 4.5 bits/param. That overhead is precisely what double quantization exists to attack.

**⚠ Trap:** NF4 is a *lossy compression of the base model*, applied before you train anything. Your QLoRA run starts from a base that is already slightly worse than the fp16 base, and no amount of adapter training is guaranteed to recover the specific capabilities the quantization damaged. On easy tasks this is invisible; on hard reasoning and on long-tail factual recall it is measurable. So when someone reports "QLoRA matched full fine-tuning," check what was evaluated. My rule: **if I can afford bf16 LoRA, I use bf16 LoRA.** QLoRA is what I reach for when the model does not otherwise fit.

### Derive the double-quantization saving. Show me the bits.

Double quantization quantizes the quantization constants — the fp32 `absmax` values NF4 produces one of per 64-weight block.

**First level, before double quantization:**
```
one fp32 constant per 64 weights = 32 bits / 64 = 0.5 bits per weight
```

**Second level:** take those fp32 constants and quantize them, 8-bit, in blocks of 256 constants, with their own fp32 second-level scales.

```
8-bit quantized constants:  8 bits / 64 weights            = 0.125    bits/weight
second-level fp32 scales:   32 bits / (64 × 256) weights   = 0.001953 bits/weight
-----------------------------------------------------------------------------
total constant overhead                                    ≈ 0.127    bits/weight
```

**Saving:** `0.5 − 0.127 = 0.373 bits per parameter`.

At 65B parameters: `65e9 × 0.373 / 8 = 3.03e9 bytes ≈ 3.0 GB` — which matches the figure reported in the QLoRA paper. At 70B: `70e9 × 0.373 / 8 = 3.26 GB`.

So the effective storage cost of a double-quantized NF4 weight is about **4.127 bits/param**, versus 4.5 without. In bytes-per-param terms: `4.127 / 8 = 0.516 bytes/param`.

Is 3 GB worth a second quantization scheme? On an 80 GB card holding a 70B model at 36 GB, 3 GB is 3.7% of the card — roughly one extra sequence of activations, or 4k more tokens of context in your training batch. On a 48 GB card it is the difference between running and not. That is the honest framing: double quantization is not a quality technique, it is a fitting technique, and it costs a small amount of dequantization work (two lookups instead of one) on the critical path.

**🗣 Say this in the room:** "Double quantization takes the per-64-block fp32 scales — which cost 0.5 bits per weight — and stores them as 8-bit values in blocks of 256, with a second level of fp32 scales. That drops constant overhead from 0.5 to about 0.127 bits per weight, roughly 3 GB on a 65B model. It buys headroom, not accuracy."

### What problem do paged optimizers solve, and when would you actually see them fire?

They solve *spikes*, not *averages*, and that distinction is the whole answer.

Your steady-state memory in a QLoRA run might be a comfortable 55 GB on an 80 GB card. But memory in a training loop is not steady. Gradient checkpointing recomputes a block, which transiently dequantizes NF4 weights to bf16. A long sequence arrives in a batch of otherwise-short ones and the attention and MLP intermediates balloon. The allocator has cached free blocks in the wrong size classes and cannot satisfy a large contiguous request. Any one of these produces a momentary spike that exceeds capacity — and CUDA's answer to that is a hard OOM that kills your run at step 40,000.

Paged optimizers allocate the optimizer state (Adam's `m` and `v`) using **NVIDIA unified memory**, which means those pages have a valid host-RAM backing. Under GPU memory pressure the driver evicts optimizer pages to host DRAM and faults them back when the optimizer step touches them. This is demand paging, with the GPU as physical memory and host RAM as the swap device — the identical mechanism to `mmap`-backed anonymous memory under pressure, with the identical trade-off: you survive, and you pay PCIe latency for the pages you touch.

Why optimizer state specifically? Because of *access pattern*. `m` and `v` are touched exactly once per step, at the end, all at once, and never during forward or backward. That is the most page-out-friendly tensor in the whole training loop — it is cold for 95% of wall-clock time. Weights and activations are hot and would thrash if you paged them.

**The cost when it fires.** PCIe Gen4 x16 is roughly 25–32 GB/s of practical bidirectional bandwidth (**📅 Volatile:** Gen5 systems roughly double this — check the actual host). Paging a 3 GB optimizer state out and back once per step costs `2 × 3 / 28 ≈ 0.21 s`. If your step time is 1.5 s, that is a 14% throughput hit — annoying, survivable. If paging fires on *every* step because you are genuinely over budget, you have not solved a problem, you have converted a crash into a slow crawl. **That is the operational rule: paging firing occasionally is insurance working; paging firing every step means reduce batch size or sequence length.** Watch it with `nvidia-smi` reported memory versus `torch.cuda.max_memory_allocated()` — a growing gap between allocator peak and device peak is the tell.

### Can I fine-tune a 70B on one 80GB card? Do the full arithmetic in front of me.

Yes, with QLoRA, and here is the budget line by line. Llama-3-70B: 80 layers, `d_model = 8192`, 64 query heads / 8 KV heads with head-dim 128 (so `k_proj`/`v_proj` are `8192 → 1024`), FFN intermediate 28,672, vocabulary 128,256.

**1. Base weights, NF4 with double quantization.**
```
70e9 params × 4.127 bits / 8 = 36.1 GB
```

**2. LoRA adapters, r=16, all seven linear projections.** Per layer:
```
q: 16·(8192+8192) = 262,144      o: 262,144
k: 16·(8192+1024) = 147,456      v: 147,456
attention subtotal               = 819,200
gate: 16·(8192+28672) = 589,824  up: 589,824   down: 16·(28672+8192) = 589,824
MLP subtotal                     = 1,769,472
per layer                        = 2,588,672
× 80 layers                      = 207.1M trainable parameters   (0.30% of 70B)
```
Optimizer-side memory: `207.1e6 × (2 param + 2 grad + 4 master + 4 m + 4 v) = 207.1e6 × 16 = 3.31 GB`.

**3. Activations, gradient checkpointing on, batch 1, seq 2048, bf16.**
```
boundary states: 1 × 2048 × 8192 × 2 = 33.5 MB per layer × 80 = 2.68 GB
one block's recompute peak (attention + MLP intermediates)  ≈ 0.5 GB
subtotal ≈ 3.2 GB
```

**4. Logits and loss.** `1 × 2048 × 128256 × 4 bytes (fp32) = 1.05 GB`, roughly doubled for the backward buffer = **2.1 GB**. With a fused chunked cross-entropy this drops to near zero — and given how tight this budget is, I would enable it.

**5. CUDA context, cuBLAS workspaces, kernel scratch:** ≈ **1.5 GB**.

**Sum:** `36.1 + 3.31 + 3.2 + 2.1 + 1.5 = 46.2 GB`. Add 10% fragmentation headroom → **≈ 50.8 GB on an 80 GB H100.**

That leaves ~29 GB, which you spend on sequence length or batch. Activations scale linearly in `batch × seq`, so batch 1 @ 8k, or batch 4 @ 2k, both land near `3.2 × 4 ≈ 12.8 GB` of activations → total ≈ 66 GB with headroom. Comfortable. **Batch 8 @ 4k would be 25.6 GB of activations → ~79 GB. That is where you fall off the cliff.**

**Contrast, to show you know the alternatives:**
- Full fine-tune 70B: `16 × 70e9 = 1,120 GB` → 16+ GPUs with FSDP.
- bf16 LoRA 70B: `140 GB` of frozen weights alone → 2 cards minimum, realistically 4.
- QLoRA 70B: **fits on one.**

**💰 Math:** at ~$3/hr for an on-demand H100 (**📅 Volatile:** verify current pricing), a 3-hour QLoRA run on 70B costs **$9**. Two 8×H100 nodes for a 70B full fine-tune at ~$24/hr/node for 12 hours is `2 × 24 × 12 = $576`, a 64× difference — and that is before you account for the multi-node orchestration engineering. The economic argument for PEFT at 70B is not close.

**🏋 Drill:** 8 minutes, unaided, on paper. Given Qwen-style 32B (64 layers, `d_model` 5120, 40 query / 8 KV heads at head-dim 128, FFN 27,648, vocab 152k), compute: (a) full-FT static memory, (b) rank-32 all-linear adapter parameter count, (c) QLoRA total at batch 2 seq 4096 with checkpointing, (d) whether it fits on a 48 GB L40S. **Pass criterion:** every number within 10% of a careful recomputation, and you correctly identify which single term you would attack first if it did not fit.

### You trained an adapter with QLoRA. Now you want to merge it into fp16 weights for serving. What goes wrong?

This is the trap in this section, and it is worth being precise about because the failure is quiet.

During QLoRA training, every forward pass computes `dequant(W_nf4) · x + (α/r)·BA·x`. The adapter's gradients are therefore computed against `dequant(W_nf4)`, **not** against the original `W_fp16`. Define the quantization error:

```
ε = W_fp16 − dequant(W_nf4)
```

`ε` is not noise from the adapter's point of view — it is a fixed, structured perturbation present in every single training step. The optimizer will happily spend part of the adapter's rank-`r` budget **compensating for `ε`**, because doing so lowers the loss. Your learned `BA` is therefore something like "the task update, plus a partial correction for this specific quantization error."

Now merge into fp16:

```
W_merged = W_fp16 + (α/r)·BA
         = dequant(W_nf4) + ε + (α/r)·BA
```

You have added back the `ε` that the adapter was trained to cancel. The quantization correction is now applied on top of weights that never had the error, so it is pure, structured damage. The magnitude depends on how much of the rank budget got spent on `ε`, which grows with lower rank, longer training, and worse quantization. The symptom is a small, uniform, hard-to-attribute quality drop — a couple of points on your eval, no crash, no warning, and it looks exactly like "fine-tuning didn't help much."

**Your four options, in the order I would consider them:**

1. **Serve unmerged, on the NF4 base.** The inference-time computation then matches training exactly. This is the highest-fidelity option and it is what I default to. It costs the unmerged adapter's runtime overhead, which is small.
2. **Merge into the dequantized base and serve in bf16.** `W_serve = dequant(W_nf4) + (α/r)·BA`. Faithful to training, but you are now serving a bf16 model whose weights carry the NF4 error baked in — worse base quality than the true fp16 base, though *consistent* with what the adapter expects.
3. **Merge and re-quantize for serving.** Merge into dequantized weights, then quantize the merged result (to NF4, AWQ, GPTQ, FP8, whatever your engine wants). Re-quantization introduces *new* error on top of the merged weights, so measure. This is common and workable, but it must be evaluated, not assumed.
4. **Use a quantization-aware initialization (LoftQ) so the adapter never absorbs `ε` in the first place.** Best structural fix; see the next question.

**⚠ Trap, stated as the reviewable rule:** *the base weights you merge into must be the same base weights you trained against.* I have seen a team train QLoRA on an NF4 base, merge into the original fp16 safetensors, deploy, watch quality drop ~3 points on their eval, and spend a week blaming the dataset. The two-line check that would have caught it: run your eval on (a) the unmerged NF4 + adapter path and (b) the merged fp16 path, on 200 examples, before shipping. If they disagree, you have this bug.

### What is LoftQ, and when should I bother with it?

LoftQ addresses the previous question at the root, and the framing is elegant enough to be worth stating carefully.

Vanilla QLoRA initializes `B = 0`, so at step 0 the model computes `dequant(W_nf4)·x` — you begin from the *quantized* model's quality, which is strictly below the fp16 model's. The adapter's first job is therefore to claw back quantization damage before it can start learning your task. That is a waste of capacity and of steps, and it is the mechanism behind the merge problem above.

LoftQ instead solves, jointly, for a quantized base `Q` and low-rank factors `A, B` that together approximate the original weights:

```
minimize over Q (quantized) and A, B (rank r):   ‖ W_fp16 − Q − B·A ‖_F
```

It does this by alternating: quantize the residual `W − BA` to get `Q`, then take a rank-`r` SVD of the residual `W − Q` to get new `A, B`, and iterate a few times. The result is an initialization where `Q + BA ≈ W_fp16` at step 0, so **training starts from approximately full-precision quality rather than from quantized quality**, and the adapter is not carrying quantization error as a hidden tax.

**📄 Paper:** Li et al. (2023), *LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models* — alternating quantization and low-rank factorization to initialize the adapter so that the quantized base plus adapter reconstructs the original weights, replacing QLoRA's zero-init in the low-bit regime.

**When it matters, and when it does not.** The reported gains concentrate at **aggressive quantization** — 2-bit and 3-bit, and to a lesser extent 4-bit at low rank. At NF4 with rank 16–64 on a well-behaved model, the quantization error is small enough that vanilla QLoRA is fine and LoftQ's benefit is within noise. So my rule: **NF4 + rank ≥ 16 → don't bother. Sub-4-bit, or rank ≤ 8, or a model known to quantize badly → LoftQ init is worth the extra preprocessing pass.**

The related idea worth knowing by name is **QA-LoRA** (quantization-aware LoRA), which arranges the adapter so the *merged* result can be quantized cleanly for deployment, targeting the merge problem rather than the initialization problem. Same family of concerns, different end of the pipeline.

**⚠ Trap:** LoftQ initialization is per-checkpoint preprocessing. If you swap the base model — a point release, a different quantization block size, a different set of target modules — the LoftQ init is invalid and must be recomputed. Caching a LoftQ init and reusing it across base versions is a silent correctness bug.

### Walk me through the actual training stack. bitsandbytes, PEFT, TRL, Unsloth — what do you use and why?

Here is the stack I would defend, layer by layer, with the honest role of each.

**`transformers` + `peft`** is the baseline and the thing I would ship at a company that values stability over the last 30% of throughput. `peft` gives you `LoraConfig`, adapter save/load, `merge_and_unload()`, and multi-adapter management. It is broadly correct, well-tested, and integrates with everything.

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(NAME, quantization_config=bnb, device_map="auto",
                                             attn_implementation="sdpa")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_rslora=False,
))
model.print_trainable_parameters()   # ALWAYS. this is your step-0 sanity check.
```

**`bitsandbytes`** is what actually implements NF4, double quantization, the paged optimizers and 8-bit Adam. It is a CUDA-kernel library with real version sensitivity — it must match your CUDA toolkit, and version skew between `bitsandbytes`, `torch` and the driver produces import-time errors or, worse, silently wrong numerics. Pin it.

**`trl`** gives you `SFTTrainer` (loss masking, packing, chat-template application) and the preference trainers. Use it rather than hand-rolling the collator; the completion-only loss masking is the part people get wrong.

**Unsloth** replaces the hot path with hand-written Triton kernels — fused RoPE, fused RMSNorm, fused cross-entropy, and a manual autograd implementation of the LoRA path that avoids materializing intermediates. Its headline claims are roughly 2× throughput and substantially lower memory versus the HF+PEFT baseline on single-GPU runs (**📅 Volatile:** the multiples move with every release; benchmark on your own model before repeating a number in an interview). The fused cross-entropy alone is often worth it at large vocabulary, per the logits arithmetic above. **Liger Kernel** is the other option in this space and composes with plain HF training. The trade-off is coverage: fused kernels support a specific list of architectures, and a model outside that list either falls back silently to the slow path or breaks.

**My decision rule.** Single GPU, one of the supported architectures, throughput matters: Unsloth or Liger. Multi-GPU, unusual architecture, or a stack that has to be maintained by people who did not build it: plain `transformers` + `peft` + `trl` + FSDP. **The rule I enforce in review: no fused-kernel library goes in without a numerical equivalence test** — same seed, same data, 50 steps, loss curves within a tight tolerance of the reference path. Fused kernels are exactly the kind of dependency that is 2× faster and 0.5% wrong.

### The run doesn't fit on one GPU even with QLoRA. How do you scale a LoRA fine-tune across eight cards?

First, be honest about which axis is binding, because the answer differs completely.

**If activations are binding** — you need 32k context, or a large batch for stable gradients — the cheapest fix is not more GPUs. It is gradient accumulation (`micro_batch=1`, `accum=64`), gradient checkpointing, a fused cross-entropy to kill the logits tensor, and sequence packing. I would exhaust those before adding a node, because every distributed strategy adds a communication tax and an operational surface.

**If the model itself does not fit**, you have two families.

**Data parallel with sharding (FSDP / DeepSpeed ZeRO-3).** Each rank holds `1/N` of the parameters and gathers the full weights of whatever layer it is currently executing, then frees them. For a LoRA run the interesting property is that the *shardable* state is mostly the frozen base — the optimizer states that ZeRO exists to shard are already tiny. So what you get from FSDP in a LoRA run is **weight sharding**, not optimizer sharding: 70B in bf16 is 140 GB, sharded over 8 cards is 17.5 GB per rank, which fits comfortably and lets you avoid QLoRA entirely. That is often the better trade — bf16 LoRA across 8 GPUs is faster per step and higher fidelity than QLoRA on one.

**Pipeline or tensor parallel.** Rarely what you want for a fine-tune. Tensor parallel adds an all-reduce inside every layer and wants NVLink; pipeline parallel introduces bubbles and needs careful microbatching. For fine-tuning workloads, sharded data parallel is almost always the right first answer, and saying so confidently is the correct signal.

**The FSDP + LoRA gotchas, which are real:**

1. **Mixed `requires_grad` within a shard.** FSDP flattens parameters into contiguous units, and historically a unit containing both frozen base weights and trainable adapter weights caused errors or wrong behaviour. `use_orig_params=True` is the mechanism that makes mixed-trainability wrapping work; set your auto-wrap policy at the decoder-layer boundary so each unit is coherent.
2. **FSDP + quantized base is genuinely awkward.** NF4 weights are a custom storage format with attached quantization state, and generic sharding logic does not automatically know how to split them. Support for FSDP+QLoRA exists (it was a notable 2024 engineering effort) but it is version-sensitive; if I can afford bf16 LoRA with FSDP instead, I take it and skip the whole class of problem.
3. **Checkpointing an adapter under FSDP.** You must gather the full state dict before saving, or you write eight shard files that no serving stack can load. Save the adapter with a full-state-dict context; the artifact should be a plain 84 MB safetensors file, identical to a single-GPU run's output.

**📐 Numbers you must know — the communication cost.** Sharded data parallel all-gathers each layer's weights on every forward and backward. For 70B bf16 that is roughly `140 GB` gathered per forward pass, spread across the step. Over NVLink at ~450 GB/s intra-node this is `0.31 s` of pure communication per pass; over 100 Gb/s Ethernet (12.5 GB/s) it is `11.2 s` — a 36× difference. **That single ratio is why fine-tuning jobs stay inside one node whenever possible**, and why an 8×H100 box with NVLink is qualitatively different from eight H100s in different racks. If someone proposes multi-node for a 70B QLoRA run that fits on one card, that is the number I would put on the whiteboard.

### Everybody says "just install flash-attn." What does it actually buy you, and why is installing it a running joke?

What it buys, first, because the number is startling.

Standard attention materializes the `[batch, heads, seq, seq]` score matrix in HBM. At batch 4, 32 heads, sequence 8192, bf16:

```
4 × 32 × 8192 × 8192 × 2 bytes = 17.2 GB   — per layer, transiently
```

That is unaffordable and it is why naive attention hits a wall around 4k context. FlashAttention never materializes it: it tiles the computation into blocks that fit in SRAM, uses the online-softmax recurrence to accumulate the correct normalization incrementally, and streams outputs. Memory goes from `O(seq²)` to `O(seq)` in the attention block, and because the kernel is IO-aware — it minimizes HBM round trips, which is the actual bottleneck, not FLOPs — it is also *faster*, typically 2–4× on the attention portion at long sequence.

**📄 Paper:** Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — tiling plus online softmax to compute exact attention without materializing the score matrix, replacing memory-efficient-but-approximate attention variants with an exact one.

Note the word **exact**. This is not an approximation; the outputs are numerically equivalent up to floating-point reassociation. That matters because it means adopting it is not a quality decision.

Now the install reality, which is a genuine interview icebreaker because everyone has suffered it. `flash-attn` ships as a source distribution whose CUDA extensions must be compiled against your exact `torch` version, CUDA toolkit version, Python version and C++ ABI. `pip install flash-attn` without `--no-build-isolation` will pull a *different* torch into an isolated build environment and either fail or produce a binary linked against the wrong runtime. A from-source build takes 30–90 minutes and can need 10+ GB of RAM per parallel job; the standard mitigation is `MAX_JOBS=4` to keep it from OOMing the box. Prebuilt wheels exist but are indexed by the full `(torch, cuda, python, abi)` tuple and you must pick the exact matching one.

**The practical answer that shows judgment:** for most fine-tuning work you do not need the `flash-attn` package at all. PyTorch's `scaled_dot_product_attention` dispatches to a built-in FlashAttention backend, and `attn_implementation="sdpa"` in `transformers` gets you most of the benefit with zero build risk. Reach for the standalone package when you need something SDPA does not expose — variable-length packed sequences without padding, specific sliding-window or ALiBi variants, or the newer Hopper-optimized kernels. In containers, pin the wheel and build it once in the image; never let it compile at deploy time.

**⚠ Trap:** installing `flash-attn` but not passing `attn_implementation="flash_attention_2"` when loading the model. It compiles, it imports, nothing errors, and you are still running the SDPA or eager path. Verify by printing `model.config._attn_implementation` — actually check, do not assume.
