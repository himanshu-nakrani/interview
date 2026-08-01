### Walk me through fp8 training. What is per-tensor scaling and why is delayed scaling a thing?

fp8 is the first format where the numbers genuinely do not fit, so you cannot just cast and go — you have to carry a **scale factor per tensor** and manage it as state. That is the conceptual difference from bf16, and everything about the engineering follows from it.

Two fp8 formats exist, and the split is deliberate:

- **E4M3** — 4 exponent bits, 3 mantissa bits, max magnitude **448**, ~2 decimal digits of precision. Used for **forward** tensors: weights and activations, which are well-conditioned and benefit from the extra mantissa bit.
- **E5M2** — 5 exponent, 2 mantissa, max **57,344**. Used for **backward** tensors: gradients, whose dynamic range is enormous and whose precision matters less.

Transformer Engine's `Format.HYBRID` is exactly this pairing. **📄 Paper:** Micikevicius et al. (2022), *FP8 Formats for Deep Learning* — it standardized E4M3/E5M2 and the argument for using different formats in forward and backward.

**Per-tensor scaling.** Before casting a tensor `X` to fp8 you compute `amax = max(|X|)` and a scale `s = FP8_MAX / amax`, store `round_fp8(s·X)`, and record `s`. The GEMM runs on the scaled fp8 operands and its fp32 accumulator output is multiplied by `1/(s_A · s_B)` to recover the true result. Every fp8 tensor in the graph carries its own scale — hence "per-tensor."

**Why delayed scaling exists.** Computing `amax` requires a full reduction over the tensor *before* you can cast it, which means an extra memory pass over data you were about to stream into a GEMM anyway. On a 400 MB activation that is a real cost, and it serializes with the matmul. Delayed scaling avoids it: keep a rolling **amax history** of the last `N` steps (default 1024 in TE), compute this step's scale from `max(history)` with a safety margin, and update the history with the amax you observe *this* step for use next step. You get the scale for free, at the price of being one step stale.

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=1024,
                        amax_compute_algo="max")
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    out = te_model(x)          # only the GEMMs inside TE modules run in fp8
loss.backward()                # gradients in E5M2, master weights still fp32
```

The failure mode of delayed scaling is a **sudden distribution shift**: if a tensor's amax jumps by more than the safety margin between steps — which happens on a loss spike, an LR warmup discontinuity, or a data-distribution change — the stale scale overflows and you get inf in fp8. TE handles this with the margin parameter and by skipping. The newer direction is **finer-grained scaling** — per-block or per-tile rather than per-tensor, e.g. 1×128 tiles for activations and 128×128 for weights — which is much more robust to outlier channels because a single outlier no longer squashes the whole tensor's scale. DeepSeek-V3's technical report describes training at scale with this style of fine-grained fp8 quantization. **📅 Volatile** — the recipe landscape (delayed vs current vs blockwise scaling, and hardware-native support on Blackwell) is moving every few months; verify what your framework version implements.

**⚠ Trap:** "fp8 gives 2× throughput because H100 fp8 tensor cores are 2× bf16." Peak is 2× (≈1,979 vs ≈989 dense TFLOPS), but **end-to-end fp8 training typically delivers 1.3–1.5×**, because only the GEMMs are fp8 — attention softmax, norms, residuals, the optimizer, and all communication stay higher precision — and the quantize/dequantize and amax bookkeeping cost real time. Quoting 2× is a tell that you have read the spec sheet and not run the job.

### In an fp8 run, what stays in higher precision, and how do you tell fp8 apart from a real bug when the loss diverges?

The rule is simple to state and it is the thing to say first: **fp8 is a GEMM format, not a training format.** Everything that is a reduction, an accumulation over many terms, or a state that persists across steps stays in bf16 or fp32.

Specifically, in every production fp8 recipe I have seen:

- **fp32:** master weights, Adam `m` and `v`, the loss computation, LayerNorm/RMSNorm statistics, softmax accumulation, and the gradient reduce-scatter at large world size. Norms and softmax are sums over the hidden or sequence dimension; doing them in low precision produces a biased mean and variance and the model degrades in a way that looks like a bad learning rate.
- **bf16:** the residual stream and residual adds, all non-GEMM elementwise work, RoPE buffers, the router logits in an MoE, and typically the embedding and LM head.
- **fp8:** the QKV projection, the attention output projection, and the two MLP GEMMs. That is it. Those are ~90% of the FLOPs, which is why it is still worth doing.

Many teams additionally keep the **first and last transformer blocks in bf16**, because the first block sees raw embedding statistics and the last block feeds the loss, and both empirically have wider activation ranges. That is a heuristic, not a theorem, but it is cheap insurance — two of eighty layers is 2.5% of the FLOPs.

**🔍 Failure taxonomy — the fp8 run diverged from the bf16 baseline:**

1. **Divergence at a specific step that reproduces exactly** → not fp8. That is a data bug or a NaN in a batch. Bisect the data.
2. **Loss tracks bf16 for a while then slowly separates upward, no inf** → precision loss in something that should not be fp8. Check that norms, softmax and the loss are not inside your `fp8_autocast` region. Check whether your embedding or LM head got captured.
3. **Sudden inf/NaN right after a loss spike or an LR change** → delayed-scaling staleness. Raise the margin, shorten `amax_history_len`, or move to current/blockwise scaling.
4. **Divergence only above some world size** → gradient reduction precision, not fp8. Set `reduce_dtype=torch.float32`.
5. **Divergence that goes away when you disable one specific layer's fp8** → outlier channels in that layer. This is what fine-grained (per-tile) scaling exists to fix; the per-tensor scale is being dominated by a handful of channels.

**The debugging instrument that matters:** log per-tensor `amax` histories and the fp8 *overflow/underflow rate* per layer, per step, as a first-class metric. An fp8 run without amax observability is undebuggable — you cannot distinguish "the model is diverging" from "one tensor's scale is wrong" after the fact.

**🗣 Say this in the room:** "fp8 covers the four GEMMs in a transformer block and nothing else. If the loss diverges, my first question is whether the divergence correlates with an amax spike in a specific layer — if it does it is a scaling problem and I move to finer-grained scaling; if it does not, fp8 is a red herring and I bisect the data."

### When does activation offloading to host memory pay, and how do you decide?

Activation offload copies saved activations from HBM to pinned host memory during forward and streams them back during backward, on a side CUDA stream, hoping the copy hides behind compute. It is the last lever I reach for, and the calculation of whether it works is a pure bandwidth-versus-time race that you can do at the whiteboard.

The condition for it to be free is:

```
activation_bytes_per_layer / PCIe_bandwidth  <  compute_time_per_layer
```

Concretely, 70B-class, `h=8192`, `L=80`, `s=8192`, microbatch `b=1`, FlashAttention on. Activations per layer = 34 · s · b · h = 34 × 8192 × 8192 = **2.28 GB**. Compute per layer forward = 2 × (70e9/80) × 8192 tokens = 14.3 TFLOP; at 400 TFLOPS that is **36 ms**.

PCIe Gen5 x16 gives ~50 GB/s theoretical, ~40 GB/s achieved for a single stream, and on an 8-GPU node the host root complexes are shared, so budget **10–15 GB/s per GPU** under load. Transfer time = 2.28 / 12 = **190 ms**.

190 ms of copy versus 36 ms of compute. The copy loses by 5×, and you have converted a compute-bound layer into a PCIe-bound one. Offload does not pay here.

Now change the operating point: microbatch 1, `s = 2048`. Activations per layer = 34 × 2048 × 8192 = 570 MB → 48 ms of copy against 9 ms of compute. Still loses. The ratio does not improve with sequence length because both terms are linear in `s`.

That is the honest general result: **on a modern node, activation offload rarely wins for the transformer body**, because activation bytes scale with `s·b·h` while per-layer FLOPs scale with `s·b·h²/L`-ish — the ratio is governed by `h`, and `h` is not large enough relative to the PCIe/compute ratio. Where it *does* win:

- **Grace-Hopper / Grace-Blackwell superchips**, where NVLink-C2C gives ~450 GB/s host-to-device instead of PCIe's 50. Now the 2.28 GB copy is 5 ms against 36 ms of compute and it hides completely. This is the actual architectural reason those parts exist, and it is a good thing to name.
- **Offloading a subset** — the largest few tensors per layer rather than all of them — so you are trading a small copy against the whole layer's compute.
- **Single-GPU or single-node fine-tuning** where the alternative is not running at all, and the host PCIe is uncontended.

**⚠ Trap:** you must use **pinned** host memory, or the copy goes through a staging buffer and runs at a third of the bandwidth, and it must be on a **separate CUDA stream** with events, or it serializes with compute and you have made things strictly worse. Most naive implementations get one of these wrong and then report that "offload doesn't help," which is true but for the wrong reason.

**🗣 Say this in the room:** "I compute activation bytes per layer over achievable PCIe bandwidth and compare it to per-layer compute time. On a standard PCIe node that ratio is about 5:1 against you, so offload only makes sense on a C2C-coherent part or when you are otherwise not going to run at all."

### Define MFU and derive the FLOPs formula. What number should I expect on a real run?

MFU — Model FLOPs Utilization — is the only throughput metric that lets you compare two runs on different hardware, different parallelism layouts, and different models, because it normalizes out everything except "how much of the machine did you use for arithmetic the model actually requires."

```
MFU = (model_FLOPs_per_token × tokens_per_second) / (num_GPUs × peak_FLOPS_per_GPU)
```

**Deriving model FLOPs per token.** A matmul of `[m, k] × [k, n]` costs `2·m·k·n` FLOPs (one multiply, one add). Forward, every parameter is used in exactly one multiply-accumulate per token, so forward = `2N` FLOPs per token for `N` parameters. Backward computes gradients with respect to both inputs and weights, which is two more matmuls of the same shape, so backward = `4N`. Total:

```
FLOPs_per_token ≈ 6N    +    12 · L · h · s
                (weights)   (attention scores, which have no parameters)
```

The second term matters when `s` is comparable to `h`. At `s = 4096, h = 8192, L = 80`: `12 × 80 × 8192 × 4096` = 3.2e10 versus `6 × 70e9` = 4.2e11 — so attention is 7% of the total. At `s = 128k` the same term is 1.0e12, more than double the parameter term, and any MFU you report that ignores it is inflated. Always state which formula you used.

**MFU vs HFU.** Hardware FLOPs Utilization counts *all* FLOPs the GPU executed, including recomputation. With full activation checkpointing you re-run forward once, so HFU ≈ (8N)/(6N) = **1.33× MFU**. Papers quote MFU because it is the honest measure of useful work; vendors sometimes quote HFU because it is bigger. Know the difference and ask which one an interviewer means.

**📐 Numbers you must know:** on H100 with bf16, peak dense is ~989 TFLOPS. A well-tuned large dense pretraining run lands at **35–50% MFU**, i.e. 350–500 TFLOPS per GPU. Meta reported roughly 380–430 TFLOPS/GPU on the 16,384-H100 Llama 3 405B run — about 38–43% MFU. ByteDance's MegaScale paper reported 55.2% MFU for a 175B model on 12,288 GPUs, which is near the top of what has been publicly claimed. Google's PaLM paper reported 46.2% MFU (57.8% HFU) on TPU v4. **Below 30% means something is wrong** and you should go hunting rather than accept it.

**⚠ Trap:** MoE models have a *much* lower natural MFU (often 20–30%) and this is not a bug. All-to-all is a hard barrier, expert GEMMs are smaller and more ragged, and load imbalance is inherent. Comparing an MoE run's MFU to a dense run's MFU and concluding the MoE run is broken is a common and embarrassing error. The right comparison for MoE is *tokens per dollar at fixed quality*, not MFU.

### Give me the GPU generation table. What actually changed between A100, H100, H200 and Blackwell, and what does each change buy a training run?

Four numbers per part are enough, and each generation moved a different one, which is the interesting part.

| Part | HBM | HBM bandwidth | bf16 dense | Intra-node link |
|---|---|---|---|---|
| A100 80GB SXM | 80 GB HBM2e | ~2.0 TB/s | ~312 TFLOPS | NVLink 3, 600 GB/s |
| H100 SXM | 80 GB HBM3 | ~3.35 TB/s | ~989 TFLOPS | NVLink 4, 900 GB/s |
| H200 SXM | 141 GB HBM3e | ~4.8 TB/s | ~989 TFLOPS (same silicon) | NVLink 4, 900 GB/s |
| B200 | 192 GB HBM3e | ~8 TB/s | ~2.2 PFLOPS | NVLink 5, ~1.8 TB/s |

**📅 Volatile:** these are spec-sheet figures for the SXM parts and vendors restate them; verify against current documentation before quoting in an interview, and never quote a sparsity-inflated number (which is 2× the dense figure) without saying so.

What each step bought:

**A100 → H100** is a *compute* jump: 3.2× the bf16 throughput, plus fp8 tensor cores at another 2×, plus the Transformer Engine, plus 1.5× the memory bandwidth. Since training is compute-bound, this is close to a real 3× on training throughput. It also brought NVLink 4 and NVSwitch, keeping the collective/compute ratio roughly constant — that matters, because a compute jump without a network jump would have made TP infeasible.

**H100 → H200** is a *memory* jump with identical compute: 141 GB instead of 80, 4.8 TB/s instead of 3.35. For **training**, more HBM means fewer sharding constraints — bigger microbatches, less recompute, less need for aggressive FULL_SHARD — but the raw throughput ceiling is unchanged. For **inference** it is a much bigger deal, because decode is memory-bandwidth-bound and 1.43× the bandwidth is 1.43× the decode tokens/s. Knowing that H200 helps inference more than training is a good differentiator answer.

**Hopper → Blackwell** moves everything: ~2.2× compute, 2.4× memory, ~2.4× HBM bandwidth, 2× NVLink, plus fp4 tensor cores. But the architecturally important item is **GB200 NVL72**: 72 Blackwell GPUs and 36 Grace CPUs in one liquid-cooled rack forming a *single 72-GPU NVLink domain*. Every parallelism decision in this section was constrained by "TP must fit inside 8 GPUs." NVL72 raises that ceiling to 72, which changes the optimal mesh — you can run TP=16 or expert parallelism across 64 experts entirely on NVLink, which is worth far more to MoE training than the raw FLOPs increase. Power and cooling become the binding constraint instead: a GB200 NVL72 rack draws on the order of 120 kW, versus ~10–15 kW for a conventional air-cooled 8-GPU node, which is why most existing datacenters cannot host them without retrofit.

**🗣 Say this in the room:** "A100→H100 was a compute generation, H100→H200 was a memory generation that helps inference more than training, and Blackwell's real story for training is not the FLOPs — it is NVL72 extending the NVLink domain from 8 to 72 GPUs, which relaxes the hard constraint that tensor and expert parallelism must fit inside a node."

### When would you deliberately not use Nvidia? Walk me through MI300X, TPUs and Trainium.

The honest framing: for most teams the answer is "you would not, because CUDA is the moat," and being able to say *why* the alternatives are attractive anyway — and exactly what it costs to use them — is what makes this a senior answer rather than a brand-loyalty answer.

**AMD MI300X.** 192 GB HBM3 at ~5.3 TB/s and roughly H100-class bf16 throughput, in the same generation where H100 had 80 GB. That memory advantage is real and it is why MI300X first landed in *inference* deployments — you fit a 70B and a large KV cache on fewer devices. For training, the gap is software: ROCm has improved substantially, PyTorch upstream supports it, and Triton/FlashAttention have ROCm paths, but the long tail — a custom fused kernel, a specific Transformer Engine feature, a distributed-checkpointing path, a profiler workflow — is where weeks disappear. The decision rule I would give: consider MI300X when you have a stable, unexciting workload and a real engineering team to absorb porting cost, and when supply or price makes it materially cheaper. Do not consider it for a research workload that changes its kernels weekly.

**Google TPUs.** A different architecture, not just a different vendor: large systolic arrays, no CUDA, XLA compilation, and — the important part — **ICI, a 3D-torus interconnect that scales to pod-level all-reduce without leaving the vendor's fabric.** TPU v5p is ~95 GB HBM per chip with pods scaling into the thousands of chips; Trillium (v6e) and Ironwood (v7) continue the line with more memory and more per-chip throughput. **📅 Volatile** — per-generation HBM, FLOPs and pod sizes change with every announcement; verify. The programming model is JAX + `jit` with `shard_map`/GSPMD sharding annotations, which is genuinely *nicer* than PyTorch distributed for expressing 4D parallelism — you annotate the mesh and the compiler emits the collectives. The costs are lock-in to GCP, a smaller ecosystem, and the fact that debugging an XLA compilation problem is a different and less-documented skill. TPUs pay when you are training large and are willing to standardize on JAX.

**AWS Trainium2.** Purpose-built training silicon with high per-chip HBM, sold at a deliberate discount to Nvidia capacity on AWS, programmed through the Neuron SDK with a PyTorch/XLA front end. The pitch is price-per-token on a workload you are willing to freeze. The reality is the same long tail as ROCm plus a smaller community. **📅 Volatile.**

**The judgment answer.** Accelerator choice is a **portability decision, not a performance decision.** If your training code is written against PyTorch's distributed abstractions with no custom CUDA, porting is days. If you depend on Transformer Engine's fp8 recipes, a hand-written FlashAttention variant, or NCCL-specific tuning, porting is a quarter. I would push back hard on any plan to move a research training workload off CUDA, and I would seriously evaluate alternatives for a frozen inference workload or a well-understood periodic retraining job — which is exactly the pattern you see at the companies that have actually done it.

### Explain the interconnect hierarchy in a real cluster. InfiniBand versus RoCE versus plain Ethernet — how do you choose?

Every parallelism decision in this section reduces to placing traffic on the right tier, so you need the tiers and their numbers memorized.

**Tier 1 — inside a node: NVLink + NVSwitch.** On an H100 SXM node, each GPU has 18 NVLink-4 links at 50 GB/s bidirectional = 900 GB/s per GPU, and NVSwitch gives full all-to-all bandwidth between all 8 GPUs. Achievable all-reduce bus bandwidth is 350–450 GB/s. This is where TP and EP live.

**Tier 2 — between nodes: RDMA over InfiniBand or Ethernet.** A well-built training node has **one 400 Gb/s NIC per GPU** — 8 NICs, 3.2 Tb/s per node — with GPUDirect RDMA so the NIC DMAs straight from HBM without a host bounce. NDR is 400 Gb/s = 50 GB/s per port, achieving ~45 GB/s. XDR at 800 Gb/s is the successor. **📅 Volatile.**

**Tier 3 — the fabric topology.** Fat-tree / rail-optimized designs are standard: GPU `i` in every node connects to the same leaf switch ("rail `i`"), so a collective that only ever talks GPU-`i`-to-GPU-`i` across nodes stays one hop. This is exactly why NCCL's rings are constructed to respect rails, and why a badly-placed job that has to cross rails eats spine bandwidth and gets congested.

**InfiniBand versus RoCE versus Ethernet.**

- **InfiniBand** is a purpose-built lossless fabric with credit-based flow control, hardware RDMA, sub-microsecond switch latency, adaptive routing, and SHARP in-network reduction. It just works for collectives. It costs more and locks you to one vendor's switches.
- **RoCEv2** is RDMA semantics over Ethernet. It gets you standard Ethernet switching, multi-vendor supply, and reuse of existing datacenter operational expertise. The catch is that Ethernet is lossy by default and RDMA hates packet loss, so you must configure PFC and ECN correctly across every switch — and misconfigured PFC produces **congestion spreading and head-of-line blocking** that manifests as random stragglers rather than as an obvious error. This is genuinely hard; it is a network-engineering discipline, not a checkbox. Meta reported building Llama-3-scale clusters on both InfiniBand and RoCE, which is the strongest available evidence that RoCE is viable at scale if you invest in it.
- **Plain TCP Ethernet**, no RDMA: fine for inference serving, hopeless for synchronous training. You lose GPUDirect (every byte goes GPU→host→NIC), you pay kernel networking overhead per packet, and achieved bandwidth is often 5–10 GB/s against a 50 GB/s link. If NCCL falls back to `NET/Socket` on your cluster, that is the single highest-impact bug you have.

**The decision rule:** InfiniBand if you are buying a cluster and training is the point. RoCE if you have a strong network team, want vendor flexibility, or are extending an existing Ethernet estate. Never accept non-RDMA Ethernet for synchronous data-parallel training — and verify with `NCCL_DEBUG=INFO` that you actually got RDMA, because the fallback is silent.

### NCCL picks ring or tree for my all-reduce. Explain the difference, and what SHARP changes.

Ring and tree are the two ends of the classic bandwidth-versus-latency trade in collective algorithms, and NCCL chooses between them per call based on message size and world size — which means your all-reduce can change algorithm as you change microbatch size, and your scaling curve can have a discontinuity you did not put there.

**Ring all-reduce.** Ranks form a logical ring. Reduce-scatter phase: `N−1` steps, each rank sending a `S/N` chunk to its neighbour. All-gather phase: another `N−1` steps. Bytes per rank = `2(N−1)/N · S` — **independent of N** for large N, which is why ring is bandwidth-optimal. But latency is `2(N−1)` sequential hops, so at `N = 1024` a small message pays 2,046 hop latencies. Ring wins for **large messages**.

**Tree all-reduce.** Build a binary tree. Reduce up the tree, broadcast down. Depth is `2·log₂(N)` — 20 steps at `N = 1024` instead of 2,046. But the root and internal nodes handle disproportionate traffic, so the effective bandwidth is worse. Tree wins for **small messages and large world sizes**. NCCL's double-binary-tree construction fixes the leaf-node-idle problem of a single tree by running two complementary trees so every rank is internal in one of them.

NCCL's tuner picks by comparing modelled latency+bandwidth cost. You can force with `NCCL_ALGO=Ring` or `Tree`, and this is a legitimate tuning knob when you know your message-size distribution better than the tuner does.

**SHARP** (Scalable Hierarchical Aggregation and Reduction Protocol) changes the shape of the problem: the **switch ASIC does the reduction**. Instead of data circulating among endpoints, each rank sends its buffer once to the switch, the switch tree reduces it in the network, and the result is multicast back. Bytes on the wire per rank drop from `~2S` to `~2S/…` — in the ideal case one send up and one receive down, so roughly `2S` becomes `S` up plus `S` down but with *no* per-hop endpoint involvement and dramatically lower latency, and critically the endpoint NICs are freed. NVIDIA's Quantum InfiniBand switches implement it; NCCL uses it through the CollNet path (`NCCL_COLLNET_ENABLE=1`). The intra-node analogue on Hopper is **NVLS** — NVLink SHARP — where the NVSwitch performs the reduction (`NCCL_NVLS_ENABLE`).

**⚠ Trap:** SHARP has a limited number of hardware reduction trees per switch and they are an allocated resource. On a shared cluster, if another job has claimed them, your job silently falls back to ring and your all-reduce gets slower with no error. If your step time varies day to day on an unchanged config, check whether CollNet is actually engaging — `NCCL_DEBUG=INFO` prints the selected algorithm per collective.

**📐 Numbers you must know:** ring all-reduce moves `2(N−1)/N · S ≈ 2S` bytes per rank. `nccl-tests` reports "busbw" which is already normalized this way, so **busbw is directly comparable to your link's line rate** — a 400G NDR port should show ~45 GB/s busbw on a large all-reduce. If it shows 15, you have a problem, which is the next question.

### `nccl-tests` shows 45 GB/s busbw but my training job's all-reduce achieves 15. Debug it.

A 3× gap between the synthetic benchmark and the real job is one of the most common and most valuable things to be able to debug, because it is usually one config line and it is usually worth 20–30% of a cluster. Work it as a decision procedure, cheapest checks first.

**1. Is it actually the same message size?** `nccl-tests` at 1 GB is bandwidth-bound; your job's buckets might be 25 MB, which is partly latency-bound. Re-run `all_reduce_perf` at *your* bucket size before concluding anything. This resolves it maybe a third of the time and costs 60 seconds.

**2. Is NCCL using the right transport?** Set `NCCL_DEBUG=INFO` and read the init log. You want to see `NET/IB` with the right HCAs and `[send] via NET/IB/x` — if you see `NET/Socket`, RDMA is not engaged and 15 GB/s is exactly what TCP gives you. Causes: `NCCL_IB_DISABLE=1` inherited from someone's debugging session, the container missing `/dev/infiniband` or the `rdma_cm` device, `NCCL_SOCKET_IFNAME` pointing at the management interface, or the IB driver stack absent in the image.

**3. Is GPUDirect RDMA active?** Without it, every byte goes HBM → host DRAM → NIC, doubling PCIe traffic and adding host involvement. The log will show whether GDR is used. Requires the `nvidia-peermem` (or equivalent) kernel module and requires the GPU and NIC to be under the **same PCIe switch** — which is a hardware layout question, not a software one.

**4. Check the PCIe/NUMA topology.** Run `nvidia-smi topo -m`. You want `NV#` between GPUs (NVLink) and `PIX`/`PXB` between a GPU and its NIC. `SYS` means the path crosses the CPU root complex and the inter-socket link — that is your 3×. Fix by binding each rank to the CPU cores and memory node local to its GPU (`numactl --cpunodebind --membind`, or your launcher's affinity settings) and by setting `NCCL_IB_HCA` to pair each GPU with its rail-local NIC.

**5. Are all NICs in use?** A node with 8 NICs but a `NCCL_IB_HCA` setting that names one will use one, giving you exactly 1/8 the bandwidth. This is a classic copy-pasted-config failure.

**6. Is ACS enabled in BIOS?** PCIe Access Control Services forces peer-to-peer DMA through the root complex and destroys GPUDirect. It must be disabled on training nodes. This is a real, common, and infuriating one because everything "works," just slowly.

**7. Is another job on the node?** Shared nodes, another tenant's dataloader saturating PCIe or the NIC, or a stray process on a GPU. Check `nvidia-smi` and the NIC counters.

**8. Only after all that: is it contention in the fabric?** Congestion from other jobs on shared spine links, or your allocation being spread across many leaf switches instead of consolidated. Look at your scheduler's topology-aware placement.

**🗣 Say this in the room:** "I re-run the benchmark at my actual message size first, then read `NCCL_DEBUG=INFO` for the transport and whether GPUDirect engaged, then `nvidia-smi topo -m` for a `SYS` path between GPU and NIC, then verify all HCAs are in use and ACS is off. In my experience the answer is transport fallback or NUMA misbinding four times out of five."

### Why is training compute-bound while inference decode is memory-bandwidth-bound? Do the roofline arithmetic.

This is the single most clarifying piece of hardware literacy in the field, and it explains why training and serving have almost no optimizations in common.

**Arithmetic intensity** is FLOPs performed per byte read from memory. Every accelerator has a **ridge point**: the intensity at which you transition from memory-bound to compute-bound, equal to `peak_FLOPS / peak_bandwidth`.

For H100 bf16: 989e12 / 3.35e12 = **295 FLOPs per byte**. That is the number to memorize. To saturate an H100's tensor cores you must do 295 floating-point operations for every byte you pull from HBM.

**Training.** The dominant operation is a GEMM of shape `[M, K] × [K, N]` where `M` = tokens in the microbatch, often 8,192 or more, and `K, N` are model dimensions, often 8,192. FLOPs = `2MKN`. Bytes moved (ignoring cache reuse, worst case) = `2(MK + KN + MN)`. Intensity ≈ `MKN / (MK + KN + MN)`. With `M = K = N = 8192` that is `8192³ / (3 × 8192²)` = **2,731 FLOPs/byte** — nearly 10× above the ridge point. Compute-bound, comfortably. This is why training cares about FLOPs, fp8, and MFU.

**Inference decode.** You generate one token at a time, so `M = 1` per sequence. The weight matrix must be read from HBM in full to produce one output vector: bytes = `2KN`, FLOPs = `2KN`. Intensity = **1 FLOP per byte** — 295× below the ridge point. You are using roughly 0.3% of the tensor cores and 100% of the memory bandwidth. This is why decode cares about bandwidth, KV-cache size, quantization (fewer bytes to read), and batching (batching `B` sequences raises intensity to `B` FLOPs/byte, which is precisely why continuous batching exists and why you need `B ≈ 295` to become compute-bound).

**The consequences to name:**

- Training on an H200 versus an H100 is nearly a wash (same compute, more memory); decode on an H200 is ~1.43× faster (more bandwidth). Same silicon change, opposite conclusion.
- fp8 in training buys throughput because you are compute-bound. Weight quantization in inference buys throughput because you are bandwidth-bound. Same format, different mechanism.
- Prefill in inference behaves like training (large `M`, compute-bound); decode does not. That asymmetry is the entire justification for prefill/decode disaggregation.

**🗣 Say this in the room:** "The H100 ridge point is 989 TFLOPS over 3.35 TB/s ≈ 295 FLOPs per byte. Training GEMMs run around 2,700 — deep in compute-bound territory. Batch-1 decode runs at 1 — three hundred times into memory-bound territory. Every difference between training optimization and serving optimization follows from that one gap."

### 💰 Estimate the cost and wall-clock time to pretrain a 70B model on 1 trillion tokens on 1,024 H100s.

This is the arithmetic that a training-systems round eventually reaches, and doing it fluently in two minutes is worth more than any individual technique in this section. Four steps.

**Step 1 — total FLOPs.** `6N × tokens` = 6 × 70e9 × 1e12 = **4.2e23 FLOPs**. (Add the attention term if the context is long: at `s = 8192, L = 80, h = 8192`, `12·L·h·s` = 6.4e10 per token versus 4.2e11 for the parameter term — 15%, so a more careful number is 4.83e23. I would state the 6N number and mention the correction.)

**Step 2 — achieved throughput per GPU.** H100 peak bf16 = 989 TFLOPS. At a realistic 40% MFU: 0.40 × 989e12 = **396 TFLOPS = 3.96e14 FLOP/s per GPU**.

**Step 3 — GPU-hours.**

```
4.2e23 / 3.96e14 = 1.061e9 GPU-seconds
1.061e9 / 3600   = 294,700 GPU-hours
```

**Step 4 — wall clock and dollars.** On 1,024 GPUs at perfect goodput: 294,700 / 1,024 = **287.8 hours = 12.0 days**. At a realistic 90% goodput (checkpoint overhead, restarts, stragglers): **13.3 days**.

At a blended $2.50/GPU-hour (**📅 Volatile** — on-demand H100 pricing has moved a lot; reserved capacity is materially cheaper and this number should be re-verified before any interview):

```
294,700 × $2.50 = $736,750  at 100% goodput
                = $818,600  at 90% goodput
```

**Now the sensitivities, which is the actual point of the exercise.**

- **MFU 40% → 30%** (a bad mesh, TP crossing nodes, or unnecessary full recompute): GPU-hours go to 392,900, cost to **$982,250**. That is **+$245,500** for one configuration mistake.
- **Goodput 90% → 70%** (bad checkpointing cadence, no straggler detection, frequent unclean restarts): cost goes from $818,600 to **$1,052,500**, i.e. **+$234,000**, and wall clock from 13.3 to 17.1 days.
- **fp8 at 1.4× end-to-end**: 294,700 → 210,500 GPU-hours, **$526,250** — saves **$210,500** and 3.4 days, at the cost of the amax-observability engineering and some divergence risk.

**🗣 Say this in the room:** "Six N times tokens over achieved FLOPS. 70B on 1T tokens is 4.2e23 FLOPs; at 40% MFU on H100 that is about 295,000 GPU-hours, so 12 days on 1,024 GPUs and roughly $750k at $2.50 an hour. The two levers that move that number most are MFU and goodput, and each is worth a quarter of a million dollars on this run — which is why I would spend the first week of the project on the mesh layout and the checkpointing story rather than on the model code."
