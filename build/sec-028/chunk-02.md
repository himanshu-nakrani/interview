### Explain prefill/decode disaggregation from first principles. Why would anyone buy two GPU pools instead of one?

Because prefill and decode are two different workloads that happen to share a weight matrix, and every scheduler that runs them on the same device is arbitrating between them badly. Prefill is compute-bound, embarrassingly parallel across positions, and wants every FLOP you own for a short burst. Decode is memory-bandwidth-bound, has a parallelism of exactly one token per sequence, and wants a large batch and an uninterrupted cadence. Co-locating them is like running your OLTP and your analytics scan on the same Postgres primary: it works, you tune it, and you spend your life explaining why the p99 moved.

Disaggregation says: give prefill its own pool of GPUs, give decode its own pool, and ship the KV cache between them. A request arrives, is routed to a prefill worker, is prefilled to completion, its KV cache is transferred to a decode worker, and the decode worker streams tokens. Prefill workers never do decode; decode workers never do prefill. The interference is not mitigated, it is *structurally impossible* — which is the qualitative difference from chunked prefill.

Three second-order wins follow immediately and they are the ones that make the economics work. First, **independent scaling**: if your workload shifts to long prompts and short answers, you add prefill nodes without buying decode capacity you will not use. Second, **independent parallelism**: prefill wants high tensor parallelism to minimise latency on a big burst of compute; decode wants *lower* TP (because all-reduce overhead per token is a larger fraction of a cheap step) and higher batch. On a shared pool you must pick one TP degree for both. Third, **independent hardware**: the two phases have opposite bottlenecks, so they have opposite ideal SKUs.

**📄 Papers:** Patel et al. (ISCA 2024), *Splitwise* — split the phases across machines and showed heterogeneous fleets (compute-dense for prompt, cheaper for token generation) hitting better throughput per dollar and per watt. Zhong et al. (OSDI 2024), *DistServe* — formalised the objective as goodput-under-SLO and co-optimised placement and parallelism per phase, reporting large multiples (order 7×) in requests served under tight SLOs. Qin et al. (2024, Mooncake, Moonshot AI's Kimi platform) — a KVCache-centric disaggregated architecture that treats pooled CPU/DRAM/SSD KV storage as a first-class tier and adds prediction-based early rejection under overload.

**⚠ Trap:** presenting the ~7× number as something you will get. Those figures are goodput improvements under *tight* SLOs on workloads where interference was the binding constraint. Under loose SLOs, or on short prompts, or at low load, disaggregation is a **net loss** — you have added a network transfer to every request and split your memory pool in two. Name the conditions or the number is a red flag.

### When does disaggregation NOT pay off? I want the conditions where it makes things worse.

This is the question that separates someone who read the papers from someone who deployed it. Five conditions, each with the mechanism:

**Short prompts.** With a 300-token prompt, prefill is ~13 ms on a 70B TP8 and there was never a stall to eliminate. You have added a KV transfer of 300 × 0.31 MB = 94 MB and a routing hop to save nothing. For chat-style traffic with sub-1k prompts, co-located with chunked prefill wins.

**Low utilisation.** Disaggregation partitions your fleet. Two pools each at 30% utilisation have worse effective concurrency than one pool at 60%, because a decode-pool GPU cannot absorb a prefill burst and vice versa. Below roughly 50–60% fleet utilisation, statistical multiplexing on a single pool beats specialisation. This is the classic pooled-versus-dedicated-resources argument and it has not changed since connection pools.

**Thin interconnect.** If your KV transfer does not comfortably fit under your TTFT slack, disaggregation directly *adds* to TTFT. See the bandwidth arithmetic in the next question — on 100 GbE it is disqualifying.

**High prefix-cache hit rates.** If 85% of your prefill tokens are cache hits, prefill is already cheap and interference is already small. Worse, disaggregation complicates the cache: the prefix cache lives in the prefill pool, but the decode pool needs the KV too, so you now have a distributed cache-coherence problem you did not have.

**Small models.** A 7B prefills 10× faster than a 70B; the stall it creates is 10× shorter; and the fixed costs of the transfer and the extra hop are unchanged. Disaggregation is a big-model technique.

**🗣 Say this in the room:** "I'd disaggregate when prompts are long, the fleet runs hot, and I have 400 Gb/s-class RDMA between pools. Short prompts, a half-empty fleet, or Ethernet-class interconnect and I'd stay co-located with chunked prefill — the transfer would cost more than the interference it removes."

### Do the bandwidth math for me. You have to move a 2.5 GB KV cache from a prefill node to a decode node. Does it fit?

The 2.5 GB is the Llama-3.3-70B figure for an 8k-token sequence: 2 (K and V) × 80 layers × 8 KV heads × 128 head_dim × 2 bytes = 327,680 bytes/token = 0.3125 MB/token, × 8,192 tokens = **2.56 GB**. That is the payload. Now the fabrics, all at a realistic 70–80% of nominal:

| Fabric | Nominal | Achieved | Time for 2.56 GB |
|---|---|---|---|
| NVLink 4 (intra-node, H100) | 900 GB/s bidir | ~450 GB/s effective one-way | **5.7 ms** |
| InfiniBand NDR, 1 NIC | 400 Gb/s = 50 GB/s | ~40 GB/s | **64 ms** |
| InfiniBand NDR, 8 NICs striped | 400 GB/s | ~300 GB/s | **8.5 ms** |
| 100 GbE (TCP) | 12.5 GB/s | ~9 GB/s | **284 ms** |

Read that table against a 400 ms TTFT budget where prefill already consumed 220 ms. Intra-node NVLink: fine, 5.7 ms is noise. Striped RDMA across all NICs: fine, 8.5 ms. A single NDR NIC: 64 ms is 16% of your entire budget for a pure data move — survivable but you would want to hide it. 100 GbE: **you have blown the budget with the transfer alone.** This is why every serious disaggregated deployment is RDMA (RoCE or IB) with GPUDirect, and why "we'll just put the pools in different racks on the standard network" is a design I reject on sight.

**📐 Numbers you must know:** KV per token scales as `2 · n_layers · n_kv_heads · d_head · dtype_bytes`. Memorise 0.31 MB/token for a 70B-class GQA model at FP16, and halve it for FP8. Then any transfer question is one multiplication: 128k context = 128k × 0.3125 MB = **40 GB**, which on a single NDR NIC is 1.0 second. Long context is what breaks disaggregation, and the arithmetic tells you exactly where.

**💰 Math on the network bill:** at 10k requests/hour with 8k prompts, you are moving 10,000 × 2.56 GB = 25.6 TB/hour = 7.1 GB/s of sustained east-west traffic. That is not a rounding error on your fabric; it is a design input to how many NICs you buy per node. If your cloud charges for cross-AZ traffic, putting the pools in different AZs at, say, $0.01/GB would cost 25,600 GB/hr × $0.01 = **$256/hour = $184k/month** in network egress alone. Keep the pools in one AZ, ideally one rack. **📅 Volatile:** cross-AZ pricing varies by provider; verify.

### 64 ms of transfer is a lot. How do you hide it?

You overlap it with the prefill compute that produces it, layer by layer. The KV for layer 0 is final the moment layer 0's forward pass completes — there is no reason to wait for layer 79 before starting to send it.

The arithmetic is lovely. An 8k prefill on a 70B TP8 takes ~450 ms of compute (8,192 tokens ÷ ~18,000 tokens/s). Eighty layers means one layer completes every 5.6 ms. Each layer's KV slice is 2.56 GB ÷ 80 = **32 MB**, which on a single 40 GB/s NIC takes 0.8 ms. So you have 5.6 ms of compute to hide 0.8 ms of transfer per layer: the transfer is fully overlapped, and the only exposed cost is the last layer's 0.8 ms plus a handshake. A 64 ms serial transfer becomes a ~1–2 ms exposed cost. This layer-wise streaming is the standard technique in production disaggregation stacks and is why the naive "the transfer kills you" objection is wrong for long prefills.

The mechanism requires three things you should name: **GPUDirect RDMA** so the KV goes HBM→NIC→HBM without a bounce through host memory (a host bounce doubles your PCIe traffic and adds copies); **pre-allocated destination blocks** on the decode worker, so the transfer has somewhere to land — which means the decode worker must admit the request and reserve KV blocks *before* prefill finishes, coupling the two schedulers; and **a completion protocol** so decode knows when layer L is safe to read.

**⚠ Trap:** assuming overlap works for short prompts. A 512-token prefill takes ~28 ms of compute and produces 160 MB of KV; per layer that is 0.35 ms of compute to hide 2 MB = 0.05 ms of transfer — still fine ratio-wise, but the *fixed* costs (RDMA connection setup, block reservation round-trip, scheduler handshake) are now a large fraction of a 28 ms prefill. Disaggregation's overhead is dominated by fixed costs at short prompts and by bandwidth at long ones; the middle is where it shines.

### Would you compress the KV cache before transferring it? What's the decision rule?

Only when the link is the binding constraint, and I decide it with a break-even calculation rather than a preference.

The rule: compression pays iff `compress_time + transfer_time_compressed + decompress_time < transfer_time_raw`. Say you achieve 2× compression. On striped RDMA at 300 GB/s, raw transfer of 2.56 GB is 8.5 ms; compressed it is 4.3 ms, saving 4.2 ms — but you must compress 2.56 GB and decompress 1.28 GB on the GPU, and a GPU-side quantise/dequantise kernel runs at maybe 500 GB/s–1 TB/s effective, so that is ~2.6 ms + ~1.3 ms = 3.9 ms of added compute. Net saving: 0.3 ms. **Not worth the complexity.** On a single 40 GB/s NIC, raw is 64 ms, compressed is 32 ms, saving 32 ms against 3.9 ms of compute — net 28 ms saved, clearly worth it. The rule is therefore: **compress when your per-request effective bandwidth is below roughly 100 GB/s; do not when it is above.**

The cheapest form of compression is not a codec at all — it is **casting the KV to FP8 for transfer**, which is exactly 2× and costs a fused cast kernel. If you are already serving with an FP8 KV cache (which since roughly 2024 I treat as the production default for 70B-class models), the transfer is 1.28 GB and you get the benefit for free with no extra step. That is the answer I would give first.

Beyond that there is a research line on smarter KV transfer compression — CacheGen-style approaches that encode KV tensors into compact bitstreams for network transfer and adapt the encoding level to available bandwidth, and disaggregation-specific work catalogued under names like KVServe and SplitZip. **📅 Volatile:** this area is moving fast and I would not quote a specific compression ratio or quality delta from memory; the durable point is the break-even inequality above, plus the caution that any lossy KV compression must be validated on *your* long-context eval, because degradation from KV quantisation concentrates in long-context, math, and non-English slices rather than showing up on a short-prompt benchmark.

### Prefill and decode want different parallelism. Spell out what you'd configure for each pool.

**Prefill pool: high tensor parallelism, low batch.** Prefill is compute-bound, so splitting the matmuls across more GPUs directly divides the latency. TP8 on a 70B roughly halves prefill latency versus TP4. The all-reduce cost per layer is amortised over thousands of tokens, so TP's communication overhead is a small fraction — this is the regime where TP scales well. You do not need large batches: one or two long prompts already saturate the tensor cores.

**Decode pool: lower tensor parallelism, high batch, more data/replica parallelism.** Decode's per-step compute is tiny; the step is dominated by streaming weights and by two all-reduces per layer. At TP8 you pay 160 all-reduces per token step, each of which has a latency floor set by the NVLink round-trip regardless of payload size — so TP's *fixed* cost is a large fraction of a cheap step. Going from TP4 to TP8 on decode might cut the weight-streaming term in half but doubles the collective count, and the measured ITL improvement is often much less than 2×, sometimes near zero. The right decode configuration is usually the **lowest TP degree that fits weights plus a useful KV budget**, replicated for throughput. For a 70B at FP8 (70 GB), TP2 on H100s (35 GB weights each, ~40 GB left for KV) is often better for ITL-at-scale than TP8, and it gives you 4 independent decode replicas per node instead of 1.

For MoE models the split is sharper still: prefill wants expert parallelism with large per-expert batches so the grouped GEMMs are efficient; decode with a small batch activates few tokens per expert and suffers badly from EP all-to-all latency, so decode wants either wide EP with aggressive batching or a completely different placement. Getting this wrong is the most common MoE serving mistake I see.

**🗣 Say this in the room:** "Prefill gets high TP because compute splits cleanly and the collectives amortise over thousands of tokens. Decode gets the *lowest* TP that fits weights and KV, because at one token per sequence the all-reduce latency is fixed cost on a cheap step — I'd rather have four TP2 decode replicas than one TP8."

### Talk to me about heterogeneous hardware. Which cards go in which pool, and what's the cost argument?

Once the phases are separated, they stop needing the same SKU, and the selection criteria become almost disjoint. **Prefill is bought on dense FLOPs per dollar per second.** It barely needs memory capacity — it holds weights plus one sequence's activations — and it barely needs HBM bandwidth. **Decode is bought on HBM bandwidth × capacity per dollar.** It needs bandwidth because that is the bottleneck and capacity because KV capacity is what sets your batch size, and batch size is your throughput.

Concretely, that argues for putting the newest compute-dense parts (Blackwell-class, or H100s with FP8/FP4 tensor cores) in prefill, and the highest-memory-bandwidth, highest-capacity parts in decode — H200 at 141 GB and ~4.8 TB/s, or MI300X at 192 GB and ~5.3 TB/s, are decode-shaped cards. **📅 Volatile:** all of those specs and their relative pricing change every product cycle; verify before quoting them.

**💰 Math, done the way I'd do it live.** Suppose prefill needs 3.2 PFLOPS to hit its TTFT and decode needs 20 TB/s aggregate HBM bandwidth to hit its ITL at your target batch. Card A gives 400 TFLOPS achieved and 3.35 TB/s at $3.00/hr; Card B gives 250 TFLOPS achieved and 4.8 TB/s at $4.00/hr. Homogeneous on A: prefill needs 8 cards, decode needs 20 TB/s ÷ 3.35 = 6 cards → 14 cards × $3 = **$42/hr**. Heterogeneous: prefill 8 × A = $24, decode 20 ÷ 4.8 = 4.2 → 5 × B = $20 → **$44/hr**. In this made-up instance heterogeneity *loses*, and that is exactly the point: the argument is not "different cards are better," it is "run the two-line arithmetic with your real SKU prices and your real bottleneck quantities." The published gains come from cases where the price gap is wide enough — e.g. using previous-generation, cheaper, still-high-bandwidth cards for decode while reserving scarce new silicon for prefill. That reasoning is Splitwise's, and it is a procurement argument as much as a systems one.

**⚠ Trap:** heterogeneous pools require your model to be quantised and loaded identically on both, and require your KV layout and dtype to match across the transfer. Mixing a card whose FP8 support differs (E4M3 vs E5M2 handling, or no native FP8 at all) means a conversion on every transfer. Check the format matrix before the price sheet.

### What do Dynamo and llm-d actually give you that you couldn't build yourself?

They give you the four pieces that turn "two pools of vLLM" into a serving system, and having built two of the four by hand I can tell you the value is real but the lock-in is also real.

**A KV-aware router.** Routing is no longer "least connections." The right decode worker is the one that already holds the most of this request's prefix, and the right prefill worker is the one whose prefix cache overlaps the prompt. That means the router needs a global view of which blocks live where — a distributed index of prefix hashes to workers — and a scoring function that trades cache affinity against load. This is the single hardest piece to build well and it is where most of the gain lives.

**A KV transfer layer.** A library that does GPUDirect RDMA point-to-point, layer-wise streaming, and a pluggable backend so the same code path works over NVLink intra-node and IB inter-node (NVIDIA's is NIXL). Writing this yourself means writing RDMA verbs code, and you will get the memory registration lifecycle wrong at least twice.

**A planner / autoscaler that understands the P:D ratio.** It watches prefill queue depth and decode KV utilisation separately and scales the two pools independently, which no stock Kubernetes HPA will do for you because the signal is not CPU or QPS.

**A tiered KV store.** GPU → host DRAM → NVMe → object storage, so a cache entry evicted from a decode worker is not lost. This is the Mooncake-shaped idea productised.

**📅 Volatile:** NVIDIA Dynamo and llm-d are both young (2025-era) and their feature sets, API stability and Kubernetes integration are moving quickly. My decision rule today: if you are running one or two models on under ~32 GPUs, roll with vLLM or SGLang co-located plus chunked prefill and skip the whole category — the operational complexity is not repaid. Past a few dozen GPUs with long prompts and a hard TTFT SLO, adopt one of these rather than building the router yourself. Verify their current state before an interview; asserting a stale feature matrix reads worse than saying "as of my last check."

### How do you size the prefill-to-decode ratio, and what happens when the workload drifts?

Size it from the token-flow rates, not from intuition. Let `λ` be requests/s, `T_in` mean input tokens, `T_out` mean output tokens. Prefill demand is `λ · T_in` tokens/s of prefill work; decode demand is `λ · T_out` tokens/s of decode work. Divide each by the per-pool per-GPU capability.

**💰 Worked:** λ = 20 req/s, T_in = 4,000, T_out = 600, Llama-3.3-70B. Prefill demand = 80,000 tok/s. A prefill GPU does ~2,850 prefill tok/s (400 TFLOPS ÷ 140 GFLOP/token), so with TP8 and ~85% scaling efficiency a node does ~19,400 tok/s → **80,000 ÷ 19,400 = 4.1 → 5 prefill nodes** (40 GPUs). Decode demand = 12,000 tok/s. A TP8 decode node at batch 64 does ~2,000 tok/s → **6 decode nodes** (48 GPUs). P:D ratio ≈ **5:6 by node count.** Now change one input: if T_in rises to 16,000 (agents pasting whole files — the Cursor/Harvey pattern), prefill demand quadruples to 320,000 tok/s → 17 prefill nodes, and the ratio flips to 17:6. That is a 3.4× change in your fleet composition from one workload shift.

Which is exactly why the ratio must be **dynamic**, and why the hardest operational problem in disaggregation is not the transfer, it is that the correct P:D ratio drifts with the time of day, with a customer onboarding, and with a prompt-template change your own team shipped. Three mitigations: (1) autoscale the pools independently on phase-specific signals — prefill queue depth for the prefill pool, KV utilisation for the decode pool; (2) keep a **flexible tranche** of nodes configured to run either role, switchable in minutes, so you are not blocked on provisioning; (3) alert on the ratio of prefill-pool utilisation to decode-pool utilisation, because a sustained divergence is the leading indicator of a workload shift, usually 30–60 minutes before it becomes an SLO incident.

**⚠ Trap:** sizing on mean input length. Prompt length distributions in agent products are brutally long-tailed — a p50 of 2k with a p99 of 120k is normal. Prefill demand is driven by the *mean*, but prefill *latency* SLO compliance is driven by the tail, and you must size for both: capacity from the mean, TTFT tiering from the tail.

### What breaks in a disaggregated system that doesn't break in a co-located one? Give me the failure taxonomy.

**🔍 Failure taxonomy — disaggregation-specific:**

**A prefill worker dies mid-prefill.** Recoverable and cheap: nothing has been streamed to the user, so retry on another prefill worker. Cost is duplicated prefill compute. Make the router's retry budget explicit — I cap it at one retry, because a request that has now consumed 2× prefill under load is a request that is making the incident worse.

**A decode worker dies mid-stream.** Not recoverable transparently. The user has already received 200 tokens. You can either fail the stream (and let the client show an error mid-answer) or restart on another worker, which requires re-prefilling the original prompt *plus* the 200 emitted tokens, and produces different continuation text because sampling is stochastic. My rule: **fail the stream and surface it.** Silently restarting produces answers with a repeated or contradictory seam, which is worse than an error. If you must recover, do it at the application layer with the emitted text as context, and make it a visible retry.

**The KV transfer fails or times out.** The request is in limbo: prefill compute is spent, decode has reserved blocks. You need a timeout on both sides and a clean release of the decode worker's reserved blocks, or you leak KV capacity — which manifests days later as mysteriously reduced max concurrency. This leak is the number-one operational bug in homegrown disaggregation.

**Backpressure inversion.** The decode pool is full, so it stops accepting handoffs, so prefill workers hold completed KV in HBM waiting for a destination, so prefill HBM fills, so prefill stops admitting. A queue formed in the worst place — in GPU memory. The fix is admission control at the *front*, gated on decode-pool capacity: never start a prefill you do not have a decode slot for. This is the "reserve the destination before you start" discipline and it is not optional.

**Split-brain on the prefix cache.** A block hash exists on prefill worker 3 and decode worker 7 with different eviction states; the router thinks a request is a cache hit and routes it to a worker that evicted the block. Result: a silent TTFT regression with no error anywhere. Instrument cache-hit *predictions* against cache-hit *outcomes* and alert on divergence.

**Metrics that no longer mean what they meant.** "GPU utilisation" on the decode pool is structurally ~30% because decode is bandwidth-bound; on the prefill pool it is ~85%. A single fleet-wide utilisation alert is now meaningless. Every dashboard must be per-pool.

### Give me your final decision rule: chunked prefill on one pool, or full disaggregation?

I run a three-question gate and I would say it exactly like this in an interview.

**Question one: is prefill interference actually my problem?** Measure it, do not assume it. Chart per-request max ITL gap against whether a long prefill was co-scheduled. If p99 max-gap is under your ITL SLO, you have no interference problem and the answer is neither technique. Most teams asking about disaggregation have not run this measurement.

**Question two: does chunked prefill close the gap?** Compute the token budget your ITL SLO implies (ITL headroom ÷ per-prefill-token cost). If that budget lands above ~512, chunked prefill on a single pool solves your problem for single-digit percent of prefill throughput and zero architectural change. **Take it.** This is where the large majority of production systems land, and shipping it is a one-line config change plus a benchmark.

**Question three: if the implied budget is under 512, do I have the preconditions for disaggregation?** Namely: mean input length above roughly 2k so the transfer amortises; fleet utilisation above ~50% so partitioning does not cost me multiplexing; RDMA at 400 Gb/s-class or better between the pools, in the same AZ; and enough GPUs (I use ~32 as the rough floor) that two pools are each big enough to batch well. All four yes → disaggregate. Any no → the honest answer is to relax the ITL SLO, shrink the model, or quantise, because you are trying to buy with architecture something the hardware will not sell you.

**🗣 Say this in the room:** "Chunked prefill first — it's a config change that buys a bounded ITL tail for a few percent of prefill throughput, and it's sufficient for most workloads. I'd only disaggregate when the ITL SLO implies a chunk budget under about 512 tokens, and only if I have long prompts, a hot fleet, RDMA between the pools and at least a couple of dozen GPUs. Otherwise the transfer and the loss of multiplexing cost more than the interference they remove."
