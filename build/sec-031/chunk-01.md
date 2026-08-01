### Before we talk about any parallelism scheme — I hand you a 70B model and a box of H100s. Walk me through the sizing arithmetic that tells you how many GPUs you need.

Start from the observation that inference memory has exactly three consumers, and only one of them is fixed. Weights are a constant. Activations during a forward pass are small and transient. The KV cache is the variable that eats everything else, and it is the only one that scales with *traffic*. So sizing is: subtract weights from HBM, subtract a working margin, and whatever is left is your concurrency budget — measured in tokens, not requests.

Weights first. A 70B model in bf16 is 70e9 × 2 = **140 GB**. One H100 SXM has 80 GB. So the answer is already "at least two GPUs," and because you want KV room you are really looking at 4 or 8. On 8×H100 (640 GB) you have 640 − 140 = 500 GB nominal, minus roughly 8–12 GB per GPU of CUDA context, NCCL buffers, activation scratch and fragmentation — call it 80 GB of overhead — leaving ~420 GB for KV.

Now the KV cache per token. Llama-3-70B-class geometry: 80 layers, 8 KV heads (GQA), head_dim 128. Per layer per token you store K and V: 2 × 8 × 128 × 2 bytes = 4,096 bytes = 4 KB. Across 80 layers that is **320 KB per token**. Memorize that shape of calculation; the formula is `2 × n_layers × n_kv_heads × head_dim × dtype_bytes`.

420 GB ÷ 320 KB = 1.31 million tokens of KV. If your average live sequence (prompt + generated so far) is 4,000 tokens, that is ~328 concurrent sequences in flight. If your product is a coding assistant where sequences average 30,000 tokens, it is ~44. That difference — the same hardware supporting 328 vs 44 concurrent users — is the entire content of a serving-design round, and it is why "how many QPS" is the wrong first question and "what is your token-length distribution" is the right one.

**📐 Numbers you must know:** bf16 bytes/param = 2, so *B* billion params = 2*B* GB. GQA KV per token = `2 × L × H_kv × d_head × 2` bytes. For 70B-class (L=80, H_kv=8, d=128) that is 320 KB/token; for 8B-class (L=32, H_kv=8, d=128) it is 128 KB/token. An MHA-era 70B with 64 KV heads would be 8× worse — 2.56 MB/token — which is the whole reason GQA exists.

**⚠ Trap:** candidates size for weights and forget that a paged KV allocator still needs headroom for fragmentation *and* for the largest single request you accept. If you advertise a 128k context window, one request can claim 128,000 × 320 KB = **41 GB** of KV by itself — 10% of your entire cache — and if you have not capped `max_model_len` or reserved per-tenant token quotas, a single user can evict everyone else. I enforce in review that `max_model_len` is set to the *product's* real ceiling, not the model's.

**🗣 Say this in the room:** "Weights set the floor on GPU count; the KV cache sets the ceiling on concurrency. I size by computing bytes-per-token of KV from the layer/head geometry, then dividing the leftover HBM by my measured p50 sequence length. Everything else — batching policy, autoscaling signal, cost per token — falls out of that one number."

### Explain tensor parallelism to me at the level of the matrix multiply. Where exactly do the collectives land?

Tensor parallelism splits individual weight matrices across GPUs so that each device holds a slice and does a slice of the arithmetic. The mental model that makes it inevitable: a matmul `Y = XW` can be cut two ways, and the two ways compose so that consecutive cuts cancel each other's communication. That cancellation is the whole trick — it is why a transformer block needs only *one* collective per sub-block instead of one per matmul.

Column-parallel: split `W` by columns, `W = [W₁ | W₂]`. Each rank computes `XWᵢ` and holds a column-slice of the output. No communication needed — but the output is sharded along the feature dimension.

Row-parallel: split `W` by rows, `W = [W₁ ; W₂]`, and split the *input* correspondingly, `X = [X₁ | X₂]`. Each rank computes `XᵢWᵢ`, which is a partial sum of the full result. You need an **all-reduce** to add the partials.

Now compose. In the MLP, `up_proj` is column-parallel, so each rank produces its own slice of the hidden activations, applies the elementwise nonlinearity locally (this works because GELU/SiLU are elementwise — that is why the split is legal), and feeds that slice directly into a row-parallel `down_proj`. The column split *produces* exactly the sharded input the row split *wants*. One all-reduce at the end of the MLP.

Attention is the same pattern with a physical meaning: Q/K/V projections are column-parallel, which means each rank owns a subset of attention *heads* end-to-end — it computes those heads' scores, softmax and value-weighted sums entirely locally, with no cross-rank traffic inside the attention math. The output projection is row-parallel. One all-reduce at the end of attention.

So: **two all-reduces per transformer layer** in the forward pass — one after attention, one after the MLP. For an 80-layer model that is 160 collectives per forward pass, each on a tensor of shape `[num_tokens, hidden_dim]`.

**📄 Paper:** Shoeybi et al. (2019), *Megatron-LM* — introduced the column-then-row decomposition that makes intra-layer model parallelism cost one all-reduce per sub-block. It replaced naive "put different layers on different GPUs" model parallelism for the within-node case.

**⚠ Trap:** people say "TP splits the model across GPUs so each GPU does 1/N of the work in 1/N of the time." The FLOPs do divide, but the collectives do not — they are pure overhead that grows with N. TP scaling is sub-linear and the sub-linearity is entirely the all-reduce. This matters most in *decode*, where the tensor being reduced is tiny (`[batch, hidden]`) so you are paying pure latency, not bandwidth. Two hundred small all-reduces per token is a latency problem, not a throughput problem.

### A team is running TP=8 spread across two 4-GPU nodes connected by 25 Gigabit Ethernet. What happens, and show me the numbers.

It falls off a cliff, and the number that kills them is latency, not bandwidth. This is the single most common GPU-topology mistake I see, and it is worth being able to compute on a whiteboard.

Take a 70B, hidden_dim 8,192, 80 layers, bf16. Decode step, batch size 1. Each all-reduce operates on `1 token × 8192 × 2 bytes = 16 KB`. A ring all-reduce moves `2(N−1)/N × S` bytes per rank; for N=8 that is 1.75 × 16 KB = 28 KB per rank per collective. With 160 collectives per token, each rank moves 160 × 28 KB = **4.5 MB per generated token**.

Bandwidth check on 25 GbE: 25 Gbps ≈ 3.1 GB/s usable. 4.5 MB ÷ 3.1 GB/s = **1.45 ms per token** of pure wire time. That alone caps you at ~690 tok/s aggregate before any compute.

But latency is worse. A ring all-reduce over 8 ranks is 2(N−1) = 14 sequential hops. Ethernet with a TCP-based transport gives you 30–100 µs per hop; take 50 µs. That is 14 × 50 µs = 700 µs *per collective*, and there are 160 of them serialized per token: **112 ms per token**. Your inter-token latency goes from ~15 ms to ~127 ms — roughly 8 tokens/second. A user watching a stream sees it crawl.

Now the same math on NVLink inside one node. NVLink 4 gives ~900 GB/s aggregate per H100 and sub-10 µs collective latency for small messages. 4.5 MB ÷ 900 GB/s = 5 µs of wire time; 160 collectives × ~8 µs ≈ 1.3 ms per token of overhead against a ~10 ms compute step. Entirely tolerable.

**⚠ Trap — the named mistake:** *TP must not cross the NVLink domain.* The rule I enforce is: TP degree ≤ the number of GPUs in one NVLink-connected node (8 for a standard HGX box, up to 72 for GB200 NVL72). If you need more parallelism than that, you go to pipeline parallelism across nodes, because PP sends one activation tensor per stage boundary instead of 160 collectives per token. If a candidate proposes TP=16 across two nodes and does not immediately flag the interconnect, that is a strong signal they have never operated one of these.

**💰 Math:** the same 8 GPUs cost the same $/hr either way — say $2.50/GPU-hr on-demand (**📅 Volatile:** GPU spot/on-demand pricing moves quarterly; verify before your loop). At $20/hr the NVLink layout serving ~2,000 output tok/s costs 20 ÷ (2000 × 3600 ÷ 1e6) = **$2.78 per million output tokens**. The Ethernet layout at ~8 tok/s per stream and maybe 250 tok/s aggregate costs 20 ÷ (250 × 3600 ÷ 1e6) = **$22.20 per million** — an 8× cost regression from a topology decision made in a Terraform file.

**🗣 Say this in the room:** "Tensor parallelism is a latency-bound collective on the critical path of every single decoded token, so it lives inside the NVLink domain and nowhere else. Across nodes I use pipeline parallelism, which pays one point-to-point transfer per stage boundary rather than 160 all-reduces per token."

### What constraint does grouped-query attention put on my TP degree, and what do engines do when the heads don't divide evenly?

The mental model: TP shards attention by *head*, so the number of heads is a divisibility constraint on the parallelism degree — and with GQA it is the number of **KV** heads that binds, not the number of query heads.

Under MHA, every query head has its own K and V, so sharding 64 heads across 8 ranks gives each rank 8 complete heads and 8 complete KV heads. Clean. Under GQA, you might have 64 query heads sharing only 8 KV heads (a group size of 8). TP=8 still works: each rank gets 8 query heads and 1 KV head. TP=16 does not divide — you cannot give a rank half a KV head.

Engines resolve this by **replicating** the KV heads: with TP=16 and 8 KV heads, ranks 0 and 1 both hold a full copy of KV head 0, each serving 4 of the 8 query heads in that group. This is correct but it doubles your KV cache memory, because the cache is now stored twice. That is a real and often-missed cost: going from TP=8 to TP=16 on an 8-KV-head model does not halve your per-GPU KV footprint, it keeps it flat while halving the weight footprint.

The same divisibility bites the MLP: `intermediate_size` must divide by TP degree, and for a model with, say, `intermediate_size = 28672`, TP=8 gives 3,584 per rank (fine) but TP=12 does not divide at all and the engine will simply refuse to load.

**📄 Paper:** Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — showed you can uptrain an MHA checkpoint into grouped-query form for a large KV-cache reduction at near-zero quality loss. It replaced the MQA-vs-MHA binary with a tunable middle, and it is the reason a modern 70B has a 320 KB/token cache instead of 2.5 MB.

**⚠ Trap:** "more TP is always better because each GPU holds fewer weights." Past the KV-head count, extra TP buys you weight-memory savings while *costing* you replicated KV, more collectives per token, and smaller per-rank GEMMs that underutilize the tensor cores. My rule: TP degree = the smallest power of two that fits weights + your target KV budget, capped at both the NVLink domain size and the KV-head count.

**🏋 Drill (10 min, unaided):** given `n_layers=80, n_heads=64, n_kv_heads=8, head_dim=128, hidden=8192, intermediate=28672`, state the legal TP degrees, the per-rank weight bytes and per-rank KV bytes per token at each, and the degree you would ship. Pass criterion: you produce the table without a calculator and you flag TP=16 as replicating KV.

### Pipeline parallelism for inference — how does it work, and derive the bubble cost for me.

Pipeline parallelism cuts the model by *depth*: stage 0 holds layers 0–19, stage 1 holds 20–39, and so on. A request's activations flow stage to stage as a single point-to-point send of shape `[tokens, hidden]` — for one token at hidden 8,192 in bf16 that is 16 KB crossing the wire *once per stage boundary*, versus TP's 160 all-reduces. That ratio is why PP is the cross-node scheme.

The cost is the bubble. If you feed a single batch through a *p*-stage pipeline, stage 1 sits idle while stage 0 works, and so on — utilization is 1/p. You recover it by splitting the batch into *m* microbatches so stages overlap. The classic GPipe schedule gives a bubble fraction of **(p − 1) / (m + p − 1)**: with p=4 stages and m=1 microbatch, 3/4 = 75% of your GPU-time is idle; with m=16, 3/19 = **15.8%**; with m=64, 3/67 = 4.5%.

For inference the arithmetic is different in an important way, and this is where candidates who only know training PP get caught. In *decode*, each "microbatch" is one token step for a subset of sequences, and continuous batching gives you a natural stream of them — so a well-fed decode pipeline runs at high occupancy because there is always another sequence's token step to inject. In *prefill*, a single long prompt is one big activation that traverses stages serially, so a 32k-token prefill on a 4-stage pipeline has a genuine 3-stage warm-up bubble on TTFT. Chunked prefill helps here by turning one prompt into many microbatches.

**📄 Paper:** Huang et al. (2019), *GPipe* gave the microbatched pipeline and the bubble formula; Narayanan et al. (*PipeDream*, and later 1F1B interleaved schedules in Megatron) reduced peak activation memory by interleaving forward and backward rather than doing all forwards then all backwards. For inference you only care about the forward half, which is why inference PP is simpler than training PP.

**⚠ Trap:** treating PP as free capacity. PP adds a serial hop per stage to *every* token's latency, and it makes load balancing brittle — if stage 2 holds one extra layer or a fatter LM head, that stage becomes the pipeline's clock and the others idle. Balance by measured per-stage milliseconds, not by layer count. The LM head on a 128k-vocab model is a genuinely large matmul (8192 × 128000 = 1.05B params, 2.1 GB in bf16) and putting it naively on the last stage alongside an equal layer count unbalances the pipeline.

**🗣 Say this in the room:** "Inside a node I use tensor parallelism because NVLink makes the all-reduce nearly free. Across nodes I use pipeline parallelism because it costs one activation send per boundary instead of 160 collectives per token. TP-inside, PP-across is the default layout, and I only deviate if the model doesn't fit that shape."

### When do you actually need more than one node for serving, and how do you compose TP with PP?

The honest answer for most product companies is: less often than people think. On 8×H100 (640 GB) you can serve a 70B in bf16 with 400+ GB of KV, or a 405B in fp8 (405 GB weights) with a thin cache, or a 200B-class MoE. Multi-node serving becomes mandatory in three cases: the model genuinely exceeds one node's HBM even quantized; you need a KV cache far larger than one node can hold (very long context, very high concurrency); or you are running disaggregated prefill/decode where the two roles live on different machines by design.

The composition rule follows the interconnect topology. Build a 2D mesh: TP along the fast axis (NVLink, within a node), PP along the slow axis (InfiniBand or Ethernet, across nodes). Sixteen GPUs across two HGX nodes becomes TP=8, PP=2 — each node holds half the layers, fully tensor-parallel within itself. Never TP=16, PP=1.

Data parallelism sits above both: you replicate the whole TP×PP unit and put a router in front. That is the axis you autoscale on, and it is the only axis where adding hardware gives you linear throughput.

**📐 Numbers you must know:** the interconnect hierarchy, in bytes/s, is the thing you reason with. HBM3 on H100 ≈ 3.35 TB/s. NVLink 4 ≈ 900 GB/s per GPU. PCIe Gen5 x16 ≈ 64 GB/s. InfiniBand NDR ≈ 400 Gb/s = 50 GB/s. 25 GbE ≈ 3.1 GB/s. Each step down is roughly 4–10×, and every parallelism decision is "which of these tiers does this data have to cross, and how many times per token."

**⚠ Trap:** proposing PP=8 with one GPU per stage to "avoid the all-reduce." You have now put 8 serial network hops on every token and made the pipeline maximally sensitive to stage imbalance, while giving up the ability to use NVLink at all. PP degree should be small — 2 or 4 — and TP should fill the node.

### Sequence and context parallelism — what problem do they solve that tensor parallelism doesn't?

The problem is that TP shards the *model* but every rank still materializes activations for the *full sequence*. For a 256k-token prefill, the activation tensors and the attention score matrix scale with sequence length regardless of TP degree, and the O(S²) attention intermediate is what actually explodes. Sequence parallelism shards along the token axis instead.

Two related mechanisms, and it is worth keeping them straight because interviewers do. **Sequence parallelism** in the Megatron sense is a memory optimization that pairs with TP: in the regions where TP leaves activations *replicated* — LayerNorm and dropout, which are not matmuls and so are not sharded — you instead shard along the sequence dimension and convert the TP all-reduce into a reduce-scatter followed by an all-gather. Same total communication volume, strictly less activation memory. It is basically free and modern stacks enable it with TP by default.

**Context parallelism** is the more aggressive thing: shard the sequence across ranks *including inside attention*, which requires each rank's queries to see every rank's keys and values. Ring Attention solves this by passing KV blocks around a ring while each rank computes partial attention against the block it currently holds, accumulating with the same online-softmax rescaling trick FlashAttention uses. Communication overlaps with compute, so with enough blocks the KV rotation hides behind the attention math entirely.

**📄 Paper:** Liu, Zaharia and Abbeel (2023), *Ring Attention with Blockwise Transformers* — made sequence length scale with device count by rotating KV blocks around a ring and overlapping the transfer with blockwise attention compute. It replaced "your context length is capped by one device's memory."

Where this matters for a product company: a 200k-token document-analysis prefill. Prefill FLOPs are ~2 × params × tokens for the linear layers plus the quadratic attention term; at 200k tokens the quadratic term stops being negligible. Context parallelism is what lets you spread that one request's prefill across 8 GPUs and cut TTFT roughly linearly, which is the difference between a 40-second and a 6-second wait on a Harvey- or Glean-style "analyze this contract" request.

**⚠ Trap:** enabling context parallelism for a chat workload with 2k prompts. CP adds ring communication to every layer and buys you nothing when the sequence already fits comfortably. It is a long-prefill feature. Turn it on per-deployment, not globally.

### Walk me through expert parallelism for a mixture-of-experts model. What does the all-to-all actually cost you?

An MoE layer replaces one big FFN with *E* expert FFNs plus a router that sends each token to *k* of them (typically k=2). The point is that parameters grow with E while FLOPs per token grow with k — you get a 400B-parameter model that computes like a 40B one. The serving consequence is a memory/compute inversion: you must hold all E experts in HBM but you only use k of them per token, so MoE serving is even more memory-capacity-bound than dense serving.

Expert parallelism shards experts across ranks: rank 0 holds experts 0–7, rank 1 holds 8–15, and so on. Now the routing is physical. After the router decides which experts each token wants, you perform an **all-to-all** — every rank sends each of its tokens' hidden vectors to whichever rank owns the chosen expert. The experts compute. Then a second all-to-all sends the results home. Two all-to-alls per MoE layer, on top of whatever TP all-reduces you already have.

All-to-all is the nastiest collective: unlike all-reduce, its cost is not reducible by tree/ring cleverness — every rank talks to every rank, so it scales as O(N) messages per rank, and it is exquisitely sensitive to *imbalance*. If the router sends 40% of a batch to one popular expert, that rank becomes the critical path and everyone waits. This is why MoE training uses auxiliary load-balancing losses and capacity factors, and why MoE *serving* needs expert-placement heuristics — replicating hot experts, or shuffling the expert→rank map based on observed routing statistics.

**⚠ Trap:** assuming an MoE's "active parameters" figure predicts its serving cost. A model advertised as "37B active out of 671B" needs the full 671B resident — 671 GB in fp8, meaning multiple nodes — even though each token only touches 37B worth of matmuls. Your GPU count is set by total params; your throughput is set by active params. Candidates who quote only the active number are sizing a fleet that cannot load the model.

**💰 Math:** compare serving a dense 70B (140 GB bf16, 8×H100 comfortably) against a 671B MoE in fp8 (671 GB weights, needs 2 nodes = 16 H100s minimum with almost no KV room, realistically 3–4 nodes). At $2.50/GPU-hr, that is $20/hr versus $60–80/hr of standing capacity — a 3–4× fleet cost — in exchange for materially better quality per generated token. Whether that trades well is an eval question, not an infra question, and I would insist on measuring quality-per-dollar on the actual task before committing to the MoE.

### How does the choice of parallelism interact with the KV cache?

This is the question that separates people who have read about TP from people who have sized a fleet. Each parallelism axis does something *different* to your cache, and getting it wrong produces an OOM at p99 concurrency rather than at deploy time.

**Tensor parallelism shards the KV cache.** Because each rank owns a subset of attention heads, it stores only those heads' K and V. TP=8 on a model with 8 KV heads means each rank holds 1/8 of the cache — 320 KB/token becomes 40 KB/token/rank. This is the main reason TP raises your concurrency ceiling and not just your weight capacity. But as noted, once TP exceeds the KV-head count you start replicating and this benefit stops.

**Pipeline parallelism partitions the cache by layer.** Stage 0 holds the KV for layers 0–19 only. Total cache across the pipeline is unchanged, but each stage's share is 1/p. Critically, a sequence's KV is spread across *all* stages, so a sequence cannot be evicted from one stage independently — eviction is a pipeline-wide decision, which makes preemption coarser.

**Data parallelism replicates nothing and shares nothing.** Each replica has its own cache. This is the one that creates the routing problem in the next question: a request's prefix cache lives on exactly one replica, and if you route the follow-up turn elsewhere you pay full prefill again.

**Context parallelism shards the cache by token position**, which is elegant for one long request and awkward for many short ones.

**📐 Numbers you must know:** per-rank KV bytes/token = `2 × L × H_kv × d_head × dtype / (TP × PP)` when TP ≤ H_kv. For the 70B example at TP=8, PP=1: 320 KB / 8 = 40 KB/token/rank. On an 80 GB H100 holding 17.5 GB of weights and ~10 GB of overhead, that leaves ~52 GB → 52e9 / 40e3 ≈ **1.3 M tokens** of cache per rank, which matches the whole-fleet number from the first question because the cache is sharded, not replicated. If you compute those two numbers and they disagree, you have made an error in one of them.

**⚠ Trap:** enabling fp8 KV cache and assuming it halves memory with no consequence. It does halve the cache — 320 KB/token to 160 KB/token, doubling concurrency — but K and V quantization error compounds across long contexts, and the degradation shows up specifically on long-context retrieval tasks (needle-in-haystack style), not on short chat evals. If you ship fp8 KV, your long-context eval is the gate, and I would not ship it on a legal or medical product without one.

### Your product routes requests round-robin across eight replicas of the same model. Why is that a bug, and what would you do instead?

It is a bug because round-robin destroys prefix-cache locality, and on a modern agentic or chat product the prefix is most of your compute.

Here is the mechanism. Automatic prefix caching means the engine hashes each block of tokens (typically 16-token blocks) and reuses the KV blocks for any prefix it has already computed. In a chat product, turn N+1's prompt is turn N's entire conversation plus a few hundred new tokens. If the request lands on the replica that served turn N, the engine finds the cached blocks and prefills only the delta. If it lands anywhere else, it recomputes the whole thing.

Do the arithmetic on a coding assistant. System prompt + repo context = 12,000 tokens; the user's new message = 300 tokens. Prefill FLOPs scale with tokens, so a cache hit turns a 12,300-token prefill into a 300-token prefill — a **41× reduction in prefill work** and a proportional cut in TTFT. With round-robin across 8 replicas, your hit probability is 1/8 = 12.5%, so you get 12.5% of that benefit. With session-affinity routing you get 90%+.

What to do instead, in increasing order of sophistication:

1. **Session affinity** — hash on conversation ID, route consistently. Cheap, works, and it is the 80% answer. Use consistent hashing so a replica leaving does not reshuffle everything.
2. **Prefix-aware routing** — the router hashes the leading N tokens of the prompt and routes on that hash, so two different users sharing the same 12k system prompt land on the same replica. This is what a production LLM gateway does, and it handles the case where there is no session ID.
3. **Cache-aware load balancing** — the router queries each replica for whether it holds a given prefix (or maintains a shadow radix tree of what it has routed where) and balances locality against queue depth. SGLang's router does approximately this.
4. **Shared/tiered KV** — push evicted KV blocks to a CPU-memory or NVMe tier, or to a cross-node KV store, so a miss on the local GPU cache is a fast fetch rather than a recompute. Mooncake (Qin et al., 2024) is the reference architecture for treating KV as a first-class, disaggregated, network-addressable store.

**⚠ Trap:** pure locality routing with no load term. If your router only optimizes prefix hits, one popular system prompt pins all its traffic to a single replica and you get a hot shard while seven GPUs idle. The routing objective is `score = α·(prefix match length) − β·(queue depth)`, and you tune β until the load histogram flattens. I have seen teams ship pure-affinity routers and discover the imbalance only when p99 tripled at a traffic peak.

**💰 Math:** 200k requests/day, 12k-token cached prefix, self-hosted. Prefill at 12,300 tokens costs roughly 2 × 70e9 × 12,300 = 1.72 PFLOPs; at an achievable ~400 TFLOPS/GPU effective on 8 GPUs (3.2 PFLOPS) that is ~0.54 s of pure prefill compute. Cutting to 300 tokens makes it ~0.013 s. Across 200k requests you save 200,000 × 0.53 s = 106,000 GPU-seconds/day ÷ 8 = 3.7 fleet-hours/day of an 8-GPU box — at $20/hr that is $74/day or **$2,200/month**, and more importantly it is the difference between needing 3 replicas and needing 5.

### What is disaggregated prefill/decode, and when would you split them onto separate hardware?

The mental model: prefill and decode are two completely different computational workloads that we historically ran on the same box for no better reason than convenience. Prefill is a big dense matmul over thousands of tokens — compute-bound, high arithmetic intensity, saturates tensor cores. Decode is one token at a time — it reads the entire model's weights from HBM to produce a single token, so it is memory-**bandwidth**-bound with an arithmetic intensity near 1. Running them on the same GPU means each interferes with the other: a long prefill blocks decode steps and you see inter-token latency spike (the "stutter" users notice), while decode's tiny batches leave tensor cores idle.

Disaggregation runs prefill on one pool and decode on another, transferring the computed KV cache over the network between them. Prefill nodes can be optimized for FLOPS (and can run at high TP for low TTFT); decode nodes can be optimized for HBM capacity and bandwidth (and want large batch sizes for throughput). Each pool autoscales on its own signal.

**📄 Paper:** Zhong et al. (2024), *DistServe* — showed that disaggregating prefill and decode onto separate resources lets you meet TTFT and TPOT SLOs simultaneously that were unachievable when colocated, because the two phases stop interfering. Mooncake (Qin et al., 2024) took the same idea to production scale with a KV-centric disaggregated architecture.

The alternative, and the one most teams should try first, is **chunked prefill**: break a long prefill into chunks of a few hundred tokens and interleave them with decode steps in the same batch, so no single prefill monopolizes the GPU for more than a chunk's duration. Sarathi-Serve (Agrawal et al., 2024) is the reference for this, and it is the default scheduling mode in modern vLLM.

**The decision rule I use:** chunked prefill first — it is a config flag and it solves the ITL-stutter problem at the cost of slightly worse TTFT. Go to full disaggregation only when (a) you are at a scale where the pools are each many nodes, (b) your prefill:decode token ratio is extreme in one direction (heavy RAG or long-document workloads push it toward prefill), and (c) you have a fast enough fabric to move KV between pools. Moving 12k tokens × 320 KB = 3.84 GB of KV over InfiniBand NDR at 50 GB/s takes 77 ms — acceptable; over 25 GbE at 3.1 GB/s it takes 1.24 s, which destroys the TTFT you were trying to protect. **Disaggregation is an InfiniBand feature.**

**⚠ Trap:** proposing disaggregation in a design round for a startup serving 50 QPS. It is real engineering with real operational cost, and at that scale the answer is chunked prefill plus prefix caching. Naming the right technique for the wrong scale reads as pattern-matching rather than judgment.

### Suppose I give you a fixed budget of 8 H100s and ask for the lowest possible TTFT on a 13B model. Do you use TP=8 or DP=8?

This is a good question because the naive answers are both wrong, and the right answer is "it depends on load, and here is the crossover."

A 13B in bf16 is 26 GB — it fits on one H100 with 44 GB left for KV. So both layouts are legal. TP=8 puts one model across all eight GPUs; DP=8 puts eight independent replicas behind a router.

**At batch size 1 / zero contention**, TP=8 wins on latency. Decode reads weights from HBM every step; with TP=8 each GPU reads only 26/8 = 3.25 GB instead of 26 GB, and the eight reads happen in parallel. Theoretical per-token time drops from 26 GB ÷ 3.35 TB/s = 7.8 ms to 0.97 ms, plus ~1–2 ms of collective overhead. Call it 3 ms/token versus 8 ms/token — TP is roughly 2.5× faster per stream even after paying for the all-reduces. Prefill similarly parallelizes, cutting TTFT.

**At high load**, DP=8 wins on throughput by a wide margin, because eight independent engines each running a large continuous batch achieve far better GPU efficiency than one engine paying 160 collectives per step. And the throughput number is what sets your cost per token.

So the real decision rule: **TP for latency-critical, low-concurrency; DP for throughput-critical, high-concurrency; and the tie-breaker is your KV budget.** TP=8 also gives you 8× the KV cache for a single request, which matters if you serve very long contexts.

**🗣 Say this in the room:** "For a model that fits on one GPU I default to data parallelism, because replicas scale throughput linearly and avoid all collective overhead. I only add tensor parallelism when the model doesn't fit, when a single request's KV needs more than one GPU's memory, or when I have a hard TTFT SLO that DP can't hit at batch size one. And I'd measure both — this is a 30-minute benchmark, not a debate."

**⚠ Trap:** benchmarking this at batch size 1 and shipping the winner. Almost every published "TP is faster" microbenchmark is a batch-1 measurement, and almost every production workload is not batch-1. Benchmark at your *measured* concurrency distribution, or you will ship a configuration that is 2× more expensive per token in exchange for a latency win nobody experiences under load.
