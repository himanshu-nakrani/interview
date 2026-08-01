### Every attention variant since 2019 — MQA, GQA, sliding window, MLA — what was the actual pressure driving all of them? Give me the one-sentence thesis first.

The thesis is this: **nothing in that list makes the model faster at math. Every one of them makes the model cheaper to remember.** The transformer's compute cost was never the binding constraint at serving time. The binding constraint is that autoregressive decode must re-read, from HBM, every key and value vector produced by every previous token, at every layer, for every single token it emits. That re-read is pure memory traffic. So the architecture evolution from 2019 onward is a sustained engineering campaign to shrink the number of bytes per token per layer that the cache has to hold — and every quality compromise in MQA, GQA, sliding window and MLA was accepted in exchange for bytes.

The backend bridge that makes this feel inevitable: you already know that a system whose working set exceeds cache becomes latency-bound on memory rather than CPU-bound, and that the fix is never "buy a faster CPU," it is "make the working set smaller." The KV cache is a per-request memo table whose eviction policy you do not control and whose size grows linearly with the conversation. On an H100 you have 80 GB of HBM and roughly 3.35 TB/s of bandwidth. Model weights are a fixed cost you pay once per decode step and amortize across the whole batch. KV cache is a *per-sequence* cost that does **not** amortize across the batch — batch 32 means 32 separate caches, all of which must be streamed through the attention kernel every step. That asymmetry is the whole story.

**📐 Numbers you must know:** under vanilla MHA, KV bytes per token per layer = `2 · d_model · bytes_per_elem`, because the K and V projections each have output width `d_model`. For Llama-3-70B (`d_model` 8192, 80 layers, fp16) that is 2 × 80 × 8192 × 2 = 2,621,440 bytes = **2.5 MiB per token**. A single 8k-token conversation would have needed 20 GiB of cache. You could not have fit *one user* alongside the weights on a pair of 80 GB cards. Llama-3-70B ships with GQA and 8 KV heads instead, which cuts that to 320 KiB/token and 2.5 GiB at 8k. That 8× is not a micro-optimization; it is the difference between the model being servable and not.

**🗣 Say this in the room:** "The architectural history since 2019 is a bytes-per-token story, not a FLOPs story. Prefill is compute-bound and we already solved it with FlashAttention; decode is bandwidth-bound and the only lever is making the KV cache smaller. MQA, GQA, sliding window and MLA are four different bets on which redundancy in K and V you can throw away most cheaply."

**⚠ Trap:** answering "why GQA?" with "it's faster." It is not faster in FLOPs — the attention score computation is essentially unchanged because you still have all your query heads. It is faster because it moves fewer bytes and because it frees HBM for more concurrent sequences, which raises throughput. If you say "fewer FLOPs" an interviewer who has served a model will stop trusting you for the rest of the round.

### Derive the KV cache size formula for me. I don't want the formula recited, I want it built up from what's actually stored.

Start from what a decode step needs and nothing else. To produce token `t+1`, the attention block at each layer computes a query vector for position `t`, dots it against the keys of positions `0..t`, softmaxes, and takes a weighted sum of the values of positions `0..t`. The keys and values of positions `0..t-1` are functions only of tokens already emitted — they are immutable. So you cache them. You cache *nothing else*: not the queries (a query is used once and discarded), not the attention weights, not the FFN activations. That "queries are not cached" observation is the seed of the entire MQA/GQA idea, because it means the query side and the key/value side can have *different head counts*.

Now count. Per token, per layer, you store one K vector and one V vector **per KV head**. Each is `d_head` elements. So:

```
elements_per_token_per_layer = 2 · n_kv_heads · d_head
bytes_per_token             = 2 · n_layers · n_kv_heads · d_head · bytes_per_elem
total_bytes                 = bytes_per_token · seq_len · batch_size
```

The leading `2` is K-and-V, not a fudge factor. `bytes_per_elem` is 2 for fp16/bf16, 1 for fp8, 0.5 for int4. Note what is *absent*: `d_model` does not appear, `n_query_heads` does not appear, `d_ff` does not appear, and the number of parameters does not appear. Under MHA specifically, `n_kv_heads · d_head = d_model`, so the formula collapses to `2 · n_layers · d_model · bytes_per_elem` — which is why the MHA number is so easy to compute in your head from a config file.

```python
def kv_bytes_per_token(n_layers, n_kv_heads, d_head, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * d_head * dtype_bytes

# Llama-3-8B: 32 layers, 8 KV heads, d_head 128, bf16
kv_bytes_per_token(32, 8, 128)      # 131072  -> exactly 128 KiB/token
# Llama-3-70B: 80 layers, 8 KV heads, d_head 128
kv_bytes_per_token(80, 8, 128)      # 327680  -> 320 KiB/token
```

**📐 Numbers you must know:** Llama-3-8B is **128 KiB per token**, so exactly **1 GiB of KV per 8k-token sequence**. That is the single most useful anchor in this entire section — every other model is a ratio away from it. Llama-3-70B is 2.5× that per token (80 layers vs 32, same KV width) at 320 KiB/token.

**⚠ Trap:** forgetting that this is per *sequence*, and that with continuous batching your concurrency limit is `usable_HBM / (bytes_per_token · max_context)`. Engineers routinely compute the single-sequence number, see "2.5 GiB, fine," and then are astonished that the server OOMs at 12 concurrent users. The KV cache is the only component of a serving deployment whose memory grows with *both* your traffic and your context window, multiplicatively.

### Compute Llama-3-70B's KV cache for one 8k-token sequence, out loud, and then tell me what it would have been without GQA.

Config first, because half the candidates who fail this fail it by not knowing where the numbers come from: Llama-3-70B has 80 transformer layers, `d_model` 8192, 64 query heads, **8 KV heads**, and `d_head` = 8192/64 = 128. Serving in bf16.

Per token, per layer, the KV width is `n_kv_heads · d_head` = 8 × 128 = **1024 elements** for K and 1024 for V. So 2048 elements per layer per token, × 2 bytes = 4096 bytes = 4 KiB per layer per token. Across 80 layers: 4 KiB × 80 = **320 KiB per token**. At 8192 tokens: 320 KiB × 8192 = 2,621,440 KiB = 2,560 MiB = **2.5 GiB for one sequence**.

Now the counterfactual. Without GQA — plain MHA, 64 KV heads — the KV width per layer becomes 64 × 128 = 8192 = `d_model`. That is exactly 8× wider, so **2.5 MiB per token and 20 GiB per 8k sequence**. Twenty gigabytes. For one user. The bf16 weights are 70e9 × 2 = 140 GB, so on a 2×H100 node (160 GB) you have roughly 20 GB free after weights, and MHA would give you *one* concurrent 8k conversation with nothing left for activations. GQA gives you 20 GiB / 2.5 GiB ≈ 8 concurrent sequences on the same hardware, which is the difference between a demo and a product.

**💰 Math:** put a price on it. Suppose you must serve 500 concurrent 8k sessions. Under GQA-8 you need 500 × 2.5 GiB = 1250 GiB of KV plus 140 GB of weights per replica; with TP=8 on an 8×H100 node (640 GB), weights take 140 GB leaving ~450 GB usable after activations and fragmentation, so ~180 sessions per node → 3 nodes. Under MHA you would need 500 × 20 GiB = 10 TB of KV → roughly 22 nodes. At an on-demand rate of about $2–3/GPU-hour for H100 (**📅 Volatile:** verify current pricing before your loop), 8 GPUs/node × 24h × 30d = 5,760 GPU-hours/node/month ≈ $14.4k at $2.50. Three nodes is ~$43k/month; twenty-two nodes is ~$317k/month. **GQA is a $274k/month line item on a single mid-sized deployment.** That is why Meta shipped it.

**🗣 Say this in the room:** "320 KiB per token, 2.5 GiB at 8k, and it would have been 20 GiB under MHA — that 8× is exactly the ratio of 64 query heads to 8 KV heads, because KV cache scales with KV heads only."

### Why is decode memory-bandwidth-bound while prefill is compute-bound? Show me the arithmetic intensity.

The mental model: prefill is a GEMM, decode is a GEMV, and a GEMV has essentially no data reuse. In prefill you push T tokens through the weights simultaneously — each weight byte you load from HBM gets used T times. In decode with batch size B you push B tokens through the weights — each weight byte gets used B times. Arithmetic intensity (FLOPs per byte moved) is therefore roughly T for prefill and roughly B for decode, and the hardware has a fixed break-even point.

**📐 Numbers you must know:** that break-even is the machine balance ratio — peak FLOP/s divided by peak bytes/s. An H100 SXM does roughly 990 TFLOP/s of dense bf16 and has roughly 3.35 TB/s of HBM3 bandwidth, so 990e12 / 3.35e12 = **~295 FLOP per byte**. Any kernel below that intensity is bandwidth-bound on that card, full stop. (**📅 Volatile:** re-derive for whatever silicon you are actually on — H200 is ~4.8 TB/s and B200 higher still, which *lowers* the break-even and makes decode relatively better off.) Below that intensity you are bandwidth-bound; above it, compute-bound. Prefill with T = 2048 tokens sits at intensity ~2048 — deep in compute-bound territory, which is why FlashAttention and tensor cores matter there. Decode at batch 1 sits at intensity ~1. **You need a batch of a few hundred to make decode compute-bound on an H100**, and most latency-sensitive products cannot batch that deep.

Concretely for a 70B model at batch 1: one decode step reads all 140 GB of bf16 weights and performs ~2N = 140 GFLOP. Memory time = 140 GB / 3.35 TB/s = **41.8 ms**. Compute time = 140e9 / 990e12 = **0.14 ms**. The GPU is idle 99.7% of the time waiting on HBM. Ceiling is 1/0.0418 = ~24 tokens/s. Split across TP=2 it is ~48 tok/s in theory, ~30–40 in practice after collectives.

Here is the part that connects to this whole section, and the part most candidates miss. Weight traffic amortizes across the batch: 140 GB read once serves all B sequences. **KV traffic does not.** Each sequence's attention must stream its own cache. At batch 32 with 8k contexts on Llama-3-70B, KV bytes read per decode step = 32 × 2.5 GiB = **80 GiB**, against 130 GiB of weights. So attention is already ~38% of your bandwidth budget. Under MHA it would have been 32 × 20 GiB = 640 GiB — **five times the weight traffic**, and your tokens/sec would be dominated entirely by cache reads.

**⚠ Trap:** believing that FlashAttention fixes decode. FlashAttention is a tiling and recomputation strategy that avoids materializing the T×T attention matrix — it is a *prefill* win and an *activation memory* win. At decode T_query = 1, there is no T×T matrix to avoid; the kernel (FlashDecoding and friends) is still fundamentally reading the whole KV cache. No kernel can make you read fewer bytes than you stored. Only the architecture can.

**🗣 Say this in the room:** "Machine balance on H100 is about 295 FLOP per byte. Prefill runs at intensity ~T so it's compute-bound; decode runs at intensity ~batch-size so it's bandwidth-bound until batch reaches the low hundreds. And crucially, weight reads amortize over the batch but KV reads don't — which is why the architecture had to change and the kernels couldn't save us."

### An interviewer asks what it costs to serve a 70B model. Walk me through your answer.

I want to flag the trap in the question before answering it, because this is the single most common way a strong backend engineer sounds like a paper-reader in a serving round.

**⚠ Trap:** answering "70 billion parameters × 2 bytes = 140 GB, so two H100s." Parameter count is the *floor*, not the cost. It tells you the static residency. It tells you nothing about how many users you can serve, which is the actual question being asked when someone says "cost." Cost per token is `hardware_cost_per_second / tokens_per_second`, and tokens/sec is governed by how many sequences you can hold concurrently, which is governed by leftover HBM after weights, which is governed by KV bytes per token. Parameter count and serving cost are related by a chain with three more links in it.

The structured answer has four terms, and I would enumerate them explicitly:

1. **Weights.** 70e9 × 2 bytes = 140 GB bf16; 70 GB at fp8; ~40 GB at int4 (weight-only, with scales). This is a hard floor and it sets your minimum GPU count.
2. **KV cache.** 320 KiB/token → at 8k context, 2.5 GiB per concurrent sequence. This is the term that scales with traffic.
3. **Activations and workspace.** Logits alone are `batch × vocab × 4 bytes` in fp32 — Llama-3's vocab is 128,256, so a batch of 256 is 256 × 128,256 × 4 = 131 MB just for one logits tensor, and chunked-prefill workspaces, CUDA graph pools and NCCL buffers add several GB. Budget 5–10% of HBM.
4. **Fragmentation and headroom.** vLLM's `gpu_memory_utilization` default of 0.9 exists for a reason.

Then the actual cost derivation. Take 8×H100, TP=8. Weights 140 GB; usable KV pool ≈ 640 × 0.9 − 140 − ~20 = **~416 GB**. At 8k context that is 416/2.5 ≈ **166 concurrent sequences**. If each generates at ~25 tokens/s under load, node throughput ≈ 4,150 tok/s. At $2.50/GPU-hr × 8 = $20/hr = $0.00556/s, cost per output token = 0.00556 / 4150 = **$1.34e-6**, i.e. **$1.34 per million output tokens** of raw compute (**📅 Volatile:** GPU rates move; re-derive with current numbers).

**🗣 Say this in the room:** "Parameter count sets the floor — 140 GB in bf16 — but it doesn't set the cost. Cost per token is driven by concurrency, and concurrency is driven by leftover HBM divided by KV bytes per token. For Llama-3-70B that's 320 KiB/token, so on an 8×H100 node I get roughly 166 concurrent 8k sessions and about $1.30 per million output tokens of raw GPU cost."

### Walk me through multi-query attention. What exactly changes in the tensor shapes?

MQA is one idea: **keep all N query heads, but project a single shared key head and a single shared value head, and let every query head attend against that same K and V.** Shazeer's framing was "one write-head" — many readers, one writer.

Shapes. Under MHA with `d_model` = 4096, 32 heads, `d_head` = 128, the projections are `W_Q, W_K, W_V ∈ R^{4096×4096}` and activations are `[B, 32, T, 128]` for Q, K and V alike. Under MQA, `W_Q` is unchanged at 4096×4096, but `W_K` and `W_V` shrink to **4096×128**. K and V activations become `[B, 1, T, 128]`. The score computation `Q @ K^T` broadcasts the single KV head against all 32 query heads: `[B, 32, T, 128] @ [B, 1, 128, T] → [B, 32, T, T]`. Output projection is unchanged because you still concatenate 32 heads' worth of output.

**📄 Paper:** Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need* — showed that collapsing K and V to a single head cuts decode-time memory traffic by roughly the head count with only a small quality loss, replacing the assumption that KV head count must equal query head count.

The cache effect is dramatic and immediate: `bytes = 2 · n_layers · 1 · d_head · b`. For a 32-layer, `d_head` 128 model in bf16 that is 2 × 32 × 1 × 128 × 2 = 16,384 bytes = **16 KiB/token**, versus 512 KiB/token under 32-head MHA — a 32× reduction. It also cuts parameters slightly: you shed `2 · n_layers · d_model · (d_model − d_head)` weights, about 2 × 32 × 4096 × 3968 = 1.04B params on a 7B model, which is not nothing.

**⚠ Trap:** thinking MQA reduces attention FLOPs. It does not, materially. You still compute 32 heads' worth of `QK^T` and 32 heads' worth of `attn @ V`. The K/V *projection* GEMMs get 32× cheaper, but those are a small slice of the block. MQA is a bandwidth and capacity win, not a compute win — and at prefill, where you are compute-bound anyway, MQA is nearly free of benefit.

The other thing to know: MQA's benefit at decode comes partly from cache size and partly from a second-order effect — with one KV head, the attention kernel's KV reads become a much smaller, more cache-friendly working set, and on a tensor-parallel deployment the KV head must be *replicated* across ranks rather than sharded, which is MQA's ugliest serving wart and one reason GQA won.

### MQA gives you a 32–64× cache reduction. So why isn't every model MQA?

Because it costs quality, and the loss is not uniform across capabilities — it concentrates exactly where product teams notice.

The mechanism: attention heads specialize. In a trained transformer some heads do previous-token copying, some do induction (find the earlier occurrence of the current token and copy what followed it), some do syntactic agreement, some do long-range retrieval. Specialization requires each head to be able to *look for different things* — and "what a head looks for" is jointly determined by its query projection **and** its key projection. Collapse K to a single shared head and you have forced all 32 heads to search the same key subspace. They can still weight it differently through their queries, but they can no longer disagree about what the relevant coordinate system is. Empirically the heads that suffer most are the sharp, retrieval-like heads — precisely the ones that make in-context recall work.

The reported symptoms, in the GQA paper and in practice: MQA shows measurable degradation on tasks with long inputs and precise extraction (summarization, question answering over a supplied document), and it is reported to be less stable to train — some groups needed extra care to avoid training divergence with a single KV head. The degradation on short-form multiple-choice benchmarks is small enough that you can talk yourself into MQA and then discover the problem only when a customer complains that the model quotes the wrong clause from a contract.

**📄 Paper:** Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — introduced the interpolation between MHA and MQA, and the uptraining recipe, showing GQA reaches close to MHA quality at close to MQA speed. It replaced the binary choice with a tunable knob.

**⚠ Trap:** evaluating an MQA/GQA change on MMLU and declaring victory. MMLU items are a few hundred tokens with the answer in the model's weights, not in the context. The capability MQA damages is *in-context retrieval over long inputs*, which MMLU does not test at all. If you propose reducing KV heads, the eval you must run is long-input extraction on your own corpus — needle-style multi-fact retrieval at your real context lengths — and you must run it at the P95 context you actually see in production, not at 2k.

**🗣 Say this in the room:** "MQA is a 32× cache win that costs you the heads' ability to disagree about what to look for. It shows up as degraded long-input retrieval, not as degraded MMLU, so it's easy to ship by accident. GQA exists because 8 KV heads recovers nearly all of that quality for 1/8th of MHA's cache."

### Explain GQA precisely. What is a "group," and how do the shapes work inside the kernel?

GQA partitions the `N` query heads into `G` groups of `N/G` heads each, and gives each *group* one shared K head and one shared V head. G = N is MHA; G = 1 is MQA; everything in between is GQA. Llama-3 uses N = 64 (70B) or 32 (8B) query heads with G = 8 in both cases, so 8 and 4 query heads per group respectively. Mistral-7B and Qwen2-7B likewise use 8 KV heads.

Shapes, which is what the interviewer actually wants:

```
Q: [B, N,  T, d_head]         # N  = n_query_heads,  e.g. 32
K: [B, G,  T, d_head]         # G  = n_kv_heads,     e.g. 8
V: [B, G,  T, d_head]
```

The kernel needs `Q @ K^T` per head. Two ways to get there. The naive way, which every from-scratch implementation uses, is to expand K and V along the head axis by `N/G` before the matmul — this is what HuggingFace's `repeat_kv` does. The correct way, which every production kernel uses, is to *not* materialize the expansion at all: reshape Q to `[B, G, N/G, T, d_head]` and use a broadcast matmul against `K` at `[B, G, 1, d_head, T]`, so the shared K head is read from HBM **once per group** and reused `N/G` times out of registers/SMEM.

That distinction is the entire performance story, and I ask about it in interviews. If you `repeat_interleave` K and V into a full `[B, N, T, d_head]` tensor before calling attention, you have just written the whole expanded cache back to HBM and thrown away every byte GQA saved on the read path. The *storage* is still G heads, so you keep the capacity win — but the *bandwidth* win at decode evaporates. On a bandwidth-bound workload that is most of the benefit.

```python
def repeat_kv(x, n_rep):                 # x: [B, G, T, D]
    if n_rep == 1:
        return x
    B, G, T, D = x.shape
    return x[:, :, None].expand(B, G, n_rep, T, D).reshape(B, G * n_rep, T, D)
```

Note that `expand` is a stride-0 view — free — and the cost is entirely in the `reshape`, which forces a materializing copy. `torch.nn.functional.scaled_dot_product_attention` gained an `enable_gqa=True` flag precisely so you can pass the unexpanded K and V and let the backend broadcast internally; use it when available rather than expanding by hand.

**⚠ Trap:** assuming the number of *groups* equals the number of query heads per group. A "GQA-8" model has 8 KV heads (8 groups); with 64 query heads that is 8 queries per group. People flip this constantly and then compute the cache 64× wrong in the wrong direction.

### Does GQA reduce parameters or FLOPs, or only memory? Be precise.

Precise answer: it reduces parameters slightly, reduces the K/V projection FLOPs proportionally, leaves the core attention FLOPs essentially unchanged, and reduces KV cache bytes by exactly `N/G`. Three different numbers, and conflating them is a tell.

**Parameters.** `W_K` and `W_V` shrink from `d_model × d_model` to `d_model × (G·d_head)`. For Llama-3-70B: MHA would be 2 × 8192 × 8192 = 134.2M params per layer for K+V; GQA-8 is 2 × 8192 × 1024 = 16.8M. Saved 117.4M per layer × 80 layers = **9.4B parameters**. That is a real and substantial parameter saving — and Meta spent it on a wider FFN. So "GQA saves parameters" is true but misleading: in practice architects hold total parameters roughly constant and redirect the budget.

**FLOPs.** The K and V projection GEMMs shrink 8×: from 2 × 2 · T · d_model² to 2 × 2 · T · d_model · (G·d_head). But the two attention matmuls — `QK^T` at `2·N·T²·d_head` and `attn@V` at the same — are unchanged, because N query heads still produce N attention maps. Under MHA those two terms plus the four projections are all `O(T·d_model²)` or `O(T²·d_model)`; cutting K/V projections removes maybe 8–10% of the block's FLOPs at short T and much less at long T. **GQA is not a meaningful FLOP optimization.**

**Memory.** Exactly `N/G` reduction in KV cache bytes, and exactly `N/G` reduction in KV bytes read per decode step *if your kernel broadcasts instead of expanding*. This is the number that matters.

**🗣 Say this in the room:** "GQA cuts KV cache bytes by the group ratio — 8× for Llama-3 — cuts K/V projection parameters by the same ratio, and leaves the attention FLOPs alone. If someone tells you GQA is a compute optimization, they've read the abstract and not the config."

### Your team has an MHA checkpoint and wants GQA without a full retrain. Walk me through uptraining.

This is a real and well-specified procedure, and knowing it separates people who have read the GQA paper from people who have heard of it.

**Step 1 — construct the initialization by mean-pooling.** You have `n_layers × N` key heads and `N` value heads per layer. Partition them into `G` contiguous groups of `N/G`. For each group, the new shared K head is the **element-wise mean of the projection matrices** of the heads in that group, and likewise for V. Concretely, if `W_K ∈ R^{d_model × N·d_head}` is viewed as N blocks of `d_model × d_head`, the group-g block of the new `W_K' ∈ R^{d_model × G·d_head}` is `mean(W_K_blocks[g·N/G : (g+1)·N/G])`. Query projections, output projection, embeddings, FFN — all copied unchanged.

```python
# W_k: [d_model, N * d_head]  ->  [d_model, G * d_head]
def meanpool_kv(W_k, N, G, d_head):
    blocks = W_k.reshape(W_k.shape[0], N, d_head)     # [d_model, N, d_head]
    grouped = blocks.reshape(W_k.shape[0], G, N // G, d_head)
    return grouped.mean(dim=2).reshape(W_k.shape[0], G * d_head)
```

**Step 2 — uptrain.** Continue pretraining on the original pretraining distribution for a small fraction α of the original pretraining compute. The GQA paper's headline recipe uses **α = 5%**, and reports that this recovers quality close to the original MHA checkpoint while decoding at close to MQA speed. Use the original data mix, not your fine-tuning data — you are repairing a representation, not teaching a task. A short LR warmup at a fraction of the original peak LR is standard.

**Step 3 — re-do post-training.** This is the step people skip and then get burned by. Your instruction tuning, DPO/RLHF and safety alignment were fit to the *old* attention. After uptraining, the base model has drifted. You must re-run the post-training stack on the uptrained base, and you must re-run your safety evals, because refusal behavior is notoriously brittle to base-model changes.

**💰 Math:** α = 5% is the whole selling point. If the original 70B pretrain was 15T tokens at C = 6ND = 6 × 70e9 × 15e12 = 6.3e24 FLOP, then 5% is 3.15e23 FLOP. At 400 TFLOP/s effective per H100 (≈40% MFU), that is 3.15e23 / 4e14 = 7.9e8 GPU-seconds = 219,000 GPU-hours ≈ **1,140 H100-days**, or about 6 days on a 200-GPU cluster. Compare the full pretrain at 4.4M GPU-hours. **You buy an 8× serving-memory reduction for ~5% of the training bill** — that is an obviously correct trade and it is why every open-weights family converted.

**⚠ Trap:** shipping the mean-pooled checkpoint *without* uptraining because "the eval looked okay." Mean-pooling alone typically costs a visible amount of quality; it is an initialization, not a conversion. And the eval that looks okay will be a short-context benchmark, per the earlier trap.

### Why mean-pool the KV heads rather than pick one head, or randomly initialize the new ones?

Because mean-pooling is the choice that preserves the most of the function the network already computes, and you can argue that from linearity rather than from vibes.

K and V are *linear* projections. The attention logit for query head i against a mean-pooled key is `q_i · (mean_g W_K^{(j)})^T x = mean_g (q_i · W_K^{(j)T} x)` — the mean of the logits the original heads would have produced. So mean-pooling gives you the **average of the group's attention patterns** before softmax, which is a smooth, low-variance approximation to what the group collectively did. Selecting a single head throws away `N/G − 1` heads' learned structure entirely and biases you toward whichever head you happened to pick. Random init throws away all of it and forces the uptraining to relearn key/value subspaces from scratch, which is far more than 5% of pretraining compute.

There is a second-order argument that matters in practice: the *residual stream* downstream of attention has a learned scale, and the output projection `W_O` was fit assuming attention outputs with a particular magnitude. Mean-pooling preserves magnitude far better than summing (which would inflate by `N/G`) or than random init (which produces near-orthogonal, uninformative outputs). Preserving activation statistics is what keeps the loss from spiking in the first hundred uptraining steps.

**⚠ Trap:** mean-pooling across the *wrong* grouping. Heads must be grouped **contiguously in the same order the tensor-parallel sharding will use**, because at TP time head `h` lives on rank `h // (N/tp)`. If you mean-pool heads {0, 8, 16, 24} into a group but TP splits heads {0..7} onto rank 0, then rank 0 needs a KV head assembled from heads living on four different ranks — you have just created an all-gather in the middle of your attention block. Group contiguously. Every production implementation does.

A worthwhile nuance for a senior answer: mean-pooling is the *published* recipe, not necessarily the optimal one. You could weight heads by the norm of their contribution, or do a rank-1 SVD of the stacked group projections, or learn the pooling weights during a brief warm-up. I have not seen a result showing these reliably beat the mean by enough to justify the complexity — so the rule I enforce is: mean-pool, uptrain at 5%, and spend the saved engineering effort on the long-context eval instead.

### If MQA is 32× and GQA-8 is only 8×, why did GQA stick as the industry default rather than MQA?

Because GQA is the point on the curve where the quality loss becomes indistinguishable from noise while the memory win is still overwhelming — and because of a serving detail that has nothing to do with quality at all.

The quality argument: going from 64 KV heads to 8 removes 87.5% of the cache. Going from 8 to 1 removes another 10.9% of the *original* cache. So the marginal memory benefit of MQA over GQA-8 is small — 8× versus 9.1× total reduction relative to MHA — while the marginal quality cost is disproportionately large, because you have gone from "eight independent key subspaces the heads can specialize over" to "one." Diminishing returns on the memory axis, accelerating cost on the quality axis. GQA-8 sits right at the knee.

The serving argument, which is the one people forget: **tensor parallelism**. With TP = 8 and 8 KV heads, each rank owns exactly one KV head and the attention block shards perfectly — no replication, no extra communication. With MQA and TP = 8, the single KV head must be *replicated* on all 8 ranks, so you store 8 copies of the cache across the node and your aggregate KV memory is 8× the theoretical minimum. MQA's memory advantage partially evaporates the moment you shard. GQA with `n_kv_heads` chosen equal to your intended TP degree is the configuration that composes cleanly with the rest of the stack. That is not a coincidence — 8 KV heads showing up in Llama-3, Mistral, Qwen2 and others is a deliberate alignment with 8-GPU nodes.

**📐 Numbers you must know:** relative to MHA at 64 heads, GQA-8 stores 12.5% of the cache; MQA stores 1.6%. Relative to *GQA-8*, MQA saves a further 87.5% of what remains — which sounds huge until you notice that on a 2.5 GiB cache it takes you from 2.5 GiB to 0.31 GiB while the weights are still 140 GB. The absolute win has stopped mattering.

**🗣 Say this in the room:** "GQA-8 gets you 87.5% of the cache reduction MQA offers, at a fraction of the quality cost, and it shards cleanly at TP=8 where MQA has to replicate. That's why 8 KV heads is the number you see across Llama-3, Mistral and Qwen — it's co-designed with an 8-GPU node."

### Someone proposes a model with 64 query heads and 3 KV heads because 3 divides the cache nicely. What do you say in review?

I reject it, and the reason is arithmetic rather than taste.

First, 64 is not divisible by 3, so you cannot form equal groups — you would need groups of 22, 21, 21, which breaks the clean broadcast in every attention kernel and forces either padding or a ragged implementation. Every production kernel (FlashAttention's GQA path, vLLM's paged kernels, TensorRT-LLM) assumes `n_query_heads % n_kv_heads == 0`. Violate it and you fall off the fast path onto a fallback, or the model simply fails to load.

Second, and more importantly, `n_kv_heads` must play nicely with your tensor-parallel degree. The constraint chain is:

```
n_query_heads % n_kv_heads == 0        # groups must be equal
n_query_heads % tp_size    == 0        # Q shards across ranks
n_kv_heads    % tp_size    == 0        # ...or KV gets replicated
```

With `n_kv_heads = 3` and any TP degree above 1 except 3, the third condition fails and you replicate KV across ranks. At TP=8 with 3 KV heads, vLLM will replicate to 8 (one per rank), meaning you store 8/3 = 2.67× more KV bytes cluster-wide than the architecture nominally requires. You have designed a model that is *more* expensive to serve than GQA-8 while having *worse* quality than GQA-8. Strictly dominated.

Third: powers of two are not superstition here. Head dimensions and head counts feed directly into kernel tile sizes, warp assignments, and NCCL collective shapes. A non-power-of-two head count produces tail effects in every tiled kernel it touches.

**🗣 Say this in the room:** "Pick `n_kv_heads` from {1, 2, 4, 8, 16} such that it's a divisor of `n_query_heads` and a multiple of, or equal to, the tensor-parallel degrees you intend to support. Three fails both conditions and gets you replication at TP=8 — you'd end up using more KV memory than GQA-8, not less."

**⚠ Trap:** treating `n_kv_heads` as a purely architectural choice made by the research team. It is a *joint* architecture-and-deployment decision. If the serving team is not in the room when that number is picked, you will ship a model whose headline efficiency claim doesn't survive contact with an 8-GPU node.
