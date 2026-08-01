### Estimate the maximum concurrency for Llama-3-70B on a single 80 GB H100 at 8k context. Talk me through every term.

The first honest answer is: **at bf16 it does not fit at all.** Weights are 70e9 × 2 = 140 GB against 80 GB of HBM. So the question is really "what do you have to give up to make it fit," and I would say that out loud rather than producing a number.

Path A — **fp8 weights.** 70e9 × 1 = 70 GB. Usable HBM at `gpu_memory_utilization = 0.9` is 72 GB. After weights, 2 GB remain, and CUDA graphs, NCCL buffers, the logits tensor (batch × 128,256 × 4 bytes) and the activation workspace will eat most of that. **Concurrency ≈ 0–1.** Technically loadable, practically useless. This is the configuration people benchmark, see 15 tok/s at batch 1, and mistakenly report as "70B on one H100."

Path B — **int4 weight-only quantization** (AWQ/GPTQ-style, 4 bits plus per-group scales and zeros, so ~4.25 effective bits). 70e9 × 0.53 ≈ 37 GB. Usable 72 GB − 37 = 35 GB. Reserve ~4 GB for activations, logits and workspace → **31 GB of KV pool**.

KV per sequence at 8k: 320 KiB/token × 8192 = 2.5 GiB = 2.68 GB. So **31 / 2.68 ≈ 11 concurrent sequences at 8k**, bf16 KV.

Path C — int4 weights **plus fp8 KV cache**: 160 KiB/token → 1.34 GB per 8k sequence → **23 concurrent sequences**. That is a usable single-card deployment.

Sanity-check the throughput too, because concurrency without throughput is meaningless. At int4, each decode step reads ~37 GB of weights plus 23 × 1.34 GB = 31 GB of KV = 68 GB per step. At 3.35 TB/s that is 20.3 ms per step → ~49 steps/s → 49 × 23 = **1,127 output tokens/s aggregate**, with each user seeing ~49 tok/s. That is a real, defensible number and it came from bytes, not from vibes.

**⚠ Trap:** forgetting that KV bytes-read grows with concurrency, so throughput does not scale linearly with batch. Going from 11 to 23 sequences increased weight-read amortization (good) but added 15 GB of KV traffic per step (bad). There is an optimum, and past it your per-user token rate collapses while aggregate throughput plateaus. Measure it; don't assume "more batch is better."

**🗣 Say this in the room:** "Llama-3-70B doesn't fit on one 80 GB card in bf16 — that's 140 GB of weights. At int4 weights plus fp8 KV I get about 23 concurrent 8k sessions and roughly 1,100 aggregate output tokens/s. If the requirement is bf16 quality, the answer is two cards minimum and I'd argue for four."

### Now do it properly: 8×H100 with TP=8, bf16, 32k context. And tell me what changes at fp8 KV.

Total HBM 8 × 80 = 640 GB. At `gpu_memory_utilization = 0.9`, 576 GB addressable.

- Weights bf16: 140 GB (sharded 17.5 GB/rank).
- Activations, CUDA graph pools, NCCL buffers, logits: budget ~20 GB across the node at these batch sizes.
- **KV pool ≈ 576 − 140 − 20 = 416 GB.**

KV per sequence at 32k context: 327,680 bytes/token × 32,768 = 10,737,418,240 B = **10.74 GB**.

**416 / 10.74 ≈ 38 concurrent sequences at full 32k.** That number surprises people and it should: an entire 8-GPU node, $30/hour of hardware, serving 38 long-context users.

With **fp8 KV**: 5.37 GB/sequence → **77 concurrent**. With fp8 KV *and* fp8 weights (70 GB): KV pool becomes 486 GB → **90 concurrent**.

Two refinements that matter in a real answer. First, 32k is the *maximum*, not the average. If your traffic's mean context is 6k with a P99 of 32k, you should size the pool on the mean and let the scheduler preempt the tail — vLLM's continuous batching allocates blocks on demand, so you get roughly `416 GB / (320 KiB × 6,144) = 208` concurrent at the mean, with preemption handling the tail. Sizing on the max is the single most common way teams under-provision concurrency by 5×.

Second, **prefix caching changes this materially** if your traffic shares prefixes. A 12k shared system prompt costs 3.84 GB of KV; without prefix caching each of 38 sequences stores its own copy = 146 GB, 35% of your pool, spent on 38 identical byte sequences. With prefix caching it is stored once. That is a bigger win than fp8.

**💰 Math:** cost per session-hour at 38 concurrent on a node at ~$2.50/GPU-hr → $20/hr / 38 = **$0.53 per session-hour**. At 77 (fp8 KV) it is $0.26. At 208 (mean-context sizing plus prefix caching) it is $0.096. **The same hardware and the same model span a 5.5× range in unit economics purely from KV management decisions.** That is the argument for putting a senior engineer on serving configuration rather than treating it as a deployment detail.

### How does GQA constrain your tensor-parallel degree? Specifically, what goes wrong at TP=16 with 8 KV heads?

Tensor parallelism shards attention along the head axis: rank r owns `n_heads / tp` query heads and `n_kv_heads / tp` KV heads, computes those heads' attention locally, and the results are combined by an all-reduce after the output projection. Clean and communication-light — *provided the head counts divide*.

At TP=16 with 8 KV heads, `8 / 16 = 0.5`. You cannot give a rank half a KV head without splitting `d_head`, which breaks RoPE (the rotation pairs adjacent dimensions) and breaks every attention kernel's tile assumptions. So frameworks do the only sane thing: **replicate**. vLLM's behavior is to replicate KV heads up to the TP size, so each of the 16 ranks stores a full copy of one KV head. Aggregate KV memory across the node is now 16 heads' worth of storage for 8 heads of information — **2× the KV bytes, cluster-wide, for zero benefit**.

Play it forward. TP=16 for a 70B model is already unusual (you would need two nodes and cross-node NVLink or InfiniBand for every attention all-reduce). Now you have doubled your KV footprint, so the extra HBM you bought with the second node is partly consumed by redundant cache. Your KV pool per unit of hardware went *down* relative to TP=8. Meanwhile every decode step pays a cross-node all-reduce. This configuration is worse on both axes.

The general constraint set I write on the whiteboard:

```
n_q_heads  % tp_size    == 0        # required
n_kv_heads % tp_size    == 0        # else KV replication, factor = tp_size / n_kv_heads
n_q_heads  % n_kv_heads == 0        # required by the architecture
```

For Llama-3-70B (64 Q, 8 KV) the clean TP degrees are 1, 2, 4, 8. TP=8 is the sweet spot: exactly one KV head per rank, one 8-GPU node, all-reduces over NVLink. That is not accidental — 8 KV heads was chosen to match an 8-GPU node.

**⚠ Trap:** proposing MQA for a large model served with tensor parallelism. One KV head at TP=8 means an 8× replication factor: you designed for 32× cache reduction and delivered 4×. If you need cache reduction on a TP=8 deployment, GQA-8 or MLA, not MQA. Conversely, MQA is excellent at TP=1 — which is exactly why it survives in draft models and on-device deployments.

**🗣 Say this in the room:** "`n_kv_heads` has to be a multiple of your TP degree or the framework replicates KV heads and you pay `tp_size / n_kv_heads` times the cache for nothing. Llama-3-70B's 8 KV heads make TP=8 the natural point; TP=16 doubles your KV footprint and adds a cross-node all-reduce per step. I'd solve that with pipeline parallelism across nodes and TP=8 within a node."

### How does the attention variant change prefix-cache block design in an engine like vLLM or SGLang?

Prefix caching works by hashing fixed-size blocks of tokens and keying the physical KV blocks by a rolling hash of `(all tokens up to and including this block)`. A new request walks its prompt block by block, looks up each hash, and reuses any physical block already resident. So the block is simultaneously the unit of allocation, the unit of hashing, and the unit of sharing.

**GQA** is the easy case and the one every engine assumes. The block tensor is shaped roughly `[num_blocks, block_size, n_kv_heads, d_head]` per K and per V, with `block_size = 16` by default in vLLM. One block of Llama-3-70B holds 16 × 320 KiB = 5 MB across all layers. Block size trades internal fragmentation (up to `block_size − 1` wasted token slots per sequence) against block-table overhead and hash granularity. **Smaller blocks give finer prefix sharing; larger blocks give better kernel efficiency and shorter block tables.** Sixteen is a reasonable default; on workloads with many near-identical prompts differing at the very end, a smaller block can raise hit rate meaningfully.

**MQA** changes the shape to `[num_blocks, block_size, 1, d_head]` and makes each block tiny — 16 tokens × 1 head × 128 dims × 2 bytes = 4 KB per layer. Now the *block table* and per-block metadata become a nontrivial fraction of overhead, and you probably want a larger `block_size` than you would under GQA to keep the bookkeeping ratio sane.

**MLA** breaks the layout entirely. There is no separate K and V tensor; there is one latent per token. The block becomes `[num_blocks, block_size, 576]` (512 latent + 64 rotary), a single tensor rather than a pair. Every allocator, every copy path, every prefix-hash routine that assumed "K tensor and V tensor" has to be generalized. This is a real engineering cost and the reason MLA support lagged in serving engines. It also means MLA's paged decode kernel wants a different, generally larger block granularity than GQA's, because the per-token record is small and dense.

**Sliding window** is the genuinely awkward interaction, and it is my favorite question here. Under SWA, blocks older than the window are dead and the allocator wants to free them. But prefix caching wants to *retain* them so the next request with the same prefix can reuse them. These are in direct conflict. Engines resolve it by tracking, per layer group, which blocks are still within the window for the current sequence, and by making cached-prefix reuse valid only when the reusing sequence's window covers the same range. The practical consequence: **prefix cache hit rates are structurally lower on sliding-window models**, because a long shared prefix is partly outside the window for the local layers. If you are choosing between a global-attention model and an SWA model for a workload with heavy prefix sharing (agents, long system prompts, RAG with a stable instruction block), factor that in — the SWA model's memory advantage is partly offset by a worse cache hit rate.

**⚠ Trap:** assuming prefix caching and sliding window compose for free because both save memory. They act on the same axis (time), they interact, and on a prefix-heavy workload the combination can be worse than global attention plus prefix caching.

### Your prefix-cache hit rate dropped from 70% to 12% overnight after a deploy. Walk me through the debug.

I treat this exactly like a sudden cache-miss storm in a backend service: the cache is keyed on something, and something changed the key. So my whole procedure is "enumerate everything that participates in the key, diff it across the deploy."

**Step 1 — confirm it's the key, not the capacity.** Two different failures produce a hit-rate drop. If `gpu_cache_usage` is pegged near 1.0 and preemptions are up, blocks are being *evicted* before reuse — a capacity problem, fix with a bigger pool or shorter contexts. If utilization is normal and hit rate collapsed, it is a *keying* problem. Check `num_preemptions` and cache-usage metrics first; this bisects the space in thirty seconds.

**Step 2 — diff the prompt prefix, byte for byte.** The hash chain is over token IDs from position 0. A single token changed at position 3 invalidates every subsequent block. Usual suspects, in the order I check them:

- A dynamic value moved to the front of the system prompt. `Current date: 2026-08-01` at the top means a **100% miss rate every midnight** and a partial miss on any per-request timestamp. Dynamic content belongs at the **end** of the shared prefix, always. This is the single most common cause.
- A per-user or per-request field (user name, tenant ID, session ID, A/B bucket) got prepended.
- The chat template changed — a new `tokenizer_config.json`, a different BOS handling, a trailing-whitespace difference in the Jinja template. Byte-identical text can tokenize differently after a template edit.
- The tokenizer itself changed (new model revision, added special tokens). Different token IDs, different hashes, total miss.

**Step 3 — check engine-level key participants.** Many engines salt or namespace the prefix-cache key by things you might not expect: LoRA adapter ID, sampling parameters that affect cache validity, `kv_cache_dtype`, and in multi-tenant setups an explicit tenant salt for isolation. A deploy that enabled per-tenant cache isolation will do exactly this. Also: did `block_size` change? Blocks hashed at size 16 are not reusable by a server running size 32.

**Step 4 — check for silent cache invalidation from a config change.** The nastiest one: **a RoPE base or `rope_scaling` change invalidates every cached KV block**, because the cached keys were rotated with different frequencies. If the deploy bumped `rope_theta` from 500,000 to something else, or enabled YaRN scaling, previously cached blocks are numerically wrong, not merely unreused. A well-built engine will version the cache on model config hash and cold-start; a badly-built one will happily serve you corrupted keys. Verify which you have.

**Step 5 — check routing.** Prefix caching is per-replica. If the deploy changed the load balancer from session-affinity to round-robin, or scaled from 4 replicas to 12, each replica now sees 1/12 of the shared-prefix traffic and the hit rate falls roughly proportionally. **This is my second-most-common root cause and it is invisible in the model layer entirely.** The fix is consistent hashing on the prefix hash, not on the user ID.

**🔍 Failure taxonomy, as a decision procedure:** cache utilization pegged → capacity, add HBM or cut max context. Utilization normal + prompt bytes changed → prompt ordering, move dynamic content to the tail. Prompt identical + tokenizer/template version changed → template drift, pin and re-verify. Config hash changed → RoPE/dtype/block-size invalidation, expect a cold start. All of the above unchanged → routing/replica-count change, add prefix-aware affinity.

**🗣 Say this in the room:** "Prefix caching is a content-addressed cache keyed on the token prefix, so I debug it like any cache-key bug: diff the key. First I separate eviction from invalidation using cache utilization and preemption counters, then I byte-diff the rendered prompt, then I check the chat template and tokenizer version, then I check whether routing stopped being prefix-affine. Nine times out of ten it's a timestamp that moved to the top of the system prompt or a replica-count change."

### Explain PagedAttention to me as if I were a backend engineer, and tell me what it does not solve.

It is virtual memory for the KV cache, and the analogy is exact rather than loose. Before PagedAttention, engines allocated a contiguous KV buffer per sequence sized to `max_model_len` — the equivalent of `malloc`ing your maximum possible request size up front. If your max length is 32k and the average request is 600 tokens, you waste 98% of every allocation. Worse, contiguous allocation fragments: you have 30 GB free but no 10 GB contiguous run, so admission fails while the memory sits idle.

PagedAttention splits the cache into fixed-size **blocks** (pages) of `block_size` tokens, maintains a per-sequence **block table** (page table) mapping logical block index → physical block, and modifies the attention kernel to gather from non-contiguous physical blocks via that table. Internal fragmentation drops to at most `block_size − 1` token slots per sequence — under 16 tokens' worth, versus thousands. External fragmentation goes to zero because all blocks are the same size.

The second win, which is the one that actually pays: **sharing**. Two sequences with the same prefix can point their block tables at the same physical blocks. That is prefix caching. For beam search or parallel sampling from one prompt, you get copy-on-write — the shared prompt blocks are referenced by all beams, and only divergent suffixes allocate. Refcounted blocks, exactly like a page cache.

**📄 Paper:** Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention* (the vLLM paper) — replaced contiguous per-sequence KV allocation with paged allocation plus block-level sharing, reporting large throughput gains driven almost entirely by higher achievable batch size.

Now what it does **not** solve, which is the half of the answer that shows judgment:

1. **It does not reduce bytes stored.** A 32k sequence still needs 10.74 GB on Llama-3-70B. Paging eliminates waste; it does not compress. Architecture (GQA/MLA) and quantization do that.
2. **It does not reduce bytes read.** Decode still streams the whole cache per step. Paging slightly *increases* read cost through indirection and non-contiguous gathers — a small, well-amortized cost, but not zero.
3. **It does not fix prefill compute.** A cold 100k-token prompt still costs 2 × N × 100,000 FLOPs. Chunked prefill schedules it better; it does not remove it.
4. **It does not choose your block size for you**, and the choice matters: too small and block-table overhead and kernel gather cost rise; too large and prefix sharing gets coarse and fragmentation returns.
5. **It does not solve cross-replica sharing.** The block table lives in one process on one node. Sharing a prefix across replicas requires a KV transfer layer or a separate prefix-cache tier — a genuinely harder distributed-systems problem.

**⚠ Trap:** citing PagedAttention as the answer to "how do you fit longer contexts." It is the answer to "how do you stop wasting the memory you have." Those are different questions and interviewers ask both.

### How does the KV cache interact with continuous batching and admission control? What signal do you autoscale on?

Mental model: the KV pool is a **fixed-capacity resource pool with variable-duration, growing leases**. Every admitted request holds blocks for its whole lifetime and *acquires more blocks every decode step*. That is unlike a connection pool, where a lease is constant-size. It means admission control cannot be a simple semaphore — a request that was safe to admit at step 0 can starve the pool at step 500.

The scheduler loop, concretely: at each step, the engine has a running batch and a waiting queue. It tries to admit new requests from the queue if the pool has room for their prompt blocks plus a watermark for growth. It runs one forward pass over the running batch. If a running sequence needs a new block and none is free, the engine **preempts** — it evicts a victim sequence, freeing its blocks, and either (a) **recomputes** its KV from scratch when rescheduled, or (b) **swaps** its blocks out to CPU RAM and back later. Recompute costs prefill FLOPs proportional to the victim's length; swap costs PCIe bandwidth (~64 GB/s on Gen5 x16, versus 3.35 TB/s of HBM — a 52× step down). Recompute usually wins for short sequences, swap for very long ones.

Now the operational question. **The signal you autoscale on is KV cache utilization and preemption rate — not QPS, not GPU utilization, not CPU.** The backend analogue is Kafka consumer lag: request rate tells you about arrivals, but the thing that predicts SLO violation is the depth of the resource you are actually running out of. GPU "utilization" as reported by `nvidia-smi` is nearly useless here — a bandwidth-bound decode kernel shows ~100% utilization while doing almost no useful work.

The alert set I would actually configure:

- `gpu_cache_usage_perc` sustained above ~0.85 → scale out. Above 0.95 → you are already preempting.
- `num_preemptions_total` rate > 0 sustained → hard signal. Preemption means someone's TTFT just got a full prefill added to it; a preempted 30k-token request pays seconds.
- `num_requests_waiting` (queue depth) → admission is throttling; combined with high cache usage this is a capacity problem, combined with low cache usage it is a compute problem.
- **Prefix cache hit rate** as a cost metric, alerted on drops, per the previous question.
- TTFT P95 and inter-token latency P95 separately. They fail for different reasons: TTFT for prefill/queueing, ITL for batch depth and bandwidth.

**⚠ Trap:** setting `max_num_seqs` high and assuming the scheduler will sort it out. It will — by preempting. High `max_num_seqs` with a small KV pool produces a thrashing regime where sequences are admitted, partially generated, preempted, recomputed, and preempted again. Aggregate throughput can *drop* while every dashboard shows the GPU busy. This is the LLM-serving version of swap thrashing, and the fix is the same: admit fewer, finish more.

**🗣 Say this in the room:** "I autoscale on KV cache utilization and preemption count, not QPS. QPS doesn't capture context length, and one 100k-token request consumes as much pool as forty 2.5k ones. Cache utilization is the LLM equivalent of consumer lag — it's the depth of the resource that actually runs out."

### You have a chatbot with a 12k-token shared system prompt at 200k calls per day. Do the economics of prefix caching, both hosted and self-hosted.

**Hosted API path.** Take $3.00 per million input tokens with a 90% discount on cached reads and a 1.25× surcharge on cache writes (**📅 Volatile:** these are the shape of Anthropic-style cache pricing; verify current per-model rates and TTL tiers before quoting them in a room).

```
Uncached prefix cost per call : 12,000 / 1e6 × $3.00        = $0.0360
Cached read cost per call     : 12,000 / 1e6 × $0.30        = $0.0036
Saving per call                                              = $0.0324
Daily saving  : 200,000 × $0.0324                            = $6,480
Monthly saving: × 30                                         = $194,400
```

Cache writes: with a 5-minute TTL you refresh at most 288 times/day per variant, at 12,000/1e6 × $3.75 = $0.045 each → **$12.96/day**. Rounding error against $6,480. The economics are so lopsided that the only interesting question is *why you would ever not do it*, and the answer is: because a dynamic token at the front of the prefix silently disables it.

**Self-hosted path.** Prefill FLOPs for the shared prefix on a 70B model: `2 · N · T = 2 × 70e9 × 12,000 = 1.68e15 FLOP = 1.68 PFLOP`. An 8×H100 node at ~400 TFLOP/s effective per GPU (roughly 40% MFU) delivers 3.2 PFLOP/s, so **0.525 s of node time per cold prefill**. With prefix caching that becomes a block-table lookup plus prefill of only the fresh suffix — say 200 tokens: `2 × 70e9 × 200 = 2.8e13 = 28 TFLOP` → **8.75 ms**.

```
Node-seconds saved per call : 0.525 − 0.009                  = 0.516 s
GPU-seconds saved per call  : × 8 GPUs                       = 4.13
Daily GPU-seconds           : × 200,000                      = 826,000
Daily GPU-hours                                              = 229
At $2.50/GPU-hr: daily                                       = $573
Monthly                                                      = $17,200
```

And the **latency** consequence, which the product cares about more than the money: TTFT drops from ~525 ms of pure prefill (plus queueing) to ~10 ms. That is the difference between a chat UI that feels sluggish and one that feels instant.

**The third saving, which most people miss: HBM.** Without prefix sharing, each concurrent sequence stores its own copy of the 12k prefix: 12,000 × 320 KiB = **3.84 GB per sequence**. At 38 concurrent sequences that is 146 GB of identical bytes — 35% of a 416 GB KV pool. With sharing, one copy. **Prefix caching is a capacity multiplier, not just a latency and cost optimization**, and on a system-prompt-heavy workload it is often the single largest concurrency win available.

**⚠ Trap:** measuring prefix-cache value only in dollars-per-token and missing the concurrency effect. I have seen teams enable prefix caching, see a modest TTFT improvement, and not notice they could now raise `max_num_seqs` by 50%.

### Design the serving stack for a code assistant like Cursor: up to 100k tokens of repo context, sub-200 ms TTFT on repeat requests, thousands of concurrent developers.

Let me start with the number that determines the whole architecture. On a 70B GQA-8 model, 100k tokens of context is 327,680 × 100,000 = **32.8 GB of KV per developer**. A thousand concurrent developers is 32.8 TB. That is 51 8×H100 nodes' worth of pure cache. **The naive design is off by two orders of magnitude, and saying so immediately is the strongest opening move.**

Cold prefill is equally damning: `2 × 70e9 × 100,000 = 1.4e16 FLOP = 14 PFLOP`, at 3.2 PFLOP/s per node = **4.4 seconds**. Against a 200 ms TTFT target, cold prefill is 22× over budget. So the design is forced: **essentially every request must hit a warm prefix, and the KV cache must be tiered.**

**1. Model choice is a KV decision.** I would not use a 70B dense GQA-8 model for the inline-completion path. Either an MLA model (68.6 KiB/token → 6.9 GB at 100k, a 4.8× improvement) or a smaller model with sliding-window/interleaved attention for the local layers. For the "explain this codebase" chat path, a bigger model is fine because the concurrency is 100× lower. **Two models, two paths, routed by task** — this is the standard and correct answer.

**2. Context ordering is a caching decision.** Structure the prompt so the prefix is stable in decreasing order of stability: system instructions → language/framework conventions → repo-level context (rarely changes) → open files (changes per session) → current file (changes per minute) → cursor region and recent edits (changes per keystroke). Every dynamic token you put early costs you the entire suffix's cache. This ordering discipline is worth more than any kernel optimization in the stack.

**3. Routing must be prefix-affine.** Consistent-hash requests on `hash(user_id, repo_id, prefix_hash)` so the same developer's session lands on the replica holding their blocks. Round-robin destroys a 90% hit rate down to `1/n_replicas`. Handle rebalancing with a bounded-load consistent hash so a hot repo doesn't pin one replica.

**4. Tier the cache.** HBM holds the active working set. Spill cold-but-recent blocks to host RAM (~64 GB/s over PCIe Gen5) and further to local NVMe (~7 GB/s). Restoring 32.8 GB from host RAM takes 32.8/64 = 512 ms — too slow for the 200 ms target on the full context, but perfectly fine for restoring the *stable repo prefix* while the volatile tail is prefilled fresh. That hybrid — restore the stable half, recompute the volatile half — is the design that actually hits the budget.

**5. Chunked prefill plus a priority class.** Interactive completions must not queue behind a 100k-token chat prefill. Chunked prefill lets you interleave prefill chunks with decode steps so a big prefill doesn't monopolize the batch; combine with an explicit priority so completion requests preempt chat prefills.

**6. Speculative decoding for the completion path.** Code is highly predictable — a small draft model or even an n-gram/retrieval-based drafter over the current file achieves high acceptance rates on boilerplate. Acceptance of 3 tokens per verify step turns a 25 tok/s decode into ~75 effective tok/s at the same KV bandwidth, because verification reads the cache once for k tokens.

**7. Instrument the thing that matters.** Not QPS. Prefix cache hit rate segmented by request type, TTFT P95 split into (queue / cache-restore / fresh-prefill), preemption rate, and **KV bytes per active session** as a capacity-planning primitive.

**💰 Math:** with 90% prefix hit rate, the average request prefills only the volatile tail — say 2,000 tokens: `2 × 70e9 × 2,000 = 2.8e14 = 0.28 PFLOP` → 87 ms on a node, plus ~30 ms of cache restore and queueing → **~120 ms TTFT, inside budget**. The 10% of cold requests pay 4.4 s and must be masked in the UI (optimistic local completion, streaming placeholder). Getting the hit rate from 90% to 95% halves your cold-path pain — which is why prompt ordering discipline is a P0 engineering concern here, not a nicety.

### When would you choose MQA today, in 2026?

Three situations, and I would name them precisely because "MQA is obsolete" is the lazy answer and it is wrong.

**Draft models for speculative decoding.** A drafter's job is to emit k candidate tokens as fast as possible; its quality only needs to be good enough to be accepted, and rejections are corrected by the target model, so **the drafter's quality loss is bounded by the acceptance rate, not by end quality**. That is the ideal place to spend quality for speed. A drafter with MQA has a tiny cache — you can keep it entirely resident alongside the target model's, and its decode step is nearly free. This is the single strongest current use of MQA.

**On-device and edge.** A phone or laptop has 8–16 GB of unified memory shared with the OS and every other app, and single-digit-hundreds of GB/s of bandwidth. There is no tensor parallelism, so MQA's replication problem does not exist. A 3B model with MQA at 32 layers, 1 KV head, `d_head` 128 in fp16 is 2 × 32 × 1 × 128 × 2 = **16 KiB/token**, so a 32k context costs 512 MiB. Under GQA-8 it would be 4 GiB, which on a 8 GB device is simply not available. **On-device long context is an MQA story.**

**Extreme-context research or specialized retrieval systems** where you are deliberately trading quality for the ability to hold 1M+ tokens, and where you can validate that your specific task tolerates it.

Where I would **not** choose it: any general-purpose served model at TP ≥ 2 (replication kills the benefit), and any workload whose value depends on precise long-input extraction. If someone proposes MQA for a document-QA product, that is a rejection in review.

**🗣 Say this in the room:** "MQA isn't dead, it moved. It's the right choice for draft models in speculative decoding and for on-device inference, because in both cases there's no tensor parallelism to force replication and the quality bar is set differently. For a served general-purpose model, GQA-8 or MLA."

### Speculative decoding plus GQA — what happens to the cache, and what's the gotcha?

Mental model: speculative decoding converts memory-bound decode into compute-bound mini-prefill. A draft model proposes k tokens; the target model verifies all k in **one forward pass**, reading its KV cache exactly once for all k queries. Since decode is bandwidth-bound and the cache read is the dominant term, servicing k tokens per cache read is close to a k× throughput multiplier on the attention component — capped by the acceptance rate.

Concretely with Llama-3-70B at 8k context: one ordinary decode step reads 130 GiB of weights plus 2.5 GiB of that sequence's cache. With k = 4 speculation, the verification pass reads the same 132.5 GiB and produces up to 4 accepted tokens. At a realistic 60% acceptance you get 2.4 tokens per pass → **2.4× effective tokens/s**, minus the draft model's own cost (a 1B drafter running 4 sequential steps is 4 × 2 GB of weight reads = 8 GB, ~6% overhead). GQA matters here because the smaller the target's cache, the smaller the fixed per-pass cost that speculation is amortizing — GQA and speculation are complements, not substitutes.

The gotchas, in order of how often they bite:

**Cache rollback.** The target writes K and V for all k speculated positions during verification. If only j < k are accepted, positions `j..k−1` must be logically discarded. With paged allocation this means truncating the sequence's block table and possibly freeing a partially-written block. If your cache manager cannot do a cheap logical truncate — or if it truncates the block table but leaves the block's `num_filled` counter wrong — you get **stale KV entries that are attended to on the next step**. The symptom is subtle: generation that is fluent but occasionally repeats or references content that was never emitted. This is the classic speculative-decoding correctness bug and it is invisible in unit tests that check only token-level output equivalence at k=1.

**Draft cache memory.** The drafter has its own KV cache, and it is per-sequence too. A 1B drafter with 16 layers, 8 KV heads, `d_head` 128 costs 2 × 16 × 8 × 128 × 2 = 65,536 B = 64 KiB/token — 20% of the 70B target's 320 KiB/token. At 38 concurrent 32k sequences that is 38 × 64 KiB × 32,768 = **79.7 GB**, which you must subtract from the target's pool. Speculation is not free in memory, and teams routinely forget to budget for it and then wonder why concurrency dropped.

**Batching interaction.** Speculation helps most at low batch (where you are most bandwidth-bound) and helps least at high batch (where you are already compute-bound and the extra k−1 verification tokens per sequence cost real FLOPs). At very high batch, speculation can be net *negative*. The correct implementation makes speculation depth adaptive on current batch size — deep speculation when the batch is shallow, off when it is deep.

**⚠ Trap:** validating speculative decoding only on greedy decoding. The rejection-sampling correctness argument guarantees the output distribution matches the target model's *for a correctly implemented sampler*; bugs in the rejection step (wrong normalization of the residual distribution) produce a subtly different distribution that greedy decoding will never expose. Test at temperature 1.0 with a distribution-level comparison over many samples, not just exact-match at temperature 0.

### You're told to cut serving cost 40% on a GQA-8 model, and retraining is off the table. Rank your options.

I would rank strictly by **expected saving divided by quality risk**, and I would insist on doing the zero-risk items first — I have seen too many teams reach for int4 while leaving prefix caching off.

**Tier 1 — lossless, do these before anything else.**
1. **Prefix caching**, if not already on and if prompts share structure. On a system-prompt-heavy workload this alone can be 30–50% of prefill cost plus a large concurrency gain (see the 146 GB of duplicated prefix earlier). Zero quality risk.
2. **Prompt reordering** to maximize hit rate — dynamic content to the tail. Costs an afternoon.
3. **Prefix-affine routing.** Turning round-robin into consistent hashing on the prefix hash can take hit rate from 1/n to 0.9. Zero quality risk.
4. **Chunked prefill + tuned `max_num_batched_tokens`.** Raises GPU efficiency by filling batches with prefill work instead of idling during decode. Typically 10–25% throughput.
5. **Right-size `max_model_len`.** If you advertise 128k but your P99 is 16k, the pool is being reserved against a fiction. Cut it and concurrency rises immediately.

**Tier 2 — small, measurable, well-understood risk.**
6. **fp8 KV cache.** Exactly 2× concurrency: 320 → 160 KiB/token. Gate on a long-context retrieval eval at your P95 context, not MMLU. This is usually where the remaining 40% comes from.
7. **fp8 weights.** 140 → 70 GB frees 70 GB of pool, and on Hopper the fp8 tensor cores make prefill faster too. Validate with your own evals.
8. **Speculative decoding**, adaptive on batch depth. 1.5–2.5× on the decode path for low-batch traffic.

**Tier 3 — real trade-offs, require a product conversation.**
9. **Model routing**: send the easy 70% of traffic to an 8B model. This is often the single biggest lever — 8B weights are 16 GB and its cache is 128 KiB/token — but it needs a router and an eval that proves the routed traffic didn't degrade.
10. **int4 weight quantization.** 70 → 37 GB. More quality risk than fp8; validate hard.
11. **Batch/flex tier** for anything not user-facing. If you also use a hosted API for some traffic, batch endpoints are typically ~50% off (**📅 Volatile:** verify).

**Tier 4 — only with a named owner and a rollback plan.**
12. int4 KV, KV eviction (H2O/StreamingLLM-style), aggressive sliding window at serve time. These fail on specific inputs rather than degrading uniformly, which makes them the hardest to detect and the easiest to regret.

**🗣 Say this in the room:** "I'd get the 40% from prefix caching plus prefix-affine routing plus fp8 KV, in that order, and I'd expect to overshoot. All three are measurable, two are lossless, and the one that isn't gets gated on a long-context retrieval eval at our P95 context length. I would not touch KV eviction for a 40% target — that's an emergency lever."
