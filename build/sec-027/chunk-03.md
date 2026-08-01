### How does automatic prefix caching work by block hashing? Tell me exactly what goes into the hash.

The mental model: once KV lives in fixed-size blocks, a block's *contents are a pure function of the tokens that produced it and everything before them*. Attention is causal, so the K and V for tokens `[i·B, (i+1)·B)` depend only on tokens `[0, (i+1)·B)`. That makes the block content-addressable, and content-addressable means cacheable — a hash table from "prefix identity" to physical block id, with no invalidation logic needed because the key *is* the content.

The hash must therefore be a **chain**, not a hash of the block alone. Block `i`'s key is `H(hash_of_block_{i-1}, token_ids_in_block_i, extra_keys)`, with block 0's predecessor being a null sentinel. The chaining is what makes the key encode position and the entire preceding context; hashing only the block's own tokens would collide two different documents that happen to share a 16-token window and hand you another request's KV.

The `extra_keys` term is the part people forget and it is where correctness lives. Anything that changes the KV values for the same token ids must be in the key: the **LoRA adapter id** (a different adapter produces different K/V for identical tokens), **multimodal input hashes** (the image tokens are placeholders; the actual embedding comes from the vision encoder), and any cache-salt or tenant-id you add for isolation. Get this wrong and you have a correctness bug that manifests as one tenant's adapter answering another tenant's request.

On a cache hit, the scheduler maps the existing physical blocks into the new sequence's block table, increments their refcounts, and **skips prefill for those tokens entirely** — the tokens are already "computed." Only the trailing partial block and everything after it are prefilled.

**⚠ Trap:** the last, incomplete block is never cached, because its content is not yet determined — more tokens will be written into it. So a 1,700-token system prompt with `block_size=16` caches `floor(1700/16) = 106` blocks = 1,696 tokens, and 4 tokens are always re-prefilled. That is fine. What is not fine is assuming the cache is token-granular; it is block-granular, and every boundary calculation you do must round down.

### SGLang's RadixAttention versus vLLM's block-hash prefix cache. Is there a real difference or is it branding?

There is a real difference, and it is a data-structure difference with a genuine workload-dependent consequence.

**Block-hash APC (vLLM)** is a flat hash map: `chained_block_hash → physical_block_id`. Lookup is O(1) per block, matching walks forward block by block until the first miss. It is simple, cheap, and its cost is invisible in a profile. Matching is inherently **block-aligned** — you share a prefix rounded down to a multiple of `block_size`.

**RadixAttention (SGLang)** maintains a **radix tree** whose edges are token sequences and whose nodes own KV ranges, with LRU eviction over the tree and reference counting to pin nodes belonging to running requests. Lookup is a longest-prefix-match walk from the root. Because tree nodes can be **split** at an arbitrary token position when a new request diverges partway through an edge, matching is effectively token-granular, and — this is the real advantage — the tree naturally represents a *branching* prefix structure rather than a flat set of chains.

Where that matters: workloads with heavy branching from a common trunk. Few-shot prompts that share 90% of their content and differ in the last example. Agent loops where one conversation forks into several tool-call variants. Tree-of-thought and self-consistency sampling. Multi-turn chat where turn `k` extends turn `k-1` — the tree makes the extension a pointer append rather than a fresh chain. SGLang's own framing is that RadixAttention is what makes *structured LM programs* (its programming model, with forks and joins) cheap.

Where it does not matter: single-turn RAG traffic with a shared system prompt and otherwise unrelated documents. That is a flat, non-branching structure, and a hash map represents it perfectly.

**📄 Paper:** Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs*. Contributed RadixAttention (radix-tree prefix KV reuse with LRU eviction) plus a compressed FSM for fast constrained decoding. It replaced the assumption that prefix reuse is only worth doing for a static system prompt.

**🗣 Say this in the room:** "They solve the same problem with different data structures. A hash chain is O(1) and block-aligned; a radix tree is longest-prefix-match and token-granular, and it represents branching prefixes natively. If my workload is one shared system prompt plus independent user turns, they perform the same. If it's agent forking or self-consistency sampling from a common trunk, the tree wins, because a hash chain has no way to express 'these eight requests share the first 3,000 tokens and then diverge.'"

### What's the eviction policy on a prefix cache, and what makes hit rate suddenly collapse?

Eviction is **LRU over blocks that are not currently referenced by a running sequence.** The refcount is the pin: a block in use by an in-flight request can never be evicted, so the cache only reclaims from the "cached but idle" population. When free blocks run out, the allocator evicts the least-recently-used unreferenced block — and here is the detail that matters, it must evict **tail-first within a chain**, because evicting a middle block orphans everything downstream (whose hashes chain through it).

Hit rate collapse has a small number of causes and they are distinguishable:

**Capacity collapse.** Running-sequence blocks crowded out the cached-idle population. This is load-correlated: hit rate falls when concurrency rises. Diagnosis: plot prefix hit rate against `gpu_cache_usage` — if they are anticorrelated, you need more KV (fewer concurrent seqs, KV quantization, or a bigger card), not a different policy.

**Key change.** Something entered the hash that used to be stable. This is *step-function* collapse — hit rate goes from 85% to near 0 at a deploy boundary, not gradually. Causes: a timestamp or request id injected into the system prompt, tool schemas serialized from a Python dict with nondeterministic ordering, a retrieved document placed *before* the static preamble, a whitespace or trailing-newline diff in a template, a model or adapter version change (which correctly changes `extra_keys`).

**Routing change.** The cache is per-GPU. If your load balancer stopped doing prefix-affinity routing — someone enabled least-connections, or a pod recycled and the hash ring rebalanced — then requests that share a prefix land on different replicas and each one misses. This is *fleet-shaped*: per-replica hit rate stays fine, aggregate falls.

**🔍 Failure taxonomy — prefix hit rate dropped:** (1) Step function at a deploy? → key change; diff the rendered prompt bytes between the old and new build, not the template source. (2) Gradual and anticorrelated with load? → capacity; check `gpu_cache_usage` and preemption count. (3) Aggregate down but per-replica flat? → routing/affinity. (4) Only for some tenants? → `extra_keys` (LoRA id, multimodal hash) changed for those tenants only.

**⚠ Trap:** monitoring "cache hit rate" as a percentage of *requests* rather than *tokens*. A request that hits 1,500 of its 1,600 prompt tokens and one that hits 16 of 1,600 both count as "a hit." The metric that predicts cost is **cached tokens ÷ total prompt tokens**, and it is the one to put on the dashboard and alert on.

### A one-character change to our system prompt dropped prefix cache hit rate to zero. Explain what happened and how you'd prevent a repeat.

It happened because the chained hash makes the cache key a function of **every preceding byte**, so a change at position 12 invalidates blocks 0 through the end of the prompt — not just the block containing the change. Prefix caching has exactly the invalidation semantics of a Merkle chain: touch anything and everything downstream is a different key.

That is the mechanism. The design rule that follows is the actually valuable part: **order your prompt by volatility, most-stable first.** Static system instructions, then tool schemas, then few-shot examples, then retrieved context, then conversation history, then the user turn. Every byte you place early is a byte you pay for once across all requests; every byte you place late costs you nothing extra. The single most expensive mistake is putting a timestamp or a request id near the top "for logging" — that gives you a 0% hit rate forever, and it is nearly invisible in review because the prompt still *works*.

**💰 Math:** an 8,000-token system prompt on a self-hosted 70B. Prefill FLOPs for those tokens ≈ 2 × 70e9 × 8,000 = 1.12e15. On a 4-way H100 node at ~1,200 aggregate effective TFLOP/s that is **0.93 seconds of GPU time per request**. At 200,000 requests/day: 200,000 × 0.93 s = 186,000 GPU-seconds = **51.7 GPU-hours/day** burned re-prefilling a constant. At $2.50/GPU-hour (📅 volatile), that is $129/day = **$3,875/month** destroyed by one character. On a hosted API the same arithmetic runs through cached-input pricing: at a 90% cache discount on input tokens, 8,000 tokens × 200,000 requests = 1.6e9 tokens/day; at $3/Mtok uncached that is $4,800/day, and at $0.30/Mtok cached it is $480/day — the character is worth **$4,320/day, ~$130k/month**.

Prevention is a test, not a convention. I put a **golden-hash test in CI**: render the production prompt template with a fixed fixture, hash the first N tokens, and assert against a checked-in value. Any change to the stable prefix fails the build with a message saying "this invalidates the prefix cache; if intentional, update the golden and expect a cache-warm period." Second, an **alert on cached-token ratio** with a 15-minute window — a drop below 60% pages, because it is a cost regression that no functional test can see.

**🗣 Say this in the room:** "Prefix caching has Merkle-chain invalidation semantics — a byte at position 12 invalidates everything after it. So prompt layout is a caching decision: stable content first, volatile content last, and never a timestamp in the system prompt. I gate it with a golden-hash test in CI and an alert on cached-token ratio, because it's a five-figure-a-month regression that passes every functional test."

### How do block size and prefix cache hit rate interact? And what's the multi-tenant hazard?

**Block size and hit rate** trade in one direction only: coarser blocks reduce shareable tokens, because matching rounds down to a boundary. If your shared prefix is `P` tokens and your block size is `B`, you share `floor(P/B) × B` tokens and re-prefill `P mod B`. Expected waste is `(B-1)/2` tokens per request. At `B=16` that is 7.5 tokens — noise. At `B=128` it is 63.5 tokens per request; at 200,000 requests/day on a 70B that is 63.5 × 200,000 × 2 × 70e9 = **1.78e18 extra prefill FLOPs/day**, which at ~1.2 PFLOP/s per 4-way node is 1,483 node-seconds ≈ **0.41 node-hours/day**, around $4/day. Real, and small. The honest conclusion is that block size affects hit rate in a direction you can name but rarely in a magnitude that matters — the reason to keep `B` small is fragmentation, and the reason to keep it from being 1 is coalescing. Say the direction confidently and the magnitude honestly; inventing a big number here is how you get caught.

**The multi-tenant hazard** is much more serious and it has two halves.

*Correctness/confidentiality:* if tenant A's prompt is a prefix of tenant B's prompt, B gets A's cached KV. That is not a leak of content by itself — the KV is a function of B's own tokens, which are identical over the shared span, so B learns nothing it did not already have. The leak is a **timing side channel**: B can measure TTFT and infer whether a given prefix is already in cache, i.e. whether *someone else recently sent that exact prefix*. On a product where prompts contain user data, that is a real oracle — an attacker can binary-search a secret prefix at roughly one bit per probe.

*The mitigation* is a **cache salt** per isolation domain, folded into `extra_keys`: `H(tenant_id, ...)` for the first block of the chain. That destroys cross-tenant sharing — which is exactly the point — and costs you the shared-system-prompt win across tenants. The middle path I actually ship: salt by tenant for *user-supplied* content and leave the platform-owned system prompt unsalted, so the 8,000-token preamble is shared globally (it is not secret) while nothing derived from user input crosses a tenant boundary.

**⚠ Trap:** assuming shared prefix caching is safe because "the KV is derived from their own tokens." That reasoning is correct about *content* and wrong about *timing*, and timing is the attack that has actually been demonstrated against hosted caches. If you are handling regulated data, salt per tenant and eat the cost.

### 📐 What throughput should I expect from a 70B on H100s, and how do I sanity-check a vendor's number?

Two things: the anchor, and — much more important — the derivation that lets you reject a bogus claim in real time.

**The anchor.** For Llama-3.3-70B in FP8 on H100 SXM5 at 128+ concurrency, aggregate output throughput lands in the low thousands of tokens/second per node — roughly **2,200–2,400 tok/s** is the figure to carry. **📅 Volatile:** this moves with engine version, parallelism degree, context length and quantization; treat it as an order-of-magnitude anchor, not a spec.

**The derivation, which is what you are actually being tested on.** Decode is memory-bandwidth-bound, so per-step time is `bytes_read / HBM_bandwidth`, and bytes read = weights + KV.

- Weights: 70B in FP8 = 70 GB. On TP=4 that is 17.5 GB/GPU.
- KV at 128 concurrent × 4,096 avg context, FP8 KV (0.156 MB/token for this model): 128 × 4,096 × 0.156 MB = 81.8 GB, or 20.4 GB/GPU.
- Total per-GPU read per step ≈ 37.9 GB. At H100's 3.35 TB/s: 37.9/3350 = **11.3 ms at 100% MBU**.
- Real MBU for a well-tuned engine is 55–70%; take 60% → **18.8 ms/step**.
- Output tokens/s = 128 sequences / 0.0188 s = **6,800 tok/s** ceiling for that configuration.

So the anchor sits well *below* the roofline ceiling, and the gap is the interesting part: TP all-reduce latency, sampling and scheduler overhead, prefill stealing decode steps under real arrival patterns, and — dominant — the fact that published numbers often assume much longer contexts, which inflates the KV term. Push the average context to 16k and the KV term becomes 128 × 16,384 × 0.156 MB = 327 GB → 81.8 GB/GPU, step time (17.5+81.8)/3350 = 29.6 ms at 100% MBU, ~49 ms at 60% → **2,600 tok/s**. That reproduces the anchor almost exactly, and now you know what configuration it implies.

**🗣 Say this in the room:** "Before I accept any throughput number I ask four things: tensor-parallel degree, input and output length distribution, quantization of weights *and* KV, and concurrency. Without those the number is unfalsifiable. With them I can reproduce it from the bandwidth roofline in thirty seconds — 70 gigabytes of FP8 weights over 3.35 terabytes a second is a 21-millisecond floor on one card, and everything else is the KV term."

**⚠ Trap:** vendors quoting "tokens/sec" that includes prefill tokens. Prefill processes thousands of tokens per step and decode processes one per sequence, so counting them together inflates the number by 5–20× on a RAG workload. Always ask for **output** tokens/sec, and separately TTFT and ITL.

### The guide claims a modern engine gets 3–5× a naive PyTorch loop on the same H100. Itemize where that factor comes from.

Decompose it, because "the engine is faster" is not an answer and the multiplier depends entirely on what you are comparing against.

Against a **naive `model.generate()` loop with batch size 1**, the gap is far larger than 5× — closer to 10–20× — and almost all of it is batching. You are reading 70 GB of weights per decode step to produce one token; at batch 64 you read the same 70 GB to produce 64 tokens. That is a ~64× improvement in weight-read amortization, throttled by KV bandwidth and the compute roofline down to something like 15×.

Against a **competent static-batching baseline** (which is what the vLLM paper actually compared to — FasterTransformer and Orca), the paper reported **2–4× throughput at the same latency**, and that gap decomposes as:

1. **Fragmentation elimination → larger batch.** 60–80% waste → <4% means roughly 2–3× more concurrent sequences fit. This is the single largest contributor.
2. **Iteration-level scheduling → no dead slots.** Removes the head-of-line idle time; worth 1.3–2× depending on how heavy-tailed your output-length distribution is.
3. **Prefix cache hits → skipped prefill.** Entirely workload-dependent; on a chat product with a fat system prompt this can be 30–50% of total prefill FLOPs.
4. **CUDA graphs → launch overhead removed.** Worth 10–30% on a small model, near-noise on a 70B.
5. **Fused kernels** (FlashAttention/FlashDecoding, fused RMSNorm+residual, fused SwiGLU, paged attention) — 10–25%.
6. **Overlapped host work** (V1's process split) — 5–20%, larger on small models.

Multiplying the middle of those ranges: 2.5 × 1.6 × 1.15 × 1.15 × 1.1 ≈ **5.8×** over static batching, which brackets the 3–5× claim once you account for the baseline already having decent kernels.

**⚠ Trap:** attributing the win primarily to "optimized CUDA kernels." Kernels are items 5 and 6 and together they are maybe 1.3× of it. The engine's advantage is overwhelmingly **memory management and scheduling** — items 1 and 2 — which is precisely why a backend engineer with allocator and scheduler instincts is well-positioned here and should say so.

### vLLM, SGLang, TensorRT-LLM, or a hosted API — give me your decision rule.

I will give the rule, then the caveat that it moves.

**Hosted API first, always, until you have a reason.** You do not run an inference engine for fun; you run one because a hosted provider cannot meet a constraint. The constraints that actually justify self-hosting are: data residency or a contract that forbids third-party processing; a fine-tuned or open-weight model no provider serves; sustained volume where the unit economics flip; or a latency floor that a shared multi-tenant provider cannot guarantee. "It'll be cheaper" is usually false at low volume once you price idle GPU hours — a node at $7,200/month must serve a *lot* of tokens to beat per-token pricing that has been deflating steadily.

**vLLM** is the default self-hosted choice: widest model coverage, fastest support for new architectures, best-documented internals, largest community, and an OpenAI-compatible server so it drops into existing clients. Pick it unless you have a specific reason not to.

**SGLang** when your workload has structure vLLM does not exploit: heavy prefix branching (agent forks, self-consistency, tree search), or heavy constrained decoding where its compressed-FSM approach to grammar masks matters. Its RadixAttention is a genuine advantage on those shapes.

**TensorRT-LLM** when you are on NVIDIA hardware, your model set is stable, and you need the last 20–30% of performance badly enough to pay for an ahead-of-time engine build per model per GPU type per parallelism config. The cost is real: build times in minutes-to-hours, a rebuild on every model or config change, and a much narrower supported-model surface. I would choose it for a single high-volume production model and would not choose it for a platform serving twenty models.

**📅 Volatile:** the capability gaps between these three narrow every quarter and the ranking above has shifted more than once. Say the *decision axes* — model coverage, workload structure, AOT-vs-JIT operational cost — rather than asserting a current winner, and say explicitly that you would re-benchmark on your own traffic before committing.

**🗣 Say this in the room:** "Default hosted, then vLLM, and I'd need a named reason to go anywhere else. The reasons are structure — SGLang if my workload branches heavily or is grammar-constrained — or squeezing the last 25% on a fixed model, which is what TensorRT-LLM buys and what an ahead-of-time engine build costs you operationally."

### 🔍 Throughput collapsed 40% after a deploy. The model didn't change. Debug it in front of me.

I work this as a decision tree, and the first move is to establish *which resource* got scarce, because the four plausible causes leave different fingerprints.

**Step 0 — confirm it is the engine.** Compare engine-reported output tokens/sec against gateway-reported successful requests/sec. If requests are down but tokens/sec per request is flat, the problem is upstream (routing, autoscaling, a rate limiter) and I am debugging the wrong box.

**Step 1 — pull four engine metrics for before and after:** prefix cache hit rate (as *cached tokens ÷ prompt tokens*), `num_preempted_requests`, `gpu_cache_usage` (KV pool occupancy), and the ratio of prefill tokens to decode tokens per step. Those four discriminate every cause below.

**Hypothesis A — prefix cache miss.** Fingerprint: cached-token ratio dropped step-function at the deploy; prefill token count per step rose sharply; KV usage roughly flat; no preemption change. Cause: the stable prefix changed — a template whitespace diff, tool schemas reordered by a dict iteration, a version string added to the system prompt, or an adapter id now entering `extra_keys`. Confirm by diffing the *rendered prompt bytes* for a fixed fixture across the two builds, not the template source. This is the most common cause and the easiest to fix.

**Hypothesis B — preemption thrash.** Fingerprint: `num_preempted_requests` went from ~0 to hundreds/minute; KV usage pinned near 100%; prefill tokens up (because recompute re-prefills). Cause: someone raised `max_num_seqs`, or the average prompt length grew (a retrieval change now returning 12 chunks instead of 6), or KV quantization got disabled. Each admitted sequence now costs more KV than the pool supports, so the engine admits and evicts in a loop and pays prefill twice. Confirm by computing `max_num_seqs × mean_total_tokens × bytes_per_token` and comparing to the pool size.

**Hypothesis C — long prefills stalling decode.** Fingerprint: throughput down, **ITL p99 up sharply while mean ITL is flat**, TTFT down or flat, no preemption change, cache hit rate fine. Cause: chunked prefill got disabled, or `max_num_batched_tokens` was raised, so a single 40k-token prefill now occupies an entire iteration and every decoding sequence stalls for its duration. Confirm by histogramming per-step token counts: you will see a bimodal distribution with a fat tail at the token budget.

**Hypothesis D — fragmentation / capacity loss.** Fingerprint: KV pool is *smaller* than before at the same `gpu_memory_utilization`. Cause: the model's non-KV footprint grew — a new LoRA adapter set loaded, CUDA graph capture for a new batch bucket allocated a pool, `expandable_segments` got turned off, or the engine version changed its workspace sizing. Confirm from the engine's startup log, which reports the number of KV blocks it allocated; compare the two builds' startup logs directly. That one line is usually the fastest possible diagnosis.

**Hypothesis E — the boring one.** `--enforce-eager` got left on from a debugging session, disabling CUDA graphs. Fingerprint: GPU busy ratio dropped, per-step time up by a roughly constant amount independent of batch size, everything else unchanged. Check the launch args diff first, honestly — it costs ten seconds and it is right more often than it should be.

**The ordering rule:** check the **startup log** (D, E) before the **metrics** (A, B, C), because config regressions are cheaper to confirm and more common after a deploy than workload shifts. And in every case, the artifact I want from the incident is a **canary that would have caught it**: a synthetic request with a fixed prompt, run every minute, asserting TTFT and cached-token ratio. That converts this entire tree into a single alert next time.

### Our mean TTFT is unchanged but p99 TTFT tripled after we turned something on. What did we turn on, and is it bad?

Almost certainly you **increased `max_num_batched_tokens`** or **disabled chunked prefill**, and it is bad for interactive traffic and fine for batch traffic — which is why "is it bad" depends on what you sell.

The mechanism: TTFT for a request has two components — queueing time until admission, and the prefill itself. Raising the per-step token budget means a large prefill can be admitted whole rather than sliced, which *lowers* TTFT for that one large request (fewer steps to first token). But it means the step containing it is enormous, so every request that arrives during that step waits an entire long iteration before it can even be considered for admission. Mean is dominated by the many small requests, which are unaffected; p99 is dominated by the unlucky ones that arrived behind a whale.

You can see it in one histogram: per-step wall time. Before, unimodal around 20 ms. After, bimodal with a second mode at 200–400 ms corresponding to full-budget prefill steps. Every request unlucky enough to arrive in one of those windows eats it.

**💰 Math:** suppose 3% of requests carry a 32k-token prompt. At a 32,768-token budget those prefill in one step of roughly 2 × 70e9 × 32,768 / 1.2e15 = **3.8 seconds** of node compute. Any request arriving in that window has ≥3.8 s of added TTFT. At 50 requests/second, a 3.8 s stall delays 190 requests. With whales at 3% of a 50 rps stream, that is 1.5 whales/second, so the stall windows are nearly continuous — which is exactly why p99 tripled. Chunk to 2,048 tokens and the same prefill takes 16 steps of 240 ms each, interleaved with decode, so the worst added TTFT for a bystander drops from 3,800 ms to 240 ms — a **15.8× reduction in the tail** at the cost of that one large request's own TTFT rising from 3.8 s to maybe 4.2 s because it now shares each step.

**🗣 Say this in the room:** "That's the classic TTFT-versus-ITL trade showing up in the tail. A bigger token budget helps the request doing the prefill and hurts everyone who arrives behind it. Mean can't see it because whales are rare; p99 is made of exactly those bystanders. If we sell interactive latency, chunk the prefill and accept a few percent lower prefill throughput."

### What goes on the inference-engine dashboard? Give me the metrics and the alert on each.

Six panels, and I would defend every one of them as load-bearing rather than decorative.

**1. TTFT and ITL distributions, p50/p95/p99, split by prefill-length bucket.** Not averages — the distributions, bucketed, because a p99 regression in the 8k-prompt bucket is invisible in the aggregate. Alert on p95 breaching the product SLO for two consecutive windows.

**2. Output tokens/sec and requests/sec, separately.** Tokens/sec is capacity; requests/sec is demand. Their ratio is the mean output length, and a shift in that ratio is often the first sign a prompt change went out.

**3. KV pool utilization (`gpu_cache_usage`) and `num_preempted_requests`.** These are the KV-pressure pair and they are the autoscaling signal — the LLM-serving analogue of consumer lag. Alert when preemptions per minute exceed a small threshold *and* utilization is above 90%; either alone is normal.

**4. Cached-token ratio (prefix cache hit rate, token-weighted).** Alert on a drop below your baseline minus 20 points over a 15-minute window. This is a **cost** alert, not a latency alert, and it is the highest-ROI alert in the whole list because nothing else can see the regression.

**5. Waiting-queue depth and time-in-queue.** Queue depth tells you whether you are capacity-limited; time-in-queue is the part of TTFT you actually control by scaling. Alert on queue time exceeding a fixed fraction of the TTFT budget.

**6. GPU busy ratio (from the engine's own step accounting, not `nvidia-smi`).** Below ~70% means host-bound; above ~90% means genuinely GPU-bound. This one metric routes every performance investigation to the right layer.

**⚠ Trap:** alerting on GPU memory used. It is always ~90% by construction — the engine pre-allocates the KV pool to `gpu_memory_utilization`. A team that pages on "GPU memory high" will page constantly and then disable the alert, and will have no alert when it matters. Alert on the *occupancy of the pool*, which is a real signal, not on the size of the pool, which is a constant.

**🗣 Say this in the room:** "The one metric most teams are missing is token-weighted prefix cache hit rate. Latency and error rate have owners; a cache-hit regression silently multiplies your bill and every functional test still passes. I alert on it like I'd alert on error rate."
