### Provider prompt caching versus self-hosted prefix caching — what's actually different, and how do you decide?

The mechanism is identical: hash the prefix, store the KV, reuse it. What differs is **which knobs you own**, and that's the whole answer.

| | Provider API | Self-hosted engine |
|---|---|---|
| Enable | Explicit `cache_control` breakpoints (or implicit, provider-dependent) | Automatic, on by default in modern engines |
| Granularity | ≤4 breakpoints, model-dependent minimum prefix | Every block; no minimum, no breakpoint budget |
| Lifetime | Fixed TTL menu (5 min / 1 hour) | LRU over your KV pool — you own capacity and policy |
| Routing | Opaque; the provider decides which replica serves you | Yours — prefix affinity, consistent hashing, per-tenant partitioning |
| Isolation | Provider's guarantee; you cannot inspect it | Yours to design and to get wrong |
| Cost model | Discrete: 1.25×/2× write, 0.1× read | Continuous: reclaimed prefill FLOPs, minus lost batch capacity |
| Observability | `usage.cache_read_input_tokens` and nothing else | Full engine metrics: hit rate, evictions, block reuse, thrash |

The important asymmetries. **Self-hosted has no breakpoint budget and no minimum prefix**, so caching is automatic and total; you never lose 8% of your prefix to a rounding rule or waste a scarce breakpoint. But **self-hosted makes cache lifetime your problem**: on a provider, an entry lives its TTL regardless of what else is happening; on your fleet, an entry survives until some other request needs its blocks, so your effective TTL is a function of load. Under a traffic spike your hit rate degrades exactly when you can least afford it — a failure mode the provider absorbs for you.

The other asymmetry runs the opposite way. **Provider routing is opaque**, so you cannot guarantee a warm-cache replica for a session; you rely on the provider doing prefix-aware routing internally and you have no recourse when it doesn't. Self-hosted, that's a solvable engineering problem and the reason prefix-affinity routing exists.

**The decision rule I'd give:** this is downstream of the self-host-versus-API decision, not an input to it. If you're on an API, caching is a prompt-*architecture* discipline — ordering, determinism, breakpoint placement, TTL selection. If you're self-hosting, caching is a *capacity and routing* discipline — KV budget, eviction, affinity, tenant partitioning. The failure modes barely overlap, which is why teams that migrate from API to self-host and assume their caching work transfers get a nasty surprise: their prompt ordering is fine, and their hit rate still collapses because every request lands on a random replica.

**🗣 Say this in the room:** "Same mechanism, different knobs. On an API it's a prompt-architecture problem — ordering by stability, four breakpoints, choosing a TTL. Self-hosted it becomes a capacity and routing problem — how much HBM the reuse pool gets, what the eviction policy is, and whether the router can land a session on the replica holding its KV. Prompt ordering discipline transfers; everything else has to be rebuilt."

### We're self-hosting on vLLM. Explain how its automatic prefix caching actually works, down to the data structure.

vLLM's automatic prefix caching is content-addressed storage bolted onto the block allocator, and if you understand a content-addressed object store you already understand it.

The KV cache is already carved into fixed-size blocks (16 tokens by default). APC adds one field per block: a hash. The hash of block `i` is computed over **the token IDs in block `i` plus the hash of block `i−1`** — a hash chain, exactly like a Merkle chain or a git commit graph. That chaining is what encodes "prefix," because a block's identity depends on its entire history, so two sequences that agree on the first `k` blocks produce identical hashes for all `k` and diverge immediately at the first differing block.

The engine keeps a `hash → physical_block_id` map plus a refcount per block. On admission, it hashes the incoming prompt block by block and walks forward until the first miss. Every hit is a **pointer assignment in the sequence's block table with a refcount increment** — no copy, no memory allocated, no compute. Prefill then starts at the first missed block, so a request whose 12,000-token prefix is fully cached prefills only its 40 new tokens.

Three consequences worth stating because they're what interviewers probe for. **Sharing is physical, not logical** — two sequences with a shared prefix literally point at the same HBM, so the memory cost of the 100th user of a shared system prompt is zero, not 1/100th. **Granularity is the block size** — a shared prefix is rounded *down* to a block boundary, so a 1,700-token system prompt with block 16 shares 1,696 tokens and re-prefills 4 forever. And **blocks with refcount 0 are not freed immediately**; they go on an eviction list and remain reusable until the space is actually needed, which is what makes the cache survive between requests.

**⚠ Trap:** candidates say "so raising the block size improves the cache hit rate." It is exactly backwards. Hit rate is measured in *tokens saved*, and a coarser block rounds every shared prefix **down** to a boundary. Block 128 with a 1,700-token prompt shares 1,664 and wastes 36; block 512 shares 1,536 and wastes 164. Larger blocks strictly reduce shareable tokens. Block size is a kernel/metadata tradeoff, not a caching lever.

**📄 Paper:** Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP). The block allocator and copy-on-write sharing it introduced are the substrate APC is built on; APC generalizes the parallel-sampling sharing case to arbitrary cross-request prefixes.

### Now contrast that with SGLang's RadixAttention. What's the actual difference and when does it matter?

RadixAttention replaces the flat hash map with a **radix tree (compressed trie) over token sequences**, where each edge holds a token span and each node owns the KV blocks for that span. Insert a new prompt and you walk the tree matching tokens; the point where you fall off the tree is your prefix-match boundary, and you attach a new branch there. Eviction is **LRU on leaves**, propagating upward: you evict the least-recently-used leaf, and if its parent becomes a childless leaf it becomes eligible next.

The difference from hash-based block caching is not "one is better," it is **what unit of sharing the structure can express**, and it shows up in three concrete places.

**Branching workloads.** Anything that forks from a shared trunk — tree-of-thought, self-consistency with n samples, a beam, an agent exploring several tool paths, few-shot prompts sharing a preamble and differing in the last example — maps directly onto tree structure. The radix tree represents "one trunk, many branches" natively, and the eviction policy understands that the trunk should outlive the branches because it has more descendants. A flat hash map with block-level refcounts gets the *sharing* right but its eviction is blind to topology: it can evict a heavily-shared trunk block because that block's own recency looks stale, even though a hundred branches depend on the subtree.

**Sub-block granularity.** The radix tree can match at token granularity along an edge, not just at block boundaries, so it doesn't round shared prefixes down.

**Multi-turn and multi-call programs.** SGLang was designed around *programs* — a chain of LLM calls with shared structure — and RadixAttention is the memory manager for that. If your workload is a scripted multi-call pipeline (extract, then classify, then rewrite, all sharing a document), the tree captures it.

**📄 Paper:** Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs* — introduced RadixAttention, replacing per-request or flat-hash prefix caching with a radix-tree KV cache plus LRU eviction across requests.

**🗣 Say this in the room:** "Both give you cross-request prefix reuse. vLLM hashes fixed-size blocks with a chained hash and a flat map; SGLang keeps a radix tree over token sequences with LRU eviction on leaves. The tree wins where the workload is genuinely tree-shaped — branching agents, n-sampling, structured multi-call programs — because eviction can respect that a trunk has many descendants. For a flat chat workload with one long shared system prompt, the two are close enough that I'd choose on operational grounds instead."

### So which one do you pick, and on what evidence?

I pick on **workload topology first, operational surface second**, and I would push back hard on anyone choosing on a benchmark blog post, because prefix-cache benchmarks are almost entirely a function of the synthetic prompt distribution used to generate them.

The decision rule I'd write down:

- **Flat, wide sharing — one big system prompt, many independent users, mostly single-turn.** Either engine gets you ~all of the available win, because the shareable set is one trunk with no branching. Choose vLLM for breadth of model and quantization coverage and the larger operational community.
- **Deeply branched or program-structured — agent trees, n-sampling, multi-call pipelines, heavy multi-turn with forks.** SGLang, because RadixAttention's tree structure and topology-aware eviction is a genuine structural advantage, not a constant factor.
- **Constrained/structured output at high volume.** SGLang has historically invested heavily here (its grammar-constrained decoding path is a first-class concern rather than a bolt-on), which matters if every request is JSON-schema-constrained.
- **Peak single-model throughput on NVIDIA with an ops team that can own a compile step.** TensorRT-LLM, at the cost of a build pipeline and much narrower model coverage.

**📅 Volatile:** this is one of the fastest-moving comparisons in the field. Feature parity between vLLM and SGLang has repeatedly converged and re-diverged, and any specific claim about which supports what is dated within a quarter. Say so in the interview — "I'd re-verify the feature matrix before committing, and here's the benchmark I'd run" is a *stronger* answer than a confident stale claim.

The benchmark I'd actually run, and this is the part that gets you hired: **replay your own production prompt distribution**, not ShareGPT. Take a week of real requests, preserve their arrival order and their prefix-sharing structure, replay at your target concurrency, and measure cache hit rate in tokens, TTFT p95, and tokens/sec goodput under your SLO. Prefix-cache performance is a property of *your traffic's sharing structure*; a benchmark on someone else's traffic tells you almost nothing.

### How much GPU memory do you give the prefix cache, and how do you tune eviction?

This is the real capacity question, and it is a direct tradeoff you should be able to state as a single sentence: **every block held for reuse is a block not available for an in-flight sequence's KV, so prefix-cache capacity trades throughput (batch size) against TTFT (hit rate).**

The budget arithmetic. On an 80 GB card running a 70B in FP8, weights take ~70 GB — that's the wrong config, so say FP8 weights ~70 GB is too tight; take a 70B in FP8 across two cards, or a 34B-class model on one. Let me do it concretely for a 70B on 2×H100 with TP=2: weights ~70 GB total (35 GB/card), CUDA context + activations ~8 GB/card, leaving roughly **37 GB/card × 2 = 74 GB of KV pool**. At a Llama-3-70B-class KV footprint of ~0.31 MB/token (FP16, GQA), that is 74,000 / 0.31 ≈ **238,000 tokens of KV capacity total**.

Now the allocation decision. If your average request is 8k prompt + 1k generation, running 24 concurrent sequences consumes 24 × 9,000 × 0.31 MB ≈ 67 GB, leaving ~7 GB ≈ 22,000 tokens for cached-but-idle prefixes. If your shared system prompt is 8,000 tokens, that is enough for **two or three distinct prefixes** — which is fine if you have one global prompt, and useless if you have 500 tenants with distinct preambles.

That gives the tuning rule: **size the reuse pool to hold your working set of distinct prefixes, then give everything else to concurrency.** Working set = (number of distinct prefixes active within one eviction window) × (prefix length). Compute it; don't guess.

Eviction policy in practice:
- **LRU is the right default** and is what both engines effectively do. Prefix reuse is strongly recency-correlated: a tenant active now will be active in 30 seconds.
- **Protect high-fanout prefixes.** In a radix tree this falls out of leaf-LRU. With flat hashing you may need to pin the global prefix explicitly, because a block shared by 400 sequences and a block used once look identical to a naive recency counter.
- **Watch for cache thrash**, the signature failure: hit rate collapses while eviction rate spikes, because the working set exceeds capacity and every entry is evicted just before its next use. It is exactly LRU thrash in a page cache. The fix is not a smarter policy — it is to shrink the working set (fewer distinct prefixes: hoist tenant-specific text out of the prefix) or raise capacity (KV quantization to FP8 roughly doubles the pool).

**⚠ Trap:** turning prefix caching on and then reporting a *throughput regression*. It happens, and the cause is that reserving reuse capacity cut your max batch size on a workload with no actual prefix sharing. Prefix caching is not free; it costs you concurrency. Measure hit rate before you keep it on.

### Design prefix-affinity routing for a fleet of 32 GPUs. Why can't I just round-robin?

You can't round-robin because a prefix cache is **node-local state**, and round-robin is a state-hostile scheduling policy. If session S's 40,000-token KV lives on GPU 7 and turn two lands on GPU 19, GPU 19 recomputes all 40,000 tokens from scratch. With 32 replicas and round-robin, the probability of landing back on the warm node is 1/32, so your effective hit rate is ~3% no matter how good the engine's caching is. This is the same problem as sticky sessions in front of an in-process cache, and it has the same family of solutions.

The design: **consistent hashing on a prefix key, not on the request.**

```python
# Route on a stable prefix identity, not on request contents.
def route_key(req) -> str:
    # Session-scoped agents: the session owns the KV.
    if req.session_id:
        return f"sess:{req.session_id}"
    # Stateless calls: hash the *stable* prefix layers only.
    return "pfx:" + hashlib.sha256(
        (req.model + "\x00" + req.tools_canonical + "\x00" + req.system).encode()
    ).hexdigest()[:16]

replica = hash_ring.get(route_key(req))   # consistent hash ring, ~150 vnodes/replica
```

Four design points that separate a real answer from a hand-wave:

1. **Consistent hashing, not modulo.** With `hash % N`, scaling from 32 to 33 replicas remaps ~97% of keys and flushes essentially every cache in the fleet at once. With a ring plus virtual nodes, adding a replica remaps ~1/33 of the key space. Virtual nodes (100–200 per physical replica) are what keeps the load distribution even.
2. **Hash the *stable* layers only.** If you hash the full prompt including the user turn, every request is a distinct key and you're back to random routing. The key must be the thing you want colocated.
3. **Two-tier routing.** Route stateless requests by prefix hash; route session-bound requests by session ID with an explicit affinity record (a Redis entry mapping `session_id → replica`, TTL'd to the KV lifetime). Sessions are the higher-value affinity because their prefix is large and unique.
4. **Bounded-load, not strict affinity** — see the next question.

**💰 Math:** 32 replicas, 8,000-token shared prefix, 300 req/s. Round-robin: ~97% miss, so ~291 req/s each pay 8,000 tokens of prefill = 2.33M prefill tokens/s. Prefix affinity at 90% hit: 30 req/s × 8,000 = 240k prefill tokens/s. You reclaimed ~2.09M tokens/s of prefill compute — roughly the entire prefill capacity of the cluster — which converts directly into either 8–10× the throughput or the ability to run the same load on a third of the fleet. On 32 H100s at ~$2.50/GPU-hr that is on the order of **$38,000/month** of hardware.

### Prefix affinity and load balancing want opposite things. How do you resolve that?

They do, and the honest framing is: **prefix affinity is a cache-locality objective, load balancing is a queueing objective, and blindly maximizing either one destroys the other.** Perfect affinity gives you a hot shard — every request sharing the popular system prompt piles onto one replica, whose queue depth explodes while 31 replicas idle. Perfect balance gives you a 3% hit rate. The answer is a bounded compromise, and there is a well-known shape for it.

**Consistent hashing with bounded loads.** Hash to the primary replica; if that replica's current load exceeds `c ×` the fleet average (with `c` typically 1.25–1.5), walk the ring to the next replica, and repeat. Under-loaded, affinity is perfect; overloaded, you spill deterministically to a *specific* neighbour, which means the spill target is consistent and builds its own warm copy of the hot prefix rather than scattering.

The critical detail is **what "load" means**. Not GPU utilization — that pins at 100% while latency collapses and tells you nothing. Not requests in flight — a 200-token request and a 200k-token request are not the same load. The right signals, in order of preference:

1. **KV cache utilization** (fraction of the block pool allocated). This is the actual admission constraint on a decode-bound engine.
2. **Waiting-queue depth in tokens**, not requests.
3. **Running-batch token count** against `max_num_batched_tokens`.

The scoring function I'd actually ship blends both objectives explicitly rather than treating spill as an exception:

```
score(replica) = w_hit * estimated_prefix_hit_tokens(replica, req)
               - w_load * kv_utilization(replica)
               - w_queue * queued_tokens(replica)
```

and route to `argmax`. With `w_hit` derived from the real saving — a hit token is worth roughly one prefill token of compute — this is a genuine cost model rather than a heuristic, and it degrades gracefully: under light load the hit term dominates and you get affinity; under heavy load the queue term dominates and you get balance. Several production routers (including vLLM's own router work and the llm-d / Dynamo-class gateways) converge on this shape.

**⚠ Trap:** hot-prefix replication is the piece people forget. If one prefix is used by 60% of traffic, no routing policy can fix a single warm copy. **Deliberately replicate the hot prefix onto every replica** by pre-warming it at pod start, then route only the *unique* portion of the key. Global prefixes should be warm everywhere; only tenant- and session-scoped prefixes need affinity.

**🗣 Say this in the room:** "I treat it as consistent hashing with bounded loads: hash to the primary, spill to the next replica on the ring if it's over ~1.3× fleet average load, where load is KV-cache utilization and queued *tokens*, never GPU utilization. And I pre-warm the globally-shared prefix on every replica so affinity only has to solve the tenant- and session-scoped case."

### What are the multi-tenant hazards of a shared prefix cache? Be specific about the leak.

There are two distinct hazards and conflating them is a bad sign in a security-adjacent interview.

**Hazard one — content leakage, which is real but structurally bounded.** A shared prefix cache stores KV tensors keyed by *token content*. Tenant B's request can only hit tenant A's entry if B's prompt is byte-identical to A's for that span. B therefore learns nothing it did not already possess — it already had the tokens; that's why it matched. There is no mechanism by which B *reads out* A's tokens: the cache returns KV for a prefix B supplied. So the direct-exfiltration story that people intuitively fear does not hold for a correctly-implemented content-addressed prefix cache.

The exception, and the one to name: **hash collisions**. If the engine keys on a truncated or weak hash and does not verify the tokens on match, a crafted colliding prefix would return another tenant's KV, which *would* be a genuine cross-tenant read. This is why serious implementations use a cryptographic-strength hash over the token chain and/or verify token equality on hit. Ask about it in a design review.

**Hazard two — timing side channel, which is real, unbounded, and the actual answer.** Cache hits are *dramatically* faster than misses; the whole point is that TTFT drops by hundreds of milliseconds. That timing difference is an oracle for **membership**: an attacker who can measure TTFT can binary-search whether a given prefix has recently been processed by the system. That leaks *existence*, not content — but existence is often the secret. "Has anyone recently asked this system about `patient_id=8837`?" "Is `<competitor>-acquisition-memo` in someone's prompt?" "Does this API key string appear in another tenant's system prompt?" With a token-granularity match you can extend it into a prefix-extension attack: guess the next token, measure, keep what's faster — a slow but real content-extraction channel against *guessable* secrets.

**Mitigations, in the order I'd apply them:**

1. **Partition the cache by tenant.** Prepend a tenant salt to the hash chain so entries are namespaced. This is the complete fix and costs you cross-tenant sharing of the global prefix — which you recover by pre-warming the global prefix per partition anyway.
2. **Share only what is provably public.** Tenant-agnostic system prompts and tool schemas: shared. Anything containing retrieved documents, user data, or tenant configuration: partitioned. Make this a typed distinction in the prompt builder, not a convention.
3. If you must share cross-tenant, **normalize the timing** on the shared span, or accept and document the membership oracle. Constant-time is expensive here; partitioning is usually cheaper than defending the channel.

**⚠ Trap:** "we're multi-tenant but everyone's on the same system prompt, so sharing is safe" — until someone adds a RAG layer *above* the breakpoint, and now retrieved customer documents are in the shared, timing-observable prefix. The invariant to enforce in review is **no user-derived or tenant-derived content above a shared breakpoint, ever.** That is a static property you can lint for.

### Make cache hit rate a production metric. Define it precisely and tell me what you'd alert on.

The definition matters more than people expect, because the obvious one is wrong.

**Wrong:** requests that got any cache hit / total requests. This is nearly always ~100% in a system with a shared system prompt and tells you nothing.

**Right:** `cache_read_input_tokens / (cache_read_input_tokens + cache_creation_input_tokens + input_tokens)` — **token-weighted hit rate**, computed over a window, per route and per model. This is the fraction of prompt tokens you did not pay full price for, and it maps linearly onto both cost and prefill latency. Every claim you make about caching should be denominated in this number.

Emit alongside it, per route:
- `p95 TTFT` split by hit and miss (the two-line chart from earlier — if the lines converge, caching is not doing anything).
- **Effective input price per 1M tokens**: `(1.25·creation + 0.1·read + 1.0·uncached) / total_prompt_tokens × base_rate`. This is the single number to put on a cost dashboard; it moves the instant caching degrades, and unlike raw spend it is normalized against traffic volume.
- **Distinct-prefix cardinality** per window. A rising cardinality is the leading indicator of a caching regression — someone just started interpolating something per-request.

The alert I actually ship: **"effective input price per 1M tokens rose more than 20% week-over-week on route X."** Not raw spend (confounded by traffic), not hit rate alone (a rate can hold while the prefix shrinks). Effective price is invariant to volume and catches every failure mode: broken breakpoints, a new invalidator, a model version change, a routing regression, a TTL mismatch.

This is the reframe I'd lead with in a systems interview: **cache hit rate is a cost-regression detector, and it is more sensitive than your billing dashboard.** Billing is daily, lagging, and confounded by traffic. Hit rate is per-request, immediate, and unconfounded. A PR that adds `f"Current time: {now}"` to a system prompt shows up in hit rate within one deploy and in the invoice three days later.

**💰 Math:** at 200k calls/day with a 12k prefix at $3/Mtok, a drop from 95% to 5% hit rate costs (0.9 × 12,000 × 200,000 × $3/1e6) × (1.0 − 0.1) ≈ **$5,830/day**. If your alerting is a monthly invoice review, you find it after **~$175,000**. If it's a hit-rate alert, you find it in fifteen minutes. That arithmetic is the entire justification for the metric, and it's how you should pitch it.

**🗣 Say this in the room:** "I define hit rate token-weighted, not request-weighted, and I alert on effective input price per million tokens rather than on raw spend — spend is confounded by traffic and lags by days. A single PR that puts a timestamp in a system prompt is a five-figure-per-day regression that hit rate catches in one deploy cycle and the invoice catches three days later."

### What is cache-augmented generation, and when would you choose it over RAG?

Cache-augmented generation is the deliberate decision to **put your entire corpus in the prompt and keep its KV cache warm**, rather than retrieving chunks per query. You prefill the whole corpus once, cache it, and every subsequent query is a short user turn appended to a warm prefix. No embedding model, no vector index, no chunker, no reranker, no retrieval hop.

It only works under a specific precondition, and naming that precondition correctly is the whole answer: **the corpus must fit in the context window with room to spare, and it must be stable relative to your TTL and reindex cadence.** Concretely — a 60,000-token internal policy handbook, a 90,000-token API reference, a company's 40,000-token style and brand guide, a single large codebase module. Not "all of Confluence."

The tradeoffs, honestly:

**In favour.** You delete an entire subsystem and its failure modes: no chunk-boundary errors, no embedding drift, no reranker, no "the answer was in chunk 7 but we retrieved 1–5," no stale-index-after-reindex incidents. Latency drops by the retrieval hop (typically 80–200 ms for embed + ANN + rerank). And critically, the model sees the *whole* corpus, so it can answer questions that require synthesizing across sections that no chunk-level retriever would co-retrieve.

**Against.** Cost scales with corpus size × query count, even at 0.1×. And quality on long context is not free — retrieval-in-the-middle degradation is real, and attention over 100k tokens of mostly-irrelevant material is measurably worse than attention over 4k tokens of well-chosen material for narrowly-scoped factual lookups.

**💰 Math:** a 60,000-token corpus at $3/Mtok. Cached read per query: 60,000 × $0.30/Mtok = **$0.018**. RAG alternative: 4,000 retrieved tokens at full price = $0.012, plus embedding (~$0.00002) plus a reranker call. So CAG is roughly 1.5× the per-query token cost — and it removes an entire pipeline you would otherwise staff, evaluate, and page on. At 20,000 queries/day that is $360/day versus $240/day: **$3,600/month to delete your retrieval stack.** For many teams that is obviously worth it; for a 5M-query/day product it obviously is not. Do the arithmetic; don't assert.

**The decision rule:** corpus under ~100k tokens and changes less often than daily → CAG. Corpus over ~500k tokens, or changes continuously, or queries are narrow lookups where precision beats synthesis → RAG. In between → hybrid: cache a stable "core" corpus as the prefix and retrieve only from the volatile tail, appending retrieved chunks *after* the cached breakpoint.

**⚠ Trap:** teams try CAG, put the retrieved documents *before* the system prompt because that's where the corpus "logically" goes, and get 0% hit rate. Corpus goes after tools and system, and the breakpoint goes at the end of the corpus. Order is stability, not logic.

### Quantify the self-hosted prefix-caching win. What are you actually saving in FLOPs and dollars?

On a self-hosted stack there is no provider discount to quote — you save *prefill compute*, and you should convert that to dollars through your GPU rate.

**The FLOP model.** Prefill of `T` tokens through a dense model with `P` parameters costs approximately `2 · P · T` FLOPs (two per parameter for the multiply-accumulate in the forward pass). Attention adds a term quadratic in `T` that matters at long context but let's take the linear term first.

For a 70B model and an 8,000-token cached prefix: 2 × 70×10⁹ × 8,000 = **1.12×10¹⁵ FLOPs saved per cache hit.** On an H100 delivering ~400 effective TFLOP/s during prefill (below the 989 TFLOP/s dense FP16 peak, which nobody hits), that is 1.12e15 / 4e14 = **2.8 GPU-seconds per hit**.

**💰 Math:** 300 req/s with a 90% hit rate is 270 hits/s × 2.8 GPU-s = **756 GPU-seconds of compute reclaimed per wall-clock second** — i.e. you would need ~756 additional H100s to serve the same traffic without prefix caching. At $2.50/GPU-hr (**📅 Volatile**), that is 756 × $2.50 = **$1,890/hour = $45,360/day = ~$1.36M/month**. Even at a tenth of that traffic it is the largest single line item in your inference budget.

Now the honest caveats, which is where seniority shows:

- **You only bank the saving if you were prefill-bound.** If your fleet is sized by decode (memory-bandwidth) and prefill was fitting in the gaps, removing prefill compute doesn't reduce GPU count — it improves TTFT and lets you run more concurrent decode. Measure which regime you're in before promising the number: compare prefill token/s against decode token/s in your actual traffic mix.
- **The reuse pool costs you KV capacity**, which costs you batch size, which costs throughput. Net it out.
- **The quadratic attention term** means the saving on very long prefixes is *larger* than `2PT` suggests. For a 128k prefix, attention FLOPs (`~4 · n_layers · d_model · T²` for the QK and AV products) become comparable to or larger than the linear term, so long-context prefix caching is superlinearly valuable. That is the technical reason CAG works economically at all.

**🗣 Say this in the room:** "Self-hosted, the saving is prefill FLOPs: about 2·P·T per hit, so 1.12e15 FLOPs for an 8k prefix on a 70B — roughly 2.8 H100-seconds. At 270 hits/second that's 756 GPU-seconds per second, which is 756 GPUs I don't have to buy. But I'd verify we're prefill-bound before quoting it; if the fleet is decode-bound the win shows up as TTFT and concurrency, not GPU count."

### How do you exploit caching in a batch or offline pipeline processing a million documents?

Offline is where caching is easiest to get right and most often left on the table, because the scheduler is *yours* — you control arrival order, which is the one thing you can never control online.

**Lever one: sort by shared prefix.** If you're classifying 1M documents against 40 different rubrics, don't process them in database order. Group by rubric, process each group contiguously, and the rubric's KV is written once and read 25,000 times. Random order gives you 40 competing prefixes thrashing a cache sized for a handful; sorted order gives you a ~100% hit rate on the rubric span. This is a one-line `ORDER BY` in your job planner and it is worth more than any prompt-engineering change you'll make that quarter.

**Lever two: length bucketing, for a different reason.** Sorting by document length within a prefix group reduces padding waste and makes batch composition predictable. Combine them: sort by `(prefix_key, length_bucket)`.

**Lever three: use the batch tier, and know the interaction.** Provider batch endpoints run around 50% of synchronous pricing with a 24-hour SLA (**📅 Volatile**), and they support caching — but the batch runner's internal scheduling is not under your control, so cross-request cache hits within a batch are best-effort. The reliable pattern is to put the shared prefix in every request with a breakpoint and accept that hit rate will be good but not guaranteed. The 50% batch discount and the 90% cache discount **compose multiplicatively on the cached span**, which is the arithmetic worth doing out loud.

**💰 Math:** 1,000,000 documents, 3,000-token shared rubric prefix, 800-token document, 200-token output. Synchronous, uncached: input 1M × 3,800 × $3/1e6 = $11,400, output 1M × 200 × $15/1e6 = $3,000 → **$14,400**. Batch tier at 50%: **$7,200**. Batch + caching on the 3,000-token prefix at 90% hit: prefix cost becomes 1M × 3,000 × ($3 × 0.5 × 0.1)/1e6 ≈ $450 (plus a negligible write), document input 1M × 800 × $1.50/1e6 = $1,200, output $1,500 → **$3,150**. From $14,400 to $3,150 — **78% off** — by choosing an endpoint and an `ORDER BY`.

**⚠ Trap:** batch jobs are exactly where the 5-minute TTL bites, because a job that stalls on a downstream write or a rate limit lets the prefix expire and then re-writes it at 1.25×. For long-running offline jobs, either keep the pipeline saturated enough that gaps stay under the TTL, or use the 1-hour TTL — with a job doing thousands of reads per prefix, the 2× write premium is irrelevant and the longer lease is pure upside. This is the clearest case where 1h beats 5m.

### What does resumability and idempotency look like in that batch pipeline, and how does caching interact with it?

Your backend instincts transfer almost completely here, with one twist that is specific to LLM output.

**Idempotent output keys.** Key every result by a deterministic hash of `(model_id, prompt_hash, decoding_params)` — not by row index, not by position in the input file. Batch results come back in arbitrary order keyed by your `custom_id`, so position-based reconciliation is a bug waiting for a retry to expose it. Write results to `outputs/{key}.json` with a conditional put; a re-run then naturally skips completed work and the job is resumable at any granularity.

**The twist:** LLM output is not deterministic even at temperature 0 (batching nondeterminism, kernel nondeterminism, and provider-side routing all break bitwise reproducibility). So idempotency here means **"exactly-once *processing*," not "identical output on retry."** If a downstream consumer depends on the exact text, you must treat the first successful write as authoritative and never overwrite — write-once semantics, not last-write-wins. I've seen a pipeline that re-ran a partial batch, produced slightly different classifications for the overlap, and silently corrupted a labeled dataset. Make the store append-only or conditional-put.

**Caching interaction, and this is the non-obvious part:** resumption order matters. A naive resume processes "all incomplete rows" in whatever order the query returns them, which destroys the prefix grouping you carefully arranged on the first pass. **The resume query must preserve the original sort** — `WHERE status = 'pending' ORDER BY prefix_key, length_bucket` — or your retry run silently costs 5× the original per document. Bake the sort into the job definition, not into the initial enqueue.

**Poison-record isolation.** The Celery dead-letter-queue instinct is exactly right: a document that fails repeatedly (a 300k-token PDF, a prompt-injection payload that trips a refusal, a malformed encoding) must be quarantined after N attempts rather than retried forever. LLM-specific addition: **cap cost per record, not just attempts.** A record that triggers a 100k-token expansion on every retry is a budget incident even if it eventually succeeds. I enforce a per-record token ceiling and route violators to a manual queue.

**💰 Math:** on a 1M-document job at $3.15 per thousand documents (from the previous question), a 3% poison rate retried 5× costs 30,000 × 4 extra attempts × $0.00315 = **$378** — annoying but survivable. The same 3% at 100k tokens each, retried 5×, costs 30,000 × 5 × 100,000 × $1.50/1e6 = **$22,500**. The failure mode isn't the retry count; it's the token blowup. Cap tokens.

### Design the caching strategy for an enterprise assistant with 800 tenants, each with a custom policy document. Where does the money go?

Let me lay out the layers and price each one, because the answer is entirely a question of which layer you make shareable.

**Layer 0 — global (tools + core system prompt), ~6,000 tokens.** Identical for all 800 tenants. This is the layer where sharing is worth the most, and it is the one you protect obsessively: deterministic tool ordering, no interpolation, pinned model.

**Layer 1 — tenant policy, ~15,000 tokens, 800 distinct values.** Cannot be shared across tenants by construction. This is where the money goes, and the design question is whether it stays warm.

**Layer 2 — session/retrieved context, variable.** Session-scoped.

**Layer 3 — turns.** Per-request.

Breakpoints at the end of layers 0, 1, and 2 — three of your four, leaving one for the sliding multi-turn position.

**💰 Math (provider API, $3/Mtok, 5-minute TTL).** Suppose 800 tenants generate 400,000 calls/day, but traffic is bursty: a given tenant's calls cluster in working hours with gaps. If the tenant policy expires between calls, every call is a 15,000-token write at 1.25× = $0.056, giving 400,000 × $0.056 = **$22,500/day**. If it stays warm, each call is a 15,000-token read at $0.0045, giving ~**$1,800/day** plus writes. The entire design problem is **keeping layer 1 warm**, and it is a $20,000/day problem.

The levers, in order:
1. **1-hour TTL on layer 1.** Write at 2× ($0.09) but hold it across intra-day gaps. If a tenant makes ≥3 calls per hour, this wins; at 400,000/day across 800 tenants that's 500 calls/tenant/day, so overwhelmingly yes. This alone probably solves it.
2. **Pre-warm on session start.** When a tenant's user opens the app, fire a `max_tokens: 0` warm-up so the first real query hits a warm cache. This is a TTFT play worth ~600 ms on a 15k prefix.
3. **Shrink layer 1.** Audit the 15,000-token policy documents; typically 60% is boilerplate identical across tenants. Hoist that into layer 0. Cutting layer 1 from 15k to 6k cuts both the write premium and the miss cost by 60% and increases layer-0 sharing.

**Self-hosted variant, because the answer changes.** 800 × 15,000 tokens of distinct policy = 12M tokens of KV. At 0.31 MB/token that is **3.7 TB** — you cannot hold the working set in HBM on any plausible fleet. So self-hosted you must either (a) accept a much smaller warm working set with LRU on the active tenants, (b) route with tenant affinity so each replica only needs its own tenants' prefixes warm (32 replicas × 25 tenants = 375k tokens ≈ 116 GB, still too much for one card but tractable with FP8 KV and a CPU offload tier), or (c) offload cold prefixes to CPU/NVMe and pay the PCIe transfer instead of the prefill. Option (b) plus (c) is what production systems converge on, and it is exactly why prefix-affinity routing exists.

**🗣 Say this in the room:** "Three tiers: a 6k global prefix shared by everyone, a 15k tenant policy that must stay warm, and per-session context. Almost all the money is in whether the tenant layer survives between calls — at 400k calls/day it's a $20k/day swing — so I'd use the 1-hour TTL there, pre-warm on session open, and audit the policy docs to hoist the boilerplate into the global layer. Self-hosted, the same problem becomes a routing problem: 800 distinct 15k prefixes is 3.7 TB of KV, so you need tenant affinity plus an offload tier, not more HBM."
