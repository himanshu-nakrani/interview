### Explain attention sinks. Why do the first four tokens of a sequence matter so much?

This is one of my favourite results because the explanation is a softmax constraint and nothing else, and once you see it you cannot unsee it.

Softmax outputs must sum to 1. At every attention head, at every position, the model is forced to distribute exactly one unit of attention mass across the visible tokens — even when *no* token is relevant to what this head is currently doing. The network needs a place to dump that mass. The only positions guaranteed visible to every query in a causal model are the first few, so during pretraining the model learns to use them as a sink: it drives their keys to a configuration that reliably absorbs leftover attention, largely independent of what those tokens actually say. Their semantic content is close to irrelevant — you can replace them with a fixed placeholder and the behaviour persists.

Now the consequence. Naive sliding-window attention — "just keep the last 4k tokens and drop the rest" — evicts the sink. The mass that used to go to position 0 now has to be redistributed among the window's tokens, which inflates their attention weights, which shifts the output distribution, and perplexity explodes almost immediately. It is not a gradual degradation; it is a cliff.

StreamingLLM's fix is embarrassingly cheap: keep the first **4 tokens** permanently plus a sliding window of the most recent $L$. That's it. No fine-tuning, no architecture change. Perplexity stays stable over millions of tokens of streaming input.

What you must be honest about, and where interviewers push: **StreamingLLM does not extend the model's effective context.** It gives you a model that can *run forever* without blowing up, not a model that *remembers* everything. Everything between the sinks and the window is gone. For a chat product where the user only cares about recent turns, that's a perfect fit. For document QA where the answer is at token 40,000 and your window is 4,000, it is a correctness bug wearing a performance costume.

**📄 Paper:** Xiao et al. (2023), "Efficient Streaming Language Models with Attention Sinks" — identified the sink phenomenon and showed that retaining a handful of initial tokens alongside a sliding window enables unbounded streaming, replacing naive window attention which collapsed.

**🗣 Say this in the room:** "Attention sinks exist because softmax has to sum to one and the model needs somewhere to put unallocated mass. The first tokens are visible to every query, so they become that dumping ground. Evict them and the distribution over the remaining tokens is wrong — perplexity blows up. StreamingLLM keeps four sinks plus a window, which buys you infinite streaming but explicitly not infinite memory."

### What is the heavy-hitter hypothesis, and what does H2O actually throw away?

The empirical observation behind H2O: attention matrices at inference are extremely sparse. A small subset of tokens — the "heavy hitters" — accumulate the overwhelming majority of attention mass across queries, and which tokens those are is largely stable as generation proceeds. If most of the cache is receiving near-zero attention, most of the cache is not earning its bytes.

The mechanism is a greedy eviction policy with a running score. For each cached token, maintain the accumulated attention it has received summed over all queries so far. When the cache budget is exceeded, evict the lowest-scoring tokens. H2O keeps two populations: the heavy hitters by accumulated score, plus a recency window (the most recent tokens always matter and haven't had time to accumulate score). The paper frames the optimal policy as a submodular maximisation and shows the greedy version has a bounded gap from it.

What it discards — and this is the honest answer, not the marketing one:

**It discards the ability to retrieve a token that becomes relevant later.** The score is accumulated over *past* queries. A token that nothing has attended to yet, but which the user's next question will make critical, has a low score and gets evicted. This is the fundamental limitation of any attention-history-based policy: it is a cache with a *past-looking* eviction signal serving a *future-looking* access pattern. In backend terms, LRU works because access patterns are temporally correlated; here the correlation is weaker and it is exactly the retrieval workloads that break it.

**It discards determinism across cache budgets.** Two runs with different budgets produce different outputs on the same prompt. That is a genuine problem for reproducibility, for evaluation, and for debugging a customer complaint.

**It adds per-step bookkeeping.** You maintain and sort scores over the cache every step, which is a real kernel cost that partially offsets the bandwidth saving.

**📄 Paper:** Zhang et al. (2023), H2O: Heavy-Hitter Oracle — accumulated-attention-score eviction with a recency window, replacing uniform-window policies that dropped tokens by position with no regard to their attention mass.

**⚠ Trap:** benchmarking H2O-style eviction on summarisation and concluding it's lossless. Summarisation is the *most* forgiving task for eviction because the answer is a diffuse function of the whole document — losing 80% of the cache degrades it gracefully. Needle-in-a-haystack and exact-citation tasks fall off a cliff at the same budget. Always evaluate eviction on precise-retrieval tasks, not on generative ones.

### SnapKV and H2O both compress the cache. What's the actual difference and when does each one apply?

They operate at different points in the request lifecycle, and confusing them is a real tell.

**H2O compresses continuously during decode.** It's an online eviction policy: the cache is at budget, a new token arrives, something must go. It applies to long *generations*.

**SnapKV compresses once, at the end of prefill.** Its insight is that the model reveals what it cares about *before* it generates anything: the attention pattern of the last few prompt tokens — the "observation window," typically the final 32 or so positions — already indicates which parts of the prompt the generation will attend to. So you let those observation tokens vote: aggregate their attention over the prompt positions, apply a pooling step so you keep contiguous clusters rather than isolated tokens (attention is locally coherent, and keeping token $i$ while dropping $i\pm1$ breaks phrases), select the top-$k$, and discard the rest. Then generate normally against a cache that is a fraction of its original size.

So: **SnapKV is a prefill-side compressor for long prompts. H2O is a decode-side evictor for long generations.** They compose — you can SnapKV the prompt and then run a recency+heavy-hitter policy on the generation — and in practice long-prompt/short-output workloads (document QA, RAG, code review over a large file) are far more common than short-prompt/long-output ones, which is why the SnapKV family gets more production attention.

The clustering detail is the part I'd want a candidate to bring up unprompted. Selecting the top-$k$ individual positions by score produces a shredded cache: you keep the token "Article" and drop "12(b)", and the model sees fragments. Pooling the scores over a small window before selecting keeps spans intact and is worth several points of accuracy on retrieval tasks for essentially zero cost.

**📄 Paper:** Li et al. (2024), SnapKV — uses an observation window at the end of the prompt to vote on which prefix KV entries to retain, with pooling to preserve contiguous spans; replaces uniform prompt truncation and position-agnostic eviction for the long-prompt case.

**🔍 Failure taxonomy:** if your workload is long prompt, short answer → SnapKV-class prefill compression, and measure on exact-retrieval evals. Long prompt, long answer (agents) → SnapKV for the prompt plus a recency-plus-sink policy for the generation. Short prompt, very long answer (story generation, long CoT) → H2O-class or sink-plus-window; SnapKV has nothing to compress. Multi-turn chat where earlier turns get re-queried → *neither*; eviction is wrong here, use offload, because the user can and will ask about turn 3.

### Cross-layer KV sharing — what does it buy and what does it break?

GQA shares KV across heads within a layer. Cross-layer attention asks the obvious next question: why not share across *layers*? Adjacent transformer layers produce highly similar key and value representations — the residual stream changes gradually — so layer $2i+1$ can attend against the KV computed by layer $2i$ instead of computing and storing its own. Sharing between pairs gives a clean **2×** reduction; sharing across larger groups gives more, with a steeper quality cost.

This is a genuinely different axis from GQA and they multiply. A model with 8 KV heads instead of 64 (8×) plus 2-layer sharing (2×) is 16× smaller than the MHA baseline, and the two techniques are close to orthogonal in their quality impact — the CLA work reports better memory/accuracy Pareto frontiers than pushing GQA alone to more aggressive head counts.

What it breaks:

**It is a pretraining decision, like MLA.** You cannot bolt it onto a served checkpoint. Anyone proposing CLA for a model already in production is proposing a retrain.

**It couples layers that were independent.** Every serving mechanism that operates per-layer now has a dependency graph. Layer-wise KV offload, layer-wise pipelining of the KV transfer in disaggregated serving, and per-layer eviction policies all have to respect the sharing groups. Pipeline parallelism gets awkward if a sharing group straddles a stage boundary — you'd have to send KV between stages, which is exactly what PP is designed to avoid.

**It reduces the model's capacity to specialise attention by depth.** Early layers attend locally and syntactically; late layers attend semantically and long-range. Forcing pairs to share a KV representation constrains that specialisation, and the quality cost is concentrated in exactly the long-range behaviours you cared about when you went looking for cache savings.

**📄 Paper:** Brandon et al. (2024), Cross-Layer Attention — shares KV projections across adjacent layers for roughly 2× cache reduction on top of GQA, improving the memory/accuracy trade-off relative to reducing KV heads alone.

### A PM asks why we can't just delete the middle of a long conversation to save memory. What breaks?

Three things, in increasing order of how much they will cost you.

**The mechanical one: you cannot "just delete" cached KV in the middle without invalidating everything after it, if you also intend to reuse the prefix cache.** Prefix caching hashes block content along with its position chain; drop block 40 and blocks 41 onward are no longer reachable by that chain. You either recompute the suffix or you accept that the cache entry is gone. So "delete the middle" is often silently "recompute from the deletion point," which costs prefill, not saves it.

**The behavioural one: the model's position encodings now have a hole.** With RoPE, each cached key carries a rotation determined by its absolute position. If you evict positions 5,000–50,000 and keep 50,001 onward, the surviving tokens still carry their original rotations, so the model perceives a 45,000-token gap. Some eviction schemes re-index positions to close the gap, which is a different lie — now the model believes those tokens were adjacent when they weren't, and long-range reasoning about ordering degrades. Both options are wrong in different ways; the literature's answer is generally to re-index, and to accept the cost.

**The product one, which is the actual answer to the PM: the middle is where the commitments are.** In a 60-turn support conversation, the middle is where the customer stated their order number, the agent promised a refund, and the policy exception was agreed. The beginning is greetings and the end is "so what's the status?" A recency-plus-sink policy keeps precisely the two least informative regions. I have seen this ship, and the failure mode is the worst kind: the bot confidently re-asks for information the customer already gave, or worse, contradicts a commitment it made twenty turns ago. It doesn't error. It doesn't show up in latency. It shows up in CSAT three weeks later.

**🗣 Say this in the room:** "Recency-based eviction keeps the greeting and the last question and throws away the middle, and in a support conversation the middle is where the commitments are. If we need to shrink the context I'd rather summarise the middle into structured state we control and re-inject it — that's a lossy compression we can evaluate — than let an attention-score heuristic decide what the customer said."

The design I'd actually push: keep the full transcript in durable storage, maintain an explicit running "conversation state" object (entities, commitments, open questions) updated by a cheap model, and rebuild the prompt each turn from sinks + state + recent turns. That is a compression you can write an eval for, version, and roll back. Cache eviction is a compression you cannot inspect.

### Pick one for me: for a 128k-context legal document assistant, do you quantise, evict, or offload? Defend it.

**Offload, with FP8 quantisation as a second lever, and eviction essentially never.** Here is the reasoning chain, because the reasoning is the answer.

Start from the workload's shape. Legal document QA is long-prompt, short-output, and — critically — **multi-query over the same document**. A user uploads a 120-page contract and asks eleven questions over twenty minutes. The prompt is stable, the queries vary, and the questions are precise-retrieval ("what's the termination notice period in the assignment clause?"), not diffuse.

**Eviction is disqualified by the access pattern.** Any attention-score-based policy is trained on the *previous* query's attention. The next question will target a different clause. You would be evicting the answer to question seven while answering question six. For precise retrieval over a document where any span may become the target, no past-looking eviction signal is valid. I would reject this in design review with that one sentence.

**Quantisation is fine and I'd take it, but it's a 2× lever, not a solution.** 128k at 320 KiB/token is 40 GiB per session; FP8 makes it 20 GiB. On a 200 GiB pool that moves us from 5 concurrent sessions to 10. With 500 lawyers on the platform, neither number is the product. And the eval discipline is non-negotiable here: legal retrieval is exactly the task class where KV quantisation error is most visible, so it ships behind a long-context exact-citation eval or it does not ship.

**Offload is the structural fix, because the access pattern is bursty per session.** During the twenty minutes the lawyer is reading an answer and typing the next question, that 20–40 GiB of cache is idle. Move it to host DRAM; pull it back when the next query arrives. The cost is a fetch on each turn, and the arithmetic says that fetch is cheap relative to recomputing the 128k prefill — I derive the exact break-even in the next question, but the headline is that at 128k the prefill takes ~12.9 seconds of a 4-GPU node while the fetch over PCIe Gen5 takes under a second.

The architecture I'd draw: GPU HBM holds active sessions; host DRAM (1–2 TB per node is cheap) holds recently-active sessions; object storage holds documents for cold re-prefill. Session affinity at the router so a returning user lands on the node holding their cache. FP8 KV throughout. A hard cap on `max_model_len`, and long documents chunked with retrieval rather than stuffed whole — because the honest answer to "128k context" is often "you don't need 128k, you need better retrieval," and I'd want that experiment run before we build the tiering.

**💰 Math:** 500 lawyers, 20% concurrently active = 100 live sessions. At FP8 128k that's 20 GiB each = 2 TB of live cache. On GPU alone at 200 GiB/node that's 10 nodes = 40 H100s = **$80/hour = $58k/month** just to hold cache. With DRAM offload and, say, 15 genuinely-in-flight sessions at any instant, you need 2 nodes of GPU (300 GiB HBM pool) plus 2 TB of host DRAM (~$10k of hardware, amortised) = 8 H100s = **$16/hour = $11.7k/month**. That is a 5× cost reduction and it is why offload exists.

### Derive the break-even: when is fetching KV from host memory cheaper than recomputing the prefill?

This is the single most useful piece of arithmetic in the offload conversation, and it is four lines.

Recomputing the prefill for $S$ tokens costs approximately $2 N S$ FLOPs (ignoring attention, which I'll add back). Fetching costs $S \cdot b$ bytes over a link of bandwidth $BW$. Set them equal:

$$\frac{S \cdot b}{BW} = \frac{2NS}{F} \;\Longrightarrow\; BW_{\text{break-even}} = \frac{b \cdot F}{2N}$$

where $b$ is bytes per token and $F$ is achieved FLOP/s. The $S$ cancels — which is why this is a clean, model-level constant rather than a per-request one.

For Llama-3.3-70B FP8 on 4×H100 at 40% MFU: $F = 4 \times 1.979\times10^{15} \times 0.40 = 3.17\times10^{15}$ FLOP/s. $b = 327{,}680$. $2N = 1.4\times10^{11}$.

$$BW_{\text{break-even}} = \frac{327{,}680 \times 3.17\times10^{15}}{1.4\times10^{11}} = 7.4\times10^{9}\text{ B/s} = \mathbf{7.4\ GB/s}$$

**Above ~7.4 GB/s, fetching beats recomputing. Below it, recompute.** Now place the real links on that axis:

| Link | Achieved BW | Verdict vs 7.4 GB/s |
|---|---|---|
| NVLink 4 (GPU↔GPU, H100) | ~450 GB/s | wins by 61× |
| Grace-Hopper NVLink-C2C (CPU↔GPU) | ~450 GB/s | wins by 61× |
| PCIe Gen5 ×16 | ~50 GB/s | wins by 6.8× |
| PCIe Gen4 ×16 | ~25 GB/s | wins by 3.4× |
| 400G InfiniBand | ~50 GB/s | wins by 6.8× |
| Single Gen4 NVMe SSD | ~7 GB/s | **loses, marginally** |
| 4-drive NVMe RAID-0 | ~25 GB/s | wins by 3.4× |
| 25G Ethernet | ~3 GB/s | loses by 2.5× |

The second-order effect makes offload look *better* than this table suggests at long context, and it's the part that separates a good answer from a great one. Recompute cost includes the quadratic attention term; fetch cost stays linear. At 128k for a 70B, total prefill is $4.1\times10^{16}$ FLOP = 12.9 s at $3.17\times10^{15}$ FLOP/s, while the cache is 40 GiB = $4.29\times10^{10}$ B. Break-even bandwidth there is $4.29\times10^{10}/12.9 = 3.3$ GB/s. **The longer the context, the lower the bandwidth you need for offload to win** — a single NVMe drive that loses at 8k wins by 2× at 128k.

**📐 Numbers you must know:** the offload break-even for GQA models in the 7B–70B range lands in the **3–9 GB/s** band, falling as context grows. That band sits almost exactly on top of single-drive NVMe, which is why NVMe offload is the genuinely contested case while DRAM offload is a settled win.

### So where does NVMe offload actually become a net loss? Give me numbers.

Three failure regions, and you need all three because a single-number answer is wrong.

**Region one: short contexts on fast GPUs.** At 8k on a 70B, prefill is $1.24\times10^{15}$ FLOP = 390 ms at 40% MFU, and the cache is 2.5 GiB. Fetching over a single 7 GB/s NVMe drive: $2.68\times10^9/7\times10^9 = 383$ ms. You have spent 383 ms of TTFT to save 390 ms of TTFT, plus you now own a storage tier. Net: nothing, for real operational cost. Below ~8k, recompute unconditionally.

**Region two: aggregate bandwidth saturation.** The per-request math is not the system math. If you are serving 80 concurrent 8k sessions and 20% of them fetch per second, that is 16 fetches/s × 2.684 GB = **43 GB/s of sustained read demand**. One NVMe drive delivers 7. You have built a queue in front of your storage tier and every fetch now waits behind others; your p99 TTFT is not 383 ms, it is 383 ms times your queue depth. **The rule I enforce: size offload storage on aggregate sustained bandwidth under peak fetch rate, never on single-request latency.**

**Region three: the write side, which everyone forgets.** Offload is not read-only — you write the cache down when you evict a session. At 80 sessions × 2.5 GiB with a 30% turnover per minute, that's 60 GB/min = 1 GB/s of sustained writes. Consumer and even many datacentre NVMe drives have write endurance measured in drive-writes-per-day; sustained 1 GB/s is 86 TB/day, which will wear out a 3 DWPD 3.84 TB drive in months. And sustained-write throughput after the SLC cache is exhausted is often a third of the headline read number.

**⚠ Trap:** benchmarking offload with a single request on an idle machine. It always wins there. The regressions appear only under concurrency, because the offload path contends for the *same* PCIe lanes your GPUs use for everything else, and because storage queueing is invisible in a single-request test. My acceptance criterion for an offload tier is a soak test at 1.5× peak concurrency measuring p99 TTFT, not a microbenchmark.

**🔍 Failure taxonomy:** offload enabled, p50 TTFT improved, p99 got worse → aggregate bandwidth saturation, you are queueing on the storage tier. TTFT improved but throughput dropped → the offload path is stealing PCIe bandwidth from something else, or the copy is not overlapped with compute. Hit rate on the offload tier is low → your session affinity is broken and requests are landing on nodes that don't hold their cache. Everything looks fine but costs went up → you are writing caches that are never read back; measure offload read-hit rate and stop writing below some session-value threshold.

### What does an LMCache-class KV layer do that vLLM's built-in prefix caching doesn't?

Built-in automatic prefix caching is deliberately narrow: it hashes fixed-size KV blocks by their content plus the chain of preceding blocks, keeps them in the GPU pool, and reuses them when a new request shares an exact prefix. It is fast, correct, and free. Its three limitations are structural.

**It is GPU-only.** Blocks live in the KV pool and are evicted when the pool needs room. Your cache is bounded by HBM, which is the scarcest and most expensive memory you own.

**It is exact-prefix-only.** The match must start at token 0 and be contiguous. In RAG this is fatal: you retrieve five chunks, the ordering varies per query, and chunk C's cached KV — computed when it sat at position 0 — is invalid at position 3,000, because RoPE encodes absolute position and because those keys and values were computed attending over different preceding text.

**It is single-instance.** Replica A's cache does nothing for replica B. With eight replicas behind a round-robin load balancer you have eight cold caches and a hit rate divided by eight.

An external KV layer attacks all three. It provides **tiering** — GPU HBM, host DRAM, local NVMe, and remote object storage as a hierarchy with its own admission and eviction policy, so a session's cache survives eviction from the GPU pool. It provides **cross-instance sharing** — a KV store the whole fleet reads from, so replica B benefits from replica A's prefill, which multiplies your effective hit rate by the replica count for shared system prompts. And the more research-flavoured capability, **non-prefix reuse**: take independently-cached text chunks, concatenate them in a new order, and instead of recomputing everything, selectively recompute only the small fraction of tokens whose KV deviates most from what full recomputation would produce — the CacheBlend line of work showed roughly 10–20% recomputation recovers most of the quality of full recomputation on RAG-shaped inputs.

**⚠ Trap:** assuming cross-instance KV sharing is free because "it's just a cache." It moves KV over the network. A 2.5 GiB cache pull over 25 Gb/s Ethernet is $2.68\times10^9/3.1\times10^9 = 864$ ms — far slower than recomputing the prefill (390 ms). Cross-instance sharing needs RDMA-class fabric or it is a pessimisation. Check the link before you check the architecture diagram.

**💰 Math:** the case where it's overwhelming is a large shared system prompt. 12,000-token system prompt, 8 replicas, 200k requests/day. Without sharing, each replica cold-starts and each replica's hit rate is limited by its own traffic; with a shared store, the prompt is prefilled essentially once per TTL. Prefill for 12k tokens on a 70B is $2\times70\times10^9\times12{,}000 = 1.68\times10^{15}$ FLOP = 530 ms of a 4-GPU node at 40% MFU. Saving that on even 50% of 200k daily requests is $100{,}000 \times 0.53 = 53{,}000$ node-seconds/day = 14.7 node-hours/day × 4 GPUs × $2/GPU-hr = **$118/day = $3.5k/month**, on one model, for a cache layer.

### Explain the Mooncake-style architecture. What problem does it solve that a bigger GPU wouldn't?

The reframing is the point: **treat the KV cache as the primary object the system is designed around, not as a side-effect of running a model.** Once you do, the cluster looks different.

The architecture has three moves. First, **disaggregate prefill from decode** into separate GPU pools, because the two phases have opposite bottlenecks — prefill is compute-bound and wants big compute-dense chips, decode is bandwidth-bound and wants cheap high-bandwidth ones — and mixing them means one phase's long prefills stall the other's decode steps. Second, **build a distributed KV pool out of the resources the GPU cluster already has and wastes**: every GPU node has hundreds of gigabytes of idle host DRAM, terabytes of idle NVMe, and an RDMA NIC that sits mostly idle between collectives. Aggregated, that is a multi-terabyte KV tier at essentially zero marginal hardware cost. Third, **make the scheduler KV-aware**: route a request to the node that already holds the largest matching prefix, rather than to the least-loaded node. Under overload, use *early rejection* — decide at admission time whether a request can be served within SLO, and reject it immediately rather than accepting it, prefilling it, and then thrashing.

Why a bigger GPU doesn't solve it: HBM capacity scales with generations slowly and expensively, while context length demand has scaled by orders of magnitude, and — decisively — **the cache's access pattern is bursty and its working set is far smaller than its total set.** Of 2 TB of live sessions, maybe 150 GB is being attended to in any given second; the rest is sitting between turns. Paying HBM prices for cold bytes is the mistake. A memory hierarchy is the correct answer to a working-set problem, and it has been the correct answer since the first CPU cache. This is precisely the argument you already make when you put Redis in front of Postgres; the only novelty is which bytes and which price ratio.

**📄 Paper:** the Mooncake work from Moonshot AI (2024) — a KVCache-centric disaggregated serving architecture with separate prefill/decode clusters, a distributed KV pool assembled from underutilised CPU DRAM and SSD across the GPU cluster, cache-aware scheduling, and early rejection under overload; it replaced the model of "each replica is an independent, stateless box."

**🗣 Say this in the room:** "The insight is that the KV cache is the stateful object in an otherwise stateless service, and once you admit that, everything follows: you tier it, you route requests to where it already lives, you bill for it, and you autoscale on it. A bigger GPU doesn't help because the problem isn't total capacity, it's that you're paying HBM prices for a cache whose working set is 10% of its footprint."

### In prefill/decode disaggregation, how big is the KV transfer and does it fit in your latency budget?

Concretely: the prefill pool produces the entire KV cache for the prompt, and every byte of it has to reach the decode pool before decoding can start. For a 70B at 8k prompt, that is **2.5 GiB, per request**. That is not a control-plane message; it's a bulk data movement that is now on your critical path.

Time it against real fabrics:

| Fabric | Achieved BW | 2.5 GiB transfer | 40 GiB (128k) |
|---|---|---|---|
| NVLink (same node) | ~450 GB/s | 6.0 ms | 95 ms |
| 400G InfiniBand / RoCE | ~50 GB/s | 54 ms | 858 ms |
| 100G Ethernet | ~12.5 GB/s | 214 ms | 3.4 s |
| 25G Ethernet | ~3.1 GB/s | 864 ms | 13.8 s |

The critical framing question — and interviewers ask it exactly this way — is **which SLO does this land in?** The transfer happens once, between prefill completing and the first decode step, so it lands in **TTFT, not ITL**. Against a 400 ms TTFT budget, 54 ms on InfiniBand is 13.5% — acceptable. 214 ms on 100G Ethernet is 54% of the budget, which leaves nothing for retrieval, prefill and queueing, so it fails. **Disaggregation requires RDMA-class fabric; on commodity Ethernet it is a pessimisation and I'd reject the design.**

The mitigation that makes disaggregation viable is **layer-wise overlap**. Layer 0's KV is final as soon as layer 0's prefill completes — you don't have to wait for layer 79. Stream each layer's KV to the decode node as it is produced, and the transfer overlaps with the remaining prefill compute. For 8k on a 70B, prefill is ~390 ms and the transfer is 54 ms on IB; overlapped, the exposed cost is roughly the last layer's transfer, ~0.7 ms, plus handshake. That turns a 13.5% TTFT tax into a rounding error, and it is why every serious disaggregated stack ships a dedicated transfer library with layer-wise pipelining and RDMA one-sided writes rather than calling `send()` on a socket.

**⚠ Trap:** sizing the fabric for average request size. The transfer is proportional to prompt length, and prompt length is right-skewed. Your p99 request moves 20× the bytes of your median. Provision the interconnect for p99 prompt length × peak request rate, and put a circuit breaker on the transfer so a single 500k-token prompt cannot saturate the fabric for everyone else.

### Would you compress the KV cache for transfer? What's the decision rule?

Yes, sometimes, and the rule is a bandwidth-delay comparison you can do in your head.

Compress when **transfer time exceeds compression time plus compressed transfer time**, i.e. when the link is slow enough that the bytes dominate. Formally, with compression ratio $r$, compression throughput $C$ bytes/s, and link bandwidth $BW$:

$$\text{compress if}\quad \frac{b}{C} + \frac{b/r}{BW} < \frac{b}{BW} \quad\Longleftrightarrow\quad C > \frac{BW \cdot r}{r-1}$$

For FP8 casting ($r=2$), which runs at essentially memory-bandwidth speed on-GPU — call it $C = 1\times10^{12}$ B/s conservatively — the condition is $BW < C(r-1)/r = 5\times10^{11}$ B/s = 500 GB/s. Every fabric except intra-node NVLink is below that. **So: cast to FP8 for transfer, always, on anything crossing a node boundary.** It's free.

Beyond FP8 the calculus changes, because now you're trading quality, not just cycles. The decision rule I use:

**Lossless-in-effect compression (FP8 cast when the cache is already FP8-tolerant, or bf16→FP8 for transit only, dequantising on arrival):** always on, for cross-node transfer. Zero quality argument needed if the receiving side is going to run FP8 attention anyway.

**Lossy compression below FP8 (INT4, low-rank projection, token selection) for transfer only:** only when the link is the binding constraint *and* you have measured quality at your deployed context length. The trap here is that "transfer-only" compression is not transfer-only — whatever you send is what the decode pool attends against for the entire generation. You have not compressed a network payload, you have permanently degraded the cache. That distinction gets lost in design docs and it is worth stating explicitly in review.

**General-purpose entropy coding (zstd, LZ4) on KV tensors:** almost never. KV activations are high-entropy floats; typical ratios are 1.1–1.3×, and the CPU cost of a few GB/s of zstd is real. You would be spending cores to save 20% of bytes. Cast to a smaller float instead — that's a 2× ratio at zero cost.

**💰 Math:** 128k-context disaggregated serving at 40 GiB per request over 400G IB (~50 GB/s achieved). bf16: 858 ms of transfer. FP8: 429 ms. Against a 2 s TTFT budget for a 128k request that is the difference between 43% and 21% of the budget. Now the aggregate: at 4,000 long-context requests/hour, bf16 demands $4{,}000 \times 4.295\times10^{10} / 3600 = 4.77\times10^{10}$ B/s = **47.7 GB/s** against a 50 GB/s link — saturated, with queueing on top. FP8 halves it to 23.9 GB/s, 48% utilisation, comfortable. The compression isn't a latency optimisation, it's the thing that makes the architecture fit at all.
