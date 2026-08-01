### Your generation quality degrades, but only under high concurrency. Nothing changed in the model or the prompts. Debug it.

The shape of this bug is important: quality that depends on *load* means quality that depends on a **scheduler decision**, because the scheduler is the only thing that knows about load. Model weights, prompts and sampling parameters are all load-independent. So I go straight to the four places where a scheduler under memory pressure changes what the model actually sees.

**Hypothesis 1 — preemption with a buggy recompute path.** Under KV pressure the engine evicts a running sequence and later recomputes its cache from the prompt plus already-generated tokens. If the recompute path reconstructs the *prompt* but forgets the already-emitted tokens, or reconstructs them with wrong `cache_position` (so RoPE rotates them at the wrong absolute positions), the sequence resumes with a subtly corrupted cache. Symptom: fluent output that loses the thread, repeats, or contradicts what it said thirty tokens earlier — and only when preemption is happening. **Check `num_preemptions_total` and correlate it with your quality metric over time.** This is my first hypothesis every time.

**Hypothesis 2 — silent context truncation under pressure.** Some gateways and some engine configurations respond to admission failure by truncating the prompt to fit, rather than by queueing or rejecting. Under load, requests get their contexts cut, and the model answers from a truncated document. This is the worst failure mode in the list because the response is confident and well-formed. Look for a truncation counter or a warning log; if you don't have one, that is the bug to fix first.

**Hypothesis 3 — an eviction policy that is only active under pressure.** If someone enabled StreamingLLM-style sink+window eviction or H2O-style heavy-hitter eviction "as a safety valve," it fires only when the pool is tight — which is exactly load-correlated. Fine on chat, catastrophic on document QA.

**Hypothesis 4 — a batching-dependent numerics or kernel path.** Attention kernels dispatch differently by batch size and sequence length: CUDA-graph-captured path versus eager, a split-K decode kernel at large batch versus a simpler one at small batch, different reduction orders in fp16. Genuine nondeterminism from reduction order is normally tiny, but a kernel bug on a path only exercised at batch > 64 is not. Test by pinning `enforce_eager` and re-running the load test; if quality recovers, it is the graph/kernel path.

**Hypothesis 5 — it isn't the model at all.** Under load, your retrieval tier may be timing out and returning fewer documents, or your reranker may be shedding load, or a tool call may be failing and getting swallowed. Always confirm the *input* to the model is identical before blaming the model.

**🔍 Failure taxonomy as a decision procedure:** (a) Log the exact rendered prompt token IDs for a sample of requests at high and low load and diff them — if they differ, it is truncation or a retrieval problem, stop here. (b) If identical, check preemption counters; if nonzero, disable preemption by lowering `max_num_seqs` and re-test. (c) If preemption is zero, check for eviction/window config. (d) If none, force eager execution and re-test to isolate the kernel path. (e) If quality still degrades with identical inputs, no preemption, no eviction and eager kernels, you have a genuine numerics issue and it is time to dump attention outputs at low and high batch and compare.

**⚠ Trap:** reproducing the bug in a single-request test and concluding it doesn't exist. Load-dependent bugs require load to reproduce. Build a replay harness that fires your production trace distribution at the server and scores outputs — this is one of the highest-value pieces of infrastructure an applied AI team can own, and almost nobody has it.

### A user reports the model "forgets" the beginning of long conversations, but only sometimes. Give me your hypothesis list, ranked.

I want to establish one thing before hypothesizing: **is it correlated with total context length, with elapsed session time, or with server load?** Those three point at completely different layers, and asking that question is half the answer.

**Correlates with context length → the context is being cut somewhere.**
1. **Application-layer conversation trimming.** Almost every chat app has a "keep the last N turns" or "keep under M tokens" policy. It fires at a threshold, which is exactly why the bug is intermittent — short conversations are fine, long ones silently lose their head. Check this first; it is the answer more often than anything model-side.
2. **Server-side truncation at `max_model_len`.** If the request exceeds the served context and the gateway left-truncates instead of erroring, the system prompt goes first. Look for a truncation log line. Fail loudly instead.
3. **Sliding-window attention in the model.** Read `sliding_window` in `config.json`. If the local layers have a 4096-token window and the fact is 30k tokens back, retrieval is going through the lossy multi-layer path and will work sometimes and not others. This produces exactly "sometimes forgets."
4. **RoPE scaling degradation.** If the model was extended to 128k with position interpolation or YaRN, quality at 100k is measurably worse than at 8k even though both are "supported." Advertised context and usable context are two different numbers and should always be quoted as two numbers.
5. **Lost-in-the-middle.** Attention to mid-context content is empirically weaker than to the head and tail. If the fact is at 40% depth in a 60k context, degraded recall is expected behavior, not a bug. Test by moving the same fact to the start and the end and measuring the recall difference.

**Correlates with server load → a serving-layer eviction.** Preemption-with-recompute bugs, sink+window eviction, heavy-hitter eviction. See the previous question's procedure.

**Correlates with elapsed time → a cache TTL.** If you use a hosted API's prompt caching with a 5-minute TTL and the user pauses, the cache expires; that costs money and latency but should *not* change output. If it does change output, something in your pipeline is rebuilding the prompt differently on a cache miss — for example, re-rendering from a summarized history. That is a real bug and a common one in agent frameworks that summarize on cache miss.

**⚠ Trap:** assuming "supports 128k" means "retrieves reliably at 128k." It does not. The honest way to state a model's capability is two numbers: advertised limit, and the length at which your own multi-fact retrieval eval still passes your threshold. For many models those differ by 2–4×. Measure it on your corpus; do not trust a vendor's needle-in-a-haystack chart, which uses a single distinctive fact in filler text and is a weak proxy for real multi-fact reasoning.

### Someone enabled fp8 KV cache. MMLU is unchanged but your long-context retrieval eval dropped three points. Explain the mechanism.

This is not a coincidence and it is not noise — it is exactly the predicted failure signature, and being able to explain why is a strong signal in a serving round.

**Mechanism part one: retrieval depends on a single sharp logit margin.** When the model needs to pull a specific fact from position 87,000, one attention head must place nearly all its mass on that one key. That requires the dot product `q · k_target` to exceed `q · k_j` for all other j — and with a 128k context there are 131,071 competitors. Quantization adds roughly independent noise to every logit. The probability that at least one distractor's noisy logit exceeds the target's grows with the number of distractors. Formally, the max of many noisy competitors grows like `σ√(2 ln T)`, so the required clean margin scales with `√(ln T)` — the retrieval task gets structurally harder as context grows, and quantization noise eats directly into the margin. MMLU items are a few hundred tokens with the answer in the weights, not the context: almost no distractors, and no dependence on a sharp attention peak at all.

**Mechanism part two: attention sinks wreck your scales.** Token 0 (and often the first few) carry key and value activations an order of magnitude larger than typical tokens, because the softmax must sum to one and heads dump unused mass there. With a **per-tensor** fp8 scale, that outlier sets the scale for the entire tensor and crushes the effective precision of all other tokens. fp8 e4m3 has only 3 mantissa bits — about 2 decimal digits of relative precision — and if the scale is set 10× too large you have effectively lost a bit or more of that. Fix: per-token or per-head scales, and/or keeping the sink tokens in bf16.

**Mechanism part three: keys have per-channel outliers.** Certain coordinates of the key vector are systematically large across essentially all tokens. Quantizing keys **per-token** lets one outlier channel dominate that token's scale. Keys want **per-channel** scaling; values are better behaved and want per-token. This asymmetry is the core finding behind KIVI-style low-bit KV quantization, and implementations that ignore it underperform the literature badly.

**What I would actually do:**
1. Re-run with per-token (or per-head) K scales rather than per-tensor, and check whether the 3-point gap closes. Usually most of it does.
2. Exclude the first 4 tokens from quantization — keep sinks in bf16. Cost: 4 tokens × 320 KiB = 1.25 MB per sequence. Free.
3. Consider **asymmetric precision**: values in fp8, keys in bf16. Keys drive the argmax; values only get averaged, and averaging is noise-tolerant. This gives you 1.33× cache reduction instead of 2× but often recovers nearly all the quality. I have found this trade underused.
4. Re-run the eval *segmented by context-length decile*. If the drop is concentrated above 32k and your P99 traffic is 8k, ship fp8 and cap `max_model_len` — that is a legitimate engineering decision, not a cop-out.

**⚠ Trap:** validating a KV-precision change on aggregate benchmarks. **KV quantization damage is concentrated in exactly the capability that aggregate benchmarks don't measure.** The rule I enforce in review: no KV-precision change ships without a long-context retrieval eval, segmented by context length, run on our own corpus at our own P95 length.

### What actually happens in memory when the scheduler preempts a sequence mid-generation? Compare recompute and swap.

Preemption is the KV pool's eviction, and like any eviction you choose between throwing state away and moving it somewhere slower.

**Recompute (the vLLM default for most cases).** The victim's physical blocks are returned to the free pool immediately — refcount drops, blocks are reusable this step. The sequence goes back to the waiting queue with its full token history (prompt + tokens generated so far) as its new "prompt." When rescheduled, the engine prefills that entire history from scratch. Cost = a full prefill of `len(prompt) + len(generated)` tokens: `2 · N · T` FLOPs. For a 70B model and a victim at 20k tokens: `2 × 70e9 × 20,000 = 2.8e15 = 2.8 PFLOP` → 0.875 s of 8-GPU node time. That work is pure waste, and it also occupies the batch, delaying everyone.

**Swap.** The victim's blocks are copied to pinned host memory, the physical blocks are freed, and the block table records the CPU location. On reschedule, blocks are copied back. Cost = `2 × bytes / PCIe bandwidth`. For that same 20k-token sequence on Llama-3-70B: 320 KiB × 20,000 = 6.4 GB, out and back = 12.8 GB at ~64 GB/s (PCIe Gen5 x16) = **200 ms**, versus 875 ms of recompute. Swap also consumes host RAM you must have provisioned, and the copies contend with any other PCIe traffic.

**The decision rule** falls straight out of the arithmetic. Recompute cost scales as `T · 2N / node_FLOPS`; swap cost scales as `T · kv_bytes_per_token · 2 / PCIe_BW`. Setting them equal:

```
2N / FLOPS  vs  2 · kv_bytes_per_token / PCIe_BW
2 × 70e9 / 3.2e15 = 43.75 ns/token      (recompute)
2 × 327,680 / 64e9 = 10.24 µs/token     (swap)
```

Recompute is **234× cheaper per token** on this hardware. That is decisive, and it is why recompute is the default: modern GPUs have absurd FLOPs relative to PCIe bandwidth. Swap only wins when you are FLOPs-starved (small batch headroom, a model so large that prefill is genuinely slow) or when you want to preserve *exact* KV state rather than recompute it — which matters if any part of your pipeline is nondeterministic.

The operational consequences that actually reach users: a preempted request's TTFT is measured from the original arrival, so a preemption injects hundreds of milliseconds to seconds into a tail latency you already promised. And preemption is **not fair** by default — the engine typically preempts the most recently admitted or the largest sequence, which means long-context requests get starved under load. If your product has a long-context premium tier, you need explicit priority in the scheduler or you will systematically punish your highest-value users.

**⚠ Trap:** treating preemption as free because throughput dashboards look fine. Aggregate tokens/s can stay flat while P99 TTFT triples. Always alert on preemption count directly.

### Write me a grouped-query attention forward pass with a working KV cache. From memory, no references. Then I'll ask three follow-ups.

```python
import torch, torch.nn as nn

class GQAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.nh, self.nkv = n_heads, n_kv_heads
        self.dh  = d_model // n_heads
        self.rep = n_heads // n_kv_heads
        self.wq = nn.Linear(d_model, n_heads    * self.dh, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(n_heads * self.dh, d_model,    bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh,  self.dh).transpose(1, 2)   # [B,nh,T,dh]
        k = self.wk(x).view(B, T, self.nkv, self.dh).transpose(1, 2)   # [B,nkv,T,dh]
        v = self.wv(x).view(B, T, self.nkv, self.dh).transpose(1, 2)

        if cache is not None and cache[0] is not None:                 # append, don't overwrite
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        S = k.shape[2]                                                 # total keys so far
        qpos = torch.arange(S - T, S, device=x.device).unsqueeze(1)    # [T,1] absolute q positions
        kpos = torch.arange(S,     device=x.device).unsqueeze(0)       # [1,S]
        mask = kpos <= qpos                                            # [T,S] causal, cache-aware

        kx = k.repeat_interleave(self.rep, dim=1)                      # [B,nh,S,dh]
        vx = v.repeat_interleave(self.rep, dim=1)
        att = (q @ kx.transpose(-2, -1)) * self.dh ** -0.5             # [B,nh,T,S]
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        out = (att @ vx).transpose(1, 2).reshape(B, T, self.nh * self.dh)
        return self.wo(out), new_cache
```

The two lines that decide whether this is correct are the mask offset and the `cat` order. `qpos` must run from `S − T` to `S − 1`, not `0` to `T − 1` — during decode with a warm cache, T = 1 and S = 4097, so the single query sits at absolute position 4096 and must see all 4097 keys. If you build the mask as `torch.tril(torch.ones(T, S))` you get a `[1, 4097]` lower-triangular row that allows exactly one key. Prefill will be perfect and decode will be silently, catastrophically wrong. **That is the off-by-one that kills more from-scratch implementations than anything else**, and it passes any test that only checks prefill.

**Follow-up 1 — what would you change for production?** Replace the manual scores with `F.scaled_dot_product_attention(q, k, v, is_causal=..., enable_gqa=True)` so the backend broadcasts K/V internally rather than materializing `repeat_interleave`, which writes an `nh/nkv`-times-larger tensor to HBM and throws away the bandwidth win. And replace `torch.cat` with writes into a preallocated buffer indexed by `cache_position` — `cat` reallocates and copies the entire cache every single decode step, turning an O(1) append into O(T) and making generation quadratic in wall-clock.

**Follow-up 2 — memory at 128k?** With 32 layers, `n_kv = 8`, `dh = 128`, bf16: `2 × 32 × 8 × 128 × 2 = 131,072 B = 128 KiB/token`; × 131,072 tokens = **16 GiB** for one sequence.

**Follow-up 3 — what changes at batch 1?** Nothing in the code; everything in the performance regime. Batch 1 decode has arithmetic intensity ~1 against a machine balance of ~295 FLOP/byte on H100, so you are ~300× off the compute roofline and your token rate is entirely determined by `weight_bytes / HBM_bandwidth`. GQA's benefit at batch 1 is smaller than at batch 64, because at batch 1 the weight read dominates the KV read; GQA's value grows with batch and with context.

**🏋 Drill:** 20 minutes, no references, no autocomplete. Write the class above, then write a test that (a) prefills 16 tokens with no cache, (b) decodes 16 more one at a time using the cache, (c) prefills all 32 at once, and asserts the final hidden states match to `atol=1e-4`. **Pass criterion: the test passes on the first run after you finish writing it.** If it fails, the failure is your mask offset with probability ~0.8. Repeat until you can do it cold in under 20 minutes.

### Here's a budget: 40 GB of KV pool, you must hold 64 concurrent sessions at 32k context, 32 layers, d_head 128. Choose n_kv_heads and defend it.

Work backwards from the constraint. This is exactly the arithmetic I want a senior candidate to do without hesitating.

```
Per-sequence budget  : 40e9 / 64                     = 625 MB
Per-token budget     : 625e6 / 32,768                = 19,073 bytes ≈ 18.6 KiB
bytes/token formula  : 2 · 32 · n_kv · 128 · b       = 16,384 · n_kv · (b/2)
```

At bf16 (b = 2): `16,384 · n_kv ≤ 19,073` → **n_kv ≤ 1.16**, so the only bf16 answer is **MQA (n_kv = 1)**: 16 KiB/token → 512 MiB/sequence → 64 × 0.537 GB = **34.4 GB**. Fits with 5.6 GB spare. n_kv = 2 at bf16 gives 32 KiB/token → 68.7 GB, which is 72% over budget.

But MQA is the wrong answer, and the point of this drill is knowing why. MQA at TP > 1 replicates, MQA has the worst quality profile of any variant, and MQA is the most fragile under any subsequent quantization. So I would not stop at "n_kv = 1 fits."

**My answer: `n_kv_heads = 2` with fp8 KV cache.** At b = 1: `2 × 32 × 2 × 128 × 1 = 16,384 B = 16 KiB/token` → identical 34.4 GB total. Same bytes as MQA at bf16, **but two independent key subspaces instead of one**. Per the earlier argument, a trained compression (fewer heads) plus a modest post-hoc one (fp8) beats an extreme trained compression alone, and two KV heads shard cleanly at TP=2.

If I can also change the attention pattern, I would go further. **`n_kv_heads = 4`, interleaved 1 global : 3 local with a 4096-token window, fp8 KV:**

```
effective tokens/seq = 0.25 × 32,768 + 0.75 × 4,096 = 8,192 + 3,072 = 11,264
bytes/token (n_kv=4, fp8) = 2 × 32 × 4 × 128 × 1 = 32,768 B = 32 KiB
per sequence = 32 KiB × 11,264 = 352 MiB = 0.369 GB
64 sessions  = 23.6 GB                        ← 41% under budget
```

Four KV heads, headroom for traffic growth, and a real long-range path through the global layers. The cost is that the local layers can only retrieve within 4k directly, so I would gate this on a multi-fact retrieval eval at 32k before committing.

**How I defend the choice in the room:** state the budget arithmetic first, name the naive answer (MQA at bf16), reject it on quality-and-sharding grounds, then present the equal-bytes alternative with more head diversity, then present the pattern-level option with the headroom. **Show that you considered three axes — head count, precision, attention pattern — and picked a point on each rather than maximizing one.**

**🏋 Drill:** 10 minutes, whiteboard only, no calculator. Given `(pool_GB, concurrency, context, n_layers, d_head)`, produce a defensible `(n_kv_heads, kv_dtype, attention_pattern)` triple with the arithmetic written out. **Pass criterion: your bytes-per-token figure is within 5% of the exact value and you name at least one alternative you rejected and why.** Run it on three random parameter sets: (40 GB, 64, 32k, 32, 128); (200 GB, 256, 8k, 48, 128); (416 GB, 40, 128k, 80, 128). The third one has no bf16 solution at any head count — noticing that is the point.

**⚠ Trap:** solving this by cutting `max_model_len` to 8k and declaring victory. That changes the product requirement rather than meeting it. If you genuinely believe 32k is over-specified, say so explicitly and bring evidence from the traffic distribution — but do not silently redefine the problem.

### Does MoE sparsity help your KV cache? And describe the memory profile of an MoE request over its lifetime.

Sparsity does **nothing** for the KV cache, and this is one of the cleanest "did you actually serve this" questions available.

Routing happens in the FFN. The attention block — Q/K/V projections, the attention itself, the output projection — is **dense and shared by all tokens**. Every token computes keys and values at every layer regardless of which experts it routes to. So an MoE's KV cache formula is identical to a dense model's with the same layer count and head geometry. DeepSeek-V3's small cache comes entirely from MLA, not at all from MoE.

The memory profile over a request's lifetime is the interesting part, and it has a crossover:

**At t = 0 (weights loaded, no traffic):** expert weights dominate completely. DeepSeek-V3 at fp8 is ~671 GB of weights, essentially all of it FFN experts. KV is zero. This is why MoE deployments are described as "memory-hungry" — the *static* footprint is enormous relative to the 37B activated parameters.

**As context and concurrency grow:** KV grows linearly in `tokens_in_flight`. The crossover point where KV equals weights:

```
671e9 bytes / 70,272 bytes-per-token = 9.55e6 tokens in flight
at 32k context: 9.55e6 / 32,768 = 291 concurrent sequences
```

So on a hypothetical machine with unbounded memory, KV would overtake weights at 291 concurrent 32k sessions. In practice you run out of HBM long before that — an 8×H200 node has 1128 GB and 671 GB is weights, so you reach ~138 sessions and stop. **For DeepSeek-V3, weights dominate throughout the practical operating range.**

Contrast Llama-3-70B: `140e9 / 327,680 = 427,246` tokens in flight, which at 8k context is only **52 concurrent sessions**. Past 52 users, the KV cache is the larger consumer. That is a completely different regime, and the practical implication is that KV optimizations (fp8 KV, prefix caching, shorter contexts) pay off much more on the dense model, while for the MoE the leverage is in weight quantization and expert placement.

**⚠ Trap:** the "37B activated so it serves like a 37B model" claim. Activated parameter count predicts *FLOPs per token*, and therefore roughly predicts compute at high batch. It does not predict memory residency — you must hold all 671B of experts in HBM (or pay a catastrophic fetch cost) because you cannot know which experts the next token needs. And it does not predict throughput at low batch, where you are bandwidth-bound and expert-parallel all-to-all traffic dominates. **A 671B/37B MoE needs roughly 671B worth of GPUs and delivers roughly 37B worth of FLOPs per token — the cost model is the worse of the two, not the better.**

**🗣 Say this in the room:** "MoE sparsity is in the FFN; attention is dense, so KV cache is unaffected — DeepSeek-V3's small cache is MLA, not MoE. And activated parameters set FLOPs, not residency: you still hold all 671B in HBM. For that model, weights dominate memory until roughly 290 concurrent 32k sessions, which you'll never reach because you run out of HBM at ~138."

### How does chunked prefill change KV write patterns, and what does it trade?

Mental model: chunked prefill is **cooperative multitasking for a GPU that would otherwise run a long uninterruptible job**. A 32k-token prefill is a single kernel sequence that monopolizes the device; every decoding user gets zero tokens for its duration. Chunking splits it into pieces that interleave with decode steps in the same batch.

The arithmetic that makes the case: a 32k prefill on a 70B model is `2 × 70e9 × 32,768 = 4.59e15 FLOP = 4.59 PFLOP`, which at 3.2 PFLOP/s of node throughput is **1.43 seconds**. During those 1.43 s, every one of your 37 other users sees an inter-token latency spike of 1.43 s. At a normal ITL of ~25 ms, that is a **57× spike** — and it will show up as a jarring stall in a streaming UI. Chunk it at 2,048 tokens per step: 16 chunks, each ~90 ms, interleaved with decode steps. Now the worst ITL spike is ~90 ms, a 3.6× blip nobody notices. The prefilling request's own TTFT rises from 1.43 s to maybe 1.7–1.9 s because it now shares batch slots with decode work.

**That is the trade, stated cleanly: chunked prefill converts a large ITL spike for many users into a modest TTFT increase for one user.** For any interactive product that is obviously the right direction, which is why modern engines enable it by default.

The KV write pattern changes in a specific way that matters for correctness. Without chunking, the prefill kernel computes K/V for all T tokens and writes them once. With chunking, chunk `i` computes K/V for its own tokens **and must attend to all previously written chunks `0..i−1`**. So each chunk is a mixed-length attention: query length 2,048, key length up to 32,768 and growing. This requires a kernel that handles arbitrary query-length-versus-key-length combinations with correct causal masking across the chunk boundary — the same varlen machinery that handles decode, generalized. The classic bug here is an off-by-one at the chunk boundary: the first token of chunk `i` must see the last token of chunk `i−1`, and a mask built with local rather than absolute positions gets this wrong. Symptom: quality that degrades only for prompts longer than one chunk, and only at the seams. It is a nasty one to find.

Tuning: `max_num_batched_tokens` sets the chunk size and is the single knob. Small chunks → smoother ITL, worse prefill efficiency (each chunk re-reads the model weights, so at 512 tokens per chunk you are paying weight-read cost 64 times for a 32k prompt instead of 16). Large chunks → better prefill throughput, worse ITL. I start at 2,048–4,096 for interactive workloads and go higher for batch/offline.

**⚠ Trap:** enabling chunked prefill and then measuring only TTFT. TTFT gets slightly *worse*; the win is entirely in ITL P99 and in aggregate throughput. If your dashboard only tracks TTFT you will conclude chunked prefill is a regression and turn it off.

### Explain the difference between what n_kv_heads costs you at training time versus at serving time.

They are almost entirely different cost structures, and conflating them is why architecture and serving teams talk past each other.

**At training time**, `n_kv_heads` barely matters. Training is one giant prefill over packed sequences: batch dimension is large, sequence length is fixed, there is no KV cache at all (every position's K and V are computed fresh in the forward pass and discarded after the backward). The only training-time effects of reducing KV heads are (a) slightly fewer parameters in `W_K`/`W_V`, freeing budget for FFN width, (b) slightly fewer FLOPs in the K/V projections — on the order of a few percent of the block, and (c) slightly less activation memory. Training a GQA-8 model versus an MHA model is maybe 2–5% cheaper. **Nobody chooses GQA to save training compute.**

**At serving time**, `n_kv_heads` is the dominant term in the equation that determines your unit economics. It sets bytes per token linearly, which sets concurrency at fixed HBM linearly, which sets cost per token roughly inversely. Eight versus sixty-four KV heads is an 8× difference in the memory resource that gates everything.

This asymmetry has a specific organizational consequence I would raise in a design review: **the team that pays for `n_kv_heads` is not the team that chooses it.** A research team optimizing loss-per-FLOP has essentially no gradient signal pushing toward fewer KV heads — MHA is marginally better on quality-per-training-FLOP. The pressure comes entirely from the serving side, and it only shows up in a metric (cost per served token, or maximum context at target concurrency) that the training team is not measured on. Every organization that has shipped a good GQA/MLA model has, structurally, had serving requirements written into the architecture spec before pretraining started. If you are ever in the room when a model architecture is being decided, the question to ask is: *"what is our KV bytes per token, and what concurrency does that give us at our target context on our actual fleet?"* If nobody can answer, the architecture is not finished.

The quality side is the mirror image: reducing KV heads costs you a small amount of quality that is **paid at training time and cannot be recovered at serving time**. Whereas quantization, eviction and windowing are serving-time decisions you can tune, revert or make per-tenant. That asymmetry — trained decisions are permanent, serving decisions are reversible — is why I want the trained compression to be conservative (GQA-8, not GQA-2) and the serving compression to carry the tuning burden.

### How would you measure whether dropping from GQA-8 to GQA-4 hurt your product, as opposed to your benchmark?

Benchmarks answer "did the model get worse in general." Products need "did the thing users actually do get worse." Those diverge, and the honest measurement plan has four layers.

**Layer 1 — a task-grounded offline eval built from production traces.** Sample real requests stratified by context-length decile and by task type (summarize / extract / answer-from-document / code-edit / multi-turn). For each, define a checkable outcome: exact-match on an extracted field, a rubric-scored judgment, a unit test passing, a citation pointing at the correct span. Run both models on **the same inputs, paired**. Pairing matters enormously: paired designs remove between-item variance and cut the sample size you need by a large factor.

**Layer 2 — a targeted probe for the capability GQA reduction actually damages.** The mechanism says head-count reduction hurts precise in-context retrieval at long input. So build a multi-fact retrieval eval on your own corpus: place 5 facts at controlled depths in real documents at 4k, 16k, 32k and 64k, and ask questions requiring 2 of them jointly. Report per-length accuracy, not an average. **If GQA-4 costs you nothing at 8k and 6 points at 32k, that is not "a 2-point average regression," it is a decision about which contexts you serve.**

**Layer 3 — power arithmetic, before you run anything.** For a two-sided test at α = 0.05 with 80% power, the standard approximation is `n ≈ 16 σ² / δ²` per arm. To detect a 2-point drop from an 85% baseline: `σ² = p(1−p) = 0.85 × 0.15 = 0.1275`, `δ = 0.02`, so `n ≈ 16 × 0.1275 / 0.0004 = 5,100` items per arm. Paired testing (McNemar on discordant pairs, or a paired bootstrap) typically cuts this substantially, but the order of magnitude stands: **you need thousands of labelled items to detect a 2-point regression, and any eval set of 200 examples cannot detect it at all.** State this before someone shows you a 200-item eval and declares parity.

**Layer 4 — online, because offline evals miss what users do.** Shadow the new model on a traffic slice and compare behavioral proxies that need no labels: regeneration rate, edit-after-accept rate (for code), conversation length before abandonment, explicit thumbs-down rate, and escalation-to-human rate for support. These are noisier per-sample but you have far more samples, and they measure the thing you are actually optimizing. Run the A/B long enough to cover a full weekly cycle.

**⚠ Trap:** running the comparison at temperature > 0 without controlling seeds or sample counts and attributing sampling noise to the architecture change. Either compare at temperature 0 for determinism, or sample k ≥ 5 per item per arm and compare means with proper variance accounting. I have watched a team ship a bad architecture change because their "no regression" result was one noisy sample per item.

**🗣 Say this in the room:** "I'd pair the comparison on real production traces, segment by context-length decile because that's where the damage concentrates, and do the power calculation up front — roughly 5,000 items per arm to detect two points at an 85% baseline, so a 200-item eval tells us nothing. Then shadow online and watch regeneration rate."

### Leadership wants to go from GQA-8 to GQA-2 to quadruple our context window. Talk me into it or out of it.

Out of it, and the strongest argument has nothing to do with quality.

**The killer: at TP=8, GQA-2 saves you exactly zero KV memory.** With 8 KV heads and TP=8, each rank stores one KV head and the node stores 8 heads' worth — the true cache, perfectly sharded. With 2 KV heads and TP=8, `n_kv_heads < tp_size`, so the framework replicates: each of the 8 ranks stores a full copy of one of the 2 heads, meaning each head is replicated 4×. Cluster-wide storage is 8 heads' worth again. **Identical KV memory, worse quality, and you paid ~5% of pretraining compute to get there.** If the deployment is TP=8 — which it is for any 70B-class model on a standard node — this proposal is strictly negative.

To capture GQA-2's benefit you would have to drop to TP=2 and use pipeline parallelism across the node instead. That is a serious change: pipeline parallelism introduces bubbles, complicates continuous batching (microbatch scheduling interacts badly with variable-length requests), and generally lowers per-request latency quality. You would be re-architecting the serving stack to unlock a memory saving you can get more cheaply elsewhere.

**Second argument: the framing is wrong.** GQA-8 → GQA-2 is a 4× cache reduction, which buys 4× context *or* 4× concurrency, not "quadruple context" for free. If you quadruple context at constant concurrency, aggregate KV is unchanged, but **decode bandwidth per token rises 4× per sequence** because each step now streams 4× more cache. Your tokens/sec per user will drop even though memory fits. Context length is not free once it fits; it is bandwidth.

**Third: the cheaper alternatives, ranked.**
- **fp8 KV cache: 2× reduction, no retraining, no architecture change, reversible with a flag, and a known validation procedure.** Start here.
- **Prefix caching plus prefix-affine routing**, if not already maximized — often larger than 2× in effective capacity on prompt-heavy workloads.
- **Interleaved local/global attention** via continued pretraining: a 1:5 pattern gives ~5.8× at 128k, more than GQA-2's 4×, and preserves all 8 KV heads' quality on the global layers. Similar uptraining cost, better outcome.
- **A different checkpoint.** If long context is the product requirement, an MLA-based or natively-long-context model probably beats surgery on this one.

**When would I be talked *into* it?** If we are training a new model from scratch anyway (so the 5% uptraining cost is zero), if the deployment is single-GPU or TP=2 (so no replication), if the workload is verified to be short-input generation-heavy rather than long-input retrieval-heavy, and if we have a long-context retrieval eval showing GQA-2 holds up on our corpus. That is four conditions, and I would want all four.

**🗣 Say this in the room:** "At TP=8, GQA-2 gives us zero KV savings — two heads replicated across eight ranks stores the same bytes as eight heads sharded. So the proposal costs quality and 5% of pretraining compute to buy nothing, unless we also move to TP=2 with pipeline parallelism. I'd get the 2× from fp8 KV this week instead, and if we need more than that, interleaved local/global gives us ~5.8× at 128k for the same uptraining budget."

### Give me the sixty-second version of the whole MHA-to-MLA arc, the way you'd tell it to a staff engineer who hasn't followed it.

Vaswani's 2017 attention gives every head its own query, key and value projections. That is the most expressive arrangement and it was the right default when sequences were 512 tokens and nobody was serving the thing. The problem is that autoregressive decode must cache every past key and value at every layer, and that cache costs `2 · n_layers · n_kv_heads · d_head · bytes_per_element` per token. Under MHA `n_kv_heads · d_head = d_model`, so a 70B-class model costs about 2.5 MiB per token — 20 GiB for a single 8k conversation. Decode is bandwidth-bound, that cache is streamed every step, and it does not amortize across the batch. The cache, not the FLOPs, became the binding constraint.

**Shazeer 2019 (MQA)** noticed queries are never cached, so query heads and KV heads need not match: keep all N query heads, share **one** key and value head. Cache drops by N. Quality drops too, concentrated in long-input retrieval, because all heads are forced into one key subspace — and at tensor parallelism the single head must be replicated, eating much of the win.

**Ainslie 2023 (GQA)** interpolated: G groups of query heads, one KV head per group. Eight groups recovers nearly all of MHA's quality at one-eighth the cache, and — the part that made it stick — eight KV heads shard perfectly onto an eight-GPU node. Crucially they showed you can convert an existing MHA checkpoint by mean-pooling KV heads within groups and uptraining for about 5% of the original pretraining compute. That made GQA nearly free to adopt, and it became the industry default: Llama-3, Mistral, Qwen.

**DeepSeek 2024 (MLA)** changed the axis. Instead of deleting heads, compress the *rank*: cache a 512-dimensional latent per token per layer and reconstruct per-head K and V from it with per-head up-projections, so heads keep independent subspaces. RoPE blocks the matrix absorption that makes decode cheap, so they split off a small 64-dim shared rotary key. Net: 576 elements per token per layer — about 3.6× smaller than GQA-8 and 57× smaller than MHA, at better quality. It needs bespoke kernels, which is why it hasn't spread beyond DeepSeek's own family yet.

Running alongside all of that, three orthogonal levers that multiply with the head-count story: **sliding window** (bound the number of cached tokens), **interleaved local/global layers** (pay full cache on only a fraction of layers), and **cross-layer KV sharing** (share one layer's cache with its neighbors). Stack three of them at 3–4× each and you get the order-of-magnitude reductions production systems actually run.

**🗣 Say this in the room, verbatim:** "The whole arc is one equation: KV bytes per token equals two, times layers, times KV heads, times head dim, times bytes per element. MQA attacked the KV-head term hard and lost quality. GQA attacked it moderately and won, because eight KV heads also happens to shard onto eight GPUs. MLA attacked the head-dim term instead by caching a low-rank latent, which is better but needs custom kernels. And sliding window, local/global interleaving and cross-layer sharing attack the layers and tokens terms independently, so they multiply. Every architecture decision since 2019 is a choice about which term in that equation to shrink and what you're willing to pay for it."
