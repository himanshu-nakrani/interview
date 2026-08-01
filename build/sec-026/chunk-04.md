### For a long-running agent, why is the cache and not the weights your capacity limit? Work the numbers.

Because an agent's context is monotonically increasing and never resets, while the weights are a fixed cost you paid once at process start. This inverts the intuition every backend engineer brings from stateless services, and it is the single most important thing to understand about serving agents.

Take a realistic coding or research agent: 50 turns, each turn appending roughly 2,000 tokens of tool output, retrieved file contents, and the model's own reasoning. By turn 50 the context is **100,000 tokens**. On a 70B at 320 KiB/token:

$$100{,}000 \times 327{,}680 = 3.28\times10^{10}\text{ B} = 30.5\text{ GiB — for one session}$$

Against the 200.4 GiB KV pool on a 4×H100 node: **six concurrent agents.** The weights are 65.2 GiB and serve all six. You are running a $250k node for six users, and five-sixths of your HBM is holding conversation history.

Now the growth pattern, which is what makes agents distinctive. The context grows linearly across turns, so the *integral* — the resource you actually consume — is quadratic in turn count. Averaged over the session, the agent holds ~50k tokens = 15.3 GiB. If a turn takes 30 seconds (tool call, network, generation), the session runs 25 minutes and consumes $50{,}000 \times 0.4167 = 20{,}833$ **token-hours**.

The second-order effect is prefill, and it is larger than most people expect. Without prefix caching, turn $n$ re-prefills the entire context from scratch. Summed over 50 turns:

$$2N \times 2{,}000 \times \frac{50 \times 51}{2} = 2\times70\times10^9 \times 2{,}000 \times 1{,}275 = 3.57\times10^{17}\text{ FLOP}$$

At $3.17\times10^{15}$ FLOP/s (4×H100, FP8, 40% MFU) that is **112.6 seconds of the entire node, per agent session.** With prefix caching, only the 2,000 new tokens per turn are prefilled: $2N \times 2{,}000 \times 50 = 1.4\times10^{16}$ FLOP = **4.4 seconds**. A 25× reduction.

**💰 Math:** at $2.50/GPU-hour, 112.6 node-seconds × 4 GPUs = 450 GPU-s = 0.125 GPU-hr = **$0.313 per session** without prefix caching, versus $0.0122 with. At 10,000 agent sessions/day that is $3,130/day versus $122/day — **$90k/month** of difference from one boolean config flag. This is why prefix caching is not an optimisation for agents; it is a precondition.

**🗣 Say this in the room:** "For chat, the weights dominate memory and the cache is small. For agents it's the reverse — a 50-turn agent is 100k tokens, 30 GiB of KV on a 70B, so a 4×H100 node holds six of them. And because context grows every turn, prefill cost is quadratic in turn count unless prefix caching is on. The capacity unit for an agent platform is token-hours held, not requests per second."

### How would you bill the KV cache in a multi-tenant platform? What's actually fair?

The framing that resolves this: **the KV cache is rented memory with a time dimension, and every other resource on your platform is billed per unit of work.** Tokens generated is a work metric. Cache held is a *rent* metric. Billing rent as if it were work is why so many platform cost models blow up.

Derive the unit price from the hardware. A 4×H100 node at $2.50/GPU-hour is $10/hour. The KV pool is 200.4 GiB out of roughly 286 GiB of total budgeted memory across the node, so ~70% of the node's memory value is the pool; attribute $7/hour to it:

$$\frac{\$7/\text{hr}}{200.4\text{ GiB}} = \$0.035\text{ per GiB-hour}$$

Convert to the unit users understand. One GiB holds $1{,}073{,}741{,}824 / 327{,}680 = 3{,}277$ tokens, so:

$$\frac{\$0.035}{3{,}277} = \$1.07\times10^{-5}\text{ per token-hour} = \mathbf{\$10.68\ per\ million\ token\text{-}hours}$$

That is the number I would put on an internal chargeback dashboard. It makes the agent example land instantly: a 25-minute session holding an average of 50k tokens costs $0.22 in cache rent alone, before a single generated token is billed. At 10,000 sessions/day that is $2,220/day = **$66.6k/month of pure holding cost**, and it appears on no per-token invoice.

The public providers have already converged on this shape, which is useful validation. Anthropic prices cache writes at a premium over base input (higher for the longer TTL) and cache reads at a steep discount — you are paying to *populate* a cache and paying rent implicitly through the write premium and TTL choice. Google's explicit context caching charges storage per token per hour outright, which is the purest form of the model. (**📅 Volatile:** all specific multipliers and storage rates change; verify against current pricing before quoting.)

The design consequence I'd argue for internally: **bill agents on token-hours, chat on tokens.** A chat turn holds its cache for two seconds; rent is noise and per-token billing is honest. An agent session holds 30 GiB for 25 minutes; per-token billing lets a customer pin a third of your node for the price of a few thousand output tokens. Every platform that has shipped agents has discovered this, usually via a customer who ran 200 concurrent sessions and paid $40.

**⚠ Trap:** billing cache reads as if they were free because "it's cached." Reading a 100k-token cache during decode costs real HBM bandwidth on every single token — from the earlier arithmetic, at 1M context the KV read is 3× the weight read. A cached prefix is cheap to *establish* and not free to *use*. Pricing that ignores this loses money precisely on your largest customers.

### What are the isolation risks when tenants share a KV cache?

Three surfaces, and the first one is a genuine security issue that gets missed in design review because "it's a cache."

**Timing side channel through the shared prefix cache.** If tenant A's proprietary system prompt is in the shared prefix cache, tenant B can send a candidate prefix and measure TTFT. A cache hit is dramatically faster than a cold prefill — for a 12k-token system prompt on a 70B, prefill is ~530 ms of node time versus a cache hit of a few milliseconds, a signal you can detect over noise with a handful of samples. Extend the guess block by block and you can extract another tenant's prompt token-block by token-block. This is a real, published class of attack against shared prefix caching.

The mitigation is namespace partitioning: incorporate the tenant identifier into the block hash chain so tenant A's blocks are unreachable by tenant B's requests. Some engines expose a per-request cache salt for exactly this (**📅 Volatile:** verify the feature and its name in your engine version). The cost is your hit rate for genuinely-shared content — a common system prompt across tenants is now cached $T$ times. My rule: **partition by default, and whitelist specific shared prefixes (your own system prompt, public documents) into a shared namespace explicitly.** Never the reverse.

**Noisy-neighbour capacity exhaustion.** One tenant opening ten 128k sessions consumes 400 GiB — more than the entire pool. Every other tenant's requests queue or get preempted. Default schedulers are tenant-blind: they preempt by arrival order or by some recency heuristic, not by who is over quota, so the heavy tenant's admitted work survives and the light tenants get evicted. The fix is a **per-tenant live-token quota enforced at admission**, plus weighted fair share on the pool. The quota is on tokens, not on requests, because requests are not the resource.

**Cross-tenant KV in a shared external store.** Once you tier KV to a fleet-wide store, you have a distributed cache holding verbatim customer content — prompts, retrieved documents, PII — in a system that was probably designed as a performance optimisation rather than a data store. It needs the same treatment as any other datastore: encryption at rest, per-tenant key scoping, TTL that satisfies your retention commitments, and a deletion path that actually reaches it when a customer exercises a right-to-erasure request. "We have a KV cache in Redis-shaped storage and nobody wrote a deletion path" is a compliance incident waiting to be found in an audit.

**🗣 Say this in the room:** "Shared prefix caching across tenants is a timing side channel — you can extract another tenant's system prompt by measuring TTFT on guessed prefixes. I partition the cache namespace by tenant by default and whitelist shared prefixes explicitly. And once KV is tiered to a fleet-wide store it's a datastore holding customer content, so it needs encryption, TTL and a deletion path, not just an eviction policy."

### Which signal do you autoscale on, and why is the obvious one wrong?

The obvious signal is QPS or CPU, and both are wrong for the same reason: **they do not measure the resource that runs out.** Two requests per second can mean 4,000 live tokens or 400,000 live tokens depending on context length, and only one of those saturates the node. Autoscaling an inference fleet on QPS is like autoscaling a database on connection count while ignoring row scans.

The correct primary signal is **KV pool utilisation**, with one crucial caveat: it saturates. Once you are at 100%, utilisation stops telling you *how* overloaded you are — 101% and 400% both read as 100%. So you need a pair:

**Utilisation** (`gpu_cache_usage_perc`-class) tells you *whether* you are full. Target 70–80% steady state on interactive traffic, leaving headroom for arrival burstiness.

**Queue depth and queue time** (`num_requests_waiting`, plus the queue-time component of your TTFT histogram) tells you *by how much* you are past full. This is the unbounded signal and it is what drives the scale-up magnitude.

Add a third, which I regard as the best leading indicator and which nobody instruments by default: **admitted-token rate versus pool capacity.** Sum the estimated total tokens (prompt + expected output) of everything admitted in the last window and divide by pool capacity. This rises *before* utilisation does, because a long request's tokens are committed at admission and materialise over the following seconds.

Then the part that actually determines whether autoscaling works: **lag.** A new GPU node is not a container start. You schedule a pod onto a GPU node (possibly waiting on a cloud provider), pull a multi-gigabyte image, load 65–140 GB of weights, initialise NCCL, capture CUDA graphs, and warm the prefix cache. Weight loading alone from object storage at 1 GB/s is 65–140 seconds; from local NVMe at 5 GB/s it is 13–28 seconds. End to end, **5 to 15 minutes** is typical.

**💰 Math:** with a 10-minute scale-up lag and traffic that can double in 10 minutes, reactive autoscaling is structurally too late — you must carry standing headroom. On a 10-node fleet, 30% headroom is 3 idle nodes × $10/hour = $30/hour = **$21.6k/month** of insurance. That number is the argument for the three things that reduce it: predictive scaling on a traffic forecast, keeping warm pool nodes with weights already loaded (paying storage instead of compute), and graceful degradation — routing overflow to a smaller model instead of queueing — which is almost always cheaper than the headroom it replaces.

**⚠ Trap:** scaling *down* on the same signal you scale up on. KV utilisation drops the instant traffic dips, and scaling down evicts every cached prefix on that node; when traffic returns 90 seconds later you eat both the scale-up lag and a cold prefix cache. Use a long scale-down cooldown — I use 15–30 minutes — and treat prefix-cache warmth as a resource you are destroying when you terminate a node.

### Design admission control for this system. Write me the policy.

The mental model: **your KV pool is a fixed token budget and admission control is a reservation system against it.** Every request that enters must reserve its worst-case footprint up front, because discovering mid-generation that the pool is full means preemption, and preemption is strictly more expensive than never having admitted the request.

```python
# Admission policy. Pseudocode over the engine's scheduler hooks — the exact
# integration surface differs per engine, but the decision logic is portable.

HIGH_WATER = 0.85          # never admit past this fraction of pool
POOL_TOKENS = 656_670      # measured from engine startup logs, not estimated

def admit(req, state) -> Decision:
    # 1. Reserve worst case, not observed case.
    reserve = req.prompt_tokens + req.max_tokens

    # 2. Hard structural limits first — cheap, deterministic rejections.
    if req.prompt_tokens + req.max_tokens > MAX_MODEL_LEN:
        return Reject(400, "context_length_exceeded")

    # 3. Per-tenant fairness, in tokens (not requests).
    t = state.tenant[req.tenant_id]
    if t.live_tokens + reserve > t.quota_tokens:
        return Reject(429, "tenant_kv_quota", retry_after=t.eta_free())

    # 4. Global capacity, with priority-aware preemption of batch work.
    if state.live_tokens + reserve > POOL_TOKENS * HIGH_WATER:
        if req.priority is INTERACTIVE and state.preemptible_batch_tokens >= reserve:
            state.preempt_batch(reserve)       # batch work is restartable
        else:
            eta = state.estimated_queue_time(reserve)
            if eta > req.slo_ttft:
                return Degrade(route_to="small-model-pool")  # shed, don't queue
            return Queue(eta)

    state.reserve(req.tenant_id, reserve)
    return Admit()
```

Four decisions in there I would defend line by line.

**Reserve on `prompt_tokens + max_tokens`, not on prompt length.** This is the one people get wrong. Admitting on prompt length means the pool fits at admission and overflows 400 tokens into generation, at which point you preempt someone. Reserving the worst case wastes capacity when clients set `max_tokens` far above their real output — so the accompanying policy is to enforce a sane default `max_tokens` and reject absurd ones, or to reserve on a p95 output-length estimate per route and accept a small preemption rate as the price of higher utilisation. Both are defensible; what is not defensible is reserving on prompt alone.

**Quota in tokens, not requests.** A tenant with a 500-request limit and 128k contexts can consume 64M tokens. Requests are not the resource.

**Shed rather than queue when the queue-time estimate blows the SLO.** A request that will miss its SLO anyway is pure waste — it consumes capacity to produce an answer nobody is still waiting for. Returning a fast 429, or degrading to a smaller model, is strictly better than delivering a correct answer 40 seconds late. This is the goodput-under-SLO framing and it is the single biggest lever on perceived reliability under load.

**Preempt batch, never preempt interactive.** Background agent work and offline batch jobs are restartable and have no human waiting; they exist precisely to be the shock absorber. Give them their own priority class and let interactive traffic evict them.

**⚠ Trap:** implementing admission control at the API gateway using request counts while the engine does its own scheduling on tokens. You now have two admission systems with different units disagreeing with each other, and the gateway lets through exactly the traffic the engine cannot serve. Admission control must be token-aware and must read the engine's actual pool state, or it is decoration.

### KV utilisation is sitting at 40% but requests are queueing. Diagnose it.

Low utilisation with a non-empty queue is diagnostic gold: it tells you immediately that **KV is not your bottleneck**, which eliminates the three fixes everyone reaches for first (quantise the cache, buy more GPUs, lower `max_model_len`). All three would be wasted effort. Work down this list.

**`max_num_seqs` is binding.** The scheduler has a hard cap on concurrent sequences independent of memory. If it's set to 64 and your average context is 4k, you top out at 256k tokens = 39% of a 656k pool, forever, with a queue behind it. This is the most common cause and it's a one-line fix. Signature: `num_requests_running` pinned at exactly `max_num_seqs`.

**`max_num_batched_tokens` is binding.** The per-iteration token budget caps how much prefill work can enter each step. With a workload of long prompts, you can be prefill-throughput-limited while the pool sits half empty — new requests cannot get their prefill scheduled fast enough to fill the cache. Signature: high time-in-queue for new requests, low ITL for running ones, iteration token counts pinned at the limit.

**You are compute-bound, not memory-bound.** If prompts are long and outputs are short, most node time goes to prefill FLOPs. The cache never fills because sequences complete quickly. Signature: high GPU SM utilisation, low cache utilisation, TTFT dominated by prefill time rather than queue time. The fix is more compute or chunked prefill tuning, not more memory.

**The bottleneck isn't the engine at all.** Gateway connection limits, an undersized retrieval service, a tokeniser running single-threaded in the request path, a rate limiter, TLS termination. I have seen a "GPU capacity problem" turn out to be a 100-connection pool in front of a reranker. Signature: queue depth measured at the gateway is high while the engine's own `num_requests_waiting` is near zero — which is why you must instrument both and never conflate them.

**Requests are queueing on something request-specific.** LoRA adapter swapping (a request needing an unloaded adapter waits for a slot), guided-decoding grammar compilation, or a structured-output schema being compiled per request rather than cached.

**🔍 Failure taxonomy — the two-signal decision procedure:** high cache utilisation + queue → genuinely KV-bound; quantise KV, reduce `max_model_len`, or add nodes. Low cache utilisation + queue + `num_requests_running` at its cap → a scheduler knob is binding, raise `max_num_seqs`. Low cache + queue + running below cap → the bottleneck is upstream or non-KV; go look at the gateway. High cache + *no* queue → healthy and efficient; raise limits and take the throughput.

### Preemption: would you configure swap or recompute, and why?

When the pool fills, the scheduler must free blocks, and it has two options for a victim sequence. **Recompute**: discard the KV entirely and re-prefill it when the sequence resumes. **Swap**: copy the blocks to host memory and copy them back on resume.

Run the arithmetic, because it decides the case. Swap pays the transfer *twice* — out and back. Recompute pays prefill once. Setting them equal gives a break-even bandwidth of $2b \cdot F / (2N)$ = twice the offload break-even, so **~14.8 GB/s** for a 70B. PCIe Gen5 at ~50 GB/s beats it by 3.4×; Gen4 at ~25 GB/s by 1.7×. On paper, swap wins.

In practice I configure **recompute**, and the reasons are all second-order effects that the bandwidth comparison misses.

**Recompute frees memory instantly; swap frees it only when the copy completes.** You preempt because you are under memory pressure *right now*. A mechanism whose relief arrives 100 ms later, after competing for the same PCIe lanes as everything else, is the wrong tool for a pressure spike.

**With prefix caching on, recompute is often nearly free.** The preempted sequence's prefix blocks may still be resident in the cache — you dropped the sequence, not necessarily its blocks. Recompute then reduces to re-linking a block table plus prefilling the divergent suffix. This is the decisive argument, and it is why modern engines default to recompute.

**Swap competes with the offload path.** If you are already tiering KV to host DRAM, swap traffic and offload traffic contend for the same lanes, and you have coupled two mechanisms that should fail independently.

**Swap adds a failure mode.** Host memory can fill; a swapped sequence can be stranded; you need eviction policy on the swap space itself. Recompute has no state to lose.

**⚠ Trap:** treating preemption rate as a tuning parameter rather than an alarm. Some preemption under burst is normal. A *sustained* nonzero preemption rate is a positive feedback loop: preempted sequences re-enter and re-prefill, prefill work competes with decode for iteration slots, ITL rises, sequences live longer, the pool stays fuller, more preemption. This does not equilibrate gently — it collapses. Alert on the derivative of the preemption counter, not on its absolute value, and treat sustained preemption as "the admission controller is misconfigured," not "the GPU is busy."

### Post-mortem time. p99 TTFT tripled after a deploy that only changed `max_num_seqs` from 128 to 256. Walk me through it.

This is a preemption cascade, and it is the most instructive KV incident because the change looks obviously safe.

**The arithmetic that was skipped.** Pool capacity is 656,670 tokens. Average context in this workload is 4,000 tokens. At `max_num_seqs = 128`: $128 \times 4{,}000 = 512{,}000$ tokens = **78% of pool** — comfortably inside the safe band. At 256: $256 \times 4{,}000 = 1{,}024{,}000$ tokens = **156% of pool.** The scheduler will happily admit sequences up to its sequence cap; the memory cap is enforced by *preemption*, not by refusing admission. So the new limit does not produce 2× concurrency — it produces sustained preemption.

**The cascade.** The pool fills. The scheduler preempts a victim, which re-enters the waiting queue and must re-prefill 4,000 tokens: $2 \times 70\times10^9 \times 4{,}000 = 5.6\times10^{14}$ FLOP = 177 ms of the whole node at $3.17\times10^{15}$ FLOP/s. At a preemption rate of 20/s, that is $20 \times 0.177 = 3.5$ node-seconds of redundant prefill demanded per wall-clock second — **3.5× the node's entire capacity, spent on work that was already done.** That prefill competes for iteration slots with both decode and with new arrivals' prefill, so new requests wait longer for their first token. p99 TTFT triples. Meanwhile ITL rises too (more sequences per step means more KV bytes read per step), so sequences live longer, so the pool stays fuller, so preemption continues. It is a self-reinforcing loop and it will not recover until arrival rate drops.

**What the dashboard showed and what it should have shown.** GPU utilisation looked *great* — near 100%, because the node was extremely busy doing redundant prefill. Throughput in completed-requests-per-second was down. The signal that named the cause in one glance is the preemption counter's derivative, which went from zero to 20/s at the exact deploy timestamp. If preemption rate is not on your primary dashboard, this incident takes hours instead of minutes.

**The fix, and the durable rule.** Roll back, then set the knob from the arithmetic rather than by intuition:

$$\texttt{max\_num\_seqs} \leq \frac{0.85 \times \text{pool tokens}}{p95\ \text{context}} = \frac{0.85 \times 656{,}670}{4{,}000} = 139$$

So 128 was right and 256 was never going to work. Note it uses **p95** context, not mean — the pool is consumed by the tail. I encode this as a startup assertion in the deployment: read the engine's reported block count, read the configured `max_num_seqs`, compare against the p95 context from last week's traffic, and refuse to boot if the product exceeds 85% of capacity. A config change that violates a capacity invariant should fail at deploy, not at 3am.

**🗣 Say this in the room:** "`max_num_seqs` is a memory commitment in disguise. The scheduler enforces it by preempting, not by declining, so setting it above what the pool can hold buys you a preemption cascade, not concurrency. The rule is `max_num_seqs ≤ 0.85 × pool_tokens / p95_context`, and I'd assert it at boot."

### Give me the eight scenarios: 7B, 70B and an MoE, at 8k, 128k and 1M, at batch 1 and batch 64. Full arithmetic.

**🏋 Drill:** unaided, on paper, in 12 minutes. Pass criterion: every cell within 5% of the values below, and a correct fit/no-fit verdict against named hardware for each. Do not proceed to the rest of this section until you can do this cold — it is, verbatim, the most frequently asked quantitative question in AI-engineering loops.

**Step 1 — per-token cost.** Same formula, three configs, bf16 KV:

- **Llama-3.1-8B**: 32 layers, 8 KV heads, `d_head` 128 → $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B = **128 KiB/token**
- **Llama-3.3-70B**: 80 layers, 8 KV heads, 128 → $2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ B = **320 KiB/token**
- **Mixtral 8×7B (MoE, 46.7B total / ~12.9B active)**: 32 layers, 8 KV heads, 128 → **128 KiB/token, identical to the 8B.** MoE does not touch the cache.

**Step 2 — the grid.** KV bytes = tokens × batch × per-token:

| | 8k, b=1 | 8k, b=64 | 128k, b=1 | 128k, b=64 | 1M, b=1 | 1M, b=64 |
|---|---|---|---|---|---|---|
| **8B** | 1.0 GiB | 64 GiB | 16 GiB | 1,024 GiB | 128 GiB | 8 TiB |
| **70B** | 2.5 GiB | 160 GiB | 40 GiB | 2,560 GiB | 320 GiB | 20 TiB |
| **MoE 8×7B** | 1.0 GiB | 64 GiB | 16 GiB | 1,024 GiB | 128 GiB | 8 TiB |

(8k = 8,192; 128k = 131,072; 1M = 1,048,576. Note $8{,}192 \times 131{,}072 = 2^{30}$ exactly, which is why the 8B/8k/b=1 cell is precisely 1 GiB — a useful anchor to rebuild the whole table from.)

**Step 3 — fit verdicts against real pools.**

*8B, FP8 weights, 1×H100.* Weights $8\times10^9$ B = 7.5 GiB. Pool $= 71.6 - 7.5 - 1.2 - 3 = 59.9$ GiB → 491k tokens. **8k b=64 (64 GiB): does not fit** — max batch is 59 at 8k. **128k b=1 (16 GiB): fits**, max batch 3. **1M b=1 (128 GiB): does not fit**; FP8 KV halves it to 64 GiB — still no; INT4 KV → 32 GiB — fits, one user, using 53% of the card's pool.

*70B, FP8 weights, 4×H100.* Pool 200.4 GiB → 656k tokens. **8k b=64 (160 GiB): fits at 80% occupancy** — tight, expect preemption on any burst. **128k b=1 (40 GiB): fits**, max batch 5. **1M b=1 (320 GiB): does not fit**; FP8 KV → 160 GiB, fits at 80% of the pool, serving exactly one user on four H100s. **b=64 at anything above 10k context: does not fit.**

*Mixtral 8×7B, FP8 weights, 2×H100.* Weights 46.7 GB = 43.5 GiB, 21.8/card. Pool $= 2 \times (71.6 - 21.8 - 1.2 - 3) = 91.2$ GiB → 748k tokens. **8k b=64 (64 GiB): fits at 70%.** **128k b=1 (16 GiB): fits**, max batch 5. Per GPU that is 46 concurrent 8k sessions against the 8B's 59 — the MoE's weights ate the pool.

**Step 4 — the three sentences you say out loud after building this.** First: the 1M column is a different product, not a longer version of the same one — every 1M cell either doesn't fit or consumes an entire node for one user. Second: batch 64 and 128k are mutually exclusive on any hardware you can afford; at 2,560 GiB you are asking for 32 H100s of pure cache. Third: the MoE row is identical to the 8B row, which is the fact that catches most candidates and is the one worth saying unprompted.

### Napkin-size a deployment for me. Model, SLO and QPS given, five minutes, no laptop.

**🏋 Drill:** given a model, a target concurrency, a context distribution and a price per GPU-hour, produce a node count and a cost per million tokens in 5 minutes. Pass criterion: you follow all five steps in order and you state your assumptions out loud. Interviewers grade the *procedure* more than the number — if you produce a correct answer without naming your assumptions, you get less credit than someone who is 20% off with every assumption stated.

**Step 1 — bytes per token.** $2 \cdot L \cdot H_{kv} \cdot d_h \cdot \texttt{dtype}$. Say it out loud as you write it. Ten seconds.

**Step 2 — fit the weights and derive the pool.** Parameters × bytes/param, pick a TP degree that fits (and check it does not exceed `num_key_value_heads`), subtract 1.2 GiB context and 3–5 GiB activations per card from `0.9 × 79.6` GiB. Multiply by cards. Thirty seconds.

**Step 3 — pool ÷ bytes-per-token = live token budget.** Then divide by p95 context to get concurrent sessions per node. This is the whole capacity answer and it is one division. Twenty seconds.

**Step 4 — check the *other* bottleneck.** Capacity is memory; throughput is bandwidth. Per decode step, bytes moved = weights + (live tokens × bytes/token); divide by aggregate HBM bandwidth for the step time; multiply concurrency by (1/step time) for aggregate tok/s; derate to 60% for real memory-bandwidth utilisation. If the tok/s number doesn't meet demand, you are bandwidth-bound and more memory won't help. Ninety seconds.

**Step 5 — nodes and money.** Nodes = required concurrency ÷ concurrency per node, then add 30% for autoscaling lag and burst headroom. Cost per Mtok = (nodes × GPUs × $/GPU-hr) ÷ (aggregate tok/s × 3600) × $10^6$. Sixty seconds.

Worked, out loud, for "70B, 200 concurrent users, p95 context 8k, $2.50/GPU-hr":

> "320 KiB per token. FP8 weights, 65 GiB, TP=4 fits with 16.3 per card; pool is about 50 per card, 200 GiB per node, 656k live tokens. At 8k p95 that's 80 sessions per node, so 200 users needs 2.5 nodes — call it 3, plus headroom, 4 nodes of 4×H100. Throughput check: 80 sessions × 8k = 640k live tokens × 320 KiB = 200 GiB of KV plus 65 GiB of weights is 265 GiB per step over 13.4 TB/s aggregate is 21 ms, so 80 tokens per 21 ms is 3,800 tok/s theoretical, derate to 60% is 2,300 tok/s per node, 9,200 across four nodes. Sixteen GPUs at $2.50 is $40/hour; $40 over 9,200 tok/s times 3600 gives $1.21 per million output tokens at full utilisation, about $3 at realistic 40% duty cycle. If the SLO needs better than 25 ms ITL I'd cut concurrency per node and add a fifth node."

That paragraph, delivered in under two minutes with the arithmetic visible, is a strong hire signal on its own.

### What does a single 1M-token request actually cost you, and would you offer it?

Full accounting, because "we support 1M context" is a claim with a price tag most teams have never computed.

**Prefill compute.** Linear part: $2 \times 70\times10^9 \times 1{,}048{,}576 = 1.47\times10^{17}$ FLOP. Attention part (causal): $2 \times 80 \times (1.048576\times10^6)^2 \times 8192 = 1.44\times10^{18}$ FLOP. Note attention is **9.8× the entire rest of the model** — this is the 107k crossover point, blown past by 10×. Total $1.59\times10^{18}$ FLOP. At $3.17\times10^{15}$ FLOP/s (4×H100, FP8, 40% MFU): **501 seconds = 8.4 minutes of all four GPUs.**

**KV footprint.** 160 GiB at FP8 (320 GiB at bf16, which does not fit). That is **80% of the node's entire KV pool for one request.**

**Decode speed.** Each token reads 160 GiB of KV plus 65 GiB of weights = $2.42\times10^{11}$ B over 13.4 TB/s aggregate = **18.1 ms/token = 55 tok/s**, with the whole node dedicated. Compare 5.4 ms/token (185 tok/s) at 8k context — long context makes decode **3.4× slower per token**, and that is pure KV bandwidth.

**💰 Math:** 501 s × 4 GPUs = 2,004 GPU-seconds = 0.557 GPU-hours × $2.50 = **$1.39 of raw prefill compute.** A 1,000-token response adds 18.1 s × 4 GPUs = 72 GPU-s = $0.05. So ~$1.44 direct. At a list price of $3/Mtok input (**📅 Volatile**) you charge $3.15 — a 54% gross margin *if the node is otherwise fully utilised*. It is not: this request held 80% of the pool for 8.4 minutes, so you forfeited roughly 8 minutes × 0.8 of a node's alternative revenue. Include that opportunity cost and the request is marginal or negative.

**Would I offer it?** Yes, but not on the interactive pool and not at the interactive price. The design:

**Separate pool.** 1M-context requests go to their own node group with their own SLO (minutes, not milliseconds) and their own autoscaling. Mixing them with chat traffic means one request destroys the p99 for everyone, and no amount of chunked prefill fully hides an 8.4-minute prefill.

**Asynchronous by default.** Submit, poll, retrieve. A request that takes 8.4 minutes to reach first token is not a streaming interaction; pretending otherwise produces a UI that appears hung.

**Priced on token-hours, not tokens.** From the chargeback derivation, 160 GiB held for 8.4 minutes is $0.035 \times 160 \times 0.14 = \$0.78$ of cache rent on top of compute.

**And I would fight the requirement first.** In my experience the honest answer to "we need 1M context" is "we need better retrieval" about eighty percent of the time. The experiment that settles it costs a day: take the 1M-token corpus, run your task with a good retriever feeding 32k of context, and compare quality. If retrieval matches full-context quality — and for most extraction and QA tasks it does — you just saved $1.39 and 8.4 minutes per request and turned a capacity nightmare into an ordinary workload. Bring that experiment to the design review rather than an opinion.

**🗣 Say this in the room:** "A 1M-token prefill on a 70B is about 1.6 × 10^18 FLOPs — 8.4 minutes of a 4×H100 node — and 160 GiB of FP8 KV, which is 80% of that node's pool. So one request costs roughly $1.40 in compute and monopolises the node. I'd serve it from a separate async pool priced on token-hours, and before building any of that I'd run the retrieval-versus-full-context ablation, because usually retrieval wins and the requirement evaporates."

### Last one — you're at a whiteboard, the interviewer says "size me a cluster," and you have 90 seconds. What comes out of your mouth?

**🏋 Drill:** rehearse this until it's muscle memory. Pass criterion: 90 seconds, no notes, assumptions stated, one number at the end, and a stated next step. Record yourself once; the failure mode is almost always burying the answer under caveats instead of leading with the structure.

The structure, and it never changes:

**"Let me state what I'm assuming."** Model config, dtype for weights and KV, GPU and its HBM and bandwidth, p95 context, target concurrency, price per GPU-hour. Six items, fifteen seconds. Stating them converts every subsequent number from a guess into a derivation, and if the interviewer disagrees with an assumption they'll correct you now rather than watching you build on sand.

**"Bytes per token is two times layers times KV heads times head dim times dtype."** Compute it. This is the anchor and everything hangs off it.

**"Weights are params times bytes, TP degree is the smallest that fits without exceeding KV heads, and the pool is what's left after weights, about a gig of context, and three to five gigs of activations."** One subtraction per bucket.

**"Pool over bytes-per-token is my live token budget; over p95 context is sessions per node."**

**"Then I check the bandwidth side, because memory gives me capacity and bandwidth gives me throughput, and they're different limits."** Bytes moved per decode step over aggregate HBM bandwidth, derated to 60%.

**"Nodes, plus thirty percent for scale-up lag. Dollars per million tokens is node cost over tokens per second times 3600."**

**"And the thing I'd want to measure before committing"** — always end here — **"is the actual context-length distribution, because everything above is driven by p95 context and I've assumed it. If the p95 is 30k instead of 8k, the node count is four times what I just said."**

That last move is what separates a senior answer from a correct one. Anyone can produce a number. What gets you hired at this level is naming the input your answer is most sensitive to, and saying how you'd go measure it.

**🗣 Say this in the room:** "Bytes per token, then weights, then the residual is the pool, then pool over bytes-per-token is my token budget, then divide by p95 context for concurrency, then sanity-check throughput against HBM bandwidth. The number I'm least confident in is p95 context length, and it's the one the whole estimate is most sensitive to — so that's the first thing I'd instrument."
