### How does a serving engine run a batch where every request wants a different LoRA adapter?

The mental model: batching is only cheap when every request in the batch multiplies by the *same* weight matrix. That is the entire economics of LLM serving — one weight read from HBM amortized over many sequences. Per-request LoRA breaks that assumption for the adapter path, and multi-LoRA serving is the set of engineering tricks that restores it.

Naively, if request `i` needs adapter `a_i`, the adapter contribution is `(α/r)·B_{a_i}·A_{a_i}·x_i`, which is a different matmul per row of the batch. Doing that as a Python loop over requests destroys throughput. The fix is a **grouped/segmented kernel**: sort the batch by adapter id so requests using the same adapter are contiguous, stack all the `A` matrices into one tensor indexed by adapter id, and launch a single kernel that, per segment, gathers the right adapter slice and does the small GEMM. That is what Punica's SGMV (Segmented Gather Matrix-Vector multiplication) does, and S-LoRA extends it into a full serving system.

**📄 Paper:** Chen et al. (2023), *Punica: Multi-Tenant LoRA Serving* — introduced SGMV, a kernel that batches the low-rank matmuls of many distinct adapters into one launch, making heterogeneous-adapter batching practical. Sheng et al. (2023), *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* — added a **unified memory pool** that pages adapter weights and KV cache blocks out of the same allocator, plus host-memory tiering, so adapter count is bounded by host RAM rather than by GPU memory.

The unified-paging idea is the part a systems engineer should appreciate. Adapters and KV blocks are both variable-size, dynamically-lifetimed GPU allocations. Giving them separate pools guarantees that one pool is starving while the other is fragmented — the exact pathology you would recognize from running two independent slab allocators against one arena. S-LoRA puts them in one paged pool with a common page size, so an idle adapter's pages can immediately become KV cache pages.

The three-tier residency is: **GPU (active this batch) → host DRAM (warm, PCIe-fetchable) → object storage (cold)**. Promotion happens on a request for an adapter not currently resident, and that promotion latency is what shows up as a cold-start tail. At 84 MB per rank-16 8B adapter over ~28 GB/s of practical PCIe Gen4 bandwidth, a host→GPU fetch is `0.084 / 28 ≈ 3 ms`. From S3 at, say, 300 MB/s it is `84 / 300 ≈ 280 ms` — two orders of magnitude worse, and it lands entirely in TTFT for the unlucky request.

**🗣 Say this in the room:** "You sort the batch by adapter id and run a segmented gather-GEMM so all the distinct adapters go through one kernel launch instead of a loop. S-LoRA additionally pages adapters and KV blocks out of a single unified pool, so adapter capacity is limited by host RAM rather than VRAM. The cost is not FLOPs — the adapter FLOPs are under 1% — it is extra HBM traffic proportional to the number of *distinct* adapters in the batch."

### What is multi-LoRA actually costing you in latency? Give me the mechanism, not a hand-wave.

The right frame is: **decode is memory-bandwidth-bound, so count bytes read per decode step, not FLOPs.**

First, dispose of the FLOPs argument. A rank-16 all-linear adapter on Llama-3-8B is 41.9M parameters. The adapter's forward FLOPs per token are `2 × 41.9e6 = 83.8 MFLOP`, against the base model's `2 × 8e9 = 16 GFLOP`. That is **0.52% extra compute**. Compute is not the problem.

Now count bytes. In a decode step, the engine must read the base weights once — 16 GB in bf16, shared across the whole batch — plus **every distinct adapter present in that batch**, because each one is a different set of weights.

```
bytes_per_decode_step = 16 GB  +  (n_distinct_adapters × 84 MB)
```

On an H100 with ~3.35 TB/s of HBM bandwidth:

| distinct adapters in batch | bytes read | step time floor | slowdown |
|---|---|---|---|
| 0 (base only) | 16.0 GB | 4.78 ms | — |
| 8 | 16.67 GB | 4.98 ms | +4.2% |
| 16 | 17.34 GB | 5.18 ms | +8.4% |
| 32 | 18.69 GB | 5.58 ms | +16.8% |
| 64 | 21.38 GB | 6.38 ms | +33.6% |

**That table is the whole story of multi-LoRA latency, and it explains the production behaviour people find mysterious:** throughput is fine when your traffic is concentrated on a few tenants and degrades roughly linearly as adapter diversity in a batch grows. It is not a function of how many adapters you have *registered*; it is a function of how many are *simultaneously hot*.

Two knobs follow directly. **`max_loras`** — a cap on distinct adapters per batch — is a direct cap on the right-hand column; requests for a capped-out adapter wait for the next batch, converting latency variance into queueing. **Adapter-aware scheduling** groups requests by adapter so batches are less diverse, at the cost of fairness and head-of-line blocking for low-traffic tenants. Both are the same trade you make with any partitioned consumer: batch affinity buys throughput and costs tail latency for the sparse partitions.

**⚠ Trap:** benchmarking multi-LoRA with one adapter and concluding "the overhead is negligible." Of course it is — you measured the `n=1` row. Your benchmark must sample adapter ids from your actual production traffic distribution, including the long tail. A synthetic uniform draw over 200 tenants is *also* wrong, in the pessimistic direction, because real traffic is Zipfian.

**📐 Numbers you must know:** a rank-16 all-linear adapter is **~0.5% of base parameters**, so **~0.5% extra FLOPs** and **~0.5% extra bytes per distinct adapter**. Multiply by the number of distinct adapters in the batch to get your decode slowdown. That one line lets you answer any multi-LoRA capacity question on the spot.

### Fifty enterprise tenants each want a fine-tuned model. Price out one base with fifty adapters versus fifty deployments.

Assume Llama-3-8B, bf16, rank-16 all-linear adapters, and that the fleet must serve 100 req/s aggregate with an SLO.

**Option A — fifty independent deployments.** Each needs the full 16 GB of weights plus KV cache. On an 80 GB A100 that is one GPU per tenant minimum, and you need at least two replicas per tenant for availability during deploys and node failures.

```
50 tenants × 2 replicas = 100 GPUs
100 × $2.00/hr  =  $200/hr
$200 × 24 × 30  =  $144,000/month
```
(**📅 Volatile:** $2.00/hr for an on-demand A100-80GB is a plausible 2025–2026 cloud figure; reserved and neocloud pricing is materially lower. Verify before quoting.)

The deeper problem is not the price, it is the **utilization**. Aggregate 100 req/s spread over 50 tenants is 2 req/s each. One 8B replica on an A100 handles far more than that, so every one of those 100 GPUs runs at low single-digit percent utilization. You are paying for 100 GPUs of capacity to serve one GPU's worth of work. This is the argument that lands in a room — not "adapters are cheaper," but "your fleet is 3% utilized."

**Option B — one base, fifty adapters, multiplexed.**

```
base weights (bf16):        16.0 GB
50 adapters × 84 MB:         4.2 GB
------------------------------------
resident:                   20.2 GB   on an 80 GB card
KV cache headroom:          ~59.8 GB
```

Llama-3-8B KV per token = `2 × 32 layers × 8 kv_heads × 128 head_dim × 2 bytes = 131,072 bytes = 128 KiB`. So `59.8e9 / 131,072 ≈ 456,000 tokens` of cache — about **111 concurrent 4k-token sequences on one GPU**. For 100 req/s with headroom and HA, four GPUs is generous:

```
4 GPUs × $2.00/hr = $8/hr  →  $5,760/month
```

**💰 The comparison:** `$144,000 / $5,760 = 25×`. Even against a stingy one-replica-per-tenant version of Option A ($72,000/month), it is 12.5×. Annualized, that is **$1.66M/year saved** on a fifty-tenant deployment — which is a number that gets an architecture approved.

**What you give up, stated honestly, because the interviewer will push:**
1. **Noisy neighbours.** One tenant's traffic spike consumes shared KV cache and queues. You need per-tenant admission control and token-rate limits — the same per-tenant token-bucket you would build for any shared API, just keyed on tokens rather than requests.
2. **Blast radius.** A bad base-model upgrade breaks all fifty tenants at once, whereas fifty deployments fail independently. Mitigate with canary fleets and per-tenant eval gates, not with more GPUs.
3. **Adapter-diversity latency**, per the previous question — up to tens of percent of decode slowdown at high diversity.
4. **Compliance.** Some enterprise contracts specify physical isolation. That is a legal constraint, not a technical one, and the right answer is a small dedicated tier for those tenants and the shared tier for everyone else — not abandoning multiplexing.

**🗣 Say this in the room:** "Fifty dedicated 8B deployments is roughly $144k/month at two replicas each and runs at single-digit utilization. One base with fifty 84 MB adapters fits in 20 GB, serves the same traffic on four GPUs for about $5.8k/month, and leaves 60 GB for KV. That is 25×. I'd carve out a dedicated tier only for tenants with contractual isolation requirements."

### Merge the adapter for serving, or hot-swap it at request time? Give me the decision rule.

These are genuinely different products and I would not let a team pick by taste.

**Merge** means folding `(α/r)·BA` into `W` offline and shipping a standalone model. Consequences: zero inference overhead, no adapter kernels, quantize and compile the merged model however you like, works on any engine including ones with no LoRA support. Costs: one full model artifact per variant (16 GB each for 8B), full deployment per variant, and a rebuild for every adapter update.

**Hot-swap** means keeping the base loaded and applying adapters at request time. Consequences: `N` variants share one base, adapters are ~84 MB artifacts you can add and remove at runtime, per-request routing. Costs: the diversity-dependent decode overhead, engine LoRA support required, rank constraints, and complexity.

**My decision rule, in order:**

1. **One or two variants, latency-critical, high volume?** Merge. There is no reason to pay adapter overhead to serve a single variant. This is the code-completion case: Cursor-style inline completion is a p50 game where 5% of decode matters and there is exactly one model.
2. **More than about five variants, or the set changes weekly?** Hot-swap. The operational cost of `N` deployments dominates before you get very far past five.
3. **Per-tenant customization at any scale?** Hot-swap, obviously — this is the case the technology was built for.
4. **Serving on an engine or hardware target without LoRA kernels** (edge, an ONNX/TensorRT export, an embedded runtime)? Merge, you have no choice.
5. **Trained with QLoRA and you need maximum fidelity?** Serve unmerged on the quantized base, per the merge-mismatch problem. If you must merge, merge into the *dequantized* base and evaluate before and after.
6. **You need the base model available alongside the adapted one** (routing simple requests to base, hard ones to adapted)? Hot-swap gives you both from one deployment for free. Merging gives you two deployments.

**The hybrid I actually ship in multi-tenant products:** hot-swap for the long tail, merge for the top few tenants by volume. If two tenants are 60% of your traffic, they get merged dedicated deployments with zero overhead, and the other 198 share a multiplexed fleet. This is straightforward Pareto reasoning and it reads as senior in a design round.

**⚠ Trap:** `merge_and_unload()` is destructive and irreversible on the in-memory model, and — because merging changes `W` — repeatedly merging and unmerging accumulates floating-point error. If your deployment pipeline merges, evaluates, unmerges, merges a different adapter, you are slowly corrupting weights. Always merge from a freshly-loaded base. Make that a hard rule in the pipeline, not a convention.

### Design the adapter lifecycle for a platform serving two hundred tenant adapters. What are the moving parts?

Treat adapters as **versioned, immutable artifacts with a routing layer** — the design is much closer to a container registry plus a feature-flag service than to anything ML-specific, and saying that out loud is the point.

**Registry.** Object storage holds `s3://adapters/{tenant}/{adapter_id}/` containing the adapter weights, the `adapter_config.json`, and — this is the part teams skip — a manifest recording base model identity and revision hash, tokenizer hash, chat template hash, rank/alpha/target modules, training data snapshot id, and the eval report that gated promotion. **The base-model revision hash is load-bearing:** an adapter trained against `base@v1` applied to `base@v2` is not an error, it is a silent quality regression. Refuse to load on hash mismatch; do not warn.

**Metadata plane.** Postgres, because you need transactions across `(tenant, adapter_id, status, traffic_weight)`. Statuses: `training → evaluated → staged → active → deprecated → deleted`. Traffic weight enables canaries and A/B without a redeploy.

**Routing.** The gateway resolves `tenant_id → active adapter_id` (cached, with a short TTL and an invalidation channel), and attaches it to the inference request. The critical design decision is that **adapter selection is a routing decision made before the request hits the engine**, not something the engine infers. That keeps the engine stateless with respect to your tenancy model and lets you shadow, canary and roll back at the gateway.

**Residency management.** GPU slots are finite (`max_cpu_loras` / `max_loras` in vLLM terms). Run an LRU over adapters with pinning for your top-`k` tenants by traffic, and prewarm on a predictable schedule if traffic is diurnal. Instrument **adapter cache hit rate** as a first-class metric alongside prefix-cache hit rate — it is your equivalent of a buffer-pool hit rate and it predicts tail latency directly.

**Promotion gate.** No adapter reaches `active` without: an eval on that tenant's held-out set beating the currently-active adapter *and* beating the base model (if it does not beat base, delete it — you have shipped a regression with extra steps), a general-capability guardrail suite proving no catastrophic forgetting, and a safety/refusal check. Automate this; it is a CI pipeline.

**Deletion.** Contractual and regulatory deletion means the adapter object, its metadata rows, its training snapshot, and any merged artifact derived from it. This is a strong argument for *never merging tenant adapters into shared weights* — a merged model has that tenant's learned behaviour smeared irreversibly across 8 billion parameters and there is no clean deletion story. **An unmerged adapter is a deletable object; merged weights are not.** I raise this unprompted in enterprise design rounds because it is the kind of constraint that reorders an architecture.

**🔍 Failure taxonomy for this platform.** (1) *Base drift* — you upgrade the base and 200 adapters silently degrade; prevented by the hash gate. (2) *Template drift* — someone changes the serving chat template; every adapter trained on the old one degrades; prevented by hashing the template into the manifest and asserting at load. (3) *Rank overflow* — a tenant trains at rank 64 while the fleet is configured `max_lora_rank=16`; some engines pad, some truncate, some error; the manifest must be validated against fleet configuration at promotion time, not at load time. (4) *Thundering herd on cold adapters* — a dormant tenant's Monday-morning traffic all misses the adapter cache simultaneously; fix with a single-flight fetch and a prewarm job. (5) *Cost attribution* — with a shared fleet you cannot bill by GPU-hour; you must meter tokens per tenant per adapter, which needs to be designed in, not bolted on.

### Your multi-LoRA fleet's p99 TTFT tripled after onboarding thirty new tenants, while p50 barely moved. Debug it.

The p50/p99 divergence is the diagnostic gift here: something is affecting a minority of requests severely rather than everything mildly. That immediately rules out the diversity-driven decode slowdown from the bandwidth table — that would move p50 too, roughly proportionally.

**Hypothesis ranking, and how I'd separate them:**

**1. Adapter cache misses (most likely).** Thirty new tenants means thirty more adapters competing for a fixed number of GPU adapter slots. Requests that hit a resident adapter are fast; requests that miss pay a host→GPU or, worse, S3→host→GPU fetch. `84 MB / 28 GB/s = 3 ms` from host DRAM — invisible. From object storage at 300 MB/s: `280 ms`, plus TLS and request overhead — that is your tripled p99 in one line. **Test:** log adapter cache hit/miss per request and correlate misses with TTFT. If misses account for the slow tail, you are done in ten minutes. **Fix:** raise `max_cpu_loras` so the warm tier holds all 200 adapters in host RAM (200 × 84 MB = 16.8 GB — trivially affordable), pin the top-`k` by traffic, and prewarm.

**2. `max_loras` queueing.** If distinct adapters per batch is capped and you now have far more concurrently-active adapters, requests for adapters that do not make the cut wait for a subsequent batch. This produces exactly a bimodal latency distribution. **Test:** look at scheduler queue-wait time bucketed by adapter, and check whether slow requests belong to low-traffic tenants. **Fix:** raise `max_loras` and eat the bandwidth cost, or accept it and set per-tenant SLOs honestly.

**3. Prefix-cache dilution.** This one is easy to miss. Thirty new tenants means thirty new system prompts, so your prefix cache is now spread over far more distinct prefixes and the hit rate falls. A prefix-cache miss means full prefill — for a 4k system prompt on an 8B model that is `2 × 8e9 × 4096 = 65.5 TFLOP`, tens of milliseconds even at good utilization, versus near-zero on a hit. **Test:** prefix-cache hit rate before versus after onboarding. **Fix:** more KV capacity, or per-tenant prefix pinning.

**4. Rank heterogeneity.** If some new tenants trained at rank 64 and the fleet's `max_lora_rank` had to be raised to accommodate them, *every* adapter's memory footprint and kernel work may now be padded to the maximum rank depending on the engine's implementation. That would move p50 as well, so it is lower on my list, but it is worth a config diff.

**5. The boring one.** Thirty new tenants is more traffic. Check whether you are simply over capacity — GPU utilization, KV cache utilization, and scheduler queue depth. Do not diagnose an exotic adapter problem when you have crossed a throughput knee.

**The instrumentation I would have had in place, and would add if not:** per-request labels for `tenant_id`, `adapter_id`, `adapter_cache_hit`, `prefix_cache_hit`, `distinct_adapters_in_batch`, `queue_wait_ms`, `prefill_ms`, `decode_ms`. With those six fields, this becomes a single query rather than an investigation. That is the same discipline as tagging spans in any distributed system, and it is the right answer to "how would you know?"

### An adapter scores 91% in the training repo's eval script and 74% in production. Nothing errored. Where do you look?

Ranked by how often it is actually the cause, in my experience:

**1. Chat-template mismatch.** Training applied one template; the serving stack applies another — a different system-message wrapper, a missing generation prompt, doubled BOS, different whitespace around role tags. The adapter learned to respond to an exact token prefix and is now seeing a slightly different one. **Check:** decode the exact token ids from one training example and one production request, and diff them byte-for-byte. Not the strings you *think* are sent — the actual tokenized ids from the actual serving path.

**2. Merge-precision mismatch.** If the adapter was QLoRA-trained and got merged into fp16 weights for serving, this is the error described earlier and the magnitude is right for it. **Check:** run the same 200 examples through unmerged-on-quantized-base versus the production merged path.

**3. Sampling parameters.** The eval script ran greedy (`temperature=0`); production runs `temperature=0.7, top_p=0.95` because that is the API default. For a format-following or extraction task, sampling alone can cost double-digit accuracy. This one is embarrassing and extremely common. **Check:** the actual sampling params on the serving path, not the ones in the config file you think it reads.

**4. Adapter not actually applied.** The request omitted the adapter id, the routing layer fell back to base on a cache miss, or the engine silently ignored an unknown adapter name. **Check:** does the production response differ from the base model's response on a prompt where you know the adapter changes behaviour? Build that canary prompt deliberately — one input where base and adapter give visibly different outputs — and assert on it in a health check.

**5. Distribution shift between the eval set and production traffic.** Your held-out set came from the same curated pool as training; production has typos, truncated inputs, other languages, adversarial users, and inputs 3× longer than anything you trained on. **Check:** sample 200 real production inputs, label them, and re-run. If the adapter scores 74% on real inputs and 91% on curated ones, the adapter is fine and your *eval set* is the bug.

**6. Truncation.** Production inputs exceed `max_model_len` and get silently truncated — often from the *left*, removing the system prompt the adapter depends on. **Check:** distribution of input token counts in production versus training.

**⚠ Trap:** reaching for retraining. Every one of the six causes above is a *plumbing* bug fixed in an afternoon, and none of them is fixed by more training. The rule I enforce: **before any retrain is approved after a production-quality gap, someone must show me a byte-for-byte diff of the training-time and serving-time token sequences, and the sampling parameters from both paths.** That check has resolved this class of incident more often than any model change I have shipped.

### Before LoRA there were bottleneck adapters. How did Houlsby adapters work and why did they lose?

Bottleneck adapters insert **new small modules into the network** rather than modifying existing weights. Each adapter is a down-projection to a bottleneck dimension `m ≪ d`, a nonlinearity, an up-projection back to `d`, wrapped in a residual connection:

```
h ← h + W_up · σ(W_down · h)     with W_down ∈ R^{m×d}, W_up ∈ R^{d×m}, m ≈ 16–64
```

Initialized near-zero so the adapter starts as approximately the identity. Houlsby et al. place **two per transformer block** — one after the attention sublayer, one after the FFN. Pfeiffer et al. later showed **one per block** (after the FFN) performs comparably at half the parameters and half the added depth, which became the standard configuration.

**📄 Paper:** Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP* — the original bottleneck-adapter formulation, showing near-full-fine-tuning quality on GLUE at ~3% of parameters, and establishing PEFT as a field. Pfeiffer et al. (2020–21) — the single-adapter-per-block placement and AdapterFusion for composing multiple adapters.

**Why they lost, and it is one reason above all others: they cannot be merged.** A bottleneck adapter is a genuinely new computation — a nonlinearity sits between the two projections, so there is no way to fold `W_up σ(W_down ·)` into the surrounding linear layers. The adapter is in your forward pass forever.

That cost is not FLOPs; it is **sequential depth and kernel launches**. You have added two extra small matmuls plus a nonlinearity per block, 32 blocks deep, and they cannot be fused into the existing GEMMs or overlapped, because each depends on the previous sublayer's output. At small batch — exactly the autoregressive decode regime where you are latency-bound and the GPU is starved — the LoRA paper measured double-digit percentage inference-latency overhead for adapters relative to the unmodified model. In a product where you serve one token at a time to a user watching a cursor blink, a permanent double-digit latency tax to save training memory is not a trade anyone takes.

Secondary reasons: bottleneck adapters change the module graph, so every architecture needs an integration, whereas LoRA wraps any `nn.Linear` generically. And LoRA's mergeability is what enables the whole merge/hot-swap duality and model merging downstream. Adapters have none of that.

**🗣 Say this in the room:** "Bottleneck adapters insert a down-project, nonlinearity, up-project block into every layer. The nonlinearity is why they can never be merged into the base weights, so you pay their latency on every token forever — and at batch 1 decode, that is a double-digit percentage tax. LoRA has no nonlinearity between its factors, so `BA` is just a weight delta. That single structural difference is why LoRA won."

### Explain prefix tuning, P-tuning v2 and soft prompts. Why don't we use them?

These are the "learn in activation space instead of weight space" family, and they are conceptually elegant and operationally awkward.

**Prompt tuning** (Lester et al., 2021) is the simplest: prepend `k` trainable continuous vectors to the *input embedding sequence*. The model is entirely frozen; you learn `k × d_model` parameters — 20 tokens at `d=4096` is 82k parameters, four orders of magnitude below LoRA. The famous finding is that at very large model scale it approaches full fine-tuning quality, while at smaller scale it lags badly.

**Prefix tuning** (Li & Liang, 2021) goes deeper: rather than only at the input, prepend trainable key and value vectors at **every attention layer**. You are directly writing into the KV cache. Capacity is much higher than prompt tuning; training is notoriously unstable and typically requires reparameterizing the prefix through an MLP that is discarded afterwards.

**P-tuning v2** (Liu et al., 2022) is essentially deep prefix tuning applied across scales and NLU tasks, demonstrating it can match full fine-tuning across model sizes where shallow prompt tuning could not.

**Why they lost, in three concrete costs:**

1. **They consume your context window.** A 32-token prefix at every layer occupies 32 KV positions, permanently, on every request. At Llama-3-8B's 128 KiB/token that is `32 × 128 KiB = 4 MB` of KV per sequence, per request — and at batch 64 that is 256 MB of KV cache you are not using for user content. More importantly, in an agent or RAG product where context is the scarce resource, spending 32 tokens of every request on a learned prefix is a real budget line.
2. **They interact badly with prefix caching.** Your engine's automatic prefix cache keys on token ids. A *learned continuous* prefix is not token ids, so depending on the engine it either fails to participate in prefix caching or requires special handling. Losing prefix-cache hits on a long system prompt is far more expensive than anything the method saves.
3. **They are hard to train.** Prefix tuning is genuinely finicky — sensitive to initialization, prone to divergence, requiring the reparameterization trick. LoRA trains on the first attempt with default hyperparameters. That reliability difference, more than any benchmark, is why practitioners moved.

**Where soft prompts are still interesting:** when you need thousands of *ultra-cheap* task variants and quality demands are modest, an 82k-parameter prompt beats an 84 MB adapter on storage by 1000×. And conceptually, they are the bridge to understanding that in-context learning and weight updates are two points on one continuum — a nice thing to be able to say when someone asks why prompting works at all.

### IA³, BitFit, ReFT — one paragraph each, and tell me whether any of them beats LoRA on some axis.

**IA³** learns three **elementwise rescaling vectors** per layer: `l_k` scaling the keys, `l_v` scaling the values, and `l_ff` scaling the FFN intermediate activations. So `K ← l_k ⊙ K`, and so on. Parameter count is a handful of vectors of length `d_head·n_kv_heads` or `d_ffn` per layer — on the order of 0.01% of the model, roughly 50× smaller than a rank-16 LoRA. Crucially, **it merges**: scaling the rows of `K` elementwise is the same as scaling the rows of `W_k`, so `diag(l_k)·W_k` folds in cleanly and inference is free. Its home turf is the few-shot regime.

**📄 Paper:** Liu et al. (2022), *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning* (the T-Few recipe) — introduced IA³ and argued that a tiny fine-tune beats stuffing examples into the prompt on both quality and per-request cost.

**BitFit** trains only the **bias terms**, roughly 0.08% of parameters, and was a striking result on BERT-scale encoders. There is a sharp modern caveat worth knowing: **Llama-family and many other current decoder LLMs have no bias terms in their linear layers at all.** BitFit is therefore largely inapplicable to the models you would actually fine-tune today. Knowing that is a good tell that you have read a config file rather than a survey paper.

**ReFT** (representation fine-tuning) breaks the frame: instead of editing weights, it learns **interventions on hidden representations** at chosen layers and token positions, editing `h` within a learned low-rank subspace. LoReFT reports competitive quality at roughly 10–50× fewer trainable parameters than LoRA. The catch is that an intervention on activations at specific positions is **not a weight delta**, so it cannot be merged, and it requires engine support for position-conditional hooks that essentially no production serving stack has.

**📄 Paper:** Wu et al. (2024), *ReFT: Representation Finetuning for Language Models* — learns low-rank interventions on hidden states rather than weight updates, achieving strong parameter efficiency at the cost of mergeability.

**Does any of them beat LoRA on an axis? Yes — storage.** IA³ and ReFT adapters are one to two orders of magnitude smaller than LoRA adapters. If your product is "50,000 per-user personalizations," `50,000 × 84 MB = 4.2 TB` of LoRA adapters versus `50,000 × ~1 MB` of IA³ is a real architectural difference. That is the one scenario where I would seriously evaluate IA³ over LoRA. On every other axis — quality ceiling, training reliability, ecosystem, kernel support, mergeability, and the sheer weight of everyone else's experience — LoRA wins, and I would need a specific reason to deviate.

### So why did LoRA win? Give me the argument as a list of properties, not a vibe.

I want six properties, because the win is overdetermined and no single one explains it.

1. **Zero inference cost when merged.** `BA` has the same shape as `W`, and there is no nonlinearity in between, so `W + (α/r)BA` is a legal weight matrix. Every alternative that inserts computation (bottleneck adapters), consumes context (prefix tuning), or intervenes on activations (ReFT) pays forever. This is the property that matters most in a product.
2. **Zero architecture cost.** LoRA wraps `nn.Linear`. It does not know or care whether the model is Llama, Qwen, Mixtral or something released next month. Every method that inserts modules needs per-architecture integration work, which is why their support matrices are always stale.
3. **It composes with quantization.** NF4 base + bf16 adapters works because the adapter is a separate additive path that never has to be quantized. That composition is QLoRA, and QLoRA is what made 70B fine-tuning accessible to people without clusters. No other PEFT method got a comparable multiplier.
4. **The dual serving mode.** The same artifact can be merged for a dedicated deployment or hot-swapped for multi-tenancy. You choose per deployment, not per training run. That optionality is worth a lot operationally.
5. **Training reliability.** Default hyperparameters work. Zero-init `B` guarantees a clean start. There is no reparameterization trick, no instability regime, no warmup schedule you must discover. Boring is a feature.
6. **The artifact is small, standard, and portable.** 84 MB, a config JSON and a safetensors file, in a format every framework reads. It can be versioned, signed, cached, shipped over a CDN, and deleted. That makes it a *product primitive*, not just a training technique — which is exactly what per-tenant customization needs.

**🗣 Say this in the room:** "LoRA won because it is the only method that is simultaneously mergeable, architecture-agnostic, quantization-compatible, and trivially reliable to train. Bottleneck adapters lose on mergeability, prefix tuning on context cost and stability, IA³ and ReFT on capacity and engine support. Nothing else has all four properties, and the four together are what turn a training trick into a serving architecture."

### When is full fine-tuning actually the right call? Argue against your own default.

There is a real return to full fine-tuning and PEFT enthusiasm sometimes obscures it. Five cases where I would push for it:

**1. Small models — under about 3B.** The memory equation makes this concrete: `16 × 3e9 = 48 GB` of static state fits on one 80 GB card with room for activations. If it fits, the LoRA memory argument evaporates entirely, and you get the full-rank update, faster convergence, and no rank hyperparameter. Since the industry's actual cost endgame is usually "distill into a small model and serve it cheaply," the model you end up shipping is frequently in exactly this range. **My rule: at ≤3B, full fine-tune unless you specifically need multi-tenant adapter serving.**

**2. Continued pretraining or large domain shift.** Tens of billions of tokens of new-domain text, a new language, a genuinely different register. This is precisely the regime where "LoRA learns less" bites, and no rank fixes it. If you are spending a five-figure compute budget anyway, do not cap your quality with a low-rank constraint to save a fraction of it.

**3. New modality or new tokens.** Training a vision projector or a large block of new embedding rows is not a low-rank operation. Full training of those specific modules (with or without LoRA elsewhere) is the correct structure.

**4. You will only ever serve one variant, at very high volume.** If you serve one model to everyone at a million requests a day, the multi-tenant argument is worth nothing to you, and the only question is quality per dollar of serving. Full fine-tuning gives you a slightly better model for the same serving cost.

**5. When forgetting is the goal.** Occasionally you *want* the model to move decisively away from base behaviour — a strict domain assistant that should refuse general chit-chat, a heavily constrained format. LoRA's regularization actively works against you there.

**⚠ Trap — the reflex in the other direction.** Having learned that LoRA is the default, engineers start proposing LoRA for a 1B model where full fine-tuning costs `16 GB` of state, trains faster, and is strictly better. Defaults are not rules. The question to ask is always: *does the memory equation actually bind, and do I need more than one variant?* If the answer to both is no, LoRA is buying you nothing but a rank hyperparameter to tune.

### With two hundred tenants of wildly different sizes, how do you set per-tenant training policy without running two hundred hyperparameter sweeps?

You tier them, and you make the tiering a function of **data volume and traffic**, because those are the two variables that actually change the right answer.

**Tier 0 — fewer than ~200 labeled examples.** Do not fine-tune. This is the single most important policy line and it saves the platform from itself. Below a couple of hundred examples you cannot build a held-out set that distinguishes a real improvement from noise, and a LoRA trained on 80 examples overfits into a model that repeats the training set. These tenants get a tenant-specific *prompt* and *retrieval index* instead — same customization outcome, no training, no artifact to maintain, instantly updatable. I would enforce this as a hard gate in the platform, not advice in a doc.

**Tier 1 — 200 to ~5,000 examples.** Fixed recipe, zero sweeping: rank 16, alpha 32, all linear layers, LR 1e-4, 2 epochs, dropout 0.05, cosine schedule. The recipe is chosen once by sweeping on three representative tenants and then frozen. The variance across tenants at this scale is dominated by data quality, not hyperparameters, so a sweep per tenant is spending compute to learn nothing.

**Tier 2 — 5,000 to 100,000 examples.** Same recipe with a small automated LR selection: run three LRs (`5e-5, 1e-4, 2e-4`) for a fraction of an epoch, pick the best by held-out loss, complete that run. Three short runs, not a grid. Consider rank 32 if training loss plateaus.

**Tier 3 — over 100,000 examples, or a tenant paying for it.** Real experimentation, human in the loop, and the question of whether a dedicated merged deployment beats a shared adapter should be revisited with their actual traffic numbers.

**The gate that matters more than any of this** is the eval. Every tier-1-and-above adapter must beat two baselines on the tenant's held-out set: the base model, and the current production configuration (which for a first fine-tune is prompt-only). **If the adapter does not beat prompt-only, you delete it.** Roughly speaking, expect a meaningful fraction of tier-1 fine-tunes to fail this gate — and that is the system working, not failing. A platform that ships every adapter it trains is a platform with no eval.

**💰 Math on why the tiering pays.** A tier-1 QLoRA run on 8B is about 20 minutes of H100 time: `0.33 hr × $3 = $1`. Two hundred tenants retrained monthly is `200 × $1 = $200/month` in compute — nothing. But a per-tenant 12-point hyperparameter grid is `200 × 12 × $1 = $2,400/month` *and* 2,400 runs to orchestrate, monitor and evaluate. The compute is not the cost; the **evaluation and operational surface** is. That is the real reason to freeze the recipe, and it is the argument I would make in the design review.
