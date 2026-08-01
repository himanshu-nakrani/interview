### What does an inference engine actually do that a naive Hugging Face `generate` loop doesn't? Be specific.

Three things, and each one is worth an order of magnitude on some axis.

**Continuous batching.** The naive loop takes a batch of *B* sequences, runs them all to completion, then takes the next batch. Because generation lengths vary wildly — one request stops at 20 tokens, another runs to 2,000 — most of the batch is padding by the end, and no new request can enter until the longest one finishes. Continuous batching (also called iteration-level or in-flight batching) makes the scheduling decision *per forward pass*: after every token step, finished sequences leave the batch and queued ones join. Your effective batch size stays near the maximum instead of decaying, and queueing latency stops being "wait for the slowest peer."

**Paged KV cache.** The naive approach allocates a contiguous KV buffer sized to `max_seq_len` for every sequence in the batch. If your max is 32k and the average request uses 2k, you have wasted 94% of your cache to internal fragmentation. PagedAttention borrows the OS virtual-memory idea: KV lives in fixed-size blocks (typically 16 tokens), a per-sequence block table maps logical positions to physical blocks, and the attention kernel gathers through that indirection. Waste drops to under one block per sequence. This is the single change that took production LLM serving from "batch size 8" to "batch size 200."

**A scheduler with admission control and preemption.** Real engines decide, every step, which requests to run, which to keep waiting, and which to *preempt* — either by swapping their KV to CPU memory or by dropping it and recomputing later. That gives you a graceful degradation curve under overload instead of an OOM.

On top of those three: fused CUDA kernels (FlashAttention variants, fused RMSNorm+residual, fused RoPE), CUDA graph capture to eliminate per-step kernel-launch overhead (which is significant when a decode step is 5 ms and you are launching hundreds of kernels), quantization support, speculative decoding, and structured-output grammar masking.

**📄 Paper:** Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention* — the vLLM paper; replaced contiguous per-sequence KV allocation with paged blocks and reported 2–4× throughput improvements over the then-standard servers. Yu et al. (OSDI 2022), *Orca* — introduced iteration-level scheduling, i.e. continuous batching, replacing request-level batching.

**💰 Math:** the practical delta is roughly 10–24× throughput on the same GPU. Concretely: an 8B model on one H100 with naive `generate` at batch 8 might sustain ~300 output tok/s; the same GPU under vLLM at high concurrency sustains 3,000–5,000 output tok/s (**📅 Volatile:** exact numbers move with engine versions and kernels — benchmark yours). At $2.50/GPU-hr that is the difference between $2.31 and $0.16 per million output tokens. This is why "we wrote our own serving loop" is a red flag in a design round unless the candidate can name what they reimplemented.

### Your engine's logs show requests being preempted. What does that mean, and which knobs do you turn?

Preemption is the engine telling you the paged KV cache is full and the scheduler had to take memory back from a sequence that was already running. It is the LLM-serving analogue of the OS killing a process to reclaim pages — and like swapping, a little is fine and a lot is a death spiral.

The mechanism. Every running sequence grows its KV allocation by one block every 16 generated tokens. The scheduler admits sequences based on available blocks, but it cannot know in advance how long each will run, so it can over-admit and then discover mid-generation that there are no free blocks. Its options are:

**Recompute preemption** — drop the victim's KV blocks entirely, put the request back at the head of the waiting queue, and when it is rescheduled, prefill its entire prompt *plus everything it has generated so far* from scratch. Cheap to implement, no data movement, and the cost is quadratic in how late the preemption happens: a sequence preempted after generating 1,800 tokens on a 4,000-token prompt must re-prefill 5,800 tokens.

**Swap preemption** — copy the victim's KV blocks to pinned host memory over PCIe and copy them back on resume. For a 5,800-token sequence at 320 KB/token that is 1.86 GB each way; at ~25 GB/s effective PCIe that is 74 ms out and 74 ms back. Cheaper than recompute for long sequences, more expensive for short ones, and it consumes PCIe bandwidth you may want for other things.

**What the counter means for your health.** A non-zero, steady preemption rate means you are running past your admitted-concurrency ceiling and paying compute to redo work. Latency degrades superlinearly, because preempted requests re-enter the queue and compete with new arrivals for the cache that was already full. This is exactly the thrash curve you know from an over-subscribed connection pool.

**The knobs, in the order I turn them:**

1. **Lower `max_num_seqs`.** This is admission control: fewer concurrent sequences means each has room to grow. Counterintuitively this usually *raises* throughput when you are thrashing, because you stop wasting compute on recomputes. It is the same result as lowering a thread-pool size on a saturated service.
2. **Lower `max_model_len`** if your product does not need the ceiling — it directly caps the worst-case per-sequence footprint.
3. **Raise `gpu_memory_utilization`** if you have verified headroom, giving the cache more blocks.
4. **fp8 KV cache**, halving bytes per token — gated on a long-context eval.
5. **More replicas or higher TP**, which is the honest answer when the first four are exhausted.

**⚠ Trap:** treating preemption as a benign log line because throughput still looks acceptable. Preemption is the leading indicator of the latency cliff — it starts several minutes before p99 blows up, and it is the single best early-warning metric on an LLM fleet. Alert on `preemption_rate > 0` sustained over 5 minutes; do not wait for the latency alert.

### Give me the real state of vLLM. What is the V1 engine and what does it change?

vLLM is the default. It has the widest model coverage of any open engine, the largest contributor base, and it is what most companies should reach for first — the burden of proof is on anything else.

The V1 engine (**📅 Volatile:** V1 became the default around vLLM 0.8; verify current version behavior before you cite specifics) is a rewrite of the core execution loop motivated by a specific problem: on fast GPUs and small models, vLLM's Python-side scheduling and detokenization overhead was becoming a meaningful fraction of step time. A 5 ms decode step with 3 ms of Python overhead around it means you are running at 60% efficiency and no kernel optimization will save you.

The changes that matter operationally: the engine core runs in its own process, isolated from the API server so HTTP handling and tokenization do not contend with the scheduler; the scheduler is unified so prefill and decode are not separate code paths (chunked prefill is the normal case, not a mode); prefix caching is on by default rather than opt-in; and piecewise CUDA graphs cover more of the step so kernel-launch overhead largely disappears.

The practical advice I give: **turn prefix caching on and verify your hit rate is actually high**, set `--max-model-len` to your product's real ceiling rather than the model's, size `--gpu-memory-utilization` conservatively at first (0.85–0.90) and raise it once you have observed fragmentation behavior, and treat `--max-num-seqs` and `--max-num-batched-tokens` as your two latency/throughput knobs.

**⚠ Trap:** setting `--gpu-memory-utilization 0.98` to maximize cache. The remaining fraction has to absorb activation peaks during a full-width prefill, NCCL buffers, and any fragmentation. You will pass your load test and OOM three days later on a burst of long prompts. I keep 10% headroom and measure the actual peak with a synthetic worst-case prompt before tightening it.

**⚠ Trap:** assuming prefix caching is free. It is nearly free in compute but it consumes KV blocks that would otherwise serve active requests, and under memory pressure the engine evicts cached prefixes. If your hit rate mysteriously drops during peak traffic, that is the mechanism — you are being evicted by your own concurrency, and the fix is more replicas or a CPU/NVMe KV tier, not a config flag.

### Where does SGLang win over vLLM, and what is RadixAttention actually doing?

RadixAttention is the reason to look at SGLang, and it is a genuinely different data structure rather than a tuning difference.

vLLM's automatic prefix caching hashes fixed blocks and reuses blocks whose hash chain matches — effectively a hash-based lookup of linear prefixes. SGLang maintains a **radix tree** (a compressed trie) over all cached token sequences, with the KV blocks hanging off the tree nodes, and an LRU eviction policy that respects the tree structure. The difference shows up when your workload branches: one shared prefix with many divergent continuations.

That is exactly the shape of agentic and structured workloads. Consider a few-shot prompt that you fan out into 8 parallel tool calls, or a tree-of-thought search that explores 5 continuations from a common state, or a multi-turn conversation where you re-run the last turn with different sampling. With a radix tree the shared trunk is stored once and matched automatically; with linear prefix hashing you get the trunk match too, but the tree structure additionally makes eviction and sharing across *sibling* branches coherent.

**📄 Paper:** Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs* — introduced RadixAttention for automatic cross-request KV reuse plus a co-designed frontend language for structured generation. It replaced manual prompt-prefix management with an automatic, tree-structured cache.

The other SGLang strengths: a fast constrained-decoding path (compressed FSM / grammar handling that can advance multiple tokens at once when the grammar forces them), a cache-aware router for multi-replica deployments that keeps affinity without hot-shard-ing, and generally aggressive kernel work.

**My decision rule:** default to vLLM. Move to SGLang when your workload is (a) heavily multi-turn or agentic with high prefix reuse, (b) dominated by structured/JSON output where constrained-decoding overhead is on your critical path, or (c) you have measured a real win on your own traffic. Both projects move fast and leapfrog each other on benchmarks, so **the only defensible answer in an interview is "I'd run a bake-off on my own token-length distribution."** Saying "SGLang is faster" as a flat claim is a trap — the interviewer will ask "on what workload?" and you need the answer.

### When is TensorRT-LLM worth the operational burden?

TensorRT-LLM buys you the highest achievable tokens/second on Nvidia hardware, and it charges you for it in build complexity and flexibility.

The mechanism: rather than dispatching PyTorch ops at runtime, you *compile* the model into a TensorRT engine — a serialized plan that fuses kernels, selects tactics per GEMM shape by autotuning on your actual GPU, bakes in the precision policy (fp8, int4 AWQ, fp8 KV cache), and fixes things like max batch size, max input length and max output length at build time. Nvidia's kernels here are typically ahead of open implementations by a real margin on their own silicon, especially for fp8 and for the newest architectures.

The costs, which are the part candidates underestimate:

- **The engine is specific to the GPU architecture and often the exact TensorRT/driver version.** An engine built for H100 does not run on A100 or L40S. That means your CI builds N engines for N SKUs, and a driver upgrade can invalidate all of them.
- **Build time is 10–40 minutes for a large model**, so "roll back to the previous model version" is not instant unless you keep built engines in a registry.
- **Build-time shape constraints** mean supporting a longer context or a bigger batch is a rebuild, not a flag.
- **Model coverage lags.** A brand-new open-weights architecture appears in vLLM within days and in TensorRT-LLM later.
- **You need someone who can debug it.** When a build fails on a shape mismatch deep in a plugin, that is a specialist's afternoon.

**My rule:** TensorRT-LLM when you are serving one or two stable models at a scale where 20–30% throughput is worth a headcount — i.e. when 20% of your GPU bill exceeds an engineer's cost. **💰 Math:** if you run 200 H100s at $2.50/hr, that is $500/hr = $365k/month. A 25% throughput gain is worth $91k/month, which trivially funds the operational burden. If you run 8 GPUs at $14.6k/month, 25% is $3.6k/month and you should be shipping features instead.

**🗣 Say this in the room:** "TensorRT-LLM is the right answer when the model is stable, the hardware is homogeneous Nvidia, and the fleet is large enough that a 20–30% throughput gain pays for a dedicated owner. Below roughly a hundred GPUs I'd stay on vLLM and spend that engineering on retrieval quality instead, because that's where the product wins are."

### Cover the rest of the engine landscape for me — TGI, LMDeploy, llama.cpp, Ollama, MLX. When does each one legitimately win?

**TGI (Text Generation Inference)** is Hugging Face's Rust-based server. Its historic strengths are a well-engineered production surface — clean OpenTelemetry integration, sensible defaults, tight integration with the Hub and with Inference Endpoints. It is a reasonable pick if you are already deep in the HF ecosystem, and less commonly the raw-performance winner. Its licensing history is worth knowing (it went to a restrictive license and later returned to Apache-2.0) because enterprise legal review will ask.

**LMDeploy** (from the InternLM/OpenMMLab side) ships a high-performance TurboMind backend and is notably strong on 4-bit weight-only quantized serving and on the Chinese open-model families. Worth benchmarking if you are serving Qwen or InternLM variants at 4 bits.

**llama.cpp / GGUF** is the CPU-and-everything-else stack: a C++ implementation with its own quantization format (GGUF) and k-quant schemes, running on CPU with AVX/NEON, on Apple Metal, on CUDA, on Vulkan. Its production niche in a company like the ones you are targeting is *on-device and edge*: a desktop app that runs a small model locally for privacy or offline capability. Cursor-style local autocomplete, a Figma plugin doing local inference, an enterprise deployment where data cannot leave the laptop. It is not the answer for a datacenter fleet — its batching and multi-tenancy story is much weaker than vLLM's.

**Ollama** is a packaging and UX layer over llama.cpp with a model registry and a simple API. Its correct use is developer experience: every engineer on the team can run the model locally in one command, which makes local eval loops and demos fast. **⚠ Trap:** shipping Ollama as your production server because it worked on a laptop. It defaults to serving a small number of concurrent requests, its scheduling is not designed for datacenter multi-tenancy, and it will quietly serve a quantized variant you did not consciously choose — meaning your production quality differs from the eval you ran against the full-precision model. I have seen exactly this cause an unexplained quality regression.

**MLX** is Apple's array framework for Apple silicon with unified memory, so an M-series machine with 128 GB of unified memory can hold a model that would need two datacenter GPUs. Its role is Mac-native applications and local research on Apple hardware. Not a server story.

**🗣 Say this in the room:** "vLLM or SGLang for the datacenter, TensorRT-LLM when the fleet is large and the model is stable, llama.cpp/GGUF for on-device, and Ollama for the developer inner loop. The only one of those choices I'd litigate in a design review is the first, and I'd settle it with a bake-off on our own traffic."

### Build me the engine selection matrix. What are the axes, and what's your default?

The axes that actually decide it, in the order I evaluate them:

1. **Model coverage** — does it support the architecture you need, today, at the quantization you need? This is a hard gate and it eliminates candidates fastest. New architectures land in vLLM first.
2. **Quantization support** — which formats, and are the kernels fast or merely present? "Supports AWQ" and "has a fast fused AWQ kernel for this GPU" are different claims.
3. **LoRA multiplexing** — can it serve N adapters against one base model with per-request adapter selection? If your product is per-customer fine-tunes, this axis is the whole decision.
4. **Structured output** — does it have a fast grammar/JSON-schema constrained decoder, and what does it cost per token? If 80% of your calls are tool calls, this is on your critical path.
5. **Prefix caching** — present, on by default, and does it survive under memory pressure? Plus: is there a cache-aware router for multi-replica?
6. **Multi-node** — TP/PP across nodes, and does it integrate with Ray or a native distributed executor?
7. **Operational burden** — build step or not, config surface, upgrade cadence, how good the failure messages are at 3am, and whether anyone on your team can read the source.

Then there are the things people forget to put on the matrix and regret: **speculative decoding support**, **fp8/int8 KV cache**, **OpenAI-compatible API surface** (which determines how much client code you rewrite), **metrics quality** (does it export queue depth and KV utilization as Prometheus metrics, because you are going to autoscale on those), and **graceful shutdown / drain behavior** (does it finish in-flight requests on SIGTERM, because your rolling deploy depends on it).

**My default, stated as a decision procedure:** start on vLLM. Move to SGLang if a bake-off on your traffic shows a win, which it most often does for agentic/structured/multi-turn workloads. Move to TensorRT-LLM only above roughly 100 GPUs with a stable model. Use a managed provider if you are pre-product-market-fit or if your traffic is under ~10 GPU-equivalents of load. Use llama.cpp only on the client.

**⚠ Trap:** picking an engine on a published benchmark. Every engine's benchmark uses the input/output length distribution that flatters it. A 512-in/128-out benchmark tells you nothing about a 30k-in/500-out coding assistant, where prefill dominates and prefix caching is worth more than kernel speed. **The benchmark that matters is yours, replayed from production traces.**

### Triton Inference Server, Ray Serve, KServe — what layer is each, and would you run more than one?

They are not competitors; they sit at different layers, and a candidate who treats them as alternatives reveals they have only read the landing pages.

**Triton Inference Server** (Nvidia) is a *model server*: a process that loads one or more models across possibly-heterogeneous backends (TensorRT, TensorRT-LLM, PyTorch, ONNX, Python) behind one HTTP/gRPC surface, with dynamic batching, model ensembles, and model-instance management. Its distinctive value is heterogeneity — if you serve an LLM, an embedding model, a reranker and a classical ranker, Triton can host all of them in one process with one metrics surface. For LLMs specifically it usually wraps a TensorRT-LLM backend.

**Ray Serve** is a *distributed application framework* for serving. Its value is composition and multi-node: you write Python deployments with independent replica counts and autoscaling, wire them into a graph (retriever → reranker → LLM → post-processor), and Ray handles placement, actor lifecycle, and cross-node model parallelism. It is the natural fit when your inference is a *pipeline* rather than a single model call, and when you need multi-node TP/PP — vLLM's multi-node executor uses Ray for exactly this.

**KServe** is a *Kubernetes control plane*: an `InferenceService` CRD that gives you declarative deployment, canary traffic splitting by percentage, scale-to-zero, and a standard inference protocol, delegating the actual serving to a runtime (which can be vLLM, Triton, or a custom container).

So the honest answer to "would you run more than one" is yes, routinely: **KServe (or a plain Deployment + your own CRD-free setup) as the K8s layer, vLLM as the engine, and Ray only if you need multi-node model parallelism or a real multi-stage pipeline.** Adding Ray when you have a single-node model and a simple call graph is complexity you will pay for during on-call.

**⚠ Trap:** using Triton's dynamic batching for an LLM and expecting continuous batching semantics. Triton's generic dynamic batcher groups requests arriving within a time window into one batch — that is request-level batching, the thing continuous batching replaced. For LLMs you need the in-flight batching that lives in the TensorRT-LLM backend, not the generic path. Configuring the wrong one gives you correct outputs at a fraction of the throughput, and nothing errors.

### Talk me through managed inference — Bedrock, Vertex, Azure AI Foundry, Together, Fireworks, Baseten, Modal. What do you actually give up?

Group them by what they are, because the trade differs:

**Hyperscaler model gateways (Bedrock, Vertex AI, Azure AI Foundry)** give you first-party and third-party models behind one API, inside your cloud's IAM/VPC/compliance perimeter, on your existing enterprise agreement. That last point is the real reason big-tech and regulated enterprises use them: the security review is already done. What you give up is model freshness (a new model appears on the vendor's own API first and on the hyperscaler weeks or months later), some API surface fidelity (the hyperscaler's wrapper may lag features like a new caching or thinking parameter), and pricing flexibility.

**Open-model API providers (Together, Fireworks)** serve open-weights models per-token at high performance, with dedicated-capacity options and often custom fine-tune hosting. You give up control over the exact serving configuration — quantization, KV-cache precision, engine version — which means a silent provider-side change can move your eval numbers. Pin versions where the provider lets you, and run a daily canary eval against your own gold set.

**Infra platforms (Baseten, Modal, and similar)** are the middle path: you bring a container/model and they handle GPU provisioning, autoscaling, scale-to-zero, and cold-start optimization. You keep engine control; you give up the ability to do exotic placement and you pay a margin over raw GPU cost. Modal in particular is strong for *batch* and bursty workloads because of fast container starts and a Python-native job model.

**What you give up in every case:** the ability to run a custom kernel or a patched engine; sub-provider-level latency control; the p99 tail during someone else's incident; and a real cost floor, because you are paying a margin over hardware.

**What you gain, and interviewers want to hear you say this plainly:** you do not staff a GPU platform team. That is two to four engineers, and at a product company those engineers are worth more on retrieval quality and eval infrastructure.

**🗣 Say this in the room:** "I default to a managed API until one of three things is true: the token volume makes the margin exceed a platform team's cost, we need a model or a serving configuration nobody hosts, or data residency forbids it. Self-hosting is a capacity decision, not a sophistication signal — I've seen teams burn a quarter on a GPU platform to serve traffic that fit inside a $4k/month API bill."

### Why do you say embedding and reranker serving should be a separate fleet? Isn't it just another model?

It is another model with a completely different performance profile, and colocating them wastes money in both directions.

The differences that matter:

**No KV cache and no autoregression.** An embedding model runs one forward pass over the input and emits a vector. There is no decode phase, so there is no memory-bandwidth-bound token loop and no cache to manage. The workload is pure prefill — compute-bound, high arithmetic intensity, perfectly batchable.

**Tiny models, huge batches.** A strong embedding model is often 0.1–7B parameters. On one A10G or L4 you can hold the model in a couple of GB and batch hundreds of documents per forward pass. The right hardware is cheap inference GPUs, not H100s. **💰 Math:** an L4 at roughly $0.70/hr (**📅 Volatile**) embedding at, say, 2,000 short documents/second is 7.2M docs/hour for $0.70 — about **$0.10 per million documents embedded**. Putting that same work on an H100 you are already renting for the LLM costs 3.5× more per hour for maybe 3× the throughput, and it steals HBM from your KV cache.

**Bursty, batch-shaped traffic.** Query-time embedding is one short string per request — trivial. But *ingestion* is a million documents at once when someone connects a new data source. That is a batch job with a completely different scaling curve than your chat fleet, and if it shares GPUs with chat, your ingestion job will destroy chat's p99. Separate fleets means separate autoscaling and separate priority.

**Rerankers are different again.** A cross-encoder reranker scores (query, document) pairs jointly, so scoring 50 candidates is 50 forward passes over query+doc — much more expensive per query than embedding, and it sits *synchronously* on the user's latency path. A reranker fleet needs low-latency scaling; an embedding fleet needs throughput scaling.

**The serving stack:** Hugging Face's **TEI (Text Embeddings Inference)** and **Infinity** are the two purpose-built servers here. Both do token-based dynamic batching (batching by token count rather than request count, which matters because document lengths vary), support ONNX/Candle backends, and are much lighter than running an LLM engine. vLLM can serve embedding models too, and that is a reasonable consolidation if you want one stack, but you are carrying a lot of machinery you do not use.

**⚠ Trap:** autoscaling the embedding fleet on the same metric as the LLM fleet. Embedding has no KV cache, so KV utilization is meaningless there; you scale it on queue depth and batch-fill rate. Copy-pasting the LLM's KEDA scaler produces a fleet that never scales.

### How does LoRA multiplexing work in a serving engine, and what breaks when you have fifty adapters?

The mental model that makes this feel inevitable: a LoRA adapter is a low-rank delta, `W + BA` where `B` is `d×r` and `A` is `r×d` with `r` typically 8–64. Because it is a *delta*, you can keep the base weights resident once and apply each request's adapter as a small extra matmul in the same batch. So multiplexing is possible at all because the adapter is tiny relative to the base — for a 70B with r=16 on the attention projections, an adapter is on the order of tens of megabytes against 140 GB of base.

Mechanically, the engine batches requests from different adapters together and computes `xW` once for the whole batch (shared), then computes the per-request low-rank term with a grouped/segmented GEMM that indexes each row into its own adapter's `A` and `B`. Punica and S-LoRA are the reference works here; vLLM and SGLang both ship multi-LoRA support built on that idea, with adapters held in GPU memory and paged in from host memory on demand.

What breaks at fifty adapters:

**Adapter memory adds up.** Fifty × 60 MB = 3 GB of HBM that is no longer KV cache. On a box with 52 GB of usable cache that is a 6% concurrency haircut — acceptable. Five hundred adapters is 30 GB and is not.

**Cold adapter swap-in.** If an adapter is not resident, the first request for it stalls while it loads from host memory or disk. With a long tail of rarely-used tenants, your p99 is dominated by swap-ins. The fix is an LRU with pinned-hot adapters and a pre-warm on tenant activity, plus routing tenants to replicas that already hold their adapter — the same affinity problem as prefix caching, one level up.

**Throughput degrades with adapter diversity in the batch.** A batch where all 64 requests share one adapter runs the low-rank term as a single clean GEMM. A batch with 64 distinct adapters runs a maximally-scattered grouped GEMM with poor memory locality. Expect a real throughput hit — measure it, do not assume it is free. Adapter-aware batching (grouping same-adapter requests) recovers some of it at the cost of fairness.

**Rank heterogeneity.** Adapters with different `r` values complicate the batched kernel; some implementations require a uniform max rank. Standardize rank across your tenants in the training pipeline — this is a governance decision you make early or regret later.

**⚠ Trap:** offering per-customer fine-tuning as a product feature without capping adapter count, rank, and target modules. Every one of those is a memory and throughput liability on a shared fleet, and once customers have adapters you cannot change the rules. Fix `r`, fix which modules get adapted, and put a hard ceiling on adapters per replica before the first customer.

### Structured output at the engine level — how does constrained decoding work and what does it cost per token?

The mental model: the model produces a probability distribution over the whole vocabulary at every step; constrained decoding masks that distribution so only tokens that keep the output within a formal language survive. You are not asking the model to obey a schema, you are making disobedience unrepresentable.

The mechanism, concretely. Compile the JSON Schema (or regex, or grammar) into a finite-state machine or pushdown automaton over *token* sequences. At each step, given the current FSM state, compute the set of allowed next tokens, build a boolean mask over the vocabulary, add `-inf` to the logits of disallowed tokens, then sample. Advance the FSM by the sampled token.

The expensive part is computing that mask. A vocabulary is 128k–256k entries; determining which of them are legal continuations from an arbitrary FSM state is not trivial, and doing it naively per step in Python will cost more than the forward pass. The engineering that makes this viable: precompute the token→state transitions where possible (Outlines' key insight — for a regex the FSM is static, so you can build an index from state to allowed-token-set ahead of time), cache masks by state, and run the mask construction on GPU or in compiled code. XGrammar and similar libraries push this further with byte-level tries and by identifying tokens that are context-independent.

There is also a **jump-ahead / fast-forward** optimization worth naming: when the grammar admits exactly one continuation — after `{"na` the schema may force `me":` — you can emit those tokens without running the model at all. On a heavily-schema'd output like a tool call, this can skip a meaningful fraction of decode steps and is a *speedup*, not a cost.

**📐 Numbers you must know:** well-implemented constrained decoding costs single-digit percent overhead per token; a naive Python implementation can cost 50%+ and turn your decode step from memory-bound into CPU-bound. If you enable structured output and your throughput halves, you have a mask-construction problem, not a model problem — profile the CPU side of the step.

**⚠ Trap — the important one:** constrained decoding guarantees *syntactic* validity, not *semantic* correctness. The model will happily emit `{"amount": 0, "currency": "XXX"}` — schema-valid, factually wrong. Worse, aggressive constraints can degrade quality: if the model's natural high-probability path is blocked, you force it onto a lower-probability continuation, and on hard schemas this measurably hurts. My rule: constrain the *shape*, validate the *content* with Pydantic plus business rules, and always keep an eval that measures semantic accuracy under constraint versus unconstrained — because the second number is sometimes higher and you need to know that.

**🗣 Say this in the room:** "Constrained decoding masks logits against a compiled grammar, so invalid JSON becomes impossible rather than unlikely. It costs a few percent per token if the mask construction is compiled, and it can actually be net-faster on heavily-templated output because of jump-ahead. But it only buys syntax — I still validate semantics downstream and I keep an eval on whether the constraint hurt answer quality."
