### You said classical ML is a skim, not a skip. What exactly stays in?

Everything you keep serves one purpose: **being able to say "I would not use an LLM here" and be right.** That is the maturity signal interviewers explicitly listen for, and it is the highest ratio of hiring signal to study hours in the entire foundations layer. You are not being tested on gradient boosting internals. You are being tested on whether you reach for a 12B-parameter probabilistic text generator to parse a fixed-format invoice number.

Keep exactly four things.

**The substitution table.** For each common "AI feature" request, the deterministic alternative and the condition under which it wins. Fixed-format extraction → a regex or a real parser. Routing among ten known intents → an embedding plus a threshold, or a logistic regression over TF-IDF, trained in an afternoon. A known aggregation over structured data → SQL, always, with the LLM at most writing the query. A compliance gate → deterministic rules, with the LLM allowed to draft and a human to approve. The head of your query distribution → precomputed answers in a cache, because in most products the top 200 queries are 40–60% of traffic.

**💰 Math, the one to have ready:** intent classification at 1M requests/month. A frontier API call with a 600-token prompt and 5 output tokens at $3/M in and $15/M out costs 1M × (600×3 + 5×15)/1e6 = 1M × ($0.0018 + $0.000075) = **$1,875/month** at ~700 ms p50. A scikit-learn logistic regression over sentence embeddings: embeddings at ~$0.02/M tokens × 600M tokens = $12/month, inference in-process at ~2 ms, and if you cache embeddings for repeat queries it approaches free. **156× cheaper, 300× faster.** The LLM is the right answer only if the label set changes weekly or the long tail is genuinely open-ended — which is a real condition, and naming that condition is what makes the answer senior rather than contrarian. (**📅 Volatile:** prices move; the ratio is the durable part.)

**Metrics, because eval rounds reuse them wholesale.** Precision/recall/F1, macro vs micro averaging, PR-AUC vs ROC-AUC under class imbalance, and the retrieval set — Recall@k, MRR, nDCG. You will use these in RAG evaluation constantly. The specific tell interviewers watch for: reporting ROC-AUC on a moderation task where positives are 1% of traffic, where PR-AUC is the honest metric and ROC-AUC flatters you badly.

**Clustering for error analysis.** HDBSCAN over embeddings of your failure cases is how you turn 4,000 bad outputs into nine named failure modes. This is a working technique, not trivia — it is the single most useful classical-ML tool in an applied AI job.

**⚠ Trap:** skimming this into contrarianism. "Just use a regex" is as bad an answer as reflex-LLM if the input is genuinely unstructured natural language across 200 formats. The senior position is the *gate*, not a preference: state the condition, then choose.

**🗣 Say this in the room:** "Before I design this I'd want to know the cardinality of the output space and how often it changes. If it's a fixed set of twelve intents that changes quarterly, this is a classifier and an embedding cache — about $12 a month and 2 milliseconds. If the intent space is open and shifts weekly, then the LLM's zero-shot generality is worth paying 150× for, and I'd still put a cached classifier in front of it for the head of the distribution."

### How much CNN and RNN history do I actually need?

One page, and I mean that literally — you should be able to write the whole thing on one page and then never open it again. The reason it is on the skim list rather than the skip list is that a specific kind of interviewer, usually a research-adjacent one at Google or DeepMind or Meta, will ask "why did transformers win?" as a warm-up, and a candidate who cannot answer it in mechanistic terms reveals that their knowledge starts at the API.

Here is the one page.

**RNNs and their defect.** A recurrent net processes a sequence one step at a time, carrying a hidden state. Two consequences follow, and both are fatal. Sequentially, you cannot parallelize across the time dimension during training — step `t` needs step `t−1` — so your GPUs idle. And the gradient from position `T` back to position `1` is a product of `T` Jacobians; if their spectral radius is below 1, the signal vanishes exponentially, and if above 1, it explodes. **📄 Paper:** Hochreiter & Schmidhuber (1997) — LSTM added multiplicative gates and an additive cell-state path so gradients could flow without repeated multiplication, which mitigated vanishing but did not touch the parallelism problem. GRU (2014) is a cheaper two-gate variant of the same idea.

**Attention's actual origin.** **📄 Paper:** Bahdanau et al. (2014) — added a learned soft alignment so a translation decoder could look at *all* encoder states rather than a single fixed-size context vector. This is the moment attention was invented, and it was a fix for the seq2seq bottleneck, not a new architecture. Luong et al. (2015) simplified the scoring function. The transformer's contribution was to notice that if attention is good enough, you can **delete the recurrence entirely** and keep only attention — **📄 Paper:** Vaswani et al. (2017).

**Why transformers won, in two sentences.** Parallelism over the sequence dimension: every position's representation at layer `L+1` is computed simultaneously from all positions at layer `L`, so a training step is a few big matmuls instead of `T` sequential small ones, and GPUs are matmul machines. And O(1) path length: any two positions are one attention hop apart, so gradient signal between distant tokens does not decay with distance the way it does through `T` recurrent steps. The price is O(T²) attention cost, which the field has spent a decade paying and optimizing.

**Everything else on the page, as one-liners.** Teacher forcing and exposure bias — training on ground-truth prefixes, generating on your own prefixes, hence compounding error. word2vec/GloVe give one static vector per word type; contextual models give one per token occurrence, which is why "bank" finally works. BERT is encoder-only with masked-LM, good for classification and embeddings; GPT is decoder-only causal, good for generation; T5 is encoder-decoder with span corruption. CNNs: convolution is weight sharing plus locality; ResNet's skip connection is the same residual idea the transformer uses; ViT showed you can just cut an image into patches and feed a transformer.

**⚠ Trap:** saying transformers won "because attention is better than recurrence at capturing long-range dependencies." That is the folk answer and it is only half right — LSTMs handle long range tolerably. The decisive advantage was training throughput. If you get this wrong in front of someone who trains models, you have signalled that you learned this from a blog post.

### And GPU scheduling in Kubernetes — you called that delta-only. What's the delta list?

Short, and I would timebox it to two hours because past that you are studying an infra role you are not interviewing for.

The delta is five items. **Extended resources and the device plugin** — GPUs are integers, allocated whole, advertised by a DaemonSet, and the scheduler has no idea what a "GPU-hour" is. **Sharing modes** — MIG for hardware-partitioned isolation, time-slicing for oversubscription without memory isolation, MPS for concurrent kernels with weak isolation; know which failure each one permits. **Gang scheduling** — an 8-GPU distributed job must get all 8 or none, or you deadlock with two half-placed jobs each holding what the other needs; this is why Volcano, Kueue and similar batch schedulers exist on top of the default one. **Topology awareness** — 8 GPUs on one node connected by NVLink is a completely different machine from 8 GPUs spread across two nodes over PCIe and Ethernet; a collective operation on the latter can be an order of magnitude slower, so placement is a correctness-of-performance concern. **Node fragmentation and drain semantics** — evicting a pod holding a 90-second-to-load model is not the same as evicting a stateless web pod, so your PDBs and preStop hooks need to account for warm-up.

**⚠ Trap:** conflating training-cluster scheduling with inference-cluster scheduling in an answer. They have opposite objectives. Training wants gang-scheduled, long-lived, checkpointed, preemptible-with-resume jobs where the metric is cluster utilization. Inference wants low-latency, elastic, warm-pooled replicas where the metric is SLO attainment and idle capacity is deliberately purchased. Saying "we'd use Kubernetes for GPU scheduling" without picking one of these tells the interviewer you have done neither.

**🧪 Verify before you skip the rest:** (1) Two 4-GPU jobs, one 8-GPU node with 6 free GPUs across two nodes — what does the default scheduler do and why is it wrong? (2) Why does NVLink vs PCIe placement change your effective throughput on a tensor-parallel deployment? (3) What does `nvidia-smi`'s utilization percentage actually measure, and why is it a bad autoscaling signal?

### Of everything on the ATTACK list, why do you put transformer internals first?

Because it is the only item on the list where fifteen years of shipping software buys you nothing, and because it is *load-bearing* for three other items. You cannot reason about KV-cache budgeting without knowing what K and V are and why queries are not cached. You cannot understand why PagedAttention exists without understanding that the cache is per-sequence and grows. You cannot evaluate a quantization scheme without knowing where the outliers live in the activation distribution. Attack it first because everything downstream compiles against it.

"Attack" has a specific operational meaning here, and it is not "read more." It is: **you can produce the thing unaided, from memory, on a blank page, in a fixed time.** The pass criteria for this topic, concretely —

- Write single-head scaled dot-product attention with a causal mask in NumPy or PyTorch, ~15 lines, no reference, in under 10 minutes.
- Extend it to multi-head with correct reshapes and transposes, writing `[B, H, T, d_h]` shapes at every line, in another 10.
- Derive why the scale is √d_k from the variance of a dot product of two `d_k`-dimensional vectors with unit-variance components (the dot product has variance `d_k`, so dividing by √d_k restores unit variance and keeps softmax out of its saturated regime).
- Explain a full block's data flow — embed, RoPE, attention, residual, RMSNorm, SwiGLU FFN, residual — and say which parts are per-token and which are per-pair.
- State parameter counts: attention is ~4·d² per layer (Q, K, V, O), FFN is ~2·d·d_ff, and with the standard `d_ff = 4d` that is 8d², so **FFN is roughly two-thirds of the parameters** and attention one-third. That single ratio kills a lot of muddled thinking.

**🏋 Drill:** 25 minutes, no autocomplete, no reference. Blank file. Implement multi-head causal self-attention, then a full transformer block with RMSNorm and a SwiGLU FFN. Then, on paper, compute the KV cache bytes per token for a model with 32 layers, 8 KV heads and `d_head` 128 in bf16. Pass criterion: the code runs on a random `[2, 16, 512]` input and produces the right output shape, AND your KV number is exactly 128 KiB with the derivation written out. Repeat weekly until it is boring.

**⚠ Trap:** learning the transformer through a diagram instead of through shapes. Diagrams with boxes and arrows produce a candidate who can *narrate* attention and cannot *write* it. The distinguishing question at the whiteboard is always a shape question — "what's the shape after the transpose?" — and the diagram does not encode it.

### Same question for KV-cache math. Why does that get its own attack slot rather than living inside transformers?

Because it is the highest-frequency arithmetic in the entire interview surface and it is asked in a distinctive way: **out loud, in your head, with the interviewer watching.** It shows up in serving rounds, in system design rounds, in cost questions, in capacity questions. A candidate who can do it fluently reads as someone who has operated a model; one who reaches for a calculator reads as someone who has read about one.

The core formula is short enough to be permanent: `bytes_per_token = 2 · n_layers · n_kv_heads · d_head · dtype_bytes`. The leading 2 is K and V. Notice what is absent — `d_model`, query head count, `d_ff`, and total parameter count. That absence is the whole point, and it is why GQA works.

**📐 Numbers you must know:** anchor on Llama-3-8B — 32 layers, 8 KV heads, `d_head` 128, bf16. `2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB per token`, so **1 GiB per 8k-token sequence**, exactly. Every other model is a ratio from that anchor. Memorize the anchor, derive the rest.

**💰 Math you should be able to run in 30 seconds under questioning:** an 80 GB H100 serving Llama-3-8B in bf16. Weights: 8B × 2 bytes = 16 GB. Reserve ~4 GB for activations, CUDA context and fragmentation. Free for KV: 80 − 16 − 4 = **60 GB**. At 128 KiB/token that is 60 × 1024 × 1024 / 128 = **491,520 tokens of cache**. At an average 4k-token context per request, that is ~120 concurrent sequences. Push average context to 32k and it collapses to ~15. **The same hardware, the same model, an 8× concurrency swing driven entirely by context length** — and that is the number product managers never see when they ask to "raise the context limit."

**⚠ Trap:** computing the single-sequence number, finding it comfortable, and concluding you have capacity. The KV cache is the one resource in a serving deployment that scales with traffic *and* context *multiplicatively*. Engineers who size on the single-sequence figure ship a service that OOMs at a concurrency the load test never reached because the load test used short prompts.

**🗣 Say this in the room:** "Decode is memory-bandwidth-bound, not compute-bound, and the KV cache is why. It's a per-request memo table whose eviction policy I don't control, it doesn't amortize across the batch the way weights do, and it grows linearly with context — so my capacity model is `free_HBM / (bytes_per_token × p95_context)`, not requests per second."

### Serving-engine internals is on the attack list, but I'm not going to be writing vLLM. Why does it matter for an applied role?

Two reasons, and the second is the real one.

The first is that you will make deployment decisions that depend on it: whether to self-host or use an API, which engine, what batch and context limits to configure, what to autoscale on, whether speculative decoding is worth the complexity for your workload. Those decisions have five- and six-figure consequences and they are unanswerable without the internals.

The second reason is that **this is the round where your backend experience converts into visible seniority, and nothing else on the list does that as efficiently.** Every serving concept is a systems concept you already own, wearing an ML costume:

- **Continuous batching** is a work-stealing scheduler that admits new requests at token granularity instead of waiting for the whole batch to finish. **📄 Paper:** Yu et al. (2022), Orca — introduced iteration-level scheduling, replacing static batching where a batch of 32 runs at the speed of its longest member. You have built the equivalent for a job queue.
- **PagedAttention** is virtual memory for the KV cache. **📄 Paper:** Kwon et al. (2023), vLLM — the KV cache is stored in fixed-size blocks with an indirection table instead of one contiguous per-sequence buffer, which eliminates internal fragmentation from over-allocating to `max_seq_len` and enables copy-on-write sharing of a common prefix. If you have ever explained why a slab allocator beats malloc for fixed-size objects, you already have this argument; only the nouns change.
- **Prefix caching / RadixAttention** is a shared-prefix trie over KV blocks, so a 12k-token system prompt reused across 200k daily calls is computed once. **📄 Paper:** Zheng et al. (2024), SGLang — RadixAttention with an LRU eviction policy over the radix tree.
- **Prefill vs decode** is the single most important asymmetry: prefill processes all input tokens in parallel and is compute-bound; decode emits one token at a time and is memory-bandwidth-bound. They contend for the same GPU, which is why a long prefill stalls everyone else's inter-token latency, and why disaggregating them onto separate pools is now a mainstream architecture.

**🗣 Say this in the room:** "PagedAttention is virtual memory for the KV cache — fixed-size blocks plus a page table, so you stop pre-allocating to max context and stop fragmenting. The win isn't a faster kernel, it's that you can run 2–4× more concurrent sequences on the same card because you're no longer wasting the gap between allocated and used."

**⚠ Trap:** describing continuous batching as "dynamic batching." Dynamic batching (the Triton-style kind) waits a few milliseconds to assemble a batch and then runs it to completion — every request in that batch finishes when the longest one does. Continuous batching evicts and admits *between token steps*. The difference in tail latency is enormous, and using the wrong term signals you learned this from a vendor page.

### Quantization made the attack list too. What's the bar there for an applied engineer?

The bar is: you can say which quantization scheme you would use, what it costs you in quality, where the risk actually lives, and how you would *measure* whether it was acceptable. You are not expected to write a CUDA kernel.

The mental model that makes everything else fall out: **quantization is a lossy compression of the weights and/or activations, and it buys you memory bandwidth, not FLOPs.** Since decode is bandwidth-bound, halving the bytes you stream per token roughly halves your decode time — that is the entire economic case. It also buys you HBM, which converts directly into KV cache headroom and therefore concurrency, which is often the larger win.

The taxonomy worth holding:

- **Weight-only, post-training.** GPTQ (**📄 Paper:** Frantar et al., 2022 — layer-wise second-order-informed rounding) and AWQ (**📄 Paper:** Lin et al., 2023 — protect the ~1% of weight channels that matter most, identified by activation magnitude) take a trained model to 4-bit weights with small quality loss and no retraining. This is the default for self-hosting.
- **Weight-and-activation.** Harder, because activations have outliers. **📄 Paper:** Dettmers et al. (2022), LLM.int8() — found that a small number of outlier feature dimensions dominate and must be kept in higher precision. **📄 Paper:** Xiao et al. (2022), SmoothQuant — migrates activation outlier difficulty into the weights via a per-channel scaling, making both quantizable. FP8 on Hopper-class hardware and FP4 on Blackwell-class make this much easier because the format has an exponent and handles outliers gracefully. (**📅 Volatile:** hardware format support moves generation to generation; verify.)
- **Quantization-aware training and QLoRA.** **📄 Paper:** Dettmers et al. (2023), QLoRA — 4-bit NF4 base weights frozen, LoRA adapters trained in bf16 on top, which is what makes fine-tuning a large model on one GPU feasible.
- **KV-cache quantization.** Separate axis, often the highest-value one. Taking the cache from fp16 to fp8 halves bytes per token and therefore doubles concurrency.

**💰 Math:** Llama-3-70B in bf16 is 140 GB — two 80 GB cards minimum, with almost no room for KV. At 4-bit it is ~35 GB, which fits on one card with ~40 GB left for cache. At 320 KiB/token that is 40 × 1024 × 1024 / 320 = **131,072 tokens** of cache, or ~32 concurrent 4k sequences on a single card. You went from a 2-GPU deployment serving few sequences to a 1-GPU deployment serving many. On a $2/GPU-hour rental that is $1,440/month saved on hardware alone, before the throughput gain.

**⚠ Trap:** benchmarking a quantized model on perplexity and declaring victory. Perplexity is remarkably insensitive to quantization damage; the degradation concentrates in long-context reasoning, multi-step arithmetic, code generation, and — most treacherously — instruction adherence and tool-call format compliance. I have seen a 4-bit model with near-identical perplexity emit malformed tool arguments at 3× the rate. **The rule I enforce: quantization is validated on your task eval, not on a language-modelling metric.**

### Post-training and RLVR — how deep does an applied AI engineer need to go?

Deep enough to hold a real conversation and to *not fine-tune*. That framing sounds glib; it is the actual bar.

You need the map, not the terrain. Pretraining gives you a next-token predictor. Supervised fine-tuning on instruction/response pairs gives you something that follows instructions. Preference optimization aligns it to human taste: **📄 Paper:** Ouyang et al. (2022), InstructGPT — RLHF as reward model plus PPO, which is what made GPT-3.5 usable; **📄 Paper:** Rafailov et al. (2023), DPO — showed the RLHF objective has a closed form that lets you optimize directly on preference pairs with no separate reward model and no RL loop, which is why most teams outside frontier labs use DPO. And then the 2024–2025 shift: **RLVR**, reinforcement learning from *verifiable* rewards, where the reward is not a learned preference model but a program — unit tests pass, the math answer matches, the JSON parses, the SQL returns the right rows. **📄 Paper:** Shao et al. (2024), DeepSeekMath — introduced GRPO, which drops the value network and normalizes advantages within a group of sampled completions, making RL affordable. DeepSeek-R1 (2025) showed rule-based verifiable rewards at scale producing long-chain reasoning behaviour.

Why RLVR matters to *you*, in an applied role, is the thing to actually say: **RLVR's constraint is that you need a verifier, and building verifiers is engineering, not research.** That is your job. If your product domain has a checkable ground truth — code that compiles and passes tests, a query that returns the right rows, an extraction that matches a database — you have the substrate for RL. If your domain is "write a good summary," you do not, and no amount of GPU budget fixes that. Being able to draw that line is the whole competency.

**⚠ Trap — the reflex-fine-tuning rejection trigger.** The single fastest way to be downgraded in an applied AI loop is to answer "the model isn't good at our domain" with "we'd fine-tune it." It is the answer of someone who has read about ML and not shipped it. Fine-tuning teaches *form*, not *facts*: it is excellent at style, format, tone, tool-call conventions and domain vocabulary, and it is a bad and expensive way to install knowledge that changes — that is retrieval's job. Before fine-tuning is legitimate you need, named explicitly: a working eval, a prompt/context baseline you have already exhausted, at least a few thousand high-quality examples, a data-refresh story, and a serving story for the adapter.

**🗣 Say this in the room:** "I'd hold fine-tuning until last. In my experience the first 80% of a domain gap is prompt and context engineering, the next 15% is retrieval and tool design, and fine-tuning is for the residual — consistent output format, house style, a tool-calling convention the base model keeps getting wrong. It teaches the model how to behave, not what's true, and anything that changes weekly has to come from retrieval or it's stale the day after you train."

### Why do you call evaluation "the spine" rather than one more topic?

Because it is the only topic that appears in every other round. A retrieval design answer without an eval is a guess. A prompt-optimization answer without an eval is superstition. A model-migration answer without an eval is a coin flip with your production traffic. And empirically, "skipped evaluation of AI outputs" is the named #1 critical error on take-home rubrics — which means it is not a knowledge gap, it is a *habit* gap, and habits need to be built early rather than crammed.

The habit is: **open every design answer with how you would know whether it works.** Not close with it. Open with it. When an interviewer says "design a support agent," the first sixty seconds should be "before I pick an architecture, here's what I'd measure and how I'd build the dataset" — because that reorders everything downstream and it immediately separates you from candidates who describe a box diagram.

What to actually attack, in order:

**Dataset construction**, because it is the part everyone skips. Where do the 200 labelled examples come from? Production logs, stratified by intent and by whether the current system succeeded. Who labels them and what is the inter-annotator agreement? What is your holdout discipline, and are you sure your eval set has not leaked into your prompt? A candidate who talks about dataset construction before metrics has already passed.

**Metric selection under the task.** Exact match and pass@k where there is ground truth. Retrieval metrics (Recall@k, nDCG, MRR) for the retrieval stage measured *separately* from generation, because otherwise you cannot localize a regression. LLM-as-judge where there is no ground truth, with its own validation.

**LLM-as-judge, validated.** **📄 Paper:** Zheng et al. (2023) — established judge models as a scalable proxy and documented their biases: position bias (favouring the first response), verbosity bias (favouring longer answers), self-enhancement bias (favouring their own family's outputs). The discipline is: pairwise beats absolute scoring, randomize position, calibrate the judge against a few hundred human labels and report the agreement rate, and re-calibrate whenever you change judge model. A judge you have not measured against humans is a random number generator with good prose style.

**Statistics.** You are comparing two prompts on 200 examples and one scores 74% and the other 78%. Is that real? Paired bootstrap or McNemar's test on the paired outcomes, because the examples are shared. Without this you will ship noise and then debug the wrong thing for a week.

**🏋 Drill:** 30 minutes. Take any feature you have shipped in the last year and write its eval plan on one page: the task, the dataset and its provenance, the primary metric and why, the guardrail metrics, the sample size needed to detect a 3-point difference, and the failure taxonomy you would tag against. Pass criterion: someone else could execute it without asking you a question.

### Agent harness engineering is the last big attack item. What does "harness" mean, and why isn't this just framework knowledge?

The harness is everything around the model call: the loop, the tool registry and schemas, the context assembly, the error surfaces, the termination conditions, the budget accounting, the persistence, the replay. The model is one function call inside it. The framing I would use in a room is that **roughly 20% of a production agent system is the LLM; the other 80% is engineering you already know how to do** — and the interview is testing whether you know which 80%.

It is not framework knowledge because the frameworks make the easy part easy and leave the hard part to you. LangGraph will give you a state machine; it will not tell you that your tool error strings are the highest-leverage prompt surface in the system, or that you need a per-run token budget with a hard kill, or that a tool returning 40k tokens of JSON has just destroyed your context and your latency. The core loop itself is small enough to write from memory, and you should be able to:

```python
# The whole pattern, ~25 lines. If you can write this, framework choice is a detail.
def run(task, tools, max_steps=12, token_budget=60_000):
    messages = [{"role": "user", "content": task}]
    used = 0
    seen = collections.Counter()
    for step in range(max_steps):
        resp = model.call(messages, tools=[t.schema for t in tools])
        used += resp.usage.input_tokens + resp.usage.output_tokens
        if used > token_budget:
            return Halt("budget", messages)
        if not resp.tool_calls:
            return Done(resp.text, messages)
        messages.append(resp.assistant_message())
        for call in resp.tool_calls:
            key = (call.name, canonical(call.args))
            seen[key] += 1
            if seen[key] > 3:                      # poison-trajectory guard
                return Halt("loop", messages)
            try:
                out = tools[call.name].run(**call.args)
            except ToolError as e:
                out = f"Error: {e}. {tools[call.name].recovery_hint}"   # error-as-prompt
            messages.append(tool_result(call.id, truncate(out, 4000)))
    return Halt("max_steps", messages)
```

Everything interesting is in that snippet's guards: the budget check, the repeat-call counter, the error string written as an instruction, the truncation. Those are the four things that separate a demo from a system, and none of them is in a framework tutorial.

**⚠ Trap:** believing agent quality is a prompt problem. In production the dominant failure causes are, in my experience and in that order: tool schemas the model misreads, tool errors returned as opaque codes, context blown out by a verbose tool result, and no termination guard. All four are engineering defects with engineering fixes. Candidates who answer "we'd improve the system prompt" to an agent-failure question are describing the least effective lever.

### What about research literacy and product sense — how do you attack something that soft?

They are the two items where "attack" means building a repeatable artifact rather than accumulating knowledge, so treat both as habits with a weekly cadence.

**Research literacy** is not "have you read every paper." It is: given a new paper, can you extract in fifteen minutes what it replaces, what it costs, and whether it changes any decision you would make? The artifact is a one-paragraph template you fill for every paper you read — *what was the state of the art before; what is the single mechanism change; what does it cost in compute, memory or quality; what would have to be true for me to adopt it.* Do that for the twenty papers this guide names and you have research literacy, because interviewers do not quiz you on paper contents, they ask "have you seen anything interesting lately?" and grade the shape of your answer.

The genuinely useful move here is knowing the *lineage* rather than the leaves: attention → transformer → scaling laws → Chinchilla's compute-optimal correction → instruction tuning → RLHF → DPO → RLVR is one coherent story, and being able to tell it as a story with the pressure that motivated each step is far more impressive than reciting eight abstracts.

**Product sense** is the round Cursor, Notion, Figma, Sierra, Harvey, Glean and Perplexity weight most heavily and that backend engineers underrate most badly. It is testable, and the test is usually one of three questions: "what would you build next for us," "what AI product do you think is badly designed and why," or "how would you know this feature was working." The competency underneath all three is *knowing which failure modes users forgive and which they do not*. Users forgive latency far more than they forgive confident wrongness. They forgive a refusal more than a fabrication. They forgive a bad answer they can see is bad; they do not forgive a plausible one that is wrong. This asymmetry should drive your design choices — it is why citations, showing retrieved sources, and "I don't know" paths are product features and not safety theatre.

**🏋 Drill:** 15 minutes weekly, unaided. Pick one AI product you use. Write: the three failure modes you have personally hit, which of them the team clearly knows about (evidenced by UI affordances), what you would instrument to catch the third one, and one feature you would cut. Pass criterion: no sentence in it could apply to a different product.

**🗣 Say this in the room** when asked what you would build: "I'd start from the failure mode users punish hardest, which for this product is confident wrongness on a question they can't verify. So the first thing I'd ship isn't a feature, it's the instrumentation to tell how often that happens — because right now nobody in this conversation, including me, can name the number."
