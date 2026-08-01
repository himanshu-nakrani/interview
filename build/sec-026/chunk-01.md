### Derive the KV cache size formula for me from first principles. I want to see where every term comes from.

Start from what attention actually needs and the formula writes itself. At decode step $t$ the model has exactly one new token. It projects that token into a query vector, and then it must attend over the keys and values of *every* token before it. Those earlier keys and values are pure functions of tokens that have already been fixed — they cannot change, because the model is causal. So recomputing them every step is redundant work. The KV cache is the memo table for that redundancy. Framed in terms you already own: it is a per-request memoization of an append-only, immutable computation, except the eviction policy is not yours and the table lives in HBM.

Now count bytes. For each transformer layer, each attention head keeps one key vector and one value vector per token, each of dimension `d_head`. That gives you:

```
bytes = 2 · n_layers · n_kv_heads · d_head · seq_len · batch · dtype_bytes
```

Term by term. The **2** is K and V — two tensors, not one; this is the term people drop and then wonder why their measured usage is exactly double their estimate. **n_layers** because every layer has its own attention and its own cache; nothing is shared across depth in a vanilla transformer. **n_kv_heads**, *not* `n_attention_heads` — with grouped-query attention several query heads share one KV head, and this is the single largest lever in the whole formula. **d_head** is the per-head dimension, usually `hidden_size / n_attention_heads` (128 for essentially every Llama-family model). **seq_len** is prompt + generated so far, which is why the cache grows *during* the request. **batch** because caches are per-sequence and share nothing. **dtype_bytes** is 2 for fp16/bf16, 1 for fp8, 0.5 for int4.

Two things that are *not* in the formula, and interviewers probe both. There is no `seq_len²` term — the attention *score* matrix is quadratic in compute, but it is never materialised and never cached (FlashAttention exists precisely so that it isn't). And there is no `d_model` term except as it hides inside `n_kv_heads · d_head`; a wide-but-shallow model and a narrow-but-deep model with the same product cost the same cache.

**🗣 Say this in the room:** "Two times layers times KV heads times head-dim times dtype gives you bytes per token; multiply by sequence length and batch. The two is K and V, and the heads term is KV heads not query heads — that's where GQA buys its 4× to 8×."

**⚠ Trap:** using `num_attention_heads` from the config file. On Llama-3-70B that is 64 instead of 8 and you will size your cluster 8× too large. The correct field is `num_key_value_heads`, and when it is absent from the config the model is multi-head and the two are equal.

### Compute the per-token KV footprint for Llama-3-70B and tell me what a single 8k conversation costs.

Llama-3-70B's config: 80 layers, 64 attention heads, 8 key-value heads, hidden size 8192, so `d_head = 8192 / 64 = 128`. In bf16:

```
2 · 80 · 8 · 128 · 2 bytes = 327,680 bytes/token = 320 KiB/token ≈ 0.31 MB/token
```

That number — **0.31 MB per token for a 70B** — is the single most useful constant in serving. Memorise it as 320 KiB and you can do most capacity questions in your head.

An 8,192-token conversation therefore holds `8,192 × 327,680 = 2,684,354,560 bytes` = exactly **2.5 GiB**. One user. One conversation. Two and a half gigabytes of HBM, held for the entire duration of the session, doing nothing but existing so the model doesn't have to recompute it.

Scale that and the implications land immediately. A 4×H100 node has roughly 200 GiB left for KV after weights and overheads (I derive that below), so 200 / 2.5 = **80 concurrent 8k sessions**. At 32k context: 10 GiB each, 20 sessions. At 128k: 40 GiB each, **five** sessions on a node that cost you roughly $250k. That is the whole reason long context is expensive, and it is expensive in a way that has nothing to do with the price of FLOPs.

**📐 Numbers you must know:** 0.31 MB/token for a 70B (80 layers, 8 KV heads, 128 head-dim, bf16); 0.125 MB/token for an 8B (32 layers, 8 KV heads, 128, bf16). Both derive from the same formula in ten seconds — never memorise them as bare facts, memorise the shapes and rederive. If you quote a number you cannot rederive on the whiteboard when the interviewer changes the dtype, you have failed the question you appeared to pass.

**💰 Math:** at an on-demand H100 rate of ~$2–3/GPU-hour (**📅 Volatile:** spot and reserved rates move quarterly; verify before your loop), a 4×H100 node is $8–12/hour. Divided across five concurrent 128k sessions, that is $1.60–2.40 per session-hour of pure *idle holding cost* — before a single token is generated. This is why "keep the agent's context warm between turns" is a business decision, not an engineering convenience.

### Why is the KV cache linear in sequence length when attention itself is quadratic?

Because the quadratic object is transient and the linear object is persistent, and modern kernels make sure the quadratic one never exists in memory at all.

At decode step $t$, the query is a single vector of shape `[batch, n_heads, 1, d_head]`. The scores it produces against the cache are shape `[batch, n_heads, 1, t]` — linear in $t$, one row. The full $T \times T$ score matrix only appears conceptually during prefill, and FlashAttention's whole contribution is to tile that computation so the matrix is never materialised in HBM: you stream tiles of K and V through SRAM and maintain a running softmax. So attention's *compute* is $O(T^2)$ and its *memory* is $O(T)$.

The confusion is worth naming because it drives a real design mistake. People conclude "long context is quadratic, therefore doubling context quadruples my memory," size for that, and get a cluster that is oversized in HBM and undersized in FLOPs. The truth is the reverse of the intuition in each phase: **memory scales linearly with context; prefill compute scales quadratically.** Doubling context doubles your KV footprint and roughly quadruples your prefill time.

**📐 Numbers you must know:** for a 70B, the attention FLOPs equal the FLOPs of everything else in the model at a sequence length of about **107k tokens**. Derivation: the linear (weight-matmul) part of a forward pass is $2 N S$ FLOPs for $N$ parameters and $S$ tokens; the attention part with causal masking is about $2 \cdot n_{layers} \cdot S^2 \cdot d_{model}$. Setting them equal: $S = 2N / (2 \cdot n_{layers} \cdot d_{model}) = 1.4\times10^{11} / (160 \times 8192) = 106{,}800$. Below 100k, attention is a rounding error on the FLOP bill. Above 500k, attention *is* the bill.

**📄 Paper:** Dao et al. (2022), FlashAttention — IO-aware exact attention that keeps memory at $O(T)$ by tiling and online softmax, replacing the standard implementation that materialised the $T \times T$ matrix and made 4k context an OOM problem rather than a latency problem.

### Explain why multi-query and grouped-query attention exist. What do they cost you?

MQA and GQA are not accuracy techniques and they are not compute techniques. They exist for exactly one reason: `n_kv_heads` is a free multiplier in the KV-cache formula, and shrinking it shrinks the cache proportionally with almost no effect on the FLOP count, because K and V projections are a small fraction of the parameters.

In vanilla multi-head attention, 64 query heads pair with 64 key heads and 64 value heads. Shazeer's observation was that the *query* heads are where the representational diversity lives; the keys and values can be shared. Multi-query attention takes that to the limit — one KV head for all queries, a 64× cache reduction. That turned out to be too aggressive: quality degrades and training becomes unstable. GQA is the interpolation: partition the query heads into $G$ groups, one KV head per group. Llama-3 uses $G = 8$, so 64 query heads share 8 KV heads — an 8× cache reduction.

Run the arithmetic on what that bought. Llama-3-70B with full MHA would be `2 · 80 · 64 · 128 · 2 = 2,621,440 bytes/token` = 2.5 MiB/token. An 8k conversation would be **20 GiB** instead of 2.5 GiB. On a 200 GiB KV pool that is 10 concurrent users instead of 80. GQA is the difference between a viable product and a science project, and it costs — per the GQA paper's own ablations — a small, sometimes barely-measurable quality delta relative to MHA, far smaller than MQA's.

The cost you actually pay in production is subtler and worth naming: GQA reduces the arithmetic intensity of the attention kernel itself. With 8 KV heads feeding 64 query heads, each loaded KV block is reused 8 times, which is good; but the total KV bytes moved per decode step drops, which means attention becomes a *smaller* share of decode time and weight streaming becomes a larger one. That shifts where your optimisation effort should go.

**📄 Paper:** Shazeer (2019), "Fast Transformer Decoding: One Write-Head Is All You Need" — introduced MQA, replacing per-head KV with a single shared head. **📄 Paper:** Ainslie et al. (2023), GQA — showed you can uptrain an existing MHA checkpoint into a GQA one with a small fraction of original pretraining compute, which is why GQA propagated across every open-weight family within a year rather than waiting for new pretraining runs.

**🗣 Say this in the room:** "GQA is a KV-cache technique that happens to live in the attention layer. It's an 8× cache reduction on Llama-3 for a quality delta small enough that everyone took it. If someone asks me to reduce KV memory and the model doesn't have GQA, that's the first thing I check — there's usually a GQA-uptrained variant of the same family."

### Walk me through multi-head latent attention. How does DeepSeek get to roughly 70 KB per token?

GQA reduces the number of KV heads. MLA asks a different question: why store K and V at all, when you could store a compressed latent and reconstruct them?

The mechanism is a low-rank bottleneck on the KV path. Instead of caching $K$ and $V$ per head, you cache a single compressed vector $c^{KV}$ of dimension `kv_lora_rank` per token per layer, produced by a down-projection of the hidden state. At attention time, up-projection matrices reconstruct the per-head K and V from that latent. Crucially, those up-projections can be algebraically folded into the query and output projections, so at inference you never actually materialise the full K and V — you attend directly against the latent. RoPE breaks that folding (it's position-dependent and doesn't commute with the up-projection), so MLA carves out a small "decoupled" RoPE key of its own dimension that is cached separately and un-compressed.

For a DeepSeek-V3-class configuration — 61 layers, `kv_lora_rank` 512, decoupled RoPE dim 64 — the cache is `61 × (512 + 64) × 2 bytes = 70,272 bytes/token ≈ 68.6 KiB/token`. Note there is **no factor of 2**: you store one latent, not a K and a V. Against a 70B dense GQA model's 320 KiB/token, that is a 4.7× reduction on a model with almost 10× the parameters. At 128k context, one session is 8.6 GiB instead of 40 GiB.

The catches are real. MLA is a *pretraining* architecture choice — you cannot retrofit it to an existing checkpoint the way GQA could be uptrained. It requires a bespoke attention kernel; the generic FlashAttention path does not apply, and for a long stretch after DeepSeek-V2 the open-source kernels were meaningfully behind hand-tuned ones. And the folding trick means your inference-time weight layout differs from your training-time layout, which is a class of bug that produces subtly wrong outputs rather than crashes.

**📄 Paper:** DeepSeek-AI (2024), DeepSeek-V2 — introduced MLA, replacing GQA's "fewer heads" approach with "low-rank latent plus decoupled RoPE," and reported a large reduction in KV cache per token relative to a comparable dense model with MHA.

**⚠ Trap:** claiming MLA "compresses the KV cache" as if it were a post-hoc quantiser. It is an architectural change baked into pretraining. In an interview, saying "we'd switch to MLA to cut our cache" for a model you're already serving is a tell that you don't know where the technique lives in the stack.

### Account for every byte on an 80 GB H100 running a large model. Where does it all actually go?

Five buckets, and if you cannot name all five, your capacity estimate will be optimistic by 15–25% — which in practice means you OOM at 3am under load rather than in staging.

**1. Model weights.** Parameter count × bytes per parameter, divided by tensor-parallel degree. A 70B in FP8 is $70\times10^9$ bytes = 65.2 GiB; at TP=4 that is 16.3 GiB per card. In bf16 it is 130.4 GiB, 32.6 GiB per card. I count memory in GiB ($2^{30}$) because that is what `nvidia-smi` and `torch.cuda.memory_allocated` report, and parameters in decimal billions, because that is how models are named. Mixing the two silently is a classic 7% error.

**2. CUDA context, driver, and communication buffers.** The CUDA context alone is ~300–500 MiB per process before you allocate anything. NCCL registers pinned buffers for collectives — with TP=4 and 80 layers you are doing 160 all-reduces per forward pass, and the buffers for those are real. Budget **1–1.5 GiB per card** and verify with an empty-model `nvidia-smi` reading.

**3. Activations and workspace.** At decode this is negligible: batch × hidden × a few buffers, single-digit MiB. At **prefill** it scales with `max_num_batched_tokens`. The largest transient is the MLP intermediate: for Llama-70B, `d_ff = 28,672`, so 8,192 batched tokens in bf16 is `8192 × 28672 × 2 = 448 MiB` per buffer, ÷4 for TP = 112 MiB per card, and you need several live at once. Add cuBLAS/cuDNN workspace. **Budget 3–5 GiB per card at `max_num_batched_tokens = 8192`**, and understand that it is roughly linear in that knob.

**4. Fragmentation headroom.** The caching allocator does not hand back a perfectly packed heap. PagedAttention removes fragmentation *within* the KV pool (that's its entire point), but everything outside the pool still fragments. This is why every engine ships a utilisation fraction below 1.0.

**5. Everything left over is the KV pool.** That is the residual, and it is what determines your concurrency.

Worked, for Llama-3.3-70B FP8 on 4×H100 SXM5, `gpu_memory_utilization=0.90`:

```
per card:  79.6 GiB reported × 0.90            =  71.6 GiB budget
           − weights (65.2 GiB / 4)            =  16.3
           − CUDA context + NCCL               =   1.2
           − activations @ 8192 batched tokens =   4.0
           ─────────────────────────────────────────
           KV pool per card                    =  50.1 GiB
           × 4 cards                           = 200.4 GiB total
```

**🗣 Say this in the room:** "Weights, KV, activations, CUDA context, fragmentation headroom. The first two you compute exactly, the third scales with your prefill token budget, the fourth is about a gig a card, and the fifth is why nobody runs at utilisation 1.0. KV is the residual — which is why every capacity conversation is really a conversation about what you spent on the other four."

### Given that budget, how many concurrent users fit? And what happens at 128k context?

Take the 200.4 GiB KV pool from the previous derivation and divide by per-session footprint. At 0.3125 MiB/token, the pool holds `200.4 × 2^30 / 327,680 = 656,670` tokens — call it **655k tokens of live cache**, total, across all users.

| Context per session | KV per session | Max concurrent sessions |
|---|---|---|
| 4k | 1.25 GiB | 160 |
| 8k | 2.5 GiB | 80 |
| 32k | 10 GiB | 20 |
| 128k | 40 GiB | **5** |
| 1M | 320 GiB | **0 — does not fit at all** |

Two conclusions that should reshape how you think about the product.

First, **your concurrency limit is a token budget, not a user count.** 655k tokens is the resource. Whether that is 80 chat users or 5 document-analysis users is a product decision, and it is the right unit for capacity planning, rate limiting, and admission control. Stop thinking "how many requests per second" and start thinking "how many live tokens."

Second, the 1M row is not a typo. A single 1M-token request needs 320 GiB of bf16 KV, which exceeds the *entire* HBM of a 4×H100 node including weights. To serve 1M context on that node you must quantise KV to FP8 (160 GiB — fits, at 80% of pool, serving exactly one user) or move to a larger topology. When a PM says "the model supports 1M context, let's expose it," this table is your answer.

**⚠ Trap:** sizing on *average* context length. Context length distributions in production are viciously right-skewed — a support bot's median might be 1,200 tokens with a p99 of 60k because someone pasted a log file. Your KV pool is consumed by the p99, not the median. The rule I enforce in review: size the pool on p95 context × target concurrency, and set a hard `max_model_len` that admission control enforces, so the p99.9 outlier is rejected at the gateway rather than evicting forty other users' caches.

### `gpu_memory_utilization` is set to 0.90 and we OOM under load anyway. Debug it.

This is the most common vLLM production incident and it has four causes, checkable in order.

**First: the utilisation fraction is of *total* memory, not free memory.** If anything else on that GPU has already allocated — a leftover process, a monitoring agent, a sidecar doing embeddings, a Triton server — the engine profiles free memory at startup, sizes its pool, and then that other process grows. You have overcommitted the card. Check `nvidia-smi` for other PIDs on the device before every deploy; in Kubernetes this means confirming you actually got an exclusive device and not a time-sliced or MPS-shared one. Time-slicing a GPU across two inference pods is the single most reliable way to produce this incident.

**Second: profiling ran with a smaller activation footprint than production.** The engine profiles peak memory during startup with a synthetic worst-case run. If your production traffic uses a feature that startup profiling didn't exercise — LoRA adapters loaded lazily, multimodal image encoders, a larger `max_num_batched_tokens` than the profiling shape, guided decoding with a large grammar's FSM — the peak is higher than what was measured. Everything that allocates outside the KV pool must be exercised at startup.

**Third: the KV pool is *not* where you OOM'd.** PagedAttention makes the pool a fixed pre-allocated arena; it cannot grow. So an OOM is almost never "we ran out of KV" — that manifests as preemption and queueing, not a crash. The OOM is in activations or a transient. Read the traceback: if it's inside the attention kernel or the sampler, it's activations, and the fix is lowering `max_num_batched_tokens`, not `gpu_memory_utilization`.

**Fourth: memory fragmentation outside the pool.** Long-running processes with variable-shape allocations fragment the caching allocator. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` genuinely helps here.

**🔍 Failure taxonomy:** OOM at startup → your arithmetic is wrong, recompute the five buckets. OOM within minutes of first traffic → activation shapes not covered by startup profiling. OOM after hours of stable operation → another process on the card, or allocator fragmentation. *Never* OOM but throughput collapses → that's KV pool exhaustion causing preemption, a completely different bug with a completely different fix.

**⚠ Trap:** responding to an OOM by lowering `gpu_memory_utilization` from 0.90 to 0.85. This shrinks the KV pool, which reduces concurrency, which reduces throughput — while doing nothing about an activation-side overflow. You have traded a loud failure for a quiet 15% capacity loss. Diagnose which bucket overflowed first.

### Why is prefill compute-bound and decode memory-bandwidth-bound? Derive the arithmetic intensity of each.

This is the single most important fact in LLM serving and almost every design decision downstream is a consequence of it. Arithmetic intensity is FLOPs performed per byte moved from HBM. Compare it to the hardware's ridge point — peak FLOP/s divided by peak bandwidth — and you know which resource you're pinned on.

**Prefill.** You push $B$ sequences of $S$ tokens through the model. FLOPs $\approx 2 N B S$ for $N$ parameters (the factor 2 is multiply-add). Bytes moved: you read each weight exactly once, $N \cdot \texttt{dtype}$. In bf16:

$$\text{AI}_{\text{prefill}} = \frac{2NBS}{2N} = BS$$

For a single 2,048-token prompt, that's **2,048 FLOP/byte**. Each weight byte you drag out of HBM does two thousand FLOPs of work. You are drowning in arithmetic and starved for nothing.

**Decode.** One token per sequence. FLOPs $\approx 2NB$. Bytes: still the whole weight matrix, $N \cdot \texttt{dtype}$, plus the KV you read.

$$\text{AI}_{\text{decode}} \approx \frac{2NB}{2N} = B$$

**Decode arithmetic intensity is approximately the batch size.** At batch 1 you are doing one FLOP per byte. That is the entire story.

Now the hardware. An H100 SXM5 does roughly 989 TFLOP/s dense bf16 against 3.35 TB/s of HBM3 bandwidth. Ridge point:

$$\frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \text{ FLOP/byte}$$

So prefill at AI 2,048 sits far to the right of the ridge — **compute-bound**. Decode at AI = batch sits far to the left until batch reaches ~295 — **memory-bandwidth-bound**, by a factor of nearly 300 at batch 1. In FP8, peak roughly doubles to ~1,979 TFLOP/s and the ridge moves to ~590 FLOP/byte, making decode *relatively worse*: faster tensor cores don't help a phase that's waiting on memory.

**📐 Numbers you must know:** H100 SXM5 ridge point ≈ 295 FLOP/byte in bf16, ≈ 590 in FP8, from 989/1979 TFLOP/s over 3.35 TB/s. Decode AI ≈ batch size. Therefore **decode needs a batch of ~300 to stop wasting the GPU**, which is precisely why continuous batching exists, why prefix caching matters, and why single-user self-hosted inference is economically absurd.

**🗣 Say this in the room:** "Prefill's arithmetic intensity is batch times sequence length — thousands. Decode's is just batch size. The H100's ridge point is around 295 FLOP per byte, so decode at batch 1 is running the GPU at well under one percent of peak FLOPs. Everything in serving — continuous batching, chunked prefill, disaggregation, speculative decoding — is a scheme to fix that one number."

### If decode arithmetic intensity is just the batch size, what's the floor on single-stream tokens per second for a 70B?

You can answer this without running anything, and being able to is a strong signal.

Every decode step must stream the entire weight matrix out of HBM at least once — there is no way around it, because every parameter participates in producing the token. So:

$$t_{\text{token}} \geq \frac{\text{weight bytes per GPU}}{\text{HBM bandwidth per GPU}}$$

For 70B in FP8 on 4×H100 with TP=4: each card holds $65.2/4 = 16.3$ GiB = $1.75\times10^{10}$ bytes, and reads it at $3.35\times10^{12}$ B/s:

$$t \geq \frac{1.75\times10^{10}}{3.35\times10^{12}} = 5.2 \text{ ms} \Rightarrow \leq 191 \text{ tok/s}$$

In bf16 it doubles to 10.4 ms, ceiling 96 tok/s. That is a hard physical ceiling for a single stream. No kernel, no compiler, no vendor will beat it.

Reality lands at 30–50% of that ceiling at TP=4, and the gap is worth understanding rather than hand-waving. Each layer requires two all-reduces across the TP group; 80 layers means 160 collectives per token, each with launch and synchronisation cost measured in tens of microseconds. Add per-kernel launch overhead — a 70B forward pass is hundreds of kernels — and at 5 ms of useful work the overhead is a large fraction. CUDA graphs recover a good chunk by replaying the whole decode step as one graph launch, which is why every serious engine captures graphs for the decode path.

**⚠ Trap:** believing more GPUs make a single stream proportionally faster. TP=8 halves the per-card weight bytes but doubles the collective count and shrinks the per-card work until you're launch-bound. Single-stream latency typically improves sublinearly from TP=4 to TP=8 and can regress. TP is a memory-capacity and aggregate-throughput lever far more than a single-stream-latency lever, and I push back hard on designs that assume otherwise.

**🗣 Say this in the room:** "Weight bytes per GPU over HBM bandwidth gives you the per-token floor. For a 70B FP8 on four H100s that's 5.2 milliseconds, so about 190 tokens per second is physics. If someone promises me 400, they've either quantised further, they're speculating, or they're measuring aggregate throughput and calling it latency."

### At what point does reading the KV cache cost more bandwidth than reading the weights?

This crossover is where serving intuition breaks for most people, and it explains why long-context workloads behave nothing like chat workloads on identical hardware.

Every decode step reads two things from HBM: the weights (once, shared across the whole batch) and the KV cache (once per sequence, proportional to each sequence's length). Weight bytes are constant. KV bytes are $B \cdot S \cdot \texttt{bytes\_per\_token}$. They cross when:

$$B \cdot S \cdot \texttt{bytes\_per\_token} = N \cdot \texttt{bytes\_per\_param}$$

For a 70B in FP8 with bf16 KV: $B \cdot S = 70\times10^9 / 327{,}680 = 213{,}600$ tokens.

**Above roughly 214k tokens of live cache, you are spending more HBM bandwidth on KV than on weights.** Batch 64 at 3,340 tokens each. Or batch 8 at 27k each. Or one 214k-token agent session.

Work a concrete step. Batch 64, 8k context each = 524,288 live tokens × 327,680 B = 171.8 GB = 160 GiB of KV. Aggregate bandwidth across 4×H100 is 13.4 TB/s. KV read: $1.718\times10^{11} / 1.34\times10^{13} = 12.8$ ms. Weight read: $7.0\times10^{10} / 1.34\times10^{13} = 5.2$ ms. Total floor 18 ms per step, producing 64 tokens → **3,555 tok/s** aggregate ceiling. At the ~60–65% memory-bandwidth utilisation a good engine achieves in practice, that lands around 2,200–2,400 tok/s — which is exactly the number you see quoted for this configuration, and now you know where it comes from rather than having memorised it.

**💰 Math:** at 2,300 tok/s on a node costing $10/hour, output tokens cost $10 / (2{,}300 \times 3600) = \$1.21\times10^{-6}$ each, or **$1.21 per million output tokens** at 100% utilisation. Real utilisation of 40% makes it $3.02/Mtok. Compare that to a hosted API's price for a 70B-class model and you have the actual build-versus-buy calculation, with the break-even sitting squarely on whether you can keep the node busy.

**⚠ Trap:** assuming quantising *weights* to FP8 speeds up a long-context batch. If KV traffic is 12.8 ms of an 18 ms step, halving the 5.2 ms weight term buys you 2.6 ms — a 14% improvement, not the 40% the marketing implies. On long-context workloads, quantise the **KV**, not the weights. Which term dominates is a two-line calculation, and I expect an engineer to do it before choosing a quantisation strategy.

### Write me something that reads a model config and tells me my maximum concurrency.

This is a script I actually keep in my toolbox, and being able to produce it from memory is worth more in an interview than any amount of recited theory.

```python
import math
from transformers import AutoConfig

DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int4": 0.5}

def kv_bytes_per_token(model_id: str, kv_dtype: str = "bf16") -> int:
    c = AutoConfig.from_pretrained(model_id)
    n_layers  = c.num_hidden_layers
    n_kv      = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    d_head    = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return int(2 * n_layers * n_kv * d_head * DTYPE_BYTES[kv_dtype])

def plan(model_id, n_params_b, n_gpus, hbm_gib=79.6, util=0.90,
         weight_dtype="fp8", kv_dtype="bf16", ctx=8192,
         ctx_overhead_gib=1.2, act_gib=4.0):
    per_token = kv_bytes_per_token(model_id, kv_dtype)
    weights_gib = (n_params_b * 1e9 * DTYPE_BYTES[weight_dtype]) / 2**30 / n_gpus
    pool_gib = n_gpus * (hbm_gib * util - weights_gib - ctx_overhead_gib - act_gib)
    tokens = int(pool_gib * 2**30 / per_token)
    return {
        "kv_bytes_per_token": per_token,
        "kv_kib_per_token": round(per_token / 1024, 1),
        "kv_pool_gib": round(pool_gib, 1),
        "total_live_tokens": tokens,
        "max_concurrent_at_ctx": tokens // ctx,
        "gib_per_session": round(ctx * per_token / 2**30, 2),
    }

print(plan("meta-llama/Llama-3.3-70B-Instruct", 70, 4, ctx=8192))
# {'kv_bytes_per_token': 327680, 'kv_kib_per_token': 320.0, 'kv_pool_gib': 200.4,
#  'total_live_tokens': 656670, 'max_concurrent_at_ctx': 80, 'gib_per_session': 2.5}
```

Three things I would defend in review. `ctx_overhead_gib` and `act_gib` are *measured*, not assumed — you run the engine with a tiny model, read `nvidia-smi`, and calibrate them for your exact stack and TP degree; hard-coding 1.2 and 4.0 without measuring is how the estimate drifts 20% from reality. The `getattr` fallbacks matter: many configs omit `head_dim`, and MHA models omit `num_key_value_heads` entirely, in which case query heads and KV heads are the same and the fallback is correct. And the function returns `total_live_tokens` as the headline, not the user count — because as established, tokens are the resource and users are a derived quantity.

**🏋 Drill:** unaided, in 10 minutes, write this function from a blank file and run it against three models of different families. Pass criterion: your `kv_bytes_per_token` matches an independent hand calculation for all three, and you correctly handle at least one model that lacks `head_dim` in its config. If you reach for a search engine for the config field names, you have not passed.
