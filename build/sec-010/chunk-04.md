### Your model scores fine on teacher-forced perplexity but generation turns to garbage after roughly 200 tokens. Debug it.

The shape of this bug tells you almost everything before you look at any code. Teacher-forced scoring is a single forward pass with no cache and known-correct positions; generation is a loop with a cache and computed positions. So the defect is in what generation does *differently*, and there are only about five such things. Perplexity being fine rules out the weights, the tokenizer's encode path, and the model architecture entirely.

**🔍 Failure taxonomy — good perplexity, bad generation, in the order I check:**

1. **Cache position off-by-one.** The signature is exactly this one: fine for 50 tokens, degrading past a few hundred, catastrophic past a thousand. Test: the cached-vs-uncached greedy equivalence check — generate 256 tokens with the cache, then run one teacher-forced forward pass over `prompt + output` and assert the argmaxes match position for position. If they diverge at token `k`, print `cache_position` and `position_ids` at step `k`.
2. **Positions not reset for left padding.** Symptom: the failure depends on *what else is in the batch*. Test: run the same prompt at batch 1 and inside a batch with a much longer sibling; if the outputs differ, it's positions or masking, not sampling.
3. **The padding mask not being extended as the cache grows.** Symptom: quality decays smoothly with generated length, and it's worse for the most-padded sequence. Test: assert `keep_mask.shape[-1] == k_cache.shape[2]` at every step. This is a one-line assertion that should live in the loop permanently.
4. **RoPE applied to the wrong tensor or with a stale position table.** Symptom: total garbage immediately, or garbage that starts exactly at the boundary of the precomputed sin/cos table length. Test: print `max(position_ids)` against the table size.
5. **Sampling configuration, not attention at all.** Temperature 1.0 with no top-p on a base model produces exactly the "starts fine, wanders into nonsense" pattern. Test: set `do_sample=False` and see if the degradation disappears. Check this *first*, it's free.
6. **EOS never supervised during SFT**, so the model runs past its natural stopping point into a region it was never trained on. Symptom: the first ~150 tokens are excellent and then it starts repeating or drifting. Test: look at whether the degradation begins right where a well-formed answer would have ended.

Note that (1), (2), (3) are all invisible to perplexity because teacher forcing never exercises them, and (5), (6) are invisible because perplexity doesn't sample. That's the general lesson worth stating: **teacher-forced metrics do not test your inference path.** Every serving stack I've been responsible for has a generation-level golden test — a fixed set of prompts, greedy decoding, output hashes compared in CI — precisely because eval loss will not catch a decode-loop regression.

**🗣 Say this in the room:** "Perplexity is teacher-forced and cache-free, so it tests the model and not the decode loop. Everything that differs between the two — cache positions, padding-side, the mask growing with the cache, RoPE indices, sampling config — is the suspect list. My first test is the cached-versus-uncached greedy equivalence check, which localizes the divergence to a specific token index."

### Same prompt, same seed, different batch composition, different output. Explain the mechanism and tell me whether you can fix it.

The mechanism is floating-point non-associativity meeting batch-dependent kernel selection. `(a + b) + c ≠ a + (b + c)` in floating point, and GPU reductions choose their summation order based on how the work is tiled — which depends on the shape of the tensors, which depends on the batch size. So `batch=1` and `batch=8` can dispatch to different cuBLAS kernels, different split-K factors, different numbers of thread blocks doing atomic accumulation, and produce results that differ in the last few bits.

Normally last-bit differences are irrelevant. In autoregressive generation they are not, because **argmax over a 128k-vocabulary softmax is a discontinuous function of the logits**. When the top two tokens are within a few ULPs — which happens constantly, since "the" versus "a" is often a near-tie — a 1e-7 perturbation flips the choice. Then the sequences diverge, and after that they're conditioned on different prefixes, so they diverge completely. It's chaos in the technical sense: bounded numerical noise, exponential divergence.

There are three separate sources to distinguish, because the fixes differ:

- **Batch-shape-dependent kernel selection.** Fixed by using kernels whose reduction order is invariant to batch size — "batch-invariant" implementations. This is achievable but it costs throughput, because the fastest kernel for batch 1 genuinely isn't the fastest for batch 64. **📅 Volatile:** a 2025 write-up from Thinking Machines Lab (*Defeating Nondeterminism in LLM Inference*) laid out the batch-invariance argument and shipped kernels for it; check the current state of engine support before promising a customer bitwise determinism.
- **Atomics and split reductions inside a single kernel.** `atomicAdd` ordering is nondeterministic across runs even at fixed shape. Fixed with `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, at a real throughput cost.
- **Continuous batching in the serving engine**, where your request is co-resident with whatever other requests arrived at that instant. This is the one people don't anticipate: even with a fixed seed and greedy decoding, a production engine gives you different outputs at different times of day, because the batch composition is a function of *other users' traffic*.

What I actually tell a product team: **do not build anything that depends on bitwise-reproducible LLM output.** Cache on the input hash, not on output equality. Write evals that measure a metric distribution over N samples with a confidence interval, not golden-string equality. If you genuinely need reproducibility — a regulated audit trail, a scientific claim — the practical route is to log the output alongside the input and treat the log as the record, rather than promising you can regenerate it. If you must have it, greedy decoding plus a pinned engine version plus batch-invariant kernels plus fixed batch size gets you there at maybe 10–20% throughput cost, and you should make someone sign for that trade.

**⚠ Trap:** "set the seed and it'll be deterministic." The seed controls the sampling RNG, and it fixes nothing about reduction order. A team that sets a seed, sees determinism at batch 1 in a notebook, and ships a golden-string test will have a flaky CI suite within a week of enabling batching.

### Why is a transformer parallelizable over the sequence during training but strictly sequential during inference? What follows from that?

Because during training you already know the whole sequence, so all `T` positions can be computed in one forward pass with a causal mask enforcing the information constraint; during inference you don't know token `t+1` until you've produced token `t`, so there is a genuine serial dependency of length equal to your output.

That asymmetry is the defining operational fact of LLM systems, and I'd unpack four consequences:

**Training scales with hardware; decode scales with clock.** Training throughput is `tokens/sec` and you buy more of it with more GPUs. Decode latency for one sequence is `n_output × time_per_step`, and `time_per_step` has a hard floor set by how fast you can stream the weights out of HBM. Adding a second GPU via tensor parallelism cuts the per-step time somewhat (each card reads half the weights) but adds an all-reduce per layer, so the scaling is sublinear and saturates quickly. **You cannot buy your way to low per-sequence latency the way you can buy throughput.**

**This is what makes decode memory-bandwidth-bound.** At batch 1, the model does ~2 FLOPs per parameter *byte* read, versus an H100's ratio of roughly `989e12 FLOP/s ÷ 3.35e12 byte/s ≈ 295` FLOPs per byte before you saturate compute. You're off by two orders of magnitude, running the tensor cores at ~1% utilization, and the only lever is reading fewer bytes: quantize the weights, batch more sequences, or produce more than one token per weight read.

**Which is exactly the design space of every decode optimization.** Speculative decoding produces `k` candidate tokens per weight read using a cheap draft model and verifies them in one pass — it converts serial steps into parallel verification. Medusa-style multi-head prediction does the same with extra heads instead of a draft model. Quantization to fp8 or int4 halves or quarters the bytes read per step. Continuous batching amortizes the weight read across concurrent users. Every one of these is attacking the same denominator.

**💰 Math:** an 8B model in bf16 is 16 GB. At 3.35 TB/s: `16/3350 = 4.78 ms` per decode step, floor, ignoring KV reads and all overhead — so **209 tokens/s** maximum for one sequence. Quantize to fp8 (8 GB): 2.39 ms, 418 tok/s. Add speculative decoding at an accepted-length of 2.5: ~1000 tok/s effective. Those three numbers are the honest ceiling of what you can promise a product team for single-stream latency, and they're derived from one division, not from a benchmark.

**🗣 Say this in the room:** "Training parallelizes over positions because the sequence is known; decode can't, because token `t+1` depends on token `t`. That serial dependency at batch 1 gives you ~2 FLOPs per byte of weights read, versus an H100's ~295 FLOP-per-byte balance point, so you're bandwidth-bound at about 1% tensor-core utilization. Every decode optimization is either reading fewer bytes or producing more tokens per read."

### Someone proposes replacing softmax with a cheaper nonlinearity in your production model. Evaluate the proposal.

My first question is *which* problem they're solving, because the answers diverge sharply.

If the goal is **asymptotic complexity** — getting off `O(T²)` — then they mean kernelized linear attention: replace `exp(q·k)` with `φ(q)·φ(k)` for some feature map, at which point associativity lets you compute `φ(Q) (φ(K)^T V)` and cost becomes `O(T · d²)` instead of `O(T² · d)`. **📄 Paper:** Katharopoulos et al. (2020), *Transformers are RNNs*, made the associativity argument explicit and showed the resulting model has an O(1) recurrent state; Choromanski et al. (2021), *Performer*, gave a random-feature map (FAVOR+) that approximates the softmax kernel with unbiased estimates. The honest state of the art: linear attention consistently loses on tasks requiring exact recall from a long context, because you have compressed the entire history into a fixed-size `d × d` state matrix and there is no mechanism for lossless lookup. That's information-theoretic, not an engineering gap — a fixed state cannot store an unbounded number of retrievable key-value pairs. Which is why the field converged on *hybrids* (a few full-attention layers interleaved with many linear/SSM layers) rather than pure linear models.

If the goal is **kernel speed** at unchanged complexity, they mean something like ReLU or sigmoid attention. ReLU attention with a `1/T` normalization has been shown competitive in vision transformers (Wortsman et al., 2023), and sigmoid attention has been studied with a length-dependent bias to compensate for the missing normalization. Both are legitimate research directions. Neither has displaced softmax in any frontier text model, and I would want to know why the person proposing it thinks they've found the thing several very well-funded teams didn't.

My review position, stated as a rule: **the burden of proof is on the replacement, and the acceptance criterion is not perplexity.** Softmax alternatives routinely match on perplexity and general benchmarks while failing on exactly the capabilities that make an LLM useful in a product — long-context retrieval, in-context learning from few-shot examples, and instruction-following over long inputs. So the eval gate I'd insist on is: (a) needle-in-a-haystack at your real context length, on your real corpus, with multiple distractor needles; (b) a many-shot in-context learning task where accuracy should climb with shot count, verifying induction still works; (c) exact-copy tasks — reproduce a UUID that appeared 30k tokens ago. If any of those regress, the answer is no regardless of how good the throughput number is.

**⚠ Trap:** conflating "we removed softmax" with "we got FlashAttention's speedup." FlashAttention keeps softmax exactly and speeds things up by 2–4× through IO-awareness alone. If someone's motivation is speed, the first question is whether they're already on a fused kernel, because the free 3× is much more attractive than the lossy 5×.

### Design a 128k-context document-QA service on a fixed GPU budget. Where does attention put the pressure, and what do you change?

Let me put numbers on the pressure first, because the design follows from the arithmetic.

**Prefill cost.** 128k tokens through an 8B model: `2 × 8e9 × 131072 = 2.1e15` FLOPs for the dense part. Plus the attention term, which is `4 · T² · d_model` per layer, so across 32 layers: `4 × 131072² × 4096 × 32 = 9.0e15` FLOPs. Attention is **four times** the dense cost at this length, which is the `T/2D = 131072/8192 = 16×` ratio from earlier applied per-layer. Total ≈ `1.1e16` FLOPs. At 400 TFLOP/s achieved: `1.1e16 / 4e14 = 27 seconds` of TTFT for one request. That is the headline problem, and no one will accept it.

**KV memory.** At 128 KB/token (32 layers, 8 KV heads, 128 dims, bf16), 128k tokens is **16 GB per concurrent request**. On an 80 GB card holding 16 GB of weights, you fit `(80 − 16 − 8 overhead) / 16 ≈ 3.5` concurrent 128k requests. Three and a half. That's your entire concurrency budget on a $30k GPU.

Now the design, in the order I'd apply it — and the first three moves are all about *not putting 128k tokens in the context*:

1. **Retrieve instead of stuff.** The overwhelming majority of "128k document QA" workloads are answerable from 4–8k tokens of the right chunks. Chunk, embed, retrieve top-k with a reranker, and prefill 8k instead of 128k. Prefill drops from 27 s to `(2 × 8e9 × 8192 + 4 × 8192² × 4096 × 32)/4e14 = (1.3e14 + 3.5e13)/4e14 = 0.41 s`. That's a **65× TTFT reduction** and a 16× concurrency increase, for the cost of building a retrieval layer. I would push back hard on any design that skips this step, and "we'll just use the long context" is the single most expensive default decision in applied AI.
2. **Prefix caching for the document.** If users ask multiple questions of the same document — which is the whole shape of document QA — the document's KV is computed once and reused. Structure the prompt as `[system][document][question]` so the shared prefix is maximal, and never put a timestamp or a request ID before the document. With a 90% hit rate you pay full prefill on one in ten requests.
3. **Chunked prefill** so a 128k prefill doesn't monopolize the GPU and stall every other user's decode. Slice into ~2k-token chunks, interleave decode steps between them. This trades a little TTFT on the long request for a lot of ITL stability across the fleet, and it's the difference between a p99 that tracks the mean and one that's 30 s.
4. **Then, and only then, the model-level levers:** GQA or MLA to cut KV bytes; fp8 KV quantization for a further 2×; sliding-window or interleaved local:global layers if you control the architecture. These are §11 and §12 territory and they multiply with everything above.

**💰 The full economics.** Say 50k document-QA requests/day. Naive 128k stuffing: 27 s of GPU-exclusive prefill each = `50,000 × 27 = 1.35e6` GPU-seconds/day = 375 GPU-hours/day, which needs ~16 H100s running flat out; at $2.50/hr that's `375 × 2.50 = $938/day ≈ $28k/month`. With retrieval to 8k plus 90% prefix-cache hit: `50,000 × 0.41 × 0.1 (miss rate) + 50,000 × 0.04 (cached path) = 2,050 + 2,000 = ~4,050` GPU-seconds/day ≈ 1.1 GPU-hours/day ≈ **$85/month**. A 300× cost reduction that costs you a retrieval pipeline and a chunking strategy. This is the arithmetic I want a candidate to produce unprompted when someone says "we'll just use the long context window."

**⚠ Trap:** treating the advertised context limit as a capability rather than a limit. Quality degrades measurably well before the advertised maximum — the lost-in-the-middle effect and general context rot are real and reproducible. Long context is a *fallback* for when retrieval fails, not a replacement for retrieval.

### You replaced `model.generate()` with a custom decode loop and outputs diverge from the reference after about thirty tokens. Walk me through the debug.

Thirty tokens is the tell. It's too long for an immediate structural error (wrong mask polarity, wrong padding side — those break at token 1) and too short for slow positional drift (which takes hundreds). Thirty tokens means the two paths are *nearly* identical and something small compounds, or something changes state at a boundary.

My procedure, and I'd say it in this order:

**Step 0 — eliminate sampling.** Force greedy on both paths (`do_sample=False`, temperature irrelevant). If they now agree, the bug was RNG state or sampling config, not attention. Half of these end here.

**Step 1 — compare logits, not tokens.** Tokens are argmaxes and hide magnitude. Run both paths in lockstep, and at each step compute `(logits_mine - logits_ref).abs().max()`. You'll see one of two patterns. If the max diff is ~1e-3 from step 0 and constant, it's a dtype or kernel difference (bf16 vs fp32, fused vs unfused attention) and the divergence at 30 is just when a near-tie finally flipped — that's not a bug, that's chaos, and you should verify with a longer greedy run at fp32. If the max diff is ~0 for 29 steps and then jumps, something *changed* at step 30.

**Step 2 — what changes at step 30?** The candidates are concrete. A preallocated cache whose initial capacity was `prompt_len + 32` and now needs to grow. A CUDA graph captured for one shape and replayed for another. A sliding window whose width you crossed. A `max_position_embeddings` or precomputed sin/cos table boundary. A chunk boundary in chunked prefill. Print the step index against every buffer size in the loop and look for a coincidence.

**Step 3 — bisect the model, not the loop.** Register forward hooks on both paths and compare the hidden state after each layer at the diverging step. The first layer where they differ is where the bug is. If layer 0's *input* differs, it's the embedding or the token you fed in. If layer 0's input matches and its output differs, it's attention or the mask at layer 0 — dump the mask and the `position_ids` for both.

**Step 4 — the three usual culprits, checked directly.** `position_ids` / `cache_position` at the diverging step. The attention mask's last dimension versus the cache's length. Whether you fed the *sampled* token or the *argmax* token back in — a surprisingly common mismatch when the reference path applies a repetition penalty or a logit processor you didn't replicate.

**⚠ Trap:** assuming any divergence is a bug. Greedy decoding is an argmax over ~128k logits; near-ties are common, and any legitimate numerical difference (a different attention kernel, a different tensor-parallel degree, a different batch size) will eventually flip one. The correct acceptance test for a rewritten decode loop is not "identical tokens" but "identical tokens under fp32 with the same kernel, and statistically indistinguishable quality on a 200-prompt eval otherwise." I've seen a week burned chasing a "bug" that was FlashAttention versus the math backend.

**🗣 Say this in the room:** "First force greedy on both. Then compare logits rather than tokens, because a constant 1e-3 logit gap that flips an argmax at token 30 is numerics, not a bug. If the logits match exactly and then jump, something changed state at that step — cache reallocation, a CUDA graph shape, a window boundary — and I'd bisect with forward hooks to find the first layer that differs."

### Explain how prefix reuse and chunked prefill interact with the causal mask. What mask does a new chunk need?

Both are the same trick — process the sequence in pieces, carry the KV forward — and the mask for a new piece is the general form you already wrote: causal *with a diagonal offset equal to the number of cached tokens*.

Concretely, suppose 4,000 tokens are already cached (from a reused prefix, or from a previous prefill chunk) and you're now processing a chunk of 1,024 new tokens. Then `T_q = 1024`, `T_kv = 5024`, and the mask must be:

- Every new query sees **all 4,000 cached keys unconditionally** — they're all in its past.
- Within the new chunk, query `i` sees new key `j` iff `j ≤ i` — the ordinary triangle.

That is exactly `tril(ones(1024, 5024), diagonal=5024 - 1024 = 4000)`: a full rectangle for the first 4,000 columns and a triangle for the last 1,024. Which is why the single expression `tril(diagonal=T_kv − T_q)` covers prefill (`offset 0`), chunked prefill (`offset = tokens_so_far`), single-token decode (`offset = T_kv − 1`, all visible), and speculative-decoding verification of `k` draft tokens (`offset = T_kv − k`, a small triangle at the end) with no branches. If you write one mask function that takes `T_q` and `T_kv` and nothing else, you get all four for free — and I consider a decode implementation with four separate mask code paths a design smell.

Two correctness requirements that ride along, and they're the ones that break in practice:

**Positions must be absolute.** The new chunk's tokens are at positions `4000 … 5023`, not `0 … 1023`. If you regenerate `position_ids` per chunk with `arange(T_q)` you have restarted the sequence, and RoPE will place chunk 2 on top of chunk 1. This is the chunked-prefill version of the left-padding position bug and it has the same signature — fine on chunk 1, degraded after.

**Prefix reuse requires that the cached prefix was computed under an identical model configuration.** The KV vectors bake in the RoPE base, the layer weights, the quantization scheme, and the exact tokenization. Change any of those — a model version bump, a `rope_theta` change during long-context extension, switching from bf16 to fp8 KV — and every cached block is silently wrong rather than merely stale. So a prefix cache key must include a *config fingerprint*, not just the token hash. I've seen a team ship a model upgrade with a prefix cache keyed on token IDs alone; the symptom was gibberish for exactly the users whose prompts happened to hit warm blocks, which is a distribution of failure nobody's dashboard was shaped to catch.

**⚠ Trap:** believing chunked prefill changes the result. It does not — the attention output for a token is identical whether you computed the prefix in one pass or ten, because attention over the cached keys is the same reduction either way (this is the online-softmax associativity again). What it changes is *scheduling*: it lets the engine interleave decode steps between chunks, which trades a small TTFT increase on the long request for a large ITL improvement on everyone else's. If someone claims chunked prefill hurt their quality, they have a position or mask bug, not a chunking problem.

### How do you test an attention implementation? Give me the actual suite.

Six tests, and I'd argue every one of them earns its runtime. The theme: attention has almost no loud failure modes, so the tests have to be constructed to *make* failures loud.

**1. Reference equivalence on small shapes.** Keep the naive materialized implementation in the repo forever. `torch.allclose(fast(q,k,v,mask), naive(q,k,v,mask), atol=1e-2, rtol=1e-2)` in bf16 at `T=64`, `H=4`, with a randomly generated boolean mask, and tighter tolerances in fp32. This catches mask polarity, softmax axis, transpose errors, and head-reshape errors — the entire class of "silently trains anyway" bugs.

**2. Causality by construction.** Perturb the input at position `j` and assert the output at every position `i < j` is *bitwise* unchanged. This is a stronger and much better test than inspecting the mask, because it tests the whole path including RoPE and the cache. Three lines, catches every causality leak including the exotic ones in custom kernels.

```python
out1 = model(x)
x2 = x.clone(); x2[:, j] = torch.randn_like(x2[:, j])
out2 = model(x2)
assert torch.equal(out1[:, :j], out2[:, :j])     # future must not leak backward
```

**3. Cached-versus-uncached greedy equivalence.** Generate 256 tokens with the cache; teacher-force `prompt + output` through one cache-free pass; assert the argmax at each position equals the emitted token. Catches every `cache_position` and mask-growth bug, and localizes the failure to an index.

**4. Padding invariance.** Encode a prompt alone and the same prompt inside a left-padded batch with a much longer sibling. Assert the logits agree to within numerical tolerance. Catches position-id derivation bugs, mask-extension bugs, and padding-side bugs — the whole family whose symptom is "quality depends on the batch."

**5. Attention weights sum to one.** On the naive path, `assert torch.allclose(attn.sum(-1), ones)` for every non-fully-masked row. Trivial, and it catches the softmax-axis error that otherwise costs you a training run.

**6. Packing / varlen equivalence.** Run three documents separately, then run them packed with a document mask, and assert per-document outputs match. This is the only test that reliably catches cross-document attention leakage, and I'd add the adversarial version: pack document B as an exact copy of document A and assert B's loss does *not* drop relative to running B alone. If it drops, B is reading A.

**⚠ Trap:** testing only with a full causal mask and uniform lengths. That configuration exercises none of the interesting paths. Every one of these tests needs ragged lengths and a nontrivial mask, or you're testing the happy path that was never going to break.

**🗣 Say this in the room:** "Attention fails silently, so the tests have to manufacture loud signals: causality by perturbation rather than by mask inspection, cached-versus-uncached greedy equality rather than short-generation eyeballing, and padding invariance because the worst bugs make output depend on batch composition. Plus a naive reference kept in-tree as the oracle."

### Give me the full failure taxonomy for attention in production. Not anecdotes — a decision procedure.

Organized by the symptom you actually get paged about.

**🔍 Symptom: quality depends on batch composition.**
Almost always positions or masks, never the model. Check in order: `position_ids` derived by `arange` under left padding; the padding mask not extended as the cache grows; a nondeterministic kernel reduction flipping near-ties (benign if the effect is small and unbiased). Discriminator: run the same prompt at batch 1 and inside a padded batch; if logits differ by more than numerical tolerance, it's (1) or (2), not (3).

**🔍 Symptom: fine for short outputs, degrades over long ones.**
Cache-position off-by-one; RoPE table exhausted past its precomputed length; sliding window crossing into a regime you didn't test; EOS never supervised so generation runs past the trained region. Discriminator: the cached-vs-uncached greedy equivalence test localizes the exact token.

**🔍 Symptom: NaN in the loss, or NaN logits at inference.**
A fully-masked row (empty sequence, masked query rows, empty cross-attention context); fp16 logit overflow from massive activations; softmax without max subtraction. Discriminator: `torch.isnan(...).any()` hooks on each layer's output localize the layer, then dump the mask row-sums for that batch — a row summing to zero in the keep mask is your answer.

**🔍 Symptom: loss plateaus or spikes during training, gradients healthy elsewhere.**
Attention entropy collapse. Discriminator: log max attention logit and mean attention entropy per layer on a probe batch; climbing logits past ~50 with entropy trending to zero confirms it. Fix with QK-norm.

**🔍 Symptom: prefix cache produces garbage for a subset of users.**
Cache key doesn't include the model/config fingerprint, so blocks computed under a different `rope_theta`, quantization, or checkpoint are being reused. Discriminator: correlate failures with prefix-cache hit; if 100% of failures are cache hits, you have your answer. This one is nasty because it affects exactly the users you're serving best.

**🔍 Symptom: long-context retrieval regressed while general benchmarks are flat.**
KV quantization (fp8/int4) hurting the precision of distant keys; sink token evicted by an aggressive cache-eviction policy; positional extension applied without validation. Discriminator: run needle-in-a-haystack at varying depths — a uniform drop points at quantization, a depth-dependent drop points at positions or eviction.

**🔍 Symptom: TTFT p99 blew up while mean is fine.**
A long prefill monopolizing the GPU and stalling queued requests. Discriminator: correlate p99 TTFT with the concurrent presence of >32k-token prompts. Fix with chunked prefill and prompt-length-based admission control.

The meta-rule I'd offer: **attention bugs are positional bugs until proven otherwise.** Masks and position indices account for the overwhelming majority of real incidents; the softmax itself almost never breaks.

### Drill: build me a mask factory under time pressure.

**🏋 Drill (20 minutes, unaided, no autocomplete).** Write a single function

```python
def build_keep_mask(pad_keep, doc_id=None, T_q=None, window=None, n_sinks=0):
    """Return a [B, 1, T_q, T_kv] boolean keep-mask (True = attend)."""
```

that correctly composes: (a) causal with the `T_kv − T_q` diagonal offset so it works for prefill, chunked prefill, and single-token decode; (b) key padding from `pad_keep: [B, T_kv]`; (c) optional document-boundary blocking from `doc_id: [B, T_kv]` and the corresponding query doc ids; (d) an optional sliding window of width `window` applied to *relative* distance; (e) `n_sinks` leading positions always visible regardless of the window.

**Pass criteria, all of which must hold on the first unaided run:**
1. With `T_q == T_kv` and no options, the result equals `torch.ones(T,T).tril()`.
2. With `T_q == 1` and `T_kv == 500`, every real key is visible (row is all-True where padding allows).
3. With `T_q == 128`, `T_kv == 1128`, the first 1000 columns are fully visible and the last 128 form a triangle.
4. With `window=4` and `n_sinks=2`, query 100 sees exactly keys `{0, 1, 96, 97, 98, 99, 100}`.
5. With two documents of lengths 5 and 7 packed into 12 tokens, no query in document 2 sees any key in document 1, and vice versa.
6. `build_keep_mask` never allocates a tensor with `H` in its shape — output leading dims are `[B, 1, ...]` and broadcasting does the rest.
7. No row of the returned mask is all-False for any real query.

If you can pass 1–3 you can implement decode. If you can pass 4–5 you can implement long-context and packing. Criterion 6 is the memory discipline and criterion 7 is the NaN guard. I would consider failing 3 or 7 disqualifying for a serving role, because those are the two that produce silent production incidents rather than test failures.

### Drill: instrument a real model's attention and report what you find.

**🏋 Drill (35 minutes, with a machine and a small open-weight model, no reference implementation).** Load any small instruction-tuned open-weight model. Write a probe that, for a fixed batch of 16 prompts of ~512 tokens each, reports per layer:

1. **Mean attention entropy** `-Σ a log a`, averaged over heads and query positions.
2. **Sink mass** — the fraction of total attention probability landing on positions 0–3, averaged over heads and queries.
3. **Max attention logit** (pre-softmax, post-scale).
4. **Attention distance** — the mean of `(q_idx − kv_idx)` weighted by attention probability, per head, which tells you how far back each head typically looks.

You'll need `output_attentions=True` or an equivalent hook, and you must run the *eager* attention path since fused kernels don't return weights — say that out loud, because knowing it is part of the exercise. **📅 Volatile:** the exact flag name and which backends support returning weights change across library versions.

**Pass criteria:**
- The numbers are produced without a `for` loop over positions (vectorized reductions only), and memory stays bounded — you must not hold all layers' `[B,H,T,T]` tensors simultaneously. At `B=16, H=32, T=512, 32 layers` in fp32 that would be `16 × 32 × 512² × 4 × 32 = 1.7e11` bytes = 172 GB. Reduce per layer and free.
- You can state, from your own numbers, which layers are the sink-heaviest and which heads are the most local versus most global.
- You correctly observe that sink mass is *low* in the first layer or two and high thereafter, and can explain why (early layers haven't yet built the representations that make token 0 a strong key, and their heads are doing local positional work).

The point of this drill is not the numbers. It's that you will have touched real attention tensors with your own hands, and the next time someone says "the attention looks weird" you'll have a reflex for what to measure rather than a vocabulary for what to say.

### An interviewer stops you mid-derivation: "why not just learn the attention pattern as parameters instead of computing it from the data?"

Because a learned static pattern is a *convolution with extra steps*, and it throws away the one property that made attention worth its quadratic cost: the routing decision depends on content, so it can be different for every input.

Make the alternative concrete. A learned pattern means `A ∈ R^{T × T}` is a parameter matrix, and `out = A V`. Three things immediately go wrong. **It's not length-generalizable** — the parameter matrix has a fixed `T`, so a model trained at 4k cannot process 4,097 tokens at all, whereas attention's parameters (`W_Q, W_K`) are length-independent by construction and the pattern is recomputed for whatever `T` you hand it. **It's position-addressed, not content-addressed** — "attend to the token 5 back" is expressible, "attend to the last mention of the entity I'm currently discussing" is not, and the second is what language actually requires. **It can't do in-context learning** — induction is fundamentally "find where this token appeared before and copy what followed," and where it appeared before is a property of *this* input, unknowable at training time.

This isn't hypothetical; the field ran the experiment. MLP-Mixer-style architectures replace token mixing with a learned position-wise MLP and work respectably in vision, where spatial structure is fixed and translation-equivariant. They do not work for language, where the relevant structure is referential rather than positional. Similarly, the "attention-free" architectures that *do* work — SSMs like Mamba, and modern gated linear attention — did not go back to static patterns; they kept input-dependence and instead made the state fixed-size. Mamba's whole innovation over earlier SSMs was making the state-space parameters a *function of the input*, i.e. reintroducing selectivity. The field's revealed preference is unanimous: you can give up the quadratic, you cannot give up input-dependence.

There's a useful backend framing for this: a learned pattern is a static routing table compiled at build time; attention is dynamic dispatch on the payload. You'd never accept a service mesh that routed requests by their arrival index rather than their content, and for the same reason.

**🗣 Say this in the room:** "A learned pattern is content-independent, so it can't length-generalize and it can't do in-context learning — induction requires knowing where a token appeared *in this input*, which is unknowable at training time. Every successful attention alternative kept input-dependence and gave up something else, usually the unbounded state. That tells you which property was load-bearing."

### Give me the sixty-second whiteboard version of attention, for a staff backend engineer who's never touched ML.

You have a sequence of `T` vectors, one per token, each of width `d`. You want each position to be able to pull information from any other position, with the choice of *which* positions made at runtime based on content, not baked into weights.

So each token computes three projections of itself: a **query** (what am I looking for), a **key** (what do I advertise), and a **value** (what I'll hand over if selected). Score every query against every key with a dot product — that's one matmul producing a `T × T` matrix. Divide by `√d_head`, because dot products of `d`-dimensional random vectors have standard deviation `√d` and softmax saturates if you don't. Mask out the future by setting those scores to `−inf` *before* the softmax, not by zeroing weights after, because the softmax denominator sums over everything it's given. Softmax each row into a distribution over source positions. Multiply by the values: each position's output is a weighted average of every visible position's value. Do that `n_heads` times in parallel on disjoint slices of the width, concatenate, and project once more. Add the result back into the residual stream.

Three facts that follow and that drive everything downstream. It's `O(T² d)` compute, which only exceeds the rest of the block past `T ≈ 2 · d_model` — so quadratic cost is a long-context problem, not a 2k-context problem. During generation, the keys and values of past tokens never change, so you cache them; that cache is `2 · n_layers · n_kv_heads · d_head · bytes` per token — around 128 KB/token for a typical 7B-class model — and it, not the weights, is what limits how many users you can serve at once. And because generating token `n+1` requires token `n`, decode is serial and memory-bandwidth-bound at roughly 2 FLOPs per byte read, against hardware built for ~300, which is why every serving optimization in the field is about reading fewer bytes or emitting more tokens per read.

**🗣 Say this in the room** (the thirty-second compression, if they cut you off): "Attention is content-addressed lookup: every token emits a query, every token advertises a key, dot products give you a soft distribution over sources, and you return the weighted average of their values. It's the only sequence operation with an O(1) path between any two positions, it costs `O(T²)`, and in production the thing that hurts isn't the FLOPs — it's the KV cache, at roughly 128 KB per token per sequence."
