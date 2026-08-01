### You're training a tokenizer for a new pretraining run. What decisions do you make, and what goes wrong?

The mental model: a tokenizer is a **lossy compression scheme that you are permanently baking into your model's cost structure**. Every downstream number — context window in "real" content, dollars per document, tokens/sec, even the loss value itself — is denominated in a unit that this one decision defines. It is also nearly impossible to change later, because changing it invalidates every token shard and every checkpoint's embedding matrix.

**The decisions, in order of consequence:**

**Vocabulary size.** The trade is compression versus embedding cost. A bigger vocab means fewer tokens per document (cheaper training and inference per unit of content) but a bigger embedding and output matrix. Llama-2 used 32k; Llama-3 moved to 128,256, which Meta reported yields roughly 15% fewer tokens on English text. Do the arithmetic on the cost side: at `d_model = 4096`, a 128k vocab embedding is `128,256 × 4096 = 5.25e8` params — 0.53 GB at bf16, and doubled if the LM head is untied. On an 8B model that is ~13% of parameters spent on lookup tables. On a 1B model it would be 50%+, which is why small models use smaller vocabularies. There is also a compute term: the final logit projection costs `2 × d_model × V` FLOPs per token, `= 2 × 4096 × 128256 = 1.05` GFLOP, against `6 × 8e9 = 48` GFLOP total per token in training — about 2%, plus a large *memory* cost for the logits tensor at long sequence length, which is a real OOM source.

**Algorithm.** Byte-level BPE is the default and correct answer for a general model. "Byte-level" matters: it guarantees no out-of-vocabulary input is possible, ever, because the base alphabet is the 256 bytes. Unigram (SentencePiece) is a defensible alternative with somewhat better morphological segmentation in some languages.

**Pre-tokenization regex.** This is the underrated decision. The regex that splits text before BPE merges determines whether numbers are split per-digit, whether whitespace runs are their own tokens (critical for code indentation), and whether a leading space attaches to a word. GPT-4-style tokenizers split numbers into groups of at most 3 digits; Llama-3 splits digits individually. Per-digit splitting measurably helps arithmetic because it makes place value positionally explicit.

**Training corpus for the tokenizer.** It must be a stratified sample of your **final** data mixture. Train the tokenizer on English web and then train the model on 30% code, and your code will tokenize at maybe 1.4× the token count it should — you have thrown away 40% of your code compute budget to a preprocessing decision.

**⚠ Trap:** forgetting reserved special tokens. Add 128–256 unused placeholder tokens to the vocabulary at training time. When post-training later needs `<|tool_call|>`, `<|thinking|>`, or a new chat-role marker, you can repurpose a reserved slot instead of resizing the embedding matrix and disturbing a trained model. Llama-3 ships exactly such reserved tokens. Skipping this costs you nothing today and a genuinely annoying surgery in six months.

**⚠ Trap:** comparing loss between runs with different tokenizers. Per-token cross-entropy is meaningless across tokenizers — the denominator changed. Normalize to **bits per byte**: `bpb = loss_nats × n_tokens / (ln 2 × n_bytes)`.

### Explain document packing. What is wrong with just concatenating documents until you hit the sequence length?

The mental model: your training step operates on fixed-shape tensors of length `L`. Documents are variable-length and mostly much shorter than `L`. You have three options — pad, truncate, or pack — and only one of them is economically viable.

**The arithmetic that kills padding.** Web documents have a heavy-tailed length distribution with a mean around 1,000–1,500 tokens. If `L = 8192` and mean document length is 1,200 tokens, padding each document into its own sequence means `1200/8192 = 14.6%` of every batch is real tokens. You are spending 85% of a $1.26M compute budget computing attention over `<pad>`. Nobody does this in pretraining; it exists only in fine-tuning code where someone forgot.

**Packing** concatenates documents end to end and cuts at `L`, so every position is a real token. The problem is what "concatenate" does to attention.

**What is wrong with naive concatenation.** Causal attention lets position `i` attend to all positions `< i`. If document B follows document A in the same packed sequence, every token of B can attend to every token of A. This is wrong in three specific ways:

1. **Cross-document contamination of context.** The model learns to condition on unrelated preceding text. At best it learns to ignore it (wasting capacity on that skill); at worst it learns spurious dependencies, and in-context learning behavior degrades because the model's notion of "the context" is polluted.
2. **Position IDs run across boundaries.** Document B's first token gets position 1,341 instead of 0. With RoPE, the model never sees document B's tokens at small relative offsets from a document start, which distorts what it learns about beginnings.
3. **The loss at boundaries is nonsense.** Predicting the first token of an unrelated document from the last token of the previous one is an unlearnable task; those tokens contribute pure noise to the gradient.

**The fix: intra-document causal masking.** Attention is restricted to within-document, and position IDs reset at each document boundary. Published work analysing sequence composition finds intra-document masking improves in-context learning and reduces the model's tendency to be distracted by irrelevant preceding context, particularly at longer sequence lengths.

**⚠ Trap:** believing packing without masking is fine "because the model learns to ignore it." Partly true and completely beside the point — the fix costs nothing. FlashAttention has a variable-length API (`flash_attn_varlen_func`, taking `cu_seqlens` cumulative-offset arrays) that implements block-diagonal masking with *no* materialized mask tensor and essentially no throughput cost. Turning it on is a config change. Not turning it on is a defect.

**💰 Math:** the case for packing is stark. On a `$1.26M` run, going from padded (14.6% useful) to packed (~100% useful) is not a 6.8× speedup in practice — you would never have run padded — but the equivalent framing is that padding would cost `$1.26M × (1/0.146) = $8.6M` for the same tokens seen. Packing is the difference between a feasible run and an infeasible one.

### Implement packing with correct masking. Write me the data-side code.

Two pieces: the packing itself, and the metadata the attention kernel needs.

```python
import numpy as np

def pack_documents(doc_token_lists, seq_len, eos_id):
    """Greedy end-to-end packing. Yields (tokens, seq_starts) per sequence.
    seq_starts are offsets WITHIN the sequence where each document begins."""
    buf, starts = [], [0]
    for doc in doc_token_lists:
        toks = list(doc) + [eos_id]
        i = 0
        while i < len(toks):
            room = seq_len - len(buf)
            take = min(room, len(toks) - i)
            buf.extend(toks[i:i + take]); i += take
            if len(buf) == seq_len:
                yield np.array(buf, dtype=np.uint32), starts
                buf, starts = [], [0]
        if buf:                      # next doc starts here
            starts.append(len(buf))
    if buf:                          # tail: pad the final partial sequence
        starts_out = starts[:-1] if starts[-1] == len(buf) else starts
        buf += [eos_id] * (seq_len - len(buf))
        yield np.array(buf, dtype=np.uint32), starts_out

def position_ids_and_cu_seqlens(starts, seq_len):
    """Per-document position ids (reset at each doc) + cumulative seqlens for varlen attn."""
    bounds = list(starts) + [seq_len]
    pos = np.concatenate([np.arange(b - a) for a, b in zip(bounds[:-1], bounds[1:])])
    cu = np.array(bounds, dtype=np.int32)     # [0, s1, s2, ..., seq_len]
    return pos.astype(np.int32), cu
```

On the model side, you pass `cu_seqlens` into FlashAttention's varlen entry point (the sequence is flattened to a single `(total_tokens, n_heads, head_dim)` tensor and the kernel treats each `[cu[i], cu[i+1])` slice as an independent sequence), and you pass `position_ids` into your RoPE application so rotations restart per document. In PyTorch's `FlexAttention` you express the same thing as a `mask_mod` returning `(q_doc_id == kv_doc_id) & (q_idx >= kv_idx)`, which compiles to a block-sparse kernel.

If you must materialize a mask (small-scale, debugging, or a framework without varlen support):

```python
def block_diag_causal_mask(doc_ids):          # doc_ids: (L,) int array
    same = doc_ids[:, None] == doc_ids[None, :]
    causal = np.tril(np.ones((len(doc_ids), len(doc_ids)), dtype=bool))
    return same & causal
```

**⚠ Trap:** the materialized mask is `L²` booleans. At `L = 8192` that is 67 MB per sequence per layer if you are careless about broadcasting, and it defeats FlashAttention entirely — the whole point of FlashAttention is never materializing the `L×L` attention matrix, and handing it a dense mask forces the memory-bound path. Use `cu_seqlens` or `FlexAttention`. If you see a `torch.finfo(dtype).min` additive mask of shape `(B, 1, L, L)` in a pretraining codebase, that is a performance bug worth 2× on long sequences.

**🏋 Drill (25 minutes, unaided):** write `pack_documents` and the block-diagonal mask from scratch, then assert three properties on random input: (1) every output sequence has exactly `seq_len` tokens; (2) `position_ids` restart at 0 at every document start; (3) for a packed batch, the loss computed with block-diagonal masking equals the sum of losses computed by running each document separately. **Pass criterion:** all three assertions green, no reference material. Property (3) is the one that catches real bugs and almost nobody writes it.

### How do you lay out the tokenized data on disk so the dataloader isn't your bottleneck?

The mental model: at 40% MFU on 1000 H100s you are consuming `~1e6` tokens/second. At 2 bytes/token that is **2 GB/s of sustained sequential read**, forever, across a job that must resume deterministically after a crash at 3am. That is an ordinary storage engineering problem and it should be treated as one, not as an afterthought in the training script.

**Format.** A flat `uint16` or `uint32` memory-mapped binary array of tokens, plus a separate index of document offsets. This is the layout Megatron and nanoGPT-style codebases use, and it is right: `np.memmap` gives you zero-copy page-cache-backed reads, sequences are `arr[i*L : (i+1)*L]`, and there is no parsing. `uint16` if your vocab is under 65,536 (halves your I/O); `uint32` for a 128k vocab. Shard into files of a few GB so you can distribute and re-generate them independently.

**Sampling and shuffle.** The requirement is that the shuffle be **a deterministic function of `(seed, epoch, global_step, rank)`** with no state. Then resumption is trivial: you do not restore a dataloader's iterator state, you recompute the index. The standard implementation is a seeded permutation of sequence indices; for a corpus too large to permute in memory, use a block shuffle (permute shard order, then permute within a buffer of a few thousand sequences). Perfect global shuffling is not required — locality within a shard is fine as long as shard order is randomized and shards are not domain-homogeneous.

**⚠ Trap:** domain-homogeneous shards. If shard 47 is entirely Python code, then for a few hundred steps every rank sees mostly Python and your gradient is a code gradient. This shows up as a periodic wobble in the loss curve with a period equal to your shard traversal, and people misdiagnose it as an LR problem. **Interleave domains at shard-write time**, so every shard is a miniature of the full mixture.

**Throughput mechanics.** Prefetch with enough workers to cover storage latency (`num_workers` such that `workers × per-worker throughput > 2 GB/s`), pin memory, and overlap H2D transfer with compute. The tell that your dataloader is the bottleneck: step time has a bimodal distribution, or GPU utilization sawtooths. Instrument it — emit `data_wait_ms` per step as a metric next to `step_time_ms`. If `data_wait_ms` is more than ~2% of step time you are lighting money on fire, and it is the cheapest MFU you will ever recover.

**Resumability contract, which I insist on in review:** given `(checkpoint, global_step)`, the dataloader must produce *exactly* the sequence of batches it would have produced without the crash. Violating this means restarts silently re-show or skip data, and after 40 restarts your effective epoch count is unknown. Test it: run 100 steps, kill at 50, resume, and assert the batch token hashes at steps 50–100 match the uninterrupted run.

### How do you decide the data mixture weights across domains?

The mental model: mixture weights are hyperparameters with an unusually bad optimization surface — each evaluation costs a training run, there are ~10–20 of them, and their effects interact. So the entire field's answer is: **use small proxy models to search, and validate at scale.**

**Three approaches, in increasing sophistication:**

1. **Ablate at 1B scale.** Train a 1B model on 20–50B tokens for each candidate mixture, evaluate on your benchmark suite, pick the winner, and hope it transfers. Cost per ablation: `6 × 1e9 × 3e10 = 1.8e20` FLOPs = `1.8e20 / (989e12 × 0.4) = 4.5e5` GPU-seconds = **126 GPU-hours** ≈ $315. You can afford 50 of these for $16k against a $1.26M run — this is the single best-value spend in the whole project, and teams consistently underspend it.
2. **Learned reweighting.** **📄 Paper:** Xie et al. (2023), *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining* — train a small reference model on a baseline mixture, then train a small proxy with group-DRO that upweights domains where the proxy has high excess loss relative to the reference; use the resulting weights to train the large model. It replaced pure human judgement with a measurable procedure, and reported both faster convergence and downstream gains.
3. **Fit a scaling law over the mixture.** Train a grid of small models across mixture weights, fit a parametric loss surface as a function of the weights, and optimize it. More sample-efficient than a grid search but more machinery.

**What the weights typically look like** for a general model — and treat these as a starting distribution to ablate against, not gospel: web text 55–75%, code 10–20%, curated/reference (books, papers, encyclopedic) 5–15%, math 2–8%, multilingual whatever your target languages require. Code is upweighted far past its natural web frequency because there is strong evidence it improves general reasoning and structured output, not just coding.

**The subtlety that matters most:** weights are usually expressed as *sampling* probabilities, which decouples them from corpus size. If code is 3% of your corpus by tokens and you set its weight to 15%, you are running code at 5 epochs while web runs at 0.6 — which is fine (up to ~4 epochs) but you must *know* that is what you did. Always report both the sampling weight and the implied epoch count per domain. I have seen a mixture that put a 40B-token math corpus at 12% of a 2T-token run, i.e. 6 epochs of math, silently.

**⚠ Trap:** treating the mixture as static. It is not, anymore. The modern practice is a schedule: general mixture for the bulk, then a very different high-quality mixture in the final decay phase. That is mid-training, and it is the biggest change to pretraining practice since Chinchilla.

### Teach me to read a loss curve. What does healthy look like, and what shapes mean something is wrong?

The loss curve is your only real-time instrument and it has a small vocabulary of shapes. Learn them the way you learned to read a latency histogram.

**Healthy.** On a log-x, linear-y plot, the curve is close to a straight line — that is the power law. Sharp drop in the first few hundred steps (learning the unigram distribution: loss falls to roughly the entropy of token frequencies, typically 4–6 nats), then smooth power-law decay, then a visible acceleration downward when the LR decay kicks in at the end. Small high-frequency jitter proportional to `1/sqrt(batch_size)` is normal and healthy — a *perfectly* smooth curve at small batch means someone is smoothing the plot.

**The shapes and their diagnoses:**

- **A single sharp spike that recovers within tens of steps.** Normal-ish. Usually a bad batch or a rare numerical event. Log it, move on.
- **A spike that does not recover, or recovers to a permanently higher level.** The optimizer state is corrupted — typically the Adam second moment `v` was poisoned by an enormous gradient, so the effective LR collapses for the affected parameters. Restart from the last good checkpoint. PaLM's authors reported exactly this and the fix that works: **restart from a checkpoint ~100 steps before the spike and skip forward past the offending batches**. The fact that skipping the data fixes it is itself the diagnostic that the data caused it.
- **Loss plateaus flat and then resumes.** Often benign (the model is reorganizing a representation), but if it lasts thousands of steps, check for a dead LR schedule, gradient clipping firing on every step, or a `NaN`-to-zero gradient path.
- **Loss decreases, then slowly *rises*.** Almost always LR too high for the current batch size, or a divergent numerical mode building up (attention logits growing without bound). Check the max attention logit and the grad-norm trend.
- **The data-bug shapes, which is what you actually asked.** The tell is *periodicity or a step discontinuity*, because both are things gradient descent does not do on its own.
  - **A sawtooth with a fixed period** = shard traversal with domain-homogeneous shards, or a repeating dataloader.
  - **A step change down** that is too fast for the LR schedule = you started seeing duplicated data, so the model is re-predicting text it has already memorized. This looks like a *win* and it is a bug. Cross-check with a held-out loss: if train loss steps down and held-out loss does not, you are memorizing.
  - **A step change up at an exact restart boundary** = the dataloader did not resume correctly, or the LR schedule was recomputed from step 0.
  - **Loss suspiciously low from step ~0** in a fine-tune or CPT = your held-out set is in your training set.

**⚠ Trap:** looking only at aggregate loss. Track **per-domain held-out loss** on a fixed set of a few thousand sequences per domain. Aggregate loss hides the case where code loss is exploding while web loss improves enough to mask it, which is exactly what a broken code-tokenization change looks like.

**🗣 Say this in the room:** "Healthy is a straight line on log-x with a downward kick at LR decay. Spikes that don't recover mean corrupted Adam state — restart before the spike and skip the batches. Anything *periodic* or *step-shaped* is a data-pipeline bug, because SGD doesn't produce those on its own."

### It's 3am on day nine, loss spiked to 6.2 and hasn't recovered in 400 steps. Talk me through what you do.

First: **stop the job and preserve state**. Every minute it runs is $700 of a 1000-GPU cluster at $2.50/GPU-hour (`1000 × 2.50 / 60 = $41.67/min`... call it $42/min, so 400 steps at 5s/step is 33 minutes ≈ $1,400 already burned). But do not delete anything — you need the poisoned optimizer state for diagnosis.

**The triage ladder, in order, because ordering by cost-to-check is the whole skill:**

1. **Is it NaN or a finite spike?** Check `grad_norm`, `loss`, and parameter norms. If any parameter is `NaN`/`Inf`, the run is dead and no amount of continued training recovers it — every subsequent update propagates the NaN. Straight to restart.
2. **Was it one bad batch?** Log the batch indices for the 20 steps around the spike (you *are* logging global batch indices — if not, that is the first fix after this incident). Decode those sequences and read them. The classic culprits: a document of 8,192 repetitions of the same character, a corrupted shard producing garbage token IDs, a document in an unexpected encoding. This takes 10 minutes and resolves it maybe 40% of the time.
3. **Check the numerics that predict spikes.** Max attention logit (growth toward the bf16 range is the classic pre-spike signal), output logit magnitude (this is what `z-loss` regularizes — PaLM added `1e-4 · log²(Z)` to keep the softmax normalizer near 1), and the Adam `v` statistics. If `v` for some parameter group collapsed toward zero right before the spike, the effective LR exploded there.
4. **Check hardware.** A silent data corruption on one rank produces a bad gradient that gets all-reduced into everyone. Run the NCCL/`dcgmi` health checks, look for a rank with anomalous grad norms in the steps before the spike, and check for ECC errors. This is more common than people expect at 1000-GPU scale.

**The recovery, which is standard practice and you should state it confidently:** restart from the checkpoint **before** the spike (not the most recent one, if the most recent one is post-spike), and **skip forward past the batches implicated**, typically a few hundred batches. Because the dataloader is a deterministic function of `(seed, step)`, this is a config change, not surgery. Then lower LR modestly for a few thousand steps if you are nervous.

**The preventive measures I would add to the run afterward** — these are cheap and every one of them is standard in production training stacks: gradient clipping at global norm 1.0 (you already have this; verify it is actually firing), z-loss on the output logits, QK-norm (RMSNorm on queries and keys before the dot product, which directly bounds attention logit growth), skipping any batch whose grad norm exceeds `k×` the running median, and bf16 rather than fp16 everywhere (fp16's 5-bit exponent overflows at 65,504 and is a spike factory; bf16 has fp32's exponent range).

**💰 Math on checkpoint cadence, decided by this incident:** if you were checkpointing every 2,000 steps at 5s/step, you just lost up to 2.8 hours of a 1000-GPU cluster = `1000 × 2.8 × 2.50 = $7,000` of redone work, plus 33 minutes of spiked training. Applying the Young/Daly optimum with `t_ckpt = 90s` and observed `T_fail ≈ 4h = 14,400s`: `T = sqrt(2 × 90 × 14400) = sqrt(2.59e6) ≈ 1,610s` ≈ 320 steps. Checkpoint every ~300 steps, not every 2,000.

**🗣 Say this in the room:** "Stop the job, check for NaN, then read the actual text of the batches around the spike — it's a bad document 40% of the time. Recovery is restart-before-the-spike and skip the batches, which works because the dataloader is deterministic in (seed, step). Then I'd add z-loss and QK-norm, since attention-logit growth is the usual precursor."

### What's your checkpointing strategy, and what exactly is in a checkpoint?

**What's in it,** and the byte accounting matters because it determines everything else. For mixed-precision AdamW training, per parameter you store:

- fp32 master weights: 4 bytes
- Adam first moment `m` (fp32): 4 bytes
- Adam second moment `v` (fp32): 4 bytes

= **12 bytes/param** in the checkpoint. (During training you additionally hold bf16 params and bf16 gradients — 2 + 2 — for the well-known ~16 bytes/param resident cost, but gradients need not be checkpointed.) Plus the RNG states, the LR-schedule position, the global step, the dataloader seed, and — this is the one people forget — **the exact data mixture config and code commit**.

**💰 Math:** an 8B model → `8e9 × 12 = 96 GB` per checkpoint. A 70B → 840 GB. A 405B → 4.9 TB. At 10 GB/s aggregate write bandwidth, the 405B checkpoint takes 8 minutes of blocked training. On 1000 GPUs that is `1000 × 8/60 × 2.50 = $333` per checkpoint, and if you keep 50 of them at 4.9 TB each that is 245 TB of storage.

**Therefore, the three techniques that make this tractable:**

1. **Sharded checkpointing.** Under FSDP/ZeRO the optimizer state is already partitioned across ranks; each rank writes only its shard, in parallel. Aggregate write bandwidth scales with rank count, so the 8-minute write becomes seconds. PyTorch's Distributed Checkpoint (DCP) is the standard API. The cost is that a checkpoint is now a directory of shards tied to a world size — you need a resharding tool to restart on a different GPU count, and you *will* need to restart on a different GPU count.
2. **Asynchronous checkpointing.** Copy state to pinned host memory (fast, blocking for seconds), then write to storage from a background thread while training continues. This turns an 8-minute stall into a ~20-second one. It is now standard and it directly changes the Young/Daly optimum by shrinking `t_ckpt`.
3. **Tiered retention.** Keep every checkpoint for the last N steps (for spike recovery), every 1,000th for the last 20k (for ablation), and every 10,000th forever (for the record, and because someone will want to branch mid-training experiments off them). This is a retention policy problem identical to backup retention and should be automated on day one.

**⚠ Trap:** checkpointing weights but not optimizer state, to "save space." Restarting from weights alone throws away `m` and `v`, so Adam restarts with zero momentum and effectively re-warms up — you get a visible loss bump and lose real progress. It is only acceptable for the *final* artifact you ship, never for a mid-run checkpoint.

**⚠ Trap:** not verifying that a checkpoint loads. A silently truncated checkpoint discovered during a 3am recovery is a career-defining experience. I run a load-and-forward-pass verification on every Nth checkpoint as a separate cheap job, and assert the loss on a fixed batch matches the training log.

### What does the eval harness look like during a pretraining run, and how often do you run it?

The mental model: you have three tiers of signal at three different costs and latencies, and the mistake is running only the expensive one.

**Tier 1 — every step.** Training loss, grad norm, LR, per-rank step time, `data_wait_ms`, max attention logit, tokens/sec, MFU. These are counters, they cost nothing, and they are what you page on. Alert on: grad norm exceeding `k×` its trailing median, MFU dropping more than 10% from baseline, any NaN, and step-time variance.

**Tier 2 — every few hundred steps.** **Held-out loss, per domain**, on a small fixed set (a few thousand sequences per domain, frozen for the life of the project). This costs one forward pass over ~10M tokens — for an 8B model that is `2 × 8e9 × 1e7 = 1.6e17` FLOPs, about 7 GPU-minutes, entirely negligible. This is your highest-value metric and the one that catches data bugs, because it is *continuous* and *decomposed*.

**Tier 3 — every few thousand steps, or at checkpoints.** The actual benchmark suite: MMLU, ARC, HellaSwag, GSM8K, HumanEval, whatever matters for your target. Run it as a **separate job on separate GPUs against the written checkpoint**, never inline in the training loop — inline evaluation is a synchronization barrier across the whole cluster and a great way to lose 3% of your run to benchmark harness bugs.

**The design rules I would enforce:**

- **Freeze the harness before the run.** Every prompt template, few-shot count, and scoring rule pinned in a versioned config. Comparing step 10,000 to step 50,000 is only valid if nothing about the measurement changed. Changing the eval mid-run and then comparing across it is the most common self-inflicted wound in this whole area.
- **Prefer continuous metrics early.** For the first ~20% of training, most multiple-choice benchmarks are at chance and give you zero information. Use log-prob margin on the correct answer, or bits-per-byte on domain-specific held-out text, which move from step one.
- **Run a known-good reference model through the same harness.** If your harness reports Llama-3-8B at 45% MMLU when the world says ~66%, your harness is broken, not the model. This is the single check that saves the most wasted debugging, and it takes an hour to set up.
- **Keep one private eval set that never leaves your infrastructure**, for the contamination reasons discussed earlier.

**⚠ Trap:** deciding to kill a run early based on benchmarks in the first 10–20% of training. Benchmark scores are near-chance and near-random then; two mixtures that are genuinely different will look identical, and a mixture that looks better at 5% frequently loses at 100%. Kill runs on *loss* and on *engineering* signals (MFU, instability), not on benchmarks, until you are well past the point where benchmarks have separated from chance.

### What is the LR schedule and warmup for a pretraining run, and what does each piece actually do?

**Warmup** exists because Adam's second-moment estimate `v` is initialized to zero and needs a few hundred steps of gradients before it is a meaningful denominator. Before that, the update `m/(sqrt(v)+ε)` is badly scaled and a full-size LR blows up the weights, permanently. Typical: linear ramp from 0 to peak over **0.1–1% of total steps** — a few hundred to a few thousand steps. Long warmup is cheap insurance; the cost of too-short warmup is a dead run.

**Peak LR** scales inversely with model width. The rough family: ~3e-4 at 1B, ~1.5e-4 to 3e-4 at 7–8B, ~1e-4 at 70B, lower still at frontier scale. The principled version is **μP** (maximal update parametrization), which reparametrizes initialization and per-layer LR so that the *optimal* LR is invariant to width — meaning you can tune LR on a 40M-param model and transfer it to a 40B one. **📄 Paper:** Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* — this is the reason a modern lab does not have to guess the LR for its largest run.

**Decay.** Cosine decay to ~10% of peak was the default for years. Its defining property is also its defining flaw: **the schedule depends on the total token count you declared at step 0.** Stop early and you get a model that never had its LR annealed, which is measurably worse than a model trained on fewer tokens with a proper decay. Want to extend the run? You cannot, without a discontinuity.

**WSD (Warmup–Stable–Decay)** fixes exactly this: warm up, hold LR **constant** for the bulk of training, then decay rapidly over the final ~10% of tokens. **📄 Paper:** Hu et al. (2024), *MiniCPM* — popularized WSD and showed it matches or beats cosine while making the run *branchable*: because the stable phase has no schedule state, any point in it is a valid starting point, so you can fork one long stable trunk into multiple short decay phases with different data mixtures and get several finished models for the price of one trunk plus `k` short decays.

**💰 Math on why that matters:** suppose the trunk is 10T tokens and each decay phase is 500B. Producing 4 model variants with cosine requires 4 full runs: `4 × 10.5T = 42T` tokens. With WSD: `10T + 4 × 0.5T = 12T` tokens. That is a **3.5× reduction**, or on our earlier numbers the difference between $1.26M and $4.4M. This is the actual reason WSD won — not a loss improvement, an experimental-throughput improvement.

**Other pieces of the recipe:** AdamW with `β1 = 0.9`, `β2 = 0.95` (lower than the 0.999 default — LLM gradients are noisier and a shorter second-moment window adapts faster), `ε = 1e-8`, weight decay 0.1 applied to weights but **not** to biases, LayerNorm/RMSNorm gains, or embeddings, and global gradient-norm clipping at 1.0.

**⚠ Trap:** applying weight decay to the embedding matrix. Rare tokens' embeddings receive gradient signal only when those tokens appear, but decay shrinks them *every step*, so rare embeddings decay toward zero between appearances. Excluding embeddings and norm parameters from decay is standard and it is a one-line config error to get wrong.

### Our MFU was 47% and this morning it's 31% with no code change. Debug it.

A 16-point MFU drop with no code change is an infrastructure question, and the discipline is to bisect by *where the time went* before theorizing. On a 1000-GPU cluster that drop costs `(0.47−0.31)/0.47 = 34%` of throughput; at $2.50/GPU-hour that is `1000 × 24 × 2.50 × 0.34 = $20,400/day`. Worth an hour of careful work.

**Step 1 — is it uniform across ranks, or is one rank slow?** Emit per-rank step time. Synchronous data-parallel training runs at the speed of the slowest rank, so *one* degraded GPU drags all 1000. Look for a rank with consistently higher compute time. Causes: thermal throttling (check `nvidia-smi` clocks and power), an ECC error triggering row remapping, a GPU that fell back to a lower clock after a fault. **This is the most common cause of an overnight MFU regression and it is a hardware replacement, not a code fix.**

**Step 2 — is it compute time or wait time?** Decompose step time into forward, backward, optimizer, all-reduce, and data wait. A profiler trace on one rank for 20 steps answers this in minutes.
- **`data_wait_ms` grew** → dataloader or storage. Did someone else start a big job on the same storage backend? Did a shard get moved to colder storage? Is the page cache cold after a node restart?
- **All-reduce time grew** → network. Check for a degraded NIC or link (one flapping InfiniBand link makes the whole ring slow), check whether NCCL fell back to a different algorithm or a slower transport (this happens silently — `NCCL_DEBUG=INFO` will tell you which channels and protocol it chose), and check whether some other job is sharing the fabric.
- **Compute time grew uniformly** → clocks, or a driver/library change. Verify nothing auto-updated.

**Step 3 — did anything about the *data* change?** This is the one people miss. If your batch's sequence composition changed — say a new shard has far more short documents, so packing produces more documents per sequence — the varlen attention kernel's block-sparsity pattern changes and throughput moves. Similarly, if you just entered a long-context stage, sequence length went up and the attention quadratic now dominates, so MFU computed with `6ND` legitimately drops even though the hardware is working harder. **Check the sequence length and the mean documents-per-sequence before concluding anything is broken.**

**Step 4 — is it a straggler in the collective rather than in compute?** With pipeline parallelism, a single slow stage creates bubbles that show up as idle time on every other stage. Look at the pipeline bubble fraction directly if you have it instrumented.

**🔍 Failure taxonomy for MFU regressions, as a decision procedure:**
1. Per-rank step time bimodal → **one bad GPU/node**. Drain it, replace it, restart. (Most common.)
2. Uniform slowdown, all-reduce dominant → **network degradation or noisy neighbor**. Check link health, NCCL topology.
3. Uniform slowdown, data wait dominant → **storage contention or cold cache**. Check IOPS on the backing store.
4. Uniform slowdown, compute dominant, clocks normal → **library/driver drift**. Diff the container image.
5. No slowdown in wall-clock, only in reported MFU → **you changed sequence length or model shape**; the metric moved, the machine did not.

**🗣 Say this in the room:** "First question is whether the slowdown is uniform or one rank, because synchronous DP runs at the speed of the slowest rank and a single throttling GPU costs the whole cluster. Then I decompose step time into compute, collective and data-wait — those three point at three different teams."

### Give me the ninety-second version of scaling laws for a VP who wants to know why we can't just train a bigger model.

**🗣 Say this in the room** — this is the whole answer, delivered as one:

"Model quality follows a predictable curve in three inputs: model size, data volume, and compute. Because it is predictable, we can plan — we run small cheap experiments and extrapolate, so we know roughly what a big run will produce before we spend the money. That is the good news.

The bad news is the shape of the curve. It is a power law, which means **equal improvements cost exponentially more each time**. Roughly, every time we want to move the quality needle by the same increment, we need about ten times the compute. So the question is never 'can we train a bigger model,' it is 'is the next increment of quality worth ten times the budget.'

And there is a second constraint that surprises people. A bigger model does not just cost more to train once — it costs more *every time anyone uses it, forever*. For a model we actually ship, the lifetime serving bill is typically ten to a hundred times the training bill. So the industry deliberately trains *small* models on *enormous* amounts of data: it costs the same to train, and it is five to ten times cheaper to run.

Which is why my recommendation is usually not 'train a bigger model.' It is: take the best available base model, spend a small fraction of that budget adapting it to our domain, and spend the rest on the data and evaluation work that actually determines whether the thing is useful. The scaling curve says the marginal compute dollar is the *worst*-returning dollar we can spend right now."

If they push for the number: "Chinchilla's rule is about 20 training tokens per parameter for the cheapest possible training. Everyone who ships ignores it and uses 100 to 2,000 tokens per parameter, deliberately over-training a small model, precisely because serving cost dominates. Llama-3's 8-billion-parameter model saw 15 trillion tokens — that is 1,875 per parameter, ninety times the 'optimal' ratio, and it was the right call."
