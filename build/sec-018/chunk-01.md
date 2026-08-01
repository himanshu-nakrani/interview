### Before we go deep on any one piece — walk me end to end through what a pretraining run actually is, as a pipeline.

Think of it as an ETL job whose output artifact happens to be a tensor instead of a table, and whose "unit test suite" is a set of benchmarks that only tell you the truth after you have already spent a million dollars. That framing gets the priorities right: almost everything that determines the quality of the final model is decided in the data stages, and almost everything that determines whether the run *finishes* is decided in the infrastructure stages. The modelling stage — "which architecture" — is the smallest lever of the three, which is exactly the opposite of what people expect.

The stages, in order, with what actually gets produced:

1. **Corpus acquisition.** Common Crawl WARC files, curated sources (books, papers, code hosting, Wikipedia), licensed data, and increasingly synthetic rewrites. Output: petabytes of raw compressed bytes in object storage.
2. **Extraction.** HTML → text. `trafilatura` or `resiliparse`. Output: JSONL documents with provenance metadata (URL, crawl date, extractor version).
3. **Filtering.** Language ID, Gopher/C4 heuristic rules, a learned quality classifier, PII redaction, safety filters. This is where 90–99% of raw web bytes die.
4. **Deduplication.** Exact-substring (suffix array) plus MinHash-LSH near-dedup, optionally SemDeDup on embeddings. Global, not per-shard.
5. **Decontamination.** N-gram overlap removal against every eval set you will ever report, plus canary insertion.
6. **Tokenizer training and tokenization.** Usually BPE, trained on a stratified sample of the *final* mix. Output: `uint16`/`uint32` token arrays.
7. **Shard and pack.** Concatenate documents into fixed-length sequences with document boundaries recorded, written into shards sized for your dataloader, with a deterministic shuffle seed.
8. **Mixture and curriculum.** Per-domain sampling weights, and a schedule if those weights change over training (they almost always do now — see mid-training).
9. **The training loop.** Parallelism strategy, optimizer, LR schedule, batch-size ramp, precision policy, checkpointing cadence, restart-from-failure machinery.
10. **The eval harness.** Cheap in-loop metrics every few hundred steps, a full benchmark suite at checkpoints, and a fixed held-out loss set that never changes for the life of the project.

**⚠ Trap:** treating the eval harness as stage 10. Build it as stage 0. If you cannot run your full eval suite on a randomly-initialized model and on a public 1B baseline *before* the run starts, you have no way to distinguish "the model is bad" from "the harness is broken," and you will discover that ambiguity at 3am on day nine.

**🗣 Say this in the room:** "Pretraining is a data-engineering problem wearing a machine-learning costume. The architecture is maybe 5% of the outcome; the data pipeline and the failure-recovery machinery are the other 95%, and the eval harness has to exist before step one."

### Derive C = 6ND for me. Where does the 6 come from?

This is the single most useful equation in the field and it is derivable in thirty seconds, so an interviewer asking it is testing whether you memorized a formula or understand a forward pass.

Start with one matrix multiply. A linear layer with weight matrix `W ∈ R^{m×n}` applied to one token's activation vector costs `m×n` multiply-accumulate operations, and by universal convention one MAC is counted as 2 FLOPs. So the **forward pass costs 2 FLOPs per parameter per token**. That is the whole content of the "2".

Now the backward pass. For each layer you must compute two gradients: the gradient with respect to the *inputs* (to propagate backward) and the gradient with respect to the *weights* (to update). Each is another matmul of the same shape as the forward one. So backward costs 2× the forward: **4 FLOPs per parameter per token**.

Total: `2 + 4 = 6` FLOPs per parameter per token. For `N` parameters and `D` training tokens, `C = 6ND`.

What the formula silently ignores: attention's quadratic term. The `QK^T` and `attn·V` products cost roughly `12 · n_layers · d_model · seq_len` FLOPs per token extra, which is *not* proportional to parameter count. As a fraction of total, that term is `seq_len / (6 · d_model)`-ish. For Llama-3-8B at `d_model = 4096` with `seq_len = 8192`, that is `8192 / (6 × 4096) ≈ 33%` — very much not negligible. At `seq_len = 2048` it is 8%. So `6ND` is a good approximation at short context and a *systematic underestimate* at long context.

**📐 Numbers you must know:** 2 FLOPs/param/token forward, 4 backward, 6 total. Inference *prefill* is 2ND. Inference *decode* is also 2N FLOPs per generated token, but decode is memory-bandwidth-bound, not FLOP-bound, so that number tells you almost nothing about decode latency — it tells you about cost on a saturated batched server.

**⚠ Trap:** using `6ND` for a Mixture-of-Experts model with total parameters. Use **active** parameters. DeepSeek-V3 has ~671B total and ~37B active; its training compute is `6 × 37e9 × D`, not `6 × 671e9 × D`. Getting this wrong overestimates the cost by 18×, and an interviewer will catch it instantly.

**🗣 Say this in the room:** "Six FLOPs per parameter per token — two forward because a MAC is two FLOPs, four backward because you compute both input-grads and weight-grads. It ignores the attention quadratic, which becomes a third of the budget once you're at 8k context on a small model."

### Kaplan 2020 said one thing, Chinchilla 2022 said another. What changed, and was Kaplan simply wrong?

**📄 Paper:** Kaplan et al. (2020), *Scaling Laws for Neural Language Models* — established that loss falls as a smooth power law in parameters, data and compute, and that these are predictable enough to plan a run. It replaced "train the biggest thing you can and hope."

**📄 Paper:** Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla) — re-ran the compute-optimal analysis and found parameters and tokens should scale roughly **equally**, giving the famous ratio of about **20 tokens per parameter**. It replaced Kaplan's prescription that you should spend marginal compute mostly on parameters.

The mental model for the difference: both papers ask "given a fixed compute budget `C = 6ND`, how do I split it between `N` and `D` to minimize loss?" Kaplan's answer was roughly `N ∝ C^0.73`, `D ∝ C^0.27` — grow the model much faster than the data. Chinchilla's was `N ∝ C^0.5`, `D ∝ C^0.5`. The practical consequence was brutal: GPT-3 at 175B params and 300B tokens is about 1.7 tokens/param, roughly **12× under-trained** by Chinchilla's rule. Chinchilla at 70B/1.4T beat it on nearly everything while being 2.5× cheaper to serve.

Was Kaplan wrong? Partly, and for boring methodological reasons rather than conceptual ones. The two differences that later re-analysis has focused on: (a) Kaplan held the learning-rate schedule roughly fixed rather than decaying it to match each run's token count, which systematically penalizes the long-data runs and biases the fit toward more parameters; and (b) Kaplan counted non-embedding parameters and fit over a smaller model range, which distorts the exponents at small `N` where embeddings are a large fraction. There is also honest ongoing debate about the exact Chinchilla exponents — an independent replication of the Chinchilla fits found the reported parametric-loss coefficients did not reproduce cleanly from the published data, though the ~20:1 headline held up qualitatively.

**⚠ Trap:** calling 20 tokens/param "the optimal ratio." It is the *training-compute-optimal* ratio, and it is optimal for exactly one objective: minimizing pretraining loss for a fixed pretraining FLOP budget. Nobody deploying a model has that objective. This is the setup for the next question and it is the single most common place a candidate reveals they learned scaling laws from a blog post.

**🗣 Say this in the room:** "Chinchilla's 20:1 answers 'what minimizes loss per training FLOP.' That is the wrong objective function for anyone who has to serve the model, which is why every production model since 2023 is deliberately, massively over-trained relative to it."

### Then why is Llama-3 8B trained on 15 trillion tokens — about 1,800 tokens per parameter, 90× past Chinchilla?

Because the objective function changed from "minimize training cost" to "minimize training cost **plus lifetime inference cost**," and for any model that actually gets deployed, the second term dominates by an order of magnitude or more.

Here is the arithmetic that makes it inevitable. Take a fixed training budget of `C = 7.2e23` FLOPs.

**Option A — Chinchilla-optimal.** With `D = 20N`, `C = 6N(20N) = 120N²`, so `N = sqrt(7.2e23 / 120) = sqrt(6e21) ≈ 7.75e10`. That is a **77B model trained on 1.55T tokens**.

**Option B — heavily over-trained.** `N = 8e9`, so `D = C / (6N) = 7.2e23 / 4.8e10 = 1.5e13`. That is an **8B model trained on 15T tokens**.

Both cost identically to train. Now serve them. Inference forward cost is `2N` FLOPs per token, so per served token the 77B model costs `2 × 7.75e10 = 1.55e11` FLOPs and the 8B costs `2 × 8e9 = 1.6e10` — a **9.7× ratio**. Memory is worse in practice: 8B at bf16 is 16 GB and fits on one accelerator with room for KV cache; 77B at bf16 is 155 GB and needs at least two H100s with tensor parallelism, which adds all-reduce latency to every single decode step.

**💰 Math:** suppose the model serves `1e15` tokens over its life (that is 2.7 billion tokens/day for a year — a mid-size product feature, not a frontier chatbot). Inference FLOPs for the 8B: `2 × 8e9 × 1e15 = 1.6e25`. That is **22× the entire training budget** of `7.2e23`. For the 77B: `1.55e26`, or 215× training. The training run is a rounding error against the inference bill. So you spend your training compute buying a *smaller* model that is as good as you can make it, not the biggest model your budget allows.

The countervailing force is that over-training has diminishing returns — loss keeps falling with more tokens but the power law flattens, and past roughly 100–200 tokens/param you are paying a lot of compute for small deltas. The reason labs still do it is that the alternative (a bigger model) costs *more forever*, so even a small quality gain per training dollar is worth it.

**⚠ Trap:** stating "over-training is free quality." It is not. Beyond the Chinchilla point you are trading training compute for inference compute at a worsening exchange rate, and past ~4 epochs of repeated data you also start paying a data-repetition penalty. The honest statement is that the *inference-adjusted* optimum sits far to the right of Chinchilla, and where exactly depends on your projected serving volume.

**🗣 Say this in the room:** "Chinchilla optimizes training FLOPs. We optimize training-plus-lifetime-inference FLOPs, and for anything that ships, inference is 10–200× training. That pushes the optimum toward small, heavily over-trained models — which is exactly the 8B-on-15T shape everyone converged on."

### You have 1000 H100s for three weeks. What do you train?

Let me do the arithmetic first and the judgement second, because the arithmetic constrains the judgement completely.

**Compute available.** Three weeks is `21 × 86,400 = 1,814,400` seconds. An H100 SXM does **989 TFLOP/s dense BF16** (the 1,979 number you see quoted is with 2:4 structured sparsity — do not use it). Cluster peak: `989e12 × 1000 = 9.89e17` FLOP/s. At a realistic **40% MFU** for a dense model on a well-tuned 1000-GPU job: `3.96e17` FLOP/s sustained. Total: `3.96e17 × 1.8144e6 = 7.2e23 FLOPs`. That is 504,000 GPU-hours.

**💰 Math:** at a 2026 reserved rate of roughly $2.50/H100-hour, that is `504,000 × 2.50 = $1.26M` of compute. **📅 Volatile:** H100 spot/reserved pricing has fallen steeply and varies 2–4× across providers — verify before you quote it.

**What that budget buys.** From `C = 6ND`: `ND = 1.2e23`. So — 8B params × 15T tokens, or 30B × 4T, or 70B × 1.7T, or a 100B-total MoE with 12B active on ~10T tokens. Note that 8B/15T is *exactly the Llama-3-8B shape*, which is a useful sanity anchor: three weeks on a thousand H100s is one frontier-class small model, and not one frontier-class large one.

**Now the judgement, which is the actual answer.** I would not train from scratch. Here is my decision procedure:

- **Do I have 15T tokens of clean, deduped, decontaminated data ready?** If not — and a team that has to ask this question does not — then the data pipeline is a 3–6 month project on its own and the GPUs will sit idle. Building FineWeb-scale data is harder than the training run.
- **Does an open-weight model at this scale already exist?** Yes, several, trained on more data than I can curate, by teams with more pretraining experience. Matching Qwen or Llama at 8B from scratch with $1.26M is not a good bet; the honest expectation is that I land meaningfully worse.
- **So what do I do with the compute?** I would spend it on **continued pretraining plus mid-training on a strong open base**, which is where the marginal return per FLOP is enormous for an applied team. Concretely: 200–500B tokens of domain-heavy mix with 20% general replay on a 8B–32B base is maybe 5% of the budget; the remaining 95% goes to (a) a long-context extension stage, (b) an annealed high-quality decay phase, (c) SFT and RLVR runs, and (d) — this is the part people underspend — **dozens of small ablation runs at 300M–1B scale** to actually learn which of your data decisions help.

**⚠ Trap:** answering this question with a model size and no data plan. The interviewer is checking whether you know that compute is the *easy* resource to acquire. If you say "I'd train a 30B on 4T tokens" and cannot immediately say where 4T clean tokens come from, you have failed the question regardless of the arithmetic.

**🗣 Say this in the room:** "A thousand H100s for three weeks is about 7×10²³ FLOPs, which is one 8B-on-15T run. But I wouldn't spend it that way — I'd take a strong open base, spend 5% on domain CPT and mid-training, and spend the rest on ablations and post-training, because I can't beat Qwen at general pretraining and I can beat everyone on my domain."

### Define MFU precisely and compute it for a run I describe to you.

**Model FLOPs Utilization** is the fraction of your hardware's peak arithmetic throughput that is actually doing useful model math. It is the pretraining equivalent of asking what fraction of your CPU cycles are doing your business logic versus syscall overhead — except the denominator is a vendor number and the numerator is a formula you choose, which is why MFU numbers across papers are not always comparable.

```
MFU = (6 · N · tokens_per_second) / (peak_FLOPs_per_GPU · num_GPUs)
```

Worked example. Suppose you train a 7B dense model on 512 H100s and your logs report 3.9 seconds per step with a global batch of 4,194,304 tokens (512 sequences × 8192 tokens per GPU-group — whatever, take the token count).

- Tokens/sec = `4.194e6 / 3.9 = 1.0754e6`.
- Numerator = `6 × 7e9 × 1.0754e6 = 4.517e16` FLOP/s.
- Denominator = `989e12 × 512 = 5.064e17` FLOP/s.
- **MFU = 4.517e16 / 5.064e17 = 8.9%.** That is terrible, and now you have a number to debug against instead of a feeling.

**📐 Numbers you must know:** healthy dense-transformer MFU on H100 clusters is **35–55%**. Below 30% means something structural is wrong (bad parallelism config, dataloader starvation, unfused kernels, tiny microbatch, communication not overlapped). Above 60% is essentially only reachable on very large models with long sequences and near-perfect overlap; if someone claims 70% dense BF16, ask whether they are using peak-with-sparsity in the denominator. As a public anchor: Meta's disclosed H100-hours for the Llama-3 405B run (~30.8M hours for 15.6T tokens) implies `6 × 405e9 × 15.6e12 / (30.8e6 × 3600 × 989e12) ≈ 34%` — a real frontier run, in the mid-30s. **📅 Volatile:** verify the published hour counts before quoting them.

Two variants you must be able to distinguish:

- **MFU** uses `6ND` — model FLOPs only, no recomputation counted.
- **HFU (Hardware FLOPs Utilization)** counts FLOPs the hardware actually executed, *including* activation-recomputation. If you use full activation checkpointing you re-run the forward pass, so HFU ≈ MFU × (8/6) = 1.33× MFU. Papers that quote HFU look better for free.

**⚠ Trap:** comparing your MFU to a published number without checking sequence length. The attention quadratic is not in the `6ND` numerator, so a run at 32k context is doing far more real work than `6ND` credits it for, and its MFU will look artificially *low*. Conversely people sometimes add the attention term to the numerator, which makes long-context MFU look artificially high. Always state which convention you used.

### What is goodput and why does it diverge from throughput on a big cluster?

Throughput is tokens/second while the job is running. **Goodput is tokens/second amortized over wall-clock, including everything that is not forward-backward**: crashed jobs, restarts, checkpoint writes, straggler nodes, failed health checks, the four hours you spent bisecting a NCCL hang. It is exactly the distinction between "requests/sec while healthy" and "requests/sec over the quarter including the incidents," and at cluster scale the gap is enormous.

The mechanism that makes it enormous is a hardware-reliability argument you can do in your head. If a single GPU node has a mean time between failures of `M` hours, a job that requires all `n` nodes simultaneously has an effective MTBF of roughly `M / n`. Synchronous data-parallel training is an AND across every rank — one dead GPU kills the step, and therefore the job.

**📐 Numbers you must know:** at 16,000 GPUs with a generous per-GPU MTBF of 50,000 hours, cluster MTBF is `50,000 / 16,000 ≈ 3.1 hours`. You will crash multiple times a day, forever, and that is the *normal* state. Meta's Llama-3 405B run publicly reported on the order of 400+ unexpected interruptions across a ~54-day window on a 16k-GPU cluster — roughly one every three hours — with the large majority hardware-attributed, while still achieving over 90% effective training time. Both halves of that sentence matter: the failures are constant, and good engineering still recovers 90% goodput.

Goodput decomposes as:

```
goodput = MFU × uptime_fraction × (1 − checkpoint_overhead) × (1 − wasted_work_fraction)
```

where `wasted_work_fraction` is the training you redo after each crash — on average half a checkpoint interval per failure. That term is why checkpoint cadence is an optimization problem, not a preference:

**💰 Math:** if checkpointing costs `t_ckpt = 60s` and you checkpoint every `T` seconds with failures every `T_fail = 11,000s` (3.1 hours), expected overhead is `t_ckpt/T + (T/2)/T_fail`. Differentiating, the optimum is `T = sqrt(2 · t_ckpt · T_fail) = sqrt(2 × 60 × 11000) = sqrt(1.32e6) ≈ 1,150s` — checkpoint every ~19 minutes. Total overhead at that cadence: `60/1150 + 575/11000 = 5.2% + 5.2% = 10.4%`. That is the Young/Daly result and it is the single most useful formula in training infrastructure.

**⚠ Trap:** optimizing MFU while ignoring goodput. I have seen a team win 6% MFU with a more aggressive parallelism config that raised the OOM-crash rate, and net-lose 15% goodput. The metric that pays for the cluster is tokens-per-wall-clock-dollar, not tokens-per-healthy-second.

### How do you choose batch size, and why is it always quoted in tokens rather than sequences?

Tokens, because the only thing the optimizer cares about is how many gradient samples went into the average, and a "sequence" is an arbitrary unit that changes when you change context length. If you report "batch size 512" and then switch from 4k to 8k context, you have silently doubled your true batch size and your learning rate is now wrong. Quoting `global_batch_tokens = micro_batch × grad_accum × data_parallel_degree × seq_len` removes the ambiguity.

The mental model for *why* there is an optimum: gradient descent with a batch of size B has gradient noise that falls as `1/sqrt(B)`. Below a certain size, doubling the batch genuinely halves the noise and lets you take a proportionally bigger step — you get near-linear speedup in wall-clock for the same number of examples seen. Above that size the gradient is already essentially the true gradient, so doubling B buys you almost nothing and you are simply burning 2× the FLOPs per step. That crossover is the **critical batch size**.

**📄 Paper:** McCandlish et al. (2018), *An Empirical Model of Large-Batch Training* — introduced the *gradient noise scale*, a measurable quantity that predicts the critical batch size, and showed it grows over the course of training as the loss landscape gets easier. This is the reason modern runs **ramp** batch size rather than fixing it.

**📐 Numbers you must know:** frontier dense runs sit in the **2M–16M tokens per step** range, ramping upward during training. Llama-3's 405B run is publicly described as ramping — starting around 4M tokens/step at 4k sequence length, doubling to 8M, and later to 16M as training progressed. **📅 Volatile:** verify exact figures against the model card.

The practical constraints that actually pick the number for you:

- **Lower bound from parallelism.** Global batch must be `≥ data_parallel_degree × micro_batch × seq_len`. On 1024 GPUs with DP degree 128 and 8k sequences, one microbatch each already forces ≥1M tokens. You cannot go below your cluster's shape without idling GPUs.
- **Upper bound from sample efficiency.** Past critical batch size you waste compute. Symptom: loss-per-token curve gets worse when you double batch at fixed LR-scaled-appropriately.
- **LR coupling.** Batch and LR move together. The rules of thumb are linear scaling (`lr ∝ B`) at small batch and square-root scaling (`lr ∝ sqrt(B)`) at large batch; in practice for LLM pretraining people tune LR at one batch size and use square-root-ish scaling with a warmup long enough to absorb the error.

**⚠ Trap:** raising batch size to fix a loss spike and forgetting to re-tune LR or warmup. Larger batch with the same LR is *more* stable, so it often does fix the spike — and simultaneously costs you sample efficiency you will never notice because you have no counterfactual. Fix spikes with the data and the numerics, not by inflating batch.

### Data is finite. What happens if I train four epochs on the same corpus instead of one epoch on 4× the data?

Much less than you would fear, which is one of the genuinely useful empirical results of the last few years.

**📄 Paper:** Muennighoff et al. (2023), *Scaling Data-Constrained Language Models* — trained models under a fixed unique-token budget with varying repetition and fit a scaling law with a data-repetition decay term. Headline finding: **repeating data for up to ~4 epochs yields loss almost indistinguishable from fresh data**; returns then decay quickly, and by roughly 16+ epochs additional repetition contributes essentially nothing. It replaced the folk rule "never repeat data, it memorizes," which came from an era of much smaller models relative to corpus size.

The mental model: a repeated token is not worthless, it is *discounted*. The model has already extracted some of the signal, so the second pass provides less new information than the first — but "less" is not "none," and for the first few passes the discount is small enough that it beats the alternative, which is padding your corpus with lower-quality data you filtered out for a reason. The paper's practical corollary is exactly that: **repeating good data beats adding bad data**, up to a point.

The consequence for planning is a decision procedure. Given a unique-token budget `U` and a compute budget implying `D` training tokens:

- `D / U ≤ 4` → just repeat. Cost you essentially nothing. This is the regime almost every applied team is in.
- `4 < D / U ≤ 16` → repeat, but also shrink `N` and spend the freed compute elsewhere; the effective value of your data is decaying, so the compute-optimal model gets *smaller* under data constraints.
- `D / U > 16` → stop. Go acquire, license or synthesize data, or accept a smaller model. More epochs are not the answer.

**⚠ Trap:** conflating "repetition is fine" with "duplication is fine." These are opposites. Controlled repetition means you see the whole deduped corpus `k` times, in shuffled order. Duplication means one document appears 400 times because it is a boilerplate footer, so the model sees it 1,600 times while seeing a good document 4 times. Duplicated data measurably degrades models and increases verbatim memorization; controlled epoching does not. Dedup first, then epoch.

**💰 Math:** for a domain CPT run this decides your budget. If you have 20B unique domain tokens and want a 100B-token CPT run, that is 5 epochs — right at the edge. I would instead run 4 epochs of domain (80B) plus 20B of general replay, which gets me the token count *and* the forgetting mitigation from the same decision.

### How do scaling laws change for Mixture-of-Experts models?

The mental model: an MoE decouples the two things that dense scaling ties together. In a dense model, "capacity to store knowledge" and "compute spent per token" are the same knob — parameters. An MoE lets you buy capacity with total parameters while paying compute only for active parameters. Scaling laws therefore gain a dimension: loss depends on active params, total params (or expert count), and tokens, roughly separately.

**📄 Paper:** Clark et al. (2022), *Unified Scaling Laws for Routed Language Models* — the first careful fit across routing methods, finding that routed models follow a scaling law in both parameter count and expert count, with the benefit of more experts diminishing as the base model grows. Later work on *fine-grained* MoE (splitting into many smaller experts with higher top-k) found that expert granularity is itself a scaling dimension, and this is the design DeepSeek popularized with shared plus fine-grained routed experts.

The operational summary I would give in an interview:

- **At fixed training FLOPs, an MoE beats a dense model.** The commonly-cited efficiency multiplier is in the range of 2–7× depending on sparsity and how you define "equivalent," meaning an MoE reaches a given loss for a fraction of the dense training compute.
- **At fixed memory, dense wins.** An MoE with 8× the total parameters needs 8× the weight memory for serving, even though it only computes with a slice. That memory is real money.
- **The right comparison for a serving team is: total params drive your GPU count and cost floor; active params drive your per-token compute and your batch-1 latency.**

**💰 Math (the anchor number):** DeepSeek-V3 is ~671B total / ~37B active, trained on ~14.8T tokens, at a publicly reported ~2.79M H800-GPU-hours, which the team costed at roughly $5.6M at $2/GPU-hour. Sanity-check with `6 × N_active × D`: `6 × 37e9 × 14.8e12 = 3.29e24` FLOPs. Capacity: `2.79e6 × 3600 × 989e12 = 9.93e24`. Implied MFU ≈ **33%** — which is honest and good for a sparse model, where the all-to-all expert routing communication is a serious throughput tax. **📅 Volatile:** these are reported figures for the main run and exclude prior research compute; verify before citing.

**⚠ Trap:** quoting "$5.6M to train a frontier model" as if it were the total cost of building DeepSeek-V3. That number is the final pretraining run's GPU rental at a notional hourly rate. It excludes the cluster's capital cost, all failed and ablation runs, the data pipeline, and every human. Repeating it uncritically in an interview signals you read a headline rather than the report.

### Loss keeps going down but the benchmark scores didn't move. What is going on?

You have hit **upstream–downstream decoupling**, and understanding it is what separates someone who has actually watched a training run from someone who has read about one.

The mental model: cross-entropy loss is an average over *all* tokens in a held-out corpus, and the overwhelming majority of those tokens are easy — function words, code punctuation, boilerplate. Benchmarks measure a tiny, adversarially-selected tail. A loss improvement of 0.01 nats is a real and reliable improvement in the model's average next-token distribution, and it can be almost entirely composed of getting slightly sharper on tokens no benchmark ever asks about.

Three specific mechanisms produce the decoupling:

1. **Metric discreteness.** Multiple-choice accuracy is a thresholded function of continuous log-probs. The model's margin on a question can improve steadily for a long time while the argmax never flips. Then several flip at once — which is why capability curves look like steps even though loss is smooth. This is the honest deflation of most "emergent ability" claims: **📄 Paper:** Schaeffer et al. (2023), *Are Emergent Abilities of Large Language Models a Mirage?* argued that many apparent emergences are artifacts of discontinuous metrics, and become smooth under continuous ones like Brier score or per-token log-prob.
2. **Domain composition of the loss set.** If your held-out loss is 90% web text and your benchmarks are math and code, loss can improve for reasons irrelevant to the benchmarks. This is fixable and you should fix it: track **per-domain held-out loss**, not one aggregate.
3. **Capability genuinely lagging.** Some capabilities appear only past a compute threshold, and no amount of loss improvement below it produces them.

What I do in practice: maintain a small set of **continuous** proxy metrics that move early — per-domain bits-per-byte, log-prob margin on a benchmark's correct answer versus distractors (not accuracy), and a handful of cloze-style probes. These move smoothly and give you signal thousands of steps before accuracy does.

**⚠ Trap:** comparing loss across runs with different tokenizers. Cross-entropy per *token* is tokenizer-dependent — a tokenizer with a bigger vocabulary produces fewer, harder tokens and a higher per-token loss for an identically-good model. Convert to **bits per byte** (`loss_nats × n_tokens / (ln(2) × n_bytes)`) whenever you compare across tokenizers. Every team makes this mistake exactly once.

**🗣 Say this in the room:** "Loss is an average over mostly-easy tokens; benchmarks are a thresholded read of the hard tail. I track per-domain bits-per-byte and answer-margin rather than accuracy, because those move smoothly and give me signal thousands of steps earlier."

### Where does inference-time compute fit into the scaling picture?

It is a third axis, and its arrival is the most consequential change to the field's planning calculus since Chinchilla. The classical picture had two dials: parameters and training tokens. The new picture has a third: **how many FLOPs you are willing to spend at answer time**, through longer chains of thought, sampling many candidates and selecting, or a search procedure over reasoning steps.

The mental model that makes it feel inevitable: training compute buys you a better *prior* over answers; inference compute buys you better *search* within that prior. These substitute for each other over some range. A smaller model that thinks for 30 seconds can match a larger model that answers immediately, on tasks where verification is easier than generation — which is exactly math, code, and anything with a checkable output.

The mechanism behind the trade is a coverage argument. If a model has per-sample success probability `p` on a task, then with `k` independent samples the probability at least one is correct is `1 − (1−p)^k`. At `p = 0.2`, `k = 1` gives 20% and `k = 16` gives `1 − 0.8^16 = 1 − 0.028 = 97%`. That is pass@16, and it is an *upper bound* on what you can achieve — you only realize it if you have a selector (a verifier, a reward model, majority vote) good enough to pick the right one. **The quality of your verifier is the entire ballgame**, which is why verifier engineering became a real job.

**💰 Math:** the cost side is unforgiving. Take a task where a 70B model answers in 200 output tokens and succeeds 60% of the time, versus an 8B model that emits 4,000 reasoning tokens and succeeds 60%. Per-token decode cost scales with active params, so 70B is ~8.75× the per-token cost of 8B, but the 8B emits 20× the tokens: `20 / 8.75 = 2.3×` — the "cheaper" small model is actually 2.3× *more* expensive per solved task, and it is also 20× slower in wall-clock because decode is sequential. Inference-time compute is not free scaling; it is a trade whose sign depends on the token ratio, and you must compute it rather than assume it.

**⚠ Trap:** treating "test-time compute scaling" as a universal replacement for a better base model. It works where verification is cheap and the base model's sample distribution already contains the right answer somewhere. On tasks where the model is confidently and consistently wrong — factual recall it never learned, a domain it has never seen — sampling 64 times gives you 64 confidently wrong answers and a 64× bill. Search cannot find what the prior does not contain.

**🗣 Say this in the room:** "It's a third scaling axis that substitutes for parameters, but only on verifiable tasks. The bound is pass@k, the realized value is capped by your verifier's quality, and the economics only work when the extra token count is smaller than the parameter-count saving — which you have to actually compute."
