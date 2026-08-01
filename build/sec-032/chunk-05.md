### How often should I checkpoint a 1,024-GPU run? Derive it.

Checkpointing is a pure optimization problem with a closed-form answer, and the reason to derive it rather than pick "every hour" is that the optimum moves by an order of magnitude between 8 GPUs and 16,000 GPUs — so intuition trained at small scale is actively wrong at large scale.

**Step 1 — job MTBF.** Failures are roughly independent per GPU, so the job's mean time between failures is the per-GPU MTBF divided by the GPU count. You can calibrate the per-GPU number from published data: Meta's Llama 3 405B run used 16,384 H100s and reported 419 unexpected interruptions over a 54-day snapshot. 54 days = 1,296 hours, so job MTBF = 1,296/419 = **3.1 hours**, implying a per-GPU MTBF of 3.1 × 16,384 ≈ **50,600 hours ≈ 5.8 years**. That is a good number to carry.

So on 1,024 GPUs: MTBF ≈ 50,600 / 1,024 = **49.4 hours**, roughly two days.

**Step 2 — the Young/Daly optimum.** If a checkpoint blocks for `δ` and failures arrive at mean interval `M`, expected wasted time per interval is `δ` (the write) plus `τ/2` (average work lost since the last checkpoint, occurring once per `M`). Minimizing `δ/τ + τ/(2M)` gives:

```
τ_opt = sqrt(2 · δ · M)
```

**Step 3 — plug in numbers.** A 70B model's full training state is bf16 params (140 GB) + fp32 master (280 GB) + Adam `m` and `v` (560 GB) ≈ **1 TB**. Say a *synchronous* checkpoint blocks the training loop for **60 s** (gather, serialize, write, fsync to a parallel filesystem).

```
M = 49.4 h = 177,840 s
τ = sqrt(2 × 60 × 177,840) = sqrt(21,340,800) = 4,620 s = 77 minutes
overhead = 60/4620 + 4620/(2 × 177,840) = 1.30% + 1.30% = 2.6%
```

Note the symmetry: at the optimum, write cost and lost-work cost are exactly equal. That is a useful sanity check on any answer.

**Step 4 — now switch to asynchronous checkpointing**, where the blocking portion is only the HBM→pinned-host copy, say **3 s**, with the write happening on a background thread.

```
τ = sqrt(2 × 3 × 177,840) = sqrt(1,067,040) = 1,033 s = 17 minutes
overhead = 3/1033 + 1033/355,680 = 0.29% + 0.29% = 0.58%
```

**4.5× less overhead and 4.5× less work lost per failure**, from one implementation change.

**Step 5 — the punchline at frontier scale.** At 16,384 GPUs, `M` = 3.1 h = 11,160 s. Synchronous: `τ = sqrt(2 × 60 × 11,160)` = 1,157 s = **19 minutes**. Asynchronous: `τ = sqrt(2 × 3 × 11,160)` = 259 s = **4.3 minutes**. You must checkpoint a 1 TB state every four minutes. That is simply not possible synchronously — the write alone would be a third of your time. **Async distributed checkpointing is not a nice-to-have at frontier scale; it is the enabling technology.**

**💰 Math:** on the 1,024-GPU, 13-day, $818k run from earlier, moving from a naive "checkpoint hourly, synchronously" (overhead 60/3600 + 3600/355,680 = 1.67% + 1.01% = 2.7%) to optimal async (0.58%) saves 2.1% ≈ **$17,000** on one run. Not huge. Do the same arithmetic at 16k GPUs on a three-month run and it is the difference between finishing and not.

### How does asynchronous distributed checkpointing actually work, and what can go wrong?

The mental model is a write-ahead buffer: you pay only for the copy out of the resource that is on the critical path (HBM), then release the training loop and let the slow part (serialization and I/O) happen concurrently with the next steps.

The mechanism, as implemented by PyTorch's `torch.distributed.checkpoint.async_save`:

1. **Staging (blocking).** Every rank copies its local shards — parameters, optimizer state, and any extra state you registered — from HBM into **pinned host memory**. This is the only part the training loop waits on. For a 1 TB state across 1,024 ranks, each rank stages ~1 GB at ~20 GB/s over PCIe = ~50 ms, plus overhead; call it a few seconds including the collective coordination.
2. **Release.** The training loop resumes immediately and starts step `N+1`.
3. **Background write.** A thread (or a separate process) serializes the staged buffers and writes them to the parallel filesystem or object store. Each rank writes its own files; there is no gather to rank 0. DCP writes per-shard files plus a `.metadata` file describing global shapes and shard offsets.
4. **Commit.** The checkpoint is only valid once *every* rank's write has completed and the metadata is durable. DCP handles this with a coordinated finish.

The failure modes, in order of how often I have seen them:

**Torn checkpoints.** The job dies while the background write is in flight. Half the ranks' files exist, half do not, and the metadata may or may not be there. If your resume logic globs the checkpoint directory and picks the newest, it picks a corrupt one. **The fix is the same one you already use for any distributed write: write to a temporary path and atomically publish a pointer only on full success.** Keep a `latest` file that is updated last, and validate on load that every expected shard exists before you start reading. I treat this as non-negotiable in review.

**Mutation during staging.** If your staging is not a true copy — if it hands the writer a reference to a live tensor — the optimizer's next step mutates the buffer mid-write and you get a checkpoint that is a mixture of step `N` and `N+1`. Silently. The model will load and be subtly wrong. This is why staging into pinned host memory is a *copy*, not a view.

**Overlap collapse.** If the background write takes longer than your checkpoint interval, the next checkpoint's staging blocks on the previous write, and you are back to synchronous behaviour with extra complexity. Monitor write duration as a first-class metric and alert when `write_duration > 0.5 × checkpoint_interval`.

**Host memory pressure.** Staging 1 TB across 128 nodes is 8 GB of pinned host memory per node, on top of the dataloader's buffers. Pinned memory is not swappable. I have watched a job OOM the *host* — which looks like a mysterious node failure — because someone doubled the checkpoint frequency.

**⚠ Trap:** the checkpoint is not just the model and optimizer. It must also contain the **LR scheduler state, the global step, the dataloader position, and the RNG states** for CPU, CUDA and every worker process. A checkpoint missing any of these loads without error and gives you a run that is subtly not the run you were doing. The test that catches all of them is in the next answer but one.

### I need to restart on 960 GPUs instead of 1,024. What has to be true, and what silently breaks?

Three things must be true mechanically, and one thing breaks statistically that nobody thinks about.

**Mechanically:**

1. **The checkpoint must be resharding-capable.** Sharded format with global metadata — `SHARDED_STATE_DICT` via DCP — so a loader at any world size can compute which byte ranges it needs. `LOCAL_STATE_DICT` cannot do this. Optimizer state must reshard too, which is why DCP's `get_state_dict`/`set_state_dict` pair exists: it flattens optimizer state into parameter-keyed, shardable tensors rather than the positional integer-keyed structure `optimizer.state_dict()` gives you.
2. **The mesh must still factor.** If your layout was TP=8 × PP=16 × DP=8, then 960 GPUs must factor compatibly: 960/(8×16) = 7.5. It does not. You would have to drop to 896 (DP=7) or restructure. This is why real elastic restarts happen at node granularity and in multiples of the TP×PP product.
3. **The launcher must support membership change.** `torchrun --nnodes=MIN:MAX --max-restarts=N` with the c10d rendezvous backend restarts *all* workers on a membership change, so your `main()` must be re-entrant and must restore from checkpoint on every start, not just when a `--resume` flag is passed.

**What silently breaks: the global batch size, and therefore the learning-rate schedule.**

`global_batch = dp_degree × micro_batch × grad_accum`. Drop from DP=8 to DP=7 with everything else fixed and your global batch drops by 12.5%. Your LR schedule was tuned for the original batch. Now every step is noisier, and if you are anywhere near the edge of stability you get a loss spike two hours later that looks like a data problem.

**The rule I enforce: hold the global batch constant across restarts by adjusting `grad_accum`, never by letting DP degree change it.** If DP goes 8 → 7, `grad_accum` goes 8 → 9.14 — which is not an integer, so you accept a small change and *log it*, or you pad with a hot spare. A run whose global batch changed mid-flight and was not recorded is a run whose results are not reproducible.

**The pragmatic senior answer.** True elasticity is rarely used in frontier pretraining precisely because of this coupling. What actually happens is **hot spares**: you allocate 1,024 GPUs of work plus 32–64 idle spare nodes, and when a node fails the scheduler swaps in a spare so the world size never changes. Restart cost becomes reload-from-checkpoint plus re-init, with zero schedule perturbation. Elastic shrinking is for jobs where throughput matters more than exact reproducibility — data preprocessing, embedding generation, evaluation sweeps — not for the pretraining run itself.

**🗣 Say this in the room:** "Mechanically it needs a DCP sharded checkpoint and a mesh that still factors. Statistically the thing that bites you is that DP degree sets global batch, which invalidates your LR schedule — so I hold global batch constant with gradient accumulation, or better, I run hot spares so the world size never changes and elasticity never has to be exercised."

### What does it take to make a training run bit-reproducible across a restart? Give me the test.

The goal is not bit-exactness in general — collectives are not associative in floating point and NCCL's reduction order can vary — but **restart-invariance**: a run that crashes at step `N` and resumes must produce the same trajectory as a run that never crashed. That is achievable, it is testable in five minutes, and almost no hand-rolled training loop has it.

**What must be in the checkpoint beyond weights and optimizer:**

- `global_step` and `epoch`.
- **LR scheduler state.** Restarting a cosine schedule at step 0 after 40k steps is a spectacular and common bug — the LR jumps back to peak and the model destabilizes over ~100 steps.
- **Dataloader position.** Not just "which epoch" — which sample index, and for a packed-sequence dataset, the packing buffer state.
- **RNG state** for `torch` (CPU), `torch.cuda` (per device), `numpy`, and Python's `random`, on every rank — plus each DataLoader worker's seed derivation.
- Gradient-scaler state if you are on fp16.
- Any EMA / model-averaging buffers.

**The design that makes this easy: make the dataloader a pure function of the step.** Rather than pickling iterator state, define

```python
def sample_ids(global_step, rank, micro_idx, base_seed, n_docs):
    # a seeded, index-based permutation — no mutable iterator state to save
    g = torch.Generator().manual_seed(hash((base_seed, global_step, rank, micro_idx)) & 0x7fffffff)
    return torch.randint(0, n_docs, (micro_batch,), generator=g)
```

Real implementations use a seeded permutation over shard/document indices rather than sampling with replacement, but the property is what matters: **resume state is one integer.** Everything else is derived. I strongly prefer this over `torch.utils.data` state-dict machinery, because it survives changes to worker count, prefetch depth, and world size.

**The RNG traps:**

- `DataLoader` workers each get their own RNG. Without a `worker_init_fn` that seeds from `(base_seed, epoch, rank, worker_id)`, either all workers produce identical augmentation or the seeding is nondeterministic across restarts.
- Dropout must differ per rank (or you have correlated noise across the data-parallel group and effectively less regularization) but must be *deterministic* given the seed. Derive it as `base_seed + rank`, and save it.
- Activation checkpointing re-runs forward, so it must restore the RNG state around the recomputed region — `preserve_rng_state=True`, the default. Turning it off for speed silently corrupts gradients wherever dropout is present.
- Prefetching means the dataloader has already consumed `num_workers × prefetch_factor` batches beyond what the training loop used. On resume you either account for that depth or accept a small duplicate/skip. Log which you chose.

**🏋 Drill — the restart-invariance test, and I require it in CI:** run 20 steps and record the loss at each. Then run 10 steps, checkpoint, kill the process, resume, and run 10 more. The loss at steps 11–20 of the second run must match the first run to within reduction noise (~1e-5 relative). **Pass criterion: you catch a deliberately-broken scheduler restore, a missing dataloader position, and a missing RNG state, each in a separate run of the test.** Ten minutes to write; it will save you a week.

### How do you detect a straggler in a 1,024-GPU job? What is the signature?

The signature is counterintuitive and knowing it is most of the answer: **in a synchronous training job, the slow rank is the one that is NOT waiting.** Everyone else piles up inside the collective. So if you look at "time spent in NCCL" per rank, 1,023 ranks show a large number and one shows nearly zero. That one is your straggler. Teams that only look at aggregate step time see "the job got 8% slower" and have no idea where.

**The instrumentation.** You need two per-rank time series emitted every step:

- `step_wall_time` per rank.
- `collective_wait_time` per rank — measurable with CUDA events around the collective, or from a profiler, or by putting an explicit `dist.barrier()` with timing immediately *before* the collective (which converts wait-in-collective into wait-at-barrier and is easier to attribute).

Then the alert is one line: `max(step_time) / median(step_time) > 1.05`, plus "which rank has minimum collective wait."

**The causes, roughly in frequency order:**

1. **Thermal or power throttling.** One GPU in a hot part of the rack downclocks. Check `nvidia-smi -q -d PERFORMANCE` for clock-throttle reasons (`SW_Power_Cap`, `HW_Thermal_Slowdown`, `HW_Slowdown`), and watch SM clock as a DCGM metric. A GPU running at 1,200 MHz instead of 1,755 MHz is 32% slow and reports no error whatsoever.
2. **Dataloader stall on one rank.** Its shard of the dataset lives on a slow or contended storage path, or one rank's local NVMe cache missed. Instrument dataloader wait time separately from compute; this is often the whole answer and it is fixable without touching hardware.
3. **ECC error correction / row remapping.** A GPU doing heavy single-bit-error correction runs measurably slower. Watch `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` and retired-page counts.
4. **A bad NIC or a degraded link.** One rail negotiating at a lower rate, or high retransmit counters, makes every collective involving that rank slow — which looks like a straggler even though the GPU is fine. Check the NIC's port rate and error counters.
5. **Host-side noisy neighbour.** Another process, a monitoring agent, or a runaway logging thread eating CPU on one node and starving the launcher.
6. **Uneven work.** Under pipeline parallelism, stage imbalance is a *structural* straggler; under MoE, expert load imbalance is a *per-step, moving* straggler. Neither is a hardware fault and neither is fixed by draining a node — this distinction matters, because the reflex is to blame the machine.

**The proactive control.** Run a small in-band self-test — a fixed 4096³ GEMM plus a small all-reduce — on every rank at startup and every few thousand steps, and compare achieved TFLOPS across ranks. A rank more than 5% below the median gets flagged before it costs you a day. This is cheap (a few hundred milliseconds) and it converts a mysterious slow run into a named bad node.

**💰 Math:** on 1,024 H100s at $2.50/GPU-hr = $2,560/hr, a single straggler running 20% slow costs 20% of the *entire cluster* = $512/hour = **$12,288/day**. One bad GPU. This is the argument for building straggler detection on day one rather than day sixty.

### Your loss is fine on 1,023 ranks and occasionally produces garbage on one. Walk me through diagnosing silent data corruption.

Silent data corruption is a GPU that computes the wrong answer without raising an ECC error, a Xid, or any other signal. It is real, it is documented at fleet scale by multiple large operators, and it is the failure mode that most training stacks are completely blind to — because every layer of the system assumes arithmetic is correct.

**The symptoms that should make you suspect SDC rather than a bug:**

- A loss spike or NaN whose *first appearance* is always on the same rank, across multiple occurrences.
- A per-rank gradient norm that is an outlier by orders of magnitude, on one rank, intermittently.
- A run that diverges and then trains perfectly when relaunched on a different node allocation with the same seed and same data.
- Eval quality that degrades over a window and then stabilizes after an unrelated restart.

**The detection stack you need in place beforehand,** because SDC is essentially undiagnosable after the fact without it:

1. **Per-rank loss and per-rank gradient norm, logged every step.** This is the single highest-value diagnostic in the whole list and it costs one extra scalar per rank per step. Without it you only have the reduced value and the bad rank is invisible.
2. **A deterministic self-test.** At startup and periodically, every GPU computes a fixed matmul on a fixed input and compares a checksum against a reference. This catches gross SDC in seconds. It does not catch rare, input-dependent SDC.
3. **Redundant recomputation on a sample.** Every `k` steps, recompute one microbatch's forward on a second rank and compare outputs to a tolerance. Expensive; used at frontier scale, overkill below it.
4. **Non-finite detection with rank attribution.** When a NaN is detected, all-gather a per-rank boolean and log *which* ranks saw it. A NaN that originates on one rank and propagates through the all-reduce is a very different problem from a NaN that appears everywhere simultaneously (which is a data or LR problem).

**The response procedure:**

1. Halt. Do not "restart and see."
2. Identify the earliest step at which the suspect rank's metrics deviated. This is your contamination boundary.
3. Roll back to the last checkpoint **before** that boundary, not the most recent one. This is why per-rank metric history retained across checkpoints matters — otherwise you cannot locate the boundary and you have to discard back to a known-good eval.
4. Drain the suspect host, remove it from the pool, and run vendor diagnostics (DCGM diagnostic level 3 or 4, which runs targeted stress and memory tests). Field-replace or RMA.
5. Resume on a spare.

**⚠ Trap:** the instinct is to restart from the *latest* checkpoint. If the corruption has been happening for 5,000 steps, the latest checkpoint contains corrupted weights and you have baked the damage in permanently. The whole reason to keep more than one checkpoint, and to keep per-rank metrics, is to be able to answer "since when."

**🗣 Say this in the room:** "SDC is invisible unless you log per-rank loss and gradient norm — the reduced values hide it completely. My first question is whether the anomaly always originates on the same rank; if it does, I roll back to a checkpoint before the first deviation, not the newest one, and I drain the host for DCGM diagnostics."

### Design the NaN watchdog and the loss-spike protocol. What is "rewind and skip"?

Loss spikes are a normal, expected phenomenon in large-model pretraining, not a sign that something is broken — and the difference between a team that loses a week to one and a team that loses ten minutes is entirely whether the protocol was written before the spike happened.

**The watchdog, which runs every step:**

```python
loss_ok = torch.isfinite(loss)
gnorm = clip_grad_norm_global(model, max_norm)          # the correctly-reduced version
gnorm_ok = torch.isfinite(gnorm)
ok = torch.tensor([loss_ok and gnorm_ok], device="cuda")
dist.all_reduce(ok, op=dist.ReduceOp.MIN)                # any rank bad -> all ranks skip
if ok.item():
    optimizer.step()
else:
    skipped += 1
    log_bad_batch(global_step, batch_ids)                # you must record WHICH data
optimizer.zero_grad(set_to_none=True)
```

Three things to notice. First, the check must be **collective** — if one rank sees a NaN and steps anyway while others skip, the replicas diverge permanently and every subsequent all-reduce averages two different models. Second, you must **skip, not clip**: clipping a NaN gives you a NaN, since `NaN / anything` is `NaN`. Third, you must **log the batch identifiers**, because that is the only way to implement the skip half of rewind-and-skip.

**The spike detector,** which is separate from the NaN detector because a spike is finite:

```
if loss > running_mean + 5·running_std   or   gnorm > 10 × running_median(gnorm):
    -> classify as a spike
```

Track running statistics over the last few hundred steps. Grad-norm spikes usually *precede* loss spikes by a step or two, which makes grad norm the better early-warning signal.

**Rewind and skip, the protocol.** A single skipped step is fine — do nothing. If spikes cluster (say, three within 200 steps), the model state and the incoming data are in a bad interaction and continuing will diverge. Then:

1. Stop. Roll back to a checkpoint from *before* the cluster began — typically a few hundred steps back.
2. Skip a window of data: advance the dataloader past the batches associated with the spikes, typically a few hundred batches.
3. Resume.

The empirical finding reported in Google's PaLM work is the key insight: after restarting from an earlier checkpoint and skipping a few hundred batches, the spike **did not reproduce**, and re-running the *same* data from a different checkpoint also did not reproduce it. So a spike is not "bad data" in isolation — it is an interaction between a specific model state and a specific batch. That is why the protocol is rewind *and* skip rather than just "remove the bad documents."

**Preventive measures worth having, in rough order of value:** gradient clipping (global, correctly reduced); a long enough LR warmup; z-loss on the output logits to keep the softmax normalizer from drifting; careful embedding/output-layer initialization; and bf16 rather than fp16 so range is not the trigger.

**⚠ Trap:** an "automatic rewind" that runs unattended without recording what it did produces a training run whose data ordering nobody can reconstruct. Every skip and every rewind must be written to a durable, queryable log with the step range and batch ids. Otherwise your run is not reproducible and you cannot answer "did we train on this document" — which for some customers is a legal question, not an engineering one.

### Define goodput. Why is it the metric you would put on the team's dashboard instead of MFU?

Because MFU measures how well you use the GPU **while you are training**, and it tells you nothing about the fraction of your $818,000 that produced model progress you actually kept. Goodput is the composite metric, and it is the one that maps to money.

```
goodput = (model FLOPs in steps that survived into a usable checkpoint)
          / (allocated GPU-seconds × peak FLOPS)
        = MFU × ETTR
```

where ETTR — effective training time ratio — is the fraction of allocated time spent on steps that were not later lost. Decompose the losses:

1. **Checkpoint blocking** — `δ/τ`, ~0.3–1.7% depending on sync vs async.
2. **Work lost to failures** — `τ/(2M)`, equal to the above at the optimum.
3. **Restart overhead** — failure detection + rescheduling + checkpoint load + NCCL init + `torch.compile` / autotune warmup. This is the one people forget and it is usually the largest term. Loading 1 TB, re-initializing 1,024 NCCL communicators, and re-warming compiled kernels can take **10–20 minutes**.
4. **Stragglers** — the max/median gap, applied to the whole cluster.
5. **Discarded runs** — the honest one nobody puts on a dashboard: days of training thrown away because of a bug found later. This is often the biggest term of all over a project's lifetime, and it is the argument for spending week one on eval and instrumentation.

**Worked example, 1,024 GPUs, 13-day run, MTBF 49.4 h → 6.3 failures.**

*Bad configuration:* failure detected only when the NCCL watchdog fires (default timeout is on the order of tens of minutes — verify your `TORCH_NCCL_*` settings, because this is the single most impactful default in the stack), checkpoint interval 4 h, reload 15 min.

Per failure: 30 min detection + 120 min average lost work + 15 min reload = **165 min**. × 6.3 = 17.3 hours out of 312 = **5.5% lost**. Plus checkpoint write overhead, plus stragglers.

*Good configuration:* a 60-second application heartbeat that kills the job on a missed beat, async checkpointing every 17 min, hot spare so no rescheduling wait, reload 8 min.

Per failure: 1 min detection + 8.5 min average lost work + 8 min reload = **17.5 min**. × 6.3 = 1.8 hours out of 312 = **0.6% lost**.

**💰 Math:** the gap is 4.9% of $818,600 = **$40,100** on one 13-day run, and 4.9% of 13 days = **15 hours** of wall clock. Scale to a 90-day frontier run on 16k GPUs and the same engineering is worth millions. The three highest-leverage items, in order: **a short heartbeat instead of the NCCL default timeout; async checkpointing; hot spares.**

**🗣 Say this in the room:** "MFU is how fast you go while you are going. Goodput is MFU times the fraction of allocated time that produced checkpoints you kept. I put goodput on the dashboard because the biggest wins are almost never in the kernel — they are in cutting failure detection from thirty minutes to one, and in making checkpointing cheap enough to do every fifteen minutes."

### The run died at 60%. What do you do in the first 30 minutes?

The answer that gets you hired is a **procedure with an explicit default action**, not a list of possible causes. The single most important property of the procedure is that it biases toward restarting quickly on known-good hardware, because 90% of interruptions at scale are single-node hardware faults and time spent investigating them is time the other 1,023 GPUs are idle at $2,560/hour.

**Minutes 0–2: is it dead or hung, and is anything still burning money?**

Check the scheduler state. A hung job is worse than a dead one — it holds the allocation and produces nothing. If ranks are alive but no step has completed in five minutes, treat it as dead and kill it. Confirm no orphaned processes are holding GPUs.

**Minutes 2–6: get the *first* failure, not the loudest one.**

This is where people lose an hour. Rank 0's log usually shows a NCCL timeout, which is a *symptom* — rank 0 was waiting for someone who died. Sort all ranks' logs by timestamp and read the earliest anomalous entry. NCCL timeout messages name the ranks that failed to arrive; those are your suspects. Pull `dmesg` and Xid events from those nodes. This requires that you already ship per-rank structured logs to a central store with synchronized clocks — if you do not have that on day one, this step is impossible and you will restart blind.

**Minutes 6–10: classify.**

- **Hardware fault** — Xid events (double-bit ECC, GPU fallen off the bus, NVLink errors), node unresponsive, NIC down. ~60–75% of unexpected interruptions at scale. → Drain the node, restart on a spare. Do not investigate now.
- **NCCL hang with no dead rank** — a network or fabric issue, a deadlock from divergent collective ordering, or a rank stuck in a slow I/O path. → Capture stack traces from every rank (`py-spy dump` or a registered `SIGQUIT` handler) *before* killing, because this evidence is unrecoverable afterward.
- **OOM** — read which rank, and at which step. If it is step-dependent, you hit a long-sequence batch; if it is rank-0-only under PP, it is the `p`-microbatch activation pileup.
- **NaN / divergence** — this is the one case where you must NOT immediately restart from the latest checkpoint, because it may be contaminated. Go to the spike protocol.
- **Infrastructure** — storage unavailable, scheduler preemption, image pull failure, expired credentials. Common and boring.

**Minutes 10–20: execute the default action.**

For the hardware case, which is most cases: blacklist the node, restart from the last verified checkpoint on a hot spare, unchanged config. **Change nothing else.** A restart that also changes a hyperparameter is unattributable — if it works you have learned nothing, and if it fails you have two variables.

**Minutes 20–30: verify the restart is actually healthy.**

Three checks, and they matter as much as the restart:

1. Loss at step `N+1` matches the pre-crash trajectory to within noise. If it jumped to ~`ln(vocab_size)` you loaded the wrong thing.
2. Grad norm and LR match the pre-crash values — catches scheduler and optimizer-state restore bugs.
3. Step time / MFU is back to baseline. If it is 10% worse, you restarted onto a bad spare and you are about to spend a day confused.

Then, and only then, file the drained node for diagnostics and write the incident up.

**⚠ Trap:** the reflex to "just restart and see if it happens again." At 1,024 GPUs, a restart costs ~15 minutes of the whole cluster = $640. Two blind restarts cost more than the ten minutes of log reading that would have identified the node. The other reflex to resist is restarting from the newest checkpoint when the failure was numerical — verify the checkpoint's eval loss before trusting it.

**🗣 Say this in the room:** "First I decide dead-or-hung, because a hung job burns the whole allocation. Then I find the *earliest* failure across all ranks, not rank zero's timeout, and classify: hardware, hang, OOM, numerical, or infra. For hardware — which is most of them — the default action is drain the node and restart on a spare with an unchanged config, and then I verify loss, grad norm and step time match pre-crash before I go investigate. The only case where I do not restart immediately is a numerical failure, because the latest checkpoint may be contaminated."

### 🏋 Drill: implement ZeRO stage 1 from scratch. Forty minutes, no reference.

Pass criterion: a working optimizer that shards Adam state across ranks, produces the same updates as single-GPU AdamW to within floating-point noise, and that you can explain the communication volume of. This is a real whiteboard-and-laptop exercise at infra-flavoured loops, and the point is not the code — it is that writing it forces you to internalize where every collective goes.

```python
import torch, torch.distributed as dist

class ZeRO1AdamW:
    """Optimizer-state-sharded AdamW. Each rank owns a contiguous slice of the
    flattened parameter space and holds fp32 master weights + moments only for it."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, wd=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.lr, (self.b1, self.b2), self.eps, self.wd, self.t = lr, betas, eps, wd, 0

        self.numel = sum(p.numel() for p in self.params)
        self.chunk = (self.numel + self.world - 1) // self.world      # padded shard size
        self.lo = min(self.rank * self.chunk, self.numel)
        self.hi = min(self.lo + self.chunk, self.numel)

        flat = self._flat([p.data for p in self.params])
        self.master = flat[self.lo:self.hi].float().clone()           # 4 B/param / world
        self.m = torch.zeros_like(self.master)                        # 4 B/param / world
        self.v = torch.zeros_like(self.master)                        # 4 B/param / world

    def _flat(self, ts):
        return torch.cat([t.reshape(-1) for t in ts])

    @torch.no_grad()
    def step(self):
        self.t += 1
        g = self._flat([p.grad for p in self.params]).float()
        dist.all_reduce(g, op=dist.ReduceOp.AVG)        # 2Psi; reduce_scatter would be Psi
        g = g[self.lo:self.hi]

        self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        self.v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        self.master.mul_(1 - self.lr * self.wd)                        # decoupled weight decay
        self.master.addcdiv_(mh, vh.sqrt().add_(self.eps), value=-self.lr)

        shard = torch.zeros(self.chunk, dtype=self.params[0].dtype, device=self.master.device)
        shard[: self.hi - self.lo] = self.master.to(shard.dtype)
        out = [torch.empty_like(shard) for _ in range(self.world)]
        dist.all_gather(out, shard)                                    # Psi
        new = torch.cat(out)[: self.numel]
        off = 0
        for p in self.params:
            p.data.copy_(new[off: off + p.numel()].view_as(p)); off += p.numel()
        for p in self.params:
            p.grad = None
```

**What the drill is testing, and what to say about your own code:**

- The `all_reduce` above is deliberately the naive version. A real ZeRO-1 uses `reduce_scatter` — each rank only needs its own slice of the reduced gradient — which moves Ψ instead of 2Ψ. Combined with the Ψ all-gather that gives 2Ψ total, exactly DDP's volume. **Being able to say "my version costs 3Ψ and the correct one costs 2Ψ, here is the one-line change" is worth more than getting it right the first time.**
- Padding the last shard matters: `numel` rarely divides by `world`, and `all_gather` requires equal-sized tensors. Forgetting this is the most common bug in this exercise.
- Memory saved: optimizer state goes from 12 bytes/param to 12/world. On 8 ranks with a 7B model, 84 GB → 10.5 GB per rank.
- Extending to ZeRO-2 means reduce-scattering gradients *during* backward via hooks and freeing the non-owned portions, rather than materializing the full flat gradient. Extending to ZeRO-3 means the same trick applied to parameters, with all-gathers in forward and backward — which is FSDP, and at that point you should use FSDP.

**⚠ Trap in the exercise:** flattening all parameters into one buffer each step is `O(model size)` of extra allocation and copy per step — fine for a teaching implementation, unacceptable in production, which is why FSDP1 pre-allocates persistent flat buffers and FSDP2 uses DTensor views. If you write this in an interview, say that out loud before they ask.

### It is day one of a 1,024-GPU run. What instrumentation do you require before the first step?

I would refuse to launch without these, and the argument is purely economic: at $2,560/hour, a metric that saves one day of confusion pays for itself sixty times over, and every item below has saved me at least that.

**Per-step, per-rank, emitted to a central store:**

- `step_time`, plus the derived `max/median` ratio. This is your straggler alarm.
- `loss` and `grad_norm` **per rank**, not just the reduced values. Without these, silent corruption and single-rank NaN are invisible.
- `dataloader_wait_time` separated from compute time. Storage stalls masquerade as GPU problems constantly.
- `collective_time` per rank.
- Non-finite counter and skipped-step counter.
- `tokens_consumed` and, for MoE, the **token drop rate** and per-expert load histogram.

**Derived and displayed prominently:**

- **MFU**, computed live from `6N × tokens/s / (GPUs × peak)`, with the formula written on the dashboard so nobody argues about which variant it is.
- **Goodput / ETTR**, cumulative since job start.
- Wall-clock and dollars burned versus the plan.

**Hardware telemetry, via DCGM:**

- SM clock and **clock-throttle reasons** — thermal and power throttling is the most common invisible slowdown.
- ECC single- and double-bit counters, retired/remapped pages.
- NVLink and PCIe replay/error counters.
- **Xid events** scraped from `dmesg` on every node, forwarded and alerted on.
- Per-NIC throughput and error counters, so a degraded rail is visible as a rail rather than as a mystery.

**Job control plane:**

- A **60-second application heartbeat** with an automatic kill on a missed beat. This one item is typically worth more than all the others combined, because it turns 30 minutes of failure-detection latency into one minute — 4.9% of the run, from the goodput arithmetic.
- Per-rank structured logs with synchronized timestamps in a central store. Non-negotiable; without it, the "first 30 minutes" procedure is impossible.
- Checkpoint index: step, path, write duration, verified-complete flag, and the eval loss at that checkpoint.
- A registered `SIGQUIT`/`faulthandler` dump so a hang produces stack traces from every rank before you kill it.
- A node blacklist that the launcher consults, so a drained bad node cannot be re-scheduled into the next restart.

**Quality, not just throughput:**

- A held-out loss evaluated every `k` steps on a *fixed* small set, plotted against a reference run. Training loss alone will not tell you that your data pipeline started emitting duplicates.
- The full config, git SHA, and data manifest hash recorded with every checkpoint.

**🗣 Say this in the room:** "Per-rank loss and grad norm, per-rank step time, throttle reasons, Xid scraping, a sixty-second heartbeat, and a checkpoint index with verified-complete flags. If I could keep only two, I would keep the heartbeat — because failure-detection latency is usually the largest goodput loss — and per-rank grad norm, because it is the only thing that makes a single bad GPU visible."

### You are interviewing at an AI product company, not a lab. How much of this do you actually need, and where would you push back on a plan to train?

This is the judgment question, and I would answer it directly rather than diplomatically, because the reflex to fine-tune is a documented rejection trigger at applied-AI loops.

**What you will actually run at Cursor, Perplexity, Notion, Harvey, Glean, Sierra, Ramp:** single-node or small-multi-node jobs — 8 to 64 H100s — doing LoRA or full fine-tuning of a 7B–70B model, or training an embedding/reranker model, or distilling a frontier model into a small fast one for a latency-critical path. That workload needs **FSDP, mixed precision, activation checkpointing, gradient accumulation, correct global gradient clipping, and resumable checkpointing.** It does not need tensor parallelism, pipeline parallelism, context parallelism, or expert parallelism, because the model fits comfortably in one NVLink domain.

So the calibration is: know **FSDP cold** — wrap policy, sharding strategy, mixed-precision policy, the clipping bug, DCP resharding. Know TP/PP/CP/EP well enough to derive the collectives and explain the mesh-ordering rule, because the design round will ask and because it is how you demonstrate you understand the constraint hierarchy. And be honest that you have not run a 16,000-GPU job — claiming otherwise is trivially caught by one follow-up about Xid codes or straggler signatures.

**Where I push back on a plan to train.** The escalation ladder is: better prompting and context construction → retrieval quality → tool and schema design → constrained/structured output → model routing → distillation → LoRA → full fine-tune → continued pretraining. Each rung is roughly an order of magnitude more expensive in engineering time and roughly an order of magnitude slower to iterate. I would push back on any proposal that skips rungs, and the specific preconditions I want to see before approving a fine-tune are:

1. **A measured eval gap on an in-domain benchmark you built**, not a vibes assessment and not a public leaderboard.
2. **Evidence that prompting and retrieval cannot close it** — meaning someone actually tried, with numbers.
3. **At least ~10k high-quality domain examples**, ideally far more, and a plan for how you get more as the product changes.
4. **A reason the frontier API cannot serve it** — latency, cost at your volume, data residency, or a genuinely off-distribution domain.
5. **A retraining and eval plan**, because a fine-tuned model is a permanent maintenance obligation that must be re-run every time the base model or the product changes.

**💰 Math that makes the argument concrete.** A LoRA fine-tune of a 70B on 8×H100 for 12 hours = 96 GPU-hours × $2.50 = **$240** in compute — but 3–6 engineer-weeks in data curation, eval construction, serving integration and ongoing maintenance, which at a loaded senior rate is **$30k–60k**. Full pretraining of a 70B is ~$750k in compute alone. Meanwhile the same eval gap can often be closed by a better retrieval strategy in two engineer-days. **📅 Volatile** on the GPU rate. The cost of training is almost never the GPUs; it is the maintenance obligation you just signed.

**🗣 Say this in the room:** "For an applied role I would expect to own single-node and small-multi-node FSDP fine-tuning, and I know that layer in detail. I know the 3D-parallelism material because it is how you reason about the constraint hierarchy, not because I have run a 16k-GPU job. And on any proposal to fine-tune, the first thing I ask for is the in-domain eval and the evidence that prompting and retrieval could not close the gap — because the model is the cheap part and the maintenance obligation is the expensive part."
