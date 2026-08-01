### Walk me through what FSDP actually does during one forward-backward. Where does memory peak?

FSDP is ZeRO-3 implemented as a PyTorch module wrapper, and the mental model is a **just-in-time materialization** pattern you already know from lazy loading: nothing is resident until the moment it is needed, and it is dropped the instant it is not. The unit of laziness is not the model and not the parameter — it is the *wrapped unit*, typically one transformer block. Getting that granularity right is 80% of using FSDP well.

The cycle, for each wrapped unit, in forward:

1. **All-gather** the unit's sharded bf16 parameters from all ranks, producing the full unwrapped weights in a temporary buffer.
2. Run the unit's forward compute.
3. **Free** the unwrapped buffer immediately (under `FULL_SHARD`). Only the `1/N` shard remains resident.

And in backward, per unit, in reverse:

4. **All-gather** the unit's parameters again (they were freed).
5. Compute gradients for the unit.
6. **Reduce-scatter** the gradients so each rank ends up holding the reduced `1/N` slice matching its parameter shard.
7. Free the unwrapped parameters and the full gradients.

So per step you pay all-gather (Ψ) + all-gather (Ψ) + reduce-scatter (Ψ) = 3Ψ, exactly the ZeRO-3 volume.

**Where memory peaks** is the question that separates people who have used FSDP from people who have read about it. It peaks in **backward**, at the moment when you simultaneously hold: (a) the sharded params for the entire model, (b) the fully-gathered params of the current unit, (c) the fully-gathered params of the *next* unit if backward prefetch is on, (d) the full unsharded gradients of the current unit, and (e) all the saved activations from forward that have not yet been consumed. For a 70B model wrapped per-block on 64 GPUs: sharded state 70e9 × 16 / 64 = 17.5 GB, plus two gathered blocks at 70e9/80 × 2 B = 1.75 GB each, plus one block of gradients at 1.75 GB — so ~22.75 GB of model state, and activations get the rest.

**⚠ Trap:** the transient all-gather buffer is sized by your *largest wrapped unit*, not by the average. If your model has one enormous module — a tied 128k-vocabulary embedding at hidden 8192 is 128000 × 8192 × 2 B = **2.1 GB** — and your wrap policy leaves it as its own unit or, worse, lumps it with the root, that one buffer sets your peak. I always print per-unit parameter counts after wrapping and look at the max, not the mean.

**🗣 Say this in the room:** "FSDP all-gathers a unit's weights right before using them and frees them right after — twice per step, since backward needs them again — and reduce-scatters gradients into shards. Peak memory is in backward and equals sharded state plus roughly two gathered units plus one unit of gradients plus live activations. Which means the wrap granularity is the memory knob, not the sharding strategy."

### What did FSDP2 change, and why does DTensor matter?

FSDP1's central data structure is the **FlatParameter**: every parameter inside a wrapped unit is flattened to 1D, concatenated into one giant tensor, and that tensor is chunked into `world_size` contiguous pieces. It is a clever performance decision — one contiguous buffer means one all-gather instead of dozens, and NCCL loves large messages — but it destroys per-parameter identity, and everything painful about FSDP1 descends from that.

The consequences you actually hit:

- **You cannot mix frozen and trainable parameters in one unit.** A FlatParameter has one `requires_grad`. So LoRA-style training, where you freeze the base and train adapters that live inside the same transformer block, forces awkward wrapping or does not work.
- **You cannot mix dtypes in one unit** — relevant the moment you want fp8 weights for the linear layers and bf16 for the norms.
- **Per-parameter optimizer settings are lost.** "No weight decay on biases and layernorms" is standard practice; under FlatParameter those parameters are bytes in the middle of a flat buffer, so param-group construction requires index gymnastics.
- **State dicts require reconstruction.** The checkpoint does not look like the model.

FSDP2 (`torch.distributed.fsdp.fully_shard`) replaces FlatParameter with **DTensor**: each parameter individually becomes a distributed tensor annotated with a `DeviceMesh` and a placement — `Shard(0)` for FSDP, `Replicate()`, or `Partial()`. The parameter keeps its own identity, dtype, `requires_grad`, and shape metadata; the sharding is a property carried alongside it. The performance gap versus FlatParameter is closed by grouping all-gathers per unit under the hood.

What this unlocks is composition. Because a DTensor's placement is per-mesh-dimension, you can build a 2D mesh `("dp", "tp")` and say a parameter is `Shard(0)` on the `dp` axis and `Shard(1)` on the `tp` axis — FSDP and tensor parallelism on the same tensor, expressed declaratively, with the collectives derived rather than hand-written. That is what makes torchtitan's 4D/5D parallelism tractable in a few hundred lines instead of Megatron's tens of thousands.

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
model = parallelize_module(model, mesh["tp"], tp_plan)   # TP first
for block in model.layers:                                # then shard each block on dp
    fully_shard(block, mesh=mesh["dp"])
fully_shard(model, mesh=mesh["dp"])                       # root wrap last
```

**⚠ Trap:** the ordering above is not stylistic. Tensor parallelism must be applied *before* FSDP sharding, and the root module must be wrapped *last*, after its children. Wrap the root first and its parameters get absorbed into the root unit, so the child wraps do nothing useful and you silently get one giant unit — the exact failure in the next question.

**📄 Paper:** Zhao et al. (2023), *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*. It documented the FlatParameter design and the prefetch/rate-limiter machinery; FSDP2's DTensor rewrite is a PyTorch engineering evolution rather than a paper result, so cite the repo and the design docs, not an invented citation.

### Your FSDP job runs but memory barely improved over DDP. What is the first thing you check?

The wrap policy. Nine times out of ten someone wrapped the model as a single unit, which means FSDP all-gathers *all* parameters at the start of forward and holds them until the end — you get exactly DDP's peak memory plus the overhead of having sharded and re-gathered everything. The sharded-state savings still apply to optimizer state, but the transient buffer eats them back.

The mental model: **the wrap boundary is the free boundary.** FSDP can only release parameters at the granularity you told it about. One unit means one gather, held for the whole forward.

The correct policy for a transformer is to wrap per decoder block:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import functools

policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)
model = FSDP(model, auto_wrap_policy=policy, sharding_strategy=ShardingStrategy.FULL_SHARD, ...)
```

Under FSDP2 the equivalent is a loop calling `fully_shard(block)` per block, then `fully_shard(model)` on the root.

Now the *other* direction, which people never consider: **wrapping too finely is also a bug.** If you wrap every `nn.Linear`, you turn one 0.65 GB all-gather per block into seven all-gathers of 30–180 MB each. NCCL's per-collective launch overhead is roughly 5–20 µs and small messages do not saturate NVLink, so you go from being bandwidth-bound to being latency-bound. On a 40-layer model that is 280 collectives per forward instead of 40, and I have seen this cost 25–40% of step time.

The heuristic I use: **wrap so that a unit is 50–500 MB of parameters.** Below 50 MB you are latency-bound; above 500 MB the transient buffer starts dominating peak memory and prefetch gets expensive. For most transformers "one decoder block" lands in that window naturally, which is why it is the default advice.

**🔍 Failure taxonomy — FSDP memory did not improve:**

1. Print `sum(p.numel() for p in module.parameters())` for every FSDP-wrapped instance after construction. If there is one unit containing >50% of the model, your policy did not fire.
2. If the policy fired but units are tiny (<10 MB), you over-wrapped — check for a `size_based_auto_wrap_policy` with too small a `min_num_params`.
3. If units look right, check the sharding strategy — `SHARD_GRAD_OP` keeps parameters unsharded after forward by design, so peak looks like DDP for params.
4. If all three are right, the memory is activations, not model state. Profile with `torch.cuda.memory._record_memory_history()` and look at the allocation timeline, not the totals.

### Explain FSDP's prefetching knobs and the rate limiter. What are you actually trading?

You are trading peak memory against communication/compute overlap, and the knobs let you slide along that curve. The default settings are conservative on memory; a well-tuned large run usually pushes toward more overlap.

**Backward prefetch.** During backward, FSDP must all-gather unit `k`'s parameters before it can compute unit `k`'s gradients. `BackwardPrefetch.BACKWARD_PRE` issues the all-gather for unit `k−1` *before* running unit `k`'s backward compute, so the collective overlaps with real work. The cost is that you now hold two units' worth of gathered parameters instead of one. `BACKWARD_POST` issues it after, using less memory and getting less overlap. `None` disables it and serializes — never do this except to isolate a bug. `BACKWARD_PRE` is the default and is almost always correct; the units are small if you wrapped correctly.

**Forward prefetch.** `forward_prefetch=True` issues unit `k+1`'s all-gather before unit `k`'s forward compute. This only helps when your CPU cannot issue kernels fast enough to keep the all-gather queue full — a real problem with small models, many layers, and Python overhead. It uses the *previous iteration's* execution order as its prediction, so it is unsafe for models with data-dependent control flow (MoE with dropping, early exit). If your forward order changes between iterations, forward prefetch will gather the wrong thing and either hang or waste bandwidth.

**The rate limiter.** `limit_all_gathers=True` is the subtle one. CUDA is asynchronous: the CPU can run far ahead of the GPU, queueing all-gathers for units 5, 6, 7, 8 while the GPU is still on unit 2. Each queued all-gather allocates its destination buffer *at queue time*. Without a limiter, a fast CPU can allocate a dozen unsharded units simultaneously and OOM — on a machine that has plenty of memory in steady state. The rate limiter blocks the CPU until the GPU has consumed prior gathers, capping in-flight allocations.

**⚠ Trap:** the rate limiter's failure signature is *nondeterministic OOM that depends on host CPU speed*. The same config OOMs on a fast head node and runs fine on a slower one, or OOMs only when the dataloader is not competing for CPU. Teams chase this as a "flaky GPU" for days. If your OOM moves around between runs with identical shapes, suspect allocation-time queueing before you suspect hardware.

**💰 Math:** turning `BACKWARD_PRE` off to save memory on a 70B run wrapped per block: you save one gathered unit ≈ 1.75 GB, and you lose overlap on an all-gather of the same 1.75 GB. At 400 GB/s intra-node that gather is 4.4 ms; over 80 layers that is 350 ms per step exposed. On a 3 s step that is **12% throughput**. On 512 H100s at $2.50/GPU-hr = $1,280/hr, 12% is $154/hr = **$110k/month** to save 1.75 GB. Buy the memory elsewhere.

### Gradient clipping under FSDP — show me why the naive call is silently wrong and write the correct one.

This is my favorite FSDP question because the bug produces no error, no NaN, and no crash — it just makes your clipping threshold wrong by a factor of √N and you find out when a loss spike that clipping should have absorbed blows up your run at step 40,000.

Global gradient clipping requires the **global** norm across every parameter on every rank:

```
total_norm = sqrt( Σ_ranks Σ_params ||g||² )
```

Under FSDP, each rank holds only its `1/N` shard of every gradient. So `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` computes `sqrt(Σ_local ||g||²)` — the norm of one shard. If the gradient energy is roughly uniform across the shard dimension, the local norm is `true_norm / √N`. On 64 ranks that is **0.125× the true norm**. Your clip threshold of 1.0 behaves like a threshold of 8.0. Clipping essentially never fires. Worse, if ranks have unequal shard norms, each rank scales its shard by a *different* factor, so the assembled gradient is no longer a scaled version of the true gradient — you have applied a per-shard nonlinearity to the update direction.

The correct computation is three lines of collective logic:

```python
def fsdp_clip_grad_norm_(params, max_norm, group=None):
    # local sum of squares over this rank's shards (grads may be DTensor or plain)
    local_sq = torch.zeros((), device="cuda", dtype=torch.float32)
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
        local_sq += g.detach().float().pow(2).sum()
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=group)   # the missing collective
    total_norm = local_sq.sqrt()
    coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(coef)                                       # same coef on every rank
    return total_norm
```

In practice: **FSDP1 exposes `model.clip_grad_norm_(max_norm)` as a method on the FSDP module and you must call that one**, not the free function — it does exactly the above. Under FSDP2, gradients are DTensors that carry their placement, so `torch.nn.utils.clip_grad_norm_` can dispatch the collective correctly; but this is exactly the kind of thing that changes between PyTorch releases, so verify empirically rather than trusting the docs.

**🏋 Drill:** the verification that catches this in 60 seconds. Run the same tiny model on 1 GPU and on 8 GPUs with identical data ordering and a fixed seed, and print `total_norm` at step 1. If the 8-GPU number is not within floating-point noise of the 1-GPU number, your clipping is broken. Pass criterion: you write this test *before* you write the training loop. I make it a required smoke test in CI for any distributed training repo I own.

**⚠ Trap:** the same √N bug appears in *anything* that reduces over parameters — gradient-norm logging, update-to-weight-ratio diagnostics, per-layer norm dashboards. If your observability computes statistics from local shards, every number on your training dashboard is wrong by an unknown factor and you will make decisions from it.

### Your FSDP checkpoint loads without error but the model is garbage. Diagnose.

Almost always a **state-dict-type mismatch between save and load**, and the reason it does not error is that PyTorch will happily load a tensor of the right shape that contains the wrong rank's data.

FSDP1 offers three state dict types and they are not interchangeable:

- `FULL_STATE_DICT` — all-gathers everything to produce the unsharded, wrap-agnostic checkpoint that looks exactly like the original `nn.Module`. Portable, loadable by `from_pretrained`, resharding-agnostic. Also: it materializes the whole model on rank 0, so a 70B model needs 280 GB of host RAM with `offload_to_cpu=True` and will OOM without it.
- `SHARDED_STATE_DICT` — each rank saves its DTensor/ShardedTensor shard *with metadata describing global shape and offsets*. This is the one that supports resharding: because the metadata is global, a loader at a different world size can compute which bytes it needs.
- `LOCAL_STATE_DICT` — raw local flat-parameter bytes with no global metadata. Fast, small, and a **trap**: it is only loadable by exactly the same world size and exactly the same wrap policy. Change your `auto_wrap_policy` (say, you added a wrap around the embedding), and the flat-parameter boundaries shift, so rank 3's bytes now correspond to different parameters. Shapes still match. No error. Garbage model.

The modern answer is to stop hand-rolling this and use `torch.distributed.checkpoint` (DCP), which saves sharded tensors with global metadata and *resharding is a first-class operation*:

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

msd, osd = get_state_dict(model, optimizer)
dcp.save({"model": msd, "optim": osd}, checkpoint_id=path)     # any world size
# ... later, possibly on a different number of GPUs:
msd, osd = get_state_dict(model, optimizer)                    # gives DCP the target layout
dcp.load({"model": msd, "optim": osd}, checkpoint_id=path)
set_state_dict(model, optimizer, model_state_dict=msd, optim_state_dict=osd)
```

The `get_state_dict` → `load` → `set_state_dict` dance matters: DCP loads *into* the destination layout you hand it, so it needs to see the target sharding before it reads bytes. Calling `dcp.load` with an empty dict gives you the source layout and defeats the whole point.

**🔍 Failure taxonomy — "checkpoint loaded, model is wrong":**

1. **Loss jumps to ~ln(vocab_size) on resume** (e.g. 11.7 for a 128k vocab) → you loaded random or misaligned weights. Check state dict type and wrap policy equality.
2. **Loss resumes correctly but immediately diverges over ~100 steps** → weights loaded, optimizer state did not. Adam with zeroed `m`/`v` takes a huge first step. Verify `osd` is non-empty and that `exp_avg_sq` tensors are not all zeros.
3. **Loss resumes correctly and drifts slowly worse** → the dataloader state was not restored, so you are re-training on data you have already seen, or your LR scheduler restarted at step 0. Print the LR and the global step on resume.
4. **Everything correct on 1 node, wrong on 8** → the checkpoint was saved `LOCAL_STATE_DICT` or the shard metadata is missing.

**🗣 Say this in the room:** "The rule I enforce is: sharded checkpoints for resumption, one full checkpoint at the end for distribution, and never `LOCAL_STATE_DICT`. And the resume test is not 'does it load' — it is 'does the loss at step N+1 after a restart match the loss at step N+1 without a restart, to within reduction noise.' That test catches optimizer state, RNG state, and dataloader state in one shot."

### How do activation checkpointing and FSDP interact? People get the ordering wrong — explain why it matters.

They interact through the backward pass, and the failure is a *double* all-gather that doubles your communication bill without changing a single line of visible behavior.

Recall what each does. Activation checkpointing discards the interior activations of a block during forward and **re-runs the block's forward** during backward to regenerate them. FSDP frees a block's parameters after forward and all-gathers them again during backward. Now compose them: if the checkpointed region and the FSDP unit are the same block, backward does one all-gather, one recompute-forward, one gradient compute, one reduce-scatter. Fine — 3Ψ total, as expected.

But if you apply activation checkpointing at a *finer* granularity than FSDP wrapping — say, checkpointing each attention and each MLP separately while wrapping per block — the recompute of the attention sub-module and the recompute of the MLP sub-module each trigger their own parameter access. Depending on how the wrapping resolved, you can end up with the unit gathered, freed, and gathered again inside the same backward. That is 4Ψ instead of 3Ψ: a **33% increase in communication** for zero benefit.

The rule: **apply activation checkpointing at the same granularity as the FSDP wrap, or coarser.** Both on the transformer block. In PyTorch the composition is:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
)
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
    check_fn=lambda m: isinstance(m, LlamaDecoderLayer),   # same class as the wrap policy
)
# then FSDP-wrap on the same class
```

**⚠ Trap:** `CheckpointImpl.REENTRANT` (the old default of `torch.utils.checkpoint`) is incompatible with a pile of things — it does not support gradients flowing to inputs that do not require grad, it breaks with `torch.compile`, and its interaction with FSDP's hooks is fragile. **Always use `NO_REENTRANT`** (which in current PyTorch is what `use_reentrant=False` selects). This is one of the highest-value single flags in the whole stack and most tutorial code still shows the wrong one.

**⚠ Trap, second one:** activation checkpointing recomputes forward, and if your forward is not deterministic — dropout, any RNG — the recomputed activations differ from the originals, and your gradients are wrong. PyTorch handles this by saving and restoring RNG state around the checkpointed region (`preserve_rng_state=True`, the default). If you disable it for speed, or if you use a custom RNG (a fused dropout kernel with its own counter), you get silently incorrect gradients. Test by running with and without checkpointing at fixed seed and comparing gradients elementwise.

### Walk me through the mixed-precision configuration in FSDP. Which dtype goes where, and why is `reduce_dtype` the interesting one?

FSDP's `MixedPrecision` has three independent dials, and treating them as one "use bf16" switch is where people lose accuracy.

```python
from torch.distributed.fsdp import MixedPrecision
mp = MixedPrecision(
    param_dtype=torch.bfloat16,    # dtype of the all-gathered params used for compute
    reduce_dtype=torch.float32,    # dtype of the gradient reduce-scatter
    buffer_dtype=torch.float32,    # dtype of non-parameter buffers
)
```

**`param_dtype`** is the easy one: bf16, so tensor cores do the work and all-gathers move half the bytes.

**`reduce_dtype`** is the one worth arguing about. The gradient reduce-scatter sums contributions from `N` ranks. In bf16, with 8 mantissa bits, each addition has ~0.4% relative rounding error, and summing 1,024 values in a ring accumulates error that grows roughly with the number of sequential additions. At world size 8 this is invisible. At world size 1,024 it is a real and measurable bias in the update direction, and it shows up as a training run that tracks the fp32 baseline for 10k steps and then slowly diverges.

The cost of `reduce_dtype=torch.float32` is exactly 2× the gradient communication: the reduce-scatter moves 4 bytes/param instead of 2. For a 70B model that is 280 GB instead of 140 GB reduce-scattered per step, i.e. `(N−1)/N × 280 ≈ 280 GB` on the wire per rank instead of 140. Whether you can afford it is the same overlap calculation as always.

My decision rule: **bf16 reduce below world size ~128, fp32 reduce above it**, and always fp32 reduce if you have ever seen an unexplained slow divergence. Some frameworks offer a middle path — reduce in bf16 but keep an fp32 accumulator per rank (gradient accumulation in fp32 across microbatches) — which recovers most of the stability for none of the bandwidth.

**`buffer_dtype`** is the sleeper. Buffers are the non-parameter tensors registered on modules: **RoPE inverse frequencies**, causal masks, running statistics. Casting RoPE's `inv_freq` to bf16 is genuinely destructive. The rotation angle at position `p` is `p × inv_freq`, and at `p = 100,000` with bf16's 0.4% relative error, the angle error is 400 radians — the position encoding is noise. Long-context models trained with `buffer_dtype=bfloat16` degrade specifically at long positions, which is exactly the regime you built them for and exactly the regime your short-context eval does not test.

**🗣 Say this in the room:** "Parameters in bf16, gradients reduced in fp32 once world size passes about 128, and buffers in fp32 always — because RoPE frequencies in bf16 destroy long-position accuracy and you will only notice it on the eval you did not run."

### How do you get a bit-comparable baseline between a single-GPU run and an FSDP run? Interviewer's version: "the FSDP loss doesn't match single-GPU and I don't know why."

You do not get bit-exact — collective reductions are not associative in floating point and NCCL's reduction order depends on world size and algorithm — but you should get *statistically indistinguishable*, and you can get bit-exact at step 0. The debugging procedure is a binary search over the sources of divergence, and it is worth having memorized because it is a live-debug interview prompt.

**Step 0 — establish the invariant that must hold exactly.** With the same seed, same data, same global batch, the *loss at step 0* on 1 GPU and on N GPUs must match to ~1e-5 relative. Step 0 involves no optimizer update and no gradient reduction affecting the forward. If step 0 already differs, the problem is data or model init, not distribution.

**The ordered checklist:**

1. **Global batch size mismatch.** The most common cause by an enormous margin. `per_device_batch × world_size × grad_accum` must be equal in both runs. If the single-GPU run used batch 32 and the 8-GPU run uses per-device 32, you are training at 8× the batch — different loss curve, not a bug.
2. **Data ordering.** `DistributedSampler` requires `sampler.set_epoch(epoch)` every epoch or every rank replays the same permutation forever. Also: the single-GPU baseline must consume the same *sequence* of examples, which means the comparison run should use the same sampler with `world_size=1`.
3. **Initialization.** Every rank must start from identical weights. If you build the model on each rank with a per-rank seed, or use `torch.device("meta")` init plus `param_init_fn` that does not seed identically, ranks diverge from step 0 and the all-reduce averages garbage. The check: all-reduce the parameter norm at step 0 and confirm max-minus-min across ranks is 0.
4. **Loss normalization.** As covered earlier — `/k` for accumulation, and token-count normalization under padding. A run that normalizes per-microbatch and one that normalizes per-global-token will differ by a few percent forever.
5. **Gradient clipping.** The √N bug. Log `total_norm` in both runs; they must match.
6. **Dropout / RNG.** Each rank should have a *different* dropout seed (otherwise you have correlated noise and effectively less regularization) but a *deterministic* one derived from a base seed plus rank. Getting this backwards — same dropout mask on every rank — makes the FSDP run behave like a smaller-batch run.
7. **`reduce_dtype`.** bf16 reduction introduces a real, small bias. Set `reduce_dtype=torch.float32` for the comparison.

**🗣 Say this in the room:** "First I check that global batch and data order are actually identical, then that all ranks initialized to the same weights, then that loss normalization and grad clipping are globally correct. Step-0 loss must match to 1e-5; if it does not, it is data or init and nothing about FSDP is involved yet."

### FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, NO_SHARD — give me the table and tell me which one you reach for.

Four strategies, and they are four points on one axis: how much state you shard, and therefore how much you communicate.

| Strategy | Params | Grads | Optim | Bytes/param on N ranks | Comm volume | ZeRO equivalent |
|---|---|---|---|---|---|---|
| `NO_SHARD` | replicated | replicated | replicated | 16 | 2Ψ all-reduce | DDP |
| `SHARD_GRAD_OP` | replicated | sharded | sharded | 2 + 14/N | 2Ψ | ZeRO-2 |
| `FULL_SHARD` | sharded | sharded | sharded | 16/N | 3Ψ | ZeRO-3 |
| `HYBRID_SHARD` | sharded in-node, replicated across | " | " | 16/G (G = group size) | 3Ψ intra + 2Ψ inter | ZeRO-3 within replica groups |

`HYBRID_SHARD` is the one people underuse and it is usually the right answer for the 7B–70B range. It builds a 2D mesh: shard across the `G` GPUs inside a node (fast NVLink), replicate across the `R = N/G` nodes. Within a node you do the ZeRO-3 dance at 400+ GB/s where it is nearly free; across nodes you do a single all-reduce of the reduce-scattered gradients — DDP-level traffic on the slow link.

Concretely, 64 GPUs = 8 nodes × 8 GPUs, 13B model:

- `FULL_SHARD` over all 64: memory 13e9 × 16/64 = **3.25 GB/GPU**, but the all-gathers cross the network. Per step 3Ψ = 3 × 26 GB = 78 GB of which most is inter-node at 45 GB/s → **~1.2 s** (before accounting for the fact that a 64-way all-gather is latency-heavy).
- `HYBRID_SHARD` with groups of 8: memory 13e9 × 16/8 = **26 GB/GPU** — still fits an 80 GB card with room for activations — and the only cross-node traffic is one gradient all-reduce of 26 GB → 2 × 26 × (7/8) / 45 = **~1.0 s**, with the two all-gathers now intra-node and effectively free.

You gave up memory you did not need and bought back overlap. That is the trade to name out loud.

**⚠ Trap:** `HYBRID_SHARD` with a shard group that spans nodes is worse than either extreme — you get inter-node all-gathers *and* inter-node all-reduces. The shard group must align with the NVLink domain. Verify with `nvidia-smi topo -m` that your 8 ranks-per-node are actually the 8 NVLink-connected GPUs, and set your process-group construction accordingly. On a badly-configured Slurm allocation this is not automatic.

### What is `DeviceMesh`, and how does torchtitan use it to express 4D parallelism?

`DeviceMesh` is the abstraction that turns "which of my 1,024 ranks talk to each other for which purpose" from a pile of hand-built `ProcessGroup` objects into a named n-dimensional array. It is the single most important API change in PyTorch distributed in the last few years, because composing parallelism schemes is fundamentally a *topology* problem and there was no type for topology.

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh(
    "cuda",
    (2, 8, 2, 8),                                  # 256 GPUs
    mesh_dim_names=("pp", "dp_shard", "cp", "tp"),
)
tp_group = mesh["tp"]            # 8 ranks, must be inside one NVLink domain
dp_group = mesh["dp_shard"]      # 8 groups for FSDP sharding
```

The ordering of the tuple is the ordering of rank assignment: the **last** dimension varies fastest, so consecutive global ranks are in the same `tp` group. That is not a detail — it is the mechanism by which you place the highest-traffic parallelism dimension on the fastest link. TP does two all-reduces of `s×b×h` per layer and must be inside NVLink; put it last. PP does point-to-point sends of one activation tensor per microbatch boundary and tolerates a slow link; put it first, spanning racks.

`DTensor` then annotates each parameter with a mesh and a per-dimension placement, e.g. `distribute_tensor(w, mesh, [Shard(0), Shard(1)])` meaning "sharded on dim 0 along the first mesh axis and dim 1 along the second." Every collective is derived from placement mismatches: if an op needs `Replicate()` on an axis where the tensor is `Shard(0)`, DTensor inserts an all-gather. If a matmul produces a `Partial(sum)` result, the next op that needs it materialized inserts an all-reduce. You describe the layout; the collectives fall out.

**torchtitan** is Meta's PyTorch-native reference training stack built on exactly this: FSDP2 for the `dp_shard` axis, `parallelize_module` with a TP plan for the `tp` axis, `torch.distributed.pipelining` for `pp`, context parallelism for `cp`, plus float8 via `torchao`, DCP async checkpointing, and `torch.compile` per block. Its value as an interview reference is that it is small enough to read end to end — the point to make is that **the modern stack expresses parallelism as declarative sharding over a named mesh rather than as hand-written collectives**, which is the architectural difference from Megatron-LM's explicit-collective style.

**📅 Volatile:** torchtitan's supported parallelism axes and the exact `fully_shard` / pipelining APIs move fast. Talk about the mesh-and-placement design, not the function signatures, unless you have used the version in question that week.

**🗣 Say this in the room:** "DeviceMesh names the axes; DTensor places tensors on them; the collectives are inferred from placement mismatches instead of hand-written. The engineering judgment is in the axis *ordering* — TP last so it lands inside NVLink, PP first so it spans the slow links."
