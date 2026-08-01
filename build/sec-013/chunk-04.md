### When is an MoE the wrong choice? Give me a decision procedure, not a vibe.

I would run these gates in order and stop at the first failure. Any one of them is disqualifying on its own.

**Gate 1 — Do the total parameters fit in the HBM you are willing to buy, with room for KV?** Not the active parameters. The total. If a 141 GB model forces you from 2 GPUs to 8, the MoE has already lost unless it is replacing a model whose *quality* you cannot otherwise reach. This gate alone kills MoE for most single-GPU and single-node deployments.

**Gate 2 — Will your steady-state decode batch exceed roughly `E/k × 100` tokens?** That is the concurrency needed for expert weight-reuse comparable to a dense model at batch ~100. Mixtral: ~400 tokens. DeepSeek-V3-class: ~3,200. If your traffic and your KV budget cannot sustain it, you will run the MoE permanently in its worst regime. **This is the gate people skip and it is the one that decides most real cases.**

**Gate 3 — Is your workload prefill-heavy?** MoE's FLOP saving is real in prefill and illusory in decode. A long-output agent loop is the wrong shape. A RAG summarizer over long documents is the right shape.

**Gate 4 — Is your latency SLO tight at low batch?** Inline code completion with a 200 ms TTFT budget forces short queues and therefore small batches, which is exactly the regime where MoE is 2× worse. Latency SLOs and MoE are structurally in tension, because the SLO caps the batch and the batch is what MoE needs.

**Gate 5 — Can you keep the expert-parallel group inside one NVLink domain?** If the model forces you across nodes, budget for all-to-all being ~3× communication-bound and staff someone who can do overlap engineering.

**Gate 6 — Do you have the operational maturity?** Per-expert utilization telemetry, straggler diagnosis, placement management, dropless-versus-padding decisions. A dense model needs none of this. If your team is three people shipping a product, an MoE is a distraction with a monthly cost.

The clean summary of where MoE is *right*: high-concurrency, short-to-medium context, prefill-heavy, throughput-graded (not latency-graded) workloads on multi-GPU NVLink nodes — plus the entirely separate and genuinely excellent case of **local/edge inference at batch 1**, where you own all the memory, there is exactly one user, and the machine's bandwidth is terrible. A 30B-total / 3B-active model on a unified-memory laptop is the correct architecture precisely because there the memory is cheap and the bandwidth is the scarce thing, which inverts the datacenter trade.

**🗣 Say this in the room:** "I'd disqualify MoE if the total parameters push us to more GPUs than the quality gain justifies, if our concurrency can't reach roughly `E/k × 100` decode tokens, if the workload is decode-heavy, or if we have a tight low-batch latency SLO. The one place I'd argue *for* it against instinct is single-user local inference, where the memory-versus-bandwidth trade flips."

### How would you turn a dense checkpoint into an MoE? What is sparse upcycling and what goes wrong?

**📄 Paper:** Komatsuzaki et al. (2022), "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints" — the result is that you can convert a trained dense model into an MoE and continue training, recovering the MoE's advantage for a fraction of the cost of pretraining one from scratch. It replaced the assumption that MoE required a from-scratch run.

The recipe, mechanically: pick which layers become MoE layers (commonly every other one, or all but the first few — DeepSeek-V3 keeps its first three layers dense, which is a good instinct because early layers do generic low-level work that specialization does not help). For each chosen layer, **copy the dense FFN weights into all `E` experts**, so every expert starts identical to the original. Add a randomly-initialized router. Then continue pretraining on a data mix resembling the original, for on the order of a few percent of the original token budget.

Why copying rather than random init matters: at step zero, an MoE whose experts are all identical copies computes *exactly* the same function as the dense model regardless of routing (up to the gate weights summing to 1). So you start at the dense model's loss rather than at random, and you never take a quality regression you have to climb back out of.

What goes wrong, in order of how often I have seen it:

**Symmetry.** Identical experts receive identical gradients from the language-modeling loss, so nothing differentiates them and the router has nothing to learn from. The standard fixes are router jitter (multiplicative noise on the hidden state before routing) and a load-balancing loss strong enough to force the router to spread tokens, after which the experts diverge because they see different data. If you skip both, you get a beautifully balanced MoE whose experts remain near-identical and which costs `E×` the memory for zero quality gain.

**Gate-scale mismatch.** With top-2 and softmax gates summing to 1, the layer output at step zero equals the dense output. With top-1 and a gate probability of, say, 0.6, the output is scaled by 0.6 and you have introduced a systematic magnitude shift into the residual stream. Check what your gating convention does to the output norm at initialization and correct it, or eat a loss spike.

**Too little continued training.** Upcycling with a token budget that is too small gives you a model with MoE serving costs and dense-model quality — the worst outcome available. Budget real tokens.

**Learning-rate schedule.** You are resuming from a converged model, so you cannot restart at peak LR without destroying it, and you cannot use a near-zero LR or the experts will never differentiate. A warmup to a fraction of the original peak, then decay, is the pattern.

**⚠ Trap:** upcycling and then evaluating only on aggregate benchmarks. The failure mode of an under-trained upcycle is that experts remain nearly duplicates, which shows up as *normal* benchmark scores (the model still works) and *abnormal* routing entropy (the router is near-uniform because there is nothing to distinguish). Your routing telemetry catches this; MMLU does not.

### Your team fine-tuned an MoE and the eval collapsed after about 200 steps. Walk me through your diagnosis.

MoE fine-tuning is genuinely more fragile than dense fine-tuning, and there are a small number of characteristic causes. I would work the list in this order, because it is roughly ordered by frequency and by cheapness to check.

**First: is routing collapsing?** Pull the per-expert dispatch fractions and the two entropies. The signature of collapse is batch-marginal entropy falling toward zero while a handful of experts absorb most tokens. The mechanism is a positive feedback loop: an expert that gets slightly more tokens gets more gradient, becomes better, attracts more tokens. During pretraining the load-balancing loss holds this in check — but **fine-tuning recipes routinely drop the auxiliary loss**, because most fine-tuning frameworks were written for dense models and the loss is not in the trainer. That is the single most common cause. Fix: reinstate the balancing loss (or the bias-based balancer) during fine-tuning.

**Second: is your fine-tuning data narrow?** If you are fine-tuning on 50k examples of one domain, the *correct* routing for that data may genuinely be concentrated on a few experts. The router adapts fast (it is a tiny layer), routes everything to three experts, and those three experts overfit while the other 253 rot. You then evaluate on anything outside that domain and it has collapsed. This is catastrophic forgetting expressed through the router rather than through the weights, and it is specific to MoE. Fix: **freeze the router.** A frozen router preserves the pretrained routing distribution and forces the fine-tune to update experts within the existing specialization structure. ST-MoE and subsequent practice both point at freezing or heavily down-weighting router updates during fine-tuning, and it is my default.

**Third: is the learning rate wrong for a sparse model?** Each expert sees roughly `k/E` of the tokens, so with `E=256, k=8` an expert receives 3% of the gradient signal per step that a dense FFN would. The effective learning rate per expert parameter is far lower — which sounds safe, but it means the *router*, which sees 100% of tokens, is moving `32×` faster relative to the experts. Divergent timescales between router and experts is a real instability source. Fix: separate parameter groups with a lower LR on the router.

**Fourth: numerics.** Check router `logsumexp` magnitude and whether routing is done in fp32. If logits are growing, add z-loss.

**Fifth: capacity/dropping.** If your trainer uses a capacity factor and your fine-tune data is skewed, drop rates can go to 20%+ and you are training on a model that is randomly no-op-ing layers.

**🔍 Decision procedure, compressed:** entropy collapsed + load skewed → balancing loss missing; entropy fine but eval collapsed on out-of-domain only → router overfit, freeze it; loss spikes with logit growth → numerics, add z-loss; drop rate high → raise capacity factor or go dropless; nothing anomalous in routing → it is an ordinary fine-tuning problem, stop blaming the MoE.

**🗣 Say this in the room:** "First thing I check is whether the fine-tuning trainer dropped the load-balancing loss — most dense-model trainers never had it. Second is whether the router overfit to a narrow domain, which is MoE-specific catastrophic forgetting; my default is to freeze the router during fine-tuning and let the experts move. I'd have per-expert dispatch fractions and routing entropy on a dashboard before I ran a single step."

### The p99 on your MoE endpoint is 4× the p50. The dense model at the same throughput doesn't do this. What's happening?

The dense-versus-MoE contrast in the question is the clue: whatever this is, it is a property of *routing*, not of the model's size or the scheduler in general. Four candidate mechanisms, and I would separate them with specific measurements rather than guessing.

**Candidate 1 — batch-composition variance.** A dense model's decode step time is a constant given batch size. An MoE's step time is `max` over GPUs of that GPU's expert load, which varies with what happens to be in the batch. A batch that happens to be topically homogeneous (three tenants all sending Rust code) concentrates load on the same experts and stretches the step. This produces a *distribution* of step times with a long right tail, and since every user in the batch shares that step, tail steps hit everyone. Measurement: log per-step `max_gpu_load / mean_gpu_load` alongside step time and correlate. If the correlation is strong, this is it.

**Candidate 2 — a specific tenant or content class.** The 1% is not random; it is a particular customer. Measurement: join p99 requests against your tenant and language labels. If p99 requests are 80% one tenant, you have semantic skew, and the fix is a scheduler change — deliberately mix tenants within a batch rather than batching by arrival, which is a small change to your continuous-batching admission policy and often fixes it outright.

**Candidate 3 — a straggler GPU.** One card is thermally throttled, or has a degraded NVLink, or holds a ragged expert assignment (11 experts where its peers hold 10). Under a barrier, that GPU sets the pace whenever it is on the critical path. Measurement: per-GPU step time histograms. If one GPU is consistently the max, this is hardware or placement, not routing.

**Candidate 4 — padding/graph bucket transitions.** If you are running bucketed CUDA graphs with capacity padding, a batch that pushes one expert over a bucket boundary jumps to the next graph and does substantially more padded work. This produces a *bimodal* latency distribution rather than a smooth tail. Measurement: is the p99 a separate mode or a tail? A bimodal histogram is diagnostic.

**💰 What it costs.** Suppose p50 ITL is 10 ms and p99 is 40 ms on an 800-token response. p50 completion: 8.0 s. p99: 32 s. For a streaming chat UI, 32 s is a churned session. If 1% of 80,000 daily requests hit that, it is 800 bad sessions a day. And you cannot fix it by adding GPUs: adding GPUs to an expert-parallel deployment *increases* the number of participants in each barrier, so the max-of-N gets worse, not better. **The counterintuitive fix is often to make the EP group narrower and run more replicas.** That is worth saying out loud because it inverts the usual scaling reflex, and it is one of the genuinely MoE-specific pieces of operational judgment in this section.

### How does quantization interact with MoE? Anything different from quantizing a dense model?

Two things are different, and both bite in production.

**First: calibration data starvation.** Post-training quantization methods that need activation statistics — anything computing per-channel scales, outlier thresholds, or Hessian-based error compensation — need calibration tokens flowing through each weight matrix. In a dense model, all 512 calibration sequences flow through every FFN. In an MoE with 256 experts and top-8, each expert sees roughly `8/256 = 3%` of them. If your calibration set is 512 sequences × 2048 tokens = 1.05 M tokens, each expert sees ~33k tokens — and the *cold* experts, the ones at the bottom of the utilization histogram, might see a few hundred. Their quantization scales are then fit on noise.

**💰 Math:** to give every expert the same calibration signal a dense FFN would get, you need `E/k = 32×` the calibration data — 33.6 M tokens instead of 1.05 M. That is a real, quantifiable cost, and the rule I enforce is: **scale calibration set size by `E/k`, and verify per-expert token counts before accepting a quantized MoE.** Log the histogram; if the minimum is under a few thousand tokens, that expert's quantization is unvalidated.

The failure mode is nasty because it is invisible on aggregate evals: the badly-quantized experts are by definition the *rarely used* ones, so they contribute little to average benchmark scores and a lot to the specific rare inputs that route to them. You ship a quantized model that scores identically on MMLU and is broken on the long tail — which, for an enterprise product, is the traffic you were paid for.

**Second: mixed precision across experts is available and underused.** Because experts are physically separate weight blocks, you can quantize them at different precisions — keep the shared expert and the top decile of hot experts in fp8, push cold experts to int4. That is not possible in a dense FFN, where the matrix is monolithic. The memory win is real: for a 671 GB fp8 model, taking 70% of expert weights to int4 saves roughly `0.7 × 671 × 0.5 ≈ 235 GB` — three H100s' worth. The catch is that mixed precision within one grouped GEMM is a kernel problem, so check that your engine supports it before designing around it (**📅 Volatile:** per-expert mixed precision support varies by engine and version).

**⚠ Trap:** quantizing the router. It is 32k parameters out of 46 billion — 0.00007% of the model — and it makes discrete decisions where a small logit perturbation flips a top-k selection and changes the computation path entirely. **Keep the router in bf16 or fp32, always.** I have seen this cause a "quantization is broken" investigation that was one line of a quantization config's layer-exclusion list.

### Where do you put LoRA adapters when fine-tuning an MoE, and what's the trap?

The default answer — attach LoRA to `q_proj`, `k_proj`, `v_proj`, `o_proj` and leave the FFN alone — is not just acceptable for an MoE, it is the *recommended* configuration, and for a reason specific to sparsity rather than the usual "attention adapters are enough" hand-wave.

If you put a LoRA adapter on expert `e`'s matrices, that adapter only receives gradient from the tokens routed to expert `e`. With 256 experts at top-8, an average expert sees 3% of your tokens; a cold expert sees a fraction of a percent. Now recall that LoRA's whole premise is a small number of trainable parameters learning from your whole dataset. Split that dataset 256 ways and the rank-16 adapter on a cold expert is fitting a handful of examples. You get adapters that are excellently trained on hot experts, undertrained on warm ones, and pure noise on cold ones — a *non-uniform* fine-tune where quality depends on which expert a query happens to route to. That is a horrible property to debug, because the same prompt phrased two ways can route differently and produce inconsistent quality.

So my ordering:

1. **Attention-only LoRA.** Safe, uniform gradient coverage (every token passes through attention), and the standard result that attention adapters carry most of the adaptation benefit holds here too.
2. **Attention LoRA + the shared expert.** If the architecture has an always-on shared expert, that is a dense component every token traverses, so an adapter there gets full gradient coverage. This is the best of both worlds and is underused.
3. **Expert LoRA, only with a lot of data and only with per-expert token accounting.** If you do it, instrument the per-expert token count during training and treat any expert below a threshold as untrained. Consider a single *shared* adapter applied to all experts, which restores uniform gradient coverage at the cost of not letting experts specialize their adaptation.
4. **Router LoRA / router training — almost never.** As covered, an updated router on narrow data is MoE-specific catastrophic forgetting. Freeze it.

**⚠ Trap:** setting `target_modules` to a regex that happens to match `gate_proj` / `up_proj` / `down_proj` inside expert modules. On a 256-expert model that quietly creates `256 × 3 × n_layers` adapters — for a 58-MoE-layer model that is 44,544 LoRA modules, which will blow up your adapter file size, slow training substantially, and give you the non-uniform training problem above. It does not error. It just produces a 6 GB "lightweight" adapter and a confusing eval. **Print the trainable-parameter count and the module list before every MoE LoRA run.** If the count is not what you predicted on paper, stop.

### Design this with me. You're standing up an internal coding assistant for 2,000 engineers on a fixed GPU budget. Dense or MoE, and show your work.

I will start with the measurements that determine the answer, because the architecture choice is downstream of the workload shape, and stating that first is the answer to the real question being asked.

**Workload characterization.** Two distinct products hiding under one name, and they must be served by different models.

*Product A — inline completion.* Fires on keystroke pause. Context ~8k tokens of surrounding code, output ~30 tokens, TTFT budget under 200 ms or engineers turn it off. Volume is enormous — call it 2,000 engineers × 400 accepted-or-rejected completions/day = 800k/day.

*Product B — chat and agentic edits.* Context ~30k tokens (files, diffs, retrieved symbols), output ~800 tokens, TTFT budget ~2 s, ITL budget ~30 ms so the stream reads faster than reading speed. Volume: 2,000 × 40/day = 80k/day, over an ~8-hour working window = 2.8 requests/second.

**Product A is not an MoE question.** A 200 ms TTFT SLO caps queue depth, which caps batch size, which puts you permanently on the wrong side of the MoE crossover. Serve this with the smallest dense model that clears your acceptance-rate bar — a 3B-to-7B-class code model, fp8, TP=1, many replicas, aggressive prefix caching on the file context. If someone proposes an MoE here I would push back hard: at batch 8 an MoE reads its full weight set to serve 8 tokens, and you will miss the SLO to buy FLOPs you were never short of.

**Product B is the real question.** Size the concurrency first, because it determines everything: at 2.8 req/s with each request occupying the decoder for `800 tokens × 30 ms = 24 s`, Little's law gives `2.8 × 24 = 67 concurrent sequences`. That is the number that decides the architecture.

*Memory, both candidates at fp8 weights and fp8 KV.*

- **Llama-3-70B dense:** weights 70 GB. KV = `2 · 80 layers · 8 kv_heads · 128 · 1 byte = 163,840 B = 160 KiB/token`. At 67 × 30k = 2.0 M resident tokens → `2.0e6 × 163840 = 328 GB`. Total 398 GB → **6 GPUs**, round up to 8 for TP divisibility and headroom.
- **Mixtral 8x22B MoE:** weights 141 GB. KV = `2 · 56 · 8 · 128 · 1 = 114,688 B = 112 KiB/token` → `2.0e6 × 114688 = 229 GB`. Total 370 GB → also **8 GPUs**.

Note the MoE's *KV* is smaller (fewer layers), which partly offsets its larger weights. Same GPU count. So memory does not decide it.

*Decode speed on 8× H100 (3.35 TB/s, ~990 TFLOP/s fp8 each).*

- Dense: `70 GB / 8 = 8.75 GB` per GPU per step → `8.75e9 / 3.35e12 = 2.6 ms` bandwidth floor. Compute at B=67: `2 · 70e9 · 67 = 9.4 TFLOP / 7.92 PFLOP/s = 1.2 ms`. Bound: **2.6 ms/token.**
- MoE: `141 / 8 = 17.6 GB` per GPU → **5.3 ms/token** (compute is only 0.66 ms; irrelevant).

Both clear the 30 ms ITL budget with room, but the dense model is **2× faster per token** and gives you 2× the headroom for traffic growth before you need more GPUs. `B_eff` for the MoE is `67 × 2/8 ≈ 17` — it is running at an effective batch of seventeen, deep in its bad regime.

*Prefill / TTFT.* 30k prompt tokens. Dense: `2 · 70e9 · 30000 = 4.2 PFLOP`; at 50% MFU on 7.92 PFLOP/s → **1.06 s**. MoE: `2.34 PFLOP` → **0.59 s**. The MoE wins TTFT by ~470 ms — genuinely meaningful. But this is a coding assistant where the file/repo context repeats across turns, so prefix caching will eliminate most prefill on turns 2+, and the advantage shrinks to the first turn of a session.

**My call: dense 70B-class for Product B.** Same GPU count, 2× better ITL, 2× more headroom, dramatically simpler operations — no expert telemetry, no straggler management, no placement table. The MoE's only win is first-turn TTFT, which prefix caching largely erases. **I would revisit if concurrency grew past ~300 decode tokens** (roughly 12 req/s, a 4× traffic increase) — at that point the crossover flips and the MoE's FLOP advantage starts to matter.

**💰 Build-versus-buy, because this is what actually gets asked next.** 8× H100 on-demand at ~$2.50/GPU-hour (**📅 Volatile**) = $20/hour = **$14.6k/month** for Product B. Against a frontier API at, say, $3/Mtok input and $15/Mtok output: input is `80,000 × 30,000 = 2.4 B tokens/day`; with prefix caching hitting 80% of it at a 90% discount, that is `0.8 × 2400 Mtok × $0.30 + 0.2 × 2400 Mtok × $3.00 = $576 + $1,440 = $2,016/day`. Output: `80,000 × 800 = 64 Mtok × $15 = $960/day`. Total ≈ **$3.0k/day ≈ $89k/month**. Self-hosting is ~6× cheaper *if* you sustain utilization — which at 67 concurrent sequences on 8 GPUs, you do. That 6× is the entire business case, and it is also why the answer changes completely at 200 engineers instead of 2,000: at one-tenth the traffic the API bill is $8.9k and the GPUs still cost $14.6k, and you should buy.

**How I would know it works:** acceptance rate for Product A, edit-survival rate (does the model's code still exist 24 hours later) for Product B, TTFT and ITL percentiles per product, and cost per accepted suggestion. Not perplexity.

### 🎯 How much of the training-side routing math should I actually know? Where's the line between Applied AI Engineer and Research Engineer here?

I will be direct about the boundary, because over-preparing here is a real waste of your time and under-preparing on the serving side is what actually loses the loop.

**What every Applied/AI Engineer candidate must be able to derive on demand** — this is fair game in any MoE-touching loop:

*How the router gets a gradient at all, given that top-k is non-differentiable.* This is the elegant bit and it is worth being able to state cleanly. The selection `topk(logits)` is a discrete operation with zero gradient almost everywhere. But the layer output is `y = Σ_j g_j · E_j(x)` where `g_j` is the router's (softmax) probability for the chosen expert. So `∂L/∂g_j = ⟨∂L/∂y, E_j(x)⟩` — the router receives gradient through the *magnitude* path, not the selection path. The learning signal is: "increase the weight on experts whose output was aligned with the direction that reduced loss." Which experts get *considered* is frozen by the argmax; which of the considered ones get *emphasized* is learned. That is why the balancing loss is needed at all — nothing in the primary gradient encourages the router to explore experts it did not already select.

*The auxiliary loss and z-loss formulas*, and what each is defending against.

*Why fine-tuning an MoE is fragile*, and that freezing the router is the standard mitigation.

**What is Research-Engineer territory** and where "I know the shape of this but I haven't implemented it" is a completely acceptable and honest answer:

Straight-through and Gumbel-style estimators for differentiable routing; the theory of why top-1 trains at all when everyone expected it not to; batch-prioritized routing orderings; the interaction of router jitter magnitude with expert count; deriving the optimal `E`, `k` and granularity for a given compute budget (MoE scaling laws); the convergence analysis of the bias-based balancer's control loop; and expert-parallel training-time optimizer sharding and communication scheduling.

**🗣 Say this in the room** if pushed past your line: *"The router gets gradient through the gate magnitude, not through the top-k selection — the selection is frozen by the argmax, which is exactly why you need an explicit balancing objective. Past that, the training-side routing literature — straight-through estimators, MoE scaling laws for choosing E and k — I've read but not implemented; my depth is on the serving side, where I can tell you why an MoE's effective batch is `B·k/E` and what that does to your GPU bill."* That sentence trades a weakness you were going to be caught on for a strength, which is the correct move.

**⚠ Trap:** bluffing the training math. An interviewer at a lab will follow up two levels deep and the failure is much more expensive than the honest boundary. And for the Applied/Product roles you are targeting — Cursor, Perplexity, Notion, Sierra, Harvey, or Meta/Google applied teams — the serving arithmetic is what they are actually testing. Nobody at those companies is training an MoE from scratch; several of them are serving one.

### 🏋 Drill: the MoE budget sheet, 12 minutes, no calculator beyond mental arithmetic.

**Setup.** You are handed this config and nothing else:

```
n_layers = 48        d_model = 6144       n_heads = 48      n_kv_heads = 8
d_head = 128         vocab = 128000       tied_embeddings = false
n_routed_experts = 64    n_shared_experts = 1    top_k = 6
moe_intermediate_size = 1536     first_k_dense_layers = 2
```

**Produce, in 12 minutes, on paper:**

1. Total parameters, to two significant figures.
2. Activated parameters per token, to two significant figures.
3. KV cache bytes per token at fp8, and total KV for 40 concurrent sequences at 24k context.
4. Minimum H100-80GB count to serve it at fp8 weights with that KV load, plus 15% headroom.
5. The effective decode batch `B_eff` at 40 concurrent sequences, and whether that is a good or bad operating point.
6. One sentence: would you deploy this, and what single measurement would change your mind?

**Pass criteria.** Parts 1–3 within 5% of the true value. Part 4 exactly right. Part 5 correct formula and correct verdict. Part 6 must name a *measurement*, not a preference.

**Worked answer, to check yourself.** Attention per layer: Q `6144²=37.75M`, O `37.75M`, K `6144·1024=6.29M`, V `6.29M` → **88.1 M**. Dense FFN layers (2 of them) — assume the same `8/3`-ish width the MoE would imply is wrong here; a dense layer in such configs typically uses a much wider `d_ff`, so state your assumption; taking `d_ff = 16384` gives `3·6144·16384 = 302 M` per dense layer. MoE layer experts: `(64+1) · 3 · 6144 · 1536 = 65 · 28.3M = 1840 M`; router `6144·64 = 0.4M`. So MoE layer ≈ `88.1 + 1840 = 1928 M`, × 46 layers = **88.7 B**; dense layers `2 × (88.1 + 302) = 0.78 B`; embeddings `128000 · 6144 × 2 = 1.57 B`. **Total ≈ 91 B.** Active per token in an MoE layer: `88.1M + (6+1) · 3 · 6144 · 1536 = 88.1 + 198 = 286 M`, × 46 = 13.2 B, plus dense layers 0.78 B plus embeddings 1.57 B → **≈ 15.5 B active.** Ratio 5.9×. KV: `2 · 48 · 8 · 128 · 1 byte = 98,304 B = 96 KiB/token`; 40 × 24k = 960k tokens → `960e3 × 98304 = 94.4 GB`. Weights fp8 = 91 GB. Total 185 GB × 1.15 = 213 GB → **3 GPUs**. `B_eff = 40 · 6 / 64 = 3.75` — **terrible**; you are reading 91 GB of weights per step to serve an effective batch of under four. Verdict: do not deploy for this traffic; the measurement that would change my mind is sustained concurrent decode tokens — I need roughly `64/6 × 100 ≈ 1,070` before this architecture earns its memory.

### 🏋 Drill: implement a capacity-limited MoE layer in 30 minutes, then answer three follow-ups in 5.

**Task, unaided, no autocomplete, 30 minutes.** Implement `class CapacityMoE(nn.Module)` with:

- top-`k` token-choice routing with an fp32 softmax over the selected logits,
- a `capacity_factor` producing `capacity = ceil(cf · N · k / E)` slots per expert,
- **token dropping**: tokens beyond an expert's capacity are dropped, and their contribution is zero (the residual carries them),
- gate renormalization over the *surviving* experts for each token,
- a returned `aux` dict containing `drop_rate`, per-expert `dispatch_fraction`, batch-marginal `routing_entropy`, and the Switch auxiliary loss `α·E·Σ f_i P_i`.

**Pass criteria (all must hold):**
1. Output shape equals input shape, and with `capacity_factor = 10.0` (effectively infinite) your layer's output matches a reference no-capacity implementation to within `1e-5` in fp32.
2. `drop_rate` is exactly `0.0` at `capacity_factor ≥ k` and strictly positive on a deliberately skewed router (set the router bias so one expert wins for 90% of tokens) at `capacity_factor = 1.0`.
3. Setting all router logits equal produces `routing_entropy == log(E)` to within `1e-4` and `aux_loss == α` to within `1e-4`. (If your aux loss does not land on `α` at uniform routing, your formula is wrong — this is the single best self-check for that function.)
4. A token whose every selected expert dropped it produces exactly the input's residual contribution — verify by asserting that row of the MoE output is all zeros.
5. Gradients flow to the router: `router.weight.grad` is non-zero after a backward pass. (If it is zero you have detached the gates somewhere — a real and common bug.)

**Then, in 5 minutes, out loud, no notes:**

1. *You raise `capacity_factor` from 1.0 to 2.0. What exactly gets worse, and by how much?* (Expected: dispatch buffer and all-to-all payload double; drop rate falls toward zero; wasted padded FLOPs rise; give the buffer size in tokens for your `N, k, E`.)
2. *Your layer runs at `E=64, k=2` with 32 concurrent decode sequences. What is `B_eff`, and what dense batch size is it equivalent to?* (Expected: `32 · 2 / 64 = 1.0` — the MoE is being read at an effective batch of **one**, which is the worst possible operating point, and you should say so without hedging.)
3. *Why is this implementation not what ships?* (Expected: the per-expert Python loop and launch overhead; dynamic shapes breaking CUDA graph capture; production uses grouped GEMM or block-sparse dropless kernels; and inference should generally be dropless because you cannot tell a user their token was dropped.)

**How to grade yourself honestly:** if you needed to look up `index_add_` versus `scatter_add_`, or you got the `order // k` mapping from sorted slot back to source token wrong on the first try, you are not yet at the bar — that indexing is the entire difficulty of the exercise and it is exactly what a whiteboard version tests. Redo it cold in two days.
