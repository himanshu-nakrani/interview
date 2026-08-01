### Design the RL environment for an agentic task. What are the requirements a training loop imposes that a production agent doesn't?

The mental model: a production agent environment needs to work *once*, correctly, for a real user. A training environment needs to work *ten million times*, identically, cheaply, and while being actively attacked by an optimizer. Those are different systems, and the second one has requirements the first has never had to think about.

**Reset semantics are the first-class requirement.** Every rollout must start from a byte-identical initial state. If rollout 4,097 sees leftover rows from rollout 4,096, you have injected correlated noise into the reward and it is undebuggable. Concretely: snapshot-and-restore, not cleanup-and-reuse. A container image or VM snapshot restored per episode; a database restored from a template (`CREATE DATABASE x TEMPLATE y` in Postgres is exactly the right primitive and is fast); a filesystem overlay discarded at episode end. **Cleanup scripts are how you get a slow, correlated, irreproducible environment. Never clean up; always destroy and recreate.**

**Seeding must be explicit and total.** Every source of nondeterminism — the environment's RNG, any simulated user, mock service responses, ordering of concurrent operations — takes a seed derived from `(episode_id, task_id)`. That gives you replay: given a trajectory that produced a weird reward, you can re-run it exactly and step through. Without replay, debugging an RL run is guesswork.

**Flakiness is a reward-noise budget, and you must measure it.** Run the same reference policy through the same task 20 times and count verdict disagreements. That flake rate is a hard floor on your signal-to-noise. At a 5% flake rate and a policy pass rate of 40%, roughly 12% of your positive rewards are noise. **My gate: flake rate below 1%, and any task above 2% is quarantined out of the pool rather than tolerated.** External network calls are the number-one cause; mock or record-replay every third-party dependency.

**Throughput and concurrency.** You need hundreds of concurrent episodes. That means the environment cannot hold a global lock, cannot bind a fixed port, cannot assume a single working directory, and cannot share a database. Per-episode isolation at every layer. This is ordinary backend multi-tenancy work and it is exactly where your existing skills transfer directly.

**Cost per rollout is a design constraint, not an afterthought.** If an episode costs $0.05, your batch size is a budget decision. Instrument cost per episode from day one and put it on the dashboard next to reward.

**Timeouts and partial episodes.** An agent that loops forever must be killed, and you must decide what reward a killed episode gets. Scoring it 0 makes "time out" competitive with "fail," which is fine; scoring it *better* than failing teaches stalling. Mask it out if the truncation rate is high.

**🗣 Say this in the room:** "The training environment has three requirements production never imposed: snapshot-based reset so every episode starts byte-identical, total seeding so any trajectory is replayable, and a measured flake rate — because flakiness is reward noise and it sets a hard floor on your signal-to-noise. I quarantine any task above about a 2% flake rate rather than living with it."

### Where does the wall-clock time actually go in an RLVR step, and how do you rebalance it?

Break the step into four phases and measure each. For a long-CoT run on a single node this is roughly what you see, and the shape is what matters more than my exact digits:

| Phase | Share | Why |
|---|---|---|
| Generation (rollout) | 60–80% | Autoregressive decode, memory-bandwidth-bound, G× the samples |
| Verification | 1–10% | CPU work, parallelizable, hidden if you stream |
| Logprob recomputation | 5–15% | Forward passes over policy and reference on all rollout tokens |
| Backward + optimizer | 10–20% | The only phase that actually updates weights |

The headline: **you spend the large majority of an RLVR run generating tokens, and only 10–20% of it learning.** That single fact drives every infrastructure decision in this area.

**The rebalancing moves, in order of leverage:**

1. **Use a real inference engine for rollouts.** Generating with the training framework's `model.generate()` is the most common and most expensive mistake in this space — no continuous batching, no paged KV cache, poor kernels. vLLM or SGLang gives you a large multiple on decode throughput for the same hardware. Every serious RL framework (veRL, OpenRLHF, TRL's vLLM integration, SkyRL) is built around this.
2. **Colocate vs disaggregate.** Colocated: the trainer and the inference engine share GPUs, alternating phases. Simple, no weight transfer over the network, but the GPUs are idle for whichever phase is not running and you pay a memory tax for holding both. Disaggregated: separate GPU pools, generation runs continuously, weights are pushed periodically. Better utilization, more moving parts, and it forces you into async (next question).
3. **Increase inner epochs `μ`.** Amortize expensive rollouts over more gradient steps. Watch the clip fraction; stop raising `μ` when it exceeds ~15%.
4. **Overlap verification with generation.** Stream completions to the verifier as they finish. This is free and it removes verification from the critical path entirely.
5. **Reuse rollout logprobs as `old_logprobs`.** If your inference engine returns logprobs, you avoid a full forward pass — but only if the engine's numerics match the trainer's closely enough (see the `ratio ≈ 1` invariant). Validate before you rely on it.
6. **Cap generation length aggressively.** Decode time is linear in tokens and the tail dominates: if 5% of your samples run to 16k tokens while the median is 2k, that 5% can be 30% of your generation wall-clock. Truncating the tail is often the single biggest wall-clock win available.

**💰 Math:** concretely, 512 prompts × G=8 × 3,000 mean output tokens = 12.3M tokens per step. On 8×H100 running a 7B at ~10k output tok/s aggregate with continuous batching, that is 1,230 s ≈ 20.5 min of pure decode. At $2/GPU-hr × 8 = $16/hr, generation costs `20.5/60 × $16 = $5.47` per step. If instead you generate with HF `generate()` at ~1.5k tok/s, the same step takes 137 minutes and costs $36.50 — a **6.7× cost increase from one library choice**. Over 500 steps: $2,735 vs $18,250. This is the arithmetic to have ready when someone proposes skipping the vLLM integration to ship faster.

### Explain async RL and staleness. How stale is too stale?

Synchronous RL has a structural inefficiency: while the trainer computes gradients, the inference workers are idle, and while the inference workers generate, the trainer is idle. On a disaggregated setup that is a lot of expensive silicon doing nothing.

Async RL decouples them. Generation workers pull prompts and produce trajectories continuously into a buffer, using whatever policy weights they currently hold. The trainer consumes from the buffer and updates. Weights are pushed from trainer to generators every `k` steps. Nobody waits.

**The cost is staleness.** A trajectory in the buffer was generated by policy `π_{t−s}` but is being used to update `π_t`. That is off-policy data, and the importance ratio `π_t/π_{t−s}` is now genuinely far from 1 rather than exactly 1. The clip is doing real work instead of being inert, and if `s` grows large enough, most of your gradient gets clipped away and you are burning compute to produce truncated updates.

**How stale is too stale — the measurement, not a rule of thumb.** Do not guess a number; instrument it:

- **Clip fraction.** Below ~10%, staleness is fine. Above ~20–30%, most of your update is being truncated and you should reduce the staleness bound.
- **Mean and p99 importance ratio.** If p99 `ρ` is drifting far from 1 (say beyond 1.5 or below 0.6), the buffer is too old.
- **KL between the generating policy and the current policy.** The most direct measurement of what staleness means. Bound it explicitly.

**The mechanism I implement** is a hard staleness bound: tag every trajectory with the policy version that produced it, and **drop any trajectory older than `s_max` versions** at consumption time. Start with `s_max = 1` (one-step-off-policy, which is nearly free and captures most of the utilization win), and raise it only while the clip fraction stays acceptable. This is the same pattern as a bounded staleness read replica — you are trading freshness for throughput and you must set the bound explicitly rather than discover it.

**⚠ Trap:** treating async RL as a pure infrastructure optimization with no algorithmic consequence. It changes the algorithm. Vanilla GRPO's derivation assumes samples come from `π_old` and you correct with a bounded ratio; large staleness violates the bound the clip was designed for. Teams have shipped async setups where the effective algorithm was heavily-clipped garbage and the reward curve still looked plausible, because clipped updates are small and small updates look like slow learning rather than broken learning. **Log the clip fraction and the policy-version lag distribution or you cannot tell the difference.**

**📐 Numbers you must know:** synchronous disaggregated RL typically leaves 30–50% of GPU-time idle across the two pools. One-step-off-policy async recovers most of that at a clip-fraction cost of a few points. That is the trade: roughly a 1.5–2× throughput gain for a small, *measurable* bias, and the measurement is what makes it engineering rather than hope.

### How do the trainer's weights get to the inference engine, and why is that hard?

It sounds like a file copy and it is the piece that most often eats a week.

**The scale of the problem.** A 7B model in bf16 is 14 GB; a 70B is 140 GB. If you push weights every step and a step is 20 minutes, 14 GB over a 25 GB/s interconnect is under a second — fine. Over a 10 Gb/s network it is 11 seconds — tolerable. Naively serializing to disk, uploading to object storage, and having the inference workers reload from a checkpoint takes minutes and can easily exceed your step time. That is the failure mode: a weight-sync path that costs more than the training step it enables.

**The three mechanisms, in ascending order of sophistication:**

1. **Checkpoint round-trip.** Trainer writes a checkpoint; inference workers reload. Simple, framework-agnostic, and far too slow for frequent sync. Acceptable when you sync every 50+ steps.
2. **Direct memory transfer.** Trainer and inference engine live in the same process group; weights move via NCCL broadcast or, when colocated on the same device, a direct pointer update into the engine's weight tensors. This is what the mature frameworks do — veRL's hybrid-engine design and the vLLM weight-update APIs used by OpenRLHF and TRL exist precisely for this. Sub-second for a 7B.
3. **Colocation with in-place update.** Trainer and inference engine share the same GPU memory for weights; "syncing" is copying a sharded tensor into the engine's layout. Fastest, and the reason "hybrid engine" designs exist.

**Why it is genuinely hard, beyond the copy:**

- **Layout mismatch.** The trainer holds FSDP- or ZeRO-sharded, possibly fp32-master weights. The inference engine wants a tensor-parallel layout in bf16, possibly with fused QKV projections and a different expert layout for MoE. The reshard-and-convert is fiddly, model-architecture-specific, and a rich source of silent bugs — a transposed weight will not crash, it will just make your model worse.
- **Consistency during the swap.** In-flight requests on the inference engine must not see half-old, half-new weights. You need a barrier: drain, swap, resume.
- **Memory pressure.** During the swap you may transiently hold two copies. On an 80 GB card already holding a training shard and a KV cache, that can OOM.
- **Validation.** The single most valuable test in this entire subsystem: **after a sync, generate one fixed prompt at temperature 0 in the inference engine and compare the logprobs to the trainer's forward pass on the same tokens.** They should match to within bf16 noise. This catches layout bugs, dtype bugs, and stale-shard bugs immediately, and it costs one forward pass.

**⚠ Trap:** assuming a small logprob mismatch between trainer and inference engine is cosmetic. It is not — it is exactly the mismatch that breaks the `ratio ≈ 1` invariant on the first inner step, so the clip starts binding on numerical noise. If you take one thing from this question: **assert trainer/inference logprob agreement after every weight sync, and alert on drift.**

### Compare the RL libraries. What would you actually pick and why?

The honest framing: this is the most volatile part of this section, the projects move fast, and an interviewer is testing whether you have *used* one rather than whether you can list six. **📅 Volatile — verify feature sets and maturity before your loop.**

**TRL (`GRPOTrainer`)** — Hugging Face's post-training library. Best on-ramp: it is the same API surface as `SFTTrainer` and `DPOTrainer`, integrates with PEFT and Accelerate, and has vLLM-backed generation. You write a Python reward function that takes completions and returns floats, which is exactly the right abstraction for the verifier-first mindset. Where it strains: very large models and multi-node scale-out, where the more specialized systems are built for it. **This is where I start every project**, and I would say so — starting with the heavier framework before you know your reward function is a classic waste of two weeks.

**veRL** (volcengine) — production RL framework built on the HybridFlow architecture (Sheng et al., 2024), which separates the single-controller dataflow from multi-controller compute so you can express PPO/GRPO variants without rewriting the parallelism. Strong FSDP/Megatron backends, mature vLLM/SGLang rollout integration, designed for multi-node. Pick it when you are training large models at scale or need to customize the algorithm.

**OpenRLHF** — Ray-based, one of the earliest mature open PPO/GRPO stacks for LLMs, with a well-worn disaggregated design (separate actor/rollout/reference/reward placement) and vLLM integration. Good when you want explicit control over resource placement.

**SkyRL** (Berkeley Sky Computing) — oriented toward long-horizon *agentic* RL with real environments, modular so you can plug in your own environment and trainer. Pick it when your rollout is an agent trajectory in a real environment rather than a single completion.

**ART** (OpenPipe's Agent Reinforcement Trainer) — aimed at making agent RL accessible from an existing agent codebase: you keep your agent loop and it handles the RL. Attractive for a small applied team that already has a working agent and wants to try RL without rebuilding it.

**VerlTool** — tooling around veRL for tool-use/agentic RL, giving you a unified interface for tool-calling environments.

**My decision procedure:**
- Single-turn verifiable task, ≤14B model, one node → **TRL GRPOTrainer**. Ship the verifier, not the trainer.
- Large model, multi-node, custom algorithm → **veRL** (or OpenRLHF if you prefer Ray's placement model).
- Multi-turn agent in a real environment → **SkyRL** or **ART**, and expect to write the environment yourself regardless.

**🗣 Say this in the room:** "I'd start with TRL's GRPOTrainer because the reward function is just a Python callable and the interesting work is the verifier, not the trainer. I'd move to veRL when I outgrow single-node or need to modify the algorithm, and to SkyRL or ART if the rollout is a multi-turn agent trajectory rather than one completion. The thing I would not do is start on the heavyweight framework before I know whether my reward signal is any good."

### GRPO gives every token the same advantage. What does that cost on a 40-step agent task, and what would you do instead?

The cost is that you punish correct behavior, and on long horizons that is not a rounding error.

**Concretely.** A 40-step trajectory: the agent searches, reads three files, forms a correct hypothesis, writes a correct patch at step 30, and then at step 38 runs the wrong test command and the episode fails. Outcome reward: 0. GRPO's advantage: negative, applied uniformly to every token of all 40 steps. Steps 1–37 were good and you just pushed them down. Meanwhile another trajectory blundered through 39 steps and got lucky at step 40; every one of its bad tokens gets positive credit.

The signal is not *wrong* — averaged over many samples, genuinely-good prefixes appear more often in successful trajectories, so the expectation points the right way. But the **variance is enormous**, and variance is what you pay for in sample count. The number of rollouts you need scales badly with horizon length: an outcome reward carries roughly one bit of information about a trajectory, and you are trying to assign credit across hundreds of decisions. This is why agentic RL is 10–100× more sample-hungry than single-turn math RL, and why it costs dollars per rollout instead of cents.

**What you can do, in ascending order of cost and cleverness:**

1. **Shorten the horizon.** Decompose the task so each RL episode is a few steps, not forty. Train the sub-skill, not the saga. Frequently the right answer and frequently ignored because it is unglamorous.
2. **Add programmatic intermediate rewards where they are genuinely verifiable.** Did the tool call have valid arguments against the schema? Did the file the agent opened exist? Did the patch apply cleanly? Each is a deterministic per-step check. These are dense, cheap, and unhackable in the ways judges are — but they reward *procedure*, so keep their weight small relative to the outcome or you will train a model that performs the ritual without solving the task.
3. **Turn-level advantages.** Treat each turn as the unit and assign advantages per turn, with a group of samples per state. Requires branching rollouts from intermediate states, which requires your environment to support snapshot-and-fork — another reason snapshot-based reset matters.
4. **Monte-Carlo value estimation at intermediate states.** From state `s_t`, roll out `m` continuations and use the empirical success rate as `V(s_t)`; the per-step advantage is then the change in estimated value. This is essentially how automatic process-reward labelling works (the Math-Shepherd line of work does exactly this for math), and VinePPO applies MC value estimates in place of a learned critic for LLM reasoning. It is principled and it is expensive — `m` extra rollouts per state.
5. **Bring back a critic.** PPO with GAE exists for precisely this problem. For genuinely long-horizon agentic RL, "GRPO replaced PPO" stops being true.

**⚠ Trap:** adding a dense shaped reward for "good agent behavior" defined by a judge. You will get an agent that performs *legibly* rather than effectively — long deliberate-sounding preambles, ostentatious tool use, careful-sounding summaries — because that is what the judge rewards and it is far easier than solving the task. Dense rewards on long-horizon tasks are where specification gaming is most severe. Keep dense terms programmatic and keep them small.

### Tell me about verifier-gated step-level credit assignment. Why is this the interesting research direction right now?

The framing: outcome rewards are unhackable but uninformative; process rewards are informative but hackable. The 2026 research line is trying to get both by **using a verifier to decide which process labels you are allowed to trust.**

The problem it solves. A process reward model trained on human step-labels (the "Let's Verify Step by Step" approach, Lightman et al., 2023, and its PRM800K dataset) gives you dense per-step signal, and that is exactly what long-horizon RL needs. But a PRM is a learned model, so optimizing hard against it reproduces every reward-model pathology you were trying to escape by moving to RLVR — and worse, it does so *densely*, at every step, which gives the policy far more surface to attack.

The idea. Do not let the process signal be the ground truth; let it be a *proposal* that a verifier confirms. Several shapes of this:

- **MC-verified step labels.** Label a step as good if continuations rolled out from it succeed at a higher verified rate than continuations from the previous step. The label's ground truth is the outcome verifier; the process signal is derived, not learned independently. This is the automatic-process-annotation idea (Math-Shepherd) used as a credit-assignment mechanism rather than as PRM training data.
- **Gating the PRM by outcome.** Only apply the PRM's step credit within trajectories whose outcome the verifier confirmed. A hacked-looking high PRM score on a trajectory that failed the outcome check is discarded rather than reinforced.
- **Per-step programmatic checks in agentic settings.** Each step's tool call is validated by the schema, the environment's state transition is checked against a spec, and only steps that pass the deterministic gate are eligible for positive credit. This is the version most relevant to an applied AI engineer, because it needs no PRM at all.

Why it is interesting now rather than in 2023: agentic tasks have become the frontier target — SWE-bench Verified, Terminal-Bench, τ-bench-style tool use — and their horizons are exactly where outcome-only GRPO breaks down. So the credit-assignment problem that math RL could ignore has become the binding constraint.

**📅 Volatile:** this is an unsettled area and the specific methods will have moved by the time you read this. Discuss it as a *problem shape* — how do you get dense signal without inheriting dense reward hacking — and name the constraint (any process signal must be anchored to something a program checks). That framing will still be correct; specific method names may not be.

**🗣 Say this in the room:** "The tension is that outcome rewards are hard to hack and carry about one bit per trajectory, while process rewards are dense and hackable at every step. The direction I find convincing is anchoring the process signal to the outcome verifier — either deriving step labels from verified rollout success rates, or gating a process model's credit on trajectories the outcome verifier confirmed — so the dense signal never becomes an independent objective the policy can attack."

### A startup founder tells you "we should RL our agent on our own task." What do you tell them?

I ask for four preconditions, and if any is missing I recommend against it — not out of conservatism, but because each missing precondition converts the project from "expensive with a chance of working" to "expensive."

**Precondition 1: a verifier you trust.** Can a program decide whether a trajectory succeeded, at ≥99% agreement with a careful human, with a flake rate under ~1%? If the answer is "we'd have an LLM judge it," the project is at high risk of optimizing the judge. If the answer is "our users tell us," the signal is too delayed and too sparse.

**Precondition 2: a prompt pool of the right size and difficulty.** You need thousands to tens of thousands of task instances with gold artifacts, and the current model must succeed on them between roughly 20% and 80% of the time. Below 20% you get no signal; above 80% you get no gradient. **If your agent already succeeds 90% of the time on everything you have, RL has nothing to grip.** This one kills more projects than any other and it is checkable in a day.

**Precondition 3: a held-out eval sensitive enough to detect the gain you expect.** If the improvement you are hoping for is 5 points and your eval has ±4 points of noise at n=200, you cannot tell whether it worked. Size the eval first: at a 60% base rate, the standard error at n=200 is `√(0.6×0.4/200) = 3.5%`, so a 95% CI is ±6.8 points. To resolve a 5-point gain you need roughly n=800 (SE 1.7%, CI ±3.4). **Build the eval before the trainer. Always.**

**Precondition 4: you have exhausted the cheap ladder.** Better prompting, better context assembly, better tool design, better retrieval, structured outputs, a stronger base model, few-shot examples, rejection-sampling SFT. Each of these is days rather than months and each has a higher expected value per engineer-week. RL is the last rung, and I have never regretted making a team climb the ladder first.

**When the answer is genuinely yes:** you have a narrow, high-volume, verifiable task; you have hit the ceiling of prompting and SFT; a 3–5 point gain is worth six figures of value; and you can afford one engineer for a quarter plus the compute. The classic profile is a code-agent company where tests are the verifier and every customer repo is a task generator, or a structured-extraction company with gold-labelled documents at volume.

**💰 Math — the honest total.** One engineer for 12 weeks at a fully-loaded $350k/yr ≈ **$81k**. Compute: 400 steps × $102/step for agentic rollouts at G=8 ≈ **$41k**, plus two or three failed runs before the successful one — realistically **$100–120k of compute**. Eval infrastructure and data curation: 4 engineer-weeks ≈ **$27k**. Serving your own weights instead of an API endpoint: a dedicated 8×H100 deployment at roughly $16/hr is **$11.5k/month** ongoing, versus whatever your API bill was. **Total first-run cost: $200–230k and one quarter.** That is the number to put in front of the founder, and then ask what the 3–5 point gain is worth. If the answer is not clearly seven figures of annual value, the answer is no.

**🗣 Say this in the room:** "I'd ask four questions: do you have a program that can grade a trajectory at 99% agreement with a human, do you have thousands of tasks where the current model succeeds 20–80% of the time, do you have an eval powered to detect the gain you're hoping for, and have you exhausted prompting, tool design, and rejection-sampling SFT? If any answer is no, RL is not the next move. If all four are yes, it's roughly $200k and a quarter, and I'd want the expected value to be clearly seven figures."

### Give me the full cost model for a 7B RLVR run. Show the arithmetic.

Assumptions stated up front, because the discipline of stating them is half the answer: 7B policy, 500 GRPO steps, 512 prompts/step, G=8, mean output 3,000 tokens, 8×H100 at $2/GPU-hr = $16/hr for the node. **📅 Volatile:** GPU rental prices move; redo with current rates.

**Tokens generated.**
`512 × 8 × 3,000 = 12,288,000` tokens per step. Over 500 steps: `6.14 × 10^9` tokens — 6.1 billion tokens of generation for one run. Worth pausing on: that is more tokens than most companies serve to customers in a quarter.

**Generation time and cost.**
At ~10,000 output tok/s aggregate on the node with continuous batching: `12,288,000 / 10,000 = 1,229 s = 20.5 min/step`. Cost: `20.5/60 × $16 = $5.47/step`.

**Training-side time and cost.**
Logprob recomputation (policy + reference over 12.3M tokens) plus backward: call it 40% of generation time ≈ 8.2 min/step → `8.2/60 × $16 = $2.19/step`.

**Verification.**
4,096 executions × 1 CPU-s = 1.14 CPU-hr/step at $0.04/vCPU-hr = **$0.046/step**. Negligible; hidden behind generation if streamed.

**Per-step total:** `$5.47 + $2.19 + $0.05 = $7.71`. Wall-clock ≈ 29 min/step.

**Per-run total:** `500 × $7.71 = $3,855`, wall clock `500 × 29 min = 242 hours ≈ 10.1 days`.

**Now the honest multipliers, which is where candidates lose the question:**

- **Failed runs.** Nobody's first RL run works. Budget 3–5 runs. → `$15,400–19,300`.
- **Hyperparameter search.** Even a small sweep over learning rate, G, and ε is 6+ runs, often shortened to 150 steps each: `6 × 150 × $7.71 = $6,940`.
- **Difficulty pre-scoring.** 30k prompts × 16 samples × 3k tokens = 1.44B tokens; at 10k tok/s that is 144,000 s = 40 node-hours ≈ **$640**.
- **Evaluation.** Every 50 steps on a 500-problem eval at 16 samples: `10 evals × 500 × 16 × 3,000 = 240M tokens` = 24,000 s ≈ 6.7 node-hours ≈ **$107** per run. Cheap; run it more often than you think.
- **Engineering.** 8–12 weeks of one engineer at $350k/yr fully loaded = **$54k–81k**.

**Grand total: roughly $25k–30k of compute and $55k–80k of engineering for one successful 7B reasoning run — call it $85k–110k.** And 10 days of wall-clock per attempt means your iteration loop is measured in weeks, which is the real constraint. **The thing to say last:** at that price, the comparison is not "RL vs no RL," it is "RL vs buying 30 billion tokens of a frontier API," which at $3/Mtok input is $90k. Frame the decision that way and it becomes a business question with a defensible answer instead of a technical preference.

### Correct the 2026 misconception for me: does RLVR create capability or sharpen existing capability?

This is the question I would use to separate someone who tracks the field from someone who read a blog post in 2025, and the correct answer is "mostly sharpens, the boundary is contested, and here is the evidence on both sides."

**The sharpening case.** Pass@k analyses show RLVR models beating their base models at k=1 by large margins and matching or *underperforming* them at large k — the curves cross. If RL had created new capability, the trained model should reach solutions the base model cannot reach at any sample budget; instead the base model, given enough shots, often covers a *wider* solution set. Add the entropy story: RL monotonically narrows the output distribution, which is precisely a description of sharpening. Add the "spurious rewards" observations from 2025 — that on certain base models (notably Qwen math variants) even random or incorrect reward signals produced benchmark gains, which is only explicable if the RL is eliciting a latent format or behavior the base already had rather than teaching anything. **📄 Paper:** Yue et al. (2025) is the standard citation for the pass@k crossover argument.

**The creation case.** Work on prolonged RL with explicit entropy management (NVIDIA's ProRL line, 2025) reports that with long enough training, sufficient exploration pressure, and diverse enough tasks, models solve problems the base model fails at any k — i.e. the boundary does move. And the R1-Zero result itself is hard to square with pure sharpening: the *length* of reasoning traces grew several-fold and the trace structure changed qualitatively, which is a bigger behavioral change than reweighting a few modes.

**The synthesis I would give, and it is defensible either way:**

Most short RLVR runs — a few hundred steps, one domain, tight KL, standard settings — are doing elicitation and sharpening. That is the regime almost every practitioner is in, and it explains almost everything they observe. Whether sufficiently long, sufficiently diverse, sufficiently exploration-preserving RL can move the capability boundary is genuinely open, and the fact that it *requires* long runs and explicit entropy control is itself evidence that the default regime does not.

**Why it matters practically, which is the part that gets you hired:**

- If it is sharpening, the **base model is your ceiling**, and "which base" is a higher-leverage decision than any RL hyperparameter.
- If it is sharpening, you should expect **pass@k degradation**, so measure it — and if your product does best-of-n or agentic retry at inference, RL may make the product worse even as pass@1 improves.
- If it is sharpening, **distillation from a stronger teacher will usually beat RL on your own model**, which is exactly what the R1 paper found for small models and is the cheaper path for almost every applied team.

**🗣 Say this in the room:** "The evidence mostly supports sharpening rather than creation: pass@k curves cross, so the base model often covers more solution space at large k, and entropy falls monotonically, which is what sharpening looks like mechanically. The counter-evidence is prolonged-RL work with explicit entropy control reporting boundary expansion, so I'd call it contested rather than settled. Practically I act on the sharpening reading — I treat base-model choice as the dominant decision, I always measure pass@k alongside pass@1, and I try distillation before RL."

### Give me the drills for this material. What would you practice, timed, before an RLVR interview?

Five drills, ordered by how likely each is to appear in a real loop. Do them unaided — no autocomplete, no reference — because Anthropic, DeepMind, xAI and several others ban AI tools in live rounds.

**🏋 Drill 1 — GRPO from memory. 25 minutes.** On paper, write the GRPO objective including the importance ratio, the clip, the group-normalized advantage, and the KL term with its estimator. Then implement `group_advantages` and `grpo_loss` in PyTorch. *Pass criterion:* the objective is correct including both normalization denominators; your code produces exactly zero advantage for a constant-reward group; you can state the tensor shape at every line without checking; and you can explain in one sentence why `ρ = 1` on the first inner step.

**🏋 Drill 2 — the degenerate-group arithmetic. 5 minutes, no calculator beyond mental math.** Given G=8 and per-prompt pass rates of 0.5, 0.9 and 0.97, state the fraction of groups producing zero gradient. *Pass criterion:* you write down `p^G + (1−p)^G` without prompting, you land near 0.8%, 43% and 78% respectively, and you can immediately name the three fixes (dynamic sampling, difficulty filtering and retirement, adaptive G on the hard tail).

**🏋 Drill 3 — design the verifier. 20 minutes, whiteboard, spoken aloud.** Given "we want to RL a model on our internal SQL generation task," design the verifier end to end: execution sandbox, correctness definition, normalization, tolerance, caching, throughput, and the reward-hacking detection kit. *Pass criterion:* you name at least four distinct hacking vectors specific to SQL (returning the expected result by hardcoding it, exploiting a result-set comparison that ignores ordering, querying the answer table directly, exploiting a loose row-count-only check), and you specify the trivial-baseline probe.

**🏋 Drill 4 — the R1 pipeline, spoken, 90 seconds.** State the four stages in order with the purpose of each, plus what R1-Zero showed and what broke. *Pass criterion:* you get SFT → RL → SFT → RL in order, you name the ~800k rejection-sampling corpus and the ~600k/200k split, you name readability and language mixing as R1-Zero's failures, and you say why stage 3 exists (restoring general capability after reasoning RL degraded it) — because that last point is what most candidates miss.

**🏋 Drill 5 — the incident. 15 minutes, spoken.** "Reward is climbing, held-out eval is flat, entropy is at 8% of its starting value, and 55% of groups are degenerate. Diagnose and give me your next three actions." *Pass criterion:* you check the eval harness against the base checkpoint first, you read high-reward samples by hand, you identify that reward-on-training-batch is conditioned on a shifting sample and is not comparable across steps, and your three actions include re-scoring prompt difficulty, enabling dynamic sampling, and rolling back to an earlier checkpoint with entropy headroom rather than continuing.

**🏋 Drill 6 — the cost defense. 10 minutes, spoken.** A founder asks whether to RL your agent. Deliver the four preconditions and the full cost model with arithmetic, from memory. *Pass criterion:* you state the 20–80% success-band precondition, you size the eval with a standard-error calculation, and your total lands within a factor of two of $200k for an agentic run — with the digits shown, not asserted.
