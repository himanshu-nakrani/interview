### Give me your default hyperparameters for an SFT run and defend every one of them.

I want to be able to say these from memory, because an interviewer asking this is checking whether you have actually run one. The defaults below are what I start from and then justify moving.

**Epochs: 2, range 1–3.** SFT datasets are small and high-signal; you are not trying to fit them, you are trying to nudge a prior. One epoch is right when the dataset is large (>100k) or heavily synthetic; three when it's small (<5k) and hand-curated. Above three epochs you are memorizing — the model starts reproducing training answers verbatim on adjacent prompts and its diversity collapses. LIMA-style runs go higher because 1,000 examples at 3 epochs is only 3,000 gradient contributions.

**Learning rate: 1e-5 to 2e-5 for full fine-tuning, 1e-4 to 2e-4 for LoRA.** The order-of-magnitude gap is not folklore, it's structural — see the next question.

**Schedule: cosine decay to ~10% of peak, warmup ratio 0.03.** Warmup exists because Adam's second-moment estimate `v` starts at zero and is badly calibrated for the first tens of steps, so the effective step size is enormous exactly when your weights are still the well-behaved pretrained ones. Three percent of a 1,000-step run is 30 warmup steps, which is plenty. I use ratio rather than absolute steps so it survives dataset-size changes.

**Effective batch size: 64–256 sequences, or better, 128k–512k tokens per optimizer step.** Expressed in tokens because that's the invariant across bucketing and packing. Achieved via `per_device_batch × grad_accum × world_size`. Small batches on SFT produce noisy gradients that manifest as unstable style — the model's tone wobbles between checkpoints.

**Optimizer: AdamW, β = (0.9, 0.999) — or 0.95 for β₂ on larger models — weight decay 0.0 to 0.1, grad clip 1.0.** I use weight decay 0 for LoRA (the adapter is already a strong constraint) and 0.01–0.1 for full FT. Grad clipping at 1.0 is non-negotiable; a single bad example with a pathological length can otherwise blow up a run.

**Precision: bf16 compute with fp32 master weights and fp32 optimizer state.** Not fp16 — fp16's 5-bit exponent overflows on attention logits and forces loss scaling, and bf16 has the same exponent range as fp32. If your hardware is pre-Ampere and bf16 isn't available, that constrains you, but on H100/A100 there is no argument for fp16 in 2026.

**Max sequence length: the 99th percentile of your data, rounded up.** Not 4,096-because-it's-round. Every unnecessary token in `max_seq_length` costs memory quadratically in attention activations if you're not using packing.

**📐 Numbers you must know:** the full-FT memory formula for AdamW in mixed precision, per parameter: 2 bytes bf16 weight + 2 bytes bf16 grad + 4 bytes fp32 master + 4 bytes Adam `m` + 4 bytes Adam `v` = **16 bytes/param.** For an 8B model that's 128 GB before a single activation — which is why full FT of an 8B does not fit on one 80 GB H100 and needs ZeRO-2 across 2+ GPUs or ZeRO-3/offload. For a 70B it's 1.12 TB, i.e. 16× H100 minimum with sharding. Derive it in the room; don't recite it.

**⚠ Trap:** copying a learning rate from a blog post that used a different effective batch size. LR and batch size are coupled; the commonly-used heuristic is that LR should scale roughly with √(batch) for Adam. If a config says 2e-5 at effective batch 8 and you run effective batch 256, you have changed the effective step size by ~5.7× without meaning to. Always report LR *with* effective batch size, or the number is meaningless.

### Why is the LoRA learning rate an order of magnitude higher than the full fine-tuning learning rate?

Because they are learning rates on different objects, and the LoRA update is deliberately attenuated at initialization.

Recall the parameterization: for a frozen weight `W ∈ R^{d×k}`, LoRA learns `ΔW = (α/r)·B·A` where `A ∈ R^{r×k}` is initialized from a small random distribution and `B ∈ R^{d×r}` is initialized to **zero**. At step 0, `ΔW = 0` exactly, which is why you can attach an adapter without changing the model's outputs. The scaling factor `α/r` is applied on top.

Three things make the required LR larger:

1. **B starts at zero, so gradients through B are proportional to A's magnitude and gradients through A are proportional to B's — which is zero.** The product structure means the update starts in a slow, near-degenerate regime. It needs a larger step to escape.
2. **Far fewer parameters carry the entire behavioral change.** A full fine-tune spreads a behavior across 8 billion parameters, so each moves microscopically. Work it out for Llama-3-8B at r=16 on all seven linear projections: per layer, `r × (fan_in + fan_out)` summed gives roughly 1.3M parameters, times 32 layers ≈ **42M trainable — about 0.5% of the model.** The same *functional* change must be expressed by ~190× fewer numbers, so each must move proportionally further.
3. **The `α/r` scaling shrinks the effective update.** With the common `α = 2r` convention the factor is 2; with `α = r` it's 1; with `α = 16, r = 64` it's 0.25, which means at r=64 you are quietly *reducing* your effective LR by 4× relative to r=16 unless you also raise α. This is why "I increased rank and it got worse" is such a common report.

The practical consequences I enforce: **when you change `r`, change `α` proportionally** (keep `α/r` fixed) so the LR you tuned stays valid; use `r = 16` and `α = 32` as a starting point for behavioral SFT, `r = 64` or higher only when you're teaching something closer to a new capability; and **target all linear layers, not just q and v** — attaching to `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` consistently outperforms attention-only in my experience, and matches the finding that LoRA's gap to full FT narrows substantially when the MLP is included.

**📄 Paper:** Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models" — replaced adapter layers (which add inference latency) and prefix tuning (which consumes context) with a reparameterization that can be merged into `W` at deploy time for literally zero inference overhead. That merge property is the reason it won.

**⚠ Trap:** rsLoRA. The standard `α/r` scaling makes higher ranks *underperform* because the effective update shrinks as `1/r`. Rank-stabilized LoRA rescales by `α/√r` instead, and it's the reason rank sweeps sometimes look flat or inverted. If you sweep rank and see no improvement, check whether your library is applying the `1/r` or `1/√r` scaling before concluding "rank doesn't matter."

### Eval loss went down every epoch, but our human reviewers say the model got worse. What's going on?

This is the single most important diagnostic conversation in SFT and I want to give the full taxonomy, because "eval loss and eval quality disagree" has at least five distinct causes and they demand different fixes.

**Cause 1: loss is measuring style-matching, not correctness.** Cross-entropy on held-out completions rewards producing *these particular tokens*. A model that learns your dataset's phrasing tics — "Certainly! Here's a breakdown:" — scores better on token-level loss while being no more correct. If your reference completions came from one generator (say, one strong model), loss is substantially a measure of "how well do you imitate that generator's surface style." This is the most common cause and the fix is that loss is a training diagnostic, never a model-selection criterion.

**Cause 2: capability regression outside the eval distribution.** Your eval set is drawn from the same distribution as training. Catastrophic forgetting damages everything *else*, and your held-out loss is structurally blind to it. Fix: a general-capability suite run at every checkpoint, covered later in this section.

**Cause 3: diversity collapse.** By epoch 3 the model's outputs have low entropy — it produces one canonical answer shape. Token-level loss on a held-out set can improve while the model becomes unusable for anything requiring variation. Measure it: sample 8 completions at temperature 1.0 for 100 prompts and compute distinct-n or pairwise embedding similarity across the samples. If mean pairwise cosine similarity between samples for the same prompt climbs from 0.62 to 0.91 across checkpoints, you have collapsed.

**Cause 4: the eval set is contaminated or trivial.** If eval prompts leaked into training, loss drops for the wrong reason. If eval completions are short and formulaic (a lot of "Yes."/"No."), loss is dominated by easy tokens.

**Cause 5: the disagreement is real and your reviewers are right about something loss cannot see** — refusal rate, hedging, verbosity, tool-call validity, a subtle factual regression. Loss aggregates over tokens; humans evaluate at the response level. A response that is 98% token-identical to the reference and wrong in one number scores brilliantly on loss and fails completely on utility.

**🔍 Failure taxonomy — the decision procedure.** Given loss↓ / quality↓: (a) run the general-capability suite on the same checkpoints — if *it* also dropped, it's forgetting, mix in replay; (b) measure output diversity across checkpoints — if it collapsed, reduce epochs or LR; (c) run contamination detection between train and eval — if positive, rebuild the eval set; (d) diff 30 outputs from the best-loss checkpoint against the worst-loss checkpoint, by hand, and characterize the difference in one sentence — if you can't, your eval set is too small to be informative; (e) if none of these fire, your loss is measuring style and you should stop looking at it.

**🗣 Say this in the room:** "Held-out cross-entropy measures token-level agreement with one reference completion, which is a proxy for style much more than for correctness. I use it to detect divergence and to sanity-check that training is happening, and I select checkpoints on task-level evals — win rate against the prompted baseline, a capability-regression suite, and format/tool-call validity — never on loss."

### So how do you detect overfitting in practice? What's on your dashboard?

Six signals, and I look at them in this order.

**1. Train/eval loss gap widening.** The classic. Train loss falling while eval loss flattens or rises is textbook overfitting, and for SFT it usually appears between epoch 2 and 3. It is necessary but not sufficient evidence — you can overfit behaviorally without the curves separating much, because SFT losses are all quite close together.

**2. Verbatim regurgitation rate.** Take 200 *training* prompts, generate greedily, and compute the fraction where the generated answer matches the training answer above some n-gram overlap threshold (I use ROUGE-L ≥ 0.9). At epoch 1 this should be low; if it's above ~40% you have memorized. This is a much sharper instrument than loss and it costs one generation pass.

**3. Output entropy / diversity across samples.** As above — 8 samples per prompt at T=1.0, pairwise similarity. Rising similarity across checkpoints is mode collapse.

**4. Length drift.** Mean output length per checkpoint. SFT on data with long answers pushes length up monotonically; if your mean response goes 180 → 340 tokens across three epochs and quality didn't improve, you're paying 1.9× on output tokens for nothing. At $15/Mtok output and 200k calls/day, 160 extra tokens/call is 200,000 × 160 × $15/1e6 = **$480/day = $14.4k/month** of pure verbosity tax. **📅 Volatile:** verify output pricing.

**5. Format validity on a held-out structured task.** If your model emits JSON or tool calls, parse-success rate is a hard, cheap, non-subjective metric. Overfitting frequently shows up here first as the model starts emitting your training set's exact field ordering even when the schema differs.

**6. The general-capability suite.** MMLU-style knowledge, GSM8K-style reasoning, HumanEval-style code, IFEval-style instruction following — a fixed battery, run at every checkpoint, plotted on the same axes as your task metric. This is the forgetting detector.

The checkpoint policy that follows: **save every half-epoch, evaluate all of them, and pick on the task metric with a hard floor on the capability suite.** "Best eval loss" as a checkpoint selection rule is how you ship a memorized model. I write the selection rule down before the run starts — e.g. "maximize win rate vs. the prompted baseline subject to ≤2 point drop on the capability suite" — so the choice isn't made retroactively by whoever is looking at the dashboard.

**⚠ Trap:** running your task eval only at the end. Three epochs of an 8B run is a couple of GPU-hours; the eval sweep across 6 checkpoints might be more expensive than the training. Budget for it explicitly, and make your eval harness batched and cached, or you will skip it under time pressure and select on loss — which is exactly the failure this whole question exists to prevent.

### How many examples do you actually need? Sell me on LIMA, then tell me where it's wrong.

**📄 Paper:** Zhou et al. (2023), "LIMA: Less Is More for Alignment" (Meta AI). They fine-tuned LLaMA-65B on **1,000 carefully curated prompt–response pairs** with no RLHF at all, and in human preference comparisons it was competitive with — and in some pairings preferred over — models trained with vastly more data and full RLHF pipelines. The framing they introduced is the **Superficial Alignment Hypothesis**: essentially all of a model's knowledge and capability is acquired in pretraining, and alignment is mostly teaching it which *sub-distribution of formats and styles* to use when interacting with users. If that's true, you need very little data — just enough to specify the format — and quality dominates quantity.

That hypothesis is directionally right and it is the single most useful prior for an applied engineer, because it changes your budget allocation completely. **A thousand excellent examples that a domain expert wrote and a second expert reviewed will beat a hundred thousand scraped ones.** I would rather spend $30k on 2,000 expert-written examples than $30k on 200k crowd-sourced ones, and I have seen that trade play out in favor of the small set more than once.

Now the limits, because an interviewer who knows the paper is waiting for them:

1. **It was evaluated on single-turn, open-ended, human-preference-judged prompts.** Human preference on freeform responses rewards fluency, structure and confidence — precisely what 1,000 well-formatted examples install. It says almost nothing about correctness on verifiable tasks. Nobody has shown 1,000 examples getting you competitive math or code.
2. **It started from a very strong base model (65B).** The Superficial Alignment Hypothesis is a claim about what pretraining already contains. On a 1B model, or on a domain the base model genuinely never saw, "just format it" is not enough. The smaller and more out-of-distribution your target, the more data you need.
3. **They explicitly acknowledged multi-turn weakness** and had to add a small set of multi-turn dialogues to make the model usable conversationally. Multi-turn coherence, tool use and long-horizon behavior are not free.
4. **No robustness, no adversarial behavior, no safety at scale.** Refusal boundaries, jailbreak resistance and abstention are behaviors with long tails; 1,000 examples cannot cover a long tail.
5. **Reproducibility is data-dependent in a way that doesn't transfer.** Their curation was heavy manual work by researchers with taste. "1k examples" is not the finding; "1k examples *curated like this*" is, and that part doesn't fit in a config file.

**🗣 Say this in the room:** "LIMA's real contribution is the Superficial Alignment Hypothesis — that capability comes from pretraining and alignment mostly selects a response distribution. So for style, format and tone I budget for hundreds to low thousands of expert examples and spend the money on curation. For anything verifiable — tool-call correctness, structured extraction, math — the data volume is set by coverage of the failure modes, not by that result, and I'd expect tens of thousands."

**📐 Numbers you must know:** my working scale ladder. Format/tone/persona: 500–2,000 examples. A single structured-output schema: 1,000–5,000. Domain-specific reasoning behavior with tool calls: 10,000–50,000. Replacing a large system prompt on a narrow task: 3,000–10,000. New capability the base model lacks: you are in continued-pretraining territory, and the number starts at hundreds of millions of tokens.

### We want the fine-tuned model to call tools. How do you format tool calls in the training data?

The governing rule is one sentence: **the training format must be byte-identical to the serving format, produced by the same code path.** Everything else follows from that, and every failure I've seen here is a violation of it.

Concretely, modern instruct models don't emit tool calls as free text — they emit them inside a structured region the template defines, and the inference server parses that region back into a JSON object for your application. Llama-3.1-style models use a dedicated `ipython`/tool role with a JSON payload; ChatML-family models often wrap calls in `<tool_call>...</tool_call>`; Anthropic and OpenAI expose the parsed object over the API and hide the wire format from you entirely. The template is also usually responsible for **rendering the tool schemas into the system prompt** — `apply_chat_template(messages, tools=[...])` is a real parameter on modern tokenizers and it injects the JSON-schema descriptions.

So the data-construction procedure:

1. Represent an example as a full message list including tool messages: `[system, user, assistant(with tool_calls), tool(result), assistant(final)]`.
2. Render it through `apply_chat_template` with the `tools=` list, exactly as production will.
3. Unmask **both** assistant turns — the one that emits the call and the one that answers after seeing the result. Mask the `tool` role message; that's an observation, not a behavior.
4. Include the tool schemas in the render, and **vary them across examples**. If every training example ships the same 5 tools, the model learns those 5 tools rather than learning to read a schema. Sample from a pool of 50+ tools, include distractors, vary the order, vary the number available.

**⚠ Trap:** training on tool calls with only the tools that were actually used present in the schema list. The model learns "if a tool is offered, call it," and its abstention rate collapses — you get tool calls on "hello." Always include examples where 6 tools are offered and the correct behavior is to call none of them, and examples where the right tool is 5th in a list of 8. Position bias in tool selection is real and it is trainable.

**⚠ Trap:** hand-writing tool-call JSON in your training set with keys in a different order or with different whitespace than the serving parser produces. If serving emits `{"name": "x", "arguments": {...}}` and you trained on `{"arguments": {...}, "name": "x"}`, you have taught a format your parser handles fine — until a partial-parse or streaming path in the inference server assumes name-first. Generate your training-set tool calls by *serializing the same objects through the same code* the runtime uses.

**🔍 Failure taxonomy — tool-calling fine-tunes.** (1) Parse failure rate > 0: template mismatch or hand-written JSON. (2) Correct tool, wrong arguments: not enough schema variety; the model is pattern-matching on the tool name. (3) Calls a tool that doesn't exist in the offered list: you never trained the negative case. (4) Emits the call as prose ("I'll use the search tool") instead of a structured call: your assistant turns weren't unmasked at the right span, or the special tokens for the tool region weren't in the vocab. (5) Never stops after the tool result: missing stop token on the final assistant turn.

**💰 Math:** why a tool-calling fine-tune can pay for itself. Suppose you offer 40 tools with schemas averaging 180 tokens — that's 7,200 tokens of system prompt on every call. At $3/Mtok input and 500k calls/day: 500,000 × 7,200 × $3/1e6 = **$10,800/day** uncached, ~$1,080/day with a 90% prefix-cache discount. A fine-tune that internalizes the 12 hot tools and only ships schemas for the long tail cuts that prompt to ~1,500 tokens: $2,250/day uncached, $225/day cached. The saving is real but it is entirely swallowed if the fine-tune breaks tool-call validity — a 2% parse-failure rate on 500k calls is 10,000 failed requests/day. Ship the eval before the model.

### What about reasoning or "thinking" traces? Should those be in the SFT data?

Yes, if and only if the serving format has a place to put them, and this is a place where getting the format wrong is unusually expensive.

Models trained to produce an internal reasoning region emit it inside delimiters — `<think>...</think>` is the widely-adopted convention in open models — and the inference stack is expected to strip that region before showing the user, while the API may or may not return it. The essential structural facts:

1. **The thinking region is part of the assistant turn and must be unmasked in training.** If you mask it and train only on the final answer, you're teaching the model to jump straight to a conclusion — the exact opposite of what you want.
2. **In multi-turn data, prior turns' thinking is usually *stripped* from the context.** This is the detail people miss. Most reasoning-model serving conventions do not carry previous thinking blocks forward into subsequent turns — the context contains prior answers but not prior scratchpads. Your training data must match: for a 3-turn conversation, turn 3's context should contain turns 1 and 2's *answers only*, while turn 3's own thinking is unmasked and trained. If you train with all prior thinking present but serve with it stripped, you have a context-distribution mismatch on every turn after the first.
3. **The delimiters must be real tokens in the vocabulary** if the base model was trained that way, or plain text if it wasn't. Adding `<think>` as a multi-token string to a model with no such token is workable but weaker.

Where the traces come from matters as much as the format. Three sources, in descending order of how much I trust them:

- **Rejection-sampled traces from the model itself, filtered by outcome correctness.** Sample k=8 traces per problem, keep the ones whose final answer is verifiably right, dedupe, train. This keeps the traces on-policy — they're in the model's own idiom — which is why it works better than importing someone else's reasoning style.
- **Traces from a stronger model, filtered by correctness.** Works, but check the licence and terms of service on the source model's outputs before you build a business on it; several providers explicitly prohibit using outputs to train competing models.
- **Human-written reasoning.** Expensive, and humans write reasoning post-hoc and tidily, which is not how the model reasons. I use these for the *format* and the rejection-sampled ones for the volume.

**⚠ Trap:** training on traces where the reasoning is wrong but the final answer is right. Outcome filtering alone lets these through, and they teach the model that sloppy reasoning is rewarded. For math and code you can partially catch this by checking intermediate steps; for general reasoning you need a process judge or human spot-checks. This is the SFT-side shadow of the reward-hacking problem.

**📐 Numbers you must know:** thinking tokens are output tokens and they are billed as such. A model that emits 900 thinking tokens plus a 150-token answer costs 7× the answer alone. At $15/Mtok output, that's 1,050 × $15/1e6 = **$0.0158 per call** vs $0.00225 for the answer alone. Over 100k calls/day: $1,575/day vs $225/day — **$40k/month** for the reasoning. That is often worth it, and it is always worth measuring; a fine-tune that shortens traces from 900 to 400 tokens while holding accuracy is a $20k/month product decision.

### Do you put refusals and other negative examples in the SFT set? How many?

Yes, and the ratio matters more than the count, because refusal training has the sharpest over-correction failure mode in all of SFT.

The mental model: a refusal example teaches a *boundary*, and a boundary learned from one side generalizes badly. If your dataset contains 500 "here is a harmful request → I can't help with that" examples and zero "here is a request that superficially resembles a harmful one but is fine → helpful answer," you have taught the model that certain surface features predict refusal. The model will then refuse "how do I kill a process on port 8080," "what's the best way to attack this optimization problem," and every medical or legal question your product exists to answer. This is not hypothetical; over-refusal is the most common complaint about aggressively safety-tuned models.

So the construction rule: **every refusal category ships with matched near-miss positives, roughly 1:1 to 1:3 refusal:near-miss.** For each refusal cluster, generate adjacent requests that share vocabulary and framing but are legitimate, and label them with helpful answers. This is where synthetic generation earns its keep — take a refused prompt, ask a strong model for "five requests that use similar words but are clearly benign," verify by hand, and you have your near-misses cheaply.

On volume: refusals should be a small fraction of the total set, on the order of **2–5%**. I've seen teams push to 20% because safety felt important, and the resulting model refused everything with a paragraph of moralizing. Refusal is a high-salience behavior; it does not need volume parity.

On *form*: train the refusal you actually want. A refusal that says "I can't help with that." is worse product than one that names the boundary and offers the adjacent legitimate path — "I can't provide instructions for that, but if you're trying to test your own system's resilience, here's how to set up a permission audit." Whatever form you want, it must be in the data, consistently, because the model will imitate the form far more reliably than the judgment.

Beyond safety refusals, the other negative classes I always include: **out-of-scope** ("that's outside what this assistant covers"), **malformed input** (the user pasted half a JSON blob), **insufficient permission** (the user asked for data their role can't see — the model should decline and say why, not fabricate), and **the tool isn't available** case discussed earlier.

**🗣 Say this in the room:** "Refusals are 2–5% of the set and every refusal category ships with matched near-miss positives at roughly 1:1, because a boundary trained from one side generalizes into over-refusal. And I hold out an over-refusal eval — a set of benign prompts that look adversarial — as a hard gate on the checkpoint, since it's the metric that regresses invisibly."

### How do you teach a model to say "I don't have enough information"? That's the thing our RAG system is worst at.

Abstention is a capability, and the reason models are bad at it is that essentially nothing in pretraining or in typical SFT data rewards it. Web text does not contain many examples of someone being asked a question and correctly saying "the provided documents don't answer this." The training distribution is overwhelmingly "question → confident answer," so that's the mapping the model learned. You cannot prompt your way out of a strong learned prior reliably; you can fine-tune your way out of it.

The data construction is the whole answer, and it's more subtle than "add some IDK examples."

**Construct abstention examples by ablation, not by invention.** Take a real (question, gold-documents, gold-answer) triple from your RAG corpus. Now *remove* the document that contained the answer and replace it with topically-similar but non-answering documents retrieved from the same corpus. The correct label for that example is an abstention that names what's missing: "The provided documents cover the 2024 policy but don't state the 2025 rate." This is critical — the abstention has to be *grounded*, not a generic "I don't know," because a generic one is indistinguishable to the model from laziness and it will over-trigger.

**Build four buckets and balance them:**
1. Answer fully present in context → answer, with citation.
2. Answer partially present → answer the covered part, explicitly flag the uncovered part. This is the hardest and most valuable bucket, and most teams skip it.
3. Answer absent, topically similar distractors present → grounded abstention.
4. Answer absent, retrieval returned nothing relevant at all → abstention plus a suggestion to rephrase or a hand-off.

My working ratio is roughly 50/20/20/10. Bucket 2 being underrepresented is why models answer half a question confidently.

**Include contradictory-context examples.** Two retrieved documents disagree. The correct behavior is to surface the conflict, not to silently pick one. Nobody trains this and every production RAG system encounters it.

**The eval must come first, and it must be two-sided.** Measure both false-answer rate (answered when it should have abstained — the dangerous direction) and false-abstention rate (abstained when the answer was right there — the annoying direction, and the one that makes users abandon the product). A model that abstains on 30% of answerable questions is worse than the one you started with. Report both, always, and set the operating point deliberately: for a legal or medical product I'll take 15% false abstention to get false-answer rate under 1%; for a general search product that trade is unacceptable.

**⚠ Trap:** generating abstention examples by asking a strong model "write a question this document doesn't answer." The questions come out unnaturally disjoint from the document, the model learns "topically unrelated → abstain," and it fails completely on the actual production case, which is a question that is *90% covered* by the context. The ablation method above is the fix — it guarantees your negatives are hard.

**💰 Math:** the value of abstention is asymmetric and you should quantify it. At Harvey-or-similar stakes, one confidently wrong cited answer that reaches a client is a remediation incident: partner review time, a client call, possibly a disclosure. Put a conservative $5,000 on it. If your baseline false-answer rate is 8% on the 12% of queries that are unanswerable, at 50,000 queries/month that's 50,000 × 0.12 × 0.08 = 480 wrong-answer events/month. Cutting that to 1% saves 420 events. Even at a 1-in-100 chance any given one escalates, that's 4.2 incidents/month × $5,000 = **$21,000/month** of avoided cost against a fine-tune that cost single-digit thousands. That is the business case, and it is the version of this answer that gets you the offer.

### Compare the fine-tuning stacks — TRL, Axolotl, LLaMA-Factory, Unsloth, torchtune. Which do you reach for?

The honest framing: these are not five competing implementations of the same thing, they're four config layers over roughly two or three real training engines plus one kernel-optimization play. Knowing which layer you're in tells you where a bug can live.

**TRL (HuggingFace).** The library, not a wrapper. `SFTTrainer` subclasses HF `Trainer`, so you write Python and you get the full `TrainingArguments`/`SFTConfig` surface, DeepSpeed and FSDP integration, and PEFT for LoRA. This is what I reach for when the job is non-standard — custom loss masking, a bespoke collator, curriculum ordering, a metric callback that runs my task eval mid-training. It's also the reference implementation everyone else follows, so reading its source is how you resolve arguments about what a flag actually does. Cost: you write more code and you own more of the correctness.

**Axolotl.** A YAML front-end over the HF stack with sensible defaults, dataset-format adapters, and packing/varlen wired up. This is my default for a *standard* run because the config file is the artifact — it's reviewable, diffable, and reproducible, which matters far more than people admit. When someone asks "what config did you run," a 60-line YAML in the repo is a much better answer than "let me find the notebook."

**LLaMA-Factory.** Similar positioning to Axolotl — YAML plus a web UI, very broad model-family coverage, and it bundles SFT/DPO/PPO/reward modeling in one place. Strong choice if you're evaluating many base models quickly or if non-engineers need to launch runs.

**Unsloth.** A kernel and memory-optimization layer: hand-written Triton kernels and manual backward passes for the common architectures, giving substantially faster training and substantially lower memory for single-GPU LoRA/QLoRA. This is what you use when the constraint is "one A100/4090 and a deadline." **📅 Volatile:** the specific speed and memory-reduction multipliers they advertise change per release and per model — verify on your hardware rather than quoting a number in an interview. The durable point is *why* it's faster: fused kernels and avoiding materialization, not a different algorithm. Multi-GPU support has historically been the limiting factor; check current status.

**torchtune (PyTorch).** Native PyTorch, minimal dependencies, YAML recipes, first-class FSDP and distributed support, designed to be *readable* — the recipes are single files you're expected to fork. This is the right choice inside a big-tech environment where "depends on the HuggingFace ecosystem" is a procurement conversation, and it's the one I'd pick if I needed to modify the training loop itself rather than configure it.

**My decision rule:** single GPU, LoRA, moving fast → Unsloth. Standard multi-GPU SFT that I want reproducible → Axolotl. Anything with custom loss, custom data logic, or mid-training evals → TRL directly. Need to own and modify the loop, or in a PyTorch-native shop → torchtune. And regardless of choice, **the correctness checks from the pre-flight list are mine, not the library's** — none of these tools will tell you your labels are wrong.

**⚠ Trap:** believing the config. Every one of these has had at least one release where a flag was accepted, silently ignored, or changed meaning. `packing`, `neftune_noise_alpha`, `max_seq_length` semantics (truncate vs. filter vs. pack), and whether completion-only masking is on by default are the four I always verify empirically on a new version, via the label dump, before spending money.

### Walk me through an actual run. Hardware, hours, dollars, and the eval delta.

This is the question the whole section exists for — the "Mastering this proves..." line for this material is precisely that you have shipped one rather than read about one. Here's the shape of the answer, with the arithmetic you should be able to reconstruct live. Treat the specific scenario as a template to fill with your own run.

**The task.** Replace a 9,000-token system prompt on a support-triage assistant that classifies an incoming ticket, extracts five structured fields, and either answers from retrieved docs or escalates. Baseline is a frontier model with the long prompt. Target is an 8B open-weight model, self-hosted.

**The data.** 14,000 examples: 9,000 from consented production logs where the frontier model's output was accepted by the human agent without edit, 3,000 where the agent edited (kept the *edited* version as the label), and 2,000 hand-built negatives — escalation cases, insufficient-context abstentions, and near-miss non-refusals. Average rendered length 1,450 tokens, 41% label density.

**The config.** LoRA r=32, α=64, dropout 0.05, on all linear layers. LR 2e-4, cosine to 10%, warmup ratio 0.03, 2 epochs, effective batch 128 sequences via 4 per device × 8 grad-accum × 4 GPUs, bf16, grad clip 1.0, max_seq_len 2,560 (99.5th percentile), packing on with varlen attention verified by the leak test.

**💰 The compute math.** 14,000 × 1,450 = 20.3M tokens per epoch, 40.6M for two. LoRA forward+backward is roughly 4·N·D FLOPs rather than the 6·N·D of full FT (you skip the weight-gradient GEMM for frozen weights): 4 × 8e9 × 4.06e7 = **1.30e18 FLOPs.** Four H100s at ~990 TFLOP/s peak bf16, 40% MFU → 4 × 3.96e14 = 1.58e15 FLOP/s. So 1.30e18 / 1.58e15 = **823 seconds ≈ 14 minutes of pure compute.** Reality with dataloading, checkpointing and eval callbacks: about 45 minutes wall clock. At $2.50/GPU-hour × 4 GPUs × 0.75h = **$7.50 of compute.** **📅 Volatile:** GPU rates.

That number is the point. **The compute is never the expensive part of an SFT program.** The 14,000 examples cost weeks of pipeline work and, if any were human-written, five figures. The eval harness cost more engineer-time than the training. Anyone who quotes you a fine-tuning project cost dominated by GPU spend has not built one.

**The eval delta, which is what you actually report.** Against the prompted-frontier baseline on a held-out 400-example set: classification accuracy 91.2% → 92.8%, field-extraction exact match 84% → 89%, escalation F1 0.71 → 0.79, false-answer rate on unanswerable tickets 9% → 2%, format validity 99.1% → 99.9%. General-capability suite: −1.4 points MMLU-style, −0.8 IFEval, within the ≤2-point gate set before the run.

**💰 The serving math, which is the actual business case.** Baseline: 9,000-token prompt + 300 output at frontier pricing, say $3/Mtok in and $15/Mtok out with 90% caching on the prompt: (9,000 × $3/1e6 × 0.1) + (300 × $15/1e6) = $0.0027 + $0.0045 = **$0.0072/call.** At 400k calls/day = $2,880/day = **$86.4k/month.** Self-hosted 8B with a 600-token prompt: an H100 serving an 8B at reasonable batch achieves on the order of thousands of output tokens/sec aggregate; at 400k calls/day × 300 output tokens = 120M output tokens/day = **1,389 tokens/sec sustained**, so a handful of H100s with headroom — call it 6 GPUs at $2.50/h = $360/day = **$10.8k/month**, plus an on-call rotation and a deploy pipeline. Roughly an 8× reduction, ~$75k/month, against maybe $150k of one-time engineering. Payback in two months, and after that you own a serving problem you didn't have before. **That last clause is the senior part of the answer** — I would not do this at 40k calls/day, where the saving is $7.5k/month and doesn't cover the on-call.

**🗣 Say this in the room:** "Compute was $8 and 45 minutes on four H100s; the program cost was the data pipeline and the eval harness. The delta that mattered was false-answer rate on unanswerable tickets going 9% to 2%, and the serving economics only worked because we were above roughly 200k calls/day — below that I'd have kept the prompted model and spent the effort on retrieval."

### How do you decide between LoRA and full fine-tuning for a given job?

Four inputs: how much you need to change, how much VRAM you have, how many variants you must serve, and how much you care about forgetting.

**Change magnitude.** LoRA constrains the update to a low-rank subspace per matrix. For behavior, format, tone, refusal boundaries and tool-call structure — all low-dimensional changes — that constraint costs you essentially nothing. For teaching genuinely new capability or a new language, the constraint bites. The empirical picture that matches my experience: on instruction-following-style targets LoRA is close to full FT, and the gap widens as the target task moves further from the pretraining distribution.

**📄 Paper:** Biderman et al. (2024), "LoRA Learns Less and Forgets Less" (Databricks) — a systematic comparison on code and math domains showing LoRA underperforms full fine-tuning on target-domain gains while preserving base-model capability substantially better, and behaving as a stronger regularizer that maintains output diversity. It's the paper to cite when someone frames LoRA as strictly worse: the forgetting side is a real advantage, not a consolation.

**Memory.** From the earlier formula, full FT of an 8B needs ~128 GB of model/optimizer state; LoRA needs the frozen base in bf16 (16 GB) plus adapter weights, grads and optimizer state for ~0.5% of params (well under 1 GB), plus activations. That's the difference between "multi-node with ZeRO-3" and "one 80 GB card." QLoRA — 4-bit NF4 base with bf16 LoRA on top — takes the base to ~5 GB and puts an 8B fine-tune on a 24 GB consumer card, at the cost of some quality and slower steps due to dequantization.

**Serving multiplicity.** This is the argument people underweight. LoRA adapters are ~100 MB; serving engines can hold many adapters against one base model's weights and route per-request. If you need per-customer or per-vertical variants — very common at Ramp, Glean, Sierra, Harvey — LoRA turns 40 fine-tunes into one deployment. Forty full fine-tunes of an 8B is 40 × 16 GB = 640 GB of weights and 40 separate deployments. That's not a quality argument, it's an architecture argument, and it usually decides the question by itself.

**Forgetting.** LoRA's constraint means the base's general capability is largely preserved. If your capability-regression gate is tight and your task is narrow, LoRA gets you there with less replay data.

**My rule:** LoRA by default. Escalate to full FT when (a) you have run LoRA at r=64+ on all linear layers with a tuned LR and it plateaus below your bar, or (b) you're doing continued pretraining on a large corpus, or (c) you're distilling into a small model where you want every parameter working. And I always ask the multiplicity question first, because if the answer is "we'll need one per customer," the decision is made regardless of quality.

**⚠ Trap:** merging LoRA adapters and expecting bit-identical behavior to the unmerged adapter. `merge_and_unload` computes `W + (α/r)BA` in the base weight's dtype; if the base is quantized, the merge is lossy and the merged model can measurably differ from the adapter-on-quantized-base you evaluated. Evaluate the artifact you're going to ship, in the precision you're going to ship it in. This has bitten enough teams that I treat "did you eval the merged model?" as a standing review question.
