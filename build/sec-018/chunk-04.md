### What is "mid-training," and why did it become a named stage rather than just the end of pretraining?

The mental model: pretraining used to be one homogeneous phase — one mixture, one schedule, then hand off to SFT. Mid-training is the recognition that **the last 5–20% of pretraining tokens are worth vastly more per token than the first 80%, because the learning rate is decaying and the model is consolidating.** When LR is high, the model is exploring and individual documents barely stick. When LR is decaying toward zero, updates are small and the model is settling into a basin — data seen during that window has an outsized effect on the final weights. So you put your best data there.

That observation turns the final phase into its own engineering surface with its own mixture, its own evals, and its own iteration loop. Concretely, mid-training now usually bundles four things that happen after the bulk pretraining trunk and before post-training:

1. **The annealed high-quality mix** — swapping the general web mixture for a much higher-quality one during LR decay.
2. **Capability upsampling** — heavy math, code, and reasoning-trace data, in proportions that would be wasteful across the whole run.
3. **The long-context extension stage** — raising sequence length from, say, 8k to 128k on a small dedicated token budget.
4. **Format and instruction priming** — a small amount of instruction-shaped and multi-turn data so that SFT has something to grab onto.

The reason it got a name is economic, and it ties directly to WSD schedules. With a constant-LR trunk, the trunk is reusable: you train it once and then run *several* short decay phases with different mid-training mixes, evaluating each. Mid-training is where the experimentation moved because it is where experiments are affordable.

**💰 Math:** on a 10T-token trunk, a 500B-token decay phase is 5% of the compute. Using our earlier cost frame, if the whole 10.5T-token run costs $1.26M, one mid-training variant costs `$1.26M × 0.05 = $63k`. You can run eight mid-training variants for $504k — half the cost of a *single* full alternative pretraining run, and you get eight measured data points instead of one. That asymmetry is why every serious team now spends most of its data-research effort on this phase.

**⚠ Trap:** believing mid-training substitutes for a good trunk. It does not. You cannot anneal your way out of a badly-filtered, badly-deduped base corpus — the capability ceiling is set by the trunk, and the decay phase moves you toward the top of that ceiling rather than raising it. The correct framing is that the trunk buys capability and mid-training buys *access* to it.

**🗣 Say this in the room:** "Mid-training is the LR-decay phase treated as its own stage. Data seen while LR is annealing has outsized influence on the final weights, so you put your highest-quality data there — plus long-context extension and capability upsampling. With a WSD schedule the trunk is reusable, so you can run eight mid-training variants for the price of half a pretraining run."

### What actually goes into the annealed decay-phase mixture, and how do you decide the proportions?

The composition is a deliberate inversion of the trunk mixture. Where the trunk is dominated by filtered web because that is where the tokens are, the decay mix is dominated by data you could never afford at trunk scale.

**What goes in, in rough order of weight:**

- **The top slice of your quality classifier's output** — the FineWeb-Edu-style highest-scoring web, not just above-threshold. If your trunk kept everything above score 3, the decay phase keeps score 4–5 only.
- **Curated reference sources** — textbooks, papers, encyclopedic content, high-quality documentation. These are token-poor and quality-dense, exactly the wrong shape for a trunk and exactly the right shape for a 500B-token phase.
- **Math and code, heavily upweighted** — often 25–40% of the decay mix versus 10–15% in the trunk.
- **Reasoning traces and worked solutions**, increasingly synthetic — step-by-step derivations, chain-of-thought-formatted problem solving.
- **Instruction-shaped data**, at a small percentage — Q&A pairs, task-formatted examples. Enough that the base model is not surprised by the shape of an instruction, not so much that it becomes an instruct model.
- **General replay** at whatever fraction prevents regression — typically 30–50% of the mix stays "normal" data.

**How you decide proportions: you ablate, and the decay phase is cheap enough to ablate properly.** This is the one place in the whole pipeline where you can actually run a proper experiment at near-full scale. Fork the trunk checkpoint, run four 50B-token decay phases with different mixes, evaluate all four on the same frozen harness. Cost per variant on our earlier numbers: `50e9/10.5e12 × $1.26M = $6,000`. Six thousand dollars per real experiment at full model scale is an extraordinary bargain and it is why this is where the research happens.

**⚠ Trap:** over-concentrating the decay phase and getting a model that is excellent on benchmarks and strange in conversation. If the last 200B tokens are 60% math and textbooks, the model's *prior over text style* shifts toward that register, and it will produce textbook-flavored answers to casual questions. This is a real and commonly-observed effect — the model's output distribution reflects its most recent data disproportionately. The mitigation is the general-replay fraction, and you should tune it by reading model samples, not only by looking at benchmark scores.

**⚠ Trap:** measuring a decay-phase variant against the trunk checkpoint rather than against a *baseline decay phase*. The decay itself improves everything, so every variant will look like a win. The comparison that means something is variant-vs-control-decay, both annealed identically.

**📐 Numbers you must know:** the decay phase is typically **5–20% of total tokens**, with LR going from the stable value to ~0 (or ~1–10% of peak). Shorter than ~2% of tokens and the anneal is too abrupt to consolidate; longer than ~25% and you have just built a two-phase run with a very long tail and lost the reusable-trunk benefit.

### Walk me through the long-context extension stage. How do you take an 8k model to 128k?

The mental model: long context is **not** something you train for from scratch, because the attention cost is quadratic in sequence length and you would be paying it for all 15 trillion tokens. Instead you train the bulk at short context, where it is cheap, and then spend a small dedicated budget teaching the model to use longer ranges. The reason this works is that almost everything the model needs to learn — grammar, facts, reasoning — is learnable in 8k windows; only the positional machinery needs extending.

**The arithmetic that forces this design.** Attention FLOPs per token scale with sequence length. Using the earlier estimate, the attention term is roughly `seq_len / (6 · d_model)` as a fraction of the `6ND` term. At `d_model = 4096`: at 8k that ratio is `8192/24576 = 33%`; at 128k it is `131072/24576 = 533%`. So training at 128k costs roughly `(1 + 5.33)/(1 + 0.33) = 4.8×` per token versus 8k. Training the entire 15T-token run at 128k would turn a $1.26M run into a $6M run to buy a capability that 2% of the tokens can deliver.

**The mechanism, in three parts:**

**1. Positional encoding extension.** RoPE encodes position as a rotation whose frequency for dimension pair `i` is `θ_i = base^(−2i/d)` with `base = 10,000` conventionally. A model trained at 8k has never seen rotation angles beyond what 8k positions produce, and extrapolating past that is where it falls apart. Two families of fix:

- **Base scaling ("NTK-aware" / RoPE-theta increase).** Raise `base` from 10,000 to something like 500,000 or 1e6. This lowers all frequencies, so position 100,000 produces angles the model has seen. Llama-3 uses a RoPE base of 500,000. Simple, effective, and the current default.
- **📄 Paper:** Peng et al. (2023), *YaRN: Efficient Context Window Extension of Large Language Models* — interpolates position frequencies non-uniformly (leaving high-frequency dimensions alone, interpolating low-frequency ones) plus an attention-temperature correction, achieving long-context extension with far fewer training tokens than naive fine-tuning. It refined the earlier *Position Interpolation* idea of simply dividing positions by a scale factor.

**2. Data.** You need documents that are actually long *and* whose long range matters. This is harder than it sounds: concatenating short documents to 128k teaches nothing, because there is no real dependency spanning the window. Sources that work: books, long code repositories concatenated by import graph, legal filings, long-form transcripts, and — increasingly — **synthetically constructed tasks** where the answer provably requires information from far back (multi-hop QA over a long document, "find the function defined 90k tokens ago"). Mix in short data too, or short-context performance regresses.

**3. Schedule.** Extend progressively — 8k → 32k → 128k — rather than in one jump, with a small token budget at each stage. Typical total for the long-context stage is on the order of tens to a few hundred billion tokens, i.e. **a few percent of the run**.

**⚠ Trap:** validating with needle-in-a-haystack and declaring victory. NIAH tests *retrieval* of a verbatim string, which is the easiest possible long-context task and which models pass long before they can actually reason over long context. Use tasks requiring aggregation across many positions (multi-needle, counting, long-document summarization scored against references) and measure the **degradation curve** across context lengths, not just the endpoint. Effective context is almost always well below advertised context, and the honest metric is "at what length does accuracy fall below X."

**💰 Math on the serving consequence:** 128k context is not free at inference either. KV cache for an 8B model with GQA (32 query heads, 8 KV heads, head_dim 128, 32 layers) at fp16 is `2 (K&V) × 32 layers × 8 heads × 128 dim × 2 bytes = 131,072 bytes/token = 128 KB/token`. At 128k tokens that is **16 GB for one request's KV cache** — as much as the model weights. You just made your maximum concurrent batch size 1 on an 80 GB card. Long context is a memory decision at serving time, and the pretraining team should tell the serving team before shipping.

### Why upsample math and code far past their natural frequency? And what does "curriculum" mean here?

Math and code are upsampled because they are the highest-density source of two things web text almost never contains: **long chains of strictly-valid deduction**, and **an unambiguous notion of correctness**. Web prose lets you be vague; a program either compiles or does not, and a proof either follows or does not. Training on that shape appears to improve reasoning behavior beyond the domains themselves — the widely-replicated observation is that adding code to a pretraining mix improves performance on non-code reasoning benchmarks. I would present that as a strong empirical regularity rather than a settled mechanism, because the causal story is still contested.

The quantitative shape: code is maybe 1–3% of raw filtered web by tokens, and appears at 10–20% in trunk mixtures and 25–40% in decay mixtures. Math is even scarcer naturally and gets similar treatment. Since these corpora are small relative to their sampling weight, you are running them at multiple epochs — which is fine (recall the ~4-epoch result) but must be tracked explicitly.

**On "curriculum."** This word means two different things and interviewers use both:

1. **Ordering within the run** — the classical curriculum-learning idea of easy-to-hard. In LLM pretraining this is largely *not* what people do; the evidence for strict difficulty ordering helping is weak, and it costs you the ability to shuffle, which costs you gradient decorrelation.
2. **Mixture scheduling across phases** — which is what actually happens and what people now mean. The mixture is a function of training progress: general web-heavy early, capability-heavy and quality-heavy late, long-context near the end. This is real, it works, and it is the mid-training story.

The mechanism for why phase ordering works when strict curriculum does not: it is about the **learning rate**, not about difficulty. Early, high-LR training is where you want maximum diversity because the model is learning broad structure and individual examples are washed out. Late, low-LR training is where individual documents stick, so you want your most valuable data there. Ordering by "what the model can currently handle" is a much weaker signal than ordering by "how much does this document's influence persist."

**⚠ Trap:** upsampling a small, high-quality math corpus so aggressively that you run it for 20+ epochs. Past roughly 4 epochs the returns collapse and memorization rises — and memorized benchmark-adjacent math is exactly how you get an inflated GSM8K score that does not generalize. Compute and report the implied epoch count for every domain; if any domain exceeds 4, either find more data, synthesize more, or lower the weight.

**🗣 Say this in the room:** "Code and math get upweighted 5–10× past natural frequency because they're the densest source of verifiable multi-step deduction, and adding code measurably helps non-code reasoning. But 'curriculum' in modern practice means mixture-scheduling by training phase, not easy-to-hard ordering — and the reason it works is that low-LR late data sticks, not that the model is 'ready' for it."

### Give me the actual recipe for domain-adaptive continued pretraining. I have 30B tokens of legal text and a base model.

Here is the recipe I would run, with the reasoning for each number, because every one of these is a place teams get it wrong.

**1. Choose the checkpoint: base, not instruct.** More on this in a moment — it is the single biggest failure mode.

**2. Learning rate: far lower than pretraining, but not as low as SFT.** Peak LR of roughly **10–30× lower than the original pretraining peak** — so if the model was pretrained at 3e-4, use 1e-5 to 3e-5. Too high and you blow away the pretrained representations (catastrophic forgetting in its most literal form); too low and nothing transfers into the weights and you have run an expensive no-op.

**3. Re-warm and re-decay.** **📄 Paper:** Ibrahim et al. (2024), *Simple and Scalable Strategies to Continually Pre-train Large Language Models* — the practical finding is that **LR re-warming followed by re-decaying, combined with a modest fraction of replay data, matches full retraining on the union of old and new data at a fraction of the cost.** Warm up over ~1–2% of CPT steps; decay to near zero at the end. Do *not* just continue at a constant LR and stop abruptly — you lose the consolidation benefit.

**4. Replay: 5–30% of the mix is general pretraining data.** This is the forgetting mitigation and it is remarkably effective for how simple it is. My default is 20%. Sources: ideally the original pretraining distribution; realistically, an open corpus of similar composition (FineWeb, Dolma) plus some code and math to protect those capabilities specifically. The dial: more replay = less forgetting, less domain gain. If your general-capability suite regresses more than ~2 points, raise replay.

**5. Token budget.** With 30B unique domain tokens, at 20% replay a run of 100B tokens is 80B domain = **2.7 epochs of domain data**, which sits comfortably under the 4-epoch line. That is my target. Compute: `6 × 8e9 × 1e11 = 4.8e21` FLOPs → at 40% MFU on H100s, `4.8e21/(989e12 × 0.4) = 1.21e7` GPU-seconds = **3,370 GPU-hours** ≈ $8,400 at $2.50/hr. On 32 H100s that is 4.4 days. That is a completely ordinary applied-team budget, which is the point: CPT is affordable, pretraining is not.

**6. Sequence length and packing:** match the base model's pretraining context (or the length your documents actually need), pack with intra-document masking, and if you need long context for the domain, do the extension as a separate later stage rather than mixing it in.

**7. Everything else stays the same as the base model.** Same tokenizer (unless you are doing vocabulary expansion — separate question), same architecture, same optimizer family, same precision. Changing two things at once means you cannot attribute the result.

**⚠ Trap:** running CPT without a *pre*-measured baseline on the general-capability suite. You must evaluate the base checkpoint on your general suite and your domain suite *before* CPT, on the frozen harness. Without the "before" number, "did we forget anything" is unanswerable and you will end up arguing from vibes in a review meeting.

**🗣 Say this in the room:** "Base checkpoint, LR about 20× below the original pretraining peak with re-warmup and re-decay, 20% general replay, and a token budget that keeps domain data under four epochs. For 30B legal tokens that's roughly a 100B-token run — about 3,400 H100-hours, call it $8–9k. And I'd measure the general-capability suite before and after, because the replay fraction is the dial I tune against that number."

### When do you expand the vocabulary, and how do you do it without breaking the model?

Vocabulary expansion is worth it when the base tokenizer is **inefficient** on your domain, and the way you decide is by measuring, not guessing: tokenize a representative sample of your domain corpus with the existing tokenizer and compute characters-per-token. Compare against the tokenizer's characters-per-token on general English (typically ~4). If your domain lands at 2.0 while English is 4.0, every domain document costs 2× the tokens, which is 2× the training compute, 2× the inference cost, and half the effective context window.

**📐 Numbers you must know:** the canonical cases are non-Latin scripts (a Llama-2-era tokenizer might use 2–3 tokens per Devanagari or Thai character, making those languages 3–8× more expensive than English), specialized notation (chemistry SMILES strings, protein sequences, DNA), and heavily-jargoned domains. General English legal or medical text is usually *not* a strong case — the words are English words, and the gain is 5–15%, which does not justify the risk.

**💰 Math on whether it pays:** suppose expansion improves your domain from 2.0 to 3.5 chars/token — a 43% token reduction. On a 100B-token CPT run at $8,400, that saves `$8,400 × 0.43 = $3,600` of training. On serving, if the feature handles 200k requests/day at 3,000 domain tokens each, you go from `6e8` to `3.4e8` tokens/day; at $3/Mtok that is `$1,800/day → $1,020/day`, saving **$285k/year**. *That* is the argument for vocabulary expansion — it is an inference-cost argument, not a training-cost argument.

**The mechanism, done correctly:**

1. **Train a new tokenizer on the domain corpus**, then take the top-`k` merges/tokens that are *not* already in the base vocabulary. Typically `k` is a few thousand up to ~30k; adding 100k tokens to a model is a different project.
2. **Resize the embedding matrix and the LM head.** New rows are appended.
3. **Initialize the new rows intelligently — this is the whole game.** Random initialization is wrong; the new embeddings start as noise in a space where every other embedding is meaningful, and the model spends a lot of training just recovering. The standard good initialization is: **for each new token, take the mean of the base-model embeddings of the sub-tokens the old tokenizer would have split it into.** So a new token `"hemoglobin"` is initialized as the average of the embeddings for `["hem", "oglo", "bin"]`. This puts the new vector in a semantically sensible neighborhood immediately. A weaker but still-acceptable fallback is initializing from the mean of all existing embeddings plus small noise, which at least puts new tokens in the right region of the space.
4. **Consider a short warmup where only the new embedding rows are trainable** (everything else frozen) for a few hundred million tokens, then unfreeze. This lets the new tokens settle without perturbing the rest of the model.
5. **Then run CPT normally.**

**⚠ Trap:** expanding the vocabulary and then evaluating with the old tokenizer, or shipping a serving stack that still has the old tokenizer file. This produces a model that appears to work (most tokens are unchanged) but is subtly, non-deterministically broken on exactly the domain text you added tokens for. Version the tokenizer with the checkpoint and assert their hashes match at load time.

**⚠ Trap:** untied embeddings. If the model ties input embeddings and the LM head, you resize one object; if untied, two, and forgetting the head is a fun bug where the model understands the new tokens perfectly and can never emit them.

### I gave the team an instruct-tuned checkpoint to continue pretraining on our domain corpus. What happened?

You destroyed the instruction-following, and this is the single most common and most expensive mistake in applied continued pretraining. I have watched it happen twice.

**The mechanism.** Instruction-following is not a robust capability distributed across the whole network; it is a comparatively thin behavioral layer installed by SFT and preference optimization on a few hundred thousand to a few million examples. Continued pretraining on raw domain text is next-token prediction on unstructured documents, with **no chat template, no assistant turns, no instruction/response structure**. Every gradient step pushes the model toward "continue this document" and away from "answer this request." After a few billion tokens of that, the model reverts substantially toward base-model behavior: it completes your prompt instead of answering it, ignores the system prompt, and never emits the stop token where a chat template expects it.

The magnitude depends on token count and LR. A few hundred million tokens at low LR degrades it noticeably; 100B tokens at CPT-typical LR effectively erases it. And the failure is *visible in a way that panics people*, because the model looks broken rather than merely worse.

**The correct decision procedure:**

- **Default: CPT the base checkpoint, then re-do post-training.** This is the clean path. You need an SFT dataset — and if you are doing domain adaptation seriously, you needed one anyway to teach domain-specific task behavior. Budget for it.
- **If you have no post-training capability and must start from instruct**, then you must include **instruction-formatted replay in the CPT mixture** — not just general web replay, but actual chat-templated instruction data at maybe 5–15% of the mix, so the format keeps receiving gradient. This partially preserves the behavior. It is a mitigation, not a fix, and you should expect measurable degradation on instruction-following benchmarks.
- **If the domain gain you need is modest, use LoRA instead of full CPT.** Low-rank adapters constrain the update to a small subspace, which mechanically limits how far you can move from the base behavior. It buys less domain knowledge and forgets less. This is a real and defensible option when the goal is style/format adaptation rather than knowledge injection.

**⚠ Trap (the named version):** *continued pretraining on an instruct checkpoint with unstructured text destroys instruction-following, and you will not notice from loss.* The domain loss goes down beautifully. Nothing in your training metrics indicates a problem. You find out when someone tries the model in the chat UI. **The guard is an instruction-following eval in the in-loop suite** — IFEval-style constraint-following, plus a handful of "does it respond to a chat-templated prompt at all" smoke tests — run every few thousand steps.

**🗣 Say this in the room:** "Instruction-following is a thin post-training layer, and raw-text CPT gradients point away from it. Default is: CPT the base, then redo SFT. If you must start from instruct, you have to keep chat-templated data in the mixture at 5–15%, and either way you need an instruction-following eval in the training loop, because domain loss will look perfect while the behavior dies."

### How do you know whether the continued pretraining actually worked? What do you measure?

You need four measurements, and teams routinely run only the first one, which is the one that matters least.

**1. Domain held-out loss / bits-per-byte.** Cheap, continuous, and it tells you the model absorbed the distribution. Necessary but weak — this number will always improve, because you just trained on that distribution. It answers "did the run do anything," not "is the model better at the job."

**2. The general-capability suite, before and after.** This is your forgetting measurement. A fixed set — MMLU, ARC, HellaSwag, GSM8K, HumanEval, plus an instruction-following eval — run on the base checkpoint before you start and on every CPT checkpoint. Report **deltas**. My acceptance threshold in review: no more than 1–2 points of regression on any of them, and if there is more, raise the replay fraction and re-run.

**3. The domain task suite.** This is the one that justifies the project, and it must be **task-shaped, not perplexity-shaped**. For a legal model that means questions a lawyer would actually ask with answers a lawyer would grade, not perplexity on contracts. Crucially, you must build this *before* you start, from real user needs, and it must be held out of the CPT corpus and decontaminated against it. If you build the eval after seeing the model, you will build an eval the model passes.

**4. The comparison that actually decides the question: CPT model vs. the base model with good retrieval and a good prompt.** This is the honest baseline and it is the one skipped most often. Most domain adaptation projects fail this comparison, because a strong base model with the right context in its prompt is very hard to beat on knowledge-intensive tasks. CPT wins on *format*, *jargon*, *domain-specific reasoning patterns*, and *cases where retrieval cannot supply what is needed* (like fluency in a low-resource language or a notation system). It usually loses on "what does this specific document say," because that is retrieval's job.

**⚠ Trap:** using perplexity on domain text as the success metric. Perplexity on the domain always goes down after CPT on the domain — it is nearly tautological. I have seen a project declare success on a 30% perplexity reduction and then discover the model was *worse* on the actual downstream task, because the CPT corpus was full of boilerplate the model learned to predict very well and nobody cares about. Perplexity is a training diagnostic, not a product metric.

**💰 Math for the decision:** put a dollar value on the comparison. If CPT costs $8,400 of compute plus, realistically, 6 engineer-weeks (~$40k fully loaded) plus ongoing maintenance every time the base model updates, that is ~$50k of first-year cost. If it lifts task success from 71% to 76% on a workload of 200k tasks/day where each failure costs $0.40 of human fallback, the saving is `200,000 × 0.05 × 0.40 = $4,000/day = $1.46M/year`. Clear win. If it lifts 71% to 72%, the saving is `$292k/year` — still a win, but now the maintenance treadmill (redoing CPT on every base-model upgrade) matters, and a prompt-plus-retrieval improvement that takes two days might get you the same point.

### Where does synthetic data fit into pretraining, and what are the real risks?

The mental model: the web is a fixed and finite corpus of a few tens of trillions of usable tokens, and frontier runs are now consuming a meaningful fraction of it. Synthetic data is the field's answer to that constraint, and it works — but "synthetic data" covers three very different operations with very different risk profiles, and conflating them is how people get this wrong.

**1. Rephrasing and rewriting existing data.** Take a document your quality filter rejected — badly formatted, poorly written, full of boilerplate — and have a model rewrite it as clean prose or as a Q&A pair. The information content is still grounded in the original human document; you are changing the presentation. **This is the lowest-risk form and it is the one that has been demonstrated at pretraining scale** — NVIDIA's Nemotron-CC work is the main public example, using LLM rewriting to convert low-quality Common Crawl into usable tokens and thereby substantially increasing the yield of high-quality tokens from a fixed crawl.

**2. Generating new content from a seed.** Take a topic, a document, or a code repository and generate exercises, worked solutions, dialogues, or reasoning traces about it. Still grounded, but now the model is contributing content and therefore contributing errors. Verification matters: for math and code you can verify by execution or by checking against a known answer, and you should. For prose you generally cannot, and you are relying on the generator's accuracy.

**3. Free-generation from the model's own distribution.** Ask a model to write documents about nothing in particular. This is where the danger is concentrated.

**The real risks, named:**

- **Model collapse.** **📄 Paper:** Shumailov et al. (2024), *AI models collapse when trained on recursively generated data* — training generation after generation on the previous generation's outputs causes the distribution's tails to vanish first and then the whole distribution to narrow, because sampling error and the model's own bias compound. The important caveat, which you should state: collapse is demonstrated under *fully recursive replacement* of real data. When synthetic data is **accumulated alongside** real data rather than replacing it, the effect is dramatically weaker. The operational rule that follows: never let synthetic data replace the real corpus; always mix, and keep a real-data floor.
- **Homogenization.** Even without full collapse, synthetic data inherits the generator's stylistic and topical preferences. A corpus rewritten by one model is more uniform than the human text it came from, and you lose exactly the diversity that makes pretraining data valuable. Mitigate by using multiple generators, high sampling temperature, and persona/style conditioning.
- **Error amplification.** The generator's factual mistakes become training data, and unlike a human error on a web page, they are systematic — the same wrong belief, restated a million times, with no contradicting sources.
- **Licensing and ToS.** Generating pretraining data with a commercial API very often violates that provider's terms regarding training competing models. This is a legal question that has ended projects, and it is a legitimate thing to raise in a design review.

**🗣 Say this in the room:** "I'd separate rephrasing from generation. Rephrasing rejected web data is well-demonstrated at scale and low-risk because the information is still grounded in a human document. Free generation risks collapse — and the collapse results are specifically about *replacing* real data recursively; accumulating synthetic alongside real is much safer. So the rule is a hard real-data floor, multiple generators for diversity, and execution-based verification wherever the domain allows it."

### Give me the decision framework: prompting vs RAG vs SFT vs continued pretraining. With numbers.

This is the question I would actually ask a candidate for an applied AI role, because the wrong answer — reflexively reaching for fine-tuning — is the most reliable rejection signal there is. The framework is: **each technique injects a different kind of information, and you should be able to say which kind you are missing.**

| Technique | What it changes | What it cannot fix | Time to first result | Marginal cost |
|---|---|---|---|---|
| Prompting / few-shot | The model's conditioning for this request | Missing knowledge; missing capability | Hours | +input tokens/request |
| RAG | What facts are in context | Format, style, reasoning patterns, jargon fluency | Days–weeks | +retrieval infra +input tokens |
| SFT / LoRA | Output format, style, task behavior, tool-use patterns | Genuinely missing world knowledge | 1–3 weeks | Training run + eval + maintenance |
| Continued pretraining | Domain distribution, jargon, notation, low-resource fluency | Anything a strong base already knows | 1–3 months | Large training run + full re-post-training |

**The escalation ladder I enforce**, in order, with the precondition for moving to the next rung:

1. **Better prompt + few-shot.** Move on only when you have a failing eval that better prompting demonstrably cannot fix, measured.
2. **Retrieval.** Move on only when you can show the right document *was* in context and the model still got it wrong. That is the diagnostic that separates a knowledge problem from a capability problem, and it is a single experiment: hand-place the gold document in the prompt and re-measure. If accuracy jumps, your problem is retrieval, not the model.
3. **Structured output / tool design / decomposition.** Cheap, often the actual fix.
4. **SFT or LoRA.** Move on to this only when the failure is about *behavior* — wrong format, wrong style, wrong tool-calling pattern, wrong verbosity — and you have ≥1,000 good examples plus a held-out eval.
5. **Continued pretraining.** Only when the base model demonstrably lacks the domain's *language*: a low-resource language, a notation system, an internal codebase's idioms at a scale retrieval cannot carry.

**💰 Math on the decision, made concrete.** A support-automation feature, 200k requests/day, currently 71% autonomous resolution, each failure costing $0.40 in human handling.

- **Prompt engineering:** 1 engineer-week ≈ $6.5k. If it moves 71% → 74%: saves `200,000 × 0.03 × 0.40 = $2,400/day = $876k/year`. **ROI: 134×.**
- **RAG build:** ~8 engineer-weeks ≈ $52k, plus ~$3k/month infra ≈ $36k/year, plus 2,000 extra input tokens/request: `200,000 × 2,000 = 4e8 tok/day` at $3/Mtok = `$1,200/day = $438k/year`. Total year-one ≈ $526k. If it moves 74% → 82%: saves `200,000 × 0.08 × 0.40 = $6,400/day = $2.34M/year`. **ROI: 4.4×.** Still clearly worth it.
- **SFT:** ~6 engineer-weeks + labeling ≈ $60k, training ~$500, plus rebuild every base-model upgrade. If it moves 82% → 85%: saves `$876k/year`. **ROI: ~14×** in year one, less thereafter — and it composes with RAG rather than replacing it.
- **CPT:** ~$50k+ and 2–3 months, and against a strong base with working retrieval, the realistic delta on this workload is 0–2 points. **Often negative ROI**, and it is the option people propose first.

**🗣 Say this in the room:** "The question I ask before any fine-tune is: if I hand-place the perfect document in the context, does the model get it right? If yes, it's a retrieval problem and fine-tuning is the expensive wrong answer. If no, is it wrong about *format* or wrong about *knowledge*? Format is SFT. Missing knowledge in a domain the base has genuinely never seen — a language, a notation — is the only case where I'd argue for continued pretraining."
