### Support escalates: "the model got noticeably worse starting last Tuesday." Nothing shipped to the prompt. Walk me through it.

I have run this exact investigation and the answer was a sampling default both times, so let me give the ladder in the order that finds it fastest rather than in the order that sounds most thorough.

**First, define "worse" as a measurable.** "Worse" is not actionable. Get 20 flagged conversations and cluster the complaint: is it *more repetitive*, *more verbose*, *less accurate*, *less varied*, *malformed output*, or *slower*? Those five point at completely different causes, and the sampler is the prime suspect for exactly two of them — more repetitive and less varied — and a strong secondary suspect for malformed output.

**Second, diff the effective request bodies across the boundary.** Not the config in git — the *serialized payload the gateway sent*, logged. This is the single highest-value piece of observability in an LLM system and most teams do not have it. If you do not have it, that is your first fix, and you can usually reconstruct the last week from provider-side logs. What you are looking for is a parameter that changed value or changed from present to absent.

The specific mechanisms by which a sampling default moves without anyone shipping a prompt change:

- **An SDK upgrade changed a client-side default.** A library that previously omitted `temperature` (letting the provider default apply, often 1.0) now sends `temperature=0.0`, or vice versa. Renovate merged it on Monday night.
- **A framework upgrade.** LangChain, LlamaIndex, an agent framework, or a serving-engine bump each carry their own defaults for `top_p`, `repetition_penalty`, `max_tokens`. vLLM and HuggingFace do not agree on defaults, and neither agrees with the hosted APIs.
- **`generation_config.json` shipped with the model.** If you self-host and pulled a new checkpoint revision, the model author's recommended sampling parameters came along with it and your engine may be honoring them. This one is genuinely invisible — the change is in a file nobody reviews.
- **A gateway or proxy injected a default** when a field was `None`, and the injection logic changed.
- **A model alias moved.** `-latest` pointing at new weights is not a sampling change, but it presents identically and must be ruled out in the same pass.

**Third, reproduce with a bisect on parameters, not on code.** Take one flagged prompt, run it 20 times under last week's parameter set and 20 times under this week's, at the same model version. If the quality difference reproduces, you have it, and you can attribute it to a single parameter by flipping one at a time. Twenty samples is the right number: at nonzero temperature a single sample tells you nothing, and you need enough to see a distributional shift.

**Fourth, if parameters are identical, look at what else feeds the sampler.** Did the tokenizer or chat template change (a new BOS, a changed role marker), so the model is now generating from a subtly different distribution? Did a constrained-decoding schema get added, which silently suppresses tool calling and reasoning quality? Did speculative decoding get enabled — which should be distribution-preserving, but is not if the draft and target logits processors are mismatched, and is definitely not if someone shipped a "lossy" spec-decode mode?

**🗣 Say this in the room:** "First I turn 'worse' into one of five measurable symptoms, because repetitive and inaccurate have different causes. Then I diff the effective request payloads across the change boundary — not the config in git — because the usual culprit is a dependency bump that moved a sampling default. HuggingFace, vLLM and the hosted APIs all have different defaults, and `generation_config.json` ships with the checkpoint."

**🔍 Failure taxonomy — sampling defaults that bite:** (1) `temperature` unset → provider default 1.0, not 0. (2) `top_p` unset in an engine that defaults it to 1.0 while your old stack used 0.9 → more tail, more hallucination. (3) `repetition_penalty` defaulting to something above 1.0 in a local engine → broken code and drifting variable names. (4) `max_tokens` defaulting low → silent truncation reported as "the model stopped explaining things." (5) `n` or `best_of` defaulting to something other than 1 → a 4× bill nobody attributes to a default.

**The prevention** is a config-as-code invariant: the gateway sends every sampling parameter **explicitly on every request**, never relying on a provider or library default, and the effective payload is logged at a sampled rate. That single rule eliminates this entire failure class, and it is the thing I would say I'd ship on day one.

### Design the sampling-parameter layer of a multi-tenant LLM gateway.

You already know how to build a gateway; the delta is that sampling parameters are a *policy* surface with correctness, cost and safety consequences, and treating them as pass-through key-values is how teams get into the mess described in the previous question. I would build four layers.

**Layer 1 — a typed, validated parameter object.** Pydantic v2 model, one per provider family, with real constraints: `temperature: float = Field(ge=0.0, le=2.0)`, `top_p: float = Field(gt=0.0, le=1.0)`, `max_tokens: int = Field(ge=1, le=MODEL_MAX)`. Cross-field validators reject the incoherent combinations — `temperature=0` together with `n>1` is almost certainly a bug (you are paying `n×` for `n` identical outputs) and should be a hard 400 with a message, not a silently accepted request. Same for `top_p < 0.3` combined with `temperature < 0.3`. This is exactly the discriminated-union-plus-field-validator work you already do; the novelty is knowing *which* combinations are incoherent.

**Layer 2 — named profiles, not raw parameters, as the public interface.** Callers request `profile="extraction"` or `profile="creative"`, and the gateway resolves that to a parameter set per model. This is the highest-leverage design choice in the whole component. It means (a) a change to what "extraction" means ships once, centrally, with an eval gate, instead of being copy-pasted into 40 services; (b) you can port across models and providers, since the same profile maps to different concrete parameters for a model whose `top_p` semantics differ; (c) your telemetry is aggregable by profile, which is what you actually want to slice quality metrics by. Raw parameter overrides stay available but require an explicit escape hatch and get flagged in review.

**Layer 3 — explicit serialization, always.** Every parameter is sent explicitly on every request; nothing is left to a provider or SDK default. The resolved payload is hashed and the hash is attached to every trace span alongside the model version string. When someone asks "did sampling change on Tuesday," the answer is a `GROUP BY payload_hash` over your traces, answered in thirty seconds.

**Layer 4 — policy enforcement per tenant and per surface.** Cap `max_tokens` per tenant tier (this is a cost control and, per the KV-reservation arithmetic earlier, a capacity control). Force `temperature=0` and structured output on surfaces marked as extraction. Refuse `n > 4` outside an allowlisted profile. And version the profiles: `extraction@v3`, pinned per caller, so a profile change is a deliberate migration rather than a global blast radius.

**⚠ Trap:** letting sampling parameters flow through untouched from an end-user-controllable field. If a customer can set `temperature` on a support-bot surface, they can set it to 2.0 and generate something quotable and embarrassing that your brand owns. If they can set `max_tokens` and `n`, they control your bill. Sampling parameters are a privileged control plane and the default posture is "not user-settable."

**💰 Math:** the `max_tokens` cap alone pays for the component. Suppose 30% of your 2 M calls/month currently run to a defensively-high `max_tokens=4096` with a real need of ~500, and that a third of those actually run long because of a rambling failure mode. That is `2e6 × 0.30 × 0.33 = 198,000 calls` emitting an excess `~3,000` tokens each = `594 M` excess output tokens/month. At $15/Mtok that is **$8,910/month** of pure waste, before counting the concurrency you get back.

### What can I legitimately do with the `logprobs` a provider returns — and what can't I?

Logprobs are the model's log-probability for the chosen token plus, optionally, the top-`k` alternatives at each position. They are the only quantitative signal a hosted API gives you about the model's internal state, and they are simultaneously the most over-interpreted number in applied LLM work.

**What they legitimately support.**

*Selection among a fixed set.* The classification-by-logit pattern from earlier — softmax over the label tokens — is exactly right, because you are comparing alternatives at a single position under a single distribution. This is the strongest use.

*Relative confidence thresholding.* Sequence-level mean logprob correlates usefully with output quality: route the bottom decile to a larger model or a human. You do not need calibration for this, only monotonicity, and monotonicity holds well enough.

*Perplexity-based reranking.* Score `n` candidates by their length-normalized logprob under the same model. Useful for best-of-n when you have no verifier.

*Detecting the "I'm guessing" regime.* A sudden spike in per-token surprisal in the middle of a factual answer is a genuine signal that the model has left the region it knows. It is noisy, but as a cheap flag on a citation or a numeric value it earns its keep.

*Constrained-choice evaluation.* For a multiple-choice eval, reading the logprob of each option letter is far cheaper and lower-variance than generating and parsing.

**What they do not support.**

*Calibrated probability of correctness.* A logprob of −0.05 (p=0.95) on the token `" Paris"` means "given my context and my training, this is the token I'd emit 95% of the time." It does not mean "there is a 95% chance this claim is true." Instruction tuning and RLHF systematically sharpen distributions — the model becomes more confident without becoming more accurate — so post-RLHF models are *worse* calibrated than their base models on this measure. Reporting a logprob to a user as a confidence percentage is a thing I would block in review.

*Cross-model comparison.* Logprobs are per-tokenizer and per-vocabulary. A model with a 32k vocabulary and one with a 200k vocabulary put mass on different-sized units; their per-token logprobs are not comparable, and neither are perplexities computed from them. This is the same invariance problem that makes bits-per-byte the honest cross-model metric.

*Hallucination detection as a solved problem.* Low logprob correlates with hallucination weakly. A confidently-stated fabrication — which is the dangerous kind — often has *high* logprob, because the model has confabulated fluently. Sequence logprob is a useful feature in a detector; it is not a detector.

**⚠ Trap:** using logprobs from a *reasoning* model or a model behind a router and assuming they describe the answer you see. If the visible answer was produced after a long hidden thinking phase, the logprobs on the answer tokens are conditioned on that thinking and are near-1.0 by construction — the model is essentially reading off its own conclusion. They tell you almost nothing about whether the conclusion was right.

**📅 Volatile:** which endpoints expose `logprobs`, how many alternatives (`top_logprobs`) they return, and whether reasoning models expose them at all — all of this varies by provider and changes. Verify before designing around it.

### I want to change our default temperature from 0.7 to 0.3. How do you evaluate that change?

The trap here is that a sampling change is a **distributional** change, and the standard "run the eval set once before and once after" procedure has enough variance to hide it or to manufacture it. I want to lay out the procedure precisely because this is a graded competency.

**Step 1 — decide what could plausibly move, and pick metrics for each.** Lowering temperature should improve factual accuracy and format compliance, and should reduce diversity and possibly perceived helpfulness. So I need at least one metric in each direction, otherwise I will "prove" an improvement by measuring only the thing I expected to improve. Concretely: accuracy or groundedness on a labelled set; format-compliance rate; distinct-n or embedding-based diversity across repeated generations for the same prompt; and a preference-judge win rate for overall quality.

**Step 2 — paired design.** Run both configurations on the *same* prompts, and compare per-prompt. Paired comparison removes prompt-difficulty variance, which is by far the largest variance component, and it typically cuts the sample size you need by a large factor. Never compare "the eval we ran in March" to "the eval we ran today."

**Step 3 — multiple samples per prompt, because the thing you are changing is a distribution.** At `T=0.7`, one sample per prompt is a draw from a distribution whose spread is exactly what you are manipulating. I run `k = 5` samples per prompt per arm and report the *mean per-prompt score*, which reduces the sampling-noise component by roughly `1/√5`. If your metric is pass/fail, report per-prompt pass rate rather than a single pass/fail.

**Step 4 — size it.** To detect a 2-percentage-point difference in a pass rate near 80% with 80% power at α=0.05, the paired-proportion calculation lands you in the neighborhood of a thousand prompts if the two arms disagree often, and far fewer if they mostly agree — because McNemar's test keys on the *discordant pairs* only. That is the point worth making out loud: for paired binary outcomes, what determines your power is how many prompts flip between arms, not how many prompts you have. Report the discordant count; if it is 12, no amount of elegance in the analysis makes your result meaningful.

**Step 5 — the right test.** Paired binary outcomes → McNemar. Paired continuous scores → paired bootstrap over prompts (resample prompts with replacement, recompute the difference, take the 2.5/97.5 percentiles), which I prefer to a t-test because LLM scores are rarely normal and often bimodal. And if you swept several temperature values, correct for multiple comparisons — a 6-value sweep at α=0.05 gives you about a 26% chance (`1 − 0.95^6 = 0.265`) of a spurious "winner" if you do not.

**Step 6 — slice.** A global temperature change will help some task types and hurt others; the aggregate can be flat while both tails moved a lot. Slice by task type at minimum. This is usually the finding that turns "change the default" into "add a profile," which is the better outcome.

**⚠ Trap:** evaluating a temperature change with an LLM judge and a fixed seed, at temperature 0 on the judge, and calling the result significant because the judge is deterministic. The judge's determinism does not reduce the *generator's* variance, which is the variance you care about. Determinism in the measuring instrument is good; it is not statistical power.

### Does letting each request have its own sampling parameters cost the serving engine anything?

Yes, more than people expect, and it is a good question because it forces you to think about what the sampler actually does when 256 sequences with 256 different parameter sets are decoded in one step.

The engine cannot run a per-request Python branch — that would serialize the batch. Instead it gathers the parameters into tensors (`temperatures: [B]`, `top_ps: [B]`, `penalties: [B, ...]`) and applies every enabled operation to the *whole* `[B, V]` logit matrix in a vectorized way, using the per-row parameter to modulate. Requests that did not ask for an operation get the identity value for it. **So the cost of a sampling feature in a batch is paid by every request in that batch the moment one request uses it.** One tenant enabling top-p makes the sort run for all 256 rows.

The expensive one is any mass-based truncation, because the textbook implementation sorts. A `[256, 128256]` sort is 32.8 M float32 elements plus 32.8 M index elements. GPU radix sort throughput is in the low tens of billions of elements per second, so this lands in the low-single-digit milliseconds. Compare that to the decode step itself: a 70B model at batch 256 might take 20–30 ms per step. A 2–3 ms sampler is a **10% throughput tax on the entire deployment**, paid because some requests set `top_p`. That is a number worth having, and the shape of the argument matters more than the exact digits — measure yours.

The mitigations are real and worth naming, because they show you have read an engine. **Sort-free top-p/top-k** via a rejection-sampling scheme: draw a candidate from the full distribution, check whether it satisfies the top-p condition using only order statistics you can compute without a full sort, and retry on failure. FlashInfer implements this family, and it removes the `O(V log V)` term. **Partial sorts** — you only need the top-`m` for some `m` bounded by the practical nucleus size, so `topk` with a modest `m` plus a fallback path handles the common case at a fraction of the cost. And **fusing** temperature, penalties and masking into a single kernel pass over the logit matrix instead of five separate `[B, V]` reads, which matters because at `[256, 128256]` fp32 each pass is 131 MB of HBM traffic — five passes is 655 MB, or about 0.3 ms at 2 TB/s just in memory movement.

**⚠ Trap:** assuming the sampler is free because "it's just a softmax." At large vocabulary and large batch it is a non-trivial fraction of the decode step, and it is a fraction that grows as you add features. When you propose adding min-p support to a gateway, the honest cost accounting includes the batch-wide kernel cost, not just the per-request one.

**📐 Numbers you must know:** logits at batch 256, vocab 128k, fp32 = `256 × 128256 × 4 = 131 MB`. Every sampler stage that reads and writes that tensor costs ~131 MB of read plus 131 MB of write; at 2 TB/s that is ~0.13 ms per stage. This is why engines fuse the sampler and why they keep the logits in bf16 where they safely can — and why they upcast only the surviving candidates rather than the whole vocabulary when precision matters.

### How do sampling parameters change for a reasoning model with a long hidden thinking phase?

The governing insight: **a reasoning model's output is two documents with different requirements glued into one generation — a long exploratory thinking phase and a short committed answer — and a single set of sampling parameters has to serve both.** That tension is why providers have been restricting the knobs rather than exposing more of them.

Practical consequences, in order of how often they bite.

**Penalties become actively dangerous.** A 4,000-token chain of thought legitimately repeats itself: it restates the problem, re-derives an intermediate value, checks an earlier step. A frequency penalty accumulating over 4,000 tokens subtracts an ever-growing amount from exactly the tokens the reasoning depends on — the variable names, the quantities, the operation words. By token 3,000 the model is being pushed to paraphrase its own arithmetic. Set all penalties to identity on reasoning models; I treat a nonzero penalty on a thinking model as a review blocker.

**`max_tokens` must budget the thinking.** Thinking tokens are output tokens: you pay for them, and they count against the cap. A `max_tokens=1024` on a model that wants 3,000 tokens of thought gets you a truncation *inside the reasoning block* with no answer at all — you paid full price for nothing. The budget must be `thinking_budget + expected_answer_length + headroom`, and where the API separates them (Anthropic's `budget_tokens` for extended thinking, Gemini's `thinkingBudget`), set both explicitly. **📅 Volatile:** the minimums, defaults and interactions here change; verify against current docs.

**Temperature moves quality non-monotonically.** Some exploration in the thinking phase genuinely helps — it is what self-consistency exploits — while the final answer wants to be near-greedy. With one temperature for both you are compromising, and the compromise usually lands around 0.6–1.0 depending on the model's training. This is one of the reasons several providers now **fix or reject** `temperature` and `top_p` on reasoning endpoints: the vendor tuned it and does not want you moving it. That is a defensible product decision, not a limitation.

**🗣 Say this in the room:** "Constrain the output, not the thinking. I let the reasoning block generate free-form with penalties off and no grammar, and I apply structured output only to the final answer block. Constraining the whole generation measurably degrades reasoning quality, and penalties over a 4,000-token chain of thought penalize the exact tokens the derivation needs."

**⚠ Trap:** applying a JSON schema constraint across the entire generation of a reasoning model. The grammar forbids the model from thinking in prose, so it either reasons badly inside string fields or does not reason at all — you have bought well-formed JSON at the cost of the capability you were paying a reasoning model for. The correct architecture is a two-phase generation: reason freely, then constrain a final block (or a second call) to the schema.

### A generation never terminates — it always runs to `max_tokens`. Debug it.

Non-termination is a stop-condition failure, and there are exactly five causes. I would walk them in this order because it is roughly the order of frequency.

**One — the EOS token id is wrong.** This is the most common cause on self-hosted models and it is almost always a config mismatch. Modern instruct models frequently have *multiple* terminal tokens: a general `<|end_of_text|>` and a chat-specific `<|eot_id|>` or `<|im_end|>`. If your generation config lists only the former and the chat-tuned model emits the latter, the engine sees a normal token and keeps going. The tell: decode the raw output and look for the end-of-turn marker sitting in the middle of your text, followed by the model starting a *new* fake user turn. That signature — the model hallucinating the next turn of the conversation — is diagnostic and means "you did not stop at the right token," not "the model is confused." The fix is passing the full set of terminal ids (`eos_token_id` accepts a list in most stacks).

**Two — the chat template is wrong or missing.** If you concatenated messages by hand instead of applying the model's template, the model is off-distribution and has no reason to emit the end-of-turn token it was trained to emit at a boundary it does not recognize. Same class of bug, upstream.

**Three — EOS is being suppressed.** Check for a `logit_bias` on the EOS id, a `min_new_tokens` / `min_tokens` setting that masks EOS until a floor is reached (this is a *feature* that behaves exactly like this bug), or a `SuppressTokensLogitsProcessor` left over from a previous experiment. Also check that a constrained-decoding grammar has an accepting state that permits EOS — a grammar that never accepts is a grammar that never lets the model stop, and this is a genuinely common bug in hand-written GBNF.

**Four — the sampler is preventing EOS from being drawn.** Two mechanisms. A repetition penalty penalizes EOS after it appears once (rare in output, but it appears in few-shot examples in the prompt if penalties cover the prompt). And a truncation filter can exclude it: EOS often sits at moderate probability — say 0.06 — while the model debates whether to continue; a `top_k=5` or an aggressive `min_p` can mask it out at every step, and the cumulative probability of never stopping compounds.

**Five — it is a base model.** Base models do not reliably emit EOS at a semantic boundary because there is no such thing as a "turn" in their training data. This is not a bug and no sampler setting fixes it. Use an instruct model or a stop sequence.

**💰 Math on why you care:** a summarization endpoint that should emit 300 tokens but runs to `max_tokens=4096` wastes 3,796 output tokens per call. At $15/Mtok and 50,000 calls/day, that is `50,000 × 3,796 × 15e-6 = $2,847/day = $85,410/month`, plus the latency: 3,796 extra tokens at 25 ms inter-token latency is **95 seconds** of extra generation per request, which means your p99 is not a p99, it is a timeout. This is the single most expensive class of sampler bug and it hides behind a green dashboard because the requests all return 200.

### How is the token actually drawn from the distribution? Give me two methods and tell me why an engine might prefer one.

Two equivalent methods, and knowing both is a genuine differentiator because the second one explains several things that otherwise look like magic.

**Inverse-CDF sampling** is what `torch.multinomial` does: compute the probabilities, take the cumulative sum, draw `u ~ Uniform(0,1)`, and return the first index where `cumsum ≥ u`. Straightforward, requires a normalized probability vector and a cumulative sum over the vocabulary.

**The Gumbel-max trick** is the one worth having in your pocket. Draw `g_i ~ Gumbel(0,1)` independently for each vocabulary entry — computed as `g = -log(-log(u))` with `u ~ Uniform(0,1)` — add it to the *logits*, and take the argmax:

```python
def gumbel_sample(logits, temperature=1.0):
    u = torch.rand_like(logits).clamp_(1e-20, 1.0)
    g = -torch.log(-torch.log(u))
    return torch.argmax(logits / temperature + g, dim=-1)
```

The remarkable fact is that this is **exactly** equivalent to sampling from `softmax(logits/T)` — not an approximation. The proof is a short computation with the Gumbel CDF, and it is a fair whiteboard ask.

Why an engine cares. **No softmax and no cumulative sum are needed** — you never normalize, which saves a pass over the `[B, V]` tensor and avoids the numerical care that a 128k-wide softmax demands. **It is a pure argmax**, which is a single well-optimized reduction with no sequential dependency, unlike a cumsum. **It extends to sampling `k` items without replacement** for free: take the top-`k` of `logits + g` (Gumbel-top-k), which is the clean way to draw `n` distinct candidates rather than sampling `n` times independently and getting duplicates.

There is a third property that is genuinely useful and rarely mentioned: **shared Gumbel noise couples your samples.** Fix the Gumbel draws and sweep temperature, and you get a coupled family of samples where the differences are attributable to temperature alone, not to independent randomness. That is a much better experiment design for a temperature sweep than independent sampling, and it is a nice thing to volunteer.

**⚠ Trap:** implementing `-log(-log(u))` without clamping `u` away from 0 and 1. At `u = 0` you get `-log(inf) = -inf`... actually `log(0) = -inf`, so `-log(-log(0))` is `-log(inf) = -inf`, and at `u = 1`, `log(1) = 0` and `-log(0) = +inf`, giving `-inf`. Either way you poison the argmax. `torch.rand` returns values in `[0, 1)`, so `u = 0` is reachable. Clamp. This is the entire content of a class of "one in ten million requests returns token 0" bugs.

### Product asks for a "regenerate" button. Design its sampling behavior.

I like this question because it is where sampling theory meets product judgment, and the naive implementation is wrong in both directions depending on your defaults.

**The requirement, stated properly:** a regenerate must produce an output that is (a) different from the previous one in a way the user perceives as a genuine second attempt, and (b) not worse in expectation. Those pull against each other, and the naive implementations fail one or the other.

If your surface runs at `temperature=0`, hitting regenerate re-runs the identical computation and returns the identical answer. Users report it as "the button is broken," and they are right. If your surface runs at `temperature=0.7`, regenerate returns a genuinely different sample — but a *random* one, which is worse than the first roughly half the time. Users learn that regenerate is a coin flip and stop trusting it.

**What I would build**, in increasing order of investment:

*Baseline.* Keep the surface's normal parameters but force a fresh seed and add a modest temperature bump — if the base is `T=0.3`, regenerate at `T=0.6`. This makes the second attempt meaningfully explore rather than jitter, and it is one config line. Regenerate is by definition a signal that the user did not like sample one, so widening the search is the right response.

*Better: condition on the rejection.* Append the previous response to the context with a short instruction to produce a materially different approach. This turns a resample into a conditional generation and gets you diversity that is *semantic* rather than *lexical* — which is what the user actually wanted. It costs the previous response's tokens in input; on a 600-token answer at $3/Mtok that is `600 × 3e-6 = $0.0018`, which is nothing.

*Better still: pre-generate and select.* If the surface has budget, generate `n=2` or `n=3` on the first call (paying `n×` on output but `1×` on the shared prefill), return the best by whatever selector you have, and hold the others. Regenerate then returns the second candidate at **zero latency and zero marginal cost** — it is already computed. This is a genuinely superior experience: regenerate becomes instant. The trade is that you pay the extra generation on 100% of requests to serve the ~10–20% who regenerate.

**💰 Math on that trade:** 1 M requests/month, 600 output tokens, $15/Mtok, and a 15% regenerate rate. *Reactive:* `1e6 × 600 × 15e-6 = $9,000` plus `0.15 × 1e6 × 600 × 15e-6 = $1,350` = **$10,350/month**, with a full generation's latency on regenerate. *Pre-generate n=2:* `1e6 × 2 × 600 × 15e-6 = $18,000` = **$18,000/month**, with instant regenerate. You are paying $7,650/month to make 150,000 regenerates instant — about **$0.051 per instant regenerate**. Now it is a product decision with a number attached, which is the whole point of doing the arithmetic. Below roughly a 50% regenerate rate, reactive wins on cost; pre-generation wins only if latency on that path is a strategic priority.

**⚠ Trap:** implementing regenerate as "same request, new seed" on a surface where a prefix cache is in play and then being surprised that the outputs are *sometimes* identical anyway. If the engine's RNG is derived from batch position rather than a per-request seed, and the request lands identically, you can get the same trajectory. Verify empirically — generate 20 regenerates on a fixed prompt and count distinct outputs — rather than trusting that a new seed was honored.

### Is there anything in the decoding literature past nucleus sampling that you actually watch?

Honest answer: the *parameter* end of decoding research is largely settled and mostly noise — every year produces another truncation heuristic and the practical difference between well-tuned top-p and well-tuned min-p is small. Where I do pay attention is decoding methods that use *extra forward computation* to change the distribution rather than reshaping the one you have, because those can improve capability rather than just style.

**Contrastive search** samples from the top-`k` but penalizes candidates whose hidden state is too similar to the already-generated context, explicitly optimizing for a "degeneration penalty." **📄 Paper:** Su et al. (2022), "A Contrastive Framework for Neural Text Generation." It is exposed in HuggingFace as `penalty_alpha` with `top_k`, needs no extra model, and genuinely reduces repetition on base models. It is largely irrelevant on modern instruct-tuned models, which do not degenerate the way 2022 models did — which is itself the general pattern in this literature.

**Contrastive decoding** (Li et al., 2023) takes the difference between a large expert model's logits and a small amateur model's logits, on the theory that failure modes shared by both are artifacts of scale-independent bias rather than knowledge. **DoLa** (Chuang et al., 2024) applies the same idea *within* one model, contrasting the final layer's logits against an earlier layer's via the logit lens, and reports factuality gains. Both are interesting because they are *capability* interventions at decode time, and both cost extra compute — DoLa less so, since it reuses intermediate activations you already have.

What I would tell an interviewer about the state of play: **this area is genuinely contested and the effects are model-dependent.** Every one of these methods was validated on the models of its moment, and most of the reported gains shrink or vanish on models that have had heavy RLHF and reasoning post-training, because post-training has internalized the correction. My decision rule is that I do not adopt a decoding-time method unless it (a) survives an eval on my own task with my own model, (b) is implemented in the engine I actually serve on — not just in HuggingFace — and (c) has a cost I can state. Almost nothing passes all three, which is why production stacks in 2026 overwhelmingly run temperature plus top-p and spend their innovation budget on context, tools and post-training instead.

**🗣 Say this in the room:** "Truncation-heuristic research is basically done — the delta between tuned top-p and tuned min-p is small. What I watch is decode-time methods that add computation to change the distribution: contrastive decoding against a weaker model, DoLa contrasting layers, and speculative decoding as the one that's unambiguously won because it's exactly distribution-preserving. Most of the rest were validated pre-RLHF and don't replicate on modern post-trained models."

### Give me the drill set. What should I be able to do unaided, and how do I know I've passed?

Four drills, in the order I would run them. All are unaided — no autocomplete, no docs — because Anthropic, DeepMind, xAI and several quant shops prohibit AI tools in live rounds, and because the muscle you are building is recall under pressure.

**🏋 Drill 1 — the generation loop. 15 minutes.** From an empty file, write a `generate()` that takes a HuggingFace causal LM and tokenizer, does one prefill and then single-token decode with a `past_key_values` cache, applies temperature and top-p, samples, and stops on EOS or `max_new_tokens`. *Pass criteria:* it runs on a small model (GPT-2 is fine) and produces coherent text; the decode step passes exactly one token; logits are upcast to fp32 before sampling; you did not need to look anything up. *Failure tells:* re-feeding the whole sequence each step, forgetting `use_cache=True`, sampling from `logits[:, 0, :]` instead of `[:, -1, :]`.

**🏋 Drill 2 — top-p, and then *verify* it. 20 minutes.** Write `top_p_filter(logits, p)` from memory. Then write the verification, which is the part people skip and the part that is actually being graded:

```python
def verify_top_p():
    torch.manual_seed(0)
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    p = 0.9
    # 1. analytic nucleus
    probs = logits.softmax(-1)                       # descending already
    cum = probs.cumsum(-1)
    expected_k = int((cum < p).sum()) + 1            # first index reaching p
    # 2. empirical support from many draws
    filt = top_p_filter(logits, p)
    draws = torch.multinomial(filt.softmax(-1).expand(50000, -1), 1)
    assert draws.unique().numel() == expected_k
    # 3. conditional distribution matches renormalized truncation
    emp = torch.bincount(draws.flatten(), minlength=6).float() / 50000
    ref = filt.softmax(-1)[0]
    assert (emp - ref).abs().max() < 0.01
    # 4. edge cases
    assert torch.equal(top_p_filter(logits, 1.0), logits)          # identity
    conf = torch.tensor([[20.0, 1.0, 0.0]])
    assert torch.isfinite(top_p_filter(conf, 0.5).softmax(-1)).all()  # no NaN
```

*Pass criteria:* all four assertions pass on the first run, and specifically assertion 4's confident-distribution case — that is the `cum > p` empty-nucleus bug, and if your implementation NaNs there you have written the bug that ships.

**🏋 Drill 3 — beam search with length normalization. 25 minutes.** Write it, including finished-beam handling and `α`-normalization. *Pass criteria:* it beats greedy on average sequence log-probability over 10 prompts (that is the correctness check — if it does not, your scoring or your beam bookkeeping is wrong); it accumulates log-probs by addition, not probabilities by multiplication; setting `α=0` visibly biases it toward shorter outputs, which you should be able to demonstrate.

**🏋 Drill 4 — the verbal battery. 3 minutes each, spoken aloud, timed.** (1) Why is temperature 0 not deterministic? Must include non-associativity of fp addition, batch-dependent reduction order, batch invariance as the missing property, and autoregressive amplification. (2) Derive the effect of temperature on the odds ratio between two tokens, with a worked number. (3) Name three ways a repetition penalty breaks code generation, mechanistically. (4) Given `α = 0.75` acceptance and `γ = 4` draft tokens in speculative decoding, compute expected tokens per target forward pass — `(1 − 0.75^5)/(1 − 0.75) = (1 − 0.2373)/0.25 = 3.05` — and say what your sampling configuration does to `α`. *Pass criteria:* no hedging, no "it depends" without immediately giving the decision rule, and every number derived rather than recalled.
