### Explain to me why `temperature=0` still gives different answers on the same prompt. Take your time.

This is the flagship question of the section and the one that most cleanly separates people who have operated a serving stack from people who have read about one. The wrong answer — and I hear it constantly — is "GPUs are nondeterministic because of concurrent atomic adds." That is a real phenomenon in *training*, where scatter-adds into gradient buffers race. It is essentially not what is happening in inference.

The right mental model is: **floating-point addition is not associative, and the order in which a GPU kernel sums a reduction depends on the *shape* of the work it was given — which depends on who else was in your batch.** `(a+b)+c ≠ a+(b+c)` in fp32; the difference is at the last-bit level, around 1e-7 relative. A GPU matmul or a norm reduction splits its sum across thread blocks, and the *number* of splits, the tile size, and whether the kernel uses a split-K strategy are all chosen by heuristics keyed on the matrix dimensions. In continuous batching, your request is co-batched with whatever else arrived in that microsecond, so the M dimension of every GEMM in the forward pass changes from run to run. Different M → different kernel configuration → different reduction order → bitwise-different logits.

Note carefully what this means. Your request is deterministic *given the batch*. It is not deterministic given only itself. The property you are missing has a name: **batch invariance** — that a kernel produce bitwise-identical results for a given row regardless of how many other rows accompany it. Standard cuBLAS/cuDNN/FlashAttention kernels are not batch-invariant, because giving up that property is what lets them pick the fastest configuration per shape.

Then autoregression amplifies. A logit perturbation of ~1e-6 only matters when it flips the argmax, which requires a near-tie at the top. Near-ties are not rare across a 128k vocabulary and 500 decode steps — think of the many positions where `" the"` and `" a"`, or two equivalent phrasings, sit within 1e-5 of each other. One flip changes the context for every subsequent token, and the two trajectories diverge irreversibly. So the observable behavior is: identical for 200 tokens, then completely different from token 201 onward. That signature — long identical prefix, sudden total divergence — is diagnostic, and it is the thing to describe out loud.

Three secondary sources compound it and are worth naming because they are engine-level, not hardware-level. **Chunked prefill**: a long prompt is split into chunks whose boundaries depend on scheduler load, and different chunking gives different reduction shapes. **Prefix caching**: a cache hit reuses KV computed under one chunking; a miss recomputes under another. The values differ in the last bits, so cache-hit and cache-miss paths for the *same* prompt produce slightly different logits. **MoE routing under a capacity factor**: which tokens get dropped from an over-subscribed expert depends on which other tokens are in the batch competing for it — that is a genuinely first-order, not last-bit, batch dependence.

**🗣 Say this in the room:** "It's not atomics. It's that fp addition isn't associative and GPU kernels pick their reduction strategy from the tensor shape, so co-batching your request with different neighbors changes the summation order and gives bitwise-different logits. Temperature 0 is argmax, and argmax on near-tied logits flips. Then autoregression amplifies one flipped token into a totally different completion. The fix is batch-invariant kernels, and it costs throughput."

**⚠ Trap:** promising a customer "we set temperature to 0, so the output is reproducible." I have seen this written into a contract. It is false on every hosted API and on every default self-hosted deployment. What you can honestly promise is described two questions down.

### Suppose I told you to actually fix that. What would you do, and what does it cost?

You make the kernels batch-invariant. That means every reduction in the forward pass must use a summation order that is a function of the *tensor's own shape*, not of the batch dimension or of a runtime heuristic.

Three families of kernel need the treatment, and each has a specific fix.

**Norms (RMSNorm/LayerNorm).** The reduction is over `d_model` within a row, so the batch dimension is trivially parallel — the danger is a kernel that switches to a split-row strategy when the batch is small, in order to keep the GPU busy. Fix: pin a single data-parallel strategy where one block owns one row, always, and accept lower utilization at batch 1.

**Matmuls.** The killer is split-K: when the batch dimension M is small, a GEMM kernel splits the K reduction across blocks and combines partials, changing the summation order. Fix: forbid split-K and pin one tile configuration for a given weight shape, regardless of M. You lose throughput at small batch, which is exactly where split-K was earning its keep.

**Attention.** FlashAttention-style kernels split the KV sequence across blocks and combine with an online-softmax rescale. If the kernel chooses the *number* of splits to fill the GPU, the split count varies with sequence length and batch, changing the order. Fix: choose a *fixed split size* rather than a fixed split count, so the boundaries land at the same KV positions regardless of how much other work is present. This also has to hold across the prefill/decode boundary and across a prefix-cache hit, which is the hardest part — the same KV positions must be reduced in the same groups whether they were just computed or loaded from cache.

**📄 Source:** the clearest public write-up of this is Thinking Machines Lab's 2025 engineering post "Defeating Nondeterminism in LLM Inference," which localized the cause as batch invariance rather than concurrency and shipped batch-invariant RMSNorm, matmul and attention kernels for vLLM. **📅 Volatile:** treat the specific throughput numbers below as illustrative of the *shape* of the cost, and re-measure on your stack.

**💰 Math:** the cost is real and roughly a 1.5–2× throughput hit in an unoptimized batch-invariant implementation, narrowing with kernel work. Take a deployment serving 200 M output tokens/month on H100s at, say, 2,500 tok/s aggregate per node. That is `200e6 / 2500 = 80,000 node-seconds = 22.2 node-hours/month`... clearly you are not capacity bound at that volume, so scale it: at 20 B output tokens/month you need `20e9 / 2500 / 3600 = 2,222 node-hours`, or about 3.1 nodes running continuously. A 1.8× slowdown takes that to 5.6 nodes. At an H100-node rate of roughly $25/hour that is `2.5 extra nodes × 730 h × $25 = $45,600/month` for determinism. That is the number you put in the doc, and then the product decision is whether determinism is worth $45k/month.

**⚠ Trap:** thinking batch-invariance also gives you determinism across *hardware or software versions*. It does not. Change the GPU model, the CUDA version, the kernel library, the tensor-parallel degree, or the quantization, and the numerics change. Batch invariance buys you "same input, same output, same deployment." Pin the deployment as part of the guarantee.

### Would you sell deterministic inference as a product feature? Argue both sides.

I would, for specific segments, and I would price it as a distinct tier rather than making it the default — and I want to make the argument in terms of who actually needs it, because "determinism is good" is not an argument.

**Who needs it.** Regulated workflows where an auditor asks "why did the system produce this output on March 3rd" and "we re-ran it and got something else" is not an acceptable answer — Harvey-style legal work product, credit and underwriting decisions at a Ramp or a Stripe, clinical documentation. Anything with a **reproducible-evidence** requirement. Also: RL training loops, where the sampler is the environment and nondeterministic rollouts turn an on-policy algorithm silently off-policy — this is a real correctness bug, not a nicety, and it is arguably the strongest technical case. And **debugging**: a nondeterministic system cannot be bisected, so every quality investigation costs 5× more.

**Who does not.** Consumer chat, where users expect variation and where a 1.8× cost increase directly compresses margin. Creative tools, where determinism is actively wrong. And anything where the *real* requirement is "stable enough," which caching solves better and cheaper than kernels do — for a fixed prompt, a response cache gives you bitwise determinism at zero inference cost, and for many enterprise surfaces that is the honest answer.

**The argument against making it default** is the $45k/month I derived above plus the latency: batch-invariant kernels are slower at small batch, which is exactly the interactive regime, so TTFT and inter-token latency both degrade for every user to serve the needs of a few. That is a bad trade to make globally.

**🗣 Say this in the room:** "I'd offer it as an enterprise tier, not a default. The people who need it are regulated workflows that need reproducible evidence and RL training loops where nondeterminism makes an on-policy algorithm off-policy. Everyone else is better served by a response cache. And I'd scope the guarantee honestly: same input, same output, *same pinned deployment* — a CUDA or TP-degree change voids it."

### A `seed` parameter exists on the API. What does it actually guarantee?

Much less than the name implies, and the gap is worth being precise about because customers read "seed" as "reproducible."

A seed controls exactly one thing: **the pseudo-random draw in the multinomial sampling step.** Given a fixed probability distribution over the vocabulary and a fixed seed, you get the same token. It does nothing about how that distribution was computed. So the seed removes the sampler's randomness while leaving all the batch-dependent numerical nondeterminism from the previous two questions completely intact.

Concretely: with a seed, at `temperature=0.7`, two runs that produce bitwise-identical logits will produce identical tokens. Two runs whose logits differ in the last bits will *still* produce different tokens, and now they diverge for a subtler reason — the sampler maps a uniform draw `u` onto the inverse CDF of the distribution, and a tiny change in the distribution moves a bin boundary past `u`. In fact seeded sampling is *more* fragile to numerical noise than greedy, because greedy only flips on a top-2 near-tie while inverse-CDF sampling flips whenever `u` lands near any bin boundary.

There is a second, sharper failure specific to batched serving: in some engines the RNG state is drawn from a **per-batch generator advanced by batch position**, so your seeded request gets a different random stream depending on where it landed in the batch. Engines that support per-request seeds keep a generator per sequence to avoid this; verify which one you have.

**📅 Volatile:** the provider surface here changes and you should verify it, not recite it. Broadly: OpenAI's `seed` is documented as best-effort and pairs with a `system_fingerprint` that changes when the backend changes — the honest reading is "if the fingerprint matches and you were lucky with batching, you probably get the same output." Some providers do not expose a seed at all. Self-hosted vLLM supports a per-request seed, which gives genuine sampler determinism but still not logit determinism without batch-invariant kernels.

**⚠ Trap:** using a seed to make an *eval* reproducible and concluding your eval is now low-variance. It is not. You have removed one variance source and left the batching one, and worse, you have now fixed the sampler to one particular trajectory — so your eval measures the quality of *one sample*, not the quality of the distribution. For eval you usually want the opposite: many seeds, and report a mean with a confidence interval.

**🗣 Say this in the room:** "Seed fixes the multinomial draw, not the logits. It's necessary but nowhere near sufficient for reproducibility, and on batched serving it doesn't even give you that unless the engine keeps per-sequence RNG state. I'd never build a compliance story on a seed alone."

### An enterprise customer opens a P1: "the same prompt returns different answers." Walk me through your triage.

I want to establish *which* kind of "different" before touching anything, because there are five distinct causes and they have nothing in common but the symptom.

**Step 0 — get the raw evidence.** Two full request bodies and two full response bodies, with headers, not a screenshot of the UI. More than half the time the requests are not actually identical.

**Step 1 — is the input actually identical?** Check for a timestamp or date in the system prompt (extremely common — "Today is 2026-08-01"), a session id, a user name, retrieved RAG chunks that changed because the index was updated, conversation history, a randomized few-shot sample, or a non-deterministic tool result injected as context. In my experience this is the answer roughly 50% of the time and it is not a model problem at all.

**Step 2 — are the sampling parameters what you think?** Log the effective parameters the gateway sent, not the ones the client set. Defaults get injected at three layers — SDK, gateway, provider — and a `temperature` that is `None` in your config may become 1.0 at the provider. If temperature > 0, the answer is "it's supposed to vary," and the conversation becomes about whether the *variation* is acceptable, which is a different (better) conversation.

**Step 3 — did the model change under you?** Check the model string. If it is an alias (`-latest`, or an undated name), the underlying weights can change without notice. Check any fingerprint field the provider returns. Pinned, dated model versions are non-negotiable for enterprise surfaces, and if this is the cause, the fix is a policy fix.

**Step 4 — at temperature 0, look at the divergence point.** Diff the two outputs character by character. **A long identical prefix followed by total divergence is the batch-invariance signature** — numerics flipped one token and autoregression did the rest. Divergence from token 1 is something else: different prompt, different model, different template, or a cache-key collision serving another tenant's cached response.

**Step 5 — cache and routing.** Is a semantic cache in the path returning a near-neighbor's answer? Is a router sending some requests to a different model tier based on load? Is a canary deployment serving 5% of traffic from a new version? These produce "sometimes different, sometimes identical" patterns that look like nondeterminism and are actually two populations.

**Step 6 — only now** do you talk about kernels, and the deliverable is not a fix, it is a scoped guarantee: what you can promise, at what cost, on what tier.

**🗣 Say this in the room:** "First I confirm the inputs are byte-identical, because half of these are a timestamp in the system prompt or a changed retrieval result. Then effective sampling params, then whether the model alias moved. If it's temperature 0 with identical inputs and I see a long identical prefix then a hard divergence, that's kernel batch-invariance and I explain the cost of fixing it rather than pretending we can."

### Explain self-consistency and give me the cost model.

Self-consistency is the observation that **for tasks with a checkable, canonical final answer, the marginal distribution over answers is more accurate than any single sampled trajectory.** Sample `k` reasoning chains at nonzero temperature, extract the final answer from each, and take the plurality vote. You are marginalizing over reasoning paths instead of committing to one — the model can reach 42 by three different valid routes and by one arithmetic slip, and the vote drowns the slip.

**📄 Paper:** Wang et al. (2023), "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (ICLR) — replaced greedy CoT decoding with sample-and-vote and reported large double-digit absolute accuracy gains on arithmetic and commonsense reasoning benchmarks with the frontier models of that era.

Two design constraints are non-negotiable. **Temperature must be > 0** — usually 0.6–0.8 — because at temperature 0 all `k` samples are the same trajectory and you have paid `k×` for nothing. And **the final answer must be extractable and comparable**, which is why it works for math, multiple choice, structured extraction and classification, and does not work for "write me a summary" — you cannot majority-vote over prose.

**💰 Math.** Take a reasoning task: 800-token prompt, 600-token chain of thought, at $3/Mtok input and $15/Mtok output (📅 verify current prices).

```
Single call:
  input   800 × $3e-6   = $0.0024
  output  600 × $15e-6  = $0.0090
  total                 = $0.0114

k = 5, using n=5 on one request (prefill shared, input billed once):
  input   800 × $3e-6         = $0.0024
  output  5 × 600 × $15e-6    = $0.0450
  total                       = $0.0474   →  4.16× cost, not 5×

k = 5 as five separate calls with prefix caching at a 90% input discount:
  input   $0.0024 + 4 × $0.00024 = $0.00336
  output  $0.0450
  total                          = $0.0484 → 4.25×
```

So the practical dial is roughly `k × output cost`, and input caching or `n>1` buys you back about 15% of the naive `5×`. At 100,000 reasoning calls/day: single-sample is `100,000 × $0.0114 = $1,140/day = $34,200/month`; k=5 is `$4,740/day = $142,200/month`. The delta is **$108,000/month**, and the only justification is a measured accuracy gain on your own eval, with a business value attached to each recovered point.

**Latency is the good news.** The `k` samples are independent, so they run concurrently — self-consistency is a *cost* dial, not a *latency* dial, as long as you have capacity. Under capacity pressure it becomes both, because you have multiplied your token demand by 5 and the queue depth reflects it.

**⚠ Trap:** applying self-consistency to modern reasoning models and expecting the same lift. Models with extensive RL-trained chain of thought have internalized a lot of what sample-and-vote was recovering; the marginal gain from `k=5` on top of a reasoning model is typically much smaller than the gains reported in 2022–2023, sometimes within noise. Measure it on your task before you sign up for a 4× bill. This is the single most common way I see the technique cargo-culted.

### Self-consistency versus best-of-n with a reward model — which do you pick?

They are the same family — spend `k×` compute at inference to buy accuracy — and they differ in *how you pick the winner*, which determines where each one applies.

**Majority vote (self-consistency)** needs the answers to be *comparable and discrete*. It has no model dependency, no extra infrastructure, and no training. It fails silently when the model is *consistently* wrong: if a systematic misreading of the question produces the same wrong answer four times out of five, the vote confidently ratifies it. Voting reduces variance, not bias — say exactly that, it is the crisp version of the limitation.

**Best-of-n with a verifier** samples `k` candidates and scores each with a separate model or program, then returns the top-scored one. Its great advantage is that it applies to *unstructured* outputs — you can best-of-n a summary, a code patch, an email draft — where voting is meaningless. Its cost is that you need the scorer, and the quality ceiling is the scorer's, not the generator's.

The decision rule I actually use:

- Is there a **deterministic verifier**? Unit tests, a compiler, a JSON-schema validator, a SQL EXPLAIN, a MIP solver checking feasibility? Then best-of-n against the verifier, always, and it is the strongest technique in this section — you get a hard correctness signal rather than a learned proxy. A code agent that samples 5 patches and keeps the one that passes the test suite is doing this.
- Is the answer **discrete and canonicalizable** (a number, a label, an entity)? Majority vote — free, no scorer to maintain, no reward hacking.
- Is the output **free-form** with no verifier? Best-of-n with a reward model or an LLM judge, and now you own an evaluation problem: the scorer needs its own eval, and `k` large enough will find and exploit the scorer's blind spots. Reward over-optimization is real; the gain from best-of-n is typically non-monotone in `k`, rising and then falling as you select harder against a flawed proxy. I would not run `k > 8` against a learned reward model without measuring the turn-over point.
- Neither? Then you are not choosing between these, you are choosing to spend the compute on a better model or better context instead — which is usually the right answer at the same price.

**💰 Math:** best-of-8 on a code task with a test-suite verifier: 8 × the generation cost plus 8 test runs. If generation is $0.02 and a sandboxed test run is $0.001 of compute, that is `8 × 0.021 = $0.168` versus `$0.021`. If pass@1 is 55% and pass@8-with-verifier-selection is 82%, you paid 8× to convert 27% of failures into successes. Whether that is a good trade is entirely a question of what a failed patch costs downstream — for an autonomous agent that opens a PR a human must review, one avoided bad PR is worth many dollars of inference.

### How do sampling parameters interact with prefix caching?

The clean statement: **prefix caching operates on the KV tensors, which are a function of the input tokens only. Sampling parameters do not affect what is cacheable and are not part of the cache key.** Temperature, top-p, penalties and seeds all live strictly downstream of the KV cache, in the sampler. So you can serve the same cached prefix to a request at `T=0` and one at `T=1.2` with no interaction at all. If someone tells you "we can't cache because we sample," they have the layering wrong.

What *does* interact, and matters:

**`n > 1` shares the prefill.** Requesting 5 samples from one prompt in a single request lets the engine compute the prefix KV once and fork `k` decode streams from it. That is the mechanism behind the "4.16×, not 5×" arithmetic in the self-consistency answer, and in a paged-attention engine it is even better than the billing suggests: the `k` sequences share the same physical KV blocks for the prompt via copy-on-write, so the *memory* cost is one prompt plus `k` short suffixes, not `k` prompts. On an 8k prompt with a 600-token continuation at k=5, naive is `5 × 8600 = 43,000` cached positions; shared is `8000 + 5 × 600 = 11,000` — a 3.9× memory saving that directly becomes concurrency.

**Cache hits change numerics.** This is the subtle one and it connects back to determinism. KV computed during a fresh prefill and KV loaded from a prefix cache are not guaranteed bitwise identical, because the fresh computation may have been chunked differently. So the *same* prompt at `temperature=0` can produce different output depending on whether it hit the prefix cache. If you are chasing a determinism bug, `cache_hit` is a variable you must log and control for. Nobody logs it. Start logging it.

**Penalties over a cached prefix.** If your repetition penalty is scoped over the full sequence including the prompt, and the prompt is a long cached system prompt, then every request pays a penalty derived from tokens it did not generate — and the effect scales with how long your cached prefix is. Another argument for scoping penalties to the generated window.

**⚠ Trap:** putting sampling parameters into a *response* cache key and then not putting them into the *prompt* cache reasoning. The two caches are different objects. A response cache keyed on `(prompt, model, temperature, top_p, seed, ...)` is correct but will almost never hit at nonzero temperature — that is fine and expected. A KV prefix cache keyed on token ids alone is also correct. Conflating them produces either a cache that never hits or one that serves a `T=0` answer to a `T=1` request.

### Speculative decoding is on. How does my sampling configuration affect it?

Speculative decoding runs a small draft model to propose `γ` tokens, then verifies all of them with one forward pass of the target model, accepting a prefix of the proposal. The essential and non-obvious property is that **it is exactly distribution-preserving** — the accepted tokens are drawn from the target model's distribution, not an approximation of it. That is what makes it safe to turn on.

The mechanism is modified rejection sampling. Draft proposes token `x` from its distribution `q`. Target computes `p` at that position. Accept with probability `min(1, p(x)/q(x))`. On rejection, sample the replacement from the normalized residual `max(0, p − q) / Σ max(0, p − q)`. The algebra works out so the marginal is exactly `p`.

**📄 Paper:** Leviathan, Kalman & Matias (2023), "Fast Inference from Transformers via Speculative Decoding" (ICML), with Chen et al. (2023), "Accelerating Large Language Model Decoding with Speculative Sampling" arriving independently. Both establish the exactness of the acceptance rule; that exactness is the contribution, since draft-and-check heuristics existed before.

Now the sampling interactions, which are what the question is really asking.

**Temperature changes acceptance rate.** Expected acceptance per proposed token is `1 − TV(p, q)`, the total-variation distance between target and draft. In the practical range, acceptance is highest near greedy — where a good draft agrees with the target's argmax perhaps 70–85% of the time — and degrades as temperature rises, because the two distributions' tails diverge more than their modes do. The operational consequence: your measured speedup is not a constant. A benchmark run at `T=0` and a production surface at `T=0.8` will not see the same numbers, and teams routinely report a regression that is just this.

**Logits processors must be applied to both models, identically.** This is the failure I have actually seen. If your repetition penalty, logit bias or min-p filter is applied to the target's logits but not the draft's, then `q` is the unpenalized distribution and `p` is the penalized one — you have manufactured TV distance out of nothing, and acceptance collapses. The generation stays *correct*, because the rejection rule guarantees the target distribution regardless, but the speedup evaporates and you are now paying for the draft model's forward passes for no benefit. Symptom: "spec decode gives us 1.05× instead of 2.1×." Check processor parity first.

**Constrained decoding is the extreme version.** A grammar mask sets most of the target's logits to `-inf`. If the draft is unconstrained, it proposes tokens the grammar forbids, `p(x) = 0`, and every such token is rejected. Acceptance can fall toward the floor. The correct implementation runs the same FSM mask over the draft's logits too — which means your grammar engine has to advance state speculatively and roll back on rejection. Not every stack does this; verify before assuming spec decode and structured outputs compose.

**📐 Numbers you must know:** with acceptance rate `α` and draft length `γ`, expected tokens per target forward pass is `(1 − α^(γ+1)) / (1 − α)`. At `α = 0.8, γ = 4`: `(1 − 0.8^5)/(1 − 0.8) = (1 − 0.328)/0.2 = 3.36` tokens per target pass. Subtract the draft's own cost — if the draft is 1/10 the size, four draft passes cost ~0.4 target-equivalents — giving a net speedup of about `3.36 / 1.4 = 2.4×`. At `α = 0.5`: `(1 − 0.031)/0.5 = 1.94` tokens per pass, net `1.94/1.4 = 1.39×`. Acceptance rate is the whole ballgame, and sampling configuration moves it.

### Where does a constrained-decoding grammar mask sit in the sampler pipeline, and what does it do to my sampling parameters?

The mask goes **first among the truncation stages and before the final softmax** — it sets the logits of grammar-illegal tokens to `-inf`, and everything else operates on the survivors. The ordering is not aesthetic: a grammar constraint is a hard correctness requirement while top-p is a soft quality preference, so the hard constraint must be applied to the raw distribution and the soft one applied within the legal set.

Get that order backwards and you get the pathology worth being able to describe: apply top-p first, keeping the nucleus, then apply the grammar mask to the nucleus — and if no legal token was in the nucleus, you have an empty candidate set. The engine's options are all bad: NaN, fall back to the argmax of the full masked distribution (silently ignoring top-p), or error. Correct ordering makes this structurally impossible, because the grammar mask always leaves at least one legal token (a well-formed grammar guarantees it) and top-p applied afterwards always keeps the top survivor.

What the mask does to your effective parameters is the more interesting half. **Masking renormalizes, so the meaning of `top_p` changes.** Suppose at some position the model's true distribution puts 0.85 on `" the"` — illegal under the grammar, which requires a `"` here — and 0.02 on `"\""`. After masking and renormalization, `"\""` might carry 0.6. Your `top_p = 0.9` now admits a nucleus computed over a completely different distribution than the model produced. Practically this means **truncation parameters are much less meaningful under constraint**, because the constraint has already done the truncation, and it did it with a hard guarantee rather than a probabilistic one.

The rule I ship: **under structured decoding, drop temperature to 0–0.2 and set `top_p = 1.0`.** You have a grammar guaranteeing well-formedness; what you want from the sampler is the model's best guess at the *content* within that form, and there is no creativity budget to spend on a field value in an extraction task. Adding sampling noise on top of a constraint mostly buys you wrong-but-well-formed values, which is the worst failure class because it passes validation.

**⚠ Trap:** the constraint mask also interacts with penalties, and badly. A repetition penalty over a JSON output penalizes `"`, `:`, `,`, `{` and `}` — the tokens the grammar *requires*. The grammar then forces them anyway (they are the only legal option, so the penalty is irrelevant), but at positions where the grammar allows several tokens, the penalty is now steering among structural options for no reason. Set penalties to identity under structured output. Every time.

**🗣 Say this in the room:** "Grammar mask first, then temperature and truncation over the legal set — never the reverse, or you can end up with an empty candidate set. And under constraint I set temperature near zero, `top_p=1.0`, and all penalties off: the grammar owns the form, and I don't want the sampler adding entropy to field values."

### Give me your default sampling parameters for extraction, chat, code and creative work, and defend each.

I want to give the reasoning rather than a table, because the table is worthless without it. The single question behind every one of these is: **how many outputs are acceptable here, and what does a bad draw cost?**

**Structured extraction, classification, routing, tool-argument generation.** `temperature = 0`, `top_p = 1.0`, all penalties off, structured output or a grammar on. There is exactly one right answer — the value in the document, the correct label, the right function name — and sampling can only move you off it. Every unit of entropy here is pure downside. The one nuance: if you are running self-consistency over extraction to catch errors, you need `T ≈ 0.5`, but then you are deliberately buying variance to vote it away.

**General chat and assistant responses.** `temperature = 0.7`, `top_p = 0.9`, penalties off. This is the community default for a reason: 0.7 preserves the model's ranking strongly enough for factual reliability while producing enough variation that repeated questions do not feel canned, and 0.9 clips the tail that generates non-sequiturs. If your chat surface is customer-facing and factual — a support agent at Sierra, a docs assistant at Notion — I drop to `T = 0.3`, because the value of variety is near zero and the cost of a fabricated detail is high.

**Code generation.** `temperature = 0.0` to `0.2`, `top_p = 0.95`, and **all repetition machinery off** for the reasons I gave earlier. Code is verifiable, so if you want diversity, get it by sampling `n` candidates at `T ≈ 0.6` and selecting with the test suite — not by raising the temperature of a single sample, which just gives you one worse sample. That distinction (diversity for *selection* versus diversity in the *output*) is the senior framing. A Cursor-style inline completion runs near-greedy because a single wrong token in a completion is a bad experience and there is no verifier in the loop.

**Creative writing, brainstorming, ideation.** `temperature = 1.0`–`1.2` with `min_p ≈ 0.05` if the stack supports it, otherwise `top_p = 0.95`. Min-p is genuinely better here for the reason established earlier — it stays anchored to the model's peak, so high temperature widens the *shape* of the distribution without admitting the absurd tail. A modest `presence_penalty` around 0.3 is defensible for brainstorming specifically, where the goal is topical coverage rather than fluent prose.

**Agentic loops with tool calls.** `temperature = 0` on the tool-selection and argument-generation steps, and if you want the model's reasoning to be more exploratory, that belongs in a separate call at a separate temperature. Mixing them means a stochastic draw picks your tool, and a wrong tool call costs a round trip, tokens, and possibly a side effect. I have seen an agent at `T=0.9` call `delete_record` instead of `get_record` because they were near-tied in the model's distribution. That is a sampling incident with a data-loss consequence.

**⚠ Trap:** setting temperature and top-p both aggressively low — `T = 0.2, top_p = 0.5` — under the belief that the effects add up to "more deterministic." They compound in a way that is hard to reason about: low temperature already concentrates mass on the top token, so top-p at 0.5 will usually admit exactly one candidate, and you have written a very confusing spelling of greedy decoding. Pick one truncation mechanism and one temperature; if you want determinism, say `temperature = 0` and leave `top_p = 1.0`.
