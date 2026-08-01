### Someone says "softmax is just a Boltzmann distribution." Unpack that, and tell me what temperature physically is.

Mental model: **the softmax is the maximum-entropy distribution consistent with a set of energies, and the logits are negative energies.** In statistical mechanics, a system in thermal equilibrium occupies state i with probability p_i ∝ exp(−E_i / kT). Set E_i = −z_i and absorb k, and you get p_i ∝ exp(z_i/T), which is the softmax with temperature. That is not an analogy — it is the same functional form derived from the same variational principle (maximize entropy subject to a fixed expected energy), and knowing that is why the temperature parameter behaves the way it does rather than being a magic knob.

The physics gives you the intuition for free. **Temperature is inverse inverse-coldness**: write β = 1/T and p_i ∝ exp(βz_i). Large β (low temperature) means the system settles into its lowest-energy state — the argmax. Small β (high temperature) means thermal noise swamps the energy differences and every state becomes equiprobable. The entropy of the distribution is monotonically increasing in T. In sampling terms:

- **T → 0⁺:** the distribution converges to a point mass on argmax(z). Greedy decoding.
- **T = 1:** the model's calibrated distribution, i.e. the one it was trained to produce by minimizing cross-entropy.
- **T → ∞:** uniform over the vocabulary. Every logit difference is divided into irrelevance.

The mechanism at the tensor level is trivially cheap: divide the logit vector by T before the softmax. One elementwise op on a `[V]` vector, ~128k FLOPs, utterly free compared to the forward pass that produced the logits.

There is one non-obvious consequence worth stating unprompted, because it is what makes temperature different from top-p. **Temperature is a monotone transformation — it never changes the ranking of tokens, only the gaps.** So T can never make an impossible token likely relative to another; it can only redistribute mass along the existing order. Top-k and top-p, by contrast, are *truncations* — they set tails to exactly zero, changing the support. That's why the two compose usefully: top-p removes the garbage tail, temperature reshapes what's left.

**⚠ Trap:** treating temperature as a "creativity" dial in a design doc without saying what it does to the failure distribution. Raising T raises entropy, and in a structured-output or tool-calling path the extra entropy lands on schema violations and hallucinated argument names. My rule in review: any code path that parses the model's output into a typed object runs at T = 0 unless there is an eval showing otherwise, and any path whose output a human reads may run hotter.

### What actually happens at temperature 0, and why do people say "temperature 0 is a lie"?

At T = 0 the expression exp(z_i/T) is a division by zero, so no provider literally computes it — the implementation branches to `argmax` and skips sampling entirely. That part is fine. The lie is in what people *infer* from it: that T = 0 gives you a deterministic, reproducible function from prompt to output. It does not, and this catches backend engineers hard because determinism is something we're used to being able to demand.

The sources of nondeterminism, in the order I'd check them:

**Floating-point non-associativity under variable batching.** A GPU reduction sums partial products in an order determined by how the kernel splits work across thread blocks, and that split depends on the shape of the batch. Your request landing in a batch of 3 versus a batch of 47 produces a *different summation order*, and (a+b)+c ≠ a+(b+c) in floating point. The logits differ in the last few bits. Usually irrelevant — but when the top two tokens are within that epsilon, argmax flips, and because generation is autoregressive, one flipped token at position 40 produces a completely different completion from position 41 onward. **This is the dominant cause, and it is invisible in single-request testing** because you only see it under production batching.

**Mixture-of-experts routing.** In implementations with per-expert capacity limits, which tokens get routed where depends on the other tokens in the batch. Your token can be dropped from its preferred expert because someone else's request filled it. Batch composition becomes an input to your output.

**Atomics and non-deterministic kernels.** Some reduction kernels use atomic adds whose ordering is not fixed run to run.

**Infrastructure drift.** A different GPU SKU, a different engine version, a different tensor-parallel degree, or a silently-updated model behind a stable API name — all change the numerics. **📅 Volatile:** provider-side model updates behind an unversioned alias are a real and recurring cause of "my prompt stopped working"; pin versioned model IDs where the provider offers them.

**🗣 Say this in the room:** "Temperature 0 is greedy decoding, not determinism. Floating-point reductions aren't associative, and the reduction order depends on batch shape, so the same prompt in a batch of 3 and a batch of 50 can produce different logits in the last bits. When the top two tokens are close, argmax flips and autoregression amplifies it into a totally different answer. So I never write a test that asserts exact output equality — I assert on a property, a schema, or a scored threshold."

**⚠ Trap:** golden-output snapshot tests in CI against a hosted model. They pass on your machine, then flake at 2% in CI, and the team's response is to add retries — which hides a real regression the day the provider does update the model. Test properties and evals, not strings. This is the single most common testing mistake a strong backend engineer makes on their first LLM system.

### Define entropy and cross-entropy in the units the training loop actually prints. Connect them to the loss I see on the dashboard.

Entropy H(p) = −Σ_i p_i log p_i is the average surprise of a distribution — the expected number of nats (if log is natural) or bits (if log₂) needed to encode a sample from p under an optimal code. Cross-entropy H(p, q) = −Σ_i p_i log q_i is the average cost of encoding samples from p using a code built for q. It is always ≥ H(p), and the gap is exactly the KL divergence: **H(p,q) = H(p) + D_KL(p‖q)**. That identity is the whole reason "minimizing cross-entropy" and "matching the data distribution" are the same sentence.

In a language model, p is the empirical one-hot target — the actual next token — so H(p) = 0 and cross-entropy *equals* the KL divergence from the data to the model. The per-token loss collapses to −log q(y_true), and the reported training loss is the mean of that over the batch:

  loss = −(1/N) Σ_{t} log q_θ(y_t | y_{<t}),  **in nats**, because PyTorch's `cross_entropy` uses natural log.

That gives you the conversion table you need for reading a dashboard:

- **loss in nats × 1.4427 = loss in bits** (1/ln 2).
- **perplexity = exp(loss_in_nats)**.
- Loss 2.0 nats = 2.885 bits/token = perplexity 7.39.
- Loss 1.6 nats = 2.31 bits/token = perplexity 4.95.
- **A uniform model over a 128k vocabulary has loss ln(128256) = 11.76 nats.** That's your step-0 sanity check: if your loss doesn't start within a hair of ln(V), your initialization or your label alignment is broken. This one check has saved me more debugging hours than any other.

Two subtleties that come up. **Masking** — padding positions and prompt tokens in an instruction-tuning setup must be excluded via `ignore_index`, and getting that wrong silently rescales your loss by the fraction of masked tokens, so your loss curve looks fine and your model learns to predict padding. **Reduction** — `mean` over a batch with variable-length sequences weights short sequences more per token than a global token-mean does; for gradient-accumulation correctness you want to sum the token losses and divide by the global token count, not average the per-microbatch means.

**⚠ Trap:** comparing loss numbers across runs with different tokenizers or different sequence packing. Loss is per-token, and "a token" is not a fixed unit. More on this below — it is the single most common information-theory error in this field.

### Forward KL versus reverse KL. Which one does pretraining use, which one does RLHF use, and what's the behavioral difference?

Mental model: **forward KL is a coward and reverse KL is a specialist.** Forward KL punishes you for assigning low probability to anything the target does; reverse KL punishes you for assigning probability to anything the target doesn't. One makes you hedge, the other makes you commit.

The asymmetry falls straight out of which distribution is doing the weighting.

**Forward KL, D(p ‖ q) = Σ p log(p/q).** The expectation is under p, the target. Wherever p has mass and q is near zero, log(p/q) → ∞ and the loss explodes. So q is *forced to cover every mode of p* — it is **zero-avoiding / mode-covering**. Where p has no mass, q is unpenalized for putting some there. Result: a broad, hedging q that spreads mass over the union of the target's modes and smears across the valleys between them.

**Reverse KL, D(q ‖ p) = Σ q log(q/p).** The expectation is under q, your own model. Wherever q has mass and p is near zero, you pay heavily — so q is **zero-forcing / mode-seeking**. But wherever p has mass and q has none, the integrand is 0·log 0 = 0 and there is no penalty at all. Result: q collapses onto one high-probability mode of p and confidently ignores the rest.

Where each lives:

- **Pretraining and SFT use forward KL.** Maximum-likelihood on next-token prediction is exactly minimizing D(p_data ‖ q_θ). This is why base models hedge — they are trained to cover, not to commit, and it's the information-theoretic root of "the model gives you a wishy-washy answer that mentions every possibility."
- **RLHF's regularizer uses reverse KL.** The PPO objective is maximize E_{q_θ}[r(x)] − β·D(q_θ ‖ π_ref), with the expectation under the *learned* policy — mode-seeking with respect to the reference. That is intentional: you want the aligned policy to commit to the high-reward mode, not to keep covering everything the base model would have said.
- **Distillation defaults to forward KL** (match the teacher's full distribution) and there is a real line of work that switches to reverse KL specifically to stop students from hedging over teacher modes they don't have the capacity to represent. When a small student must approximate a big teacher, forward KL makes it smear and reverse KL makes it pick — and for generation quality, picking usually wins.
- **DPO's implicit reward is a KL artifact.** The closed-form optimum of "maximize reward subject to a reverse-KL leash" is π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β). Rearranged, r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + const. That log-ratio *is* the reward, which is what lets DPO train directly on preference pairs with no separate reward model.

**🗣 Say this in the room:** "Forward KL takes the expectation under the data, so it's zero-avoiding — the model must cover every mode, which is why MLE-trained base models hedge. Reverse KL takes the expectation under the model, so it's zero-forcing — the model commits to one mode and ignores the rest. That's why RLHF uses reverse KL as its leash: you want the aligned policy to pick the high-reward mode, not to keep covering everything."

**⚠ Trap:** calling KL a distance. It is not symmetric and does not satisfy the triangle inequality. Saying "the KL distance between the models" in a room with a research engineer is a small but real credibility cost, and the asymmetry is exactly the thing that makes it useful here.

### Where does the KL term actually sit in an RLHF objective, and what breaks if I set β too low or too high?

The objective is: maximize over θ, E_{x∼D, y∼π_θ(·|x)} [ r_φ(x, y) ] − β · D_KL(π_θ(·|x) ‖ π_ref(·|x)). The reward model r_φ scores completions; π_ref is the frozen SFT checkpoint; β is the leash length. In implementation, the KL is not computed exactly over the vocabulary at every position (that would be a `[T, 128k]` reduction per sample, and it's usually estimated on the sampled tokens instead) and it is commonly folded into the per-token reward as a shaping term rather than kept as a separate loss.

**β too low** — the leash is long, and you get **reward hacking**. The policy discovers regions of output space where the reward model is wrong, because the reward model was trained on completions from π_ref's distribution and has no idea what to do off-distribution. The classic signatures: outputs get longer and longer (length is correlated with human preference in most preference datasets, so the RM learned "longer = better"), the model develops verbal tics that happen to score well, sycophancy increases, and eventually the text degrades into something that scores 9/10 on the RM and is unreadable. Your monitoring signal is **the gap between RM score and held-out human/judge score widening over training steps** — RM score climbing while your independent eval falls is the definitive diagnosis.

**β too high** — the leash is short, and nothing happens. The policy stays glued to π_ref, KL stays near zero, the reward barely moves, and you've spent a lot of GPU-hours producing your SFT model back. Less obviously, an overly tight leash can *look* like it's working for a while and then plateau early, which people misdiagnose as "the reward model isn't good enough."

The practical loop: **treat KL as a controlled variable, not a hyperparameter.** Plot KL(π_θ ‖ π_ref) against training step and against reward. What you want is a monotone reward increase at bounded KL — a "reward per nat of KL" efficiency curve. Many production setups use an adaptive controller that adjusts β to hold KL near a target rather than fixing β. And regardless of β, keep a held-out eval that the reward model never saw, evaluated by something other than the reward model, as the actual stopping criterion.

**⚠ Trap:** believing GRPO/DPO-style methods removed this problem. They moved it. DPO's β is the same leash — set it too low and you get degenerate outputs that maximize the implicit reward margin; the failure just looks different (probability mass collapsing off *both* chosen and rejected responses is a documented DPO pathology). Any method with a reference-model regularizer has a β, and a β always has these two failure modes at its ends.

### What's Jensen–Shannon divergence, and why does it barely show up in LLM training?

JSD is the symmetrized, bounded cousin of KL: with m = (p + q)/2, JSD(p, q) = ½D(p‖m) + ½D(q‖m). Two properties make it attractive on paper. It is symmetric, so there's no forward/reverse choice to argue about. And it is bounded — by log 2 with natural log, or exactly 1 bit with log₂ — so it never explodes when the supports disagree, unlike KL, which goes infinite the moment p has mass where q has none. Its square root, √JSD, is a genuine metric satisfying the triangle inequality.

It barely shows up in LLM training for a simple reason: **the boundedness that makes it safe also makes its gradients weak exactly where you need them strong.** If your model assigns near-zero probability to the true token, forward KL screams; JSD shrugs, because the mixture m still has half the target's mass there. For maximum-likelihood training you *want* the unbounded penalty — that's the learning signal. JSD's boundedness was the reason it appeared in the GAN literature (where an unbounded objective destabilizes the discriminator), and LLM training doesn't have that problem.

Where you do see it in an applied LLM stack is **outside the training loop, as a measurement**:

- **Distribution drift monitoring.** Compute the token or embedding-cluster distribution of this week's production traffic against last month's; JSD gives you a bounded, symmetric, interpretable-in-bits number you can alert on. KL would go infinite the first time a new token appears.
- **Diversity metrics** across generations — JSD between the n-gram distributions of two decoding configurations.
- **Ensemble/routing agreement** — how far apart are two models' output distributions on the same prompt, as a routing or escalation signal.

**⚠ Trap:** alerting on JSD without a baseline distribution of JSD. A JSD of 0.05 bits means nothing in isolation — you need the week-over-week JSD distribution during a known-healthy period to know whether 0.05 is Tuesday or an incident. This is the same discipline as any drift metric; the failure mode is a threshold picked by vibes.

### Mutual information — give me a real use for it in an LLM system.

I(X;Y) = H(X) − H(X|Y) is "how many bits does knowing Y save you when encoding X," and equivalently D_KL(p(x,y) ‖ p(x)p(y)) — the KL between the joint and the product of marginals, so it's exactly zero iff X and Y are independent. It's symmetric, non-negative, and measures *any* dependence, not just linear correlation, which is what distinguishes it from correlation coefficients.

Four places it genuinely earns its keep in this stack:

**Contrastive embedding training.** The InfoNCE objective — the loss behind essentially every modern text embedding model — is a lower bound on the mutual information between the two views (query and positive passage). Maximizing InfoNCE over a batch of size K is maximizing a bound that saturates at log K, which is the formal reason **larger batch sizes produce better embedding models**: the bound you can achieve is capped by log(batch size), so a batch of 64 caps you at 6 bits of MI regardless of how good your encoder is. **📄 Paper:** van den Oord et al. (2018), *Representation Learning with Contrastive Predictive Coding* — introduced InfoNCE and proved the log-K mutual-information bound; it replaced ad-hoc triplet losses as the standard contrastive objective.

**Pointwise mutual information in retrieval.** PMI(term, doc) = log[p(term,doc)/(p(term)p(doc))] is, structurally, what IDF weighting approximates. When you're debugging why BM25 beats your dense retriever on a corpus, the answer is usually that the corpus has high-PMI rare terms — part numbers, error codes, proper nouns — that the embedding model has smeared into a generic subspace.

**Eval-set construction and contamination checks.** MI between a candidate feature (query length, language, tenant) and the outcome (task success) tells you which slice actually explains your failures, which is how you decide what to stratify your eval on rather than guessing.

**Deduplication and diversity in data curation.** Selecting a training or few-shot subset that maximizes information about the target task while minimizing redundancy among the selected items is, formally, an MI-maximization problem, and submodular selection algorithms operationalize it.

**⚠ Trap:** estimating MI from a few thousand samples in a high-dimensional continuous space and believing the number. MI estimation is notoriously biased upward with limited samples, and the neural MI estimators have known variance pathologies. Use it as a *ranking* signal between candidate features on the same sample size, not as an absolute quantity you report to a stakeholder.

### Define perplexity, derive it from cross-entropy, and tell me what perplexity 8 actually means.

Perplexity is the exponentiated mean negative log-likelihood:

  **PPL = exp( −(1/N) Σ_{t=1}^{N} log p(x_t | x_{<t}) ) = exp(mean NLL in nats) = 2^(mean NLL in bits)**.

The derivation is one line: the model's per-token cross-entropy loss *is* the mean NLL, so PPL is just `exp(loss)`. Which is why, on any PyTorch training run, `math.exp(loss.item())` is your perplexity and you never need a separate metric.

The interpretation is the part worth being crisp about. **Perplexity is the effective branching factor** — the size of the uniform distribution that would leave you equally uncertain. PPL = 8 means that, averaged over the corpus, the model is as uncertain as if it were choosing uniformly among 8 equally-likely tokens at each position. A model with PPL = 1 predicts the text perfectly. A model that hasn't learned anything on a 128k vocabulary has PPL = 128,256 (loss ln V = 11.76 nats).

Some anchors for calibration. A well-trained modern model on general English web text lands somewhere in the single digits to low teens on a held-out sample of its own distribution — but that range is nearly meaningless without knowing the tokenizer and the eval corpus, which is the next question. On a *very* predictable corpus (repetitive logs, structured JSON), perplexity in the low single digits is normal. On a corpus of dense technical prose or a language underrepresented in training, double digits.

Two mechanical cautions. **Perplexity is corpus-dependent to a degree people underestimate** — the same model on Wikipedia versus on a private codebase can differ by 3× — so PPL numbers are only comparable within a fixed eval corpus. And **stride matters**: evaluating a long document by chopping it into disjoint 2,048-token windows means every window's first tokens are predicted with no context, inflating perplexity. The correct procedure is a sliding window with a stride, scoring only the tokens that had full context. Getting the stride wrong is a common way to "discover" that your fine-tune made the model worse.

### I have two models. Model A has perplexity 6.2 and Model B has 7.9. Which one is better?

**You cannot tell, and if the two models have different tokenizers, the question is not just unanswerable but malformed.** This is the flagship information-theory trap in this field and it is asked deliberately.

Here's why. Perplexity is per *token*, and a token is not a unit of language — it is a unit of a particular BPE vocabulary. A tokenizer that produces longer tokens has fewer, harder predictions; a tokenizer that produces shorter tokens has more, easier ones. The *total* information content of the text is fixed; how you slice it into prediction steps is not.

**💰 Worked demonstration.** Suppose two models are *exactly equally good* — both compress the eval corpus to 0.75 bits per byte. Model A's tokenizer averages 3.5 bytes per token; Model B's averages 4.5.

- Model A's per-token loss = 0.75 bits/byte × 3.5 bytes/token = 2.625 bits → PPL = 2^2.625 = **6.17**.
- Model B's per-token loss = 0.75 × 4.5 = 3.375 bits → PPL = 2^3.375 = **10.37**.

Identical modeling quality; Model B's perplexity is 68% higher. Under the naive reading you'd ship A and be wrong. In fact if A had *better* PPL than B on the same corpus with those tokenizers, A could still be the worse model — you'd need to convert to a common unit to know.

So what do I actually ask before answering?

1. **Same tokenizer?** If not, convert to bits-per-byte and compare that. If yes, proceed.
2. **Same eval corpus, byte-for-byte, with the same preprocessing?** Different whitespace normalization alone moves PPL by percent.
3. **Same stride and context length?** A 2,048-window evaluation and an 8,192-sliding-window evaluation of the same model give different numbers.
4. **Is the eval corpus in either model's training data?** Contamination makes perplexity arbitrarily low and is the easiest metric in the world to game.
5. **And the real question: does perplexity predict what I care about?** Below a certain quality floor it correlates well with downstream capability; among frontier-adjacent models it is a weak predictor of instruction-following, tool use, or reasoning. An RLHF'd chat model typically has *worse* perplexity on raw web text than its own base model, because alignment moved it off the pretraining distribution. It is not worse.

**🗣 Say this in the room:** "I'd refuse to answer until I know whether they share a tokenizer. Perplexity is per token and tokens aren't a fixed unit — two identically-good models can differ 60% in perplexity purely from average token length. I'd convert both to bits-per-byte, which is tokenizer-invariant, and even then I'd only use it as a pretraining health metric, not as a decision metric for a product model."

**⚠ Trap:** the corollary trap, which is subtler and gets senior candidates: using perplexity to compare a base model against its own RLHF'd or instruction-tuned descendant. Same tokenizer, same corpus, and the comparison is still invalid, because post-training deliberately shifts the output distribution away from the raw-text distribution the perplexity is measured against. Perplexity going up after post-training is expected, not a regression.

### Then give me bits-per-byte. Derive it and show me the conversion from a training loss.

Bits-per-byte is the tokenizer-invariant version of perplexity: instead of "how surprised am I per token," it asks "how many bits do I need per byte of the original UTF-8 text." Since bytes are a property of the text and not of your vocabulary, two models with different tokenizers become directly comparable. It is, literally, a compression rate — a model with 0.75 BPB compresses the corpus to 0.75/8 = 9.4% of its original size, which is a legitimate and rather beautiful way to think about what a language model is.

The derivation. Total negative log-likelihood over the corpus, in nats, is Σ_t NLL_t. Convert to bits by dividing by ln 2. Divide by the number of *bytes* in the raw text:

  **BPB = (Σ_t NLL_t) / (ln 2 × n_bytes) = mean_NLL_nats / (ln 2 × bytes_per_token)**

where bytes_per_token = n_bytes / n_tokens for that tokenizer on that corpus.

Worked both directions:

- Training loss 2.0 nats/token, tokenizer averaging 4.0 bytes/token → BPB = 2.0 / (0.6931 × 4.0) = 2.0 / 2.7726 = **0.721 bits/byte**.
- Same model, and you want the perplexity a different tokenizer would report at 3.2 bytes/token → per-token loss = 0.721 × 0.6931 × 3.2 = 1.599 nats → **PPL = e^1.599 = 4.95**. Same model, same quality, PPL drops from 7.39 to 4.95 by changing nothing but the tokenizer.

```python
import math
def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    return total_nll_nats / (math.log(2) * n_bytes)

# accumulate over the eval set:
#   total_nll_nats += F.cross_entropy(logits, targets, reduction="sum").item()
#   n_bytes        += len(raw_text_chunk.encode("utf-8"))
```

The implementation detail people get wrong: **n_bytes must be the byte length of the raw text, not of the decoded tokens after any normalization.** If your tokenizer strips or adds whitespace, or your loader lowercases, you're dividing by the wrong denominator and your "invariant" metric isn't. Compute the byte count from the original file, before any preprocessing touches it.

**📐 Numbers you must know:** BPB and PPL relate as BPB = log₂(PPL) / bytes_per_token. Typical BPE tokenizers for English average roughly 3.5–4.5 bytes per token; code tokenizers on code are lower; non-Latin scripts under an English-centric tokenizer can drop toward 1–2 bytes/token, which is simultaneously a perplexity artifact and a real cost problem — the same paragraph costs 3–4× more tokens in Hindi than in English on many tokenizers, which is a billing and context-budget issue, not just an academic one.

### Derive the Gumbel-max trick and tell me why anyone bothers.

The claim: if g_1…g_V are i.i.d. Gumbel(0,1) noise, then **argmax_i (z_i + g_i) is distributed exactly as Categorical(softmax(z))**. You sample from a softmax without ever computing the softmax.

Generating the noise is one line: if U ~ Uniform(0,1), then G = −log(−log U) is Gumbel(0,1). So:

```python
def gumbel_sample(logits, temperature=1.0):
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-20)))
    return ((logits / temperature) + g).argmax(-1)
```

Sketch of the proof, which is what they want if they ask you to derive it. The Gumbel CDF is F(x) = exp(−e^{−x}). For argmax to be i, you need z_i + g_i > z_j + g_j for all j ≠ i. Condition on the value of z_i + g_i = m: each other term must be below m, contributing ∏_{j≠i} exp(−e^{z_j − m}). Multiply by the density of z_i + g_i at m, integrate over m, and the exponentials collect into exp(z_i)/Σ_j exp(z_j). The Gumbel is precisely the max-stable distribution that makes this integral collapse — that's why this particular noise and no other.

Why it matters:

- **Reparameterization.** The argmax is not differentiable, but replacing it with a softmax over (z + g)/τ gives the Gumbel-Softmax / Concrete relaxation — a differentiable, temperature-controlled approximation to a categorical sample. That is how you backprop through a discrete choice, which is how differentiable MoE routing and various discrete-latent models are trained. **📄 Paper:** Jang, Gu & Poole (2017) and, concurrently, Maddison, Mnih & Teh (2017) introduced the Gumbel-Softmax / Concrete distribution, replacing high-variance REINFORCE estimators for categorical latents with a low-variance biased one.
- **Sampling without replacement.** Take the **top-k** of z + g rather than the top-1, and you get an exact sample of k distinct items from the categorical without replacement. That's the basis of stochastic beam search, and it's a genuinely useful trick for diverse candidate generation in a best-of-n pipeline.
- **Decoupling randomness from the distribution.** Because the noise is drawn independently of the logits, you can fix the Gumbel noise per position with a seed and get reproducible sampling that is *still* a correct sample. That's operationally useful for debugging a sampler and for A/B tests where you want the same random draw across two model variants.

**⚠ Trap:** believing Gumbel-max is faster than ordinary softmax sampling. It isn't, meaningfully — both are O(V) and neither is your bottleneck. It also *worsens* numerical behavior if you're careless: the double log needs a clamp on the uniform draw or you'll get infinities. Its value is structural (differentiability, without-replacement sampling, reproducibility), not performance.

### Write rejection sampling, prove it's correct, and then show me that speculative decoding is exactly this.

**Plain rejection sampling.** You want samples from p but can only sample from q. If you know a constant M with p(x) ≤ M·q(x) everywhere, then: draw x ~ q, draw u ~ Uniform(0,1), accept x if u ≤ p(x)/(M q(x)), else redraw. Accepted samples are exactly distributed as p, and the acceptance probability is 1/M. Correctness: P(accept and X = x) = q(x)·p(x)/(M q(x)) = p(x)/M, so conditioning on acceptance normalizes to p(x). The whole scheme is a way to convert "I can evaluate p up to a constant" into "I can sample from p," and its cost is the 1/M acceptance rate, which is why it's useless in high dimensions where M is astronomical.

**Speculative decoding is a modified rejection sampler**, and this is the part that makes the algorithm *correct* rather than merely fast — which is the point interviewers are probing. A small draft model q proposes k tokens; the large target model p scores all k in a single forward pass (cheap, because verifying k tokens is one batched prefill, not k decode steps). For each drafted token x in order:

- Accept with probability **min(1, p(x)/q(x))**.
- On the first rejection, discard the remaining drafts and **sample the replacement token from the normalized residual distribution, proportional to max(0, p(x) − q(x))**.

The residual step is the non-obvious part, and it's what makes the output distribution exactly p. The proof is short enough to say out loud: the probability of emitting token x is

  P(x) = q(x)·min(1, p(x)/q(x)) + P(reject)·[p(x) − q(x)]₊ / Σ_y [p(y) − q(y)]₊.

The first term is min(q(x), p(x)). The rejection probability is 1 − Σ_y min(p(y), q(y)) = Σ_y [p(y) − q(y)]₊, which cancels the denominator of the second term exactly. So P(x) = min(p(x), q(x)) + [p(x) − q(x)]₊ = p(x). **The output is drawn from the target model's distribution, exactly — not approximately.** Speculative decoding is lossless, and that's the claim that lets you deploy it without re-running your evals.

**📄 Paper:** Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers via Speculative Decoding*, and concurrently Chen et al. (2023), *Accelerating Large Language Model Decoding with Speculative Sampling* — both established the modified-rejection scheme that guarantees the target distribution while amortizing the target model's forward pass over multiple tokens. It replaced the assumption that autoregressive decoding must be strictly sequential in target-model forward passes.

**💰 Math on the speedup.** With per-token acceptance rate α and k drafted tokens per verification round, the expected number of tokens emitted per round is (1 − α^{k+1})/(1 − α). At α = 0.8 and k = 4: (1 − 0.8⁵)/(0.2) = (1 − 0.328)/0.2 = **3.36 tokens per target forward pass**. If the draft model costs 10% of the target per token, the cost per round is 1 target pass + 4 draft passes = 1 + 0.4 = 1.4 target-equivalents, so throughput improves by 3.36/1.4 = **2.4×**. Push α down to 0.6 and it falls to (1−0.078)/0.4 = 2.31 tokens per round, 2.31/1.4 = 1.65×. **Acceptance rate is the whole ballgame**, and it's what you should measure first when speculative decoding underdelivers.

**⚠ Trap:** assuming the speedup is free. It is free in *quality* (exactly lossless) but not in *memory* — you're holding a second model and a second KV cache in HBM — and not in *throughput at high batch*, because when you're already compute-bound from large batches, the extra draft FLOPs cost you. Speculative decoding is a latency optimization for low-to-medium batch regimes, and turning it on under heavy load can make aggregate throughput worse.

### Importance sampling — where does it show up in RLHF, and what's its failure mode?

The identity is E_{x∼p}[f(x)] = E_{x∼q}[f(x)·p(x)/q(x)]. You wanted an expectation under p, you only have samples from q, so you reweight by the likelihood ratio. That's it — a correction for evaluating one distribution's expectation using another's samples.

In RLHF this is the entire reason PPO is allowed to reuse rollout data. You generate trajectories from the current policy π_old, then take several gradient steps. After the first step, θ has moved, so your data is off-policy with respect to π_θ. Importance sampling corrects it: the policy-gradient term becomes r_t(θ)·Â_t where **r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)** and Â_t is the advantage. GRPO uses the same ratio with a group-normalized advantage in place of a learned value function.

**The failure mode is variance, and it is severe.** The importance weight is a ratio of probabilities, and ratios of small numbers are wild. If π_θ has drifted so that a sampled token now has 20× the probability it had under π_old, that one token contributes a weight of 20 to the gradient and dominates the batch. In a sequence of hundreds of tokens, per-sequence products of ratios explode or vanish exponentially in length. The formal diagnostic is **effective sample size**, ESS = (Σw_i)²/Σw_i², which tells you how many of your N samples are actually contributing; when ESS collapses to a handful, your gradient estimate is a few lucky trajectories with enormous weights and training destabilizes.

The mitigations, and each is worth naming:

- **Clipping.** PPO's clipped surrogate, min(r_t Â_t, clip(r_t, 1−ε, 1+ε)Â_t) with ε ≈ 0.2, hard-bounds the weight's influence. It's a biased estimator, deliberately — bounded bias in exchange for bounded variance is the right trade here.
- **Few epochs per rollout batch.** The further θ drifts from θ_old, the worse the ratios. One to four epochs is standard; ten is asking for it.
- **The KL leash** discussed above does double duty — it also keeps ratios near 1.
- **Token-level rather than sequence-level ratios**, so you never multiply hundreds of ratios together.

**🔍 Failure taxonomy — RL run destabilizes mid-training:** (1) Plot the distribution of importance ratios per batch; if the tail exceeds ~3–5, the policy has drifted too far — reduce epochs per batch or raise β. (2) Plot the fraction of tokens hitting the clip boundary; above ~20% you are mostly training on a clipped, biased signal. (3) Plot ESS; a collapse precedes the loss blowup by many steps and is your best early warning. (4) **Check for a sampling/training numerics mismatch** — if you generate rollouts with an inference engine and compute log-probs in a training framework, kernel and precision differences make π_old subtly wrong, so your ratios are wrong even at step 0. This is a real, common, and extremely confusing production bug in RLHF pipelines; the diagnostic is to recompute log-probs of the generated tokens in *both* stacks and compare them directly before you debug anything else.
