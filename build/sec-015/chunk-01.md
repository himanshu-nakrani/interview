### Start me at the beginning. A model has just finished a forward pass. What actually happens between that and a token appearing in my terminal?

The single most useful reframe here is that **the model never produces a token. It produces a vector of unnormalized scores over the entire vocabulary, and every interesting decision in this section happens *after* the model is done.** Decoding is a separate, cheap, entirely deterministic-if-you-want-it program that runs on the CPU-side of a serving engine — or on the GPU as a tiny epilogue kernel — and it is where roughly half of the "the model got worse" incidents you will ever debug actually live. Backend analogy: the model is a scoring service; the sampler is the routing policy in front of it. Nobody blames the routing policy first, and that is exactly why it stays broken for three weeks.

Mechanically: the final transformer layer emits a hidden state `h` of shape `[B, T, d_model]`. During decode you only care about the last position, so you slice `h[:, -1, :]` to `[B, d_model]`. That gets multiplied by the unembedding matrix (the "LM head") of shape `[d_model, V]`, giving **logits** of shape `[B, V]`. For Llama-3-8B that is `[B, 128256]` — one float per vocabulary entry, and note that this is a 128k-wide matmul against a 4096-dim vector, which is why the LM head alone is 525 M parameters and why a large vocabulary is a real serving cost, not just an embedding-table cost.

Those logits are real numbers on `(-∞, +∞)`. They are not probabilities and they are not comparable across models or even across checkpoints. The sampler then does, in order: apply penalties and biases (which mutate logits additively or multiplicatively), apply temperature (divide), apply truncation filters like top-k/top-p (set the losers to `-inf`), softmax the survivors into a probability distribution, and draw one index from it. That index is a token id. The detokenizer turns it into bytes, and the bytes get streamed to you — usually after a buffering step, because a single token is frequently *not* a valid UTF-8 sequence on its own.

**⚠ Trap:** thinking softmax happens inside the model. In every serving stack I have read — vLLM, SGLang, TensorRT-LLM, HF `generate` — the model's forward returns raw logits and the sampler owns softmax. This matters because it means every logit manipulation you do (bias, penalty, constraint mask) happens in *log space* on unnormalized scores, and adding a constant to one logit changes that token's relative odds multiplicatively after the exponential. A `logit_bias` of +5 is not "5% more likely"; it is `e^5 ≈ 148×` the odds, before renormalization.

**🗣 Say this in the room:** "The model outputs a `[B, V]` logit vector, not a token. Everything from penalties through temperature through truncation to the multinomial draw is a separate sampler pipeline that operates on those logits, and it's fully under my control — including making it deterministic, mostly."

### Write me a generation loop from scratch. No `model.generate`.

The reason this gets asked is that it is the smallest program that forces you to have the KV cache, the position handling, the sampler and the stop condition all correct simultaneously. Here it is at the level I would expect written unaided in ten minutes.

```python
import torch, torch.nn.functional as F

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=128,
             temperature=0.7, top_p=0.9, eos_id=None):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    eos_id = eos_id if eos_id is not None else tokenizer.eos_token_id
    past, out_ids = None, []

    # --- prefill: one forward over the whole prompt, fills the cache ---
    cur = ids
    for _ in range(max_new_tokens):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()      # [1, V]  <- fp32 for the sampler

        logits = logits / max(temperature, 1e-5)
        probs = sample_top_p(logits, top_p)         # returns a distribution
        nxt = torch.multinomial(probs, num_samples=1)  # [1, 1]

        tok = nxt.item()
        if tok == eos_id:
            break
        out_ids.append(tok)
        cur = nxt                                   # --- decode: one token in ---

    return tokenizer.decode(out_ids)
```

Four things in that loop are the actual test. **One:** the first iteration passes the whole prompt (`prefill`, `T` tokens, compute-bound), every later iteration passes exactly one token (`decode`, `T=1`, memory-bandwidth-bound) — the `cur = nxt` line is the entire difference and it is the line people forget, producing a loop that re-runs the full prompt every step at O(n²) total cost. **Two:** `past_key_values` must be threaded through; drop it and you get correct output at catastrophic cost. **Three:** `.float()` on the logits before sampling — bf16 has ~8 bits of mantissa, and a softmax over 128k entries in bf16 will quantize small probabilities to zero and give you a visibly different tail. Every serious engine upcasts logits to fp32 for the sampler. **Four:** the loop samples *after* slicing the last position; slicing before the LM head is the real optimization (`logits_to_keep=1` style), because you otherwise materialize a `[1, T, 128256]` tensor during prefill — at T=8192 that is `8192 × 128256 × 4 bytes = 4.2 GB` for a tensor you throw away.

**⚠ Trap:** on modern HuggingFace the cache is a `Cache` object rather than a legacy tuple, and correct position handling for RoPE depends on `cache_position` being right. If you hand-roll batched generation with left padding and do not offset positions per sequence, every padded sequence gets rotated at the wrong absolute positions and generates fluent nonsense. Batch-1 works; batch-8 quietly degrades. If you are unsure of the exact keyword in your version, say so and describe the invariant — "position ids must equal the token's true index in its own sequence, not its index in the padded tensor" — rather than inventing a signature.

**📐 Numbers you must know:** prefill of an 8k prompt on a 7B model is one forward over 8192 positions; decode of 500 tokens is 500 forwards over 1 position each. Same model, wildly different bottleneck. Prefill saturates tensor cores; decode moves the full 14 GB of bf16 weights across HBM once per token, so at ~2 TB/s you get a hard floor of `14 / 2000 s ≈ 7 ms/token ≈ 143 tok/s` at batch 1 no matter how fast your GPU's math units are. That floor is why batching exists.

### What does temperature actually do to the distribution? Derive it, don't describe it.

Temperature is not a "creativity knob." It is the inverse of the β parameter in a Boltzmann distribution, and once you see that, every one of its behaviors is forced.

Softmax is `p_i = exp(z_i) / Σ_j exp(z_j)`. Statistical mechanics writes the same object as `p_i ∝ exp(-E_i / kT)` — the probability of a system occupying state `i` at temperature `T`. Map `z_i = -E_i / k` and you have exactly the LLM sampler with `T = 1`. Introducing temperature means `p_i(T) = exp(z_i / T) / Σ_j exp(z_j / T)`, i.e. dividing logits by `T` before the exponential.

Now read off the limits. As `T → 0⁺`, the ratio between the top two probabilities is `exp((z_1 - z_2)/T)`, which diverges for any positive gap, so all mass collapses onto the argmax — temperature 0 *is* greedy, in the limit. As `T → ∞`, all logits go to 0, exponentials go to 1, and you get the uniform distribution over the whole vocabulary. At `T = 1` you get the model's own calibrated-ish distribution as trained by cross-entropy.

The crucial quantitative fact is that temperature acts on **log-odds linearly**, so its effect on probabilities is exponential. Take two tokens with logits 12.0 and 10.0 — a gap of 2.

```
T = 1.0 : odds ratio = e^(2/1.0)  = e^2   =   7.39×
T = 0.7 : odds ratio = e^(2/0.7)  = e^2.857 =  17.4×
T = 0.5 : odds ratio = e^(2/0.5)  = e^4   =  54.6×
T = 1.5 : odds ratio = e^(2/1.5)  = e^1.333 =  3.79×
```

Going from T=1.0 to T=0.7 — which sounds like a modest 30% change — more than doubles the odds ratio. This is why "just nudge the temperature down a bit" is never a small change, and why sampling-parameter changes deserve the same eval gate as a prompt change.

**⚠ Trap:** temperature is monotone — it never reorders tokens, only re-weights them. So "raise the temperature to get a more creative answer" is *only* true because it moves mass into the tail; it cannot make token 5000 beat token 1 in rank. Conversely, lowering temperature cannot fix a factual error whose correct token is not already ranked highly. If the right answer is not in the model's top-50, no sampling parameter on earth will produce it. That is the honest boundary between "a sampling problem" and "a model/context problem," and being able to draw it fast is a senior tell.

**🗣 Say this in the room:** "Temperature is inverse β in a Boltzmann distribution. It divides logits before softmax, so it's linear in log-odds and exponential in probability ratios. T→0 is argmax, T→∞ is uniform, and it's rank-preserving — which means it can change *how often* the model picks the right token but never *whether* the right token is reachable."

### Why isn't greedy decoding the same as finding the most likely sequence?

Because greedy is a locally optimal choice under a factorized model, and the product of locally-maximal factors is not the maximal product. This is elementary once stated, and interviewers ask it to see whether you conflate "the model's next-token distribution" with "the model's sequence distribution."

Concretely, `P(y₁...y_n | x) = Π_t P(y_t | x, y_<t)`. Greedy picks `argmax P(y_t | ·)` at each step, which commits irrevocably. Construct the counterexample in two steps: suppose at t=1 token A has p=0.6 and token B has p=0.4. Greedy takes A. But suppose after A the distribution is flat — best continuation p=0.3, giving a sequence probability of `0.6 × 0.3 = 0.18` — while after B the model is confident, best continuation p=0.9, giving `0.4 × 0.9 = 0.36`. B's path is twice as likely and greedy will never see it, because greedy has no lookahead. That is the entire motivation for beam search: keep `k` hypotheses alive so a strong second-place prefix gets a chance to prove itself.

Two more things worth saying, because they are where the answer gets interesting. First, exact MAP decoding — finding the true argmax over sequences — is intractable; the search space is `V^n`, and beam search is a heuristic that gives no optimality guarantee, just a better one than greedy. Second, and this is the part that surprises people: **for open-ended generation, the most likely sequence is usually a bad output.** Push beam width up on a chat model and quality gets *worse*, converging on bland, repetitive, high-probability text. The maximum-probability string under a well-trained LM tends to be degenerate — often the empty string or a loop. Human text is not the mode of the distribution; it is a typical sample from it.

**📄 Paper:** Holtzman et al. (2020), "The Curious Case of Neural Text Degeneration" — showed that maximization-based decoding produces repetitive, unnaturally-flat-entropy text, that human text occupies a much wider probability band than beam search output, and introduced nucleus sampling as the fix. This is the paper that made stochastic decoding the default for chat.

**🗣 Say this in the room:** "Greedy is locally optimal, not globally — a lower-probability first token can lead to a much higher-probability sequence, which is why beam search exists. But for open-ended text, high sequence probability is actively the wrong objective; the mode of the distribution is degenerate. That's why we sample for chat and search for translation."

### Implement top-k sampling, and tell me where it fails.

Top-k is the crudest truncation you can do: sort the logits, keep the `k` highest, set everything else to `-inf`, renormalize, sample. It exists because the tail of a 128k-vocabulary distribution holds a large amount of aggregate probability mass spread across tokens that are individually absurd, and sampling from the untruncated distribution occasionally draws one of them — and one absurd token derails the whole continuation, because autoregression conditions on its own mistakes.

```python
def top_k_filter(logits, k):
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]   # [B, 1]
    return logits.masked_fill(logits < kth, float("-inf"))
```

Three lines, and the only subtlety is `[..., -1, None]` to keep the broadcast dimension, plus using `<` rather than `<=` so exact ties at the k-th value are kept (you may keep more than k tokens; that is the correct behavior — silently dropping one member of a tie is a worse bug than returning k+1 candidates).

**📄 Paper:** Fan, Lewis & Dauphin (2018), "Hierarchical Neural Story Generation" — introduced top-k sampling for open-ended generation. It replaced full-distribution sampling, which was too noisy, and pure beam search, which was too bland.

The failure is that `k` is a **fixed count against a variable-shape distribution.** Consider two positions in the same generation. At a position right after `"The capital of France is"`, the distribution is a spike: `" Paris"` might hold p=0.97, and the next 49 tokens hold 0.03 between them. `k=50` here means you have a 3% chance of *not* saying Paris — you have deliberately built in a 3% error rate for no benefit. Now consider a position after `"She opened the door and saw"`, where the distribution is genuinely flat: maybe 800 tokens are plausible continuations with no clear winner. `k=50` here truncates 750 legitimate options and makes the output more repetitive than the model wants to be.

So top-k is simultaneously too permissive where the model is confident and too restrictive where it isn't. That is precisely the gap top-p was invented to close.

**⚠ Trap:** treating `k` as a quality dial and tuning it globally. There is no single `k` that is right across a generation, let alone across tasks, because the entropy of the next-token distribution varies by orders of magnitude *within a single response*. I would push back on any config that ships a tuned `top_k` without also shipping a top-p or min-p, because you have picked a constant to approximate a function of entropy.

### Derive nucleus sampling and implement it. Why did it win over top-k?

Top-p fixes exactly the defect above by making the truncation **adaptive to the distribution's shape**: instead of "keep k tokens," it says "keep the smallest set of tokens whose cumulative probability is at least `p`." The size of that set — the *nucleus* — automatically shrinks to 1 when the model is confident and expands to hundreds when it isn't. You have replaced a constant with a function of entropy, which is what the previous answer said was missing.

Formally: sort tokens by descending probability, find the smallest prefix `S` such that `Σ_{i∈S} p_i ≥ p`, mask everything else, renormalize.

```python
def top_p_filter(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum   = probs.cumsum(dim=-1)
    # keep token i iff the mass strictly BEFORE it is still under p
    remove_sorted = (cum - probs) > p          # always False at i=0 -> top token survives
    remove = torch.zeros_like(remove_sorted).scatter_(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))
```

The `cum - probs > p` formulation is the one I write from memory, because it is the version that cannot produce an empty nucleus. `cum[0] - probs[0] = 0`, which is never `> p` for any `p ≥ 0`, so the argmax token is structurally guaranteed to survive. The naive `cum > p` version drops the top token whenever `p_max > p` — set `top_p = 0.5` on a distribution where the best token has p=0.9 and you mask *everything*, softmax over an all-`-inf` row gives NaN, and `torch.multinomial` throws or returns garbage. I have seen this ship.

Why it won: it is one parameter that means something task-independent ("how much of the model's own probability mass am I willing to sample from"), it degrades gracefully at both ends (`p=1.0` is pure sampling, `p→0` is greedy), and it composes with temperature in a way people find intuitive. Every major provider exposes it; several expose *only* it plus temperature.

**📐 Numbers you must know:** the community defaults are `top_p = 0.9` for chat-like generation and `top_p = 0.95` for code, and the reason is entropy: code has more genuinely-valid continuations at any given point (variable names, whitespace, ordering) than prose does at a factual position, so truncating at 0.9 clips real options. The Holtzman paper's headline setting was `p = 0.95`.

**⚠ Trap:** believing top-p bounds the number of candidates. It does not. On a near-uniform distribution — which happens at the start of a creative generation or on a badly-out-of-domain prompt — `top_p = 0.9` can admit tens of thousands of tokens. If you need a hard bound on the candidate set for a latency or safety reason, you must *also* set top-k; they compose (apply k first, then p, on the survivors). Most engines let you set both, and the intersection is the intended behavior.

### I'm going to show you a top-p implementation from a PR. Find the bugs.

```python
# PR under review
def top_p_sample(logits, top_p=0.9, temperature=1.0):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, 1)
    return sorted_idx[choice] / temperature
```

There are four defects and they are of escalating severity.

**Bug 1 — temperature is never applied, and then is applied to a token id.** `temperature` appears only on the return line, where it divides an *integer index*. That returns a float token id, which will either throw on `tokenizer.decode` or, worse, be silently cast and index the wrong token. Temperature must divide the *logits* before softmax. This one is loud enough that a smoke test catches it, which makes it the least dangerous of the four.

**Bug 2 — off-by-one on the nucleus boundary, which can empty the nucleus.** `mask = cum > top_p` removes the token that *crosses* the threshold. Since `cum[0] = p_max`, any position where the top token's probability exceeds `top_p` masks the top token itself. With `top_p = 0.9` and a confident model (`p_max = 0.97`), *every* token is masked, the sum is 0, the division produces NaN, and `multinomial` errors out — or on some versions returns index 0 silently. The fix is `mask = (cum - sorted_probs) > top_p`, or equivalently shift the mask right by one and force `mask[..., 0] = False`.

**Bug 3 — in-place mutation of a tensor that may be a view.** `sorted_probs[mask] = 0.0` writes into the output of `torch.sort`. Under autograd this raises; under `no_grad` it is fine but it is still the kind of write that breaks when someone later wraps this in a `torch.compile` region or shares the buffer. Use `masked_fill` and return a new tensor. Style, not correctness — but it is the line a reviewer should flag.

**Bug 4 — the real one: zeroing probabilities is not the same as masking logits, once anything else is in the pipeline.** Setting probabilities to zero and renormalizing is mathematically equivalent to setting logits to `-inf` and re-softmaxing *for this operation alone*. But the moment a second filter, a logit bias, or a constrained-decoding grammar mask runs after this, the pipeline expects logits, and it has been handed a probability vector. Everything downstream that adds a bias in log-space now adds it in probability-space. The convention every engine enforces is: **every stage of the sampler consumes logits and emits logits; softmax happens exactly once, at the end.** This PR breaks that invariant and the resulting bug is invisible until someone enables JSON mode.

**🗣 Say this in the room:** "Three functional bugs — temperature applied to the token id instead of the logits, a `cum > p` boundary that can empty the nucleus and NaN on a confident distribution, and an in-place write into a sort view. The architectural one is that it returns probabilities where the sampler pipeline contract is logits-in, logits-out."

### What is min-p sampling and what's it fixing that top-p doesn't?

Top-p adapts to distribution shape, but it does so through a *cumulative* quantity, and cumulative mass is a lossy summary. Min-p replaces it with a direct relative-quality criterion: **keep every token whose probability is at least `min_p × p_max`.** With `min_p = 0.05` and a peak of `p_max = 0.90`, the threshold is 0.045, so you keep only tokens within about a 20× odds factor of the best one — a nucleus of maybe two or three. With a flat distribution where `p_max = 0.02`, the threshold is 0.001, and you keep a very wide set. Same parameter, and it scales with the model's own confidence rather than with an integral.

```python
def min_p_filter(logits, min_p):
    probs = torch.softmax(logits, dim=-1)
    thresh = min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < thresh, float("-inf"))
```

The concrete failure of top-p that this addresses is the **long-flat-tail-plus-spike** case. Imagine `p_max = 0.5` and then 4,000 tokens each at roughly 0.000125 (summing to the remaining 0.5). With `top_p = 0.9`, you keep the spike plus enough of the flat tail to reach 0.9 — that is 3,200 tail tokens, each of which is 4,000× less likely than the best one and most of which are nonsense. After renormalizing over the nucleus you have a `0.4 / 0.9 = 44%` chance of drawing one. Top-p cannot see this because it only knows cumulative mass, not the *ratio* between the head and the tail. Min-p sees it immediately and clips the tail.

The practical consequence people care about is that min-p tolerates high temperature much better. Because the threshold is relative to the peak, you can run `temperature = 1.5` with `min_p = 0.05` and still get coherent text, which is a genuinely useful operating point for creative generation. Top-p at 1.5 falls apart, because raising temperature flattens the distribution and therefore *widens* the nucleus at fixed p — exactly the wrong direction.

**📄 Paper:** Nguyen et al. (2024) introduced min-p as a dynamic, peak-relative truncation for high-temperature creative generation. Worth knowing that its headline quality claims drew substantive public criticism during peer review over reproducibility of the human-eval comparisons; the *mechanism* is sound and widely implemented (vLLM, llama.cpp, HuggingFace all expose `min_p`), while the size of the quality win is contested. Saying that out loud is a credibility gain, not a loss.

**⚠ Trap:** ordering. Min-p's whole value proposition is that it is relative to the peak — but if temperature runs *first* and flattens the distribution, `p_max` shrinks and the absolute threshold shrinks with it, so min-p admits more junk. Many practitioners recommend applying min-p *before* temperature so that the nucleus is chosen on the model's native confidence and temperature only reshapes the survivors. Different libraries order these differently and versions have changed. Verify empirically against your stack — print the surviving candidate count at a fixed prompt — rather than trusting a config comment.

### Explain locally typical sampling. Would you ship it?

Typical sampling starts from information theory rather than from probability mass, and the reframe is worth understanding even if you never enable it. The claim is that natural language is not the *most probable* continuation and it is not a *uniform* draw — it is a continuation whose surprisal is close to the distribution's **expected** surprisal. Human speakers spread information roughly evenly across an utterance; they neither say things that are totally predictable (uninformative) nor things that are wildly improbable (incomprehensible). So the target is not "high probability" but "typical information content."

Mechanically: compute the conditional entropy `H = -Σ p_i log p_i` of the next-token distribution. For each token compute its surprisal `-log p_i`. Score each token by `|(-log p_i) - H|` — the absolute deviation from the expected surprisal. Sort *ascending* by that score, and keep the smallest set whose cumulative probability reaches `τ`. Note what this does that no other sampler does: **it can exclude the argmax.** If the top token is far too predictable relative to the distribution's entropy, typical sampling drops it. Every other truncation method in this section is a prefix of the descending-probability order; typical sampling is not.

**📄 Paper:** Meister, Pimentel, Wiher & Cotterell (2023), "Locally Typical Sampling" (TACL) — formalized decoding as targeting the expected information content, and showed reduced degenerate repetition versus nucleus sampling on open-ended generation.

Would I ship it? For a product surface, no, and I would say so directly. It is exposed in HuggingFace (`typical_p`) and llama.cpp but not by the major API providers, so any pipeline using it is locked to self-hosted inference; the quality delta over well-tuned top-p or min-p is small and task-dependent; and its ability to drop the argmax is exactly wrong for anything factual — an extraction or classification task wants the confident token *because* it is confident. Where it earns its place is open-ended creative generation on self-hosted models, and as a diagnostic: if typical sampling noticeably improves your output, that is evidence your distribution has a degenerate over-confident mode, which is usually a fine-tuning or prompt problem worth fixing at the source.

**🗣 Say this in the room:** "Typical sampling targets tokens whose surprisal is near the distribution's entropy, rather than tokens with high probability — so it can drop the argmax. Elegant, and the only sampler that isn't a prefix of the sorted order. I don't ship it: no provider exposes it, and dropping the argmax is precisely wrong for extraction."

### Epsilon and eta sampling — what are they, and where do they fit?

These are the "principled truncation" family, and the framing that makes them click is **desmoothing**. A language model trained with cross-entropy is implicitly smoothed: it can never assign exactly zero probability to any token, so it assigns tiny nonzero mass to tokens that are genuinely impossible in context. Truncation sampling, on this view, is not a hack — it is an attempt to undo the smoothing the training objective imposed and recover the model's "true" support.

Epsilon sampling is the simplest possible version: keep every token with `p_i > ε`, an absolute probability floor. Typical `ε` is around `3e-4` to `9e-4`. It has no adaptivity to the peak at all, which is its weakness — on a flat 10,000-way distribution, every token might fall below the floor.

Eta sampling fixes that by making the floor adapt to entropy. The threshold is `min(ε, √ε · exp(-H))` where `H` is the entropy of the next-token distribution. Read it as: use the fixed floor `ε` normally, but when the distribution is high-entropy (large `H`, so `exp(-H)` is small), lower the floor so you do not over-truncate a genuinely uncertain position. It is the same instinct as min-p — make the cut relative to how confident the model actually is — arrived at from a different direction.

**📄 Paper:** Hewitt, Manning & Liang (2022), "Truncation Sampling as Language Model Desmoothing" — reframed top-k/top-p as approximate desmoothing, and introduced epsilon and eta sampling with the entropy-dependent threshold above.

In practice these are HuggingFace-only knobs (`epsilon_cutoff`, `eta_cutoff`) that essentially nobody sets in production. Know them for two reasons. First, the desmoothing framing is the best single sentence explaining *why* truncation improves quality at all — you are removing mass the model was forced to assign but does not believe. Second, an interviewer asking "why not just sample from the raw distribution?" is asking for exactly this answer.

### What is Mirostat and do you use it?

Mirostat is a closed-loop controller, and if you have ever written a rate limiter with an adaptive window, you already have the mental model: instead of setting a truncation parameter and hoping the resulting surprisal is right, you set a **target surprisal** and let a feedback loop adjust the truncation on every step to hit it.

**📄 Paper:** Basu, Ramachandran, Keskar & Varshney (2021), "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity" (ICLR 2021).

The mechanism: you specify `τ` (target surprisal in nats, roughly `log` perplexity) and `η` (learning rate). At each step the algorithm truncates using a `k` derived from an estimate of the tail's Zipf exponent, samples a token, measures that token's actual surprisal `s = -log p`, computes the error `e = s - τ`, and updates the internal parameter `μ ← μ - η·e`. If recent tokens were too surprising, tighten; too boring, loosen. It is proportional control on perplexity.

The observation motivating it is real and worth repeating: with a fixed `k` or `p`, the *observed* perplexity of the generated text drifts over the course of a long generation — usually downward, into the repetition attractor, because once the model starts repeating, repetition becomes high-probability, which makes the nucleus narrow, which makes repetition more likely. It is positive feedback. Mirostat breaks the loop by construction: as surprisal falls below target, the controller widens the candidate set.

Do I use it? No, and here is the decision rule. It is available in llama.cpp and a few local-inference UIs, not in vLLM's standard sampling params and not in any hosted API, so it is unavailable in the deployments that matter for these roles. It adds two coupled hyperparameters and a stateful controller to a component whose greatest virtue is being stateless and debuggable. And the repetition-attractor problem it solves is better addressed upstream — by a decent instruct-tuned model, a repetition penalty, or fixing the prompt that induced the loop. I would bring it up in an interview as evidence I know the design space, and then say I would not ship it.

**🔍 Failure taxonomy — "the output degenerates into a loop after ~500 tokens":** work this ladder in order. (1) Is the model a base model rather than instruct-tuned? Base models loop; this is not a sampler bug. (2) Is temperature ≤ 0.3 or top-p ≤ 0.7? Over-truncation is the single most common cause — widen before you add penalties. (3) Does the prompt contain a repeated structure the model is pattern-matching? Check the last 200 tokens of context. (4) Only now reach for `repetition_penalty` around 1.05–1.15, and measure. (5) If it is a self-hosted long-form creative surface and 1–4 fail, Mirostat is a legitimate last resort.
