### Start me off simply — what is distillation, and why do you keep calling it the cost lever rather than a quality lever?

Distillation is the operation of moving a capability out of an expensive artifact and into a cheap one, using the expensive artifact's own outputs as the supervision signal. That is the whole idea. Everything else — logit matching, on-policy sampling, rejection sampling, context distillation — is a variation on *what signal you copy* and *from which distribution you sample the inputs*.

The mental model I use is a compiler with profile-guided optimization. A frontier model is a general-purpose interpreter: it can do anything, and it pays for that generality on every token. Your product does not need anything; it needs one narrow distribution of tasks — classify this support ticket, rewrite this paragraph in the user's voice, produce a function-call plan for this schema. Distillation is a specializing compiler: you run the interpreter over a representative workload, record what it did, and emit a small binary that reproduces that behaviour on that workload only. The student is not smarter than the teacher and will never be. It is smaller, and on the slice you compiled for, it is close enough.

Why "cost lever": the ceiling of distillation is the teacher. If your teacher gets 84% on your eval, no distillation procedure makes the student get 90%. What distillation buys is throughput and price. A 7B student on your own hardware serves the same task at roughly 20× lower marginal cost than a frontier API call, and at 5–10× lower latency because both prefill FLOPs and decode memory-bandwidth scale with parameter count. So the honest framing in an interview is: *distillation is how you take a quality level you already achieved with prompting and make it affordable at production volume.* It is the last rung before fine-tuning on the escalation ladder, and it is far and away the most common post-training job an AI Engineer actually ships — most engineers at product companies will never run RLHF, but many will run a distillation.

**🗣 Say this in the room:** "I reach for distillation when I already have a prompted frontier pipeline that hits the quality bar and the only remaining problem is unit economics or p99. Distillation cannot raise the ceiling — the teacher is the ceiling — so if the prompted pipeline is not good enough, distilling it is the wrong move and I'd go back to retrieval or tool design first."

**⚠ Trap:** treating distillation as a way to get frontier capability for free. Students inherit teacher errors *and* add their own compression error. If the teacher hallucinates on 3% of your traffic, the student will hallucinate on at least 3%, usually more, and with less-calibrated uncertainty. Distillation launders the teacher's mistakes into a model that can no longer be prompted out of them.

### Walk me through logit distillation. What exactly is the loss, and why is temperature in there?

Here is the intuition that makes it inevitable. A hard label says "the next token is `cat`." That is one bit of routing information out of a 128k-way vocabulary. The teacher's *full distribution* says "`cat` 0.62, `kitten` 0.11, `dog` 0.04, `feline` 0.03, …" — and those relative magnitudes among the wrong answers encode the teacher's learned similarity structure. Hinton called this the dark knowledge: the ratio of `kitten` to `dog` tells the student something about the geometry of the problem that no one-hot label can. Logit distillation is training on that whole vector instead of on the argmax.

Mechanically, at each position `t` you have teacher logits `z^T_t` and student logits `z^S_t`, both of shape `[V]`. You soften both with a temperature `T`, then minimize the KL between them:

```
p_T = softmax(z^T / T)
p_S = softmax(z^S / T)
L_KD = T² · KL(p_T ‖ p_S)
```

Temperature exists because a well-trained teacher is extremely peaked — the top token often carries >0.9 mass, and everything informative is buried at 1e-4. Raising `T` flattens the distribution and surfaces that structure. The `T²` factor is not decoration: the gradient of a softmax cross-entropy through a `1/T` scaling picks up a `1/T²`, so multiplying the loss by `T²` keeps the gradient magnitude roughly constant as you tune `T`. Without it, changing temperature silently changes your effective learning rate. In practice you also mix in the ordinary hard-label cross-entropy: `L = α·L_KD + (1−α)·L_CE`, with α around 0.5–0.9 depending on how much you trust the teacher.

**⚠ Trap:** you cannot do logit distillation against a closed API. OpenAI, Anthropic and Google do not return full next-token distributions; at most you get top-k logprobs for a handful of candidates, and often not for the reasoning models at all. So logit distillation is an *open-weight teacher* technique — Llama-70B → Llama-8B, Qwen-72B → Qwen-7B. If the interviewer's scenario says "distill GPT-class model X," the honest answer is sequence-level distillation, not logit KL. Candidates who describe KL-matching against a closed API have revealed they have not done it.

**📄 Paper:** Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network* — introduced temperature-softened soft targets and the `T²` gradient correction, replacing the earlier practice of training small models on hard labels alone.

**📐 Numbers you must know:** storing full teacher logits for offline distillation is usually infeasible. 10M tokens × 128k vocab × 2 bytes (bf16) = 2.56 TB. That is why you either (a) run the teacher online in the training loop, paying a forward pass per step, or (b) transfer only top-k. Top-8 with indices: 10M × 8 × (2 + 4) bytes = 480 MB. Two-thousand-fold reduction, and empirically top-8 to top-64 captures nearly all the usable signal because the tail is noise. Renormalize the top-k over its own mass before taking the KL.

### Offline sequence-level distillation versus on-policy distillation — what's the actual difference and when does it matter?

The difference is *whose mistakes the student gets trained on*, and it is exactly the exposure-bias problem you already know from any autoregressive system.

Offline sequence-level distillation is: prompt the teacher over your input distribution, sample or greedy-decode a response, save `(prompt, teacher_response)` to a JSONL, and run ordinary SFT on it with loss masked to the completion. That is it. It is embarrassingly simple, it parallelizes trivially, it works against any API, and it is what 90% of shipped distillations actually are. The catch is that the student only ever sees *teacher-generated prefixes*. At inference it sees *its own* prefixes, and the moment it makes an early token error it is off the training manifold with no idea how to recover. On a 30-token classification output that barely matters. On a 2,000-token agent trajectory it is the dominant failure mode: the student drifts, compounds, and produces something the teacher would never have produced.

On-policy distillation fixes exactly that. You sample the *student's* own completions, then ask the teacher to score them — either by computing the teacher's token-level distribution over the student's sequence and taking a KL, or by having the teacher rewrite/critique. The training distribution is now the deployment distribution. This is the generalized knowledge distillation framing: interpolate between teacher-generated and student-generated data, and use reverse KL (or a Jensen-Shannon-style divergence) rather than forward KL.

The forward-vs-reverse KL choice matters and is a favourite follow-up. Forward KL `KL(p_T ‖ p_S)` is mass-covering: the student is penalized wherever the teacher has mass and it does not, so it spreads out and hedges, which for a small student means blurring across modes and producing bland or incoherent text. Reverse KL `KL(p_S ‖ p_T)` is mode-seeking: the student is penalized only where *it* puts mass the teacher does not, so it picks one mode of the teacher and does it cleanly. For a student with far less capacity than the teacher, mode-seeking is usually what you want, because it cannot represent all the teacher's modes anyway and hedging produces mush.

**📄 Paper:** Kim & Rush (2016), *Sequence-Level Knowledge Distillation* — showed that training a small NMT model on teacher-decoded outputs beats token-level KD, establishing the offline recipe. Agarwal et al. (2024), *GKD: Generalized Knowledge Distillation* — trains on student-sampled sequences with a divergence family that includes reverse KL, directly addressing the train/inference distribution mismatch.

**🔍 Failure taxonomy — which one do I pick?** Short outputs (<200 tokens), single-turn, no state: offline sequence-level. Ship it, it is a week of work. Long-form generation or multi-turn agent trajectories where errors compound: you need on-policy, and you need an open-weight teacher because you need per-token teacher scores over student text. Teacher is closed-API only and outputs are long: your on-policy option collapses to rejection sampling — sample from the student, have the teacher judge/verify, train on the survivors. That is the practical middle ground and it is what I would propose in most product settings.

### Implement on-policy distillation. Torch, from memory, no library.

The shape of it is: generate with the student under `no_grad`, forward both models over the generated sequence, take a per-token reverse KL on the completion tokens only, backprop into the student. Roughly:

```python
import torch, torch.nn.functional as F

def on_policy_distill_step(student, teacher, tok, prompts, opt, max_new=128, T=1.0):
    batch = tok(prompts, return_tensors="pt", padding=True).to(student.device)
    plen = batch.input_ids.shape[1]

    # 1. roll out the STUDENT — this is what makes it on-policy
    with torch.no_grad():
        seq = student.generate(**batch, max_new_tokens=max_new, do_sample=True,
                               temperature=1.0, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
    attn = (seq != tok.pad_token_id).long()

    # 2. score the same sequence under both models
    s_logits = student(input_ids=seq, attention_mask=attn).logits[:, :-1]
    with torch.no_grad():
        t_logits = teacher(input_ids=seq, attention_mask=attn).logits[:, :-1]

    # 3. completion tokens only; never train on the prompt
    mask = attn[:, 1:].clone()
    mask[:, :plen - 1] = 0

    log_p_s = F.log_softmax(s_logits / T, dim=-1)
    log_p_t = F.log_softmax(t_logits / T, dim=-1)

    # reverse KL: KL(student || teacher), mode-seeking
    kl = (log_p_s.exp() * (log_p_s - log_p_t)).sum(-1)      # [B, L-1]
    loss = (T * T) * (kl * mask).sum() / mask.sum().clamp(min=1)

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
    return loss.item()
```

Three things I look for when reviewing this. First, the rollout is under `no_grad` and is *not* the thing you backprop through — the gradient path is the second forward pass over the fixed sequence. People try to backprop through `generate()` and get either an error or a silent no-op because of the argmax/sampling. Second, the mask must zero out prompt positions and padding; a missing mask means you are distilling the teacher's opinion about the user's own tokens, which is at best wasted capacity and at worst teaches the student to continue prompts instead of answer them. Third, normalizing by `mask.sum()` rather than by batch size — otherwise long sequences dominate the gradient and you get a length bias baked into the student.

**⚠ Trap:** tokenizer mismatch. This loss is only defined if the student and teacher share a vocabulary, because you are comparing distributions over the same index space. Llama-70B → Llama-8B: fine. Qwen teacher → Llama student: the loss is meaningless even though the tensors have compatible shapes only by coincidence, and usually they do not even have that. Cross-tokenizer distillation requires either sequence-level distillation or a token-alignment scheme, and "just distill Qwen into Llama with KL" is a review-blocking error I have caught more than once.

### Explain rejection-sampling fine-tuning. Why is it the first thing you should try before any RL?

Rejection-sampling fine-tuning is the recognition that if you have a *verifier*, you do not need a policy-gradient algorithm — you can turn RL into SFT by filtering.

The loop: for each prompt, sample `n` completions from the current model at a decent temperature. Run each through a verifier — unit tests, an exact-match checker, a schema validator, a reward model with a threshold, an LLM judge. Keep only the ones that pass. Fine-tune the model on the survivors with plain cross-entropy. Optionally repeat. That is best-of-n, except instead of throwing away the good sample after serving it, you write it into the weights.

Why this is not a hack: it is a legitimate approximation to the RL objective. If you sample from policy π and keep only trajectories with reward 1, you are doing SFT on samples from the reward-reweighted distribution `π(y|x)·r(y)/Z` — which is exactly the optimal KL-constrained policy from the RLHF derivation, in the limit of a binary reward and enough samples. Rejection sampling is policy improvement by importance weighting with hard 0/1 weights. It just does not need advantages, a value head, a clipped surrogate, or a rollout/train synchronization system.

The operational argument is the one that wins interviews. GRPO requires an RL framework, a rollout engine sitting next to the trainer, weight synchronization, and a person who can read entropy and KL curves and tell you why the run collapsed. Rejection sampling requires a `for` loop, a batch inference call, a filter, and your existing SFT stack. In engineer-weeks that is maybe 1 versus 6. It captures a large fraction of the gain on most tasks and it fails *loudly* — if your filter keeps 0% of samples you find out in the first hour, whereas an RL run can burn a GPU-week before you notice entropy collapsed at step 200.

**🗣 Say this in the room:** "My default is rejection-sampling fine-tuning, and I only escalate to GRPO when I can point at a specific thing rejection sampling cannot do: I need the negative signal from failures, or the pass rate is so low that sampling is prohibitively expensive, or I've plateaued and the remaining headroom needs many small on-policy steps rather than a few big supervised ones."

**💰 Math:** suppose your model solves 40% of prompts and you want 50k training examples. At `n=4` samples per prompt, `P(at least one pass) = 1 − 0.6⁴ = 0.87`, so you need `50,000 / 0.87 ≈ 57,500` prompts and `230,000` generations. At 600 output tokens each on an open-weight 8B served locally at, say, 4,000 tok/s aggregate on one H100, that is `230,000 × 600 / 4,000 = 34,500 s ≈ 9.6 GPU-hours`, roughly $25 at $2.50/H100-hour. The filtering and SFT are noise next to that. A GRPO run on the same task is 200–600 GPU-hours plus the engineering. The ratio is not close.

### Explain STaR. What is the "rationalization" step and why does it matter?

STaR is rejection sampling applied specifically to *reasoning*, with one clever addition that keeps the hard problems in the dataset.

The base loop: you have questions with known final answers but no worked solutions. Prompt the model with a few CoT exemplars, sample a chain of thought plus an answer, and check the answer against the gold label. If it matches, keep the whole rationale as a training example. Fine-tune on the kept rationales. Repeat from the *original* base model each round (not from the previous fine-tune) to avoid compounding overfit. You are bootstrapping the ability to produce reasoning traces using only answer-level supervision — the rationale is never labelled, it is only *validated by its consequence*.

The problem is the survivorship filter. Problems the model already solves generate data; problems it cannot solve generate nothing, forever. So the dataset is systematically biased toward the easy tail and the model never improves on the frontier of its ability. Rationalization is the fix: for a problem the model failed, re-prompt it *with the correct answer given in the prompt* and ask it to produce a reasoning chain that arrives there. Now you get a rationale for a hard problem. Strip the hint, keep `(question, rationale, answer)`, train on it.

**⚠ Trap:** rationalization generates post-hoc justifications, not the reasoning that would have found the answer. When you hand a model the answer, it will happily produce a chain that contains a subtly invalid step and still lands on the right conclusion — and you have now trained on an invalid derivation with a correct label, which is precisely how you teach a model to produce confident-looking wrong reasoning. In production I gate rationalized examples harder than sampled ones: a second model verifies the *chain* step by step, not just the answer, and I cap rationalized data at maybe 20–30% of the mix. The original paper reports rationalization helps; my read is it helps *on benchmarks with checkable answers* and is a liability on open-ended tasks where nobody checks the chain.

**📄 Paper:** Zelikman et al. (2022), *STaR: Bootstrapping Reasoning with Reasoning* — bootstrapped chain-of-thought supervision from answer-only labels, with rationalization to recover failed problems. It is the direct ancestor of the rejection-sampling SFT stage in modern reasoning-model recipes.

### ReST and ReST-EM — what do they add on top of STaR, and what's the EM framing actually saying?

ReST formalizes the loop into two named phases so you can reason about the cost of each separately. **Grow**: sample a large batch of completions from the current policy over your prompt set — this is the expensive, inference-bound phase, and it produces a static dataset. **Improve**: filter that dataset by a reward threshold and fine-tune on the survivors, possibly several times with an *increasing* threshold, reusing the same generated data. The point of the split is that Grow is where the GPU-hours go, so you amortize it: one expensive generation pass feeds several cheap improvement passes at progressively stricter filters. That is a genuinely useful piece of ops engineering, not just a renaming.

ReST-EM (the "Beyond Human Data" line of work) reframes the same loop as expectation-maximization on a latent-variable model. The latent is the reasoning trace; the observation is the correct answer. **E-step**: sample traces from the current model and keep those consistent with the observed answer — that is your posterior sample over the latent. **M-step**: maximize the likelihood of the model on those traces. Iterate. The framing earns its keep in two ways: it tells you *why* you should re-filter with the current policy every round rather than reusing round-1 data (your posterior moved), and it predicts the empirical result the paper actually found — gains saturate after a small number of iterations, typically two or three, because the E-step posterior stops changing much. If someone tells me their self-training loop is on iteration nine and still improving, my prior is that they have a leak, not a discovery.

**📄 Paper:** Gulcehre et al. (2023), *Reinforced Self-Training (ReST) for Language Modeling* — the Grow/Improve decomposition with threshold-annealed reuse. Singh et al. (2023), *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models* — ReST^EM, the EM interpretation, and the result that model-generated verified data can outperform equivalent human data on math and code, with saturation after a few iterations.

**📐 Numbers you must know:** the practical iteration count for self-training loops is 2–3, not 10. Every published variant — STaR, ReST, ReST-EM, and the rejection-sampling stages in modern reasoning recipes — reports saturation quickly. Budget for three rounds, instrument diversity from round one, and treat "still improving at round six" as a signal to check for contamination between your generated data and your eval set.

### Where does RAFT fit, and how is it different from the others?

RAFT — reward-ranked fine-tuning — is the same skeleton with a *relative* filter instead of an absolute one, which is what you use when you have a scalar reward model rather than a binary verifier.

The distinction is the whole point. With a verifier you filter on `reward == 1`. With a reward model you have a continuous score whose absolute scale is meaningless and drifts across prompts — an easy prompt might score 3.2 for a mediocre answer while a hard prompt scores 1.1 for an excellent one. Thresholding globally on that will select for easy prompts, not for good answers. RAFT instead samples `k` completions per prompt, ranks them *within the prompt*, and keeps the top-1 (or top-m). That is a per-prompt normalization, structurally the same trick GRPO uses with group-relative advantages, applied at the data-selection layer rather than in the gradient.

So the family sorts cleanly by what signal you have: binary verifier and you want reasoning traces → STaR. Binary verifier and you want an ops-friendly loop → ReST/ReST-EM. Scalar reward model, no verifier → RAFT. Scalar reward model and you also want the negative gradient from the bad samples → you have left rejection sampling and you are doing DPO (pair the best with the worst) or GRPO.

**⚠ Trap:** RAFT's top-1-of-k filter is a reward-model amplifier. Any bias in the RM — verbosity preference, markdown preference, sycophancy — is exactly what top-1 selection concentrates, because you are deliberately picking the sample that maximizes it. Best-of-n at *serving* time has the same bias but at least it does not accumulate; RAFT bakes it into the weights, and then next round's samples are drawn from the biased policy. I always log mean output length per iteration on a RAFT loop; a monotone climb from 180 → 240 → 310 tokens with a flat human-preference score means the RM is paying you in length and you should stop.

### Be blunt with me: can a model actually teach itself something it doesn't already know, or is self-improvement just sharpening?

The honest 2026 answer is that the loops we have described **sharpen and consolidate existing capability rather than create new capability**, and the mechanism makes that inevitable — but the word "just" in the question is doing unfair work, because sharpening is worth an enormous amount.

The mechanism first. Every self-improvement loop samples from the model and filters. You cannot sample what the model cannot produce. If the probability of a correct solution to a problem class is exactly zero under the current policy, no amount of sampling surfaces one, no filter finds one, and the loop is provably blind to that class. What the loop *does* do is take capability that exists in the tail — solvable at pass@64 but not at pass@1 — and move it into the mode. You are converting sample-efficiency into reliability, redistributing probability mass toward behaviours the model could already produce occasionally. That is exactly why measured gains from self-training show up as large pass@1 improvements with much smaller pass@k improvements at large k, and it is why the loops saturate after a couple of iterations: once the tail is consolidated, there is nothing left to consolidate.

The critical qualifier, and the thing that makes this a genuinely contested question rather than a settled one: **the filter is external information.** A unit-test suite, a formal checker, a database-state comparison — these inject bits from outside the model into the loop. So the correct statement is not "a model cannot exceed itself"; it is "a model plus a verifier can exceed the model, bounded by what the verifier can certify." That is why self-improvement works spectacularly on math and code, where cheap perfect verifiers exist, and stalls on essay quality, taste and strategy, where the best available verifier is another model with the same blind spots. The frontier of this research line is exactly the question of how far verifier-mediated bootstrapping goes and whether weak verifiers can supervise strong generators.

**🗣 Say this in the room:** "Self-training loops sharpen — they turn pass@64 capability into pass@1 reliability, which is why gains saturate in two or three iterations. But the filter is external information, so a model plus a strong verifier genuinely can go beyond the model. The practical consequence is that I spend my engineering effort on the verifier, not on the training loop, because the verifier is the only thing in the system that can be a source of new information."

**⚠ Trap:** citing a self-improvement result on a benchmark with a leaky verifier as evidence of capability creation. If the verifier can be satisfied without solving the task — visible tests, an answer-format match that admits lucky guesses, a judge that rewards confidence — the loop concentrates on the exploit and the benchmark rises while nothing was learned. Before I believe any self-improvement number I want to see the held-out-verifier score next to the training-verifier score.

### Explain model collapse. Is it a real production risk for the loops we've been discussing, or is it a paper result about a pathological setup?

Both, and the distinction is where the interesting answer lives.

The mechanism first. Every generation step is a lossy resampling. When you sample from a model and train the next model on those samples, you are estimating the distribution from a finite sample, and finite samples under-represent the tails. Train on that, sample again, and the new model's tails are thinner still. Iterate and the distribution contracts toward its high-density modes: first the rare-but-valid outputs disappear (early collapse — variance loss), then the model converges toward a low-entropy point mass that does not resemble the original data at all (late collapse). The Nature-published recursion work demonstrates this cleanly, and importantly it happens even with no approximation error — finite sampling alone is sufficient. It is the same statistical failure as repeatedly photocopying a photocopy.

Now the production-relevance answer, which is what actually distinguishes a senior response. The catastrophic version of the result requires *fully replacing* the training data with the previous generation's output each round. If you *accumulate* — keep the original human data and add synthetic data on top — the follow-up literature finds collapse is substantially mitigated, because the original distribution keeps anchoring the tails. And critically, every loop we have discussed has a *filter* attached: STaR keeps only verified-correct traces, RAFT keeps only top-ranked ones. A filter is external information injected into the loop; it is not pure self-recursion. That is why self-training with a verifier works at all, and it is why "model collapse means synthetic data is bad" is a misreading I would push back on in a design review.

**🔍 Failure taxonomy — collapse symptoms, in the order you will see them.** (1) Output-length variance shrinks; the model produces suspiciously uniform-length answers. (2) Distinct-n / self-BLEU across samples at fixed temperature degrades — sample 8 completions per prompt and they start being paraphrases. (3) Rejection-sampling yield *rises* while eval score is flat — the model is producing more of what the filter already liked, not more capability. (4) Rare formats, rare languages, rare entity classes vanish from outputs entirely. (5) Only at the end does aggregate benchmark score drop, which is why you cannot use benchmark score as your collapse detector.

**Preservation practices I actually enforce:** keep a fixed replay fraction of original human/production data in every round (I use 20–40%); measure diversity every round with a cheap metric — mean pairwise embedding distance among `n` samples per prompt, plus distinct-3 — and set a regression gate on it, not just on quality; sample at temperature ≥ 0.8 during the Grow phase even though you serve at 0.2, because generation temperature is your only diversity dial; deduplicate synthetic data semantically, not just by exact string, because near-duplicates are the actual mass; and hold out a *frozen* set of original-data prompts to evaluate on every round so you can see distribution drift rather than measuring the model on data it produced.

### How many distillation examples do I actually need, and how do I choose which prompts to distill on?

The prompt distribution is the entire experiment. I have seen teams agonize over student architecture and learning rate while sampling their distillation prompts from a hand-written list of 300 examples an engineer invented on a Tuesday, and that choice dominates everything else.

The rule: distillation prompts must be drawn from production traffic, or from something you can defend as a sample of it. If you have logs, sample stratified — by intent class, by tenant, by input length bucket, by whether the request historically failed. Oversample the tail deliberately: if 2% of your traffic is a hard intent, at uniform sampling it is 2% of your training data and the student will be terrible at it, whereas at 10% you spend a little capacity and cover it. If you have no logs, you generate prompts synthetically but you seed them from real artifacts — real documents, real schemas, real ticket titles — because a language model asked to invent user queries produces a distribution that is far narrower and far more polite than reality.

On volume: for a narrow, well-specified task (classification, extraction, fixed-schema transformation) 3k–10k examples usually saturates a 7B student, and going to 100k buys you almost nothing. For open-ended generation with style requirements, 20k–50k. For agentic multi-turn tool use, you need more and the unit is *trajectories* not turns — 5k–20k complete trajectories, which is a very different data-collection problem because each trajectory is expensive to generate and to verify. The reliable way to find your number is a data-scaling curve: train at 1k, 3k, 10k, 30k and plot eval score against log(n). It is nearly always a clean line that bends, and the bend tells you where to stop spending. Running that curve costs four LoRA runs — a day of work — and it has saved me a five-figure data budget more than once.

**⚠ Trap:** deduplicating your distillation prompts by exact string and calling it done. Production traffic is full of near-duplicates ("reset my password", "how do I reset password", "password reset pls"), and after teacher generation you have thousands of near-identical `(prompt, response)` pairs that inflate your dataset size, inflate your apparent eval score if any leaked into eval, and teach the student to be extremely confident about one narrow region. Embed and cluster, cap per-cluster counts, then sample.

**🏋 Drill:** take any dataset you have, 5,000 examples. In 90 minutes, unaided: embed the prompts with a small local encoder, cluster to 200 clusters, cap each cluster at 25 examples, and report the before/after count plus the cluster-size histogram. Pass criterion: you can state what fraction of your original data was concentrated in the top 10 clusters. If it is over 40% — which it usually is — you have just explained why your last fine-tune generalized badly.

### Show me the distillation loss you'd actually write for a mixed setup — hard labels, teacher logits, and a KL term. And tell me how you'd pick the weights.

The production loss is almost never pure KD. It is a mixture, and each term is there for a specific reason:

```python
def distill_loss(student_logits, teacher_logits, labels, mask, T=2.0, alpha=0.7):
    # student_logits, teacher_logits: [B, L, V]; labels: [B, L]; mask: [B, L] float
    ce = F.cross_entropy(
        student_logits.transpose(1, 2), labels, reduction="none")          # [B, L]
    ce = (ce * mask).sum() / mask.sum().clamp(min=1)

    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t     = F.softmax(teacher_logits / T, dim=-1)
    kd = -(p_t * log_p_s).sum(-1) + (p_t * p_t.clamp_min(1e-9).log()).sum(-1)
    kd = (T * T) * (kd * mask).sum() / mask.sum().clamp(min=1)

    return alpha * kd + (1.0 - alpha) * ce, {"ce": ce.item(), "kd": kd.item()}
```

The cross-entropy term anchors the student to the actual ground-truth token when you have one — for distillation on verified data, `labels` *is* the teacher's sampled sequence, so CE is sequence-level distillation and KD is the soft-target refinement on top of it. Where you have genuine human labels (a curated set) the CE term is the thing that keeps the student from inheriting teacher errors on that slice, which is why I always mix in a few thousand human-verified examples even in a 100%-synthetic pipeline.

Picking the weights: `T` in 1–4, `alpha` in 0.5–0.9. My starting point is `T=2, alpha=0.7` and I do not tune them before I have a data-scaling curve, because data quantity and prompt distribution move the eval far more than these do. The one diagnostic worth watching is the *ratio* of the two logged terms. If `kd` is two orders of magnitude below `ce` the KD term is doing nothing and you have effectively run plain SFT — usually because your teacher is so peaked that at `T=2` the soft targets are still near-one-hot, which means raise `T`.

**⚠ Trap:** applying the temperature to the student at training time and then forgetting that at inference the student runs at `T=1`. That is fine and expected — the temperature is a training-time lens, not a property of the student. But the mirror error is real: people distill at `T=4`, observe the student's logits are unusually flat, and then serve at high temperature "to match," which produces an incoherent model. Serve at whatever your task needs, independent of the distillation temperature.

### When would you deliberately *not* distill, even when the cost math looks good?

Four situations, and being able to name them is the judgment signal.

**The task distribution is not stable.** Distillation compiles a snapshot. If your product's inputs shift monthly — a new document type, a new tenant vertical, a new language — you are signing up for a retraining treadmill, and each retrain is a new eval, a new rollout, a new rollback plan. A prompted frontier model absorbs distribution shift for free. My threshold: if the input distribution is stable enough that a 3-month-old eval set still predicts production quality, distillation is safe. If it is not, fix the eval problem first.

**You need the tail.** Distillation to a small student is a capacity trade: the student gets excellent at the modal 95% and noticeably worse at the weird 5%. For a support-deflection bot that is fine — route the weird ones to a human or to the frontier model. For a legal or medical assistant where the weird cases are the ones that matter, you have optimized the wrong percentile. The architecture that resolves this is a router: distilled student handles the modal traffic, an escalation predicate sends the rest to the teacher, and you *measure* the escalation rate as an SLO. That hybrid usually captures 80% of the cost saving with 5% of the quality risk, and proposing it unprompted is how you sound like you have shipped one.

**The model upgrade treadmill will strand you.** Frontier capability at a given price point has been improving fast enough that a student you distilled and quantized and deployed can be beaten, on quality *and* on cost, by next quarter's cheap tier from the same provider. That is a real risk and the honest framing is an option-value argument: distillation converts a variable cost into a fixed engineering commitment, and that is only correct when your volume is high enough and stable enough that the payback period is short. I want payback inside two quarters, not four.

**📅 Volatile:** the rate of price-per-capability decline is itself the input to that decision, and it has been steep. Do not carry a number from this guide into a room — re-derive the payback with the current small-tier price the week of your loop.

**Licensing forbids it.** Which is the next question, and it is the one that actually kills projects.
