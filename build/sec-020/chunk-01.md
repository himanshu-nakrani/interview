### Start me at the beginning: if you already have SFT, why do you need a reward model at all? Why not just train on more good answers?

Because SFT can only teach you to imitate a demonstration, and for most of what we actually care about, nobody can write the demonstration. The mental model is the difference between a unit test and a code review. SFT is a unit test: here is the input, here is the exact expected output, match it. A reward model is a code review: I cannot write your function for you, but I can look at two versions and tell you which one I'd merge. Preference data exists because *ranking is cheap and generating is expensive*, and that asymmetry is the entire economic justification for RLHF.

There is a second, deeper reason that gets missed. SFT's loss is maximum likelihood on a single target, which means it can only ever push probability mass *toward* one string. It has no mechanism for pushing mass *away* from a bad string. If your model has a bad habit — hedging, sycophancy, fabricating a citation with confident formatting — SFT can only fight it indirectly by showing more counterexamples and hoping the good behavior wins on volume. Preference learning is the first objective in the stack that has an explicit negative gradient: this output, less likely. That is why alignment behaviors (refusal calibration, tone, not-making-things-up) respond to preference training in a way they stubbornly do not respond to more SFT.

Third: SFT is bounded by the annotator. If a human writes the target, the model asymptotes at human-writing quality on that task. Preference data is bounded by the annotator's *discrimination* ability, which is strictly higher than their generation ability. I cannot write a better legal memo than a Harvey user's associate, but I can reliably tell you which of two memos cited a case that doesn't exist. That gap — generate-hard, verify-easy — is where RLHF gets its headroom.

**🗣 Say this in the room:** "SFT maximizes likelihood of one target, so it can only imitate and can never explicitly downweight a bad behavior. Preference learning gives you a signed gradient over a *comparison*, which is both cheaper to collect and can exceed demonstration quality, because humans discriminate better than they generate."

**⚠ Trap:** treating "we did RLHF" as a capability claim. RLHF on a preference dataset does not make a model smarter at math; it makes it produce outputs that the reward model scores highly, which mostly means better-formatted, better-hedged, more helpful-*seeming* outputs. The capability gains people attribute to RLHF in the 2022–2023 era were mostly *elicitation* — the base model could already do it, and alignment made it reliably do it on request. Say that out loud and you sound like someone who has read the ablations.

### Summarize the InstructGPT paper for me precisely. I want the three stages and the numbers.

**📄 Paper:** Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback*. Its contribution was not any single algorithm — Bradley-Terry is from 1952, PPO from 2017, KL-regularized LM fine-tuning from Ziegler et al. (2019). Its contribution was demonstrating the full three-stage pipeline at scale on a general instruction distribution and showing that it dominated raw scale: a 1.3B InstructGPT was preferred by human labelers over the 175B GPT-3 base. That single result — a 100× smaller model winning on human preference — is what redirected the field's spend from pretraining-only to post-training.

The three stages, in order:

**Stage 1, SFT.** Labelers write demonstration responses to prompts drawn from the API prompt distribution plus labeler-written seeds. On the order of ~13k training prompts. Fine-tune GPT-3 on these with standard next-token loss. This gets you a model that follows instructions at all, and — critically — it becomes the *initialization for everything downstream*.

**Stage 2, reward modeling.** For each prompt, sample K responses from the SFT model (K between 4 and 9), have a labeler rank all K, and expand into C(K,2) pairwise comparisons — 6 pairs at K=4, 36 at K=9. Train a scalar reward head on a 6B model (they explicitly note the 175B RM was unstable to train) with the Bradley-Terry log-loss. On the order of ~33k prompts, but far more comparisons.

**Stage 3, PPO.** Optimize the SFT policy against the frozen RM with a per-token KL penalty against the SFT reference, on ~31k prompts. They also introduce **PPO-ptx**, which mixes pretraining gradients into the PPO update to counter the alignment tax — the observed regression on academic NLP benchmarks caused by aligning.

Three details that separate someone who read the paper from someone who read the abstract:

1. **All C(K,2) comparisons from one prompt go in the same batch element**, not shuffled independently into the dataset. If you shuffle them, each response is seen in many separate gradient steps, the RM overfits in a single epoch, and accuracy degrades. Putting them together also makes it K forward passes instead of C(K,2), so it's cheaper *and* better. This is the single most quotable implementation detail in the paper.
2. **The RM is trained for exactly one epoch.** More than one and it overfits — preference data is small and noisy.
3. **The alignment tax is real and measured**, and PPO-ptx is the mitigation. That honesty is why the paper aged well.

**⚠ Trap:** saying "InstructGPT is how ChatGPT was trained." It is the direct ancestor and the public reference, but the deployed systems diverged substantially and immediately. In an interview, cite it as *the canonical published description of the RLHF pipeline*, not as a description of any current production system.

### I'm building a preference dataset for a coding assistant. Walk me through how you'd actually collect it — the interface, the units, the volume.

Start from the decision the data has to support, because that determines the unit. If I'm training a reward model for a Cursor-style inline-edit product, the unit is not "which response is better" in the abstract — it is "which diff would this engineer have accepted." So the collection interface should show the *same context* (file, cursor position, instruction) and two candidate diffs, rendered exactly as the product renders them.

**The three collection modes and when each is right:**

*Pairwise (A vs B, forced choice).* Cheapest per judgment, highest agreement, the default. Downsides: no magnitude — "A is barely better" and "A is vastly better" produce identical training signal — and you need O(n) pairs to cover the space. Always include a tie option internally and then *drop ties from training* rather than forcing a coin flip; a forced coin flip on a genuine tie injects pure label noise, and Bradley-Terry has no way to represent it.

*k-wise ranking (rank 4–9 responses).* This is what InstructGPT did and what I default to when I control the generation. You get C(K,2) pairs from one annotator session at maybe 2.5× the time cost of a single pairwise judgment, so the pairs-per-dollar improves by ~2–4×. Annotators are also more consistent when they see the whole set — comparative context calibrates them. Ranking beyond K≈7 degrades fast; humans lose track.

*Ratings converted to pairs.* Absolute Likert scores (1–5) on individual responses. Attractive because it's the interface product teams already have. I'll cover the conversion pitfalls in the next question — the short version is that inter-annotator scale drift makes cross-annotator pairs unreliable.

**Sourcing the candidates matters as much as the labels.** The pairs must come from *your policy's* output distribution, not from a mix of GPT-4 and a 1B model, because a RM trained on easy pairs learns "detect the weak model," which is a feature that does not exist in the region where the policy actually operates. Concretely: sample K completions from the current SFT checkpoint at temperature ~1.0 with different seeds. Hard, near-tie pairs are the expensive, valuable ones.

**📐 Numbers you must know:** volume. InstructGPT's RM used on the order of 33k prompts with K-wise rankings, so roughly 10⁵–10⁶ pairwise comparisons. For a *narrow domain* RM — one task, one output format — I have seen useful signal at 5k–10k well-curated pairs, and I would not plan a program below 3k. At a realistic vendor rate of $0.50–$2.00 per pairwise judgment for general text and $5–$20 for expert-domain judgments (legal, medical, senior-engineer code review), 10k expert pairs is $50k–$200k. **📅 Volatile:** vendor rates move; verify with Surge/Scale/Mercor before quoting a number in a planning doc.

**💰 Math:** this is why the archetype matters. For a consumer product, 100k general pairs at $1 = $100k and you can afford to iterate quarterly. For Harvey, 10k pairs judged by licensed attorneys at $15 = $150k for a dataset one-tenth the size, and you *cannot* iterate quarterly — which is the actual reason enterprise-vertical AI companies lean on production implicit feedback (accept/reject/edit-distance signals) far harder than consumer companies do.

### Product wants to give me their existing 5-star rating data instead of paying for pairwise labels. What do I do with it and what breaks?

You can convert ratings to pairs, and often you should, but you have to be explicit about the three things the conversion destroys.

The mechanical conversion: for two responses to the *same prompt* with ratings r_A > r_B, emit the pair (A preferred). Discard equal ratings. That's it. The subtlety is entirely in which pairs you're allowed to form.

**Break #1: cross-annotator scale drift.** Annotator 1's "4" and annotator 2's "4" are not the same quantity. Some people never give 5s; some never give below 3. If you form pairs across annotators, you are encoding personality differences as preference signal. **The rule I enforce: only form pairs between responses rated by the same annotator on the same prompt.** If your rating collection didn't guarantee that — and product-telemetry rating data almost never does — you must either restrict to the subset that did, or per-annotator z-score the ratings first and only pair when the normalized gap exceeds a threshold. Z-scoring assumes each annotator saw a comparable difficulty mix, which is itself worth checking.

**Break #2: no shared prompt.** Ratings from production telemetry are one rating per *conversation*, so you have no second response to the same prompt. That data cannot become preference pairs at all. It can become a filtering signal for rejection-sampling SFT, or a binary label for KTO-style training, but there is no pair in it. I have watched a team spend six weeks trying to force thumbs-up/thumbs-down telemetry into a Bradley-Terry pipeline; it was the wrong objective for the data they had.

**Break #3: magnitude information is thrown away, and then quietly leaks back as noise.** A 5-vs-1 pair and a 4-vs-3 pair produce the same loss target. The 4-vs-3 pair is probably 60% noise. My default is to *drop* pairs where the rating gap is 1 unless I've verified they're same-annotator and my agreement stats are strong, and to keep the wide-gap pairs. You lose data volume and gain effective signal.

**⚠ Trap:** the "thumbs-down means the response was bad" assumption. In production, thumbs-down is heavily confounded with *task difficulty* and *user mood*, and thumbs-up is confounded with response length and confidence. Raw production feedback has a base rate of maybe 1–3% of turns receiving any rating at all, and that 1–3% is a wildly non-random sample — dominated by users who hit a failure or who are unusually engaged. Before it enters any training pipeline I want a human audit of a few hundred thumbs-downs categorized by *why*, because typically 30–50% of them are "model was right, user was wrong" or "user wanted a different task."

**🗣 Say this in the room:** "Ratings convert to pairs only within-annotator and within-prompt. Across annotators, a 4 and a 4 aren't the same number, and you'd be training the reward model on annotator personality. If the data doesn't have two responses to the same prompt, it isn't preference data at all — it's a binary label, and I'd point it at KTO or rejection sampling instead."

### How do you measure whether your annotators agree, and what number would make you stop the collection and rewrite the guidelines?

For binary pairwise preference, the metric is **Cohen's κ** for two annotators on the same items, or **Krippendorff's α** when you have variable overlap and missing data — which describes every real labeling program, so α is what I actually run. Both correct raw agreement for chance agreement, which matters enormously here: on a forced binary choice, two annotators flipping coins agree 50% of the time, so raw agreement of 65% is nearly worthless and κ makes that visible (κ ≈ 0.30).

The operating procedure I run on every labeling program:

1. **Overlap set.** Route 5–10% of items to at least three annotators. This is a permanent tax, not a pilot-only thing — agreement drifts as annotators get tired, as guidelines get reinterpreted, and as the policy's output distribution shifts.
2. **Gold questions.** 2–3% of items are pre-adjudicated by me or a senior domain expert with a known correct answer. This catches an individual annotator degrading, which α over the pool will average away.
3. **Per-annotator α, not just pooled.** Pooled α of 0.6 can mean "everyone is moderately noisy" or "eight people at 0.75 and two people at 0.1." Those need completely different interventions.

**📐 Numbers you must know — the agreement bands I use:**
- α or κ **below 0.4**: stop. Your guidelines are underspecified or the task is genuinely ambiguous. Don't buy more data; rewrite the rubric and re-run the pilot.
- **0.4–0.6**: typical for open-ended helpfulness judgments and workable, but it caps your reward model. A useful rule of thumb: RM test accuracy will land near the human agreement rate, because that rate is the ceiling on label consistency. If humans agree 70% of the time, an RM scoring 72% is *at* the noise ceiling, not underfit.
- **0.6–0.8**: good. This is what you get when the rubric decomposes the judgment (correctness, then instruction-following, then safety, then style) instead of asking "which is better."
- **Above 0.8** on subjective helpfulness: be suspicious. Usually it means the pairs are too easy — you're comparing a good model to a bad one, and the RM you train will be useless in the regime that matters.

That last one is the counterintuitive part and it's a great thing to volunteer. **High agreement plus high RM accuracy is often a symptom of an easy dataset, not a healthy one.**

**⚠ Trap:** treating disagreement as noise to be averaged out. Sometimes disagreement is *signal* that the population genuinely splits — one annotator prefers terse, one prefers thorough. Averaging those into a single scalar reward trains a model toward the bland midpoint, which is a real and observed cause of the "RLHF makes models boring" complaint. If you find a systematic split, the right move is either to pick a side explicitly in the guidelines (an editorial decision, not a data decision) or to condition the reward model on a persona/preference token so you can serve both.

**🏋 Drill:** take 100 preference pairs from any public dataset, label them yourself blind, then compute Cohen's κ against the shipped labels — by hand, from the confusion matrix: κ = (p_o − p_e)/(1 − p_e). Twenty minutes, no library. Pass criterion: you compute p_e correctly from the marginals (not assume 0.5) and you can state in one sentence why your κ is what it is.

### Derive the Bradley-Terry loss and write me the reward-model training step from memory.

Bradley-Terry (1952) is a model of paired comparisons: assign each item a latent positive strength, and the probability that i beats j is strength_i / (strength_i + strength_j). Parameterize the strength as exp(r) with r the output of your network, and the whole thing collapses to a logistic:

P(y_w ≻ y_l | x) = exp(r(x,y_w)) / (exp(r(x,y_w)) + exp(r(x,y_l))) = σ(r(x,y_w) − r(x,y_l))

Maximum likelihood on observed comparisons gives the loss:

L = −E[ log σ( r(x, y_w) − r(x, y_l) ) ]

That is the entire objective. The thing to internalize is that **only the difference is identified.** Adding any constant to r for all responses to a given prompt leaves the loss unchanged. The reward model's absolute scale is meaningless; only within-prompt differences carry information. Everything painful about reward models downstream traces back to this one fact.

The architecture: take a decoder-only transformer, drop the LM head (vocab-sized projection), and attach a single linear layer d_model → 1. Run the full sequence (prompt + response), take the hidden state at the **final token** of the response, project to a scalar. Last token, because causal attention means only that position has seen the whole response.

```python
import torch, torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, base_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_name, torch_dtype=torch.bfloat16)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1, bias=False)
        nn.init.normal_(self.head.weight, std=1.0 / (self.backbone.config.hidden_size + 1) ** 0.5)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1                    # index of final real token
        pooled = h[torch.arange(h.size(0), device=h.device), last]
        return self.head(pooled.float()).squeeze(-1)            # (B,) scalar reward

def rm_step(model, batch, opt):
    # batch: chosen/rejected already tokenized to the SAME max length
    r_w = model(batch["chosen_ids"],   batch["chosen_mask"])
    r_l = model(batch["rejected_ids"], batch["rejected_mask"])
    loss = -torch.nn.functional.logsigmoid(r_w - r_l).mean()
    acc  = (r_w > r_l).float().mean()
    loss.backward(); opt.step(); opt.zero_grad()
    return loss.item(), acc.item()
```

Three implementation details that are graded:

**Concatenate chosen and rejected into one forward pass** in practice (`torch.cat` then split the outputs) so that batch-norm-free but still batch-order-sensitive kernels and any dropout see them identically, and so you halve the launch overhead. I wrote it as two calls above for readability; in production it's one.

**Initialize the head small.** A default `nn.Linear` init gives you rewards of magnitude ~10 at step 0, the sigmoid saturates, gradients vanish, and the run looks dead for a thousand steps. Scaling init by 1/sqrt(d+1) is the standard fix.

**Left-pad or mask correctly.** If you right-pad and take `h[:, -1]` you are pooling a pad token. The `attention_mask.sum(dim=1) - 1` indexing above is the safe form.

**⚠ Trap:** adding a margin or a length term to this loss "to help." Some variants do add a margin proportional to the rating gap (r_w − r_l − m), and that's defensible when you have magnitude information. But adding an explicit length penalty *into the RM loss* is a mistake I'd push back on in review: you've now baked a length prior into the reward function, which the policy will then optimize against in ways you cannot inspect. Handle length bias at the data level and at the eval level, not by hand-editing the objective.

### Do you initialize the reward model from the base checkpoint or the SFT checkpoint, and how big should it be relative to the policy?

Initialize from the SFT checkpoint, essentially always. The reason is representational, not sentimental: the reward model has to score responses drawn from the SFT policy's distribution, and the SFT checkpoint's residual stream already encodes the features that distinguish good from bad *within that distribution* — instruction adherence, format conformance, whether it stopped when it should. A base checkpoint has to relearn all of that from a preference dataset three orders of magnitude smaller than pretraining. In practice, SFT-init reward models converge faster and score a couple of points higher on held-out pair accuracy.

The exception worth knowing: if your SFT model was heavily specialized (say, fine-tuned to emit only JSON) and you want a general-purpose RM, SFT-init drags in the specialization as a prior. Then a base or a general-instruct init is better. But the default is SFT-init, and if a candidate says "base" without qualification I'd probe why.

**On size:** the RM does not have to match the policy, and often shouldn't. InstructGPT used a 6B RM for a 175B policy and explicitly reported that the 175B RM was unstable to train. The pragmatic considerations:

- **Bigger RM = later overoptimization onset.** This is the most important empirical fact about RM size. Larger reward models resist Goodharting for longer, which means you can push further up the KL axis before gold-reward performance turns over.
- **The RM runs on every rollout token batch during PPO**, so its inference cost is a first-order term in your training throughput. A 70B RM against a 7B policy will dominate your step time.
- **Below ~1B the RM is usually too weak** to distinguish subtle quality differences; it latches onto surface features (length, markdown, confidence words) because those are the only features it can represent well.

My default for a 7–8B policy is a 7–8B RM, same family, SFT-init. For a 70B policy I'd still consider an 8B RM first and measure whether pair accuracy on my hard eval slice is the bottleneck before spending 70B-worth of inference on every rollout.

**💰 Math:** RM inference cost inside PPO. A 7B RM scoring a batch of 512 sequences of ~1,024 tokens is a prefill of 512 × 1024 = 524k tokens. At 2·N·T FLOPs for a forward pass, that's 2 × 7e9 × 5.24e5 ≈ 7.3e15 FLOPs. One H100 at a realistic 400 TFLOP/s bf16 achieved does that in ~18 seconds. Swap to a 70B RM and it's 180 seconds per PPO step on one GPU — you now need ten times the RM serving fleet to keep the trainer fed. That arithmetic, not model quality, is usually what decides RM size.

### How do you evaluate a reward model? Give me the metric and tell me what a good number looks like.

The primary metric is **pairwise accuracy on a held-out preference set**: the fraction of held-out pairs where r(chosen) > r(rejected). It is the only metric that directly measures the thing you trained. But raw accuracy is close to useless without three decompositions, and volunteering those decompositions is the difference between a mid and a senior answer.

**Decomposition 1: by slice.** Accuracy on easy pairs (large model vs small model) will be 90%+ and tells you nothing. Accuracy on *hard* pairs — two samples from the same policy at the same temperature, where a human annotator took 40 seconds to decide — is the number that predicts RLHF outcomes. I report both and I lead with the hard slice. A healthy general-purpose RM lands around 65–75% on genuinely hard same-policy pairs; anything at 90% on that slice means your "hard" slice isn't hard.

**Decomposition 2: against the human agreement ceiling.** If your annotators agree with each other 72% of the time on this slice, an RM at 72% is *at ceiling* and further training is fitting noise. Always report RM accuracy next to human-human agreement on the same items. Teams that don't do this chase a phantom 20 points of headroom that doesn't exist.

**Decomposition 3: by confound.** Compute accuracy on the subset where the chosen response is *shorter* than the rejected one. If your RM is 78% overall but 51% on the shorter-chosen subset, you have not trained a quality model, you have trained a length model with a quality-shaped veneer. Same test for markdown presence, for the presence of a numbered list, for "Certainly!"-style openers, and for refusal.

Beyond accuracy:

**Calibration.** σ(r_w − r_l) should be a calibrated probability that a human prefers w. Bin your held-out pairs by predicted probability and plot observed win rate; you want the diagonal. Reward models are typically overconfident, and overconfidence in the RM translates directly into aggressive, high-variance PPO updates.

**Best-of-n gold correlation.** The most decision-relevant eval: sample n=1,2,4,8,…,64 responses per prompt from the policy, pick the RM's argmax, and score *that* selection with a trusted gold judge (human, or a much stronger model on a small set). If the RM is good, gold score rises monotonically in n. If the curve rises then falls, you have found your overoptimization onset before spending a single PPO GPU-hour. **This is the cheapest possible early warning and I run it before every RLHF run.**

**⚠ Trap:** reward-model *loss* as a progress metric. Because only differences are identified, the loss can decrease by the model simply spreading rewards further apart — increasing the scale of r without improving ordering. Watch accuracy and the reward *margin distribution*, not loss.

### What is RewardBench, and what does a high score on it actually predict?

**📄 Paper:** Lambert et al. (2024), *RewardBench* (AI2) — the first standardized public benchmark for reward models, structured as curated prompt/chosen/rejected triples across categories (chat, harder chat, safety, and reasoning), scored as simple pairwise accuracy. Before it, every RM paper reported accuracy on its own private held-out split, which made cross-paper comparison meaningless. Its contribution is comparability, and it also surfaced that many RMs were substantially worse at safety and reasoning than their headline chat numbers suggested. **📅 Volatile:** there is a successor generation of the benchmark and the leaderboard composition changes; verify the current version and the top entries before your loop rather than quoting a name I may have stale.

What it predicts: **that your RM is not broken.** It is a floor test. An RM that scores near chance on RewardBench's reasoning slice will not produce a useful RLHF run, and that is genuinely worth knowing cheaply.

What it does not predict, and this is the answer the interviewer is fishing for:

**It does not predict performance on your distribution.** RewardBench prompts are general assistant prompts. If your policy emits SQL, or legal citations, or code diffs, RewardBench accuracy tells you approximately nothing about whether your RM can rank two candidate diffs.

**It does not predict resistance to overoptimization.** Two RMs with identical RewardBench accuracy can have wildly different Goodhart onsets, because accuracy on a fixed static dataset says nothing about behavior *off-distribution*, and off-distribution is exactly where a policy goes once it starts optimizing against you. A benchmark of static pairs cannot measure a dynamic, adversarial property.

**It is a static benchmark and therefore contaminable.** Once a benchmark is public and used for selection, the field selects for it. Treat leaderboard position as weak evidence.

**🗣 Say this in the room:** "RewardBench is a smoke test, not a fitness test. It tells me an RM isn't broken and lets me compare public checkpoints. It cannot tell me the two things I actually need: how the RM behaves on my task distribution, and how far up the KL axis I can push before it Goodharts. For both of those I need my own hard-pair eval set and a best-of-n gold-correlation curve."

### The scalar the reward model outputs — what does the number mean? Can I compare a 3.2 on one prompt to a 3.2 on another?

No, and this is one of my favorite screening questions because the wrong answer is very confident.

Bradley-Terry identifies rewards only up to a **per-prompt additive constant**. The loss depends solely on r(x, y_w) − r(x, y_l), both conditioned on the same x. Nothing in the training signal ever compares a response to prompt A against a response to prompt B. So the model is free to assign prompt A's responses rewards around +8 and prompt B's around −2, and that offset is arbitrary — it typically encodes something like "how easy is this prompt," or worse, some surface feature of the prompt text.

The consequences are concrete:

**You cannot threshold on the raw reward.** "Reject any response scoring below 0" is meaningless across prompts. If you need a quality gate, you need a per-prompt reference — for example, score the SFT model's response to the same prompt and threshold on the *difference*.

**Cross-prompt reward variance leaks into PPO as advantage noise.** If prompt difficulty creates a ±5 offset and the actual quality signal within a prompt is ±0.5, then batch-level advantage estimates are dominated by which prompts happened to be sampled. This is precisely why PPO implementations whiten advantages, and it's a large part of why GRPO's group-relative normalization (subtract the mean reward of the group of responses to the *same* prompt) works so well — it removes the per-prompt offset by construction, exactly matching what Bradley-Terry actually identified.

**Reward scale drifts across RM training runs**, so a KL coefficient β tuned against RM v1 is not valid against RM v2. Any time you retrain the RM, you must re-tune β or renormalize the reward. Teams get burned by this: RM v2 has better accuracy, they swap it in, the run explodes, and they blame the RM.

**The standard mitigations:** (1) normalize rewards so that the reward of the SFT model's own response is centered at zero — InstructGPT does a version of this; (2) whiten rewards within the PPO batch; (3) use group-relative advantages. All three are attacking the same identifiability hole.

**⚠ Trap:** "our reward model outputs 7.5, so the response is good." I have seen this shipped as a production quality gate. It silently becomes a prompt-difficulty detector. The fix is always a paired baseline.

### Tell me about position, length and formatting bias in raw preference data. How do you detect it and what do you do about it?

These are three distinct contaminants that all arrive in the same shipment, and every one of them is a shortcut the reward model would rather learn than learn quality.

**Position bias.** Annotators systematically prefer whichever response is shown first (or, for some interfaces, second). The magnitude in a poorly designed interface can be 5–15 percentage points. **Detection:** randomize A/B order per item, record the presented order, and compute the win rate of "position 1" across the dataset. If it deviates from 50% by more than a couple of points, you have position bias. **Mitigation:** randomize order (mandatory), and for high-stakes slices, present each pair twice in both orders to the same annotator and keep only the consistent judgments. That halves throughput and roughly doubles effective label quality on hard pairs. It also gives you a free per-annotator consistency metric.

**Length bias.** The big one. Humans prefer longer responses at roughly-equal quality, because length correlates with thoroughness and effort. In most raw preference datasets the chosen response is longer than the rejected one in something like 60–70% of pairs. A reward model will absolutely exploit this: length is a trivially extractable feature, and it explains a large chunk of the label variance for free. **Detection:** the shorter-chosen accuracy slice from earlier, plus a simple regression of reward on token count across your held-out set. If reward correlates with length at r > 0.4, you have a length model. **Mitigation, in order of preference:** (a) balance the dataset — subsample so that chosen-longer and chosen-shorter are near 50/50, which is expensive in data but clean; (b) instruct annotators explicitly that length is not quality and enforce with gold questions; (c) length-controlled evaluation downstream so you at least *measure* it honestly; (d) as a last resort, a length penalty in the RL reward, which is a blunt instrument that the policy will route around.

**Formatting bias.** Markdown headers, bullet lists, bolded key terms, and a confident opening line all raise human ratings independent of content. This one is insidious because it's *partially legitimate* — formatted answers genuinely are easier to read — so you can't just null it out. **Detection:** hold formatting constant. Take 200 pairs, strip all markdown from both sides, re-collect labels, and compare flip rate. A flip rate above ~15% means formatting is driving a meaningful share of your labels.

**⚠ Trap:** believing you've fixed length bias because you added a length penalty to the RL reward. You have not fixed the reward model; you have added a second, opposing bias and hoped they cancel. The observable failure mode is a policy that produces *truncated-feeling* answers on genuinely complex prompts (where long is correct) and still-padded answers on simple ones, because the penalty is uniform and the bias is not. Fix it in the data.

**💰 Math:** what length bias costs at serving time. If your RLHF'd model's mean output grows from 280 to 520 tokens with no quality gain, at $15/Mtok output that's 240 extra tokens × $15/1e6 = $0.0036 per call. At 2M calls/day that is $7,200/day — $216k/month of pure verbosity, plus the latency: 240 tokens at 40 tok/s of decode is +6 seconds on every response. Length bias is not an academic concern; it is a line item.

### Should I be using an LLM-as-judge as my reward model instead of training a scalar head? Sell me one side.

They are different tools and the honest answer is that in 2026 you often want both, wired differently. Let me separate the two things people conflate.

A **scalar reward model** is a fine-tuned encoder-style scorer: one forward pass, one number, ~10–50ms for a 7B model on a short sequence, differentiable-adjacent (you can use it inside RL). A **generative reward model / LLM-as-judge** prompts a strong instruct model to compare two responses (or grade one against a rubric) and emit a verdict, optionally with a chain of thought.

**Where generative judges win:**
- **Zero training data to start.** You can have a working judge in an afternoon with a good rubric. For a startup with 200 preference labels, this is decisive.
- **Rubric legibility.** When it says "B is better," it says *why*, and you can read it, argue with it, and fix the rubric. A scalar RM's 4.7 is unfalsifiable. In a customer-facing product where PMs need to reason about quality, this matters more than people admit.
- **Better on reasoning-heavy comparisons**, because the judge can spend inference compute. A scalar head does one forward pass and cannot deliberate.
- **Trivially updatable.** New quality criterion? Edit the prompt. A scalar RM needs a new dataset and a retrain.

**Where scalar RMs win:**
- **Cost and latency inside an RL loop.** This is the killer. A PPO run scores every rollout: 512 rollouts per step × 2,000 steps = 1M judgments. A 7B scalar RM does that in-cluster. A frontier-model judge at ~$3/Mtok input, ~1,500 tokens per judgment, is 1M × 1,500 × $3/1e6 = **$4,500 per run in judging alone**, plus the API latency serialized into your training step, plus rate limits. That is why production RLHF uses scalar RMs and reserves generative judges for evaluation.
- **Determinism and reproducibility.** A scalar RM at temperature 0 with fixed weights gives the same number forever. A judge behind an API silently changes underneath you when the provider updates the model, and your reward function moved without a commit. I treat "judge model version" as a pinned dependency for exactly this reason.
- **No self-preference bias.** Judges systematically favor their own outputs and outputs from their own family. If you're RLHF-ing a model and judging with a relative of it, you have a closed loop.

**The hybrid I actually recommend:** use a strong generative judge to *generate and audit preference labels* at scale (RLAIF-style), train a scalar RM on those labels, and run RL against the scalar RM. You get the judge's rubric legibility at data-creation time and the scalar RM's throughput at training time. Then keep the generative judge as your held-out evaluator — but only if it did *not* produce the training labels for the RM you're evaluating, or you've closed the loop again.

**🗣 Say this in the room:** "Generative judges for data creation and evaluation; scalar reward models inside the RL loop. The deciding factor is that RL scores a million rollouts per run — at frontier-API prices a judge costs thousands of dollars and serializes API latency into every training step, while a co-located 7B scalar head costs GPU-seconds. It's a throughput decision, not a quality decision."
