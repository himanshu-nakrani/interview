### You keep saying "build the eval set before the training set." Why is that ordering non-negotiable?

Because otherwise the eval set is defined by the training set, and you have built a machine that cannot tell you it failed.

The mechanism of the failure is specific. If you gather 20,000 examples first and then hold out 2,000, your eval is drawn from exactly the distribution your training data happened to cover. Every gap in your data collection — the customer segment you didn't sample, the query type your logs under-represent, the failure mode nobody thought to capture — is a gap in *both* sets simultaneously. Your eval will happily report 94% and the model will fall over on the 6% of production traffic that neither set contains. You have measured coverage of your own blind spot, which is by construction zero.

Building the eval first inverts the dependency. The eval set is an assertion about **what the product must do**, written from product requirements, support tickets, user interviews and adversarial imagination — not from whatever data was convenient. Then the training set's job is to make that eval pass. Any capability in the eval you can't get training data for is now a visible, tracked gap rather than an invisible one.

The concrete practice I enforce:

1. **Write the eval spec before writing any data pipeline.** A list of capability slices with target metrics: "field extraction on scanned invoices, exact-match ≥ 90%"; "abstention on unanswerable queries, false-answer rate ≤ 2%"; "no regression >2pt on the general suite."
2. **Populate the eval from sources structurally different from your training sources.** If training comes from production logs, eval should include hand-authored adversarial cases and cases from a *later* time window. Same-source eval is how temporal drift hides.
3. **Freeze it and version it.** The eval set is an artifact with a hash, checked into the repo, never edited to make a number go up. If you change it, you change its version and you re-baseline everything.
4. **Hold out a genuinely blind slice** that only one person can see, opened once, before ship. Everything you look at repeatedly, you overfit to — including by hand, through a hundred small decisions about which examples to add.
5. **Baseline the prompted model on it before you train anything.** Half the fine-tuning projects I've seen killed were killed here, because the prompted baseline already cleared the bar and nobody had checked.

**⚠ Trap:** the eval set that grows during the project. Someone finds a failure, adds it to eval, fixes it in training, repeats. After three cycles the eval is a list of things you already fixed and it has zero predictive power for anything new. Failures found during development go into a *separate* growing regression set; the frozen eval stays frozen.

**🗣 Say this in the room:** "The eval set encodes the product requirement; the training set is an attempt to satisfy it. If I build training data first, my eval inherits its blind spots and I lose the ability to detect them. So: eval spec, frozen eval set with a version hash, prompted baseline measured on it, and only then do I decide whether a fine-tune is even the right instrument."

### Where does SFT data come from? Give me the full taxonomy and the constraint on each.

Five sources, and every real program is a blend. I'd score a candidate on whether they name the legal and consent constraints unprompted, because that's the part that gets companies in trouble.

**1. Human-written from scratch.** Domain experts writing prompt–response pairs against guidelines. Highest quality ceiling, highest cost, slowest. Non-negotiable for domains where the model has no competent teacher: specialist legal reasoning, clinical judgment, your company's internal procedures. **📅 Volatile / 💰:** expert-authored examples in specialist domains have been quoted to me in the $20–$150 per item range depending on length and expertise; verify current vendor pricing rather than quoting mine.

**2. Human-edited model output.** A model drafts, an expert corrects. This is the highest-leverage mode and it's what I default to — you get maybe 3–5× the throughput of writing from scratch at 80–90% of the quality, and the edit distance itself is a free quality signal (large edits flag hard cases; zero edits flag either easy cases or a lazy annotator). Keep the edited version as the label and log the pre-edit draft; the (draft, edit) pairs are also preference data later.

**3. Synthetic generation from a model you're allowed to use.** Self-Instruct-style bootstrapping, Evol-Instruct, persona-driven, seed-grounded — covered in the next several questions. Cheap, scales to millions, and its failure mode is homogeneity rather than error.

**4. Distillation from a stronger model.** Same mechanism as (3), but with the explicit intent of transferring capability from a frontier model into a small one. Extremely effective. **The constraint is legal, not technical:** several major providers' terms of service prohibit using their outputs to develop or train competing models. Some open-weight licences carry output-usage and anti-distillation restrictions too, and some require naming derivatives after the source family. This is a question for counsel and it is a real gating item — I've seen a fine-tuning project reach eval before anyone read the ToS. Ask it on day one, in writing, and record the answer.

**5. Consented production logs.** Your own users' interactions. The best distribution match you will ever get, because it *is* the production distribution. The constraints are consent, PII, and contract — see the next question.

The mixing judgment: I want a **spine of human-authored or human-edited data defining the target behavior**, and synthetic data for *volume and coverage* around it. Synthetic-only sets look fine on eval and are subtly flat in production — the diversity of real user phrasing is very hard to synthesize. Human-only sets are too small and too expensive to cover the tail. A blend I've been happy with is roughly 20% human-authored/edited, 50% synthetic verified, 30% production logs, but the right ratio is entirely task-dependent and the number to defend is *why you chose it*, not the number itself.

**⚠ Trap:** treating a public instruction dataset as free data. Many popular ones are themselves distilled from a commercial model with restrictive terms, and the licence on the *dataset* does not launder the terms on the *outputs*. Trace provenance one level up before you use anything, and record the provenance per-example in your schema so you can excise a source later without rebuilding everything. Provenance as a first-class column has saved me twice.

### We have three years of production chat logs. Walk me through turning them into a training set responsibly.

This is the highest-value data you own and the fastest way to create a legal incident. I run it as a pipeline with hard gates, and the first three gates have nothing to do with ML.

**Gate 1 — legal basis.** Does your ToS, DPA and privacy policy actually permit using customer content for model training? For enterprise contracts, the answer is very often "no, unless the customer opted in," and it may vary per-customer. This has to be resolved per-tenant with a machine-readable flag, not per-product. Build the pipeline so `tenant.training_consent == True` is a filter at the source, and make it fail closed. If someone asks you to "just use the ones that seem fine," that's the moment to escalate rather than comply.

**Gate 2 — PII and secrets.** Production chat contains names, emails, account numbers, addresses, API keys, and occasionally full credit-card numbers people pasted. Run detection (regex for structured identifiers plus an NER pass plus a secret-scanner for key formats), then either drop the example or replace with consistent surrogates. Consistent surrogates matter: if you replace every email with `user@example.com`, you teach the model to emit that string; if you replace with varied realistic fakes, you preserve the format-learning without memorizing anyone. Then verify — sample 500 processed examples and have a human look for leakage, because your detector's recall is not 100%.

**Gate 3 — the training-memorization risk.** Even after scrubbing, a model trained on production data can regurgitate. Fine-tuning on a small dataset for 3 epochs is exactly the regime where memorization happens. Test for it: take 200 rare strings that appear exactly once in the training data and check whether the model can be prompted to complete them. If it can, reduce epochs, increase data, or drop that content.

Now the ML gates:

**Gate 4 — quality labelling, per turn.** Not every logged assistant turn is a good label. Signals I use, in order of reliability: (a) the human agent sent the model's draft unedited — strong positive; (b) thumbs-up — weak positive, sparse and biased; (c) the conversation resolved without escalation — moderate positive; (d) the user did not immediately rephrase the same question — moderate; (e) thumbs-down or an escalation — strong negative, and *keep these*, they're your negative set and your eval material. Attach the flag per-turn, as discussed earlier.

**Gate 5 — the feedback-loop problem.** Your logs contain the *current model's* outputs. Training on them is self-training, and it compounds the current model's biases and blind spots — the model's own quirks become the target. The mitigations: prefer human-edited turns over accepted-as-is turns (the edit is genuine human signal), cap the fraction of the training set that comes from unedited model output, and keep a fixed injection of human-authored data every cycle. This is the applied version of model collapse, and it's the version you'll actually hit.

**Gate 6 — temporal split for eval.** Split by time, not randomly. Train on months 1–30, evaluate on months 31–36. A random split leaks the same customer, the same incident, the same product version into both sides and inflates your numbers. Also split by *tenant* if you're multi-tenant, so you're measuring generalization to a new customer rather than to a new sentence from a known one.

**🔍 Failure taxonomy — production-log training sets.** (1) Model is great on old queries, weak on new features → your logs predate the feature; add synthetic coverage. (2) Model leaks another customer's data → PII gate failed or the log itself contained cross-tenant content. (3) Model inherits a bad habit nobody can trace → unedited model output was over-weighted. (4) Eval says 95%, prod says 70% → random rather than temporal/tenant split. (5) Legal asks you to delete one customer's contribution → you can't, unless provenance is per-example and you can retrain. That last one is why the provenance column is mandatory.

**🗣 Say this in the room:** "Production logs are the best distribution match and the biggest liability. I gate on per-tenant training consent that fails closed, scrub PII with consistent surrogates and verify by human sample, prefer human-edited turns over model-accepted turns to avoid self-training drift, split eval temporally and by tenant, and carry per-example provenance so a deletion request is executable."

### Implement Self-Instruct for me. What does the filtering step actually do?

**📄 Paper:** Wang et al. (2022/2023), "Self-Instruct: Aligning Language Models with Self-Generated Instructions" — bootstrapped ~52k instructions from **175 seed tasks** using a base GPT-3, and showed the resulting fine-tune closed much of the gap to instruction-tuned models. It replaced "pay humans to write 100k instructions" with "pay humans to write 175 and let the model expand." Alpaca (Stanford, 2023) is the famous application of the recipe.

The loop, and the filtering step is the whole trick:

```python
import random, re
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def self_instruct(llm, seeds, target=20_000, sim_thresh=0.7):
    pool = list(seeds)                     # ~175 human-written (instruction, input, output)
    while len(pool) < target:
        # 1. sample few-shot exemplars: mostly machine-generated, some human
        shots = random.sample([p for p in pool if p["src"] == "human"], 6) \
              + random.sample([p for p in pool if p["src"] == "machine"] or pool, 2)
        # 2. generate new instructions
        new_instrs = llm.generate_instructions(shots, n=8)
        for instr in new_instrs:
            # 3. ROUGE-L filter against the ENTIRE pool
            if max(scorer.score(instr, p["instruction"])["rougeL"].fmeasure
                   for p in pool) > sim_thresh:
                continue
            # 4. heuristic filters
            if not (3 <= len(instr.split()) <= 150): continue
            if re.search(r"\b(image|picture|graph|file|draw)\b", instr): continue
            # 5. classify input-first vs output-first, then generate the instance
            inp, out = llm.generate_instance(instr)
            pool.append({"instruction": instr, "input": inp, "output": out,
                         "src": "machine"})
    return pool
```

**The ROUGE-L filter against the entire existing pool is the load-bearing line.** Without it, the model regenerates the same twenty instruction archetypes forever — "Write a poem about X," "Summarize the following," "Translate to French" — with different nouns. The similarity gate forces each new instruction to be lexically distant from everything already collected, which is a crude but effective proxy for novelty. The threshold (0.7 in the original) is the diversity/yield dial: lower it and you get more diverse but fewer instructions per API call; raise it and your set collapses toward the seeds.

The other structural choice worth naming: **the few-shot exemplars are drawn mostly from human seeds and only partly from machine-generated ones.** This is a damping term against drift. If you sample exemplars purely from the growing machine pool, each generation is conditioned on the previous generation's quirks and you get recursive amplification — the model-collapse dynamic in miniature. Keeping human seeds in the prompt anchors the distribution.

**⚠ Trap:** the yield curve is brutally nonlinear and people budget for the average. Early iterations pass the similarity filter almost always; by 20k instructions you may be rejecting 70–90% of generations because the pool already covers the space. If you budgeted API spend on the early acceptance rate you'll be 5× over. Measure acceptance rate as a function of pool size before committing, and know that the honest way to get past the plateau is not more sampling but **new seeds and new grounding** — see the seed-grounded question.

**⚠ Trap:** ROUGE-L over the whole pool is O(n²) and will become your bottleneck at 50k+. Use MinHash/LSH for candidate retrieval and only ROUGE the top-k neighbours, or switch to embedding-space nearest-neighbour with a FAISS index. I've watched a generation job spend 90% of wall clock in the deduplication scorer.

### What does Evol-Instruct add on top of that?

**📄 Paper:** Xu et al. (2023), "WizardLM: Empowering Large Language Models to Follow Complex Instructions" — introduced **Evol-Instruct**, which rewrites existing instructions into harder ones rather than sampling new ones from scratch. It attacked the specific weakness of Self-Instruct sets: they are diverse in *topic* but flat in *difficulty*, clustering around simple single-step requests, because that's what a model produces when asked for "a new instruction."

Two evolution operators:

**In-depth evolving** makes one instruction harder along five axes: add constraints, deepen (ask for more detail or a further consequence), concretize (replace general terms with specific ones), increase reasoning steps (turn a one-hop question into a multi-hop one), and complicate the input (add a table, add irrelevant context, add a nested structure).

**In-breadth evolving** generates a *new* instruction in the same domain but on a different topic — mutation for coverage rather than difficulty.

You apply these iteratively — evolve the pool, evolve the evolved pool, several rounds — and each round produces a harder generation. The key operational detail is the **elimination step**: after each evolution you must discard failed evolutions. An evolution fails when the evolved instruction is semantically identical to the original (the model just reworded it), when it became unanswerable or self-contradictory, when the response is a refusal or an apology, or when the response is just a copy of the instruction. Skip the elimination step and your pool fills with degenerate mutations that are longer but not harder — and "longer" is exactly what a length-biased eval will reward, so you'll think it worked.

The practical reason I use Evol-Instruct: **difficulty coverage is what separates a training set that moves your eval from one that doesn't.** If your model already handles everything in your dataset, the gradient signal is near zero and you're spending compute to reinforce what it already does. Evolution is a cheap way to manufacture examples in the band just above current capability.

**⚠ Trap:** evolving without measuring difficulty, and mistaking length for difficulty. Verify with the model you're training: sample k=4 responses per evolved instruction from the *current* checkpoint and score them. Instructions where all 4 pass are too easy — deprioritize. Where 0 of 4 pass, they may be too hard *or* broken — inspect by hand, because unanswerable instructions look identical to hard ones from the pass-rate alone. Where 1–3 of 4 pass, that's your high-value band. This pass-rate binning is one of the highest-return things you can do to a synthetic dataset and almost nobody does it.

**💰 Math:** three rounds of evolution over a 20k pool with 4 in-depth variants per instruction, at ~600 input + 400 output tokens per evolution call and ~1,200 output tokens per response generation. Round 1: 20k × 4 = 80k evolutions. Evolution calls: 80,000 × (600 × $3/1e6 + 400 × $15/1e6) = 80,000 × ($0.0018 + $0.006) = **$624**. Response generation for survivors, assume 50% survive elimination: 40,000 × (800 × $3/1e6 + 1,200 × $15/1e6) = 40,000 × $0.0204 = **$816**. Round 1 total ≈ $1,440; three rounds with a shrinking pool, call it **$3,000–$4,000** for a difficulty-laddered 100k-instruction set. That is cheap against one week of an engineer's time, which is the correct comparison to make out loud. **📅 Volatile:** prices.

### Explain persona-driven and seed-grounded synthetic generation, and why they beat pure bootstrapping.

Both are answers to the same problem: a model asked to "generate a diverse instruction" samples from its own mode and produces a narrow, repetitive distribution no matter how you prompt it. The fix in both cases is to **inject external entropy** — condition generation on something the model didn't choose.

**Persona-driven generation** conditions each request on a described persona: "You are a maritime insurance underwriter in Rotterdam with 12 years of experience." The persona changes vocabulary, assumed background, question framing and what counts as a good answer, and because you supply a large library of personas, the diversity of the output set inherits the diversity of the library rather than the model's sampling entropy. The scaled version of this idea — a curated library on the order of a billion personas mined from web text, used to drive synthetic data generation across math, logic, instructions and more — was published by a Tencent group in 2024 as "Persona Hub." **📅 Volatile:** check the licence terms on any public persona library before using it commercially. You can also build a small one cheaply from your own customer segmentation, which is usually better-targeted anyway: 200 personas derived from real user roles will out-perform 100,000 generic ones for a product fine-tune.

**Seed-grounded generation** conditions on a real document: a support ticket, a page of your API docs, a contract clause, a code file, a Postgres schema. The prompt is "given this document, write three questions a user might ask that this document answers, and their answers." Now the *content* is anchored in reality rather than in the model's priors, which does three things at once: it guarantees factual grounding, it gives you exact topical coverage of your corpus, and it hands you the retrieval-context for free so you can build RAG-shaped training examples with real distractors.

Seed-grounded is my default for any product fine-tune, and I'd push back hard on a plan that started with pure Self-Instruct when the team has a document corpus sitting right there. The coverage argument is decisive: with seed grounding, "did we cover the whole product?" becomes "did we generate from every document?", which is a checkable claim. With pure bootstrapping it's unanswerable.

The composition I actually run: **grounding document × persona × task-type**, sampled as a cross-product. Take a doc, sample a persona, sample a task type from {factual lookup, multi-hop across two docs, comparison, unanswerable-from-this-doc, ambiguous-needs-clarification}, and generate. That third axis is what gives you the abstention and clarification buckets discussed earlier, and getting them from the same generator as everything else keeps the style consistent.

**⚠ Trap:** persona conditioning that leaks into the output. If the persona is in the generation prompt, the generated *answer* often addresses the persona explicitly ("As a maritime underwriter, you'll know that...") — and now you've trained your model to make assumptions about the user. Generate with the persona, then run a scrub pass that strips persona references from the response, or instruct the generator to produce responses that are persona-appropriate in depth but persona-neutral in address. Check for this explicitly; I've seen it ship.

**🗣 Say this in the room:** "Pure self-bootstrapping saturates because the diversity ceiling is the generator's own sampling entropy. Persona and document grounding inject external entropy — the diversity of the output set becomes the diversity of your persona library or your corpus. For a product fine-tune I generate from the cross product of grounding document, persona, and task type, because that makes coverage a checkable property rather than a hope."

### How do you verify synthetic data? Give me the ladder from cheapest to most expensive.

Verification is the difference between synthetic data and synthetic noise, and the ladder matters because the cheap rungs handle most of the volume.

**Rung 1 — structural validation, ~free.** Does the JSON parse? Does it match the schema? Is the tool name in the offered list? Are required fields present? Does the response contain the citation markers you require? Is the length within bounds? This catches 5–20% of generations depending on task and costs microseconds. Run it first, always.

**Rung 2 — execution / ground-truth verification, cheap and decisive where available.** This is the highest-value rung and its availability determines your whole strategy:
- **Code:** run it. Unit tests, or at minimum does it compile/import and does it produce the expected output on given inputs. A generated code example that doesn't run has no business in a training set.
- **Math:** check the final answer against a reference, or verify with SymPy, or check numerically.
- **SQL:** execute against a real schema and compare result sets — not string-compare the query, which fails on equivalent formulations.
- **Structured extraction:** compare against the source document with a deterministic check (does the extracted date literally appear in the text?).
- **Retrieval-grounded answers:** verify every claim is supported by a span in the provided context via entailment or exact-match on key entities.

Where you can execute, execute. **📄** The general principle here — that verifiable rewards beat learned judges when they exist — is the same principle underneath RLVR, and it applies just as much to SFT data filtering. If your task has a verifier and you're using an LLM judge instead, you've chosen the worse tool.

**Rung 3 — a second model as critic, cheap-ish and noisy.** Have a *different* model (different family if possible, to decorrelate errors) score the pair against a rubric, or answer the same prompt independently and check for agreement. Two flavors: **critique-and-filter** (score 1–5, keep ≥4) and **consistency filtering** (generate the answer k times, keep only prompts where the answers agree — a strong proxy for "this question has a determinate answer"). The consistency variant is underused and it's excellent for building an abstention set, because the *inconsistent* ones are exactly the ambiguous or unanswerable questions.

**Rung 4 — human spot-check, expensive and mandatory.** See the next question. This isn't a filter, it's a *measurement* of the rungs above.

**⚠ Trap:** the generator and the critic being the same model. A model scores its own output higher than a third party's — self-preference bias is well documented in LLM-as-judge work — and more importantly its errors are correlated with its own generation errors, so it systematically cannot see its own failure modes. If GPT-family generated it, don't have GPT-family judge it, and if you must, at minimum use a different model version and never the same call chain.

**⚠ Trap:** filtering so hard on a judge score that you strip all the difficulty out. A rubric judge rewards clean, confident, well-structured answers. Filter at ≥4/5 and you keep the easy questions and drop the ones where the correct answer is hedged, partial, or an abstention — precisely the behaviors you were trying to teach. I stratify: apply the judge filter within difficulty bands rather than globally, so hard examples compete against hard examples.

**💰 Math:** verification cost on a 100k-example set. Structural: free. Execution (say 40% of examples are code/SQL): 40,000 × 2 seconds of sandbox = 22 CPU-hours ≈ **$3**. Second-model critique at 1,500 input + 200 output tokens on a cheap model at $0.25/$1.25 per Mtok: 100,000 × (1,500 × $0.25/1e6 + 200 × $1.25/1e6) = 100,000 × ($0.000375 + $0.00025) = **$62.50**. Human spot-check of 500 examples at 4 minutes each = 33 hours at $40/h = **$1,320.** Total ≈ $1,400 to verify 100k examples, dominated entirely by the human rung — which is the right shape, and which tells you the optimization target is *what you sample for humans*, not the model calls. **📅 Volatile:** prices.

### How many human spot-checks, and how do you sample them?

Uniform random sampling is the wrong default and it's what most teams do. The purpose of human review on a synthetic set is not to filter the set — you can't afford to review 100k items — it's to **estimate the error rate of your automated filters and to find failure modes you didn't anticipate.** Those two purposes want different sampling.

**For error-rate estimation: uniform random, and size it from the confidence interval you need.** If you want to know your post-filter error rate to within ±2 percentage points at 95% confidence, and you expect an error rate around 5%, the normal approximation gives n ≈ 1.96² × 0.05 × 0.95 / 0.02² = 3.84 × 0.0475 / 0.0004 = **456 examples.** That's the number, and I'd round to 500. Notice that if you only need ±5pp, n drops to 73 — so decide the precision you need before committing 33 hours of expert time. This arithmetic is worth doing out loud in an interview; it demonstrates you treat data quality as a measurement problem.

**For failure-mode discovery: stratified and adversarial.** Sample deliberately from: the lowest-confidence decile of your judge scores; the boundary band right at your filter threshold (that's where the filter's errors concentrate); the longest and shortest 1%; each generation source and each persona bucket; every cluster in an embedding-space clustering, so you touch every region of the set; and every example the automated verifier *rejected*, at a lower rate, because false rejections are invisible otherwise.

**The one everyone skips: review the rejects.** If your verifier is throwing away 40% of generations, sample 100 rejects and confirm they deserved it. I have seen a schema validator rejecting 30% of a set because of a trailing-comma bug, which nobody noticed because rejects don't appear in any dashboard.

**Structure of the review itself.** Reviewers get a rubric with named error categories, not a thumbs up/down — "factually wrong," "doesn't follow the instruction," "format violation," "correct but low quality," "instruction is broken/unanswerable," "should have abstained." The category distribution is the actual output of this exercise, because it tells you what to fix. A 6% error rate is not actionable; "4 of the 6 points are the model inventing citation numbers on grounded questions" is.

**Cadence.** Spot-check *every* generation batch, not just the first. Generation quality drifts when you change a prompt, change a model version, or move to a new grounding corpus. I run a 100-example smoke review on every batch and the full 500-example measurement at each major version.

**🏋 Drill:** given a 50k synthetic set and 8 hours of one expert reviewer, design the review plan. Pass criterion: you allocate a fixed uniform sample sized to a stated confidence interval, a stratified discovery sample across generation sources and score deciles, a reject sample, and you name the error taxonomy in advance. If your plan is "review 500 random ones," you've failed the drill.

### How do you measure whether a dataset is actually diverse? Give me metrics I can compute.

Diversity is the property people claim and never measure, so having concrete instruments here is a differentiator. Four levels, and I want all four because they catch different collapses.

**1. Lexical — distinct-n and self-BLEU.** `distinct-n` is the count of unique n-grams over total n-grams across the corpus; it falls as the set becomes repetitive. `self-BLEU` samples pairs and computes BLEU between them — *higher* self-BLEU means *less* diversity. Both are cheap and both are dominated by surface form, so they catch "every instruction starts with 'Write a'" but miss semantic redundancy. Track the distribution of first-3-tokens across your instructions as a poor-man's version; if 30% of your set starts with the same trigram, you have a problem you can see immediately.

**2. Semantic — embedding-space coverage.** Embed every instruction, then measure: (a) mean pairwise cosine similarity (rising = collapsing); (b) the number of clusters at a fixed distance threshold, or the effective rank / participation ratio of the embedding covariance matrix, which gives you "how many independent directions does this dataset span"; (c) **cluster-size entropy** — cluster into k=100 and compute the entropy of the cluster-size distribution against log(100). Entropy near log(k) means uniform coverage; a low value means a few giant clusters and a long tail of singletons, which is what pure Self-Instruct produces.

**3. Task-type and difficulty — categorical coverage.** Classify each example into your task taxonomy (extraction, summarization, multi-hop, abstention, tool-call, refusal, clarification) and look at the histogram against your *target* distribution, which should be derived from production traffic. This is the one that actually predicts eval movement. A set can be beautifully diverse in embedding space and contain zero abstention examples.

**4. Coverage against the eval set.** For each eval example, find its nearest training neighbours in embedding space. Eval examples with no close training neighbour are the ones you'll fail. This is the most directly actionable diversity measurement there is, and it's cheap — one FAISS index and one query per eval item. Plot the distribution of nearest-neighbour distances; the right tail is your prioritized data-collection list.

**⚠ Trap:** optimizing a diversity metric directly. If you filter for low pairwise similarity you will preferentially keep weird, malformed, and off-distribution examples, because garbage is maximally distant from everything. Diversity metrics are *diagnostics*, and the correct response to a bad one is "generate more from an under-covered region," never "delete the dense region." The one exception is deduplication, which removes near-identical items rather than optimizing a global score.

**📐 Numbers you must know:** my rough working thresholds, offered as starting points rather than laws. Mean pairwise cosine similarity of instruction embeddings above ~0.55 in a general instruction set means you're collapsing. Any single cluster holding >5% of a 100-cluster partition is a redundancy flag worth inspecting. And more than ~10% of eval examples having no training neighbour within your typical intra-cluster distance means your training set doesn't cover your requirement.

### Design me the deduplication pipeline. Why isn't exact hashing enough?

Because the duplicates that hurt you are near-duplicates, and near-duplicates are the majority.

Exact hashing catches copy-paste. It misses: the same instruction with a different name substituted; the same document chunk with different whitespace; the same question generated twice by your synthetic pipeline with one word changed; the same support ticket appearing in your logs under two ticket IDs. In a Self-Instruct-style set, exact duplicates might be 1% and near-duplicates 25%.

Why it matters concretely: duplicates are **implicit up-weighting.** An example appearing 40 times gets 40× the gradient contribution, and there is no principled reason for it to. The classic symptom is a model that emits one specific phrasing constantly — trace it back and you'll find that phrasing was duplicated in the training set. Duplicates also **inflate your eval** if the duplicate landed on the other side of the split, and they **waste compute** in direct proportion to their share.

The three-stage pipeline:

**Stage 1 — exact.** Normalize (lowercase, collapse whitespace, strip punctuation for the hash only), SHA-256, drop collisions. Microseconds per document. Do this on the instruction, the response, and the concatenation, separately — identical instructions with different responses are a *different* problem (label inconsistency) and you want to see them, not silently drop them.

**Stage 2 — near-duplicate via MinHash + LSH.** Shingle each document into overlapping word n-grams (n=5 is standard for text), MinHash to a signature of ~128 permutations, band the signature into LSH buckets, and only compute exact Jaccard within a bucket. This gets you approximate-Jaccard dedup in roughly O(n) instead of O(n²), and it's what every large pretraining pipeline uses. Threshold around Jaccard 0.8 for aggressive dedup, 0.9 for conservative. `datasketch` in Python does this in a few dozen lines.

**Stage 3 — semantic dedup via embeddings.** MinHash still misses paraphrase: "How do I reset my password?" and "What's the process for changing my login credentials?" share almost no 5-grams. Embed, build a FAISS index, and for each item find neighbours above a cosine threshold (~0.95 for near-identical, ~0.90 if you're being aggressive), then keep one representative per cluster. **📄 Paper:** Abbas et al. (2023), "SemDeDup: Data-efficient learning at web scale through semantic deduplication" — showed you can remove a large fraction of a web-scale corpus by embedding-space clustering with no loss, and sometimes a gain, in downstream quality. It's the reference for "semantic dedup is not just cleanup, it's a training-efficiency win."

**Which representative to keep** is a decision people make randomly and shouldn't. Keep the one with the highest verifier score, or the human-authored one over the synthetic one, or the longest response if length correlates with quality in your set — but decide it explicitly and record it.

**⚠ Trap:** deduplicating across the train/eval boundary in the wrong direction. You must dedup *eval against train* and drop from **train**, never from eval — dropping the eval item hides the contamination instead of fixing it, and it silently shrinks and biases your eval set. Order of operations: build eval, freeze it, then dedup train against the frozen eval, then dedup train against itself.

**💰 Math:** dedup as a compute saving. A 300k-example set with 22% near-duplicates at Jaccard ≥ 0.85, average 1,400 tokens: removing 66,000 examples removes 92.4M training tokens per epoch. At 2 epochs and 6·N·D for a full FT of an 8B: 6 × 8e9 × 1.85e8 = 8.88e18 FLOPs saved, which at 1.58e15 FLOP/s on four H100s at 40% MFU is 5,620 seconds ≈ **1.6 GPU-hours × 4 GPUs ≈ $16**. Small in dollars — but it's also 1.5 hours off every iteration of your experiment loop, and iteration speed is the thing that actually determines whether the project succeeds.

### How do you balance difficulty in a training set, and why does it matter?

Because gradient signal comes from error, and a dataset full of things the model already does perfectly contributes almost nothing while costing full compute.

The measurement is the same pass-rate binning from the Evol-Instruct answer, and it's the single most useful thing you can do to a dataset. Take your **current** model (the actual checkpoint you're about to fine-tune, not a frontier model), sample k=4 or k=8 responses per training prompt, score them with whatever verifier you have, and bin by pass rate:

- **k/k pass (too easy).** The model already does this. Keep some — they're your anti-forgetting anchor and they stabilize format — but they should not dominate. If 70% of your set is here, your fine-tune will barely move.
- **1..k−1 pass (the productive band).** The model can sometimes do it. This is where learning happens. I want this to be the plurality of the set, 40–60%.
- **0/k pass (too hard or broken).** Two very different populations mixed together: genuinely hard examples the model needs to learn, and *broken* examples where the instruction is unanswerable, the gold label is wrong, or the verifier is buggy. **You must hand-inspect this bin** — I sample 50 and categorize. Training on a mislabeled example at 0/k pass rate is actively harmful; it's a maximally-confident wrong gradient.

The trap here is real and worth naming: **hard examples and wrong examples are indistinguishable by pass rate.** Every automated difficulty-filtering scheme I've seen fails on this, and it's why the hand-inspection of the 0/k bin is non-negotiable rather than nice-to-have. In my experience 20–40% of a 0/k bin in a synthetic set is broken data, not hard data.

Beyond pass-rate binning, three other balancing axes:

**Length balance.** If your hard examples are systematically longer, the model learns "hard → verbose" and you've trained a length bias into it. Check the correlation between difficulty bin and response length; if it's strong, either shorten the hard responses or lengthen a sample of the easy ones.

**Task-type balance against production traffic.** Your synthetic pipeline will over-produce whatever it finds easy to produce. Reweight to match the *deployment* distribution, not the generation distribution. If 30% of production queries are abstention-worthy and 3% of your training set is, you've built the wrong model regardless of how clean the data is.

**Curriculum ordering — and my honest position: don't bother by default.** Ordering easy-to-hard has a long literature and mixed results for LLM SFT specifically. With shuffled data and a couple of epochs, the model sees everything anyway. I'd rather spend the effort on the composition of the set than on the order, and I'd say so directly if asked, because "we implemented a curriculum" is a common resume line with weak evidence behind it.

**🗣 Say this in the room:** "I bin training examples by the pass rate of the model I'm about to fine-tune, sampling k=8. The 0-of-8 bin gets hand-inspected because hard and mislabeled look identical from the outside, and in synthetic sets a large minority of it is broken data. Then I target the partial-pass band as the plurality of the set, keeping some already-solved examples as a format anchor."

### What does "format normalization" mean and what breaks if you skip it?

It means making every example in your dataset agree on the conventions the model will learn, because **the model will learn the conventions whether or not you chose them.**

The concrete list I run on every dataset:

**Markdown consistency.** Some sources use `**bold**`, some use `<b>`. Some use `-` bullets, some `*`, some `1.` for everything. Some wrap code in triple backticks with a language tag, some without, some in single backticks. If your set is mixed, the model learns a probabilistic mixture and emits inconsistent formatting — which then breaks any downstream parser you built assuming one convention.

**Whitespace and newlines.** Trailing whitespace, `\r\n` vs `\n`, leading blank lines, number of blank lines between sections. These are *tokens*. A response starting with `\n\n` in half your set teaches the model to sometimes lead with blank lines.

**Unicode.** NFC normalization, smart quotes vs straight quotes, non-breaking spaces (which come in from HTML scraping and tokenize as separate tokens), zero-width characters. I've debugged a "model output looks fine but the regex doesn't match" ticket that was a U+00A0 in the training data.

**Structural preamble.** Does every response start with a restatement of the question? Does it start with "Sure!"? Some sources do and some don't. Pick one and enforce it — including *removing* the preamble if you don't want it, which is a cheap way to cut output tokens across every future request.

**Citation and JSON conventions.** Key order, whether numbers are quoted, date format, how citations are marked (`[1]` vs `[doc_3]` vs inline). One convention, enforced by a validator, or you get a model that emits three formats depending on which part of your training distribution the prompt lands near.

**Language and locale.** If 4% of your set is in a different language, the model will occasionally answer in it. If dates are mixed US and ISO, it will mix them.

The reason this is worth real engineering effort: **format inconsistency is the cheapest quality loss in the entire pipeline.** It costs nothing intellectually to fix and it directly determines parse success rate, output-token count, and whether the product's UI renders correctly. And unlike most data problems, it's fully automatable — a normalization function plus a validator that hard-fails on violations, run as a pipeline stage with a report.

**⚠ Trap:** normalizing so hard you destroy legitimate variation. If you strip all markdown, the model stops producing structure entirely, including when it would help. If you force every response into the same skeleton, you get a model that emits section headers on "hi." Normalize *conventions* (which bullet character), not *content decisions* (whether to use bullets at all) — and hold out a set of examples whose correct answer is a bare sentence, to keep the model capable of brevity.

**🏋 Drill:** take any public instruction dataset, write a normalization + validation pass, and report the violation counts by category before and after. Pass criterion: you find at least four distinct convention conflicts and your validator hard-fails on all of them afterward. Thirty minutes. Every dataset has them.

### You have 200,000 candidate examples and the budget to train on 20,000. What's your selection function?

The framing I want to establish first, because it inverts most people's instinct: **this is not "throw away 90% of our data," it is "choose the 20,000 that move the eval."** More data is not better past the point where it's redundant, and a smaller curated set routinely beats a larger noisy one — that's the operational content of the LIMA result. So selection is a design problem with a stated objective, and I'd write the objective down before touching the data.

The pipeline, as a sequence of filters and then a scored selection:

**Filter first, score second.** Hard filters remove anything you'd never want regardless of score: format-validation failures, examples that fail execution/entailment verification, exact and near-duplicates (three-stage: hash → MinHash/LSH → embedding), anything flagged by contamination detection against the frozen eval, PII survivors, and examples above your truncation limit. On a raw 200k set I'd expect these to take out 25–45%, which is already most of the reduction. Track the count removed per filter and *look at the biggest one* — an unexpectedly large bucket is almost always a bug in the filter, not a property of the data.

**Then score the survivors on four axes and select, not sort.** The distinction matters: if you rank by a single quality score and take the top 20k, you get the 20k easiest, cleanest, most homogeneous examples, and your fine-tune barely moves. What you want is a *constrained* selection.

- **Quality** — verifier score, judge rubric score, human-authored/edited bonus, source-provenance prior. Use as a floor, not a ranking: drop below threshold, don't sort by it.
- **Difficulty** — the k-sample pass rate of the model you're about to fine-tune. Target the partial-pass band as the plurality.
- **Diversity** — cluster the survivors in embedding space into ~500 clusters and select *within* clusters proportionally, so no cluster can dominate. This is the single most important constraint and it's the one a naive top-k destroys.
- **Task-type distribution** — hard quotas per bucket, set from production traffic, not from what the generator produced. If abstention is 12% of production, it is 12% of the selection, and if you don't have 2,400 good abstention examples you go generate them rather than filling the quota with weak ones.

Concretely: allocate the 20,000 across task-type quotas, then within each quota allocate across embedding clusters, then within each cluster pick by difficulty band with a quality floor. It's stratified sampling with a scoring tiebreak, and it takes an afternoon to implement.

**⚠ Trap:** running the selection once and never revisiting it. Selection is conditional on the *current* checkpoint's pass rates, so after round 1 the difficulty bands have shifted — examples that were in the productive band are now solved. The second round should re-score, not reuse. Teams that skip this find that round 2 of fine-tuning does nothing and conclude "we've saturated," when actually they retrained on already-learned data.

**🗣 Say this in the room:** "Hard filters for correctness, dedup and contamination first — that's usually a third of the set. Then a *constrained* selection rather than a top-k on a quality score, because top-k gives you the easiest and most homogeneous examples: task-type quotas from production traffic, stratified across embedding clusters, picking within the partial-pass difficulty band subject to a quality floor. And I re-score before every subsequent round, because difficulty is relative to the current checkpoint."
