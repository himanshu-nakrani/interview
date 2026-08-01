### We're a legal-AI company. Our lawyers say the model doesn't "think like a lawyer." Design the training program, with a budget.

I want to interrogate the complaint before I spend anything, because "doesn't think like a lawyer" is a symptom description that maps onto at least four different technical problems, and they have wildly different price tags.

**Step 1 — decompose the complaint into measurable failures.** I would sit with three lawyers and collect 200 real failing cases, then bucket them:

- **(a) Cites a statute that does not exist, or misstates one that does.** → Retrieval/grounding failure. Not a training problem.
- **(b) Gets the law right but structures the answer wrong** — no IRAC, no citation format, hedges where a memo should be definitive. → Format/behavior. SFT.
- **(c) Reasons correctly about the facts but misses a doctrine a second-year associate would apply.** → Domain reasoning. SFT with expert-written reasoning traces, possibly RLVR against attorney-graded rubrics.
- **(d) Misreads the jargon** — treats "consideration" as its ordinary-English meaning, mishandles citation syntax, misparses a defined-terms section. → This is the *only* bucket where CPT is the right instrument.

In my experience the distribution is roughly (a) 40%, (b) 35%, (c) 20%, (d) 5%. That distribution is the answer to the question, and it says: **most of your budget should not go to training.**

**Step 2 — build the eval before anything else.** 500 attorney-authored tasks with attorney-authored gold answers and a grading rubric, split into dev (200) and a locked holdout (300) that never touches any training corpus and is decontaminated against every corpus you build. Cost: 500 tasks × 45 min of attorney time at ~$150/hr fully loaded = `500 × 0.75 × 150 = $56,000`. That number shocks people and it is the best money in the project — without it, every subsequent decision is unfalsifiable.

**Step 3 — the actual program, in ROI order.**

- **Retrieval over the corpus of authority** (statutes, case law, the client's own document set), with citation-verification as a hard post-check: every citation the model emits is looked up, and an unresolvable citation blocks the response. This alone kills most of bucket (a). ~10 engineer-weeks ≈ $65k plus infra.
- **SFT on 3,000–10,000 attorney-authored or attorney-reviewed examples** in the exact output format you want, including refusals and "insufficient authority" responses. Labeling cost at 20 min/example and $150/hr: `5,000 × 0.333 × 150 = $250k`. This is the largest line item and it is the one that buys buckets (b) and (c). It is also the item you can stage — 1,000 examples first, measure, then decide.
- **CPT: I would probably not run it, and I would say so.** With 30B tokens of legal text, the recipe from earlier costs ~$8.4k of compute and ~6 engineer-weeks, and it buys bucket (d), which is 5% of failures. Against a strong base model that has already seen a great deal of public legal text, the realistic delta is 0–2 points on the task suite. **I would run it as a cheap experiment at the *end*** — after retrieval and SFT — because by then I have an eval good enough to measure whether it helps, and if it does not I have lost $8k rather than three months.

**💰 Full budget:** eval $56k + retrieval $65k + SFT $250k + CPT experiment $50k ≈ **$421k** and about five months, with 80% of the value landing in the first two months from retrieval and the first 1,000 SFT examples. Against a workload where, say, each attorney-hour saved is worth $300 and the product saves 2 hours/attorney/week across 400 attorneys, that is `400 × 2 × 52 × 300 = $12.5M/year` of value — the program is not close to marginal.

**⚠ Trap:** starting with CPT because it feels like the most "serious" intervention. It is the most expensive, the slowest to measure, the hardest to maintain (you redo it on every base-model upgrade), and it addresses the smallest failure bucket. In a design review I would ask directly: "show me the failure taxonomy that says knowledge injection is the bottleneck." If the team cannot produce one, the CPT proposal is not ready.

**🗣 Say this in the room:** "Before I train anything I'd taxonomize 200 real failures. My prior is that 40% are grounding, 35% are format, 20% are reasoning, and 5% are genuinely 'the base model doesn't speak this language' — and only that last 5% is a continued-pretraining problem. So: eval first, retrieval second, SFT third, and CPT as a cheap experiment at the end once I can actually measure it."

### An enterprise customer wants a model continued-pretrained on their private monorepo. Talk me through it — including why you might refuse.

The code case is genuinely more favorable to CPT than the legal case, and it is worth saying why: a large private codebase contains *idioms, internal APIs, naming conventions and architectural patterns that appear nowhere in the base model's training data*, and those are exactly the "language of the domain" that CPT is good at. Retrieval helps but incompletely, because the useful knowledge is diffuse — it is not "which document says X," it is "what does code here look like."

**The sizing question first, because it usually settles it.** A large monorepo is 10–50M lines of code. At roughly 10 tokens per line, that is **100–500M tokens**. Against a CPT run that wants tens of billions of tokens, this is 0.1–1% of a reasonable mixture. You cannot run 100 epochs over it — you would memorize it and forget everything else. So the realistic design is: 500M unique tokens at 3 epochs = 1.5B domain tokens, plus general code and general text replay at 80%, for a ~7.5B-token run. That is `6 × 8e9 × 7.5e9 = 3.6e20` FLOPs = `3.6e20/(989e12 × 0.4) = 9.1e5` GPU-seconds = **253 GPU-hours ≈ $630 of compute**. The compute is trivial. Everything else is not.

**What I would actually build, in order:**

1. **Repository-aware retrieval first**, because it is cheaper, updates instantly when the code changes, and handles the 60% of queries that are "where is X defined" and "how do we usually do Y." A model CPT'd on Monday's monorepo is wrong about Wednesday's.
2. **CPT with repo-structure-aware packing.** This is the interesting technical content: do not pack files randomly. Pack files that belong together — same module, or ordered by import/dependency graph — into the same sequence, so intra-document attention can actually learn cross-file relationships. This is a real and material design decision and it is what separates a thoughtful answer from a generic one.
3. **Fill-in-the-middle formatting.** If the product is autocomplete rather than chat, the training objective should match: FIM reformats a file as `<prefix><suffix><middle>` so the model learns to complete with both left and right context, which is what an editor actually provides. Training on plain left-to-right and serving in an editor is a train/serve mismatch that costs you real accuracy.
4. **Then measure against the base-plus-retrieval baseline**, on a held-out set of real completions from the customer's own commit history — with a strict time split, training on commits before date T and evaluating on commits after.

**When I would refuse, and these are the reasons I would put in writing:**

- **The repo changes daily and the model does not.** A per-customer CPT'd model is a snapshot with a staleness half-life measured in weeks. Retrieval has no such property. If the customer's real complaint is "it doesn't know about the module we shipped last month," CPT is structurally the wrong tool.
- **Per-customer model economics.** One model per enterprise customer means one deployment, one eval suite, one upgrade path, one on-call surface, *per customer*. At 20 customers that is 20 models to re-run every time the base model is upgraded, which is every few months. **💰** If each re-run is $630 of compute plus 1 engineer-week ($6.5k) of eval and validation, 20 customers × 4 upgrades/year = `20 × 4 × $7,130 = $570k/year` of pure treadmill. LoRA adapters make this much better (one base, N adapters, adapters swappable at serve time) and should be the default for per-customer adaptation.
- **Data governance.** Training on a customer's proprietary code means their code is in your weights. You must be able to answer: can it be extracted? (Yes, sometimes, especially for text repeated in the corpus.) Can you delete it on request? (Not without retraining.) Is it isolated from other customers' models? (It had better be.) If the contract has a deletion-on-termination clause, a trained-in weight is a compliance problem you cannot engineer around after the fact.

**🗣 Say this in the room:** "Code is one of the better CPT cases because internal idioms genuinely aren't in the base model — but I'd still build retrieval first, because the repo changes daily and the weights don't. If I did train, I'd pack by dependency graph rather than randomly, use FIM to match the serving format, and use LoRA adapters rather than full CPT so I'm not maintaining twenty forked base models."

### Two weeks after we shipped a CPT'd model, a customer found it reproducing a proprietary internal document verbatim. Walk me through the incident.

This is a data-governance incident, not a model-quality incident, and the first thing I do is treat it as one — which means the response has a legal and communications track running in parallel with the technical one, and I do not get to decide alone when it is closed.

**Immediate containment.** Roll back to the pre-CPT model or to a retrieval-based configuration. You cannot patch weights on a deadline, so serving-layer mitigation is the only fast lever: an output filter that checks generated text against a hash index of the sensitive corpus (n-gram bloom filter over the documents, same machinery as decontamination, applied at output time). This is a real mitigation you can ship in a day and it should already have existed.

**Then the diagnosis, because "why did it memorize this" has a specific and usually boring answer.**

Memorization in language models is driven overwhelmingly by **duplication count**. The well-established empirical picture — Lee et al.'s dedup work and the memorization-quantification literature that followed — is that verbatim regurgitation rises sharply with the number of times a sequence appears in the training data, and rises with model size. A document seen once is rarely reproducible; a document seen a few hundred times often is. So my first query is: **how many times does this document appear in the CPT corpus?**

The usual answers, in order of frequency:

1. **The corpus was not deduplicated**, or was deduplicated per-shard rather than globally. The document exists in 300 copies because it is in an email thread, a wiki, an archive, and a backup directory.
2. **Epoch count.** If the domain corpus was small and you ran 15 epochs to hit a token target, every document was seen 15 times *on top of* whatever duplication existed. This is the trap from earlier — controlled epoching is fine, epoching *duplicated* data multiplies.
3. **It was in a boilerplate position.** Legal footers, headers and templates appear in every document in a corpus and are therefore massively over-represented.
4. **The document was also in the eval or the SFT set** and got reinforced at low LR, where things stick hardest.

**The permanent fixes, which go in the postmortem:**

- Global exact-substring and MinHash dedup on the CPT corpus, non-negotiable, before any future run.
- **A memorization eval as a gate.** Take 1,000 sequences from the training corpus, feed the first 50 tokens, greedily decode 100, and measure exact-match rate against the continuation. Track it per checkpoint. A rising curve is your early warning, and it is a five-minute eval. Any run that ships must have this number in its report.
- Cap epochs per source and report them.
- Sensitive-document classification *before* corpus assembly, with a hard exclusion list, plus canary strings inserted into genuinely sensitive documents so you can test extraction directly.
- Reconsider the architecture: if the requirement is "customer data must be deletable," then customer data belongs in a retrieval index (deletable in milliseconds) and not in weights (deletable only by retraining). That is the architectural lesson and it is the one worth escalating.

**⚠ Trap:** concluding "the model is too big, we need a smaller one." Model scale does increase memorization, but duplication is the dominant term and the one you control. Shrinking the model to fix a dedup bug trades a large capability loss for a partial fix to the wrong variable.

**💰 Math on the eval that would have caught it:** the memorization check is 1,000 generations of 100 tokens on an 8B model — `1000 × 100 × 2 × 8e9 = 1.6e15` FLOPs, seconds of GPU time, call it $0.01. The incident cost: an emergency rollback, a customer escalation, a legal review, and likely a contractual remediation. Any nonzero estimate of that puts the ROI in the millions-to-one range. This is the archetype of the cheap gate nobody adds until after the incident.

### Our new corpus build came out with 30% fewer tokens than the last one and nobody knows why. How do you find it?

This is a pipeline observability problem and it is solvable in an hour if you built the pipeline correctly, or a week if you did not — which is itself the lesson.

**The method: bisect by stage using per-stage counters.** Every stage should emit `documents_in`, `documents_out`, `bytes_in`, `bytes_out`, and `dropped_by_reason{reason=...}`. Diff those counters between the two builds. The 30% appears at exactly one stage, and now you have localized it. If you do not have those counters, add them first — resist the urge to start reading data, because you will spend three days confirming what a counter diff would have told you in ten minutes.

**The likely culprits, ordered by base rate:**

1. **A filter threshold moved.** Someone tuned the quality classifier threshold from 2.5 to 3.0 in a PR that looked like a config tweak. Diff the config hashes between builds — this is why content-addressed, config-hashed stage outputs matter.
2. **A dependency changed behavior.** `trafilatura` upgraded and its default extraction became more conservative. Language-ID model version changed and its confidence distribution shifted, so the same 0.65 threshold now rejects more. **This is the one people never suspect and it is extremely common** — pin your extraction and classifier versions in the container image and record them in the corpus metadata.
3. **A source went missing.** One of your input prefixes did not get listed — an S3 listing timed out and the job continued with fewer shards, silently. Assert input shard counts against an expected manifest; a pipeline that silently proceeds on partial input is a defect.
4. **Dedup got more aggressive.** Either the LSH parameters changed (remember: raising `P` without adjusting `b`/`r` silently moves the threshold), or the new build includes overlapping crawls so there is genuinely more duplication to remove. Check the dedup stage's removal *rate*, not just its output count.
5. **Tokenizer changed.** A different tokenizer produces a different token count from identical text. If token count dropped but *byte* count did not, this is your answer immediately — which is why you count bytes as well as tokens at every stage.
6. **A silent encoding failure.** A shard of documents failed UTF-8 decoding and was dropped by an exception handler that logged at DEBUG.

**The structural fixes I would require afterward:**

- **A golden-sample regression test.** 10,000 fixed documents run through the full pipeline on every change, asserting output token count within a tolerance and a hash of the output. This is an ordinary CI test and data pipelines almost never have one.
- **Counters as first-class metrics** with a dashboard, alerting on stage-level yield deviating more than X% from the previous build.
- **Pinned versions for every extraction and classification dependency**, recorded in corpus metadata.

**🗣 Say this in the room:** "Diff the per-stage document and byte counters between the two builds — the 30% lives at exactly one stage. If I don't have those counters, that's the actual bug and I'd add them before touching data. My prior on the cause is: a config threshold moved, or a library version bumped and changed its default behavior. Counting bytes as well as tokens immediately separates 'we lost documents' from 'the tokenizer changed.'"

### Give me your failure taxonomy for a pretraining or continued-pretraining program. How does this break in production?

**🔍 Failure taxonomy**, organized as a decision procedure from the symptom you actually observe.

**Symptom: the run crashes or hangs.**
1. NCCL hang with no error → a rank died silently or a link degraded. Check per-rank heartbeats; the last rank to report is the failed one. Fix: node drain and restart. Prevention: watchdog with per-rank liveness, automatic node exclusion.
2. OOM after N hours of healthy training → activation memory grew with sequence length (long-context stage started), or memory fragmentation. Fix: recomputation policy, expandable segments allocator. Prevention: run the max-length batch shape at step 0, not at step 40,000.
3. Repeated crash at the same step → deterministic data bug. Decode the batch.

**Symptom: loss misbehaves.**
4. Spike that recovers → benign, log it.
5. Spike that does not recover → corrupted optimizer state. Restart before the spike, skip batches. Prevention: z-loss, QK-norm, grad-norm-based batch skipping, bf16 not fp16.
6. NaN → dead run, restart. Prevention as above.
7. Periodic sawtooth → domain-homogeneous shards. Fix the shard writer to interleave.
8. Step-change *down* in train loss without a matching move in held-out loss → duplicated data entered the stream. Your dedup is broken.
9. Step-change *up* at a restart boundary → dataloader or LR schedule did not resume correctly. This corrupts your effective epoch count.

**Symptom: throughput degrades.**
10. Per-rank step time bimodal → one bad GPU. Drain it.
11. All-reduce time up → network degradation or noisy neighbor.
12. `data_wait_ms` up → storage contention or cold cache.
13. MFU down but wall-clock unchanged → you changed sequence length; the metric moved, not the machine.

**Symptom: the model is finished and something is wrong with it.**
14. Benchmarks jumped, siblings flat → contamination. Run the n-gram scan and check the private eval.
15. Domain loss great, downstream task worse → you optimized perplexity on boilerplate. The eval was wrong, not the model.
16. Model completes prompts instead of answering → you CPT'd an instruct checkpoint on raw text.
17. General benchmarks regressed 5+ points after CPT → replay fraction too low, or LR too high.
18. Model emits training data verbatim → duplication in the corpus, or too many epochs.
19. Model is excellent on benchmarks and stylistically strange → decay-phase mixture over-concentrated; raise general replay.
20. Long-context benchmark passes, real long-context tasks fail → you validated on needle-in-a-haystack, which tests retrieval and not aggregation.

**Symptom: the program is finished and the business is unhappy.**
21. The model is better and nobody can prove it → you built the eval after the model. This is the most common terminal failure of applied training programs and it is entirely preventable.
22. The model is better and cannot be maintained → you forked a base model and the base model shipped a new version. Budget the treadmill or use adapters.
23. The model is better and cannot be deployed → nobody costed the serving side. A 70B model that beats the 8B by 3 points and costs 9× per token is not automatically a win.

**🗣 Say this in the room:** "I'd group it by observed symptom: crash, loss shape, throughput, model quality, and program outcome. The two that actually kill projects aren't technical — they're 'the eval was built after the model, so nothing is provable' and 'we forked a base model and now we own a maintenance treadmill.'"

### 🏋 Here's the drill: you have 512 H100s and 10 days. Give me a training plan with numbers, in ten minutes, no calculator.

**🏋 Drill.** Ten minutes, no references, no calculator — do the arithmetic in round numbers and state your assumptions out loud. **Pass criterion:** you produce a defensible model size and token count, a dollar figure, and a stated MFU assumption, and your FLOP total is within 2× of the reference answer below.

**Reference solution.**

*Compute.* 512 GPUs × 10 days. `10 × 86,400 ≈ 864,000` seconds. H100 dense BF16 peak ≈ `1e15` FLOP/s (round 989 TFLOP/s to 1 PFLOP/s — this is the approximation that makes the whole thing mental). At 40% MFU: `4e14` FLOP/s per GPU. Cluster: `512 × 4e14 = 2e17` FLOP/s. Total: `2e17 × 8.64e5 ≈ 1.7e23` FLOPs.

*Cross-check in GPU-hours:* `512 × 240 = 123,000` GPU-hours. At $2.50/hr → **$307,000**. **📅 Volatile:** verify the rate.

*What it buys.* `C = 6ND` → `ND = 1.7e23/6 ≈ 2.9e22`. So:
- 3B params → 9.5T tokens (3,150 tok/param — extremely over-trained, aggressive but a real design point)
- 8B params → 3.6T tokens (450 tok/param — a sensible over-trained small model)
- 30B params → ~1T tokens (33 tok/param — near Chinchilla)

*The plan I would present.* Not from scratch. Take an 8B base, and spend the budget as:
- 5% — **ablations**: ~20 runs at 1B params × 20B tokens each. Each is `6 × 1e9 × 2e10 = 1.2e20` FLOPs, and 20 of them is `2.4e21` = 1.4% of budget. Cheap, and it is where you learn what your data is worth.
- 60% — **CPT + mid-training trunk**: ~2T tokens on the 8B at `6 × 8e9 × 2e12 = 9.6e22` FLOPs = 56% of budget. Domain-heavy with 20% general replay.
- 15% — **decay-phase variants**: four annealed mixes at 150B tokens each = `4 × 7.2e21 = 2.9e22` = 17%.
- 10% — **long-context extension**: 100B tokens at 128k, which costs ~4.8× per token, so `4.8 × 6 × 8e9 × 1e11 = 2.3e22` = 13%.
- Remainder — post-training and slack. And there is *always* slack, because you will lose ~10% to failures.

*Sanity checks you should state:* 60% + 15% + 13% + 1.4% ≈ 90%, leaving 10% for goodput loss — that is the right shape. And 2T domain tokens against, say, 300B unique domain tokens is 6.7 epochs, which is over the 4-epoch line, so either I need more data or I lower the domain weight. **Catching that in the plan rather than in the run is the point of the drill.**

### 🏋 Second drill: implement MinHash near-duplicate detection from scratch. Thirty minutes.

**🏋 Drill.** Thirty minutes, no references. **Pass criterion:** your implementation, run on 1,000 documents where you have injected 50 known near-duplicate pairs at Jaccard ≈ 0.85, recovers at least 45 of them with fewer than 10 false positives — and you can state your `b`, `r`, and the resulting threshold from memory.

Reference implementation, which you should be able to write from memory:

```python
import numpy as np, hashlib
from collections import defaultdict

MERSENNE = (1 << 61) - 1

def shingles(text, n=5):
    w = text.lower().split()
    return {" ".join(w[i:i+n]) for i in range(max(0, len(w) - n + 1))}

def hash_shingle(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")

def minhash(text, a, b_coef, n=5):
    """a, b_coef: (P,) random coefficients for the universal hash family
       h_i(x) = (a_i*x + b_i) mod (2^61 - 1)"""
    hs = np.array([hash_shingle(s) for s in shingles(text, n)], dtype=object)
    if len(hs) == 0:
        return np.full(len(a), MERSENNE, dtype=np.uint64)
    perm = [(int(ai) * hs + int(bi)) % MERSENNE for ai, bi in zip(a, b_coef)]
    return np.array([int(p.min()) for p in perm], dtype=np.uint64)

def lsh_buckets(signatures, b, r):
    """signatures: (n_docs, b*r). Returns candidate pairs."""
    cands = set()
    for band in range(b):
        table = defaultdict(list)
        for doc_id, sig in enumerate(signatures):
            key = tuple(sig[band*r:(band+1)*r].tolist())
            table[key].append(doc_id)
        for group in table.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    cands.add((group[i], group[j]))
    return cands

def threshold(b, r):
    return (1.0 / b) ** (1.0 / r)

# usage: P = b*r = 13*18 = 234 -> threshold ~ 0.80
rng = np.random.default_rng(0)
b, r = 18, 13
a = rng.integers(1, MERSENNE, size=b*r)
bc = rng.integers(0, MERSENNE, size=b*r)
```

**The three things graders look for:** (1) you used a *universal hash family* with random coefficients rather than `P` calls to a fixed hash, because `P` independent permutations is the whole premise; (2) you banded correctly and can state `t ≈ (1/b)^(1/r)`; (3) you noted that the candidate set still needs verification — LSH gives you candidates, and you either accept the approximation or compute true Jaccard on the (much smaller) candidate list.

**⚠ Trap in the drill itself:** hashing the shingle *set* in Python without deduplicating shingles first. Jaccard is defined on sets; if you use a list you are computing something else and your threshold math no longer holds.

### Should a startup pretrain a foundation model from scratch in 2026? Defend your answer.

Almost always no, and I want to give the reasoning rather than the slogan, because the slogan is what a candidate who has not thought about it says too.

**The case against, with numbers.** To reach parity with a current open-weight 8B model you need roughly 15T tokens of curated, deduplicated, decontaminated data and ~7e23 FLOPs. The compute is ~$1.3M at current rates and falling — genuinely affordable for a funded startup. **The compute is not the barrier.** The barriers are:

- **Data.** Building a FineWeb-quality 15T-token corpus is a 6–12 month project for a specialized team, and the resulting corpus is not obviously better than the public ones. You would be re-deriving DCLM.
- **Iteration count.** The teams that produce good models did not produce them on the first run. They ran hundreds of ablations and several full-scale attempts. Budget 3–5× the headline number.
- **Post-training.** A base model is not a product. The distance from base checkpoint to something a user tolerates is SFT plus preference optimization plus safety work plus tool-use training, and that is where most of the differentiating effort now lives.
- **The treadmill.** Open weights improve every few months. A model you spend nine months building is competing against whatever ships the week you finish, and you now own the obligation to do it again.
- **Opportunity cost.** Those nine engineer-months spent on evals, retrieval, agent harness quality and product would almost certainly produce more customer value.

**The cases where I would change my mind** — and being able to name these is what distinguishes judgement from reflex:

1. **A modality or data type with no good open model.** Genomics, protein sequences, time-series telemetry, EDA netlists, a proprietary sensor format. There is no base model to adapt because nobody has built one, and your data is genuinely unique.
2. **A low-resource language where existing tokenizers and models are badly served.** Here the tokenizer alone is a real cost advantage, and CPT on a base model whose tokenizer wastes 3× on your script hits a ceiling that only pretraining removes.
3. **A hard architectural bet you cannot test by fine-tuning** — a fundamentally different attention mechanism, a new positional scheme, an alternative objective. Fine-tuning cannot evaluate a pretraining-time architectural claim.
4. **Extreme deployment constraints.** A model that must run in 2 GB on a device, at a quality nobody has published for that size, on a fixed domain. Distillation is the first answer, but a small purpose-built model trained on a narrow distribution can genuinely beat a distilled general model on that distribution.
5. **Regulatory or sovereignty requirements** that forbid any dependency on an externally-trained model's provenance — this is a real driver of national and sector-specific model programs, and it is a policy reason rather than a technical one, which is fine as long as everyone knows that.

**💰 The honest comparison for a typical applied startup:** $1.3M of compute plus ~$3M of engineer-time for a from-scratch 8B that lands at, optimistically, parity with a free open model — versus $50k of compute plus $500k of engineer-time for CPT, SFT, evals and retrieval on top of that free model, landing meaningfully *above* it on your specific tasks. The second option is 8× cheaper and better on the only axis your customers measure.

**🗣 Say this in the room:** "No, with five named exceptions. The compute is affordable now — it's about $1.3M for an 8B-class run — but the data pipeline, the ablation budget, the post-training and the upgrade treadmill are what actually cost you, and at the end you've matched a free model. I'd change my mind for a modality with no open base, a badly-tokenized low-resource language, an architectural bet you can't test by fine-tuning, an extreme on-device constraint, or a sovereignty requirement."

### You're interviewing for an applied AI role. You will never run a pretraining job. Why should you know any of this?

Because four of the decisions you *will* make every week are downstream of it, and because the failure mode this section prevents is the one that gets applied engineers rejected: proposing a fine-tune when the problem is retrieval, or accepting a benchmark number that is contaminated, or costing a plan without arithmetic.

**Concretely, here is what transfers.**

**1. The escalation ladder and its preconditions.** Knowing what pretraining actually installs in a model tells you what fine-tuning can and cannot change. Instruction-following is a thin post-training layer, so it is cheap to modify and easy to destroy. World knowledge is distributed through the whole network by trillions of tokens, so you are not going to inject a meaningful amount of it with 5,000 SFT examples. That single distinction resolves most "should we fine-tune" arguments correctly and immediately.

**2. Cost arithmetic as a reflex.** `C = 6ND` for training, `2N` FLOPs per token for inference, KV cache bytes per token, MFU as a utilization metric. These are the same skill as knowing your database's row size and your service's p99 — they let you sanity-check a proposal in your head before the meeting ends. An engineer who can say "that's 3,400 GPU-hours, about $8,500, four days on 32 cards" is trusted with budgets. One who says "we'd need to fine-tune it" is not.

**3. Evaluation skepticism.** Everything in this section about contamination, decontamination's limits, upstream-downstream decoupling, and building the eval before the model applies directly to shipping a product feature. When a vendor shows you a benchmark, you now know to ask about the private held-out set and the n-gram overlap. When your own eval jumps, you know to check the correlation structure first.

**4. Speaking the language.** In a company with a research or model team, you will be in rooms where people say "we're going to change the decay-phase mix" or "the base checkpoint has weak long-context." If those sentences parse, you can participate — you can tell them what the serving consequences are, you can flag that raising context to 128k costs 16 GB of KV cache per request, and you become the person who connects the model team to production. That role is disproportionately valuable and it is exactly the AI Engineer job at a company that trains its own models.

**🗣 Say this in the room:** "I won't run a pretraining job, but I make four decisions a week that depend on understanding one: whether a failure is a knowledge problem or a behavior problem, what a training proposal actually costs, whether a benchmark number is real, and what a model-team decision will do to my serving budget. Those are the same skills, and the arithmetic is the part I refuse to hand-wave."
