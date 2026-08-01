### Do the break-even. At what monthly volume does a fine-tuned 3B beat a prompted frontier model?

This is the arithmetic I want to be able to do on a whiteboard without notes, because some version of it is asked in nearly every applied-AI loop. **📅 Volatile:** all prices below are illustrative — verify current numbers before your interview; the *structure* is what's durable.

**The workload.** A per-request shape of: 5,000-token static preamble (system prompt + tool schemas + few-shots), 800 tokens of dynamic context, 150 output tokens.

**Option A — prompted frontier, prefix-cached.** At $3/Mtok uncached input, $0.30/Mtok cache-read, $15/Mtok output:

```
5,000 × $0.30/1e6  = $0.00150   (cached preamble)
  800 × $3.00/1e6  = $0.00240   (dynamic context)
  150 × $15.0/1e6  = $0.00225   (output)
                    ─────────
                     $0.00615 per request
```

Marginal cost only. Zero fixed cost, zero ownership.

**Option B — fine-tuned 3B, self-hosted.** The fine-tune deletes the preamble (task and format are in weights), so the request is 800 in / 150 out and the cost is entirely fixed infrastructure and amortized engineering:

```
one-time build (6 eng-weeks @ ~$5,800/wk + data + sweeps) ≈ $40,000
  amortized over 12 months                                =  $3,333/mo
serving: 2 × L40S-class @ $1.50/GPU-hr × 730 hr           =  $2,190/mo
maintenance: 2 re-tunes/yr @ 1.5 eng-weeks each ≈ $24k/yr =  $2,000/mo
                                                            ─────────
                                                 fixed    =  $7,523/mo
```

**Capacity check, because a break-even you can't serve is fiction.** A 3B in bf16 is 6 GB of weights. An L40S-class card at ~864 GB/s HBM does 864/6 = 144 weight-passes/second; batching amortizes that read across the batch, so at batch 64 the theoretical ceiling is 64 × 144 ≈ 9,200 output tok/s. Take 25–30% of theoretical for real attention, KV traffic and scheduling → **~2,500 tok/s** sustained. At 150 output tokens per request that is **16.6 req/s**, and at a realistic 30% average utilization across a diurnal curve, **~13M requests/month** fits inside that two-GPU footprint. Prefill sanity check: 800 tokens × 2 × 3e9 = 4.8 TFLOPs per request; at 16.6 req/s that is 80 TFLOP/s against ~150 TFLOP/s of realistic bf16 throughput — tight but fits, and it is why you size prefill separately rather than assuming decode is the only constraint.

**The crossover.**

```
$0.00615 × V = $7,523   →   V = 1,223,000 requests/month ≈ 41,000/day
```

**So: below ~1.2M requests/month, the prompted frontier model wins on pure cost.** Above it the fine-tune wins, and the gap widens fast because the fine-tune's cost is flat up to 13M/month. At 5M requests/month: frontier = 5e6 × $0.00615 = **$30,750/mo** vs fine-tune **$7,523/mo** → **$23,227/month saved, $279k/year**. At 500k requests/month: frontier = $3,075/mo vs $7,523/mo → the fine-tune costs **2.4× more** and you also own a checkpoint.

**Latency, which is often the real driver.** The 3B: 800-token prefill ≈ 32ms of compute, TTFT under 60ms locally, then 150 tokens at ~120 tok/s single-stream = 1.25s → **~1.3s end to end.** The frontier call: 5,800 tokens of prefill plus network, TTFT ~500ms, 150 output at ~60 tok/s = 2.5s → **~3.0s.** A 2.3× latency win that no prompt change can produce, because 5,000 of those prefill tokens are the prompt.

**📐 Numbers you must know**, with their derivation so you can rebuild them for any shape: (a) a frontier request of ~6k input / 150 output lands near **$0.006–$0.02** depending on cache hit rate — derived as `in_tok × in_price + out_tok × out_price`, and the cache read is ~10% of uncached input; (b) a self-hosted small-model serving lane costs **~$1,100/GPU-month** at $1.50/GPU-hour × 730 hours, so any two-GPU deployment starts at ~$2,200/month before a single request; (c) a fully-loaded senior engineer-week is **~$5,800** at $300k/year, which is how you convert "six weeks of work" into a number finance will argue with; (d) decode throughput for a dense model is bounded by `HBM_bandwidth / weight_bytes × batch` — 864 GB/s ÷ 6 GB × 64 ≈ 9,200 tok/s theoretical, take 25–30%.

**🗣 Say this in the room:** "At my request shape the prompted frontier path costs about six-tenths of a cent per call, and a self-hosted fine-tuned 3B costs roughly $7.5k a month all-in including amortized engineering and two re-tunes a year. That crosses over around 1.2 million requests a month. Below that I prompt; above that I need to look seriously, and above 5 million it's a quarter-million dollars a year and the conversation is over."

### Token prices have been falling roughly 80% a year. Redo that decision with price deflation in it.

This is the question that separates a spreadsheet from judgment, and it is the strongest argument *against* fine-tuning that exists — stronger than anything about hallucination.

**Redo the crossover at 80%-lower token prices** ($0.60/Mtok in, $0.06/Mtok cache-read, $3/Mtok out):

```
5,000 × $0.06/1e6 = $0.00030
  800 × $0.60/1e6 = $0.00048
  150 × $3.00/1e6 = $0.00045
                    ─────────
                     $0.00123 per request

break-even V = $7,523 / $0.00123 = 6,116,000 requests/month ≈ 204,000/day
```

**The break-even moved 5×.** A project that was comfortably above the line at 2M requests/month is now well below it, and it took twelve months to happen — which is roughly the useful life of the checkpoint you built.

Now the asymmetry that makes this decisive: **your fixed side does not deflate at the same rate.** GPU rental prices have fallen, but on the order of tens of percent, not 80%. Engineer salaries went up. Your maintenance line went up, because there were more base-model upgrades to chase. So the variable side of the comparison — the side you were trying to escape — is the side improving fastest.

There is a second-order effect in the same direction: the *quality floor* rises. The frontier-minus-one model in twelve months is better and cheaper than the model you distilled from, so your fine-tuned 3B is now competing against a hosted small model that is both smarter than your student and priced below your amortized cost. This is the treadmill in its purest form.

**The decision rule I actually use:** commit to a fine-tune for cost reasons only if your **projected steady-state volume is at least 3–5× the break-even computed at next year's expected prices**, not today's. Concretely: take today's break-even, multiply by 5 for a year of deflation, then require your volume forecast to clear *that*. In the example above, that means you want ≥6M requests/month before a pure cost argument justifies the build. Below that, the honest answer is "we'd be building an asset that depreciates faster than we can amortize it."

**⚠ Trap:** the counter-force nobody prices in. Deflation of ~80%/year in unit price has been accompanied by enterprise LLM spend *increasing*, because cheaper tokens unlock workloads that were previously uneconomic — agentic loops that make 40 model calls per user action, background evaluation, per-document processing at corpus scale. This is a Jevons dynamic: your bill can grow while your per-token price collapses. So "prices are falling, don't optimize" is also wrong. The correct reading is: **optimize the levers that are configuration (caching, routing, context size) aggressively and continuously; be conservative about levers that are artifacts (fine-tunes) because their payback window is shrinking.** That sentence is the whole section in one line.

**💰 Math, the Jevons version:** if unit price falls 80% (×0.2) but your per-task token consumption grows 6× because you moved from single-shot to a 6-call agent loop, your bill is 0.2 × 6 = **1.2× — it went up 20%**. I have watched exactly this happen to three teams. Track cost *per resolved task*, never cost per token.

### Explain the model-upgrade treadmill to me and tell me how you hedge against it.

The treadmill is this: you fine-tune model M at time t. At t+4 months the provider or the open-weight community ships M′, which is better, cheaper, and faster than M *base*. Your fine-tune of M is now competing not against the old baseline but against M′ prompted — and frequently loses, because a generational jump in the base model is worth more than your task-specific tuning was.

To move to M′ you must re-run the whole pipeline: regenerate or re-verify training data against M′'s chat template and tokenizer, re-sweep hyperparameters (the LR that worked for M is often wrong for M′), re-run the full eval, re-run the capability-regression suite, re-canary, and keep both checkpoints live during the migration. That is 1–2 engineer-weeks each time, 2–4 times a year, forever. **Your fine-tune is not a project, it is a subscription.**

Worse, sometimes you *can't* upgrade. If you fine-tuned a hosted model and the provider deprecates that base, your checkpoint is stranded — you cannot port weights, and depending on the provider you may get a fixed sunset window to re-train or lose the endpoint. If you fine-tuned an open-weight model, you at least own the artifact, but you own it on an architecture that is falling behind.

**How I hedge, concretely:**

**Keep the pipeline, not the checkpoint, as the asset.** Everything — data generation, filtering, training config, eval, canary — lives in a reproducible pipeline that takes `base_model` as a parameter. If re-tuning on a new base is a one-command job that finishes overnight, the treadmill costs hours instead of weeks. This is the single highest-leverage engineering decision in the whole area and it is exactly the CI/CD discipline you already have; teams fail here for the same reason they fail at reproducible builds.

**Never delete the prompted baseline.** Keep it running on a small traffic slice, or at minimum keep it green in CI. It is your fallback *and* your continuous measurement of whether the fine-tune still earns its keep. The day the baseline catches the fine-tune, you have a decision to make and you want to make it with data.

**Prefer LoRA over full fine-tuning when quality allows.** Adapters are small, fast to re-train, and cheap to keep multiple versions of. Re-training a LoRA on a new base is hours; a full fine-tune is days.

**Version the data, not just the model.** Your training set outlives every base model. Treat it as the durable artifact: versioned, deduped, contamination-checked against your eval, with provenance per example. When M″ ships, the data is the thing you still have.

**Write the sunset criterion into the design doc on day one.** "We retire this fine-tune when the prompted baseline reaches within 2 points on the eval at under 1.3× the cost." Deciding that up front is how you avoid the sunk-cost argument in eleven months.

**💰 Math:** treadmill cost alone, at 3 upgrades/year × 1.5 engineer-weeks × $5,800/week = **$26,100/year** of pure maintenance, before a single quality improvement. That figure belongs in the break-even from the first slide, and leaving it out is how projects that "obviously pay for themselves" quietly don't.

### It's Monday. The new frontier model just beat your six-month-old fine-tune zero-shot on your own eval. What do you do?

First, I don't panic and I don't get attached. This outcome was in the design doc as the sunset criterion, so it is a planned branch, not a crisis.

**Step 1 — verify the comparison is fair, in both directions.** Is the new model being evaluated with the same context assembly, the same tools, the same output constraints? And is my fine-tune being evaluated on the *current* traffic distribution, or on an eval set frozen six months ago that no longer looks like production? Both errors are common and they point opposite ways. I want the head-to-head on a freshly-sampled 300-item set from last month's traffic, graded by the same rubric, run three times for a noise floor.

**Step 2 — compare on the full vector, not just accuracy.** Quality, p50/p95 latency, cost per request at current volume, data-residency and privacy constraints, and rate-limit exposure. A fine-tune that loses by 2 accuracy points but runs at 1.3s instead of 3.0s and never hits a provider rate limit may still be the right production choice for an interactive feature. Conversely, if the frontier model wins on quality *and* the deflated price puts it under my amortized fixed cost, the decision is trivial.

**Step 3 — if the frontier model wins outright, migrate, and do it in the cheapest order.** Route a 5% canary to the prompted path with the same eval instrumentation. Watch the metrics that offline evals miss: abstention rate, escalation rate to human, tail latency, and complaint volume. Ramp over two weeks. Keep the fine-tune warm for one release cycle, then decommission it and *keep the pipeline and the data*.

**Step 4 — the interesting branch: re-tune the new base.** Often the right answer is not "abandon the fine-tune" but "run the same pipeline on the better base." If your pipeline is parameterized (which is why you built it that way), this is an overnight job. The new fine-tune usually beats both the old fine-tune and the new prompted baseline, because base-model improvements and task-specific adaptation compose. Then you re-run the break-even with the new prices, because that is what decides whether to keep the serving lane at all.

**🗣 Say this in the room:** "This is the branch I wrote into the design doc as the sunset criterion, so it's a decision, not a fire. I re-run the head-to-head on freshly-sampled current traffic, compare on quality *and* p95 *and* cost, and then either re-run my pipeline on the new base — which is an overnight job because the pipeline is parameterized on base model — or canary the prompted path at 5% and decommission. The thing I never do is defend the checkpoint because we paid for it."

**⚠ Trap:** the sunk-cost defense, dressed up as "but we own our model, we're not dependent on a vendor." Independence has value, but it is a *strategic* argument with a price, and you should be able to state that price: in this example, roughly $7.5k/month plus a quarter-million-dollar build. If the strategic argument is real (data residency, an air-gapped deployment, a contractual bar on third-party processing), say that and stop pretending it's about cost. Mixing the two arguments is how teams lose credibility with finance.

### Does LoRA change the calculus? Cheap adapters, multi-tenant serving — does the ladder still hold?

The ladder holds, but LoRA moves two of the numbers in it substantially, and one of them is a step change.

**What LoRA changes on the training side:** the marginal cost of a fine-tune drops by roughly an order of magnitude in wall-clock and memory. You can tune an 8B on a single 80GB card, iterate in hours, keep ten adapter versions around, and re-tune on a new base overnight. That collapses the treadmill cost from ~1.5 engineer-weeks to ~2 engineer-days per upgrade. It also reduces catastrophic forgetting: the low-rank constraint means the model moves less, so it forgets less — and correspondingly learns less on large distribution shifts, which is the documented trade (Biderman et al., 2024, *LoRA Learns Less and Forgets Less*).

**What LoRA changes on the serving side — this is the step change.** Multi-adapter serving (S-LoRA-style, and the adapter multiplexing built into modern engines) lets one base model in memory serve many adapters, selecting per request, with the adapter weights swapped in as small per-layer additions. So the fixed serving cost stops being per-fine-tune and becomes per-*base*.

**💰 Math:** in the earlier break-even, serving was $2,190/month for a dedicated pair of GPUs. If you are a platform serving 50 tenant-specific adapters, the naive architecture is 50 × $2,190 = **$109,500/month**. With multi-LoRA on one base, you carry roughly the same two-GPU footprint plus adapter memory — a rank-16 LoRA on an 8B targeting all linear layers is on the order of tens of megabytes, so 50 adapters is a couple of GB — call it $2,500/month total, or **$50/month per tenant**. That is a **~44× reduction in per-tenant fixed cost**, which drops each tenant's break-even volume by the same factor. A per-tenant fine-tune that made no sense at 1.2M requests/month makes sense at ~30k.

So: for a *platform* with many tenants each wanting bespoke behaviour, LoRA plus multi-adapter serving genuinely changes when you climb the ladder. That is a real architectural pattern and worth naming in an interview.

**What LoRA does not change, and this is what I'd push back on:**

- **It does not lower the data cost or the eval cost.** Those dominated the itemization and they are per-adapter, not per-GPU. Fifty tenant adapters means fifty eval sets, or one eval set that doesn't actually test any of them.
- **It does not change what fine-tuning is good at.** A LoRA still will not inject facts. Cheaper wrong answers are still wrong answers.
- **It adds per-request latency overhead** when adapters are heterogeneous within a batch, because you lose some of the clean batched-GEMM path. Modest, but measure it — a few percent of decode throughput is normal, and it degrades as the number of distinct adapters in a batch grows.
- **It adds an operational surface**: which adapter, which version, what happens when a tenant's adapter is missing or corrupt, and how you canary fifty artifacts. Merge-for-serving removes the multiplexing cost but reinstates the per-model footprint, so you choose per tenant based on volume.

**⚠ Trap:** treating "LoRA is cheap" as permission to skip the preconditions. The gate list does not get shorter because training got cheaper — the eval, the plateau measurement and the held-out split are all still required. Cheap training makes it *easier* to produce fifty confidently-wrong adapters that nobody can evaluate.

### Design case — a Sierra-style enterprise support agent, 2M conversations a month. Walk the whole ladder with numbers.

Let me set the scope: a deployed support agent for a mid-size SaaS customer, 2M conversations/month, average 6 model calls per conversation (retrieval-augmented answer, two tool calls, a policy check, a summary), so **12M model calls/month**. Success metric is **resolution rate without human escalation**, not accuracy. Starting point: 62% autonomous resolution, and the customer wants 80%.

**Rung 0 — instrument before you touch anything.** A stable eval: 400 real conversations, labeled resolved/escalated/wrong by two annotators with adjudication, plus a live A/B harness. Two engineer-weeks. Nothing below happens without this. I also want the failure taxonomy: on the 152 failures in that sample, I expect roughly 40% "the answer wasn't in retrieved context," 20% "needed a tool it didn't have or used wrong," 15% "correct but escalated anyway due to policy ambiguity," 15% "tone/handling," 10% ambiguous label.

**Rung 1 — prompt and context.** Mine 15 few-shot examples from the *resolved* conversations that were hardest. Add explicit abstention and escalation criteria. Restructure so the static preamble (instructions, tool schemas, few-shots ≈ 5,500 tokens) is first and cacheable and the conversation is last. Expected: 62% → 68%, one engineer-week. Cost effect: preamble goes from uncached to ~85% cache-hit, saving 5,500 × $3/1e6 × 0.85 = **$0.014/call × 12M = $168,000/month.** That one change pays for the entire program.

**Rung 2 — retrieval.** Hybrid BM25 + dense over the help center and past resolved tickets, k=40, cross-encoder rerank to 5, plus query rewriting to resolve "it" and "that error." Also index resolved tickets, not just docs — past resolutions are the highest-value corpus in a support product and most teams forget them. Expected: 68% → 76%. Three engineer-weeks. Adds ~80ms of rerank latency and ~2,500 tokens of context per answering call.

**Rung 3 — tools.** `get_account(user)`, `get_subscription_status`, `check_entitlement(feature, plan)`, `create_refund(amount)` behind a human approval gate, `get_policy(tier, issue_type)`. This kills the entire "used a stale policy" and "guessed the plan" failure class. Expected: 76% → 81%. Three engineer-weeks. **This is the rung that hits the target**, and it is worth saying out loud that the customer's goal was reached without touching weights.

**Rung 4 — structured output** on the escalation decision and the action calls, with an explicit `unknown` branch. Doesn't move resolution; removes a class of 3am pages. One engineer-week.

**Rung 5 — routing.** Now optimize cost. Classification, summarization and policy lookup (about 4 of the 6 calls) go to a small model; the answering call and the escalation decision stay frontier. Blended cost per conversation: 4 calls at ~$0.0008 + 2 calls at ~$0.0062 = $0.0032 + $0.0124 = **$0.0156/conversation → $31,200/month** at 2M, versus all-frontier 6 × $0.0062 = $0.0372 → $74,400/month. **Saves $43,200/month for ~two engineer-weeks.**

**Rung 6 — distillation, only now.** The answering call is the expensive one. Take 40k production (context, answer) pairs where the outcome was "resolved, no escalation, positive CSAT," filter with a verifier for citation groundedness, and distill into an 8B. If it holds resolution within 1.5 points, cost per answering call goes from $0.0062 to roughly $0.0004 amortized at this volume, saving another ~$27,000/month. **But** — check the deflation rule first: at 4M answering calls/month against a $7.5k–$12k/month all-in fixed cost, we're at 5×+ break-even even after a year of price cuts, so this one clears the bar.

**Rung 7 — fine-tuning for capability?** I would not, in this design, and I'd say so explicitly. The residual failures after rung 3 are dominated by ambiguous labels and genuinely novel issues, neither of which gradients fix. What I *might* fine-tune for is the escalation decision boundary specifically, because there we have 2M/month of labeled outcomes and it is a learned boundary, not a fact — that is precondition-4-compliant and it is a small, well-scoped model with a clean eval.

**🔍 Failure taxonomy for this system in production:** resolution drops → check retrieval recall first (did the help center get restructured?), then prefix-cache hit rate (did someone add a timestamp?), then tool error rates (is the entitlements API returning 500s that the model is narrating around?), then escalation-classifier drift. In that order, because that is the order of both frequency and cheapness-to-check.

### Now argue the other side. Give me a case where fine-tuning early is correct and I should stop lecturing you about the ladder.

Gladly — a candidate who applies the ladder dogmatically is a different flavour of the same problem. The ladder is ordered by cost-of-being-wrong, but **if a lower rung is structurally incapable of meeting a hard constraint, you skip it and you say why.**

The canonical case is a Cursor-shaped product: inline code completion or next-edit prediction, fired on every pause in typing.

**The constraint that breaks the ladder: latency.** The interaction budget is on the order of tens of milliseconds to feel like part of the editor — call it a p95 of 150ms end to end. A hosted frontier call cannot meet that: a round trip to a provider is 30–80ms of network before any compute, TTFT on a large model with several thousand tokens of file context is 300–800ms, and you need multiple tokens out. **No amount of prompting, retrieval, routing, or caching reduces a 500ms TTFT to 100ms.** The lower rungs are not merely insufficient, they are structurally unable to reach the target. Skipping them is not impatience, it is reading the constraint.

**The second constraint: the output is form.** A next-edit prediction is a *diff in a specific dialect*, conditioned on cursor position, recent edits, open buffers and the language's idiom. That is exactly the "tacit format that the prompt cannot express" bucket. And you have essentially unlimited supervision: accepted completions are labeled positives, rejected ones are labeled negatives, generated at a rate of millions per day. Precondition 3 (≥1,000 labeled examples) is satisfied a thousand times over on day one.

**The third: volume makes the economics trivial.** At tens of millions of completions per day, the break-even computation isn't close — the fixed cost of a serving fleet is amortized across a volume where a per-call frontier price of even $0.001 would be $10,000/day. 10M completions/day × $0.001 = **$10,000/day = $3.65M/year**; the same traffic on self-hosted small models is a fleet cost measured in tens of thousands per month.

So the honest answer is: for latency-bound, form-dominated, high-volume, self-labeling tasks, you go to a small trained model immediately, and the ladder's role is not to stop you but to tell you *which* rung you're skipping and why. Speech, real-time translation, ranking, moderation at scale, and structured extraction at corpus scale have the same shape.

**🗣 Say this in the room:** "I skip rungs when a lower rung is structurally incapable, not when it's inconvenient. A hosted call has a floor of roughly 300–500ms TTFT; if my SLO is 150ms, no prompt fixes that, so I go straight to a small self-hosted model. The ladder is about not paying an artifact's cost for a config-level problem — it isn't a rule against training."

**⚠ Trap:** using this reasoning to justify skipping the *eval*. Even in the skip-the-ladder case, precondition 1 stands: you still need a stable offline eval plus an online acceptance-rate metric before you train, because "acceptance rate went up" and "quality went up" are different claims and the first is gameable by making completions shorter.

### Debug this: they fine-tuned, the offline eval went up eight points, and production complaint volume went up. What happened?

This is the most common post-fine-tune incident and there are six candidate causes. I would work them in this order, because that is the order of frequency and of cheapness to check.

**1. Train/serve context skew.** Did the training examples' context format match what the serving path actually assembles — same tags, same chunk count, same ordering, same fields populated? A model trained on 5 reranked chunks and served 20 unreranked ones is out of distribution on every request. *Check:* dump 20 raw serving prompts and 20 raw training prompts and diff them character by character. This finds the bug maybe 30% of the time and it takes twenty minutes.

**2. Abstention collapse.** Compare the correct / incorrect / **abstained** three-way split before and after. If abstention fell from 18% to 2% while accuracy rose 8 points, the model stopped saying "I don't know" and started guessing. Aggregate accuracy went up; user-visible *wrongness* went up more, and wrong-with-confidence generates complaints where "I'm not sure" does not. This is the failure the whole section warns about and it is invisible to any single-number eval.

**3. Eval/train provenance leakage.** Were the eval items generated by the same pipeline, from the same documents, with the same question templates as the training set? Then the 8 points measure template fit. *Check:* build a 100-item slice from real user queries in the last two weeks and re-score. If the delta collapses, that's your answer.

**4. Distribution shift between eval and traffic.** The eval was frozen when it was built. Compare the topic and length distribution of eval items against last month's traffic. Support corpora drift fast — a new product launch can make 20% of live traffic unlike anything in the eval.

**5. Catastrophic forgetting on capabilities not in the eval.** The eval measured the target task. Did tool-calling accuracy, multi-turn coherence, refusal calibration, or non-English handling regress? This is exactly what the capability-regression suite exists for, and its absence is why you can't answer this question quickly. *Check:* run a general instruction-following and tool-calling suite on both checkpoints.

**6. Tail versus mean.** Offline evals report averages; complaints are generated by the tail. An 8-point mean improvement with a fatter tail of catastrophic outputs is a net loss in user perception. *Check:* score the distribution, not the mean — what fraction of outputs are rated ≤2/5 before and after? A rise there with a rising mean is the signature.

**🔍 The decision procedure, compressed:** diff the prompts (skew) → check the abstention split → re-score on fresh human queries (leakage) → run the regression suite (forgetting) → look at the bottom decile, not the mean (tail). Four of those five take under an hour each. If all five come back clean, then and only then do I suspect the fine-tune itself and start looking at LR, epochs and overfitting on a small dataset.

**⚠ Trap:** rolling back and calling it done. If you don't identify which of the six it was, you will reproduce it on the next attempt, and you will have burned the team's appetite for the whole approach. The incident review has to name the cause.

### Ninety seconds. A staff engineer asks "should we fine-tune?" Give me the whole decision narrative, with numbers, as you'd actually say it.

**🗣 Say this in the room** — this is the rehearsed version, and I would practice it until it comes out without thinking:

"Maybe, but not yet, and here's the sequence I'd want to see first.

First: what's the failure? I'd sample a hundred production failures and classify them — was the needed information in the context, was it a tool the model should have called, was it a parse failure, or was the answer factually right and still wrong for us? In every system I've done this on, forty to sixty percent land in retrieval and under fifteen percent are genuinely behavioural. Fine-tuning only addresses that last bucket.

Second: fine-tuning does not reliably install facts. Facts get into weights through thousands of pretraining exposures in many phrasings. One exposure in an SFT set teaches the question-to-answer *template* instead, and the published result is that examples carrying unfamiliar knowledge are learned slowly and raise hallucination on facts the model previously had right. So if the complaint is 'it gets our product details wrong,' training makes it worse, more confidently.

Third: my gate is a stable eval that's been stable two weeks, a documented plateau on prompting, retrieval, tools and structured output — with an oracle-context ablation showing the remaining headroom as a number — at least a thousand clean labeled examples with a genuinely held-out split, a capability-regression suite, an owner, and a serving fallback. Seven checks. Any 'no' sends it back down the ladder.

Fourth: the money. All-in, a first fine-tune is fifty to a hundred and ten thousand dollars of loaded cost in year one, of which under half a percent is GPU. Against a prompted frontier path at about six-tenths of a cent per request, that breaks even around one-point-two million requests a month. And token prices have been falling roughly eighty percent a year while my engineering costs haven't, so I'd want three to five times that volume before I'd commit — call it five million requests a month — because the asset depreciates faster than we amortize it.

Where I *would* fine-tune, today, without argument: tacit house style, a domain output format, a learned decision boundary with tens of thousands of labeled outcomes behind it, or a hard latency SLO a hosted call structurally cannot meet. Facts go in context; behaviour goes in weights."

That is roughly 320 words — about ninety seconds at interview pace. The structure to memorize is five beats: **classify the failure → why training doesn't fix facts → the seven-item gate → the break-even with deflation → what I would train for.** The numbers are the part that makes it land; delivering the same answer without them reads as received wisdom.

### Drills — how would you rehearse this so it's automatic, and what's the anti-pattern on the other side of it?

**🏋 Drill 1 — the napkin break-even, 5 minutes, unaided.** Given a request shape (input tokens, output tokens, static preamble size), a price table, and a target model size, compute: cached and uncached per-request cost, the fine-tune's monthly fixed cost with amortized engineering, the crossover volume, and the crossover volume after a year of 80% price deflation. Pass criterion: four numbers with the arithmetic written out, in five minutes, no calculator beyond mental arithmetic on powers of ten. Do this once a day for a week with different shapes; it is the single most-asked quantitative question in applied-AI loops.

**🏋 Drill 2 — the ladder, spoken, 90 seconds.** Record yourself answering "should we fine-tune?" against a stopwatch. Pass criterion: all seven rungs named in order, at least three specific numbers, and the phrase "facts go in context, behaviour goes in weights." Listen back for hedging — "it depends" without a decision rule is the failure.

**🏋 Drill 3 — the taxonomy under time pressure, 30 minutes.** Take twenty invented-but-realistic failure descriptions ("the summary cited a policy that was retired last quarter"; "the JSON had a `category` of `billing` for a login issue"; "it computed a refund of $1,240 when the correct figure was $1,204"). Assign each a rung and name the specific fix. Pass criterion: 18/20 correct rung assignment and no answer that is just "fine-tune."

**🏋 Drill 4 — the ablation table from memory.** Write the eight-row ablation ladder table, with columns, for a RAG system you have never seen. Pass criterion: the oracle row is present, a noise-floor row is present, and cost and p95 are columns, not afterthoughts.

**The anti-pattern on the other side — and it is real.** A candidate who has over-learned this section becomes the person who *never* trains, treats every proposal as naive, and answers every training question with "have you tried a better prompt?" That reads as dogma, and at a company that actually post-trains — Databricks, Meta, Snowflake, any lab, any product team with a bespoke model — it is disqualifying in the opposite direction. The tell that you have the balance right is that you can, unprompted, name four concrete cases where you *would* train (tacit style, domain output format, a learned decision boundary with tens of thousands of labels, a latency SLO a hosted call cannot meet) and describe the pipeline you'd build, including chat-template correctness, loss masking on completions only, the held-out split, and the regression suite. Say the "no" and the "here's when yes" in the same breath.

**🗣 Say this in the room, as the closer:** "I'm not against training — I've costed it and I know exactly what it buys. I'm against paying an artifact's price for a configuration-level problem, and against installing facts through a mechanism that converts ignorance into confident error. Show me the plateau and the labeled data and I'll build the pipeline this quarter."
