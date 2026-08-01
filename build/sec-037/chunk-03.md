### Explain Self-RAG. What are reflection tokens actually doing?

**Mental model: Self-RAG moves the retrieval decision out of your router and into the model's own output distribution.** Instead of a separate classifier deciding "should I retrieve," the model emits a special token that *is* that decision, sampled from the same softmax as everything else — so retrieval becomes a first-class action in generation rather than a wrapper around it.

**📄 Paper:** Asai, Wu, Wang, Sil & Hajishirzi (2023) — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. It replaced the "always retrieve top-k and hope" pattern with on-demand retrieval plus self-assessment, trained into the weights rather than prompted.

The mechanism is four token families added to the vocabulary and trained via data distilled from a critic model:

- **`Retrieve`** — `yes` / `no` / `continue`. Emitted as generation proceeds; `no` means the model believes it can answer from parametric memory, `continue` means keep using what is already retrieved.
- **`ISREL`** — relevant / irrelevant, emitted per retrieved passage. A learned relevance judgment on the passage the model just conditioned on.
- **`ISSUP`** — fully supported / partially supported / no support. **This is a groundedness judgment on the model's own generated segment against the passage** — a self-check on whether it just hallucinated.
- **`ISUSE`** — a 1–5 usefulness rating on the response.

At inference you exploit these for **tree-decoding with critique-weighted beam search**: generate candidate continuations conditioned on different retrieved passages, score each by a weighted combination of the reflection-token probabilities, and keep the best. The weights are tunable at inference time, which is the elegant part — you can dial up the `ISSUP` weight to trade fluency for groundedness **without retraining**, which is a knob no prompt-based system gives you.

Why it matters for your job even though you will not train one: it is the clearest existence proof that **"should I retrieve" is a model-level decision, not an orchestration-level one**, and it names the three checks any serious RAG system needs regardless of implementation — is this passage relevant, is my claim supported by it, is my answer useful. In a frozen-API world you approximate all three with separate cheap calls (a reranker for `ISREL`, an NLI or LLM-judge check for `ISSUP`), pay three round trips for what Self-RAG gets in one forward pass, and that gap is precisely why retrieval-aware post-training keeps being interesting.

**⚠ Trap:** candidates describe Self-RAG as "the model decides whether to retrieve," stop there, and miss that the load-bearing contribution is the **per-segment groundedness token**. The retrieve-or-not decision is the headline; `ISSUP` is the part that actually reduces hallucination, because it is a supervised signal on attribution rather than on relevance.

### What does CRAG add on top of that, and would you ship its web-search fallback?

**Mental model: CRAG accepts that retrieval will sometimes return garbage and asks the obvious follow-up nobody implements — what should the system do when it knows the retrieval was bad?** Most pipelines have no answer; they stuff the garbage in and let the generator improvise.

**📄 Paper:** Yan et al. (2024) — *Corrective Retrieval Augmented Generation*. The contribution is a lightweight **retrieval evaluator** that scores query-document relevance and drives a three-way branch:

- **Correct** (confidence high): run **knowledge refinement** — decompose the retrieved documents into fine-grained strips, score each, drop the irrelevant ones, recompose. This is the underrated half: even a correct retrieval carries a lot of unrelated text, and stripping it improves generation.
- **Incorrect** (confidence low): discard the internal retrieval entirely and **fall back to web search** with a rewritten query.
- **Ambiguous**: do both and combine.

The evaluator is small — a fine-tuned T5-scale model in the paper — which matters because a heavyweight evaluator would eat the entire latency budget for a check that fires on every query.

Would I ship the web fallback? **In a consumer or research product, yes. In an enterprise assistant, almost never, and I would say why forcefully.** The web fallback silently changes the trust boundary of your answer: your corpus is curated, permissioned, and versioned, and the open web is none of those. If a legal assistant at Harvey answers a question about a client's contract from a random web result, that is not a degraded answer, it is an incident. The corporate versions of the branch that I do ship:

- fall back to a **broader internal index** (all-company instead of team-scoped) with an explicit provenance label,
- fall back to **abstention with a scoped explanation** — "I don't have documentation on X; here are the two nearest things and here's how to request it,"
- fall back to a **human handoff** with the failed query attached.

**The generalizable idea, which is what I actually take from CRAG: a calibrated retrieval evaluator between retrieval and generation, whose output drives a branch.** You already have most of it — your cross-encoder reranker produces a score. Calibrate it on a few hundred labeled pairs including known-irrelevant ones, fit a logistic regression from logit to P(relevant), pick a threshold at your target precision, and now `P < 0.3` is a *decision*, not a number in a log.

**💰 Math:** if 8% of queries fail retrieval and you abstain instead of generating, you skip `0.08 × 1e6 = 80,000` calls/month at ~$0.030 each = **$2,400/month saved while improving quality**. Cases where cost and quality point the same direction are rare enough that you should take every one.

### Explain FLARE. What problem does mid-generation retrieval solve that pre-generation retrieval can't?

**Mental model: retrieving once, before generation, assumes you know what evidence the answer needs before you have written it. For a long answer, you don't.** The information need emerges as the answer unfolds — a three-paragraph summary of a policy might need the retention rules in paragraph one and the deletion SLA in paragraph three, and the initial query gave no signal about the second.

**📄 Paper:** Jiang, Xu, Gao, Sun, Liu, Dwivedi-Yu, Yang, Callan & Neubig (2023) — *Active Retrieval Augmented Generation* (FLARE). The mechanism is genuinely clever and worth being able to describe precisely:

1. Generate a **temporary next sentence** without retrieving.
2. Inspect the token-level probabilities of that sentence. If every token's probability is above a threshold θ, the model is confident — accept the sentence and move on, no retrieval.
3. If any token falls below θ, the model is uncertain, and uncertainty is a proxy for "I am about to make something up." Take that temporary sentence, **mask out the low-confidence tokens**, use the result as a retrieval query, retrieve, and regenerate the sentence conditioned on the new evidence.

The masking step is the insight. The low-confidence tokens are exactly the ones the model is unsure about, so including them in the query would bias retrieval toward the model's own guess. Masking them leaves the scaffolding — "The retention period for EU customer data is ___" — which is a far better search query than either the original question or the hallucinated completion.

Why you probably cannot ship it as written: **it requires token-level logprobs and the ability to interrupt and resume generation mid-stream.** Several hosted APIs expose logprobs; not all do, and not all support cleanly regenerating from a partial completion with modified context. `📅 Volatile — verify the logprob and prefill/continuation support of your specific provider before designing around this.` It also multiplies latency for long answers: a 10-sentence answer with a 30% trigger rate is 3 extra retrieve-and-regenerate cycles.

What survives and is shippable: **confidence-triggered retrieval as a general pattern.** The tractable version is at the answer level rather than the sentence level — generate, check groundedness of the claims against retrieved context with a cheap NLI or judge call, and if a claim is unsupported, re-retrieve on that specific claim and regenerate. Slower granularity, same idea, works with any API.

**⚠ Trap:** token probability is a proxy for *fluency* uncertainty, not *factual* uncertainty, and a well-tuned model hallucinates fluently and confidently. FLARE's trigger catches the model that is groping for a word; it does not catch the model that is confidently wrong. Treat low confidence as a strong signal to retrieve and high confidence as **no signal at all** — asymmetric, which is how all confidence heuristics in this space should be treated.

### Adaptive-RAG routes by question complexity. Is that different from what we've already discussed as intent routing, and is it worth it?

It is the same family, and the paper's specific contribution is worth stating because it makes the economics explicit rather than the taxonomy.

**📄 Paper:** Jeong, Baek, Cho, Hwang & Park (2024) — *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*. A small classifier routes each query into one of three strategies: **no retrieval** (the model knows it), **single-step retrieval** (one pass suffices), or **multi-step iterative retrieval** (needs a loop). The training labels come cheaply — run all three strategies on a training set and label each question with the *simplest* strategy that got it right, plus dataset-level priors for the rest. That labeling trick is the part I actually reuse.

The economic argument, which is the whole point: iterative multi-step retrieval is 3–5× the cost and latency of single-step. If you route everything through it, you pay that multiple on the 80% of queries that never needed it. If you route nothing through it, you fail the 10% that did. **Routing converts a fixed cost multiplier into an expected one.**

**💰 Math:** say single-step costs $0.030 and 900 ms, and iterative costs $0.11 and 3.4 s. Everything-iterative at 1M queries/month is `1e6 × 0.11 = $110,000` and a 3.4 s p50. Everything-single-step is `$30,000` and 900 ms, but fails the 10% multi-hop slice outright. Routing with a classifier at, say, 85% accuracy sending 12% to the iterative path: `0.88 × 0.030 + 0.12 × 0.11 = 0.0264 + 0.0132 = $0.0396`/query = **$39,600/month**, with a p50 still around 950 ms and a p95 that carries the iterative tail. **You bought 90% of the quality of always-iterative for 36% of the cost.** That arithmetic is the answer to "is routing worth it," and it is worth memorizing the shape.

Where it differs from intent routing as I described it earlier: intent routing is about *what kind of thing* the user wants (lookup, action, chit-chat), Adaptive-RAG is about *how much work* the answer needs. In production I collapse them into one classifier with one output schema, because they are both "understand the query" and neither justifies its own round trip. But they are conceptually distinct axes and an interviewer may be probing whether you see that.

**⚠ Trap:** router misclassification is asymmetric and most people build it symmetric. Routing a multi-hop question to the single-step path produces a **confidently wrong answer** — the worst outcome. Routing a simple question to the iterative path produces a correct answer that cost 3× too much — an annoying outcome. **Bias the router's threshold toward the expensive path**, and tune it on the cost of the error, not on classification accuracy. A router at 85% accuracy with the errors in the right direction beats one at 90% with them in the wrong direction.

### What is Speculative RAG, and what does it borrow from speculative decoding?

**Mental model: it borrows the shape, not the mechanism — a small fast model proposes, a large model verifies, and the win comes from the verification being cheaper than generation.** Speculative decoding does this at the token level with exact distribution matching. Speculative RAG does it at the *draft answer* level, with no exactness guarantee, which is why the analogy is a teaching device rather than a technical equivalence and you should say so.

**📄 Paper:** Wang et al. (2024) — *Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting*. The mechanism:

1. Retrieve a larger candidate set than usual.
2. **Cluster** the retrieved documents so that the clusters represent distinct perspectives or subtopics, then sample subsets that each draw from different clusters — so each subset is a diverse, non-redundant slice of the evidence.
3. A **small specialist drafter** model generates one answer per subset, **in parallel**, along with a rationale.
4. A **larger generalist verifier** scores the drafts and selects the best.

The two wins. **Latency**: the drafts are generated concurrently on a small model over short contexts, rather than one large model reading all documents serially over a long context — so wall-clock is roughly one small-model call plus one verification, not one huge prefill. **Quality**: each draft sees a small, coherent evidence subset without cross-document interference, which is the FiD isolation argument again — the drafter is not confused by five documents arguing with each other, because it only sees one perspective.

Is this shippable? It is the most operationally demanding architecture in this section: you need a second model deployed, a clustering step, and a verifier prompt. **I would reach for it only when the corpus is genuinely multi-perspective** — competing vendor documentation, conflicting policy versions, a research literature review — where the failure mode you are fixing is "the model blended two incompatible sources into one incoherent answer." For a homogeneous internal FAQ it is enormous machinery for nothing.

**The idea worth stealing even if you never build it: draft-then-verify decouples reading evidence from committing to an answer.** A much cheaper version of the same insight is map-reduce — summarize each retrieved chunk against the question independently, then synthesize from the summaries. Same isolation property, one extra parallel fan-out, no second model to operate. That is the version I have actually shipped.

### I have five architectures on the whiteboard — Self-RAG, CRAG, FLARE, Adaptive-RAG, Speculative RAG. Which would you actually build, and in what order?

None of them as published, and I want to be direct about that because it is the honest senior answer. **These are research artifacts that name mechanisms; production systems steal the mechanisms and discard the packaging.** Two of them require training a model, one requires token-level logprobs and mid-stream interruption, one requires a second deployed model. What I actually ship is a straight-line pipeline that has absorbed one idea from each.

In build order, with the trigger for each:

**1. A calibrated retrieval evaluator (from CRAG).** First, always, because it is nearly free — you already have reranker scores, you just have to calibrate them against labels and pick a threshold. It gives you abstention, it gives you conditional query transformation, and it gives you the branch condition for everything below. **This one thing changes the system from "always does the same thing" to "knows when it failed," which is the entire difference between tier 1 and tier 3 thinking.**

**2. Complexity routing (from Adaptive-RAG).** Second, because it is the cost lever — the $110k-to-$40k arithmetic above. One small classifier, folded into the query-understanding call you are already making.

**3. Groundedness checking on the output (from Self-RAG's `ISSUP`).** Third, because it converts hallucination from an unmeasured risk into a monitored metric. Implemented as a cheap post-hoc entailment check per claim, not as trained reflection tokens.

**4. Confidence-triggered re-retrieval (from FLARE).** Fourth, at claim granularity rather than token granularity: if the groundedness check fails on a claim, re-retrieve on that claim and regenerate. Only when step 3 shows the failure rate is high enough to justify the latency.

**5. Draft-then-verify (from Speculative RAG), as map-reduce.** Last, and only for multi-perspective corpora.

**🗣 Say this in the room:** "I wouldn't implement any of these as published. What I'd take is: CRAG's retrieval evaluator, because a calibrated confidence score is the branch condition everything else needs; Adaptive-RAG's complexity router, because that's the cost lever; and Self-RAG's groundedness check, because it makes hallucination a measured quantity. FLARE's active retrieval I'd approximate at claim level since token logprobs and mid-stream interruption aren't reliably available. And I'd build them in that order because each one's value depends on the previous one existing."

That answer works because it demonstrates you have read the literature and are not impressed by it, which is exactly the register these companies hire for.

### Design the iterative retrieve-reason loop. I want the termination conditions and the budget.

The loop is easy to write and easy to ship broken; everything interesting is in the stopping rules, so that is where I spend the answer.

```python
async def iterative_rag(question, retrieve, llm, *,
                        max_hops=3, max_tokens=24_000, deadline_s=8.0):
    ctx, seen, hops = [], set(), 0
    subq = question
    t0 = time.monotonic()
    while hops < max_hops:
        hits = await retrieve(subq, k=6)
        new = [h for h in hits if h.chunk_id not in seen]
        if not new:                                   # STOP: no new information
            break
        seen.update(h.chunk_id for h in new)
        ctx.extend(new)
        if tokens(ctx) > max_tokens:                  # STOP: budget
            ctx = trim(ctx, max_tokens); break
        if time.monotonic() - t0 > deadline_s:        # STOP: wall clock
            break
        step = await llm(NEXT_STEP_PROMPT, question=question, context=ctx)
        hops += 1
        if step.done:                                 # STOP: model says sufficient
            break
        subq = step.next_query
    return await llm(FINAL_PROMPT, question=question, context=ctx,
                     complete=(hops < max_hops))
```

**Five termination conditions and you need all five.** They fail independently:

**No new chunks.** The single most valuable stop, and the one people omit. Without it, a loop that re-retrieves the same six chunks will happily run to `max_hops` while making zero progress — you pay 3× for 1× of information. Deduplicate by chunk ID across hops and stop when the new-chunk count hits zero.

**Hop cap.** Hard, low, and non-negotiable: **3 for interactive, maybe 5 for async**. Multi-hop questions in real corpora are 2-hop with a long tail; a cap of 3 covers the overwhelming majority and bounds the worst case.

**Token budget.** Context accumulates across hops and a loop with no budget will blow past the window and either error or trigger silent truncation that drops the earliest — often most relevant — evidence.

**Wall-clock deadline.** Because a hop is retrieval plus an LLM call, ~1.2 s, and three hops is 3.6 s before the final generation begins. If the product SLO is 5 s total, the loop must be able to give up mid-flight and answer with what it has.

**Model-declared sufficiency.** The soft one. Never trust it alone — the model will declare done prematurely on hard questions and never on ambiguous ones — but as one of five it is fine.

**The output contract matters as much as the loop:** return whether the loop terminated by sufficiency or by budget exhaustion, and *tell the user*. "Based on what I found in three searches" is an honest answer; presenting a budget-truncated answer as complete is how you build a system nobody trusts twice.

**⚠ Trap:** the query for hop N+1 is generated from context that already contains hop N's chunks, so the model tends to drift toward whatever it just read rather than toward what is still missing. Prompt explicitly for the gap: *"List the facts still needed to answer the original question, then write a search query for the first one."* Forcing the gap to be named before the query is written is a small change that measurably reduces the loop's tendency to circle.

### Self-correcting RAG sounds great until it costs 8× per query. How do you keep the loop bounded and how do you know it's actually helping?

Two separate questions and both are usually unanswered in the designs I review.

**Bounding it.** Every self-correction loop needs a **per-request budget object** threaded through the call graph — token budget, wall-clock deadline, and a hard call count — exactly the way you would thread a `deadline` through an RPC chain. Not a config constant read at each site; an object that decrements, so that any component can ask "how much do I have left" and degrade rather than fail. **The specific bug this prevents:** a retry inside the reranker and a retry inside the loop compose multiplicatively, and 3 × 3 = 9 model calls for one user request. Retry amplification is a backend concept you already know; it is worse here because each unit is 400 ms and $0.01 rather than 3 ms and nothing.

Also: **make the correction path cheaper than the original path, not more expensive.** The reflex is to retry with a bigger model. The better design is to retry with the *same* model and better evidence, since the failure was usually retrieval, not reasoning. If the loop escalates model tier on every attempt, your worst-case cost is unbounded in a way that will show up on a bill before it shows up in a dashboard.

**Knowing it helps** — the part almost nobody does. The correct measurement is a **paired comparison on the queries where the loop actually fired**, not an aggregate. Aggregate metrics wash it out: if the loop fires on 12% of traffic and improves those by 15 points, the aggregate moves 1.8 points, which is inside the noise of most eval sets and will be dismissed.

So: log every triggered correction with the pre-correction answer and the post-correction answer, sample 100, and judge them pairwise — better, same, worse. **You are looking for the `worse` bucket specifically.** In every self-correcting system I have measured, there is a non-trivial rate of corrections that make the answer worse: the model had it right, the critic invented an objection, the "corrected" answer is now hedged into uselessness or has drifted to a different question. If `worse` is above roughly 10% of firings, the loop is a net negative even with a healthy `better` rate, because a regression on an answer that was already correct is worth more than an improvement on one that was already wrong.

**🔍 Failure taxonomy — self-correction, as a decision procedure:**
1. **Loop never fires** → the trigger threshold is miscalibrated; check the score distribution, not the threshold value.
2. **Loop fires on >30% of traffic** → your base retrieval is broken; fix that instead, the loop is a symptom.
3. **Loop fires and terminates at max_hops frequently** → the stopping condition is unreachable, usually because the critic prompt demands a certainty the evidence cannot support.
4. **Loop improves the eval set but users complain about latency** → you optimized the wrong metric; add p95 to the same dashboard as quality and make the trade explicit.
5. **`worse` bucket >10%** → the critic is over-triggering; raise its threshold or restrict corrections to specific detected failure classes (unsupported claim, missing entity) rather than a general "is this good?" judgment.

### Where does the golden evaluation set come from? Nobody has labeled data on day one.

From your query logs, and the process for mining them is a real technique with real pitfalls rather than "look at some logs."

**Step 1 — get logs at all.** On day one you have none, so bootstrap: pull the questions from support tickets, the search box of your existing docs site, the `#help` Slack channel, and the FAQ your solutions team already maintains. **These sources are better than synthetic queries because they carry the real vocabulary** — users write "can't log in," not "authentication failure remediation."

**Step 2 — cluster and stratify, do not sample uniformly.** Embed the queries, cluster them, and sample **within clusters proportionally to cluster size but with a floor**. Uniform random sampling gives you a golden set that mirrors the head of the distribution — 40 variants of "how do I reset my password" — and tells you nothing about the tail where your failures live. My split is roughly **50% head (weighted by frequency), 30% torso, 20% deliberate tail**, plus explicit slices you care about: multi-turn follow-ups, identifier queries, multi-hop, and out-of-scope queries that *should* be abstained on. **A golden set with no out-of-scope queries cannot measure over-abstention**, and over-abstention is the regression you will accidentally ship the first time you tighten a threshold.

**Step 3 — label chunk-level relevance, not just answers.** For each query, the annotation you need is "which chunk IDs contain the evidence." That is what lets you compute recall@k and nDCG@k and decouple retrieval evaluation from generation evaluation — the single most important structural property of a RAG eval harness. Answer-level labels alone cannot tell you whether a failure was retrieval or generation, and that is the first question you will ask on every incident.

**Step 4 — LLM-assisted labeling with a human gate.** Have a strong model propose the relevant chunks; have a human confirm or correct. This is roughly 5× faster than labeling cold and keeps the human in the loop where the judgment is.

**⚠ Trap — the synthetic-query bootstrap and its specific bias.** The common shortcut is: take each chunk, ask an LLM to generate a question it answers, and use those as the golden set. It works for getting started and it has a bias that will fool you. **The generated question is derived from the chunk, so it inherits the chunk's exact vocabulary** — and your retriever then looks brilliant, because the query and the target share surface form. Real users use different words. I have watched a synthetic set report recall@10 of 0.94 while the real-query set sat at 0.61 on the same system. Mitigations: prompt for questions in a *different register* than the source ("write it as a frustrated customer would"), and never report a synthetic number without a real-query number beside it.

**📐 Numbers you must know:** **100 labeled queries is the minimum for a decision, 300–500 is a good set, and 1,000+ is a luxury.** The reason for 100 as a floor is the arithmetic: a 5-percentage-point difference in recall on `n = 100` has a standard error of roughly `sqrt(0.5 × 0.5 / 100) ≈ 5%`, so a 5-point delta is barely one standard error and you cannot call it. At `n = 400` the standard error is 2.5% and a 5-point delta becomes a two-sigma result. **Sizing your eval set is the same power-analysis reasoning as sizing an A/B test, and saying that out loud is a strong signal.**

### What is head-of-distribution precomputation and why does it change the cost curve so much?

**Mental model: query distributions in real products are brutally Zipfian, and a system that treats every query as novel is paying full price for work it has already done thousands of times.** Precomputation exploits the head; everything else in this section optimizes the tail.

The observation to measure first: in support and internal-assistant traffic, the **top 100 distinct question intents typically cover 30–50% of volume**. Not the top 100 query strings — the top 100 *intents*, after clustering. That distinction matters because the strings vary endlessly and the intents do not.

The mechanism: cluster your query logs, identify the top N intents by volume, and for each one **precompute and human-review a canonical answer with its citations**. Store them in a lookup keyed on a canonical form. At query time, match the incoming query to the intent and serve the reviewed answer.

The matching step is where this gets dangerous and where the design decision lives. Exact-match keys are safe and have a low hit rate. Embedding-similarity matching has a high hit rate and **is a semantic cache, with all the hazards** — cosine 0.95 between "how do I cancel my subscription" and "how do I cancel my subscription refund" is entirely plausible, and serving the wrong reviewed answer with full confidence is worse than generating a fresh mediocre one. My rules: **a high threshold (0.93+ on a reranker, not a bi-encoder), a mandatory cross-encoder confirmation on the match, and a hard exclusion for any query containing a number, date, identifier or negation** — those are exactly the tokens that flip meaning while barely moving cosine.

**💰 Math:** 1M queries/month, 35% head coverage, 90% of those matched safely. That is `1e6 × 0.35 × 0.9 = 315,000` queries served from a lookup. At $0.032/query for the full pipeline, that is `315,000 × 0.032 = $10,080/month` saved, and those queries return in **~40 ms instead of ~1.8 s** — a 45× latency improvement on a third of your traffic, which moves your *aggregate* p50 more than any model or infrastructure change available to you.

**The second benefit is the one I actually pitch:** the head answers are **human-reviewed**. The 35% of traffic that matters most is now guaranteed correct rather than probabilistically correct. For a support product that is a quality argument, not a cost argument, and it is the argument that gets the work prioritized. **Invalidation is the cost:** every precomputed answer is a stale-cache risk, so each entry needs a source-document version stamp and must be invalidated when any cited document changes. Without that you have built a machine for confidently serving last quarter's policy.

### Design the routing layer for a system with several different indexes — docs, code, tickets, and a SQL warehouse. How does the model decide?

**Mental model: this is tool selection, and the reliability of tool selection is dominated by tool *descriptions*, not by model quality.** The most common failure I see is a router prompt listing four indexes by name — "docs, code, tickets, warehouse" — and then everyone blaming the model when it picks wrong. The model was given four nouns and no decision criteria.

The design that works:

**Describe each index by the questions it answers, with examples and explicit negatives.**

```
docs      — product behaviour, policies, how-to. Use for "how does X work",
            "what is our policy on Y". NOT for "why did X break on Tuesday".
tickets   — customer-reported incidents, resolutions, known issues. Use for
            "has anyone hit X", "what was the fix for Y". NOT for policy questions.
code      — implementation. Use when the question is about actual behaviour of a
            function, config default, or version. NOT for intended behaviour.
warehouse — aggregates, counts, trends over structured data. Use for "how many",
            "what percentage", "trend over time". Never for narrative questions.
```

**Allow multi-select and default to it when uncertain.** "Why is the retry limit 3?" plausibly hits docs (the policy), code (the constant), and tickets (the incident that caused it). Retrieving from all three and fusing with weighted RRF costs three concurrent searches — under 100 ms, since they parallelize — and is dramatically more robust than forcing a single choice. **Routing is only worth the risk of being wrong when the alternatives are expensive; concurrent cheap retrievals should be fanned out, not routed.** The warehouse is the exception: text-to-SQL is expensive and risky, so that branch stays exclusive and gated on high confidence.

**Fuse across indexes with per-index weights, not raw scores.** Scores from a dense index over docs and BM25 over tickets are not comparable in any sense. RRF with per-index weights tuned on your golden set is the correct primitive — and the weights encode real editorial judgment, e.g. down-weighting tickets because they are numerous, noisy, and often describe problems rather than answers.

**⚠ Trap:** routing on the *raw* query in a multi-turn conversation. The router sees "what about in the EU?" and has no idea which index the conversation was about. **Route on the rewritten standalone query, always** — which means rewriting must happen before routing, and is another reason to fold both into one structured-output call rather than chaining them.

**🏋 Drill:** 25 minutes. Write the router prompt for these four indexes with a structured output schema supporting multi-select and per-index confidence, plus the fusion code that takes four ranked lists and per-index weights and produces one ranking. *Pass criterion:* the descriptions contain explicit negative examples, the schema allows multi-select, the fusion is rank-based rather than score-based, and there is a defined fallback when confidence is low — fan out to all text indexes, never to the warehouse.
