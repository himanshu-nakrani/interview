### Tell me about an LLM feature you shipped. I want the numbers.

The mental model for constructing this answer: an AI-feature story is graded on whether you had a **closed loop** — a way to know whether the thing worked that existed *before* you shipped it. Anyone can describe a feature. The senior signal is that you can state, in order, the baseline, the intervention, the measurement, and the delta, and that the measurement wasn't invented after the fact to justify the launch.

Here is the shape, with example arithmetic you must replace with your own real figures. "We had a support-answer feature where a first-pass retrieval-plus-generation pipeline was resolving about 41% of tickets without escalation, measured on a rolling 7-day window against human-labelled resolution. Two problems: p95 was 6.4s against a 3s target, and cost was $0.019 per answered ticket at roughly 28k tickets/day, so about $530/day, $16k/month. I did three things in priority order: moved the 11k-token system prompt behind a provider prefix cache, which at $3/Mtok uncached versus $0.30/Mtok cached takes that portion from 11,000/1e6 × 3 = $0.033 to $0.0033 per call — call it $0.03 saved per call, so 28k × $0.03 = $840/day gross, though our real cache-hit rate was 78%, so ~$655/day; second, moved reranking off the critical path for the 60% of queries where the top-1 BM25 score cleared a threshold, which took p95 from 6.4s to 2.9s; third, added 140 labelled cases to the eval set from the escalation logs, which is what caught that the reranker skip was costing us 3 points of resolution on multi-part questions, so I gated it to single-intent queries."

Notice the structure: three interventions, each with a number, and **one of them partially backfired and I found it because of the eval**. That last clause is worth more than the other two combined. It proves the loop was closed.

**⚠ Trap:** reporting cost per call instead of cost per resolved task. At this tier the interviewer will convert for you and it is much better if you've already done it. Cost per call goes *down* when you route to a cheaper model; cost per resolved task can go *up* at the same time, because the cheap model retries. State the denominator you're using and why.

**💰 Math:** the retry amplification is worth internalizing. If a cheap model at $0.30/Mtok input resolves 62% first-pass and you retry once on failure, expected calls per resolution = 1/0.62 = 1.61, plus the failed-attempt tokens are pure waste. A model at 4× the price resolving 88% needs 1/0.88 = 1.14 calls. Cost ratio is 4 × 1.14 / 1.61 = 2.83×, so the expensive model is still 2.8× the spend — but if each escalation to a human costs $4.20 in agent time, the cheap model's extra 26 points of escalation on 28k tickets/day is 0.26 × 28,000 × $4.20 = $30,576/day. The model price difference is noise against that. This is the arithmetic that makes you sound like you've shipped.

**🗣 Say this in the room:** "The number I care about is cost per resolved task, not per call — cost per call went down 40% in that project and total spend went *up*, because the cheaper model retried."

### Walk me through an eval you built. What did it measure, and what decision did it change?

This is the highest-signal story for any AI-engineering role and the one most career-switchers cannot tell, which makes it the best possible use of your prep time if you don't have one yet. The mental model: **an eval is a test suite for a system whose output space is unbounded, so its value is entirely in how you chose the cases.** Interviewers will therefore spend most of the follow-ups on set construction, not on metrics.

The answer needs six things. What the system was and what "correct" meant for it — the hard part, and you should say so. Where the cases came from — production logs beat synthetic, and stratified sampling from real traffic beats a hand-picked list, because hand-picked lists encode the failures you already know about. How many cases, and why that many. Who labelled and what the inter-annotator agreement was, or if it was just you, say that plainly. What the metric was and what it doesn't capture. And the decision it changed.

Concretely: "156 cases stratified across five query intents from a month of production logs, weighted to match real traffic, plus 30 deliberately adversarial ones I wrote from the escalation queue. Scored by exact-match on the extracted field for the structured half and by a rubric-based LLM judge for the free-text half, where I validated the judge against 60 of my own labels and got 87% agreement — which I'd call adequate for ranking two prompt variants and not adequate for a release gate. The decision it changed: we were about to ship a 'summarize the thread' improvement that scored better on the aggregate and 9 points *worse* on the refund-policy slice, which is 4% of traffic and 100% of our regulatory exposure. We didn't ship it."

**📐 Numbers you must know:** for a binary metric near 50%, the standard error on n cases is about 0.5/√n. At n=100 that's 5 points, so a 3-point improvement is inside the noise; you need roughly n ≈ (0.5/0.01)² = 2,500 to resolve 1 point, or a paired design on the same cases which kills most of the variance and is why paired bootstrap is the default. If you say "our eval had 40 cases and we saw a 5% improvement," a good interviewer will ask whether you could distinguish that from nothing. Have the answer.

**⚠ Trap:** describing an LLM-as-judge eval without saying how you validated the judge. An unvalidated judge is a random number generator with good manners. Always state the agreement number against human labels, and always state what you would and would not use it for.

**🏋 Drill:** in 10 minutes, on paper, design the eval for a feature you actually shipped — cases, source, count, labelling, metric, slice breakdown, and the specific decision it would gate. Pass criterion: it contains a slice where you expect performance to be worse than average, and a stated sample size with a justification.

### Have you fine-tuned a model? Walk me through it — hardware, cost, and the measured delta.

Answer this honestly at whatever depth you actually have, because the follow-ups are unforgiving and the penalty for overclaiming is far worse than the penalty for a modest but well-understood project. The mental model to lead with: **fine-tuning is the last rung of the escalation ladder, and the interesting part of the story is why the four rungs below it didn't work.** If your answer opens with hyperparameters, you've signalled that you reach for fine-tuning reflexively, which is a known rejection trigger in applied-AI loops.

The structure: the failure that prompted it (a behavior prompting could not reliably produce — format adherence at high volume, a domain vocabulary, a latency budget that forced a smaller model), what you tried first and why it was insufficient, then the run. For the run itself, be able to state: base model and size, adapter method and rank, dataset size and provenance, sequence length, hardware, wall-clock, and cost.

Example of the shape: "Qwen-class 7B base, LoRA rank 16 on the attention projections, 8,400 examples curated from production traces where a human had corrected the output, 2 epochs at 2k sequence length, one A100-80GB, about 3.5 hours. At a rented ~$1.80/GPU-hour that's roughly $6.30 of compute for the run, and about four *days* of my time on data curation — which is the real number and the one I'd emphasize. The delta was format-compliance on our structured output going from 91.2% to 99.1% on a held-out 400-case set, which mattered because each malformed response cost a retry, and 8.8% of 200k daily calls at $0.004 a retry is 17,600 × 0.004 = $70/day, $2,100/month, against a one-off $6 of compute plus serving." **📅 Volatile:** GPU rental prices move constantly; state yours as "at the time, roughly $X/hr" rather than as a current fact.

If you haven't fine-tuned anything: say so directly and then say what you'd do, with the same specificity. "I haven't run one in production. If I had to, the trigger would be a behavior I couldn't get with prompting plus few-shot plus structured decoding, and my first move would be LoRA rather than full fine-tuning because of the memory arithmetic — full fine-tuning a 7B in bf16 needs weights plus gradients plus two Adam moments in fp32, so roughly 14 + 14 + 56 = 84GB before activations, which doesn't fit an 80GB card, whereas LoRA at rank 16 trains well under 1% of parameters." That answer scores better than a vague claim of experience, because it demonstrates the model that makes the decision.

**⚠ Trap:** the reflex-fine-tuning rejection. When asked "how would you improve quality here?", a candidate whose first answer is "fine-tune" is filtered at many applied loops. The expected first answers are: fix the retrieval, fix the context, fix the tool interface, constrain the decoding, then route. Fine-tuning is rung six.

### Your resume says you built an agent with LangGraph. What did you actually write, and what did the framework do?

This is the inflated-resume probe and it is deployed constantly in 2026 loops because framework-shaped resumes became indistinguishable from each other. The mental model for surviving it: **the interviewer wants to know whether you can rebuild the abstraction, not whether you can call it.** The winning move is to volunteer the boundary before they draw it.

The answer has three parts. First, name what the framework gave you, accurately and without diminishing it: "state persistence across steps, the graph execution semantics, checkpointing, and the streaming plumbing." Second, name what you wrote: "the tool schemas and their validation, the retry-and-backoff policy including which errors are retryable, the token budget enforcement, the termination conditions, the trace instrumentation, and about 200 lines of the actual control-flow decisions." Third — and this is the part that converts — **demonstrate you could do it without the framework**: "the core is a while loop over messages that appends `tool_use` blocks and their `tool_result` responses back into the conversation until the model stops requesting tools or you hit a budget; the framework's value for us was the persistence and the visualization, not the loop."

Being able to sketch the raw loop in ten lines of pseudocode on demand is worth rehearsing:

```
messages = [system, user]
for step in range(max_steps):
    resp = model(messages, tools=tool_schemas)
    messages.append(resp)
    if not resp.tool_calls: return resp
    for call in resp.tool_calls:
        result = dispatch(call)          # validate args, timeout, catch
        messages.append(tool_result(call.id, result))
    if tokens_used > budget: return degrade()
raise StepLimitExceeded
```

**⚠ Trap:** defensiveness. Candidates hear this question as an accusation and start justifying the framework choice. It isn't an accusation; it's a depth probe. The answer "the framework did most of it and here's exactly which parts, and here's the piece I'd have had to write myself" is a *strong* answer. The failing answer is a vague "well, I built the whole agent."

**⚠ Trap #2:** the mirror-image overcorrection — claiming you wrote everything from scratch when you didn't. This gets caught by one question about a detail of the thing you claim to have built, and it ends the interview in the debrief even if nobody says so at the time.

**🗣 Say this in the room:** "Let me draw the line myself, because I think it's the interesting part — LangGraph gave us persistence, checkpointing and the graph semantics. What I wrote was the tool layer, the budget enforcement and the termination logic, and honestly the loop underneath is about ten lines, which we'd have written ourselves if we hadn't needed the durable-state story."

### Tell me about a production incident you owned end to end.

The mental model: an incident story is a **debugging-under-uncertainty demo with a stakes multiplier**, and the ending they care about is not the fix — it's whether the class of failure can recur. Everyone lands the fix. Fewer candidates land the systemic change, and almost nobody lands "here is the specific test that now exists."

Structure it as: detection (how did you find out, and how *should* you have found out), the wrong hypothesis, the measurement that killed it, the real cause, the mitigation versus the fix — these are different and saying so signals experience — the customer impact quantified, and the durable change.

An AI-flavored incident is worth more than a generic one in these loops because it demonstrates the failure taxonomy of the new domain. **🔍 Failure taxonomy** for AI incidents, which doubles as a way to classify your own story: *silent quality regression* (a prompt or model version change degrades output while all HTTP metrics stay green); *retrieval staleness* (index rebuilt, alias not swapped, or embeddings from a different model version mixed into one index); *cost blowout* (a retry loop or an agent that doesn't terminate); *tail-latency collapse* (a long-context request monopolizing a serving batch); *safety/PII leak* (context assembled across tenant boundaries); *upstream provider change* (a model deprecation or a silent behavior change under the same model name). Being able to name six categories cold makes the follow-up conversation much stronger than any single anecdote.

The quantification is mandatory. "Roughly 9% of requests for 4 hours 20 minutes — about 3,400 affected sessions at our 8k/hour rate — and because the failure returned 200 with a plausible answer, none of our alerts fired; we found it from two support tickets." That sentence is worth more than the whole rest of the story, because it names the reason this domain is hard: **wrong is not an error code.**

**🗣 Say this in the room:** "The thing I took away is that our monitoring could tell us the system was *up* and could not tell us it was *right*, and those had never been different before. The durable fix was a canary set of 40 questions with known answers running every 15 minutes against production, which would have caught this in under an hour."

**⚠ Trap:** ending on the fix. If your story ends "and we rolled back and it was fine," you have described an outage, not an incident you owned. End on the detection gap you closed.

### Tell me about a system you argued against building.

This is a costly-signal question and one of the highest-scoring stories in the bank, because arguing against work is uncomfortable, career-risky, and therefore hard to fake. The mental model: they are testing whether your product judgment is real enough that you'd spend political capital on it, and — critically — whether you can distinguish between *"this is hard"* and *"this shouldn't exist."* Engineers who argue against everything hard are a liability; engineers who never argue are a different liability.

The strong version of this story has four elements. A specific proposal with a plausible case for it — steelman it, because if the idea was obviously bad your judgment wasn't tested. The reason you objected, stated in terms of *outcome* rather than effort. What you proposed instead. And the ending, which can go either way: you can have lost and it's still a strong story, provided you executed well afterward.

For an AI role, the highest-value instance is a case where the objection was "the technology won't do this reliably and the failure is expensive." Example shape: "Sales wanted an agent that would autonomously issue account credits up to $2,000 based on a support conversation. I pushed back — not because the extraction is hard, it isn't, but because the error is unbounded and unrecoverable: a false positive is money out the door and a false negative is invisible. I proposed the model draft the credit with a justification and a confidence, and a human approve; we measured that the drafting alone cut handling time from 6.2 minutes to 2.1, which captured most of the value at none of the risk. The commitment I made was that if the draft-approval rate exceeded 95% over a quarter on the under-$200 band, we'd revisit auto-approval for that band — and we did."

That ending is the senior move: **you didn't block, you sequenced.** You converted an unbounded risk into a measured one with an explicit graduation criterion.

**⚠ Trap:** telling this as a story about being right. The tone should be "here's how I reasoned and here's what I committed to that would have proven me wrong." Certainty about a judgment call reads badly in this exact round.

### Describe something you shipped that you later concluded was wrong.

The mental model: this question has two failure modes and they're symmetric. **Defensiveness** — "it was the right call with the information we had" — signals you don't update. **Performative self-flagellation** — an over-rehearsed confession delivered with too much emotion — signals you're managing the interviewer's impression rather than answering. What scores is a flat, specific, slightly boring account of a real mistake, with a clear statement of what you now believe and why.

The selection criterion for which story to use: it must be a **judgment error, not an execution error**, and it must be one where you had the information or could have gotten it. "We had an outage because of a race condition I missed" is an execution error and a weak answer. "I designed the whole thing around an assumption I never tested, and testing it would have taken an afternoon" is a judgment error and a strong one.

Example shape: "I built a semantic cache in front of an answering pipeline — embed the query, and if cosine similarity to a cached query exceeded 0.94, return the cached answer. It cut cost meaningfully and I was pleased with it. What I got wrong is that semantic similarity is not the same as answer-equivalence: 'can I cancel my order' and 'can I cancel my subscription' sat above the threshold in our embedding space and got the same answer. It was a small fraction of traffic and it took us six weeks to notice, because a wrong-but-fluent answer doesn't page anybody. What I'd do now is either scope the cache to a narrow, verified intent set, or keep the cache and re-verify the retrieved answer against the new query with a cheap model before returning — and, more generally, I don't ship a cache whose failure mode is 'confidently wrong' without a sampled correctness monitor on the cache hits specifically."

Note the three moves: name the mechanism precisely, quantify the blast radius honestly (including "it was small"), and state a *general rule* you now hold. The general rule is what makes it a learning rather than an anecdote.

**🗣 Say this in the room:** "The error wasn't the implementation, it was that I treated a similarity threshold as if it were an equivalence relation. I'd been careful about the cache invalidation and completely uncritical about the cache *key*."

**⚠ Trap:** using a story where the mistake was ultimately someone else's, or where the "mistake" is secretly a virtue ("I cared too much about test coverage"). Every interviewer has a name for that answer and it costs you more than the honest one would.

### Tell me about a time you disagreed with someone more senior and lost.

The scored variable is not who was right. It is **what you did on day two.** A candidate who lost an argument and then quietly under-invested in the outcome is describing a form of sabotage, and interviewers listen for it in the texture of the story — a certain relish in "and then it failed exactly like I said."

The strong structure: state your position and the strongest version of theirs, describe how you argued it (in writing? with data? in a room?), describe the decision, then describe your commitment to the decision, then the outcome and what you learned about *how you argue* rather than about who was right.

The most useful detail you can include is a **disagree-and-commit artifact**: something concrete you did to make the chosen path succeed while preserving the ability to detect if it failed. "I wrote down the two things I thought would break and the metric that would show it, put it in the design doc as an open risk, and then built the instrumentation for both. Six weeks later one of them did happen — the tail-latency one — and because the instrumentation existed we caught it in a day instead of a quarter, and we changed course without anyone having to relitigate the original argument." That is a masterclass answer: you lost, you committed, you made the loss cheap to reverse, and you removed ego from the correction.

If you were wrong, say so plainly and say what you'd missed. Candidates dramatically underrate "I lost and they were right" as an answer. It's rare, it's credible, and it directly demonstrates the update capability the round is testing.

**🗣 Say this in the room:** "I lost that one, and I still think my read on the risk was right — but I was wrong about the timeline, and my argument leaned on the risk without pricing how much the delay would cost, which is why it didn't land. Now when I disagree I try to bring both numbers."

### How do you talk about a project where the metric didn't move?

Carefully, and it's worth having one, because a bank of eight unbroken successes reads as either lucky or edited. The mental model: a null result is a strong story **if and only if** you can show that the experiment was designed well enough for the null to be informative. A badly-run project that failed teaches nothing; a well-run project that failed teaches the interviewer that you can distinguish those two cases.

The elements: the hypothesis, stated as a falsifiable prediction with a number ("I expected reranking to add 4–6 points of answer accuracy on the multi-hop slice"); the design (paired, same queries, held-out set, what you controlled for); the result (no significant change, with the confidence interval); the diagnosis (why — and "I don't fully know" is acceptable if you say what you'd measure next); and the decision (you removed it, because it added 180ms of p95 for nothing, and un-shipping is a real skill).

**💰 Math on why un-shipping matters:** a reranker that adds 180ms of p95 and 2 extra model calls at, say, $0.0004 each, across 400k requests/day, is 400,000 × 0.0008 = $320/day = $9,600/month for a measured-zero benefit. Being the person who deleted that is worth as much as being the person who added something. Say the number.

**⚠ Trap:** dressing a null up as a win ("it didn't improve accuracy but it improved consistency"). If you didn't measure consistency prospectively, don't claim it. Post-hoc rescue narratives are exactly what a good interviewer is scanning for, and getting caught inventing a secondary metric contaminates everything else you said.

**🗣 Say this in the room:** "It didn't work, and the useful part is that we knew within nine days instead of shipping it and wondering. The design was paired on 300 held-out queries, so we could rule out anything above about 3 points."

### What would your last manager say you need to improve? — how do I answer this without a fake weakness?

The mental model: this question is a **calibration test**, not a confession. They are checking whether your self-model matches the one a reasonable observer would have. A candidate with no accurate weakness has a broken self-model, which is the actual disqualifier — the weakness itself is almost never the problem.

The formula that works: a real weakness, in a dimension that is genuinely a weakness and not a disguised strength, plus the specific mechanism you've built to contain it, plus an honest statement of how well the mechanism works. The third clause is what makes it credible — "I've mostly fixed it" is suspicious; "it's better, and it still shows up when I'm under time pressure" is human.

Good raw material for a senior backend engineer moving into AI: a tendency to go deep on mechanism when the situation called for a fast approximate answer; over-indexing on correctness in early-stage work where speed dominates; writing for the reader who's already deep rather than the stakeholder who isn't; taking on too much of the hard work personally instead of distributing it, which caps team throughput at your own. Each of these is a real cost to a real team, which is why they're believable.

Example shape: "The consistent feedback is that I under-communicate in the middle of hard problems. I'll go quiet for two days while I'm converging and my manager finds out where things are on Thursday. The mechanism I use now is a written end-of-day line in the channel — status, current hypothesis, what would change my mind — which fixes it about 80% of the time. It still slips when I'm deep in a debugging session, and my current fix for that is a calendar block rather than willpower."

**⚠ Trap:** "I'm a perfectionist," "I work too hard," "I care too much about code quality." These are not answers; they are the absence of an answer, and every interviewer maps them to "not self-aware" or "not being straight with me." Either one is fatal in the values round specifically.

### My best public artifact is a written guide and a side project rather than a shipped company system. How do I present that without it sounding like a hobby?

Reframe from content to engineering, and do it with the vocabulary of a system rather than the vocabulary of a publication. The mental model: an interviewer does not care that you wrote a guide; they care whether you built something with scale, constraints, and decisions in it. A 1,600-question technical corpus with a search layer, an ingestion pipeline and a deployed front end *is* a system, and it has the exact properties this job cares about — a corpus, a retrieval problem, a quality bar you had to define yourself, and a deployment.

Present it as four things. **Ingestion and structure** — how content is chunked, what the schema is, how you handle updates without breaking links. **Retrieval** — what the search actually does, and what you measured; if you built lexical search over it, say why you didn't reach for embeddings, or if you did, say what recall you got and on what query set. **Deployment and scale** — hosting, build times, page count, latency, what broke at what size. **Quality control** — how you keep 900k words internally consistent, which is a genuinely hard problem and maps directly onto the data-quality work that dominates applied AI.

Then preempt the obvious probe, because it's coming: *"did you just compile this from other sources / did an AI write it?"* Answer it before they ask, with the boundary drawn explicitly — what you used a model for, what you verified, and what your verification process was. That is exactly the same disclosure discipline the job requires, so answering it well is itself the signal.

**🗣 Say this in the room:** "I'd rather talk about it as a system than as writing. It's a corpus with an ingestion pipeline, a retrieval layer and a deployment, and the hardest engineering problem in it was consistency at scale — the same concept explained in eleven places has to not contradict itself, and that's a data-quality problem, not a prose problem. I'll tell you exactly where I used a model and exactly what I verified by hand."

**⚠ Trap:** leading with the volume ("1,633 questions, 900,000 words"). Scale as an opener sounds like a content-marketing pitch. Lead with a decision — "I chose lexical over vector search for this corpus, and here's why that was right and where it now fails" — and let scale come out as context.

### How do I claim a metric I can no longer verify because I've left the company?

State it with its provenance and its uncertainty, in that order, and never round in your favor. The mental model: interviewers are not fact-checking your dashboards; they are testing whether your relationship with numbers is careful. A candidate who says "roughly 60%, measured over a two-week window against the prior release, and I'd want to caveat that our traffic mix shifted in that period" is *more* trustworthy than one who says "62.4%," because the first one demonstrably knows what would make the number wrong.

The four questions to have an answer for on every metric in your bank, because they are the standard probe sequence: **against what baseline** (the previous system, a no-op, a human?), **over what window** (a day of traffic is not a result), **on what population** (all traffic, or the slice where it works?), and **with what variance** (was it stable, or did you sample the good week?). If any of your headline numbers can't survive all four, demote it in the bank — put it in the story as color and lead with a different, defensible one.

**📐 Numbers you must know** about your own bank: every story should carry three numbers of *different kinds* — one scale number (traffic, corpus size, tenants), one performance number (latency, accuracy, cost), and one consequence number (dollars, incidents avoided, headcount-hours). Three numbers of the same kind reads as one fact repeated.

**⚠ Trap:** the un-anchored percentage. "Improved performance by 40%" invites an unanswerable follow-up because there's no unit. Say "cut p95 from 4.2 seconds to 2.5" — absolute values, both ends. It's more informative, it's harder to fake, and it makes the follow-up conversation technical instead of forensic.

**🗣 Say this in the room:** "I don't have the dashboard anymore, so treat it as approximate — p95 went from just over 4 seconds to a bit under 2.5, measured on a week of production traffic against the previous release, and the caveat is that we shipped it alongside a client-side change, so I'd attribute most but not all of it to the backend work."
