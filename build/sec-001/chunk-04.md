### You keep referring to "archetypes." Define them, and tell me why you'd stamp one on every prep item.

Because loop structure correlates far more strongly with company type than with job title, an archetype tag is the highest-compression routing device available. There are six that matter:

**Frontier lab** (OpenAI, Anthropic, DeepMind, xAI, Mistral). Deepest technical bar, heaviest weight on judgment and depth-of-ownership, work trials and research deep dives, values rounds with real teeth.

**AI infrastructure** (Nvidia, Together, Fireworks, Baseten, Modal, lab infra teams). Systems loops with an unfamiliar bottleneck: GPU scheduling, batching, memory, preemption, throughput per dollar.

**AI product** (Cursor, Perplexity, Notion, Figma, Sierra, Harvey, Glean). Practical build rounds, take-homes, retrieval and agent design, product sense, evaluation methodology as the recurring discriminator.

**Big-tech applied** (Meta, Google, Amazon, Microsoft). Classical loop machinery plus one or two AI rounds; structured behavioral scoring; reliability and cost discipline over capability enthusiasm.

**High-comp non-AI-native** (Stripe, Ramp, Snowflake, Databricks, Confluent). Very high classical engineering bar, moderate AI-specific bar, heavy emphasis on treating the model as a governed production dependency.

**Forward-deployed** (OpenAI FDE, Anthropic Applied, Palantir, Scale, Sierra, Harvey). Scoping and customer-conversation rounds, fast end-to-end building, ambiguity tolerance.

The reason to stamp the tag on every prep item is subtraction. There is more material in this domain than any human can hold, and the tag is what lets you *honestly skip* rather than guiltily skim. If your list is three AI-product companies and two big-tech applied, then distributed-training-systems design is a skip, not a gap — and knowing that with confidence is worth more than a shallow pass over it.

**⚠ Trap:** Targeting all six. Every candidate wants to keep options open, and the result is uniformly shallow preparation that fails everywhere. Pick two archetypes, accept that you are choosing not to be competitive at the other four this cycle, and revisit next cycle.

### I have eight weeks and I'm targeting AI-product companies. Give me the critical path and the time budget.

Eight weeks at a realistic 15 hours a week is 120 hours. Here is how I would spend it, and the discipline is that every block ends in an artifact, not a feeling of having read something.

**Weeks 1–2 (30h) — routing and foundations.** The taxonomy and loop map (this material), the skip/skim/attack subtraction pass so you stop re-learning what you already ship, the provider and model landscape, and the process mechanics of take-homes. Then the math that actually gets asked and the "is an LLM even right here" gate. Artifact: your target list with archetype tags and a predicted loop per company.

**Weeks 3–4 (30h) — the internals floor.** Attention from scratch until you can write multi-head attention with a correct KV cache in under 25 minutes unaided; KV-cache arithmetic; tokenization; sampling and decoding; embeddings. You are not going for research depth, you are going for the ability to answer a follow-up two levels down without flinching. Artifact: a from-scratch attention implementation with a memory-versus-context-length measurement.

**Weeks 5–6 (35h) — the core of the job.** RAG end to end (chunking, hybrid retrieval, reranking, evaluation), agent design and tool contracts, structured output, and **evaluation as a discipline** — which is the single highest-return block in the entire plan for this archetype. Artifact: a working RAG-plus-agent system on a real corpus with an eval harness that produces a scored report and a cost-per-request number.

**Weeks 7–8 (25h) — loop simulation.** Mock system design rounds out loud and timed, a full take-home under real time constraints followed by a self-administered defense, rehearsed behavioral and product-sense stories, and a light algorithmic maintenance track if Perplexity or a similar exception is on your list. Artifact: three recorded 45-minute mock rounds you have watched back.

**📐 Numbers you must know about your own plan:** 120 hours is roughly 2.5% of a working year and it is genuinely enough for this archetype *only* if you subtract aggressively. The failure mode is spending 40 of those hours re-reading distributed systems material you already own. Run the skip list first; it is the highest-leverage hour in the plan.

**⚠ Trap:** Leaving evaluation for last because it is the least fun. It is the most-graded and least-prepared competency in this archetype's loops — take-home rubrics weight it explicitly and system design rounds reward opening with it. If you are behind schedule, cut internals depth before you cut evaluation.

### Same question but big-tech applied AI, ten weeks. What changes?

The shape inverts: you spend *more* time on things you already know, because the classical rounds are still there and they are still scored hard.

Ten weeks at 15 hours is 150 hours, and I would allocate roughly: 35 hours to classical loop machinery (algorithmic coding maintenance, general system design at their scale, and — this is the part backend engineers skip and regret — structured behavioral preparation), 45 hours to the AI content (RAG, agents, evaluation, plus enough internals to survive a depth probe), 30 hours to the ML-system-design flavor these companies favor (end-to-end thinking including data pipelines, offline and online metrics, A/B testing, and rollout), 20 hours to cost-and-reliability framing at their scale, and 20 hours to mocks.

Three specific differences from the AI-product plan.

*Behavioral preparation is real work here, not a night before.* Amazon's Leadership Principles are evaluated explicitly by a bar raiser with veto power; Google's hiring committee reads written packets, which means your interviewers must be able to *write you up legibly*. Prepare 6–8 stories in STAR form with metrics attached, each mapped to two or three principles, and rehearse them to 2 minutes. This alone is 10 hours and it is not optional.

*Scale changes the design answer.* At 50M requests/day the interesting part of the design is unit economics and graceful degradation, not model capability. Practice the arithmetic until it is automatic.

*Internals depth is less load-bearing.* You will rarely be asked to derive attention at Meta or Amazon for an applied role. Know the mechanism well enough to reason about cost and latency; do not spend 30 hours on it.

**⚠ Trap:** Assuming the AI round is the hard one. In big-tech applied loops the modal rejection I see is on the *classical* system design or the behavioral packet, not on the AI content. Prepare in proportion to where you actually lose.

### I want to keep both an AI-infra and an AI-product option open. Can I prep both, and where do I cut?

Partially, and the honest answer is that you cut *depth in one direction*, not breadth uniformly — because the two archetypes share a surprising amount at the middle layer.

The shared core is bigger than people assume: transformer mechanics at the level of shapes and memory, KV-cache arithmetic, tokenization, sampling, batching intuition, latency decomposition (TTFT / ITL / total), cost modeling, and general distributed-systems reasoning. That is maybe 60% of what either loop needs, and it is one body of study.

The divergence is at the ends. AI-infra wants you *below* the API: serving-engine internals, PagedAttention-style memory management, quantization, parallelism strategies, GPU scheduling and preemption, kernel-level intuition. AI-product wants you *above* it: retrieval quality, agent design, evaluation methodology, product metrics, prompt and context engineering.

The cut I would make: study the shared core to full depth, then pick **one** end to go deep on and take the other only to "can hold a conversation." Trying to be deep at both ends in a single cycle produces a candidate who is mid at both, and both loops probe depth.

The asymmetry worth naming: it is easier to go product→infra later than the reverse, because infra depth compounds slowly and requires access to hardware, while product depth can be built from a laptop and an API key. If you genuinely cannot choose, choose product for this cycle and treat infra as the next one.

**🗣 Say this in the room** (when asked which direction you want): "I'm strongest above the model — retrieval, agents, evaluation, cost. I have real systems depth underneath it, so I can reason about serving tradeoffs and I'd want to grow there, but I'd be overclaiming if I told you I've tuned a serving engine in production."

### How should I sequence practice loops versus target loops, and why does the order matter so much?

Because interviewing is a skill that decays and improves on a fast timescale, and because offers expire. Those two facts fully determine the sequence.

The plan I would run: **two practice loops first**, at companies you would accept but are not optimizing for, deliberately started 2–3 weeks before anything else. Their function is to surface the gap between what you can explain at a desk and what you can explain out loud at minute 40 with someone probing. You will find, reliably, that two or three answers you thought were solid collapse under follow-up. That information is not available any other way — self-assessment systematically overrates verbal fluency.

Then **target loops clustered**, with start dates staggered by expected process length so that offers arrive within the same 10–14 day window. This is the arithmetic that most candidates get wrong: a fast AI-product company can run 11–19 days end to end, while big tech can run 4–8 weeks including committee and offer review. If you start them on the same day, the fast offer arrives with a 5–7 day exploding deadline while the slow one is still in the middle of its loop, and you have no leverage.

**📐 Numbers you must know:** to have offers land together, start the slowest process first. If big tech is ~6 weeks and an AI-product company is ~2.5 weeks, you start big tech in week 0 and the product company in week 3.5. Add a one-week buffer, because take-home scheduling and interviewer availability always slip in the direction of slower.

**💰 Math on why clustering is worth this much trouble:** competing offers at this level move total compensation on the order of 10–25%. On a $400k package that is $40k–$100k per year, and it is entirely a function of *calendar arithmetic*, not of how well you answer any single question. There is no other preparation activity with that return per hour. **📅 Volatile:** bands and negotiation norms shift; verify current market data before anchoring.

**⚠ Trap:** Using your favorite company as your first loop. Your first loop of a cycle is your worst, always, and it is worst on exactly the dimension — verbal fluency under probing — that no amount of reading fixes. Spend it somewhere you can afford to lose.

### What questions do you ask the recruiter to de-risk the loop, and what do the answers actually tell you?

Six questions, in this order, on the first call. Each one changes what I do afterward.

**"Which team and org is this for, and what does it own?"** — Routes everything. At OpenAI in particular, team variance is larger than company variance. A vague answer means the req is speculative or the recruiter is agency-side, which changes how much I invest.

**"Can you walk me through the stages and what each is evaluating?"** — Gives me the predicted loop. A recruiter who cannot describe their own process is a signal about the org, and a recruiter who describes it precisely is worth trusting later.

**"Is there a data-structures-and-algorithms round, or is the coding practical?"** — The single highest-value factual answer, because it determines whether I spend 30 hours on algorithms.

**"What's the most common reason strong candidates don't get through?"** — The hidden-weight question. It works because it is framed as helping them, not as gaming them, and it gets a real answer surprisingly often.

**"What's the AI-tool policy in the technical rounds?"** — Determines whether I practice with or without autocomplete, and whether the round is proctored.

**"What's the band for this level, and how is the level determined?"** — Anchors before I do, and surfaces whether leveling is decided by the loop or pre-set by the req. If it is pre-set below where I want to be, that is a conversation to have *now*, not after five rounds.

**🗣 Say this in the room:** "I want to prepare for the right things rather than guess — could you tell me what each stage evaluates, and whether the coding portion is algorithmic or practical?"

**⚠ Trap:** Treating the recruiter as an obstacle. They are the highest-information, lowest-cost source in the entire process and their incentive is aligned with yours: they get credit for candidates who convert. Candidates who are terse and defensive on the screen lose access to information that is freely available for the asking.

### Here's a debugging scenario. A candidate keeps reaching onsites and failing at the same stage every time. How would you diagnose it?

I'd treat it exactly like a production incident: get the failure signal, localize it, form a hypothesis with a testable prediction, and fix one variable at a time. The mistake candidates make is the same one engineers make with a flaky test — they change five things at once and learn nothing.

**🔍 Failure taxonomy — by stage of failure:**

*Fails at the recruiter screen.* Cause is almost never technical: it is legibility or band. The resume does not contain a numbered AI artifact, or the comp expectation disqualified them. Fix: rewrite the top third of the resume as claim + metric + mechanism, and stop giving a number first.

*Fails at the technical screen.* Two distinct causes with different fixes — algorithmic screens fail on speed and fluency (fix: volume practice, unassisted), practical build screens fail on environment friction and scoping (fix: rehearse a cold-start build, have a known-good project skeleton, and practice cutting scope out loud).

*Fails after the take-home.* Nearly always the missing evaluation section, or a submission with no README explaining decisions. This is the most common silent failure in the entire market and it is completely fixable in four hours of work per submission.

*Fails at the take-home defense.* The build was assembled rather than decided. Fix: a decision log written *during* the build, with alternatives.

*Fails at system design.* Two sub-causes to distinguish by asking what the interviewer probed. If they kept asking "how would you know," you skipped evaluation. If they kept asking "what does that cost," you skipped arithmetic. If they kept asking "what happens when," you designed the happy path.

*Fails at behavioral or values.* Vagueness, no metrics, no escalation, or stories that collapse under follow-up. Fix: write the stories down, put a number in each, and have someone hostile probe them.

*Passes everything and gets no offer, or gets down-leveled.* This is a scope problem, not a knowledge problem. Every story lives inside one service. Fix: find and rehearse two stories where your decisions bound people you did not manage.

The meta-move: **ask for feedback every single time**, and specifically ask "was there a round where I was weakest?" Companies frequently decline, but roughly one in three recruiters will tell you something specific, and three data points localize the bug faster than ten more loops.

### Drill: I'm going to hand you three job descriptions cold. Predict the loop for each. How do you practice that, and what's a passing standard?

**🏋 Drill — JD-to-loop prediction, 15 minutes total, unaided.** Pull three real job descriptions from your actual target list: one AI-product company, one big-tech applied req, one that is deliberately ambiguous (a title that says "AI Engineer" at a company you cannot immediately classify). Give yourself five minutes each. For each, write on one page:

1. The archetype tag, with the two JD phrases that decided it.
2. Which side of the model boundary the role sits on (calls an API vs owns weights), with the evidence.
3. The predicted stage list, named, in order.
4. Which stage carries the hidden weight, and why.
5. The one round you are currently weakest at, and the single artifact you would build this week to fix it.
6. Two questions you will ask the recruiter that would falsify your prediction.

**Pass criterion:** all three completed in 15 minutes without looking anything up; every archetype call supported by quoted JD language rather than by the company's reputation; and at least one prediction that you can state you are *unsure* about, with the specific recruiter question that resolves it. If every prediction feels certain, you are pattern-matching on the brand rather than reading the document — that is the failure this drill is designed to catch.

**Extension, and this is the part that compounds:** keep the page. After the recruiter call, mark each prediction right or wrong. After three cycles you will have a calibrated model of your own JD-reading, which is worth more than any single loop's outcome — and it converts "I think their process is X" into "I predicted their process correctly 8 of the last 11 times," which is genuinely how a senior engineer should hold beliefs about anything.

### Last one. Give me your sixty-second positioning statement — a backend engineer walking into an AI Engineer loop.

Sixty seconds is roughly 150 words, and the structure that survives it is: substrate in one sentence, delta in three, evidence with numbers, and a specific reason for being here. What kills it is chronology — nobody wants your career in order.

**🗣 Say this in the room:** "I'm a senior Python backend engineer — distributed systems, high-throughput services, Postgres and Redis at scale, and I'm the person on my team who debugs the concurrency problems nobody else wants. For the last stretch I've been working on the layer above the model. I own a retrieval-and-agent feature end to end: the chunking and hybrid retrieval, the tool contracts, the structured output validation, and — the part I care most about — the eval harness that gates its deploys, because before we had it we were shipping regressions we couldn't see. I took cost per resolved request from about three cents to under one, mostly through prefix caching and routing the easy 70% to a smaller model, with an eval proving quality held. I'm here because [specific, company-particular reason], and the round I'd most like to be pushed on is evaluation."

Three things that sentence is doing deliberately. It establishes that the systems depth is *real* and not a hedge, because that depth is a genuine advantage in these loops and underselling it is common. It puts evaluation at the center, which is the competency this market most consistently under-supplies. And it invites a specific probe, which is a confidence signal and also lets you steer the conversation toward your strongest ground.

**⚠ Trap:** Apologizing for the transition. "I don't have a formal ML background, but…" is a sentence that costs you a level. You are not a deficient ML engineer; you are a systems engineer entering a discipline that is currently 80% systems work and desperately short of people who can make a nondeterministic dependency behave in production. State it that way, because it is true.
