# PART XVII — Human Rounds

At frontier labs the values and customer rounds reject more candidates than the coding rounds do, and the FDE decomposition case is the lowest-pass-rate stage in the industry. All three are learnable skills with drills, not soft-skill filler.

## Contents

1. [84. Behavioral, Mission Alignment and Safety Reasoning](#84-behavioral-mission-alignment-and-safety-reasoning) — 46 questions
2. [85. Forward-Deployed Engineering: Decomposition Cases and Customer Simulation](#85-forward-deployed-engineering-decomposition-cases-and-customer-simulation) — 47 questions
3. [86. Research Literacy, the Paper Canon and Working With Research Teams](#86-research-literacy-the-paper-canon-and-working-with-research-teams) — 42 questions


---

## 84. Behavioral, Mission Alignment and Safety Reasoning

*Mastering this proves you can pass the round that reportedly rejects the most candidates at Anthropic — and that generic STAR prep actively hurts.*

### Why does the behavioral round at a frontier lab reject more candidates than the coding round, when at most big-tech companies it's a formality?

Start from the economics of the filter rather than from anything mystical about "culture." A technical screen measures a skill that is cheap to acquire and increasingly cheap to fake. Between 2023 and 2026 the marginal cost of looking like a competent coder in a 60-minute screen collapsed — public question banks, AI assistants, and the fact that most AI-engineering work is now composition rather than invention. So the screen's discriminative power fell. Meanwhile the cost of a bad hire at a lab went *up*, because these engineers make unilateral judgment calls about what gets shipped to millions of people with no deterministic test that catches a bad call. When one signal gets noisier and the other gets more consequential, weight migrates. That migration is the entire story.

The mechanism inside the round: at a frontier lab the behavioral interviewer is trying to answer a question a coding round structurally cannot answer — *when this person is alone at 11pm with a deadline and a model doing something subtly wrong, what do they do?* They probe for whether your judgment is load-bearing or decorative. A candidate who says "I'd escalate to my manager" for every hard case is telling them the judgment is decorative. A candidate who has three stories where they held a line, and one where they held the wrong line and can say so, is telling them it's load-bearing.

Anthropic is the most-cited example: the values round is widely reported by candidates as the primary filter, and the loop is short enough (~19-day reported median) that there is no long tail of extra technical signal to rescue you. OpenAI runs a mission-alignment conversation. DeepMind runs GCA/"Googleyness." Founder rounds at Perplexity, Cursor and Scale are the same filter with less process around it.

**⚠ Trap:** treating this as the "easy round" and allocating prep time proportional to how technical it feels. I have watched strong engineers spend eight weeks on attention internals and forty minutes on this, then get rejected with feedback that says "technically strong, not enough evidence of ownership." That feedback is not a euphemism for anything. It is literal.

**📐 Numbers you must know:** budget your prep by rejection probability, not by discomfort. If a five-round loop rejects at the values round at a rate comparable to the coding round — and multiple public candidate reports put it there or higher at Anthropic — then the correct allocation is *at least* 1/5 of your prep, which for a 10-week plan is two full weeks. Most candidates spend under a day. **📅 Volatile:** pass-rate figures, median-timeline figures and the loop structure itself (round names, ordering, how many stages) are candidate-reported and drift; verify against recent Blind/Levels threads and with your recruiter before your loop rather than quoting a number in the room.

**🗣 Say this in the room:** "I've assumed this round is where the decision actually gets made, so I came with specifics rather than themes — happy to go as deep as you want on any of them, including the ones that went badly."

### My recruiter told me Anthropic's round is a "values interview." What is actually being scored in there, and how is it different from a culture-fit chat?

A culture-fit chat asks "would I enjoy sitting next to this person." A values round asks "does this person's default behavior, under pressure and without supervision, produce outcomes we'd endorse." Those are different questions and the second one is answerable from evidence. The interviewer is not looking for enthusiasm about the mission; they are looking for *costly signal* — moments where your stated values cost you something, because uncostly values are unfalsifiable.

Mechanically, most values rounds decompose into four probes, and you should be able to name which one you're in at any moment. First, **ownership under ambiguity**: did you define the problem, or was it handed to you? Second, **honest self-assessment**: can you describe a decision you got wrong without either defending it or performing contrition? Third, **judgment about harm and risk**: when you shipped something with a known failure mode, what did you do about it, and did you tell anyone? Fourth, **communication under disagreement**: what happens when you and a stakeholder want different things.

The scoring is comparative and evidence-weighted. A story that ends "and so I raised it with the team and we decided to hold the launch" scores far above "I believe strongly in responsible deployment," because the first has a decision, a cost, and a counterfactual. Interviewers are trained — informally at least — to keep asking "and then what happened?" until they hit either a specific consequence or a wall. The wall is the signal.

**⚠ Trap:** answering values questions with values. If you're asked "tell me about a time you had to make a call with incomplete information" and you answer with a philosophy of decision-making, you have failed the question and you will not be told. Answer with a Tuesday. A date, a system, a number, a person who disagreed.

**⚠ Trap #2 — the performative-safety tell.** Candidates who have read the company's blog and nothing else produce answers that sound like the blog. Interviewers at labs hear forty of those a week and it reads as flattery, not alignment. Your position is more credible when it contains something the company would mildly disagree with, defended calmly.

**🗣 Say this in the room:** "I'll give you the concrete version rather than the principle, because I think the principle only means something if it cost me something — this one cost us a three-week slip."

### You said generic STAR prep actively hurts here. Why? What should replace it?

STAR is a compression format designed for a hiring process that has to compare 400 candidates on the same rubric with interviewers who have twenty minutes and no domain context. It optimizes for *legibility*. Frontier-lab and founder rounds optimize for *depth*, and those objectives conflict: the STAR discipline of "keep Situation short, land on Result" trains you to summarize away exactly the middle section — the part where you were confused, chose between three bad options, and picked one — that this round is trying to inspect.

Concretely, here is what a STAR-drilled candidate sounds like: forty seconds of context, twenty seconds of "my task was to reduce latency," thirty seconds of "I profiled, found the bottleneck, added caching," and a result. Total 90 seconds, perfectly structured, and utterly unfollowable — the interviewer cannot ask a real question because there is no surface area. Then they ask "what was the hardest trade-off?" and you improvise, badly, because you rehearsed the summary and not the substance.

The replacement I use is what I'd call **STAR-D: Situation, Tension, Action, Result, Delta.** The Tension replaces "Task" — instead of what you were assigned, state the *conflict*: two things you couldn't have at once, the constraint that made the obvious answer wrong. The Delta replaces the throwaway "and I learned a lot": what you would do differently and what specifically changed in how you work. Tension makes the story interesting; Delta makes it honest. Both are the parts an interviewer will actually dig into.

Length calibration matters too. Aim for a **90-second opening** that ends on a hook you *want* them to pull, then let them pull. If they don't pull, you offer: "the interesting part was actually the part where I was wrong about the cause for two days — want that?" You are steering toward depth on purpose.

**⚠ Trap:** the over-rehearsed story. There's a specific vocal texture to a story told for the ninth time in the same word order, and experienced interviewers hear it instantly and discount everything in it. Rehearse the *facts* — the numbers, the names, the dates, the decision points — not the sentences. Practice telling each story three different lengths (30 seconds, 2 minutes, 8 minutes) so the phrasing has to regenerate each time.

**🏋 Drill:** take your single best project. Set a timer for 90 seconds and tell it out loud ending on a deliberate hook. Then set 6 minutes and tell it again with a completely different entry point (start from the incident, not the design). Pass criterion: both versions contain at least three numbers and at least one thing you got wrong, and no sentence appears in both.

### Walk me through how you'd build a story bank for these rounds. How many stories, and what has to be in each one?

The mental model: you are not preparing answers, you are preparing a **small indexed corpus with high coverage**, and the interview is a retrieval problem. Roughly 90% of behavioral questions across every one of these loops map onto about eight underlying situations. If you have eight well-instrumented stories, every question becomes "which story, and from which angle," which is a far easier live task than composition.

Here is the coverage set I'd insist on — eight slots, each of which must carry numbers:

1. **The hardest technical thing you shipped.** Needs a mechanism you can whiteboard and a measured outcome.
2. **A latency or cost win with the arithmetic.** "p99 from 4.2s to 900ms by moving the embedding call off the request path and batching at 32" — the shape matters more than the specific numbers, but the numbers must be yours and real.
3. **An eval you built.** What it measured, how many labelled cases, what decision it changed. This is the highest-leverage story for an AI role and most candidates don't have it — which is precisely why building one is worth three weeks of your prep.
4. **An incident you owned.** Detection, mitigation, root cause, and the *specific test or eval case added so it cannot recur*. That last clause is the part that scores.
5. **A system you argued against building.** Costly signal, ownership, and product judgment in one artifact.
6. **A disagreement with someone more senior.** Ideally one where you lost and executed well anyway, and one where you won.
7. **Something you shipped that you later concluded was wrong.** Not a fake weakness. A real call.
8. **Something you built with genuine ambiguity** — no spec, no requester, you decided it mattered.

Each entry in the bank should be a written index card with: the two-sentence situation, the tension in one sentence, **three numbers**, the name of the person who disagreed with you, one thing you got wrong, and the delta. That's it — resist writing prose, because prose is what you'll end up reciting.

**💰 Math on where the numbers come from:** you do not need permission to reconstruct metrics you legitimately knew. If your service ran 1.2M requests/day and you cut mean latency 300ms, that's 1.2e6 × 0.3s = 360,000 seconds = 100 engine-hours of latency removed per day; if it let you drop from 12 pods to 9 at ~$140/pod/month, that's 3 × 140 = $420/month = $5,040/year. Small money — say so, and say what it *unblocked*, which is usually the real result. What kills candidates is not small numbers; it's no numbers.

**⚠ Trap:** inflating. Every senior interviewer has a probe for it, covered later in this section, and it is trivially effective — they ask about the denominators. "40% faster" invites "faster than what baseline, measured over what window, with what variance." If your number cannot survive three levels of denominator questions, downgrade it to something that can. A defensible 15% beats an indefensible 60% by a wide margin.

### Tell me about the most technically difficult thing you've built.

This is the single most-asked opening in every loop on your target list, and it is not a warm-up — it sets the interviewer's prior for the next 45 minutes. The mental model: they are calibrating your *ceiling*, and the ceiling is set by the hardest thing you can explain at three levels of depth, not by the biggest thing you were adjacent to.

The structural move that works: **pick the story where the difficulty was epistemic, not just laborious.** "We migrated 400 endpoints" is large but not hard. "The p99 was 40× the p50 and every metric we had said the system was healthy" is hard, because the difficulty was in *not knowing*, and every follow-up question then has a real answer.

Here is the skeleton, with the shape of the numbers you should substitute your own into. Situation in two sentences, including scale ("~8k req/min, 40-odd tenants, hard 2s SLO"). Tension in one ("the obvious fix — bigger pool — made it worse, which meant my model of the failure was wrong"). Then the *investigation*, which is where you spend your time: the hypothesis you formed, the measurement that killed it, the second hypothesis, the instrumentation you had to add because you couldn't see it. Then the fix, then the number, then the delta.

For an AI-role loop, if you have an AI system that qualifies, lead with it — but if your genuinely hardest work is backend, **lead with the backend one and land the plane on transfer**. Something like: "the reason I tell that one is that the debugging shape carried over directly — when we started running an LLM pipeline, the same problem appeared as a tail of 30-second requests that our success-rate dashboard was blind to, because a wrong answer returns 200." That sentence does more for you than a mediocre AI story would.

**⚠ Trap:** choosing the story with the biggest headline number rather than the one you remember at the tensor/packet/scheduler level. You will be interrupted with "why did that work?" and the honest answer to that question is what's being scored. I would rather hear a beautifully-understood 12% than a hand-wavy 10×.

**🗣 Say this in the room:** "The hard part wasn't the fix — the fix was forty lines. The hard part was that for two days every dashboard we had said the system was fine, and I want to walk you through why they were all lying."

### What is OpenAI's mission-alignment round actually testing, and how do you prepare for it without sounding like a press release?

The mental model: mission-alignment is not a loyalty test, it's a **stability test**. They are checking whether your stated motivation is load-bearing enough to still be there in eighteen months when the work is unglamorous, or whether you're momentum-chasing. Momentum-chasers are expensive: they onboard for four months and leave when the next thing gets hot.

So the round probes for *derivation*. Not "do you believe AGI will be transformative" — everyone says yes — but "what did you do about it before anyone was paying you to." Anything you built, wrote, read, or changed your mind about *at cost* is evidence. A 1,600-question technical guide written on nights and weekends is evidence. Reading three papers because a production system confused you is evidence. Having a considered, non-default opinion about a deployment question is evidence.

The preparation is not memorization; it's construction. Write down, in your own words and on paper, three things: (a) the specific reason you moved from backend to this work, phrased as a problem you find interesting rather than a market you find lucrative; (b) one thing about the current trajectory of the field you find genuinely concerning, at a level of specificity that shows you thought about it rather than absorbed it; (c) one thing you think the consensus gets wrong. Then stress-test each against "why?" three times. If any of them bottoms out in "because that's what people say," it isn't yours yet.

**⚠ Trap:** the compensation-shaped answer wearing a mission costume. Interviewers at labs are extremely well-calibrated on this. It is entirely fine — and more credible — to acknowledge that the work is also the highest-leverage engineering on the planet right now and that this is part of the appeal. What is not fine is pretending money and impact aren't both operating.

**🗣 Say this in the room:** "The honest version is that I got here from the infrastructure side — I was debugging a retrieval system that was confidently wrong, and I realized I couldn't tell whether the failure was in my chunking or in the model, and I didn't like not being able to tell. That's a fairly narrow reason to change careers, but it's the real one, and it's why I've spent the last year on evaluation more than on prompting."

### How is DeepMind's "Googleyness"/GCA round different from a lab values round, and does the same prep transfer?

Different objective, overlapping evidence. Googleyness and General Cognitive Ability are Google-lineage constructs and they are testing for *scalable collaboration inside a large matrixed org*: comfort with ambiguity, intellectual humility, bias to action, and the ability to make progress without authority. A frontier-lab values round is testing whether your judgment about consequences is sound. Both want ownership stories; they want different endings.

Concretely: for a lab values round, the strongest ending to a disagreement story is often "I escalated it and we didn't ship." For a Googleyness round, the strongest ending is often "I found the three people who owned the pieces, got them in a room, and we shipped a scoped version in two weeks." The first rewards a held line; the second rewards navigated complexity. Same story, different emphasis — which is exactly why the story bank should store facts rather than scripts.

GCA also has a genuine cognitive component: hypothetical, ambiguous problems ("how would you estimate X," "how would you decide between A and B with no data"). Treat those exactly like a system-design round — clarify, state assumptions out loud, decompose, quantify, name what you'd measure to know you were right. The failure mode is answering the hypothetical as if it were a quiz with one answer rather than a thinking-out-loud exercise.

**🗣 Say this in the room** (for a Googleyness ambiguity question): "Let me state my assumptions first so you can correct the ones that are wrong, because the answer changes a lot depending on whether we control the data source."

**⚠ Trap:** importing the frontier-lab register wholesale. A safety-flavored answer to "tell me about a time you influenced without authority" reads as evasive at DeepMind's GCA round even though it would score at Anthropic. Read the room's objective, not just the company's logo.

### Founder rounds at Perplexity, Cursor and Scale — what's actually being scored, and how do you prep for a round with no rubric?

The mental model: a founder round is a **conviction and taste interview** conducted by someone with total hiring authority and no obligation to be consistent. There is no rubric because the founder is the rubric. What they are almost always sampling for is three things: do you have opinions about the product, are you fast, and would you be miserable in an environment with no process.

Product opinions are the highest-yield preparation and the most commonly skipped. Before a Cursor round, use Cursor for two weeks — heavily — and come with one thing you love, one thing that's broken, and a hypothesis about why the broken thing is hard. Before Perplexity, run twenty real queries and be able to say something specific about where the retrieval fails and what you'd measure. Cursor's onsite is widely reported to include a paid multi-hour build project with "do you actually use AI coding tools" as a hidden rubric line; showing up as a non-user is disqualifying in a way no one will tell you. **📅 Volatile:** onsite formats at fast-growing startups change every couple of quarters — treat this as a reported pattern and confirm the current shape with your recruiter rather than asserting it.

Speed is sampled through how quickly you converge. Founders read long preamble as slowness. Give the answer first, then the reasoning — the inverted-pyramid discipline. If they ask "would you build this on top of an off-the-shelf agent framework?", say "no, and here's the one condition that would change my mind" rather than a two-minute survey of options.

The environment question is real and you should answer it honestly: if you need a spec, a design review and a QA gate to be effective, a 40-person AI product company will be unpleasant for both parties. Have a real answer for "how do you work when nobody tells you what to do."

**⚠ Trap:** treating a founder round as the "chat" at the end of the loop and coasting. Multiple companies on your target list have the founder as a hard veto after a clean technical loop. Prep it like a technical round.

**🗣 Say this in the room:** "I've been using it daily for three weeks. The thing I'd change first is [specific]. I don't know your constraints, so I might be wrong about why it is the way it is — but that's the first thing I'd want to dig into."

### What is Anthropic's customer-conversation simulation, and what is the rubric?

Format first, because candidates are blindsided by it: an interviewer role-plays a customer or internal stakeholder — typically non-technical or semi-technical — and you have a live conversation, usually 30–45 minutes, about a real-ish use case. It is reported to eliminate a large fraction of candidates who cleared the coding stages, and the reason is that it tests a skill nothing else in the loop touches: **can you be technically honest and commercially useful at the same time, in real time.** **📅 Volatile:** exact format, length and stage ordering shift; confirm with your recruiter, and it is entirely fine to ask them "what does the simulation round look like and who plays the customer?"

The rubric, as far as it can be reconstructed from candidate reports and from what the role actually requires, has four lines. **Discovery** — do you find out what they're actually trying to do before proposing anything? **Honesty under commercial pressure** — when they ask for something the technology can't reliably do, do you say so plainly, without either capitulating or lecturing? **Translation** — can you explain a model limitation without jargon and without condescension? **Forward motion** — does the conversation end with a concrete next step the customer would agree to?

The single most common failure is jumping to architecture. The customer says "we want to automate claims triage" and the candidate starts describing a retrieval pipeline. You have now proposed a solution to a problem you have not defined, and every subsequent minute compounds the error. The correct first move is always questions: what happens today, who does it, how many per day, what does a mistake cost, how would you know if it worked.

The second most common failure is capitulation. The role-play customer will push for something unwise — "can it just make the decision automatically?" — and a candidate who says "sure, we can do that" has failed the round even if the technical answer that follows is excellent.

**🗣 Say this in the room:** "Before I say anything about how we'd build it, can I ask what happens today when one of these comes in — who touches it, and what does it cost you when one goes wrong?"

**🏋 Drill:** have a friend play a non-technical operations director who wants an LLM to approve refunds under $500 automatically. 20 minutes. Pass criterion: you asked at least six questions before proposing anything, you explicitly stated one thing the system will get wrong and how often, and you ended with a named next step that involves a measurement rather than a build.

### How do I explain moving from backend engineering into AI without it sounding like résumé-driven development?

The mental model: hiring managers are not worried that you're chasing a hot field — everyone is. They're worried about **two specific risks**: that you'll discover you dislike the actual work (which is 70% evaluation, data plumbing and failure analysis, not model design), and that your depth is a veneer over a weekend of tutorials. Your answer should retire both risks explicitly.

The structure that works is: a concrete trigger, a costly investment, and an accurate picture of the work. Trigger — a specific moment in a system you owned where the LLM layer became the interesting part of the problem. Costly investment — what you did about it that took real time and produced an artifact someone else can inspect. Accurate picture — say out loud that the job is mostly evals and error analysis, which proves you've actually done it.

Then convert the backend background from a liability into the thesis. This is the strongest available frame and most career-switchers underuse it: the hard part of shipping AI products in 2026 is not model knowledge, it's that these systems are *nondeterministic distributed systems with no unit tests and an unbounded input space*. The people who are good at that are people who are good at observability, idempotency, backpressure, and failure taxonomies. That is a backend skill set applied to a new failure domain.

**⚠ Trap:** the apologetic frame — "I know I don't have an ML background, but..." Never open a sentence with your gap. State the gap once, factually, in the place where it's relevant ("I haven't trained a model from scratch and I wouldn't claim to; the closest is a LoRA run on 8k examples"), then move on. Confidence about a stated limitation reads as calibration; anxiety about it reads as a bigger gap than it is.

**🗣 Say this in the room:** "I moved because the interesting failures in my systems stopped being on my side of the API. A retrieval service returning 200 with a confidently wrong answer isn't a class of bug my monitoring stack understood, and I wanted to be on the side of that problem where you can actually fix it."

### What's the difference between ownership under genuine ambiguity and execution against a spec, and how do I tell which of my stories is which?

This is the single highest-leverage distinction in the whole section, so let me make it mechanical rather than vibes-based. A story is an **execution** story if someone else could have written the ticket. It's an **ownership** story if the hardest part was deciding *what the ticket was*. Interviewers at this tier are almost exclusively hiring for the second, because the first is what they expect from a strong mid-level engineer and they are trying to distinguish IC5 from IC6.

The test I apply to my own stories, and to candidates': **remove the protagonist and ask whether the outcome still happens.** If a competent engineer assigned the same JIRA ticket would have produced roughly the same result, it's execution. If the project would not exist — because nobody had noticed the problem, or everyone had noticed and nobody had framed it — it's ownership. A second test: **who did you have to convince, and what did they say?** Ownership stories always have a skeptic in them. Execution stories have a reviewer.

Concretely, the difference sounds like this. Execution: "We needed to cut inference cost, so I implemented prompt caching and semantic dedup and got cost per request down 62%." Ownership: "Nobody was tracking cost per *resolved task*, only cost per call, so the dashboard looked fine while our unit economics got worse every week as retries went up. I built the per-task metric first, which is what showed us that 30% of spend was on conversations that never resolved — the caching work came second and was the smaller half."

Notice the second version contains a *reframing*. That is the tell. Senior ownership almost always involves changing the denominator someone was measuring.

**⚠ Trap:** overclaiming ownership on a team project. Do not say "I decided"; say "I proposed X, Y pushed back on latency, we compromised at Z, and I owned the implementation and the rollback plan." Specific shared credit reads as more senior than sole credit, and it survives the reference check.

**🗣 Say this in the room:** "The part I'd claim credit for is the framing, not the build — the build was two weeks and any of us could have done it. What took the longest was convincing the team we were measuring the wrong thing."

### How much of this round can I actually prepare, given that the questions are unpredictable?

More than you'd think, because the questions are only superficially unpredictable. The mental model: the question space is large but the *answer* space is your eight stories plus three positions. Preparation is building the corpus and then practicing retrieval, not enumerating questions.

Here's the concrete allocation I'd defend for a two-week run-up. **Days 1–3: the story bank.** Write eight index cards, facts only, three numbers each. This is the bulk of the work and it is genuinely hard because reconstructing real numbers takes digging through old dashboards, PRs and Slack. Do the digging. **Days 4–5: the positions.** Write out your actual view on deployment risk, on open weights, and on where you'd draw a line, and stress-test each to three follow-ups. **Days 6–7: company research** — their published research, their product decisions, and one specific thing you'd ask about. **Days 8–10: live reps.** Out loud, with a person if possible, recording yourself if not. **Days 11–14: the simulation and the drills**, including the customer role-play and the "what did you build vs the framework" probe.

The reps are the part everyone skips and they're what converts. There is a large gap between a story you *know* and a story you can *tell* under mild adversarial pressure at 11pm IST after four hours of technical rounds. **📐 Numbers you must know:** for a US-timezone loop from IST you will likely be interviewing between 21:00 and 02:00 local; a 5-round onsite is 5–6 hours, meaning your values round lands at roughly hour four, near 01:00. Rehearse at that hour at least twice. Energy management is not a soft variable here — the round that rejects most candidates is scheduled when you are worst.

**🏋 Drill:** record yourself answering "tell me about a time you were wrong" cold, with no preparation, once per day for five days. Play back only the first 20 seconds of each. Pass criterion: by day five the first 20 seconds contain a specific system, a specific date-ish anchor, and no throat-clearing ("that's a great question," "let me think," "so basically").
### Tell me about an LLM feature you shipped. I want the numbers.

The mental model for constructing this answer: an AI-feature story is graded on whether you had a **closed loop** — a way to know whether the thing worked that existed *before* you shipped it. Anyone can describe a feature. The senior signal is that you can state, in order, the baseline, the intervention, the measurement, and the delta, and that the measurement wasn't invented after the fact to justify the launch.

Here is the shape, with example arithmetic you must replace with your own real figures. "We had a support-answer feature where a first-pass retrieval-plus-generation pipeline was resolving about 41% of tickets without escalation, measured on a rolling 7-day window against human-labelled resolution. Two problems: p95 was 6.4s against a 3s target, and cost was $0.19 per answered ticket at roughly 28k tickets/day, so about $5,300/day, $160k/month. I did three things in priority order: moved the 11k-token system prompt behind a provider prefix cache, which at $3/Mtok uncached versus $0.30/Mtok cached takes that portion from 11,000/1e6 × 3 = $0.033 to $0.0033 per call — call it $0.03 saved per call, so 28k × $0.03 = $840/day gross, though our real cache-hit rate was 78%, so ~$655/day; second, moved reranking off the critical path for the 60% of queries where the top-1 BM25 score cleared a threshold, which took p95 from 6.4s to 2.9s; third, added 140 labelled cases to the eval set from the escalation logs, which is what caught that the reranker skip was costing us 3 points of resolution on multi-part questions, so I gated it to single-intent queries." **📅 Volatile:** the $3/Mtok-uncached and $0.30/Mtok-cached figures are illustrative provider prices, and cache-read discounts and cache-write surcharges differ by vendor and generation — recompute with your provider's current rate card rather than quoting these.

Notice the structure: three interventions, each with a number, and **one of them partially backfired and I found it because of the eval**. That last clause is worth more than the other two combined. It proves the loop was closed.

**⚠ Trap:** reporting cost per call instead of cost per resolved task. At this tier the interviewer will convert for you and it is much better if you've already done it. Cost per call goes *down* when you route to a cheaper model; cost per resolved task can go *up* at the same time, because the cheap model retries. State the denominator you're using and why.

**💰 Math:** the retry amplification is worth internalizing. If a cheap model at $0.30/Mtok input resolves 62% first-pass and you retry once on failure, expected calls per resolution = 1/0.62 = 1.61, plus the failed-attempt tokens are pure waste. A model at 4× the price resolving 88% needs 1/0.88 = 1.14 calls. Cost ratio is 4 × 1.14 / 1.61 = 2.83×, so the expensive model is still 2.8× the spend — but if each escalation to a human costs $4.20 in agent time, the escalation gap swamps it. Be careful to compute that gap *after* the retry, not from the first-pass rates: on the same retry-once policy the cheap model still escalates 0.38² ≈ 14.4% of tickets against 0.12² ≈ 1.4% for the expensive one (that squaring assumes the retry fails independently, which is optimistic — correlated failures make the cheap model look worse), so the ~13-point gap on 28k tickets/day is 0.13 × 28,000 × $4.20 ≈ $15,300/day. The model price difference is noise against that. This is the arithmetic that makes you sound like you've shipped.

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

If you haven't fine-tuned anything: say so directly and then say what you'd do, with the same specificity. "I haven't run one in production. If I had to, the trigger would be a behavior I couldn't get with prompting plus few-shot plus structured decoding, and my first move would be LoRA rather than full fine-tuning because of the memory arithmetic — full fine-tuning a 7B in bf16 needs weights plus gradients plus two Adam moments in fp32, so roughly 14 + 14 + 56 = 84GB before activations — and most mixed-precision trainers also keep an fp32 master copy of the weights, another 28GB, which is where the familiar ~16 bytes/parameter ≈ 112GB figure comes from — either way it doesn't fit an 80GB card, whereas LoRA at rank 16 trains well under 1% of parameters." That answer scores better than a vague claim of experience, because it demonstrates the model that makes the decision.

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

The quantification is mandatory. "Roughly 9% of requests for 4 hours 20 minutes — about 3,100 affected sessions at our 8k/hour rate — and because the failure returned 200 with a plausible answer, none of our alerts fired; we found it from two support tickets." That sentence is worth more than the whole rest of the story, because it names the reason this domain is hard: **wrong is not an error code.**

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
### Do you think AI is dangerous?

The mental model for this entire family of questions: you are not being asked for a position, you are being asked whether you *have* one — meaning something you arrived at, can defend, and would apply to a decision on Tuesday. The tell for a manufactured position is that it doesn't constrain anything. If your view on AI risk has never made you do or refuse to do something concrete, it isn't a view, it's a vibe, and three follow-ups will expose that.

So build the answer from the bottom up: from risks you have personally observed, then out to the ones you haven't. Concretely, there are at least four distinguishable categories and being able to name them separately is most of the signal. **Near-term deployment harms** — a system confidently wrong in a consequential domain, an automation that removes the human who would have caught the error, differential failure rates across user groups you never sliced for. **Misuse** — the capability is working as designed and the user is the threat: fraud, targeted manipulation, cyber-offense uplift, CSAM and NCII, bio and chem uplift at the frontier. **Structural/societal** — labor displacement, epistemic pollution, concentration of capability. **Loss-of-control** — the long-horizon concern about highly capable systems pursuing objectives we didn't intend and cannot correct.

The honest thing for an applied engineer to say is that your direct evidence is overwhelmingly in the first two categories, that you take the third seriously as an economic matter, and that on the fourth you have a considered view of the *argument* even though nothing you've shipped bears on it. That calibration — being clear about where your evidence is strong and where you're reasoning from others' work — is worth more than confident agreement with whatever you think the interviewer believes.

**⚠ Trap:** the mirror. Candidates at Anthropic answer with x-risk framing and candidates at a fast-moving product company answer with "I think the risks are overblown," and both are transparently reflecting the perceived house view. Interviewers at labs specifically probe for this by pushing *against* their own position to see if you fold. A view that moves under mild pressure is scored as no view at all.

**⚠ Trap #2:** treating "dangerous" as a yes/no. The strongest answers immediately disaggregate — "dangerous how, and on what timescale?" — because the interesting disagreements are all between the categories, not about the total.

**🗣 Say this in the room:** "Yes, but I'd want to separate four things, because I think they have different evidence bases and different mitigations — and I'd say my direct experience is entirely in the first one, which is systems that are confidently wrong in places where somebody acts on the answer."

### Let's push on that. You said deployment harms are your main concern — but if the model is only wrong 2% of the time and humans are wrong 5%, isn't deploying it strictly better?

This is follow-up two of a standard three-deep ladder, and it's a good argument, which is why you need a real answer rather than a deflection. The mental model: aggregate accuracy comparisons hide three things that determine whether a deployment is actually safer — **the correlation structure of the errors, the distribution of the errors across people, and the recoverability of each error.**

Correlation: human errors are approximately independent — two adjusters make different mistakes on different claims. A model's errors are systematic and correlated: it fails the same way on every instance of a pattern, at machine scale, simultaneously. A 2% independent error rate and a 2% correlated error rate are very different risk objects. The second one produces a single class-action-shaped incident; the first produces noise.

Distribution: a 2% aggregate can be 0.5% on the majority slice and 14% on a minority slice, and aggregate metrics are precisely the instrument that hides it. This is why I insist on sliced reporting as a release gate rather than as an afterthought — the question I ask in review is not "what's the accuracy" but "what's the worst slice with more than 100 cases in it."

Recoverability: an error that produces a draft a human rejects costs a minute. An error that issues a payment, deletes a record, sends an email or files a document costs an incident. So the honest form of the comparison is not "2% vs 5%" but expected *harm* per unit, and harm is error rate × blast radius × 1/(probability someone catches it).

**💰 Math:** take 40,000 decisions/month. Human at 5% with independent errors and a supervisor catching 60%: 40,000 × 0.05 × 0.4 = 800 uncaught. Model at 2% but errors correlated into ~6 systematic patterns, with automation bias dropping human catch rate to 20% because reviewers rubber-stamp fluent output: 40,000 × 0.02 × 0.8 = 640 uncaught — genuinely better on the count, but concentrated in six failure modes, so the tail risk is a single 640-case class of identical wrong outcomes rather than 800 scattered ones. Whether that's better depends entirely on whether the harm function is linear. For refunds, it's linear and the model wins. For eligibility denials, it isn't, and it doesn't.

**⚠ Trap:** automation bias is the number people forget. Human review of fluent machine output is measurably weaker than human review of human output — reviewers approve at a higher rate and spend less time. If your safety case depends on "a human checks it," your safety case has to include how you keep the human's attention, e.g. by surfacing uncertainty, injecting known-bad cases as an attention audit, or requiring the reviewer to fill in the rationale rather than click approve.

**🗣 Say this in the room:** "The aggregate comparison isn't the right one, because model errors are correlated and human errors mostly aren't — a 2% correlated error rate is one incident with 800 instances, not 800 independent small ones."

### Third push: doesn't all of this just mean you'd never ship anything? Give me a case where you'd ship despite known risk.

Yes — and if you can't, you're not employable in an applied role, so answer this one decisively. The mental model: the goal is not zero risk, it's **bounded, measured, and recoverable risk with someone accountable for the bound.** Safety work that only ever says no is indistinguishable from an absence of engineering.

The frame I use has four questions, and I'll answer them for a real case. *Is the worst case bounded?* *Is it reversible?* *Will we detect it?* *Is there a kill switch and who can pull it?* If all four are yes, ship, and ship faster than the risk-averse instinct wants to.

Case: an internal support-answer assistant, retrieval-grounded, over our own docs, shown to agents rather than customers. Worst case is bounded — an agent gets a wrong suggestion and either catches it or relays a wrong answer that a customer disputes. Reversible — yes, we can correct and re-contact. Detectable — yes, if we instrument agent edit-rate and thumbs-down and sample 50 conversations a week. Kill switch — a feature flag any on-call can flip, tested in a game day. That ships on day one, at 10% of agents, with the eval running. I would not wait for a perfect eval before that launch, because the fastest way to build the eval is from the first two weeks of real usage.

Compare to the same system answering customers directly with no human in the loop and the ability to state policy commitments. Now the worst case is unbounded (we said something contractually binding), partially irreversible, and detection is much worse because the customer just leaves. Same model, same prompt, different deployment surface, and the second one needs a materially different bar.

That distinction — **the risk lives in the deployment surface, not in the model** — is the single most useful thing you can say in a safety conversation as an applied engineer, and it is what distinguishes you from someone reciting principles.

**🗣 Say this in the room:** "I'd ship it — internally, to 10% of agents, behind a flag, with an edit-rate metric and a weekly sample. The thing I wouldn't do without a lot more evidence is put the same system in front of customers with the authority to make commitments, and that's a deployment-surface decision, not a model decision."

### Walk me through the deployment risk for a specific feature: we want an agent that can read a user's inbox and send emails on their behalf.

Good — this is the format that actually appears, and the winning behavior is to reason aloud in a structured way rather than to produce a list of principles. My structure for any capability review is five passes: **capability boundary, adversary model, blast radius, detection, and the graduated rollout.**

*Capability boundary.* Read and send are wildly different. I'd split them immediately: read-only summarization and draft-generation is a different product with a different risk profile from autonomous send. The first thing I'd propose is that v1 drafts and never sends, and that "send" is a separate capability gated behind explicit per-message confirmation, with autonomous send as a v3 conversation that requires evidence we won't have for months.

*Adversary model.* This is the part most candidates miss and it's the highest-signal thing you can raise: **the inbox is attacker-controlled input.** Anyone can email your user. If the agent reads email content and that content enters the model's context alongside its instructions, you have a prompt-injection surface with a built-in exfiltration channel — the send capability. The canonical attack is an email containing text that instructs the agent to forward the last 20 messages to an external address. This is not hypothetical; indirect prompt injection through retrieved content is the central unsolved problem in agent security, and I'd say plainly that there is no known complete defense.

*Mitigations that actually reduce it*, in order of value: never let untrusted content and privileged actions share an unconstrained loop — separate the "read and summarize" model call from the "decide to act" call and pass only structured, validated fields between them; allowlist recipients to addresses already in the user's sent history for autonomous sends; strip or neutralize instruction-looking content from retrieved bodies; require a human confirmation for any send to a new domain or with an attachment; rate-limit sends per session; log every action with the full context that produced it.

*Blast radius and detection.* Worst case is silent exfiltration of a user's correspondence, which is irreversible and reputationally catastrophic. Detection: alert on sends to novel domains, on bulk forwards, on any action whose triggering context contained imperative language from an untrusted body.

*Rollout.* Internal dogfood, then opt-in beta with confirmation-on-every-send, then relax confirmation only for the allowlisted-recipient case, with a red-team exercise between each stage.

**⚠ Trap:** answering this with content-safety framing (toxicity, PII redaction) when the dominant risk is injection and exfiltration. Content filters do essentially nothing against an attacker who controls the retrieved text. Naming the right threat model is most of the score.

**🗣 Say this in the room:** "The thing that makes this hard isn't that the model might write a bad email — it's that the inbox is attacker-controlled input and the send tool is an exfiltration channel. I'd design the whole thing around never letting untrusted text and privileged actions live in the same loop."

### A customer asks for something you believe is unsafe. Show me how you handle it live — they want the system to auto-approve insurance claims under $5,000 with no human review.

The mental model for this simulation: your job is not to refuse and it is not to comply. It is to **find the version of what they want that you can actually stand behind, and get them to want that instead.** Refusal loses the deal and the round; compliance loses the round more quietly. The move is discovery, then a reframe, then a concrete alternative with a graduation path.

Sequence it. First, questions, and they should be business questions rather than technical ones: how many claims a day, what's the current cycle time and where is the pain, what does a wrong approval cost you, what does a wrong denial cost you, what's your regulator's position on automated decisions in this line, and what happens today when an adjuster gets one wrong — is there an appeal? You are looking for the asymmetry, and in claims there always is one: wrong approvals cost money and are recoverable-ish; wrong denials cost customers, regulatory exposure and sometimes a lawsuit, and they're often invisible because the customer just goes away.

Then the honest statement, delivered without hedging or apology: "I can build something that gets this to a few seconds instead of two days. What I can't do is build something that's right every time, and the specific reason is that the model will fail in *patterns* rather than at random — so when it's wrong, it'll be wrong the same way on every claim that looks like that one, and you'll find out in a batch."

Then the reframe: split the decision. Auto-approve is one action; the system can do several others safely. Extract and structure the claim, check policy coverage against the document, flag the six things an adjuster looks for, and produce a recommendation with a confidence and a citation to the policy clause. Measure the agreement rate against adjusters on a few hundred historical claims *they* label. Then, if agreement on the under-$500 low-complexity band exceeds a threshold you agree on in advance, auto-approve that band with a sampled audit — and expand the band on evidence.

Then the commitment: "Here's what I'd want to do in the first two weeks — take 200 of your historical claims where you know the outcome, run them, and show you the confusion matrix by claim type. If it's good, you'll have more confidence than my opinion could give you. If it's bad, you've spent two weeks instead of two quarters."

**⚠ Trap:** lecturing. The moment you say "responsible AI requires..." you've lost the room. The customer does not need your ethics; they need your engineering judgment translated into their risk language — dollars, regulators, cycle time, and appeals.

**🗣 Say this in the room:** "I don't want to say no to this, I want to sequence it. Give me the band where a mistake is cheap and recoverable, let me prove it there against your own historical claims, and we expand on evidence rather than on my say-so."

### A PM wants to ship an unevaluated feature on Friday. What do you actually do?

Not what you believe — what you do, in order, with timestamps. The mental model: "I'd push back" is a non-answer. Escalation without a proposal is just obstruction, and interviewers grade on whether you produced a path rather than a veto.

Here's the actual sequence I'd run. **Step one, quantify the unknown rather than assert it.** "We don't have an eval" is weak; "we have no measurement of how often this is wrong, and at 40k requests/day even a 3% failure rate is 1,200 bad outputs a day" is a fact the PM can reason about. Convert the abstraction into their units.

**Step two, offer the fastest possible evidence, and make it cheap.** In four hours you can hand-label 80 stratified cases from staging traffic and get a point estimate with a ±11-point confidence band at n=80 (0.5/√80 ≈ 5.6% standard error, so roughly ±11 at 95%). That's not a release gate, but it does distinguish "90% works" from "60% works," and 60% you do not ship. Say it as: "give me until Thursday noon and I'll tell you whether we're arguing about the last 5% or about whether it works at all."

**Step three, propose the risk-reduced ship** rather than the delay. Ship to 2% of traffic. Ship behind a flag. Ship with the output labelled as a draft. Ship to internal users. Ship without the action-taking part. Almost always there's a version that captures the deadline and caps the exposure, and finding it is the senior contribution.

**Step four, write it down.** One paragraph in the channel or the doc: what we know, what we don't, the specific failure I expect, the metric that would show it, and the rollback. This is not CYA — it converts a personality conflict into a documented decision, and it means the correction later is impersonal.

**Step five, escalate only on an unbounded-and-irreversible risk**, and escalate with the same paragraph, not with feelings. If the feature can commit the company to something, touch money, or affect a protected decision, that's the line where I stop negotiating and go to the person who can accept the risk. Note that escalating is a real cost and using it on a low-stakes disagreement burns the credibility you'll need for a real one.

**⚠ Trap:** treating every unevaluated ship as equally serious. A PM who ships an unevaluated tone change to an internal tool is fine. Reserve the machinery for irreversible or unbounded surfaces, and be visibly relaxed about the rest — that contrast is what makes your objection credible when you do raise it.

**🗣 Say this in the room:** "I almost never block a launch. What I do is convert 'this might be bad' into a number by Thursday, and then propose the smallest version that still hits Friday — 2% of traffic behind a flag is usually the deal."

### Explain to a non-technical VP why the system sometimes makes things up.

Plain language, one mechanism, no jargon, no apology, and — critically — no anthropomorphizing. The mental model to convey: **the system is optimizing for plausible continuation, not for truth, and it has no internal representation of "I don't know this" that's reliably connected to its output.**

Here's a version I'd actually use: "The model works by predicting what text should come next, given everything it's seen. It's extraordinarily good at that. But 'what should come next' and 'what's true' are different targets that happen to agree most of the time, because true things are common in its training data. When they disagree — when a plausible-sounding answer exists and the true one is rare, absent, or specific to us — it produces the plausible one, and it produces it in exactly the same confident tone as everything else. It isn't lying and it isn't guessing in the way a person guesses; it has no mechanism that reliably distinguishes 'I know this' from 'this is the shape of an answer.'"

Then the crucial second half, which is what a VP actually needs: what you do about it. "So we don't try to make it never wrong — we can't. We do three things instead: we give it the source documents so it's summarizing rather than recalling, which cuts this a lot; we require it to cite, so a human can check in five seconds instead of five minutes; and we measure how often it's wrong on a fixed set of questions so we notice when it gets worse. What I can promise you is a *known* error rate on a defined set of tasks, not a zero error rate."

**⚠ Trap:** using the words "hallucination," "temperature," "stochastic," or "token" with a non-technical audience. They're either meaningless or, worse, misleading — "hallucination" implies a malfunction that could be patched, and it sets the expectation that a fix is coming. "It optimizes for plausible, not for true" sets a durable expectation.

**⚠ Trap #2:** over-reassuring. If you say "RAG solves this," you have created a future incident *and* an angry stakeholder. Grounding reduces the rate and changes the failure mode — it doesn't eliminate it, and a model will still misread a retrieved document or answer from memory when retrieval returns nothing useful.

**🗣 Say this in the room:** "It's optimizing for a plausible answer, not a true one, and those usually coincide — the problem is that when they don't, it sounds exactly the same. So we design for a known error rate that a human can check quickly, not for zero."

### The same VP asks why they got two different answers to the same question yesterday and today. Explain nondeterminism.

Short answer first, because that's what they need: "Because the system is designed to sample rather than to look up, and separately, the thing underneath us changes without telling us." Two distinct causes, and conflating them is the common mistake.

Cause one is intentional: at each step the model produces a probability distribution over next words and we draw from it. We can turn that down — set the randomness to zero — and we do for anything structured. It gets us *more* consistent, not perfectly consistent, because floating-point arithmetic on a GPU isn't associative and the result can depend on how requests were batched together, which depends on who else was using the system at that moment. That's an honest and slightly surprising fact and it lands well: **your answer can depend on the batch you were in.**

Cause two is the one with business consequences: the provider updates the model. Same name, different weights, different behavior. Your prompt that worked for six months quietly gets worse. Say this explicitly, because it sets up the ask: "this is why I want the eval suite — it's how we find out that something changed before a customer does."

Then translate to their world: "Think of it as a very experienced contractor rather than a database. Ask the same question twice and you'll get answers that mean the same thing in different words. Where we need exactly-the-same, we don't ask the model — we cache the answer, or we constrain the output to a fixed set of choices, or we have it fill in a form rather than write prose."

**💰 Math worth having ready if they push on "just cache everything":** caching only helps where questions repeat. If your query distribution has 22% of volume in the top 200 queries, exact-match caching those caps out at a 22% hit rate; at $0.012/call and 90k calls/day that's 0.22 × 90,000 × 0.012 = $238/day saved, and the other 78% is still nondeterministic. Consistency has to come from constraints, not from caching.

**🗣 Say this in the room:** "Two reasons — one we chose, and one we don't control. We chose sampling, and we can dial that down. What we don't control is that the vendor can change the model under the same name, which is exactly why I want a fixed set of test questions running daily."

### What's your actual position on releasing open-weight models?

This is a genuinely contested question and the right answer says so. The mental model: nobody has a clean argument here, because the empirical claim on both sides — "open weights meaningfully increase misuse" versus "open weights meaningfully increase defense and scrutiny" — is hard to measure, and most public positions correlate suspiciously well with the speaker's commercial interest. If you present it as settled you either sound naive or sound like you're reciting a corporate line.

My decision rule, which I'd offer as a rule rather than a verdict: **the release question should turn on the marginal uplift relative to what's already available, not on the model's absolute capability.** A model that's strong at coding and reasoning but no better than three existing open models at any dangerous-capability evaluation adds little marginal risk and adds real value in auditability, research access, and reducing dependence on three API vendors. A model that meaningfully advances the state of the art on a specific uplift axis — bio/chem synthesis planning, autonomous cyber-offense — is a different object, and "it's only slightly better" is a bad argument there because capability thresholds are where the harm function is nonlinear.

The second thing I'd say is that "open weights" is doing too much work as a phrase. Open weights, open data, open training code, and permissive licensing are four separate decisions with different risk and different benefit, and the interesting policy positions live in the combinations. Note also the license reality on the applied side — several widely-used "open" models carry usage clauses (large-MAU restrictions, output-usage and anti-distillation terms) that matter commercially and that many engineers never read. **📅 Volatile:** license terms change between model generations; check the actual license file rather than the family's reputation.

**⚠ Trap:** picking the side you think the interviewer holds. Anthropic's public position on release policy and Meta's are far apart, and a candidate who flips between them is worse than one who holds either consistently. What travels everywhere is the decision rule plus an honest statement of uncertainty.

**🗣 Say this in the room:** "I don't think this is settled and I'd be suspicious of anyone who says it is. My rule is that release decisions should turn on marginal uplift over what's already downloadable, not on absolute capability — which means the answer differs by capability axis rather than by model."

### How do you think about misuse when your product is enterprise software and the risk is your customer's behavior, not a random attacker's?

This is the shape the question actually takes at Harvey, Glean, Sierra, Ramp and Stripe, and it's more interesting than the consumer version because the abuse is contractual rather than adversarial. The mental model: **in enterprise, the misuse you have to design for is usually mundane and organizational — using the system for a decision it wasn't validated for — not a bad actor jailbreaking you.**

Concretely: you build a document-analysis system validated on contract review, and a customer starts using it for employment decisions. You build a search system over a company's internal corpus, and it turns out the ACL model in their legacy SharePoint is wrong, so the system faithfully surfaces the compensation spreadsheet to everyone. You build a summarizer, and a customer routes regulated communications through it that they're required to retain and can't. None of these involve an attacker. All of them are incidents.

The controls that actually work here are boring and mostly not model-level. **Scope statements in the product** — an intended-use description that appears where the work happens, not buried in an MSA. **Permission inheritance as a hard invariant** — the system must never be able to surface a document the requesting user could not open directly, enforced at retrieval time with the user's identity, not filtered post-hoc from a shared index, which is a design I would reject in review outright. **Per-tenant isolation with no cross-tenant caching**, which is a rule I'd enforce even when the cache-hit-rate argument is compelling, because the failure mode is unrecoverable. **Audit trails that record what context produced what output**, because six months later someone will ask why the system said something and "we don't retain that" is a bad answer for a regulated customer. **Slice-level evaluation on the customer's own data**, since your benchmark says nothing about their document distribution.

**💰 Math on the cross-tenant caching temptation:** a shared semantic cache might lift hit rate from 31% to 47% — at 300k calls/day and $0.008/call that's 0.16 × 300,000 × 0.008 = $384/day, ~$11.5k/month. Real money. It is also one bug away from serving Tenant A's answer to Tenant B, which is a breach-notification event, a contract termination, and possibly the company. I take the $11.5k loss. That trade, stated with the number and then declined, is exactly the kind of answer that scores in a values round — you priced it and still said no.

**🗣 Say this in the room:** "In enterprise the misuse I plan for isn't an attacker, it's a customer quietly using the system for a decision we never validated it for. The control for that is a scoped intended-use statement plus per-slice evaluation on their data — and an ACL model where retrieval runs as the user, not as the index."

### If you joined us on Monday, what safety-relevant work would you actually do in the first 90 days?

Answer this as a plan, not as a philosophy, because it doubles as evidence that your safety views are operational. The mental model: **most safety value in an applied role comes from measurement and blast-radius control, not from policy.** Say that out loud and then show the plan.

*Weeks 1–3: find out what we can't see.* Inventory every place a model output reaches a user or takes an action, and for each one write down the surface (draft vs autonomous), the worst case, whether it's reversible, and whether we'd detect it. This artifact — a one-page risk register keyed on deployment surface — usually does not exist and usually surprises people, because the riskiest surface is rarely the one with the most attention on it.

*Weeks 3–6: build the canary.* A fixed set of 40–80 questions with known-correct answers running against production on a schedule, alerting on a drop. This catches the failure class that page-based monitoring is structurally blind to — the system is up and returning 200s and the answers got worse. If nothing else got done in 90 days, this alone justifies the quarter.

*Weeks 4–8: sliced evaluation on the highest-stakes surface.* Not an aggregate score — the worst slice with enough volume to be meaningful. Publish it. The act of publishing a number that isn't flattering is itself a cultural contribution.

*Weeks 6–10: reduce blast radius on one surface.* Pick the place where an error is irreversible and make it reversible or gated: add a confirmation, an allowlist, a spend cap, a rate limit, a rollback path that's been tested.

*Weeks 8–12: a red-team session and an incident drill.* An hour with three colleagues actively trying to get the system to do the wrong thing, and a game day where we exercise the kill switch, because an untested kill switch is a comment.

**🗣 Say this in the room:** "My first 90 days would be almost entirely measurement and blast radius, not policy. The single highest-value thing is usually a canary eval running against production every fifteen minutes, because 'the service is up' and 'the answers are right' are different questions and most stacks only monitor the first one."

### Do you think the field should slow down?

Answer honestly and specifically, and expect the interviewer to argue the other side regardless of what they believe. The mental model: this is the purest test of whether your position is yours, because there's no way to know what answer the room wants — labs contain people who hold both views strongly and interviewers are explicitly looking for whether you fold.

What makes an answer good is decomposition rather than a verdict. "Slow down" bundles at least four separable things: pace of capability research, pace of deployment into consequential domains, pace of scaling compute, and pace of regulation. You can coherently hold that deployment into irreversible domains is running ahead of the evaluation science while also holding that slowing capability research unilaterally mostly transfers capability to actors who won't do the safety work. Say the combination you actually believe and name the tension in it — that unilateral restraint has a competitive-dynamics problem, and that "someone else would build it anyway" is an argument that justifies everything and therefore justifies nothing.

The applied-engineer version, and the one I'd land on: the bottleneck I see from where I sit isn't capability, it's **evaluation science and deployment discipline**. We can build systems whose behavior we cannot characterize, and the gap between what we can build and what we can measure is widening faster than either. That is a claim I can defend from direct experience — I've shipped systems I couldn't fully characterize — which makes it a stronger position than an opinion about compute trends I've read about.

**⚠ Trap:** answering in slogans in either direction ("safety is a tax on progress" / "we're building something we don't understand"). Both are unfalsifiable and both read as imported. Ground the claim in something you've observed.

**⚠ Trap #2:** conceding when pushed. If the interviewer says "but you'd agree the risks are speculative, right?" and you say "yes, absolutely," you've just demonstrated the exact failure mode. The correct response is to hold the distinction: "some of them are speculative and I'd label them that way; the deployment ones aren't speculative at all, I've caused a couple."

**🗣 Say this in the room:** "I'd separate capability from deployment. On deployment I think we're clearly running ahead of our ability to measure what we've built, and that's the part I've personally experienced. On capability I hold a weaker view, and I'd rather tell you it's weak than pretend otherwise."
### How do you research a company well enough that "why us" doesn't sound generic?

The mental model: generic enthusiasm is free, so it carries no information. What carries information is evidence that you engaged with a **decision they made** — a product trade-off, a published result, a public position — closely enough to have an opinion about it, including a mild disagreement. Praise is cheap; engagement is not.

The research procedure I'd run, in about three hours per company. **One: use the product for real, for at least a week**, on a real task, and keep notes on where it breaks. This is non-negotiable for the AI-product archetype and it is the single most common gap. **Two: read their engineering and research blog for the last twelve months**, not the marketing site, and extract the decisions — what they chose to build in-house, what they bought, what they published and what they conspicuously didn't. **Three: find one technical artifact** — a paper, an open-source repo, a conference talk, a detailed changelog — and read it well enough to summarize its contribution and name a limitation. **Four: read a recent postmortem, status page or public incident** if one exists; it tells you more about their engineering culture than any blog post. **Five: write down one question you genuinely can't answer from the outside.** That question is your closing move.

The output of that research is not a list of compliments. It's three sentences you can deploy: an observation about a decision, your read on why they made it, and the thing you'd want to know. "You moved from an agent that plans up front to one that re-plans after each tool call — I'd guess that's about error recovery on long tasks, and I'd want to know what it did to your token cost per task" is worth more than any amount of "I love your mission."

**⚠ Trap:** researching the company and not the *team*. Loops at Meta, Google, Amazon and Microsoft are team-matched; the org you'd sit in matters more than the logo, and asking the recruiter "which team, and what's their current top priority" is both allowed and clarifying. At smaller companies, research the person interviewing you — their talks and their GitHub — and it is not creepy, it's preparation.

**🗣 Say this in the room:** "The thing that made me look harder at you specifically was [decision]. My outside read is [hypothesis], and the part I can't figure out from here is [question] — which is honestly one of the things I wanted to ask you."

### Give me a worked "why this company" for two very different targets — a frontier-adjacent lab and an AI product company.

The structural difference: at a lab, "why us" is answered with **mission and judgment**; at a product company it's answered with **product taste and speed**. Same candidate, different emphasis, and using the wrong register is a real cost.

*Frontier lab / Anthropic-shaped.* Three beats: the work, the constraint, and the honest reservation. "The work I want to be doing is the applied layer — taking a model that's capable in the abstract and making it reliable for one specific consequential task, which is mostly evaluation and failure analysis. I want to do that somewhere the evaluation is treated as the product rather than as a gate before launch, and from the outside you're one of a small number of places where that's true — the interpretability and evals work being public is a signal about what gets rewarded internally. The reservation I'd want to talk about is how much an applied engineer here actually gets to influence what the model does versus building around it, because I'd find the second one less interesting after a year." That last sentence is the one that converts: a real reservation, calmly stated, reads as someone evaluating a decision rather than auditioning.

*AI product / Cursor-shaped.* Three beats: usage, an opinion, and a build instinct. "I've had it as my primary editor for a month. The thing I'd defend is [specific behavior] — I think the decision to [specific trade-off] is right and it's the reason I stopped using the alternative. The thing that's broken for me is [specific], and my guess is it's hard because [hypothesis about the constraint]. If I were here, the first thing I'd want to look at is whether [measurable thing]." Notice there's no mission talk at all. A founder round wants to know you'd be useful on Tuesday.

**⚠ Trap:** the transferable "why us" — an answer that would work verbatim for four companies. Test yours by swapping the company name in; if it still parses, it's not an answer. The specificity that survives the swap test is always a *decision they made*, never a value they hold, because values are shared across the whole industry.

### I've read a lot of papers I don't fully understand. How do I use frontier research in an applied interview without it backfiring?

Sparingly, precisely, and only when the paper is load-bearing for a claim you're making. The mental model: **citing a paper is a promise that you can answer follow-ups about it**, and in an applied round the citation buys you almost nothing while the failed follow-up costs you a lot. The expected value is negative unless you actually know the work.

The failure looks like this. Candidate mentions "we used speculative decoding" or drops "like in the Chinchilla paper" to establish credibility. The interviewer, who has read it, asks one specific question — what the acceptance-rate/draft-length trade-off does to your throughput, or what Chinchilla actually changed about the compute-optimal ratio — and the candidate produces a summary that could have come from a tweet. Everything the candidate said before that is now discounted. This is the ⚠ trap the spec warns about, and it is common precisely because name-dropping feels like a low-risk credibility play.

The rule I'd apply: **cite a paper only when it changed something you did.** "We were sizing a fine-tuning dataset and the compute-optimal scaling work is why I argued for more tokens rather than a bigger base model" is a legitimate use — the paper is doing work in the argument. "As Vaswani et al. showed..." is not. And if you're going to name it, be ready with the four-part summary: what problem, what the key idea was, what it replaced, and what its acknowledged limitation is. If you can't produce the limitation, you don't know the paper well enough to name it.

The honest fallback, which scores better than a bluff every time: describe the mechanism without the attribution. "There's a line of work on extending context by rescaling the rotary frequencies rather than retraining from scratch, and the practical upshot is that you can take an 8k model to 128k with a modest continued-pretraining budget — I'd have to look up which variant we'd want." That sentence demonstrates you understand the technique and are careful about attribution, which is exactly the combination a research-adjacent interviewer is scanning for.

**🗣 Say this in the room** (when caught at the edge of your knowledge): "I know the mechanism but I don't trust my memory on the attribution or the exact numbers, so let me describe what it does and flag that I'd verify the citation."

### What recent development in AI actually changed how you work?

This is asked in almost every loop and it is a rehearsed-90-seconds question, not a spontaneous one — treat it accordingly. The mental model: they're testing whether you consume the field as news or as engineering input. The difference is whether your answer contains a **behavior change**.

The template: the development, the mechanism in one sentence, what you did differently as a result, and a limitation you've personally hit. That last part is what separates you from someone who read a summary. "Reasoning models changed how I budget — the mechanism is that the model spends output tokens on intermediate reasoning before answering, so the cost model shifted from 'prompt length dominates' to 'thinking tokens dominate' and my per-request cost function needed a third term. What I do differently: I now set an explicit thinking budget per route rather than a global one, because the classification step doesn't need it and the multi-hop synthesis step does. The limitation I've hit is that more reasoning isn't monotonically better — on some short factual tasks the extra reasoning talks itself out of the correct first answer, and there's been reporting that heavy reasoning degrades calibration, which matches what I saw."

Note the structure: mechanism, behavior change, personally-observed limitation. All three, and the third one is where most candidates stop short.

**📐 Numbers you must know** for the reasoning-cost version of this answer: if thinking tokens are billed as output, a route that produces a 200-token answer but 3,000 thinking tokens is billed on 3,200 output tokens. At an output price of, say, $15/Mtok that's 3,200/1e6 × 15 = $0.048 versus $0.003 without thinking — a 16× swing on the output side alone. That's the arithmetic that justifies per-route budgets rather than a global switch. **📅 Volatile:** prices and whether thinking tokens are billed at the output rate differ by provider and change; verify before quoting.

**⚠ Trap:** naming something that is merely recent and famous, with no personal contact. "DeepSeek-R1 was really impressive" invites "what specifically about the training recipe?" and if the answer is "the reasoning," you've lost the exchange. Pick something you actually used, even if it's less glamorous — a structured-output mode, a caching feature, a batch tier — and the answer gets stronger, not weaker.

### What questions should I ask them, and what do my questions signal?

Your questions are graded. Treat the last five minutes as a scored segment, because interviewers write down what you asked and it is one of the few signals that distinguishes candidates who are choosing from candidates who are hoping.

The categories that carry signal. **Questions about how decisions get made:** "When an eval says a change is a regression and a customer wants it anyway, who decides?" This is a values-round-shaped question and it's the best one on the list because the answer genuinely varies and tells you something. **Questions about the failure you'd inherit:** "What's the thing about this system that everyone on the team knows is wrong and nobody's had time to fix?" **Questions about measurement:** "How do you currently know a model change made things worse?" — asked at an AI company, this often produces a slightly uncomfortable answer, which is informative for you and demonstrates your priorities to them. **Questions about the role's actual shape:** "What fraction of this job is evaluation and data work versus building features?" If the honest answer is 70/30 and you're excited about that, say so.

What to avoid: anything on the careers page, anything about promotion timelines in a first round, and — the one that reads worst — no questions at all, which is universally recorded as low interest regardless of how good your reasons are.

**🗣 Say this in the room:** "One I ask everywhere, because the answers vary a lot: when your eval says a change is a regression and a customer really wants it, what actually happens?"

**⚠ Trap:** asking a question you already know the answer to in order to demonstrate knowledge. Interviewers spot the performance and it converts a genuine signal (curiosity) into a negative one (posturing). If you want to show knowledge, show it in the answer segment where it belongs.

### Compensation and leveling come up in the behavioral round. How do I not lose money there?

Two rules, and they're mechanical. **Rule one: never name a number first in an early conversation, and never anchor on your current compensation** — especially with a large geographic differential, where your current number is not evidence about your market value and will be used as an anchor if you supply it. Deflect once, politely, with a redirect to scope: "I'd rather get aligned on the level and the scope first — my expectation is that the offer will be competitive for the level, and I'm happy to go into detail once we know what that is." If pressed a second time, give a **range you've derived from public data for that level and location**, not your history, and say where you got it.

**Rule two: leveling is decided by the evidence in your stories, not by the negotiation.** This is the part that connects back to the rest of this section. The difference between an IC5 and IC6 offer at these companies is usually six figures over a couple of years, and it's determined largely by whether your stories demonstrate scope — did you set the direction, did you influence other teams, did you own an outcome rather than a component. If your best stories are execution stories, you will be levelled on execution, and no amount of negotiation at the end recovers that. **The leveling conversation happens in the behavioral round; the salary conversation just ratifies it.**

**💰 Math on why the level dominates:** a one-level gap at a large AI employer is commonly on the order of $80k–$150k/year in total comp including equity refresh; over a four-year vest that's $320k–$600k. The most aggressive salary negotiation you will ever run might move base by $20k. The stories are worth an order of magnitude more than the negotiation, which is why "prep the behavioral round" is also financial advice. **📅 Volatile:** band sizes move; check Levels-style data for the specific company and level before your loop.

**⚠ Trap:** treating the recruiter screen as non-evaluative and giving a sloppy version of your ownership story there. Recruiters at these companies pass level recommendations forward, and the hiring committee reads the recruiter notes.

### Drill bank one — ownership and impact. Give me the weak and strong versions side by side.

Run these unaided, out loud, recording yourself, then compare against the strong version. The gap is your prep list.

**1. "Tell me about a project you're proud of."** *Weak:* "I led the migration of our monolith to services — it was a big project, took about eight months, and it made everything a lot faster and easier to maintain." *Strong:* "I'll pick a smaller one because it had a clearer result. Our support-answer pipeline resolved 41% of tickets and p95 was 6.4 seconds against a 3-second target. I found that reranking was on the critical path for every query even though 60% of queries had an unambiguous top hit, gated it on a score threshold, and p95 went to 2.9. The part I'd claim is noticing the 60% — the implementation was two days." *Why:* scope, numbers, and a claim limited to the part that was actually his.

**2. "What's the largest scale you've worked at?"** *Weak:* "We had millions of users." *Strong:* "About 8,000 requests a minute at peak across 40-odd tenants, with a hard 2-second SLO. The number that shaped my design decisions more than the RPS was the tenant count, because it meant noisy-neighbor isolation, not throughput, was the hard problem." *Why:* the strong version reports the constraint that mattered, not the biggest number available.

**3. "Tell me about a time you influenced without authority."** *Weak:* "I convinced the team to adopt better testing practices." *Strong:* "I wanted us to gate releases on a retrieval eval and nobody owned that. Rather than pitch it, I built it for one surface, ran it silently for three weeks, and brought the log of five changes it would have caught — two of which we'd actually shipped and later reverted. The argument made itself at that point. The mistake I made was doing it silently; the tech lead was reasonably annoyed I hadn't looped him in earlier." *Why:* a mechanism, evidence, and a self-critique that makes it believable.

**4. "Have you mentored anyone?"** *Weak:* "Yeah, I've onboarded a few juniors and done a lot of code review." *Strong:* "Two engineers over about a year. The concrete thing I changed with one of them: he was writing PRs that were correct and unreviewable — 900 lines, no description. We spent two sessions on decomposition and PR narrative, and his median PR went from about 600 lines to about 150 and his review turnaround went from three days to under one. That's a number I can point at rather than a feeling." *Why:* mentorship claims are usually unfalsifiable; this one isn't.

**5. "Why should we hire you over someone with an ML PhD?"** *Weak:* "I'm a fast learner and I'm really passionate about AI." *Strong:* "For a research role you shouldn't. For this one, the hard part isn't the modeling — it's that these are nondeterministic distributed systems with an unbounded input space and no unit tests, and the failure modes are tail latency, silent quality regression, cost blowouts and cross-tenant leakage. That's the exact set of problems I've spent eight years on, and I've done the work to close the model-layer gap rather than assuming it away." *Why:* concedes the true comparison, then reframes onto the actual job.

### Drill bank two — technical judgment and the framework boundary.

**6. "How would you improve the quality of this RAG system?"** *Weak:* "I'd fine-tune the model on our domain data." *Strong:* "I'd want to know where it's failing first, and there are only three places: retrieval didn't return the answer, retrieval returned it and the model ignored it, or the answer isn't in the corpus. Those have completely different fixes, and I'd separate them by measuring recall@k against a set of queries with known gold documents. Fine-tuning is the last thing I'd try, and only for behavior, not for knowledge." *Why:* the reflex-fine-tune answer is a known rejection trigger; the strong version is a diagnostic procedure.

**7. "You have vector search on your resume. Which index and why?"** *Weak:* "We used a vector database, I think it was HNSW under the hood." *Strong:* "HNSW at M=32, ef_construction 200, and we tuned ef_search per query class because the recall/latency curve is steep in the 40–120 range. The reason I know those numbers is that the default ef_search cost us about 4 points of recall@10 and nobody noticed for a month, because there was no recall metric — which is really a story about the missing eval, not about the index." *Why:* specificity, plus a pivot to the systemic lesson.

**8. "What did you actually write versus the framework?"** *Weak:* "I built the whole agent." *Strong:* "The framework gave us durable state, checkpointing and the graph semantics. I wrote the tool schemas and their validation, the retry classification, the token budget enforcement and the termination conditions. The loop underneath is about ten lines — model call, append tool results, repeat until no tool calls or budget exhausted — and I could write it here if that's useful." *Why:* volunteering the boundary is the whole answer.

**9. "Why didn't you use [framework/vendor/technique]?"** *Weak:* "We looked at it but it didn't fit our use case." *Strong:* "We evaluated it and rejected it on one specific thing: we needed per-step traces joined to our existing tracing backend, and at the time getting that out required patching internals. The trade we made was writing about 300 lines ourselves and giving up their retry logic, which we then had to rebuild worse. If I did it again with the current version I'd probably take the framework." *Why:* a named reason, a named cost, and a willingness to say the decision might have been wrong.

**10. "This design seems over-engineered."** *Weak:* "It needs to be robust for production." *Strong:* "For the current traffic, yes — the queue and the fallback path are unnecessary at 200 requests a day. I put them in because the provider rate limit was the thing that took us down twice, and I'd rather carry that complexity than get paged. If you told me we'd never exceed 500/day I'd delete both and save about 400 lines." *Why:* agrees where the critique is right, names the specific risk being bought, and states the condition under which he'd remove it.

### Drill bank three — safety, stakeholders and pressure.

**11. "A customer wants the model to make the final decision. Go."** *Weak:* "That's not really something I'd be comfortable with — there are a lot of risks with fully automated decisions." *Strong:* "Tell me what happens today when one of these is decided wrongly — is there an appeal, and what does it cost you? …Right, so wrong denials are the expensive direction. I'd propose the model recommends with a citation and a confidence, we measure agreement against your adjusters on 200 of your own historical cases, and if the low-complexity band clears a bar we set in advance, we automate that band with a sampled audit." *Why:* discovery, asymmetry, a sequenced path with a graduation criterion.

**12. "Explain to me, non-technically, why it made that up."** *Weak:* "It hallucinated — that's a known limitation of LLMs, they're stochastic parrots essentially." *Strong:* "It predicts what should come next rather than looking anything up, and 'plausible' and 'true' usually agree — but when they don't, it produces the plausible one in exactly the same confident tone. So we don't design for zero errors, we design for a known error rate that a human can check in five seconds, which is why every answer carries a citation." *Why:* no jargon, one mechanism, and it ends on what you do about it.

**13. "The PM wants to ship Friday with no eval."** *Weak:* "I'd tell them we can't ship without proper evaluation." *Strong:* "I'd convert it to a number by Thursday — 80 hand-labelled cases from staging gets us a point estimate to about ±11 points, which is enough to tell 'works' from 'doesn't.' And I'd propose we ship Friday to 2% behind a flag either way. I only actually block if the surface is irreversible — if it can send money or make a commitment." *Why:* a plan with a date, and an explicit statement of where his line actually is.

**14. "Do you worry about AI safety?"** *Weak:* "Absolutely, I think it's one of the most important issues of our time and I really admire the work you're doing on it." *Strong:* "In four separate senses that I'd keep apart, and I'd say my direct evidence is only in one of them — systems that are confidently wrong where someone acts on the answer. I've shipped one of those. On loss-of-control I've read the arguments and find them worth taking seriously, but I'd be overstating my position if I claimed direct evidence." *Why:* disaggregation plus honest calibration; the weak version is flattery and every lab interviewer has heard it forty times this month.

**15. "You seem to have a lot of caveats. Are you slow?"** *Weak:* "No, I ship fast, I just care about quality." *Strong:* "Fair challenge. My actual record is that I ship the internal or flagged version in week one and spend the caution on the surfaces where a mistake is irreversible. The thing I won't ship on day one is anything that can move money or make a commitment on the company's behalf without a human in the path — everything else goes out behind a flag." *Why:* accepts the premise as a fair question, then answers with a policy rather than a protest.

### Drill bank four — curveballs — and how do I rehearse all of this so it doesn't sound rehearsed?

**16. "What would you do in your first month here?"** *Weak:* "Learn the codebase and ramp up on the team's priorities." *Strong:* "Ship something small in week one, even if it's trivial, because that's how I find out where the deploy pipeline actually hurts. Then I'd want to write down every place a model output reaches a user or takes an action, with the worst case and whether we'd detect it — my experience is that document doesn't exist and it surprises people when it does."

**17. "What's a technical opinion you hold that most people disagree with?"** *Weak:* "I think most companies over-complicate their infrastructure." *Strong:* "That semantic caching is usually a bad idea in its common form. It's the highest cost-savings-per-line change available, so it gets shipped constantly, and its failure mode is returning a confidently wrong answer to a similar-but-different question with no error anywhere. I'd take the cost hit unless the cache is scoped to a verified intent set — and I've been on the wrong side of that one, which is why I hold it strongly."

**18. "Tell me about a time you failed."** *Weak:* "We missed a deadline on a project because of unexpected complexity." *Strong:* "I built a semantic cache on a similarity threshold, treating similarity as if it were equivalence. 'Cancel my order' and 'cancel my subscription' collided and got the same answer for six weeks before anyone noticed, because a fluent wrong answer doesn't page anyone. The general rule I took from it: I don't ship anything whose failure mode is 'confidently wrong' without a sampled correctness monitor on that specific path."

**19. "Where do you want to be in five years?"** *Weak:* "In a senior leadership position." *Strong:* "Still building, with more surface. Concretely I'd like to own the reliability and evaluation layer for a product that a lot of people depend on — the thing that decides whether a model change ships. Whether that has a manager title on it I genuinely don't care about, and I'd rather be honest about that than tell you a ladder answer."

**20. "Anything you want to tell us that we didn't ask about?"** *Weak:* "No, I think we covered everything." *Strong:* "One thing — you didn't ask what I'd be bad at here, and I'd rather say it than have you find out. I haven't worked at this pace of shipping; my instincts are calibrated for a team where a bad deploy costs a customer money, and I'd expect to spend the first two months recalibrating what 'good enough to ship' means. I think that's a real adjustment and not a fake weakness."

**The rehearsal method.** The reason over-rehearsed answers fail is that you memorized *sentences*; the reason unrehearsed answers fail is that you had to compose under load. The resolution is to memorize **facts and structure, then regenerate language every time.** Concretely: for each of your eight stories, memorize six atoms — the system, the tension, three numbers, and the thing you got wrong. Never write the prose. Then do reps where the *entry point* changes: tell the same story starting from the incident, from the disagreement, from the metric, from the person who pushed back. Four entry points per story means the language cannot calcify.

**🏋 Drill (the full simulation, 60 minutes, unaided):** have someone ask you twelve questions drawn at random from the twenty above, in a random order, with the instruction to push back once on every answer — including on the ones you got right, because folding under a push on a correct answer is the failure mode this round is built to detect. Pass criteria: (a) every answer contains at least one number; (b) at least two answers contain something you got wrong; (c) you did not change a position under a single push; (d) no answer exceeded three minutes without the interviewer asking for more; (e) you can name, after the fact, which of your eight stories you used and which you never reached for — the unused ones are the gap in your coverage set.

**🗣 Say this in the room, when you genuinely don't have a story for what was asked:** "I don't have a clean example of that, and I'd rather say so than stretch one. The closest I have is [adjacent story] — want that, or would a hypothetical about how I'd handle it be more useful?" That answer costs you almost nothing and buys you enormous credibility on everything else you said, because it establishes that when you *do* claim something, you mean it.


---

## 85. Forward-Deployed Engineering: Decomposition Cases and Customer Simulation

*Mastering this proves you can pass the highest-weight, lowest-pass-rate round in the industry — reportedly ~40% pass rate and ~30% of total evaluation weight.*

### Walk me through what actually happens in the forward-deployed decomposition case. What's the format, and what do you think is really being scored?

The format is deceptively simple and that is what makes it lethal. An interviewer plays a customer executive — VP of Claims at an insurer, GC's office at a bank, head of support at a retailer — and hands you a sentence like "we spend too much money processing claims, can AI help?" You have 45 to 60 minutes. There is no data, no repo, no spec, and no correct answer. Most candidates treat it as a system design round with a business skin on it, and that misreading is the single largest cause of failure.

What is actually being scored is whether you can convert an unbounded business complaint into a **scoped, sequenced, measurable engagement that a real customer would sign and a real team could execute**. Concretely, the rubric that shows up again and again across OpenAI FDE, Anthropic's Applied AI loop, Palantir, Scale and Sierra decomposes into five things: (1) do you ask questions before you propose, and are they the *load-bearing* questions; (2) can you restate the problem in the customer's own vocabulary such that they would say "yes, that's it"; (3) do you cut scope out loud and defend the cut; (4) is there a demonstrable artifact inside two weeks that de-risks the biggest unknown; (5) is there a number, agreed in advance, that decides whether the pilot succeeded. Architecture is maybe 15% of the score. Most candidates spend 80% of the time there.

The second thing being scored is quieter: **can the interviewer imagine putting you in a room alone with their customer.** That is a judgment about temperament, not knowledge. Do you interrupt the customer? Do you agree with something that is technically impossible because they seem excited? Do you lecture? Do you say "great question" and then not answer it? An FDE is a person the company deploys unsupervised into a room where the deal is worth seven figures. They are hiring for someone who does not need to be chaperoned.

**📐 Numbers you must know:** the widely-reported shape of this round is a ~40% pass rate and roughly ~30% of total loop weight at FDE-heavy employers, with the customer-simulation variant reported to eliminate around 60% of candidates who already cleared the coding stages. **📅 Volatile:** these are candidate-reported aggregates, not published figures — treat them as *directionally* true (this is the highest-weight, lowest-pass-rate stage) and never quote a percentage in the room.

**🗣 Say this in the room:** "Before I design anything I want to make sure I'm solving the problem you have rather than the one I find interesting — can I ask you six or seven questions about how claims move through your shop today?"

### I'm going to say one sentence — "our claims processing costs too much, can AI help?" — and then stop talking. What do you do next?

I do not design. I do not say "so we'd build a RAG pipeline over your claims corpus." The move is to buy information cheaply and visibly, and the way you do that is to ask for a *walkthrough of one real unit of work*, because concrete beats abstract every time and because it flushes out constraints the customer would never think to volunteer.

My literal opening: "Can you walk me through one claim, start to finish, from the moment it arrives to the moment money moves? Pick a boring one." Then I shut up and take notes on a shared surface they can see. I am listening for four things: every **handoff** (each one is a queue with a latency and an owner), every **decision** (each one is a candidate for automation and a candidate for a compliance problem), every **system** named (each is an integration with an auth story), and every **exception path** ("well, if it's over $10k, Denise looks at it") because exception paths are where the true cost lives and where the 80/20 hides.

Only after that do I ask the volume-and-money questions, because now I can ask them precisely: how many claims per month, how many minutes of adjuster time per claim, what does an adjuster cost fully loaded, what fraction hit the exception path, what is the current error rate and how do you know. Those five numbers are the entire business case and most customers have never had anyone ask for all five at once.

**⚠ Trap — the architecture reflex.** Backend engineers are trained to be helpful by proposing. In this round, proposing early reads as *not listening*, and it is scored as a failure even if the architecture is correct. I have seen candidates sketch a genuinely good agentic pipeline in minute four and get a no-hire with the note "did not discover the constraint that the client's claims data is only available as a nightly export." The architecture was fine. They just designed for a company that didn't exist.

**⚠ Trap #2 — asking questions that don't change your answer.** "What's your tech stack?" is a filler question at minute two; it changes nothing about scope. "Which of these steps, if it were 10× faster, would actually let you close claims faster?" changes everything. Before you ask a question, know which branch of your plan each answer sends you down. If both answers lead to the same plan, don't ask it.

**🗣 Say this in the room:** "Pick one real claim and walk me through it — I'd rather understand one case deeply than the process in the abstract, because the exceptions are usually where the cost is."

### Give me your decomposition method, end to end. Assume I'll interrupt you.

Eight beats, in order, and the order is the method — every failed case I have watched came from running them out of sequence.

**1. Clarifying questions (10–12 min).** Walk one unit of work. Extract handoffs, decisions, systems, exception paths. Get the five business numbers: volume, minutes per unit, loaded cost per person, exception rate, current quality and how it's measured.

**2. Stakeholder map (3 min).** Who signs, who uses, who can veto, who loses if this works. Say it out loud and validate it: "So the budget is yours, the users are the 40 adjusters, and compliance can stop us — is there anyone else who can say no?"

**3. Current-state process, drawn.** A box-and-arrow of today with the times and volumes attached. This is the artifact that makes the customer trust you, because it is the first time anyone has shown them their own process with numbers on it.

**4. Constraint discovery.** Data access, auth, regulatory, latency, change management, procurement. Constraints are not caveats; they are the design. "Only available as a 2am export" and "an adjuster must sign every denial" are the two facts that determine the whole architecture.

**5. Problem restatement the customer would sign.** One sentence with a number in it, in their vocabulary. "You want to cut adjuster touch-time on the 62% of auto claims that are routine from 18 minutes to under 6, without changing who signs the denial." If they say "yes, exactly," you have earned the right to design.

**6. What you will not build,** said out loud, with the reason. This is the senior move and I'll defend it separately.

**7. A two-week demonstrable slice** that de-risks the largest unknown, not the easiest component.

**8. A success metric they will accept**, defined before you build, with a threshold and a measurement procedure.

Only then, if there's time, architecture — and even then it's one diagram with the integration points and the human-in-the-loop boundary marked, not a component inventory.

**🗣 Say this in the room:** "I'll run this in a fixed order: understand today, find the constraints, restate the problem back to you until you agree with the restatement, then cut it down to something we can show you working in two weeks. If I start drawing boxes before step three, stop me."

**⚠ Trap:** treating this as a checklist you recite rather than a loop you run. The interviewer will throw a fact at you in minute 30 ("oh, by the way, adjusters are unionized and the contract specifies human review") that invalidates steps 5–8. The score comes from *visibly re-running the loop* — "that changes the restatement, let me redo it" — not from having gotten it right the first time. Candidates who absorb a plan-invalidating fact and keep talking about their original plan fail. Every time.

### Draw me the stakeholder map for an enterprise AI deployment. Who actually matters, and which one do candidates always miss?

The mental model: an enterprise AI pilot does not fail on model quality, it fails because someone with veto power was discovered in month three. So the stakeholder map is really a **veto map**, and you build it in the first fifteen minutes so that nobody surfaces later as a surprise.

Five roles, and they are almost never the same person:

**The economic buyer** signs the contract and owns the budget line. They care about one number and it is money or headcount, not accuracy. Your success metric must be translatable into their number.

**The champion** brought you in. They are betting personal credibility on you. They will tell you the truth about internal politics if you ask privately, and they need artifacts they can forward internally — which is why the two-week demo matters more than a perfect quarter-three system.

**The end users** — the 40 adjusters, the 12 paralegals, the support agents. They can kill the pilot by not using it, and they often have a rational reason to: if the tool measures them, or threatens them, or adds a click. Nobody asks them anything, which is why adoption dies.

**The blockers.** Security review, legal/compliance, privacy, procurement, and — the one candidates always miss — **IT/platform**, who own the systems you need to integrate with and who were not consulted about your existence. IT does not want you; you are unplanned work with a deadline attached to someone else's promotion. Getting a service account provisioned can genuinely take six weeks, which is longer than your pilot.

**The threatened party.** Whoever runs the BPO contract or the offshore team that this displaces. If your success means their budget shrinks, they will be helpful, slow, and unavailable.

**⚠ Trap:** treating the champion as the customer. The champion is your ally, not your buyer, and champions systematically under-report the blockers because they want you to say yes. The question I always ask them, privately: "If this works perfectly, who inside your company is unhappy about that?" The answer to that question has saved more engagements than any technical decision I've made.

**💰 Math on why IT is the schedule risk:** a 2-week slice needs data access on day 2. If a service-account request queues behind a standard 4-week change-advisory-board cycle, your 2-week demo becomes a 6-week demo and the champion's credibility is spent before you write a line of code. The mitigation is explicit and it belongs in your answer: on day one, ask for a **de-identified export of 200 real cases over email or SFTP** as the day-2 unblock, and start the production access request in parallel. Two tracks, one of which cannot be blocked by a ticket queue.

### Why do you insist on mapping the current-state process before designing anything? Isn't that consulting theater?

It is the opposite of theater — it is the only reliable way to find where the money actually is, and it protects you from the most expensive mistake in applied AI, which is automating a step that costs nothing.

Here's the mechanism. Business complaints arrive as aggregates ("claims cost too much") but cost is never uniformly distributed across the process. When you draw the current state with time and volume on each arc, you almost always find that 70% of the labor sits on one or two steps, and those steps are frequently *not* the ones the customer named. The customer says "document review is slow." You draw the process and find document review is 6 minutes and the subsequent *reconciliation of the reviewed document against three internal systems* is 22 minutes. If you had built a document-review assistant you would have delivered a 6-minute win on a 28-minute process — a 21% improvement that no one notices — and lost the account.

Mechanically, what I draw is a swimlane with three annotations per step: **volume** (how many units per month reach this step), **touch time** (minutes of human attention), and **wait time** (elapsed clock time, which in queue-driven back-office processes is routinely one to two orders of magnitude larger than touch time and is what the customer's *customers* actually feel). Those two clocks matter differently: touch time is cost, wait time is experience. If the customer's complaint is "our NPS is bad," the fix is wait time and may involve no AI at all — just removing a queue. If the complaint is "we're hiring 30 more adjusters," the fix is touch time. Say which one you're optimizing and why.

**⚠ Trap — optimizing touch time when the customer meant wait time.** I have watched a team cut per-claim handling from 18 minutes to 7 and get told the pilot "didn't do anything," because claims still sat in a queue for four days waiting for a batch job. The AI was fine. The bottleneck was a nightly cron.

**💰 Math that makes the case concrete:** 40,000 claims/month, 18 min touch time = 12,000 person-hours/month. A fully-loaded adjuster at $75k salary is roughly $110k/yr all-in ≈ $53/hour at 2,080 hours. So the process costs 12,000 × $53 ≈ **$636,000/month**. Now the important part: if 62% are routine and you cut those from 18 to 5 minutes, you save 0.62 × 40,000 × 13/60 = 5,373 hours/month ≈ **$285,000/month**, or $3.4M/year. That is the number the economic buyer cares about, and you got it from a process map and five questions — not from a model.

### What are you hunting for in constraint discovery, and which constraints have actually killed projects you've seen?

Constraints are the design. A senior FDE hunts them deliberately because every one of them either eliminates a whole architecture or converts a 6-month plan into a 6-week plan. I run six categories, in roughly this order of how often they blow up a plan.

**Data availability and shape.** Is there an API or is there an export? Is history retained, and for how long? Are the fields you need actually populated, or are they nullable and empty 40% of the time? Is there labeled ground truth anywhere — a QA sample, an audit log, a dispute record? That last one is gold: audit and QA records are pre-existing labels and they are the cheapest eval set you will ever get.

**Auth and permission semantics.** Who is allowed to see what, and is that encoded in a system you can query, or in a person's head? This is the constraint that determines whether retrieval is even legal.

**Regulatory and contractual.** Adverse-action notice requirements, "a human must decide," records-retention rules, model-explainability obligations, data residency, whether the customer's contract with *their* customers permits sending data to a third-party model provider at all. In healthcare and lending this is usually the binding constraint.

**Latency and mode.** Is this interactive (an agent waiting) or batch (overnight)? Batch changes everything: you can use a bigger model, retry aggressively, use the provider's batch tier at roughly half price, and you don't need a streaming UI.

**Change management.** Who has to change their behavior for this to produce value, and what is in it for them? If the answer is "they get measured more closely," you have a very hard project.

**Procurement and security review.** How long is their vendor security review, do they require a specific cloud, do they need the model in their VPC, is there a DPA in place with the model provider. A "we can only use models running in our own Azure tenant" discovered in week five re-architects your entire solution.

**⚠ Trap — accepting "we have an API" at face value.** The correct follow-up is "can you send me a sample response body?" I have twice been told an API existed and twice found it was a SOAP endpoint returning a base64-encoded PDF of a screen render. Ask for the payload, not for the noun.

**🗣 Say this in the room:** "I'm going to spend a few minutes on constraints because they usually determine the design more than the requirements do — specifically I want to know what data I can actually touch on day one, who has to approve the output, and whether this is interactive or overnight."

### Give me the problem restatement you'd hand back to the customer. What makes a restatement good enough that they'd sign it?

A problem restatement is a contract disguised as a sentence. Its job is to convert an unbounded complaint into a bounded, falsifiable objective that the customer recognizes as theirs. If it doesn't have a scope boundary and a number, it isn't a restatement — it's a paraphrase.

The template I use has four slots: **[population] × [current state, quantified] → [target state, quantified] × [invariant that must not change]**.

Bad: "Use AI to make claims processing more efficient." No population, no numbers, no invariant. Unfalsifiable, and it means everyone can be disappointed differently.

Good: "For the ~62% of auto claims under $5,000 with no injury and a police report attached, reduce adjuster touch time from 18 minutes to under 6, with no change to who signs the settlement and no increase in the 30-day reopen rate." That sentence does five things at once: it names the slice you'll work on (and by omission, the slice you won't), it states the baseline, it states the target, it protects the compliance invariant, and it names the quality guardrail the customer would otherwise discover the hard way.

The invariant clause is the part that separates senior from mid. Every automation has a metric that will silently get worse — reopen rate, appeal rate, escalation rate, complaint volume, downstream rework. Naming it yourself, before the customer does, is the single most credibility-generating move available in this round. It signals you've shipped something before.

Then you say the magic words: **"Does that match how you'd describe it? If any part is wrong, I'd rather fix it now than in six weeks."** And you wait. Half the time they correct one clause, and that correction is worth more than the previous twenty minutes.

**⚠ Trap:** restating in your vocabulary instead of theirs. If they say "adjuster," never say "operator." If they say "file," never say "record." Vocabulary mismatch reads as "this person is going to build a generic thing." Mirror their nouns exactly — it costs you nothing and it is the cheapest trust you can buy.

**🗣 Say this in the room:** "Let me play back what I think the problem is, and I'd like you to correct me — for routine auto claims under five thousand dollars, get touch time from eighteen minutes to under six, without changing who signs and without the reopen rate going up. Right?"

### Tell me about choosing what not to build. Why do you say it out loud instead of just quietly leaving it out of scope?

Because unstated scope is *assumed* scope, and every enterprise AI pilot that ended in a fight ended in a fight about something nobody wrote down. Saying the exclusion out loud, early, and with the reason, is the highest-leverage sentence in the entire engagement — and interviewers score it explicitly because it is the clearest available signal of seniority.

There is also a psychological mechanism worth understanding. A customer who hears "we'll do all of that" gets a warm feeling and zero information. A customer who hears "we are *not* going to touch injury claims in this pilot, because they involve medical records, a different regulatory regime and a much harder judgment call, and if we include them we'll spend the whole pilot on 8% of your volume" learns three things: you understand their domain, you've done this before, and you are optimizing for their outcome rather than for the size of the statement of work. Counterintuitively, cutting scope makes the deal *more* likely to close, not less, because it makes the plan believable.

What I cut, in roughly this order: (1) anything requiring a new data source that isn't already accessible; (2) the long tail of case types below ~10% volume; (3) full automation where assisted-drafting captures 80% of the value at 20% of the risk; (4) UI polish and custom frontends — I use their existing surface or a bare internal tool; (5) anything downstream of a system I'd have to write into rather than read from, in phase one. Writes require change control, rollback design and a much scarier security review than reads.

The phrasing matters. Every exclusion gets a **reason** and a **door**: "not in this pilot, because X — and here's what would make it worth doing in phase two." That way you're sequencing, not refusing.

**⚠ Trap — the "we can do that too" reflex under enthusiasm.** The customer gets excited, riffs on five adjacent use cases, and the candidate nods along. Every nod becomes an expectation. In the simulation the interviewer will do exactly this on purpose. The correct response is warm and firm: "That's a genuinely good idea and I think it's a phase-two thing — if we try to do both, the risk is we do neither well by the end of Q3."

**🗣 Say this in the room:** "Here's what I'd deliberately leave out of the pilot, and why — I'd rather be explicit about it now than have it show up as a surprise in the readout."

### How do you scope a two-week demonstrable slice? What goes in, what gets cut, and why two weeks?

Two weeks is not a productivity target, it is a **feedback-loop constraint**. The purpose of the first slice is not to deliver value; it is to convert your riskiest assumption into a fact while the customer still has the political capital to help you. Every week you go without contact with real data and real users, you are compounding assumption risk on a champion's borrowed credibility.

The slice has four mandatory properties. It must run on **their real data** (de-identified is fine, synthetic is not — synthetic data hides exactly the mess you need to discover). It must produce an output a **real user can react to**, because "this is wrong and here's why" from a domain expert is the highest-bandwidth signal in the engagement. It must be **measurable against the eval set** you built in week one. And it must **touch the hardest integration**, even if only in read-only, single-record, hand-triggered form — because integration is where schedules die and you want that discovered in week two, not week nine.

What gets cut: authentication (run it as yourself with a service account or a manual export), scale (50 cases is a demo, 50,000 is a project), the UI (a Streamlit page or a spreadsheet with a column of model outputs is genuinely fine and reads as *focused*, not lazy), error handling, retries, multi-tenancy, and any second use case.

Concretely, for the claims case: a script that reads 200 de-identified routine claims from an export, produces a structured extraction plus a recommended action plus a citation to the policy clause it relied on, writes the result to a spreadsheet with one row per claim, and scores itself against 50 adjuster-labeled cases. Two engineers, ten working days. That is a demo that changes the conversation, because the customer's own adjusters can sit down and mark it up.

**⚠ Trap — building the two-week slice out of the *easiest* components so it demos cleanly.** A polished demo on synthetic data that avoids the real integration is worse than useless; it manufactures confidence and defers the discovery of the thing that kills you. If your slice can't fail, it isn't testing anything.

**🏋 Drill:** take any of the seven cases later in this section, set a 6-minute timer, and write the two-week slice as: data source, exact output artifact, the one integration touched, the eval set size, the named riskiest assumption it tests, and three things explicitly excluded. Pass criterion: an engineer who has never seen the case could start Monday from your six bullets, and the "riskiest assumption" is not "can the model do this."

### Sequencing — which piece do you build first, and how do you decide?

The rule I enforce is: **the first demo de-risks the largest unknown, not the largest component.** This is the inverse of how backend projects are usually sequenced (foundations first, then features) and getting it backwards is the most common senior-level mistake in this round.

The reasoning is about the shape of the risk distribution. In a normal backend project, uncertainty is roughly uniform — you know the system will work, you're just estimating effort. In an applied AI engagement, uncertainty is savagely bimodal: 80% of the plan is boring plumbing you could estimate to the day, and there are one or two binary unknowns that determine whether the project is possible at all. "Can a model read these particular scanned adjuster notes accurately enough to be useful" is a coin-flip, and everything downstream is conditional on it. Building the ingestion pipeline first is spending three weeks to learn nothing about the coin flip.

So the procedure: list every assumption, mark each with (a) probability it's false and (b) blast radius if false, and attack the top-right quadrant first, in the cheapest possible way. Usually the cheapest possible way is embarrassingly manual — paste 30 real documents into a chat window and eyeball the outputs. That is a legitimate first-day activity and I will defend it in any room. It answers the binary question in two hours instead of two weeks.

Second-order sequencing: after the model-capability unknown, the next-riskiest thing is almost always **data access**, then **whether the users will change their behavior**, then everything else. Notice that only the first of those is a model question. In my experience the ranked killers of enterprise AI pilots are: change management, data access, unclear success criteria, and only then model capability.

**⚠ Trap — de-risking in the order the architecture diagram flows.** Left-to-right sequencing (ingest → chunk → embed → retrieve → generate → evaluate) feels natural and is exactly wrong, because the unknown lives at the right end and the plumbing lives at the left. Build right-to-left: prove the generation quality on hand-fed data, then work backwards toward automated ingestion.

**🗣 Say this in the room:** "The first thing I'd build is whatever answers the question 'is this even possible' — usually that means hand-feeding thirty real documents through a model on day one, before I write any pipeline. If that fails I'd rather know in an afternoon than in a month."

### What's a success metric the customer will actually accept? How is it different from the eval number you'd report internally?

They are two different objects and conflating them is a classic failure. Your eval number is a **model-quality measurement**: 91% field-level extraction accuracy, 0.83 macro-F1 on triage category, 4% hallucinated-citation rate. The customer's success metric is a **business outcome under an operating policy**: minutes per claim, percentage of tickets fully deflected, dollars of rework avoided, days-to-close. The bridge between them is the *policy* — the routing thresholds and human-review rules that convert a probability into a business process.

A metric the customer accepts has five properties. It is **measurable with data they already have** (if it requires new instrumentation, you're now also delivering an analytics project). It has a **baseline measured before you start** — this is non-negotiable and everyone forgets it; without a pre-measured baseline, any result is arguable. It has a **threshold agreed in writing** ("≥40% of routine claims fully auto-processed at ≥95% agreement with adjuster decisions"). It has a **guardrail metric that must not regress** (reopen rate, appeal rate, CSAT). And it has a **named owner and a measurement date**.

The key structural insight: your metric must be a **rate on a defined population**, not an aggregate. "Reduce claims cost 30%" is a trap because it depends on volume mix, on staffing decisions you don't control, and on a hundred confounders. "Of auto claims meeting criteria X, ≥55% complete with zero adjuster touches, at ≥95% decision agreement, with 30-day reopen rate not exceeding the current 3.1%" is defensible because every term is measurable and attributable.

**⚠ Trap — accepting "accuracy" as the success metric.** Accuracy is not a business metric and its definition will be relitigated at the readout. Worse, the customer's mental accuracy target is almost always ~99%, which is above human baseline and unachievable. Reframe immediately to **coverage at a fixed quality bar**: fix the quality (agreement with a human expert) at a level they accept, then compete on what fraction of volume you can handle at that bar. That framing is honest, it is how the system will actually be tuned, and it makes the precision/recall trade legible to a non-technical buyer.

**💰 Math showing why coverage-at-quality is the right frame:** suppose your triage model at threshold τ auto-handles 55% of volume at 97% agreement, or 80% of volume at 91% agreement. At 40,000 claims/month with $53/hr adjusters and 18→0 minutes saved on auto-handled claims: 55% coverage saves 0.55 × 40,000 × 0.3 hr × $53 = **$350k/month**; 80% coverage saves $509k/month but the 9% error rate on 32,000 claims = 2,880 wrong decisions/month vs 660 at the tighter threshold. If a wrong decision costs $400 in rework and appeal, that's 2,880 × 400 = $1.15M/month in error cost versus $264k. The tighter threshold wins by a mile, and now you can *show* the customer the curve rather than argue about accuracy.

### The customer says "just tell me what it'll cost and how long it'll take." How do you size and price this under genuine uncertainty?

I never give a single number, and I say why in one sentence: a point estimate on a project with a binary unknown in it is not an estimate, it's a coin flip with a decimal point. What I give instead is a **staged commitment with a decision gate**, which is both more honest and — importantly — easier to get approved, because it converts a large scary number into a small one.

The structure: **Stage 0 (1 week, fixed price or free):** build the eval set from their examples, hand-run 50 cases, produce a feasibility memo with measured numbers. **Stage 1 (2–3 weeks):** the demonstrable slice on real data, one integration read-path, a measured quality number against the eval. **Gate:** a written go/no-go with pre-agreed thresholds. **Stage 2 (6–8 weeks):** production pilot with real users, monitoring, and the guardrails. **Gate.** **Stage 3:** rollout and handover.

For effort, I estimate the plumbing (which I can estimate) and bound the unknown (which I can't). "Ingestion, extraction pipeline, eval harness and the review UI is 6–8 engineer-weeks and I'm confident in that. The unknown is whether the scanned notes are legible enough; if they are, add two weeks of tuning, if they aren't, we need OCR remediation and that's a different project which I'd scope after Stage 0."

Then I name the assumptions explicitly, as a list, because that list *is* the estimate: "This plan assumes (1) we get a de-identified 500-case export within 5 business days, (2) two adjusters can give us 4 hours each for labeling, (3) claims data is available via the existing warehouse rather than the mainframe, (4) we're read-only in phase one, (5) your security review for a SaaS model provider is already complete. If (1) or (2) slips, everything slips one-for-one. If (3) is wrong, add three weeks. If (5) is wrong, this is a Q4 project, not a Q3 one."

**🗣 Say this in the room:** "I can give you a confident number for the parts I've done before and an honest range for the part I haven't, and I'd rather do that than give you one number that's wrong. What I'd propose is a one-week feasibility stage that turns the biggest unknown into a measurement — and then the real estimate is worth something."

**⚠ Trap — the estimate that assumes zero customer-side latency.** Every plan I've seen blow up did so on the customer's clock, not the vendor's: waiting on data, waiting on access, waiting on SME time, waiting on legal. Put the customer's obligations in the plan as dated line items with owners, and say plainly: "these three items are on your side and they're on the critical path." Doing this in the interview is a strong senior signal, because it shows you've been burned.
### You've said your first deliverable is an eval, not a demo. The customer wants to see something working. Defend that.

I don't defend it as a methodology point, I defend it as *their* interest, and I never use the word "eval" in the room. What I say is: "Before I build anything, I want us to agree on what 'good' looks like using your own cases — otherwise in six weeks we'll be arguing about whether it works based on whichever three examples you happened to try."

That sentence lands with every enterprise buyer who has been burned by a vendor demo, which is all of them. It reframes the eval from *engineering hygiene* to *your protection against me*. And it's true: the eval set is the only artifact that makes the pilot's outcome non-negotiable, which is as much in the customer's interest as mine.

The mechanism underneath: an LLM system's quality is not a scalar you can eyeball, and the failure mode of demo-driven development is brutally specific. You show ten cases, they work, everyone's delighted; the customer then tries their own case, it fails, and now the entire project's credibility rests on an n=1 anecdote from a VP. With a 50-case labeled set built from *their* examples, that same failure becomes "yes, that's in the 12% we don't handle yet, here's the cluster it belongs to and here's what it would take." One is a crisis, the other is a project update. Same underlying system.

Practically, the eval is also cheap in a way customers don't expect. Stage zero is: get 50–100 real cases, sit with two domain experts for four hours, have them label the correct output for each, and write a scoring function. That's a week including scheduling. And it doubles as requirements discovery — the labeling session is where you discover that "urgent" means three different things to three adjusters, which is a requirement you would otherwise have found in UAT.

**⚠ Trap — building the eval from cases *you* find interesting.** If you sample cases yourself, you sample toward what the model handles or toward what looks hard to an engineer. The set must be a stratified sample of *their* real distribution: pull by volume across case types, oversample the exception paths they named, and deliberately include the five cases their most experienced person considers hardest. Ask for those explicitly — "give me the five that would trip up a new hire" — because those five determine whether the system is trusted.

**🗣 Say this in the room:** "The first thing I'd deliver isn't a demo, it's a scorecard built from fifty of your own real cases with your team's answers on them. That way when we show you something in two weeks, you're not deciding whether you like it — you're reading a number you already agreed was the right number."

### Realistically, how do you get a busy customer SME to label 50 cases for you? They have a day job.

You do not send them a spreadsheet and a link. That fails roughly always, and it fails silently — you get seven rows back three weeks later, half of them blank. Labeling is a scheduling and incentive problem, not a tooling problem, and treating it as tooling is the mistake.

What works: **book a two-hour session, in person or on video, with two experts at once, and do the labeling live with you driving the screen.** Three reasons this dominates async. First, it converts an open-ended obligation into a bounded calendar block, which is the only kind of commitment busy people keep. Second, two experts disagreeing in real time is the single richest source of specification you will ever get — every disagreement is an ambiguity in the task definition that would otherwise have surfaced as a "bug" in month two. Third, you are present to ask "why?" after each label, and the *why* is what becomes your rubric.

Reduce the work per item ruthlessly. Never ask them to write the ideal output from scratch — that's 5–10 minutes an item and they'll quit. Instead pre-fill a draft (yes, from a model; say so plainly) and ask them to **accept, edit, or reject with a one-word reason code**. Accept/edit/reject on a pre-filled draft runs about 60–90 seconds per case, so 50 cases is genuinely one 90-minute session with a break.

Then sequence it: 10 cases together while you calibrate and argue about the rubric, then 40 more with them working and you watching. If they will only give you one hour, do 25 and say so honestly in the readout.

**💰 Math on why this is a bargain:** two SMEs × 2 hours = 4 expert-hours. At a fully-loaded senior claims adjuster or paralegal rate of ~$60–110/hour that's roughly $250–450 of their time. Against a pilot that costs the vendor 6–10 engineer-weeks (call it $60k–100k of loaded engineering), spending $400 to make the outcome measurable is a 0.5% insurance premium. Say exactly this to the economic buyer when the champion tells you nobody has time: "I'm asking for four hours of expert time to protect a hundred thousand dollars of build."

**⚠ Trap — accepting one labeler.** A single labeler gives you a rubric with no measured ambiguity, which means you cannot distinguish "the model is wrong" from "the task is underspecified." Have both experts independently label an overlapping subset of 15–20 items and compute raw agreement. If they agree on only 78% of cases, then **78% is your practical ceiling** and you must say so out loud, early — it reframes the entire success conversation and it is the most useful number you will produce in week one.

### How many labeled examples do you actually need? Someone's going to say fifty isn't statistically meaningful.

They're half right, and the honest answer is that 50 is a *decision* instrument, not a *publication* instrument, and you should say which one you're building.

Do the arithmetic out loud. If you measure 90% correct on n = 50, the standard error is √(0.9 × 0.1 / 50) = √0.0018 = 0.042, so the 95% interval is roughly 90% ± 8.3pp — call it **82% to 98%**. That is a wide band, and it means 50 cases can tell you "this is clearly viable" or "this is clearly not," but it cannot tell you whether prompt A at 88% beats prompt B at 84%.

For that comparison you use the fact that both variants ran on the *same* cases. A paired comparison (McNemar's test on the discordant pairs — cases where one variant is right and the other wrong) is dramatically more powerful than treating them as independent samples, because the shared case difficulty cancels. In practice with n = 50 paired items, a difference is detectable if the discordant pairs lean heavily one way — roughly, you need about 10+ discordant cases splitting something like 9–1 before you should believe it. Below that, say "inside the noise" and mean it.

So my staging is: **50 cases to decide feasibility and to find failure clusters** (which is qualitative work and needs far fewer items — you'll see the same four failure modes by case 30); **200–300 cases before you tune thresholds**, because you're now estimating a rate at a specific operating point and you want ±3–4pp; **and a frozen holdout of ~100 you never look at**, opened once at the pilot readout. If you tune against the same 50 cases for six weeks you have overfit to them and your reported number is fiction.

**⚠ Trap — reporting a single accuracy number on 50 cases without the interval, then having it fall by 6 points on the next batch and looking like you regressed.** You didn't regress; you were always inside the band. State the interval from day one and the second number is a confirmation instead of an incident. This is the cheapest credibility protection in the job.

**🗣 Say this in the room:** "Fifty cases gives us roughly plus-or-minus eight points at ninety percent, so it's enough to decide go/no-go and to find the failure patterns — it's not enough to tune on. Before we set thresholds I'd want two to three hundred, and I'd want a hundred we never look at until the readout."

### Walk me through building the bespoke eval itself. What does the scoring actually look like for a messy real task?

Start from the shape of the output, because that determines the scoring, and most enterprise tasks are not free-form generation — they're structured extraction plus a decision plus a justification. Score those three parts separately, because they fail for different reasons and a blended score hides which.

**Structured fields** get exact or normalized match: claim amount, policy number, date of loss, jurisdiction. Normalize aggressively (dates, currency, casing, whitespace) and report **per-field** accuracy, never averaged — a system at 99% on policy number and 61% on date-of-loss is a very different system from one at 80% on both, and only the per-field view tells you where to spend a week.

**The decision** (triage category, approve/deny/escalate) gets a confusion matrix against the expert label, and you report the classes separately. The cost of the errors is asymmetric — a false "auto-approve" on a fraudulent claim is catastrophic; a false "escalate" costs four minutes. So I report **precision on the auto-action class** as the headline safety number and recall as the coverage number, and I never report a single F1 to a customer because it averages away the asymmetry they care about.

**The justification / free text** is where LLM-as-judge earns its keep, but only against a rubric derived from the labeling session, and only after you've validated the judge. Concretely: write the rubric as 3–5 binary checks ("cites a specific policy clause"; "the cited clause actually exists in the retrieved document"; "does not assert a fact absent from the source"), have the judge answer each as a yes/no with the evidence quoted, and then **measure the judge against the humans on the 50 labeled cases**. If the judge agrees with your experts less often than your experts agree with each other, the judge is not usable yet.

Underneath all three sits the cheapest and most valuable layer: **hard assertions**. A cited clause ID must exist in the source document. A dollar amount in the summary must appear in the extracted fields. A date must be within the policy period. These are deterministic checks, they cost nothing, they catch the failure class customers find most alarming (confident fabrication), and they can run in production as guardrails, not just in the eval. In a customer conversation, "we check every citation against the source document programmatically and refuse to show one that doesn't resolve" is worth more than any accuracy number.

**⚠ Trap — a single blended score.** "Our system scores 87%" invites the question "87% of what?" and you will not like where the conversation goes. Report a small dashboard: per-field extraction, decision precision/recall on the automatable class, citation-resolution rate, and the guardrail metric. Four numbers, each actionable.

### The customer has no API. There's a nightly CSV export from a system written in 1998, and one guy named Raj who understands the schema. Walk me through it.

Honestly? This is the normal case, and my first reaction is mild relief, because a nightly export removes an entire class of problems: no rate limits, no auth token rotation, no production write path, no risk of my code taking down their system of record. The instinct to say "we need a real API" is a junior instinct — it converts a solvable data problem into a six-month IT project that isn't yours.

The plan. **Day one: get one file.** Not a spec, not a schema doc — one real file, de-identified, over whatever channel legal permits. Everything you'll learn about that system is in the bytes: encoding (it will be latin-1 or cp1252, not UTF-8), line endings, whether the delimiter appears unescaped inside free-text fields, how nulls are represented (empty string, `NULL`, `\N`, or a single space), whether dates are `MM/DD/YY` with a 2-digit year and a Y2K windowing rule, and whether the "amount" column has been silently truncated to 2 decimal places or is in cents.

**Day two: an hour with Raj, recorded.** Raj is the highest-value asset in the engagement and he will retire or leave. What I want from him is not the schema — I want the *tribal semantics*: which columns are actually populated, which are abandoned, what the status codes mean, which combination of flags means "this row is a correction of a previous row," and what the known data-quality landmines are. Every legacy system has three or four "oh, you have to ignore rows where type = 7, those are ghost records from the 2011 migration" facts, and they are not written down anywhere.

**Then: build a profiling report before you build a pipeline.** Per column: null rate, cardinality, top-10 values, min/max, and length distribution. This takes an hour with pandas and it is the artifact that makes the customer trust you, because you will find things they didn't know. I have delivered a column-profile report on day three and had the customer's own data team ask for a copy.

Architecturally, treat the export as an event source: land raw files immutably with a checksum and an ingestion timestamp, parse into a typed staging table, and make every downstream step reproducible from the raw file. The freshness constraint (data is up to 24 hours stale) then becomes a *product* constraint you surface honestly — it may rule out the interactive use case entirely, and finding that out in week one is a win.

**⚠ Trap — assuming the export is complete and idempotent.** Nightly exports are frequently deltas mislabeled as snapshots, or snapshots with a silent row cap, or they re-export corrected records with the same primary key and a new timestamp. Check: does row count roughly match their stated volume? Do primary keys ever repeat across files? Reconcile your ingested count against a number they can verify — total claims opened last month — and show them the reconciliation. That single check has caught a missing 8% of records for me more than once.

### How do you handle permissions? The customer wants search over their internal documents, and not everyone can see everything.

This is the constraint that most often turns a two-week demo into a one-quarter project, and it is the one that gets waved away in interviews. The mental model to lead with: **an embedding index is a permission-laundering machine.** Text goes in from documents with wildly different ACLs and comes out as a similarity score with no memory of who was allowed to read it. If you don't design the permission model in, you have built a system whose entire function is to leak the salary spreadsheet into an answer for an intern.

There are exactly three workable designs and I'd name them in this order.

**Pre-filter (the only one I'd ship by default):** at query time, resolve the user's identity to a set of group/ACL identifiers, and push that as a hard filter into the vector search itself — every chunk carries the ACL of its source document as a metadata field, and the ANN query is constrained to chunks the user can read. Correct, and it works with every serious vector store. The cost is that heavy filtering degrades ANN recall — if a user can see 2% of the corpus, an HNSW search constrained to that 2% either has to over-fetch dramatically or degrade to brute force over the permitted subset. Budget for it: over-fetch k by 5–20× and re-filter, or partition indexes by major security boundary.

**Post-filter:** retrieve top-k globally, then drop what the user can't see. Simple, and wrong in two ways — you leak result *counts* and timing, and more importantly you get silent quality collapse for restricted users when all 20 hits are filtered away and the model answers from nothing.

**Index-per-tenant/boundary:** the right answer when boundaries are coarse and legally hard (per-client matters in a law firm, where cross-contamination is a malpractice event, not a bug). Expensive in index count, but the isolation is structural rather than a filter you might get wrong.

Two things people forget. **Permissions are dynamic** — someone leaves the deal team on Tuesday and your index still says they can read it, so you need ACL sync with a defined staleness bound, and you must state that bound to security review ("ACL changes propagate within 15 minutes; for immediate revocation we check live at query time for the top-k documents"). And **the ACL must be enforced at generation time too**, not just retrieval: if any part of the pipeline (a summary cache, a "related documents" feature, a conversation history shared between users) crosses the boundary, you've leaked.

**⚠ Trap — the shared cache.** A semantic cache or a summary cache keyed on query text alone will happily serve user B the answer computed from user A's documents. Every cache key in a permissioned system must include the ACL context, or the cache must live strictly downstream of the filter. This bug is invisible in testing because your test users all have admin.

**🗣 Say this in the room:** "Permissions have to be a hard filter inside the retrieval query, not a filter on the results, and every cache key has to include the user's access context. The failure mode otherwise is that the system works perfectly and occasionally shows someone a document they'd be fired for reading."

### Their API is half-documented, the sandbox doesn't match production, and rate limits are undocumented. How do you de-risk the integration?

Treat their API the way you'd treat a third-party payment provider you don't trust: assume the docs are aspirational and the sandbox is a lie, and make the discovery cheap and early rather than accurate and late.

The concrete sequence. **Get one real response body in week one**, from production if legally possible, from a screenshot-and-retype if not. Docs describe the intended shape; the payload describes the actual shape, including the fields that are documented as required and are null 30% of the time, and the enum with an undocumented eighth value. Write a contract test against that real payload immediately and run it on every build — it's the only thing that will tell you when they change something without telling you, which they will.

**Probe the limits empirically, with permission.** Ask "who owns this API and can I have 20 minutes with them," then ask that person the four questions that matter: what's the actual rate limit and is it per-token or per-IP; what's the p99 latency and does it change during the nightly batch window; what happens on overload — 429, 500, or a slow 200 with truncated data; and is there a maintenance window. The truncated-200 case is the one that will silently corrupt your pipeline, and only a human will tell you about it.

**Assume the sandbox differs and plan a production smoke test as a milestone.** Sandbox environments in enterprises typically have a 6-month-stale schema, different auth, and 1/1000th the data volume, which means every scale and every edge case is invisible there. I put "first successful production read of one real record" on the plan as an explicit dated milestone with a named owner on their side, because it is the moment the integration risk actually retires.

**Then design around them, not through them.** Own your side: idempotent, resumable, checkpointed pulls; a durable landing zone so you never re-hit their API for data you already have; a circuit breaker so their degradation doesn't become your incident; and a backfill path that's decoupled from the live path. All of this is your existing backend competence and you should say so — it's one of the few places in this round where your background is directly, obviously valuable.

**⚠ Trap — building the integration before proving the model can do the task.** If the model can't read the documents well enough, a perfect integration is worth zero. Hand-feed data first (see the sequencing rule), integrate second. I've seen a team spend three weeks on OAuth against a legacy IdP for a use case that turned out to be infeasible.

### Palantir-derived FDE lore says you should build something useful within 48 hours of landing onsite. Is that real, and what do you actually build?

It's real, and it's not about heroics — it's about **converting an abstract vendor relationship into a concrete working relationship before anyone's enthusiasm decays.** The 48-hour artifact buys you three things you cannot buy any other way: it proves you can touch their data (which means access actually works, which is the #1 schedule risk retired on day two), it gives your champion something to forward internally, and it changes the conversations you get to have — people bring you their real problems once they've seen you produce something, and they don't before.

What I actually build is almost never AI. In order of how often it's the right answer:

1. **A profiling report on their real data.** Row counts, null rates, cardinality, top values, the distribution of case types by volume, and three anomalies. Costs two hours. Customers are consistently astonished, because nobody has looked at their own data in years.
2. **The current-state process map with their numbers on it**, which turns a hallway conversation into a shared artifact they will use in meetings you aren't in.
3. **Thirty real cases hand-run through a model, in a spreadsheet, with a column for the model output and an empty column for "is this right?"** This is the highest-value 48-hour AI artifact by a wide margin. It answers the feasibility question, and it doubles as the start of the eval set.
4. **A one-page memo** with the restated problem, the constraints found, the two-week slice, and the list of things you need from them with dates and names.

Note what is *not* on the list: a deployed service, a UI, an architecture diagram, a pipeline. Those are week-three artifacts and building them in 48 hours means you skipped the discovery.

**⚠ Trap — mistaking the 48-hour ethos for "ship a prototype fast."** The point isn't speed of building, it's speed of *learning and of establishing credibility*. A polished prototype built from assumptions is worse than a spreadsheet of 30 real cases, because the prototype forecloses the conversation where they tell you your assumptions were wrong.

**🗣 Say this in the room:** "Inside the first two days I'd want something on their screen made from their own data — usually a data profile and thirty real cases run through a model with a blank column for their expert to mark up. Not because it's impressive, but because it's the fastest way to find out what I've got wrong."

### How do you price this? Your own cost of goods is token-metered and the model prices keep dropping.

Two separate questions and I'd untangle them explicitly: what the *engagement* costs (people-time, which dominates) and what the *steady-state system* costs (tokens and infra, which is usually a rounding error and which people wildly over-worry about).

Do the token math first and out loud, because the result is usually a relief and it kills a whole category of customer anxiety. Take the claims case: 40,000 claims/month, each needing roughly 8,000 input tokens (claim record + retrieved policy clauses + system prompt) and 800 output tokens. That's 40,000 × 8,000 = 320M input tokens and 32M output tokens per month. At a mid-tier frontier price of, say, $3/Mtok input and $15/Mtok output: 320 × $3 = $960 plus 32 × $15 = $480, so **$1,440/month**. Against $636,000/month of adjuster labor, inference is **0.23% of the cost of the process it's automating**. **📅 Volatile:** per-token prices move constantly and downward — re-derive with the current published rates before your loop rather than quoting these.

That arithmetic is the single most useful thing you can say to a CFO-adjacent buyer, because their mental model is "AI is expensive." It isn't; *people* are expensive, and the correct framing is cost-per-resolved-case versus cost-per-human-touch.

Then the levers, so the number is defensible rather than a guess: the ~5.5k-token system-plus-policy prefix is identical across calls, so **prefix caching** — the discount on cached input runs roughly 50–90% depending on the provider — takes that portion from $3/Mtok down toward ~$0.30/Mtok at the best end of that range; if ~5.5k of the 8k input is shared prefix, a 90% cache discount cuts roughly 60% of input cost, taking the $960 to around $370. If it's batch (overnight), the provider batch tier at roughly 50% cuts it again. And routing the easy 70% to a small model at ~1/10th the price cuts more. Present it as: base cost, then three levers with the arithmetic, then the honest statement that this is not where the money is.

For engagement pricing, the model I'd argue for is **staged fixed-fee with a gate** rather than time-and-materials or outcome-based. Outcome-based pricing sounds appealing and is a trap in year one: attribution is contested, the baseline is disputed, and you've made your revenue depend on the customer's change management, which you don't control. Fixed-fee stages give them a bounded downside and give you a defined scope.

**⚠ Trap — pricing on the demo's token usage.** Demos run at low volume with generous retries, long contexts and the biggest model. Production runs with routing, caching, truncation and a smaller model on the easy path. Extrapolating demo cost per call to production volume routinely overstates the bill by 5–10×, and I've seen that inflated number get a project killed in a finance review. Always present production cost with the levers applied, and state which levers you've already validated.

### They want us to write back into the system of record — auto-post the adjudication into the claims platform. How do you think about that?

I'd push back on doing it in phase one, and I'd give the reason in risk terms rather than engineering terms, because the buyer only hears risk.

The asymmetry is stark. A read-only system that's wrong produces a bad suggestion that a human ignores. A write-enabled system that's wrong produces a **wrong record in the system of record**, which in a regulated shop means an audit finding, a possible regulatory notification, a manual remediation project, and a customer-facing letter that has to be retracted. The blast radius of a write is not 10× a read, it's 1000×, and it converts a quality problem into a compliance incident.

So the ladder I propose, and I'd present it as a sequence they'll get to walk up rather than a refusal: **(1) Suggest** — output lands in a queue or a side panel, human takes the action. **(2) Pre-fill** — output populates the form fields, human reviews and clicks submit. This is where most of the value actually is: the human is still the actor of record, the audit trail is unchanged, and you've removed the typing, which is most of the touch time. **(3) Auto-act with human sampling** — the system writes, and a defined sample (say 10%, weighted toward low-confidence) gets reviewed after the fact, with the review rate tied to the measured error rate. **(4) Auto-act on a narrow, well-evidenced slice** — only after the eval and pilot data justify it for a specific sub-population.

When you do build the write path, the engineering is straight backend and you should say so crisply: idempotency key per case so a retry can't double-post; a reversal/compensating path designed *before* launch, not after; a kill switch that's a config flag, not a deploy; every automated write tagged with the model version, prompt version and confidence so that when you find a systematic error you can query exactly which records are affected. That last one has saved projects — "we can identify all 1,240 affected records in one query" is a very different conversation from "we don't know which ones."

**🗣 Say this in the room:** "I'd want to earn the write path rather than start with it. Phase one, we pre-fill the form and your adjuster clicks submit — that captures most of the time savings and keeps your audit trail exactly as it is today. Once we've got several thousand cases of measured agreement, we can talk about which narrow slice earns full automation."

### The customer's document corpus is a mess — duplicates, superseded policy versions, three copies of the same handbook. How does that change your design and your promises?

This is the most common cause of the "the AI gave a wrong answer" complaint in enterprise deployments, and the crucial framing to deliver early is that **it is a data problem wearing a model costume.** If three versions of the underwriting manual are in the corpus and two are obsolete, a perfect retriever and a perfect model will confidently give you a correct answer from a policy that was retired in 2022. No amount of prompting fixes that. Saying this clearly, before it happens, is one of the highest-trust moves available.

Design consequences, concretely. **Every chunk carries provenance metadata**: source system, document ID, version, effective date, expiry date, and status (current/superseded/draft). **Retrieval filters on validity by default** — superseded content is excluded unless the user is explicitly asking a historical question, which is a different mode. **The answer surfaces the provenance in the UI**: document name, version, effective date, and a deep link. Enterprise users don't trust an answer, they trust a citation they can click, and the click-through is also your best quality telemetry — a high click rate with a low "was this helpful" rate means retrieval is finding the neighborhood but not the answer.

Then the deduplication and staleness work, which is unglamorous and is where a chunk of the pilot budget actually goes: near-duplicate detection (hash the normalized text, then embedding-similarity clustering above a threshold, then keep the newest by effective date), a decision about drafts and personal copies in the file share (my default: exclude anything outside the designated system of record in phase one, and say so), and an ownership question you must ask out loud — **"when the handbook is updated, who tells the system?"** If the answer is "nobody," you've just found the reason the pilot degrades in month four.

**🔍 Failure taxonomy — stale-answer triage, in order:** (1) Is the retrieved chunk from the current document version? If no → provenance/filter bug, most common. (2) Is the current version even in the index? If no → ingestion lag; check the sync job and its dead-letter path. (3) Did retrieval find the right document but the wrong section? If yes → chunking boundary problem; check whether the answer spans a chunk break. (4) Did retrieval find it and the model ignored it? If yes → context ordering or prompt problem, and check whether the relevant chunk was ranked below position ~10 in a long context. (5) Only if all four pass is it a model reasoning failure — which, in my experience, is maybe 15% of reported "hallucinations" in enterprise search. The other 85% are pipeline bugs, and telling a customer that honestly is how you keep their trust when the first bad answer arrives.

**⚠ Trap — promising a "single source of truth" you don't control.** You can promise the system will only answer from documents in the designated repository, with the version stamped on the answer. You cannot promise the repository is correct. Draw that line explicitly, in writing, in the pilot success document, because when a wrong-but-faithfully-retrieved answer surfaces, that line is the difference between "the AI is broken" and "we found a document-governance gap, which is worth knowing about."

### Their security review is 12 weeks and they say data can't leave their tenant. What happens to your plan?

First reaction: this is a *sequencing* fact, not a *feasibility* fact, and my job is to keep the technical work moving while the paperwork runs, because the failure mode is a team idling for three months and losing the champion's momentum.

The two-track move. **Track A, immediately:** get a de-identified or synthetic-but-real-shaped sample under a narrow one-off approval — usually far easier than a full vendor review, because it's a data-sharing question with a bounded scope rather than an architecture question. With 200 de-identified cases you can build the eval, prove feasibility, and produce the feasibility memo, all of which happen in the window where you'd otherwise be blocked. **Track B, in parallel from day one:** the full review, with you actively helping — offering the architecture diagram, the data-flow document, the sub-processor list, the retention settings, and the specific contractual terms about training on customer data, before they ask. Vendors who arrive with a completed security package cut weeks off, and this is a genuinely differentiating behavior.

Then the architecture branch, which you should name as a decision with a cost, not a preference. If data truly cannot leave their tenant, your options are: a model provider available inside their cloud tenant (the major clouds all offer first-party and partner model endpoints within a customer's own subscription, with the data-flow story that implies); a self-hosted open-weight model on GPUs in their environment; or a hybrid where de-identified or field-level-redacted content goes to an external API and the sensitive join happens locally. Each has a real cost you should state: self-hosting adds GPU capacity planning, an inference-serving stack to operate, a quality gap versus frontier models on hard reasoning, and an ops burden the customer may not want — but it can be the only path in some regulated environments.

**💰 Math on the self-host branch, so it isn't hand-waving:** at 40,000 claims/month with an 8k-token input, that's 320M input tokens/month ≈ 10.7M tokens/day ≈ 124 tokens/second sustained if spread evenly — which a single modern 8×GPU node serves with enormous headroom, but you're paying for the node continuously. A dedicated multi-GPU instance in a major cloud runs on the order of $10–30/hour on-demand, so ~$7,000–22,000/month of committed capacity versus roughly $1,440/month of API tokens at frontier prices. **📅 Volatile:** instance pricing and token pricing both move; re-derive. The point that survives: self-hosting at this volume is 5–15× more expensive *and* adds an ops team, so it should be chosen for compliance reasons, stated as such, and never sold as a cost saving. Customers respect being told that plainly.

**🗣 Say this in the room:** "Twelve weeks of security review doesn't have to be twelve weeks of nothing happening. I'd ask for a narrow approval on two hundred de-identified cases so we can prove feasibility and build the scorecard while the full review runs, and I'd want to hand your security team the data-flow doc and sub-processor list in week one rather than waiting for the questionnaire."
### Tell me about the customer-conversation simulation round. What separates the people who pass from the people who don't?

The format: an interviewer role-plays a specific non-technical stakeholder — a VP, a GC, a head of operations — with a specific agenda, and they push. They will ask for something impossible, or express disappointment, or misunderstand what the system does in a way that's flattering to you and dangerous to them. It's typically 30–45 minutes and it is reported (candidate-reported, not published — **📅 Volatile**) to eliminate a majority of candidates who already cleared the technical stages, which tells you it's testing something orthogonal to engineering skill.

What it's testing is a narrow, specific competence: **can you be simultaneously honest and useful when those two pull apart.** Nearly every failure I've seen is a collapse to one pole. **Capitulation** is agreeing that yes, we can get to 100%, yes we can auto-deny claims, yes we'll have that by end of quarter — because disagreeing feels rude and because engineers are conditioned to be accommodating with stakeholders. **Lecturing** is the other pole: responding to "why did it say something different the second time?" with a paragraph about temperature and sampling. Both fail, and they fail for the same underlying reason — neither one leaves the customer with a *decision they can make*.

The pass pattern has a consistent shape, and I'd drill it as a four-move sequence: **acknowledge the real concern under the question** (not the literal question), **give the honest mechanism in one plain sentence** (no jargon, no more than one sentence), **name the consequence for them specifically**, and **offer the alternative you'd stand behind.** Never end on the bad news. Ending on the bad news is what makes honesty feel like an attack; ending on an offer makes the same honesty feel like partnership.

The second thing being scored is whether you can hold a position under three rounds of pressure without either hardening into defensiveness or softening into agreement. The interviewer will push a third time specifically to see which way you break. The correct behavior on the third push is to become *more concrete*, not more insistent — offer a test, a number, a pilot design, a small experiment that would resolve the disagreement with evidence.

**⚠ Trap — treating the roleplay as a roleplay.** Candidates who narrate ("well, I'd probably tell them that...") instead of speaking in character score badly, because the round is testing delivery, not knowledge of what to deliver. Speak directly to the person. Use their name. Say "you" and "your team."

**🗣 Say this in the room (as the opening frame when they start pushing):** "I want to give you the honest version even where it's not what you were hoping, because if I over-promise now you'll find out in eight weeks and that's a much worse conversation for both of us."

### I'm the VP of Claims. I asked your system the same question twice and got two different answers. That seems broken. Explain it to me.

"It's not broken, but I understand why it looks that way, and it's a real thing you need to plan around rather than something I'd wave off.

These systems generate text one piece at a time, and at each step there are several reasonable next words. The model picks among them with some randomness — that's what lets it write fluently rather than robotically, but it means two runs of the same question can phrase the answer differently, and occasionally reach a different conclusion on a genuinely borderline case.

For you, what matters is which parts I can make stable and which I can't. **The decision and the extracted fields I can make effectively deterministic** — I turn the randomness down to zero for those, I constrain the output to a fixed set of values, and I check them against your source documents programmatically. If it says the claim amount is $4,200, that number came out of your record and we verify it did. **The wording of the explanation will vary**, and honestly that's fine — your adjusters don't need identical prose, they need the same call and the same evidence.

Where I'd focus your attention instead: not 'does it say the same thing twice,' but 'does it agree with your best adjuster.' That's what we measured on the fifty cases your team labeled, and that's the number I'd hold myself to. If you want, I'll add a check to the pilot where we run two hundred cases three times each and report how often the *decision* changes — my expectation is under 2%, and if it's higher than that, that's a defect I own and fix."

**⚠ Trap — saying the word "temperature."** Or "sampling," "stochastic," "top-p," or "nondeterministic." Every one of these makes the customer feel talked down to and none of them changes what they'll do. The plain-language version — "it picks among several reasonable next words, and I can turn that off for the parts that matter" — carries the same information and reads as mastery rather than jargon.

**⚠ Trap #2 — over-promising determinism.** Do not say "I'll set temperature to zero and it'll be deterministic." Greedy decoding reduces variation dramatically but does not guarantee bit-identical output across a fleet — batching, hardware, kernel selection and provider-side model updates all introduce variance, and a silent provider model update can shift behavior overnight. Promise *decision stability, measured*, not determinism. Then pin model versions where the provider allows it and put "provider changed the model under us" in the risk register, because it will happen.

### The same VP now says: "My team found an answer that was completely made up. It cited a policy section that doesn't exist. How do I trust anything this thing says?"

"That's the right thing to be alarmed about, and I want to tell you exactly what happened and exactly what we do about it, because 'trust it more' is not an answer.

The model is a very good pattern-completer. When it's asked to produce a citation and the retrieved documents don't contain a clean one, it will sometimes produce something that *looks* exactly like your citations — right format, plausible section number — because producing plausible-looking text is precisely what it's built to do. It has no internal sense of 'I don't know this one.' That's a genuine limitation, it's not a bug I can patch, and any vendor who tells you they've eliminated it is selling you something.

What we do instead is make it impossible for that to reach a user. Every citation the system produces gets checked against the actual document before it's shown — the section has to exist, in the version that was current on the date of loss, and the sentence it's supporting has to appear in the retrieved text. If the check fails, we don't show the answer; the case routes to an adjuster with a note saying we couldn't substantiate it. That turns an invisible failure into a visible one, which is the whole game.

Two things I'd ask of you. First, I want that specific case — send it to me and I'll tell you within a day whether it was the checker missing, the document not being in the index, or something else, and it goes into the scorecard permanently so it can't come back. Second, I'd like us to agree on the guardrail number: what fraction of cases getting routed to a human because we couldn't substantiate them is acceptable to you? Because I can make that number as close to zero as you like by being more cautious, and the cost is coverage. That's the actual trade, and it should be your call, not mine."

**⚠ Trap — the two failure responses.** Response one, defensive: "well, all LLMs do that." True, useless, and it sounds like you don't care. Response two, over-correcting: "I'll fix it so it never happens." You can't, and you've now written a check that bounces at the readout. The third path — *here's the mechanism, here's the containment, here's the number we should agree on together* — is the only one that survives a follow-up.

**🗣 Say this in the room:** "I can't make the model incapable of producing a wrong citation. I can make it structurally impossible for an unverified citation to reach your adjuster, and I can show you the rate at which that check fires. That's the honest version of 'trustworthy.'"

### The demo two weeks ago looked amazing. The system now in front of real users looks noticeably worse and the customer is unhappy. Handle this conversation.

The mental model I bring, and I'd say a version of it out loud: **a demo measures the best case; production measures the distribution.** The gap between them is not a regression, it's the demo having been a biased sample — and the mistake that created this situation was mine, at the demo, for not framing it as a sample.

The conversation, in order. **Own it first and specifically**, because a customer who senses deflection stops listening: "The demo was on twelve cases that I chose, and I should have said clearly at the time that those weren't a random draw. That's on me and it's why you're surprised now." That sentence costs nothing and buys the rest of the meeting.

**Then convert the vibe into a number**, immediately, because "it feels worse" is unmanageable and "it's at 71% versus the 89% you saw" is a project. "Here's what we're seeing on the two hundred cases from the last ten days: 71% agreement with your adjusters, versus 89% on the labeled set. So there's an eighteen-point gap and I've spent two days on where it comes from."

**Then the diagnosis, in clusters, with proportions**, because customers can hold three buckets and cannot hold seventeen anecdotes: "Sixty percent of the gap is one thing — claims with attached scanned handwritten notes, which weren't in our original sample at all and are 14% of your real volume. About a quarter is a document type our ingestion is silently dropping. The rest is spread out."

**Then the offer with dates and a fork:** "Two of those three I can fix in this pilot and I'd expect us back above 85% on the routable subset in three weeks. The handwriting one I don't think is worth fixing — I'd rather we route those to a human by rule and stop pretending. That drops coverage from 62% to 53% of your volume and I think that's the right trade. Here's what that does to the business case."

**⚠ Trap — the demo that created this problem.** The real fix happens two weeks earlier: never demo without stating the sample. "These are twelve cases drawn at random from the export you sent, including the two hardest ones your team flagged; on the full fifty we're at 84%." One sentence, and you have inoculated yourself against this entire conversation. I would rather show a slightly worse demo with an honest denominator than a perfect one without.

**💰 Math for the fork above:** at 40,000 claims/month, dropping coverage 62% → 53% costs 0.09 × 40,000 × 13/60 hr × $53 = **$41,300/month** of forgone savings. Fixing handwriting recognition to a usable standard is realistically 4–6 engineer-weeks plus an OCR vendor evaluation, call it $60k of effort against $496k/year of recovered value — so it's positive, but it's a *phase two project with its own gate*, not a pilot-scope patch. Presenting the fork with both numbers is what makes the recommendation land as judgment rather than as retreat.

### The customer asks for something you think is genuinely unsafe — "can it just auto-deny the claims it flags as likely fraud?" Handle it without capitulating and without lecturing.

The move here is to **redirect from capability to consequence**, because arguing about capability invites "well, can it or can't it," which you lose either way. Reframe onto the thing they actually own: risk.

"It can produce that flag, and it'll be reasonably good at it. What I'd push back on is the auto-deny, and let me give you the specific reason rather than a general one.

If we're at 92% precision on the fraud flag — which would be a strong result — then on your volume, roughly one in twelve denials is a legitimate claimant getting denied by a machine with no human in the loop. At the fraud-flag rate you described, that's a few dozen people a month. Every one of those is a complaint, an appeal, potentially a regulator letter, and if it clusters in a way that correlates with a protected class, it's a much bigger problem than any of that. And denials are the one action here that is hard to walk back — you can always approve a claim later, but a wrongful denial has already reached your customer.

So here's what I'd build instead, and I think it captures most of what you want: the flag goes into a dedicated review queue with the evidence assembled — the specific patterns matched, the prior claims history, the documents pulled and highlighted. Your SIU investigator gets a case that's pre-built instead of one they have to construct from scratch. From what your team described, that's the majority of the time cost on a fraud review. We'd be automating the investigation prep, not the decision.

If in six months the data says the top-confidence decile is running at 99%+ precision against your own investigators' conclusions, then we have an evidence-based conversation about auto-actioning that specific slice. I'd want the evidence first. Would that work as the phase-one shape?"

**⚠ Trap — moralizing.** "That would be unethical" ends the conversation and gets you scored as someone who can't be put in front of a customer. The reasoning that persuades a VP is *their* exposure: appeals, regulators, brand, reversibility. Say the safety thing in the language of risk they already manage. It's not a compromise of the principle; it's the version of the principle that actually gets adopted.

**⚠ Trap #2 — the flat no.** "We can't do that" without an alternative reads as an inability to solve their problem. Every refusal in this round must come with a counter-offer that captures a defensible fraction of the value. The senior formulation is always "not that — this, and here's the path to that."

**🗣 Say this in the room:** "I'd separate detection from decision. I'll automate everything up to the decision, including assembling the evidence, and I want a human on the irreversible action until the data earns the alternative."

### The stakeholder says "it just needs to be 100% accurate, that's non-negotiable." Where do you go?

I go to the human baseline, immediately, because it's the only move that reframes without contradicting them.

"Totally fair, and I want to make sure I'm aiming at the right target. Can I ask — when two of your senior adjusters look at the same borderline claim today, how often do they reach the same call?

[They'll say something like: honestly, maybe 85-90%.]

Right, and that's normal — it's not a criticism of your team, it's that these are judgment calls. So a hundred percent isn't a bar that exists anywhere in this process today, including for people. What I think you actually mean, and tell me if I'm wrong, is: *it must not make a category of mistake that costs us badly, and when it's unsure it must not pretend otherwise.* That I can build.

Concretely, that turns into three commitments. On the irreversible actions — denials, payments over your threshold — a human decides, always, so the system's error can't become your error. On the automated slice, the system only acts when it's confident, and we set that threshold from measured data on your own cases, so you're choosing the operating point rather than accepting mine. And on the mistakes that do happen, we detect them: everything the system does is logged with its inputs and its version, so when something's wrong we find every other case like it in one query instead of wondering.

That's the shape I'd sign up to. The number I'd want us to agree on isn't accuracy, it's: *of the cases we automate, we agree with your experts at least 97% of the time* — and then we compete on how much volume we can bring under that bar. Does that feel like it protects what you're worried about?"

**⚠ Trap — arguing about whether 100% is achievable.** You will be technically right and lose the room. Never contest the number; excavate the fear underneath it. The fear is always one of three things: an irreversible harm, a regulatory exposure, or personal career risk for the person asking. Address the one they have.

**📐 Numbers you must know:** human inter-annotator agreement on genuinely subjective enterprise tasks commonly sits in the 70–90% range — support-ticket intent classification, claims severity triage, legal document relevance. Measure it in your labeling session (two SMEs, 15–20 overlapping items) and you will have a real number for *this* customer. It is the single most useful number in the entire engagement, because it converts "the AI got it wrong" into "the AI disagreed with one of your experts, and so does your other expert, 18% of the time."

### Say the sentence "this use case is not going to work." Then tell me what you offer instead.

The willingness to say it is the whole point of the question, and the way you say it determines whether you keep the account.

Here's a real shape of it. The customer wants a system that reads incoming claim documents and predicts which claims will end up in litigation, so they can staff early. The honest assessment: the signal for litigation is mostly *not* in the claim document — it's in claimant behavior over subsequent weeks, whether an attorney appears, and jurisdiction. A model reading intake documents will produce a confident-looking score with very little predictive power, and it'll be worse than the actuarial baseline they already have.

"I don't think this one works, and I'd rather tell you now than in ten weeks. The reason isn't the model — it's that the information you'd need to predict litigation mostly doesn't exist yet at intake. Whatever we build will look like it's working and won't beat the rules your actuarial team already uses, and my guess is we'd spend a quarter proving that.

What I'd redirect the effort to, if the underlying goal is reducing litigation cost: two things in the same neighborhood that I think do work. One, first-contact quality — a large share of litigation starts with a slow or contradictory first response, and drafting a consistent, complete, fast first response is very much in scope for this. Two, once an attorney letter *does* arrive, the intake and summarization of that correspondence is a real time sink and a clean automation.

If you want me to be wrong about the prediction piece, here's the cheap test: give me eighteen months of closed claims with the litigation outcome and I'll spend a week seeing whether intake-time features carry any signal at all above your baseline. A week, and then we both know. But I wouldn't build it on the assumption that they do."

Three things that make this work. It **arrives early** — killing something in week two is professional, killing it in week ten is a failure. It **redirects to adjacent value** so the meeting doesn't end on a subtraction. And it **offers a falsification test with a bounded cost**, which lets the customer disagree with you without a fight and makes you look confident rather than pessimistic.

**🗣 Say this in the room:** "I'd rather lose this piece of scope than lose your team's confidence in the next three things I tell you will work."

### The champion wants to skip the labeled evaluation and go straight to a production pilot — there's a board deadline. What do you do?

I don't refuse, because refusing to a person under executive pressure just gets you routed around. I make the eval *smaller and faster* so it stops being the thing blocking them, and I make the cost of skipping it concrete and personal to them.

"I hear the deadline and I don't want to be the reason you miss it. Let me propose a version that costs you three days instead of two weeks.

Instead of two hundred cases, we do fifty, in one ninety-minute session with two of your adjusters this week. I'll drive, they'll accept-or-correct pre-filled answers, and I'll have the scorecard by Thursday. That's it — three days, four hours of their time, and it doesn't move your date.

Here's why I'm pushing for even that small version. At the board readout someone will ask 'how well does it work?' If we have the scorecard, you answer with a number your own team produced and the conversation moves on. If we don't, the honest answer is 'it looked good in the cases we tried,' and in my experience that answer gets followed by someone in the room trying a case live, hitting an edge case, and the demo becoming about that one case. The scorecard is mostly protection for *you* in that room.

And if we genuinely can't get the adjusters this week, here's the fallback: we run the pilot, and I instrument it so every single case a user overrides is captured with the correction. That builds the labeled set as a byproduct of the pilot — we'll have four hundred labeled cases in three weeks instead of fifty in three days. It's a worse position going into the board meeting but it's not nothing, and I'd want your explicit sign-off that we're accepting that trade."

Two moves in there worth naming. **Shrink rather than refuse** — the eval's value is highly non-linear and 50 cases captures most of it, so trading rigor for schedule is a legitimate senior call. And **make the risk land on the person who owns it**, factually and without threat: "here's what happens to you in that room." Then get the trade acknowledged out loud, which is not CYA — it's the thing that keeps you from being blamed in six weeks for a decision you flagged.

**⚠ Trap — the eval that never gets built because the pilot succeeded.** If phase one ships without a labeled set, you will never get one, because success removes the urgency and failure removes the goodwill. Build the override-capture instrumentation into the pilot on day one so the labeled set accumulates whether or not anyone prioritizes it.

### Translate an eval result into something the buyer cares about. Show me.

The failure mode here is reporting a model metric to a person who has no way to convert it into a decision. The bridge is always the same three-step chain: **quality metric → operating policy → business quantity.** You have to walk all three out loud, because the middle step is where the judgment lives and it's the step candidates skip.

Take a concrete result: on 300 labeled support tickets, the system produces a resolution the customer's own senior agents rate as acceptable on 78% of cases; its self-reported confidence, calibrated on a holdout, lets us pick a threshold at which 46% of tickets are handled with 94% acceptable quality.

Now the translation, said to a VP of Support:

"Of your incoming tickets, we can fully resolve about 46% without an agent, at a quality your senior agents rate as acceptable 94% of the time. The other 54% go to an agent with a drafted response attached, which our timing says saves about 40% of handle time on those.

Your numbers: 120,000 tickets a month, 8 minutes average handle time, $28 fully-loaded per agent-hour. Today that's 120,000 × 8/60 = 16,000 agent-hours a month, or 16,000 × $28 = **$448,000/month**.

Deflected: 0.46 × 120,000 = 55,200 tickets × 8/60 hr × $28 = **$206,000/month saved**. Assisted: 64,800 tickets × 8/60 × 0.40 × $28 = **$96,800/month saved**. Total about **$303,000/month**, or $3.6M a year, against inference cost I estimate at under $6,000/month.

The number I'd hold back from that: on the deflected 46%, 6% get an answer your senior agents wouldn't have given. That's 55,200 × 0.06 = 3,312 tickets a month. Most are 'unhelpful' rather than 'wrong,' and they escalate, which costs you a re-contact — call it 5 minutes each, so 3,312 × 5/60 × $28 = **$7,700/month of rework**, plus a CSAT hit I can't price and would want to measure. So the honest net is around $295,000/month with a customer-experience risk we should track weekly and a threshold we can tighten if CSAT moves."

That last paragraph is why this answer passes. Volunteering the cost of your own errors — with arithmetic — is the most credible thing you can do in front of a buyer, and it preempts the question that would otherwise be asked adversarially at the readout.

**⚠ Trap — headline savings computed as if 100% of theoretical time savings converts to money.** It doesn't. Saved agent-minutes only become dollars if headcount changes or volume growth is absorbed without hiring, and the customer's HR reality may make neither true. The senior framing: "this is $3.6M of capacity; whether it shows up as cost reduction or as absorbing next year's 30% volume growth without hiring is your call, and I'd suggest we write down which one we're claiming before the readout." Buyers respect this enormously because every one of them has been handed an inflated savings model by a vendor.

### The economic buyer loves it. The forty adjusters who have to use it clearly hate it. What do you do?

I treat this as the highest-priority defect in the engagement, above any model quality issue, because a system nobody uses scores zero regardless of its accuracy — and because user hostility is usually *rational* and therefore diagnosable.

First, find out which of the four causes it is, by sitting with three of them for an hour each and watching them work — not by sending a survey. **(1) It's threatening.** They believe this is the prelude to headcount cuts, and nobody has told them otherwise credibly. **(2) It's slower.** The tool adds clicks, or it lives in a second window and they have to retype, or its 6-second latency breaks their rhythm on a task they do 200 times a day. **(3) It's wrong in a way that costs them.** They're measured on quality and the tool's errors land on their record, so using it is personally risky. **(4) It doesn't fit their real workflow**, which differs from the documented one in ways nobody told you.

Each has a different fix and only one of them is a model fix. For (2) the answer is usually integration and latency — putting the output where they already work, streaming so the first token appears in under a second, and pre-computing on case assignment rather than on click. For (3) the answer is a policy change you must negotiate with their manager: **errors originating from an accepted suggestion do not count against the adjuster's quality score during the pilot.** That single sentence has flipped adoption for me more than any model improvement, and I'd say it in the interview because it demonstrates you understand that incentives beat features. For (1) the answer is not yours to give — it's the buyer's, and my job is to tell them plainly that adoption will not happen until someone senior says something specific and believable about jobs.

Structurally, I'd insist on three things from the start of any pilot: **name two or three users as design partners** with their manager's explicit time allocation, ship them something in week two, and put a **usage metric in the success criteria alongside the quality metric** — "≥70% of eligible cases have the assistant opened" — so adoption is a tracked, owned number rather than a surprise at the readout.

**🔍 Failure taxonomy — adoption collapse, as a decision procedure:** check usage telemetry first. If usage is near zero from day one → onboarding/access problem or active resistance; go talk to a manager. If usage started high and decayed → the tool disappointed them; pull the cases where they overrode it and cluster the reasons. If usage is high but override rate is above ~40% → it's producing output they have to fix, which is often *worse* than nothing; consider narrowing to the subset where override is low and turning it off elsewhere. If usage is high and overrides are low but they still complain → it's likely latency or workflow friction, not quality; sit and watch someone use it.

**⚠ Trap — measuring adoption by license count.** Seats provisioned, accounts created, and "monthly active" are all vanity. The only metric that matters is per-eligible-case usage: of the cases where the tool *could* have helped, in what fraction did the user actually engage with it, and what fraction of its output survived to the final answer.

### During a readout, the customer says the model is unreliable — but you know the bad outputs came from a bug in your chunking, not the model. Their CTO is in the room. What do you say?

I correct it, immediately and without defensiveness, because letting "the model is unreliable" stand unchallenged does long-term damage that is very expensive to undo. Once an organization decides "AI doesn't work for us," you don't get that back in this budget cycle, and the belief will outlive your engagement.

But I correct it in a way that doesn't sound like blame-shifting, which means leading with ownership of the *outcome* before correcting the *attribution*:

"Those outputs were bad and that's on us. I do want to correct one thing though, because it changes what we should do about it. Those specific failures weren't the model reasoning badly — they were our pipeline splitting the policy documents at the wrong boundary, so the model was answering from half a clause. It answered correctly given what it was handed; we handed it garbage. I've reproduced it on nine of the eleven cases your team flagged.

That matters because it's a fix with a date on it rather than a fundamental limitation. It's a chunking change plus a re-index, roughly four days, and I'd want to re-run the full scorecard afterward and show you the before-and-after rather than just telling you it's fixed.

It also tells us something we should have caught: our eval set didn't contain any documents with that structure. So I'm adding those eleven cases plus twenty more of that document type permanently, which is how we make sure this specific thing can't come back."

Three moves. **Own the outcome first** — "those outputs were bad and that's on us" — which earns you the right to the correction. **Correct with evidence**, not assertion: "I've reproduced it on nine of eleven." **Close the loop into the eval**, because a fix without a regression test is a promise, and a fix with one is an engineering practice they can verify.

**⚠ Trap — the technically-correct correction delivered first.** "Actually, that's not a model problem, it's a chunking bug" is true and lands as defensive, especially in front of a CTO who is watching how you behave under criticism. Order matters more than content here. Ownership, then correction, then the loop closed.

**🗣 Say this in the room:** "That output was wrong and I own it. The cause was our retrieval splitting the document at the wrong place rather than the model reasoning badly — which I say only because it means it's a four-day fix with a regression test, not a limitation we have to live with."

### Write me the pilot success-criteria document. What's in it, and who signs it?

This is the single artifact the deal is judged against, and the rule I enforce is that **it is written and signed before any production code ships** — because after the results exist, every clause becomes a negotiation and the customer's memory of what "success" meant will have drifted toward whatever you didn't achieve.

It's two pages. Nine sections.

**1. Problem statement** — the restatement they signed, verbatim, with the population definition and the numbers.

**2. Scope in / scope out** — explicit lists. The "out" list is longer and each item has a one-line reason. This section prevents the most expensive class of disagreement.

**3. Baseline** — the current-state numbers, how they were measured, when, and by whom. If the baseline is estimated rather than measured, say so in the document. Un-baselined pilots cannot succeed, because "better than what?" becomes unanswerable.

**4. Primary success metric** — one metric, a threshold, a population, and a measurement window. "Of auto claims meeting criteria X received between Nov 1 and Dec 15, ≥50% complete with zero adjuster touches, at ≥95% agreement with adjuster decisions on a 150-case audit sample."

**5. Guardrail metrics that must not regress** — with their current values and tolerances. Reopen rate ≤ 3.5% (currently 3.1%). Customer complaints on automated claims not exceeding the manual rate. Median cycle time not increasing.

**6. Adoption criteria** — because a technically successful, unused system is a failed pilot. "≥70% of eligible cases have the assistant engaged by week 6."

**7. Measurement procedure** — the data source for every number, who computes it, on what cadence, and the audit-sample protocol including who labels it. This section is what makes the readout uncontestable, and it's the section everyone skips.

**8. Assumptions and dependencies** — the customer-side obligations with dates and named owners. Data access by Nov 3 (Raj). Two adjusters for four hours in week 1 (Denise). Security review complete by Nov 10 (InfoSec). Each with an explicit "if this slips, the pilot end date moves by the same amount."

**9. Decision rule** — what happens at each outcome. Not just "success = expand." Write the three branches: metric met → production rollout with scope Y at price Z; metric partially met → defined extension of N weeks with a specific hypothesis; metric missed → we stop, and here is what you keep (the eval set, the process map, the data-quality findings). Naming the stop condition, in writing, before you start, is the strongest credibility signal available to a vendor, and it is what senior interviewers listen for.

Signatures: the **economic buyer** (owns the money and the decision), the **operational owner** (owns the users and adoption), and **you**. If you can get compliance or security to acknowledge scope, do — it prevents the week-nine ambush.

**⚠ Trap — a success criteria doc with no stop condition and no named guardrail.** Both omissions are optimism, and both make the readout a fight. The doc that says "and if we're below 40% coverage at 95% agreement, we recommend not proceeding, and here's what you'll have gotten for the spend" is the one that gets you the second engagement even when the first one fails.
### Run the insurance claims case live for me, top to bottom, in six minutes. Go.

"Six minutes, so I'll be compressed and you should stop me wherever you want more.

**Questions I'd ask first.** Walk me through one routine auto claim end to end. How many claims a month, what's the average adjuster touch time, what does a fully-loaded adjuster cost, what fraction hit an exception path, and how do you know today when an adjustment was wrong? Then: where does the claim data live, is there an API or an export, and who signs a denial?

**What I'd assume you told me** — 40,000 claims/month, 18 minutes of touch time, ~$53/hour loaded, 62% are routine auto under $5k with no injury, data comes out as a nightly export from the claims platform, and by statute an adjuster signs every denial.

**Stakeholders.** You're the buyer. Ops manager owns the 40 adjusters and owns adoption. Compliance can veto. IT owns the export and is my schedule risk. And the union contract, if there is one, is a constraint I need to know about today, not in month two.

**Restatement:** for the 62% of auto claims under $5,000 with no injury and a police report attached, cut adjuster touch time from 18 minutes to under 6, with no change to who signs the settlement and no increase in the 3.1% thirty-day reopen rate. Is that the problem?

**Not building:** injury claims — different regulatory regime, medical records, and only 8% of volume. No auto-denial. No writes into the claims platform in phase one; we pre-fill the adjuster's form and they submit. No custom UI — we use a queue view and a spreadsheet if we have to.

**Two-week slice:** 200 de-identified claims from the export, hand-triggered. For each: structured extraction of the fields your adjusters key by hand, a recommended action, and a citation to the specific policy clause it relied on with a programmatic check that the clause exists. Scored against 50 cases your adjusters label with me in a 90-minute session in week one. The riskiest assumption it tests is whether the scanned attachments are legible enough — which I'd test on day one by hand-feeding thirty of them before writing any pipeline.

**Success metric:** of routine auto claims meeting the criteria, ≥50% complete with under 6 minutes of adjuster touch, at ≥95% agreement with adjuster decisions on a 150-case audit sample, with 30-day reopen rate at or below 3.5%, and ≥70% of eligible cases having the assistant engaged by week six.

**The business number:** 62% of 40,000 is 24,800 claims; 12 minutes saved each is 4,960 hours/month at $53 = **$263,000/month**, $3.2M/year. Inference is roughly 24,800 × 8,000 input tokens = 198M input tokens plus ~20M output — around **$1,000/month** at current mid-tier frontier pricing, so under 0.4% of the value. Even if I'm wrong by 5× on tokens it doesn't change the decision.

**What would change the plan:** if the scans aren't legible, this becomes an OCR remediation project and I'd want to scope that separately. If IT can't get us a de-identified export inside five business days, everything slides one-for-one. If compliance requires explainability beyond a policy-clause citation, that's a design change I'd want to discuss now."

**⚠ Trap — filling six minutes with architecture.** Notice there is no diagram in that answer and no component list. If the interviewer wants the architecture they will ask, and by then you have earned it. Candidates who lead with retrieval strategy and model choice in a case round consistently score below candidates who never mention either.

### Now the legal one. A firm wants to use AI for contract review in due diligence. Decompose it.

The domain-specific thing to lead with, and it changes everything downstream: **in a law firm, the unit of confidentiality is the matter, and cross-matter contamination is a malpractice event, not a bug.** That fact determines the architecture before any model question — it pushes you toward index-per-matter isolation rather than a filtered shared index, and it means no shared cache, no cross-matter fine-tuning on client documents without explicit consent, and a retention story that matches the engagement letter.

**Current state.** A mid-size deal has a data room with 8,000–40,000 documents. Associates review contracts for a fixed issues list — change of control, assignment, exclusivity, non-compete, MFN, termination for convenience, limitation of liability, governing law — and produce a summary chart plus a risk memo. It's 200–600 associate-hours per deal, done under time pressure by people billing at $400–700/hour, and the quality varies with how tired the associate is at 2am.

**The restatement:** for the eight issues on the standard diligence checklist, produce for every contract in the data room a determination (present / absent / ambiguous) with the exact quoted clause and page cite, at a recall on "present" high enough that a partner will accept it as a first pass — with an associate verifying every flagged item and spot-checking the negatives.

Note the metric shape: **recall on the risky class is the headline, not accuracy.** A missed change-of-control clause in a $400M acquisition is a career event; a false positive costs an associate ninety seconds to dismiss. So I'd tune deliberately toward over-flagging and I'd say that out loud to the partner, because it inverts the usual instinct and demonstrates domain judgment. Target something like ≥98% recall on the issues list against a partner-labeled set, accepting precision in the 60–75% range in phase one.

**What I would not build:** no legal advice, no risk *assessment* — the system finds and quotes clauses, humans assess them. No negative assurance ("this data room contains no MFN clauses") because the liability shape of that claim is completely different from "here are the MFN clauses we found." No drafting of the client-facing memo in phase one.

**Eval:** 40 contracts, partner-or-senior-associate labeled per issue, drawn from three closed deals across different industries so you catch the drafting-style variance that will otherwise ambush you. Plus the hard-negative set: contracts where the clause is *almost* present — a change-of-control clause that only triggers on a majority-of-assets sale — because that's where both models and junior associates fail.

**💰 The business math is different here and you must get it right:** law firms bill hours, so "saving associate hours" reduces revenue unless the work is fixed-fee, or unless it lets them take on more deals with the same headcount, or unless it's a client-driven cost pressure. Ask which. At 400 hours/deal and 40 deals/year that's 16,000 associate-hours; a 50% reduction on the reviewable portion (say 60% of those hours) is 0.6 × 16,000 × 0.5 = 4,800 hours. At $500/hr realized that's $2.4M of billings — which is either a $2.4M revenue *loss* or a $2.4M capacity gain, depending entirely on the firm's business model and their clients' fee pressure. Asking that question is the whole difference between a vendor and a partner in this room.

### Healthcare intake — a provider group wants AI to handle patient intake and referrals. What's your decomposition?

Lead with the bright line, because it defines the whole engagement: **the system does not diagnose, triage acuity, or recommend treatment.** Anything that constitutes clinical decision-making brings in a regulatory regime (device oversight, clinical validation) that no pilot survives, plus a liability exposure no provider group will accept. The value here is entirely in the administrative layer, and there is an enormous amount of it.

**Current state, and it's worse than outsiders expect.** Referrals arrive as faxes — genuinely, in 2026, fax is still a dominant channel in US healthcare — plus portal messages, phone calls and PDFs. A staff member reads each one, extracts demographics, insurance, referring provider, reason for referral and requested specialty, keys it into the EHR, verifies insurance eligibility, and schedules. This is 8–15 minutes per referral, error-prone, and the errors surface as denied claims weeks later.

**Restatement:** for inbound faxed and PDF referrals, extract the 14 fields intake staff key by hand and pre-populate the EHR intake form, cutting handling from 12 minutes to under 4, with a staff member reviewing and submitting every one, and no reduction in the accuracy of insurance-eligibility capture.

**Constraints that shape it.** PHI everywhere, so a signed BAA with the model provider is a precondition — verify the covered services list, not just that a BAA exists. Data residency and retention settings must be explicit and documented. Faxes are 200-dpi scans of photocopies of handwriting, so OCR quality is the binary risk and I test it on day one with thirty real (de-identified) faxes before anything else. The EHR is very likely to have an integration surface that is slow, awkward, or requires a certified partner — assume pre-fill via the clipboard or an intermediary worklist rather than a direct write in phase one.

**Eval and its asymmetry.** Field-level accuracy, reported per field, never averaged. Two fields carry almost all the downstream cost — insurance member ID and date of birth — because an error there produces a denied claim and a rework loop weeks later. So the eval reports those separately with their own thresholds, and the system's confidence on those two fields drives whether the field is pre-filled or left blank with a highlight. **Leaving a field blank is a feature**, not a failure: a blank field gets typed, a wrong field gets accepted.

**⚠ Trap — the pre-fill that's confidently wrong.** Humans reviewing pre-filled forms rubber-stamp; this is a well-established automation-bias effect and it means a 95%-accurate pre-fill can produce *more* errors than manual entry if reviewers stop reading. My mitigation is structural: low-confidence fields are left empty and outlined, high-confidence fields are pre-filled, and the two highest-cost fields require an explicit confirmation click regardless of confidence. Then measure it — compare downstream claim-denial rates for pre-filled versus manual cohorts for the first six weeks. If denials go up, the pilot has failed even if extraction accuracy went up, and you want to be the one who says so.

### Bank back-office automation and financial reconciliation. Where does the LLM actually belong, and where would you refuse to put it?

This is the case where the "is an LLM even the right tool" gate earns its keep, and I'd open with it, because a candidate who proposes an LLM for the matching engine reveals they've never done this work.

**Reconciliation matching is not an LLM problem.** Matching 400,000 payments against 400,000 ledger entries is a join with fuzzy keys — amount tolerance, date window, reference-number normalization, many-to-one aggregation. Deterministic rules plus, at most, a gradient-boosted classifier over engineered features will beat an LLM on accuracy, cost and auditability by margins that aren't close, and it produces an explanation an auditor accepts. At 400,000 items/month, an LLM call per item at even 1,500 tokens is 600M tokens/month — on the order of $1,800/month at $3/Mtok, which is affordable, but the real problem is that it's non-auditable, non-deterministic, and slower by orders of magnitude than a SQL join. I would say this plainly and it is one of the strongest senior signals available in this round.

**Where the LLM does belong is the breaks.** Typically 2–5% of items don't auto-match, and each break is a human reading unstructured evidence — a remittance advice PDF, an email chain, a SWIFT MT103 free-text field, a counterparty's spreadsheet — and reasoning about *why*. That is genuinely language work and it's where 90% of the labor sits.

So the restatement: for the ~12,000 monthly unmatched items, read the associated remittance and correspondence, propose a resolution category (partial payment, FX difference, duplicate, timing difference, fee deduction, wrong reference) with the supporting evidence quoted, pre-fill the break-resolution note, and route to the right team — cutting average break-handling time from 14 minutes to under 5, with an analyst approving every resolution.

**💰 Math:** 12,000 breaks × 9 minutes saved = 1,800 hours/month; at ~$45/hour loaded for an ops analyst that's **$81,000/month**, $972k/year. Inference at ~6,000 input tokens per break is 72M tokens/month ≈ $216 at $3/Mtok, plus output. The ratio again is ~0.3%.

**Constraints specific to banking:** SOX and audit mean every automated action needs a reproducible trail — inputs, model version, prompt version, output, approver, timestamp — and I'd design that on day one, not retrofit it. Segregation of duties means the system cannot both propose and approve. Data almost certainly cannot leave the tenant, which forces the in-tenant or self-hosted branch and its cost. And month-end close is a hard latency window: if the batch doesn't finish by 6am on close day, it is worthless that day, which makes throughput a real requirement rather than a nice-to-have.

**🗣 Say this in the room:** "I'd keep the matching engine deterministic — it's a join, and an auditor can't accept a probabilistic explanation for why two payments matched. I'd point the model at the exceptions, where someone is currently reading a PDF and an email chain to figure out why something didn't match. That's where the hours are and it's genuinely a language problem."

### Retail support deflection. The customer wants an agent that resolves tickets end to end. How do you scope it?

The framing that matters: **deflection is not a model capability, it's a policy decision about which actions you'll let a machine take.** Most of what limits deflection isn't whether the model can write a good answer — it's whether the company will let it issue a refund, change an address, or cancel an order without a human. So my first questions are about actions and authority, not about the model.

**Segment the volume before anything else.** Pull a month of tickets and cluster them, then bucket by two axes: does resolving it require *information* (where's my order, what's your return policy) or an *action* (refund me, change the address, cancel this)? And is the action *reversible*? Information-only is the easy 30–40% and it needs read-only integrations. Reversible actions (resend a confirmation, update an address before dispatch) are the next tranche. Irreversible actions with money attached need a policy and a limit.

**The restatement:** for the top eight intents covering ~65% of ticket volume, resolve end to end without a human — including issuing refunds under $50 where the order meets the policy criteria — with a target of ≥45% full containment, CSAT on contained conversations not below the human-handled baseline, and zero refunds issued outside policy.

That last clause is the one I'd insist on and it's implemented not by prompting but by **the tool boundary**: the refund tool itself enforces the amount cap, the order-age check and the one-refund-per-order rule server-side. The model can request a refund; it cannot issue one outside policy, because the API won't let it. Enforcing policy in the tool rather than the prompt is the single most important design decision in agentic customer support and I'd say so explicitly — prompts are advisory, tools are load-bearing.

**Metrics, and the trap in them.** Containment (resolved without human) is the business metric, but containment alone is gameable in the worst way: a system that stonewalls users and ends conversations shows fantastic containment and destroys your brand. So containment must always be reported alongside **re-contact rate within 72 hours** (the honest measure of whether it actually resolved anything) and CSAT on contained conversations. **🔍 If containment is up and re-contact is up, the agent is deflecting, not resolving — that is a failure wearing a success's clothes**, and it's the most common way a support-deflection pilot posts great numbers and gets cancelled in month six.

**💰 Math:** 120,000 tickets/month at $28/hr loaded and 8 minutes each is $448,000/month. 45% containment saves 0.45 × 120,000 × 8/60 × $28 = **$201,600/month**. But subtract re-contact: if 8% of contained conversations re-contact and cost 12 minutes each (the human now has context to unpick), that's 0.08 × 54,000 × 12/60 × $28 = **$24,200/month** back. Net ≈ $177,000/month. Reporting the net, unprompted, is what separates you from the vendor who quoted $201,600.

### Internal knowledge search across an enterprise's Slack, Drive, Confluence and Jira. What kills this rollout?

Three things, and none of them is model quality. I'd name them in the first minute, because the customer will otherwise assume the hard part is the AI.

**Permissions.** Covered at length elsewhere in this section, but for this case specifically: the corpus spans systems with incompatible permission models — Slack channel membership, Drive per-file ACLs and link-sharing, Confluence space permissions, Jira project roles. Reconciling those into one queryable filter, keeping it fresh, and handling the file that's shared "anyone with the link" is the majority of the engineering. This is the work.

**Freshness and provenance.** The retired 2022 handbook problem. Every chunk needs source, version, last-modified and author; retrieval must prefer current documents and surface the date on the answer; and the connectors need incremental sync with a monitored lag, because a 6-hour indexing lag means a policy announced this morning doesn't exist to the system this afternoon.

**The query distribution.** Sample a thousand real queries before you design and you'll find a shape most people don't expect: a large share are **navigational** ("expense policy," "Q3 OKRs," "vpn setup") where the user wants the *document*, not a synthesized answer, and a generated paragraph is strictly worse than a link. A middle tranche are genuine synthesis questions. And a long tail are questions the corpus simply cannot answer because the knowledge is in someone's head or in a DM. Designing one generative answer path for all three is why these deployments feel disappointing: the head of the distribution is a search problem and you replaced it with a slower, chattier search.

**Restatement:** for the ~60% of queries that are answerable from the indexed corpus, return a correct answer with a clickable citation to the current version of the source, at ≥70% "helpful" rate on a labeled 200-query set drawn from real logs, with strict permission enforcement verified by an audit, and ≥40% weekly active usage among the pilot department by week eight.

**What I'd cut:** don't index everything. Start with two or three high-value, well-governed sources (the HR/IT knowledge base and the engineering wiki) rather than all of Slack. Slack is the highest-volume, lowest-signal, hardest-permission source and indexing it first is how these projects drown. Personal drives: excluded, explicitly.

**⚠ Trap — measuring success with search metrics on a task problem.** Click-through and MRR tell you about ranking; they don't tell you whether anyone got their job done. The metric I'd actually fight for is a **weekly two-question in-product survey on a sample of sessions** ("did you find what you needed? did you have to ask a person anyway?"), plus the deflection proxy the customer can already measure: volume of questions in the #it-help and #hr-questions channels before and after. That second one is measurable from data they already have, requires no instrumentation, and is exactly the kind of metric an economic buyer believes.

### The pilot succeeded. Walk me through handover — what do you actually hand over, and to whom?

The mental model: **an FDE engagement that ends with a working system and no owner has failed with extra steps.** The deliverable is not the system, it's a system plus a team that can keep it working after you leave, and the handover is designed from week one rather than assembled in the last week.

Six artifacts, and I'd name the owner for each before I build them.

**The runbook**, written for someone who was not there. What the system does, its inputs and outputs, its dependencies with their owners, how to restart it, how to turn it off (the kill switch and who's authorized to pull it), what to do when the provider returns 429s or 5xx, how to roll back a prompt or model version, and the top five alerts with the exact first action for each. If the runbook can't be executed by an on-call engineer at 3am who has never seen the system, it isn't done.

**The eval harness and the golden set**, with instructions to run it, in *their* CI, with a named owner. This is the most valuable thing you leave behind and the one most often skipped. Without it they cannot safely change a prompt, upgrade a model, or add a document source — which means the system freezes and rots.

**Monitoring**, and it's the piece where your backend background transfers directly, with three additions to the usual golden signals. Quality proxies: the guardrail-fire rate (citations that failed verification), the human-override rate, the fraction of outputs falling back to escalation. Cost: tokens per resolved case and cost per resolved case, plotted daily — this is the metric that catches a prompt change that silently tripled context. And drift: the distribution of input types and query clusters versus the pilot period, because the most common cause of "it got worse" is that the inputs changed, not the system.

**The threshold and policy document** — which confidence thresholds are set where, what they trade off, and how to change them safely (change the threshold, re-run the eval, look at the coverage/quality curve, then deploy).

**A named on-call rotation and an escalation path** including who at the model provider to contact and under which support agreement.

**The known-limitations register** — the failure modes you know about and chose not to fix, written down. Handing over a list of what doesn't work is a mark of seniority and it prevents the successor team from rediscovering each one as an incident.

**⚠ Trap — handing over to "the platform team" who were never in the room.** Adoption of the *system* by users and adoption of the *ownership* by an engineering team are two separate change-management problems and both need a named human who agreed. Get that person into the last three weeks of the pilot doing real work — let them fix a bug, run the eval, respond to one alert — before you leave. A handover meeting is not a handover.

### How do you run the expansion conversation after a successful pilot without sounding like a salesperson?

You don't sell the next phase; you present the evidence and let the arithmetic make the argument, and you show up with the *problems* as well as the wins because that's what makes the wins believable.

Structure the readout in five parts, in this order. **The metric against the criteria we signed** — met, missed, or partial, said plainly in the first sixty seconds, with the guardrails alongside. **The business translation**, with the arithmetic shown and the honest net after error cost. **What we learned that you didn't know before** — this is usually the data-quality findings, the inter-expert disagreement rate, the process-map numbers — because these have standalone value and they demonstrate that the engagement produced knowledge, not just software. **What didn't work and what I'd stop doing.** **Then, and only then, the three options.**

Three options, not one, because a single proposal invites yes/no and three invites a choice: (a) **deepen** — same use case, more coverage, tightening the threshold or fixing the two failure clusters we identified, with the incremental value quantified; (b) **widen** — same pattern, adjacent use case, and now you can estimate it credibly because you've measured the first one; (c) **harden** — production-grade the pilot, monitoring, handover, no new capability. For each: effort, cost, expected value with the arithmetic, and the risk.

The strongest expansion move is one you set up months earlier: during discovery you heard about four other processes with the same shape. Name one specifically. "Your commercial-lines team does the same document extraction on a smaller volume — the pattern transfers and the marginal cost is maybe a third of what this one cost, because the eval harness, the ingestion and the review workflow already exist." That is not selling; it's the natural inference from what you both now know.

**⚠ Trap — expanding before the first thing is stable.** The most common way a good pilot becomes a bad program is agreeing to use case two while use case one has no owner, no monitoring and a manual weekly babysitting ritual you're personally performing. My rule in review: no new use case until the current one has a named owner, a runbook, an eval in CI, and two weeks of unattended operation. Say that in the room — declining expansion revenue for a stated engineering reason is an extremely strong seniority signal.

**🗣 Say this in the room:** "Before we talk about the next use case I'd want this one running for two weeks without me touching it, with your team on the alerts. If I'm still the person who restarts it when it breaks, we haven't finished."

### How does this round differ across OpenAI FDE, Anthropic's applied roles, Palantir, Scale, Sierra, Harvey and Glean?

The underlying skill is the same everywhere — turn ambiguity into a scoped, measurable, sequenced plan — but the emphasis shifts, and preparing the wrong emphasis is a cheap way to lose. **📅 Volatile:** role definitions and loop structures at all of these companies change frequently; treat this as a directional map and verify against recent candidate reports before your loop.

**Palantir** is the origin of the discipline and the round is the most purely decompositional: heavy ambiguity, heavy emphasis on data integration and modeling the customer's domain, and the strongest weighting on "what would you build in the first 48 hours onsite." Expect the interviewer to be genuinely adversarial about scope and to reward you for cutting it. Data plumbing is not beneath the role here, it *is* the role.

**OpenAI FDE / deployment-engineering roles** lean product and prototyping: how fast can you get something working against the API in front of a customer, plus real coding. Expect the case to be paired with a build round, and expect model-capability judgment (what's actually feasible today, what isn't) to carry weight.

**Anthropic's applied and customer-facing roles** weight the *conversation* stage most heavily — the customer simulation is a named stage — and safety reasoning is scored inside it rather than bolted on. Expect at least one push toward a use case with a real harm dimension, and expect the follow-ups to test whether your position survives pressure.

**Scale** biases toward cost-constrained design and data-operations reality: labeling pipelines, throughput, unit economics. Expect an explicit "what does this cost per unit and how does that change at 10×" line of questioning.

**Sierra** is agent-centric customer experience: containment, escalation design, tone, action boundaries, and what an agent is permitted to do. Prepare the tool-boundary and re-contact-rate material.

**Harvey** and other vertical-AI companies test **domain fluency plus eval rigor**. You are not expected to be a lawyer, but you are expected to know why matter isolation matters, why recall beats precision on a risk-flagging task, and why "the model summarized the contract" is not a product.

**Glean** and enterprise-search companies test deployment engineering: connectors, permission models, freshness, and the rollout/adoption problem, which is most of the job.

**DeepMind** runs less of a classic FDE case; where an applied/customer element appears, it tends to be fused with evaluation-infrastructure design — how would you measure this, at scale, reliably.

The preparation implication: build one case you can run cold in six minutes (the claims one works), then prepare *three* domain overlays — a regulated-decision domain, a document/knowledge domain, and a conversational-agent domain — because every company above draws from one of those three.

### What are the specific ways candidates lose this round? Give me the taxonomy, not anecdotes.

**🔍 Failure taxonomy — run this as a self-check on any practice case.**

**1. Architecture-first (the most common, ~40% of failures I'd guess).** *Symptom:* boxes on the board before minute ten; the words "vector database" before the words "how does this work today." *Root cause:* comfort-seeking — architecture is the part you know. *Detection during the round:* if you've said more sentences than the interviewer in the first ten minutes, you're in it. *Fix:* an internal rule that no architecture leaves your mouth until you've delivered a problem restatement and had it confirmed.

**2. Over-promising on capability.** *Symptom:* "yes, we can do that" more than once; agreeing to 99% accuracy; accepting an auto-decision on an irreversible action; nodding through scope creep. *Root cause:* conflict avoidance plus not having a rehearsed refusal. *Detection:* count the number of times you've said no. If it's zero at minute thirty, you have failed, because no real engagement has zero exclusions. *Fix:* pre-write two refusals — one scope, one safety — and deliver at least one.

**3. No measurable outcome.** *Symptom:* the plan ends with "and then we'd deploy it"; the word "accuracy" used with no population, threshold or measurement procedure; no baseline. *Root cause:* treating evaluation as a phase rather than as the frame. *Detection:* can you state, in one sentence, the number that decides whether this pilot succeeded, and who measures it? *Fix:* the success sentence is a required beat, not an optional one.

**4. Ignoring change management.** *Symptom:* the humans in the process appear only as a cost to be removed; no mention of who has to change behavior; no adoption metric. *Root cause:* engineering framing of a socio-technical system. *Detection:* did you ask a single question about the end users, as distinct from the buyer? *Fix:* the stakeholder map is a required beat and it has a users row.

**5. Solving the stated problem instead of the real one.** *Symptom:* you built exactly what they asked for, on the step that costs six minutes out of twenty-eight. *Detection:* did you ever quantify where the time actually goes? *Fix:* the current-state process map with numbers.

**6. Losing the room in the conversation stage.** *Symptom:* jargon; lecturing; capitulating; ending on bad news; narrating instead of speaking in character. *Fix:* the four-move sequence — acknowledge, one plain sentence of mechanism, the consequence for them, the alternative you'd stand behind.

**7. Rigidity.** *Symptom:* the interviewer introduces a plan-invalidating fact at minute thirty and you continue with your original plan. *Fix:* say "that changes the restatement" out loud and redo it. Visible re-planning scores; silent absorption does not.

### Give me the drill set. How do I actually practice this alone?

**🏋 Drill 1 — the six-minute cold case (do this 15 times).** Write seven prompts on cards: "our claims processing costs too much," "we want AI for contract review," "our support costs are out of control," "we can't find anything in our wiki," "reconciliation takes three analysts full time," "patient intake is a bottleneck," "we want to use AI in underwriting." Draw one at random, start a 6-minute timer, and speak the full arc out loud, recorded: clarifying questions, stakeholders, restatement, exclusions, two-week slice, success metric, business math. **Pass criterion:** the restatement contains a population and two numbers; you name at least two exclusions with reasons; the two-week slice names its riskiest assumption; the success metric has a threshold, a population and a guardrail; and there is at least one piece of arithmetic with digits. Fail yourself if you drew a diagram.

**🏋 Drill 2 — the number under pressure.** Pick any of the seven cases. In 90 seconds, unaided and out loud, produce the annual value: volume × time saved × loaded rate, then subtract an error cost, then state inference cost as a percentage of the value. **Pass criterion:** all arithmetic spoken with digits, no calculator, and you volunteered the error cost without being asked. Do this daily for two weeks; the fluency is what makes you sound like you've shipped.

**🏋 Drill 3 — three refusals.** Write out, verbatim, three sentences: one declining a scope request, one declining an unsafe automation, one telling a customer a use case will not work. Each must contain a reason and an alternative. Say them into a recording and listen for defensiveness or apology. **Pass criterion:** none of the three contains the word "unfortunately," none contains "we can't" without a following "what I'd do instead," and none is longer than four sentences.

**🏋 Drill 4 — the hostile roleplay.** You need a partner for this one; a colleague is fine and it takes twenty minutes. Give them four lines to deliver in order: "it gave two different answers to the same question," "it made up a citation," "the demo looked better than this," "we need it to be 100% accurate." They deliver each and then push twice on your response. **Pass criterion:** you never use the words temperature, stochastic, embedding, hallucination, or LLM; every answer ends on an offer rather than a limitation; and on the third push you get more concrete rather than more insistent.

**🏋 Drill 5 — write the one-pager, timed at 25 minutes.** For one case, produce the pilot success-criteria document: problem statement, in/out scope, baseline, primary metric, guardrails, adoption criterion, measurement procedure, dependencies with named owners and dates, and the three-branch decision rule including the stop condition. **Pass criterion:** it fits on two pages, every metric names its data source, and the stop condition is written explicitly. Then hand it to a non-technical friend and ask them what happens if the pilot half-works. If they can answer from the document, you've passed.

**🏋 Drill 6 — the labeling session simulation.** Take 20 items from any public dataset in a domain you don't know, write a rubric, label them yourself, then label them again a week later without looking. Measure your own self-agreement. **Pass criterion:** you can state your self-agreement rate as a number and explain the three items where you disagreed with yourself. This is the fastest way to internalize why inter-annotator agreement is your real ceiling — and having done it, you will say so in a room with genuine conviction rather than as a memorized talking point.


---

## 86. Research Literacy, the Paper Canon and Working With Research Teams

*Mastering this proves you can hold your own in a paper round and be useful to a research team — the difference between an applied engineer at a lab and one anywhere else.*

### There's a "paper discussion" round on your schedule. What is that round actually testing, and how does it differ across the companies you're likely to see it at?

The mental model that fixes this round: nobody is testing whether you have read a paper. They are testing whether *reading papers changes what you build*. A paper round is a proxy for one question — when a new technique lands on arXiv on a Tuesday, are you the engineer who can tell in twenty minutes whether it is relevant to your serving stack, or the one who forwards the tweet? Everything about how the round is scored follows from that.

Mechanically the round takes three shapes and you should identify which one you are in inside the first ninety seconds. **Shape one, the canon check**: "tell me about a paper that influenced how you build." This is depth-on-demand — they will pick one thing you said and drill until you hit bottom. **Shape two, the cold read**: you are handed a paper (or an abstract plus two figures) you have not seen, given 15–25 minutes, and asked to critique it. This tests method-reading, not memory, and it is the fairest version. **Shape three, the applied translation**: "we're seeing X in production; is there literature that speaks to it?" — the round frontier labs and AI-product companies actually care about, because it is the job.

Company weighting, honestly stated. DeepMind and xAI research-adjacent loops run the deep dive as a graded stage with an actual researcher in the room, often on *your* claimed area. OpenAI and Anthropic Applied loops fold it into a technical conversation — less formal, but the interviewer is frequently someone who has published, so bluffing has a very short half-life. Cursor, Perplexity, Sierra and Harvey will rarely run a formal paper round, but will absolutely ask "why did you pick that retriever / that decoding strategy," and the honest answer is a paper. Big-tech applied loops (Meta, Google, Amazon, Microsoft, Databricks) tend to ask it as a currency check: are you a 2023 engineer or a current one.

**⚠ Trap:** preparing breadth for a round that is scored on depth. Twenty papers you can summarize and zero you can criticize is a losing hand — the second follow-up finds the floor. Three papers you can attack, plus honest "I've skimmed that, here's what I took from it" for the rest, beats it every time.

**🗣 Say this in the room:** "I read papers as a serving engineer, not as a reviewer — so I mostly index on what a result changes about a system I'd build. Happy to go deep on the ones where that actually happened, and I'll flag where I've only skimmed."

### Give me your protocol for summarizing a paper in ninety seconds.

Five beats, always the same order, and the discipline is that you say all five even when you are nervous — because the fifth is the one that scores. **Problem. Key idea. What it replaced. The evidence. The limitation.**

Why this order works: *problem* establishes you understand the constraint that made the work necessary, which is the part a summary-from-the-abstract candidate always skips. *Key idea* is one sentence and one mechanism, not a paragraph of architecture. *What it replaced* is the beat that proves you know the field's shape rather than one point in it — you cannot state what a paper replaced unless you know the prior art. *The evidence* forces you to name the actual experimental setup, which is where bluffing dies (anyone can say "it improved accuracy"; only a reader says "on WMT'14 EN-DE at 28.4 BLEU with a single model, trained about 3.5 days on eight P100s"). *The limitation* signals you read past the conclusion section, and it is the natural hook for the interviewer's follow-up — which means you control what gets asked next.

Here it is executed on FlashAttention, at real speed: "Problem — attention is exact but its memory traffic is quadratic in sequence length because you materialize the T×T score matrix in HBM, and on modern GPUs that traffic, not the matmul, is the bottleneck. Key idea — tile the computation into blocks that fit in SRAM and use an online-softmax recurrence so you never write the full matrix out, then recompute it in the backward pass instead of storing it. What it replaced — approximate-attention work (Linformer, Performer, sparse patterns) that traded quality for asymptotics; FlashAttention gave you exactness *and* speed, which mostly ended that line. Evidence — wall-clock speedups on BERT and GPT-2 training and the ability to train at longer contexts at the same memory. Limitation — it is a kernel, not an algorithm change: it does nothing about the quadratic *compute*, so at very long contexts you still eat T², and the original kernel was Ampere-tuned, which is why FlashAttention-2 and -3 exist."

That is about 75 seconds spoken. **📄 Paper:** Dao et al. (2022) — IO-aware exact attention via tiling and recomputation; it replaced the approximate-attention research program for most practical purposes.

**⚠ Trap:** the abstract-recital. If your summary contains the words "we propose a novel," you are reading the abstract back, and the interviewer knows because abstracts have a distinctive rhythm. Always convert to mechanism: what tensor moved where, what got dropped, what got cheaper.

**🏋 Drill:** pick five canon papers. Record yourself doing all five beats for each with a 90-second timer. Pass criterion: every one contains a specific number from the experiments and a limitation the authors themselves acknowledge, and none exceeds 100 seconds.

### Walk me through Attention Is All You Need as if I've read it and want to see whether you have.

Problem: sequence transduction in 2017 meant recurrence, and recurrence serializes along the time axis — you cannot compute step *t* before step *t−1*, so training throughput is bounded by sequence length no matter how many GPUs you own. Convolutional alternatives (ByteNet, ConvS2S) parallelize but grow the path length between two distant positions with distance, so long-range dependencies still have to propagate through many layers. The paper's real contribution is architectural *and* economic: an architecture with O(1) path length between any two positions and full parallelism across the sequence during training.

Key idea: scaled dot-product attention, `softmax(QKᵀ/√d_k)V`, run in *h* parallel heads over projections of dimension d_k = d_model/h, concatenated and re-projected. The √d_k is not cosmetic — if q and k have i.i.d. unit-variance components, their dot product has variance d_k, so without the scale the softmax saturates as d_k grows and gradients vanish. Around each sublayer: residual connection and LayerNorm (Post-LN in the original), plus a position-wise FFN of width 4·d_model with ReLU. Because attention is permutation-equivariant, position is injected via fixed sinusoidal encodings added to the embeddings.

What it replaced: recurrent encoder-decoder machine translation with Bahdanau/Luong attention bolted on. The title is a genuine claim — attention had existed since 2014 as an *addition* to RNNs; the contribution was deleting the RNN.

The evidence: WMT 2014 English→German at 28.4 BLEU and English→French at 41.8 BLEU for the big model, beating prior ensembles at a fraction of the training cost — the base model trained roughly 12 hours on eight P100s, the big model about 3.5 days. There is also an English constituency-parsing transfer result, which most people forget and which was the paper's evidence for generality.

Limitations, stated honestly: O(T²·d) compute and O(T²) attention memory; fixed sinusoidal positions that extrapolate poorly in practice; Post-LN, which turned out to need careful warmup and was later abandoned for Pre-LN; and the encoder-decoder framing, which the decoder-only lineage discarded within three years. The paper also used dropout inside attention at 0.1, which modern large-scale pretraining largely dropped.

**⚠ Trap:** describing the transformer as "the architecture behind GPT" and stopping. The paper is an *encoder-decoder* for translation. If you cannot say what changed to get from it to a decoder-only LM — drop the encoder and cross-attention, causal-mask the self-attention, swap the objective to next-token prediction, later swap Post-LN→Pre-LN, LayerNorm→RMSNorm, ReLU→SwiGLU, sinusoidal→RoPE, MHA→GQA — you have read *about* the paper, not the paper.

**🗣 Say this in the room:** "The thing I actually take from it as an engineer is that the win was parallelism, not accuracy — it made compute the binding constraint instead of wall-clock sequence dependence, and everything about the scaling era follows from that."

### What did GPT-3 actually demonstrate that GPT-2 didn't, and why does an applied engineer care?

The mental model: GPT-3 is not "a bigger GPT-2 that scored better." Its claim is that at sufficient scale, a task specification can move from the *weights* to the *prompt*. That single relocation is why your job exists. Before it, shipping an NLP feature meant collecting a labelled dataset and fine-tuning a per-task model; after it, it meant writing a prompt and iterating on it in an afternoon. Every product company on your target list is downstream of that shift.

Mechanism and scale: 175B parameters, 96 layers, d_model 12288, 96 heads, 2048-token context, alternating dense and locally-banded sparse attention. Trained on roughly 300B tokens of filtered Common Crawl plus WebText2, books and Wikipedia. The experimental design is the interesting part — they evaluated zero-shot, one-shot and few-shot with *no gradient updates*, across dozens of tasks and across a model-size ladder from 125M to 175B, and showed the few-shot-minus-zero-shot gap *widening* with scale. That widening gap is the actual evidence for in-context learning as an emergent capability rather than a prompt-format artifact.

What it replaced: the BERT-era "pretrain then fine-tune a task head" pipeline as the default for new tasks.

The limitations the paper itself names, which candidates forget: weak performance on tasks requiring bidirectional comparison, poor sample efficiency relative to humans, no way to tell whether few-shot learning is genuinely learning at inference time or retrieving something seen in pretraining, and contamination — they ran a train/test overlap analysis and found it, and reported some results with the caveat.

**💰 Math on why Chinchilla immediately made this look wrong:** GPT-3 spent roughly C = 6ND = 6 × 1.75e11 × 3e11 ≈ 3.15e23 FLOPs. Compute-optimally at *that same budget* you solve 6·N·(20N) = C for N, which gives a ~50B model on ~1.0T tokens. Two useful bracketing views on top of that: hold the data at 300B tokens and 20:1 says a ~15B model; hold the model at 175B and 20:1 says about 3.5T tokens — and the actual run was 175B on 300B, i.e. roughly 1.7 tokens per parameter. It was massively under-trained, and that is *why* it is a historically important but practically obsolete artifact.

**⚠ Trap:** citing GPT-3 as evidence that "bigger is better." It is evidence that in-context learning emerges with scale. The parameter-count-as-quality reading is exactly what Chinchilla falsified eighteen months later.

### Explain Chinchilla to me, and then tell me what people get wrong when they quote it.

Problem: given a fixed training compute budget C, how do you split it between model size N and training tokens D? Kaplan et al. (2020) had answered this and the field had internalized their answer — scale parameters aggressively, data much less so. That prior is what produced GPT-3, Gopher, MT-NLG: enormous, under-fed models.

Key idea: Hoffmann et al. (2022) re-ran the estimation with three independent methods — (1) fix model sizes and vary tokens, reading off minima; (2) isoFLOP curves, training many (N, D) pairs at each of several fixed compute budgets and finding the parabola's minimum; (3) fitting a parametric loss surface of the form L(N, D) = E + A/N^α + B/D^β and minimizing under the constraint C ≈ 6ND. All three said N and D should scale *roughly in proportion*, giving the famous heuristic of about 20 training tokens per parameter at the compute-optimal point.

The evidence is the part to memorize because it is one clean comparison: Chinchilla, 70B parameters trained on 1.4T tokens, beat Gopher, 280B trained on 300B tokens, across a broad benchmark suite — at approximately the same training compute. Check it: 6 × 7e10 × 1.4e12 = 5.88e23 versus 6 × 2.8e11 × 3e11 = 5.04e23. Same order, same budget, and the smaller model won *and* is four times cheaper to serve.

Now the misreadings, which is what the question is really after.

**⚠ Trap 1 — treating 20:1 as a serving rule.** Chinchilla optimizes *training* compute for a target loss. It says nothing about inference. If you will serve a model billions of times, the total-cost-optimal choice is a *smaller* model trained far past 20:1 — which is precisely why Llama-3-8B was trained on ~15T tokens, about 1,875 tokens per parameter, roughly 90× "Chinchilla-optimal." That is not a mistake; it is a different objective function. Being able to say this sentence is a strong senior tell.

**⚠ Trap 2 — quoting 20:1 as settled science.** A 2024 replication attempt argued that the third, parametric estimator in the Chinchilla paper was mis-fit and inconsistent with the paper's own approaches 1 and 2, implying somewhat different coefficients and wider uncertainty. The 20:1 heuristic survives as an order-of-magnitude anchor; the third decimal place does not exist. Say the number *and* say it is an anchor.

**⚠ Trap 3 — forgetting the scaling laws were fit on dense models on a specific data distribution.** They do not transfer unchanged to MoE, to heavily-repeated data, or to post-training compute.

**🗣 Say this in the room:** "Chinchilla is a training-compute-optimal result, and almost nobody I work with has a training-compute-optimal objective. Once you amortize inference, you deliberately over-train a smaller model — Llama-3-8B at ~15T tokens is roughly 90× past the Chinchilla point and that's the correct call for a model you serve."

### Walk me through InstructGPT. Why does an applied engineer need to know the three stages?

Mental model: pretraining optimizes for "what token comes next in text like this," which is not the same objective as "be useful to the person who typed this." InstructGPT is the paper that named the gap — *alignment* in the boring, practical sense of objective mismatch — and gave the three-stage recipe that every subsequent instruction-tuned model is a variation on. You need the stages because every post-training decision you will ever argue about maps onto one of them.

Stage 1, supervised fine-tuning: human labellers write demonstrations of desired behavior on a prompt distribution drawn from real API traffic, and you fine-tune the base model on them. Stage 2, reward modelling: for a given prompt, sample K completions, have labellers *rank* them, and train a reward model on the pairwise comparisons with a Bradley-Terry-style loss — ranking rather than absolute scoring because humans are far more consistent at comparisons than at scalar ratings. Stage 3, RL: optimize the policy against the reward model with PPO, plus a per-token KL penalty against the SFT policy to stop the model drifting into reward-model blind spots, plus (in their PPO-ptx variant) a mixed-in pretraining gradient to reduce capability regressions on standard benchmarks.

The evidence, and it is the number to remember: labellers preferred outputs from the 1.3B InstructGPT over the 175B GPT-3. Two orders of magnitude of parameters, beaten by post-training on the axis users actually care about. **💰 Math on why that matters commercially:** serving cost scales roughly with active parameters, so 175B → 1.3B is on the order of a 100× reduction in FLOPs per token for a *better*-rated product. That is the single strongest argument in existence for spending your budget on post-training and evaluation instead of on a bigger base model.

Limitations the paper states: the labeller pool was small and demographically narrow, so "aligned" means aligned to about forty contractors' preferences; the model remains overly deferential and will confidently follow a false premise; and there is an explicit alignment tax on some NLP benchmarks that PPO-ptx mitigates but does not eliminate.

**⚠ Trap:** saying "RLHF" when you mean the whole pipeline. Two of the three stages are supervised. Most quality you will ever ship comes from stage 1 data quality, and I have never seen a team whose SFT data was good enough that RL was the bottleneck.

**📄 Paper:** Ouyang et al. (2022) — SFT → reward model → PPO with a KL anchor; it replaced prompt-engineering-only deployment of raw base models and established preference data as the central asset.

### What is Constitutional AI, and where does it show up in work you'd actually do?

The mental model: human preference labelling is the bottleneck and the cost centre of RLHF, and it is worst exactly where you need it most — harmlessness data requires humans to read and rank harmful content at volume. Constitutional AI's move is to replace the *human* in "human feedback" for the harmlessness axis with a model conditioned on an explicit written set of principles, so the value judgment lives in a document you can read, version and argue about rather than in an opaque pile of labels.

Two phases. **Supervised phase**: sample responses from a helpful-only model to red-teaming prompts, ask the model to *critique* its own response against a principle sampled from the constitution, then *revise* it; fine-tune on the revised responses. This gets the model into a roughly-harmless basin cheaply. **RL phase (RLAIF)**: generate response pairs, have a model choose the better one against a sampled constitutional principle, train a preference model on those AI-generated comparisons, and run RL against it — with human preference data still used for helpfulness.

What it replaced: exhaustive human labelling of harmfulness comparisons, and the opacity of "our values are whatever the labellers happened to prefer."

Where this touches applied work, which is the real question. First, **AI feedback as an eval and data-generation technique is now standard practice** — LLM-as-judge with a written rubric is structurally the same idea, and the CAI paper is the honest citation when you defend it. Second, **the constitution-as-artifact pattern**: when I build a judge or a content policy for a product, I write the principles as a versioned document with test cases attached, precisely because it makes disagreement about behavior a document review rather than a data-relabelling project. Third, it is the intellectual root of the self-critique/revise loop you will implement in an agent harness.

**⚠ Trap:** claiming RLAIF removes human judgment. It relocates it into the constitution's wording and into whoever chose the principles — and it inherits every bias of the model doing the judging, including a well-documented tendency to prefer longer and more confidently-worded responses. If you deploy an AI judge without measuring its agreement against a human-labelled set, you have not removed the labelling cost, you have hidden it.

**📄 Paper:** Bai et al. (2022), Anthropic — self-critique and revision plus RL from AI feedback against an explicit constitution; it replaced human harmlessness comparisons as the primary safety-training signal.

### Explain LoRA well enough that I'd believe you've implemented it.

Mental model: fine-tuning updates a weight matrix W by some ΔW. The empirical bet of LoRA is that for adapting a pretrained model to a task, the *useful* ΔW has low intrinsic rank — the model already contains the capability and you are steering it, not installing something new. If ΔW is low rank you never need to materialize it: write ΔW = BA where A is r×k and B is d×r with r ≪ min(d, k), train only A and B, and freeze W.

Mechanism, concretely. Forward pass is `h = xW₀ᵀ + (α/r)·x AᵀBᵀ` — the α/r scaling exists so that changing r does not force you to retune the learning rate. A is initialized Gaussian, B is initialized to **zero**, so at step 0 the adapter contributes exactly nothing and the model is bit-identical to the base; without that, you inject noise into a converged model on the first forward pass. Because both branches are linear in x, at deployment you can fold W ← W₀ + (α/r)BA and the adapter costs zero extra inference latency — the property that killed the earlier serial-adapter line of work, which added layers and therefore latency.

Parameter arithmetic: for a d=4096 square projection, full fine-tuning trains 4096² = 16.8M parameters. At r=16, LoRA trains 2 × 4096 × 16 = 131,072 — a 128× reduction for that matrix. The paper reported roughly a 10,000× reduction in trainable parameters and about a 3× reduction in GPU memory versus full fine-tuning of GPT-3 175B. The memory win is mostly *optimizer state*: Adam keeps two fp32 moments per trainable parameter, so 8 bytes each; 16.8M trainable → 134 MB per matrix, versus 1 MB at r=16.

Practical judgment: apply it to attention projections at minimum, and in modern practice to the MLP projections too, which usually matters more than raising r. The rule I enforce in review is *widen the target-module set before you raise the rank* — going from q,v-only at r=64 to all-linear at r=16 is almost always the better trade.

**⚠ Trap:** expecting LoRA to install new factual knowledge. It is very good at style, format, tone, tool-call syntax and task adherence, and poor at teaching facts the base model does not have — that is a retrieval problem, and answering "we'll fine-tune it on our docs" is the single most reliable way to get rejected from an applied AI loop.

**📄 Paper:** Hu et al. (2021) — low-rank update matrices with zero-init on B and mergeable weights; it replaced serial adapter modules, which added inference latency, and full fine-tuning, which was memory-prohibitive.

### QLoRA gets cited constantly. What are its actual components, and which one is doing the work?

Problem: LoRA cuts optimizer and gradient memory but not *base weight* memory — a 65B model in bf16 is 130 GB of frozen weights, which does not fit on one card no matter how few parameters you train. QLoRA's target is that residual.

Three components, and they are separable. **(1) 4-bit NormalFloat (NF4)** — a 4-bit datatype whose quantization bins are placed at the quantiles of a standard normal, on the argument that pretrained weights are approximately normally distributed, so equal-probability bins are information-theoretically better than the equal-width bins of int4. **(2) Double quantization** — the per-block quantization constants themselves are quantized, which sounds trivial and is: at block size 64 with fp32 constants you pay 32/64 = 0.5 bits per parameter of overhead, and double-quantizing takes that to roughly 0.127 bits/param, worth ~0.37 bits × 65e9 ≈ 3 GB on a 65B model. **(3) Paged optimizers** — using unified memory so optimizer state pages to CPU on gradient-checkpointing spikes instead of OOMing.

The critical mechanism people mis-state: the base weights are *stored* in NF4 and **dequantized to bf16 on the fly for each matmul**. QLoRA is not 4-bit compute. It buys memory, and it costs you time — you pay a dequantization kernel on every forward and backward pass, which in practice makes QLoRA meaningfully slower per step than bf16 LoRA. The LoRA adapters themselves stay in bf16 and are what receives gradients.

The headline evidence: fine-tuning a 65B model on a single 48 GB GPU, with the Guanaco family matching much of ChatGPT-level performance on their evaluation at that time, and — importantly for the argument — 4-bit NF4 base weights not degrading task performance relative to 16-bit LoRA in their comparisons.

**💰 Math:** 65B params × 2 bytes (bf16) = 130 GB, versus 65e9 × 0.5 bytes (4-bit) = 32.5 GB, plus roughly 1 GB of quantization constants after double quantization, plus adapters and activations — which is how you land under 48 GB. At a single 48 GB card (A6000-class) rental instead of a 4×A100-80GB node, that is roughly a 4–6× hourly-cost reduction for the experiment. **📅 Volatile:** GPU hourly rates move fast; verify before quoting a dollar figure.

**⚠ Trap:** using QLoRA when you have the memory for bf16 LoRA. If it fits, don't quantize — you are trading throughput for memory you already have. I have reviewed several projects that ran QLoRA on a 7B model on an 80 GB card, which is strictly worse than LoRA on every axis.

**📄 Paper:** Dettmers et al. (2023) — NF4 + double quantization + paged optimizers to make single-GPU fine-tuning of 65B-class models possible; it replaced multi-node full fine-tuning for the adaptation use case.

### DPO replaced PPO in a lot of pipelines. Derive the idea for me and tell me where it falls down.

Mental model: the RLHF pipeline trains a reward model and then runs RL to find the policy that maximizes it under a KL constraint. But the *solution* to that constrained optimization is known in closed form — the optimal policy is the reference policy reweighted by the exponentiated reward. DPO reads that identity backwards: if the optimal policy is a function of the reward, then the reward is a function of the policy, so you can substitute it into the preference likelihood and skip having a reward model at all.

The derivation, which you should be able to sketch. KL-regularized RL has optimal policy π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β). Rearranging: r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x). Plug that into the Bradley-Terry preference model P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l)); the intractable partition term log Z(x) depends only on x and therefore *cancels in the difference*. What remains is a plain classification loss over preference pairs:

```
L_DPO = −E[ log σ( β·(log π_θ(y_w|x) − log π_ref(y_w|x))
                  − β·(log π_θ(y_l|x) − log π_ref(y_l|x)) ) ]
```

Two forward passes (policy and frozen reference), a sigmoid, done. No reward model, no rollouts, no value network, no PPO clipping heuristics — a four-model, sampling-in-the-loop pipeline collapses to a supervised-looking one, which is why every fine-tuning library shipped it within months.

Where it falls down, and this is the part that separates a reader from a user. It is **offline and off-policy**: you optimize against a fixed preference dataset, and the policy drifts away from the distribution those pairs were sampled from, so late in training you are extrapolating. There is a well-observed pathology where the log-probability of *both* the chosen and rejected responses falls — DPO can satisfy the objective by pushing the margin apart while making the whole region less likely, which is not what you wanted. It inherits length bias from the preference data with no reward-model regularizer to fight it. And β is genuinely finicky: too low and it drifts off π_ref into degeneracy, too high and nothing moves.

The honest current state: for a fixed, decent preference dataset, DPO is the right default because the engineering cost is a tenth of PPO's. For anything with a *verifiable* reward — math, code, tool-use success — you want on-policy RL with a rule-based reward instead, which is the GRPO/RLVR line. And on-policy variants (iterative DPO, online DPO, sampling fresh pairs and re-labelling) recover much of PPO's advantage at lower cost. This area is genuinely contested; say that rather than pretending there is consensus.

**📄 Paper:** Rafailov et al. (2023) — the closed-form KL-regularized optimum turns preference learning into a classification loss; it replaced the reward-model-plus-PPO pipeline as the default for offline preference data.

### Explain GRPO. Why did DeepSeek drop the value network, and what did that buy them?

Mental model: PPO's value network exists to reduce gradient variance by estimating a baseline — "how good is this state on average" — so you learn from the *advantage* rather than the raw return. That network is roughly as large as the policy, must be trained alongside it, and is hard to fit for long language trajectories where the reward arrives only at the end. GRPO's observation: if you are going to sample multiple completions per prompt anyway, the *group* gives you an empirical baseline for free. Use the group's own mean reward as the baseline and delete the critic.

Mechanism. For each prompt q, sample a group of G outputs {o₁…o_G} from the current policy. Score each with the reward function. Compute the advantage as the standardized reward within the group, Â_i = (r_i − mean(r)) / std(r), and assign it to every token of that output. Then apply the usual clipped policy-gradient surrogate with a KL penalty toward the reference model. No value head, no GAE, no critic optimizer state.

What that buys, in resources: you delete a model of comparable size from the training loop, which removes its parameters, its gradients and its optimizer moments. **💰 Math:** for a 7B critic in bf16 with Adam, that is 7e9 × (2 weights + 2 grads + 8 optimizer bytes) = 7e9 × 12 = 84 GB of state removed, plus its forward/backward compute. That is the difference between fitting an RL run on one node and needing two.

The cost: your variance reduction is now empirical and depends on G, so groups that are too small give you noisy advantages, and prompts where every sample gets the same reward produce a zero-variance group that contributes nothing (and, if you divide by std naively, can produce numerical trouble). It also biases toward prompts of intermediate difficulty, which is a real effect and is sometimes exploited deliberately as curriculum.

The context matters: GRPO was introduced for math with a rule-based correctness reward, where sampling many attempts per problem is natural and the reward is cheap and exact. That is the regime where deleting the critic is clearly correct. Generalizing it to fuzzy, model-judged rewards is doable but you lose the property that made it safe.

**📄 Paper:** Shao et al. (2024), DeepSeekMath — group-relative advantage estimation removes the value network from PPO-style RL for LLMs; it replaced PPO's learned critic for verifiable-reward settings and became the workhorse of the reasoning-RL era.

**🗣 Say this in the room:** "GRPO is PPO with the critic replaced by the group mean. You give up a learned baseline and you get back an entire model's worth of memory and a much simpler training loop — a very good trade when your reward is a cheap exact check, a worse one when it's a fuzzy judge."
### Take me through DeepSeek-R1. What was genuinely new, and what part do people over-claim?

Mental model: everything before it treated chain-of-thought as a *prompting* phenomenon or something you distilled in with SFT. R1's claim is that extended reasoning is a *behavior you can grow with reinforcement learning against an automatically-checkable reward*, without teaching the model what a good chain of thought looks like. You do not supervise the reasoning; you supervise only the answer, and the reasoning lengthens on its own because longer reasoning gets the answer right more often.

Two artifacts, and conflating them is the most common error. **R1-Zero** is the scientific result: pure RL (GRPO) applied directly to the base model with rule-based rewards — an accuracy reward that checks the final answer against ground truth, and a format reward that enforces the reasoning-tags structure. No supervised warm start at all. Response length grew over training, self-verification and backtracking emerged, and reasoning benchmark performance climbed substantially. It also had real defects the paper states plainly: poor readability and language mixing mid-chain. **R1** is the product recipe built to fix those: a small cold-start SFT set of readable long-CoT examples, then reasoning-oriented RL, then rejection sampling to harvest good trajectories into a much larger SFT set covering non-reasoning tasks too, then a final RL stage for helpfulness and harmlessness.

The third contribution, and the one with the most practical consequence for an applied engineer: **distillation of R1's outputs into small dense models** (Qwen and Llama bases) produced small models that substantially outperformed running RL directly on those same small models. That is a strong empirical statement — for small models, imitating a large reasoner beats discovering reasoning yourself — and it is the reason "distill from a frontier reasoner" became a standard product move.

What is over-claimed: that R1 shows RL "creates" capabilities from nothing. The more defensible reading, and the one a researcher will respect, is that RL with verifiable rewards *elicits and sharpens* behaviors already latent in the base model's distribution and reallocates probability mass toward the ones that verify — which is consistent with the observation that pass@k for large k on the base model is often already decent while pass@1 is not. Whether RLVR expands the capability frontier or mainly collapses sampling onto the good tail is genuinely contested in the literature. Saying that you know it is contested is worth more than picking a side.

**⚠ Trap:** describing R1 as "trained with RLHF." There is no human preference model in the core reasoning loop — the reward is a deterministic checker. That distinction (RLHF vs RLVR) is exactly what the question is probing.

**🗣 Say this in the room:** "R1-Zero is the result and R1 is the recipe. The result is that long chain-of-thought emerges from pure RL against a rule-based correctness reward; the recipe adds a cold start and a final alignment pass because Zero's chains were unreadable and mixed languages. The part I actually use is the distillation finding."

### Why does FlashAttention exist if the math is unchanged, and what changed in FA2 and FA3?

Because on modern accelerators the bottleneck is not arithmetic, it is moving bytes between HBM and on-chip SRAM. An H100 does roughly 990 TFLOP/s of BF16 tensor-core math with structured sparsity (about 495 dense) against roughly 3.35 TB/s of HBM3 bandwidth. That is a ratio of hundreds of FLOPs per byte, so any kernel that reads and writes a large intermediate to HBM is bandwidth-bound no matter how good its matmul is. Naive attention materializes an T×T score matrix, writes it, reads it back for softmax, writes again, reads again for the ×V matmul. For T=8192 with batch 8 and 32 heads at fp16 that intermediate is 8 × 32 × 8192² × 2 bytes = 34.4 GB — written and re-read multiple times.

The mechanism: tile Q, K and V into blocks sized to fit SRAM, and compute the output block by block using the **online softmax** recurrence — carry a running max m and a running sum ℓ, rescale the accumulated output when a new block raises the max, and you get the exact softmax without ever having the full row in memory at once. For the backward pass, do not store the score matrix; recompute it from the saved statistics. Net result: HBM traffic drops from O(T² + T·d) to roughly O(T²·d² / M) where M is SRAM size (that is the paper's Θ(N²d²M⁻¹) bound — note the d², not d) — asymptotically better in the term that actually costs wall-clock. The output is **bit-comparable-to-exact** attention; this is not an approximation.

The successor deltas, which is what the second half of the question wants. **FlashAttention-2** (Dao, 2023) is a work-partitioning rewrite: fewer non-matmul FLOPs (the rescaling was eating tensor-core throughput because non-matmul ops run far slower per FLOP), parallelization over the sequence dimension as well as batch and heads so that long-sequence low-batch cases keep the GPU busy, and better warp-level partitioning to cut shared-memory traffic. **FlashAttention-3** (2024) targets Hopper specifically: asynchrony between tensor cores and the TMA copy engine so data movement overlaps math, warp specialization, and FP8 support with incoherent processing to control quantization error.

**⚠ Trap:** believing FlashAttention reduces attention's *compute*. It does not — you still do O(T²·d) FLOPs. It reduces memory traffic and peak memory. If someone says "we used FlashAttention so quadratic attention isn't a problem at 1M tokens," they have confused the two axes; at 1M tokens the FLOPs alone will destroy you.

**📐 Numbers you must know:** arithmetic intensity is the whole game. H100 SXM: ~3.35 TB/s HBM3, ~495 TFLOP/s dense BF16 → ~148 FLOPs per byte before you are bandwidth-bound. Any kernel below that ratio is memory-bound, which is why decode (one token, whole weight matrix read) is bandwidth-bound and prefill (many tokens, same weights) is compute-bound. **📅 Volatile:** per-SKU bandwidth and FLOPs figures change with each hardware generation; verify against the current datasheet.

### Explain PagedAttention in terms a backend engineer would find obvious, then tell me what it actually cost.

This is the paper where your background is worth the most, so lean into it: PagedAttention is virtual memory for the KV cache. Before it, a serving engine allocated one contiguous buffer per sequence sized to the maximum possible length, because attention kernels wanted contiguity. That is `malloc(max_len)` for every request. The waste is exactly the waste you would predict — **internal fragmentation** (a request that generates 200 tokens holding a 2048-token reservation), **external fragmentation** (free space that exists but not contiguously), and **reservation waste** (space held for tokens not yet generated). The vLLM paper measured that existing systems wasted a large majority of KV memory this way.

Mechanism: chop the KV cache into fixed-size *blocks* (commonly 16 tokens per block per layer), keep a **block table** per sequence mapping logical block index → physical block, and write an attention kernel that gathers through the block table instead of assuming contiguity. Now a sequence grows one block at a time, so internal fragmentation is bounded by one block, external fragmentation is eliminated (all blocks are the same size), and — the part that pays for itself twice — blocks can be **shared** across sequences with copy-on-write. Parallel samples from one prompt, beam search branches, and multiple requests hitting the same system prompt all share prefix blocks physically. That sharing is the foundation of automatic prefix caching.

What it bought: the paper reports 2–4× throughput improvements over the strong serving baselines of its time at the same latency, with the gain largest for long sequences, large models and complex decoding — because those are the cases where memory, not compute, capped batch size. That mechanism is the one to state: **PagedAttention does not make any single request faster. It raises the number of concurrent sequences you can hold, which raises batch size, which raises throughput.** Per-request latency at low load is unchanged or very slightly worse from the indirection.

**💰 Math on why it matters to your bill:** take Llama-3-70B with GQA (8 KV heads, d_head 128, 80 layers) at fp16. KV bytes per token = 2 × 80 × 8 × 128 × 2 = 327,680 ≈ 320 KiB. Note the weights alone are 140 GB in fp16, so this does not fit on a two-card 80 GB pair with room to spare — take a four-card 80 GB node and suppose ~30 GiB of the remainder is available for KV; 30 GiB / 320 KiB ≈ 98,000 tokens of cache. If your average request is 2k prompt + 500 output but you were reserving 8k, you held 8k × 320 KiB = 2.5 GiB per sequence and fit 12 concurrent; paged, you hold ~2.5k tokens ≈ 800 MiB and fit ~39. Three times the batch at the same hardware is close to three times the throughput in a decode-bound regime, which is a direct 3× reduction in $/token.

**⚠ Trap:** citing PagedAttention as a latency optimization in an interview. It is a memory-management optimization whose effect is on throughput and admission. Getting that causal chain wrong is a tell that you read the abstract.

**📄 Paper:** Kwon et al. (2023), SOSP — block-based non-contiguous KV allocation with a block table and copy-on-write sharing; it replaced per-sequence contiguous pre-allocation and made prefix sharing physically possible.

### Speculative decoding sounds like it should change the output distribution. Prove to me that it doesn't.

Start with why it works at all: decode is memory-bandwidth-bound. Generating one token from a 70B model in fp16 requires reading ~140 GB of weights; at ~3.35 TB/s that is ~42 ms of pure weight movement at one card's worth of HBM bandwidth (140 GB does not fit on one 80 GB card, so in practice you shard tensor-parallel over two and both the bytes per card and the time roughly halve — the ratio is the point, not the absolute), while the arithmetic (2 × 70e9 = 140 GFLOPs) takes well under a millisecond on a card doing hundreds of TFLOP/s. So the target model is idle-ish per token, and *verifying* several tokens costs almost the same as generating one — the weights get read once either way. Speculative decoding monetizes that: let a cheap draft model propose γ tokens, then have the target model score all γ+1 positions in a single forward pass.

Now the correctness argument, which is the actual question. Let p be the target distribution at a position and q the draft's. For each drafted token x sampled from q, accept it with probability min(1, p(x)/q(x)). If rejected, sample a replacement from the **residual** distribution, normalized `max(0, p(x) − q(x)) / Σ_x max(0, p(x) − q(x))`. The claim is that a token emitted by this procedure is distributed exactly as p.

Sketch the proof in the room like this: P(emit x) = P(draft proposed x and accepted) + P(rejected) × P(residual sampled x). The first term is q(x)·min(1, p(x)/q(x)) = min(q(x), p(x)). The rejection probability is Σ_x q(x)·(1 − min(1, p(x)/q(x))) = Σ_x max(0, q(x) − p(x)), which equals the normalizer of the residual, Σ_x max(0, p(x) − q(x)) (both equal the total variation distance between p and q). So the second term is exactly max(0, p(x) − q(x)). Adding: min(q,p) + max(0, p−q) = p(x). Exactly p, for any draft q, however bad. That is the elegant part — a *terrible* draft model costs you speed and never costs you correctness.

Speed math. With acceptance rate α and γ drafted tokens, expected tokens per target forward pass is (1 − α^(γ+1))/(1 − α). At α = 0.7, γ = 4: (1 − 0.7⁵)/0.3 = (1 − 0.168)/0.3 = 2.77 tokens per target pass. Subtract the draft cost: if the draft is 1/20th the size and you run it 4 times, that is 4/20 = 0.2 target-equivalents, so effective speedup ≈ 2.77/1.2 ≈ 2.3×. That arithmetic is the answer to "how much will this help us," and you should be able to do it live.

**⚠ Trap:** two of them. First, "we're at temperature 0 so acceptance is deterministic" — at greedy decoding the scheme degenerates to exact-match checking, which is fine, but the acceptance rate is then far more brittle to draft/target divergence. Second, and more expensive in production: speculative decoding helps **latency at low batch size** and can *hurt* throughput at high batch, because once you are compute-saturated with a large batch the target forward is no longer free, and you are spending real FLOPs verifying tokens you will throw away. The rule I enforce: enable it for interactive low-concurrency traffic, measure before enabling it on a saturated batch server.

**📄 Paper:** Leviathan et al. (2023) and, concurrently, Chen et al. (2023) — modified rejection sampling that provably preserves the target distribution while amortizing the memory-bound decode step over multiple tokens.

### Everyone says "RAG" and cites Lewis 2020. What did that paper actually do, and why does the mismatch matter?

This is a favourite trap question at labs, and the honest answer scores. The paper is *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, Lewis et al. (2020), and it describes a **trained end-to-end system**, not a prompting pattern. The architecture is a DPR dense retriever (BERT-based query and document encoders) over a Wikipedia dense index, feeding a BART-large generator, with the **query encoder and the generator fine-tuned jointly** by backpropagating through the retrieval marginalization; the document encoder and index are frozen because re-indexing every step is impractical.

The genuinely important technical content is the two marginalization variants. **RAG-Sequence** retrieves top-k documents once and marginalizes over documents at the sequence level — p(y|x) ≈ Σ_z p(z|x)·p(y|x,z). **RAG-Token** marginalizes per generated token, so different tokens in one answer can be sourced from different documents, which helps when an answer combines facts. That per-token marginalization is a real probabilistic mechanism and it has essentially nothing to do with what you do today. The framing that mattered most conceptually: parametric memory (the weights) plus **non-parametric memory** (a swappable index), with the property that you can update knowledge by swapping the index without retraining — which the paper demonstrates by hot-swapping a Wikipedia snapshot.

Why the mismatch matters: what the industry calls RAG in 2026 — embed chunks, top-k nearest neighbours, stuff them into a prompt, generate — is *retrieval-augmented prompting* with a frozen instruction-tuned model and no learned retriever, no marginalization, and no joint training. If you cite Lewis 2020 as the origin of your pipeline in front of someone who has read it, you have shown you have not. The correct move is to cite it for the *framing* and be explicit about the delta.

**🗣 Say this in the room:** "Lewis 2020 is where the parametric-plus-non-parametric framing comes from, and it's a jointly-trained DPR-plus-BART system with sequence- and token-level marginalization over retrieved documents. What we ship isn't that — it's retrieval-augmented prompting against a frozen model. I cite it for the concept and the index-swap property, not for the architecture."

**⚠ Trap:** the reverse error — dismissing the paper as obsolete. The joint-training idea is coming back: learned/embedded retrievers fine-tuned on downstream signal outperform off-the-shelf embeddings on domain corpora, and that is exactly the paper's thesis.

### What is ColBERT's late interaction, and when would you actually pay for it?

Mental model: there are two extremes in neural retrieval. A **cross-encoder** puts query and document in the same forward pass so every query token can attend to every document token — maximum quality, and completely unusable for first-stage retrieval because you would run the model once per document. A **bi-encoder** embeds query and document independently into one vector each, so retrieval is a single ANN lookup — fast, and it destroys information by compressing a document into one point. ColBERT's late interaction is the interpolation: encode independently (so documents can be precomputed and indexed) but keep **one vector per token**, and compute relevance at query time by summing, over query tokens, the maximum similarity against any document token — the MaxSim operator.

Score(q, d) = Σ_{i ∈ query tokens} max_{j ∈ doc tokens} (E_qi · E_dj). The interaction is "late" because it happens after independent encoding, which is what preserves the offline-indexable property. The intuition for why it beats a single vector: a document about three different things gets three regions of embedding space instead of an average that is about none of them, and exact-ish term matching survives — ColBERT tends to be markedly more robust than dense bi-encoders on out-of-domain corpora and on rare entities, product codes, and identifiers, which is exactly the failure mode enterprise RAG hits.

The cost, which is the second half of the question. You store a vector per token instead of per chunk. **💰 Math:** a 512-token chunk with a 128-dim ColBERT vector at fp16 is 512 × 128 × 2 = 131 KB, versus a single 1024-dim fp16 chunk embedding at 2 KB — roughly 65× the index. For 10M chunks that is 1.3 TB versus 20 GB. ColBERTv2 attacks precisely this with residual compression — cluster centroids plus low-bit residuals — bringing it to a few tens of bytes per token, so call it a 6–10× index inflation rather than 65×. That is the honest number to quote.

When I would pay for it: domain corpora full of identifiers and jargon where dense embeddings are weakest, or where you have no in-domain training data to fine-tune a bi-encoder with. When I would not: general web-ish text where a good bi-encoder plus BM25 hybrid plus a cross-encoder reranker over the top 100 gets you there for a tenth of the operational complexity. That hybrid-plus-rerank stack is the default I would defend in a design round; ColBERT is the answer when the reranker cannot fix what first-stage recall already lost.

**📄 Paper:** Khattab & Zaharia (2020) — token-level late interaction with MaxSim; ColBERTv2 (Santhanam et al., 2022) added residual compression and denoised supervision, which is what made the index size survivable.

### Self-RAG proposes retrieval as a decision rather than a step. Walk me through it and tell me whether you'd ship it.

Mental model: standard RAG retrieves unconditionally, a fixed k, every time. That is wrong in two directions — for "what's 2+2" or "rewrite this email politely" retrieval is pure noise injection and latency, and for a hard multi-hop question one round of k=5 is not enough. Self-RAG's move is to make retrieval and its use into *decisions the model itself makes*, by training it to emit special **reflection tokens** interleaved with the output.

Mechanism: four kinds of reflection token. A `Retrieve` token decides whether retrieval is needed at this point (yes/no/continue). Then, for each retrieved passage, `ISREL` judges whether the passage is relevant, `ISSUP` judges whether the generated segment is actually supported by that passage, and `ISUSE` rates the overall usefulness of the response on a scale. Training is a distillation: a strong critic model labels a corpus with these tokens offline, and then the *generator* is trained on that augmented corpus so it emits them itself at inference — meaning you do not need to run a separate critic at serving time. At inference you can generate several continuations conditioned on different passages and score them with a weighted combination of the reflection-token probabilities, effectively a self-critique beam search, and the weights are tunable at inference time to trade fluency against citation-faithfulness without retraining.

That last property is the one I care about as an engineer: a knob that trades groundedness against helpfulness *at inference time* is exactly what a product needs, because legal and medical surfaces want it cranked one way and a brainstorming surface the other.

Would I ship it? Not as-published, usually, and I would say so directly. It requires fine-tuning your generator, which forecloses using the strongest hosted frontier model — and in my experience the delta from switching to a stronger model exceeds the delta from Self-RAG on the same base. What I *do* ship is the decomposition: a cheap retrieval-necessity classifier in front of the pipeline, per-passage relevance filtering before context assembly, and a support/groundedness check on the generated answer with a re-ask on failure. Those are the three reflection tokens implemented as pipeline stages with a frozen model, which captures most of the value at zero training cost.

**⚠ Trap:** treating "adaptive retrieval" as free. Each extra decision point is another model call and another failure mode; a naive Self-RAG-shaped pipeline with per-passage relevance judging at k=10 is eleven calls where you had one. **💰 Math:** at $3/Mtok input and 600 tokens per relevance judgement, 10 judgements/query = 6,000 tokens = $0.018 per query on top of the answer call; at 100k queries/day that is $1,800/day, or $54k/month, to filter passages. Batch those judgements into one call with a structured output and you pay the rubric and the query *once* instead of ten times — but you still send all ten passages, so the saving is bounded by the boilerplate fraction, not a free 10×. If the rubric plus query is 300 of the 600 tokens, you go 6,000 → ~3,300 tokens, roughly 2×; prompt caching on the shared prefix takes it further. Do the split explicitly before quoting a number. **📅 Volatile:** per-token prices move; re-price against the current rate card.

**📄 Paper:** Asai et al. (2023) — self-reflection tokens that make retrieval and its use learned decisions with inference-time controllability.

### ReAct and Toolformer are both "tool use" papers. What is each one's actual contribution, and which one still shapes how you build?

They solve different problems and conflating them is a common tell.

**ReAct** (Yao et al., 2022) is a *prompting/inference structure*. Its observation: chain-of-thought is reasoning without grounding, so it hallucinates facts and cannot recover from a wrong premise; act-only agents take actions without a reasoning trace, so they cannot plan or handle exceptions. Interleave them — Thought, Action, Observation, repeat — and each fixes the other's failure mode: the thought decides which action to take, the observation injects external ground truth that can correct the thought. Evidence was on HotpotQA/FEVER with a Wikipedia search API and on ALFWorld/WebShop for decision-making, with the interesting result that ReAct beat act-only substantially on the interactive benchmarks and that combining ReAct with CoT-SC beat either alone. What it replaced: CoT-only and act-only agents, and it is the direct ancestor of essentially every agent loop shipped since.

**Toolformer** (Schick et al., 2023) is a *training* paper. Its problem: how do you teach a model to call APIs without a hand-labelled dataset of tool calls? Its answer is beautifully self-supervised — prompt the model to propose candidate API calls at plausible positions in ordinary text, actually execute them, and **keep a call only if inserting the call and its result reduces the perplexity of the subsequent tokens** compared to no call or an empty result. Then fine-tune on the filtered corpus. The model learns *when* and *with what arguments* to call a tool from a signal that requires zero human annotation.

Which still shapes how I build: ReAct, overwhelmingly — but not in its original form, and you should say why. Modern models have native tool-calling trained in, so you no longer parse `Action: search[...]` out of free text; you get structured tool-call objects with schema-validated arguments. The ReAct *loop* survives (reason → act → observe → repeat, with a termination condition); the ReAct *prompt format* is obsolete and using it on a modern model is actively worse than the native API because you lose constrained decoding on arguments. Toolformer's direct recipe is largely superseded — providers ship tool-calling in post-training — but its idea is alive as the standard way to *generate synthetic tool-use training data*: propose, execute, keep what verifies. That is the same filter, and it is how you would build a fine-tuning set for a bespoke internal tool today.

**⚠ Trap:** presenting ReAct as an architecture. It is a prompting pattern and a loop invariant. If an interviewer asks "how would you improve a ReAct agent," the answers live in tool design, observation truncation, and termination — not in the pattern itself.

**🗣 Say this in the room:** "ReAct's contribution is that grounding and reasoning fix each other's failure modes, and that loop is still what I build. The specific text format is dead — native tool calling gives you schema validation the free-text parser never had."

### Explain sparse MoE using Mixtral and DeepSeek-V3, and tell me why activated-parameter count is a misleading serving metric.

Mental model: in a dense transformer every token pays for every FFN parameter. But an FFN is doing many unrelated things, and most tokens need only a few of them. MoE replaces one FFN with N experts and a router that sends each token to the top-k, so parameter count (capacity, knowledge) decouples from FLOPs per token (compute). You buy quality with memory instead of with compute.

**Mixtral 8x7B** (Jiang et al., 2024): each layer has 8 experts, a router picks top-2 per token, outputs are combined weighted by the router's softmax. Totals are 46.7B parameters — note it is *not* 8 × 7B, because attention and embeddings are shared — with roughly 12.9B active per token. Its evidence was matching or beating Llama-2-70B while computing like a ~13B model.

**DeepSeek-V3** pushes the same idea much harder with three refinements worth naming. **Fine-grained experts**: many more, smaller experts, so top-k selects a richer combination and specialization is sharper. **Shared experts**: one or more experts every token always goes through, which absorbs the common-to-everything computation so the routed experts do not each have to relearn it. **Auxiliary-loss-free load balancing**: instead of an auxiliary balancing loss that fights the main objective, add a per-expert *bias* to the routing scores and adjust it dynamically — push it down for overloaded experts, up for underloaded ones — so balancing happens without a gradient that degrades quality. It totals 671B parameters with ~37B active.

Now the real question. Activated parameters predict **FLOPs per token**; they predict almost nothing about serving cost, for three reasons.

**First, memory.** You must hold *all* the weights resident. DeepSeek-V3 at fp8 is ~671 GB of weights — nine H100-80GB cards for weights alone, realistically 16 with KV cache and activation space — even though each token only touches 37B of them. A dense 37B model needs one card. **💰 Math:** Mixtral at bf16 is 46.7e9 × 2 = 93.4 GB, so two 80 GB cards minimum, to compute like a 13B model that would fit on one. Your $/GPU-hour is set by the 93.4 GB, not by the 12.9B.

**Second, memory bandwidth at decode.** Decode reads weights, and with a batch of B different tokens routing to different experts, the union of experts touched grows with B. At batch 1 you read ~12.9B params' worth; at batch 64 across 8 experts you likely touch all of them, so you read the full 46.7B and your arithmetic intensity per token collapses toward the dense case.

**Third, all-to-all communication.** With expert parallelism the routed tokens must be dispatched to the GPUs holding their experts and gathered back — twice per MoE layer. That is a latency floor set by your interconnect, and it is why MoE serving is much more sensitive to NVLink-vs-Ethernet topology than dense serving.

**🗣 Say this in the room:** "Activated parameters tell you FLOPs. Total parameters tell you how many GPUs you rent. For MoE the second number sets your bill, plus an all-to-all you don't have in a dense model — so I'd never quote 'it's a 13B-active model' as a cost estimate."

### Explain YaRN and what it replaced. Then tell me what breaks in production when someone applies it.

Mental model: RoPE encodes position by rotating each 2-dimensional slice of the query and key vectors by an angle proportional to position, with per-slice frequencies θ_i = base^(−2i/d). Low-index dimensions rotate fast (short wavelength, encode local order), high-index dimensions rotate slowly (long wavelength, encode global position). If you trained at 4k and run at 32k, the slow dimensions see rotation angles they have never seen — out of distribution — and attention degrades. Context extension is the problem of getting those angles back into distribution.

The lineage matters. **Position Interpolation** (Chen et al., 2023) divides all positions by the extension factor s, so position 32000 is presented as 4000. Uniform, simple, and it works with a short fine-tune — but it squeezes the *high*-frequency dimensions too, which is where fine-grained local ordering lives, so you lose short-range precision. **"NTK-aware" scaling** instead raises the RoPE base, which spreads the compression non-uniformly, better but ad hoc.

**YaRN** (Peng et al., 2023) formalizes this as **NTK-by-parts**: compute each dimension's wavelength, and decide per dimension. Dimensions whose wavelength is much shorter than the original training context are already fully-cycled and seen every phase — leave them alone. Dimensions whose wavelength exceeds the original context have never completed a cycle — interpolate them fully. Blend in between with a ramp. Second component, and the one people forget: **attention temperature**. Extending context increases the number of tokens competing in the softmax, so entropy rises and attention flattens; YaRN scales the attention logits by a factor derived from the extension ratio (implementable for free by folding a constant into the q/k scaling), which recovers sharpness. The result is much better perplexity at extended context with roughly an order of magnitude less fine-tuning data and far fewer steps than PI-style approaches.

Now the production failures, which is the half that earns the point.

**⚠ Trap 1 — the silent prefix-cache invalidation.** Changing `rope_scaling` or the RoPE base changes every key and value tensor for the *same* token sequence. Any KV cache — a serving engine's automatic prefix cache, a persisted session cache, an offline-precomputed document cache — is now numerically wrong but structurally valid, so nothing errors. You get a model that is subtly, unreproducibly worse for the requests that hit warm cache and fine for cold ones. The rule I enforce: RoPE config is part of the cache key, alongside model version and quantization.

**⚠ Trap 2 — extending without validating short-context.** YaRN's whole point is to protect short-range behavior, but any extension plus fine-tune can regress it. You must re-run your short-context evals; a 128k model that lost two points on your 2k-token production distribution is a net loss, because that is where your traffic is.

**⚠ Trap 3 — assuming static scaling is free at short lengths.** With static YaRN the scaling applies at every length, including the ones the model was originally fine on. Dynamic variants that scale as a function of current sequence length exist for exactly this reason — but they change the transformation mid-sequence, which again means cached keys computed at one length are inconsistent with a later one unless the implementation handles it.

**📐 Numbers you must know:** advertised context and usable context are two different numbers and you should always state both. Extension gives you the *positional* capability; it does not give you retrieval quality at that length. Always pair a context-extension claim with a multi-fact eval on your own corpus at the target length. **📅 Volatile:** every vendor's advertised limit; verify before quoting.
### I'm going to hand you a paper you've never seen and give you twenty minutes. Talk me through how you read it.

The mental model is triage, not reading. Twenty minutes is not enough to understand a paper; it is enough to build a correct *model of the claim and its support*, which is all the round needs. So I read in a deliberately non-linear order designed to answer four questions as early as possible: what is being claimed, against what, measured how, and what would have to be true for it to be wrong.

My order, with rough time budget. **Minute 0–2: title, abstract, and the last figure or table.** The final table is usually the headline result, and reading it before the prose stops the paper from framing my interpretation. I want to see the axes, the baselines listed, and whether there are error bars. **Minute 2–4: introduction's last paragraph and the contributions list.** This is where the authors state the claim in their own strongest terms — I write it down verbatim, because I will hold them to it. **Minute 4–8: the method figure and the one or two equations that carry the idea.** I try to write the mechanism in one sentence of my own words; if I cannot, I have not understood it and I go back rather than forward. **Minute 8–14: the experimental setup.** This is where the actual information density is — what datasets, what model sizes, what baselines, what hyperparameter search, what was held fixed, how many seeds. **Minute 14–17: ablations.** Which component is load-bearing, and is there an ablation missing that would be embarrassing. **Minute 17–20: limitations and related work**, both read adversarially — related work tells me what the authors think they are beating, and any prior art conspicuously absent is a signal.

I skip, on a first pass: proofs, most of the related-work prose, appendix hyperparameter tables (unless the setup section was suspiciously thin, in which case that *is* the story), and all qualitative examples, which are selected and therefore uninformative.

Then I talk. And the way I open matters: I state the claim, then the mechanism in one sentence, then the single experiment that most supports it, then the single experiment I most wish existed. That last item is what the round is scoring — "what would you run next" is the question a researcher actually cares about, because it is the question they ask themselves.

**⚠ Trap:** reading linearly and running out of time in section 3. I have watched candidates spend eighteen of twenty minutes on the method and then say "I didn't get to the results." That is a fail, because the method is the part you can reconstruct from the figure and the results are the part you cannot.

**🗣 Say this in the room:** "Before I go through it — my read is that the claim is X, the mechanism is Y, and the load-bearing evidence is table 3. I'll say where I think that's well supported and where I'd want another experiment. Stop me if you'd rather I go deeper on the method."

### Give me your critique framework. What are you actually looking for when you attack a paper?

Five axes, in this order, because each one is cheaper to check than the next and can kill the paper on its own.

**1. Is the claim what the evidence supports?** Read the abstract's strongest sentence and the results table side by side. The most common gap in ML papers is a claim of generality supported by evidence on one model family, one scale and two datasets. Not fraud — just a scope mismatch, and naming it precisely ("the claim is architecture-level, the evidence is one 7B decoder on two English benchmarks") is a strong opening.

**2. Are the baselines honest?** This is where most of the value is and I will give it its own answer, but the checks are: was the baseline tuned with the same effort as the method; is the baseline the *current* best or a convenient older one; is the comparison at matched compute, matched parameters, or matched inference cost — and did they say which; is there a trivially strong baseline missing (for retrieval papers, BM25; for reasoning papers, majority-vote self-consistency at matched sample budget; for agent papers, a single well-prompted call).

**3. Do the ablations isolate the contribution?** If the method has three components and there is one ablation table with one row removed at a time, that tells you about marginal contribution but nothing about interactions. If the paper introduces a new component *and* a new training recipe *and* more data, and the ablation only removes the component, the result is unattributable.

**4. Threats to validity.** Contamination — could the eval data be in pretraining, and did they check? Test-set selection — were the benchmarks chosen after seeing results? Evaluation protocol — for LLM-judged results, which judge, and was the judge's agreement with humans measured, and is the judge from the same family as the method (self-preference bias is real and large)? Variance — how many seeds, and are the error bars over seeds or over the eval set's sampling? Those are different quantities and papers conflate them.

**5. What would I run next?** Three flavors: the ablation that would attribute the gain, the scale point that would tell you whether the effect survives (effects that shrink with scale are extremely common and are the single most important thing to ask about), and the deployment-realistic version (does the gain survive quantization, batching, or a different prompt).

**⚠ Trap:** critiquing by attacking novelty ("this is just X with Y"). It is almost always partly true, it is almost never interesting, and at a lab it reads as a reviewer-2 reflex rather than an engineer's judgment. Attack the *support*, not the originality.

**🗣 Say this in the room:** "My main concern isn't the mechanism, it's attribution — they changed three things and ablated one. I'd want the run with the new recipe and the old component before I believed the component is what's doing the work."

### How do you tell whether a reported improvement is real or inside the noise band?

The mental model backend engineers already have: this is exactly a latency-regression question. You would never accept "p99 dropped 3 ms" from a single run of a load test, because you know the run-to-run variance. ML papers report single-run deltas constantly, and the discipline is to reconstruct the variance the paper did not report.

There are two independent noise sources and you must name both. **Evaluation sampling noise**: your benchmark is a sample from a population of possible items, and accuracy on it has a standard error. **Training/inference stochasticity**: seeds, data order, and — for anything sampled at temperature > 0 — decoding randomness.

Worked example, and you should be able to do this arithmetic live. A benchmark with n = 1,000 items, baseline 72.0%, method 73.8%, so Δ = 1.8 points. The standard error of a proportion is √(p(1−p)/n) = √(0.72 × 0.28 / 1000) = √0.0002016 = 0.0142, i.e. 1.42 points. For two *independent* samples the SE of the difference is √2 × 1.42 ≈ 2.0 points, so a 95% interval on the difference is ±3.9 points. A 1.8-point gain is nowhere near significant under that analysis.

But that analysis is too pessimistic, because both systems were evaluated on the **same items** — this is a paired design, and you should use it. **McNemar's test** looks only at discordant pairs: let b = items the baseline got right and the method got wrong, c = the reverse. Suppose b = 60 and c = 78; the net is 18/1000 = 1.8 points, consistent with what was reported. χ² with continuity correction = (|b − c| − 1)² / (b + c) = (18 − 1)² / 138 = 289/138 = 2.09, against a critical value of 3.84 at 1 df — p ≈ 0.15. Still not significant, and now you have said so with the right test. The paired analysis is *more* powerful than the unpaired one and it still fails, which is a much stronger statement.

**📐 Numbers you must know:** to detect a 2-point absolute difference around 70% accuracy at 80% power and α = 0.05, unpaired, you need roughly n = 2 × (1.96 + 0.84)² × p(1−p) / δ² = 2 × 7.85 × 0.21 / 0.0004 ≈ 8,200 items per arm. Most published benchmarks have 500–2,000. That single calculation tells you why most 1–2 point deltas in the literature are uninterpretable, and it is the most useful number in this entire section. Paired designs and paired bootstrap cut the requirement substantially — often by 3–5× depending on how correlated the systems are — which is why you should always evaluate on identical items and report the paired statistic.

For continuous or composite metrics where no closed form applies, use the **paired bootstrap**: resample the eval set with replacement 10,000 times, recompute the delta each time, and report the 2.5th and 97.5th percentiles. Twenty lines of code, no distributional assumptions, and it is what I would ask a candidate to write.

**⚠ Trap:** reporting the *best* of several seeds and comparing to the baseline's single seed. This is endemic and often unintentional. The rule I enforce in review: the number of seeds must be equal across arms, stated, and the reported statistic must be the mean with a spread — never the max.

### Design an ablation for me. Say I've built a RAG pipeline with a rewriter, a hybrid retriever, and a reranker, and it beats the old one by 6 points.

The mental model that makes ablation design non-arbitrary: an ablation is not "turn things off," it is **an attribution experiment for a specific causal question**, and you should write the question down before you design the run. Here the question is "how much of the 6 points is attributable to each component, and do they interact?" — and the answer determines what you delete to cut cost, which is the reason a business pays for the experiment.

Start with the crucial control that almost everyone skips: **hold the compute and the context budget fixed.** If the new pipeline puts 4,000 tokens into the prompt and the old one put 1,500, then part of your 6 points is "more context," not "better retrieval," and you have not measured what you think. So the first ablation arm is the old pipeline with k raised until the context budget matches. I have seen this single control erase half of a reported gain more than once.

Then the design. With three binary components there are 2³ = 8 configurations, and if each eval run is cheap you should just run the **full factorial** — eight arms gives you all three main effects *and* all the interactions, which one-at-a-time removal cannot give you. The interactions are the interesting part here and you should predict them out loud: I expect the rewriter and the reranker to be **substitutes**, not complements, because a reranker over a wider candidate set can recover documents a bad query would have missed, so the rewriter's marginal value should be much smaller when the reranker is on. If the factorial confirms that, you delete the rewriter and save a model call per query.

If eight arms is too expensive, run the leave-one-out set (3 arms) plus the single-component set (3 arms) plus full and none — that *is* the factorial, so the real economy is to reduce the eval set size per arm, which you should do only after computing whether the reduced n can still resolve the effect sizes you expect (see the power calculation above; if you expect a 1.5-point component effect you need thousands of items).

Additional arms I would insist on. **An oracle-retrieval arm**: feed the known-correct passages and measure end-to-end accuracy. That gives you the ceiling and tells you whether your remaining errors are retrieval failures or generation failures — the single most useful diagnostic in RAG, and it costs one run. **A no-retrieval arm**: the model alone. If the model alone gets 70% of your 6-point gain, your retrieval is mostly decorative on this eval, which means your eval is wrong.

Report as a table with per-arm mean, seeds, and a paired confidence interval against the full system — not against "none," because the decision you are making is *what to remove*, and the comparison should match the decision.

**💰 Math on why this pays:** if the rewriter costs one extra call at 800 tokens in / 60 out per query, at $3/Mtok in and $15/Mtok out that is 800 × 3e-6 + 60 × 15e-6 = $0.0024 + $0.0009 = $0.0033 per query. At 500k queries/day that is $1,650/day = ~$49,500/month. An eight-arm ablation that costs a few hundred dollars of eval compute to find out the rewriter contributes 0.3 points is the highest-ROI experiment on the roadmap.

### How many seeds do you need, and what actually varies between them?

Mental model: "seed" is not one knob. In a modern training or eval stack, run-to-run variance comes from at least five places, and knowing which ones you controlled is the difference between a reproducible result and a coincidence. **Weight initialization** (for anything trained from scratch or with newly-initialized adapters — LoRA's A matrix is random). **Data order** — the shuffle, and for packed sequences, which documents landed in which batch. **Dropout and any other stochastic layer.** **Non-determinism in the kernels themselves** — atomics and reduction order in cuDNN/Triton kernels make even a fixed-seed GPU run non-bit-reproducible unless you force deterministic algorithms, and doing so costs throughput. **Decoding randomness** at eval time if temperature > 0.

The practical answer on count: fine-tuning results should be reported over at least 3 seeds and ideally 5, with mean and standard deviation, because the seed-to-seed spread on downstream benchmarks is frequently the same magnitude as the effect people are trying to claim. RL results need more — reinforcement learning is notoriously high-variance and a 3-seed RL comparison is weak evidence. If the compute genuinely does not allow multiple seeds, the honest move is to say "single seed, so treat the ordering as a hypothesis" rather than to report a bolded number.

For *evaluation-only* comparisons where nothing is trained, the seed question collapses to decoding: fix temperature to 0 for a deterministic protocol, or — better for anything where sampling is part of the product — run k samples per item and report the mean with its interval, because temperature-0 evaluation systematically over-states reliability of a system you will ship at temperature 0.7.

**⚠ Trap:** believing `torch.manual_seed` gives you reproducibility. It fixes Python/PyTorch RNG streams. It does not fix cuDNN algorithm selection, atomic reduction order, or the fact that changing batch size changes the reduction tree and therefore the numerics. Two runs with the same seed and different tensor-parallel degrees will diverge. The rule I enforce: reproducibility claims must state the hardware, the library versions, the parallelism configuration, and whether deterministic kernels were forced.

**🗣 Say this in the room:** "Before I believe a one-point improvement I want to know the seed-to-seed spread of the baseline. If nobody has measured it, that's the first run I'd do — it costs one extra training run and it tells you whether the rest of the experiment grid is worth running at all."

### You decide to reproduce a published result. What usually breaks?

Mental model: papers report the *destination*, not the road. The reported hyperparameters are the ones the authors thought mattered, which is a subset of the ones that do. Reproduction is mostly an exercise in recovering unstated defaults.

The failure taxonomy, in the order I hit them.

**🔍 Failure taxonomy — reproduction:**

**1. Data.** The most common and the most expensive. The stated dataset is a version, and versions drift — dataset cards get updated, deduplication differs, the split you downloaded is not the split they used, and preprocessing (whether you strip the system prompt, how you template chat turns, whether you count the prompt tokens in the loss) is described in one sentence that admits four implementations. Chat templating alone is responsible for a shocking share of failed fine-tuning reproductions: applying the wrong template silently produces a model that trains fine and evaluates badly.

**2. Evaluation harness.** Two harnesses can score the same model on "MMLU" and differ by several points because of answer extraction, whether they score by log-likelihood of the option letter versus generating and parsing, few-shot count, prompt format, and normalization. Before concluding the *model* differs, verify your harness reproduces the *baseline* number the paper reports. That is the single highest-value control in a reproduction and it is cheap.

**3. Numerics and hardware.** bf16 versus fp16 versus fp32 master weights, whether attention is computed with a fused kernel, TF32 matmul settings, and gradient accumulation ordering all move results by fractions of a point — and occasionally more when training is near an instability.

**4. Unstated hyperparameters.** Warmup steps, LR schedule shape, weight decay on which parameter groups (norm and bias parameters are usually excluded — is yours?), gradient clipping threshold, Adam β₂ and ε, effective batch size after accumulation and data parallelism.

**5. The thing that was never in the paper.** A pretraining data mix, an internal filtering model, a checkpoint selected on a validation set that is not released. When the gap survives everything else, this is usually it, and the honest reproduction report says so instead of blaming itself.

**💰 Math on scoping a reproduction before you start:** suppose you want to reproduce a LoRA fine-tune of an 8B model on 50k examples averaging 1,000 tokens. LoRA's backward skips weight-gradient matmuls for frozen weights, so cost is roughly 4N per token rather than the 6N of full training: 4 × 8e9 × 5e7 = 1.6e18 FLOPs. On an H100 at ~495 TFLOP/s dense BF16 peak and a realistic 40% MFU ≈ 198 TFLOP/s, that is 1.6e18 / 1.98e14 ≈ 8,080 seconds ≈ 2.25 hours per epoch. At roughly $3/GPU-hour that is under $7 per run — so a 5-seed reproduction is $35 of compute and two days of your time, and the compute was never the constraint. Knowing that changes what you are willing to attempt. **📅 Volatile:** GPU rental rates; re-verify.

### What goes in a reproduction report, and why is that a hiring artifact?

The mental model for why this converts: a reproduction report is the cheapest possible demonstration of the two things labs cannot screen for otherwise — that you can run an experiment correctly, and that you will report an unflattering result honestly. Almost nobody publishes one. That scarcity is the entire value.

Structure I would insist on, and it maps almost exactly onto a good incident report, which is a format you already own:

**1. The claim under test, quoted.** One sentence, verbatim from the paper, plus the specific table and row you are targeting. Reproductions fail most often because the target was vague.

**2. What you ran.** Model checkpoint with its exact revision hash, dataset with version, code with commit, hardware, library versions, parallelism configuration, and every hyperparameter — including the ones you guessed, explicitly marked as guesses. The "marked as guesses" column is the credibility signal.

**3. The control.** Did you reproduce the paper's *baseline* number with your harness? If not, stop and fix that first, and report the gap. A reproduction that cannot reproduce the baseline has not measured the method.

**4. The result, with variance.** Mean over N seeds, standard deviation, and a paired interval against your own baseline. Not the best seed.

**5. The delta and your explanation of it.** If you got 71.2 where they reported 73.9, you owe a hypothesis and, ideally, one experiment testing it — "our chat template omitted the trailing newline; adding it recovers 1.4 points" is worth more than the original reproduction.

**6. What you could not test and why.** Compute limits, unreleased data, missing code. State it plainly.

**7. A conclusion with a confidence word.** "Reproduced," "reproduced with caveats," "directionally reproduced, magnitude smaller," "not reproduced." Pick one and defend it.

**8. A runnable repo.** One command, a pinned environment, a seed, and the raw result files committed — not just the summary table.

**⚠ Trap:** framing a failed reproduction as an accusation. It almost never is one — the usual cause is an unstated detail — and a report written with a prosecutorial tone reads as junior and burns the relationship with authors who would otherwise answer your email. Write it as "here is what I could and could not recover, and here is my best guess at the delta," and then actually email the authors. In my experience they reply more often than people expect, and a cited reply in your report is a very strong artifact.

**🏋 Drill:** pick one table row from any of the canon papers whose full pipeline you can run for under $50. Reproduce it end to end and publish the report with all eight sections. Pass criterion: a stranger can rerun it from your README in one command, and your report contains at least one number that disagrees with the paper along with your hypothesis for why.

### There's a training loss curve on the whiteboard. Talk me through what you can and cannot conclude from it.

Mental model: a loss curve is a compressed log of an optimization process, and almost every visually striking feature has a *systems* explanation before it has a science explanation. The senior instinct is to ask what changed in the pipeline, not what changed in the model.

What I read, in order. **The axes** — is the x-axis steps, tokens, or wall-clock? Steps hide changes in batch size; tokens is the only axis on which two runs with different batch sizes are comparable. Is the y-axis log-scale? Loss curves look deceptively flat on a linear axis late in training when the interesting differences are in the third decimal. **The overall shape** — a healthy LM pretraining curve is a steep initial drop then a long, slowly-flattening power-law-ish decline; on a log-log plot it should be roughly linear.

Then the features. **A spike that recovers within a few hundred steps** is usually a bad data shard or a numerical event, not a model problem; the standard responses are to lower Adam β₂, add z-loss on the logits, or rewind to the last checkpoint and skip the offending data window. **A spike that does not recover** means you have left the basin and rewind-and-skip is the only reliable fix. **A sudden clean drop** is almost always a schedule event — the decay phase of a WSD/trapezoid schedule, or the tail of a cosine — and it is the single most-misread feature on any loss curve: you cannot compare two runs mid-cosine, because the run whose cosine is closer to its end will look better and that advantage is an artifact of the schedule, not of the model. **A step change at a round step number** is a restart from checkpoint with a changed config; check the data loader state was restored, because a silently-reset shuffle means you are re-training on data you already saw.

What you cannot conclude, and this is the part that scores. **Loss is not quality.** Training loss going down while your downstream eval goes down too is entirely normal in fine-tuning — it is overfitting to the SFT distribution or catastrophic forgetting, and the loss curve is blind to both. **Loss is not comparable across tokenizers**, because it is per-token cross-entropy and a different tokenizer changes the denominator; bits-per-byte is the invariant version. **Loss is not comparable across data mixes**, because an easier validation mix gives a lower number for a worse model. And the gap between train and validation loss tells you much less than it does in classical ML, because at single-epoch pretraining scale there is effectively no memorization gap to see.

**🗣 Say this in the room:** "Before I interpret this I need three things: is x-axis tokens or steps, what's the LR schedule, and is the val set held fixed across the runs being compared. Most of what looks like a modelling difference on a loss plot turns out to be a schedule or a data-mix difference."

### Show me you can argue about a scaling plot. What are the traps?

Mental model: scaling plots are log-log because power laws are straight lines in log-log, and everything that makes them treacherous follows from that. On a log-log axis, a factor-of-two error looks small, extrapolation looks safe, and a slightly wrong slope compounds into an order of magnitude.

The two plot types you must distinguish. **A loss-versus-compute (or -parameters, or -tokens) curve** shows a fitted power law of the form L(N) = E + A/N^α, where E is the irreducible entropy term — and note that the *presence* of E is why these curves bend away from a straight line at large N, and why fitting a pure power law without an offset systematically over-predicts future gains. **An isoFLOP plot** fixes total compute and sweeps the parameter/token split, producing a parabola in log-parameter space whose minimum is the compute-optimal model size at that budget; the envelope of those minima across budgets is what gives you the scaling exponent. If you cannot say which of these two you are looking at, you cannot argue about it.

The traps, which is what the question is for.

**Extrapolation range.** A fit over three orders of magnitude of compute is being used to predict two orders beyond the largest point. That is the whole methodology and it is also the whole risk. Always ask where the largest fitted point is relative to the claim.

**The fit's own uncertainty.** Scaling-law coefficients have confidence intervals that are frequently not shown. The Chinchilla replication episode is the cautionary example: a 2024 replication attempt argued the parametric estimator in the original was mis-fit and inconsistent with the paper's own other two approaches, implying wider uncertainty on the exponents than the crisp "20 tokens per parameter" suggests. The heuristic survives as an anchor; the precision does not.

**Curves that cross.** Two methods compared at one scale tell you nothing about their ordering at another. A very large fraction of published "our method beats X" results at 1B parameters shrink or reverse at 70B, because many interventions are substitutes for capacity. The single best question you can ask about any scaling claim is "does the gap widen, hold, or shrink with scale?" — and if the paper only has one scale point, say that the claim is unfalsifiable at deployment scale.

**Metric choice.** Loss scales smoothly; downstream task accuracy often does not, and apparent "emergence" is substantially a function of using a discontinuous metric (exact match, multiple-choice accuracy) rather than a continuous one. This is a real and important argument in the literature and you should know that "emergence" is contested rather than settled.

**Axis and compute-definition mismatches.** Is compute training-only or does it include the cost of the data pipeline? Is it FLOPs or GPU-hours (which conflates hardware generations)? Is the x-axis total parameters or activated parameters — for MoE those differ by 18× and papers plot both.

**📐 Numbers you must know:** C ≈ 6ND for dense training (2N forward + 4N backward per token) and ≈ 2N per token for inference. Everything on a scaling plot converts through those two identities, so you can always move between the axes live. Example: a 7B model on 1T tokens is 6 × 7e9 × 1e12 = 4.2e22 FLOPs; on 512 H100s at 40% MFU ≈ 512 × 1.98e14 = 1.01e17 FLOP/s, that is 4.2e22/1.01e17 ≈ 415,000 s ≈ 4.8 days.

### A model tops a leaderboard by four points. What do you check before you believe it?

The mental model I would open with: a leaderboard number is a *joint* measurement of a model, an eval harness, a prompt, and a training-data hygiene policy, and only one of those four is the thing you want to know about. Four points can come from any of them.

**Contamination first.** Public benchmarks with fixed test sets and years of web presence are in pretraining corpora. The empirical demonstration everyone should know is the GSM8K case: a research group built GSM1K, a fresh grade-school-math set constructed to match GSM8K's distribution and difficulty, and found that several model families dropped substantially on the fresh set — with some families down by roughly ten points or more — while the strongest frontier models held roughly steady. That gap *is* the contamination measurement, and the methodology (build a distribution-matched fresh set) is the only honest way to get it. So: does the model card report a contamination analysis? Is there a held-out or private variant of the benchmark, and does the ranking hold there?

**Prompt and harness.** Was the number produced with the same few-shot count, answer-extraction rule and scoring method as the baselines it is beating? A model that generates a well-formatted answer scores better under a parsing harness than one that is equally correct but chatty, and that is a formatting win being reported as a capability win. Chat-template mismatches on the *baselines* are a common and usually accidental way to inflate a delta.

**Sampling budget.** Was the winner evaluated with self-consistency, best-of-n, or a longer reasoning budget than the baselines? If the winner used 8 samples and majority vote and the baseline used 1, the correct comparison is at matched inference cost — and at matched cost, the ordering often flips. This is now the single most important normalization for reasoning models, and asking about it is a strong signal.

**Statistics.** Benchmark sizes are often 500–2,000 items; recompute the noise band. On a 1,319-item set at ~80%, SE = √(0.8 × 0.2 / 1319) = √0.0001213 = 0.011 = 1.1 points, so a paired 95% interval is on the order of ±2–3 points and a four-point single-run gain is at best marginal.

**Judge-based benchmarks specifically.** If ranking comes from an LLM judge, check whether the judge shares a family with the winner (self-preference bias), whether position bias was controlled by swapping order, and whether length was controlled — judges systematically prefer longer answers, and length-controlled variants of preference benchmarks exist precisely because the uncontrolled versions were being gamed.

**🗣 Say this in the room:** "Four points on a public benchmark is roughly a coin flip between a real capability gain, a decoding-budget difference, and contamination. I'd want the fresh-set or private-split number, the matched-inference-cost comparison, and the paired interval before I'd let it change a routing decision."

**⚠ Trap:** letting a leaderboard drive a production model choice. The only benchmark that should move your router is your own golden set on your own traffic distribution — and if you cannot say what that set is, that is the finding.

### An interviewer says "our new prompt scored 3 points higher on our eval, ship it." What do you say?

I push back, and specifically I ask four questions in this order, because each one is cheaper than the next.

**"On how many items, and is it the same items?"** If the eval is 150 hand-written cases, a 3-point gain is 4.5 items. Standard error at 80% on n = 150 is √(0.8 × 0.2/150) = 0.0327 = 3.3 points, so the observed gain is smaller than one standard error of a single arm. It is noise, and I would say so plainly. If it is paired on identical items, compute McNemar and quote the p-value rather than arguing about it qualitatively.

**"How many prompt variants did we try to get this one?"** This is the multiple-comparisons problem and it is endemic to prompt engineering, because iterating on a prompt against an eval set *is* a search over hypotheses. If you tried 30 variants and picked the best, the probability that at least one beats baseline by chance at α = 0.05 is 1 − 0.95³⁰ = 1 − 0.215 = 0.785. A 78% chance of a false positive is not a shipping criterion. The fix is not just Bonferroni — it is a **held-out set the variants were never scored against**, and a re-run of the winner on it. Prompt engineering needs train/dev/test discipline exactly as much as model training does, and almost no team applies it. This is one of the highest-value things you can say in an applied AI interview.

**"Did anything else regress?"** A single aggregate number hides slice movement. I want the delta broken out by the slices that matter — the top intents, the long-tail intents, the adversarial set, the languages, the customers who complained last month. A 3-point aggregate gain composed of +6 on the easy majority slice and −4 on the regulated slice is a ship-blocker.

**"What did it cost?"** If the new prompt is 900 tokens longer, at $3/Mtok and 2M calls/month that is 900 × 2e6 × 3e-6 = $5,400/month, and it also changes your prefix-cache boundaries. Three points for $5,400/month may be a fine trade — but it is a trade, and it has to be stated as one.

**🗣 Say this in the room:** "I'd want the paired statistic, the number of variants we searched, a held-out confirmation run, and the per-slice breakdown. If it survives all four I'll ship it today — if we can't answer the second one, the honest read is that we've overfit the eval set and we don't know what we have."

**⚠ Trap:** the eval set that becomes a training set. Once a team has iterated prompts against a golden set for six months, that set no longer measures generalization — it measures how well you have fit it. The rule I enforce: rotate in fresh labelled items continuously and keep a locked holdout that is scored at most monthly, by one owner, with the results dated.
### How do you keep current without drowning? Be specific about your routine.

Mental model: the volume of published work exceeded any individual's reading capacity years ago, so "keeping up" is not a reading problem, it is a **filtering and indexing problem** — and you already know how to build one of those. Treat it like a stream you consume with a cheap classifier in front and a durable store behind. The failure mode is not missing a paper; it is reading forty abstracts a week and retaining nothing you can argue about in an interview.

My routine, concretely. **A weekly 45-minute triage slot, not a daily habit.** Daily arXiv skimming produces recency bias and burnout; weekly batching lets the community pre-filter for you. Sources, ranked by engineering signal per minute:

- **Engine and framework release notes and changelogs** — vLLM, SGLang, TensorRT-LLM, TRL, PyTorch. Highest signal per minute of anything, because a feature landing in a serving engine means a technique survived contact with production. If continuous batching or a new attention backend ships, that is worth more than the paper it came from.
- **Lab engineering blogs and model cards** — the deployment details (context handling, caching semantics, tool-calling behavior, safety mitigations) that never appear in papers.
- **Two or three practitioners whose taste you have verified** over months, not follower count. The verification test: did their enthusiasm about something six months ago turn out to be right?
- **arXiv cs.CL / cs.LG listings**, skimmed by title only, plus whatever gets discussed by the above.
- **Conference proceedings** as a *retrospective* filter — NeurIPS/ICML/ICLR/ACL accepted lists a year later tell you what survived, which is a better use of reading time than the preprint firehose.

The triage rule, three tiers. **Tier 1 — read the abstract and the results table (2 min).** Everything that survives a title skim. **Tier 2 — 20-minute structured read** using the cold-read order, for anything that would change a system I own. Maybe two per week. **Tier 3 — reproduce or implement.** Maybe one per quarter, and this is where actual retention comes from.

**The paper journal** is the piece that converts reading into interview performance, and it is why this question is in the guide. One file, append-only, one entry per Tier-2 paper, five fields: the five-beat summary in my own words, *what I would change in a system I own if I believed it*, my confidence that it replicates, a link, and the date. Two consequences: writing "what I would change" forces the applied translation that interviewers probe for, and a year later you can grep the file for the three papers that actually changed your mind. Before any loop I re-read the last six months of entries — twenty minutes, and it is the highest-leverage prep I do for the research round.

**⚠ Trap:** confusing consumption with currency. Being able to name the last month's releases is not currency; currency is being able to say what you changed because of one. Interviewers distinguish these instantly by asking "and did you do anything with it?"

**📐 Numbers you must know:** the AI-relevant arXiv categories publish on the order of hundreds of new submissions per weekday. At 2 minutes per abstract, reading 5% of them is 10+ hours a week. The arithmetic is why a triage protocol is mandatory rather than optional. **📅 Volatile:** submission volumes climb every year; recompute rather than quote.

### "What recent paper excited you and why?" Go.

This is the most predictable question in the round and the most commonly botched, so treat it as a rehearsed 90-second artifact with a fixed structure: **what it is, the mechanism in one sentence, why it changed something for me specifically, the limitation I would name unprompted, and the follow-up I am inviting.** The limitation is not modesty — it is the proof that you read it, and it steers the follow-up onto ground you have prepared.

Here is a fully-worked example on DeepSeek-R1, spoken at real speed:

"The one that actually changed how I think is the R1 line, and specifically R1-Zero rather than R1. The mechanism is that they ran GRPO directly on a base model with a purely rule-based reward — an exact-match check on the final answer plus a format reward — with no supervised warm start at all, and long chain-of-thought behavior with self-verification and backtracking emerged on its own, along with response length growing over training. What changed for me is that it reframed reasoning as something you can *grow with a verifier* rather than something you have to demonstrate with expensive human-written traces, which pushed me to look at every failure mode in my own systems and ask 'is there a cheap automatic checker here?' — for the code-generation surface I work on there is, because tests are the reward. The limitation I'd name is that I don't think the paper settles whether RL is expanding the capability frontier or mostly collapsing sampling onto a good tail the base model already had; the pass@k evidence in that debate points at the second reading, and it matters commercially, because if it's the second one then distillation and better sampling get you most of the way for a fraction of the cost. The distillation result in the same paper — that small models imitating R1 beat small models trained with RL directly — is at least consistent with that."

That is roughly 100 seconds, it contains a mechanism, a specific personal consequence, a named limitation, an honest statement about a contested question, and an invitation for the interviewer to push on RLVR. It cannot be delivered by someone who read a summary.

**⚠ Trap — the fatal one:** naming a famous paper you have not actually read. Interviewers at labs default to their own area, and the second follow-up is always mechanical ("what exactly does the format reward check?", "what does GRPO use as the baseline?"). One vague answer there does more damage than saying "I haven't read it" would have, because it retroactively discounts every confident thing you said earlier in the loop. Pick papers by what you can defend at depth 3, never by prestige.

**🏋 Drill:** prepare exactly three of these, from three different areas (one architecture/serving, one post-training, one retrieval/agents), and have a colleague ask you three escalating follow-ups on each without warning. Pass criterion: you reach the third follow-up on all three without a filler answer, and each one contains at least one number from the paper's experiments.

### An interviewer asks about a paper you haven't read. What do you actually say?

Say you have not read it, immediately, in one clause — and then buy the exchange back with the two things you *do* have: adjacent knowledge and a good question. The full move is roughly: acknowledge, place it, ask, offer.

"I haven't read that one." (Acknowledge — no hedging verbs, no "I've seen it go by," which every interviewer decodes as no.) "I know it roughly as being in the same space as X, which I have read — is the core idea closer to A or to B?" (Place it, using real adjacent knowledge, and ask a question whose answer you can genuinely use.) Then, when they tell you, engage properly: "okay — then the thing I'd want to know is whether the gain holds at scale, because for X the same effect shrank above 13B." (Offer — you have now had a technical conversation about a paper you have not read, which is exactly the skill the round is for.)

Why this works better than bluffing, mechanically: the interviewer's model of you updates on *calibration*, not coverage. Nobody has read everything, and an interviewer who has published knows the field's read-rate better than you do. A candidate who accurately reports their own uncertainty is a candidate whose confident statements can be trusted — and the confident statements are what they will write in the debrief. A candidate who bluffs once has poisoned every other claim they made, because the interviewer now has to discount all of them.

The one thing not to do is deflect into a different paper you *did* read without acknowledging the swap. It is transparent and it reads as evasive rather than knowledgeable.

**🗣 Say this in the room:** "I haven't read it — I know the area but not that result. What's the mechanism? ... Got it. Then the question I'd have is whether that survives at serving batch sizes, because the analogous thing in [X] only helped at batch 1."

**⚠ Trap:** over-correcting into performative humility. Saying "I'm not really an expert in anything ML" to seem calibrated is a different failure and it costs you the level. Calibration means being precise about the boundary — "I can go deep on serving and retrieval; on pretraining data curation I can follow a conversation but I wouldn't claim judgment" is the right shape.

### You find what you believe is a model bug. How do you file it so a research team actually acts on it?

Mental model: a model team's queue is not sorted by how annoyed the reporter is; it is sorted by **how cheap the issue is to reproduce and how legible its impact is**. Your job as the filer is to do the expensive half of the work — the isolation and the measurement — so that a researcher's cost to act is fifteen minutes rather than two days. Most "the model is bad at X" reports die because they force the receiving team to do that work themselves.

What a report that gets acted on contains:

**1. A minimal reproduction that is actually minimal.** Not your production prompt. Strip the system prompt, strip the tools, strip retrieval, and find the smallest input that still triggers it. If it only reproduces with your 12k-token system prompt, say so explicitly — that is itself the finding, and it points at a long-context or instruction-conflict issue rather than a capability gap.

**2. A rate, not an anecdote.** "Fails 34/100 at temperature 0.7, 41/100 at temperature 0, n=100 each" is a bug report. "It sometimes does this" is a complaint. Include the exact decoding parameters, because half of reported model bugs are sampling settings.

**3. Version and boundary information.** Exact model version string, when it started, and whether it reproduces on the previous version. A regression is a categorically higher-priority object than a longstanding weakness, and you are the only person positioned to establish which one it is. Include request IDs if the provider supports them — that lets them find the actual inference.

**4. Expected versus actual, with the expectation justified.** "Expected: it does not invent a citation. Actual: invents a plausible one 34% of the time" is clean. Avoid expectations that encode a preference rather than a defect.

**5. Impact, in the receiving team's units.** Not "this is blocking us." Rather: "this affects 12% of our document-QA traffic, which is 40k requests/day, and it is the top complaint category from our two largest accounts." Give them the sentence they will paste into their own prioritization doc.

**6. Your isolation work, stated.** "We ruled out retrieval by feeding oracle passages — it still fails. We ruled out our parser by inspecting raw completions." This is what converts you from a reporter into a collaborator.

**7. An eval case, attached.** The single highest-leverage element: hand them 20–50 labelled examples in a runnable format. A bug with an eval attached can be *verified fixed*, which means it can enter a training or eval loop; a bug without one cannot, and will be closed.

**⚠ Trap:** filing a harness bug as a model bug. Do the elimination first (see the model-vs-harness question below). Filing two harness bugs as model bugs costs you the ability to get the third, real one prioritized — reputation with a model team is a rate-limited resource and you spend it every time you file.

**🗣 Say this in the room:** "I treat a model bug report like an incident report for a system I don't own: minimal repro, failure rate with n, version boundary, impact in their units, and a small labelled eval attached so the fix is verifiable. If I can't produce those, I don't have a bug yet — I have a hypothesis."

### What makes an eval that a research team will actually trust and use?

Mental model: a research team's scarcest resource is a *signal they can optimize against without being deceived*. Most evals handed to them by product teams are unusable not because they are wrong but because they are unfalsifiable or unstable — the score moves and nobody can say why. Trust comes from properties, and you should be able to list them.

**Provenance you can state per item.** Where did each case come from — real traffic (with a date range and a sampling rule), synthetic generation (with the generator model named), or hand-authored? Mixed-provenance sets with no per-item label are the most common reason a set gets quietly abandoned, because when something looks anomalous nobody can audit it.

**Labels with measured agreement.** If humans labelled it, report inter-annotator agreement on a sample. If it was model-labelled, report the judge's agreement against a human-labelled subset, and report it *per slice*, because judges are usually fine on easy cases and worst exactly where the decision matters. An eval whose ground truth has never been checked against a second labeller is an opinion with a number on it.

**Statistical power, stated up front.** Say what effect size the set can resolve. "n = 1,200, paired design, resolves a 2.5-point difference at 80% power" tells a researcher immediately whether it can adjudicate the change they care about. Most product evals are 100–200 items and can resolve nothing under 8 points, and nobody has ever computed it.

**Contamination hygiene.** Was it constructed from public data the model may have trained on? Do you hold a private split that has never been sent to a provider whose retention terms allow training? This matters more each year, and a team will ask.

**Slices, not one number.** A single aggregate is not actionable. The set should carry per-item metadata (intent, customer segment, document type, language, length bucket) so a regression can be localized to a slice within minutes.

**Versioning and an owner.** The set has a version, a changelog, a named owner, and an immutable locked holdout. When a score changes, the first question is always "did the model change or did the eval change," and only versioning answers it.

**A baseline row and a null row.** Include the score of the current production system and the score of a trivial baseline (retrieve-nothing, or always-answer-the-majority-class). A set where the null baseline scores 68% is telling you something important about itself.

**💰 Math on what a good golden set costs, so you can defend the ask:** 500 items × 12 minutes of expert labelling = 100 hours; at a loaded $80/hour that is $8,000, plus a 20% double-labelled subset for agreement at ~$1,600 — call it $10k and three weeks. Against a team of six engineers at ~$250k loaded each shipping into the dark, $10k is under 0.7% of annual team cost. That is the arithmetic I use to get a golden set funded, and it works.

**⚠ Trap:** building the eval from the same model's outputs you are trying to evaluate or train. Every self-generated set inherits the generator's blind spots, and you will measure a capability improvement that is really an increase in self-similarity. If you must generate synthetically, have humans filter, and keep a human-authored control slice to detect the drift.

### How would you ask a model team for a new capability?

The mental model: a capability request is a prioritization argument, and prioritization arguments are won with **evidence of demand, a cheap definition of done, and an honest statement of your fallback**. A request that lacks any of the three is read as a wish. Also — and this is the part applied engineers get wrong — the model team's cost structure is not yours. A change that seems small to you (better JSON adherence) may require a post-training data campaign and a full eval sweep; a change that seems huge (a new tool-calling behavior) may be a decoding-time constraint someone can ship in a week. So you ask in terms of the *behavior you need*, and you let them choose the layer.

The structure I use:

**Failure distribution first, with volume.** "Here is a taxonomy of our 2,400 failed sessions last month, clustered: 41% is the model calling a tool with a hallucinated enum value, 22% is premature termination, 18% is citation fabrication." Clustering embeddings of failure traces is the cheap way to produce this and it is far more persuasive than examples.

**The one behavior, stated as a testable predicate.** Not "better tool use" — "when a tool schema contains an enum, the model should never emit a value outside it; we measure violation rate on our 800-case set." A predicate a machine can check is a predicate a training team can optimize.

**The eval, attached and runnable.** Same point as the bug report, and it is the difference between a request and a project. If they cannot measure it, they cannot ship it, and the person who brings the measurement usually gets to define the target.

**The business number.** Requests per day, revenue exposure, the named accounts. Model teams are prioritizing across many askers; you are competing on legibility of impact.

**Your fallback, honestly.** "If this isn't on the roadmap, our plan is constrained decoding with a grammar over the enum, which we estimate costs 30ms per call and doesn't generalize to the semantic-validity cases." Two effects: it proves you are not outsourcing your own engineering, and it tells them exactly what capability their answer is worth. I have had requests accepted mainly because the fallback made the residual gap obvious.

**⚠ Trap:** asking for a model change to fix something a schema change would fix. If your tool has fourteen parameters and three of them are mutually exclusive, the model is not the problem — your interface is. Model teams notice this immediately and it costs credibility. Redesign the tool, remeasure, *then* ask.

**🗣 Say this in the room:** "I bring three things: the failure taxonomy with volumes, a runnable eval that defines done, and what we'll build ourselves if the answer is no. If I can't articulate the fallback I don't understand the request well enough to make it."

### A customer complains that the assistant gives wrong answers on their contract questions. Translate that into something a research or post-training team can act on.

Mental model: a customer complaint is at the top of a translation stack, and each layer down narrows the space of possible fixes. Your value as an applied engineer at a lab is doing that translation faithfully — most of the information is lost if you hand a research team "customers say it's wrong."

The stack, layer by layer, with the question each answers.

**Layer 1 — complaint to trace set.** Get the actual sessions. Not summaries. 50–100 real traces with inputs, retrieved context, tool calls and outputs. Anything you conclude before you have traces is speculation.

**Layer 2 — trace set to failure taxonomy.** Cluster and label. On contract QA the clusters are usually: (a) retrieval missed the governing clause; (b) retrieval found it but the model relied on a superseded version; (c) the model read the clause correctly and reasoned wrongly about precedence or conditionals; (d) the model was right and the customer's expectation was wrong; (e) formatting/citation failure that made a correct answer look wrong. These have completely different owners and you must not skip this step.

**Layer 3 — taxonomy to locus.** (a) and (b) are *your* problem — chunking that splits clauses, missing document-version metadata, no recency or effective-date filter. Fixing those is a retrieval engineering task and you should never escalate them. (e) is a prompt/format problem. Only (c) is a candidate model or post-training issue, and even then you must first check whether it survives with **oracle context** — hand the model exactly the right clauses and see if it still gets precedence wrong. If it does, you have a genuine reasoning gap.

**Layer 4 — locus to ask.** Now, and only now, you have something a post-training team can use, and it comes in one of three shapes. **A data ask**: "we have 400 cases of nested-conditional contract reasoning with expert-labelled correct answers; this is a domain gap and here is a candidate SFT set." **A capability ask with an eval**: "here is a 300-case benchmark of clause-precedence reasoning; the current model is at 61%, GPT-class competitors at 68%, and this blocks a named vertical." **A behavior ask**: "the model should abstain when the retrieved set contains two conflicting effective dates rather than picking one" — an abstention/calibration ask, which is often the most valuable and least-requested category.

**💰 Math on why abstention is usually the right ask:** if 8% of answers are wrong and the customer's cost of a wrong contract answer is a review cycle worth roughly $400 of lawyer time, at 5,000 queries/month that is 400 × 0.08 × 5000 = $160,000/month of downstream cost. Converting half those errors into "I'm not confident, here are the two conflicting clauses" costs the customer a few minutes of reading and removes ~$80,000/month of rework. Accuracy is not the metric; **cost of the error mode** is, and framing the ask that way is what makes it fundable.

**⚠ Trap:** the reflex fine-tune. "The customer's domain is different, let's fine-tune on their contracts" is the most common wrong answer in this entire field. Domain-specific *knowledge* is a retrieval problem; fine-tuning changes style, format and task adherence far more reliably than it installs facts, and a fine-tune also freezes you off the frontier model upgrade path. Escalate through prompt/context → retrieval → tool design → structured output → routing → distillation, and put fine-tuning last with a stated precondition.

### You think it's a harness problem. The researcher thinks it's a model problem. How do you settle it?

Not by arguing — by naming the discriminating experiment, which is the only move that ends this disagreement and is exactly what the round is testing. The mental model: "model problem" and "harness problem" are hypotheses that make *different predictions*, and your job is to find the cheapest observation on which they differ.

Here is the ladder I run, cheapest first. Each rung isolates one variable and most disputes die by rung three.

**1. Oracle context.** Replace whatever your retrieval/tool layer produced with the known-correct information, hand-assembled. If the failure disappears, it is a retrieval or context-assembly problem and the model is exonerated. This single test resolves the majority of RAG disputes and it costs an afternoon.

**2. Oracle trajectory.** For agent failures, hand-write the ideal sequence of tool calls and force the model to execute it. If the outcome is still wrong, the tools or the environment are wrong. If it is right, the failure is in *choosing* the trajectory, which narrows you to planning.

**3. pass@k.** Run the same task k times at temperature > 0 and measure whether *any* sample succeeds. If pass@8 is high and pass@1 is low, the model can do the task and your harness is failing to reliably elicit it — that is a sampling, prompting, or selection problem, and the fixes are on your side (better prompt, self-consistency, a verifier, constrained decoding). If pass@32 is still zero, it is a genuine capability gap and the researcher is right. This is the single most decisive test in the ladder and I would lead with it if the task is cheap to run.

**4. Model swap, harness fixed.** Run the identical harness against a different model family. If every model fails identically, suspect the harness. If failure rate varies wide across models, it is capability-correlated.

**5. Harness swap, model fixed.** Run the same model through a minimal, dependency-free harness — no framework, raw API, hand-built loop. Framework bugs (a truncated observation, a dropped tool result, a system prompt injected in the wrong place, an incorrect chat template) are *extremely* common and this test finds them. I have resolved more of these disputes at rung five than anywhere else.

**6. Prompt/format ablation.** Vary only the tool schema, or only the output format, or only the position of the instruction. Large sensitivity to these means the model's behavior is fragile — which is a legitimate model finding, and a much better-specified one than "it's bad at this."

Then present the result, not the argument. **🗣 Say this in the room:** "I don't think we should debate it — pass@32 is the cheap discriminator. If any sample succeeds, it's my harness and I'll own it. If none do across 32 samples and two model families, it's a capability gap and I'll write it up with the eval attached." That sentence has resolved this disagreement for me more times than any amount of evidence, because it makes the disagreement falsifiable and it visibly puts your own work on the line first.

**⚠ Trap:** conceding too fast. The social gradient in a room with a researcher pushes an applied engineer toward "you're probably right, it's our harness." Being wrong about this is expensive in both directions: an unfiled model bug never gets fixed, and a misfiled one burns credibility. Run the ladder.

### Which public artifacts actually convert into interviews, and how would you rank them?

The mental model to open with: hiring managers are trying to estimate a posterior over your ability from a very noisy sample, and the artifacts that convert are the ones with the **highest signal-to-effort ratio for the reader** — things that can be verified in five minutes and that a large fraction of applicants cannot produce. That is the ranking criterion, not how impressive the thing feels to build.

**1. A merged PR to a serious infrastructure project** (vLLM, SGLang, TRL, LangGraph, Outlines, transformers). Highest signal by a distance, because it is externally verified — a maintainer with no incentive to be nice reviewed your code and merged it. A kernel or scheduler fix outranks a docs fix, but a merged docs PR outranks an unmerged kernel branch. The path that works: run the project in anger, find something broken, fix that. Issues tagged good-first-issue are crowded; a bug you hit yourself has no competition and you already have the repro.

**2. A public eval harness for a task nobody has benchmarked.** Underrated and, for the archetypes you are targeting, possibly the highest-converting artifact per hour of work. It demonstrates the exact skill every one of these companies says is scarce — evaluation methodology — and it is genuinely rare. Pick a task from a domain you know, build 300–500 labelled cases with documented provenance, run five models, publish the table with confidence intervals and a contamination note. That artifact makes you the person who *has the numbers* for that task.

**3. A from-scratch reimplementation with benchmarks.** Not "I wrote a transformer" — everyone has. It converts when it carries measurement: "GQA vs MHA KV-cache footprint and decode throughput at four context lengths on one A100, with the code and the raw numbers." The benchmark is the artifact; the implementation is the setup.

**4. A written post-mortem of a real production AI incident.** Rare because most people cannot publish internal detail, which is exactly why an anonymized one converts. Detection, contributing factors, why tests did not catch it, and the eval case added so it cannot recur. Reads as senior in a way nothing else on this list does.

**5. A reproduction report.** Covered above; scarce, cheap, and demonstrates honesty under an unflattering result.

Below the line, and I would say this bluntly: a chatbot demo, a LangChain tutorial re-implementation, and a "I built an agent that browses the web" repo do not convert at this level, because thousands exist and none of them carry a measurement. The distinguishing feature of everything above the line is that **it contains numbers somebody else can check.**

**💰 Math on effort-to-conversion:** a good eval harness is roughly 40–60 hours (labelling dominates). Against a 10-week prep plan of, say, 250 hours, that is ~20% of your budget for an artifact that is quotable in every single behavioral, design and research round — you will reference it in the eval question, the design question, the "tell me something you built" question and the paper round. Nothing else in the plan has that reuse factor, which is why I would fund it before another 40 hours of LeetCode.

**⚠ Trap:** building the artifact and not writing the README. The README is the artifact for 90% of readers — they will never run your code. Lead with the result table, then the method, then reproduction instructions. A great benchmark with a three-line README converts worse than a mediocre one with a proper write-up, which is annoying and true.

### Give me a four-week plan to go from "I've read summaries" to passing a paper round, and tell me how I'd know it worked.

Mental model for the plan: research literacy has three separable skills — **recall** (the canon, at 90 seconds each), **cold-read critique** (a procedure, not knowledge), and **applied translation** (turning a result into a system change). They need different practice and most people only practice the first, which is why they fail on the second follow-up. Budget accordingly and make the pass criteria falsifiable.

**Week 1 — the canon, breadth pass.** All 20 canon papers, 30–40 minutes each, using the cold-read order. Write a paper-journal entry for every one with the five beats plus "what I'd change in a system I own." Do not aim for depth; aim for a correct map. *Pass criterion:* you can deliver a 90-second five-beat summary for any of the 20 when someone names it at random, with at least one experimental number in each.

**Week 2 — depth on five, adversarially.** Pick five, weighted toward your target archetype (for AI-product: RAG/Lewis, ColBERT, ReAct, PagedAttention, DPO; for infra: FlashAttention, PagedAttention, speculative decoding, MoE, YaRN). Read them properly, including the experimental section and the appendix hyperparameters. For each, write down three criticisms and the experiment you would run next. *Pass criterion:* a colleague can ask you three escalating follow-ups on each of the five and you never give a filler answer.

**Week 3 — cold-read drills.** Three per week, on papers you deliberately have not seen, from areas adjacent to but not inside your comfort zone. 20-minute timer, then a 10-minute spoken critique recorded. Grade yourself against the five-axis framework: claim-vs-evidence, baselines, ablations, threats to validity, what next. *Pass criterion:* on two consecutive cold reads you identify at least one genuine methodological weakness — a missing baseline, an unablatable confound, a scale limitation — that you can defend, without having read anything about the paper.

**Week 4 — reproduce one thing, and rehearse the human half.** Pick the smallest real result you can run for under $50 and produce the full eight-section reproduction report. In parallel, rehearse the three "recent paper that excited you" answers and run one mock of the model-vs-harness disagreement with someone playing the researcher. *Pass criterion:* the report is published and runnable in one command by a stranger, and your three rehearsed answers each survive three follow-ups on the clock.

**🏋 Drill — the dress rehearsal, do this once at the end.** 60 minutes with a colleague. Ten minutes: they name five canon papers at random, you give 90-second summaries. Twenty minutes: cold read of a paper they chose without telling you, then critique. Ten minutes: "what recent paper excited you," with three follow-ups. Ten minutes: they play a researcher who insists your production failure is a harness bug; you run the discriminating-experiment ladder out loud. Ten minutes: feedback. *Pass criterion:* you got through all four blocks without a single "I think I read that one," and in the last block you proposed pass@k or oracle-context before they asked for a test.

**⚠ Trap:** spending all four weeks in week 1. Recall is the comfortable part and the least-scored part; the canon exists so that you have shared vocabulary, not so you can recite it. If you have to cut, cut week 1 to ten papers and keep the cold reads and the reproduction — those are what actually produce a hire signal.

**🗣 Say this in the room, when asked how you keep up:** "Weekly triage, a paper journal where every entry has to end with what I'd change in a system I own, and one reproduction a quarter. The journal is the part that made the difference — it turned reading into things I'd actually argue for."
