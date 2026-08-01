### What are the named critical errors that get submissions rejected? Rank them.

There are five that recur across debrief notes with enough regularity to treat as a checklist.

**🔍 Failure taxonomy — read this as a decision procedure you run on your own submission before you send it.** Ask the five questions in order and stop at the first "no": *Is there a number anywhere in this repo that came from measuring my own output?* (no → error 1). *Would a grader who timed my commits believe I gave this real effort, or have I declared my time box?* (no → error 2). *For every parameter I set, can I point to a written line saying why?* (no → error 3). *Can I find any file in this repo in ten seconds and justify five random lines?* (no → error 4). *Does every abstraction here make a change I actually made cheaper?* (no → error 5). Ranked by how often each is decisive:

**1. Skipping evaluation of AI outputs.** The system works in the demo, and there is no evidence anywhere that it works in general. This is decisive on its own at AI-product companies, because the entire job is shipping under nondeterminism and a candidate who doesn't instrument their own output has demonstrated they'd ship blind.

**2. Insufficient effort — the visibly-rushed submission.** Two hours on a four-hour brief, no tests, README that's the framework's default, TODO comments in the merged code. It reads as "I don't want this job much," and the grader has a stack of submissions where someone clearly did. Note the asymmetry: a *time-boxed* submission with the box declared reads as discipline; the identical artifact with no note reads as effort.

**3. No documented reasoning.** Choices exist but no rationale does. Chunk size 500 because 500. This one is fatal in a subtler way — it doesn't fail the code review, it fails the defense, because you have to invent the reasoning live and it shows.

**4. An unprepared defense.** Cannot navigate own repo, cannot recall why a parameter is what it is, contradicts the README. This raises the authorship question, and once that question is live in a grader's mind, every strength becomes ambiguous.

**5. Unjustified over-engineering.** Abstractions and infrastructure with no decision behind them, usually accompanied by a missing eval — because the hours came from somewhere.

**⚠ Trap:** believing these are independent. They're correlated through a single root cause, which is **starting to code before deciding what "good" means**. If you begin with the eval, you get numbers (fixes #1), the numbers force decisions (fixes #3), the decisions constrain scope (fixes #5), and the defense becomes a recital of things you actually did (fixes #4). One habit removes four of the five failure modes.

**🗣 Say this in the room:** "The first thing I built was the way to tell whether it was working, because otherwise every later decision would have been a preference rather than a finding."

### How do graders detect "insufficient effort"? I want to know what the tells are.

Because they're specific and they're cheap to eliminate, here's the actual tell list from the grader's side:

- **A README that is the framework's scaffold text**, or that is generic enough to have been written without reading the assignment. Instant signal.
- **Zero tests, or one test that asserts `True`.** Not "few tests" — zero. Three real tests reads completely differently from none.
- **`TODO`, `FIXME`, and commented-out blocks left in the submitted branch.** Not because the code is worse but because it signals you never re-read the diff before submitting. Five minutes of `git diff` prevents this entirely.
- **Default parameters everywhere with no note.** `chunk_size=1000, k=5, temperature=0.7` — every one a library default — plus a README that doesn't mention any of them. It reads as "I copied a quickstart."
- **The clean-clone failure.** Missing dependency, hardcoded absolute path, an import of a file that isn't committed. This is the single most common hard failure and it is 100% preventable by the twenty-minute fresh-clone drill.
- **A commit history of one commit, timestamped four hours before the deadline** on a five-day window.

None of these are about talent. They're about whether you gave the artifact a final pass. **📐 A useful budget rule: reserve the last 20% of your time box for finishing rather than building.** On an 8-hour real budget that's 90 minutes for README, fresh-clone test, diff review, and the results table — and it will improve your score more than 90 minutes of additional features, every single time.

**⚠ Trap:** the opposite error, which is real — spending 14 hours on a stated-4-hour assignment. Graders can often tell, from commit timestamps and scope, and at some companies it counts against you as a calibration failure ("cannot scope"). It also poisons your own pipeline, because you can't sustain it across three companies. Deliver at roughly 1.5–2× the stated estimate with the overrun declared, not at 4×.

### Say I get a question in the defense about a choice I genuinely didn't think about. What do I do?

Say so, immediately and without embarrassment, and then convert it into an experiment. There is a script for this and it works because it's honest:

> "I didn't evaluate that — I took the library default. It mattered less than X, which is where I spent the time. If I were to check it, I'd sweep chunk size across 256/512/1024 on my 40-question set and look at recall@5; I'd expect the effect to be largest on the multi-hop questions, and I'd take the change only if it moved recall outside the confidence interval on N=40, which for a difference this size it probably wouldn't at that sample size."

That answer scores *better* than a fabricated justification, and it's not close. It demonstrates four things at once: honesty under pressure, an accurate model of where your time went, the ability to design an experiment on the spot, and statistical literacy about your own sample size. Interviewers are looking for the failure mode where a candidate bluffs, because bluffing on a team is expensive — someone acts on your invented confidence.

The three-answer taxonomy from earlier is the thing to internalize because it covers every "why did you..." question that exists: **I measured it (here's the number) / I didn't measure it (here's why it was low-priority and here's the experiment) / that's a bug (here's the fix).** Rehearse all three phrasings out loud. Under pressure people default to a fourth option — vague retroactive justification — and it's the only one that fails.

**⚠ Trap:** over-correcting into self-deprecation. "Yeah, that's probably wrong, I didn't really know what I was doing there" is not honesty, it's an invitation to discount your whole submission. Honesty is *specific* and *bounded*: name what you didn't do, name why, name what you'd do. Never generalize a gap into a statement about your competence.

**🗣 Say this in the room** (when you genuinely disagree with the interviewer's implied better answer): "I'd push back on that slightly — I considered a reranker and rejected it for this corpus because my recall@5 was already 0.94, so the ceiling on reranking was six points of recall and it would have added 150 ms to p95. If recall had been 0.7 I'd agree with you." Disagreeing with evidence is a strong senior signal; agreeing reflexively with your interviewer is a weak one.

### I got a take-home before I'd spoken to a single human at the company. Do I do it?

Usually not, and there's a cleaner move than either doing it or declining.

The mental model: **an assignment issued before any human conversation is a company outsourcing its screening cost to candidates at scale.** They are spending zero of their time and asking for four to eight hours of yours, from an unknown number of applicants, with no signal that you're a real candidate rather than a résumé that matched a keyword. The expected value is bad and the process signal is worse — it tells you something true about how that org will treat your time after you join.

The move I'd make, in one polite email, is to **trade a conversation for the assignment**:

> "Happy to do the exercise. Before I invest the time, could we do a 20–30 minute call with the hiring manager or a team member? I'd like to understand the problems the team is working on so I can scope the exercise toward what you actually care about — and honestly, I want to confirm the role's a fit before we both spend time on it."

This is not a demand and it isn't adversarial; it's a proposal that's better for both parties, and it's framed as improving the assignment's usefulness. In my experience roughly half of companies agree immediately, some agree after a recruiter check, and the ones that refuse outright have told you something worth knowing. **The candidates who lose the process by asking this are, in my observation, the ones who ask it defensively** ("I don't do unpaid work") rather than collaboratively.

There's a second variant worth knowing: some companies issue the take-home first *because their process is genuinely blind-first by design*, as an anti-bias measure, and they'll say so. That's a defensible policy and the calculus changes — but they should still be able to tell you the role, the team, and the scope in writing.

**⚠ Trap:** deciding this purely on principle and losing companies you actually wanted. Weight it by how much you want the job and how much the assignment costs. A 2-hour exercise from a company that's top of your list, issued before a call, is worth just doing. A 3-day full-stack build from a company you're lukewarm on is worth declining even if they're prestigious, because the opportunity cost is a full pass through the material you'll be tested on at the company you *do* want.

### What about an assignment whose scope is genuinely exploitative? How do I push back without being screened out?

First, calibrate what "exploitative" means, because senior candidates both under- and over-detect it. The markers:

- **The deliverable is a deployable product, not a sample of work.** "Build a production-ready multi-tenant RAG service with auth, a billing hook, an admin UI, and a deployment pipeline" is not an exercise; it's a sprint.
- **The stated estimate is off by 3× or more from any honest reading.** A brief that says "should take about 5 hours" and enumerates fourteen requirements including a frontend, tests, deployment, and an eval framework is a 30–40 hour assignment. Community estimates on the most extreme public examples have run to roughly **€6,000–10,000 of professional work** at normal contract rates — and the arithmetic behind that is not exotic: 40 hours × €150–250/hour is €6,000–10,000.
- **The problem is suspiciously specific to their live roadmap** — their actual data schema, their actual customer's use case, a connector for the exact third-party API their product needs next.
- **No compensation is offered and no defense conversation is scheduled**, which means the artifact has value to them independent of you.

Note the contrast with the healthy end of the market: a **paid work trial** — a bounded, compensated, scheduled block, which some companies run as an onsite day — is the *opposite* signal. Paying for the time means the company has internalized the cost, which means they'll keep it bounded.

The renegotiation script, which works more often than candidates expect:

> "I've read through it and my honest estimate is 25–30 hours to do all of it well, which I can't commit to alongside my current role. Two options: I can do the retrieval and evaluation portions — the parts I think are most diagnostic — as a scoped 5-hour subset, and we can discuss the rest in the review. Or if the full scope is necessary, is there a paid trial structure? Either works for me."

Three things make this land: you gave a number (which demonstrates estimation, itself a senior skill), you offered a concrete alternative rather than just a refusal, and you left the decision with them. I have never seen this framing lose a company that was worth working for, and I have repeatedly seen it produce a reduced scope.

**⚠ Trap:** doing 30 hours of work resentfully and letting it show in the submission. The half-hearted 30-hour submission is the worst outcome available — you paid the full cost and got the "insufficient effort" grade anyway. Decide to do it properly or decide to renegotiate. Do not decide to do it badly.

**🗣 Say this in the room:** "My honest estimate for the full scope is about 28 hours. I'd rather deliver a scoped subset excellently than all of it thinly — here's the subset I think is most diagnostic and why."

### Can I read anything real about a company from how they run their process?

Yes, and I'd argue it's the most reliable due-diligence channel you have, because unlike Glassdoor and unlike the interviews themselves, **the process is not curated for you — it's the org's actual operating behavior leaking out.** You're watching how they handle a cross-functional workflow with a deadline and an external counterparty. That generalizes.

The signals I read, and what each one predicts:

**Positive:**
- **A written brief with an explicit rubric.** Predicts a culture where expectations get written down, which predicts fewer surprises in performance review.
- **A stated time box that they defend** — "please don't spend more than four hours, we mean it." Predicts respect for boundaries and realistic planning.
- **A defense conversation is scheduled before you submit.** Predicts they intend to actually read your work, and that it isn't a filter to discard you cheaply.
- **Same-day or next-day scheduling responses, and a named point of contact.** Predicts operational competence generally.
- **They answered your clarifying questions with substance**, including "we don't know, use your judgment, tell us what you assumed." That last answer is a *great* sign — it means they can tolerate ambiguity being surfaced rather than hidden.
- **Compensation range disclosed unprompted.**

**Negative:**
- **Silence measured in weeks, and you're the one chasing.** The strongest single negative. If a company that wants to hire you can't reply in three business days during the courtship phase, model what happens when you need a decision from another team in month four.
- **Scope creep after you started** — "oh, could you also add a UI." Predicts requirement instability, which is the number-one driver of engineering misery.
- **The interviewer hasn't read your submission** and asks you to explain what it does. Predicts that senior time is not protected and that internal work goes unreviewed.
- **Round count inflation** — a sixth and seventh round appearing after you were told five. Predicts weak internal decision-making and no clear owner.
- **Nobody can tell you what the team's roadmap is.** Predicts you'll be reorganized.
- **They rescheduled twice and neither time was explained.**

**⚠ Trap:** discounting these signals because the compensation number is large. The process signal is a leading indicator of your day-to-day experience, and the compensation number is not a hedge against a badly-run org — at this level, a mis-fit that ends in eleven months costs you more in career trajectory than the delta between two offers.

**🗣 Say this in the room** (turning it into a question for them, which reads as senior rather than suspicious): "How does this exercise map to what the team is actually working on this quarter? And who'll be reviewing it — will I be talking to them?" The quality and specificity of the answer is your data.

### What are realistic end-to-end timelines, and how do I make offers land in the same window?

The timelines cluster into three regimes, and the whole game of negotiation is putting offers from different regimes into the same two-week window.

**Fast regime — roughly 10–20 days end to end.** AI-native startups and scale-ups with a small hiring committee and a founder in the loop. Some run a median close to **11 days**; some frontier labs report a median near **19 days**. Characterized by: recruiter replies within a day, batched onsite, decision within 48 hours of the final round.

**Medium regime — roughly 3–5 weeks.** Mid-size AI companies and well-run enterprise AI teams. A take-home with a 5-day window plus scheduling friction plus a debrief cycle.

**Slow regime — 6–10 weeks, occasionally longer.** Big tech. The interviews are not the slow part; **the queues are** — recruiter-to-manager handoff, scheduling five interviewers across time zones, hiring committee convening weekly, then leveling review, then compensation approval, then an offer letter. Amazon-style loops add a bar-raiser scheduling constraint; Google-style loops add a hiring committee that meets on a fixed cadence and can request more information, which restarts a week. Some companies additionally require **manager references before an offer is extended**, which adds days and requires you to have warned your references in advance.

**📐 The scheduling arithmetic you must do:** if you want a big-tech offer (8 weeks) and a startup offer (2 weeks) to land in the same week, **you start the big-tech process six weeks before you start the startup process.** Not the same day. This one calculation is worth more leverage than any negotiation tactic, because leverage in an offer conversation is entirely a function of holding a competing deadline, and you cannot manufacture that after the fact.

Practical sequencing for a 12-week campaign: weeks 1–2 apply to the slow regime; weeks 5–6 apply to the medium regime; weeks 8–9 apply to the fast regime; weeks 10–12 everything converges. Front-load two low-stakes practice loops in weeks 3–4 so your first real onsite isn't your first onsite of the year.

**⚠ Trap:** letting a fast company's exploding offer set the clock. If a startup gives you 72 hours and your top choice is in week 5 of 8, the honest move is to tell the startup exactly that — "I have processes in flight that conclude in three weeks; I'd like to make a decision with complete information, and I'm genuinely interested." Good companies extend. Companies that refuse to extend a two-week ask are giving you a process signal, and it's the same signal as the two-week silence: they optimize their convenience over your decision quality.

**🗣 Say this in the room** (to a recruiter, in the first call, always): "What's the typical timeline from here to a decision, and are there any steps like references or committee review that add calendar time?" Asking this in call one lets you build the sequencing plan, and it reads as organized rather than presumptuous.

### How should I actually manage the day of a five-to-seven hour onsite?

Treat it as an endurance event with a known failure curve, because that's what it is. **The scores on rounds four and five are systematically lower than rounds one and two for the same candidate**, and the cause is not knowledge — it's glucose, hydration, and accumulated cognitive load. This is a solvable engineering problem and most candidates don't even model it.

The mechanics that actually matter:

- **Ask for the schedule in advance and the order of rounds.** Then decide where your hardest round falls. If the live build or the system design is round five, that's worth flagging — "would it be possible to put the build round earlier in the day?" is a reasonable ask and is granted more often than people assume.
- **Eat before, and eat *during*.** A 5-hour onsite with one 15-minute break and no food is a self-inflicted 15% performance cut on the last two rounds. Bring something you can eat in four minutes. If it's virtual, this is free; if it's onsite, ask about the lunch break when you get the schedule.
- **Protect the gaps.** Between rounds, do not review notes. Stand up, look at something far away, drink water, breathe. The marginal value of cramming in a 10-minute gap is negative — it raises arousal and degrades the working memory you need for the next round.
- **Reset the frame between rounds explicitly.** Each interviewer arrives with no knowledge of how the last one went. If round three went badly, round four's grader does not know that, and carrying the affect from a bad round into the next one is the actual mechanism by which one bad round becomes three. Say to yourself, literally, "new interviewer, new score." It sounds trivial and it is the highest-leverage thing on this list.
- **Have a stock 60-second self-introduction** so you're not composing one from scratch five times while tired.
- **For virtual onsites: check the tooling the day before** — the collaborative editor, screen share, and whether their environment has your language's tooling. Losing eight minutes of a 45-minute round to a screen-share problem is a real and common way to lose a round.
- **For an IST-based candidate interviewing with US teams,** this compounds badly: a 5-hour onsite starting at 9:30pm IST ends at 2:30am. Push hard for a split across two days, which is normal and usually granted. If it can't be split, shift your sleep schedule by two hours for the three days *before* — not the night of — because a single-night shift is just sleep deprivation with extra steps.

**📐 Numbers you must know:** budget **20–30 minutes of buffer** before the first round for setup, and treat the last 10 minutes of every round as yours for questions — that's roughly 50 minutes of your 5-hour day spent on questions you ask them, so have 3 prepared per interviewer type and don't waste them on things the careers page answers.

### There's a live build round — sometimes a full paid onsite project. How is that different from a take-home?

The live build compresses the whole take-home into a supervised window, and the grading shifts fundamentally: **in a take-home the artifact is graded; in a live build the process is graded, and the artifact is only evidence that the process was real.** Some companies run this as a genuinely long block — a paid, roughly 8-hour onsite project day is now a real format at AI-product companies — and a few run a multi-day paid work trial under NDA.

What's being observed that isn't observable in a take-home:

**How you start.** The strongest candidates spend the first ten minutes not typing: restating the problem, asking two clarifying questions, and stating the plan out loud with an explicit cut line ("I'll get ingestion and retrieval working first, then a minimal eval, then improve retrieval if there's time; if I'm short I'll cut the reranker"). Candidates who open the editor and start typing in minute one are scored down almost universally, and they never find out why.

**How you handle being stuck.** Everyone gets stuck. The graded question is what you do in minute three of being stuck: do you narrate ("the embedding call is returning a 422, I'm going to print the request body rather than guess"), do you bisect the problem, do you check the actual response instead of theorizing? Debugging out loud is the single highest-value observable behavior in this round, and it's a *learnable skill you should practice*, because most engineers debug silently by default.

**Whether you keep the thing running.** Committing frequently and keeping a working state at all times matters more here than in a take-home, because at any moment the interviewer may say "show me where we are." The candidate whose system has been broken for the last 25 minutes while they refactor has made a scheduling error.

**Whether you use the time budget deliberately.** Announce checkpoints: "we're at the halfway mark, I have retrieval working and no eval, so I'm switching to the eval now even though retrieval isn't tuned." That sentence, said out loud at the halfway point, is worth an enormous amount, because it demonstrates the exact scoping judgment the format exists to test.

**⚠ Trap:** treating the interviewer as an examiner rather than a colleague. In most live builds you are allowed to ask them things — "do you have a preference between X and Y here?", "is it fair to stub the auth?" — and candidates who never speak are read as unable to collaborate. It's a pairing session with a scorecard, not a proctored exam. If the format genuinely is a silent exam, they'll tell you.

**🏋 Drill (4 hours, weekly, unaided, timed):** pick a corpus you've never seen, start a timer, and build retrieval + an eval + a README, narrating out loud into a voice recorder the entire time as if someone is watching. Pass criteria: a working query path by the 90-minute mark, an eval producing a table by the 3-hour mark, a README by 3:45, and — listening back — no silent gap longer than 45 seconds. The narration is the part everyone skips and the part that's actually being graded.

### Some of these companies pay for the work trial. Does that change how I approach it?

It changes the calculus in three ways, and the third one is the one people miss.

**First, the scope becomes fair game.** When a company pays for 48 hours under NDA, the asymmetry that makes unpaid take-homes objectionable disappears. They have internalized the cost, which also means they have an incentive to keep it bounded and to actually use the output as a signal. My default is to accept paid trials from companies I'd work for, essentially without haggling.

**Second, the bar is higher, and the standard shifts from "sample of work" to "would we want this person's output."** A paid trial is closer to a contract engagement, and the evaluation is closer to "how did having this person around for two days feel." Communication cadence becomes a graded dimension in a way it isn't for a take-home: sending a short end-of-day note — what you did, what you found, what you're doing next, what you need from them — is expected behavior in a work trial and its absence is noticed.

**Third — the part people miss — the NDA and IP terms are real and you must read them.** A work trial under NDA typically means you cannot publish the artifact, cannot use it in your portfolio, and in some agreements assigns the IP to them. That's usually fine and is the price of the arrangement. What is *not* fine, and what I would flag before signing, is an agreement broad enough to touch work you already do publicly. If you maintain a public repository, a written guide, or an OSS contribution in an adjacent area, a broadly-drafted IP-assignment or non-compete clause in a *trial* agreement is worth a five-minute read and, if it's broad, one email asking them to scope it to the trial work product. Companies grant this routinely because their intent is narrow; the breadth is usually boilerplate. Signing it unread is the mistake.

**⚠ Trap:** treating the paid trial as guaranteed conversion because money changed hands. Conversion rates on work trials are not 100% and the payment is not a signing bonus. Keep your other processes running in parallel; do not stall a competing loop for a trial that hasn't produced an offer.

**💰 Math on whether to accept:** a 48-hour paid trial at a typical trial rate might pay $1,500–3,000 (**📅 Volatile:** rates vary widely; ask). Against that, you're spending two days you could spend on two other companies' loops. The decision rule I'd use: accept if this is a top-three target, or if the trial replaces rather than adds to the loop. Decline or defer if it's a fifth-choice company adding a trial *on top of* five rounds — that's a process signal, and it's the one about not respecting your time.

### The submission's in. Is there anything worth doing between submitting and the defense?

Three things, and they're cheap.

**One: a short submission note, sent with the link.** Four to six sentences — what you built, the headline number from your eval, the one thing you consciously cut and why, and how long you spent. This is the framing device that determines how your README is read, because it's the first thing the grader sees. It also inoculates you against the two most common misreads: that an absent feature was an oversight rather than a decision, and that the artifact took you either far more or far less time than it did.

**Two: 30 minutes of re-reading your own code, on the morning of the defense.** Not the README — the code. Re-read the retrieval path, the prompt template, and every magic number. Write yourself a one-page cheat sheet with the file layout and the numbers from your results table, and have it open. Under call pressure, retrieving "chunk size was 512 with 64 overlap" from memory five days later is not reliable, and the hesitation reads far worse than the fact is worth.

**Three: prepare the three questions you'll ask them.** Not "what's the culture like." Something that only someone who did this exercise would ask: "when you run this in production, what does your retrieval eval look like — do you have a labeled set, and how do you keep it fresh as the corpus changes?" That question does two things simultaneously: it's genuinely useful to you for evaluating the team, and it demonstrates that your thinking continued past the submission deadline.

**⚠ Trap:** continuing to improve the repo after submitting and mentioning it on the call. "I actually added reranking over the weekend" undermines the scoping story you just told and, worse, means the grader read a different artifact than the one you're describing. If you build more, keep it on a branch, don't push to the reviewed branch, and only mention it if asked what you'd do next — at which point "I actually prototyped it, here's what it did to recall" is a strong answer *because the number is real*.
