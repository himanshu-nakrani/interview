### Describe the shape of a senior AI engineering loop as it stands now — how many stages, and what is each one actually testing?

The shape stabilized around five to seven stages, and the useful thing is that the *stages* are more predictable than the *questions*. Almost every loop on your target list is a permutation of these seven:

**1. Recruiter screen (20–30 min).** Tests: is your background legible, is your comp expectation in band, do you have a reason for wanting *this* company. Nobody gets hired here; plenty get filtered.

**2. Technical screen (45–60 min) or asynchronous coding assessment.** Either a live practical build, a CodeSignal/HackerRank-style automated round, or a small API-integration exercise. At AI-product companies this is increasingly "build a tiny working thing with an LLM" rather than an algorithms question.

**3. Take-home or work trial (2–8 hours of work, 2–7 day deadline).** The most common single artifact in this market. RAG, agentic, conversational, or document-processing flavored. Graded on functional correctness, architecture, **evaluation methodology**, production readiness, and documentation — with evaluation methodology being the criterion candidates most often skip and interviewers most consistently weight.

**4. Take-home defense / deep dive (45–90 min).** You walk your own code. Tests whether you made deliberate choices or lucky ones. This is where the take-home is actually graded.

**5. AI system design (60 min).** RAG, agent, or serving design under an explicit cost and latency budget. The distinguishing move from a backend design round is that you must open with how you'd measure quality.

**6. Domain or research depth (45–60 min).** Varies hardest by archetype: transformer internals at a lab, serving internals at infra, product-and-metrics at an AI-product company, a distributed-training design at DeepMind.

**7. Behavioral / values / hiring manager (45–60 min).** At most companies a formality with teeth; at Anthropic reportedly the primary filter; at Amazon a fully structured Leadership Principles evaluation with a bar raiser.

**📅 Volatile:** stage counts, names, and ordering change per team and per quarter — everything company-specific in this section is the reported shape at the time of compilation and should be re-confirmed with your recruiter before you schedule.

**📐 Numbers you must know:** budget the calendar, not just the prep. A five-to-seven stage loop at a fast company runs roughly 11–20 days end to end; at big tech, 4–8 weeks including hiring committee and offer review. Take-homes add 2–7 days of deadline plus your actual build time. If you want three offers landing in the same two-week window for leverage, you must start the slow processes 3–4 weeks *before* the fast ones. That single scheduling fact is worth more than any individual answer in this guide.

### Is there a coding round or not? Give me the honest distribution, because I've heard both.

Both are true, and the distribution is the answer: the large majority of senior AI engineering loops — call it roughly seven in ten — contain **no LeetCode-style algorithmic round at all**. They replace it with a practical build, a take-home, or an API-integration exercise. That is a genuine market shift and it is the single most common thing candidates prepare wrong.

But the exceptions are named, consistent, and brutal, and you must know them by name: **Perplexity** runs a genuinely hard machine-coding round at LeetCode-Hard difficulty. **xAI** runs algorithmic screens. **Anthropic** runs an automated coding screen in the CodeSignal family before humans get involved. **Quant and trading firms hiring AI engineers** (Jane Street, HRT, Citadel-adjacent) run full quantitative and algorithmic gauntlets and do not care that the role says AI. Several big-tech applied loops still carry one classical coding round inside an otherwise-modern loop.

**📅 Volatile:** which specific companies run which screen changes; verify with the recruiter.

The practical decision rule I use: maintain a *floor* of algorithmic fluency rather than a peak. Two to three hours a week of medium-difficulty problems keeps you from being embarrassed by an unexpected screen, and that is cheap. Grinding two hundred hard problems for an AI Engineer loop that is going to ask you to design an eval harness is a catastrophic misallocation — that time buys you a full pass through transformer internals and a working RAG evaluation project instead.

**⚠ Trap:** Reading "70% have no coding round" as "I can stop coding." What replaced the algorithm round is *harder to fake*: you are now asked to build a working thing under time pressure with real APIs, and fluency — writing correct Python fast, without autocomplete, since several of these companies prohibit AI tools in live rounds — is more load-bearing than before, not less. The bar moved from puzzle-solving to shipping speed.

**🗣 Say this in the room** (when a recruiter can't tell you): "Should I expect a data-structures-and-algorithms round, or is the coding portion practical and API-oriented? I'll prepare for both, but I'd like to know where to put the weight."

### What is the recruiter screen actually filtering on for these roles, and what do you say in the first ninety seconds?

It filters on three things and nothing else: legibility, band, and motivation. Legibility means the recruiter can write one sentence in the ATS that makes a hiring manager say "yes, talk to them." Band means your comp expectation does not immediately disqualify you or, more dangerously, anchor you low. Motivation means you have a company-specific reason, because at this compensation level everyone is technically plausible and the tiebreaker is who will actually accept.

For a backend engineer moving into AI the legibility problem is the real risk. If your ninety seconds is a career chronology, the recruiter writes "backend engineer, interested in AI" — which is the same sentence as a thousand other applicants. You need them to write "shipped LLM features in production, owns evals and cost, deep systems background."

The structure that works: one sentence of substrate, two sentences of AI evidence *with numbers and a named failure mode*, one sentence of why this company.

**🗣 Say this in the room:** "I'm a senior backend engineer — Python, distributed systems, Postgres, high-throughput services. For the last stretch I've been building on the model layer: I own a retrieval-and-agent feature end to end, including the eval harness that gates its deploys and a cost model that got us from about three cents to under a cent per resolved request, mostly through prefix caching and routing. The failure mode that taught me the most was stale answers surviving a reindex because our eval didn't cover freshness. I'm talking to you specifically because [company-specific reason]."

**⚠ Trap:** Giving a number band before you know theirs. Ask for the range for the level first — "what's the band you have budgeted for this level?" — and if pressed, give a range anchored on total comp and explicitly flagged as flexible on structure. Anchoring low in a recruiter screen has cost candidates more money than any interview answer ever will.

### Walk me through an AI-focused system design round and tell me how it differs from a backend system design round.

Structurally it is the same interview you have already passed many times — requirements, constraints, a component diagram, data flow, scaling, failure modes — and about 60% of the content is genuinely your existing skill set. Three things are different, and each is a graded discriminator.

**First: you must open with evaluation.** In a backend design round nobody asks how you'll know the service is correct, because correctness is definitional. Here output quality is a continuous, contested variable, and the strongest single move available to you is to open with "before I design this, let me say how I'd know it works" and then name an offline eval set, a scored metric, and an online proxy. I have watched this one move flip an interview's tone. Candidates who bolt evaluation on at minute 50 get rated a level lower than candidates who lead with it.

**Second: the cost model is part of the design, with arithmetic.** Backend design rounds rarely ask you to price a request. AI design rounds do, because token cost is often the dominant marginal cost of the feature and a design that is 3× too expensive is not a design. You are expected to say things like: "the system prompt is 6k tokens, the retrieved context is 4k, output is 600 — at $3 per million input and $15 per million output that's (10,000 × 3 + 600 × 15)/1e6 = $0.030 + $0.009 = $0.039 per request; at 100k requests/day that is $3,900/day or about $117k/month, which is too much, so here's how prefix caching and routing take it down."

**Third: the latency vocabulary changes.** Your p95 becomes three numbers: time to first token, inter-token latency, and total completion time — and which one matters depends on whether the surface streams. A chat UI lives or dies on TTFT; a batch document pipeline does not care about TTFT at all and cares about throughput per dollar. Saying "p95 latency" without decomposing it is a tell that you have not shipped a streaming surface.

**⚠ Trap:** Designing the happy path and treating nondeterminism as an edge case. In these systems the model *will* return malformed JSON, call a tool with a hallucinated argument, loop, and refuse. Those are not edge cases, they are a percentage of daily traffic, and your design must show where they are caught — schema validation with a repair path, tool-call argument validation, a step budget, a fallback route.

**🗣 Say this in the room:** "Before I draw anything: I'd want to define the quality metric and build a fifty-to-two-hundred example eval set from real traffic, because every later decision — retrieval strategy, model choice, whether we can route the cheap 80% to a smaller model — is only decidable against that."

### The take-home defense round — what's being graded that isn't in the code?

Deliberateness. The code proves you can build; the defense proves you *chose*. Interviewers are probing for the difference between a candidate who assembled a working pipeline from tutorials and one who made a sequence of tradeoffs they can still defend under pressure.

The questions are predictable and you should rehearse them out loud: Why this chunking strategy and what did you compare it against? Why this embedding model — did you measure or did you pick the default? How do you know retrieval is working, separately from generation? What's your p95 and where does it go? What would break at 100× the documents? What did you deliberately not build, and why? What would you do with two more days?

That last pair is where seniority shows. A mid-level candidate defends everything they built. A senior candidate says "I deliberately skipped a reranker because my eval said recall@10 was already 0.93 and the latency budget was 2 seconds; if recall were the bottleneck, that's the first thing I'd add, and I'd expect roughly 100–300ms of added latency for a cross-encoder over 20 candidates." Naming what you skipped, with the reason and the trigger that would change your mind, is the highest-signal sentence available in this round.

**⚠ Trap:** Not re-reading your own submission before the defense. There is often a 3–7 day gap, and forgetting why you set `chunk_size=512` reads exactly like not having decided it. Keep a one-page decision log with every non-obvious choice and its alternative — write it while you build, not after.

### What is a values or culture round at an AI lab actually testing, and how do people fail it?

It is testing whether your judgment holds when the technically-optimal action and the responsible action diverge — and, at labs that take safety seriously as an institutional commitment rather than a marketing line, it is reportedly a primary filter rather than a formality. Anthropic in particular is widely reported to weight this round heavily enough that strong technical candidates fail there.

What it is not: a trivia quiz about the company's published principles. Reciting them is a mild negative — it reads as preparation for the round rather than possession of the disposition.

The actual question shapes are scenario-based: you find a serious flaw in something already shipping and fixing it slips a launch; a customer asks for a capability you think is harmful or out of scope; your eval says the feature is not ready and your PM says the demo is Thursday; you discover a colleague's benchmark number does not reproduce. What is graded is whether you (a) name the tension honestly instead of pretending there isn't one, (b) escalate rather than unilaterally deciding, (c) propose a concrete path that preserves both the launch and the standard where possible, and (d) can state what would change your mind.

The failure modes are consistent. **Performative safety** — claiming you would block every launch — reads as unserious and as someone who has never had to ship. **Pure velocity** — "I'd ship it and fix forward" — fails immediately at a lab. **Vagueness** — "I'd raise it with the team" with no mechanism, no metric, no decision owner. And **overclaiming**, telling a story with an implausibly heroic ending; interviewers at this level probe stories hard and an invented one collapses in two follow-ups.

**🗣 Say this in the room:** "I'd separate 'is this a launch blocker' from 'is this a known limitation we're comfortable documenting.' If it's a blocker I'd say so with the eval number attached, propose the smallest scope that ships safely, and put the decision in front of the person who owns the risk rather than making it myself in a Slack thread."

### How does leveling work in these orgs — what evidence gets you IC5 versus IC6?

Leveling is decided on scope evidence, not on years or on how hard the technical questions were. The rubric interviewers are filling in is roughly: what is the largest thing this person has been *accountable* for, how many people did their decisions bind, and did they define the problem or receive it.

**IC5 (senior)** evidence: you owned a significant service or feature end to end, made the architectural calls within it, handled its production incidents, mentored one or two engineers, and delivered against a spec someone else largely set. In AI terms: "I own the retrieval and agent layer for feature X, defined its evals, cut cost 60%, and it serves N requests/day."

**IC6 (staff)** evidence: your decisions bound other teams, you defined the problem rather than the solution, you set a technical direction that outlived a project, and you can point to something that did not exist as a category before you argued for it. In AI terms: "I built the evaluation platform three product teams now gate their deploys on, and I set the model-routing policy the org uses" — that is scope over people you do not manage.

The thing backend engineers underweight: at this level, *written* evidence counts. A design doc that changed a decision, an incident review that changed a process, an internal standard others adopted. Bring two of those as artifacts you can describe in ninety seconds.

**⚠ Trap:** Titles do not transfer. "Senior" at a 200-person company frequently maps to IC4/IC5 at big tech and occasionally to IC6 at a startup. Never argue level from your current title; argue it from scope evidence and let them map it. And be aware that down-leveling is the most common silent outcome of a strong-but-narrow loop — if every story you tell is inside one service, you will be leveled inside one service.

**💰 Math:** The delta is worth arguing about. At the companies on your list the IC5→IC6 step is commonly on the order of $80k–$200k/year in total compensation, dominated by the equity component. Over a four-year vest that is $320k–$800k. Spending ten hours preparing two scope stories with numbers is, on that arithmetic, the highest hourly-rate work in your entire preparation. **📅 Volatile:** bands move; verify against current levels data before you negotiate.

### You're told the onsite includes a "research deep dive." What are you expected to bring, and what if you have no papers?

You are expected to bring one piece of technical work you understand to the bottom and can defend against an expert, and — this is the part candidates miss — it does not have to be research. The round is testing depth of ownership, ability to explain a technical decision to someone smarter than you about their area, and whether your curiosity survives contact with hard questions.

If you have publications, they use those. If you do not, the acceptable substitutes at a frontier lab are: a from-scratch reimplementation with measurements (write attention and a KV cache, then show a memory and throughput curve, then explain why it deviates from the naive prediction), a genuine performance investigation (you profiled a serving stack and found the bottleneck was X, with before/after numbers), a paper you have implemented and can critique — not summarize, *critique*: what would you ablate, what does the paper not control for, what would you expect to break at a different scale, or an open-source contribution that required understanding somebody else's internals.

The structure that works is the same as a good post-mortem: what was the question, why was the obvious approach wrong, what did you try, what did the data say, what did you conclude, and what would you do differently. Twelve minutes of that, then forty of questions.

**⚠ Trap:** Presenting something you built but did not decide. If you cannot answer "why this and not the alternative" three levels deep, pick different work. The classic failure is presenting a project where you followed a framework's defaults — every "why" bottoms out in "that's what the tutorial did," and the interviewer stops asking.

**🏋 Drill:** 20 minutes, no notes, no slides. Whiteboard one technical decision you made this year: the alternatives, the measurement, the outcome, and the thing you would change. Record yourself. Pass criterion: no sentence begins with "we just" or "by default," and at least three specific numbers appear.

### How do you handle the AI-tool policy? Some loops ban Copilot, and I've heard at least one company runs both an assisted and an unassisted round.

You handle it by defaulting to *ask, then comply visibly*, and by having practiced both modes.

The landscape splits three ways. **Prohibited:** several labs and research-heavy orgs — Anthropic, DeepMind, xAI among the commonly-reported ones, plus a set of enterprises — ban AI assistance in live rounds outright and back it with proctoring: browser monitoring, screen sharing, occasionally audio analysis, and an increasing share of in-person onsites specifically to close this hole. **Dual-mode:** Microsoft is the notable example of running *both* an AI-assisted round and a raw-coding round, on the explicit theory that they need to see both your ceiling with tools and your floor without them. **Encouraged:** most AI-product companies — Cursor most obviously — expect you to use AI tools and may quietly grade you on whether you use them *well*. **📅 Volatile:** these policies changed materially in the last two years and will change again; confirm per-loop.

The practical preparation consequence is uncomfortable and worth taking seriously: you must be able to write correct Python — a class, a dataclass, an async function, a retry with backoff, a small numpy or torch tensor manipulation — from memory at speed, without autocomplete. Most working engineers have quietly lost this. Two weeks of writing code in a plain editor fixes it, and it is the cheapest insurance in the process.

In an assisted round, what is graded is not whether you use the tool but *how*: do you specify clearly, do you read what it produced, do you catch the bug it introduced, do you know when to stop prompting and write it yourself. Narrate that. "I'll let it scaffold the client and the retries, but I'm going to write the streaming parser myself because that's where the edge cases are" is a strong sentence in an assisted round.

**⚠ Trap:** Using a hidden assistant in a prohibited round. Detection is better than candidates assume — real-time transcription tools and answer overlays are specifically what the proctoring targets — and the consequence is not a rejection, it is a permanent blacklist at a company you may want for a decade. It is a terrible expected-value trade.

### Of all these rounds, which one carries the hidden weight — and how do you find out which one before you sit it?

There is always one round that is disproportionately decisive, and it is almost never the one the candidate over-prepares. The archetype predicts it:

At a **frontier lab**, it is the values or alignment-judgment round — technical competence is table stakes at the applicant pool they see, and judgment is the differentiator. At an **AI-product company**, it is the take-home defense or the product-sense conversation: can you argue about a metric and a user, not just a stack. At **AI-infra**, it is the systems design round with real numbers — GPU memory, batching, tail latency — because the whole job is that. At **big-tech applied**, it is the behavioral round evaluated against a structured rubric, plus the design round's cross-team story; Amazon's bar raiser is the explicit institutionalization of this. At **FDE**, it is the customer or scoping conversation. At **high-comp non-AI-native** companies (Stripe, Ramp, and similar), it is usually the practical build plus the reliability-and-cost argument — they are hiring an engineer who will not blow up their unit economics.

How to find out: ask. Recruiters answer this honestly far more often than candidates expect, because a prepared candidate makes their pipeline look good. The phrasing that gets a real answer is not "which round matters most" — that invites "they all matter" — it is: **"What's the most common reason strong candidates don't make it through this loop?"** That question gets you the hidden weight, in one sentence, roughly three times out of four.

**🗣 Say this in the room** (to the recruiter): "What's the most common reason strong candidates don't get through this process? I'd rather over-prepare for the right thing."

**🔍 Failure taxonomy — misallocated preparation:** (1) *Over-indexed on algorithms* for a loop with no algorithm round — symptom: you feel over-prepared and still get "not enough depth on evaluation." (2) *Over-indexed on internals* for an AI-product loop — symptom: you can derive attention but fumble "what metric would you ship on." (3) *Under-indexed on communication* for FDE or enterprise — symptom: technically clean rounds, vague "fit" rejection. (4) *Under-indexed on the take-home* — symptom: you pass every live round and get rejected after submitting; cause: you treated a graded artifact as a formality and skipped the eval section.
