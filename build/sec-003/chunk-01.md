### Walk me through the LLM feature at the top of your resume.

This is the question that decides whether the rest of the loop happens, and almost everyone answers it as a feature tour — "we built a support assistant, it uses RAG over our docs, users liked it." That answer is indistinguishable from a weekend project, so it gets scored as one.

The mental model: **a shipped LLM feature is a claim about a system under uncertainty, and a claim is only credible if you can state how you measured it, what it cost, how fast it was, and how it failed.** Four numbers. If any one is missing, the interviewer's prior is that you demoed something rather than operated it. This is not an arbitrary rubric — it maps exactly onto how you'd defend any backend service in a design review, except that the correctness axis is statistical rather than binary, and that is the axis backend engineers forget to bring.

So the shape of the answer is four beats, in this order:

**1. The eval.** Not "users liked it." A frozen set with a size, a construction method, and a metric. "182 real support tickets, stratified across the six intent classes by production frequency, labelled by two of our support leads with disagreements adjudicated by a third; primary metric was resolution-without-escalation judged by a rubric grader, secondary was groundedness — every factual claim traceable to a retrieved chunk." Interviewers are listening for whether the eval predates the system or was retrofitted to flatter it.

**2. The cost number, with arithmetic.** "3,200 input and 400 output tokens per call at $3/$15 per million meant $0.0096 + $0.0060 = $0.0156 per call; at 2.4M calls a month that's $37,440. Prefix-caching the 2,600-token system-plus-few-shot header at a 90% read discount took the cached portion from $0.0078 to $0.00078, and routing 65% of traffic to a small model at $0.25/$1.25 got the blended cost to $0.0039 a call — $9,264 a month, a 75% cut." **📅 Volatile:** those per-million prices move constantly; carry *your* prices and re-derive before the loop.

**3. The latency number, with the decomposition.** p95 TTFT and p95 end-to-end, separately, because they have different causes. "TTFT p95 was 4.2s, of which 2.9s was our own retrieval fan-out, not the model — we were doing a sequential embed → vector search → rerank chain. Parallelising embed with the metadata filter and moving the reranker to a cross-encoder on the top 40 got TTFT p95 to 1.1s."

**4. The named failure mode, and what you did about it.** This is the beat that separates seniors from everyone else, because it proves the system ran long enough to hurt you. "The router sent multi-hop questions to the small model, where the multi-hop slice of the eval dropped from 71% to 48% while the aggregate barely moved — which is exactly why the eval is sliced. We added a confidence-gated escalation that pushed about 8% of traffic back to the frontier model and cost us 3 points of the savings."

**⚠ Trap:** leading with the architecture. If you open with "so it's a RAG pipeline with a hybrid retriever and a reranker," you've told the interviewer you think the interesting part is the diagram. Every candidate has that diagram. Almost none of them can tell you what it cost per call.

**🗣 Say this in the room:** "The short version is: 182-ticket frozen eval, resolution-without-escalation as the primary metric, $0.0038 per call blended after caching and routing, 1.1s p95 time-to-first-token, and the failure mode that nearly shipped was the router quietly degrading multi-hop questions — invisible in the aggregate, obvious in the slice. Want me to go deeper on any of those?"

### My resume is eight years of Python backend work. How do I translate it into AI-engineering language without lying?

The instinct most people follow is to reword: "built REST APIs" becomes "built AI-powered APIs." That is transparently cosmetic and it burns your credibility on the first bullet an interviewer probes. The correct operation is not rewording, it is **re-anchoring the same evidence against the constraints the AI role actually has.** You are not claiming new work. You are claiming that the work you did is load-bearing for the work they need done — and that claim is true, which is why it survives probing.

Three translation rules I enforce:

**Rule 1 — translate the constraint, not the noun.** Nobody hiring an AI Engineer cares that you built a rate limiter. They care enormously that you have shipped a *token-bucket limiter over a resource whose unit of consumption is variable and only known after the fact* — because that is the exact shape of the provider TPM-limit problem, and most ML-background candidates have never solved it. So the bullet becomes: "Built the per-tenant quota layer for a multi-tenant API where cost per request varied 40× by payload; enforcement keyed on estimated-then-reconciled units, not request count." Now an AI hiring manager reads that and sees their own problem.

**Rule 2 — the LLM bullet earns the top slot only if it has all four numbers.** If your one AI project has an eval, a cost figure, a latency figure and a failure mode, it leads. If it does not, do not promote it — a weak AI bullet at the top invites twenty minutes of questioning on your weakest material, while a strong distributed-systems bullet at the top gets you credited for depth and *then* you talk about the AI work on your terms. I have watched candidates lose a screen purely on bullet ordering.

**Rule 3 — the honest verb ladder.** These verbs have specific meanings to the people reading, and using one you can't defend is the single fastest rejection in this section:

- *Used* / *integrated* — you called an API. Completely respectable, and honest.
- *Built* — you own the code and can whiteboard the data flow.
- *Evaluated* — there is a frozen set, a metric, and a decision that changed because of a number.
- *Fine-tuned* — you produced weights, you can name the base model, the data volume, the method (LoRA rank / full SFT / DPO), the hardware, and the eval that justified it against the prompting baseline. If you cannot produce all six, **delete the word.**
- *Trained* — pre-training. Almost certainly do not use this word.

**⚠ Trap: the word "fine-tuned" on a resume you cannot defend.** This is the most reliable self-inflicted rejection in AI hiring. Interviewers at every archetype on your list treat it as a free depth probe, and it is a probe you will fail with "we used a service that did it for us." Worse, the reflex-fine-tuning signal is itself a seniority downgrade: senior candidates are expected to explain why they *didn't* fine-tune. A resume that says "evaluated LoRA fine-tuning against a prompt-plus-retrieval baseline; prompting won on 5 of 6 slices at 1/9th the operational complexity, so we shipped the baseline" is strictly stronger than one that says "fine-tuned Llama."

**🗣 Say this in the room:** "I'd characterise myself as a backend engineer who ships LLM systems, not an ML engineer. I've built and evaluated retrieval and agent systems in production; I haven't trained a model, and the one time we scoped a fine-tune we killed it because prompting plus retrieval beat it on our eval."

### You've got twenty seconds of my attention on a one-page resume. What am I looking at, and what should be there?

Assume the reader is a hiring manager with 60 resumes and a full calendar. The actual scan is not reading — it is four fixations, in a predictable order: the top-left identity block, the first two bullets of the most recent role, the company names down the left rail, and the links. Everything else in the document exists to survive round two, after the twenty seconds have already bought you a closer read.

So the structure I'd enforce:

**The header does one job: it states the role you are applying for and gives the three links that constitute proof.** Name, one line of positioning ("Senior backend engineer — LLM systems, retrieval, evaluation; Python/FastAPI/Postgres"), then GitHub, the portfolio artifact, and email. No address, no photo, no "Objective" paragraph, no skills-bar graphics with four filled circles out of five. The links must be plain, short, and go somewhere that loads in under two seconds.

**Then a four-to-six-line summary — but only if it contains numbers.** A summary that says "passionate engineer with a track record of delivering scalable solutions" is worse than no summary, because it consumes one of your four fixations with zero information. A summary that says "Ships LLM features end-to-end: retrieval, evals, cost control. Most recent: cut blended inference cost 75% ($37k → $9k/mo) while holding a 182-case eval flat" spends that fixation buying you a read of the whole page.

**Then experience, reverse-chronological, with the AI-relevant role expanded and everything older compressed.** Your 2018 role gets one line — company, title, dates, and one bullet. Nobody is hiring you for 2018. The temptation to give every role equal depth comes from a fairness instinct that has no counterpart in the reader's head.

**Then projects — but only if a project outranks a job bullet.** For this transition, it very often does. A public eval harness with a benchmark table beats a fourth bullet about a Kafka migration.

**Then a skills line, flat, no proficiency ratings.** Group them: languages, then LLM/AI (whatever you can defend), then data/infra. Proficiency self-ratings are a liability — "Python: expert" is a claim you cannot win and can lose.

**📐 Numbers you must know:** one page means roughly 550–650 words of actual content. A 12-bullet experience section at 22 words a bullet is 264 words, which is already 45% of your budget. That arithmetic is why "add another bullet" is almost always the wrong move and "make this bullet carry a number" is almost always the right one.

**⚠ Trap:** two pages "because I have eight years of experience." Seniority is demonstrated by what you chose to cut. A two-page resume from a senior IC reads as an inability to prioritise, which is precisely the competency the role is being hired for. The exceptions are real but narrow — some big-tech and enterprise processes are page-count-indifferent, and academic/research CVs are a different document entirely.

### Show me how you'd rewrite a weak resume bullet. Give me the rule you're applying.

The rule is **claim + metric + mechanism, in that order, in one sentence**, and every bullet on the page obeys it or gets deleted. Claim is what changed. Metric is the number that proves it changed. Mechanism is the two or three words that prove you did it rather than watched it. Reversing the order — leading with the mechanism — is the most common defect, because engineers find the mechanism most interesting and the reader does not.

Watch four:

**Weak:** "Worked on improving the performance of the document search service."
**Strong:** "Cut search p95 from 840ms to 190ms (−77%) by replacing per-request embedding with a warm ONNX runtime pool and moving reranking to top-40 candidates only."
*Claim: latency cut. Metric: 840 → 190. Mechanism: warm pool + candidate truncation.*

**Weak:** "Used OpenAI API to build a chatbot for internal knowledge base."
**Strong:** "Shipped an internal knowledge assistant to 400 weekly users at $0.0038/query blended, holding 78% groundedness on a 182-case frozen eval; cost came from prefix caching a 2,600-token header and routing 65% of traffic to a small model."
*Note what changed: not the work, the accounting of it.*

**Weak:** "Responsible for Celery task queue and background jobs."
**Strong:** "Owned the async job tier (Celery/Redis, ~2M tasks/day); cut poison-message incidents to zero by adding a per-task idempotency key and a DLQ with automated replay."
*"Responsible for" is a job-description verb. It describes a slot, not a person.*

**Weak:** "Familiar with LangChain, vector databases, and prompt engineering."
**Strong:** delete it. A familiarity claim with no artifact behind it is a liability line — it invites a question you have no evidence for and it dilutes the lines that do have evidence.

**⚠ Trap: the unsourceable percentage.** "Improved efficiency by 40%" with no before/after pair and no mechanism reads as invented, and once one number reads as invented, every number on the page is discounted. The fix is always the same: state the two endpoints, not the delta. "840ms → 190ms" cannot be faked in a follow-up; "−77%" can.

**🏋 Drill:** take your current resume, cover the right-hand two-thirds of the page with your hand so only the first six words of each bullet are visible, and read down the column. If more than two bullets begin with "Worked on," "Responsible for," "Helped," or a technology name, you have a claim-ordering problem. Timed: 10 minutes to rewrite the offenders. Pass criterion: every bullet's first six words state an outcome.

### Your resume says you cut inference cost 75%. Walk me through the arithmetic.

They will ask this, at Ramp and Stripe and Databricks especially, because cost discipline is the thing their AI orgs are actually short of. The pass condition is that you can do it out loud, in units, without notes. Numbers on a resume that you cannot re-derive live are worse than no numbers.

Here is the derivation, in the order I'd say it:

Baseline: 2.4M calls a month, averaging 3,200 input tokens and 400 output tokens. At $3 per million input and $15 per million output, that's 3,200 × 3 / 10⁶ = **$0.0096** in and 400 × 15 / 10⁶ = **$0.0060** out, so **$0.0156** a call, times 2.4M = **$37,440 a month**.

First lever, prefix caching. Of the 3,200 input tokens, 2,600 were a fixed system prompt plus few-shot block — identical on every call. At a 90% cached-read discount that block costs 2,600 × 0.30 / 10⁶ = **$0.00078** instead of **$0.0078**, saving $0.0070 a call. New per-call: **$0.0086**, or $20,640 a month. Note what made this possible: the cacheable content had to be a *stable prefix*, so we moved the per-user context below the few-shots. That ordering decision is the whole trick, and it's the part interviewers probe.

Second lever, routing. A classifier sent 65% of traffic — single-hop, single-document questions — to a small model at $0.25/$1.25 per million. That's 3,200 × 0.25/10⁶ + 400 × 1.25/10⁶ = $0.0008 + $0.0005 = **$0.0013** a call. Blended: 0.65 × 0.0013 + 0.35 × 0.0086 = 0.000845 + 0.003010 = **$0.00386** a call → **$9,264 a month**.

$37,440 → $9,264 is a **75.3% reduction, $28,176/month, $338k/year.**

And then the part that makes it a senior answer rather than a spreadsheet: **what it cost me.** The router itself runs on every request — a small classifier at roughly $0.0002 a call adds $480/month, which is inside the noise. The real cost was quality: the multi-hop slice of the eval fell from 71% to 48% before we added confidence-gated escalation, which pushed 8% of traffic back to the frontier model and took the blended figure from $0.0034 to $0.00386. I'd rather quote the honest post-escalation number and be able to defend it than quote the pretty one.

**💰 Math worth internalising:** at these prices, **input tokens are usually the bill.** 3,200 in / 400 out at $3/$15 is $0.0096 vs $0.0060 — input is 62% of spend despite output being 4.7× more expensive per token, purely because there are 8× more of them. Engineers reflexively optimise output length. The 10× lever is almost always the prompt.

**📅 Volatile:** every price above. Re-derive with current numbers the week of your loop; quoting a stale price in 2026 is a specific, noticed failure.

### Which claims on an AI-adjacent resume get a candidate rejected, and how do you know you're overclaiming?

There is a clean test, and I'd apply it to every line before submitting: **could a competent interviewer spend eight minutes on this bullet and find the floor?** If the honest answer is "no, they'd hit the floor in ninety seconds," the bullet is an overclaim regardless of whether the words are technically true.

**🔍 Failure taxonomy — the overclaim ladder, in descending order of how fast it kills you:**

1. **"Trained a model."** Unless you ran pre-training, this is read as either a lie or a vocabulary error, and both are fatal at senior level. Fine-tuning is not training. Prompting is definitely not training.
2. **"Fine-tuned Llama/Mistral/Qwen."** Probe: base model, parameter count, method, dataset size and provenance, hardware and wall-clock, and the eval delta against the prompting baseline. Six facts. Missing two is survivable; missing four is a reject.
3. **"Built a RAG system."** Probe: chunking strategy and why, embedding model and dimension, index type, retrieval metric (recall@k on what labelled set?), and what you did about the stale-index problem. If "we used LangChain's default" answers three of those, the bullet was about a tutorial.
4. **"Built an agent."** Probe: how does the loop terminate, what does a tool error look like to the model, what's your context budget per turn, and what's your cost per *resolved task* versus per call. The absence of a termination story is the tell.
5. **"Reduced hallucinations by X%."** Probe: measured how? Hallucination is not a metric, it is a category of failures that requires an operational definition — groundedness against retrieved spans, or claim-level factual precision against a reference. If you can't state the definition, drop the claim.
6. **"Production" as an adjective.** Production means it had real users, an on-call story, and something that broke. A demo behind a login used by your team of six is "internal pilot" — and calling it that honestly reads as *more* senior, not less.
7. **"Improved accuracy by 30%."** Relative or absolute? From what baseline, on what set, with what n? A 30% relative improvement on a 40-example set is noise, and the interviewer knows the binomial width even if you don't.

**⚠ Trap:** believing you must inflate because everyone else does. In this specific market the reverse is true — the supply of resumes claiming RAG and agents and fine-tuning is effectively infinite, and the differentiator is *calibration*. An interviewer who hears "I haven't fine-tuned anything, here's why we decided not to, and here's the eval that made the call" updates hard in your favour, because that sentence is unfakeable and it is exactly what the job requires.

**🗣 Say this in the room:** "I've deliberately kept my resume to things I can be interrogated on for ten minutes. If something's on there, I own the code and I can whiteboard it. If it's not there, I probably touched it and don't consider myself credible on it yet."

### I've got one LLM project and it's not the biggest thing I've built. How do I present eight years of backend work so it's an asset instead of a liability?

Reframe the question the reader is actually asking. A hiring manager at Notion or Sierra or Ramp is not asking "has this person done more AI than the other candidates" — they are asking **"can this person be trusted to own a production system that happens to have a model in it."** The scarce skill in AI product teams right now is not model knowledge, which is a few weeks of study for a strong engineer. It is the ability to run a system with multi-tenancy, quotas, retries, observability and a cost line. That is eight years of your life.

So the presentation strategy is a barbell: **one deep AI artifact with all four numbers, plus a backend record framed entirely in the vocabulary of the constraints AI systems have.** Not a thin AI record padded out.

Concretely, the four backend competencies that translate hardest and that I'd make sure appear explicitly:

- **Nondeterministic-output idempotency.** You've built idempotency keys over deterministic handlers. LLM systems break that model — the same key can legitimately produce different output, so the interesting question becomes what you key on and what you store. If you have any experience with at-least-once delivery over non-idempotent side effects, that story is gold here.
- **Cost-per-unit-work accounting.** If you've ever built or defended a per-tenant cost model, that is directly the skill. Most people entering this field have never had to say what a request costs.
- **Streaming and cancellation.** Long-lived streaming responses with a client that can vanish mid-flight — you have this from SSE/WebSocket work, and it is precisely the token-billing-leak problem.
- **Observability with high-cardinality dimensions.** Traces where the interesting attribute is per-request and unbounded. This is exactly LLM tracing, one span per model call with token counts and latencies as attributes.

**⚠ Trap: apologising for your background in the summary line.** "Backend engineer transitioning into AI" is a self-inflicted demotion — it frames you as an entrant. "Backend engineer who ships LLM systems" is the same set of facts and frames you as an operator. The second one is also more accurate: you are not transitioning away from backend, the job *is* backend with a probabilistic dependency.

**🗣 Say this in the room:** "My view is that the hard part of an AI product isn't the model call — it's everything you already have to be good at: quotas, retries that don't quadruple your bill, evals that catch a regression before a customer does, and knowing what a request costs. That's the part I've been doing for eight years. The model layer I've learned deliberately and can go as deep as you want on."

### How much does ATS keyword-matching actually matter, and where does it stop mattering?

Most of what circulates about ATS is folklore, and acting on the folklore produces worse resumes. Here's the honest mechanism.

An applicant tracking system — Greenhouse, Lever, Ashby, Workday and friends — is fundamentally a **CRM for candidates**: it stores applications, routes them to reviewers, tracks stage transitions, and schedules. Its primary function is not filtering. Most of these systems do parse your resume into structured fields and most support keyword search and filters, and a recruiter working a 400-application req absolutely will search "Kubernetes" or "PyTorch" to triage. But the widely-repeated claim that some large fraction of resumes are auto-rejected by a robot before human eyes is not how the mainstream tools are typically configured; the more accurate model is **a human recruiter, under time pressure, using search and filters as a triage tool.** The failure mode you should actually design against is *not being surfaced by a search*, not *being scored and killed by an algorithm*.

That distinction changes the tactics:

- **Parseability matters, and it is cheap.** Single column. No text inside images, no tables for layout, no headers/footers carrying content, standard section names ("Experience," "Education," "Skills"), and a real PDF with selectable text. Multi-column resumes parse into interleaved garbage in some parsers, which is a genuinely dumb way to lose.
- **Term coverage matters, in the sense of using the industry's word.** If the JD says "RAG" and you wrote "retrieval-augmented generation," write both once. If it says "LLM evaluation" and you wrote "quality measurement," fix it. This is not keyword stuffing; it is speaking the searchable dialect.
- **Invisible white-text keyword stuffing:** don't. It is detectable, it is treated as fraud, and it will get you blacklisted at the company.

**Where it stops mattering — and this is the part that should change your behaviour:** ATS is only load-bearing on the **cold-inbound** path. A referral, a recruiter who reached out to you, a hiring-manager DM that converted, or a warm intro all enter the funnel at a stage where a human is already committed to reading you. The resume still has to be good; it just no longer has to survive a search. So the correct strategic conclusion is not "optimise the resume harder" — it is **shift volume off the cold-inbound path**, which is what the referral and outreach material later in this section is for.

**📐 Numbers you must know:** the funnel shape you should assume, and then instrument for yourself, is roughly this — cold applications to a hot AI req convert to a recruiter screen in the low single-digit percents; referrals convert to a screen at multiples of that. The exact multiple varies wildly by company and is not worth memorising. What *is* worth memorising: **one referral is worth on the order of ten to thirty cold applications of your time**, which is why spending an evening on a warm intro beats spending it on twenty more applications.

**⚠ Trap:** tailoring the resume per-application at high volume. It feels diligent and it is where the hours go. Build two versions — one AI-product-weighted, one big-tech-applied-weighted — and swap the summary line and the top three bullets. Beyond that, marginal returns are near zero and the time is better spent on the artifact that gets you referred.

### Do I need different resumes for Cursor and for Google? What actually changes?

Yes, but the delta is small and specific — two versions, not fifteen. The reason is that the two archetypes on your target list read for genuinely different evidence, and a document optimised for one reads as slightly off to the other.

**The AI-product read (Cursor, Perplexity, Notion, Figma, Sierra, Harvey, Glean, Ramp).** These teams are small, the hiring manager is often reading personally, and they are optimising for *ship rate and product judgment*. What moves them: a public artifact they can click, evidence you have made a quality/cost/latency tradeoff under real constraints, and any signal you use the category of product they're building. Cursor famously cares whether you actually use AI coding tools daily — that is a real, reported rubric item, not a personality question. So on this version: portfolio links go above the fold, the projects section sits above older experience, and the summary line is product-shaped ("ships LLM features end-to-end, from eval design to cost control").

**The big-tech applied read (Meta, Google, Amazon, Microsoft, Databricks, Snowflake, Stripe).** Structured process, recruiter-first, leveling decided partly off the document. What moves them: scope and blast radius, unambiguous ownership language, systems scale, and enough legible seniority signal to justify an IC5/IC6 slot. Portfolio links matter far less; a hiring committee reading a packet six weeks later is looking at written evidence of impact. So on this version: scale numbers get promoted (QPS, data volume, team size influenced, systems owned), the projects section moves below experience, and every bullet is checked for the ambiguity of "we" — committees discount "we" hard, so it is "I designed" or "I led" or it does not count.

**What does *not* change: the four numbers.** Eval, cost, latency, failure mode — they belong on both. Nobody has ever been penalised for knowing what their system cost.

**⚠ Trap:** the "we" habit. Engineers say "we" out of genuine team decency, and every calibration process in big tech treats an unattributed "we" as a scope-unknown signal. Write "I" for what you did and name the team's contribution explicitly in one clause. It reads as more honest, not less: "Designed and owned the eval harness for a six-person team's assistant feature."

**🗣 Say this in the room** (when asked about the same project by these two audiences, the emphasis flips): to a product company — "the interesting decision was shipping the cheaper model for 65% of traffic and building the escalation gate, because a 3-point quality difference on single-hop questions wasn't worth 6× the cost." To a big-tech panel — "I owned the eval and cost model for the feature across three product surfaces; the design had to hold up under a 40× variance in per-tenant request cost."

### What actually happens to my application at each of my target companies — who reads it, and what are they reading for?

Knowing this changes where you spend effort, so it is worth being concrete rather than generic. The mechanism differs sharply between your two archetypes, and the biggest error is applying a big-tech strategy to a 60-person AI product company or vice versa.

**Small AI-product companies (roughly under 300 people — Cursor, Sierra, Harvey, and similar).** Frequently no dedicated recruiter on the req, or one recruiter covering everything. The hiring manager or a senior IC on the team reads the pipeline directly, often in batches, often clicking links. This is the regime where **a portfolio artifact is decisive**, because the reader is technical, curious, and has the authority to say "interesting, let's talk" without a committee. It is also the regime where a cold email to the hiring manager can work, because they are the actual decision-maker and their inbox is not yet industrialised.

**Mid-size AI-product companies (Perplexity, Notion, Figma, Glean, Ramp at current scale).** Recruiter-screened first, then hiring manager. The resume must survive a non-engineer's pattern-match before a technical reader ever sees it, which is the one place the searchable-dialect point matters. Portfolio still helps a great deal in the HM read, and referrals from any engineer inside are unusually powerful because these companies are still small enough that internal referrals carry weight.

**Big tech (Meta, Google, Amazon, Microsoft, Databricks, Snowflake, Stripe).** Recruiter-first, always. The resume is the primary object, the portfolio is at best a tiebreaker, and the packet travels through stages where the reader has never met you. Referrals matter but function differently — a referral typically guarantees a human read, not an interview. Leveling starts forming here, off the document.

**🎯 The practical allocation this implies:** if your target list leans product-company, the highest-leverage hour is spent on the artifact and the outreach, and the resume just needs to be clean and honest. If it leans big-tech, the highest-leverage hour is on the resume's scope language and the referral, and the artifact is a nice-to-have. Most candidates get this exactly backwards — they polish a resume for a company that would have hired them off a repo, and build a repo for a company that will never open it.

**⚠ Trap:** assuming a "quick apply" on a job board is equivalent to applying through the company's own careers page. Aggregator applications are frequently a lower-quality inbound channel and sometimes do not carry your links intact. Apply on the company's own ATS-hosted page, every time. It costs four extra minutes.
