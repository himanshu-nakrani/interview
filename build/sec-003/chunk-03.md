### I want a merged PR in a serious AI repo. How do I pick the repo and the issue so I don't waste two months?

The failure mode is picking by prestige — "vLLM is the most impressive, I'll do vLLM" — and then discovering that the tractable issues require CUDA kernel fluency you don't have, while the ones you *could* do are claimed within an hour. Pick by **the intersection of what you can actually land and what the signal is worth to your target list**, and accept that the second-best repo you can succeed in beats the best repo you'll abandon.

The filter I'd apply, in order:

**1. Does the repo's dominant language match your actual competence?** vLLM and SGLang are Python at the orchestration layer and CUDA/C++ at the kernel layer; there is a great deal of tractable Python — server surface, scheduler-adjacent logic, model integration, benchmarking, tests — and a much harder kernel layer. TRL, LangGraph and Outlines are essentially all Python. As a senior Python engineer with no GPU-kernel background, your leverage is in the Python layer, and there is no shame in that: a well-executed fix in the serving frontend of a major engine is a real contribution.

**2. What does merging there actually say about you?** These map to different claims:
- **vLLM / SGLang** → "I can operate inside a real inference engine." The strongest signal for AI-infra and for the serving round anywhere. **📄 Paper:** Kwon et al. (2023) — PagedAttention, which brought virtual-memory-style paging to the KV cache and replaced contiguous per-sequence allocation; that idea is vLLM's core. **📄 Paper:** Zheng et al. (2024) — SGLang, whose RadixAttention shares KV-cache prefixes across requests via a radix tree rather than exact-match caching.
- **TRL** → "I have hands on post-training." Disproportionately valuable to you precisely because post-training literacy is a named rejection cause for backend engineers, and a merged PR in a trainer library is unfakeable evidence you read the code.
- **Outlines** → "I understand constrained decoding at the logits level." **📄 Paper:** Willard & Louf (2023) — compiling a regex or grammar into a finite-state machine so that at each decoding step you mask the logits of tokens that cannot continue a valid string; it replaced generate-and-retry-until-it-parses. This is a beautifully tractable area for a strong Python engineer.
- **LangGraph** → "I build agent systems." Weakest of the four as a *depth* signal to frontier-adjacent interviewers, strongest as a *relevance* signal to AI-product teams building agents.

**3. Is the issue tractable in under 15 hours and is it unclaimed?** Read the last five merged PRs in the area first: how large were they, how many review rounds, what did the maintainer ask for. That tells you the real cost, and the real cost is usually 3× the code time because of review latency.

**The issue types that actually work for a first PR**, roughly in ascending difficulty: a documented behaviour that doesn't match the code (find it yourself while using the library — this is the best source); a missing test for a path you can read; a benchmark or profiling script for something the maintainers have discussed wanting; an error message that is unactionable (my favourite class — small diff, obviously good, requires understanding the surrounding code); support for an argument or config that exists in one code path but not a parallel one.

**⚠ Trap:** opening with a refactor or an architectural proposal. Unsolicited "I restructured your module" PRs from strangers are close-on-sight in most large repos, and rightly so — the maintainer eats the review cost and the merge risk for a change they didn't ask for. **Your first PR should be boring, small, obviously correct, and tested.** Earn the right to propose.

### Walk me through the mechanics of actually landing that first PR. What's the sequence?

The sequence matters because most first-time contributors lose on process, not on code. Treat it as you would a change to a service you don't own — because that's exactly what it is.

**Step 1 — become a user first.** Run the project on something real. Every good first issue I've ever found came from friction I hit myself, and a PR that opens with "I hit this while doing X" is immediately more credible than one that opens with "I found this in the issue tracker." This also inoculates you against the most common rejection: proposing a change that misunderstands the intended usage.

**Step 2 — read CONTRIBUTING.md and the last five merged PRs in your area.** Style, test expectations, DCO/CLA requirements, whether they want an issue opened first. Some projects require sign-off on commits; some require the issue to be discussed before a PR. Violating either wastes a week.

**Step 3 — get the dev environment green before you change anything.** Build from source, run the test suite, confirm it passes. If you cannot get a clean baseline, you cannot tell your breakage from theirs. For the serving engines this step is genuinely the hard part and can consume your first several hours — budget for it.

**Step 4 — comment on the issue before you code.** Two sentences: what you intend to do and how you plan to verify it. This claims the issue socially, and it surfaces a maintainer's "actually we want it done differently" *before* you've spent ten hours.

**Step 5 — the change itself: minimum diff, plus a test that fails before and passes after.** The test is what converts your PR from "trust me" to "verified," and it is the single biggest predictor of a fast merge. No drive-by formatting. No unrelated fixes bundled in — open a second PR.

**Step 6 — write the description like an incident report.** What the problem is (with a reproduction), what you changed, why this approach, how you verified, and what you deliberately did not change. Include the before/after output.

**Step 7 — respond to review within 24 hours, and do not argue about style.** Maintainers are volunteers with limited attention; a contributor who turns around review comments quickly and without friction gets merged and gets invited back. If you disagree on substance, say so once, briefly, with reasoning, and then defer.

**📐 Numbers you must know for planning:** budget roughly **10–20 hours for a first PR** in a large engine — split about 40% environment and codebase orientation, 25% the change, 35% tests and review iterations — plus **1–6 weeks of wall-clock latency** for review in a high-volume repo. That calendar latency is the reason to start this in week one of your search, not week six. It is a background process, and it either lands before your loops or it doesn't.

**⚠ Trap:** measuring your progress in PRs opened. The metric is *merged*, and an abandoned PR that went stale after one round of review comments is mildly negative evidence if anyone looks. Finish one before starting a second.

### What does a merged PR in vLLM actually prove to you as a hiring manager? Be honest about the ceiling.

Honestly? It proves less than candidates hope and more than cynics claim, and knowing exactly which is itself a signal of calibration.

**What it does prove:** that you can get a non-trivial codebase building locally, navigate unfamiliar code without the author's help, write a change that a maintainer with no incentive to be nice found acceptable, and write a test. That bundle is genuinely rare and it is the *actual* job at a company where you'll be dropped into a large unfamiliar system. It also proves you finish things — which, at the median, is the scarcest trait in the candidate pool. And it gives an interviewer a concrete, verifiable, third-party-validated artifact to talk to you about, which is worth more than any self-reported project because you did not control the acceptance criteria.

**What it does not prove:** that you understand the system deeply. A ten-line fix to an error message in an inference engine tells me nothing about whether you understand paged attention. The claim "contributor to vLLM" on a resume, standing alone, is one I discount heavily, because I've seen it attached to a typo fix in a docstring.

Which means the whole value is in **how you frame it**, and the framing rule is: *claim the specific thing, and let the interviewer upgrade you.* Do not write "vLLM contributor." Write what you actually did, in one line, with the mechanism: "Merged a fix to the OpenAI-compatible server's handling of malformed sampling parameters, which previously surfaced as a 500 rather than a 422." That is small, honest, and it invites the good version of the follow-up question rather than the bad one.

**The archetype-dependent read:** AI-infra teams and serving-adjacent roles weight this heavily — it is the closest thing to a work sample they can get. AI-product companies weight it moderately; they care more about whether you can ship a product surface. Big-tech applied loops weight it lowest of all, because their process is designed to be evidence-blind and evaluate you in the room. So if your target list is product-and-big-tech-weighted, **an OSS PR is a nice-to-have that you should time-box, not the centrepiece of your plan.** I would rather you spend forty hours on the eval-harness artifact.

**🗣 Say this in the room:** "It's a small contribution and I'd characterise it that way — a bug in how malformed sampling params were surfaced, about thirty lines with a regression test. What I got out of it was reading the request path end to end, which is what I'd actually want to talk about."

### Tell me about your open-source contribution. (Assume it's thirty lines.)

The pressure here is to inflate, and inflating is the trap: the interviewer is going to ask a follow-up that reveals the true size anyway, and the gap between your framing and the reality is what gets scored. The winning structure is **shrink the claim, expand the context.** Be precise and modest about what you changed, then spend the rest of the answer on what reading that code taught you — because the reading is the part that's actually interesting and the part they're testing.

The four-beat structure:

**1. The precise claim, immediately.** "It's small — about thirty lines plus a test." Saying this yourself is disarming and it buys you credibility for everything after.

**2. The problem, in their terms.** "The server accepted a request with an out-of-range sampling parameter and failed deep in the sampler, so the client got a 500 with a stack trace instead of a validation error, and the request had already been scheduled." Note that this framing shows you understand the request lifecycle — that's the actual content.

**3. What you had to understand to fix it.** This is where the answer earns its keep and where you get to demonstrate real depth without claiming credit for it: "To place the validation correctly I had to follow the path from the HTTP handler through request construction into the scheduler, and understand that once a request is admitted it holds KV-cache blocks — so failing late doesn't just produce a bad error, it wastes an admission slot and the blocks aren't released until the exception unwinds."

**4. What you'd do next in that codebase.** Signals you're not a drive-by. "The thing I'd want to look at next is the preemption path — I don't yet have a good picture of what happens to a partially-decoded sequence when it gets evicted."

**⚠ Trap:** "I contributed to vLLM" as a standalone sentence with a pause after it, waiting to be impressed. It reads as an attempt to borrow the project's prestige, and experienced interviewers are specifically allergic to it. The counter-instinct — naming the size before they ask — costs you nothing and buys a great deal.

**🗣 Say this in the room:** "Small one — thirty lines and a test, moving parameter validation before request admission so a bad sampling parameter returns a 422 instead of blowing up in the sampler after the request had already taken a scheduling slot. The valuable part for me was tracing the admission path and seeing that a late failure holds cache blocks until the exception unwinds."

### Your PR has been open for five weeks with no review. What do you do, and what else counts as a contribution?

This is normal, not personal — high-traffic AI repos routinely carry hundreds of open PRs and maintainer attention is the scarcest resource in the ecosystem. The wrong responses are to bump it every three days (which burns goodwill) or to quietly give up (which converts fifteen hours into nothing).

**The escalation ladder I'd use:** at week two, a single polite comment confirming the PR is ready and CI is green, with a one-line summary. At week four, if the project has a public chat channel, mention it once there with a link — this often works because it reaches a maintainer in a context where a 20-second decision is cheap. At week six, look at whether the PR touches an area with an identifiable owner in recent history and tag exactly one person. Beyond that, let it sit; keep it rebased and green so that if it *is* reviewed, it merges. And in the meantime, **the PR is still an artifact you can talk about** — "here's a change I proposed, here's the reasoning, it's awaiting review" is a perfectly good interview answer, and the code exists whether or not it merged.

**The broader point: a merged code PR is not the only contribution that signals.** Several alternatives have better effort-to-signal ratios and shorter latency:

- **A high-quality bug report with a minimal reproduction.** A stranger who arrives with a 15-line repro, the exact versions, the expected-vs-actual, and a bisected commit is doing skilled work that maintainers value enormously — and it is visible, permanent, and attributable to you. This is genuinely underrated and it is the one I'd start with.
- **Reproducing and triaging someone else's issue.** "I can reproduce this on 0.x with the following minimal case; it does not occur on 0.y; here's the diff between those in that file." That comment can be more useful than a PR.
- **A benchmark the project doesn't have.** If you build the benchmark artifact from earlier in this section against a project's feature, offer it. Sometimes it gets upstreamed; even when it doesn't, you now have a public technical exchange with the maintainers of a well-known project.
- **Documentation for something you had to figure out the hard way.** Low prestige, high merge rate, and a real contribution — the docs gap you hit is a gap everyone hits.

**⚠ Trap:** treating docs PRs as embarrassing and hiding them. A merged docs PR that fixes a genuinely wrong page is a real contribution and reads fine when described accurately. What reads badly is a docs typo fix described as if it were an engineering contribution. The rule, again: **claim exactly what you did.**

### You've written a 1,633-question technical guide. I'm going to be blunt — why is that engineering evidence and not content marketing?

Because the questions are the input, not the output. The artifact is a **content system**: an ingestion and generation pipeline, a structured corpus, a retrieval layer, a deployed application, and a set of quality controls. If I describe it as "I wrote a big guide," it is content. If I describe it as "I built a pipeline that produces, validates, indexes and serves a 1,633-item structured corpus," it is a system with a design worth interrogating — and that is the accurate description.

The reframe has to be specific, so here is the frame I would use, and every claim in it is a claim you should be able to defend for ten minutes:

**It's a corpus with a schema.** 1,633 items is not a blog; it's a dataset. There is a unit of content with fields (question, answer, section, difficulty, prerequisites, cross-references), which means there is a schema, which means there is validation, which means there are items that failed validation and a policy for what happens to them. Talk about the schema.

**It's a generation pipeline with quality gates, and gates are the interesting part.** How did you prevent duplicate questions across 1,633 items — exact match, normalised match, or embedding-similarity dedupe with a threshold you chose and can defend? How did you enforce section-level coverage against a spec? What did you do about answers that were fluent and wrong? Every one of these is a real engineering decision and each one maps to something an AI-product team does daily. **Duplicate detection at 1,633 items via embeddings is exactly the near-duplicate problem in a RAG corpus** — same algorithm, same threshold-tuning problem, same false-positive/false-negative tradeoff.

**It's a retrieval problem with a real corpus.** Chunking a long-form Q&A corpus is non-trivial: the natural unit is the Q&A pair, which is often too long for one chunk and semantically indivisible. What did you do? That is a genuine retrieval design question and you have empirical experience of it.

**It's deployed, and deployment has a cost line.** Build times, index size, hosting cost, page-load latency, what happens on a rebuild.

**🗣 Say this in the room:** "I'd rather you look at it as a pipeline than as a document. The interesting parts are the quality gates — dedupe across 1,633 items using embedding similarity with a threshold I had to tune because exact matching missed paraphrases, coverage validation against a section spec, and the chunking decision for retrieval, where the natural semantic unit is a whole Q&A pair that's often longer than my chunk budget. Those are the same three problems as any production RAG corpus, just with a corpus I own."

**⚠ Trap:** leading with the number. "1,633 questions" as an opening line invites exactly the skeptical read, because scale without process reads as generated bulk. Lead with the pipeline and the quality gates; let the number arrive as a consequence.

### Did you just generate all of that with an LLM?

This question is coming, it is fair, and how you handle it is worth more than the artifact itself. The instinct to get defensive is the one thing that will actually hurt you. The correct posture is: **yes, LLMs were part of the toolchain, here is exactly what they did and what they could not do, and here is the verification system I built precisely because they cannot be trusted at this volume.**

Structure the answer as a division of labour, because that's the truth and it's also the most impressive available answer:

**What was mine and could not be delegated:** the curriculum design — deciding what 1,633 questions should cover, in what order, at what depth, and against which target roles. That is the hard part and it comes from having sat on both sides of technical interviews. Also mine: the quality bar, the schema, the pipeline, the dedupe thresholds, the validation rules, and every judgment about whether an answer was actually correct.

**What the model did:** drafting at volume, under a specification, with structure I defined. Exactly as it would in any production content or code pipeline.

**What the model demonstrably could not do — and this is the killer detail:** name the specific failure modes you caught. Duplicates that were textually different but semantically identical, which is why the dedupe is embedding-based rather than string-based. Answers that were fluent and confidently wrong on version-specific behaviour, which is why anything version-dependent got flagged and verified. Drift in depth across sections, which is why there's a coverage/length check. **A candidate who can enumerate the specific ways their generation pipeline failed has unambiguously operated it**, because those failure modes are not guessable from the outside.

**Then close the loop:** "the reason I can answer that question with specifics is the same reason I'd be useful on your team — I've had to build the verification layer for a generated corpus at a scale where reading everything twice wasn't possible. That's the same problem as shipping any LLM feature."

**⚠ Trap:** claiming you wrote all 1,633 answers by hand. It is not credible at that volume, it is not true, and if the interviewer probes and finds the seam, everything else you've said is retroactively suspect. **Overclaiming authorship is a strictly worse strategy than owning the pipeline**, because the pipeline is the more impressive claim to this audience anyway.

**🎯 Archetype note:** at Cursor specifically, and increasingly at most AI-product companies, "I used AI tools heavily and here is my verification system" is a *positive* signal — several of these companies treat fluency with AI-assisted workflows as a hiring criterion. At a more conservative big-tech panel, lead with the design and verification and mention the tooling as tooling. The facts are the same; the emphasis moves.

### Walk me through the architecture of the site you built for it — treat this as a system design question.

Good, because that's what it is, and answering it as a system design question rather than a project tour is the whole point. **📅 Volatile:** framework version specifics (Next.js 16 and its rendering defaults) move quickly; describe the mechanism and pin the version you actually shipped.

**Requirements first, out loud, because that's what seniority sounds like.** Corpus of ~1,633 long-form Q&A items, roughly a million words. Read-heavy to the point of read-only — writes happen at build time, never at request time. Traffic is low and bursty (a link gets shared). Two access patterns matter: navigational (reader lands on section 34 and reads linearly) and lookup (reader wants "how do I size a KV cache" and doesn't know which section it's in). Budget: it should cost single-digit dollars a month. Those requirements pre-determine most of the design, and saying so is the answer.

**Ingestion.** Source of truth is structured files, not a CMS — each section a markdown file with front matter, validated against a schema at build time so a malformed item fails the build rather than shipping. This is the same instinct as failing a deploy on a migration error. The pipeline's stages: parse → validate schema → dedupe check → build the cross-reference graph (a `see §5` that points at a nonexistent section should break CI) → emit the render corpus and the search index.

**Rendering.** With a read-only corpus, everything is static generation. A million words across ~87 section pages is entirely pre-renderable, which converts the whole serving problem into a CDN problem: no server, no database at request time, no cold-start, no per-request model call. The number to have ready is build time — if generating 87 pages plus the index takes N minutes, say N and say what dominates it.

**Search and retrieval — the part they'll actually push on.** The honest engineering answer is that there are two tiers, and the interesting content is the tradeoff between them. Tier one is a client-side lexical index (an inverted index shipped to the browser) — zero infrastructure, instant, works offline, and its size is the constraint you must quantify: an index over a million words, even after stemming and stop-word removal, is measured in megabytes, and shipping 5 MB of index to a phone is a real cost you have to trade against. Tier two is semantic search, which requires embeddings, a vector index, and a request-time service — better for the "I don't know the word for this" query, but it reintroduces infrastructure and a per-query cost. **The senior answer states which one you shipped, why, and the specific query class where it fails** — lexical search cannot find "how do I size a KV cache" if the text says "compute the memory footprint of cached keys and values," and that failure class is exactly why hybrid retrieval exists.

**Chunking, if you did semantic.** The natural unit is the Q&A pair, but pairs run long. Options: embed the question only (cheap, precise for lookup, misses answer-body content), embed the whole pair (dilutes the vector), or chunk the answer with the question prepended to every chunk as context (my default, and worth defending — it keeps each chunk self-describing at the cost of duplicating the question text).

**Deployment and scale.** Static output on a CDN. The scale question is trivially answered and you should say so confidently: static assets on a CDN don't have a QPS problem, they have a bandwidth bill. **💰 Math:** a page with 12 KB of gzipped HTML plus shared assets, at 50,000 pageviews in a launch week, is on the order of 50,000 × ~150 KB ≈ 7.5 GB of transfer — free-to-a-few-dollars on any modern CDN tier. Contrast that with the same content behind a request-time model call at even $0.002 a query: 50,000 queries = **$100**, for a worse experience. That comparison — and the fact you made it — is the answer to "did you think about cost."

**⚠ Trap:** over-architecting the answer to sound impressive. If someone describes a static content site with a Kubernetes cluster and a vector database, I mark them down, because the requirements said read-only and low-traffic. **Matching architecture to requirements, and saying explicitly what you did *not* build and why, is the senior move.**

### You said the post-mortem is the most valuable thing I can write. Show me its structure — and how do I write about a failure without it costing me?

Start from why it converts: a post-mortem is the only format where the *reader learns something they could not have derived themselves*, because it contains an outcome from the real world rather than a restatement of documentation. It is also the format that most closely resembles what senior engineers do at work, so writing a good one is a work sample. And the fear that it makes you look bad has the causality backwards — **admitting a failure with a number attached is the strongest available signal of seniority**, because juniors describe systems as working and seniors describe them as having a failure profile.

The structure, which is deliberately the same as an internal incident review because that familiarity is part of the appeal:

**1. The system in three sentences and one diagram.** What it did, for whom, at what scale. Numbers: users, requests, tokens, cost.
**2. What you expected.** State the prediction you actually held, including how confident you were. This is the beat everyone skips and it is what makes the rest interesting — a surprise is only a surprise relative to a stated prior.
**3. What happened, with a timeline.** When it started, how you found out, what the signal was. "We found out from a customer" is a valid and honest answer, and pairing it with "which is why we now alert on abstention rate" turns it into a lesson.
**4. The investigation, including the wrong hypothesis.** The false lead is the most instructive part of the whole document and the most commonly cut.
**5. The root cause, mechanistically.** Not "the model hallucinated." *Why* — the retriever returned three low-scoring chunks and the prompt had no abstention instruction, so the model filled the gap; the aggregate eval hid it because unanswerable questions were 4% of the set.
**6. The fix, and what it cost.** Latency added, dollars added, quality traded.
**7. What you'd do differently, and what remains unsolved.** Open problems are credibility, not weakness.

**On the confidentiality question, which is real and which you must handle before you publish:** the safe transformation is to keep the *engineering* and drop the *identifying*. Remove the employer, the product name, customer names, and anything that could identify a customer. Convert absolute business figures to ratios or rounded orders of magnitude — "roughly two million calls a month" rather than an exact figure tied to a public company's disclosed metrics. Never publish internal metrics, incident timelines tied to a public outage, security details, or anything you learned from a customer's data. If your employment agreement or your employer's policy requires review, get it — the downside of publishing something you shouldn't have is career-scale, and the upside of any single post is not. Written as "a retrieval-heavy assistant I worked on," a post-mortem loses almost none of its technical value.

**⚠ Trap:** the post-mortem that blames a vendor or a model. "The model was unreliable" is a non-finding, it teaches nothing, and to a hiring manager it reads as an engineer who thinks nondeterminism is someone else's problem. **The interesting root cause is always in your system's design** — the missing abstention path, the eval slice that was too small to detect the regression, the retrieval threshold you never tuned. Own the layer you control.

**🗣 Say this in the room:** "The most useful thing I've written is a post-mortem of a retrieval system I worked on — what I expected, what actually happened, the hypothesis I chased for two days that was wrong, and the root cause, which was that unanswerable questions were 4% of my eval set so a real regression was invisible in the aggregate. It's the piece I'd want you to read."

### How do you get more than one artifact out of a single body of work?

Because building things is expensive and distributing them is cheap, and most engineers do the expensive part once and the cheap part zero times. The operating principle: **one body of work should produce four assets on four different distribution channels**, and you should plan those four before you start rather than discovering them afterwards.

Take the eval-harness project from earlier and run it through:

**Asset 1 — the repo.** Discoverable by anyone who looks you up, linkable from a resume, forkable. Channel: GitHub, your resume, your outreach emails.

**Asset 2 — the write-up.** The narrative with the surprising finding, the methodology, and the limitations. Channel: your own site, wherever practitioners in this field gather. This is the asset that can travel without you.

**Asset 3 — the resume bullet.** One line, claim + metric + mechanism, with the link. Channel: every application, forever.

**Asset 4 — the interview answer.** A rehearsed 3-minute walkthrough that hits eval, cost, latency and failure mode. Channel: every loop. **This is the highest-value asset of the four and the one people never explicitly build.** Write it down. Time yourself delivering it. Three minutes.

And frequently a fifth: **the outreach hook.** "I benchmarked four retrieval configurations on a 200-question set and the unanswerable slice behaved the opposite of what I expected — I wrote it up here, thought it might be relevant to what your team's doing." That is a cold email with a reason to exist, which is the only kind that works.

The same multiplication applies to your existing work. The guide is a corpus, a pipeline, a search system, a deployment, and a writing sample — five different conversations from one project, and which one you lead with should depend on who you're talking to. To an infra-leaning interviewer, it's a build pipeline with quality gates. To an AI-product team, it's a retrieval system with a real corpus and a real chunking tradeoff. To a hiring manager wondering if you can write, it's the writing sample.

**⚠ Trap:** building artifact number two before you have extracted all five assets from artifact number one. Engineers do this because building is fun and distributing feels like self-promotion. The arithmetic is brutal: forty hours to build, maybe four hours to extract four more assets. Skipping the four hours throws away most of the return on the forty.

**🏋 Drill:** take the strongest thing you have already built. Set a 45-minute timer and produce: the resume bullet (one line, with a number), the 3-minute spoken walkthrough (written out, then delivered to a recording), and the outreach paragraph (four sentences, with a specific reason it's relevant to one named company). Pass criterion: the spoken version lands between 2:30 and 3:15 and contains at least two numbers you did not have to look up.
