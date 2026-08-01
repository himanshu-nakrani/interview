### Take me through the OpenAI loop. What should I expect and where do people fall out?

Treat OpenAI as a federation, not a company — the single most important fact about their process is extreme team-to-team variance. Research, Applied, ChatGPT, and the API platform run materially different loops with different bars and different question distributions. Interviewing for "OpenAI" is not a thing; interviewing for a named team is. Ask the recruiter which org and which team before you prepare anything.

The reported common shape is around six stages: recruiter screen, a technical screen, one or more deep technical rounds, a **paid work trial** — commonly described as a roughly 48-hour project under NDA — and then team-fit and hiring-manager conversations, followed by an offer review. The work trial is the distinguishing element and it is the thing to plan your calendar around: it is compensated, it is real work, it is NDA'd, and it is graded like a job sample rather than an exam. **📅 Volatile:** stage count, trial length, and compensation for the trial vary by org and change; confirm with your recruiter.

Where people fall out. On the Applied and ChatGPT side, the failure is shallow product judgment — building the thing asked for without arguing about whether it is the right thing, and without an evaluation story. On the API-platform side, the failure is reliability thinking: these are teams operating a service under brutal load, and they want the backend engineer's instincts about rate limits, backpressure, idempotency under nondeterminism, and multi-tenant fairness. On the Research side, the failure is depth — you will be asked to go three levels down on something and there is no bluffing.

**🗣 Say this in the room** (to the recruiter, first call): "Which org and team is this for? The preparation for Applied versus API platform versus Research is quite different and I'd like to aim at the right one."

**⚠ Trap:** Treating the paid work trial as a take-home to be finished. It is a simulation of working there, and communication during it is graded — asking a clarifying question on day one, stating your scope decision explicitly, and shipping something narrower but complete with a documented eval beats shipping something broad and unmeasured. The 48-hour clock is a scoping test disguised as a time limit.

### And Anthropic's process — what's the reported shape and what's actually filtering?

The reported shape is compact: an initial screen, an automated or live coding assessment (a CodeSignal-family screen is commonly reported), a **customer-conversation simulation** for the applied and forward-deployed roles, and then an onsite — with a **values round that is widely reported as the primary filter**. Median time from first contact to decision is reported around 19 days, which is fast by big-company standards and means you should not start this process while your calendar is full. **📅 Volatile:** verify the current stage list; Anthropic has iterated on it.

Two things distinguish this loop from every other on your list.

First, the values round is not a formality and it is not about reciting published principles. It is a genuine evaluation of whether your judgment is sound when caution and velocity conflict, and strong technical candidates are reported to fail there. Prepare it like a technical round: have three real stories where you traded off shipping speed against a correctness, safety, or user-harm concern, with what you actually did, who you escalated to, and what you'd change.

Second, the coding screen is real and automated. This is one of the named exceptions to "senior AI loops have no algorithmic round." Do not walk in cold on the assumption that AI roles skipped that phase.

**🗣 Say this in the room** (values round): "I don't think 'ship it' and 'be careful' are actually opposed most of the time — usually the tension is that we haven't defined what 'ready' means. So my first move is to make the standard explicit and measurable, and then the disagreement becomes 'is 92% good enough for this surface,' which is a decision an owner can make, instead of a vibes argument."

**⚠ Trap:** Performing safety. Candidates who have read the company's public writing sometimes arrive with an ideology instead of a disposition, and it reads as costume. The credible version is specific, mundane, and includes a case where you *shipped* something imperfect and documented the limitation — because someone who claims they never do that has never operated anything.

### Anthropic's customer-conversation simulation round — how do you even prepare for that?

You prepare for it the way you'd prepare for a design round: with a repeatable structure, because the failure mode is improvisation.

The round is a role-play. An interviewer plays a customer — often a technically-literate but non-expert stakeholder with a vague, over-scoped, or subtly wrong request — and you have to run the conversation. What is graded: do you diagnose before prescribing, do you ask questions that change the design rather than questions that fill silence, do you push back on a bad idea without being adversarial, do you set expectations honestly about what models can and cannot do, and can you translate a technical constraint into business terms.

The structure I use has four beats and it is worth rehearsing until it is automatic.

**Beat one — understand the job to be done.** Not "what do you want built," but "walk me through how this works today, step by step, and where it hurts." You are looking for the workflow, the volume, and the current cost of failure.

**Beat two — define success in numbers, out loud.** "If this works, what changes? Is it hours saved per case, resolution rate, error rate?" A customer who cannot answer this is telling you something important, and naming that gently is a strong move.

**Beat three — surface the constraints they haven't mentioned.** Data location and sensitivity, who is allowed to see what, latency tolerance, whether a wrong answer is embarrassing or catastrophic, existing systems you must integrate with, and who will maintain this after you leave.

**Beat four — propose a narrow first slice with an explicit measurement, and name what you are *not* doing.** "I'd start with the top three intents, which look like 60% of volume, run it in suggest-only mode for two weeks, and measure agreement against your current handling. I would not automate refunds in phase one."

**⚠ Trap:** Demonstrating knowledge instead of listening. The instinct under interview pressure is to prove you know how to build it, so you start designing at minute three. In this round, designing early is the failure. The interviewer is specifically watching whether you can sit in ambiguity long enough to find the real problem — and customers routinely ask for the wrong thing.

**🗣 Say this in the room:** "Before I suggest an approach, can you walk me through how someone does this today, end to end? I want to understand where the time actually goes, because in my experience the bottleneck is usually one step upstream of where it's felt."

### DeepMind's loop is the one I know least about. What's in it?

DeepMind runs the most research-weighted loop on your target list, and it is the one where a backend engineer is furthest from the mode. The reported components: an **ML coding round** (implement something in PyTorch, not solve a puzzle), a **research deep dive** on your own work, a **distributed-training-systems design** round, an **evaluation-infrastructure design** round, and a Google-style **Googleyness / general cognitive** round, with a hiring-committee decision on top — which is why the calendar runs long. **📅 Volatile:** confirm the current component list; Google-family processes vary by ladder and location.

Two of those rounds are unusual enough to name.

*Distributed-training-systems design* is a real systems interview with an unfamiliar bottleneck: shard a model across N devices, choose among data / tensor / pipeline / expert parallelism, reason about the communication volume each strategy implies, handle checkpointing, and survive a node failure 40 hours into a run. Your distributed-systems instincts transfer *very* well here — this is consistent hashing and failure domains and coordination overhead in a new costume — but you must have the vocabulary and the memory arithmetic to use them.

*Evaluation-infrastructure design* is the round nobody expects and it is close to a gift for someone with your background. It is: design the system that runs thousands of evaluations reproducibly, versions the datasets, caches results, handles nondeterminism, detects contamination, and gives researchers a fast enough loop to iterate. That is a data-pipeline and reproducibility problem — dataset versioning, idempotent job execution, caching keyed on (model, prompt, dataset) hashes, statistical significance on the reported deltas. If you get this round, lead with the fact that a 2% eval delta on 200 examples is not a result and show how your system would compute the confidence interval.

**⚠ Trap:** Treating the Googleyness round as filler. In Google-family processes the hiring committee sees written feedback from every round, and a lukewarm behavioral packet against strong technical scores is a common source of "hire, but not at this level" outcomes.

### Cursor's process is unusual. Walk me through it and tell me the hidden rubric.

Cursor's reported shape is two phone screens followed by an **8-hour paid onsite project** — a full working day, compensated, building something real with their stack and their people. **📅 Volatile:** confirm current length and structure.

The hidden rubric is the thing to internalize: **do you actually use AI coding tools, and are you good at it?** This is a company whose product is an AI code editor. A candidate who arrives having barely used agentic coding tools, or who uses them badly — accepting large diffs unread, prompting vaguely, failing to notice a subtly wrong refactor — fails on the dimension the company cares most about, no matter how clean their algorithms are. The reciprocal is also true and is the actionable part: come with opinions. Where does agentic editing break down? What's your workflow for reviewing a 400-line AI-generated diff? When do you turn the assistant off? Which failure modes have you learned to anticipate?

An 8-hour project is also a stamina and scoping test. The winning pattern is the same as any work trial: cut scope early and explicitly, get something working end to end in the first third, then deepen. Communicate a plan at hour one and a status at the midpoint — you are being evaluated as a colleague for a day, and colleagues who go silent for eight hours are not pleasant to work with.

**🗣 Say this in the room:** "I use agentic editing for scaffolding, test generation, and mechanical refactors, and I turn it off for anything with concurrency or subtle invariants — those are exactly where the diff looks right and isn't. My review rule is that I never accept a diff larger than I'd accept from a junior without reading every line."

**💰 Math:** Budget the day honestly. Eight hours at, say, a $250k salary is roughly $250,000 / 2,080 ≈ $120/hour of your time, so a $960 opportunity cost — and these trials are typically paid at or near a comparable rate, which is why they are ethical. What is *not* budgeted is the recovery day. Do not schedule two 8-hour trials in the same week; the second one will be visibly worse.

### Perplexity's loop — what am I walking into?

The most algorithmically demanding loop among the AI-product companies, and the one where "senior AI loops don't have coding rounds" will get you eliminated. The reported shape: a **machine-coding round at LeetCode-Hard difficulty**, a **retrieval and search system design** round, and a **founder round**, with a reported median around 11 days end to end. **📅 Volatile:** confirm current stages.

Prepare it in three separate tracks.

*The coding track* is genuine hard-difficulty algorithmic work, live, under time pressure, likely without AI assistance. This needs weeks of actual practice, not a weekend. If Perplexity is on your list, you must decide early whether you are paying that tax, because it does not amortize across your other target companies.

*The search-design track* is where your backend depth is a genuine advantage and where you should aim to be memorable. This is not "explain RAG." It is a real search system: query understanding and rewriting, candidate generation from an inverted index and a vector index, fusion of the two (and you should be able to say why reciprocal-rank fusion is the pragmatic default over score normalization when the two scorers have incomparable scales), reranking with a cross-encoder under a latency budget, freshness and re-crawl, index update strategy without a read-side stall, caching at the query and the passage level, and citation grounding so the generated answer is attributable. Bring numbers: an index of N documents, a p95 budget of X ms decomposed across retrieval, rerank and first token.

*The founder round* is judgment and intensity: why this problem, what would you build, what do you think is wrong with the product today. Have a real, specific critique ready — a generic compliment is a wasted round.

**⚠ Trap:** Designing a search system with a single vector index and calling it done. At web scale, lexical retrieval is not a legacy component you have replaced; it is the recall backbone for rare entities, exact identifiers, and fresh content where embeddings are weakest. Saying "I'd use hybrid retrieval" without being able to explain *which failures each half catches* is a shallow answer.

### Scale AI runs a cost-constrained system design round. What does that mean in practice, and what's the FDE variant?

The reported shape is a HackerRank-style automated assessment, then a **cost-constrained system design** round, then behavioral — with a distinct Forward-Deployed variant that trades some system design for customer scoping. **📅 Volatile:** confirm current process.

"Cost-constrained" means the interviewer hands you a budget and the design is only correct if it fits. This is a genuinely different interview and it is one you can practice mechanically, because it is arithmetic under a constraint. The pattern: they describe a workload (say, classify and extract fields from 2 million documents per month, average 5 pages each), give you a budget, and grade whether your design lands under it with the math shown.

**💰 Math — a worked version of exactly this round.** 2M documents/month × 5 pages, and take a page as roughly 700 tokens, so 3,500 input tokens per document plus a 1,200-token instruction prompt and 300 tokens of structured output. Naive frontier-model design at $3/Mtok in and $15/Mtok out: input (3,500 + 1,200) = 4,700 tokens → 4,700 × $3/1e6 = $0.0141; output 300 × $15/1e6 = $0.0045; total $0.0186/doc × 2,000,000 = **$37,200/month**. Now optimize in the order that actually pays: (1) prefix-cache the 1,200-token instruction — at a 90% cached-input discount that portion drops from $0.0036 to $0.00036, saving ~$6,500/month; (2) batch-tier the workload since documents are not latency-sensitive — commonly around 50% off, roughly halving what remains; (3) route by difficulty — send the 80% of documents that a small model handles at equal accuracy (which you prove with an eval, not an assertion) to a model roughly 10–20× cheaper. Stacking those plausibly lands you near $6k–$9k/month, a 4–6× reduction, and — the part that gets you the hire — you state which optimization you'd verify first and what eval would tell you routing is safe. **📅 Volatile:** all per-token prices and discount rates here are illustrative and move constantly; re-derive with current numbers.

The FDE variant substitutes a scoping conversation: an interviewer plays a customer with a vague problem and you must extract requirements, name what you would *not* build, and define success in a measurable way before writing anything.

**🗣 Say this in the room:** "Give me a moment to price the naive design first — I'd rather know how far over budget we are before I start optimizing, because the answer determines whether this is a caching problem or an architecture problem."

### Databricks fuses a GenAI-frameworks round with ML-infra design. How do I prepare for that combination?

Databricks is the cleanest example of the enterprise / data-platform archetype, and the fusion is not accidental: their customers do AI *on top of a lakehouse*, so they hire people who can hold both layers at once. The reported loop includes a **GenAI-frameworks round fused with ML-infrastructure design**, plus — and this one is procedurally unusual — **mandatory manager references before an offer**. **📅 Volatile:** confirm; reference requirements in particular vary by org and region.

Prepare three things.

*Frameworks, with opinions.* You will be asked about the orchestration layer — retrieval frameworks, agent frameworks, evaluation tooling, vector stores, MLflow-style experiment and model tracking. The bar is not API recall; it is judgment about when a framework earns its abstraction cost and when you should write the 200 lines yourself. Have a position: mine is that framework value is highest for evaluation harnesses and tracing, and lowest for the retrieval and prompt-assembly path, where the abstraction usually costs you more debugging time than it saves you typing.

*Data-platform grounding.* This is where you are already strong and should say so. Governance, lineage, permissions that must flow through to retrieval (a document the user cannot read must not appear in their context — and enforcing that at query time rather than post-filtering is a real design decision), incremental re-embedding when source rows change, and the batch-versus-streaming decision for an embedding pipeline. Bring your Postgres and Kafka instincts explicitly; that is the bridge.

*The reference check.* Plan for it *now*. Mandatory pre-offer manager references mean you cannot run this process entirely quietly, and you need at least one former manager who has agreed in advance. Line that up before you apply, not after you get an offer conversation.

**⚠ Trap:** Trashing frameworks to sound senior. "LangChain is bloated" is a junior-sounding opinion delivered as a senior one. The credible version names the boundary: what you would use it for, what you would not, and the specific debugging experience that formed the view.

### Nvidia and the AI-infra loops — what do those rounds actually test?

They test whether you can reason about GPUs as a scheduled, contended, expensive resource, and whether you have operated one under load rather than read about it. The recurring round topics across Nvidia and the serving companies (Together, Fireworks, Baseten, Modal, and lab infra teams) are: **GPU scheduling**, **batching systems**, **deployment platforms**, and **preemption and checkpointing**.

Concretely, what those mean:

*GPU scheduling* — multiple tenants, heterogeneous models, GPUs that cannot be oversubscribed the way CPU can. Expect: how do you pack models onto devices, how do you handle a request for a model that is not resident (cold start = loading tens of GB of weights over PCIe or from network storage), how do you avoid head-of-line blocking when one tenant sends 100k-token prompts, and what is your fairness policy.

*Batching systems* — the central object of modern LLM serving. You need to be able to explain continuous batching (requests join and leave the batch at token granularity rather than the batch running to completion), why prefill and decode have completely different resource profiles (prefill is compute-bound and parallel over the sequence; decode is memory-bandwidth-bound and serial), and why chunked prefill exists — to stop a long prompt's prefill from stalling every in-flight decode.

*Preemption and checkpointing* — this is your distributed-systems background almost verbatim. A training run across 512 GPUs where a node dies; a serving instance that must evict a request's KV cache under memory pressure and either recompute or swap it. Expect questions about checkpoint frequency versus expected failure rate.

**💰 Math — checkpoint interval, the classic version of this question.** If a node fails on average every 6 hours across the cluster and a checkpoint costs 4 minutes of wall clock, then checkpointing every T minutes gives expected lost work of T/2 on failure. Total overhead per hour ≈ (60/T) × 4 minutes of checkpointing, plus (1/6) × (T/2) minutes of lost work per hour. Minimizing: at T = 60 min you pay 4 min/hr of checkpointing and 5 min/hr expected loss = 9 min/hr (15%); at T = 30 you pay 8 + 2.5 = 10.5; at T = 90 you pay 2.67 + 7.5 = 10.2. So the optimum sits near an hour and the cost curve is flat around it — which is the real insight to state out loud: the exact interval barely matters, but being 10× off in either direction costs you 20–30% of your cluster.

**⚠ Trap:** Answering an infra round with parameter counts. "It's a 70B model" tells the interviewer nothing about whether it fits; KV cache per token, batch size, context length, and quantization decide that, and quoting parameter count when asked about serving cost is one of the loudest tells that someone has not served a model.

### Big-tech applied AI — Meta, Google, Amazon, Microsoft. What's different about those loops?

The dominant fact is that they bolted an AI round onto an existing, highly-structured process rather than inventing a new one. That has three consequences you should plan around.

**The classical rounds survive.** You will still get a coding round, still get a general system design round, and still get a structured behavioral round. The AI-specific content is usually one or two rounds, not the whole loop. So preparation splits: roughly half your time goes to the loop machinery you already know, and you should not neglect it on the theory that this is an "AI role."

**The behavioral round is scored against a rubric.** Amazon is the extreme case — Leadership Principles evaluated explicitly, with a bar raiser who has veto and is not on the hiring team, and a strong expectation of STAR-structured stories with metrics. Google routes everything through a hiring committee reading written packets, which means how legibly your interviewers can write you up matters as much as how well you did. Meta runs a distinct ML system design round for AI-adjacent roles that expects end-to-end thinking including data and metrics.

**Microsoft's notable variation** is running both an **AI-assisted coding round and a raw coding round** — the explicit position that they want to see your output with tools and your fundamentals without them. **📅 Volatile:** confirm current formats; all four companies have been actively revising.

The archetype-specific answer style: these panels reward **reliability, cost discipline, and cross-team thinking** over capability enthusiasm. A frontier-lab answer about pushing model capability lands flat; an answer about how the feature degrades gracefully when the model provider has an outage, what the per-request cost is at their scale, and how you'd roll it out behind a flag with an online metric lands well. Scale is real at these companies, so the arithmetic is worth doing: a feature at 50M requests/day costing $0.01 each is $500k/day — nobody ships that, and the design conversation is entirely about the routing and caching that make it $0.0005.

**🗣 Say this in the room:** "At your volume the design question is really a unit-economics question — so I'd start by pricing the naive version per request, multiply by daily volume, and let that decide how much of the traffic can be served by a cached or small-model path."

### What about the high-comp companies that aren't AI-native — Stripe, Ramp, Snowflake? What do their AI loops look like?

These are the archetype I'd argue is most undervalued for someone with your background, because the loop is mostly a loop you already pass. They are excellent engineering organizations with a serious existing product, and the AI work is a feature layer on top of a business that already works. The consequence: the bar on core engineering is *high* and the bar on exotic AI knowledge is *moderate*, which is exactly the inverse of a frontier lab.

Expect a loop that looks like: a strong practical coding round, a classical system design round on their real domain (payments, ledgers, collaborative documents, warehouses), one AI-focused round (build or design an LLM feature, usually with retrieval and evaluation), and a behavioral round with real weight on collaboration and product judgment.

What differentiates candidates here is not model knowledge, it is **applying senior engineering discipline to a nondeterministic dependency**. Concretely: how do you version and roll back a prompt like code; how do you flag-gate a model upgrade when the provider deprecates your model on 60 days' notice; how do you handle a provider outage without taking the product down; how do you keep a per-tenant cost ceiling so one enterprise customer's usage does not eat the margin; how do you prevent a document the user is not permitted to read from entering their context window; how do you audit an AI-generated action in a system where a wrong ledger entry is a real incident.

Those are the questions a Stripe or Ramp panel actually cares about, and they are *your* questions in new clothing — feature flags, rollback, multi-tenant quotas, authorization at the data layer, audit logs.

**🗣 Say this in the room:** "I treat prompts and model versions as deployed artifacts: versioned in the repo, rolled out behind a flag, gated by an offline eval, and monitored with an online proxy metric so we can roll back in one deploy. The failure mode I'm defending against is a silent quality regression, which is invisible to every alert we already have."

**⚠ Trap:** Under-preparing the classical rounds because the role says AI. I have seen candidates study transformers for six weeks and then fail a Stripe loop on a plain distributed-systems design question about idempotent payment retries. At non-AI-native companies the classical bar is the *primary* bar.

### Sierra, Harvey, Glean — the vertical AI-product companies. Anything specific?

Yes: domain grounding and deployment reality carry more weight than at a horizontal product company, and the loops reflect that.

These companies sell into a specific workflow — customer service resolution at Sierra, legal work at Harvey, enterprise search and assistants at Glean — and the thing that separates a demo from a business in those verticals is whether the system is *right often enough to be trusted by a professional whose job depends on it*. So expect: an evaluation-heavy conversation (what does "correct" mean for a legal research answer, and who decides), a permissions and data-governance conversation (Glean's entire technical moat is enterprise permission-aware retrieval across dozens of source systems — a document must be invisible in retrieval to a user who cannot open it in the source system, enforced at query time and re-checked at render time), and an agent-reliability conversation (Sierra's product is an agent that takes real actions on real accounts; the interesting questions are about action confirmation, reversibility, escalation to a human, and containment when the agent is wrong).

Expect also a customer-facing or scoping component, since these companies deploy into enterprises and most engineering roles there touch customer reality.

The preparation that pays: be able to talk about **the cost of a wrong answer** in the vertical. In consumer chat a wrong answer is annoying; in legal it is malpractice exposure and in customer service it is a refund issued against policy. That asymmetry should drive your entire design — where you put a human in the loop, what you refuse to automate, what confidence threshold triggers escalation, and how you measure the escalation rate as a first-class product metric rather than a failure.

**🗣 Say this in the room:** "I'd design around the cost asymmetry: for actions that are cheap to reverse I'd let the agent act and log it; for anything irreversible or externally-visible I'd require confirmation, and I'd treat escalation-to-human rate as a headline metric rather than something to minimize at all costs — driving it to zero is how you ship a confident wrong answer."
