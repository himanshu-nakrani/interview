# PART 0 — Orientation, Routing and Getting the Interview

Preparing for the wrong loop is the cheapest and most common way to lose an offer at this compensation level. Nothing here is content; all of it is routing, subtraction, and the machinery of actually getting scheduled.

## Contents

1. [1. Role Taxonomy, Loop Anatomy and the Company-by-Company Loop Map](#1-role-taxonomy-loop-anatomy-and-the-company-by-company-loop-map) — 41 questions
2. [2. The Delta Map: What to Skip, What to Skim, What to Attack](#2-the-delta-map-what-to-skip-what-to-skim-what-to-attack) — 40 questions
3. [3. Getting the Interview: Resume, GitHub, Portfolio and Positioning](#3-getting-the-interview-resume-github-portfolio-and-positioning) — 40 questions
4. [4. Geography, Sponsorship, Remote and the India-to-Frontier-Lab Path](#4-geography-sponsorship-remote-and-the-india-to-frontier-lab-path) — 31 questions
5. [5. The Model and Provider Landscape — the Volatility Sink](#5-the-model-and-provider-landscape-the-volatility-sink) — 45 questions
6. [6. Navigating the Process: Take-Homes, Work Trials, Proctoring and Scope](#6-navigating-the-process-take-homes-work-trials-proctoring-and-scope) — 40 questions


---

## 1. Role Taxonomy, Loop Anatomy and the Company-by-Company Loop Map

*Mastering this proves you can read a job description and correctly predict which five rounds you will sit in and which one carries the hidden weight.*

### Let's start at the top — you applied for an AI Engineer role. In your own words, what does that job actually own day to day?

An AI Engineer owns the system around a model that somebody else trained. That is the whole definition, and every other distinction in this taxonomy falls out of it. You take a frontier model as a fixed, expensive, non-deterministic dependency — the way you already take Postgres or Stripe as a dependency — and you build the retrieval, the prompt and context assembly, the tool layer, the structured-output contract, the caching, the routing, the guardrails, the evaluation harness, and the cost and latency budget that make it behave like a product feature instead of a demo.

Concretely, in a week I would expect to: change how context is assembled for a prompt and measure whether that moved a scored eval; add a tool to an agent loop and discover the model calls it three times when it should call it once; find that p95 time-to-first-token regressed because the prefix cache stopped hitting after someone injected a timestamp into the system prompt; and argue in review about whether a failure is a retrieval problem or a generation problem. The single most defining daily artifact is the eval set — an AI Engineer who cannot tell you their offline eval and their online proxy metric is doing prompt engineering, not engineering.

What the role is essentially never asked to do: design a pretraining data mix, own a distributed training run, write a fused CUDA kernel, or read a loss curve for a 70B model. If those appear in the interview, the title on the JD is lying and you are actually sitting an ML Engineer or Research Engineer loop — which is exactly why this section exists.

**🗣 Say this in the room:** "I think of an AI Engineer as owning everything except the weights. The model is a dependency with a price per token, a latency distribution and a nondeterministic output; my job is the retrieval, context, tools, structured output, evals and cost envelope that turn that into a product SLA."

**⚠ Trap:** Defining the role by the tools ("I use LangChain and pgvector"). Frameworks are a screening question, not the job. Define the role by what you *own*, because the leveling conversation later is entirely about ownership scope.

### What's the difference between an Applied AI Engineer and an ML Engineer, and why does that distinction matter for how you answer?

The dividing line is which side of the weights you live on. An ML Engineer owns a model's lifecycle: features, training data, a training pipeline, hyperparameters, evaluation of the *model*, a deployment artifact, and monitoring for drift and retraining triggers. An Applied AI Engineer owns a product surface built on a model whose weights are frozen and usually external. ML Engineers ship models. Applied AI Engineers ship features that call models.

The reason this matters is not vocabulary, it is the default answer you reach for under pressure. Give an ML Engineer a quality problem and their trained reflex is "collect more labels and retrain." Give an Applied AI Engineer the same problem and the reflex is "look at fifty failures, cluster them, and find out whether this is retrieval, context, instruction ambiguity, tool design, or genuine model capability." In an Applied AI loop the second reflex is the hire signal and the first one is a downgrade. In an ML Engineer loop it is precisely inverted — say "I'd just prompt it better" to someone who owns a recsys retraining pipeline and you will read as a hobbyist.

There is also an honest overlap zone that confuses people: post-training. Fine-tuning, LoRA adapters, preference optimization, distillation into a smaller model. Some Applied AI teams do this — at product companies it is usually a cost-reduction or latency move (distill a frontier model's behavior into an 8B you serve yourself), not a capability move. Owning that pipeline drifts you toward ML Engineer even if your title says otherwise.

**⚠ Trap:** The "reflex fine-tune." I have seen strong backend candidates torpedo an Applied AI loop by answering *every* quality question with fine-tuning. It signals you have never actually measured where a system loses accuracy, because in practice retrieval and context assembly account for the large majority of fixable failures in a RAG or agent system, and they are two orders of magnitude cheaper to iterate on. Fine-tuning is the *last* rung of the escalation ladder, and you should be able to name the preconditions that justify climbing to it.

**🗣 Say this in the room:** "My default is not to touch the weights. I'd run the escalation ladder — prompt and context, then retrieval, then tool design, then structured output, then routing, then distillation, and fine-tuning last, only once I have an eval that says the remaining errors are capability errors and not context errors."

### Research Engineer at a frontier lab — what do they actually do, and how would I know from a JD whether I'm even in range?

A Research Engineer is the person who makes a research idea run at scale. They are not usually the person who has the idea, and they are not a scientist with a publication requirement — they are an extremely strong systems engineer whose substrate happens to be training runs. The daily work is implementing an architecture variant correctly, building the data pipeline that feeds it, getting it to not diverge at 512 GPUs, debugging a loss spike at 3am, writing the evaluation harness that decides whether the variant won, and cutting the iteration time from six hours to forty minutes so the researcher can run twelve ablations instead of two.

The JD tells you immediately. Look for these tokens: *ablation*, *pretraining*, *data mix*, *scaling law*, *FSDP / DeepSpeed / Megatron*, *throughput (MFU)*, *checkpointing*, *reproduce results from papers*. If three or more appear, it is a Research Engineer role regardless of the title, and the loop will contain a research deep dive and a distributed-training-systems design round. If instead you see *RAG*, *agents*, *tool use*, *latency*, *customers*, *evals*, *product*, it is an Applied role.

The honest calibration for a senior backend engineer: Research Engineer is reachable but it is the longest path. It requires PyTorch fluency at the autograd and distributed-primitive level, comfort with the training-numerics failure modes, and enough research literacy to read an arXiv paper and say what it replaces. A candidate with genuine CPython, memory, and concurrency depth converts to this faster than most — the debugging instinct transfers almost perfectly — but "faster" is months, not weeks.

**⚠ Trap:** Applying to Research Engineer because it sounds more prestigious and pays similarly. It does not pay meaningfully more than senior Applied at the top companies, and it has a far higher rejection rate for people coming from backend. Route by conversion probability, not by title vanity.

### Tell me what an AI Infrastructure Engineer owns that an AI Engineer doesn't.

AI Infrastructure owns the layer between the model weights and the API call — the thing an AI Engineer treats as a black box with a price list. Concretely: the serving engine and its scheduler (continuous batching, chunked prefill, preemption policy, admission control), the KV cache as a memory allocator problem, quantization and kernel selection, multi-GPU parallelism strategy (tensor / pipeline / expert parallel), autoscaling on GPU-appropriate signals, and — on the training side — cluster scheduling, gang scheduling, checkpointing, and preemption recovery when a node dies 40 hours into a run.

The mental bridge from your background is exact and you should use it out loud: this is capacity engineering with an unfamiliar bottleneck. You already know how to think about a connection pool, a work queue, admission control, head-of-line blocking, and tail latency under load. The novelty is that the scarce resource is HBM bandwidth and GPU memory rather than CPU and file descriptors, that requests have wildly variable and *unknown-in-advance* service times because output length is not known until you generate it, and that batching *increases* throughput while *degrading* per-token latency in a way you must trade off explicitly.

The tell in a JD: *vLLM / SGLang / TensorRT-LLM*, *throughput per dollar*, *GPU utilization*, *Kubernetes device plugin*, *NCCL*, *preemption*, *checkpoint restore*, *tokens/sec/GPU*. The loop will be a systems loop — GPU scheduling, batching design, a deployment platform question, and something about how you survive a spot-instance eviction mid-run.

**📐 Numbers you must know:** The reason this is a distinct discipline is the price of the resource. An 8×H100 node rents in the rough neighborhood of $2–$3 per GPU-hour on demand, so one node is roughly 8 × $2.50 × 24 = $480/day, about $14,400/month. Driving utilization from 35% to 70% on a 10-node fleet is therefore worth roughly $144k/month × 0.5 ≈ $72k/month. That number is why the role exists and why it pays what it does. **📅 Volatile:** GPU rental prices move fast and vary 2–3× across clouds and commitment terms — verify before your loop.

### What's a Forward-Deployed Engineer, and why has every AI company suddenly started hiring them?

An FDE is an engineer who ships inside the customer's problem instead of inside your repo. They fly (or Zoom) into an enterprise, learn a workflow nobody documented, build the integration and the prompts and the evals against that customer's actual data, and stay until it works. The reason the archetype exploded is structural: foundation models are general but enterprise value is specific, and the last mile — messy PDFs, a 15-year-old case-management system, a compliance rule that only the third person you interviewed knows about — cannot be productized in advance. Someone has to go get it. Palantir invented the title; OpenAI, Anthropic, Sierra, Harvey, Scale, and Glean all now run some version of it.

What it owns: discovery conversations, scoping, building a working vertical slice fast, defining success criteria with the customer *in numbers*, running the eval that proves it, and feeding the generalizable parts back into the core product. What it does not own: the model, the platform roadmap, or usually long-term maintenance.

The loop reflects this and it surprises backend engineers. Expect at least one round that is not technical at all in the usual sense — a simulated customer conversation, a scoping exercise, or "here is a vague business problem, ask me questions." The graded skill is whether you can extract requirements from a non-technical stakeholder without either over-promising or lecturing them. Technical rounds tend toward practical building speed rather than algorithms: wire something end to end quickly, with an eval, under ambiguity.

**🗣 Say this in the room:** "The FDE bet is that the last mile can't be productized, so you send an engineer to walk it. I'd measure myself on whether the customer's workflow metric moved — resolution rate, hours saved per case — not on whether I shipped a feature."

**⚠ Trap:** Treating FDE as a lesser engineering role or a solutions-engineering title with a raise. At the companies on your target list it is often the *highest-leverage* IC track, compensated at or above core engineering, precisely because the conversion from pilot to seven-figure contract runs through it. But it is genuinely different work; take it because you want customer contact, not because it looked like an easier door.

### How does a Solutions Architect or Solutions Engineer differ from an FDE — and when should someone take one?

The difference is who owns delivery of working software. A Solutions Architect is pre-sales: they design the reference architecture, run the technical evaluation, answer security-review questions, build the demo, and hand off. An FDE is post-sale: they build the actual thing in the customer's environment and are accountable when it does not work. SA is measured on pipeline influenced and deals closed; FDE is measured on deployments that survive.

The practical consequence for you is compensation structure and career path. SA roles usually carry a sales-linked variable component and sit under a go-to-market org; FDE usually sits under engineering with standard equity. Both hire backend engineers with communication skills, and both loops emphasize the same rare combination — build fast, explain clearly, resist over-promising.

I would take an SA role in exactly one situation: as a fast lateral into an AI-native company whose core engineering loop you cannot pass yet, with an explicit internal-transfer conversation before you sign. It works — the internal bar for transfer is real but far lower than the external bar — but only if you get that conversation on the record with a hiring manager, not a recruiter. Otherwise take FDE, which keeps you in the engineering ladder.

### Here's a JD titled "AI Engineer." The bullets mention ablations, data mixes, and training throughput. How do you read that?

I read the title as marketing and the bullets as the contract. "AI Engineer" has become the most overloaded title in the industry — it is applied to at least four genuinely different jobs — so I parse the JD bottom-up: responsibilities and required-experience bullets first, title last, and team name second-to-last.

That specific JD is an ML/Research Engineer role wearing a fashionable hat. *Ablation* means you will be asked to design an experiment and defend a comparison. *Data mix* means pretraining or continued-pretraining work. *Throughput* means MFU, distributed training, and a systems-design round about a multi-node run. If I applied anyway, the loop I'd get is a PyTorch implementation round, a training-systems design round, and a deep dive on something I have supposedly done — none of which is the loop the title advertised. That mismatch is where candidates burn their one shot at a company.

The general decoding rules I run, in order:

*Look at what the model is.* If the JD names a provider API (OpenAI, Anthropic, Bedrock, Vertex), the weights are external and this is Applied. If it names a base model family plus a training framework, the weights are internal and this is ML/RE.

*Look at what the metric is.* Latency, cost per request, resolution rate, deflection rate, user satisfaction → product. Loss, perplexity, MFU, benchmark scores → research. Tokens/sec/GPU, utilization, p99 under load → infra.

*Look at who they say you'll work with.* "Partner with researchers" means you're the engineer to their science — RE-adjacent. "Partner with PMs and designers" means product. "Partner with customers" means FDE. "Partner with platform teams" means infra.

*Look at the interview process if they publish it.* A published take-home is nearly always Applied. A published "research discussion" round is nearly always RE.

**⚠ Trap:** Trusting the level in the title across companies. Titles are not comparable — a "Senior AI Engineer" at a 60-person AI-product startup and at Google are two different scopes and two different comp bands. Anchor on scope evidence and on the published band, never on the word "Senior."

### Walk me through how you'd read a job description in ninety seconds and predict the five rounds you'll sit.

I run a fixed four-pass scan, because I want a *prediction I can be wrong about* rather than a vibe.

Pass one, thirty seconds: classify the archetype. Frontier lab, AI infra, AI product, big-tech applied, high-comp non-AI-native, or FDE. The archetype alone predicts most of the loop, because loop structure is far more correlated with company type than with title.

Pass two, twenty seconds: find the model boundary. Do they call an API or own weights? This decides whether I get a transformer-internals round.

Pass three, twenty seconds: find the named systems. If the JD names vLLM, Triton, or Ray, expect a serving round. If it names LangGraph, MCP, or "agent," expect an agent-design round and probably an agent take-home. If it names pgvector, Elasticsearch, or "hybrid search," expect a retrieval-design round. If it names Spark, Databricks, or a lakehouse, expect data-platform design fused into the AI question.

Pass four, twenty seconds: find the customer. Enterprise customers named in the JD → expect a communication or scoping round. Consumer product → expect a product-sense round with a metric argument.

From that I write down my predicted loop in one line, e.g. for an AI-product company: *recruiter screen → practical coding or take-home → take-home defense / system design with an eval focus → agent-or-retrieval design → hiring manager and product sense → team fit.* Then I ask the recruiter to confirm, which costs nothing and is the highest-return question in the entire process.

**🗣 Say this in the room** (to the recruiter, verbatim): "Before I schedule, could you walk me through the stages and tell me what each one is evaluating? I want to prepare for the right things — in particular, is there an algorithmic coding round, and is there a round focused on evaluation methodology?"

**🔍 Failure taxonomy — how JD-reading goes wrong:** (1) *Title-anchored:* you prepared for Applied and got a PyTorch round — cause: you read the title, not the bullets. Fix: bullets-first, always. (2) *Framework-anchored:* you memorized LangChain APIs and got asked to design an eval — cause: you predicted the screen, not the loop. Fix: predict all stages, weight the middle ones. (3) *Archetype-blind:* you gave a frontier-lab answer (scaling, capability) to a big-tech applied panel that wanted reliability and cost — cause: no archetype tag. Fix: tag before you prep. (4) *Stale-loop:* you prepped last year's process. Fix: ask the recruiter; these processes changed materially between 2024 and 2026.

### What's the single biggest positioning mistake you see backend engineers make in AI Engineer loops?

Answering an AI Engineer question as if it were an ML Engineer question. It happens because the candidate has just spent six weeks studying transformers and gradient descent, and that material is now the most salient thing in their head, so every question becomes an opportunity to display it.

Here is the concrete shape. The interviewer says: "Our support-ticket classifier is at 78% accuracy and the team wants 90%. What do you do?" The mistake answer starts with "I'd fine-tune a smaller model on the labeled tickets" and then discusses LoRA rank. The hire answer starts with "First I'd want to see the confusion matrix and read fifty errors — I want to know whether the 22% is one dominant confusion pair, ambiguous labels, or genuinely hard cases, because those have completely different fixes and only one of them is a model problem."

The reason this is graded so harshly is that it reveals process, not knowledge. Applied AI work is 70% error analysis and 20% plumbing; the model is the part you did not build. An interviewer at Sierra or Harvey or Ramp is trying to find out whether you will spend the first two weeks of a quality regression staring at data or shipping a speculative fix. The fine-tuning answer says "shipping a speculative fix."

The inverse mistake exists too and is equally fatal in the other direction: giving a prompt-engineering answer in a Research Engineer loop, where "have you tried asking it nicely" is not an intervention.

**⚠ Trap:** Displaying breadth to prove you studied. Under pressure, candidates volunteer everything they know — attention mechanics, quantization, RLHF — into an answer that asked for a debugging plan. It reads as unfocused. Answer the question asked; the depth gets probed on follow-up, and being probed is a *good* sign.

**🗣 Say this in the room:** "Before I pick an intervention I'd want the error distribution. My experience is that most accuracy gaps in these systems decompose into retrieval misses, context assembly problems, and instruction ambiguity long before they're capability limits — and those are far cheaper to fix than touching weights."

### Why is "we'd fine-tune the model" a rejection trigger, specifically? Give me the argument you'd make in the room.

Because it is almost always the most expensive answer to the least-diagnosed problem, and saying it first proves you skipped diagnosis.

The mental model I hold is an escalation ladder ordered by cost-to-try, and I climb it in order: (1) prompt and context assembly, (2) retrieval quality, (3) tool design and structured output contracts, (4) model routing — send the hard 10% to a stronger model, (5) distillation into a smaller model for cost, (6) parameter-efficient fine-tuning, (7) full fine-tuning. Each rung costs materially more engineering time than the one below it (the two ends of the ladder are one to two orders of magnitude apart: a few engineer-days at the bottom against engineer-weeks plus ongoing ownership at the top), and — this is the part candidates miss — each rung *adds a permanent maintenance obligation*. A fine-tune is not a change, it is an asset: you now own a checkpoint, an eval to prevent regression, a retraining trigger when the base model deprecates, and a deployment surface.

**💰 Math:** Compare interventions on a task doing 500k calls/month with an 8k-token prompt and 500-token output. At $3/Mtok input and $15/Mtok output, that is (8,000 × 3 + 500 × 15) / 1,000,000 = $0.024 + $0.0075 = $0.0315 per call, so 500,000 × $0.0315 = **$15,750/month**. A prompt/context fix costs a few engineer-days and can be reverted in one deploy. A distillation to a self-hosted 8B might cut inference to roughly $1,500–$3,000/month of GPU — real money — but costs a data-generation run, a training run, an eval harness, and a serving stack you now operate; call it 4–8 engineer-weeks up front plus ongoing ownership. At a $15.7k/month bill you are not saving your way to a fine-tune decision — you would need either a quality reason or 5–10× that volume for the payback to be obvious. **📅 Volatile:** per-token prices have been falling roughly an order of magnitude per generation-and-a-half; re-run this arithmetic with current numbers before you quote it.

The preconditions I'd name before agreeing to fine-tune: the failures are capability failures not context failures (verified by error analysis); you have ≥1,000 high-quality task-specific examples or a way to generate them; the task has a stable output format or style the base model resists; you have an eval that will catch regression; and either cost at your volume or latency justifies owning a serving stack. If I can't check most of those boxes, the answer is no.

**🗣 Say this in the room:** "Fine-tuning is the last rung, not the first. I'd want error analysis showing these are capability failures rather than context failures, a thousand-plus clean examples, and a regression eval before I'd sign up for owning a checkpoint — because a fine-tune isn't a change, it's a permanent asset with a maintenance cost."
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

The landscape splits three ways. **Prohibited:** several labs and research-heavy orgs — DeepMind and xAI among the commonly-reported ones, plus a set of enterprises — ban AI assistance in live rounds outright and back it with proctoring: browser monitoring, screen sharing, occasionally audio analysis, and an increasing share of in-person onsites specifically to close this hole. **Dual-mode:** Microsoft is the notable example of running *both* an AI-assisted round and a raw-coding round, on the explicit theory that they need to see both your ceiling with tools and your floor without them. **Encouraged:** most AI-product companies — Cursor most obviously — expect you to use AI tools and may quietly grade you on whether you use them *well*. Anthropic is the cautionary case for treating any of this as fixed: it has publicly revised its stance on candidate AI use at least once, so treat policy as a per-round fact you confirm rather than a per-company reputation. **📅 Volatile:** these policies changed materially in the last two years and will change again; confirm per-loop, and assume anything you read about a specific company's policy is stale.

The practical preparation consequence is uncomfortable and worth taking seriously: you must be able to write correct Python — a class, a dataclass, an async function, a retry with backoff, a small numpy or torch tensor manipulation — from memory at speed, without autocomplete. Most working engineers have quietly lost this. Two weeks of writing code in a plain editor fixes it, and it is the cheapest insurance in the process.

In an assisted round, what is graded is not whether you use the tool but *how*: do you specify clearly, do you read what it produced, do you catch the bug it introduced, do you know when to stop prompting and write it yourself. Narrate that. "I'll let it scaffold the client and the retries, but I'm going to write the streaming parser myself because that's where the edge cases are" is a strong sentence in an assisted round.

**⚠ Trap:** Using a hidden assistant in a prohibited round. Detection is better than candidates assume — real-time transcription tools and answer overlays are specifically what the proctoring targets — and the consequence is not a rejection, it is a permanent blacklist at a company you may want for a decade. It is a terrible expected-value trade.

### Of all these rounds, which one carries the hidden weight — and how do you find out which one before you sit it?

There is always one round that is disproportionately decisive, and it is almost never the one the candidate over-prepares. The archetype predicts it:

At a **frontier lab**, it is the values or alignment-judgment round — technical competence is table stakes at the applicant pool they see, and judgment is the differentiator. At an **AI-product company**, it is the take-home defense or the product-sense conversation: can you argue about a metric and a user, not just a stack. At **AI-infra**, it is the systems design round with real numbers — GPU memory, batching, tail latency — because the whole job is that. At **big-tech applied**, it is the behavioral round evaluated against a structured rubric, plus the design round's cross-team story; Amazon's bar raiser is the explicit institutionalization of this. At **FDE**, it is the customer or scoping conversation. At **high-comp non-AI-native** companies (Stripe, Ramp, and similar), it is usually the practical build plus the reliability-and-cost argument — they are hiring an engineer who will not blow up their unit economics.

How to find out: ask. Recruiters answer this honestly far more often than candidates expect, because a prepared candidate makes their pipeline look good. The phrasing that gets a real answer is not "which round matters most" — that invites "they all matter" — it is: **"What's the most common reason strong candidates don't make it through this loop?"** That question gets you the hidden weight, in one sentence, roughly three times out of four.

**🗣 Say this in the room** (to the recruiter): "What's the most common reason strong candidates don't get through this process? I'd rather over-prepare for the right thing."

**🔍 Failure taxonomy — misallocated preparation:** (1) *Over-indexed on algorithms* for a loop with no algorithm round — symptom: you feel over-prepared and still get "not enough depth on evaluation." (2) *Over-indexed on internals* for an AI-product loop — symptom: you can derive attention but fumble "what metric would you ship on." (3) *Under-indexed on communication* for FDE or enterprise — symptom: technically clean rounds, vague "fit" rejection. (4) *Under-indexed on the take-home* — symptom: you pass every live round and get rejected after submitting; cause: you treated a graded artifact as a formality and skipped the eval section.
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

**💰 Math — a worked version of exactly this round.** 2M documents/month × 5 pages, and take a page as roughly 700 tokens, so 3,500 input tokens per document plus a 1,200-token instruction prompt and 300 tokens of structured output. Naive frontier-model design at $3/Mtok in and $15/Mtok out: input (3,500 + 1,200) = 4,700 tokens → 4,700 × $3/1e6 = $0.0141; output 300 × $15/1e6 = $0.0045; total $0.0186/doc × 2,000,000 = **$37,200/month**. Now optimize in the order that actually pays: (1) prefix-cache the 1,200-token instruction — at a 90% cached-input discount that portion drops from $0.0036 to $0.00036, saving ~$6,500/month; (2) batch-tier the workload since documents are not latency-sensitive — commonly around 50% off, roughly halving what remains; (3) route by difficulty — send the 80% of documents that a small model handles at equal accuracy (which you prove with an eval, not an assertion) to a model roughly 10–20× cheaper. Stack them explicitly rather than hand-waving: after (1) you are at ~$30.7k, after (2) ~$15.4k, and after (3) you pay 0.2 × $15.4k on the frontier path plus 0.8 × $15.4k at a tenth to a twentieth of the rate ≈ **$3.7k–$4.3k/month**. Quote the defensible range as roughly $4k–$9k — a 4–9× reduction — because providers do not always let batch and cached-input discounts compose, and the 80% routable share is a claim you have to prove rather than assume. The part that gets you the hire is stating which optimization you'd verify first and what eval would tell you routing is safe. **📅 Volatile:** all per-token prices and discount rates here are illustrative and move constantly; re-derive with current numbers.

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

**💰 Math — checkpoint interval, the classic version of this question.** If a node fails on average every 6 hours across the cluster and a checkpoint costs 4 minutes of wall clock, then checkpointing every T minutes gives expected lost work of T/2 on failure. Total overhead per hour ≈ (60/T) × 4 minutes of checkpointing, plus (1/6) × (T/2) minutes of lost work per hour. Minimizing: at T = 60 min you pay 4 min/hr of checkpointing and 5 min/hr expected loss = 9 min/hr (15%); at T = 30 you pay 8 + 2.5 = 10.5; at T = 90 you pay 2.67 + 7.5 = 10.2. So the optimum sits just under an hour — the closed form is the Young/Daly result T\* ≈ √(2 × checkpoint cost × MTBF) = √(2 × 4 × 360) ≈ 54 minutes, costing ~8.9 min/hr ≈ 15% overhead — and the cost curve is flat around it. That flatness is the real insight to state out loud: the exact interval barely matters, but being 3× off in either direction pushes overhead to roughly 25%, and being 10× off costs you about three-quarters of your cluster.

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

What differentiates candidates here is not model knowledge, it is **applying senior engineering discipline to a nondeterministic dependency**. Concretely: how do you version and roll back a prompt like code; how do you flag-gate a model upgrade when the provider announces a deprecation date for the model you depend on (notice periods vary by provider and have been as short as a couple of months); how do you handle a provider outage without taking the product down; how do you keep a per-tenant cost ceiling so one enterprise customer's usage does not eat the margin; how do you prevent a document the user is not permitted to read from entering their context window; how do you audit an AI-generated action in a system where a wrong ledger entry is a real incident.

Those are the questions a Stripe or Ramp panel actually cares about, and they are *your* questions in new clothing — feature flags, rollback, multi-tenant quotas, authorization at the data layer, audit logs.

**🗣 Say this in the room:** "I treat prompts and model versions as deployed artifacts: versioned in the repo, rolled out behind a flag, gated by an offline eval, and monitored with an online proxy metric so we can roll back in one deploy. The failure mode I'm defending against is a silent quality regression, which is invisible to every alert we already have."

**⚠ Trap:** Under-preparing the classical rounds because the role says AI. I have seen candidates study transformers for six weeks and then fail a Stripe loop on a plain distributed-systems design question about idempotent payment retries. At non-AI-native companies the classical bar is the *primary* bar.

### Sierra, Harvey, Glean — the vertical AI-product companies. Anything specific?

Yes: domain grounding and deployment reality carry more weight than at a horizontal product company, and the loops reflect that.

These companies sell into a specific workflow — customer service resolution at Sierra, legal work at Harvey, enterprise search and assistants at Glean — and the thing that separates a demo from a business in those verticals is whether the system is *right often enough to be trusted by a professional whose job depends on it*. So expect: an evaluation-heavy conversation (what does "correct" mean for a legal research answer, and who decides), a permissions and data-governance conversation (Glean's entire technical moat is enterprise permission-aware retrieval across dozens of source systems — a document must be invisible in retrieval to a user who cannot open it in the source system, enforced at query time and re-checked at render time), and an agent-reliability conversation (Sierra's product is an agent that takes real actions on real accounts; the interesting questions are about action confirmation, reversibility, escalation to a human, and containment when the agent is wrong).

Expect also a customer-facing or scoping component, since these companies deploy into enterprises and most engineering roles there touch customer reality.

The preparation that pays: be able to talk about **the cost of a wrong answer** in the vertical. In consumer chat a wrong answer is annoying; in legal it is malpractice exposure and in customer service it is a refund issued against policy. That asymmetry should drive your entire design — where you put a human in the loop, what you refuse to automate, what confidence threshold triggers escalation, and how you measure the escalation rate as a first-class product metric rather than a failure.

**🗣 Say this in the room:** "I'd design around the cost asymmetry: for actions that are cheap to reverse I'd let the agent act and log it; for anything irreversible or externally-visible I'd require confirmation, and I'd treat escalation-to-human rate as a headline metric rather than something to minimize at all costs — driving it to zero is how you ship a confident wrong answer."
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

**📐 Numbers you must know about your own plan:** 120 hours is roughly 6% of a 2,080-hour working year and it is genuinely enough for this archetype *only* if you subtract aggressively. The failure mode is spending 40 of those hours re-reading distributed systems material you already own. Run the skip list first; it is the highest-leverage hour in the plan.

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


---

## 2. The Delta Map: What to Skip, What to Skim, What to Attack

*Mastering this proves you can subtract 350 questions of material you already ship and redirect that time into the five rounds where a backend engineer actually gets rejected.*

### You've got eight weeks and a full-time job. Walk me through how you'd decide what *not* to study.

The mental model is capacity planning, and you already run it every quarter: you do not add a feature to a saturated service, you find the thing already consuming the budget and delete it. Prep time is the saturated service. Almost every backend engineer who fails an AI loop failed it while working hard — they spent week three re-reading asyncio internals because that felt productive and safe, and arrived at the onsite unable to write multi-head attention on a whiteboard. **Effort was never the constraint. Allocation was.** So the first artifact you build is not a study schedule, it is a subtraction list.

The decision rule I use has exactly three tests, applied per topic:

1. **Does this topic appear in the loop at all?** If no interviewer will ask it, it is not prep, it is comfort. Generic behavioral STAR is the cleanest example — the archetypes you are targeting have replaced it with values rounds and customer-conversation simulations that STAR answers actively fail.
2. **If it appears, do you already clear the bar cold?** Not "do you know it" — *do you clear the bar under time pressure, unaided, at the depth this level demands*. That is what the 🧪 gates in this section are for. Three questions per skip, answered correctly out loud, or the skip is revoked.
3. **Is there a delta between your version and the AI version?** This is where most engineers get it wrong in both directions. Celery is a skip; *cancelling a 90-second streaming generation whose consumer disconnected* is not, because it has no Celery analogue. K8s is a skip; GPU device plugins, MIG partitioning and node-level fragmentation are not. The skip is almost never total — it is "skip the substrate, study the six-inch delta on top."

**📐 Numbers you must know:** this guide is 4,009 questions. The SKIP list removes roughly 350 of them — about 8.7% of the corpus. At the mastery path's rate of 30 questions/day, 350 questions is 11.7 days. On an 8-week (56-day) runway that is **21% of your total prep time**, recovered and redirected into transformer internals, serving, evaluation, post-training and agent harnesses. That is the entire economic argument for this section: one-fifth of your runway, free, for the cost of one honest afternoon of self-assessment.

**⚠ Trap:** treating the skip list as permission to be rusty. A skip means "this is already at bar, do not spend prep cycles on it." It does not mean "this will not be tested." Big-tech applied loops (Meta, Google, Amazon, Microsoft) still run a coding round and a general system-design round where your backend fluency is the *floor*, not a bonus. If you skip asyncio and then stumble on an async producer/consumer in a live round, the skip was mis-taken — the gate existed to catch exactly that.

**🗣 Say this in the room:** "My prep plan was built by subtraction. I audited what the role actually tests against what I already ship daily, cut about a fifth of the material, and spent that time on transformer internals and evaluation methodology — because those are where a backend engineer's gap is real and where the loop actually rejects people."

### In a debrief, where do backend engineers most often get rejected from AI Engineer loops? Be specific.

Five rounds, and they are consistent enough across the archetypes that you should treat them as the load-bearing structure of your whole plan.

**One: transformer internals.** The from-scratch round — implement attention, explain why the softmax is scaled by √d_k, derive the KV cache size, explain what RoPE actually rotates. Backend engineers fail this not because it is hard but because they treated the transformer as an API boundary. It is the only round where "I've been shipping LLM features for a year" buys you nothing.

**Two: serving and inference internals.** Continuous batching, PagedAttention, prefill-vs-decode asymmetry, quantization trade-offs, speculative decoding. You will recognize every concept here as a scheduler or memory-allocator problem — that recognition is your unfair advantage, but only if you have actually learned the primitives to map onto.

**Three: evaluation.** This is the single most reliable rejection cause and the most under-prepared. The tell is a candidate who designs a beautiful RAG system and, asked "how would you know it works," says "we'd look at user feedback." A senior answer opens with the eval before the architecture. Every take-home rubric in the wild weights evaluation methodology as critical, and skipping it is the named #1 critical error.

**Four: post-training literacy.** You will not be asked to run an RLHF job at an applied role. You *will* be asked why you would use DPO over PPO, what RLVR needs that a chat task cannot supply, when LoRA beats prompting, and — the trap — you will be asked a problem whose correct answer is "do not fine-tune." Reflex-fine-tuning is an instant senior-level downgrade.

**Five: agent harness engineering.** Not "have you used LangGraph." Tool-schema design, error surfaces fed back to the model, context budget accounting, loop termination, replayability, and cost per *resolved task* rather than per call. This is the round Cursor, Sierra, Harvey and Glean actually care about, and it is closest to your existing skills — it is distributed systems with a nondeterministic worker.

**⚠ Trap:** believing "AI Engineer means no algorithms round, so I can coast on coding." Roughly 70% of senior AI loops contain no LeetCode-style round — but the named exceptions are exactly the companies on your target list: Perplexity runs a genuinely hard machine-coding round, xAI and Anthropic screen with timed coding, and quant-adjacent AI shops are pure algorithms. Check the specific company before exercising that skip.

**🗣 Say this in the room:** "I assumed the gap was model internals and evaluation, not engineering, so that's where I put my time. The engineering side of an AI product — queues, idempotency, observability, multi-tenancy — is the part I've been doing for years; what I had to build was the judgment about *what to measure*."

### Tell me why you'd skip asyncio and event-loop internals entirely. Isn't every LLM app I/O-bound?

It is, and that is precisely why you skip it: you already own the entire competency. You know that `asyncio` is a single-threaded cooperative scheduler over a selector, that a coroutine is a generator with a `send()` protocol and that awaiting is a suspension point, that a blocking call inside a coroutine stalls every other task on that loop, that `asyncio.gather` propagates the first exception while `TaskGroup` cancels siblings and raises an `ExceptionGroup`, and that `run_in_executor` is how you quarantine CPU-bound work. Nothing about calling a model API changes any of that. An LLM request is an HTTP request with a long tail and a streaming body — architecturally it is a slow S3 GET.

The *delta* is small and specific, and it is worth about ninety minutes, not two days:

- **Streaming response lifecycle.** Server-Sent Events from the provider, forwarded as SSE to your own client. The specific hazard is that FastAPI's `StreamingResponse` will happily keep the upstream generation alive after the browser disconnects; you need to detect disconnect and cancel, or you pay for tokens nobody receives.
- **Concurrency limits that are token-shaped, not request-shaped.** A `Semaphore(50)` is the wrong primitive when the provider's limit is tokens-per-minute. Your gate must be a token-bucket over *estimated* input+output tokens.
- **Cancellation semantics under a long stream.** `asyncio.CancelledError` in the middle of an SSE read must close the upstream connection, not just stop consuming it, or you leak both a socket and a billing meter.

```python
# The one async pattern worth rehearsing: disconnect-aware streaming proxy.
@app.post("/chat")
async def chat(req: Request, body: ChatIn):
    async def gen():
        async with client.messages.stream(**body.to_params()) as stream:
            async for text in stream.text_stream:
                if await req.is_disconnected():   # client hung up
                    break                          # context manager closes upstream
                yield f"data: {json.dumps({'delta': text})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

**🧪 Verify before you skip:** (1) Why does `await asyncio.sleep(0)` yield control but `time.sleep(0)` does not, and what breaks if you get that wrong inside a request handler? (2) In `TaskGroup`, one child raises — describe precisely what happens to the other children and what the parent observes. (3) You have a 90-second streaming generation and the client disconnects at second 4. Where does the cancellation actually have to reach for you to stop being billed? If you cannot answer all three cleanly, do the ninety minutes.

**⚠ Trap:** assuming provider SDK clients are safe to share across an event loop with unbounded concurrency. They usually are safe object-wise, but the connection pool underneath has a finite size, and 500 concurrent streams against a pool of 100 means 400 coroutines silently queued behind a connection acquire — showing up in your traces as inflated TTFT with no upstream latency to blame. That is your old connection-pool-exhaustion incident wearing a new hat.

### Same question for the GIL and free-threading. Skip or not?

Skip the GIL as a *concept*, because you already have it at a level most AI candidates never reach: you know it protects interpreter state including refcounts, that it is released around blocking I/O and inside C extensions that opt out, that it makes CPython threads useful for I/O and useless for pure-Python CPU work, and you can reason about the eval-loop switch interval. There is no version of an AI Engineer interview that goes deeper than that.

The one thing to actually internalize is *where the GIL is and is not released in an AI stack*, because it changes your architecture:

- **PyTorch, NumPy and tokenizer kernels release the GIL** for the duration of their C/Rust work. HuggingFace `tokenizers` is Rust with the GIL released, so batch-tokenizing 10k documents across a thread pool genuinely parallelizes. This surprises people who over-learned "threads don't help in Python."
- **Pure-Python glue does not.** Your chunking logic, your prompt templating, your JSON post-processing — all GIL-bound. In a document-ingestion pipeline that is often where the real CPU time hides, and the fix is processes, not threads.
- **The GPU process boundary.** In an inference server the model lives in one process; multiprocess "parallelism" would mean N copies of the weights in VRAM. This is why serving engines use a single process with an internal batching scheduler rather than a worker pool — the memory model, not the GIL, dictates the shape.

**📅 Volatile:** free-threaded CPython (PEP 703) shipped as an experimental build in 3.13 and moved to a supported-but-optional build in the 3.14 line, with the wider ecosystem still catching up on C-extension compatibility. Do not assert a specific "it's the default now" status in a room — verify before your loop. The safe framing is: "free-threading removes the GIL at a per-object-locking cost and a single-thread performance penalty; for an LLM gateway it changes little because we're I/O-bound anyway, and the extensions that matter already release the GIL."

**🧪 Verify before you skip:** (1) Name two operations in an LLM serving path that release the GIL and two that do not. (2) Why does running four `uvicorn` workers not quadruple throughput for a GPU-backed inference endpoint? (3) What is the actual cost that free-threading pays to remove the GIL, and who eats it?

**🗣 Say this in the room:** "The GIL is a non-issue for LLM serving because the whole hot path is either I/O wait or GPU work with the GIL released. Where it bites is the CPU-bound glue — chunking, templating, embedding-side preprocessing — and for that I use processes, not threads."

### Pydantic v2 is on my resume. Should I be revising validation internals?

No. Skip the mechanics — `model_validate` vs `model_construct`, validator ordering, `Annotated` metadata, the pydantic-core Rust layer, serialization aliases, discriminated unions. You use these weekly and no AI interviewer will out-depth you on them.

What you must attack instead is the *adjacent* topic that looks like Pydantic and is not: **structured output from a model**. Here is the delta that actually matters and that almost nobody transfers correctly.

Your instinct is that a schema is a validation contract applied *after* data arrives. In LLM serving, a schema can instead be a *generation constraint* applied token by token: the JSON Schema is compiled to a finite-state machine or a grammar, and at each decode step the sampler's logits are masked so only tokens that keep the output valid have nonzero probability. That is constrained decoding — it does not validate, it makes invalid output unrepresentable. Providers expose it as strict structured outputs; open-weight stacks do it with Outlines, XGrammar, llama.cpp's GBNF, or vLLM's guided decoding backends.

The consequences you should be able to state cold:

- **JSON Schema support is a subset.** Strict modes typically require `additionalProperties: false`, require every property to be listed in `required`, and reject or ignore constructs like `pattern`, `minimum`, `oneOf` in some engines. Pydantic will happily emit a schema your provider silently degrades. The rule I enforce in review: generate the schema from the model, then *assert* it round-trips through the provider's strict validator in CI, rather than trusting it.
- **Constrained decoding shifts the failure mode, it does not remove it.** You get syntactically perfect JSON containing semantically wrong values — an enum member that is valid but incorrect, a required field filled with a plausible hallucination because the grammar forbade omitting it. Making a field required under constrained decoding *forces* a fabrication when the model doesn't know. Optional fields plus an explicit `"unknown"` sentinel are strictly better.
- **It costs something.** Grammar compilation is per-schema (cacheable) and mask computation is per-token; a well-implemented backend adds low single-digit percent to decode, a naive one can cost far more. Measure it rather than assuming.

**⚠ Trap:** "we use Pydantic so our LLM output is validated." Validation catches a malformed response *after* you paid for all its tokens, then you retry and pay again. Constrained decoding prevents the malformed response. Conflating the two in an interview reads as someone who has only ever used the model through a wrapper library.

**🧪 Verify before you skip:** (1) What does `additionalProperties: false` do to a model's ability to produce output, and why do strict modes demand it? (2) Why does marking a field `required` under constrained decoding increase hallucination rate? (3) You get valid JSON with a wrong enum value — is that a schema bug, a prompt bug, or an eval gap, and how do you tell?

### What about SQLAlchemy 2.0 and Alembic — you've clearly got that. Anything left?

Skip the ORM entirely: the 2.0 style, `select()` construction, unit of work, identity map, lazy-load `N+1` and `selectinload`, session scoping in async, `Alembic` autogenerate and its blind spots, online migrations with `CREATE INDEX CONCURRENTLY`. That is all at bar and it stays at bar.

The delta is narrow and you should learn it precisely, because it is a favourite Databricks/Snowflake/Ramp-flavoured design probe: **vector storage as a schema-migration problem.**

Two things change. First, `pgvector` adds column types (`vector`, `halfvec`, `sparsevec`) and index types (IVFFlat and HNSW) with build parameters that behave nothing like a B-tree. An HNSW build is expensive and memory-hungry; IVFFlat needs representative data present before you build it because it clusters. `CREATE INDEX CONCURRENTLY` still applies but the build time is now measured in tens of minutes to hours on millions of rows, not seconds.

Second — and this is the real insight — **changing your embedding model is a migration, and it is not backward compatible.** Vectors from two different models are not comparable; a mixed index silently returns garbage neighbours rather than erroring. So the pattern is the one you already use for a zero-downtime index swap, lifted wholesale:

```
1. Add embedding_v2 column (or a whole shadow table) — nullable, no index.
2. Backfill in batches, from a queue, resumable, with a watermark column.
3. Build the v2 HNSW index CONCURRENTLY once backfill is complete.
4. Shadow-read: query both, log recall@k agreement and score distributions.
5. Flip a read alias / feature flag, keep v1 for rollback.
6. Drop v1 after the rollback window.
```

**💰 Math:** re-embedding 5M chunks averaging 400 tokens each = 2.0B tokens. At a small-embedding-model price of roughly $0.02 per million tokens (**📅 Volatile** — verify current embedding pricing), that is 2,000 × $0.02 = **$40** of API spend. The API cost is trivial; the cost that actually bites is that you must hold both indexes simultaneously. 5M vectors at 1536 dims × 4 bytes = 30.7 GB per copy of raw vectors, plus HNSW graph overhead of roughly 30–60% depending on `m`, so call it ~45 GB per version and ~90 GB peak during the swap. **The migration is a capacity-planning problem, not a cost problem** — that inversion is the answer interviewers are listening for.

**🧪 Verify before you skip:** (1) You add a new embedding model — what specifically goes wrong if you write its vectors into the existing column? (2) Why does IVFFlat need data before index build while HNSW does not? (3) What is your peak storage requirement during a re-embedding swap, and how do you avoid doubling it?

### Celery, Kafka, DLQs — you've run those for years. Where's the delta?

Skip the fundamentals: broker semantics, `acks_late`, visibility timeout, at-least-once delivery, idempotent consumers, dead-letter queues, consumer groups and partition assignment, backpressure, poison-message quarantine. You already reason in this vocabulary and it maps almost one-to-one.

Four deltas are real and worth a focused hour each:

**Task duration and shape.** A Celery task is typically sub-second to seconds. An agent trajectory is 30 seconds to 20 minutes, is *streaming* (the caller wants partial output before completion), and is *cancellable* mid-flight with real money at stake. That combination breaks the fire-and-forget pattern. The shape that works: durable run record in Postgres, worker publishes deltas to a Redis stream or pub/sub channel keyed by run ID, the HTTP layer subscribes and forwards SSE, cancellation is a flag the worker polls between model calls. You already know every piece; nobody just hands you the assembly.

**Retries are not free and not idempotent.** In your world, retrying a failed task costs a little CPU. Here, retrying a failed agent step re-runs every model call in that step. I cover the 4× amplification arithmetic later in this section — the headline is that the naive "retry 3× with backoff" you would add without thinking is a cost decision that needs a budget, not a resilience default.

**Poison messages become poison *trajectories*.** A poison message is a bad payload. A poison trajectory is a run that is *individually valid* at every step but loops — the model calls the same tool with the same arguments, gets the same error, and tries again. There is no malformed input to quarantine on. Detection is state-based: hash `(tool_name, canonicalized_args)` per step, and if the same hash repeats N times, or if step count exceeds budget, or if cumulative spend exceeds the run's cap, kill and route the whole trajectory to a review queue with its full transcript.

**Ordering matters less; budget matters more.** In Kafka you obsess over partition ordering. In an agent system the scarce resource is tokens and provider rate limit, so the scheduler you want is closer to a fair-share queue over a token budget than a partitioned log.

**🧪 Verify before you skip:** (1) An agent run has produced 40 seconds of streamed output and the worker crashes — what does the user see, and what is your recovery contract? (2) Distinguish a poison message from a poison trajectory and name the detection signal for each. (3) Why is "retry the whole task" a worse default here than in a normal Celery pipeline?

### Docker and Kubernetes — skip?

Skip the substrate completely: images, layers, multi-stage builds, Deployments, Services, HPA, probes, resource requests and limits, PDBs, rolling updates, ConfigMaps and Secrets, Helm. Nobody in this loop will beat you on that and nobody will spend a round on it.

The delta is GPUs, and it is genuinely different because **a GPU is not a compressible resource.** CPU throttles under contention; GPU memory does not — you either fit or you OOM, and there is no fractional scheduling by default.

What to learn, in order of interview yield:

- **The device-plugin model.** GPUs are advertised as an extended resource (`nvidia.com/gpu`) and are allocated in whole units. You request 1, you get 1 exclusively. There is no `0.5` the way there is with CPU millicores.
- **Sharing mechanisms and their honest trade-offs.** MIG partitions an A100/H100 into hardware-isolated slices with dedicated memory and SM allocation — real isolation, coarse granularity. Time-slicing multiplexes without memory isolation, so one tenant OOMs everyone. MPS gives concurrent kernel execution with weaker isolation. The senior answer names which one you would use and why, not just that they exist.
- **Scheduling and fragmentation.** Bin-packing 8-GPU nodes with mixed 1-, 2- and 4-GPU workloads fragments exactly like a slab allocator, and you get the same pathology: plenty of free GPUs, none of them adjacent enough to place a 4-GPU job. Gang scheduling (all-or-nothing placement) and topology awareness (NVLink-adjacent vs across PCIe) are the mitigations.
- **Cold start is the real SLO killer.** Loading a 70B model in fp16 means moving 140 GB from storage into HBM. On a 2 GB/s pull that is 70 seconds; that is why HPA on request rate is a fantasy for large models and why you scale on a leading indicator with warm pools.

**📐 Numbers you must know:** weights in bytes = `params × bytes_per_param`, and bf16/fp16 is 2 bytes, so **the GB figure is just twice the parameter count in billions**. 8B → 16 GB, 70B → 140 GB, 405B → 810 GB. Halve it for fp8, quarter it for int4. Set that against 80 GB per H100-class card and the deployment shape falls straight out: 8B fits comfortably on one card with room for cache, 70B needs two in bf16 or one at 4-bit, and 405B at 810 GB in bf16 does *not* fit an 8×80 GB node (640 GB) — it is served at fp8 on one 8-GPU node with tensor parallelism, or across two nodes if you insist on bf16. You should be able to go from a model name to a GPU count in about four seconds, because that is the pace at which this gets asked.

**⚠ Trap:** answering a GPU-autoscaling question with "we'd use HPA on CPU" or even "on GPU utilization." GPU utilization as reported by `nvidia-smi` measures whether *any* kernel is resident, not whether the SMs are busy — a memory-bandwidth-bound decode loop can show 95% "utilization" while doing very little math. The signal that actually correlates with capacity is KV-cache occupancy and the serving engine's queue depth, which I unpack later in this section.

**🧪 Verify before you skip:** (1) Why can't you set `nvidia.com/gpu: 0.5`? (2) Give me the failure mode of time-slicing that MIG prevents. (3) Your model takes 70 seconds to load — describe an autoscaling policy that survives a 3× traffic spike.

### REST and gRPC design is core to my job. Anything new in an AI product?

Skip the design vocabulary: resource modelling, idempotency keys, pagination, versioning, error taxonomies, gRPC streaming modes, protobuf evolution rules, deadlines and cancellation propagation. All at bar.

Three deltas, and the third one is where seniors separate.

**Streaming is the default, not an option.** A chat or agent endpoint is SSE or a WebSocket, because TTFT is the perceived-latency metric and buffering the full response destroys it. This has API-design consequences you should be able to enumerate: your error contract must handle failures that occur *after* a 200 OK and after partial content has been delivered, so errors become in-band events in the stream rather than status codes; retries at the HTTP layer are now dangerous because you may have already delivered half an answer; and every proxy in your path (nginx, ALB, CloudFront, Cloudflare) has buffering and idle-timeout settings that will silently break streaming until you configure them.

**MCP is the emergent tool-integration surface.** The Model Context Protocol standardizes how a model-facing client discovers and invokes tools, resources and prompts from a server, over stdio or HTTP transports. Think of it as "OpenAPI for tools, plus a discovery handshake" — it does not do anything you could not do by hand, its value is that it turns N×M integrations into N+M. **📅 Volatile:** the spec revises on a real cadence and transport details (notably the HTTP transport story) have changed since launch; verify the current revision before a loop rather than describing an old one confidently.

**The interesting design question is tool schemas as an API for a nondeterministic client.** This is the one worth thinking about hard, because your REST instincts partly mislead. You would normally design fine-grained resources and let the client compose. A model composes badly and pays tokens for every schema it reads, so you design *coarse, task-shaped* tools with few parameters, unambiguous names, and error messages written to be *read by the model as instructions* ("invalid date format, expected YYYY-MM-DD, you sent 'next tuesday'") rather than as codes. That inversion — error strings as prompts — is the single most transferable idea in agent engineering, and it violates everything a REST purist believes.

**🗣 Say this in the room:** "A tool schema is an API whose client is a probabilistic reader with a token budget. So I optimize for the opposite of what I'd do for a human-written client: fewer, coarser tools; names and descriptions that disambiguate without examples; and error payloads written as corrective instructions, because the error text goes straight back into the model's context and is the cheapest place to steer behaviour."

### I've done a hundred behavioral rounds. Why is generic STAR on the skip list?

Because at these companies the behavioral round has been replaced by something STAR answers actively underperform on, and rehearsing STAR harder makes it worse.

Three replacements you should prepare for by name:

**The values round.** Anthropic's is widely reported as a primary filter rather than a formality, and it probes reasoning about tradeoffs under uncertainty — how you behaved when safety or honesty was expensive, when you shipped something you had reservations about, how you handled being wrong in public. A polished STAR story with a tidy quantified result reads as rehearsed and evasive here. What works is specificity plus genuine ambivalence: the decision you are still not sure about, and what you would need to know to settle it.

**The customer-conversation simulation.** Anthropic's applied loop and forward-deployed loops generally include a live roleplay: an interviewer plays a customer with an ill-posed problem, and you are graded on scoping, on saying "that's not what I'd build and here's why," and on not promising capability the model does not have. STAR is useless because it is not a retrospective, it is a live performance.

**The founder or bar-raiser round.** Perplexity's founder round, Amazon's bar raiser, Cursor's "do you actually use AI coding tools daily" hidden rubric. These test taste and conviction. The question "which AI product do you think is badly built and why" is a real question with a real right answer shape: name it, be specific about the mechanism, be fair about the constraint they were under.

What to *keep* from your behavioral prep: the raw material. You need six to eight incidents with real numbers — the outage you diagnosed, the migration you led, the design you argued against and lost, the thing you shipped that failed. Keep the incidents, drop the S-T-A-R scaffolding, and be ready to be interrupted mid-story and taken somewhere else, which is the actual format.

**⚠ Trap:** bringing an AI story you cannot defend to the depth of this guide's other sections. If you say "we added semantic caching and cut cost 40%," expect "what was your false-positive rate on cache hits, and how did you measure it?" A behavioral story about an AI system is a technical round in disguise. Pick incidents you can survive three follow-ups on.

**🏋 Drill:** 20 minutes, unaided. Write out the *incident inventory*: eight bullet lines, each one sentence, each containing one number and one thing that went wrong. No STAR structure. Pass criterion: at least three involve a decision you now think was wrong, and at least two are AI-adjacent with a metric you could defend under cross-examination.

### Add it all up for me. How much time does the skip list actually buy, and where does it go?

Let me do the arithmetic explicitly, because "focus on the important stuff" is not a plan and an interviewer can smell the difference.

**What comes out.** Across the eight skips — asyncio, GIL/free-threading, Pydantic mechanics, SQLAlchemy/Alembic, Celery fundamentals, Docker/K8s basics, REST/gRPC design, generic STAR — the guide's own accounting is roughly 350 questions of the 4,009. But the honest number is larger than that, because these topics are also where you would have *voluntarily* over-invested: they are comfortable, they produce a feeling of progress, and they have infinite depth available. Call the recovered budget 350 questions of reading plus another 10–15 hours of comfort-rereading you now do not do.

**💰 Math:** on an 8-week plan at 12 hours/week you have 96 hours. 350 questions at the guide's density is roughly 20 hours of reading. The comfort-rereading is another 12. So you recover **~32 of 96 hours — one third of your entire runway.** That is not a marginal optimization; that is the difference between covering Parts II, IV and VIII properly and skimming all three.

**Where it goes,** in priority order, and this allocation is opinionated:

| Redirect to | Hours | Why this weight |
|---|---|---|
| Transformer internals + KV-cache math | 10 | Only round where prior shipping experience buys nothing; also gates comprehension of serving |
| Evaluation as a discipline | 8 | Highest rejection rate; also the thing that makes every design answer stronger |
| Serving and inference internals | 6 | Highest leverage for your existing systems intuition |
| Agent harness engineering | 5 | Closest to your strengths, and the round the AI-product archetype weights hardest |
| Post-training literacy (not depth) | 3 | Enough to not reflex-fine-tune and to hold the DPO/PPO/RLVR conversation |

**⚠ Trap:** redirecting the recovered time into *more frameworks*. The instinct after cutting Celery is to go learn LangChain, LlamaIndex and CrewAI, because those feel like the AI equivalent of the thing you cut. They are not. Framework surface area is the lowest-value AI knowledge per hour — it expires, it is trivially recoverable from docs during the job, and no strong interviewer probes it. The guide's `↔` contract exists for exactly this reason: learn the mechanism in 40 lines, then learn what the framework calls it, in that order, never the reverse.

**🗣 Say this in the room:** when asked how you prepared — "I audited the delta between what I already ship and what the role tests. About a third of what I'd have naively studied was substrate I already own — async, queues, containers, API design — so I cut it and spent that time on transformer internals, serving, and evaluation methodology. The one thing I deliberately did *not* do was study agent frameworks, because I'd rather be able to write the loop than name the library."
### You said classical ML is a skim, not a skip. What exactly stays in?

Everything you keep serves one purpose: **being able to say "I would not use an LLM here" and be right.** That is the maturity signal interviewers explicitly listen for, and it is the highest ratio of hiring signal to study hours in the entire foundations layer. You are not being tested on gradient boosting internals. You are being tested on whether you reach for a 12B-parameter probabilistic text generator to parse a fixed-format invoice number.

Keep exactly four things.

**The substitution table.** For each common "AI feature" request, the deterministic alternative and the condition under which it wins. Fixed-format extraction → a regex or a real parser. Routing among ten known intents → an embedding plus a threshold, or a logistic regression over TF-IDF, trained in an afternoon. A known aggregation over structured data → SQL, always, with the LLM at most writing the query. A compliance gate → deterministic rules, with the LLM allowed to draft and a human to approve. The head of your query distribution → precomputed answers in a cache, because in most products the top 200 queries are 40–60% of traffic.

**💰 Math, the one to have ready:** intent classification at 1M requests/month. A frontier API call with a 600-token prompt and 5 output tokens at $3/M in and $15/M out costs 1M × (600×3 + 5×15)/1e6 = 1M × ($0.0018 + $0.000075) = **$1,875/month** at ~700 ms p50. A scikit-learn logistic regression over sentence embeddings: embeddings at ~$0.02/M tokens × 600M tokens = $12/month, inference in-process at ~2 ms, and if you cache embeddings for repeat queries it approaches free. **156× cheaper, 300× faster.** The LLM is the right answer only if the label set changes weekly or the long tail is genuinely open-ended — which is a real condition, and naming that condition is what makes the answer senior rather than contrarian. (**📅 Volatile:** prices move; the ratio is the durable part.)

**Metrics, because eval rounds reuse them wholesale.** Precision/recall/F1, macro vs micro averaging, PR-AUC vs ROC-AUC under class imbalance, and the retrieval set — Recall@k, MRR, nDCG. You will use these in RAG evaluation constantly. The specific tell interviewers watch for: reporting ROC-AUC on a moderation task where positives are 1% of traffic, where PR-AUC is the honest metric and ROC-AUC flatters you badly.

**Clustering for error analysis.** HDBSCAN over embeddings of your failure cases is how you turn 4,000 bad outputs into nine named failure modes. This is a working technique, not trivia — it is the single most useful classical-ML tool in an applied AI job.

**⚠ Trap:** skimming this into contrarianism. "Just use a regex" is as bad an answer as reflex-LLM if the input is genuinely unstructured natural language across 200 formats. The senior position is the *gate*, not a preference: state the condition, then choose.

**🗣 Say this in the room:** "Before I design this I'd want to know the cardinality of the output space and how often it changes. If it's a fixed set of twelve intents that changes quarterly, this is a classifier and an embedding cache — about $12 a month and 2 milliseconds. If the intent space is open and shifts weekly, then the LLM's zero-shot generality is worth paying 150× for, and I'd still put a cached classifier in front of it for the head of the distribution."

### How much CNN and RNN history do I actually need?

One page, and I mean that literally — you should be able to write the whole thing on one page and then never open it again. The reason it is on the skim list rather than the skip list is that a specific kind of interviewer, usually a research-adjacent one at Google or DeepMind or Meta, will ask "why did transformers win?" as a warm-up, and a candidate who cannot answer it in mechanistic terms reveals that their knowledge starts at the API.

Here is the one page.

**RNNs and their defect.** A recurrent net processes a sequence one step at a time, carrying a hidden state. Two consequences follow, and both are fatal. Sequentially, you cannot parallelize across the time dimension during training — step `t` needs step `t−1` — so your GPUs idle. And the gradient from position `T` back to position `1` is a product of `T` Jacobians; if their spectral radius is below 1, the signal vanishes exponentially, and if above 1, it explodes. **📄 Paper:** Hochreiter & Schmidhuber (1997) — LSTM added multiplicative gates and an additive cell-state path so gradients could flow without repeated multiplication, which mitigated vanishing but did not touch the parallelism problem. GRU (2014) is a cheaper two-gate variant of the same idea.

**Attention's actual origin.** **📄 Paper:** Bahdanau et al. (2014) — added a learned soft alignment so a translation decoder could look at *all* encoder states rather than a single fixed-size context vector. This is the moment attention was invented, and it was a fix for the seq2seq bottleneck, not a new architecture. Luong et al. (2015) simplified the scoring function. The transformer's contribution was to notice that if attention is good enough, you can **delete the recurrence entirely** and keep only attention — **📄 Paper:** Vaswani et al. (2017).

**Why transformers won, in two sentences.** Parallelism over the sequence dimension: every position's representation at layer `L+1` is computed simultaneously from all positions at layer `L`, so a training step is a few big matmuls instead of `T` sequential small ones, and GPUs are matmul machines. And O(1) path length: any two positions are one attention hop apart, so gradient signal between distant tokens does not decay with distance the way it does through `T` recurrent steps. The price is O(T²) attention cost, which the field has spent a decade paying and optimizing.

**Everything else on the page, as one-liners.** Teacher forcing and exposure bias — training on ground-truth prefixes, generating on your own prefixes, hence compounding error. word2vec/GloVe give one static vector per word type; contextual models give one per token occurrence, which is why "bank" finally works. BERT is encoder-only with masked-LM, good for classification and embeddings; GPT is decoder-only causal, good for generation; T5 is encoder-decoder with span corruption. CNNs: convolution is weight sharing plus locality; ResNet's skip connection is the same residual idea the transformer uses; ViT showed you can just cut an image into patches and feed a transformer.

**⚠ Trap:** saying transformers won "because attention is better than recurrence at capturing long-range dependencies." That is the folk answer and it is only half right — LSTMs handle long range tolerably. The decisive advantage was training throughput. If you get this wrong in front of someone who trains models, you have signalled that you learned this from a blog post.

### And GPU scheduling in Kubernetes — you called that delta-only. What's the delta list?

Short, and I would timebox it to two hours because past that you are studying an infra role you are not interviewing for.

The delta is five items. **Extended resources and the device plugin** — GPUs are integers, allocated whole, advertised by a DaemonSet, and the scheduler has no idea what a "GPU-hour" is. **Sharing modes** — MIG for hardware-partitioned isolation, time-slicing for oversubscription without memory isolation, MPS for concurrent kernels with weak isolation; know which failure each one permits. **Gang scheduling** — an 8-GPU distributed job must get all 8 or none, or you deadlock with two half-placed jobs each holding what the other needs; this is why Volcano, Kueue and similar batch schedulers exist on top of the default one. **Topology awareness** — 8 GPUs on one node connected by NVLink is a completely different machine from 8 GPUs spread across two nodes over PCIe and Ethernet; a collective operation on the latter can be an order of magnitude slower, so placement is a correctness-of-performance concern. **Node fragmentation and drain semantics** — evicting a pod holding a 90-second-to-load model is not the same as evicting a stateless web pod, so your PDBs and preStop hooks need to account for warm-up.

**⚠ Trap:** conflating training-cluster scheduling with inference-cluster scheduling in an answer. They have opposite objectives. Training wants gang-scheduled, long-lived, checkpointed, preemptible-with-resume jobs where the metric is cluster utilization. Inference wants low-latency, elastic, warm-pooled replicas where the metric is SLO attainment and idle capacity is deliberately purchased. Saying "we'd use Kubernetes for GPU scheduling" without picking one of these tells the interviewer you have done neither.

**🧪 Verify before you skip the rest:** (1) Two 4-GPU jobs, one 8-GPU node with 6 free GPUs across two nodes — what does the default scheduler do and why is it wrong? (2) Why does NVLink vs PCIe placement change your effective throughput on a tensor-parallel deployment? (3) What does `nvidia-smi`'s utilization percentage actually measure, and why is it a bad autoscaling signal?

### Of everything on the ATTACK list, why do you put transformer internals first?

Because it is the only item on the list where fifteen years of shipping software buys you nothing, and because it is *load-bearing* for three other items. You cannot reason about KV-cache budgeting without knowing what K and V are and why queries are not cached. You cannot understand why PagedAttention exists without understanding that the cache is per-sequence and grows. You cannot evaluate a quantization scheme without knowing where the outliers live in the activation distribution. Attack it first because everything downstream compiles against it.

"Attack" has a specific operational meaning here, and it is not "read more." It is: **you can produce the thing unaided, from memory, on a blank page, in a fixed time.** The pass criteria for this topic, concretely —

- Write single-head scaled dot-product attention with a causal mask in NumPy or PyTorch, ~15 lines, no reference, in under 10 minutes.
- Extend it to multi-head with correct reshapes and transposes, writing `[B, H, T, d_h]` shapes at every line, in another 10.
- Derive why the scale is √d_k from the variance of a dot product of two `d_k`-dimensional vectors with unit-variance components (the dot product has variance `d_k`, so dividing by √d_k restores unit variance and keeps softmax out of its saturated regime).
- Explain a full block's data flow — embed, RoPE, attention, residual, RMSNorm, SwiGLU FFN, residual — and say which parts are per-token and which are per-pair.
- State parameter counts: attention is ~4·d² per layer (Q, K, V, O), FFN is ~2·d·d_ff, and with the standard `d_ff = 4d` that is 8d², so **FFN is roughly two-thirds of the parameters** and attention one-third. That single ratio kills a lot of muddled thinking.

**🏋 Drill:** 25 minutes, no autocomplete, no reference. Blank file. Implement multi-head causal self-attention, then a full transformer block with RMSNorm and a SwiGLU FFN. Then, on paper, compute the KV cache bytes per token for a model with 32 layers, 8 KV heads and `d_head` 128 in bf16. Pass criterion: the code runs on a random `[2, 16, 512]` input and produces the right output shape, AND your KV number is exactly 128 KiB with the derivation written out. Repeat weekly until it is boring.

**⚠ Trap:** learning the transformer through a diagram instead of through shapes. Diagrams with boxes and arrows produce a candidate who can *narrate* attention and cannot *write* it. The distinguishing question at the whiteboard is always a shape question — "what's the shape after the transpose?" — and the diagram does not encode it.

### Same question for KV-cache math. Why does that get its own attack slot rather than living inside transformers?

Because it is the highest-frequency arithmetic in the entire interview surface and it is asked in a distinctive way: **out loud, in your head, with the interviewer watching.** It shows up in serving rounds, in system design rounds, in cost questions, in capacity questions. A candidate who can do it fluently reads as someone who has operated a model; one who reaches for a calculator reads as someone who has read about one.

The core formula is short enough to be permanent: `bytes_per_token = 2 · n_layers · n_kv_heads · d_head · dtype_bytes`. The leading 2 is K and V. Notice what is absent — `d_model`, query head count, `d_ff`, and total parameter count. That absence is the whole point, and it is why GQA works.

**📐 Numbers you must know:** anchor on Llama-3-8B — 32 layers, 8 KV heads, `d_head` 128, bf16. `2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB per token`, so **1 GiB per 8k-token sequence**, exactly. Every other model is a ratio from that anchor. Memorize the anchor, derive the rest.

**💰 Math you should be able to run in 30 seconds under questioning:** an 80 GB H100 serving Llama-3-8B in bf16. Weights: 8B × 2 bytes = 16 GB. Reserve ~4 GB for activations, CUDA context and fragmentation. Free for KV: 80 − 16 − 4 = **60 GB**. At 128 KiB/token that is 60 × 1024 × 1024 / 128 = **491,520 tokens of cache**. At an average 4k-token context per request, that is ~120 concurrent sequences. Push average context to 32k and it collapses to ~15. **The same hardware, the same model, an 8× concurrency swing driven entirely by context length** — and that is the number product managers never see when they ask to "raise the context limit."

**⚠ Trap:** computing the single-sequence number, finding it comfortable, and concluding you have capacity. The KV cache is the one resource in a serving deployment that scales with traffic *and* context *multiplicatively*. Engineers who size on the single-sequence figure ship a service that OOMs at a concurrency the load test never reached because the load test used short prompts.

**🗣 Say this in the room:** "Decode is memory-bandwidth-bound, not compute-bound, and the KV cache is why. It's a per-request memo table whose eviction policy I don't control, it doesn't amortize across the batch the way weights do, and it grows linearly with context — so my capacity model is `free_HBM / (bytes_per_token × p95_context)`, not requests per second."

### Serving-engine internals is on the attack list, but I'm not going to be writing vLLM. Why does it matter for an applied role?

Two reasons, and the second is the real one.

The first is that you will make deployment decisions that depend on it: whether to self-host or use an API, which engine, what batch and context limits to configure, what to autoscale on, whether speculative decoding is worth the complexity for your workload. Those decisions have five- and six-figure consequences and they are unanswerable without the internals.

The second reason is that **this is the round where your backend experience converts into visible seniority, and nothing else on the list does that as efficiently.** Every serving concept is a systems concept you already own, wearing an ML costume:

- **Continuous batching** is a work-stealing scheduler that admits new requests at token granularity instead of waiting for the whole batch to finish. **📄 Paper:** Yu et al. (2022), Orca — introduced iteration-level scheduling, replacing static batching where a batch of 32 runs at the speed of its longest member. You have built the equivalent for a job queue.
- **PagedAttention** is virtual memory for the KV cache. **📄 Paper:** Kwon et al. (2023), vLLM — the KV cache is stored in fixed-size blocks with an indirection table instead of one contiguous per-sequence buffer, which eliminates internal fragmentation from over-allocating to `max_seq_len` and enables copy-on-write sharing of a common prefix. If you have ever explained why a slab allocator beats malloc for fixed-size objects, you already have this argument; only the nouns change.
- **Prefix caching / RadixAttention** is a shared-prefix trie over KV blocks, so a 12k-token system prompt reused across 200k daily calls is computed once. **📄 Paper:** Zheng et al. (2024), SGLang — RadixAttention with an LRU eviction policy over the radix tree.
- **Prefill vs decode** is the single most important asymmetry: prefill processes all input tokens in parallel and is compute-bound; decode emits one token at a time and is memory-bandwidth-bound. They contend for the same GPU, which is why a long prefill stalls everyone else's inter-token latency, and why disaggregating them onto separate pools is now a mainstream architecture.

**🗣 Say this in the room:** "PagedAttention is virtual memory for the KV cache — fixed-size blocks plus a page table, so you stop pre-allocating to max context and stop fragmenting. The win isn't a faster kernel, it's that you can run 2–4× more concurrent sequences on the same card because you're no longer wasting the gap between allocated and used."

**⚠ Trap:** describing continuous batching as "dynamic batching." Dynamic batching (the Triton-style kind) waits a few milliseconds to assemble a batch and then runs it to completion — every request in that batch finishes when the longest one does. Continuous batching evicts and admits *between token steps*. The difference in tail latency is enormous, and using the wrong term signals you learned this from a vendor page.

### Quantization made the attack list too. What's the bar there for an applied engineer?

The bar is: you can say which quantization scheme you would use, what it costs you in quality, where the risk actually lives, and how you would *measure* whether it was acceptable. You are not expected to write a CUDA kernel.

The mental model that makes everything else fall out: **quantization is a lossy compression of the weights and/or activations, and it buys you memory bandwidth, not FLOPs.** Since decode is bandwidth-bound, halving the bytes you stream per token roughly halves your decode time — that is the entire economic case. It also buys you HBM, which converts directly into KV cache headroom and therefore concurrency, which is often the larger win.

The taxonomy worth holding:

- **Weight-only, post-training.** GPTQ (**📄 Paper:** Frantar et al., 2022 — layer-wise second-order-informed rounding) and AWQ (**📄 Paper:** Lin et al., 2023 — protect the ~1% of weight channels that matter most, identified by activation magnitude) take a trained model to 4-bit weights with small quality loss and no retraining. This is the default for self-hosting.
- **Weight-and-activation.** Harder, because activations have outliers. **📄 Paper:** Dettmers et al. (2022), LLM.int8() — found that a small number of outlier feature dimensions dominate and must be kept in higher precision. **📄 Paper:** Xiao et al. (2022), SmoothQuant — migrates activation outlier difficulty into the weights via a per-channel scaling, making both quantizable. FP8 on Hopper-class hardware and FP4 on Blackwell-class make this much easier because the format has an exponent and handles outliers gracefully. (**📅 Volatile:** hardware format support moves generation to generation; verify.)
- **Quantization-aware training and QLoRA.** **📄 Paper:** Dettmers et al. (2023), QLoRA — 4-bit NF4 base weights frozen, LoRA adapters trained in bf16 on top, which is what makes fine-tuning a large model on one GPU feasible.
- **KV-cache quantization.** Separate axis, often the highest-value one. Taking the cache from fp16 to fp8 halves bytes per token and therefore doubles concurrency.

**💰 Math:** Llama-3-70B in bf16 is 140 GB — two 80 GB cards minimum, with almost no room for KV. At 4-bit it is ~35 GB, which fits on one card with ~40 GB left for cache. At 320 KiB/token that is 40 × 1024 × 1024 / 320 = **131,072 tokens** of cache, or ~32 concurrent 4k sequences on a single card. You went from a 2-GPU deployment serving few sequences to a 1-GPU deployment serving many. On a $2/GPU-hour rental that is $1,440/month saved on hardware alone, before the throughput gain.

**⚠ Trap:** benchmarking a quantized model on perplexity and declaring victory. Perplexity is remarkably insensitive to quantization damage; the degradation concentrates in long-context reasoning, multi-step arithmetic, code generation, and — most treacherously — instruction adherence and tool-call format compliance. I have seen a 4-bit model with near-identical perplexity emit malformed tool arguments at 3× the rate. **The rule I enforce: quantization is validated on your task eval, not on a language-modelling metric.**

### Post-training and RLVR — how deep does an applied AI engineer need to go?

Deep enough to hold a real conversation and to *not fine-tune*. That framing sounds glib; it is the actual bar.

You need the map, not the terrain. Pretraining gives you a next-token predictor. Supervised fine-tuning on instruction/response pairs gives you something that follows instructions. Preference optimization aligns it to human taste: **📄 Paper:** Ouyang et al. (2022), InstructGPT — RLHF as reward model plus PPO, which is what made GPT-3.5 usable; **📄 Paper:** Rafailov et al. (2023), DPO — showed the RLHF objective has a closed form that lets you optimize directly on preference pairs with no separate reward model and no RL loop, which is why most teams outside frontier labs use DPO. And then the 2024–2025 shift: **RLVR**, reinforcement learning from *verifiable* rewards, where the reward is not a learned preference model but a program — unit tests pass, the math answer matches, the JSON parses, the SQL returns the right rows. **📄 Paper:** Shao et al. (2024), DeepSeekMath — introduced GRPO, which drops the value network and normalizes advantages within a group of sampled completions, making RL affordable. DeepSeek-R1 (2025) showed rule-based verifiable rewards at scale producing long-chain reasoning behaviour.

Why RLVR matters to *you*, in an applied role, is the thing to actually say: **RLVR's constraint is that you need a verifier, and building verifiers is engineering, not research.** That is your job. If your product domain has a checkable ground truth — code that compiles and passes tests, a query that returns the right rows, an extraction that matches a database — you have the substrate for RL. If your domain is "write a good summary," you do not, and no amount of GPU budget fixes that. Being able to draw that line is the whole competency.

**⚠ Trap — the reflex-fine-tuning rejection trigger.** The single fastest way to be downgraded in an applied AI loop is to answer "the model isn't good at our domain" with "we'd fine-tune it." It is the answer of someone who has read about ML and not shipped it. Fine-tuning teaches *form*, not *facts*: it is excellent at style, format, tone, tool-call conventions and domain vocabulary, and it is a bad and expensive way to install knowledge that changes — that is retrieval's job. Before fine-tuning is legitimate you need, named explicitly: a working eval, a prompt/context baseline you have already exhausted, at least a few thousand high-quality examples, a data-refresh story, and a serving story for the adapter.

**🗣 Say this in the room:** "I'd hold fine-tuning until last. In my experience the first 80% of a domain gap is prompt and context engineering, the next 15% is retrieval and tool design, and fine-tuning is for the residual — consistent output format, house style, a tool-calling convention the base model keeps getting wrong. It teaches the model how to behave, not what's true, and anything that changes weekly has to come from retrieval or it's stale the day after you train."

### Why do you call evaluation "the spine" rather than one more topic?

Because it is the only topic that appears in every other round. A retrieval design answer without an eval is a guess. A prompt-optimization answer without an eval is superstition. A model-migration answer without an eval is a coin flip with your production traffic. And empirically, "skipped evaluation of AI outputs" is the named #1 critical error on take-home rubrics — which means it is not a knowledge gap, it is a *habit* gap, and habits need to be built early rather than crammed.

The habit is: **open every design answer with how you would know whether it works.** Not close with it. Open with it. When an interviewer says "design a support agent," the first sixty seconds should be "before I pick an architecture, here's what I'd measure and how I'd build the dataset" — because that reorders everything downstream and it immediately separates you from candidates who describe a box diagram.

What to actually attack, in order:

**Dataset construction**, because it is the part everyone skips. Where do the 200 labelled examples come from? Production logs, stratified by intent and by whether the current system succeeded. Who labels them and what is the inter-annotator agreement? What is your holdout discipline, and are you sure your eval set has not leaked into your prompt? A candidate who talks about dataset construction before metrics has already passed.

**Metric selection under the task.** Exact match and pass@k where there is ground truth. Retrieval metrics (Recall@k, nDCG, MRR) for the retrieval stage measured *separately* from generation, because otherwise you cannot localize a regression. LLM-as-judge where there is no ground truth, with its own validation.

**LLM-as-judge, validated.** **📄 Paper:** Zheng et al. (2023) — established judge models as a scalable proxy and documented their biases: position bias (favouring the first response), verbosity bias (favouring longer answers), self-enhancement bias (favouring their own family's outputs). The discipline is: pairwise beats absolute scoring, randomize position, calibrate the judge against a few hundred human labels and report the agreement rate, and re-calibrate whenever you change judge model. A judge you have not measured against humans is a random number generator with good prose style.

**Statistics.** You are comparing two prompts on 200 examples and one scores 74% and the other 78%. Is that real? Paired bootstrap or McNemar's test on the paired outcomes, because the examples are shared. Without this you will ship noise and then debug the wrong thing for a week.

**🏋 Drill:** 30 minutes. Take any feature you have shipped in the last year and write its eval plan on one page: the task, the dataset and its provenance, the primary metric and why, the guardrail metrics, the sample size needed to detect a 3-point difference, and the failure taxonomy you would tag against. Pass criterion: someone else could execute it without asking you a question.

### Agent harness engineering is the last big attack item. What does "harness" mean, and why isn't this just framework knowledge?

The harness is everything around the model call: the loop, the tool registry and schemas, the context assembly, the error surfaces, the termination conditions, the budget accounting, the persistence, the replay. The model is one function call inside it. The framing I would use in a room is that **roughly 20% of a production agent system is the LLM; the other 80% is engineering you already know how to do** — and the interview is testing whether you know which 80%.

It is not framework knowledge because the frameworks make the easy part easy and leave the hard part to you. LangGraph will give you a state machine; it will not tell you that your tool error strings are the highest-leverage prompt surface in the system, or that you need a per-run token budget with a hard kill, or that a tool returning 40k tokens of JSON has just destroyed your context and your latency. The core loop itself is small enough to write from memory, and you should be able to:

```python
# The whole pattern, ~25 lines. If you can write this, framework choice is a detail.
def run(task, tools, max_steps=12, token_budget=60_000):   # tools: {name -> Tool}
    messages = [{"role": "user", "content": task}]
    used = 0
    seen = collections.Counter()
    for step in range(max_steps):
        resp = model.call(messages, tools=[t.schema for t in tools.values()])
        used += resp.usage.input_tokens + resp.usage.output_tokens
        if used > token_budget:
            return Halt("budget", messages)
        if not resp.tool_calls:
            return Done(resp.text, messages)
        messages.append(resp.assistant_message())
        for call in resp.tool_calls:
            key = (call.name, canonical(call.args))
            seen[key] += 1
            if seen[key] > 3:                      # poison-trajectory guard
                return Halt("loop", messages)
            try:
                out = tools[call.name].run(**call.args)
            except ToolError as e:
                out = f"Error: {e}. {tools[call.name].recovery_hint}"   # error-as-prompt
            messages.append(tool_result(call.id, truncate(out, 4000)))
    return Halt("max_steps", messages)
```

Everything interesting is in that snippet's guards: the budget check, the repeat-call counter, the error string written as an instruction, the truncation. Those are the four things that separate a demo from a system, and none of them is in a framework tutorial.

**⚠ Trap:** believing agent quality is a prompt problem. In production the dominant failure causes are, in my experience and in that order: tool schemas the model misreads, tool errors returned as opaque codes, context blown out by a verbose tool result, and no termination guard. All four are engineering defects with engineering fixes. Candidates who answer "we'd improve the system prompt" to an agent-failure question are describing the least effective lever.

### What about research literacy and product sense — how do you attack something that soft?

They are the two items where "attack" means building a repeatable artifact rather than accumulating knowledge, so treat both as habits with a weekly cadence.

**Research literacy** is not "have you read every paper." It is: given a new paper, can you extract in fifteen minutes what it replaces, what it costs, and whether it changes any decision you would make? The artifact is a one-paragraph template you fill for every paper you read — *what was the state of the art before; what is the single mechanism change; what does it cost in compute, memory or quality; what would have to be true for me to adopt it.* Do that for the twenty papers this guide names and you have research literacy, because interviewers do not quiz you on paper contents, they ask "have you seen anything interesting lately?" and grade the shape of your answer.

The genuinely useful move here is knowing the *lineage* rather than the leaves: attention → transformer → scaling laws → Chinchilla's compute-optimal correction → instruction tuning → RLHF → DPO → RLVR is one coherent story, and being able to tell it as a story with the pressure that motivated each step is far more impressive than reciting eight abstracts.

**Product sense** is the round Cursor, Notion, Figma, Sierra, Harvey, Glean and Perplexity weight most heavily and that backend engineers underrate most badly. It is testable, and the test is usually one of three questions: "what would you build next for us," "what AI product do you think is badly designed and why," or "how would you know this feature was working." The competency underneath all three is *knowing which failure modes users forgive and which they do not*. Users forgive latency far more than they forgive confident wrongness. They forgive a refusal more than a fabrication. They forgive a bad answer they can see is bad; they do not forgive a plausible one that is wrong. This asymmetry should drive your design choices — it is why citations, showing retrieved sources, and "I don't know" paths are product features and not safety theatre.

**🏋 Drill:** 15 minutes weekly, unaided. Pick one AI product you use. Write: the three failure modes you have personally hit, which of them the team clearly knows about (evidenced by UI affordances), what you would instrument to catch the third one, and one feature you would cut. Pass criterion: no sentence in it could apply to a different product.

**🗣 Say this in the room** when asked what you would build: "I'd start from the failure mode users punish hardest, which for this product is confident wrongness on a question they can't verify. So the first thing I'd ship isn't a feature, it's the instrumentation to tell how often that happens — because right now nobody in this conversation, including me, can name the number."
### You've built Redis rate limiters before. Design the equivalent for an LLM gateway serving 40 internal teams.

Start from what changed, because the algorithm did not. A token bucket is still a token bucket. What changed is **the unit of the bucket and the fact that you cannot measure the withdrawal until after you have spent it.**

Your HTTP rate limiter counts requests, and a request's cost is known at admission. Here the provider limits you on three axes simultaneously — requests per minute, *input* tokens per minute, and *output* tokens per minute — and the dominant one is almost always tokens. A single request can consume 200 tokens or 200,000. So a `Semaphore(100)` or a 1,000-RPM limit is not merely imprecise, it is measuring the wrong quantity: one tenant sending 100k-token documents will saturate your token budget at 5 requests per minute while your request-counter reports 0.5% utilization.

The design that works:

**Bucket in tokens, admit on an estimate, reconcile on the actual.** At admission you know input tokens exactly (tokenize, or use the provider's counting endpoint, or estimate at ~4 characters/token for English and be conservative). You do *not* know output tokens, so you reserve `max_tokens` — the ceiling the caller requested — and refund the difference when the response completes. This is a two-phase commit against a bucket, and it is the whole trick.

```lua
-- Atomic reserve against a per-tenant token bucket. KEYS[1]=bucket, ARGV: now, rate, burst, want
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or tonumber(ARGV[3])
local ts     = tonumber(b[2]) or tonumber(ARGV[1])
tokens = math.min(tonumber(ARGV[3]), tokens + (tonumber(ARGV[1]) - ts) * tonumber(ARGV[2]))
if tokens < tonumber(ARGV[4]) then
  redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', ARGV[1])
  return {0, math.ceil((tonumber(ARGV[4]) - tokens) / tonumber(ARGV[2]))}  -- deny + retry_after_s
end
redis.call('HMSET', KEYS[1], 'tokens', tokens - tonumber(ARGV[4]), 'ts', ARGV[1])
redis.call('EXPIRE', KEYS[1], 600)
return {1, 0}
```

Then `HINCRBYFLOAT bucket tokens <refund>` when the actual output count arrives.

**Three things beyond the algorithm that the interviewer is actually probing:**

*The upstream bucket is shared and you do not own it.* Your per-tenant buckets must sum to less than your provider allocation, with headroom, or you have built a fair queue that fairly distributes 429s. Read the provider's rate-limit response headers (`x-ratelimit-remaining-tokens` / `anthropic-ratelimit-tokens-remaining` families, plus `retry-after` — **📅 Volatile**, header names differ per provider and change) and feed them back as a *measured* ceiling rather than trusting a configured constant.

*Queue, don't reject, for async traffic.* A 429 to a human-facing chat request is correct. A 429 to a batch enrichment job is waste — that work should sit in a priority queue and drain at whatever rate the bucket refills, with interactive traffic given strict priority. This is classic multi-class scheduling and it is where your background pays off.

*Fairness needs a work-conserving policy.* Strict per-tenant caps leave capacity idle when tenants are quiet. Deficit round-robin or weighted fair queueing over the shared token budget lets a quiet-hours tenant burst into unused capacity without letting them starve anyone at peak.

**💰 Math:** suppose your org tier gives 2,000,000 input tokens/minute. Forty teams, naive equal split = 50k TPM each. One team runs a document pipeline at 100k-token requests: they can issue 0.5 requests/*minute* — one request every two minutes — and are permanently throttled while 39 teams idle. Now the same budget under weighted fair queueing with a 200k burst allowance: the pipeline team drains unused capacity at 3–4× their nominal share overnight and is capped back at peak. Same money, ~4× the useful work out of the same tier.

**⚠ Trap:** rate-limiting on requests because that is what your middleware already does. It passes review, it looks correct on the dashboard, and it fails the first time a tenant changes their prompt length. **The rule I enforce: any limiter in front of a model call is denominated in tokens, or it is decorative.**

### Walk me through the LLM version of a dead-letter queue. What goes in it and how does it get there?

The mental model shift is that in your queue systems the poison is in the *message*, and here the poison is in the *trajectory*. Every individual step is well-formed and would pass any payload validator you wrote. The pathology is in the sequence: the model calls `search_tickets(status="opne")`, gets a validation error, apologizes, and calls `search_tickets(status="opne")` again. Sixty times. There is nothing malformed to quarantine on and no exception is ever raised.

So detection must be state-based, and there are exactly four signals worth wiring:

**Repeat-action detection.** Hash `(tool_name, canonicalized_args)` at every step into a per-run counter. Canonicalization matters — sort keys, normalize whitespace and casing, round floats — or the model's cosmetic variation defeats you. Threshold at 3 identical calls, or 5 near-identical ones by edit distance. This catches the majority of real loops.

**Budget exhaustion.** Per-run caps on steps (12–20 is typical), on cumulative tokens, and on wall-clock. Every one of these should be a *hard* kill with a recorded reason, not a soft warning. The step cap alone is insufficient because a single step with a 200k-token tool result can blow the whole budget.

**Non-progress detection.** Harder and worth building for high-value agents: track whether the agent's *state* changed — files written, rows returned, a distinct URL fetched. Ten steps with no state delta is a stall even if every call was distinct.

**Terminal-error classification.** Distinguish retryable (429, 503, transient tool timeout) from terminal (401, schema violation the model keeps repeating, tool that does not exist). Terminal errors should abort the run, not be fed back for another attempt, because the model cannot fix a missing credential by rewording.

What goes in the DLQ is different too. A poison message goes to the DLQ as a payload; a poison trajectory goes to a review queue as **the entire transcript plus the run's cost**, because the transcript *is* the debugging artifact and because the transcript is what you mine to build eval cases. This is the highest-value part of the whole pattern: your poison-trajectory queue is also your eval dataset generator. A run that looped is, by definition, a case your system handles badly and should be regression-tested against.

**🔍 Failure taxonomy — run this as a decision procedure when an agent misbehaves:**

1. *Did it loop on an identical call?* → tool error message is uninformative. Fix the error string first; it is the cheapest and most effective lever.
2. *Did it loop on varying calls to the same tool?* → schema is ambiguous. The model is guessing at parameter semantics. Fix the description and add an enum.
3. *Did it stop early with a confident wrong answer?* → termination condition too permissive, or the tool returned an empty result the model read as "no such thing exists" rather than "query failed."
4. *Did it blow the token budget in 2 steps?* → a tool result is unbounded. Truncate and paginate at the tool boundary, never at the context boundary.
5. *Did it call a tool that does not exist?* → too many tools in the registry, or names too similar. Consolidate.

**⚠ Trap:** retrying a poison trajectory from step zero. You just paid the full cost again for a deterministic failure with a nonzero chance of the same loop. Poison trajectories are not retried; they are killed, recorded, and — if the domain allows — resumed *from the last good state* with the failing tool disabled or with human input injected.

### Our RAG answers went stale after a reindex and nobody noticed for a week. What should the rollout have looked like?

This is your Postgres index migration, and the reason it bit you is that a vector index fails *silently* in a way a B-tree does not. If you break a B-tree, queries error or plans go sequential and latency screams at you. If you break a vector index — by mixing embedding spaces, by building HNSW with the wrong `ef_construction`, by rebuilding while writes are still landing — the query still returns exactly `k` rows, in a plausible order, with plausible scores. **The failure surface has no error signal at all.** That is why "nobody noticed for a week" is the expected outcome, not a team failure.

The rollout you want is the alias-swap you already know, with two additions that are specific to embeddings.

```
1. Freeze schema: new index/table/collection `docs_v2`, alias `docs_live` still → v1.
2. Backfill v2 from a durable queue with a watermark. Resumable. Idempotent per chunk_id.
   Track: chunks_total, chunks_embedded, chunks_failed, oldest_unembedded_updated_at.
3. Do NOT flip on "backfill complete." Flip on evidence.
4. Shadow read: for 1–5% of live queries, run both indexes, log
   overlap@10 (|topk_v1 ∩ topk_v2| / 10) and score distributions. Do not compare latency only.
5. Golden-set eval: a frozen set of ~200 (query, known-relevant-doc-ids) pairs.
   Measure Recall@10 and nDCG@10 on v1 and v2. v2 must be >= v1 minus a stated tolerance.
6. Flip alias. Keep v1 warm for the rollback window (I use 7 days).
7. Continuously: emit oldest_document_indexed_at as a gauge. This is your staleness SLI.
```

Step 4 and step 7 are the ones teams skip and the ones that would have caught your incident.

The staleness metric deserves emphasis because it is the single most under-instrumented signal in production RAG. You want a gauge — `now() − min(indexed_at)` over documents whose `updated_at > indexed_at` — alarmed at whatever your product's freshness promise is. It is exactly Kafka consumer lag, applied to an indexing pipeline, and framing it that way in an interview lands well because it shows you are reusing rather than reinventing.

**⚠ Trap:** the two-embedding-model incident, which is the specific version of this that keeps happening. You upgrade the embedding model, redeploy the *query* path, and the index still holds vectors from the old model. Cosine similarity between two different embedding spaces is not an error; it is a number, usually in a reasonable range, ranked meaninglessly. Retrieval quality collapses and every dashboard stays green. **Defence: store the embedding model ID and dimension as a column on every vector row, and have the query path assert equality at read time.** One cheap assertion prevents the whole class.

**💰 Math on the rollback window:** 5M chunks × 1536 dims × 4 bytes = 30.7 GB of raw vectors per version, plus HNSW graph overhead (call it 40% at `m=16`) ≈ 43 GB per version, ≈ 86 GB while both are live. On a managed Postgres with 128 GB RAM that is the difference between the index sitting in shared buffers and thrashing to disk — so a shadow window is a *capacity* decision you plan for, not a free safety net. If you cannot hold both, the honest alternative is a read-only maintenance window, and you should say so rather than pretending.

### You keep Kafka consumer lag on a dashboard. What's the equivalent leading indicator for an inference deployment?

Consumer lag works because it is a *leading* indicator with a physical meaning: unprocessed work is accumulating and you can see the backlog growing before latency degrades. Almost every metric people put on an LLM autoscaling dashboard is *lagging* — request latency, error rate, GPU utilization — which means you scale after users have already had a bad experience, and given 60–90 second cold starts, "after" is fatal.

The genuine equivalents, in order of usefulness:

**KV-cache utilization** is the primary one, and it maps to lag almost exactly. The serving engine knows what fraction of its KV block pool is allocated. As utilization climbs toward 100%, the scheduler stops admitting new sequences and starts *preempting* — vLLM will swap or recompute evicted sequences, which is a cliff, not a slope. Alarm at 70–80%, scale at 80%, because you need the new replica to be warm before you hit 95%. This is the number I would put in the middle of the dashboard.

**Waiting-queue depth and time-in-queue.** The engine's pending-request queue is literally a backlog. Time-in-queue rising while GPU-side step time stays flat means you are admission-limited, not compute-limited, and adding replicas will help. If step time is rising too, you are compute-limited and adding replicas helps differently (and you may instead need to cap batch size or context).

**Preemption / recompute rate.** Any nonzero preemption is capacity pain. It is the equivalent of seeing rebalances in a consumer group — the system is doing work that is not your work.

**⚠ Trap — and this is the one that most reliably separates candidates:** autoscaling on GPU utilization. `nvidia-smi`'s utilization figure reports the fraction of sampled intervals during which at least one kernel was executing. During memory-bandwidth-bound decode, kernels are essentially always resident, so it reads 90–100% whether you are serving 3 sequences or 200. **It is a boolean dressed as a percentage.** A deployment autoscaled on it will sit pinned at "fully utilized" and never scale, or will scale on nothing. Use `DCGM_FI_PROF_*` occupancy metrics if you must have a hardware signal, but prefer the engine's own scheduler metrics — they are the ones that know about queueing.

**💰 Math on why leading matters:** a 70B model at ~140 GB in bf16 pulled from an object store at 2 GB/s takes 70 s to load, plus container start and CUDA init — call it 90 s to serve traffic. Traffic doubles in 30 s. If you scale on p95 latency you begin scaling at t≈40 s and are healthy at t≈130 s, so you serve ~90 seconds of degraded traffic. If you scale at 80% KV utilization you begin at t≈12 s and are healthy at t≈102 s — but more importantly you can hold a warm pool sized off the same signal. The fix for cold start is never a faster loader; it is a leading indicator plus purchased idle capacity, and the interviewer wants to hear you say idle capacity is a *product* decision with a price tag.

### Translate "connection pool exhaustion" into this world for me.

It becomes **provider rate-limit saturation**, and the reason the analogy is worth carrying is that the observable symptoms are identical and the misdiagnosis is identical.

In your world: pool of 20, 200 concurrent requests, 180 coroutines blocked on `acquire()`. Your database looks healthy — low CPU, fast queries — and your app's p99 is catastrophic. The latency is entirely queueing at a resource whose depth is not in the trace. Every senior engineer has been burned by this once and then instruments pool wait time forever.

Here: your provider tier is 2M input TPM. You are at 1.9M. The provider is not erroring, is not slow, and its own latency metrics are perfect. But your client-side limiter (or the provider's) is queueing, so your measured TTFT includes several seconds of waiting for a token bucket to refill. The upstream looks healthy from *its* side. **You are measuring your own queue and attributing it to the model.**

The instrumentation that makes this diagnosable is the same instrumentation:

- Emit **time-in-limiter** as its own span, separate from time-to-first-token from the provider. Without this split, TTFT is an uninterpretable sum.
- Emit **429 rate and `retry-after` values** as a first-class metric, not as an error-log line. A rising `retry-after` is the provider telling you exactly how oversubscribed you are.
- Track **remaining-tokens headroom** from the response headers as a gauge. This is your "connections available" and it is the leading indicator.
- Tag every metric by tenant *and* by model, because the limits are per-model and one tenant's migration to a bigger model silently halves everyone's headroom.

The differences worth naming, because an interviewer will push:

*You cannot just raise the pool size.* Provider limits are a commercial negotiation with a tier-escalation process measured in days. So the mitigations are structural: route overflow to a second model or a second provider, shed low-priority traffic, move batchable work to a batch endpoint at roughly half price, and cache aggressively at the prefix level.

*Multi-account or multi-region load balancing is an ops decision, not a hack* — but it must be an explicit, contractually-clean one, and you should say that out loud rather than implying you would evade limits. The clean version is separate organizations for separate business units with separate quotas, which is how the provider expects large customers to operate.

*Failing open is worse here.* When a pool is exhausted you queue and the work eventually completes. When you are rate-limited and you "fail open" by retrying harder, you amplify the saturation — the classic congestion collapse. Retries against a 429 must be exponential with jitter and must respect `retry-after`, and your concurrency limiter must *decrease* on 429, AIMD-style, not stay constant.

**🗣 Say this in the room:** "This is connection-pool exhaustion with a different pool. The tell is that upstream latency looks perfect while our p99 is terrible, which means the wait is in our own admission queue. So the first thing I'd do is split TTFT into time-in-limiter and time-at-provider, because until those are separate spans nobody in the room can tell whether we have a capacity problem or a model problem."

### Your dashboards are all p50/p95/p99. What replaces that for a streaming LLM endpoint, and why isn't p95 latency enough?

Because a streaming response has no single latency. It has an onset and a rate, and users perceive those completely differently. A response that starts in 300 ms and streams for 8 seconds feels fast. A response that returns as one blob at 4 seconds feels slow. End-to-end p95 ranks the second one better. **Your existing metric literally inverts the user's preference ordering**, which is the sharpest reason to change vocabulary rather than just add jargon.

The three metrics that replace it:

**TTFT — time to first token.** Onset. Dominated by prefill, which is compute-bound and scales with input length, plus any queueing in your limiter or the engine's admission queue. This is the number that determines whether the product feels alive. For chat, I would set an SLO around 500 ms p95; for an agent step that a human is not watching, TTFT is nearly irrelevant and optimizing it is wasted effort.

**ITL — inter-token latency.** The distribution of gaps between consecutive tokens. This is what you alarm on for *stutter*: a p99 ITL spike means another request's long prefill just stalled your decode loop, which is the single most common cause of a "it froze mid-sentence" complaint.

**TPOT — time per output token.** The mean of that distribution, i.e. `(E2E − TTFT) / (output_tokens − 1)`. This is your throughput-per-user figure and it is bounded by memory bandwidth and batch size.

The relation you should be able to write instantly: `E2E ≈ TTFT + TPOT × (output_tokens − 1)`.

**📐 Numbers you must know:** human reading speed is roughly 250 words/minute ≈ 4 words/sec ≈ **5–6 tokens/sec**. So a TPOT of 200 ms (5 tok/s) is exactly reading speed; anything faster is invisible to a reading user and is only worth paying for when the output is being consumed by code, or when the user is skimming for a code block. This single number kills a lot of expensive optimization: **if a human reads the output, TPOT below ~50 ms buys nothing and you should spend that budget on TTFT or on quality instead.**

**💰 Math, worked:** a 500-token answer at TTFT 400 ms and TPOT 25 ms gives E2E = 0.4 + 0.025 × 499 = 12.9 s. Halving TPOT to 12.5 ms — which might cost you speculative decoding complexity or a second GPU — gives 6.6 s. But the user is reading at 5–6 tok/s and needs 500/5.5 ≈ 91 s to actually read it. **You spent a GPU to finish generating 6 seconds earlier into a 91-second reading task.** Halving TTFT from 400 ms to 200 ms, by contrast, is directly perceptible. That is the trade an interviewer wants you to make explicitly.

**⚠ Trap:** treating ITL and TPOT as synonyms in a room where someone serves models. TPOT is the mean; ITL is the distribution, and the whole point of tracking ITL separately is the tail. A deployment with excellent p50 ITL and terrible p99 ITL is a deployment with prefill/decode contention, and the fix is chunked prefill or prefill/decode disaggregation — a specific, nameable fix you can only reach if you kept the distribution.

### If QPS is the wrong throughput metric, what's the right one?

QPS is wrong because a "query" is not a unit of work here — one request might be 500 tokens of work and the next 500,000. Reporting QPS for an LLM service is like reporting "files per second" for a storage system without mentioning file size: technically a rate, operationally meaningless.

**Tokens per second** is the honest throughput unit, and it splits in two because the two phases have different economics. **Prefill tokens/sec** measures compute throughput and is roughly bounded by the GPU's matmul rate. **Decode tokens/sec** — aggregated across all concurrent sequences — measures how well you are amortizing weight reads across the batch, and it is bounded by memory bandwidth. A deployment can be excellent at one and terrible at the other, so a single aggregate number hides the actual bottleneck.

But the metric that actually belongs on the SLO dashboard is **goodput: the rate of requests completed *while meeting their latency SLO*.** This distinction is the one that gets you credit. Throughput and latency trade off through batch size: bigger batches raise total tokens/sec and raise per-request TPOT, because each sequence now waits behind more work per step. So you can always "improve throughput" by degrading everyone's experience, and a team optimizing raw tokens/sec will do exactly that and report it as a win. Goodput is throughput with the SLO as a constraint rather than an afterthought — it is the only number that cannot be gamed by making users wait.

**📄 Paper:** the goodput framing for LLM serving was made central by the prefill/decode disaggregation work — Zhong et al. (2024), DistServe — which argued that co-locating prefill and decode forces a single batch-size choice onto two workloads with opposite latency profiles, and that separating them onto distinct resource pools lets you tune each for its own SLO. Related: chunked prefill (Agrawal et al., 2024, Sarathi-Serve) splits a long prefill into slices interleaved with decode steps so a single 100k-token prompt does not stall every other user's stream.

**💰 Math showing the trade:** suppose at batch 8 your decode step takes 20 ms (TPOT 20 ms, aggregate 8 × 50 = 400 tok/s) and at batch 32 it takes 45 ms (TPOT 45 ms, aggregate 32 × 22 = 711 tok/s). Raw throughput is up 78%. But if your TPOT SLO is 30 ms, **goodput at batch 32 is zero** — every request violates. The correct answer is not "pick batch 8," it is "find the largest batch whose TPOT stays under 30 ms," which here might be 16, and then note that the remaining headroom should come from disaggregation or quantization, not from batching harder.

**🗣 Say this in the room:** "I'd stop reporting QPS. The unit of work is tokens, and the number I'd hold the team to is goodput — completed requests per second that met the TTFT and TPOT SLOs — because throughput and latency trade off through batch size, and any team measured on raw throughput will quietly buy it by degrading everyone's stream."

### Finance asks for cost per request. Why do you push back, and what do you give them instead?

Because cost per request is measurable, stable, and answers a question nobody cares about. A support agent that resolves a ticket in one $0.40 call is dramatically better business than one that resolves it in twelve $0.03 calls, and cost per request ranks them the wrong way round — it makes the twelve-call agent look 13× cheaper per unit while it burns more money and more of the user's time.

The number to hold is **cost per resolved task**: total spend divided by the count of tasks that reached a *verified* successful outcome. Everything hard about it is in the word "verified," which is exactly why it is a good metric — it forces you to define success, which forces you to build the eval, which is the discipline the whole loop is testing.

Build it as a chain:

```
cost_per_resolved_task = (Σ tokens_in·price_in + Σ tokens_out·price_out + retrieval + tools + human_escalation_cost)
                         ÷ (tasks × resolution_rate)
```

Two terms in there are the ones people forget and both are usually dominant. **Retries and failed trajectories are numerator, not noise** — a run that looped for 40 steps and was killed costs real money and resolved nothing. And **human escalation is a real cost line**: if 30% of tickets escalate to a human at a loaded cost of $6 each, that term swamps your token spend and it is the term your automation is actually trying to move.

**💰 Math, the comparison that makes the point:** two designs for a support agent, 100k tickets/month.

*Design A — single frontier call, big context.* 25k input tokens (full policy docs stuffed in), 600 output. At $3/M in, $15/M out: 25,000 × 3/1e6 + 600 × 15/1e6 = $0.0750 + $0.0090 = **$0.084/ticket**. Resolution rate 62%. Escalations: 38,000 × $6 = $228,000. Token spend 100,000 × $0.084 = $8,400. Total $236,400. **Cost per resolved task = $236,400 / 62,000 = $3.81.**

*Design B — retrieval + a 4-step agent.* Average 5 model calls, 4k input each, 300 output each: 5 × (4,000 × 3/1e6 + 300 × 15/1e6) = 5 × ($0.012 + $0.0045) = **$0.0825/ticket** — essentially identical token cost. Resolution rate 79%. Escalations: 21,000 × $6 = $126,000. Token spend $8,250. Total $134,250. **Cost per resolved task = $134,250 / 79,000 = $1.70.**

Cost per *request* says A is fractionally cheaper. Cost per *resolved task* says B is 2.2× better and saves **$102,150/month**. That is the argument, with digits, and it is the shape of answer that makes a hiring manager decide you have owned a budget. (**📅 Volatile:** unit prices move; recompute with current numbers before you cite them.)

**⚠ Trap:** defining resolution as "the user didn't reply again." That proxy is corrupted in both directions — users abandon when the answer is bad, and users stop replying when the answer is good. You need either an explicit signal (ticket closed by the customer, code merged, invoice matched) or a validated LLM-judge with a measured agreement rate against human labels. Say which one you would use and why; hand-waving here is the failure.

### Cache hit rate is second nature to you. What's the LLM-specific version and how do you move it?

Three different caches get called "cache" in this stack and conflating them is a common tell. Rank them by how much of the industry's actual savings they produce.

**Prefix cache hit rate** is the important one and the one you should name first. Your system prompt, tool schemas, few-shot examples and retrieved documents form a shared prefix across calls; the KV states for that prefix are deterministic given the tokens, so they can be computed once and reused. Self-hosted, that is vLLM's automatic prefix caching or SGLang's RadixAttention. Through an API, it is the provider's prompt caching, with cached input billed at a steep discount.

The lever that moves it is **prompt layout**, and this is the single most actionable piece of engineering in this whole answer: **order your prompt from most-static to most-dynamic.** System instructions, then tool definitions, then few-shot examples, then retrieved context, then conversation history, then the user turn. One timestamp, one user ID, one randomized example order near the top invalidates everything after it, because the cache is a prefix match — not a fuzzy match. I have seen a team's hit rate go from 8% to 91% by moving a `"Current time: ..."` line from the first sentence of the system prompt to the last message.

**💰 Math:** 12k-token system prompt + tools, 200k calls/day, $3/M input, cache reads at ~10% of base and cache writes at ~1.25× (**📅 Volatile** — Anthropic-style economics; OpenAI's automatic caching has used a different discount, verify both). Uncached: 12,000 × 3/1e6 = $0.036/call → 200,000 × $0.036 = **$7,200/day = $216k/month**. At a 90% hit rate: 10% of calls pay the write premium (0.1 × 200k × 12,000 × 3.75/1e6 = $900) and 90% pay the read rate (0.9 × 200k × 12,000 × 0.30/1e6 = $648), total ≈ **$1,548/day ≈ $46.4k/month**. **Saving ≈ $170k/month from reordering a prompt.** Nothing else in applied LLM engineering has that ratio of effort to dollars, which is why prefix-cache hit rate belongs on the main dashboard next to error rate.

**Exact-response cache** — hash the full normalized request, return the stored response. Safe, boring, and effective on the head of the query distribution. Hit rates of 15–40% are realistic for consumer products with a heavy head and near zero for personalized or agentic traffic.

**Semantic cache** — embed the query, return a stored answer if cosine similarity exceeds a threshold. This is the dangerous one and it gets its own treatment shortly. Its "hit rate" is meaningless without a paired *false-hit rate*, and reporting the first without the second is the exact mistake I would flag in review.

**⚠ Trap:** measuring cache hit rate at the request level when the provider caches at a block granularity (commonly 128 tokens, and with a minimum cacheable prefix length). A 300-token prompt may be entirely uncacheable, and a prefix that matches for 1,900 of 2,000 tokens gives you the first 1,792 cached and the rest recomputed. Report *cached input tokens / total input tokens* from the provider's usage fields, not a boolean hit/miss you computed yourself.

### Last one on this: here's our current service dashboard — latency percentiles, error rate, QPS, cost per request. Redesign it for an LLM product and tell me what's missing entirely.

I would keep exactly one of those four unchanged — error rate — and even that gets split.

**Row 1: user-perceived latency.** TTFT p50/p95/p99 as the headline. ITL p99 next to it, because that is your stutter detector. TPOT p95 third, with the reading-speed reference line at 180 ms drawn on the chart so nobody optimizes past the point of human perceptibility. E2E stays, but demoted, because it is a derived quantity now.

**Row 2: capacity, leading indicators only.** KV-cache utilization (or, if you are API-only, provider token headroom from the rate-limit headers). Waiting-queue depth and time-in-queue. Preemption rate. Time-in-limiter as a distinct series from time-at-provider — that split is non-negotiable, it is the difference between a diagnosable and an undiagnosable p99.

**Row 3: throughput and economics.** Goodput — completions per second meeting SLO — not QPS. Prefill and decode tokens/sec separately. Prefix-cache hit rate as *cached input tokens ÷ total input tokens*. Cost per resolved task, with cost per request kept only as a debugging sub-metric.

**Row 4: quality, which does not exist on your current dashboard at all, and that is the real finding.** This is where the interview is actually won. Your existing dashboard cannot distinguish "working" from "returning fluent nonsense," because every LLM failure mode returns HTTP 200. So: refusal rate, empty-or-degenerate-output rate, schema-validation failure rate on structured outputs, retrieval staleness (`now() − oldest indexed document`), and a sampled online judge score on a fixed rubric with its human-agreement rate published next to it so nobody over-trusts it. Plus tool-call error rate broken down by tool, which is the highest-signal single chart in any agent system.

**Row 5: safety and abuse**, sized to your product — prompt-injection detector hits, PII-in-output detections, jailbreak-classifier rate.

**⚠ Trap:** the one that motivates the entire redesign — **every catastrophic LLM failure is an HTTP 200.** A hallucinated policy, a stale retrieval, a silently truncated context, a semantic cache serving the wrong tenant's answer: all of them are green on your current board. If you take one thing from this section into a design round, make it the sentence "my existing observability stack cannot see any of the failure modes that matter here, so the first thing I'd build is the quality row."

**🗣 Say this in the room:** "The honest answer is that three of those four metrics measure the wrong quantity and the fourth can't see the failures that matter. Latency becomes TTFT and inter-token latency because streaming has an onset and a rate. QPS becomes goodput because a request isn't a unit of work. Cost per request becomes cost per resolved task. And I'd add a whole row that doesn't exist today — refusal rate, schema failures, retrieval staleness, sampled judge scores — because every bad answer this system produces returns a 200."
### Make this endpoint idempotent. It calls a model and writes the result to a database.

This is the question where backend instinct misleads most cleanly, so let me name the misleading part first: **an idempotency key guarantees you perform the operation once. It does not guarantee that a retried request would have produced the same result, and with a model, it emphatically would not.** Everything else follows from taking that seriously.

Your normal contract is "same key, same effect, and the second call returns the first call's response." That contract still holds and you build it the same way — a table keyed on `(tenant_id, idempotency_key)` with a unique constraint, holding request-hash, status and the serialized response, written in the same transaction as the effect. Nothing about LLMs changes that machinery. What changes is what you are protecting.

**What is actually nondeterministic, precisely.** Temperature > 0 samples, so outputs differ run to run. But even at temperature 0 you should not promise determinism: floating-point reduction order varies with batch composition on the GPU, and your request's batch-mates change every call, so logits differ in the last bits and occasionally flip an argmax. Provider-side model updates, silent routing between hardware generations, and MoE routing under different batching all add variance. **📅 Volatile:** some providers expose a `seed` and a `system_fingerprint` for best-effort reproducibility; treat it as best-effort, never as a contract, and verify current behaviour.

**The consequences to design around:**

*The stored response is the source of truth, not a re-computation.* If the client retries with the same key, you return the stored bytes. You never re-call the model to "reproduce" it. This is standard, but the stakes are higher because re-calling is expensive and would produce a *different* answer, which is a far worse bug than a duplicate charge — the user sees the answer change under them.

*In-flight concurrency needs a real lock, not a check-then-act.* Two retries arriving 50 ms apart both find no row and both call the model. Insert the key row with status `in_progress` *first*, let the unique constraint reject the loser, and have the loser poll or return 409 with `Retry-After`. Otherwise you pay twice and race on the write.

*Idempotency at the wrong layer is worse than none.* Making the outer HTTP handler idempotent while the agent loop inside retries individual tool calls means the *effects* — the email sent, the refund issued, the row inserted — are not covered. Every tool with a side effect needs its own idempotency key, derived deterministically from the run ID and the step index, and passed through. This is the part candidates miss.

*Some effects are inherently non-idempotent and the model chooses them.* If the model decides to call `send_email` twice with slightly different bodies, no key at the HTTP layer saves you. The control is at the tool boundary: dedupe on a semantic key (`run_id + recipient + intent`), or require confirmation for irreversible actions, or make the tool a *proposal* that a deterministic executor applies once.

**⚠ Trap:** hashing the request body to detect "same request, same answer" and treating a hash miss as a client error. Prompts contain timestamps, session IDs and retrieved context that legitimately change between a request and its retry, so the hash differs while the intent is identical. Key on an explicit client-supplied idempotency key; use the body hash only to *detect conflict* (same key, materially different body → 422), never as the key itself.

**🗣 Say this in the room:** "Idempotency here protects the side effects, not the output — I can guarantee we only charge the card once, I cannot guarantee that a regenerated answer is byte-identical, and I wouldn't design anything that depends on it. So the key goes on the write and on each side-effecting tool call, and the stored response is authoritative on replay rather than something we recompute."

### An engineer on your team wraps every model call in a three-attempt retry with backoff. What do you say in code review?

I block it, and the reason is arithmetic rather than taste.

In your normal services a retry costs a little CPU and a little connection churn, so "retry 3× with exponential backoff" is close to free and is correctly the default. Here every attempt re-runs the full generation, and the failures that trigger retries are *correlated across all your callers* — a 429 or an overloaded upstream means everyone is retrying at once. Retries stop being resilience and become a load amplifier pointed at a resource that is already saturated, with a billing meter attached. This is congestion collapse, which you already know, with the novel feature that it also empties your budget.

**💰 Math on the amplification.** The killer is that retry layers *multiply*, and modern stacks stack them without anyone deciding to. Provider SDKs retry internally by default (commonly 2 retries = 3 attempts). Your `tenacity` decorator adds 3 attempts. Your workflow engine retries the activity. Nobody wrote "27 attempts" but that is the ceiling. In steady state with a 2% failure rate the expected multiplier is ≈ 1.02 and nobody notices. During an incident where the failure rate hits 50%, each layer's expected attempts is 1/(1−0.5) = 2, and two nested layers give **2 × 2 = 4× the spend, precisely during the window when the provider is telling you to back off.** At a normal $5,000/day that is $20,000/day burned making the outage worse. And the retried traffic is what keeps the failure rate at 50%.

**What I ask for instead, as a review checklist:**

1. **Exactly one retry layer.** Disable the SDK's internal retries or disable yours. Name in a comment which one is authoritative.
2. **Retry only what is retryable.** 429 and 5xx, yes. 400, 401, 403, content-policy refusals, schema violations and context-length errors: never — the identical request will fail identically and you have paid for nothing twice.
3. **Respect `retry-after`.** Exponential backoff that ignores the server's explicit instruction is not backoff.
4. **Full jitter.** `sleep(random.uniform(0, min(cap, base * 2**attempt)))`. Without jitter, retries from a thousand callers re-synchronize into a thundering herd on every cycle.
5. **A concurrency limiter that shrinks on failure.** AIMD: additively increase in-flight capacity on success, multiplicatively halve it on 429. A circuit breaker is the coarse version and is also acceptable; a fixed semaphore is not.
6. **A per-request cost ceiling, enforced.** Retries count against the same budget as the original. A run that has spent its budget does not get retried, it gets failed and recorded.
7. **A budget for the retry itself.** For a streaming response that failed at token 400 of 500, retrying from scratch costs a full generation to recover 20% of one. Prefer resuming with the partial output as context, or degrade to a cheaper model, or return the partial with a marker.

**⚠ Trap:** the timeout-abort double charge. If you set an aggressive client timeout and abort, the generation may have already run to completion server-side, and whether you are billed for tokens you never received depends on the provider and on whether you were streaming. Assume you are billed. That converts an aggressive timeout policy into a silent cost line that never appears in your success metrics — you pay full price for every request you threw away.

**🗣 Say this in the room:** "Retries here are a cost decision, not a resilience default. I allow exactly one retry layer, only on 429 and 5xx, with full jitter and `retry-after` respected, sitting behind an AIMD concurrency limiter — because with two nested layers and a 50% failure rate you're at 4× spend at exactly the moment the provider is asking you to send less."

### All our tests pass and CI is green, but users say the assistant got worse after last week's deploy. How is that possible?

It is not only possible, it is the expected state of an LLM system under a conventional test suite, and the reason is a category error in what your tests assert.

Your unit and integration tests assert **structure**: the endpoint returns 200, the response parses as JSON, the schema validates, the tool was called with the right arguments, the database row was written. Every one of those can pass while the *content* degrades from correct to confidently wrong. The set of behaviours your tests constrain and the set of behaviours users care about barely overlap. A green build is evidence that the plumbing works; it is zero evidence about quality, and treating it as such is how teams ship regressions invisibly.

Compounding that: the things that changed last week may not be in your diff at all. Candidate causes, in the order I would check them:

**🔍 Failure taxonomy — "it got worse and CI is green":**

1. *Did the model version change under you?* Provider aliases float. A pinned snapshot ID is the only defence, and even then snapshots retire. Check the model string actually being sent in production, not the one in the config file. **📅 Volatile** — deprecation windows differ per provider.
2. *Did the retrieval corpus change?* An ingest job added 400k low-quality documents, or a reindex changed chunking, or the staleness gauge you never built has been climbing for nine days. Retrieval regressions look exactly like model regressions from the user's seat.
3. *Did the prompt change through a path that isn't code?* Prompts in a database, a feature flag, an admin UI, a CMS. If your prompts are not versioned in git with the same review gate as code, this is your most likely cause and you cannot bisect it.
4. *Did context length grow past the effective window?* Adding two more few-shot examples or a bigger `k` can push the important instruction into the middle of a long context, where adherence measurably degrades even though the context "fits."
5. *Did a tool's output format change?* An upstream API added a field, the tool result got 3× longer, and the model now truncates or gets distracted. This is a contract change no test covered because the tool test asserted on schema, not on token count.
6. *Did nothing change, and the traffic mix shifted?* A marketing campaign brought a new query distribution your prompt was never tuned for. This is the one people never check and it is common.

**What replaces the green build.** An eval suite in CI, gated statistically. Concretely: a versioned dataset of 200–500 cases with expected outcomes or judge rubrics; a job that runs it on every prompt, model, retrieval-config or code change; a gate that fails the build on a *statistically significant* regression, not on any drop. And a separate always-on production sampler scoring live traffic, because your eval set ages and your traffic does not.

**📐 Numbers you must know — how big does the eval set need to be?** To detect a 3-percentage-point change around a 75% baseline with an *unpaired* comparison you need roughly `16·p(1−p)/δ² = 16 × 0.1875 / 0.0009 ≈ 3,300` examples per arm, which nobody can label. But you are comparing two systems **on the same examples**, so use a paired test — McNemar's on the discordant pairs, or a paired bootstrap. Paired designs need order-of-magnitude fewer examples, because agreement on the easy cases cancels out; with realistic discordance rates a few hundred examples is enough to detect a change of that size. **The practical rule: 200–500 cases, always paired, always with a confidence interval reported next to the point estimate.** A dashboard showing "74% → 78%" with no interval is how teams chase noise for a week.

**⚠ Trap:** writing eval cases by asking the model to generate them. You get cases drawn from the model's own distribution — exactly the inputs it already handles — and a suite that scores 96% and detects nothing. Eval sets come from production logs, stratified over intent and over whether the current system succeeded, with the hard cases deliberately over-sampled.

### Someone proposes a semantic cache to cut spend. Sell me on it, then tell me why you'd be nervous.

The proposal is sound in outline: embed the incoming query, search a store of previous `(query, answer)` pairs, and if the nearest neighbour's cosine similarity exceeds a threshold, return the stored answer without calling the model. On a product with a heavy head — support FAQ, docs Q&A, product search — this genuinely converts a $0.05, 900 ms call into a $0.00002, 8 ms vector lookup. At a 30% hit rate on 200k daily calls that is 60,000 × $0.05 = **$3,000/day = $90k/month saved**, and the latency win is more valuable than the money.

Now the nervousness, and it is specific rather than general. **An exact cache can only be stale. A semantic cache can be wrong.** Those are different risk classes and your existing cache intuition does not carry over, because every cache you have built before had an equality test at its core.

The failure mode is that cosine similarity in embedding space is a *topical* measure, not a *semantic-equivalence* measure. Embeddings are trained to put things that are about the same subject near each other, and the distinctions that flip the correct answer are exactly the low-magnitude ones:

- **Entity substitution.** "What's the refund policy for the Pro plan?" and "...for the Enterprise plan?" typically sit above 0.95 cosine. Same topic, different answer, and the wrong answer is fluent and specific.
- **Negation and polarity.** "Can I cancel after 30 days?" vs "Can I cancel *before* 30 days?" Embedding models handle negation poorly; this is a well-documented weakness, not an implementation defect you can tune away.
- **Numeric and temporal specificity.** "How much is the Q3 invoice?" and "How much is the Q4 invoice?" are near-identical vectors.
- **Tenant leakage — the incident that gets written up.** If the cache is not partitioned by tenant, by user permissions, and by every filter that scoped the retrieval, you will serve customer A's answer to customer B. This is not a quality bug, it is a data-breach class of bug, and it is the reason I require cache keys to include the full authorization context, not just the query embedding.

**💰 Math on why the threshold cannot be tuned to safety.** Suppose at threshold 0.92 you get a 34% hit rate with a 6% false-hit rate. That is 200,000 × 0.34 × 0.06 = **4,080 confidently wrong answers per day** in exchange for $3,400/day. Raise the threshold to 0.97: hit rate falls to 11%, false-hit rate to maybe 1.5%, so 330 wrong answers/day for $1,100/day of savings. **You cannot get the false-hit rate to zero without the hit rate going with it**, because the two are the same distribution viewed from opposite ends. The decision is therefore a product decision about the cost of a wrong answer, and the only defensible way to make it is to *measure* the false-hit rate on a labelled set — never to pick a threshold because it looked round.

**What I would actually ship**, in order of how much I trust it: exact-match cache on a normalized request first, because it is free of this entire risk class; then prefix caching, which saves more money than semantic caching in most systems and cannot be wrong at all; then semantic caching *only* for a curated, high-frequency, low-variance query set with a measured false-hit rate, partitioned by tenant, with a TTL tied to the underlying data's change rate, and with a "was this helpful" signal wired to invalidate.

**⚠ Trap:** reporting semantic cache hit rate as a win metric with no paired false-hit rate. It is the most common form of AI-adjacent metric fraud I see, usually unintentional, and it is exactly the thing an interviewer will probe when you mention semantic caching. Volunteer the false-hit number before they ask.

### We added a 10-second timeout to the model call and now users complain about failures on the long answers. Diagnose it.

The diagnosis is that you applied a timeout to the wrong quantity. **A total-duration timeout on a streaming generation is a bug, because duration is a function of output length, and output length is a function of the question.** A user asking for a 2,000-token migration plan will legitimately take 40 seconds while streaming perfectly. Your timeout kills a healthy request in progress, after the user has already watched half the answer appear — the worst possible failure presentation, and one your error rate metric probably records as an upstream failure rather than as self-harm.

The correct instrument is two timeouts on two different quantities, plus a cap:

**A TTFT timeout, tight.** If no first token has arrived in 5–10 seconds, something is genuinely wrong — you are queued behind a limiter, the provider is degraded, or a huge prefill is stalling. This is the timeout that maps to your usual "is the upstream alive" check, and it should fire fast because it is the only one that can fire before you have spent output tokens.

**An inter-token-gap timeout, tighter.** Once tokens are flowing, healthy streams do not pause. A gap of more than 2–5 seconds between consecutive tokens means the stream is dead or stalled behind another request's prefill. This is a *stall* detector, and it is the one people never implement. It is what actually distinguishes "slow because the answer is long" from "hung."

**A hard cap expressed in tokens, not seconds.** `max_tokens` is your real bound on cost and on duration, and it fails cleanly with a `stop_reason` you can detect and handle, rather than by severing a connection.

**📐 Numbers you must know:** convert a latency budget into a token budget rather than a second budget, because tokens are the thing you control. `max_duration ≈ TTFT + TPOT × max_tokens`. At a measured TTFT of 0.5 s and TPOT of 30 ms, a 20-second product budget allows (20 − 0.5) / 0.03 ≈ **650 output tokens**. If the feature genuinely needs 2,000 tokens, the budget is 0.5 + 0.03 × 2,000 = 60.5 s and no timeout will fix that — you either cut scope, cut TPOT (quantization, speculative decoding, smaller model), or change the UX so the wait is not dead time. This substitution — seconds into tokens — is one of the most useful reflexes in the whole discipline.

```python
# Timeout on onset and on stalls, never on total duration.
async def stream_with_stall_detection(client, params, ttft_s=8.0, gap_s=4.0):
    async with client.messages.stream(**params) as stream:
        it = stream.text_stream.__aiter__()
        deadline = ttft_s
        while True:
            try:
                chunk = await asyncio.wait_for(it.__anext__(), timeout=deadline)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                raise StreamStalled(f"no token within {deadline}s")
            deadline = gap_s          # after first token, switch to gap budget
            yield chunk
```

**The infrastructure that will undo you regardless of application code:** every proxy between you and the client has its own idle timeout and its own buffering, and the defaults are hostile to streaming. nginx buffers proxied responses by default (`proxy_buffering off` and `X-Accel-Buffering: no` are the fixes), load balancers have idle timeouts commonly in the 30–60 second range, and some CDNs will hold a response until they see enough bytes. The classic incident is that streaming works perfectly in dev and returns as one blob in production, and everyone blames the model. **📅 Volatile:** default idle timeouts differ per cloud and change; check yours rather than trusting a remembered number.

**⚠ Trap:** setting a total timeout derived from p95 latency measured on your current traffic. The moment someone ships a feature that produces longer answers — a summarize-this-document button, a code-generation mode — the p95 shifts and your timeout starts killing your newest, most valuable feature. Timeouts must be tied to a *rate* (onset, gap) which is stable, not to a *total* which is workload-dependent.

**🗣 Say this in the room:** "I'd never put a wall-clock timeout on a streaming generation, because duration scales with output length and I'd be killing my longest, most valuable answers. I put a tight timeout on time-to-first-token, a stall timeout on the inter-token gap, and a hard cap on `max_tokens` — that bounds cost and duration without ever cutting off a healthy stream."

### The model isn't good enough on our domain. Walk me up your escalation ladder — and tell me what has to be true at each rung before you climb to the next.

The ladder exists because the cheap interventions are also the fast, reversible ones, and because every rung you skip makes the next rung's evaluation harder to interpret. The rule I enforce is that **you may not climb a rung until you can state, with a number, what the previous rung achieved and why it plateaued.** Without that, you cannot tell whether the expensive thing worked.

**Rung 0 — build the eval.** Not a rung, a precondition for the whole ladder. 200–500 cases from production logs, a primary metric, a paired comparison procedure. If someone asks me to improve a model's domain performance and there is no eval, my answer is that we are not yet in a position to improve anything, only to change things.

**Rung 1 — prompt and context engineering.** Clearer task framing, explicit output contract, 3–8 well-chosen few-shot examples, and — most underrated — *removing* material rather than adding it. In my experience this recovers the majority of a domain gap, and it costs an afternoon. **Precondition to leave:** you have run at least a dozen prompt variants against the eval, the best one is significantly better than the baseline, and the remaining errors are no longer instruction-following errors. *Climb when:* the failures are now missing-knowledge failures, not misunderstanding failures.

**Rung 2 — retrieval.** The model does not know your data. Give it the data. This is the correct fix for anything that *changes*: policies, prices, tickets, code, documents. **Precondition to leave:** you have measured retrieval separately from generation. Recall@k on a labelled query→document set tells you whether the right chunk was even in the context. If Recall@10 is 55%, you have a retrieval problem and no amount of generation work will fix it. *Climb when:* retrieval recall is high and the model still gets it wrong with the right context in front of it.

**Rung 3 — tool design.** The model does not need to *know* it, it needs to *fetch or compute* it. Wrap the exact query, the calculator, the internal API. This is where the biggest wins in agentic products actually live and it is chronically skipped because it is unglamorous engineering. **Precondition to leave:** tool call error rate broken out per tool is low, tool descriptions have been revised at least once against observed misuse, and error strings are written as corrective instructions. *Climb when:* the model is calling the right tools correctly and still producing unusable output.

**Rung 4 — structured output and constrained decoding.** If the failure is format — malformed JSON, inconsistent enums, wrong field names — do not fine-tune for it and do not prompt harder. Constrain the decode. This is a near-total fix for a whole failure class and takes a day. **Precondition to leave:** output is now schema-valid at essentially 100% and the residual errors are semantic, not syntactic.

Those four rungs, in that order, close most gaps. The three below the line are the ones that require you to have earned them, and they are the next question.

**⚠ Trap:** running the rungs in parallel to "save time." You change the prompt, add retrieval and swap the model in one week, the score moves 9 points, and you have no idea which change did it — or that two of them helped and one hurt. Sequential with a paired eval between each is not bureaucracy; it is the only way to know what to keep when you later need to cut cost.

### Now the bottom of the ladder — routing, distillation, fine-tuning. What are the actual preconditions?

**Rung 5 — routing.** Send easy requests to a cheap small model and hard ones to a frontier model, with a classifier or a heuristic making the call. Sometimes the "route" is to no model at all — a cached answer or a deterministic path. This is a cost lever primarily and a latency lever secondarily; it rarely improves peak quality.

**Precondition:** you can *measure* difficulty in advance, and you have a validated fallback for misroutes. The classifier's errors are asymmetric — routing a hard query to the small model is a visible quality failure, routing an easy one to the big model just costs money — so tune the threshold to be conservative and monitor the escalation rate.

**💰 Math:** 1M requests/month, 3k input + 500 output each. All-frontier at $3/$15 per M: 1M × (3,000 × 3/1e6 + 500 × 15/1e6) = 1M × ($0.009 + $0.0075) = **$16,500/month**. Route 70% to a small model at roughly $0.25/$1.25 per M (**📅 Volatile**): 700k × (3,000 × 0.25/1e6 + 500 × 1.25/1e6) = 700k × ($0.00075 + $0.000625) = $963. Remaining 300k at frontier = $4,950. Plus routing classifier cost, negligible if it is an embedding model. Total **$5,913 — a 64% reduction, $10,587/month saved.** Now the honest part: if 4% of the 700k requests sent to the small model are actually hard, that is 28,000 degraded answers a month, and whether that trade is correct is a product decision that requires the eval from rung 0 to even discuss.

**Rung 6 — distillation.** Use your frontier model to generate high-quality outputs on your traffic distribution, then train a small model on them. You get frontier-ish quality on *your* narrow task at small-model cost and latency. This is the right answer far more often than general fine-tuning, and it is under-used.

**Preconditions, all of them:** a task narrow enough that a small model can represent it; tens of thousands of teacher outputs, *filtered* by your eval rather than used raw; a licence that permits training on the teacher's outputs (several providers' terms restrict using outputs to train competing models — **📅 Volatile**, read the current terms, this is a real legal constraint not a formality); and a serving story for the student. **Precondition to justify:** volume high enough that the training and serving cost amortizes. At 1M requests/month the routing example above already saves $10.5k/month, so a distillation project costing $30k of engineering pays back in three months — at 20k requests/month it never does.

**Rung 7 — fine-tuning the base behaviour.** LoRA or full SFT on curated examples, and it is last for a reason. **What it is good at:** output format and structure, house style and tone, tool-calling conventions the base model keeps getting wrong, domain vocabulary and jargon, and compressing a very long prompt into weights so you stop paying for it every call. **What it is bad at:** installing facts. Facts change; weights do not. Anything that changes weekly must come from retrieval or it is stale the day after your training run.

**The named preconditions, and I would recite these as a list in an interview:** (1) a working eval with a baseline number; (2) rungs 1–4 exhausted, with the plateau documented; (3) at least a few thousand high-quality, consistent examples — consistency matters more than volume, and 2,000 clean examples beat 20,000 noisy ones; (4) a data-refresh plan, because the model will drift from your evolving product; (5) a serving plan — adapter hosting, versioning, rollback, and the fact that a custom model forfeits some provider features and cache economics; (6) an owner, because a fine-tuned model is a dependency with a maintenance cost forever.

**⚠ Trap — the reflex-fine-tuning rejection trigger, stated plainly.** Answering "the model doesn't know our domain" with "we'd fine-tune" is the single fastest way to be marked down in an applied AI loop, because it reveals that you have read about ML rather than shipped it. The interviewer is not testing whether you know what fine-tuning is; they are testing whether you know it is the seventh thing to try.

**🗣 Say this in the room:** "I'd hold fine-tuning until last and I'd want to be able to state what each earlier rung bought us. Fine-tuning teaches the model how to behave, not what's true — so format, style and tool conventions, yes; domain facts, no, those go in retrieval or they're stale immediately. Before I'd greenlight it I'd want an eval with a baseline, a documented plateau on prompting and retrieval, a few thousand consistent examples, and a named owner for the refresh."

### How would you actually find your own gaps before spending eight weeks studying? Give me the test.

Build a 40-question self-assessment, run it unaided and timed, and grade it honestly. The purpose is not a score, it is **localization** — the difference between "I need to study AI" and "I am fine on serving, weak on evaluation statistics, and cannot write attention." The first statement produces eight weeks of undirected reading; the second produces a plan.

**The structure: eight domains, five questions each, answered out loud or on paper in 90 minutes total with no references.**

*Transformer internals (5).* Write scaled dot-product attention with a causal mask. Derive √d_k. Give the parameter split between attention and FFN in a standard block. Explain what RoPE rotates and why relative position falls out. State the KV cache formula and say which quantities are absent from it.

*Serving and inference (5).* Explain why decode is memory-bandwidth-bound. Distinguish continuous batching from dynamic batching. Explain PagedAttention as a memory-allocator argument. Compute max concurrency for an 8B model on an 80 GB card at 8k context. Name the correct autoscaling signal and say why GPU utilization is not it.

*Retrieval (5).* Explain the chunking trade-off in terms of recall and precision. Contrast dense, sparse and hybrid retrieval and say when each wins. Describe reranking and its cost. Define Recall@k and nDCG@k. Describe a zero-downtime re-embedding migration.

*Evaluation (5).* Construct an eval dataset for a support agent, saying where cases come from. Choose a metric and defend it. Describe LLM-as-judge validation and name three of its biases. Compute the sample size to detect a 3-point change. Say what you would gate CI on.

*Agents (5).* Write the tool-calling loop from memory. Design a tool schema and say why it is coarse. Describe poison-trajectory detection. Give a context-budget accounting for a 10-step run. Name the four dominant agent failure causes in production.

*Post-training (5).* Contrast SFT, RLHF and DPO in one sentence each. Explain what RLVR needs that a chat task cannot supply. Explain LoRA's low-rank decomposition and its parameter count. Say when fine-tuning beats prompting. Say when it does not.

*Cost and capacity (5).* Compute monthly spend for a stated workload. Compute prefix-caching savings. Compute the routing trade with a misroute rate. Compute KV cache for a given model. Convert a latency budget into a token budget.

*Product and judgment (5).* Say when not to use an LLM, with the substitution. Name the failure mode users punish hardest. Describe the escalation ladder with preconditions. Critique an AI product you use. Say how you would know a feature is working.

**Grading, and this is the part that has to be brutal.** Score each question 0, 1 or 2. **2 = you produced it unaided, correctly, in time, including the arithmetic.** 1 = you knew the shape but needed a hint, fumbled a number, or could not write the code. 0 = you could not start. There is no partial credit for "I've read about that." A 1 is a fail for interview purposes, because a 1 under interview pressure becomes a 0.

**How to read the result.** Per-domain out of 10. **0–3: this domain is a rejection risk and gets a full week.** 4–6: real gaps, budget three focused days and re-test. 7–8: sharpen with drills only. 9–10: skip and re-test in week six to catch decay. Total out of 80 is nearly meaningless — the *shape* is the artifact, and the single lowest domain gets attacked first regardless of what the total says.

**⚠ Trap:** grading yourself while reading the answer. The entire diagnostic value of this exercise comes from the gap between recognition and production, and recognition feels exactly like knowledge. Write your answer down *first*, in full, then compare. Every engineer who skips this step scores themselves 15–20 points high and discovers the truth in a real loop.

**🏋 Drill — do this before you read another section.** 90 minutes, timer running, phone away, no autocomplete, blank paper and a blank editor. All 40 questions. Then grade against the guide over the following week, one domain at a time. **Pass criterion for the exercise itself (not for the score):** you have a per-domain vector of eight numbers and a written one-line plan for each domain scoring below 7, with a specific section to read and a date. If you finish with a total score and no plan, you did the exercise wrong — the score was never the point.

**🗣 Say this in the room,** when a hiring manager asks how you approached the transition: "I ran a written self-assessment across eight domains before I studied anything, graded it on whether I could *produce* the answer rather than recognize it, and it turned out my gaps were transformer internals and evaluation statistics — not the engineering. So I spent a third of my prep on those two and skipped the async, queueing and container material entirely, because I've been shipping it for years."


---

## 3. Getting the Interview: Resume, GitHub, Portfolio and Positioning

*Mastering this proves you can get scheduled at a frontier lab from outside the network, which is a prerequisite the other 86 sections assume away.*

### Walk me through the LLM feature at the top of your resume.

This is the question that decides whether the rest of the loop happens, and almost everyone answers it as a feature tour — "we built a support assistant, it uses RAG over our docs, users liked it." That answer is indistinguishable from a weekend project, so it gets scored as one.

The mental model: **a shipped LLM feature is a claim about a system under uncertainty, and a claim is only credible if you can state how you measured it, what it cost, how fast it was, and how it failed.** Four numbers. If any one is missing, the interviewer's prior is that you demoed something rather than operated it. This is not an arbitrary rubric — it maps exactly onto how you'd defend any backend service in a design review, except that the correctness axis is statistical rather than binary, and that is the axis backend engineers forget to bring.

So the shape of the answer is four beats, in this order:

**1. The eval.** Not "users liked it." A frozen set with a size, a construction method, and a metric. "182 real support tickets, stratified across the six intent classes by production frequency, labelled by two of our support leads with disagreements adjudicated by a third; primary metric was resolution-without-escalation judged by a rubric grader, secondary was groundedness — every factual claim traceable to a retrieved chunk." Interviewers are listening for whether the eval predates the system or was retrofitted to flatter it.

**2. The cost number, with arithmetic.** "3,200 input and 400 output tokens per call at $3/$15 per million meant $0.0096 + $0.0060 = $0.0156 per call; at 2.4M calls a month that's $37,440. Prefix-caching the 2,600-token system-plus-few-shot header at a 90% read discount took the cached portion from $0.0078 to $0.00078, and routing 65% of traffic to a small model at $0.25/$1.25 got the blended cost to $0.0039 a call — $9,264 a month, a 75% cut." **📅 Volatile:** those per-million prices move constantly; carry *your* prices and re-derive before the loop.

**3. The latency number, with the decomposition.** p95 TTFT and p95 end-to-end, separately, because they have different causes. "TTFT p95 was 4.2s, of which 2.9s was our own retrieval fan-out, not the model — we were doing a sequential embed → vector search → rerank chain. Parallelising embed with the metadata filter and moving the reranker to a cross-encoder on the top 40 got TTFT p95 to 1.1s."

**4. The named failure mode, and what you did about it.** This is the beat that separates seniors from everyone else, because it proves the system ran long enough to hurt you. "The router sent multi-hop questions to the small model, where the multi-hop slice of the eval dropped from 71% to 48% while the aggregate barely moved — which is exactly why the eval is sliced. We added a confidence-gated escalation that pushed about 8% of traffic back to the frontier model and cost us 3 points of the savings."

**⚠ Trap:** leading with the architecture. If you open with "so it's a RAG pipeline with a hybrid retriever and a reranker," you've told the interviewer you think the interesting part is the diagram. Every candidate has that diagram. Almost none of them can tell you what it cost per call.

**🗣 Say this in the room:** "The short version is: 182-ticket frozen eval, resolution-without-escalation as the primary metric, $0.0039 per call blended after caching and routing, 1.1s p95 time-to-first-token, and the failure mode that nearly shipped was the router quietly degrading multi-hop questions — invisible in the aggregate, obvious in the slice. Want me to go deeper on any of those?"

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

**⚠ Trap: the word "fine-tuned" on a resume you cannot defend.** This is the most reliable self-inflicted rejection in AI hiring. Interviewers at every archetype on your list treat it as a free depth probe, and it is a probe you will fail with "we used a service that did it for us." Worse, the reflex-fine-tuning signal is itself a seniority downgrade: senior candidates are expected to explain why they *didn't* fine-tune. A resume that says "evaluated LoRA fine-tuning against a prompt-plus-retrieval baseline; prompting won on 5 of 6 slices with no training or serving pipeline to maintain, so we shipped the baseline" is strictly stronger than one that says "fine-tuned Llama."

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
**Strong:** "Shipped an internal knowledge assistant to 400 weekly users at $0.0039/query blended, holding 78% groundedness on a 182-case frozen eval; cost came from prefix caching a 2,600-token header and routing 65% of traffic to a small model."
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

**💰 Math worth internalising:** at these prices, **input tokens are usually the bill.** 3,200 in / 400 out at $3/$15 is $0.0096 vs $0.0060 — input is 62% of spend despite output being 5× more expensive per token, purely because there are 8× more of them. Engineers reflexively optimise output length. The 10× lever is almost always the prompt.

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

**The AI-product read (Cursor, Perplexity, Notion, Figma, Sierra, Harvey, Glean, Ramp).** These teams are small, the hiring manager is often reading personally, and they are optimising for *ship rate and product judgment*. What moves them: a public artifact they can click, evidence you have made a quality/cost/latency tradeoff under real constraints, and any signal you use the category of product they're building. Cursor is widely reported to care whether you actually use AI coding tools daily — treat that as a rubric item rather than a personality question. **📅 Volatile:** company-specific hiring rubrics are second-hand and change; confirm with the recruiter rather than assuming. So on this version: portfolio links go above the fold, the projects section sits above older experience, and the summary line is product-shaped ("ships LLM features end-to-end, from eval design to cost control").

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
### If you could only show me three things you've built, what would they be and why those three?

The mistake is picking three *impressive* things. Pick three things that **jointly span the axes the role is graded on**, because a portfolio is a coverage argument, not a highlight reel. Three artifacts that all demonstrate the same competence are worth barely more than one.

The three axes that matter for AI Engineer roles at your target companies are: (1) *do you understand the model layer from the inside*, (2) *can you measure quality rather than assert it*, and (3) *can you operate a system with real users, cost and latency*. So:

**Artifact 1 — the from-scratch mechanism piece.** A transformer implemented from nothing, or a KV-cache/attention-variant benchmark, or a small inference server. Its job is to make the "does this backend person actually understand the model" question resolve in ninety seconds without an interview. It does not need to be novel; it needs to be *correct, small, readable, and numerate* — i.e. it must report numbers, not just run.

**Artifact 2 — the eval harness.** A public, reproducible evaluation of something, with a task definition, a dataset you can distribute or regenerate, a grader whose agreement with human labels you measured, and a results table with confidence intervals. This is the single highest-converting artifact in this field and I'll defend that claim in the next question.

**Artifact 3 — the operated system.** Something with users, a cost line, and a post-mortem. For you, the guide plus its Next.js site is exactly this artifact, and it is stronger than a toy chatbot precisely because it has a real ingestion pipeline, real retrieval, real deployment and real traffic.

Notice what is absent: **a chatbot demo, an "AI agent that browses the web," and a LangChain tutorial reimplementation.** These are the three most common portfolio items in this market and their marginal information content is near zero, because the reviewer has seen forty of them and cannot distinguish "I understood this" from "I followed a video."

**⚠ Trap:** breadth as a substitute for depth. Six shallow repos read worse than two deep ones — they signal that you start things. Pin three, archive or unpin the rest. Your GitHub profile is a curated portfolio surface, not an archive, and treating it as an archive is a choice you are making by default.

**🗣 Say this in the room:** "I'd show you three things: an attention implementation with a KV-cache benchmark so you can see I understand the mechanism, an eval harness with a measured grader so you can see how I decide whether something works, and a system I actually operate with a cost model and a post-mortem. They're picked to cover different questions rather than to be three of the same thing."

### Why do you keep saying an eval harness converts better than a chatbot demo? Defend that.

Because of what each artifact is *evidence of*, and evidence is the only currency in a hiring decision.

A chatbot demo is evidence that you can call an API and wire a frontend. That was a differentiating skill in early 2023. It is now the baseline capability of a competent undergraduate with a weekend, and — critically — **the reviewer cannot tell your chatbot from a generated one.** The artifact does not discriminate, so it cannot convert. Worse, a demo invites the question "how do you know it's good?", and if the repo contains no answer, the artifact has actively harmed you.

An eval harness is evidence of the exact competency that the debrief rejects people for. Building one forces you to make and defend a chain of decisions that cannot be faked: what is the task, what is a correct answer, where does the data come from and is it contaminated, is the grader an exact-match, a rubric LLM judge, or a human, **what is the measured agreement between your grader and human labels**, how many examples do you need for the difference you're claiming to be real, and how do you report uncertainty. Every one of those is a five-minute interview conversation with a right answer, and having built one means you have already had the conversation with yourself.

There is a second, less obvious reason: **an eval harness produces a table, and a table is shareable.** A demo has to be experienced; a benchmark table can be screenshotted into a Slack channel by the engineer who found it. Artifacts that other people can forward are the ones that travel.

**📐 Numbers you must know — why your eval needs to be bigger than you think.** If two systems score 72% and 78% on n = 100 examples, is that real? The standard error of a proportion is √(p(1−p)/n) = √(0.75 × 0.25 / 100) ≈ 0.043, so each estimate carries roughly ±4.3 points of one-sigma noise, and the standard error of the *difference* on independent samples is about √(2) × 4.3 ≈ 6.1 points. A 6-point gap is one sigma. It is not a result. Getting that same 6-point gap to two sigma needs roughly n ≈ 400 per arm. **This single calculation, done out loud, marks you as someone who has actually run evals** — and it is why paired evaluation on the *same* items (where you compare per-item wins and losses rather than two independent means) is the standard trick, since it removes item difficulty as a variance source and buys you the same power at a fraction of the sample size.

**⚠ Trap:** publishing an eval where the grader is an LLM and you never validated the grader. An unvalidated judge is a random number generator with good grammar. The fix is cheap and it is the thing that makes your repo credible: hand-label 100 items, report the agreement between your judge and your labels (Cohen's κ or plain agreement percentage, stated), and publish the disagreements. Publishing your grader's failure cases is the strongest credibility signal available in this entire section, because nobody fakes that.

**🗣 Say this in the room:** "The thing I'd point at is the eval harness. Building it meant deciding what a correct answer is, measuring my judge against 100 human labels before trusting it, and being honest about how much of a gap is actually resolvable at my sample size. That's the reasoning I'd bring to your quality problem."

### I'm going to spend ninety seconds on your GitHub before the screen. What should I see?

Ninety seconds is roughly: profile page, read the top of the README of one pinned repo, glance at the file tree, maybe open one file. Design for exactly that.

**The profile page.** A one-line bio that says what you build, and up to six pinned repos ordered so the first one is the strongest. The green contribution square wall is not evidence of anything and I would ignore anyone who reads it as such; do not manufacture it with daily trivial commits, which is a visible and slightly embarrassing pattern.

**The README, which is the whole game.** The reviewer will not read your code before deciding whether to care. The README decides. It must answer, in the first screen — before any install instructions — **what this is, what question it answers, and what the result was.** My template:

1. One sentence: what it is and what it's for.
2. The result, as a table or three bullets with numbers. If it's a benchmark, the table goes here, at the top, above the fold.
3. "How to reproduce" in three commands maximum.
4. Then: the design section — the interesting decisions and their tradeoffs, written as a design doc.
5. Then: limitations, honestly. What it does not do, where the numbers are shaky, what you'd do with another week.

That fifth section is disproportionately powerful. A limitations section is the written form of calibration, and calibration is what senior hiring is actually screening for.

**The file tree.** A reviewer glancing at a tree is asking one question: does this look like a system or like a notebook someone exported? Signals that read well: a `tests/` directory with tests that actually assert something, a pinned dependency file (a lockfile, `pyproject.toml` with versions), a `Makefile` or `justfile` with the three commands from the README, a `results/` or `benchmarks/` directory with committed raw output. Signals that read badly: `notebook_final_v3_copy.ipynb`, a 900-line `main.py`, committed `.env` files, no tests at all.

**⚠ Trap:** the README that opens with "Installation." That structure assumes the reader has already decided to use your thing. Your reader has decided nothing; they are deciding whether you are worth an hour of their calendar. Result first, always.

**🏋 Drill:** open your top pinned repo, start a 90-second timer, and read only what a stranger would read in that window. Write down what you learned. Pass criterion: you can state what the project does, one number it produced, and one design decision the author made. If you can't, rewrite the README — not the code.

### Your repo has a benchmark table in the README. How do I know the numbers aren't garbage?

You don't, unless the repo makes it cheap to check — and **making it cheap to check is the entire value of the artifact.** An unfalsifiable benchmark table is decoration; a falsifiable one is a credential. The difference is a handful of specific practices, each of which is also a thing an interviewer might ask you about.

**Report the setup completely enough to reproduce.** Hardware (exact GPU or CPU, count), driver/CUDA version, library versions pinned, model identity including the exact revision or quantisation, batch size, sequence lengths, sampling parameters (temperature, top-p) and seeds. "Llama on an A100" is not a setup; "Llama-3.1-8B-Instruct at bf16, vLLM 0.x, one A100-80GB, input 1024 / output 128, batch 32, greedy" is. **📅 Volatile:** engine version numbers age fast — pin them in the repo and say the date you ran it.

**Report variance, not just the mean.** Three runs minimum, report median and spread. A single-run throughput number is a sample from a distribution whose width you have not measured, and on shared or virtualised hardware that width can be double digits of percent.

**Separate warm-up from steady state.** First-iteration numbers on a GPU include kernel autotuning, memory allocator warm-up, and possibly graph compilation. Anyone who has profiled a JIT'd system knows this reflex; the same discipline applies. Discard warm-up iterations explicitly and say you did.

**State the metric precisely.** "Latency" is ambiguous in this field in a way it isn't in backend: time-to-first-token, inter-token latency, end-to-end per request, and per-request-under-concurrency are four different numbers and the honest table names which. Throughput needs its concurrency level attached or it is meaningless.

**Commit the raw output.** Not just the summary table — the JSON or CSV your harness emitted, so a skeptic can recompute your aggregates. This is the single cheapest trust-building move available and almost nobody does it.

**Show a control.** If you're claiming your optimisation helped, the table must contain the unoptimised baseline measured the same day on the same box, not a number quoted from someone else's blog post.

**⚠ Trap: the accidental apples-to-oranges comparison.** The classic version is benchmarking two inference engines where one is doing continuous batching and the other is not, or where one is serving a quantised model and the other bf16, and reporting the throughput ratio as if it measured engine quality. If you publish that, someone knowledgeable will notice in public. The defensive habit: for every comparison, write one sentence listing what is held constant — and put that sentence in the README.

**💰 Math:** benchmarks cost money and the repo should say so. Ten configurations × three runs × four minutes a run = 120 minutes of GPU time; on a rented A100 at roughly $1.50–$2.50/hour that is about **$3–$5 of compute** for a table that can carry a portfolio. **📅 Volatile** on the hourly rate. The point to internalise is that the barrier to publishing a real benchmark is not cost — it is rigour, which is why doing it rigorously is a differentiator.

### Someone clones your repo and it doesn't run. What did you fail to do, and how do you make that impossible?

Treat "a stranger cannot run this in ten minutes" as a production incident, because the blast radius is the same: the artifact silently stops converting and you never find out. The debugging discipline here is the same one you'd apply to a flaky deploy — the bug is almost never in the code, it is in the **implicit environment** the code assumed.

**🔍 Failure taxonomy — why a portfolio repo fails to run, in order of frequency:**

1. **Unpinned dependencies.** It worked in March because a transitive dep hadn't broken yet. Fix: a lockfile committed, or `pyproject.toml` with upper bounds, plus the Python version stated. This ecosystem moves fast enough that unpinned means broken within months.
2. **An undocumented secret.** The code reads `OPENAI_API_KEY` and dies with a `KeyError` in a stack trace instead of a message. Fix: a committed `.env.example`, validate config at startup, and fail with a sentence a human can act on.
3. **Undocumented data.** The harness expects a dataset that lives on your laptop. Fix: either commit a small sample, ship a `make data` that downloads it, or generate synthetic data — and make the default path use the small sample so the smoke test runs for free.
4. **Requires a GPU you didn't say it requires.** Fix: state the hardware requirement in the first screen of the README, and provide a tiny-model or CPU path for the smoke test.
5. **Costs money to run and doesn't warn.** Someone runs your eval and burns $40 of API credit. Fix: print the estimated cost and token count before executing, and default to a `--limit 20` sample. This also reads as excellent engineering judgment on its own.
6. **The README drifted from the code.** The command in the README is two refactors old. Fix: this is the one that CI solves — make the README's commands the ones CI executes.

The structural solution is a **GitHub Actions workflow that runs the README's quickstart on a clean runner** on every push, with a mocked or tiny-model path so it doesn't need a GPU or credits. A green badge on a portfolio repo is not decoration; it is a machine-checked claim that a stranger can run this. That single workflow prevents four of the six failures above.

```yaml
# .github/workflows/smoke.yml — the highest-leverage 20 lines in a portfolio repo
name: smoke
on: [push, pull_request]
jobs:
  quickstart:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: make smoke        # the exact command the README tells a stranger to run
```

**⚠ Trap:** "it works on my machine" hidden behind a Dockerfile that you last built four months ago and that pulls `:latest` base images. A Dockerfile is not reproducibility unless something builds it on a schedule. Add a weekly `schedule:` trigger to that workflow and you'll learn about the breakage from an email rather than from a hiring manager's silence.

### You have forty hours. Design me the one portfolio project you'd build, end to end.

Forty hours is two weekends and it is enough — if the scope is chosen so that the *result* is the deliverable and the code is incidental. I'd build a **benchmark-plus-eval artifact that answers a question nobody has published a clean answer to**, because that produces a table, a blog post, and a repo from one body of work, and those three assets have different distribution channels.

Here's a concrete design I'd actually run, weighted toward your target archetypes.

**The question:** *For a retrieval-heavy assistant workload, how do the quality/cost/latency tradeoffs actually land across (a) frontier model with no retrieval and a large context, (b) small model with retrieval, (c) frontier model with retrieval, and (d) small model with retrieval plus a reranker?* This is a question every AI product team at Notion, Glean or Sierra has argued about internally with vibes rather than numbers.

**Hours 0–6: the dataset.** 200 questions over a public corpus you can redistribute — pick something with real structure and real ambiguity. Stratify deliberately: 60 single-hop, 60 multi-hop, 40 unanswerable-from-corpus (this slice is where systems embarrass themselves and where your artifact earns its keep), 40 requiring recency or aggregation. Label answers yourself. This is the expensive part and it is what nobody else will do.

**Hours 6–12: the grader, and validate it.** Rubric-based judge for correctness, plus a groundedness check (is each claim supported by a retrieved span). Then hand-label 100 items and report agreement. Publish the disagreement cases.

**Hours 12–24: the four systems.** Keep the code boring. One retrieval module, one generation module, a config file selecting the arm. Instrument tokens in/out per call, wall-clock per stage, and cost from a pricing table in config.

**Hours 24–32: run, three seeds, record raw output.** Report per-slice results, not just aggregate — the unanswerable and multi-hop slices are the story.

**Hours 32–40: write it up.** README with the table above the fold, a post with the narrative and the surprising result, limitations section, and the cost of running the whole thing.

**💰 Math on what this costs you to run:** 200 questions × 4 arms × 3 seeds = 2,400 generations. Say 6,000 input tokens (retrieved context) and 300 output. Frontier arms (half of them, 1,200 calls): 1,200 × (6,000 × $3/10⁶ + 300 × $15/10⁶) = 1,200 × ($0.018 + $0.0045) = **$27**. Small-model arms: 1,200 × (6,000 × $0.25/10⁶ + 300 × $1.25/10⁶) = 1,200 × ($0.0015 + $0.000375) = **$2.25**. Judge calls at 2,400 × 2 metrics × ~$0.004 ≈ **$19**. Embeddings for the corpus, a few dollars. **Total ≈ $50–$60 for an artifact that can anchor an entire job search.** That arithmetic — in the README — is itself a hiring signal. **📅 Volatile:** prices.

**⚠ Trap:** scoping a *product* instead of a *result* in forty hours. A half-finished product with no users is worth nothing; a small, complete, honestly-measured result is worth a lot. If your forty-hour plan includes "build auth," delete the plan.

### Everyone writes a from-scratch transformer. How do I make mine worth looking at?

You're right that the baseline version is worthless as a differentiator — there are thousands of them, most descended from the same handful of well-known educational repos, and a reviewer can't distinguish yours from a fork. But the underlying activity is still the single best way to actually learn the material, so the answer is not to skip it; it is to **attach a question to it.** A from-scratch implementation with a research question attached stops being a tutorial and becomes an experiment.

Question-attached variants that are genuinely interesting and are all inside a weekend or two:

- **Measure what the textbook asserts.** Train a small model twice, once with the 1/√d_k attention scaling and once without, and plot the loss curves and the attention-logit magnitudes. Now you have *evidence* for the answer you'd otherwise recite in an interview. Same treatment for pre-norm vs post-norm stability at increasing depth, or learned positional embeddings vs RoPE on length generalisation past the training length.
- **Implement one modern component honestly and benchmark it.** Multi-head vs grouped-query vs multi-query attention: implement all three in the same codebase, then report the actual KV-cache bytes per token and the decode throughput at several batch sizes. That table is directly the thing serving interviews ask about, and building it means you'll never fumble the KV-cache arithmetic.
- **Reproduce a paper's headline plot at 1/1000 scale and report where it breaks.** The honest write-up of *where the small-scale reproduction diverged from the paper* is more interesting than a successful reproduction, and it demonstrates research literacy, which is the scarce half of "AI engineer."

**📄 Paper:** Vaswani et al. (2017) — the encoder-decoder transformer built on scaled dot-product attention, which replaced recurrence as the sequence-modelling default. Reimplementing it is table stakes; *measuring one of its claims* is the differentiator.

**⚠ Trap:** the from-scratch repo that has no numbers in it at all. If the README says "an educational implementation of the transformer" and shows no loss curve, no throughput figure, no memory measurement, it is indistinguishable from every other one and it will be treated as such. **Numbers are what make an artifact yours.**

**🗣 Say this in the room:** "I wrote the transformer from scratch mainly to stop hand-waving, but the part I'd actually show you is the GQA comparison — same codebase, three attention variants, measured KV-cache bytes per token and decode throughput at batch 1, 8 and 32. It's where the memory-bandwidth story stopped being something I'd read and became something I'd measured."

### What makes a technical blog post actually reach the people who hire, rather than sinking?

Distribution in this field is not mysterious, but it is unforgiving: **posts that get read are posts that contain something the reader cannot get elsewhere.** Explanation is not that thing — the internet has an unlimited supply of "how attention works" posts, and yours is competing against ones written by people who built the systems. Three formats reliably clear the bar, and they map exactly onto the three artifact types.

**Format 1 — the post-mortem of a system you actually ran.** "We shipped X, here is what it cost, here is what broke, here is the number that made us change the design." Nobody else can write this post because nobody else has your incident. It is the most valuable and the least written, because it requires admitting something went wrong — which is exactly why it signals seniority. Structure it like an incident review: what we built, what we expected, what happened, the timeline, the root cause, what we changed, and what we'd do differently. Include the cost and latency numbers.

**Format 2 — the from-scratch reimplementation with numbers.** As above: the reimplementation is the excuse, the measurement is the content. Title it after the finding, not the activity: not "Building a Transformer from Scratch" but something that names what you measured.

**Format 3 — the benchmark nobody ran.** The most reliably shareable of the three, because it fills a gap people already feel. Pick a comparison that practitioners argue about with anecdotes: structured-output approaches under a strict schema, chunking strategies on a real corpus, reranker cost/benefit at various k, an inference engine comparison held genuinely constant. The rigour requirements from the benchmark question above apply in full — a widely-shared benchmark with a methodological hole is worse for you than no post.

**On distribution mechanics, briefly and honestly:** posts travel when someone with an audience shares them, and that happens when the post has a *specific, quotable finding* — a table, a number, a reversal of conventional wisdom. Title and first paragraph carry almost all the weight; state the finding in both. Post where practitioners in this field actually congregate rather than where you find it comfortable, and link it from your GitHub profile and your resume so it compounds even when it doesn't go anywhere on its own.

**⚠ Trap:** writing five explainer posts on topics you just learned. Beyond the near-zero distribution value, there is a real downside risk: an explainer written from a fresh understanding usually contains a subtle error, and the audience you want is exactly the audience that will spot it. **Write from what you measured, not from what you read this week.**

**📐 Numbers you must know:** the realistic conversion from writing is lumpy and low-frequency, and anyone promising otherwise is selling something. Assume most posts do nothing and that the value is (a) the artifact existing when someone Googles you, and (b) one post in five or ten reaching someone consequential. That expected value is still excellent, because the cost is one weekend and the payoff is a warm inbound — but plan the pipeline as though writing will produce zero interviews, and treat any that arrive as upside.

### Should I keep my demo hosted and live? What does that actually cost me?

Ask what the demo is *for*, and the answer is usually: to let a reviewer see, in under a minute, that a thing works. That goal has three implementations with wildly different costs, and the reflex choice — a live hosted app with a real model behind it — is usually the worst one.

**Option A — a 60-second screen recording, embedded at the top of the README.** Costs nothing, never breaks, never gets abused, works when the reviewer is on a plane, and takes the reviewer zero setup. For most portfolio purposes this is the correct answer and I'd default to it.

**Option B — hosted but keyless and rate-limited.** Correct when the interactivity *is* the point — a search/retrieval product where the interesting thing is trying your own query. This is the case for your guide's site, where a reviewer typing a query and getting a good result *is* the demonstration.

**Option C — hosted, unauthenticated, calling a frontier model on your key.** This is the one I'd push back on in review. You have built an open proxy to a metered API and put it on the public internet. It will be found by scrapers within days.

**💰 Math on why option C is a bad idea:** suppose a scraper hits your endpoint at a modest 2 requests/second for one day: 2 × 86,400 = 172,800 calls. At even 1,500 input and 500 output tokens on a mid-tier model at $3/$15 per million, that's 172,800 × ($0.0045 + $0.0075) = **$2,074 in one day.** The mitigation stack is exactly your day job — a per-IP token bucket, a global daily token budget enforced in Redis with a hard kill, a cheap small model behind the demo rather than a frontier one, a provider-side spend cap, and a billing alert. If you do host it, **put that mitigation stack in the README**; a reviewer seeing "the demo enforces a 50k-token/day global budget and degrades to a cached response beyond it" learns more about your engineering than the demo itself does.

**On keeping things alive long-term:** an idle serverless deployment of a Next.js app or a small FastAPI service is typically a few dollars a month or free on hobby tiers — that is not the cost that hurts. The costs that hurt are the model bill and your attention: a demo that has been broken for three months because a dependency moved is actively negative evidence. **📅 Volatile:** hosting tiers and free-tier limits change constantly.

**⚠ Trap:** the dead link on a resume. A portfolio URL that 404s or times out is worse than having no portfolio, because it converts a neutral prior into "does not check their own work." Put a weekly scheduled uptime check on every URL that appears on your resume — a five-line cron that curls each link and emails you on non-200. You would do this for a production endpoint without thinking; your resume links are a production endpoint.

### What do you actually look for in someone's commit history, and does commit hygiene matter for hiring?

Honestly: almost nobody reads commit history during a screen, so optimising for it is a poor use of time. But it matters in two specific, high-stakes moments, and both are worth ten minutes of preparation rather than a habit overhaul.

**Moment one: the reviewer who is deciding whether you wrote this.** When a repo looks suspiciously polished, a skeptical reviewer will glance at the history. A repo whose entire content arrived in one commit called "initial commit" three days before you applied is a legitimate flag — not proof of anything, but it removes a form of evidence you could have had for free. A history showing incremental construction over weeks, with commits that fix your own bugs, is quietly strong evidence of authorship. **This is the cheapest defence against the "did you just generate this?" question, and it costs nothing except committing as you go.**

**Moment two: your open-source pull requests.** Here hygiene is not cosmetic — it is the actual medium of the interaction. A maintainer decides how much of their scarce attention to give you partly from the shape of your PR: one logical change per commit, a message that says why rather than what, tests in the same PR, no unrelated formatting churn, and a description that states the problem, the approach and the verification. A PR that touches 40 files because your editor reformatted them will not be merged no matter how good the fix is.

The message convention I'd hold to on anything public: a short imperative subject under ~72 characters, a blank line, then a body that explains the *why* and any tradeoff considered. "Fix bug" is a wasted line; "Reject zero-length chunks before embedding — empty inputs produced NaN vectors that silently poisoned recall@10" is a line that makes a stranger trust you.

**⚠ Trap:** rewriting your history to look better before applying. Force-pushing a fabricated multi-week history is detectable (timestamps and their relationship to file content have a way of not matching) and the downside if noticed is catastrophic relative to the tiny upside. The honest version — "I built this over three weekends, here's the history" — is available to you for free if you simply start committing incrementally now.

**🗣 Say this in the room,** if a reviewer probes how a repo was built: "It's about three weekends of evenings — the history shows it. The first weekend was the dataset and the labelling, which was most of the work; the second was the four arms; the third was the write-up and fixing the two places where my judge disagreed with my own labels."
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
### I don't know anyone at these companies. How do I actually get a referral rather than hoping for one?

Start by fixing the model of what a referral *is*, because the wrong model is why most people fail at this. A referral is not a favour. It is **an employee spending a small amount of their reputation to move you from the cold pile to the read pile**, usually for a cash bonus, and always with the risk that you embarrass them. So the question you must answer for a stranger is not "will you help me" — it is "**why is referring me a low-risk, positive-expected-value action for you?**" Everything below is a way of answering that question in advance.

**The tiers, in order of yield:**

**Tier 1 — dormant strong ties.** Former colleagues, people you shipped with. Almost everyone underestimates this list because they only think of current contacts. Enumerate every person you have worked directly with over eight years, look up where they are now, and you will typically find several inside your target set or one hop from it. This is a two-hour exercise with the highest yield in the whole section, and it is the one people skip because it feels like homework. A message to a dormant strong tie converts far better than a message to a warm stranger.

**Tier 2 — earned weak ties.** People who know your work rather than your face: the maintainer who reviewed your PR, someone who replied to your benchmark post, a person you had a substantive technical exchange with. This tier is the entire reason the artifact strategy exists — **artifacts manufacture weak ties**, which is their real economic function.

**Tier 3 — cold employees.** Lowest yield, but non-zero if you make the risk calculation easy for them. The ask that works is small, specific and pre-packaged: not "can you refer me," which asks them to evaluate you, but "I applied to req #1234 last week; if you happen to think the profile fits, a referral would help — no worries either way. Two-line summary and links below, so you don't have to write anything." You are minimising their effort to a copy-paste and giving them an easy, face-saving no.

**What to include, every time, in one short message:** the specific req (link it), one sentence on who you are, **one specific artifact with a number in it**, and an explicit low-pressure out. No attached resume unless asked — a link is lower friction.

**⚠ Trap:** the coffee-chat gambit — "can I pick your brain for 30 minutes about your career journey" as a stealth referral request. Senior engineers at hot AI companies receive these constantly and recognise them instantly. Asking directly and briefly is *more* respectful and converts better. If you genuinely want the conversation, ask a real technical question you cannot answer yourself, and don't mention jobs.

**📐 Numbers you must know:** the exact multiplier varies enormously by company and req, so don't memorise a statistic — memorise the decision rule. **A referral converts to a recruiter screen at some multiple of cold application, large enough that if a referral is available for a company, applying cold first is a mistake.** Practical consequence: for every target company, spend fifteen minutes looking for a path *before* you press apply, because most ATS flows will not let a referral attach cleanly to an application that already exists.

### Write me the cold email that actually gets a reply from a hiring manager.

The one that works has a specific shape, and the shape follows from what the recipient is doing when it arrives: scanning a full inbox on a phone, deciding in about four seconds whether this is a person or a template. **The single differentiator is evidence that you did work before writing** — because that is the one thing a mass-mailer cannot fake at volume.

Rules, then the template:

- **Under 150 words.** A long cold email is a request for a large block of attention from someone who has none. Length is itself a negative signal about your judgment.
- **The first sentence must contain something only you could have written** — a specific, correct observation about their product, their engineering blog, their public technical decisions, or a problem their domain obviously has. Not "I'm impressed by your mission."
- **One artifact, with a number, that is relevant to them.** Not three links. One.
- **A small, concrete ask.** "Is the retrieval team hiring?" or "Worth me applying to req X?" — a yes/no question is answerable in four seconds; "would love to chat about opportunities" is not.
- **No attachments. No "circling back" follow-up more than once.**

```
Subject: retrieval eval on 200 questions — relevant to [product]?

Hi [name] — I benchmarked four retrieval configurations on a
200-question set I labelled myself (60 single-hop, 60 multi-hop,
40 unanswerable-from-corpus, 40 recency). The unanswerable slice
was the interesting one: the small-model-plus-reranker arm beat
the frontier-model-with-large-context arm there by 14 points,
because the big-context arm confabulated rather than abstaining.
Write-up and code: [link]

I'm a senior Python backend engineer (8 yrs — FastAPI, Postgres,
distributed systems) who's spent the last year shipping LLM
features: evals, retrieval, cost control. [One specific sentence
about their product's retrieval or agent surface.]

Is the [team] hiring at senior level? Happy to apply through the
normal process — just wanted to put the benchmark in front of you.

[name] · github · site
```

**⚠ Trap:** the flattery opener. "I've been following your work and I'm deeply inspired by what you're building" is the signature of a template, and it is the fastest path to the archive. Replace it with a technical observation, even a mildly critical one — "your docs suggest you're doing exact-match prefix caching; I benchmarked radix-tree prefix sharing against exact-match and the hit rate difference at 30% prefix overlap was larger than I expected" gets replies, because it proves you thought.

**🗣 Say this in the room,** when a recruiter asks how you found them: "I'd been benchmarking retrieval configurations and it seemed directly relevant to what your team is doing, so I sent the write-up to [name] rather than just applying." That sentence positions you as someone who leads with work.

**📐 Numbers you must know:** treat *templated* cold outreach as a low-single-digit-percent reply channel and size your effort accordingly — twenty carefully-researched emails, not two hundred templated ones. Twenty at fifteen minutes each is five hours; at a 10% reply rate — the optimistic end, and only reachable because each one is genuinely researched — that is two conversations, which is a better return than a hundred cold applications. **The tail is what makes it worth it:** one reply from a hiring manager at a company you want is worth more than the entire rest of the funnel.

### A recruiter has twenty minutes with you on a screen. What are they deciding, and how do you make it easy?

They are deciding three things and only three, and candidates lose these screens by answering questions the recruiter did not ask. The recruiter is deciding: **(1) is this person's experience roughly the level and shape the req wants, (2) will they embarrass me in front of the hiring manager, and (3) will they accept an offer if we make one** — meaning comp expectations, timeline, and location/work-authorisation constraints.

None of those is a technical evaluation. So the failure mode is going deep on the transformer when the recruiter needs a crisp positioning sentence they can paste into a summary.

**What to prepare, verbatim:**

**A 45-second positioning answer to "tell me about yourself."** Not a career history. Present state, one proof point with a number, and what you're looking for. "I'm a senior backend engineer, eight years, mostly Python at scale — FastAPI, Postgres, event-driven systems. The last year I've been shipping LLM features end to end: I own the eval design, the retrieval, and the cost model; on the last one I got blended cost from $0.0156 to $0.0039 a call while holding quality flat on a 182-case eval. I'm looking for a role where the AI system is the product rather than a feature bolted on the side."

**A 20-second answer to "why are you looking."** Forward-looking, never grievance-shaped.

**A clean answer on comp,** with a range and a basis, delivered without flinching. Recruiters read hesitation here as inexperience.

**A clean answer on logistics** — timezone, notice period, authorisation. Say it plainly and early; ambiguity here gets you dropped silently later, which is far worse than a direct conversation now. (The strategy behind those answers is its own topic — see §4.)

**Two questions for them that only a serious candidate would ask:** what the loop actually consists of and how many stages; and what the team's current hardest problem is. The first is pure information you need for prep; the second frequently gets you a genuinely useful answer that shapes every subsequent round.

**⚠ Trap:** treating the recruiter as an obstacle. The recruiter is the single best-informed ally you have about the process — they know the loop structure, the interviewers, the leveling, and the timeline, and they are professionally motivated for you to do well. Ask them for the loop breakdown explicitly: "what are the stages, and is there anything specific I should prepare for the technical rounds?" Most will tell you, in detail. Candidates who don't ask are leaving the map on the table.

**🗣 Say this in the room:** "Before we finish — what's the shape of the loop, and is there a round that tends to be the differentiator for this team? I'd rather prepare for the right thing."

### What should my LinkedIn actually say for AI roles, and how much does it matter?

It matters in exactly one direction: **LinkedIn is an inbound channel, not an outbound one.** Nothing you post will get you a job; what it does is determine whether recruiters searching for candidates find you and whether the ones who find you bother to click. Optimise for being found and being credible in a six-second scan, then stop thinking about it.

**The four fields that do work:**

**The headline.** This is what appears in search results and it is the only text most recruiters read. It should contain the words a recruiter would search. "Senior Software Engineer at [Company]" is invisible. "Senior Backend Engineer → AI/LLM Systems | Python, RAG, Evals, Inference Cost" is findable. Yes, it is keyword-shaped; that is the mechanism of the surface.

**The About section, first two lines.** It truncates. Put the positioning sentence and one number in the first two lines and treat everything after as optional.

**"Open to work" — the recruiters-only variant.** Recruiter-visible-only signalling exists and costs you nothing; the public green banner is a personal judgment call and does carry some stigma in senior circles. **📅 Volatile:** these product features change.

**The experience section, mirroring your resume's claim + metric + mechanism bullets.** Recruiters cross-check. Divergence between resume and profile creates friction you don't need.

**On posting:** the honest position is that most engineers get near-zero return and it is not a required activity. The exception that does work is posting *artifacts* — "I benchmarked X, here are the results, here's the repo" — because it converts your existing work into inbound rather than creating new work. If you write one post a month announcing something you actually built, that is the whole strategy. If you find yourself writing engagement-shaped commentary about AI trends, you have drifted into an activity with a terrible hourly rate.

**⚠ Trap:** a profile that describes you as your current job. Recruiters search for the role they're filling. If your profile says "Backend Engineer, payments," you will not surface in a search for AI engineers no matter how much LLM work you've shipped. **Your profile should describe the job you want, constrained by what is true.**

### Do I need to be on X to get hired in AI? Give me a decision rule rather than an opinion.

The honest state of play: a meaningful part of the AI field's professional conversation happens on X, several hiring managers at the AI-product companies on your target list are active there, and there are documented cases of hiring conversations starting from a post. It is also a substantial time sink with an extremely high variance return, and plenty of people get hired at these companies with no presence at all. Anyone who tells you it is mandatory is overstating it; anyone who tells you it is worthless is ignoring where a chunk of the field talks.

**The decision rule I'd apply:**

**Do it if** your target list leans toward small AI-product companies (Cursor, Sierra, Harvey, and their peers) *and* you have artifacts to post *and* you can commit to posting only outputs. Under those conditions the mechanism is clear: you post a benchmark, someone with reach shares it, a hiring manager sees it, and you have a warm inbound you could not have bought.

**Don't do it if** your target list leans big-tech (Meta, Google, Amazon, Microsoft, Databricks, Snowflake, Stripe), because those pipelines are recruiter-and-referral-driven and a public presence contributes essentially nothing. Spend the hours on the referral list instead.

**Either way, do the ten-minute minimum:** a profile that says what you build with a link to your GitHub and your artifact. This is a lookup surface — when someone hears your name, they may check — and having it be blank or embarrassing is a small, avoidable cost.

**The one thing worth doing if you engage at all:** post the artifact, not the take. "I ran this benchmark, here's the table, here's the repo, here's what surprised me" is a post with a reason to exist. Commentary on model releases has essentially infinite supply and provides no evidence about you.

**⚠ Trap:** the engagement treadmill. Chasing reach on a platform tuned to reward volume converts prep hours into dopamine at a punishing rate. **Cap it.** My rule for a job-search period: two hours a week total, all of it spent posting outputs or replying substantively to technical threads in your area, none of it spent scrolling. If you cannot hold that cap, the correct move is to not do it at all — the opportunity cost against building the eval-harness artifact is real and it is large.

### Lay out the order I should run my interview pipeline in. Does sequencing actually matter?

It matters more than almost anything else in this section, and it is the thing strong candidates most often get wrong — because the natural instinct is to apply to the company you want most, first, while you are least prepared. **Your interview performance improves measurably across the first three or four loops** — not because you learn more content but because you learn the *format*: how to think out loud, how to scope a design question in the first three minutes, how to recover from a stumble, how to manage a 60-minute clock. Burning your top choice on that learning curve is the single most expensive mistake available.

**The three waves:**

**Wave 1 — practice loops (weeks 1–3 of interviewing).** Two to four companies you would genuinely join but are not your top choice, chosen to *match the format* of your targets. If your targets run take-homes, pick practice companies that run take-homes. Treat these as completely real — prepare fully, want the offer — because a fake-feeling practice loop teaches you nothing. Their real purpose is calibration: after each one, write down every question you were asked within an hour, and grade your own answers.

**Wave 2 — target loops (weeks 3–7).** Your actual list, clustered so the onsites land within about a two-to-three-week window. Front-load the ones with the longest processes, since a slow company started in week three finishes around the same time as a fast one started in week six.

**Wave 3 — the reach or the late arrival.** Companies you'd take over everything, or ones that responded late. There is a real tension here: an exploding offer from wave 1 can force your hand before wave 2 finishes, which is exactly the situation clustering exists to prevent.

**📅 Volatile:** loop lengths change; verify current medians with the recruiter rather than trusting any published figure. The structural point stands regardless: **process length varies by weeks across companies, so "apply at the same time" and "finish at the same time" are completely different plans.**

**⚠ Trap:** applying everywhere in one weekend because it feels efficient. You will get your top choice's screen in week one, at your worst, and you cannot re-interview for six to twelve months at most companies. Applying is not free — **each application spends a scarce, non-renewable option.**

**🗣 Say this in the room,** when a recruiter asks about your timeline: "I'm in process with a few other companies and expect to be making a decision in about three weeks. I'd like to keep this one on roughly that timeline if that's workable." This is true, it is standard, and it is the sentence that creates the cluster.

### Explain offer clustering to me like an engineer. Why does the calendar matter so much?

Because your negotiating leverage is a function of the number of live, comparable offers you hold **simultaneously**, and offers have expiries measured in one to two weeks. That is a synchronisation problem with hard deadlines, which is a shape you already know: you are trying to get N asynchronous processes with different latencies to arrive within one window.

**The mechanism.** An offer alone gives you a number. Two offers give you a market. The difference is not marginal — a competing offer is the only piece of evidence that reliably moves a compensation band, because it converts your ask from a preference into a constraint the recruiter must clear or lose you. Sequential offers give you none of this: you accept or decline in isolation, and declining to keep waiting is a genuinely bad trade.

**The mechanics of achieving it:**

**Back-solve from the target date.** If you want offers in hand the week of day 60, and company A's process runs ~5 weeks while company B's runs ~9, then B starts around day 0 and A starts around day 28. **The applications are staggered so that the *finishes* line up.** This is the single most useful sentence in this question.

**Slow down the fast ones, legitimately.** Ask to schedule the onsite two weeks out because you want to prepare properly. This is a completely normal request that also reads as diligence.

**Speed up the slow ones, legitimately.** Once you have any offer, tell the other recruiters. "I've received an offer with a decision deadline of the 20th; I'm most interested here — is it possible to compress the remaining stages?" This works remarkably often, because recruiters are measured on closed offers and losing a finalist to a scheduling problem is an outcome they will spend effort to avoid.

**Ask for an extension, once, early, with a reason.** Standard, and much better received when asked on day one of an offer window than on the last day.

**💰 Math on why this is worth the coordination pain:** at senior AI-engineer levels, the spread between an accepted first offer and a competitively-negotiated one is routinely 10–25% of total compensation, and equity refresh and level can move more than base. On a package in the mid-hundreds of thousands, **a 15% delta is well into six figures over a multi-year vest.** Two to three weeks of calendar discipline is the cheapest money in your entire career. **📅 Volatile:** bands and equity structures change; benchmark current numbers before quoting any.

**⚠ Trap:** believing you must disclose which company made the competing offer, or share the exact figures. You do not have to name the company. You do have to be truthful about the existence of an offer — inventing one is a fireable-if-discovered, industry-small, catastrophically-bad-expected-value move, and recruiters at this level talk to each other.

### Two months in, I've had almost no responses. Debug my pipeline.

Treat it exactly as you'd treat a service with no traffic reaching the handler: instrument the funnel, find the stage where the drop is anomalous, and fix that stage rather than doing more of everything. Doing more of everything is what people actually do, and it is why the second month looks like the first.

**🔍 Failure taxonomy — localise by which stage is dropping:**

**Applications → recruiter screen is near zero (under a couple of percent, across 40+ applications).** The document or the channel is the problem, not you. In order of likelihood: (a) you are applying cold to hot reqs with no referral, which is the highest-volume, lowest-yield channel in existence — fix by shifting volume to referrals and outreach; (b) the resume's top third does not say what you're applying for, so a recruiter pattern-matching for "AI engineer" sees "backend engineer" and moves on; (c) you're applying to reqs you genuinely don't match on a hard filter (years, location, authorisation) and the rejection is mechanical; (d) parse failures from a multi-column PDF. Test (b) cheaply: give the resume to someone for twenty seconds and ask them what job you're applying for. If they get it wrong, that's your bug.

**Recruiter screen → hiring manager is dropping.** Almost always the positioning answer or the comp/logistics answer. Record yourself giving the 45-second intro. If it's a career history rather than a positioning statement, that's the bug. If the drop follows the comp conversation, your range is misaligned with the band and you need to recalibrate or re-target.

**Technical rounds are dropping.** Now it is content, and this guide's other 86 sections are the fix. But localise first: which round? A pattern of failing the same round is a study-plan problem; scattered failures across different rounds usually means format — running out of clock, not thinking out loud, not scoping before designing.

**Onsite → offer is dropping repeatedly.** Usually one of: the evaluation question (you designed without ever saying how you'd measure it), leveling mismatch (you're being read a level below your ask, which sometimes shows up as "no" rather than a lower offer), or the values/behavioural round, which at Anthropic in particular is reported as a primary filter rather than a formality.

**And the stage most people forget to check: are you even in the market you think you are?** If sixty applications produced two screens and both companies were 5,000-person enterprises while your resume is artifact-shaped and product-flavoured, the mismatch is targeting, not quality.

**⚠ Trap:** interpreting silence as a quality judgment. Non-response frequently means the req was filled, frozen, or never real. **Silence carries almost no information — only stage-transition rates do**, which is the entire reason to keep a spreadsheet: company, channel (cold/referral/outreach), date applied, stage reached, date of each transition. Twenty rows makes the leaking stage obvious; zero rows makes it invisible, and you will fill the gap with anxiety instead of data.

**🗣 Say this in the room,** if asked why you're still looking after a couple of months: "I've been deliberately sequencing — I wanted a few loops under my belt before the companies I actually care about, and I'd rather have several conversations finish around the same time than take the first thing." That is both true and a positioning statement.

### How do I handle a rejection, and when can I go back?

Two separate questions — the immediate handling and the long game — and the second one is where the real value is, because at this compensation level you will very likely interview at some of these companies more than once over a career.

**Immediately: ask for feedback, once, briefly, and without any hint of arguing.** "Thanks for letting me know — if there's any specific feedback the team is able to share, I'd find it genuinely useful for how I prepare." Most companies will give you nothing, and that is policy rather than a comment on you. Occasionally a recruiter at a smaller company gives you one sentence, and that sentence is worth more than a week of self-diagnosis. Never, under any circumstances, relitigate a question in that email; the fastest way to guarantee no future consideration is to be difficult on the way out.

**Then: write your own debrief within 24 hours, while the memory is intact.** Every question you were asked, verbatim if you can. Which ones you fumbled. What you'd say now. This corpus becomes the most accurate study guide you will ever have, because it is drawn from your actual target companies rather than from someone's list of common questions. After four loops you will notice the same three question shapes recurring, and *that* is what you should be drilling.

**On timing a return:** most companies apply a cooling-off period, commonly in the range of six to twelve months, and it frequently varies by how far you got and by the specific reason. **📅 Volatile and company-specific — ask the recruiter directly**, because they will usually tell you and it converts a guess into a date. Getting rejected after an onsite and after a screen are very different situations; the former often comes with an explicit "please reapply in N months," and that is a genuine invitation rather than a politeness.

**What actually makes a second attempt succeed** is a new, legible fact: a merged PR in a project they use, a published benchmark, a role with materially larger scope, or a specific gap closed. "I've been studying" is not a new fact. If the debrief told you that evaluation methodology sank you, the artifact you build in the intervening months is an eval harness — and your re-application email says so in one sentence.

**⚠ Trap:** letting one rejection re-sequence your pipeline out of discouragement — usually by pausing for two weeks and losing the clustering window. The pipeline is a portfolio; individual outcomes are high-variance and roughly independent. **Treat a rejection as one sample, log it, and keep the calendar intact.**

### Give me the drill. How do I know my materials are actually ready to send?

**🏋 Drill — the two-hour readiness audit.** Run this once, unaided, before you send a single application. Pass criteria are explicit; anything you fail is your next work item, in order.

**Block 1 — the 20-second scan (15 minutes).** Print the resume. Hand it to someone technical for exactly 20 seconds, then take it back and ask three questions: what role am I applying for, what's the strongest thing I've done, and what number do you remember? **Pass:** they get the role right, name your lead artifact, and recall at least one number. If they recall no number, your top third has no numbers in it.

**Block 2 — the interrogation (30 minutes).** Take the top five bullets on your resume. For each, spend six minutes writing down every follow-up question a hostile interviewer could ask, and answer them out loud. **Pass:** every bullet survives six minutes without you saying "I'd have to check" more than once. **Fail:** any bullet where you hit the floor inside ninety seconds — rewrite or delete it. This block is where the word "fine-tuned" usually gets deleted.

**Block 3 — the four numbers (15 minutes).** For your lead AI project, write down, unaided: the eval (set size, construction, metric), the cost per unit with the arithmetic, the latency with its decomposition, and the named failure mode with what you did about it. **Pass:** all four, from memory, arithmetic shown, in under fifteen minutes. This is the single highest-yield block in the audit because it is the literal content of the first question you'll be asked.

**Block 4 — the cold clone (20 minutes).** On a machine that is not yours, or in a fresh container with no environment variables set, clone your lead repo and follow only what the README says. Time it. **Pass:** something meaningful runs in under ten minutes with no undocumented step, and any cost incurred was printed before it was spent.

**Block 5 — the link check (10 minutes).** Every URL on your resume, profile and email signature. Load each on mobile, on a cold cache. **Pass:** all 200s, all under two seconds, nothing 404s, nothing shows a stale error page. Then set the weekly cron so you never do this manually again.

**Block 6 — the spoken assets (30 minutes).** Record yourself delivering, from memory: the 45-second positioning answer, the 3-minute artifact walkthrough, and the 60-second "tell me about your open-source contribution" answer. Listen back. **Pass:** the positioning answer lands between 40 and 55 seconds and contains one number; the walkthrough lands between 2:30 and 3:15 and hits all four numbers; the OSS answer states the size of the contribution in the first sentence.

**⚠ Trap:** doing the audit in your head. Every block above has a physical artifact — a printed page, a recording, a fresh container, a timer — precisely because self-assessment without an external check reliably returns "yes, I'm ready." You already know this from testing: **an assertion you didn't run is not a passing test.**

**🗣 Say this in the room** — the sentence this entire section exists to make true: "Everything on my resume is something I can be interrogated on for ten minutes, and everything I've linked, a stranger can run in under ten. I'd rather show you less and have all of it hold up."


---

## 4. Geography, Sponsorship, Remote and the India-to-Frontier-Lab Path

*Mastering this proves you have a strategy rather than a hope, given an IST-based candidate and a US/UK/EU-centric target list.*

### You're based in India and every company on your target list is headquartered in the US or UK. Walk me through your actual strategy — not your hope.

The mental model that fixes this whole section: **there are exactly five ways a US or UK company can pay you, and they have wildly different comp, equity, visa and career outcomes. "Getting hired" is not one decision, it is picking which of the five you are optimizing for before you send the first application.** Almost everyone applies to 200 jobs without knowing which modality they are applying into, which is why they get 200 silent rejections — the company's ATS knows the answer to "can we employ this person" long before a human reads the resume.

The five modalities:

1. **Relocate on a company-sponsored work visa.** They file H-1B (lottery), O-1, L-1, UK Skilled Worker, EU Blue Card, or Canadian work permit; you move. Full local band, full equity, full career surface. Highest ceiling, longest lead time, most gating.
2. **Local hire into a foreign-national entity you already qualify for.** You get a visa yourself (UK Global Talent, Canada Express Entry PR, an EU job-seeker route) *before* the job, so the company hires you as a normal local candidate with no sponsorship burden. This is enormously underrated and I will come back to it.
3. **Hire into the company's Indian entity (GCC or engineering site).** Local band, real equity in many cases, and — critically — an internal-transfer path (L-1) two to three years later.
4. **Employer-of-record (EOR) employment.** Deel, Remote, Velocity Global, or similar employs you in India on the US company's behalf. You are an employee of the EOR, not of the company. Blended comp, contested equity, no visa path.
5. **Independent contractor.** You invoice them. Highest flexibility, lowest everything else — no equity in most cases, no benefits, and a permanent-outsider status inside the org chart.

**⚠ Trap: treating "remote" as a single category.** Remote-US, remote-EMEA, remote-worldwide and remote-with-EOR-in-approved-countries are four different postings and only the last two can hire you. The tell is in the posting: "Remote (US)" means the person must have US work authorization and be physically in the US; it is not a geography-agnostic role and applying to it burns your name in their ATS.

**🗣 Say this in the room:** "My plan has a primary and a hedge. Primary is a relocation-sponsored role — I'd want to understand your immigration counsel's appetite for O-1 versus H-1B, because as an Indian national the H-1B lottery plus the EB-2 backlog is a fifteen-year path and O-1 to EB-1A is a three-year one. The hedge is that I'm building the evidence portfolio for a self-obtained visa so I can be a local hire rather than a sponsorship request."

The strategic point, and the one that separates a plan from a hope: **modality 2 inverts the power dynamic.** Every other modality asks the company to take on cost, legal risk and a six-to-twelve-month delay. Modality 2 asks them for nothing. A UK Global Talent visa in your passport turns you from "an Indian candidate we'd have to sponsor" into "a candidate who can start in four weeks," and the number of London-based AI roles that opens is not marginally larger, it is categorically larger.

**📅 Volatile:** every visa route, threshold, fee and processing time in this section moves — sometimes by executive action, with weeks of notice. Re-verify each one against the official government source in the month you apply, not against this guide.

### How do you find out, empirically, whether a specific company sponsors — without asking the recruiter and burning the question?

Because sponsorship data is public, and reading it is a five-minute research task that almost no candidate does.

The mental model: **a US employer cannot file an H-1B without first filing a Labor Condition Application with the Department of Labor, and LCA filings are published as a bulk disclosure dataset. Sponsorship is therefore not a policy you have to ask about — it is a log you can grep.** Treat it exactly like you would treat any other production question you refuse to answer from vibes.

The procedure I actually run before applying anywhere:

**1. Pull the DOL LCA disclosure data.** The Office of Foreign Labor Certification publishes quarterly and annual LCA disclosure files. Third-party front-ends (h1bdata.info, myvisajobs, h1bgrader and similar) index the same underlying data and are faster to query. What you want per company: how many LCAs filed in the last two fiscal years, for what job titles, at what wage levels, and in which offices. **📅 Volatile:** these front-ends come and go; the DOL file itself is the durable source.

**2. Read the shape, not just the count.** A company with 400 LCAs across "Software Engineer II/III/IV" is a machine — Google, Meta, Amazon, Microsoft sponsor as routine infrastructure. A company with 6 LCAs, all at senior/staff titles, sponsors selectively for people they really want. A company with zero has either never done it or has a policy against it, and you will not be the exception on your first hire.

**3. Check for O-1 and L-1 signal separately.** O-1 and L-1 do not generate LCAs, so a company can be a heavy O-1 filer and look empty in the H-1B data. This is common at AI startups, where the "we don't do H-1B because the lottery is a coin flip we can't plan around" position coexists with "we'll do O-1 for someone exceptional." You cannot see O-1 in public data; this is the one question worth spending on the recruiter.

**4. Decode the posting's own language.** "Must be authorized to work in the US without sponsorship now or in the future" is a hard no. Silence is usually a soft yes for big tech and a genuine unknown at startups. "We are unable to sponsor at this time" often means "for this req," and a different req at the same company can be different.

**⚠ Trap: assuming a company that sponsors will sponsor *you*, for *this* role.** Sponsorship decisions are made per-requisition against a headcount budget that includes legal fees and a hiring-timeline risk. A team that needs someone in six weeks will not sponsor even at a company that sponsors 400 people a year. The correct read of the LCA data is "is sponsorship institutionally possible here," not "will they do it."

**🗣 Say this in the room, to the recruiter, on call one:** "I want to be efficient with your time on one logistics item — I'm an Indian national based in India. I can see from public LCA data that you sponsor, so my question is narrower: for this specific req, is relocation with sponsorship in scope, and does your counsel handle O-1 as well as H-1B? If the answer is no I'd rather know now and stay in touch for a role where it is."

That framing does three things: it shows you did research, it makes the question cheap to answer, and it signals you will not waste four rounds of their time.

### Explain the H-1B route to me as though I've never filed one — and then tell me why it's a bad primary plan for an Indian national specifically.

The mechanism first, because the details are where the strategy lives.

H-1B is a temporary work visa for "specialty occupations," tied to a sponsoring employer, initially granted for three years and extendable to six (longer if a green-card process is far enough along). The annual supply is capped: **65,000 regular visas plus a further 20,000 reserved for holders of a US master's degree or higher.** Demand has exceeded supply by multiples for over a decade, so USCIS runs a **registration lottery** — the employer registers you in a spring window, a random selection is drawn, and only if you are selected does the full petition get filed. Selected petitions with an approved start date begin at the start of the government fiscal year, **October 1**. **📅 Volatile:** the registration window, the selection mechanism (there have been repeated attempts to move from a pure random draw to a wage-weighted selection), and the fee structure have all been actively changed and litigated in recent years; a substantial new fee on certain new H-1B petitions was imposed by executive action and immediately challenged in court. Verify the current rule and cost on uscis.gov before you build any plan on it.

So the timeline, best case: register in spring, get selected (probability well under half in recent years for the regular pool), start in October. Miss the lottery, wait a year. That is the first problem.

The second problem is the one that actually matters and that almost nobody thinks about early enough. **H-1B is temporary. The green card is the destination, and for Indian nationals the employment-based green card is subject to per-country limits that produce a backlog measured in decades, not years.** The EB-2 and EB-3 categories for India have had priority-date waits that, at various points, imply a wait longer than a working career. EB-1 for India is also backlogged but by a far smaller margin. **📅 Volatile:** priority dates move monthly in the State Department Visa Bulletin, sometimes backward; check the current bulletin rather than any remembered number.

**💰 Math on the real cost of the H-1B-first plan.** Suppose you get an H-1B on your second lottery attempt — you lose one year waiting. You start in October. Your employer starts PERM (the labor-certification step for EB-2/EB-3) after, realistically, twelve months of tenure; the prevailing-wage determination and the recruitment steps that must precede filing add several more months, and your **priority date is set the day the PERM application is filed** — call it around year 2 — with I-140 approval typically a year or so behind that. **📅 Volatile:** DOL prevailing-wage and PERM processing times have swung by many months in both directions; check the current DOL processing-times page rather than this estimate. If the EB-2 India queue at that time implies a decade-plus wait, you are looking at H-1B extensions past year 6 (permitted once an I-140 is approved) and a green card somewhere north of year 12 from today. During all of that: your spouse cannot work until the I-140 is approved and an H-4 EAD is granted, changing employers is possible but requires a transfer petition, and getting laid off starts a grace-period clock measured in days. **That is not a career plan, it is a mortgage on your optionality.**

**⚠ Trap: the "just get in on H-1B and figure it out" reflex.** It is the default advice in Indian engineering circles because it was the right answer in 2010. It is now, for a candidate with a public artifact portfolio, the *slower* route than O-1 → EB-1A, and it is dramatically less flexible. I would take an H-1B if it were offered, but I would not *build the plan around it*, and I would negotiate EB-1A/NIW filing support into the offer on day one rather than accepting the company's default PERM track.

**🗣 Say this in the room:** "H-1B works but it's a lottery with a ten-plus-year backlog behind it for Indian nationals. What I'd want to know is whether your immigration counsel will file an O-1 and, once I'm in, support an EB-1A rather than defaulting me into PERM/EB-2 — because that's the difference between a three-year path and a fifteen-year one."

### What is a cap-exempt H-1B, and could you actually engineer your way into one?

The mental model: **the cap is a property of the petitioning employer, not of you.** Certain employer classes are statutorily exempt from the numerical cap, which means they can file an H-1B petition for you *at any time of year, with no lottery*. If you can get employed by one, you have converted a coin flip with a twelve-month cycle time into a normal hiring process with a normal processing time.

The exempt classes are, broadly: institutions of higher education; nonprofit entities related to or affiliated with such institutions; nonprofit research organizations; and government research organizations. **📅 Volatile:** the precise definition of "related or affiliated," and how aggressively USCIS reads it, has moved with regulation and litigation — this is a question for an immigration attorney, not for a guide.

Two structural facts make this strategically interesting for an AI engineer:

**First, a great deal of AI research infrastructure sits inside exempt employers.** University AI labs and institutes, and independent nonprofit research organizations, hire research engineers — and "research engineer at an AI institute" is a role a strong backend engineer with transformer internals and serving-stack depth is genuinely competitive for. It is not a consolation prize; it is often the most technically interesting job on the list.

**Second, cap-exempt status can carry into concurrent employment.** The well-known structure is: hold a cap-exempt H-1B with an exempt employer, and a for-profit employer may then file a *concurrent* H-1B petition, which can also be treated as cap-exempt for as long as the qualifying cap-exempt employment continues. This is real, it is used, and it is exactly the kind of arrangement that must be set up by a competent immigration attorney rather than improvised — if the exempt employment lapses, the concurrent petition's basis lapses with it.

**⚠ Trap: thinking cap-exempt is a loophole you can execute alone.** It is a legitimate statutory category, but the failure mode is a nominal part-time affiliation that USCIS finds is not bona fide employment — which does not just fail, it creates a misrepresentation record that follows you. The rule I would enforce: the exempt job must be a real job you actually do, with real duties and a real wage, or you do not do this at all.

**🔍 Failure taxonomy for this route:** (a) the exempt employer's wage is far below your market rate and you cannot afford the bridge year — check the LCA prevailing wage for the role and location before you commit; (b) the exempt role is research-track and does not build the applied-AI narrative you need for your next move — mitigate by choosing infrastructure/serving work over pure research support; (c) the concurrent-employment structure collapses if you leave the exempt job, so you have coupled two employments into a single point of failure.

### Talk me through O-1A. What are the criteria, and honestly, where do you stand today?

The mental model: **O-1A is not a "genius visa," it is an evidentiary standard.** It exists for individuals with "extraordinary ability" in the sciences, business, education or athletics, and in practice you qualify by documenting that you satisfy at least **three of eight regulatory criteria**, and then surviving a holistic "final merits" judgement that the evidence taken together shows sustained acclaim. The word that matters is *documented*. This is a paperwork exercise built on top of a career, and the paperwork is winnable by an engineer who plans for it eighteen months out.

The eight criteria, in the regulation's own terms, are: nationally or internationally recognized **awards**; **membership** in associations that require outstanding achievement judged by experts; **published material about you** in professional or major trade publications; **participation as a judge** of the work of others in your field; **original contributions of major significance** to the field; **authorship of scholarly articles** in professional journals or major media; employment in a **critical or essential capacity** for organizations with a distinguished reputation; and a **high salary** relative to others in the field. There is also a catch-all for comparable evidence where the listed criteria don't fit the occupation. USCIS has issued policy guidance specifically discussing how these criteria apply to STEM fields, which is the guidance your attorney will lean on. **📅 Volatile:** confirm the current criteria list and the current policy-manual guidance at uscis.gov — the STEM guidance in particular has been updated more than once.

Mechanically: O-1A is **employer- or agent-petitioned** (you cannot self-petition, unlike EB-1A), granted for up to **three years initially with one-year extensions**, has **no annual cap and no lottery**, permits **dual intent** (you can pursue a green card without prejudicing it), and requires an **advisory opinion** from a peer group or labor organization — in software there generally isn't a relevant union, so this is typically handled via expert opinion letters. Processing can be fast, and premium processing is available.

The honest self-assessment for a candidate in your position: today you probably do not have three criteria fully evidenced. What you likely have partial claims on are **critical or essential role** (needs a letter from leadership at a company with a documentable reputation, describing scope and impact, not a job description), **original contributions of major significance** (needs merged upstream OSS with adoption evidence *plus* independent expert letters attesting to significance — the letters do the work, the commits are the exhibit), and **authorship** (a technical guide is not a scholarly article; a workshop paper, an arXiv preprint with citations, or bylined technical writing in a recognized publication is much closer).

**⚠ Trap: believing GitHub stars are evidence.** They are not, on their own. USCIS does not know what a star is and adjudicators have seen every inflated metric. What converts is **third-party attestation**: an independent expert in the field, with no employment relationship to you, stating in a signed letter what your contribution was and why the field cares. Stars are a supporting exhibit under that letter, never the claim itself.

**🗣 Say this in the room, if immigration comes up:** "I'm building toward O-1A rather than relying on the H-1B lottery, because it's uncapped, it's filable year-round, and it feeds EB-1A rather than the EB-2 India backlog. What I'd need from an employer is a willingness to be the petitioner — the evidence portfolio is my responsibility, not yours."

### Take your actual artifacts — the interview guide, OSS pull requests, benchmarks, talks — and map them onto the O-1A criteria. Which three do you win on, and what's the twelve-month plan?

This is the question I would ask to find out whether someone has a strategy or a slogan, so let me answer it as a plan with deliverables rather than a list of hopes.

**Criterion: original contributions of major significance.** The strongest available claim. The build: land **merged, non-trivial pull requests in infrastructure the field visibly depends on** — a serving engine, a training library, a structured-decoding library, an agent framework. "Non-trivial" means a scheduler behavior, a correctness fix in a batching path, a memory accounting bug — not docs and not typos. Then collect the attestation: two to three signed letters from maintainers or from independent senior engineers at other companies, describing what the contribution changed and who it affects. Deliverable by month 9: three merged substantive PRs across two projects, plus release notes or changelog entries naming the change, plus two independent letters.

**Criterion: authorship of scholarly articles or major-media technical writing.** The guide alone does not clear this. What does: an **arXiv preprint** with a real result — the honest, achievable version for an applied engineer is a rigorous benchmark or reproduction study nobody else ran, with methodology, released code, and a released dataset. A benchmark of, say, structured-output reliability across engines under load, or the cost/quality frontier of retrieval configurations at fixed latency, is a genuine contribution and is publishable at a workshop. Deliverable by month 12: one preprint plus one workshop submission.

**Criterion: judging the work of others.** The cheapest criterion to acquire and the one most people miss. Paths: serve as a **program-committee reviewer for a workshop** (workshops actively recruit reviewers and accept applied engineers), judge a hackathon with a documented judging role, or become a **reviewer/triager on an OSS project** with a documented review history. Deliverable by month 6: one documented reviewing role with a confirmation email or public listing.

Those three are the realistic trio. Two more worth opportunistically building:

**Published material about you** — not written *by* you. A podcast appearance, a conference talk that gets written up, or a trade-publication piece that quotes you and your work. Talks are the lever: a CFP accepted at a recognized conference produces both a speaking record and, often, coverage.

**High salary** — becomes available *after* your first US-band offer, evidenced against wage survey data for the occupation and location. This is why sequencing matters: a high-band offer is itself an O-1 exhibit for the *next* filing.

**🏋 Drill (45 minutes, unaided).** Open a blank document. Write the eight criteria from memory — you should get at least six. For each, write one line: the specific artifact you would submit, its current status, and the single next action with a date. Pass criterion: at least three criteria have a named artifact with a next action inside 90 days, and zero of your evidence lines are "GitHub stars" or "my blog has traffic."

**⚠ Trap: starting the portfolio after you start interviewing.** The evidence must show a *pattern* — adjudicators are unimpressed by three artifacts that all appeared in the ninety days before filing. Start eighteen months before you need it, which means start now regardless of where you are in a loop.

### Your current employer has a US parent or a US client relationship. Is L-1 a real option for you?

The mental model: **L-1 is an intracompany transfer, so it is not something you apply for — it is something you become eligible for by working somewhere for long enough.** That makes it the slowest route to start and one of the most reliable to finish, and it is the single strongest argument for taking an India GCC role at a US company as a deliberate two-to-three-year move rather than as a compromise.

The mechanics: L-1 requires that you have been employed by a qualifying related foreign entity (parent, subsidiary, affiliate or branch of the US petitioner) for **at least one continuous year within the preceding three years**, and that you are transferring into a role that matches your category. **L-1A** is for managers and executives, granted up to a maximum of **seven years**. **L-1B** is for employees with **specialized knowledge**, maximum **five years**. There is no annual cap and no lottery. Large multinationals with an approved **blanket L** petition can process individual transfers much faster than filing individually. **📅 Volatile:** L-1B "specialized knowledge" adjudication standards have tightened and loosened repeatedly and denial rates have swung a great deal; check current trends with counsel.

The strategic asymmetry that most people miss: **L-1A leads to EB-1C**, the multinational manager/executive green-card category, which sits in the EB-1 bucket — the same bucket as EB-1A, dramatically less backlogged for India than EB-2. L-1B does not have that direct bridge; an L-1B holder typically ends up back in EB-2/EB-3 with the standard India wait. So if you are choosing between an IC track and a management track at a company that transfers people, the visa consequences of that choice are enormous and are almost never discussed in the career conversation where the choice is actually made.

**💰 Math on the L-1 detour.** Say you join a US-company India site today at ₹70 LPA. You need 12 months of qualifying employment before a transfer is even filable, and realistically an internal transfer requires a business justification and a receiving team, which pushes it to 24–36 months. Against that: an O-1 route can plausibly be filed within 12–18 months if the evidence portfolio is executed. So L-1 is *slower to first US paycheck* but *far more certain*, because it does not depend on an adjudicator's judgement about your acclaim. The correct posture is to run both: take the GCC role that starts the L-1 clock, and build the O-1 portfolio in parallel. They do not conflict — the GCC role's scope letters are themselves O-1 evidence for the "critical or essential capacity" criterion.

**⚠ Trap: assuming any Indian employer with a US client qualifies.** L-1 requires a *qualifying corporate relationship* — common ownership and control between the entities. A services company staffing a US client is not a parent/subsidiary relationship with that client, and body-shop L-1B filings are precisely the pattern that drew heightened scrutiny. Working at an Indian IT services firm with US clients does not put you on an L-1 track to those clients.

**🗣 Say this in the room, in a GCC interview:** "I want to be direct that a medium-term US transfer is part of why this role interests me. Can you tell me how often engineers on this team have transferred to a US site, and whether the company files L-1A for senior ICs or only for managers?" That is a fair question, it signals ambition rather than flight risk, and the answer tells you whether the site is a real engineering org or a cost center.

### Compare the green-card endgames for an Indian national — EB-2 with PERM, EB-2 NIW, and EB-1A. Which are you targeting and why?

The mental model: **the visa gets you in; the category decides whether you can ever leave the job you got in with.** Every employment-based green card for an Indian national queues behind a per-country limit, so the only variable that meaningfully changes your life is *which bucket you queue in*, because the buckets have wildly different queue lengths.

**EB-2 with PERM** is the employer-driven default. The employer runs a labor-market test (PERM) to show no qualified US worker is available, files an I-140, and your priority date is set. It is employer-sponsored end to end, which means the sponsorship is a leash: change jobs early and you generally restart. For India, the EB-2 queue has been the multi-decade one. **📅 Volatile:** check the current Visa Bulletin; both EB-2 and EB-3 India dates have moved and even retrogressed.

**EB-2 NIW** — National Interest Waiver — is EB-2 with the job-offer and labor-certification requirements waived because your work is in the national interest. Crucially it is **self-petitionable**: you file for yourself, and the approval travels with you rather than with an employer. The governing framework is the three-prong test from **Matter of Dhanasar (AAO, 2016)**: the endeavor has substantial merit and national importance; you are well positioned to advance it; and on balance it benefits the US to waive the job offer and labor certification. **📄 Note:** Dhanasar replaced the older, more restrictive NYSDOT framework, and is why NIW became realistically available to engineers rather than only to academics. The catch for you: NIW is still an **EB-2** category, so it inherits the EB-2 India queue. You get freedom of employer, not speed.

**EB-1A** — extraordinary ability — is also **self-petitionable**, uses a criteria structure closely parallel to O-1A but adjudicated to a higher standard, and sits in **EB-1**, whose India queue has historically been far shorter than EB-2's. This is the actual prize.

So the decision rule I would state plainly: **for an Indian national, the only green-card plans worth building around are EB-1A and, secondarily, EB-1C via L-1A. EB-2 — with or without NIW — is a fallback you file in parallel to hold an early priority date, not a plan.** You can file NIW and EB-1A concurrently and, if approved separately, port the earlier priority date, which is exactly why filing NIW early is worth it even if EB-1A is the target: it is a cheap option on a date.

**⚠ Trap: conflating O-1A approval with EB-1A approval.** They share a criteria vocabulary, and an approved O-1A is helpful evidence, but EB-1A is adjudicated more strictly and an O-1 approval is not a rubber stamp. I have seen the assumption "I got O-1, EB-1A is a formality" stated confidently; it is wrong, and the gap between them is exactly the difference between "extraordinary for a temporary role" and "sustained national or international acclaim."

**💰 The cost of getting the category wrong.** Concretely: EB-1 versus EB-2 for India, on historical Visa Bulletin behavior, is plausibly a difference of a decade in when you get a green card. A decade of green-card wait is a decade in which you cannot freely change employers without re-running sponsorship, cannot easily start a company, and your spouse's ability to work depends on an I-140 approval and an EAD. Priced as career optionality, that is worth more than any single offer's equity package — which is why I would trade meaningful cash comp for an employer that commits, in writing in the offer letter, to filing EB-1A with a firm that does them well.

### UK Global Talent versus a Skilled Worker visa — which are you pursuing, and why does the distinction matter so much for your situation?

The mental model, and this is the single highest-leverage fact in this whole section: **the UK Global Talent visa is not employer-sponsored. You obtain it yourself, before you have a job, and then you are a normal candidate.** It converts you from modality 1 (ask a company to sponsor) to modality 2 (be locally hireable), and that changes which roles will even talk to you.

**Global Talent** requires an **endorsement** from an approved endorsing body in your field, and for digital technology the endorsement assesses whether you are a recognized leader (**Exceptional Talent**) or an emerging leader (**Exceptional Promise**) — the promise route is explicitly designed for people earlier in the arc. Endorsement is evidence-based and the evidence categories will look familiar after the O-1A discussion: significant technical contributions, product/company impact, recognition beyond your employer, open-source work, publications, talks. **📅 Volatile:** the endorsing body for the digital-technology route changed hands in 2023 and endorsement criteria have been revised; check gov.uk for the current endorsing body and criteria before you assemble anything. There is also a separate route where certain prestigious fellowships or awards short-circuit the endorsement step entirely.

What it buys: **no employer sponsorship required, no job offer required, freedom to change jobs or be self-employed, and a shorter path to settlement** — the Exceptional Talent route has historically qualified for indefinite leave to remain after three years, versus five for Exceptional Promise. **📅 Volatile:** settlement rules were under active reform; verify.

**Skilled Worker** is the ordinary sponsored route: a licensed sponsor, a certificate of sponsorship, a salary at or above the general threshold and the going rate for the occupation code, and the visa is tied to that employer. The general salary threshold and the skill-level requirement were both raised substantially in recent years. **📅 Volatile:** the threshold has moved more than once and there are separate, lower thresholds for new entrants and for some occupations; check the current figure.

**The decision rule:** pursue Global Talent *as a background process starting now*, independent of any job search, because it has no downside and a nine-to-twelve-month build for the evidence. Accept a Skilled Worker sponsorship if a great role offers it, but never let it be the only plan, because it recreates the leash problem — leaving that employer means finding another licensed sponsor.

**🗣 Say this in the room:** "For the UK I'm going through Global Talent rather than asking for Skilled Worker sponsorship — it's self-obtained and not tied to an employer, so from your side I'd be a standard hire with no sponsorship cost or timeline risk."

**⚠ Trap: assuming the UK is a consolation prize versus the US.** For AI specifically it is not. London and, increasingly, Dublin and Zurich host real frontier-lab engineering, and the compensation gap versus the US, while real, is much smaller than the gap between a US band and an India band. More importantly, a UK role at a US-headquartered company is one of the most reliable *later* bridges to a US transfer, because now you are an internal candidate at a company with the corporate relationship an intracompany transfer needs.

### Sketch the non-UK European options for me — Blue Card, Netherlands, Ireland — and tell me what would actually make you pick one.

The mental model: **continental Europe generally trades ceiling for certainty.** The visa routes are more deterministic than the US lottery, the paths to permanent residence are shorter, and the compensation is materially lower — often by a factor of two to three against a US band for the same seniority. You pick Europe when you are optimizing for a fast, high-probability path to residence and an excellent quality of life, not when you are maximizing near-term comp.

**EU Blue Card** is the pan-EU highly-skilled route, implemented per member state. The common shape: a job offer at or above a **salary threshold set annually by the member state**, with a lower threshold for shortage occupations — IT roles frequently qualify for the reduced threshold. Germany's Skilled Immigration Act reforms also created a pathway for **IT specialists without a formal degree** who can evidence several years of relevant professional experience, which matters for anyone whose credential story is non-linear. Blue Card carries intra-EU mobility rights after a qualifying period and a comparatively fast route to permanent residence, accelerated further by language acquisition. **📅 Volatile:** every salary threshold here is re-set annually and the shortage-occupation list changes; the German figures in particular move each January. Verify on the member state's official portal.

**Netherlands** runs the **Highly Skilled Migrant** scheme: your employer must be an IND-**recognized sponsor** (a published register you can check before applying), and the salary must exceed an age-banded threshold that is indexed annually. Recognition means processing is fast and predictable. The historic sweetener was the **30% ruling**, a tax facility on a portion of income for incoming skilled migrants — it has been repeatedly amended, capped and restructured in recent years, so **📅 Volatile:** treat any specific percentage or duration you have heard as stale until verified. Amsterdam has a genuine and growing AI employer base, and the recognized-sponsor register is a searchable, empirical list of who can hire you — the European analogue of the LCA-grep move.

**Ireland** uses the **Critical Skills Employment Permit**, with a lower salary threshold for occupations on the critical skills list (software and AI roles are generally on it) and a higher one otherwise; the permit route leads to long-term residence relatively quickly and family members get favorable treatment. **📅 Volatile:** both thresholds were raised recently and are scheduled to keep rising. Dublin's specific relevance to your target list is that it is the EMEA hub for a large number of US tech companies including several with real AI engineering presence — which makes it, like London, a strong internal-transfer springboard.

**The decision rule I would apply:** if a role in Dublin or London at a US-headquartered AI company is on the table, take it over a comparable Amsterdam or Berlin role at a European company, purely on the internal-mobility optionality. If the choice is between a European role and continuing to wait for a US outcome, take the European role — being inside the industry in a Western market beats being outside it with a better theoretical ceiling, and the residency clock starting is itself worth a lot.

**💰 Math on the comp gap you are accepting.** A senior AI engineer band that might be $350k–500k total comp in the Bay Area is plausibly £110k–170k in London and €90k–140k in Amsterdam or Dublin at the same level — call it 30–45% of the US number on a nominal basis. **📅 Volatile:** these bands move fast in AI; re-benchmark. Before concluding Europe is a bad trade, adjust for the things the US number has to fund: healthcare, university costs, and — for an Indian national — a green-card queue that a European permanent residence route resolves in roughly five years with no lottery. I would not describe Europe as a downgrade; I would describe it as a different risk profile, and if the US path stalls, it is by a wide margin the best hedge available.

### Is Canada a detour or a shortcut?

Both, and which one it is depends entirely on whether you treat it as a destination or as a staging area — so let me separate the two cases.

**As a destination it is the most deterministic route on the list.** Canada's **Express Entry** system scores you on a points grid (age, education, language test results, work experience, arranged employment, provincial nomination) and invites the top of the pool in periodic draws — including **category-based draws** that target specific occupation groups, with STEM occupations having featured among them. The critical property: **you can obtain permanent residence without a job offer**, on your own initiative, on a timeline of roughly a year if your score clears. For an Indian national, "permanent residence, self-obtained, in about a year" versus "employment-based green card, employer-tied, in about a decade" is not a close comparison on certainty. Separately, the **Global Talent Stream** offers an employer-driven work-permit route with an aggressive processing-time service standard for eligible tech occupations. **📅 Volatile:** Canada has recently tightened overall immigration levels and adjusted Express Entry categories and cut-off scores substantially; the draw scores and category list change often — check the IRCC site.

**As a staging area, be honest about what it does and does not do.** It does put you in a North American time zone with a legal work status and a compensation band better than India's; it does make you employable by the Canadian arms of most US tech companies, which then makes an **intracompany transfer to the US a live option** on the L-1 clock; and Canadian citizenship after the residency requirement makes the **TN** category available, which is a fast, renewable US work status for citizens of Canada and Mexico in listed professions. **📅 Volatile and jurisdiction-specific:** TN's occupation list is narrow and its application to software roles has historically required mapping onto categories like computer systems analyst — this is a question for counsel, not a plan you assume.

What it does *not* do: Canadian compensation for senior engineers is well below US levels — Toronto and Vancouver senior bands commonly land at roughly half to two-thirds of a comparable US number in nominal CAD-to-USD terms, and the AI-specific premium is thinner. So the "go to Canada for two years then transfer" plan costs you real money and adds years.

**🗣 Say this in the room, if someone challenges the Canada plan:** "I treat Canada as insurance, not as the plan. Express Entry is the only route on my list where the outcome depends on a published points threshold rather than on someone's discretion, so it's the hedge I run in the background while the O-1 and Global Talent tracks play out."

**⚠ Trap: starting Express Entry the month you need it.** The scoring depends on a valid language test result and, often, an educational credential assessment — both of which take weeks to months to obtain and both of which expire. Get the language test and the credential assessment done *now*, while they are cheap and you are not under pressure; a profile you can submit in a week is worth far more than a plan you can start in three months.
### Take my target list — Cursor, Perplexity, Notion, Figma, Sierra, Harvey, Glean, Ramp, and then Meta, Google, Amazon, Microsoft, Databricks, Snowflake, Stripe — and sort it by what's actually achievable from India.

I will answer this as a **taxonomy plus a verification procedure**, not as a table of company facts, because company-by-company hiring policy is the most volatile information in this entire guide — offices open, remote policies flip, and a claim I make today is a liability in your loop next quarter. **📅 Volatile: verify every company-specific claim below against the company's own careers page and current job postings before you act on it.**

**Tier A — large employers with mature global entities and routine sponsorship.** Google, Microsoft, Amazon, Meta and similarly-sized companies operate substantial Indian engineering sites *and* sponsor relocation at volume. For these, the highest-probability path is not a US application at all: it is **apply to the Indian site, perform, transfer.** Their LCA footprint is enormous, their internal mobility programs are real, and their India AI orgs do genuine work. The failure mode here is being hired into a support or sustaining role and discovering there is no path from it to the AI org.

**Tier B — enterprise/data-platform companies with real Indian engineering.** Databricks, Snowflake, Stripe and peers have built Indian engineering sites that own production surfaces rather than being test farms. Same pattern as Tier A but with much smaller headcount and thus much more selective hiring — which is good news, because these sites hire senior backend engineers directly into product teams. Verify the *charter* of the Indian site before you apply: ask in the screen what the site owns end to end.

**Tier C — AI-product companies, mostly US-only and mostly in-person.** Cursor, Sierra, Harvey, Perplexity, Ramp and companies at that stage are typically concentrated in one or two offices with a strong in-person culture, and several are explicit that they do not hire remote. Some have opened or announced international offices; the announcements are recent and the reality lags them. For this tier the honest read is: **you are competing for relocation with sponsorship, or you are not competing.** Do not spend your application budget here without checking whether the specific req is remote-eligible, and do not assume an announced office means it is hiring engineers yet.

**Tier D — companies with genuine remote-international or EOR practice.** These exist across every tier and cannot be predicted from company size or fame; the empirical signal is a careers page that lists **specific approved countries** for remote hiring, or a job posting with an "India" or "Remote — Global" location. If India is on the approved-country list, they already have EOR infrastructure and hiring you is a solved problem for their people ops.

**The verification procedure, in order, per company (fifteen minutes each):**
1. Open the careers page; filter by location. Does India appear at all? For engineering, or only for sales/support?
2. If yes, open three engineering reqs at the Indian site and read what the team owns. "Own the X service end to end" versus "support the US team" is the entire difference.
3. If no India, check whether any req is remote-eligible and read the fine print for the country list.
4. Grep the DOL LCA disclosure data for the company's H-1B filing volume and titles.
5. Check LinkedIn for engineers at that company located in India — headcount and titles tell you what the site actually is.

**⚠ Trap: applying to a US-located req and hoping they'll make an exception.** They will not, because the exception is not the hiring manager's to make — it is a legal-entity question owned by people ops, and the answer is "we don't have an entity there" long before anyone reads your resume. The application is not rejected on merit; it is rejected on geography, and it costs you the ability to apply cleanly to a suitable req later.

**🗣 Say this in the room:** "I've sorted your postings by whether the Indian site owns product surfaces or supports US teams, and I'm applying to the former. If there's a US relocation path from that site within two to three years, that's the version of this role I'm most interested in."

### Contractor, employer-of-record, or the company's own subsidiary — what's the difference and which should you push for?

The mental model: **these three differ in who your legal employer is, and that single fact cascades into equity eligibility, benefits, termination protection, tax treatment, and — most importantly — whether you are inside or outside the org chart.**

**Independent contractor.** You invoice; they pay. Nobody withholds anything. In India you register a proprietorship or a company, you handle GST if applicable, and you take advance-tax obligations onto yourself. Advantages: highest gross rate (they are not paying employer-side costs), maximum flexibility, and you can serve multiple clients. Disadvantages that matter more than they sound: **no equity in the overwhelming majority of cases** (incentive stock options in the US are statutorily restricted to employees; contractors, if granted anything, get non-qualified options or nothing), no severance, no notice period, and a real legal risk on *their* side — many jurisdictions will reclassify a full-time-equivalent long-term contractor as a de facto employee, which is exactly why mature companies stop doing this.

**Employer of record.** A third party (Deel, Remote, Velocity Global, Globalization Partners and similar) has a legal entity in India and employs you there, invoicing the US company for your cost plus a fee. You get an Indian employment contract, statutory benefits, PF, payslips. The US company gets to treat you like an employee day to day without incorporating in India. **📅 Volatile:** the EOR vendor landscape consolidates constantly and specific vendors' India capabilities differ; the mechanism is durable, the names are not.

**Subsidiary employment.** The company has its own Indian entity and you are its employee. This is what a GCC is. Best benefits, cleanest equity story, and — the part that matters strategically — **you are an employee of the actual corporate group, which is what an intracompany transfer requires.**

**The ranking I would push for, and why:** subsidiary > EOR > contractor, and the gap between subsidiary and EOR is much larger than it looks. Subsidiary employment gives you the corporate relationship an L-1 needs; EOR does not, because your legal employer is the EOR, not the company. **An EOR employee cannot be intracompany-transferred, because there is no intracompany relationship.** That is the single most important and least-known fact in this question, and it means that accepting an EOR arrangement at a company you hoped to transfer within is quietly closing a door you thought you were opening.

**💰 Math on the EOR overhead you should know is being spent on you.** EOR vendors typically charge either a flat per-employee monthly fee or a percentage of payroll — commonly in the region of a few hundred dollars a month per head, or a mid-single-digit to low-double-digit percentage. **📅 Volatile:** pricing is negotiated and moves. So if your gross is $90,000/yr, the company's fully-loaded cost is that plus statutory India employer contributions plus the EOR margin — call it $100k–110k against a budget line that could have bought a US hire at $220k+ fully loaded. Know this number: it is your leverage, because it demonstrates you are the cheap option, and it is your risk, because a cost-cutting cycle cuts the line item that is easiest to cut and EOR contracts terminate more easily than employment ones.

**⚠ Trap: accepting "we'll convert you to full-time later."** Conversion requires the company to incorporate in India or to already have an entity — it is a corporate-structure decision, not an HR one, and no hiring manager can commit to it. Ask directly: "Do you have an Indian legal entity today?" If the answer is no, the conversion promise is a hope, and I would price the role as if the conversion never happens.

### You get a remote-from-India offer at a US AI startup. Build the comp model. Show me the arithmetic.

The mental model: **there are three reference bands, not one, and negotiation is a fight about which band the conversation anchors to.** The US band for the role, the India local-market band for the role, and the "blended" or "geo-adjusted" band the company invents to split the difference. Whoever names a band first usually wins, which is why the number you walk in with must be constructed, not felt.

**Step 1 — establish the US band.** Use levels.fyi for the company and level, cross-checked against DOL LCA disclosure wage data (which gives you the *base* the company actually certified for that title and location — a hard, filed number, not a self-report) and against Blind or Pave-derived aggregates. Say the model lands at $200k base + $150k/yr equity = **$350k total comp** for a senior AI engineer at a mid-stage US AI company. **📅 Volatile: AI-specific bands have moved sharply; re-benchmark within the month you negotiate.**

**Step 2 — establish the India band for the same work.** Senior engineers at strong Indian sites of US companies plausibly land ₹60–110 LPA all-in depending on company and level. At ₹85 LPA and ₹86/USD that is roughly **$99,000**. **📅 Volatile: both the band and the exchange rate move.**

**Step 3 — the gap is 3.5×, and that is the whole negotiation.** Companies justify the gap on cost-of-living, on local-market competitiveness, and on "we pay the market where you live." The counter-arguments that actually work are: (a) **you are not competing in the local market, you are competing for a US-team seat and delivering US-team output**; (b) **your replacement cost is the US hire they did not make**; and (c) if they want you to work US hours, they are buying your evenings, which is a real cost with a real price.

**Step 4 — construct the ask as a percentage of the US band with a stated rationale.** State the ask as a fraction of **US total compensation** and keep that denominator fixed for the whole negotiation, because the moment you let the conversation slide between "base," "cash" and "total comp" you have lost the ability to compare anything. A remote-from-India *opening* ask is typically **45–65% of US total comp**, and the ones that hold near the top of that range are the ones with a written justification. Against the $350k model: **$160–190k cash, plus the same equity grant as a US hire at that level** — the cash alone is 46–54% of the $350k denominator (the equity ask deliberately sits outside that ratio, which is why you keep the two conversations separate), and you say the percentage out loud so the concession is visible.

**💰 Worked example of a landed offer.** Ask: $165k cash. Their opening: $95k ("that's above the India market"). Your move — reframe the denominator: "I'm not asking for your US band. Your filed base for this title is $205k and levels.fyi puts total comp around $350k; $165k is 47% of that total, so I've already conceded more than half. At $165k I cost you less than a US hire's employer-side load alone, and I'm the one absorbing the time-zone cost." Realistic landing zone: **$130k–145k cash** — 37–41% of the $350k — plus equity, if you keep equity as a separate, non-tradeable ask. At ₹86/USD, $135k is roughly ₹1.16 crore, comfortably above any local band, which is exactly why this is worth twenty minutes of friction rather than accepting the first number.

**⚠ Trap: negotiating cash and equity as one number.** The company's cash budget for an EOR line item is often genuinely constrained, but equity comes from a different pool with different approvals. Negotiate them in separate conversations. Conceding on cash to "buy" equity is fine; conceding on equity to buy cash is usually a mistake, because the equity is the only part of the package with asymmetric upside and it is the part that geo-adjustment has the weakest justification for touching — a share of the company is worth the same regardless of where the holder sleeps.

**🗣 Say this in the room:** "I want to be straightforward about geography rather than dance around it. I understand you'll geo-adjust cash and I'm not going to argue for the full US band. What I will argue for is that the equity grant shouldn't be geo-adjusted — the strike price and the share count don't depend on where I live, and if I'm doing the same scope as a US hire, the ownership should reflect the scope."

### Am I even eligible for equity as a remote or EOR hire? What actually happens to the options?

Short answer: **usually yes for RSUs and non-qualified options, usually no for the US-tax-advantaged instruments, and always subject to a document you must read rather than assume.**

The mechanism. US companies grant equity under a **stock plan** whose eligibility language defines who can receive grants — typically "employees, directors and consultants of the Company and its subsidiaries." Two consequences follow directly. First, **incentive stock options (ISOs) are restricted by US tax law to employees**, so a contractor or an EOR-employed worker generally cannot receive ISOs and would receive **non-qualified stock options (NSOs)** or RSUs instead. For a non-US-resident this is mostly a non-issue — the ISO tax advantage is a US tax advantage you would not have benefited from anyway. Second, and this is the one that bites: **if your legal employer is an EOR and not the company or a subsidiary, you may fall outside the plan's eligibility definition entirely**, and the company has to grant to you as a "consultant" or amend the plan. Many do it routinely. Some discover mid-onboarding that they cannot, and the grant quietly disappears from the package.

**The rule I enforce: get the grant in the offer letter with share count, strike price (or grant value), vesting schedule, and the plan name — and then ask for the plan document and the form of award agreement before you sign.** You read a database migration before merging it; read this. The four things you are looking for in the award agreement:

1. **Eligibility** — does your engagement type qualify, in writing?
2. **Post-termination exercise window** — the standard is 90 days after termination, which for a non-US holder with a large exercise cost is often equivalent to forfeiture. A 7-to-10-year post-termination exercise window is a real and negotiable term and is worth more than a chunk of cash.
3. **Cliff and acceleration** — standard 1-year cliff, 4-year vest, and what happens on change of control (single- vs double-trigger).
4. **Transfer and settlement mechanics for a non-US holder** — can shares actually be issued to and held by someone in India, and through what broker? This is a real operational question and "we'll figure it out at liquidity" is a bad answer.

**⚠ Trap: assuming an offer letter's "equity: $150,000" line means anything by itself.** Without the share count and the current preferred/409A price you cannot compute the ownership percentage, and "$150k of equity" at a company that just raised at a stretched valuation may be worth a fraction of "$90k of equity" at an earlier-stage one. Ask for: total shares outstanding (fully diluted), your share count, the strike price, and the most recent preferred price. A company that refuses all four is telling you something.

**💰 The exercise-cost trap, with numbers.** Suppose you are granted 40,000 NSOs at a $2.00 strike and, four years later, the 409A is $12.00. Exercising all of them costs $80,000 cash, and in the US the $10.00/share spread would be ordinary income on exercise — for a non-US resident the treatment depends on the treaty and your residency, but the cash requirement does not go away. If your post-termination window is 90 days, leaving that job means finding $80,000 in ninety days or forfeiting the whole grant. **This is why the exercise window is the term I negotiate hardest on, and it costs the company nothing to give.**

### You're an Indian tax resident holding equity in a US company. What's the tax mechanism you need to understand — not as advice, as mechanism?

Stated up front: **this is mechanism, not tax advice, and the specifics change annually. Retain a chartered accountant who has actually handled foreign ESOPs before you do anything irreversible.** With that said, an engineer who cannot describe the mechanism will make an avoidable and expensive mistake, so here is the shape.

**The two taxable events.** For options, Indian tax treats the **exercise** as a perquisite — the difference between the fair market value at exercise and the exercise price is added to your salary income and taxed at your slab rate, with the employer typically obliged to withhold. Then the **sale** is a separate capital-gains event on the difference between the sale price and the FMV that was already taxed as a perquisite. For RSUs the analogous perquisite event happens at **vesting**, since there is no exercise price. The critical implication: **you can owe tax at exercise or vest on paper gains for shares you cannot sell**, which is a genuine cash-flow hazard with private-company equity and the reason "exercise early, exercise often" advice imported from US blogs can be actively dangerous for an Indian resident.

**Holding-period and rate treatment.** Foreign (unlisted-abroad) shares have their own holding-period threshold for long-term versus short-term capital gains treatment, distinct from the threshold for Indian listed shares. **📅 Volatile:** the thresholds and the rates were changed in recent Finance Acts; verify the current year's rules.

**Three compliance obligations people miss and get penalized for:**

1. **Schedule FA.** Indian residents must disclose foreign assets — including foreign shares, vested and unvested-where-applicable, and foreign bank/brokerage accounts — in the Foreign Assets schedule of the return. Non-disclosure exposes you to penalties under the black-money legislation that are grossly disproportionate to the asset value. This is the single most commonly missed obligation among Indian engineers with US equity, and it is not a rounding error.
2. **Foreign tax credit and Form 67.** If US tax is withheld, relief under the India-US double-taxation treaty is claimed through the foreign tax credit mechanism, which requires a timely filed form. Miss the filing and you can lose the credit and be taxed twice.
3. **The LRS ceiling on outbound remittance.** The Liberalised Remittance Scheme caps how much an Indian resident individual may remit abroad per financial year — a limit that becomes relevant if you need to *send dollars out* to exercise options. **📅 Volatile:** the ceiling and the TCS treatment on remittances have both been revised; verify the current figures.

**⚠ Trap: treating an EOR payslip as if the equity is handled.** The EOR administers your salary; it usually does *not* administer the US company's equity plan, and the perquisite-withholding chain that works cleanly for a subsidiary employee often has no owner in an EOR arrangement. Ask explicitly, in writing, before signing: **who computes and withholds the perquisite tax on my equity events, and who issues the documentation I need for my return?** If nobody can name an owner, you are the owner, and you should budget for it.

**💰 Why this is worth an afternoon.** On a $135,000 cash package plus a grant that produces a $60,000 perquisite event in a single year — roughly ₹1.7 crore of income at ₹86/USD — the marginal Indian rate is the 30% top slab grossed up by the applicable surcharge and the 4% cess, which at that income lands around **36%** (0.30 × 1.15 × 1.04), rising to about 39% once total income crosses ₹2 crore. **📅 Volatile:** slabs, surcharge tiers and cess are re-set in each Finance Act and differ between the old and new regimes — recompute against the current year's rules rather than carrying this number. Getting the timing, the credit and the disclosure right is worth five figures a year; getting the Schedule FA disclosure wrong is worth a great deal more than that in penalty exposure.

### How do you benchmark compensation and turn a US band into a defensible remote ask?

The mental model: **an ask is defensible when it is derived rather than asserted.** Recruiters have heard every number; they have not heard many *derivations*, and a derivation is very hard to argue with because arguing requires attacking a source rather than attacking you.

**The four sources, ranked by how much weight they carry in an argument:**

1. **DOL LCA disclosure data** — the strongest, because it is the company's own filed wage for a specific title in a specific location, submitted under penalty of perjury. It is base salary only, it lags by a quarter or two, and it is a floor rather than a midpoint (companies file at or above the prevailing wage, and top-of-band hires often exceed it). But when you say "your filed base for Senior Software Engineer in San Francisco is $X," nobody argues.
2. **levels.fyi** — self-reported, so upward-biased at the top end, but it is the shared vocabulary. Its real value is **leveling**: mapping your scope onto the company's actual ladder so that you are negotiating within the right band rather than fighting the wrong one. Getting leveled one step higher is worth more than any negotiation tactic.
3. **Blind and community aggregates** — noisy, occasionally invaluable for a specific company's current behavior (are they matching? are they doing off-cycle refreshes?), and useless as a citation.
4. **The recruiter themself** — "what's the band for this level?" is a fair, normal question and many recruiters answer it, especially where pay-transparency laws require a posted range. Ranges posted under those laws are real data.

For India-side benchmarking, levels.fyi has usable India data for large companies, and beyond that you are largely relying on network. **📅 Volatile:** everything above is a point-in-time reading; re-pull within the month.

**The derivation template I would use, out loud:**

"For this role at your level, your filed base in [location] is $205k, levels.fyi puts total comp for that level around $340k, and your posted range for the equivalent US req is $190–240k base. I'm not asking for that. I'm asking for roughly 50% of that total-comp figure, taken in cash — $165k — on the reasoning that I'm delivering the same scope on your hours, from a location that saves you the employer-side and relocation cost of a US hire, and that this number is still well above what I'd be paid locally, which is what makes the arrangement stable for both of us."

That last clause matters more than people realize. **A remote arrangement priced far above the local band is a retention machine, and saying so out loud converts your ask from a cost into a risk mitigation.**

**⚠ Trap: benchmarking against the wrong level.** Backend engineers moving into AI roles routinely accept a level downgrade because they feel like beginners in the domain, and one level at these companies is commonly $80k–150k a year in total comp (**📅 Volatile:** level deltas move with the equity market and with AI-specific band inflation — re-derive from current levels.fyi data). Your leveling is set by scope of ownership, ambiguity handled and blast radius — all of which you have from backend work — not by how many months you have been writing eval harnesses. Fight the leveling conversation, not the number.

### The recruiter asks for your compensation expectations on the first call. You're in India. What do you actually say?

This is a scripted moment and you should have the script memorized, because the improvised version of this answer costs real money — anchoring at the India band on call one caps the whole negotiation, and refusing to answer at all reads as evasive from an international candidate whose geography the recruiter is already uncertain about.

**The mechanism you are managing:** the recruiter's job on call one is to disqualify you cheaply, and one of their two disqualifiers is comp mismatch (the other is work authorization). They need enough of a signal to proceed. Giving them nothing forces them to guess, and the guess is your local market. So you give them a number, but you give them the number attached to a frame.

**🗣 Say this in the room, verbatim:**

"Happy to be concrete so we don't waste rounds. I'm calibrating against the US band for this level rather than the Indian market, because the scope and the hours are the US role's — for context, your filed base for this title is around $X. I'm not asking for the full band; for a remote arrangement I'm targeting $160–180k cash — call it half of total comp — with a standard equity grant for the level. If your budget for a remote hire is structurally below that, I'd rather you tell me now."

Four things that script does. It **answers the question**, so you are not evasive. It **anchors on the US band** and names a source, so the anchor is not arbitrary. It **pre-concedes the geo-discount**, which makes you sound reasonable and removes their best argument. And it **invites early disqualification**, which sounds risky but is the strongest possible signal of confidence and saves you four rounds when the budget genuinely is not there.

**If you have no US band data for that specific company**, substitute: "For a senior AI engineer at a company at your stage, I'm calibrating to a $320–380k US total-comp band and targeting roughly half of that total, taken in cash, for a remote arrangement." Naming the band you are calibrating to is what makes the number defensible.

**⚠ Trap: answering with your current compensation.** In several US jurisdictions the recruiter is legally barred from asking for salary history, and even where they are not, an Indian current-comp number in the conversation is an anchor you will never escape — a ₹45 LPA current package translates to about $52k and every subsequent number will be measured against it. If asked directly for current comp: "I'd rather anchor on the market for the role than on my current package — the geographies aren't comparable. Here's what I'm targeting." Nobody pushes past that.

**⚠ Trap: the "what would make you say yes today" close.** Never answer that with a number on call one. "I'd want to see the full scope and meet the team before I could answer that responsibly" is the correct response and costs you nothing.

### How do you raise time zones in a recruiter screen without disqualifying yourself?

The mental model: **the interviewer's fear is not that you are in IST, it is that you will be unavailable when something breaks or when a decision needs three people in a room. Answer the fear, not the fact.** Candidates who lead with "I'm flexible on hours!" sound like they have not thought about it. Candidates who lead with a *specific committed overlap window and a written-first working style* sound like they have already run a distributed team, which as a backend engineer you effectively have.

**The mechanism to state:** IST is UTC+05:30. Against San Francisco that is **13.5 hours ahead of PST and 12.5 hours ahead of PDT**; against New York, **10.5 hours ahead of EST and 9.5 ahead of EDT**; against London, **5.5 hours ahead of GMT and 4.5 hours ahead of BST**. So a 9:00 a.m. Pacific standup is **9:30 p.m. IST** during US daylight time and **10:30 p.m. IST** in winter — and note that the offset *changes twice a year* because India does not observe daylight saving, which is a scheduling detail that breaks recurring meetings and that you should mention because almost no candidate does.

**🗣 Say this in the room:**

"On time zones — I'm UTC+5:30, so I'd commit to a hard overlap of 9:00 p.m. to 1:00 a.m. IST, which is 8:30 a.m. to 12:30 p.m. Pacific during daylight time. That covers your standup, any design review, and the first half of your day for anything urgent. Outside that window I work asynchronously and write things down — design docs, decision records, PR descriptions that someone can act on without me. The one thing I'd ask for is that decisions that need me get made inside that window rather than at 4 p.m. your time, and I'd flag that India doesn't do daylight saving, so the offset shifts by an hour twice a year and recurring invites need to account for it."

**Why that works:** it converts an objection into a demonstration of exactly the working style distributed teams say they want, and it names a concrete constraint you are asking them to respect — which reads as someone who has done this before, not someone who is grateful to be considered.

**⚠ Trap: promising full US-hours availability.** You will say it in the screen, do it for four months, and then quietly degrade — and the degradation is read as an attitude problem rather than as the predictable consequence of a sleep schedule nobody can hold. Promise four hours you can sustain for two years, not eight you can sustain for one quarter. I have seen more remote arrangements die from over-promising overlap than from having too little.

**⚠ Trap: raising it too late.** If you get to the onsite before geography comes up, you have wasted their time and yours, and the late reveal reads as concealment. Raise it in the first five minutes of call one, as a logistics item you have already solved, not as a confession.

### Design your actual working week against a San Francisco team. Give me the hours.

The mental model: **treat your calendar like a scheduler with two classes of work and a hard deadline, because that is what it is.** Synchronous work is latency-sensitive and must land inside a window you do not control. Deep work is throughput-oriented and should be scheduled where it is cheapest — which for you is a time of day that no US team can ever take from you. The mistake is running an undifferentiated day and letting the overlap window fragment it.

**The schedule I would actually run:**

- **09:30–13:30 IST — deep work block one.** This is 9:00 p.m.–1:00 a.m. Pacific the previous day; nobody is awake, nothing will interrupt you. This is where the hard implementation goes. It is the single most valuable block in a distributed engineer's week and it is a direct consequence of the time zone, not despite it.
- **13:30–15:00 IST — break, meal, exercise, sunlight.** Non-negotiable, because the evening block is what makes the whole thing sustainable and you cannot do a 9 p.m.–1 a.m. block on a depleted body.
- **15:00–18:00 IST — deep work block two / async communication.** Write the PR descriptions, the design docs, the decision records, the review comments. Everything you produce here lands in the US team's morning inbox, which is exactly the handoff property you want.
- **18:00–21:00 IST — off.** Dinner, family, a life. Protect this like a production SLO; it is the thing that fails first.
- **21:00–01:00 IST — the overlap block.** 8:30 a.m.–12:30 p.m. Pacific during PDT. Standup, design reviews, pairing, one-on-ones, incident response. Meetings only — do not schedule implementation here, because you will be interrupted and because your cognition is worse at midnight than at 10 a.m.

**💰 The arithmetic that makes this a feature, not a sacrifice.** That is 7 hours of protected deep work per day against a US-based engineer's realistically 3–4 hours between meetings — you get roughly **double the uninterrupted implementation time**, and your output lands overnight so the team wakes up to finished work. Over a quarter that is a visible throughput difference, and it is the single strongest argument available for renewing and expanding a remote arrangement. Say this explicitly in your first performance conversation; do not assume anyone notices.

**🔍 Failure taxonomy for the distributed week, as a decision procedure:**
- **Symptom: the overlap block creeps later.** Cause: US colleagues scheduling at their 1–3 p.m. Fix: publish your working hours in your calendar as a hard boundary and decline outside them once, early. If you decline once in month one it holds; if you accept for three months it never holds again.
- **Symptom: you are always the one being caught up.** Cause: decisions made in your night, in synchronous channels, without a written record. Fix: demand written decision records as a team norm — frame it as a benefit to *them* (onboarding, audit, the person on PTO), never as an accommodation for you.
- **Symptom: sleep debt accumulating, quality dropping in week 3 of every month.** Cause: a shifted schedule that never stabilizes because the weekend snaps you back. Fix: keep the *same* schedule on weekends for the first two months to let circadian entrainment happen, and treat any night past 1:00 a.m. IST as an incident that needs a root cause, not as commitment.
- **Symptom: invisible to leadership; skipped for promotion.** Cause: you are asleep during the hours when informal visibility happens. Fix: over-invest in written artifacts with your name on them — design docs, post-mortems, benchmark reports — because writing is the only visibility channel that works across a 12.5-hour offset.

### Which India-based employers actually do frontier-adjacent AI work, and how do you tell a real one from a wrapper shop?

The mental model: **evaluate an Indian AI job the way you would evaluate a service — by what it owns in production, not by what it is called.** The title "AI Engineer" in the Indian market spans everything from "owns the retrieval and serving stack for a product with millions of users" to "writes prompts for a client demo," and the résumé consequence of the second one is severe: two years of it makes you *less* hireable at your target companies, because it reads as prompt-tinkering rather than systems ownership.

**The categories that contain real work:**

1. **Global capability centres of US big tech with genuine research or AI-platform charters.** Google and Microsoft both operate long-standing research organizations in India with real publication records; Nvidia has substantial Indian engineering; Adobe, Amazon, Databricks, Stripe and others run Indian engineering sites that own production surfaces. **📅 Volatile:** which specific teams sit in India changes constantly — verify by asking what the site owns, not by trusting the brand.
2. **Indian foundation-model and AI-product companies.** A small number of Indian companies are genuinely training or heavily post-training models and building serving infrastructure rather than reselling API calls. **📅 Volatile:** this set turns over rapidly with funding cycles — evaluate the specific company at the time you apply.
3. **Indian engineering sites of US AI-native companies.** Several US AI companies have announced or opened India offices recently. The announcements outrun the engineering headcount by a year or more; check whether they are actually hiring engineers or only go-to-market.
4. **Deep-tech and quant firms.** Trading firms and a handful of infrastructure companies do genuinely hard ML systems work in India and pay at or above GCC bands.

**The five diagnostic questions I would ask in the interview to separate real from wrapper:**

1. **"What does this team own end to end in production — name the service."** A real answer names a system, a traffic level and an SLO. A wrapper answer names a client.
2. **"Do you have an eval harness, and who owns it?"** This is the single sharpest question in the set. Organizations doing real LLM work have an eval suite with a name, an owner and a CI integration. Organizations doing demo work have a spreadsheet, or nothing, and will say "we test manually" or "the client validates it."
3. **"Do you serve any model yourselves, or is everything a provider API?"** Neither answer is disqualifying — plenty of excellent applied work is pure API — but the answer tells you whether you will get serving-stack depth, and serving depth is the scarcest thing on an AI Engineer résumé.
4. **"What's your cost per resolved task, and has anyone driven it down?"** If nobody can answer, nobody is operating the system; they are shipping it.
5. **"How many engineers here have moved to a US or European site in the last two years?"** The answer is a direct measurement of whether the site is a real part of the company or a cost centre.

**⚠ Trap: taking a "GenAI Lead" title at a services company because the title is bigger.** Your target employers read titles from Indian services firms with heavy discounting and will interrogate scope instead. A senior IC role owning a real production system at a company with an eval culture is worth vastly more in your next loop than a lead title over a team of six building client demos — and the gap widens every year you stay.

**🗣 Say this in the room, when asked why you'd take an India-based role given your US ambitions:** "Because the fastest route to a US AI role is a track record of owning AI systems in production, and this site owns one. I'd rather spend two years shipping a real serving and eval stack here than two years applying from outside with nothing to point at."
### You have two offers at similar cash: an India GCC role in a big-tech AI org, and a remote EOR role at a US AI startup. Which do you take?

The mental model: **cash is the least informative variable in this comparison. What you are actually choosing between is a visa-eligible corporate relationship plus institutional stability, versus proximity to frontier work plus the résumé line that gets you the next loop.** Name the axes and the decision stops being a coin flip.

**Five axes, weighted:**

**1. Visa path (weight: highest).** GCC employment is subsidiary employment, so it starts the L-1 clock and creates a real internal-transfer surface. EOR employment does not create any corporate relationship with the US company at all, so it starts no clock and creates no transfer path. If a US relocation is your objective, this axis alone is close to decisive for the GCC.

**2. Work quality and the résumé line.** This is where the startup usually wins — but *usually* is not always, and you verify it with the diagnostic questions rather than assuming. A GCC team that owns a production serving stack with an eval harness in CI beats a startup where you are the fourth person writing prompts. Run the same five questions on both: what does the team own end to end, who owns the eval harness, do you serve any model yourself, what is cost per resolved task, and how many people have transferred out.

**3. Stability and termination risk.** EOR contracts terminate faster and more cheaply than employment, and remote-international headcount is structurally the first thing cut in a downturn because it is the least visible and the least protected. A GCC role at a large company is not immune, but the asymmetry is real. Price it: what is your runway if the arrangement ends in month nine?

**4. Equity quality.** Big-tech RSUs are liquid and priced; startup options are illiquid, may require a large exercise cost, and may be structurally awkward to grant to an EOR worker at all. A "$150k equity" line at the startup and a "$150k equity" line at big tech are not the same instrument and should not be compared at face value.

**5. Optionality on the next move.** Two years at a recognized AI-product company reads differently in a screen than two years at a GCC — but two years at a GCC *with an L-1 transfer* reads better than either, because it ends with you in the US.

**The decision rule I would apply.** If the GCC team genuinely owns AI systems in production, take the GCC — the visa path and the equity quality dominate, and the work-quality gap is smaller than the brand gap suggests. If the GCC team is a support organization for a US team, take the startup, because a support role compounds negatively and no visa path is worth two years of unimpressive scope. **The tiebreaker is the eval-harness question**, because an organization with an owned eval harness is an organization doing real work, in either shape.

**🗣 Say this in the room, to either employer:** "I'm evaluating this against one other offer and I'll be direct about the axis I'm weighing — whether the team owns production AI systems end to end, and whether there's a medium-term path to a US or European site. Both matter more to me than the cash delta, which is small."

### How do you sequence multiple offers for visa optionality rather than just for money?

The mental model: **an offer is not only a compensation package, it is a bundle of legal options — the right to be petitioned for, the right to accumulate qualifying employment, the right to be sponsored later — and options have different expiry dates. Sequence them so the slow options start first.**

**The three clocks you are starting, and their lengths:**
- **The L-1 clock** needs at least one continuous year of qualifying employment with a related entity within the preceding three years, and realistically 24–36 months before a transfer is business-justified. Longest clock; start it earliest.
- **The O-1/EB-1A evidence clock** needs roughly 12–18 months of deliberate portfolio building — merged upstream contributions, a preprint, a judging role, expert letters. It runs *in parallel* with employment and is not blocked by it.
- **The green-card priority-date clock** starts the day a qualifying petition is filed, and an early date is portable to a later petition in some circumstances. This is why filing an EB-2 NIW early — even if EB-1A is the real target — is a cheap option on a date.

**The sequencing rules that follow:**

1. **Take the offer that starts the longest clock, unless it damages the résumé.** A GCC role at a company with real transfer volume is worth taking over a marginally better-paid remote role, because the clock is the scarce resource.
2. **Never let a visa route depend on a single employer's goodwill.** Self-petitionable routes — EB-1A, EB-2 NIW, UK Global Talent, Canadian Express Entry — should always be running in the background, because they survive a layoff and an employer-tied route does not.
3. **Negotiate immigration support into the offer letter, in writing, at offer stage.** Specifically: named willingness to petition for O-1; commitment to file a green-card petition and the *category* they will file; the month tenure at which they will start; and who pays legal fees for you and for dependents. Every one of these is standard and negotiable at offer stage and impossible to obtain at month eighteen.
4. **Cluster your loops.** Running interviews across five companies simultaneously so the offers land within a two-to-three-week window is what creates leverage on all of the above. A single offer negotiated alone gets you a small cash bump; three concurrent offers get you the immigration clauses.
5. **Do not resign until the visa-relevant terms are in the signed document.** A verbal "of course we'll do O-1" from a recruiter is not a commitment, and the person who said it will have left before it matters.

**⚠ Trap: optimizing the first offer for cash and the second for visa.** It runs backwards. Cash compounds over decades but is recoverable — you can always get paid more later. A visa clock you did not start is a year you cannot get back, and for an Indian national with a decade-scale green-card queue, a year at the front of the process is worth far more than a year's salary delta at the back of it.

**💰 Concrete comparison.** Offer A: remote EOR, $145k, no visa path. Offer B: GCC, ₹95 LPA ≈ $110k, subsidiary employment, transfer volume verified at roughly a dozen engineers in two years. The cash gap is $35k/yr, so two years of B costs you $70k. If B produces an L-1 transfer at month 30 into a $320k US band, the crossover happens inside the first year in the US and the lifetime difference is not close. **📅 Volatile: rates and bands move; re-run with your own numbers.** The point is the shape of the calculation, and the shape says: pay for the clock.

### Every round of this loop lands between 11 p.m. and 1 a.m. your time. How do you not lose on energy?

The mental model: **you are being evaluated on peak cognitive performance at your circadian trough, and no amount of preparation compensates for a bad protocol. Treat the schedule as an engineering problem with a two-week lead time, not as something you tough out.** I have watched strong candidates fail a system-design round at 12:30 a.m. purely on working-memory degradation — they knew the material and could not hold the design in their head.

**The mechanism.** Human cognitive performance follows a circadian rhythm with a trough in the small hours; working memory, reaction time and — critically for interviews — *verbal fluency and error monitoring* all degrade. Acute sleep restriction compounds it. The two levers you control are **phase** (shifting when your body thinks it is daytime) and **acute alertness** (light, caffeine timing, food, movement).

**The protocol I would run, starting fourteen days before the loop:**

- **Shift the phase deliberately.** Move your sleep window later by 30–60 minutes per day until you are sleeping roughly 2:00 a.m.–10:00 a.m. IST. Hold it on weekends. A phase shift that is re-broken every Saturday never completes.
- **Light is the control knob.** Bright light in your subjective morning (which is now late morning) anchors the phase; dim, warm light from ~7 p.m. onward, then bright light again at 9 p.m. before the block. A bright desk lamp in your face for 20 minutes before a midnight round is worth more than a coffee.
- **Caffeine with a timer, not a mood.** Roughly 45–60 minutes before the round, one dose, and none in the six hours before your intended sleep time — you need to sleep after the interview or the next day's round degrades.
- **Eat before, not during.** A heavy meal inside 90 minutes of the round produces exactly the postprandial dip you cannot afford. Light protein two hours before.
- **Ten minutes of movement immediately before.** Not exercise — a walk, stairs, anything that raises core temperature and heart rate. This is the single highest-return item on the list and almost nobody does it.
- **Warm up the specific skill.** Fifteen minutes before a coding round, write a small piece of code. Before a design round, talk out loud through a design you already know. Verbal fluency at midnight needs a runway; do not let the first sentence you speak that day be your answer to question one.

**🏋 Drill.** Two weeks before your real loop, book a mock interview with a friend at **midnight IST**, on video, with a whiteboard, for 60 minutes. Pass criterion: you can implement scaled dot-product attention from memory and narrate what you are doing at the same time. If narration collapses while your hands keep typing, your protocol is not working yet — that specific dissociation is the reliable early sign of trough-state degradation, and it is exactly what an interviewer scores you down for.

**⚠ Trap: treating the last round as the one that needs the protocol.** The onsite is usually four to five back-to-back rounds. If it is scheduled as a single day in US hours, you are looking at roughly **9:30 p.m. to 4:00 a.m. IST**, and rounds four and five are where the decision is made. Protocol for the whole block or do not bother.

### What can you actually ask for on interview scheduling, and what will get you dinged?

The mental model: **recruiters schedule hundreds of loops and they are optimizing for cycle time. Requests that reduce their work or are trivially accommodated are free; requests that add coordination cost are expensive and are remembered.** Sort your asks by which they are before you make them.

**Free asks — make these every time:**
- **A specific window rather than "any time."** "I'm available 8:00 a.m.–1:00 p.m. Pacific any weekday" is easier for them than open availability, because it removes a decision.
- **Splitting an onsite across two days.** This is very commonly accommodated and it is the single highest-value scheduling ask available to you, because it converts one 6-hour block ending at 4 a.m. into two 3-hour blocks ending at 1 a.m. Ask for it explicitly and give the reason: "I'm in IST; splitting across two days means you get me at my best in every round rather than degraded in the last two."
- **The earliest possible slot in the US day.** 8:00 a.m. Pacific is 8:30 p.m. IST in US summer and 9:30 p.m. in winter — meaningfully better than an 11 a.m. Pacific slot that lands at 11:30 p.m. or 12:30 a.m.
- **Knowing the format in advance.** Round type, duration, language, whether there is a shared editor, whether it is a take-home. This is normal and any decent recruiter provides it.
- **A week or two of lead time before an onsite.** Standard, and it lets you run the circadian protocol.

**Expensive asks — make at most one, and only if it matters:**
- Rescheduling more than once.
- Asking for interviewers in a different geography than the team.
- Asking to move an onsite by more than two weeks (this reads as low interest, and pipelines close).

**Asks that will get you dinged:**
- Requesting a different interviewer without a concrete reason.
- Asking to change the format ("can I do a take-home instead of the live coding round?") — this reads as avoiding the evaluation.
- Any implication that the time zone makes you less able to perform. Never apologize for geography; state the constraint and the accommodation in the same sentence.

**🗣 Say this in the room, to the recruiter:** "One scheduling request, and I'll be easy on everything else: I'm in IST, so a five-round onsite as a single block would end around 4 a.m. my time. If you can split it across two days I'll take any slots you have, including consecutive days. If it has to be one block, I'll do it — I'd just rather give you my best in round five."

That framing works because it makes the request about *their* signal quality rather than about your comfort, and because the pre-commitment to accept a no removes any adversarial edge.

**⚠ Trap: going silent for 12 hours because you were asleep.** Recruiter emails arrive during your night, and a 14-hour reply latency reads as low interest even when it is physics. Fix it structurally: reply to everything in your 9:00 p.m. block, and put a line in your first email — "I'm in IST, so I'll typically reply to you in your morning" — which converts a perceived attitude into an announced schedule.

### What's the recording, proctoring and AI-tool policy landscape, and how do you avoid being disqualified by it?

The mental model: **assume you are being observed by more instrumentation than you can see, and that the policy differs per round within the same loop.** The failure mode here is not cheating; it is an honest candidate violating a policy they did not read, and the penalty is usually silent — you do not get told, you get rejected.

**The four things that vary and that you must confirm before each round:**

1. **AI-tool use during the round.** Several labs and quant firms explicitly ban AI assistance in live coding rounds, sometimes with a proctored environment; other companies deliberately *require* AI-assisted workflow because using AI tools well is the thing being assessed. These are opposite policies at companies you are interviewing with in the same month. **Confirm in writing, per round, and never assume the policy carries over from the previous round or the previous company.**
2. **Proctoring on screening assessments.** Automated platforms may record screen, webcam and audio, may flag tab-switching and copy-paste, and may run plagiarism detection against submitted code. Behaviour that is innocent — looking away to think, having a second monitor, pasting your own boilerplate — can be flagged. Clear your desk, close everything, use one screen, type rather than paste, and if you must reference documentation, ask first whether it is permitted.
3. **Your recording of the interview.** Assume it is prohibited unless explicitly permitted, and understand that in some jurisdictions recording without all-party consent is unlawful independent of company policy. Do not record. Take notes by hand instead — visible note-taking is fine and reads well.
4. **Their recording of the interview.** Increasingly common, often with an AI note-taker in the call. You may be asked to consent; you may decline, though declining is a mild negative signal. The useful question to ask is what happens to the recording and how long it is retained — asking it once, calmly, reads as professionalism.

**⚠ Trap: the take-home you completed with an AI assistant, followed by a live walkthrough.** This is now a standard and deliberate pattern: the take-home is not the test, the walkthrough is. If you cannot explain and modify every line under questioning, the assessment concludes you did not write it — and that conclusion is correct even when you did, because you did not understand it. The rule: **use whatever tools are permitted to write it, then re-derive every non-trivial decision unaided before the walkthrough, and be ready to say what you would change.**

**🗣 Say this in the room, when asked whether you used AI on the take-home:** "Yes, for the boilerplate and the test scaffolding, and I'll tell you exactly where — the retrieval loop and the eval harness I wrote myself because I needed to reason about the failure modes. Happy to walk through any of it and change it live."

Honesty here is strictly dominant. Every interviewer at these companies uses AI tools daily; nobody believes you did not, and claiming you did not is the answer that actually loses.

### Your Indian employment contract has a broad IP-assignment clause and you've published a large public guide and open-source work. What's the risk, and what do you do about it?

The mental model: **IP-assignment clauses in Indian employment contracts are frequently drafted far more broadly than the employer intends to enforce, and the risk is not usually litigation — it is a diligence question at your next employer that you cannot answer cleanly.** A US company's counsel will ask whether your public artifacts are encumbered, and "I think it's fine" is not an answer that clears legal review.

**The mechanism to understand:**

- Under the **Copyright Act, 1957**, works created by an employee in the course of employment generally vest in the employer absent an agreement to the contrary — so the default already leans against you before the contract says anything. The contested question is what "in the course of employment" covers.
- Employment contracts commonly extend that with clauses assigning inventions and works "created during the term of employment," sometimes without limiting to work-related subject matter or to company time and resources. Some Indian contracts also carry **moonlighting** restrictions — this became a live issue publicly when several large Indian IT employers took disciplinary positions against dual employment.
- **Section 27 of the Indian Contract Act, 1872** voids agreements in restraint of trade, which is why post-employment non-competes are generally unenforceable in India — but that section does not help you on IP assignment, which is a property transfer, not a restraint.

**The five things I would actually do, in order:**

1. **Read your contract and find the exact clause.** Note whether it is limited by subject matter ("relating to the company's business"), by resources ("using company equipment or confidential information"), or by nothing at all. The unlimited version is the dangerous one.
2. **Create a clean provenance record.** Personal machine, personal accounts, personal time, no company confidential information, no company-specific content. Commit history with timestamps is genuinely useful evidence here — your GitHub log is a contemporaneous record of when and where work was done. This is one of the rare places where commit hygiene has a legal payoff.
3. **Never let company material leak into the artifact.** A guide that contains a question you were asked in an internal interview, or an architecture diagram that resembles your employer's, converts an ambiguous IP question into a confidentiality one, which is much worse.
4. **Ask for written acknowledgement if the relationship permits it.** A one-line email from your manager or HR acknowledging that a named personal project is outside the scope of your employment IP assignment is worth enormous amounts later and costs nothing to request. Not every employer will give it; asking is usually low risk.
5. **Disclose it at the next employer, on the prior-inventions schedule.** US offer packages typically include an employee invention-assignment agreement with an exhibit for listing prior inventions you are excluding from assignment. **List the guide, the site and your OSS repositories by name and URL on that exhibit.** Leaving it blank assigns them to your new employer. This is the step people skip because the form looks like boilerplate, and it is the step with the largest consequence.

**⚠ Trap: assuming that because nobody objected while you were employed, the claim is extinguished.** It is not; assignment claims can surface at exit, at acquisition diligence, or when the artifact becomes commercially interesting. The time to clean this up is while you are on good terms, not during a notice period.

**🗣 Say this in the room, if a US employer's counsel raises it:** "The guide and the repositories were built on personal time and personal equipment, contain no material from any employer, and I've listed them on the prior-inventions exhibit. My commit history is public and timestamped if you need provenance."

### Non-competes, non-solicits, notice periods and relieving letters — what actually bites when you're hired by a US company?

The mental model: **the clause your Indian contract enforces hardest is the one US employers care about least, and vice versa. The real friction is almost never the non-compete — it is the notice period and the paperwork of exit.**

**Non-competes.** Post-employment non-competes are generally unenforceable in India under **Section 27 of the Indian Contract Act, 1872**, which voids agreements in restraint of trade; Indian courts have consistently declined to enforce restrictions operating after employment ends, while enforcing restrictions *during* employment. In the US, California voids employee non-competes by statute, and the FTC's attempted nationwide ban was set aside in litigation, leaving it a state-by-state matter. **📅 Volatile:** US state-level non-compete law has been changing rapidly and the FTC's posture has shifted; verify for the specific state if it ever matters.

**Non-solicits and confidentiality** are a different story — both are generally enforceable in India, and confidentiality obligations survive employment indefinitely. These are the ones that actually constrain behaviour: do not take documents, do not take code, do not recruit your former team in the first year.

**Notice periods — the real problem.** Indian employment contracts, especially at large IT services companies, commonly carry **60- to 90-day notice periods**, sometimes with buyout clauses and sometimes with employer discretion to refuse a buyout. US hiring managers plan around two to four weeks. A 90-day notice period is a genuine offer risk: I have seen offers lapse over it. Handle it by **disclosing the notice period on the first recruiter call**, not at offer stage — it is a known feature of the Indian market, US recruiters have encountered it, and disclosed early it is a planning input rather than a surprise.

**The relieving letter.** This is the artifact that catches US employers off guard and catches Indian candidates at the worst moment. Indian employers issue a relieving letter (and experience/service letter) on a clean exit, and Indian background-check vendors treat its absence as a red flag. Exiting badly — abandoning notice, disputed dues, an unreturned laptop — can mean it is withheld, and that can surface years later in a check for a completely different job.

**The rules I would enforce on yourself:**
1. Serve the notice properly, hand over cleanly, and get the relieving letter and final settlement in hand before your start date.
2. Keep personal copies of offer letters, appraisal letters, payslips and Form 16 for every employer, forever. Reconstructing employment evidence for a company that has restructured or been acquired is genuinely difficult.
3. Negotiate the start date against the *worst case* notice period, then start early if released early. Promising four weeks and delivering twelve is how offers get rescinded.

**🗣 Say this in the room, on call one:** "One logistics item so it's not a surprise later: my current notice period is 90 days, with a possible buyout at my employer's discretion. I'd plan for a start date twelve weeks from signing and try to beat it. I mention it now because I know it's longer than US norms."

### The background check comes back with a discrepancy on your dates or your title. What happened, and how do you prevent it?

The mental model: **background checks do not verify what you claim, they verify what your former employer's HR system says — and those two things disagree far more often than people expect.** The failure is almost always administrative rather than dishonest, and it is entirely preventable by reconciling your own record *before* anyone checks it.

**The four discrepancies that actually occur:**

1. **Dates.** Your last working day, your relieving date, and your final settlement date can all differ, and the HR system may report a different one than you listed. A resume that says "Jan 2022 – Mar 2024" against an HR record of "Jan 2022 – Feb 2024" is flagged as a discrepancy even though nothing dishonest occurred.
2. **Title.** Internal designation frequently differs from the functional title everyone uses. Your business card and your Slack say "Senior Backend Engineer"; the HRIS says "Technology Analyst, Grade 5." The verification comes back mismatched. **Use the HRIS title on formal documents and the functional title in conversation, and if you must use the functional title on a resume, be ready to explain the mapping in one sentence.**
3. **Employment type.** Contract-to-hire, third-party payroll, and secondment arrangements are extremely common in the Indian market and are frequently misrepresented on resumes — not maliciously, but because the candidate thinks of themselves as having worked at the client. If you were on a vendor's payroll at a client site, **the vendor is your employer**, and listing the client as the employer is the single most common cause of a failed check for Indian candidates.
4. **Education.** University verification in India can be slow and occasionally fails on administrative grounds. Get a copy of your degree certificate, consolidated marksheet and any transcript now, before you need them.

**The pre-flight I would run before any offer:**
- Reconcile every date on your resume against your relieving letters and Form 16s. Fix the resume, not the record.
- Confirm each employer's HRIS title, and align the resume or prepare the one-sentence explanation.
- For any third-party-payroll engagement, list the legal employer with the client named in the description: "Vendor Pvt Ltd (deployed at Client)."
- Verify each former employer's current verification channel — for companies that have been acquired, merged or shut down, find out *now* who answers verification requests, because discovering it during a check costs weeks.

**⚠ Trap: rounding a title upward on the resume because it reflects your real scope.** It probably does reflect your real scope, and it will still fail verification, and a failed verification on a title is treated by some employers as a misrepresentation rather than as an approximation. Put the verifiable title in the header and the real scope in the bullets — the bullets are where scope belongs anyway, and nobody verifies bullets.

**🔍 Failure taxonomy for the offer-to-start window, as a decision procedure:**
- **Symptom: check stalls with no result.** Cause: the former employer's verification vendor has not responded, or the company no longer exists. Fix: proactively supply relieving letter, payslips and Form 16 — documentary evidence is an accepted fallback at most vendors. Do this the day the check starts, not the day it stalls.
- **Symptom: discrepancy flagged on dates.** Cause: relieving date versus last working day. Fix: send the relieving letter with the dates highlighted and a one-line explanation. Almost always resolved same-week if you get ahead of it.
- **Symptom: employer disputes the title.** Cause: HRIS designation mismatch. Fix: nothing, after the fact. Prevent it by using the HRIS title.
- **Symptom: offer goes quiet after the check.** Cause: usually not the check — usually headcount. Ask directly and in writing; you are entitled to know, and the answer determines whether you keep your other loops warm.

### What are the constants and dates you should be able to recite about this whole path?

**📐 Numbers you must know — with their derivation, and all of them re-verifiable at a primary source.**

**Time-zone arithmetic, which you will use weekly.** IST is UTC+05:30 and India does not observe daylight saving, so *your* offset is fixed and the other end's moves. US Pacific is UTC−08:00 in winter and UTC−07:00 in summer, giving offsets of **13.5 and 12.5 hours**. US Eastern is UTC−05:00/−04:00, giving **10.5 and 9.5 hours**. UK is UTC+00:00/+01:00, giving **5.5 and 4.5 hours**. Derivation is just 5.5 minus their offset. The consequence you must be able to compute live: **a 9 a.m. Pacific meeting is 9:30 p.m. IST during US daylight time and 10:30 p.m. IST in winter** (9:00 + 12.5, and 9:00 + 13.5) — and the fact that it *moves* is the detail that marks you as someone who has actually done this.

**H-1B supply.** **65,000 regular plus 20,000 US-master's-exempt = 85,000** per fiscal year, against registration volumes that have run at several multiples of that, which is why it is a lottery rather than a queue. Fiscal-year start and thus earliest employment start date: **October 1**. **📅 Volatile:** fees, the selection mechanism and eligibility rules have all been subject to active rulemaking and litigation; verify at uscis.gov.

**O-1A structure.** **Three of eight criteria** plus a final-merits determination. Initial validity **up to three years**, extensions in **one-year** increments. **No cap, no lottery, dual intent, employer- or agent-petitioned** (not self-petitionable). Requires an advisory opinion.

**L-1 structure.** **One continuous year of qualifying employment within the preceding three years** at a related entity. **L-1A: seven-year maximum, and the bridge to EB-1C. L-1B: five-year maximum, no equivalent bridge.** No cap.

**The green-card categories that matter for an Indian national.** **EB-1A**: self-petitionable, EB-1 queue. **EB-2 NIW**: self-petitionable under the three-prong **Matter of Dhanasar (2016)** framework, but still the EB-2 queue. **EB-2/EB-3 with PERM**: employer-tied, longest queue. The one number you should *not* memorize is any specific priority date — read the current **Visa Bulletin**, which is republished monthly and which retrogresses.

**Comp anchors, all 📅 Volatile and all to be re-derived the month you negotiate.** A senior AI-engineer US total-comp band in the low-to-mid hundreds of thousands; an Indian senior band roughly a quarter to a third of it; a remote-from-India cash *ask* typically opened at **45–65% of US total compensation** and commonly landing in the **35–45%** range, with the denominator stated so it cannot drift mid-negotiation. Carry the *ratio* and the *derivation method*, never the absolute figures — the figures in this guide will be stale before you use them, and the LCA-plus-levels.fyi derivation will not be.

**🗣 Say this in the room, when immigration comes up and you want to sound like you have run this before:** "The short version: H-1B is 85,000 a year against multiples of that in demand, so it's a lottery with an October start, and behind it is an EB-2 India queue measured in decades. O-1A is three-of-eight criteria, uncapped, filable year-round, and it feeds EB-1A which is self-petitionable and in a far shorter queue. That's why my plan is O-1 first and H-1B as a fallback rather than the reverse."

### Give me the whole thing as one plan. Where does it break, and what's the drill?

The mental model for the section as a whole: **you are running four independent processes with different failure modes, and the entire point of running them in parallel is that no single rejection ends the path.** A candidate with one process has a hope. A candidate with four has a portfolio, and portfolios are how you survive a lottery, an adjudicator's discretion, a hiring freeze and a policy change — all four of which are outside your control and at least one of which will happen to you.

**The four parallel processes:**
1. **Employment** — apply into modality-appropriate reqs, weighted toward companies with real Indian engineering sites and toward roles that own production AI systems.
2. **Evidence** — the O-1A/EB-1A/Global Talent portfolio, built to a schedule: merged upstream contributions, one preprint or workshop paper, one judging role, expert letters, one conference talk.
3. **Self-obtained status** — UK Global Talent endorsement and a submittable Canadian Express Entry profile, both of which convert you from a sponsorship request into a local hire.
4. **Readiness** — language test, credential assessment, document pack (relieving letters, Form 16s, degree certificates, payslips), and a reconciled resume that will survive a background check.

**🔍 Failure taxonomy for the geography plan, as a decision procedure:**

- **Symptom: consistent silence after applying, no screens.** Diagnose by modality mismatch first, not by resume quality. If you are applying to US-located reqs from India, the resume is not the variable. **Fix:** re-sort the target list by the five-step verification procedure and only apply where the modality exists.
- **Symptom: screens happen, then die immediately after the logistics question.** Cause: geography surfaced as a surprise or was framed apologetically. **Fix:** move it to the first five minutes with the committed-overlap script.
- **Symptom: offers arrive but at the local band.** Cause: you anchored on India, or you never named a US-band derivation. **Fix:** the LCA-plus-levels derivation, delivered on call one, with a pre-conceded geo-discount.
- **Symptom: you're two years in with no visa progress.** Cause: you took an EOR role, which starts no clock, and you did not run the evidence process in parallel. **Fix:** this is the expensive one and the reason process 2 must run regardless of employment — it is the only process that is entirely under your control.
- **Symptom: a policy change invalidates your primary route.** Cause: it will happen; fee changes, selection-mechanism changes and threshold changes have all landed with weeks of notice in recent years. **Fix:** this is precisely why there are four processes and not one. Re-verify every route quarterly and be willing to switch primaries.

**🏋 Drill (25 minutes, unaided, single page).** Write your twelve-month geography plan. Required contents: (a) the modality you are targeting per company, for your top eight companies, derived from their actual postings; (b) your three O-1A criteria with the specific artifact and a dated next action for each; (c) the self-obtained status you are pursuing and its next concrete step with a date; (d) your target cash number with the derivation shown, in one line; (e) your committed overlap window in both IST and Pacific; (f) the single dependency that, if it fails, breaks the most of the plan — and what you would do instead. **Pass criterion:** every item has a date, no item says "research," and item (f) is answered with an alternative rather than with "reassess."

**🏋 Drill (90 minutes, once, this week — the highest-return exercise in this section).** Pull the DOL LCA disclosure data for your top eight target companies. Build a table: filings in the last two fiscal years, titles, locations, wage range. Then open each company's careers page and add two columns: does an Indian engineering site exist, and is any engineering req remote-eligible with a country list that includes India. **Pass criterion:** at the end you can state, for each of eight companies, which of the five modalities applies — and you will find that at least two of your assumptions were wrong. That table is worth more than any advice in this section, because it is current and it is about your actual list.


---

## 5. The Model and Provider Landscape — the Volatility Sink

*Mastering this proves you are shipping in production daily rather than reciting a 2024 model table; it is also the single section you re-verify before every loop.*

### We're not going to test whether you memorized a model table. Tell me how you actually decide which model to put behind a feature.

The mental model I want you to hold: **model selection is a constraint-satisfaction problem with one hard constraint and eight soft ones, and the hard one is the only one you evaluate first.** The hard constraint is the capability floor — does this model clear the quality bar on *my* eval, on *my* data? Everything else (price, latency, context, caching, structured output, tool-calling, rate limits, retention terms) is a soft constraint you optimize *within* the set of models that clear the floor. Candidates fail this question by starting with price, because price is the number that's easiest to look up.

Here is the nine-axis rubric I actually run, in order:

1. **Capability floor.** Does it pass my task eval at the threshold the product needs? Binary gate. Everything below is a tiebreak.
2. **Latency budget.** TTFT and inter-token latency against the UX contract. A 200 ms TTFT product and a 4-second-TTFT product are different model markets.
3. **Price per million in / out**, weighted by *your* in:out ratio, not the vendor's marketing example.
4. **Context limit — advertised vs usable.** Two numbers, always.
5. **Caching semantics.** Prefix-cache discount, TTL, minimum cacheable prefix, what invalidates it. This routinely moves effective price by 5–10×, so it belongs above raw price in practice.
6. **Structured-output support.** Constrained decoding with a real schema guarantee, or "please output JSON" and a retry loop? The difference is a 3% malformed rate versus 0%.
7. **Tool-calling quality.** Not "does it emit valid JSON for a tool call" — does it pick the *right* tool, with the right arguments, and stop when it should.
8. **Rate limits and headroom.** Your requests-per-minute and tokens-per-minute ceiling at your current org tier, and how fast you can escalate.
9. **Data-retention and residency terms.** Often decided by someone who is not you, and it can veto axes 1–8.

**🗣 Say this in the room:** "I don't pick a model, I eliminate models. First I run my own eval to get the set that clears the capability floor — usually two or three. Then it's a constrained optimization over latency, effective price after caching, and context. The answer is almost never 'the best model'; it's 'the cheapest model that clears the floor with margin, plus a documented escalation path to a stronger one.'"

**⚠ Trap:** treating this as a static decision. It is a *routing policy*, re-evaluated on a cadence. The senior signal is saying "and here's the eval job that re-runs this decision every time a provider ships, so I find out from CI rather than from a support ticket."

**📅 Volatile:** every number in this section — lineups, prices, context limits, tier thresholds, engine feature matrices — has a shelf life measured in weeks. Everything is date-stamped. Re-verify before your loop; the arithmetic and the framework are what you're being tested on.

### "Best model wins" became "best fit wins" at some point. What actually changed?

Three things changed at once, and the interesting part is that none of them was a capability breakthrough.

**First, the capability floor for common tasks dropped below the cheap tier.** In 2023, if you wanted reliable JSON extraction from a messy invoice, there was exactly one model that could do it. By 2025 the small tier could do it, and the frontier tier was 5–20× the price for output you couldn't distinguish on your eval. Once a task's difficulty falls below the floor of the cheap tier, spending frontier money on it is pure waste — and *most production tasks are below that floor*. The frontier tier stopped being where you build and became where you escalate.

**Second, price stopped being a single number.** With prefix caching, batch tiers, and reasoning-token billing, "cost per request" became a function of your traffic shape rather than a property of the model. A model with a higher sticker price and a 90% cache discount on a 12k-token system prompt is cheaper for an agent than a model with a lower sticker price and no caching. You cannot rank models by price anymore; you can only rank *deployments*.

**Third, the axes became genuinely orthogonal.** The model with the best coding eval is not the one with the best structured-output guarantee, is not the one with the highest TPM ceiling on your account tier, is not the one your legal team will sign for. In 2023 those correlated because there was one good model. Now they don't.

**💰 Math:** a concrete illustration of why "best" is the wrong word. Take a classification endpoint at 5M requests/month, 800 input tokens and 40 output tokens each. On a frontier-tier model at $5/Mtok in and $25/Mtok out (**📅 Volatile:** Opus-class pricing as of mid-2026): (5e6 × 800 / 1e6 × $5) + (5e6 × 40 / 1e6 × $25) = $20,000 + $5,000 = **$25,000/month**. On a small-tier model at $1/$5: $4,000 + $1,000 = **$5,000/month**. If your eval shows the small model at 96.1% and the frontier at 96.4% on this task, you are paying $20,000/month for 0.3 points that is inside your confidence interval. That is the entire "best fit" argument in one calculation.

**🗣 Say this in the room:** "'Best' assumes a total ordering that stopped existing around the point where the cheap tier cleared most production tasks. What I optimize is cost per *resolved task* subject to a quality floor and a latency SLO — and those three quantities point at different models depending on the workload."

### Walk me through establishing the capability floor. How do you find it without just guessing?

The mental model: the capability floor is not a property of the model, it is a property of **the pair (your task, your quality threshold)** — so it can only be measured, never looked up. Public benchmarks tell you the ordering on *someone else's* task distribution, and the correlation with your task is unknown and usually weaker than you'd like.

Mechanism, as a procedure I'd actually run in a week:

**Step 1 — build the eval set before you touch a model.** 150–300 examples sampled from real traffic (or the closest proxy you have), stratified so the hard cases aren't drowned out by the easy head of the distribution. Label them. This is the expensive part and there is no shortcut; if you skip it, every subsequent decision is vibes.

**Step 2 — define the threshold as a product decision, in writing.** "≥95% exact match on the extracted fields, with ≤0.5% hallucinated fields" is a threshold. "Good quality" is not. The hallucination-rate side matters more than the accuracy side for most products, because a wrong-but-confident answer costs more than a refusal.

**Step 3 — descend, don't ascend.** Start at the strongest model you can afford, confirm the task is solvable *at all*. If the frontier model fails your threshold, the problem is your prompt, your retrieval, or your task decomposition — not your model choice, and swapping models will not save you. Once the frontier model passes, walk *down* the tier ladder until something fails. The cheapest passing model is your floor.

**Step 4 — measure the gap with statistics, not eyeballs.** On a 200-example set, a 96.4% vs 96.1% difference is 0.6 examples. Use a paired bootstrap over the per-example outcomes; if the 95% CI on the difference straddles zero, the models are indistinguishable *on your evidence* and you take the cheaper one. Report the CI in your decision doc.

**Step 5 — re-run on a cadence and on every provider release.** Wire it into CI as a scheduled job, not a manual notebook.

**⚠ Trap:** running the descent with a prompt tuned for the frontier model. Small models are more sensitive to prompt structure — they benefit from more explicit formatting instructions, fewer implicit inferences, and few-shot examples the big model didn't need. If you hand a small model a terse prompt written for a reasoning model and conclude "it can't do this," you've measured your prompt, not the model. Budget an hour of prompt adaptation per candidate before you call the floor.

**🏋 Drill (60 minutes, unaided):** take any task you've shipped. Write the threshold statement in one sentence with two numbers in it. Build a 50-example labeled set. Run three model tiers. Produce a table with accuracy, 95% CI, p50 latency, and cost per 1k requests. Pass criterion: you can state which model you'd ship and defend it against "why not the better one?" using only your table.

### How do you turn a latency SLO into a model constraint? Be specific about the metrics.

The mental model: **a chat product's latency contract is two numbers, not one, and they're bounded by different physics.** Time-to-first-token is dominated by queueing, prefill compute, and network round-trip; inter-token latency is dominated by memory bandwidth at the serving layer and is essentially fixed per model per deployment. Your product SLO must be decomposed onto both, because a model can be great at one and useless at the other.

The vocabulary swap you need to make from backend: p95 request latency becomes **TTFT** (time to first token), **ITL/TPOT** (inter-token latency / time per output token), and **total generation time** = TTFT + ITL × output_tokens. Throughput becomes **tokens/sec** and, more usefully, **goodput** — requests/sec that actually met the SLO, which is the only throughput number worth quoting.

The procedure:

1. **Write the UX contract.** Streaming chat: TTFT under ~500 ms feels instant, under ~1 s is fine, over 2 s users start reloading. Non-streaming API in a request path: total generation time is what matters and there is no hiding behind streaming. Autocomplete in an editor (Cursor-class): TTFT budget is 100–300 ms and the model tier is decided by that alone.
2. **Convert to a token budget.** If your SLO is 3 s total, TTFT measures 600 ms, and ITL is 15 ms/token, you have (3000 − 600)/15 = **160 output tokens**. That is your real max output length, and it is a *design constraint on the prompt*, not something you discover in prod.
3. **Check whether reasoning tokens fit.** A model that emits 2,000 reasoning tokens before its first visible token has a TTFT-equivalent of 2000 × 15 ms = 30 seconds from the user's perspective, even if the API's technical TTFT is 400 ms. For any interactive surface, a reasoning model is a different product, not a drop-in upgrade.
4. **Measure at your p99 input length, not your median.** Prefill is roughly linear in input tokens; a 32k-token context request has a TTFT several times your median. If your context is retrieval-augmented and unbounded, your TTFT distribution has a long tail by construction.

**📐 Numbers you must know:** ITL at 15–30 ms/token is the range a well-served mid-size model lands in, which is 30–65 tokens/sec — comfortably faster than human reading speed (~5 tokens/sec of *reading*, ~250 wpm). This is why streaming saves you: past roughly 10 tokens/sec, further ITL improvement is invisible to a reading user, and your entire remaining latency budget should be spent on TTFT. Derive it yourself: 250 words/min ÷ 60 ≈ 4.2 words/s × ~1.3 tokens/word ≈ 5.5 tokens/s.

**⚠ Trap:** quoting the provider's advertised tokens/sec. That figure is measured at their batch size on their hardware with a short prompt. Your number depends on your prompt length, your concurrency, and whether you got a cache hit. Measure it from your own client, at your own p95 input length, over a week, from the region you deploy in.

### Build me a per-request cost function from a pricing table. I want to see the code.

Mental model: **a provider's price list is not a price, it's a rate card with four or five distinct rates, and which one applies depends on the state of a cache you don't control.** Your cost function must therefore take the *usage object returned by the API*, not your estimate of the request, as its input. The single most common cost-modeling error I see in review is multiplying an estimated token count by a single price.

The rate structure you must model, per provider:

- **Uncached input** — full input rate.
- **Cache write** — a premium over base input (Anthropic charges ~1.25× base for the 5-minute TTL and ~2× for the 1-hour TTL; **📅 Volatile**).
- **Cache read** — a deep discount, roughly 0.1× base input on Anthropic, and providers differ (some are 0.25×, some 0.5×). This is the single highest-leverage number in the whole table.
- **Output** — typically 4–5× the input rate.
- **Reasoning/thinking tokens** — billed as output, and *invisible in your response text*. This is where budgets die.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Rates:
    """USD per 1M tokens. Verify against the provider pricing page before use."""
    input_uncached: float
    input_cache_write: float   # e.g. 1.25 * input_uncached for a 5m TTL
    input_cache_read: float    # e.g. 0.10 * input_uncached
    output: float              # reasoning tokens bill at this rate too

def request_cost_usd(usage, r: Rates) -> float:
    """`usage` is the provider's usage object, not your estimate."""
    uncached = getattr(usage, "input_tokens", 0)
    write    = getattr(usage, "cache_creation_input_tokens", 0)
    read     = getattr(usage, "cache_read_input_tokens", 0)
    out      = getattr(usage, "output_tokens", 0)   # includes thinking tokens
    return (uncached * r.input_uncached
            + write  * r.input_cache_write
            + read   * r.input_cache_read
            + out    * r.output) / 1_000_000
```

Two properties of this function that matter in review. First, **`input_tokens` on Anthropic is the uncached remainder only** — total prompt size is `input_tokens + cache_creation + cache_read`. If you log `input_tokens` as "prompt size" you will report an agent that ran for an hour as having a 4k-token prompt, and your dashboards will be quietly wrong. Second, the function must be applied per response and *summed over the whole agent trajectory*, because a single user-visible task is 10–40 API calls in an agent loop. Cost per API call is a meaningless unit; **cost per resolved task** is the unit your director cares about.

**💰 Math:** a 12k-token system prompt at $3/Mtok input, called 200k times/day. Uncached: 12,000 × $3/1e6 = $0.036/call → $7,200/day → **$216k/month**. With a prefix cache at 0.1× and a ~90% hit rate: hits cost 12,000 × $0.30/1e6 = $0.0036, misses cost 12,000 × $3.75/1e6 = $0.045 (write premium). Blended: 0.9 × $0.0036 + 0.1 × $0.045 = $0.00774/call → $1,548/day → **$46.4k/month**. That is a **$170k/month** delta from one `cache_control` marker, and it is entirely arithmetic you can do at a whiteboard.

**⚠ Trap:** forgetting that reasoning tokens bill at the output rate and are not returned to you as text. A request whose visible answer is 80 tokens can bill 3,000 output tokens. If your cost dashboard is built from `len(response_text)`, you will under-report by nearly 40× on reasoning workloads and discover it on the invoice.

### Context limits — you said "advertised vs usable, always two numbers." Defend that.

The mental model: **the advertised context limit is a memory-allocation guarantee, not a comprehension guarantee.** The provider is telling you the request will not be rejected. It is not telling you the model will use the middle of that window. Those are different claims and only one of them is contractual.

The mechanism behind the gap is well documented. Retrieval accuracy over long contexts degrades with position — the "lost in the middle" effect, where facts placed near the beginning or end of a long prompt are recovered reliably and facts in the middle are not (**📄 Paper:** Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts* — showed a U-shaped accuracy curve over document position in multi-document QA and key-value retrieval, and reframed long context as a quality question rather than a capacity question). Beyond that, degradation compounds when the task requires *multiple* facts, or reasoning over retrieved facts rather than quoting them, or discriminating between similar distractors.

So the two numbers are:

- **Advertised**: what the API accepts. 200k, 1M, 2M — read off the docs, date-stamped.
- **Usable**: the length beyond which *your* task's accuracy drops below *your* threshold. Measured, on your corpus, with your task.

How you measure the second one: build a length-stratified eval. Same questions, same gold answers, but with the supporting evidence embedded in contexts of 4k / 16k / 64k / 200k tokens, with position of the evidence randomized (not always at the end — that's how you accidentally certify a model that only reads the tail). Plot accuracy against context length. The knee in that curve is your usable limit.

**⚠ Trap:** using needle-in-a-haystack as your long-context eval and declaring victory. NIAH asks the model to find one verbatim, semantically out-of-place string. It is a weak proxy: models that ace NIAH at 1M tokens can fail multi-fact synthesis at 32k. If a candidate tells me "the model gets 100% on needle-in-a-haystack at 1M," my follow-up is "and what's its accuracy when the answer requires combining three facts from three different positions?" That is the question the product actually depends on.

**💰 Math:** the cost argument against long context is usually stronger than the quality argument. Stuffing 200k tokens of context into every request at $3/Mtok costs $0.60 per call in input alone. At 50k calls/day that is $30,000/day — **$900k/month**. A retrieval step that selects 4k relevant tokens instead costs $0.012/call → $600/day → $18k/month, plus embedding and index cost of maybe $2k/month. Long context is not free RAG; it's RAG's expensive cousin that you use when the corpus genuinely doesn't decompose.

**🗣 Say this in the room:** "I quote context as two numbers: advertised and usable-on-my-corpus. Advertised is what the API accepts; usable is where my length-stratified eval's accuracy crosses my threshold with evidence positioned randomly. For most of my workloads those differ by a factor of two to eight, and the difference is the entire justification for the retrieval layer."

### Draw me the full price surface for a modern provider — every distinct token rate — and tell me which line dominates in an agent.

There are five rates, and engineers routinely model two.

| Rate | Typical multiple of base input | Why it exists |
|---|---|---|
| Uncached input | 1× | Prefill compute you actually did |
| Cache write | 1.25× (short TTL) → 2× (long TTL) | You occupied KV memory for future reuse |
| Cache read | ~0.1× (Anthropic) — verify per provider | You skipped prefill; provider only pays memory |
| Output | 4–5× input | Decode is bandwidth-bound and doesn't batch as well |
| Reasoning / thinking | Billed as output | It *is* output, you just don't see it |

**📐 Numbers you must know (📅 Volatile — Anthropic first-party rates as cached 2026-06-24, re-verify):** Opus-class $5 in / $25 out per Mtok; Sonnet-class $3 / $15; Haiku-class $1 / $5. Cache read ≈ 0.1×; cache write 1.25× at 5-minute TTL, 2× at 1-hour TTL. Batch tier ≈ 50% off. The *ratios* are the durable part: output ≈ 5× input, cache read ≈ 0.1× input, batch ≈ 0.5×. Memorize the ratios, look up the absolutes.

Now, which line dominates? It depends entirely on your architecture, and being able to say which is a strong seniority signal:

- **Single-turn classification/extraction** (short output): *uncached input* dominates. Optimization = shorter prompts, cheaper tier, batch tier.
- **Chat with a long system prompt**: *cache reads* dominate after warm-up, and your job is protecting the cache from invalidation. Optimization = frozen prefix.
- **Agent loop with tools**: **output tokens dominate, and it isn't close.** Every turn re-sends the whole growing transcript (cheap, cached) and generates new reasoning + tool arguments (expensive, uncached, at 5× input rate). A 20-turn agent generating 600 output tokens/turn burns 12,000 output tokens = 12,000 × $15/1e6 = **$0.18 per task in output alone**, versus perhaps $0.04 of cached input.
- **Batch document processing**: *input* dominates and the batch tier is free money.

**💰 Math (the break-even for prompt caching, which you should be able to derive live):** cache write costs 1.25×, cache read costs 0.1×, uncached costs 1×. Two calls with caching = 1.25 + 0.1 = 1.35×; two calls without = 2×. **Caching pays from the second call at 5-minute TTL.** At 1-hour TTL: 2 + 0.1 = 2.1× for two calls versus 2× uncached — that *loses*; you need three calls (2.2× vs 3×) to break even. So the TTL choice is a traffic-shape decision: bursty traffic with gaps longer than the short TTL justifies the long one; steady traffic never does.

**⚠ Trap:** enabling the 1-hour TTL "to be safe" on a high-traffic endpoint. You've just doubled your cache-write cost on a prefix that would have stayed warm anyway. The rule I enforce in review: long TTL requires evidence of inter-request gaps exceeding the short TTL, from your own latency-between-requests histogram.

### When do you use the batch tier, and when does the 50% discount lose you money?

Mental model: **the batch tier is a spot market for GPU capacity — you're telling the provider "schedule me whenever you have a trough," and they pay you ~50% for the option.** It is the exact same trade as spot instances, and you should reason about it the same way: great for work with a soft deadline, catastrophic for work in a request path.

Mechanism (Anthropic's Message Batches API as a concrete instance; OpenAI's Batch API is structurally similar): you `POST` a list of requests, each tagged with a `custom_id`. You poll for `processing_status == "ended"`. You stream results. Limits as of mid-2026: up to 100,000 requests or 256 MB per batch, most complete within an hour, **maximum 24 hours**, results retained 29 days (**📅 Volatile**). Every Messages API feature works inside it — tools, vision, prompt caching.

Three rules I enforce:

1. **Key results by `custom_id`, never by position.** Results come back in arbitrary order. Indexing by array position is a bug that passes review, passes staging with 10 items, and silently mismatches 40,000 labels in prod. This is the single most common batch bug.
2. **Handle four result types, not one.** `succeeded`, `errored`, `canceled`, `expired`. `expired` means the 24h window elapsed — those requests never ran and must be resubmitted. A pipeline that only branches on `succeeded` silently drops them.
3. **Never put the batch tier in a path with a user waiting.** "Most complete within an hour" is not an SLA you can build a UI on.

Where batch **loses** you money or time:

- **Iterating on prompts.** A 20-minute batch turnaround on a prompt-tuning loop costs you a day of engineering time to save $40 of API spend. Use the sync API while iterating; switch to batch when the prompt is frozen.
- **Anything with a retry-and-refine loop.** Two dependent batch rounds is potentially a 48-hour pipeline.
- **When it breaks your cache.** Batch requests may not hit the same warm prefix cache your sync traffic maintains, so a workload whose real saving was a 90% cache-read rate can be *more* expensive at "50% off" uncached. Do the arithmetic before assuming.

**💰 Math:** a nightly eval suite, 8,000 examples × (3,000 input + 400 output) tokens against a $3/$15 model. Sync: (8000 × 3000 / 1e6 × $3) + (8000 × 400 / 1e6 × $15) = $72 + $48 = **$120/night** = $3,600/month. Batch at 50%: **$60/night** = $1,800/month. Saving $1,800/month for a job that runs at 2am and is read at 9am — obviously correct. Now the counter-case: that same suite with a shared 20k-token cached preamble across all 8,000 requests. Cached-sync input = 8000 × 20,000 × $0.30/1e6 = $48 versus batch-uncached 8000 × 20,000 × $1.50/1e6 = $240. **The "discounted" path costs 5× more.** Always compute against your *cached* baseline, not your list-price baseline.

### How would you evaluate structured-output support as a selection axis? What are you actually comparing?

The mental model: **there are three qualitatively different things vendors all call "JSON mode," and only one of them gives you a guarantee.**

1. **Prompted JSON.** You ask for JSON in the prompt. No enforcement. Failure rate on hard schemas is single-digit percent and correlates with exactly the inputs you care about (long, weird, adversarial). You need a parse-and-retry loop, and retries cost you a full extra generation.
2. **JSON mode.** The decoder is constrained to emit syntactically valid JSON. You get well-formed JSON — but not *your schema*. Missing required fields, wrong types, and invented keys all still happen.
3. **Constrained decoding against your schema** (Anthropic's `output_config.format` with a JSON schema, or `strict: true` on a tool definition; OpenAI's Structured Outputs with `strict: true`). The provider compiles your schema into a grammar/automaton and masks the logits at each decode step so that only tokens keeping the output on a valid path have nonzero probability. **Schema conformance becomes a property of the sampler, not of the model.** This is the only one where "0% malformed" is a defensible claim.

What to compare, concretely:

- **Which JSON Schema features are supported.** Universally the constrained-decoding implementations restrict the schema language. Typically supported: object/array/string/integer/number/boolean/null, `enum`, `const`, `anyOf`, `$ref`/`$defs`, and common string `format`s. Typically *not*: recursive schemas, numeric bounds (`minimum`/`maximum`), string length bounds, complex array constraints. `additionalProperties: false` is usually mandatory on every object. If your schema uses `minimum: 0`, find out whether the provider drops it silently (and validates client-side) or rejects the request — those are very different failure modes.
- **First-request latency.** New schemas incur a one-time compilation cost, then get cached (24h on Anthropic; **📅 Volatile**). If you generate schemas dynamically per request you pay that penalty every time — a real, measurable regression that looks like random latency spikes.
- **What happens on `max_tokens` or a refusal.** Constrained decoding guarantees the output is *on a valid path*, not that it *completed*. Hitting the token cap gives you truncated-but-locally-valid JSON. Your parser must handle it.
- **Interaction with other features.** On Anthropic, structured output is incompatible with citations (400) and with message prefilling; strict tool use is incompatible with programmatic tool calling and forced `tool_choice`. These matrices are real and they bite in integration, not in the prototype.

**⚠ Trap:** believing schema conformance implies semantic correctness. Constrained decoding guarantees the model emits `{"amount": 4200, "currency": "USD"}` in the right shape. It guarantees nothing about whether 4200 is the right number. I've watched teams celebrate "we eliminated our JSON errors" while their field-level accuracy sat at 82%. Constrained decoding moves your failure mode from *loud* (parse error) to *silent* (wrong value), which is strictly worse for observability unless you add field-level evals at the same time.

**🗣 Say this in the room:** "I treat structured output as a sampler feature, not a model capability — the provider masks logits against a compiled grammar, so conformance is guaranteed and correctness isn't. I select on which schema features survive compilation, and I always pair it with a field-level accuracy eval, because the whole point of enabling it is that malformed output stops being the thing that alerts me."

### Tool-calling quality — how do you measure it, rather than trusting a leaderboard?

Mental model: **"tool calling" bundles four separable skills, and models fail at different ones.** A model that emits perfectly-typed tool arguments 100% of the time and calls `web_search` for a question it already knows the answer to is a bad tool-caller, and no schema-validity metric will tell you that.

The four skills, each of which needs its own metric:

1. **Selection** — given a tool set and a request, does it call the right tool (or correctly call none)? Metric: accuracy over a labeled set that deliberately includes *no-tool-needed* cases. The no-tool cases are where over-eager models fail.
2. **Argument construction** — are the arguments not just schema-valid but *semantically* right? Metric: exact match on arguments against gold, field by field.
3. **Composition** — multi-step: does it chain tools correctly, pass outputs forward, and recover when a tool returns an error? Metric: end-to-end task success over a set of trajectories requiring 2–5 calls, plus a specific recovery-rate metric on injected tool errors.
4. **Termination** — does it stop when the task is done, or loop? Metric: distribution of turns-to-completion, with a hard cap; the mean is less interesting than the 95th percentile and the fraction that hit the cap.

The eval harness is a mock tool server with deterministic responses. Do not evaluate against live tools; you cannot separate model failure from API flakiness, and you'll spend a week chasing a rate limit.

```python
# Sketch of the harness shape — deterministic mocks, per-skill scoring.
CASES = [
    {"q": "What's 17% of 4,280?",       "gold_tool": "calculator", "gold_args": {"expr": "0.17*4280"}},
    {"q": "What's the capital of Peru?", "gold_tool": None},        # no-tool case: over-eager models fail here
    {"q": "Email the Q3 report to Sam",  "gold_tool": "send_email", "gold_args": {"to": "sam@…", "attach": "q3.pdf"}},
]
# score: selection accuracy, arg exact-match | correct selection, turns-to-completion p95.
```

Two things that move tool-calling quality more than model choice, and which you should raise unprompted because they signal you've shipped this:

- **Tool descriptions that state *when* to call, not just what the tool does.** "Get current weather" versus "Get current weather. Call this when the user asks about conditions in a named location; do not call it for historical or forecast questions beyond 7 days." The second reliably raises should-call rate on models that are conservative about tools.
- **Tool-set size.** Past roughly 15–20 tools, selection accuracy degrades measurably on every model I've tested. The fix is not a better model, it's namespacing, a router, or deferred tool loading (tool search) so only relevant schemas are in context.

**⚠ Trap:** measuring tool calling only on happy paths. The production failure mode is a tool that returns an error or an empty result and a model that either ignores it and hallucinates the answer, or retries the identical call forever. Inject error results into 20% of your eval trajectories. This single change is what separates an eval that predicts production from one that doesn't.

### Where does data retention and residency sit in your selection framework, and how has it actually vetoed a choice for you?

Mental model: **retention terms are a hard constraint that arrives late and has no engineering workaround.** They belong at the top of your framework even though they're the least technical axis, because discovering them in week six of a project costs you the project.

The axes that actually get negotiated:

- **Zero data retention (ZDR).** Provider does not persist prompts/completions beyond the request. Standard for enterprise contracts. The trap: **ZDR is not universally compatible with every model or feature.** As of mid-2026, Anthropic's Fable-class model requires 30-day retention and returns `400 invalid_request_error` on every request from a ZDR org — a perfectly valid payload that fails for a contract reason (**📅 Volatile**). If you're debugging a blanket 400 with no obvious request problem, check the org's retention configuration before you touch the payload. Similar coupling exists elsewhere: abuse-monitoring caches, prompt-caching persistence, and batch-result retention (29 days on Anthropic) are all *storage*, and a strict ZDR reading can exclude them.
- **Training on your data.** All the major API providers state that business/API traffic is not used for training by default. Consumer tiers are a different contract. Your legal team will want this in the enterprise agreement, not a docs page.
- **Residency.** Which region the inference physically runs in. Some providers expose it as a request parameter or an endpoint; some only through a cloud partner (Bedrock/Vertex/Foundry in a specific region). If you're serving EU healthcare or EU financial customers, this decides your provider before any eval runs.
- **Subprocessor chain.** Going through a cloud reseller adds a party to your DPA. Sometimes that's how you get approved (the customer already trusts AWS); sometimes it's what blocks you.

The engineering consequence you should raise: **cloud-partner deployments are feature-lagged.** Running Claude on Bedrock or Vertex rather than first-party gets you the core Messages API, but a documented subset of everything else — as of mid-2026, Message Batches, the Models API, several server-side tools, and Managed Agents are first-party-only or partner-restricted (**📅 Volatile — check the platform availability matrix, it changes every release**). So "legal says Bedrock" isn't a procurement footnote, it's an architecture constraint that can delete a feature you designed around.

**🗣 Say this in the room:** "Retention and residency are the first thing I check, not the last, because they're the only axis with no engineering workaround. And I check them against the *feature* matrix, not just the model list — a cloud-partner deployment that satisfies legal may not support the batch endpoint or the caching semantics my cost model assumed, and I'd rather find that out in week one."

**⚠ Trap:** assuming the provider's headline retention policy applies to every surface. Prompt caching stores your prefix somewhere for the TTL. Batch results are retained for weeks. Abuse-monitoring pipelines have their own window. Each of those is a separate line in the DPA, and "we use ZDR" is not an answer to "where is that 20k-token system prompt for the next hour?"

### It's the week before your loop. Walk me through your refresh checklist.

Mental model: **treat the volatile layer like a dependency lockfile — you don't memorize it, you re-resolve it, and you timestamp the resolution.** The interviewer is not testing whether you know today's price. They're testing whether you have a *process* that would have known last month's and will know next month's. Saying "I re-verify this before every loop, here's my checklist, and as of last Tuesday it was X" is a stronger answer than a confidently-recited stale number.

The checklist, in the order I run it (budget: 90 minutes):

1. **Provider model lineups and IDs.** Pull the live list programmatically, not from a blog post — Anthropic exposes `GET /v1/models` returning `id`, `display_name`, `max_input_tokens`, `max_tokens`, and a `capabilities` tree. That's your ground truth for context window and feature support, and it's newer than any doc page.
2. **Pricing pages, all three providers.** Record input, output, cache-read, cache-write, and batch rates. Compute the ratios (output/input, cache-read/input) and check whether they moved — the ratios moving is more interesting than the absolutes moving.
3. **Deprecation and retirement schedules.** Which model IDs you depend on have a retirement date inside the next two quarters.
4. **Feature/parameter deltas.** The thinking-control surface, sampling parameters, prefill support, and structured-output parameter names have all changed under stable-looking model families. Read the migration guide for the newest model in each family — it is the single densest source of "what changed."
5. **Rate-limit tier tables.** Your org's current RPM/ITPM/OTPM and what the next tier requires.
6. **Serving-engine feature matrices** if you're touching infra: which of vLLM / SGLang / TensorRT-LLM currently support the quantization format, attention variant, speculative decoding mode, and structured-output backend you're planning to claim in a design round.
7. **Open-weight releases** in the last 90 days and their licenses.
8. **Regulatory dates** — the EU AI Act milestone calendar, since one of them is probably within six months of your loop.

**🗣 Say this in the room, verbatim, when you don't know a current number:** "I don't want to quote you a stale price — that table moves monthly and I re-verify it before I make a decision. What I can give you is the structure: output is roughly 5× input, cache reads run around a tenth of base input, and batch is about half. Those ratios have been stable for two years and they're what the architecture decision actually turns on. If you want, I'll do the arithmetic with your real numbers."

That answer converts "I don't know" into a demonstration of judgment, and interviewers reward it. The failure mode is the opposite: confidently quoting a 2024 price in a 2026 room. It signals you haven't shipped recently, and it's unrecoverable — every subsequent number you say gets discounted.

**🏋 Drill (25 minutes, timed, unaided):** from memory, write down the five distinct token rates a modern provider charges, the approximate multiple of base input for each, the caching break-even in number-of-calls for a short TTL and a long TTL, and the three most common reasons a prefix cache silently stops hitting. Then verify against the live docs and mark every place you were wrong. Pass criterion: at most one wrong ratio, and all three cache invalidators correct.
### Write me the Anthropic messages loop with tools, end to end. I want to see what a correct implementation looks like.

Mental model: **the Messages API is stateless, and the "conversation" is a list you own and resend in full on every call.** There is no session on the server. Tool use is not a separate API — it's a `stop_reason` and two content-block types on the same `POST /v1/messages`. If you already ship stateless HTTP services, the only genuinely new thing here is that the response is a *heterogeneous list of typed blocks*, not a string, and that the loop terminates on a field rather than on your control flow.

The protocol in one paragraph: you send `messages` plus `tools`. The model responds with `stop_reason: "tool_use"` and a `content` array that may contain text blocks *and* one or more `tool_use` blocks, each with an `id`, `name`, and parsed `input`. You append the assistant's **entire `content` array** to history, execute the tools, and append a single user message containing one `tool_result` block per `tool_use` block, each carrying the matching `tool_use_id`. Loop until `stop_reason` is `end_turn`.

```python
import anthropic
client = anthropic.Anthropic()

messages = [{"role": "user", "content": user_input}]
for _ in range(MAX_TURNS):                      # always bound the loop
    resp = client.messages.create(
        model="claude-sonnet-5", max_tokens=16000,
        tools=TOOLS, messages=messages,
    )
    messages.append({"role": "assistant", "content": resp.content})   # full blocks, not text

    if resp.stop_reason == "refusal":            # check BEFORE reading content
        return handle_refusal(resp)
    if resp.stop_reason == "end_turn":
        return next(b.text for b in resp.content if b.type == "text")
    if resp.stop_reason == "pause_turn":         # server-side tool hit its iteration cap
        continue                                 # re-send as-is; the server resumes

    results = []
    for b in resp.content:
        if b.type == "tool_use":
            try:
                out = execute(b.name, b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            except Exception as e:               # NEVER drop a failed tool's result
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": f"Error: {e}", "is_error": True})
    messages.append({"role": "user", "content": results})   # ALL results in ONE message
```

Five invariants that are load-bearing, and each is a bug I have reviewed:

1. **Append `resp.content`, not the extracted text.** Dropping the `tool_use` blocks makes the next request reference tool results with no matching tool call → 400.
2. **Every `tool_use` block needs exactly one `tool_result` with a matching `tool_use_id`.** Missing one is a 400; extras are a 400.
3. **All parallel tool results go in a single user message.** Splitting them across multiple messages is accepted by the API but trains the model to stop making parallel calls — a silent throughput regression you'll never trace back.
4. **A failed tool returns `is_error: True`, it does not raise past the loop.** The model needs to see the failure to recover from it. Swallowing errors produces an agent that hallucinates the tool's output.
5. **Bound the loop.** An unbounded `while stop_reason == "tool_use"` is a runaway cost incident waiting for a model that decides to retry the same failing call forever.

**⚠ Trap:** parsing tool arguments by string-matching the serialized input. Models legitimately vary their JSON escaping (Unicode escapes, escaped forward slashes) across versions. The SDK already hands you `block.input` as a parsed dict — use it. Code that does `if '"city": "Paris"' in raw_input` passes review and breaks on a model upgrade.

**💰 Math:** the loop's cost is dominated by the resend. A 20-turn agent with a 15k-token prefix resends that prefix 20 times = 300k input tokens. Uncached at $3/Mtok that's $0.90 per task; with a cache breakpoint on the frozen prefix at 0.1×, roughly $0.09 + the write premium. At 100k tasks/month that's **$90k versus $10k**. The `cache_control` marker is the highest-ROI line of code in an agent.

### The thinking-control API has changed several times. Walk me through what's durable and what isn't.

This is a deliberately good question for this section, because it's the clearest example of a **volatile control surface over a durable mechanism**. Get the mechanism right and you can adapt to any parameter name; memorize the parameter name and you're stale in six months.

**The durable mechanism.** Reasoning models are trained to emit an extended chain of thought before their visible answer. Those reasoning tokens are generated by the same decoder, occupy the same context window, count against the same `max_tokens` ceiling, and **bill at the output rate**. The only real knob anyone can give you is *how many of them to spend*. Every API surface any provider has shipped is some encoding of "how much thinking," and the tradeoff is always the same three-way: more thinking → higher quality on multi-step problems → more output tokens → more cost and more latency-before-first-visible-token.

**The volatile surface, on Anthropic, in order (📅 Volatile — this is the timeline as of mid-2026; verify against the migration guide before your loop):**

- **Fixed budget era.** `thinking: {"type": "enabled", "budget_tokens": N}` with a minimum of 1,024 and a requirement that `budget_tokens < max_tokens`. You picked a hard token ceiling for reasoning. Simple, but you had to guess N per task, and a task that needed 40k got cut off at 8k.
- **Adaptive era.** `thinking: {"type": "adaptive"}` — the model decides per request how much to think. `budget_tokens` was deprecated on the 4.6 generation and **returns a 400 on 4.7 and later**. Depth is now controlled by a separate, coarser knob: `output_config: {"effort": "low"|"medium"|"high"|"xhigh"|"max"}`.
- **Default flips.** On some model generations, omitting `thinking` meant *no* thinking; on the current Opus generation, omitting it runs adaptive. That is a silent cost and truncation change on migration: a workload that sized `max_tokens` tightly around its visible answer now truncates, because thinking and answer share the ceiling.
- **Visibility.** The raw chain of thought is not returned. `thinking.display` is `"omitted"` by default (empty-text thinking blocks) or `"summarized"` for a readable summary. If you stream reasoning to a user and don't set `"summarized"`, the UX is a long pause with no output.

**⚠ Trap:** if you're asked "how do you set a thinking budget?" the wrong answer is to recite `budget_tokens`. The right answer names the mechanism, then the current surface, then flags the version dependency: *"Depth control — currently `effort` inside `output_config` on Anthropic's recent models, and `budget_tokens` before that, which now 400s on the newer ones. The durable point is that reasoning tokens are output tokens: they bill at the output rate, they share the `max_tokens` ceiling with the answer, and they're invisible in the response text, so any cost dashboard built on rendered-text length under-reports them badly."*

**💰 Math:** a request whose visible answer is 120 tokens but which spends 4,000 reasoning tokens bills 4,120 output tokens. At $15/Mtok that's $0.0618 versus the $0.0018 you'd estimate from the visible text — **34× under-reporting**. At 500k requests/month you'd budget $900 and receive an invoice for $30,900. Read `usage.output_tokens`, never `len(text)`.

**🗣 Say this in the room:** "Reasoning is output. It bills at the output rate, it eats the `max_tokens` ceiling, and it's invisible. So the two things I always do are: size `max_tokens` for thinking-plus-answer, and drive cost dashboards off the API's `usage` object rather than anything I can measure client-side."

### Place the cache breakpoints for a multi-turn agent and tell me exactly what invalidates them.

Mental model: **prompt caching is a prefix match, not a content-addressed cache.** The key is the exact bytes of the rendered prompt up to each breakpoint. One byte changed at position N invalidates every breakpoint at position ≥ N. Every caching decision follows from that single sentence, and if you internalize it you never need to memorize a rule.

**Render order is `tools` → `system` → `messages`.** So a breakpoint on the last system block caches tools *and* system together. That ordering is why "just add a tool mid-conversation" is catastrophic: tools render at position zero, so touching them invalidates literally everything.

Placement for an agent:

```python
resp = client.messages.create(
    model="claude-sonnet-5", max_tokens=16000,
    tools=TOOLS,                                    # deterministic order — sort by name
    system=[{"type": "text", "text": FROZEN_SYSTEM,
             "cache_control": {"type": "ephemeral"}}],   # breakpoint 1: tools + system
    messages=[
        *history,                                    # breakpoint 2 on the last block of the
                                                     # most-recently-appended turn
        {"role": "user", "content": turn_content},
    ],
)
```

Two breakpoints, not four: one on the frozen prefix (tools + system), one rolling on the tail of the conversation so each turn reuses the whole prior transcript. You get a maximum of four breakpoints per request; spending them all is usually a sign the prompt is badly organized.

**Design rules that matter more than marker placement:**

- **Freeze the system prompt.** No `datetime.now()`, no user ID, no `if flag: system += ...`. Dynamic context goes *after* the last breakpoint — in a user turn, or as a mid-conversation `{"role": "system", ...}` message on models that support it (which sits after the cached history rather than in front of it).
- **Serialize tools deterministically.** `json.dumps(..., sort_keys=True)`; never iterate a `set`.
- **Never switch model mid-conversation.** Caches are model-scoped. If you want a cheaper model for a sub-task, spawn a subagent rather than swapping the main loop's model.
- **Fork operations must copy the parent's `system`, `tools`, and `model` verbatim.** A summarization or compaction call that rebuilds the prefix "close enough" misses the parent's cache entirely.

**The invalidation hierarchy** — the part most candidates don't know, and it's genuinely useful:

| What you changed | Tools cache | System cache | Messages cache |
|---|:--:|:--:|:--:|
| Tool definitions, or the model | ❌ | ❌ | ❌ |
| System prompt content | ✅ | ❌ | ❌ |
| `tool_choice`, images, thinking on/off | ✅ | ✅ | ❌ |
| Message content | ✅ | ✅ | ❌ |

So `tool_choice` can vary per request without rebuilding the expensive tools+system prefix — though it does cost you the messages cache, so it is cheap rather than free — and toggling thinking is the same trade. Only tool-definition and model changes force a full rebuild.

**📐 Numbers you must know:** the **minimum cacheable prefix is model-dependent and non-monotonic** — roughly 512 tokens on the newest Opus/Fable generation, 1024 on Opus 4.8 / Sonnet, 2048 on the Haiku line and older generations (**📅 Volatile — read the exact figure off the prompt-caching docs for the specific model ID**). A 3k-token prompt caches on one model and *silently does not* on another, with no error — just `cache_creation_input_tokens: 0`. Also: each breakpoint walks back at most **20 content blocks** looking for a prior entry, so an agent turn that appends 30 tool_use/tool_result blocks blows past the lookback and silently misses. Add an intermediate breakpoint every ~15 blocks in long turns.

**⚠ Trap:** parallel fan-out on a cold cache. A cache entry only becomes readable once the first response *begins streaming*. Firing N identical-prefix requests concurrently means all N pay full price and N−1 of the writes are wasted. Send one, await the first streamed token, then fire the rest.

### Your prefix-cache hit rate went from 85% to 12% overnight. No deploy went out. Debug it.

**🔍 Failure taxonomy — run this as a decision procedure, not a hunt.**

**Step 0: confirm the measurement.** Hit rate must be computed from `usage.cache_read_input_tokens / (cache_read + cache_creation + input_tokens)`, per request, aggregated. If you're computing it from request counts rather than token counts you may be looking at an artifact. Also confirm `input_tokens` isn't being read as "total prompt size" — it's the *uncached remainder only*.

**Step 1: is the prefix still byte-identical?** Log the SHA-256 of the rendered prefix (tools JSON + system text) on every request, sampled. Diff two requests from the affected window against two from before. This localizes the change in one step and is the single highest-value instrumentation you can add. If the hashes differ, go to step 2. If they're identical, go to step 4.

**Step 2: what changed in the prefix, given no deploy?** In order of frequency:

- **A time-varying value in the system prompt.** Not necessarily a timestamp you wrote — a config value fetched from a feature-flag service, a rendered "today's date," a rotating tenant list. "No deploy" does not mean "no change" when the prompt is assembled from remote config.
- **Tool set changed underneath you.** An MCP server added a tool, or a dynamic tool registry re-ordered. Tools render at position 0, so this nukes everything.
- **Non-deterministic serialization.** A `dict` that gained a key, a `set` iterated for a tool list, `json.dumps` without `sort_keys`. This can flip on a Python minor-version or dependency bump without a deploy of your code.
- **Model alias moved.** You pinned `claude-sonnet-5` and the alias now resolves to a new snapshot. Caches are model-scoped; a silent alias update invalidates every entry at once. This one matches the "overnight, no deploy" symptom best.

**Step 3: if the prefix is identical but reads are zero — check the size and the TTL.**

- Did the prompt shrink below the model's **minimum cacheable prefix**? A refactor that trimmed 300 tokens from a 1,200-token system prompt on a model with a 1,024 minimum turns caching off with no error.
- Did traffic get sparser? Cache TTL is 5 minutes by default. A drop in QPS from 10/s to 1/min means every request is a cold write. Plot inter-request gap per prefix against the TTL — this is why the symptom can be "traffic went down and cost per request went *up*."
- Did your load balancer start spreading traffic across regions or accounts? Caches don't follow you across those boundaries.

**Step 4: the 20-block lookback.** If your agent turns got longer (a tool started returning more blocks), consecutive turns can exceed the lookback window between breakpoints.

**💰 Math on the blast radius:** 200k calls/day, 12k-token prefix, $3/Mtok input. At 85% hit: 0.85 × 12,000 × $0.30/1e6 + 0.15 × 12,000 × $3.75/1e6 = $0.00306 + $0.00675 = $0.00981/call → $1,962/day. At 12% hit: 0.12 × $0.0036 + 0.88 × $0.045 = $0.000432 + $0.0396 = $0.0400/call → **$8,006/day**. That is **+$181k/month** appearing with no code change and no alert, unless you alert on hit rate directly. Which is the lesson: **cache hit rate is a first-class SLI with a page-worthy threshold**, exactly like your Redis hit rate — and unlike Redis, a miss here costs real dollars per event rather than just latency.

### OpenAI has both Chat Completions and the Responses API. Which do you build a new product on, and why?

Mental model: **Chat Completions is a stateless text endpoint that grew tool-calling by accretion; the Responses API is a re-designed surface where the unit is a typed *item* stream and the server can hold state.** The distinction that matters architecturally is what happens to reasoning: on a stateless completions endpoint, a reasoning model's chain of thought is generated, billed, and then thrown away between turns, so the next turn re-derives it. A surface with server-side item state can carry reasoning items forward.

For a **new** product I build on the Responses API, for four reasons:

1. **Reasoning continuity across turns.** In an agent loop with a reasoning model, being able to carry the reasoning context forward (via a previous-response reference) avoids re-deriving the same chain and measurably reduces both cost and turn count on multi-step tasks.
2. **Built-in server-side tools** (web search, file search, code interpreter) that don't require me to implement, host, and secure an execution loop. For a product team, "the tool runs on their infrastructure" is a real staffing argument.
3. **One item taxonomy** rather than the completions API's split between `message.content`, `tool_calls`, and various out-of-band fields. It's less code to write and less code to get subtly wrong.
4. **It's where new features land.** The older endpoint is maintained but is not where the surface is evolving.

When I'd stay on **Chat Completions**: you have an existing integration that works and no reasoning-model or built-in-tool requirement; or you're deliberately targeting the de-facto-standard request shape that every OpenAI-compatible server (vLLM, SGLang, LiteLLM, Together, Fireworks, Groq, Ollama) implements. That last one is a serious architectural argument — **the Chat Completions schema is the lingua franca of the whole open-source serving ecosystem**, and building your internal abstraction around it means you can point at a self-hosted model by changing a base URL. If your roadmap includes "we may self-host a fine-tuned open model," that portability is worth more than the newer surface's ergonomics.

**⚠ Trap:** assuming server-side state means you can stop sending history. Stateful referencing is an optimization with a retention window and its own privacy posture; if the reference expires or the org is under a strict retention configuration, you need the full-history path anyway. Build the stateless path first and treat state as a cache, exactly as you'd treat a session store you don't own.

**🗣 Say this in the room:** "New product, OpenAI-native: Responses API, mostly for reasoning-item continuity and the hosted tools. Anything where I want provider portability or a self-hosting escape hatch: I design the internal interface around the Chat Completions shape, because that's what every open-source inference server speaks, and it turns a provider migration into a base-URL change."

**📅 Volatile:** exact parameter names, which built-in tools are available, and the state-retention window all move. Re-read the API reference the week of your loop; the architectural argument above is what's being tested.

### Reasoning effort is a single enum. How does it actually interact with cost and latency, and how do you tune it?

Mental model: **effort is a dial on the *number of output tokens the model spends before answering*, and every downstream consequence follows mechanically from that.** More effort → more reasoning tokens → linearly more output cost, linearly more time-to-first-visible-token, and (on hard multi-step tasks only) higher accuracy with strongly diminishing returns. On easy tasks it buys nothing and can actively hurt by inviting overthinking.

The mechanics you should be able to state:

- **Cost scales with reasoning tokens at the output rate.** If `high` produces 3× the reasoning tokens of `low`, and reasoning dominates your output, your cost roughly triples. Measure it: run your eval at each level and record mean `output_tokens`, not just accuracy.
- **Latency scales the same way, and it lands entirely on perceived TTFT** for a streaming UI, because nothing visible streams while the model thinks. This is why effort is a *product* decision on interactive surfaces, not a cost decision.
- **Effort also changes agentic behavior, not just depth.** Lower effort empirically means fewer and more-consolidated tool calls, less preamble, terser output; higher effort means more exploration and more delegation. So on an agent, effort changes the *shape* of the trajectory, and the token cost can move superlinearly because more tool calls means more tool results in context.
- **It's not monotonic in total cost.** On agentic work, higher effort up front often *reduces* total turn count and therefore total spend. That's counterintuitive and worth saying out loud: you cannot infer total task cost from per-request effort.

The tuning procedure: **sweep, don't default.** Run your eval at every level. Produce a table of accuracy, mean output tokens, p95 total latency, and cost per resolved task. Pick per route, not globally — a classification endpoint and an agentic coding endpoint should not share an effort setting.

**💰 Math:** eval task, 1,000 examples. `low` → 84% success, mean 900 output tokens. `high` → 91% success, mean 3,400 output tokens. At $15/Mtok output: low = 900 × $15/1e6 = $0.0135/task; high = $0.051/task. Cost **per success**: low = $0.0135/0.84 = $0.0161; high = $0.051/0.91 = $0.0560. High is **3.5× more expensive per success** for 7 points of accuracy. Whether that's worth it depends entirely on what a failure costs you — if a failure means a human handles the ticket at $4, then 7 points on 100k tasks saves 7,000 × $4 = $28,000 of human time for 100,000 × $0.0375 = $3,750 of extra inference. Obviously worth it. If a failure just means the user retries, it isn't. **The senior answer is that this question is unanswerable without the cost of a failure, and you should say so.**

**⚠ Trap:** using effort as a verbosity control. Lowering effort to make responses shorter is unreliable — on recent models it moves thinking volume without dependably changing visible output length. Verbosity is a prompt problem; effort is a depth problem. Confusing them produces a workload that's cheaper *and* worse with no shortening achieved.

### Explain how strict structured outputs actually work at the decoder, and show me the mechanism in code.

Mental model: **you are not asking the model to follow a schema; you are deleting the tokens that would violate it.** The schema is compiled into a finite automaton (a grammar over the token vocabulary). At each decode step the sampler computes which vocabulary tokens keep the partial output on a path that can still complete to a valid document, and sets the logits of all other tokens to −∞ before softmax. Conformance therefore becomes a property of the sampler, and its probability is exactly 1 — not "very high."

The from-scratch version, ~25 lines, which you should be able to write at a whiteboard:

```python
import torch

def constrained_generate(model, tokenizer, prompt_ids, automaton, max_new=256):
    """automaton.allowed(state) -> set[int] of legal next token ids
       automaton.step(state, tok) -> next state; automaton.accepting(state) -> bool"""
    ids, state = list(prompt_ids), automaton.start()
    for _ in range(max_new):
        logits = model(torch.tensor([ids])).logits[0, -1]          # [vocab]
        mask = torch.full_like(logits, float("-inf"))
        allowed = automaton.allowed(state)                          # e.g. {token ids for '"', digits, ...}
        mask[list(allowed)] = 0.0
        tok = torch.multinomial(torch.softmax(logits + mask, -1), 1).item()
        ids.append(tok)
        state = automaton.step(state, tok)
        if automaton.accepting(state):
            break
    return tokenizer.decode(ids[len(prompt_ids):])
```

Everything real (Outlines, XGrammar, llguidance, and the providers' hosted implementations) is this plus heavy engineering: precomputing the token-to-transition index so `allowed()` is O(1) rather than a vocabulary scan, handling the fact that BPE tokens span grammar symbol boundaries, and caching compiled automata across requests.

**Consequences you must know:**

- **Schema compilation is not free, and it's cached.** First request with a novel schema pays a compilation cost; subsequent ones hit a cache (24 hours on Anthropic; **📅 Volatile**). Generating schemas dynamically per request means paying it every time — a latency regression that looks like random spikes.
- **The supported schema subset is narrower than JSON Schema.** Recursion is usually out. Numeric bounds (`minimum`), string-length bounds (`minLength`), and complex array constraints usually aren't expressible in the grammar — some SDKs strip them and validate client-side, some providers reject. `additionalProperties: false` is typically mandatory on every object, and OpenAI's strict mode additionally requires every property to appear in `required` (you express optionality as a union with `null`).
- **Masking can distort the distribution.** Forcing a path the model considered unlikely can degrade content quality — the classic symptom is a model that produces correctly-shaped output with a lower-quality *value* than it would have written free-form. Mitigation: keep schemas shallow, put a free-text `reasoning` field *before* the constrained fields so the model can think inside the schema, and always measure field-level accuracy rather than parse rate.
- **Truncation still happens.** `max_tokens` cuts you off mid-document; the prefix is grammar-valid but the document is incomplete.

**⚠ Trap:** believing "0% malformed" means "0% wrong." It converts a loud failure into a silent one. Pair every structured-output rollout with a field-level accuracy eval, or you have removed your only alerting signal.

### Walk me through what's genuinely different about the Gemini surface.

Three things are structurally different, and they're the ones worth knowing rather than the parameter-name trivia.

**1. Caching is explicit and object-based, not implicit and prefix-based.** Anthropic and OpenAI infer the cacheable prefix from your request (marker-based or automatic). Gemini has you *create a cached-content resource* — you upload the context, get back a handle with a TTL, and reference the handle in subsequent `generateContent` calls. This is a genuinely different operational model:

- **Pro:** you control the lifetime explicitly, you can list and delete entries, and a shared document cached once can be referenced by many unrelated requests without a byte-identical prefix.
- **Con:** it's a resource with a lifecycle you must manage — create, refresh TTL, delete — and there's a minimum token count below which caching is unavailable. You need a small control plane, not a marker.
- **Design consequence:** for a "chat with this 300-page PDF" product, explicit caching is a better fit — one cache object per document, many users, many questions. For an agent with a frozen system prompt, implicit prefix caching is less code.

**2. Thinking is a token budget, not an enum.** `thinkingConfig` takes a `thinkingBudget` in tokens (with sentinel values for "off" and "let the model decide"), plus a flag to return thought *summaries*. Compared to an effort enum this is finer-grained and easier to reason about for cost — you can bound reasoning spend arithmetically — but it puts the burden of picking the number on you, per task. **📅 Volatile:** which models honor which sentinel values, and whether thinking can be disabled at all, varies by model generation; verify.

**3. Multimodality is native and wide, and that's the actual reason to pick it.** Long video, long audio, and large document sets are first-class inputs rather than adapters bolted on. The engineering implication is a Files API for large media plus a token-accounting model where a minute of video or audio maps to a token count you must compute — you cannot estimate media cost from file size. For a product like "answer questions over this 90-minute recording," this is the differentiator, and the very large advertised context window is what makes it usable.

Also structurally distinct: **`systemInstruction` is a separate top-level field** from `contents` (rather than a role inside the message list), and **safety settings are explicit per-category thresholds** you configure per request — which is a real operational lever, and also a real source of "why did this return empty" incidents when a threshold is stricter than you assumed.

**⚠ Trap:** treating the very large advertised context as a substitute for retrieval. Everything in the advertised-vs-usable discussion applies with more force at 1M+ tokens, and the cost arithmetic is brutal: filling a 1M-token window even at a low per-token rate is dollars per request. Long context is for corpora that genuinely don't decompose — a single long video, a codebase you must reason over globally — not as a way to skip building an index.

**🗣 Say this in the room:** "The interesting difference isn't the parameter names, it's that Gemini's caching is an explicit resource with a TTL rather than an inferred prefix. That changes the architecture: one cache object per shared document, referenced by many users, instead of requiring every request to share a byte-identical prefix. For a document-QA product that's a better fit; for an agent with a frozen system prompt, marker-based caching is less machinery."

### You're building an internal SDK over three providers. What do you normalize and what do you deliberately leak?

Mental model: **the wrong abstraction here is the one that makes all three providers look the same, because the differences that matter are exactly the ones you'd be hiding.** I've watched teams build a beautiful `LLMClient.complete(prompt) -> str` and then spend two quarters unable to use prompt caching, structured outputs, or reasoning control, because none of them fit the interface. Design for the *union* of capabilities with explicit capability detection, not the intersection.

**Normalize (safe, high value):**

- **Request envelope:** model, messages, system, tools, max_tokens, stream. These map cleanly.
- **Content-block taxonomy:** text / image / document / tool_use / tool_result / thinking. Adopt a superset and translate.
- **Usage accounting:** one `Usage` struct with `input_uncached`, `cache_read`, `cache_write`, `output`, `reasoning`. Every provider reports these under different names and some fold reasoning into output — normalizing this is where most of the value is, because it's what your cost attribution runs on.
- **Errors:** map to a typed hierarchy with a `retryable: bool`. 429 / 5xx / connection → retryable; 400 / 401 / 404 → not. Never string-match error messages.
- **Streaming events:** a single event union — `message_start`, `text_delta`, `tool_call_delta`, `thinking_delta`, `usage_update`, `done`, `error`. Adapters translate each provider's SSE shape into it.
- **Retry, timeout, and rate-limit handling.** Your policy, one place.

**Leak deliberately (attempting to normalize these creates bugs):**

- **Caching.** Marker-based prefix caching and explicit cache-object caching are not the same operation and cannot share an interface without lying. Expose both as provider-specific, and expose a common `cache_stats` in `Usage` so your *measurement* is uniform even though your *mechanism* isn't.
- **Reasoning control.** An enum on one, a token budget on another, absent on a third. Expose `effort: Literal["low","medium","high"] | None` as a *hint* that each adapter maps as best it can, and let callers pass through raw provider params for anything precise. Do not invent a fake universal token budget.
- **Structured-output schema dialects.** The supported JSON Schema subsets differ. Validate the schema against the target provider's subset at construction time and fail loudly, rather than silently dropping constraints.
- **Model identity.** Never define your own model enum that maps to provider IDs. Pin exact provider IDs in config. An internal `Model.FAST` that silently repoints is how you get an unexplained eval regression.

**The capability-detection pattern:**

```python
@dataclass(frozen=True)
class Capabilities:
    prefix_cache: bool
    explicit_cache: bool
    strict_schema: bool
    reasoning: Literal["none", "effort", "budget"]
    max_input_tokens: int
    parallel_tools: bool

# Callers branch on capabilities; they never branch on provider name.
if caps.strict_schema:
    req.output_schema = SCHEMA
else:
    req.system += JSON_INSTRUCTIONS      # and enable the parse-retry path
```

Branching on capability rather than on provider name is what makes the abstraction survive a fourth provider.

**⚠ Trap:** normalizing the token *count*. Tokenizers differ; the same text is a different number of tokens on every provider, and even across generations of the same provider (a tokenizer change of 1×–1.35× is normal on a major model revision). Any code that estimates tokens once and reuses the number across providers will misbudget context windows and truncate. Call each provider's own token-counting endpoint, and never use `tiktoken` to estimate Claude tokens — it undercounts by 15–20% on prose and much more on code.

### Streaming: what does your gateway have to get right, and what breaks in production?

Mental model: **a streaming LLM response is a long-lived HTTP response over infrastructure designed for short ones**, and every component between your client and the provider — load balancers, proxies, CDNs, serverless runtimes, and your own framework's response buffering — has an opinion about that.

What the wire actually looks like (Anthropic SSE, and everyone's is structurally similar): `message_start` carrying metadata and initial usage → `content_block_start` → a run of `content_block_delta` → `content_block_stop` (repeated per block, so text, thinking, and tool_use each get their own block index) → `message_delta` carrying the final `stop_reason` and output token count → `message_stop`. **Usage arrives at the end, not the beginning** — so any cost accounting must be attached to stream completion, and a client that disconnects mid-stream leaves you with tokens you were billed for and did not record.

What the gateway must get right:

1. **Don't buffer.** A framework that buffers the whole response before flushing turns streaming into non-streaming with extra steps. This is the number-one cause of "we implemented streaming and TTFT didn't improve."
2. **Disable proxy buffering explicitly.** nginx buffers proxied responses by default; you need the response-buffering directive off (or the `X-Accel-Buffering: no` header) or you get the same symptom one layer out.
3. **Heartbeats through idle timeouts.** A reasoning model can go 30+ seconds with no visible token. An ALB with a 30-second idle timeout will cut it. Either emit SSE comment heartbeats or raise the timeout — and if you're on a serverless runtime with a hard response-duration cap, a long reasoning turn may simply be unservable there.
4. **Accumulate tool arguments across deltas.** Tool call arguments stream as JSON fragments. You cannot parse until the block closes. Code that tries `json.loads` on each delta works in dev (short arguments arrive in one chunk) and fails in prod.
5. **Propagate cancellation.** When the user closes the tab, abort the upstream request. Otherwise you pay for output nobody reads — and on an agent, you pay for the *rest of the trajectory*.
6. **Handle mid-stream errors.** An error can arrive as an event *after* you've already emitted 200 OK and half an answer. Your protocol to the browser needs an error frame; you can't fall back to an HTTP status code you already sent.

**⚠ Trap:** timeouts in the SDKs are not wall-clock caps in the way you expect. Read timeouts in `requests`/`httpx` are **per-chunk** and reset on every byte — a trickling stream can block indefinitely under a "120-second timeout." Also, SDK retries multiply: with a 10-minute timeout and 2 retries, worst-case wall clock is 30 minutes. If you need a hard deadline, enforce it yourself with a monotonic-clock check at the loop level.

**💰 Math on cancellation propagation:** 5% of an agent product's 200k daily sessions are abandoned mid-trajectory, average 40% of the way through a 20-turn task costing $0.18 in output. Without cancellation you pay the remaining 60%: 200,000 × 0.05 × 0.6 × $0.18 = **$1,080/day = $32.4k/month** for output no human will read. Cancellation propagation is a two-line change and a $390k/year line item.

### Extended thinking plus multi-turn tool use — there's a replay rule people get wrong. What is it, and why does it exist?

Mental model: **thinking blocks are signed, provider-verified artifacts, not free text you own.** When you continue a conversation, you must echo them back **exactly as received** — including blocks whose visible text is empty. The API rejects blocks that have been *modified*; it does not reject blocks you merely read. The signature exists so the provider can verify the reasoning context wasn't tampered with, which matters because reasoning influences the model's subsequent behavior and is a natural injection surface.

The concrete rules, and the failure each one prevents:

1. **Append the assistant's entire `content` array, verbatim.** Not the text. Not a filtered list. Filtering out thinking blocks in a "clean up history" pass produces ordering/signature 400s that are baffling to debug, because the code that *causes* them lives nowhere near the code that reports them.
2. **Empty-text thinking blocks are still blocks.** When `display` is `"omitted"` — the default on several current models — you receive `thinking` blocks whose text is `""`. They look like noise and they are not. Dropping them is the single most common version of this bug.
3. **Don't reconstruct.** Copy the object. Rebuilding a thinking block from its fields, or round-tripping through a schema that drops the signature, invalidates it.
4. **Cross-model replay differs from same-model replay.** Ordinary thinking blocks generally replay fine to a different model (the server renders them). Blocks from the models whose raw chain of thought is never returned are *dropped from the prompt* when replayed to a different model — silently, and before pricing, so they lower your `input_tokens` and you aren't billed. That's fine; what's not fine is depending on either behavior.
5. **After a mid-output fallback to a different model, omit the pre-boundary model-internal blocks** — thinking, redacted thinking, unmatched tool_use, unpaired server-tool blocks — when echoing back. Text blocks and everything after the boundary echo normally.

**⚠ Trap:** the "compaction" bug. Teams write a history-compaction step that summarizes old turns into a single text block to save context. If that step drops thinking blocks or breaks tool_use/tool_result pairing, the next request 400s — and because compaction only triggers at length, it fires in production after the feature has been live for weeks and never in your test suite. **Write a test that runs your compaction over a 40-turn transcript containing thinking blocks and parallel tool calls, and asserts the result is still an accepted request.** That test has caught this for me twice.

**🗣 Say this in the room:** "Thinking blocks are signed artifacts — I append the full `content` array verbatim, never a filtered version, and I treat empty-text thinking blocks as load-bearing because on current defaults that's what you get. The failure mode is that a history-cleanup or compaction step strips them, and you get a 400 in production weeks later from code that looks unrelated."

### What actually breaks in the tool_use / tool_result protocol? Give me the failure taxonomy.

**🔍 Failure taxonomy, ordered by how often I've seen each in review:**

**1. Unmatched IDs → 400.** Every `tool_use` block needs exactly one `tool_result` with a matching `tool_use_id`, in the immediately following user message. Causes: a tool that raised and whose result was never appended; a filter that dropped a `tool_use` block; parallel calls where one path returned early. **Fix pattern:** build the results list by iterating the model's blocks, never by iterating your own dispatch table — that guarantees one result per call structurally.

**2. Results split across multiple user messages → silent behavior regression.** Accepted by the API. Trains the model to stop emitting parallel tool calls, so your agent gets slower and more expensive over time with no error. Diagnostic: track calls-per-assistant-turn as a metric; a drop from 2.3 to 1.0 after a refactor is this bug.

**3. Swallowed tool errors → hallucination.** If a tool fails and you append a generic "an error occurred" or skip the result, the model has no information and confabulates. Return `is_error: True` with the actual error text (sanitized). Models recover well from informative errors and terribly from silence.

**4. Runaway loops.** The model calls the same tool with the same arguments repeatedly. Causes: a tool that returns an empty result the model reads as "try again," or a task the tool set genuinely cannot accomplish. **Fixes:** a hard turn cap; a duplicate-call detector that injects "you already called this with these arguments and got this result" as an error result; and making empty results explicit ("no matching records found" beats `[]`).

**5. Tool-set drift breaking the cache.** Adding or reordering tools mid-conversation invalidates the entire prefix cache because tools render at position zero. Symptom: cost per task creeping up in an agent whose tool registry is dynamic. Fix: freeze and sort the tool list; use deferred loading / tool search if you genuinely need a large dynamic set.

**6. Tool-set size degrading selection.** Past ~15–20 tools, selection accuracy falls on every model I've measured. Not a bug you can fix with a prompt — it's a design constraint. Namespace, route, or defer.

**7. Server-side tools and `pause_turn`.** When a provider-hosted tool loop hits its internal iteration cap, you get `stop_reason: "pause_turn"` rather than a completion. The correct handling is to re-send the conversation with the paused assistant turn appended and *no* extra "continue" message — the server resumes on seeing the trailing server-tool block. Adding "Please continue." as a user turn is a real bug: it changes the conversation and can derail the resume. Note also that some SDK tool-runner helpers do **not** auto-resume `pause_turn` — the loop just ends and returns a silently truncated answer with no error. If you use a runner with server-side tools, check `stop_reason` on every iteration.

**8. Forced `tool_choice` incompatibilities.** Forcing a specific tool is mutually exclusive with several features (strict/programmatic tool calling, some structured-output modes, some platform-specific thinking configurations). These matrices are real; read them before designing around a forced call.

**⚠ Trap:** implementing your own dispatch loop when the SDK's tool runner already handles the protocol, on the belief that you "need control for human-in-the-loop approval." You don't — modern runners yield the assistant message *before* the tools execute and let you gate, modify, or override. Hand-rolling the loop to get approval gates is how the unmatched-ID and split-results bugs get written. Drop to a manual loop when you genuinely need control the runner doesn't expose, and be able to name what that is.
### Read me the rate-limit headers and then write the client that respects them.

Mental model: **an LLM provider's rate limiter is a multi-dimensional token bucket, and the dimension that kills you is not requests.** You are limited on requests per minute *and* input tokens per minute *and* output tokens per minute, independently. A workload of 10 RPM with 200k-token prompts saturates ITPM while sitting at 2% of your RPM ceiling — and the 429 you get back looks identical to a request-rate 429 unless you read the headers.

The headers, by provider family (**📅 Volatile — names drift; the structure doesn't**):

- **Anthropic:** `anthropic-ratelimit-requests-limit` / `-remaining` / `-reset`, plus the same triple for `input-tokens` and `output-tokens`, plus `retry-after` on a 429. `-reset` is an RFC 3339 timestamp, not a delta.
- **OpenAI:** `x-ratelimit-limit-requests` / `-remaining-requests` / `-reset-requests` and `-limit-tokens` / `-remaining-tokens` / `-reset-tokens`. Resets are durations like `6m0s`.

**📐 Numbers you must know:** the second mechanism, and the one that actually explains most surprise 429s — **admission control reserves against your declared `max_tokens`, not your actual output.** A request with `max_tokens: 128000` debits your output-token bucket by 128,000 at admission even if it generates 200 tokens; the difference is refunded after completion. So a service that lazily sets `max_tokens` to the model's ceiling everywhere will 429 at a fraction of its real throughput — and the fix is a one-line change to a sane per-route ceiling. This is one of the highest-value pieces of trivia in this section because it looks like a capacity problem and is actually a configuration problem.

```python
import time, random, anthropic

class BudgetedClient:
    def __init__(self, client, floor=0.05):
        self.c, self.floor = client, floor          # floor: pause below 5% remaining

    def call(self, **kw):
        for attempt in range(6):
            try:
                raw = self.c.messages.with_raw_response.create(**kw)
            except anthropic.RateLimitError as e:       # reactive: the 429 you didn't avoid
                time.sleep(on_429(e.response.headers, attempt))
                continue
            h = raw.headers
            msg = raw.parse()
            # Proactive: back off before the bucket empties, don't wait for the 429.
            rem  = int(h.get("anthropic-ratelimit-input-tokens-remaining", 1 << 30))
            lim  = int(h.get("anthropic-ratelimit-input-tokens-limit", 1 << 30))
            if lim and rem / lim < self.floor:
                time.sleep(_secs_until(h.get("anthropic-ratelimit-input-tokens-reset")))
            return msg
        raise RuntimeError("exhausted retries")

def on_429(headers, attempt):
    ra = headers.get("retry-after")
    if ra:                                   # ALWAYS prefer the server's number
        return float(ra)
    return min(60.0, 2 ** attempt) + random.uniform(0, 1)   # jitter is mandatory
```

Three rules I enforce in review:

1. **Honor `retry-after` over your own backoff.** Your exponential curve is a guess; the header is the answer.
2. **Full jitter, always.** Synchronized retries from N workers reconverge into a thundering herd that keeps you 429'd. This is the same lesson as any distributed retry storm — you already know it, and people forget it here because the SDK "handles retries."
3. **Back off *before* the 429, using `-remaining`.** A 429 is a wasted round trip and, on some providers, still consumes a request slot. Treating `remaining/limit` as a health signal and shedding load at 5% is the difference between graceful degradation and a cliff.

**⚠ Trap:** relying on the SDK's built-in retries as your rate-limit strategy. They default to 2 retries and they're per-call — they do nothing about the fact that your 40 workers are collectively over budget. Rate limiting is a *fleet-level* concern; you need a shared token bucket (Redis, exactly as you'd build for any upstream quota) that all workers draw from, keyed on tokens rather than requests.

### You're launching in three weeks and your projected load is 6× your current TPM ceiling. What do you do?

This is a capacity-planning question wearing an LLM costume, and the strong answer treats it like one: **compute the requirement, secure the headroom, and design the degradation path — in that order, in parallel, because the first one gates the sales conversation.**

**Step 1: compute the requirement precisely, in the provider's units.** Not "6× traffic." Peak RPM, peak input-TPM, peak output-TPM, computed at your *p95* prompt length, not the mean, and at your peak minute, not your daily average. A product with a 4× peak-to-mean ratio and a 20% weekly growth curve needs headroom over peak, not over mean:

```
peak_input_TPM  = peak_RPM × p95_input_tokens
peak_output_TPM = peak_RPM × p95_max_tokens        # remember: admission reserves max_tokens
required        = peak × (1 + growth_to_launch) × safety_factor(1.5–2×)
```

**Step 2: escalate the tier — this has a lead time, so start it today.** Provider tiers advance on cumulative spend and account age, and there is a manual path above the self-serve ceiling. Both take time. The concrete asks: current tier and its limits, the thresholds for the next tier, and whether they'll grant a temporary or custom limit for a dated launch. Providers do this routinely for a credible launch plan with numbers attached — bring the arithmetic from step 1, not a request for "more."

**Step 3: reduce demand, which is usually faster than raising supply.**
- **Cache the prefix.** Cached-read tokens still count against ITPM on some providers but the *latency* relief is immediate and the cost relief is large; verify the quota treatment for yours.
- **Cut `max_tokens` per route.** Frees output-TPM headroom immediately, at zero quality cost if your outputs are actually short.
- **Move anything with a soft deadline to the batch tier**, which has separate quota. Nightly evals, backfills, enrichment jobs — off the interactive quota entirely.
- **Route the head of the distribution down a tier.** If 60% of traffic is a task your small model passes, that's 60% of your peak TPM moved to a different quota pool.

**Step 4: design the degradation path before you need it.** Priority classes: paying-user interactive traffic never sheds; free-tier traffic sheds first; background enrichment sheds first of all. Implement it as a shared token bucket with per-class reservations, not as "retry until it works."

**Step 5: secure a second provider as capacity, not just as failover.** Two providers with independent quotas is 2× headroom, and the eval work to qualify the second one is work you should have done anyway.

**💰 Math:** peak 900 RPM, p95 input 6,000 tokens, `max_tokens` 2,000. Input-TPM = 900 × 6,000 = 5.4M. Output-TPM reserved = 900 × 2,000 = 1.8M. If your tier caps ITPM at 2M you are **2.7× over on input alone** — and the fix might be entirely free: a prefix cache that makes 5,000 of those 6,000 tokens cache-reads (if they don't count against ITPM on your provider) drops you to 900k, under the cap, with no tier change and no code beyond one marker. **Always compute whether you have a quota problem or a prompt problem before you escalate.**

**⚠ Trap:** discovering your limits during the launch. Load-test against the real provider at real peak *two weeks out*, in the region you'll serve from. A synthetic test against a mock proves nothing about admission control.

### Running requests across multiple accounts to get more throughput — smart ops or a terms violation?

Give the honest, bounded answer; interviewers are testing your judgment about a genuinely grey area, and both "always fine" and "never do it" are wrong.

**The bright line:** creating multiple accounts *to evade a rate limit that a single account would enforce* is circumvention, and every major provider's terms prohibit it. Doing it will, at best, get your accounts merged or suspended; at worst it ends an enterprise relationship. Do not build a rotation over free-tier keys and call it architecture.

**What is legitimate and is normal enterprise practice:**

- **Multiple workspaces or projects under one organization**, each with its own key and its own limits, provisioned by the provider. This is a feature, not a workaround — it exists so you can isolate prod from staging, or team A's budget from team B's, and get per-workspace attribution and quotas.
- **Multiple deployment surfaces for the same model.** First-party API, plus the same model through a cloud partner (Bedrock / Vertex / Foundry), have **separate quota pools** and separate commercial relationships. Running primary on one and burst on another is a supported architecture, and it's how large deployments actually get capacity. It's also your best availability story, since the failure domains are genuinely different.
- **Multiple providers.** Obviously fine, and the strongest form.
- **Multiple regions** where the provider offers regional endpoints with independent capacity.

**The engineering costs you must name, because they're what makes this a real design question rather than a hack:**

1. **Prompt caches don't follow you.** Caches are scoped to the account/deployment and the model. Spreading traffic across pools means each pool maintains its own cache, so your hit rate degrades roughly as `1/N` for the same traffic volume — and a 90% → 60% hit rate can cost you more than the burst capacity was worth. Route with **cache affinity**: hash on the cache key (tenant, or system-prompt version) so the same prefix consistently lands on the same pool.
2. **Feature parity is not guaranteed.** Cloud-partner deployments are feature-lagged: batch endpoints, certain server-side tools, caching semantics, and newest-model availability differ. Your abstraction must degrade, and your eval must run against each surface.
3. **Model identity drifts.** The "same" model on a partner platform is a differently-versioned snapshot with a different ID scheme. Pin explicitly and eval per surface.
4. **Cost attribution fragments.** You now have N invoices with different rate cards.

**🗣 Say this in the room:** "Multi-account to dodge a limit is circumvention and I wouldn't build it. Multi-*workspace* under one org, and the same model across first-party plus a cloud partner, are supported and are how you actually get headroom — those are separate quota pools with separate failure domains. The engineering cost is that prompt caches are pool-scoped, so I route with cache affinity rather than round-robin, or I hand back in cache misses more than I gained in capacity."

### It's 3am. Your primary provider is returning 429s and 529s across the board. Design the failover.

**🔍 Failure taxonomy first — because the correct response differs by class, and reflexive failover on the wrong class makes things worse:**

| Signal | Class | Correct response |
|---|---|---|
| 429 with `retry-after`, your `-remaining` near zero | **You** are over quota | Shed load by priority class, back off; do NOT fail over (you'll just move the overload) |
| 429 with `-remaining` healthy | Provider-side capacity | Retry with jitter; fail over if it persists past ~30s |
| 529 / overloaded | Provider-side capacity | Retry with jitter, then fail over |
| 5xx on a subset of requests | Partial degradation | Retry; circuit-break per model, not per provider |
| Latency up, error rate flat | Degradation, not outage | Do **not** fail over on latency alone — you may be seeing your own long prompts |
| 400s spiking | **Your** bug, or a contract change | Never retry. Alert. (e.g. a retention-config or model-deprecation 400) |

The design, in layers:

**Layer 1 — request-level retry with jitter, honoring `retry-after`.** Cap at 2–3 attempts. Beyond that you're amplifying the outage; this is the "just add a retry" trap that costs you 4× on a bad day.

**Layer 2 — circuit breaker per (provider, model).** Standard breaker, but the half-open probe should be a *cheap* request, not a real user's. Break on error rate over a rolling window, not on consecutive failures — LLM traffic is bursty enough that consecutive-failure counting trips spuriously.

**Layer 3 — the failover ladder, in preference order.** This is where the LLM-specific judgment lives:

1. **Same model, different deployment surface** (first-party → cloud partner). Zero quality change; you keep your evals valid. Cost: cache is cold on the new surface, and feature parity may be partial.
2. **Same family, smaller model.** Predictable quality drop that you have *measured*, because you ran your eval against the fallback tier before the incident.
3. **Different provider.** Largest quality delta and the one requiring the most prep — a prompt tuned for one model is not tuned for another, so a cross-provider fallback needs its own prompt variant and its own eval run, maintained.
4. **Degrade the feature.** Cached/canned response, "try again shortly," queue-and-notify. For many features this is *better* than a bad answer, and saying so is a senior signal.

**The three things people forget:**

- **Fail over with a quality budget, not blindly.** If your fallback scores 78% where primary scores 94%, silently serving it during a 4-hour incident may cause more damage than a clean error — especially for anything writing to a database or sending a message. Gate destructive actions behind the primary; let read-only paths degrade.
- **Emit a signal that says which model served the request**, and put it on every log line and every eval sample. Otherwise your Monday quality dashboard shows an unexplained dip and you spend a day rediscovering that Saturday was an incident.
- **Bound the fan-out cost.** Failover multiplies attempts. Cap total attempts per user task, not per API call, or one bad hour becomes a 4× invoice.

**💰 Math:** a naive "retry 5 times across 3 providers" policy during a 2-hour partial outage on a service doing 40k tasks/hour at $0.12/task. Base spend for the window: 80,000 × $0.12 = $9,600. If 30% of requests trigger the full ladder and average 3.5 billed attempts: 24,000 × $0.12 × 3.5 + 56,000 × $0.12 = $10,080 + $6,720 = **$16,800** — a 75% overspend on an incident where you served *fewer* successful requests than normal. Cap attempts per task and count them.

### How do you tell a quality regression from an availability problem, when a provider ships a change you didn't ask for?

Mental model: **availability failures are loud and instrumented; quality failures are silent and only visible against a baseline you built in advance.** If you don't have a continuously-running eval, you have no ability to detect this class of incident at all — you'll learn about it from a customer, three weeks late, and you won't be able to prove it.

The threat model, concretely:

- **A stable alias repoints to a new snapshot.** You pinned `some-model-latest` and the underlying weights changed. Your evals move; your logs show nothing.
- **A default flips.** Thinking goes from off-by-default to on-by-default, or `display` changes, or effort's default level changes. Behavior and cost both move with no code change on your side.
- **A safety classifier tightens.** Requests that used to succeed start returning refusals. This shows up as an *error* class you may not be handling — on some providers a refusal is HTTP 200 with `stop_reason: "refusal"`, so if your code reads `content[0]` unconditionally you get an IndexError, and if it doesn't, you get an empty answer counted as a success.
- **Tokenizer change on a model revision.** Same text, 1×–1.35× the tokens. Your context budgets and cost model silently shift.

The detection system, which is the actual answer:

1. **Pin exact model IDs, never aliases, in production.** Aliases are for prototypes. This is the single highest-leverage control and it costs nothing.
2. **Run a golden-set eval continuously** — 100–300 examples, hourly or on a cron, against production config, with results in your normal metrics system alongside latency and error rate. Alert on a shift beyond your CI.
3. **Log a response fingerprint on every request**: exact model ID returned by the API (not the one you requested — providers may substitute), `stop_reason`, `usage` breakdown, and system-prompt version hash. A shift in the *distribution* of `stop_reason` or mean `output_tokens` is a leading indicator that fires before your eval does, because it's computed on 100% of traffic instead of a sample.
4. **Alert on token-per-request drift.** Mean output tokens jumping 40% overnight with flat traffic is a default flip or a model swap, and it's visible within minutes.
5. **Keep a canary pinned to the previous snapshot** during a migration window so you can A/B rather than argue.

**⚠ Trap:** treating "the model got worse" as unfalsifiable and moving on. It's entirely falsifiable if you have a golden set with confidence intervals, and entirely unfalsifiable if you don't. The senior behavior is to have the artifact ready *before* the incident, and to be able to say "our golden set moved from 93.1% ± 1.8 to 87.4% ± 2.1 between the 3rd and 4th, here are the twelve examples that flipped."

**🗣 Say this in the room:** "Availability I detect from error rates; quality I can only detect against a baseline, so I run a golden-set eval on a cron into the same dashboard as p99 latency, and I log the model ID the API actually returned on every request. Aliases never go to production — a silently-updated alias is a deploy I didn't do, and pinning is free."

### Give me the open-weight landscape. Families, and where each is honestly competitive.

Mental model: **the open-weight question is never "is it as good as the frontier model" — it's "is it above the capability floor for this task class, at a cost structure I control."** For a large fraction of production tasks the answer has been yes for a while, and for a shrinking-but-real set of tasks the answer is still clearly no. Knowing *which set* is the signal.

**The families (📅 Volatile — new releases every few weeks; verify the current flagship in each line before your loop):**

- **Llama (Meta).** The ecosystem default: the best tooling support, the most fine-tuning recipes, the widest deployment coverage. Community license, not open source — see the license question below.
- **Qwen (Alibaba).** Consistently the strongest open weights on coding and multilingual work, and a very wide size ladder from sub-1B to very large MoE, which matters because it lets you pick a size for your latency budget rather than accepting whatever the family ships. Mostly Apache-2.0, with per-size exceptions.
- **DeepSeek.** MoE architectures with strong reasoning; notable for permissive licensing on weights (MIT on the R1 line) and for publishing genuine architectural contributions (MLA, auxiliary-loss-free load balancing) rather than just weights.
- **Mistral.** Strong European option, mixed licensing — some models Apache-2.0, others under a research-only license that forbids commercial use. Read the specific model card every time; this family is where people most often get the license wrong.
- **Gemma (Google).** Small, efficient, good for on-device and cost-sensitive classification. Custom terms with a prohibited-use policy, not OSI-approved.
- **Specialist lines:** embedding models, rerankers, and code-specific models where a small open model is routinely *better* than a frontier general model at the same task and 100× cheaper.

**The honest capability gap, by task class:**

| Task class | Open weights vs frontier API |
|---|---|
| Classification, routing, sentiment, extraction with a schema | **Parity.** Frontier is waste here. |
| Embeddings and reranking | **Open wins.** Cheap, fast, self-hostable, task-specific. |
| Summarization of a provided document | **Near parity** at mid size. Differences show up on long inputs. |
| Single-file code completion / infill | **Near parity** with a good code model. |
| Structured multi-step tool use in an agent | **Gap, and it's the big one.** Selection accuracy, argument fidelity, error recovery, and knowing when to stop all degrade noticeably. |
| Long-horizon agentic coding across a repo | **Clear frontier advantage.** |
| Hard multi-step reasoning, math, competition-style problems | **Gap narrowing but real**, especially in the reasoning-trained frontier tier. |
| Very long context with multi-fact synthesis | **Frontier advantage**, and the usable-vs-advertised gap is wider on open models. |
| Anything where you need a vendor to be accountable | **API wins** by definition. |

**🗣 Say this in the room:** "My default architecture is a frontier API for the agentic and reasoning-heavy path and open weights for the high-volume mechanical path — classification, routing, embeddings, reranking, extraction. That split usually moves 70–90% of *request volume* onto self-hosted models while leaving the hard 10% on the API, and it's where the cost curve actually bends. The place I don't reach for open weights is multi-step tool use, where the gap is still real."

**⚠ Trap:** benchmarking an open model against a frontier model on a leaderboard and concluding parity. Leaderboards over-represent the tasks the open community optimizes for, and contamination is a live problem. Run your own eval. And if you quote a leaderboard number in an interview without having run your own, expect the follow-up "and what did *you* measure?"

### Walk me through the license traps in open weights. Be specific.

**⚠ Trap: "open weights" is not "open source."** Most of the popular families ship under custom licenses that are not OSI-approved and carry conditions a normal open-source license does not. Getting this wrong is a legal exposure, not a style violation, and an interviewer at any company with a legal department is testing whether you know that.

The specific clauses, by family (**📅 Volatile — licenses change between model generations within the same family; read the model card for the exact checkpoint you are deploying**):

**Llama — Community License.**
- **The 700M MAU clause.** If, on the release date of the version you use, your products or affiliates' products have **more than 700 million monthly active users**, you must request a separate license from Meta, which they may grant or withhold at their discretion. This is a real gate for a handful of companies and a non-issue for everyone else — but you must be able to answer it, because "we don't know our MAU" is not an answer in a procurement review.
- **Naming and attribution.** Derivative models must carry "Llama" at the start of the name; you must display "Built with Llama."
- **Acceptable Use Policy** incorporated by reference, and it is enforceable.
- **Output usage** — this clause has *changed between generations*. Earlier Llama licenses restricted using model outputs to improve other large language models; a later generation relaxed it, subject to the naming requirement. **Do not answer this from memory for a specific version; read that version's license text.** Being able to say "this clause changed between versions, so I'd check the exact checkpoint" is the correct answer.

**Qwen.** Most sizes ship Apache-2.0, which is genuinely permissive — but not uniformly across every size in every generation. Check the specific model card.

**Mistral.** Split licensing: several models are Apache-2.0; others ship under a research license that **prohibits commercial use** outright. This is the family where I've seen the most expensive mistakes, because "Mistral is Apache" is a widely-believed half-truth.

**Gemma.** Custom Terms of Use plus a Prohibited Use Policy. You must pass the terms through to downstream recipients, and Google reserves the ability to restrict use. Not OSI-approved.

**DeepSeek.** MIT on the R1-line weights — about as permissive as it gets. Note that permissive weight licensing does not automatically mean permissive *training-data* provenance, which is a separate question your legal team may ask.

**The cross-cutting trap — anti-distillation clauses on the API side.** This is separate from weight licenses and catches more people. Frontier API providers' terms generally prohibit using their outputs to develop a **competing model**. So the pipeline "call the frontier API on 500k prompts → fine-tune an open model on the outputs → ship it" has two independent legal questions: does the open model's license allow the derivative, and does the API provider's ToS allow you to use its outputs that way. The second one is where teams get surprised, and the answer is genuinely fact-specific: distilling to serve *your own* product's task is a different posture from distilling to ship a general-purpose model that competes with the provider.

**🗣 Say this in the room:** "I treat weight licenses as a procurement input, not a footnote. The three things I check on every checkpoint: is it actually OSI-approved or a custom license; is there a scale or field-of-use gate like Llama's 700-million-MAU clause; and are there naming, attribution, or output-usage conditions that follow the derivative downstream. And separately, if training data came from a frontier API, I check that provider's terms on using outputs to develop models — that's a different agreement and it's the one people forget."

### Legal comes to you: "Can we fine-tune Llama on our customer data and sell the resulting model as a product?" Answer them.

Answer in four separable questions, because conflating them is how this goes wrong. The interviewer is testing structured thinking under a legal-flavored question, not asking you to practice law — and you should say that out loud.

**1. Does the base model's license permit a commercial derivative?** For a Llama community-license model, generally yes, subject to: the MAU gate (check yours as of the version's release date), the naming requirement (your product's model name must begin with "Llama"), the "Built with Llama" attribution, passing the license and Acceptable Use Policy through to whoever you distribute to, and including the required notice file. So "sell it" is permitted but *conditioned*, and one of the conditions constrains your product branding — which is a real product decision, not a legal footnote, and worth raising early.

**2. Does your customer data permit this use?** Almost always the harder question. Your DPA and privacy policy have to actually authorize using customer content to train a model, and "improve our services" is frequently *not* read as covering "bake it into weights we sell to third parties." Then: is the data personal data under GDPR/CCPA, and if so what's your lawful basis, and how do you honor a deletion request against a model that has already memorized it? **Model weights are not a database you can DELETE FROM.** The engineering answer is to make it never enter the weights: per-tenant retrieval instead of per-tenant fine-tuning, PII stripping and canonicalization in the training pipeline, and a documented retention window on the training corpus.

**3. Does the training data have third-party provenance issues?** If any of the fine-tuning data is outputs from a frontier API, that provider's terms on using outputs to develop models applies independently of the Llama license — see the previous question. If it's scraped, that's its own analysis.

**4. What are you actually shipping?** Selling weights, offering it as a hosted service, and embedding it in your product are three different distribution modes with different obligations under the same license. Nail this down before answering anything else, because the naming and pass-through conditions bite hardest when you distribute weights.

**The engineering counter-proposal you should have ready:** in most cases where someone asks this, fine-tuning is not the right mechanism anyway. If the goal is "the model knows our customers' data," retrieval gives you per-tenant isolation, instant updates, an actual deletion story, and no license question about the derivative. If the goal is "the model follows our format and tone," a small fine-tune on *synthetic or internal* data — not customer data — usually gets you there. **Raising that alternative unprompted is the strongest thing you can do with this question**, because it shows you understand fine-tuning is the last rung of the escalation ladder rather than the first.

**🗣 Say this in the room:** "Four questions, and I'd want counsel on the middle two: does the base license allow a commercial derivative and under what conditions; does our DPA cover training on customer data and how do we honor deletion against weights; is there third-party provenance in the training set; and are we distributing weights or hosting a service. But before any of that — I'd push back on the design. If the requirement is per-customer knowledge, retrieval gives us tenant isolation, immediate updates, and a real deletion story, and it sidesteps the entire question."

### Build me the self-host versus API crossover model. Where's the break-even and what did you leave out?

Mental model: **API pricing is fully variable cost; self-hosting is mostly fixed cost with a utilization multiplier.** So the crossover is entirely a function of *sustained* utilization, and the number that kills self-hosting projects is never the GPU price — it's the fraction of the day those GPUs are idle.

**The API side** is trivial: `tokens × rate`, already built in the cost-function question.

**The self-host side**, and the discipline is in enumerating every term:

```
monthly_cost = GPUs × hours × $/GPU-hr        # compute
             + engineer_FTE_fraction × loaded_cost   # the term everyone omits
             + observability + load-balancing + storage + egress
             + idle_capacity_you_provisioned_for_peak
```

Then the throughput side, which decides how many GPUs you need:

```
tokens/sec/GPU  ← measured, at YOUR sequence lengths and YOUR batch size
sustainable_load = tokens/sec/GPU × GPUs × utilization_fraction
```

**💰 Math, worked.** Suppose 3B output tokens/month on a mid-size open model, and your measured serving throughput is 2,500 output tokens/sec/GPU at your batch depth on an 80GB card.

- Seconds of GPU time needed: 3e9 / 2,500 = **1.2M GPU-seconds** = 333 GPU-hours.
- At 100% utilization that's 333/730 ≈ **0.46 GPUs**. But you can't run at 100% — traffic is peaky. At a realistic 30% average utilization you need ~1.5 GPUs of capacity, and for redundancy you deploy 2 (or 4, if the model needs TP=2 and you want two replicas).
- 2 GPUs × 730 hours × $2.50/GPU-hr (**📅 Volatile**) = **$3,650/month** of compute.
- Plus 0.25 FTE at a $300k loaded cost = **$6,250/month**.
- Plus observability, LB, storage, on-call: call it **$1,000/month**.
- **Self-host total: ~$10,900/month.**

API side for the same 3B output tokens, assuming a comparable-tier hosted model at $5/Mtok output plus, say, 6B input tokens at $1/Mtok: (3,000 × $5) + (6,000 × $1) = $15,000 + $6,000 = **$21,000/month**.

So at this volume self-hosting wins by roughly 2×. Now change one number: drop the volume to 300M output tokens/month. API side falls to ~$2,100/month. Self-host side barely moves — you still need the GPUs for peak, you still need the engineer — call it $9,000. **Self-hosting is now 4× more expensive.** That's the whole shape: the API line is linear through the origin, the self-host line has a large intercept, and the crossover for a small open model against a small hosted model tends to sit somewhere in the high hundreds of millions to low billions of tokens per month.

**What people leave out, in order of how much it costs them:**

1. **Engineering time.** Always the largest term below ~1B tokens/month, and always the one omitted from the spreadsheet that gets self-hosting approved.
2. **Utilization.** Everyone models 100%. Real interactive traffic runs 20–40% average against peak-provisioned capacity. That's a 2.5–5× multiplier on your compute line.
3. **The quality delta.** If the open model needs 1.4× the output tokens to reach the same answer quality, or fails 6% more often and triggers a retry or a human, that's a cost you must add. Cost per *resolved task*, not per token.
4. **Peak headroom and redundancy.** One replica is not a deployment.
5. **Model upgrades.** The API upgrades for free; self-hosting means you own re-evaluating, re-tuning, and re-deploying every time a better checkpoint lands.

**🗣 Say this in the room:** "The crossover is set by sustained utilization, not by the GPU price. Below roughly a billion output tokens a month, the loaded cost of the engineer who owns the deployment dominates every other term, and the API wins even at a 5× per-token premium. Above it, self-hosting wins — but only for a workload with steady, predictable load and a task where the open model actually clears my quality floor. Bursty traffic on self-hosted GPUs is the worst of both worlds."

### Distillation from a frontier model into a small open model — what's the technical ceiling, and what's the legal ceiling?

**The technique.** You use a strong model to generate a training set — labels, reasoning traces, tool trajectories, preference pairs — and fine-tune a small open model on it. Two flavors: *hard-label* distillation (train on the teacher's sampled outputs, which is just supervised fine-tuning on synthetic data) and *soft-label* distillation (train on the teacher's full output distribution, minimizing KL to the teacher). Soft-label is more sample-efficient but requires logit access, which frontier APIs don't provide — so in practice API distillation is hard-label SFT on generated data.

**The technical ceiling, stated honestly:**

- **It works extremely well for narrow, well-specified tasks.** A 7B model distilled on 50k teacher-generated examples of one classification or extraction task routinely matches the teacher on that task, at 1–2% of the cost. This is the highest-ROI move in applied LLM engineering and it is underused.
- **It works poorly for general capability.** You are transferring *behavior on your data distribution*, not intelligence. Off-distribution the student falls off a cliff, and it falls off silently — the failure mode is confident wrongness on inputs your distillation set didn't cover.
- **The student inherits the teacher's errors and none of its uncertainty.** If the teacher is 94% accurate, 6% of your training labels are wrong, and the student learns them as ground truth. Filtering matters enormously: use the teacher's own agreement across samples, a verifier, or a second model as a judge to drop low-confidence labels. Distilling unfiltered teacher output is the most common reason a distillation project underdelivers.
- **Reasoning traces transfer better than you'd expect but need verification.** Training on teacher chains-of-thought helps — but only when you filter to traces whose *final answer was correct*. Traces with correct answers and broken reasoning teach broken reasoning.
- **Data quality beats data volume.** 50k well-filtered, diverse, on-distribution examples beats 500k unfiltered ones, consistently.

**The legal ceiling:** frontier providers' terms generally prohibit using outputs to develop a **competing** model. That word is load-bearing and the analysis is fact-specific. Distilling a task-specific classifier that runs inside your product is a materially different posture from training and releasing a general-purpose assistant. Separately, some open-weight licenses have their own output-usage clauses that have changed between generations. And if you plan to release the student's weights, you inherit both sets of conditions. **Get counsel; don't reason your way to a conclusion from the license text alone, and say so in the room** — knowing where the boundary of your competence is reads as senior, not evasive.

**💰 Math:** an extraction task at 20M requests/month, 1,200 in / 150 out. Frontier at $3/$15: (20e6 × 1200/1e6 × $3) + (20e6 × 150/1e6 × $15) = $72,000 + $45,000 = **$117,000/month**. Distillation project: 80k teacher-generated examples at ~$0.008 each = $640 one-time, plus ~$800 of fine-tuning compute, plus two engineer-weeks. Serving the 7B student self-hosted at, say, $4,500/month of GPU including utilization slack. **Payback on the one-time cost is under a day; the run-rate saving is ~$112k/month.** This is the single most compelling cost argument in applied LLM engineering — and the reason it isn't done everywhere is that it requires an eval good enough to prove the student is safe to ship, which most teams don't have.

**⚠ Trap:** distilling before you have a stable prompt and a trustworthy eval. You'd be freezing today's prompt into weights. Distill last, after the escalation ladder — prompt, context, retrieval, tools, structured output, routing — has been exhausted and the task has stopped moving.

### For each of these — a support-ticket router, a code-review assistant, and a legal-document Q&A — which model would you pick and why?

Answer each with the same shape: capability floor first, then the constraint that actually decides it. Interviewers use this format to test whether your framework survives contact with concrete cases.

**Support-ticket router** (classify into ~40 categories, extract 6 fields, route). High volume, low difficulty, latency matters only in aggregate.

- Floor: a small hosted model almost certainly clears it; a fine-tuned open model of 7–8B parameters clears it after distillation.
- Deciding constraint: **cost at volume**. At 20M/month this is exactly the distillation case from the previous question — $117k/month on frontier versus ~$5k self-hosted, with a two-week project.
- My pick: **small hosted tier to launch and to generate the distillation set; open-weight fine-tune once the taxonomy is stable.** Ship the escalation path too: anything the classifier scores below a confidence threshold goes to a bigger model, then to a human. That confidence-routed cascade is what makes the cheap tier safe.
- Structured output with a strict schema is mandatory here, and cheap.

**Code-review assistant** (comment on a diff, flag real bugs, don't nitpick).

- Floor: this is genuinely hard. Real-bug recall on unfamiliar code is where the open/frontier gap is widest, and it's a task where being wrong is expensive in a specific way: false positives train reviewers to ignore the tool, which kills adoption permanently.
- Deciding constraint: **precision and recall on real bugs**, and the fact that latency is soft — a review comment 40 seconds after the push is fine.
- My pick: **frontier reasoning-tier model, higher effort setting**, because latency is soft and the cost per review is bounded by diff size, not by traffic. A code review at 15k input tokens and 1,500 output tokens costs (15,000 × $5 + 1,500 × $25)/1e6 = $0.1125. At 3,000 reviews/day that's $338/day — **$10k/month**, trivially justified against engineer time.
- The non-obvious part: recent models follow severity filters *literally*, so a prompt saying "only report high-severity issues" makes measured recall drop even when bug-finding improved. Have it report everything with a confidence and severity, and filter downstream.

**Legal-document Q&A** (answer questions over a 400-page contract set, with citations).

- Floor: high, because the failure mode is a confident wrong answer about a contractual obligation.
- Deciding constraint: **grounding and citation fidelity**, then **data residency and retention**, then context.
- My pick: **retrieval plus a frontier model with native citation support**, not a long-context stuff. Two reasons: cost (400 pages ≈ 300k tokens × $3/Mtok = $0.90 per question versus ~$0.02 with retrieval), and quality (multi-fact synthesis over 300k tokens is exactly where usable-context falls short of advertised-context). Native citations turn "is this grounded?" from a judgment call into a check.
- Retention and residency likely decide the provider before anything else — this is the archetype of the case where legal picks your deployment surface and you eval whatever is left.

**🗣 Say this in the room:** "Router: cheap tier, strict schema, confidence-routed escalation, and distill once the taxonomy stops moving — it's a volume problem. Code review: frontier reasoning tier at high effort, because latency is soft and a false positive permanently costs adoption. Legal Q&A: retrieval plus native citations rather than long-context stuffing, and residency probably picks my provider before I run a single eval."
### Token prices have fallen roughly 80% year over year and enterprise LLM spend has roughly doubled over the same period. Reconcile those.

Mental model: **this is Jevons' paradox with a two-week feedback loop.** When the unit cost of a capability collapses, the set of economically viable applications expands faster than the price falls, so aggregate consumption — and aggregate spend — rises. You have already lived this: cloud storage got 10× cheaper per GB and your storage bill went up, because at the new price you started keeping things you used to delete.

The mechanism specific to LLMs has three compounding legs, and naming all three is what separates an economics-flavored answer from a real one:

**Leg 1 — the capability floor moves down through the price tiers.** A task that only the frontier tier could do last year is done by the cheap tier this year. That's a per-request price cut. But it also means the task is now *worth automating at all* — a workflow that cost $0.40/item and saved $0.35 of human time was a non-starter; at $0.04/item it's a 9× ROI and gets built. **Price cuts don't just reduce the bill for existing work; they convert non-customers into customers.**

**Leg 2 — the architecture changed from one call to many.** The single biggest driver of enterprise spend growth is not traffic, it's **tokens per user-visible task**. A 2023 chat feature was one call, ~1,500 tokens. A 2026 agentic feature on the same user request is 15–40 calls with a growing transcript, tool results, and reasoning tokens — routinely 200k–2M tokens for one task. That's a 100–1000× increase in tokens per task, against an 80% price cut. Do the arithmetic: 0.2 × 300 = **60× more spend per task**, and this is the whole story.

**Leg 3 — reasoning tokens are a new line item that didn't exist.** They bill at the output rate, they're invisible in the response, and effort settings that improve quality multiply them.

**📐 Numbers you must know:** the orders of magnitude, derived rather than quoted. A 2023 single-call chat feature: **~1.5k tokens per user-visible task**. A 2026 agentic feature: 15–40 calls × a transcript that grows to tens of thousands of tokens plus reasoning = **200k–2M tokens per task**, i.e. **10²–10³×**. Against a roughly 5× per-token price cut over the same window, that is a net 20–200× increase in spend per task. Whenever someone tells you inference "got cheap," those two exponents are the counter-argument.

**💰 Math, so the shape is undeniable.** Feature X, 2024: 500k tasks/month × 1 call × (2,000 in + 400 out) at $10/$30 per Mtok = 500,000 × ($0.020 + $0.012) = **$16,000/month**. Same feature, 2026, rebuilt as an agent: 500k tasks × 18 calls × (average 9,000 cached-in + 700 out) at $3/$15 with a 90% cache-read rate. Input: 18 × 9,000 = 162,000 tokens/task; blended input rate ≈ 0.9 × $0.30 + 0.1 × $3.75 = $0.645/Mtok → 162,000 × $0.645/1e6 = $0.1045. Output: 18 × 700 = 12,600 × $15/1e6 = $0.189. Per task ≈ $0.294 → **$147,000/month**. The unit price fell ~70%, the quality is far higher, and the bill went up **9×**.

**What this means for you as an engineer, which is the part interviewers actually want:**

1. **Never forecast next year's budget by extrapolating this year's tokens at next year's prices.** You will be wrong by an order of magnitude in the expensive direction, because architecture changes dominate price changes.
2. **The unit that matters is cost per resolved task**, and it must be tracked as a first-class metric with a target, because it's the only quantity that stays comparable across an architecture change.
3. **Price cuts should be spent deliberately.** When a tier gets 60% cheaper you have a choice: bank it, or buy more quality (higher effort, more retrieval, a verification pass). Make that a decision with a number attached, not something that happens to you.

**🗣 Say this in the room:** "Prices per token fell ~80%, and on the workloads we rebuilt as agent loops with reasoning, tokens per task rose two to three orders of magnitude — so those workloads got 20–200× more expensive per task even as the unit price collapsed. Only a slice of traffic made that jump, which is why the aggregate bill roughly doubled rather than going up 50×; but that slice is where all the growth is. So I don't track cost per token or cost per call — I track cost per resolved task, with a target, because that's the only unit that survives an architecture change and it's the one the business can price against."

### Your LLM bill tripled month over month. Traffic is flat. Debug it.

**🔍 Failure taxonomy, ordered by frequency, each with the diagnostic that confirms or eliminates it in one query.**

**1. Prefix cache stopped hitting.** Diagnostic: `sum(cache_read) / sum(cache_read + cache_write + input_uncached)` over time. If it fell, run the cache-debug procedure — hash the rendered prefix, look for a moved model alias, a dynamic tool set, a config-injected value, or traffic getting sparse enough to fall outside the TTL. This is the single most common cause and it can 4× your bill with zero code change.

**2. Output tokens per request grew.** Diagnostic: `mean(output_tokens)` by route, over time. Causes: a default flip that turned thinking on; an effort setting raised; a prompt change that removed a length constraint; a model upgrade that's simply more verbose. Output bills at 5× input, so a 2× output increase is often the entire delta.

**3. Turn count per task grew.** Diagnostic: `mean(api_calls_per_task)` — which requires you to have a task/trace ID on every call. If you don't, add it now; without it you cannot distinguish "each call got more expensive" from "we make more calls," and those have opposite fixes. Causes: a tool that started failing and triggering retries; a new tool that confuses selection; a loop cap raised; results split across messages killing parallel tool calls.

**4. Input length grew.** Diagnostic: `p50/p95(total_prompt_tokens)`, remembering that total = uncached + cache_read + cache_write, not `input_tokens` alone. Causes: retrieval `k` raised; a chunker change; conversation history not being trimmed; a document type that started arriving larger.

**5. Traffic mix shifted.** "Flat traffic" usually means flat request count. Diagnostic: spend by route and by tenant. One enterprise customer onboarding onto the expensive path can triple the bill while total requests move 3%.

**6. Retry / failover amplification.** Diagnostic: attempts per task, and spend attributable to non-first attempts. A provider degradation last month plus an uncapped retry ladder is a classic tripling.

**7. Someone changed the model.** Diagnostic: spend by the model ID *the API returned*. A route silently repointed from a $1/$5 tier to a $5/$25 tier is a 5× on that route.

**8. A batch job moved onto the sync path**, or a sync job lost its batch discount.

**The instrumentation that makes this a 20-minute investigation instead of a two-day one** — and the correct answer to "how would you have caught this sooner":

```python
# One structured log line per API call. Non-negotiable fields.
log.info("llm_call", extra={
    "trace_id": trace_id, "task_id": task_id, "route": route, "tenant_id": tenant,
    "model_requested": req_model, "model_served": resp.model,   # these can differ
    "prompt_version": prompt_hash, "attempt": attempt,
    "in_uncached": u.input_tokens, "cache_read": u.cache_read_input_tokens,
    "cache_write": u.cache_creation_input_tokens, "out": u.output_tokens,
    "stop_reason": resp.stop_reason, "cost_usd": cost(u, rates),
})
```

With that, every hypothesis above is a `GROUP BY` away. Without it, you are reading an invoice.

**⚠ Trap:** starting from the invoice. The provider bill is aggregated by model and day; it cannot tell you which route, tenant, or prompt version moved. **Cost must be attributed at the call site, in your own telemetry, at emit time** — exactly as you'd never debug a database cost problem from the RDS bill alone.

**🗣 Say this in the room:** "First question: did cost per call go up, or calls per task? Those have completely different fixes. I'd check cache hit rate, mean output tokens, and calls-per-task in that order — those three explain the large majority of surprise tripling, and all three are single queries if you log the usage breakdown with a trace ID at the call site."

### Design per-tenant cost attribution and budget enforcement for a multi-tenant AI product.

Mental model: **this is a metering and quota system, and you have built one before — the only genuinely new part is that the unit of consumption is not knowable until after the work is done.** You can't price the request at admission, because the model decides how many output and reasoning tokens to spend. So the standard "reserve, consume, settle" pattern applies, with a pessimistic reservation and a settlement.

**Layer 1 — attribution.** Every API call carries `tenant_id`, `user_id`, `feature`, `task_id`, `attempt`. Cost is computed at the call site from the response's `usage` object and the rate card for the exact model served. Emit as a structured event; aggregate in your warehouse. Two rules:

- **Attribute at the task level, not the call level.** A tenant consumed one "resolved ticket," which cost you 22 API calls. The task is what you bill or budget against; the calls are implementation detail.
- **Attribute failures and retries too.** A tenant whose data causes 40% tool failures costs you 1.6× the tokens for the same delivered value, and if your attribution only counts successes you will misprice them permanently.

**Layer 2 — enforcement.** A distributed token bucket per tenant, in Redis, keyed on **dollars or tokens per period**, not on requests. The reserve/settle shape:

```python
# Admission: reserve a pessimistic upper bound.
# Rates are USD per 1M tokens — divide, or your reservation is 1e6× the settled cost.
est = (est_input_tokens * rates.input_uncached
       + max_tokens * rates.output) / 1_000_000
if not bucket.try_reserve(tenant, est):          # atomic Lua: check + debit
    raise BudgetExceeded(tenant)
try:
    resp = client.messages.create(..., max_tokens=max_tokens)
finally:
    actual = request_cost_usd(resp.usage, rates)
    bucket.settle(tenant, reserved=est, actual=actual)   # refund the difference
```

The reservation must be pessimistic (use `max_tokens`, not your guess) or a tenant can overrun by the ratio of actual to estimated. This is exactly the provider's own admission-control mechanism, and building it yourself for the same reason is a good thing to point out.

**Layer 3 — degradation, not rejection.** A hard 429 at the budget line is a bad product. The ladder: at 80% of budget, route to a cheaper tier and lower effort; at 95%, disable the expensive optional features (deep research, multi-agent fan-out); at 100%, queue non-interactive work and serve interactive work with a clear notice. Each step needs a measured quality delta so you know what you're spending in accuracy to save in dollars.

**Layer 4 — the controls that stop the bleeding.** Per-tenant caps on: `max_tokens` per call, turns per task, tool calls per task, retries per task, and concurrent tasks. Every one of those is a runaway-cost vector; a turn cap alone prevents the worst incident class.

**Layer 5 — reconciliation.** Nightly, sum your per-call attributed cost and compare to the provider's reported usage. Alert if they diverge by more than ~2%. Divergence means dropped telemetry, a rate card you didn't update, or traffic from a path you don't instrument — all three of which you want to know about before the invoice.

**💰 Math on why enforcement is not optional:** an agent with no turn cap on a task the tool set can't complete. 40 turns × 8,000 output tokens... realistically it loops: 200 turns × 800 output tokens = 160,000 output tokens at $15/Mtok = **$2.40 for one stuck task**. One tenant with a malformed integration generating 3,000 such tasks a day = **$7,200/day**. A turn cap of 25 bounds it at $0.30/task and $900/day, and a duplicate-call detector cuts it further. The cap is four lines of code.

**⚠ Trap:** building attribution on estimated tokens because "we need the number before the call." You need an *estimate* before the call for reservation, and the *actual* after for accounting. Conflating them gives you a ledger that drifts from the invoice and a finance team that stops trusting your dashboards.

### MCP has gone through several spec revisions. What changed, and why does the revision date matter operationally?

Mental model: **MCP is a wire protocol with dated revisions, and "we support MCP" is as meaningless a statement as "we support HTTP" without a version.** The revision string is the compatibility contract between your client and every server you connect to, and it has changed in ways that are not backwards-compatible.

**The revision history and what each changed (📅 Volatile — verify the current revision and any newer ones before your loop):**

- **2024-11-05** — the initial specification. JSON-RPC 2.0 over stdio or HTTP+SSE (two endpoints: one SSE stream for server→client, one POST endpoint for client→server). Core primitives: tools, resources, prompts.
- **2025-03-26** — the significant one. Replaced the two-endpoint HTTP+SSE transport with **Streamable HTTP** (a single endpoint that can optionally upgrade to SSE), which made MCP servers deployable behind ordinary infrastructure — stateless, load-balanced, serverless — rather than requiring a sticky long-lived connection. Added a comprehensive **OAuth 2.1-based authorization framework**, tool annotations, and audio content.
- **2025-06-18** — **removed JSON-RPC batching** (a genuine breaking change; clients that batched must stop), added **structured tool output**, **elicitation** (servers can request additional input from the user mid-call), and resource links. Formalized MCP servers as OAuth **resource servers** and required clients to send resource indicators so a token issued for one server can't be replayed at another. Made the `MCP-Protocol-Version` header required on HTTP transport.

**Why the date matters operationally, which is the actual question:**

1. **Transport compatibility.** A client that only speaks the original HTTP+SSE transport cannot talk to a Streamable-HTTP-only server, and vice versa. This is the number-one integration failure and it presents as a connection that hangs rather than an error.
2. **Auth model.** Pre-2025-03-26 there was no standardized authorization story, so servers rolled their own. Post-2025-06-18 there's a resource-indicator requirement designed specifically to prevent token-replay across servers. If you're integrating a third-party MCP server, "which revision do you implement" determines whether your security review passes.
3. **Breaking removals.** Batching removal means a client written against the earlier spec fails against a newer server, silently or loudly depending on implementation quality.
4. **Header negotiation.** The required version header on HTTP transport is how mismatches surface as a clean error instead of a mysterious hang — which is exactly why you should send it and reject servers that don't.

**The security posture you should raise unprompted**, because it's the thing product companies care about: an MCP server is **remote code you are granting tool access to your agent**. The risks are (a) a malicious or compromised server returning tool results crafted as prompt injections that redirect the agent, (b) token scope over-grant, (c) tool-name shadowing where a hostile server registers a tool name your agent trusts, and (d) supply chain — you're pulling a server someone else maintains. Mitigations: pin server versions, allowlist servers, keep credentials out of the sandbox (inject them at egress rather than handing them to the model's environment), treat every tool result as untrusted input, and require human confirmation for irreversible actions.

**🗣 Say this in the room:** "I'd never say 'we support MCP' without a revision date — the transport changed from dual-endpoint SSE to Streamable HTTP, batching was removed, and the authorization model went from unspecified to OAuth 2.1 with resource indicators. Those are hard compatibility boundaries. And operationally I treat every MCP server as untrusted remote code: allowlisted, version-pinned, credentials injected at egress rather than into the sandbox, and tool results treated as adversarial input."

### The EU AI Act — what actually applies to you as an applied AI engineer, and on what timeline?

Mental model: **the Act is risk-tiered and role-based. Your obligations are determined by two questions: what risk tier is your use case, and are you a *provider* or a *deployer* of the system?** Most product engineers are deployers of a general-purpose model inside a specific application, and that combination usually lands them in the transparency tier rather than the high-risk tier — but the exceptions are the ones that end careers.

**The tiers:**

- **Prohibited** — social scoring, certain biometric categorization, emotion inference in workplaces and education, untargeted facial-image scraping, some manipulative techniques. Not "regulated," *banned*.
- **High-risk** — Annex III use cases: employment and worker management (CV screening!), education access, essential private and public services including credit scoring, law enforcement, migration, justice, and critical infrastructure; plus AI as a safety component of already-regulated products. Obligations are heavy: risk management system, data governance, technical documentation, logging, human oversight, accuracy/robustness/cybersecurity, conformity assessment, registration.
- **Limited risk / transparency** — disclose that a user is interacting with an AI; label synthetic content and deepfakes.
- **Minimal risk** — most things. No specific obligations.
- **GPAI models** — a separate track with obligations on the model provider (technical documentation, copyright policy, training-content summary), plus additional systemic-risk obligations above a compute threshold.

**The milestone calendar (📅 Volatile — and genuinely contested; there has been active legislative discussion about delaying parts of the high-risk timeline. Verify the current state before your loop; do not assert a date you haven't checked this month):**

| Date | What applies |
|---|---|
| 1 Aug 2024 | Act enters into force |
| 2 Feb 2025 | Prohibited practices; AI-literacy obligations |
| 2 Aug 2025 | GPAI model obligations; governance structures; most penalty provisions |
| **2 Aug 2026** | **General applicability — including Annex III high-risk obligations and the Article 50 transparency obligations** |
| 2 Aug 2027 | High-risk as a safety component of regulated products; GPAI models placed on market before Aug 2025 must be brought into compliance |

Penalties scale to a percentage of global annual turnover, with the highest band for prohibited practices — which is why this gets executive attention rather than being a docs-page problem.

**What this means for your engineering work, concretely — this is the part that distinguishes a useful answer from a Wikipedia recital:**

1. **Classify the use case in the design doc, not at launch.** "Are we in Annex III?" is a five-minute question at design time and a three-month remediation at launch. Résumé screening, credit decisioning, and anything gating access to a service are the ones that surprise people.
2. **Transparency obligations are engineering work.** Disclosing AI interaction and marking synthetic content mean UI changes and, for generated media, machine-readable provenance marking. Budget for it.
3. **Logging and traceability are Act requirements for high-risk, and they're the same artifacts you want anyway** — trace IDs, model version, prompt version, inputs, outputs, and human-override records, retained. You are already building this for cost attribution and eval; scope it once.
4. **Human oversight must be real.** A human who rubber-stamps 400 decisions an hour is not oversight. It has to be a person with the information, the authority, and the time to override.
5. **The GPAI obligations mostly land on your model provider, and you inherit their documentation** — which is why "which provider" becomes a compliance question as well as a technical one.

**🗣 Say this in the room:** "As a deployer of a general-purpose model in a product, I'm usually in the transparency tier — disclose AI interaction, mark synthetic content. What I check first, at design time, is whether the use case falls in Annex III, because employment screening and credit decisioning are high-risk and that's a completely different obligation set: risk management, data governance, conformity assessment, real human oversight. The dates have been in motion legislatively, so I'd verify the current calendar rather than quote one from memory."

### A model you depend on is being retired in 60 days. Run the migration.

Mental model: **treat it as a dependency upgrade with a hard deadline and an unknown-magnitude behavioral change — closer to a database major-version upgrade than a library bump.** The specific hazard is that the API contract may be identical while the *behavior* is not, so your type checker and your integration tests both pass and your quality moves.

**Week 1 — inventory and classify.** Grep for the model ID across the repo and classify every hit, because the right action differs:

| Hit type | Action |
|---|---|
| Actual API call sites | Swap the ID **and** apply the breaking-change checklist |
| Model registries, routing configs, pricing catalogs, OpenAPI specs | The old entry may need to *stay* (the model is still served elsewhere). Add the new one alongside; never blind-replace |
| Capability gates (`if "opus-4" in model_id:`) | **Add** the new ID; don't replace, or you disable a feature for remaining old-model traffic |
| Test fixtures, seed data, registry assertions | Add alongside; verify the definer has an entry for the new model first |
| Suffixed variants (`-fast`, `-1024k`, dated snapshots) | These are deployment identifiers. Verify a new-model equivalent exists before assuming |

This classification step is the one people skip, and it's the one that causes "we migrated and prod broke in an unrelated service."

**Week 1–2 — read the migration guide and enumerate the breaking changes.** These are real and they 400: thinking-configuration parameters removed, sampling parameters (`temperature`/`top_p`/`top_k`) rejected on some newer models, assistant-turn prefills rejected, tool type/name pairs updated, parameter renames like `output_format` → `output_config.format`. Also enumerate the *silent* changes: default flips on thinking, tokenizer changes shifting token counts by a factor of 1×–1.35×, changed defaults on reasoning visibility.

**Week 2–3 — re-baseline the numbers.** Re-run `count_tokens` against the new model on representative prompts. A tokenizer change means your `max_tokens` ceilings, context budgets, compaction triggers, and cost model are all off. **Do not apply a blanket multiplier**; measure.

**Week 3–4 — run the eval, both models, side by side.** Same eval set, same prompts. Report accuracy with confidence intervals, mean output tokens, p95 latency, and cost per resolved task. This is the artifact that says whether you can ship.

**Week 4–6 — adapt prompts.** Expect this to be needed and budget for it. Recent-generation models follow instructions more literally, calibrate verbosity to task complexity, and differ in tool-use eagerness and self-verification behavior. Concretely: prompts written to *overcome* an older model's reluctance ("CRITICAL: you MUST use this tool") over-trigger on a newer one; verification scaffolding you added ("double-check your work") can cause over-verification; severity filters in review prompts get followed more literally and depress measured recall.

**Week 6–8 — canary and cut over.** Route 5% → 25% → 100%, with the golden-set eval and cost-per-task on the dashboard at each step, and the old model still available for rollback until the retirement date passes.

**⚠ Trap:** doing the model-ID swap in week one and declaring the migration done because CI is green. Your test suite asserts shapes; the regression is in quality and cost, and neither is in your test suite unless you put an eval there. **The deliverable of a model migration is an eval comparison table, not a diff.**

**🗣 Say this in the room:** "I'd classify every hit on that model ID first — call sites get migrated, but registries and capability gates get the new model *added*, not substituted, or I break traffic that's still on the old one. Then breaking changes, then re-baseline token counts because the tokenizer probably moved, then a side-by-side eval with confidence intervals, then prompt adaptation, then a canary. The artifact I'd bring to review is the eval table, not the PR."

### Design a router across a cheap and an expensive model. Show me when it pays.

Mental model: **a router is a cascade with a confidence threshold, and its economics are governed by one number — the fraction of traffic the cheap model handles correctly — and one hazard: the cost of the escalation itself.** If escalation means re-running the whole task on the expensive model, you pay for both, and the router only pays if the cheap model handles a large majority.

**The three routing architectures, in increasing order of how much they actually work:**

1. **Pre-classification.** A tiny model or a classifier looks at the request and picks a tier. Cheap (one extra small call), but it's predicting difficulty from the input alone, which is genuinely hard — most requests don't look hard until you try them.
2. **Cascade with a confidence signal.** Run cheap first; escalate when a confidence signal is low. The confidence signal is the whole design: a self-reported score (weak, models are overconfident), token-level logprobs on the answer span (better where available), agreement across N samples at temperature (strong but N× the cost), or a cheap verifier model checking the cheap model's answer (usually the best cost/quality point).
3. **Deterministic routing on request features.** Route by tenant tier, feature, input length, or task type. Boring, transparent, no ML, and in my experience it captures most of the available savings — because the split is usually *by feature*, not by difficulty within a feature.

Start with (3). Add (2) where a feature has genuinely mixed difficulty. Reach for (1) rarely.

**💰 The math you must be able to do live.** Let `p` = fraction handled correctly by the cheap model, `C_c` = cheap cost/task, `C_e` = expensive cost/task, `C_v` = verifier cost/task.

```
cost_router      = C_c + C_v + (1 - p)·C_e
cost_all_expensive = C_e
router wins iff   C_c + C_v + (1 - p)·C_e  <  C_e
              ⟺   C_c + C_v  <  p·C_e
```

Plug numbers: `C_e` = $0.100, `C_c` = $0.012, `C_v` = $0.004. Condition: $0.016 < p × $0.100 → **p > 0.16**. So even a cheap model that only handles 16% of traffic breaks even. At p = 0.75: cost = 0.012 + 0.004 + 0.25 × 0.100 = **$0.041 vs $0.100 — a 59% saving**. At 5M tasks/month that's $500,000 → $205,000, a **$295k/month** saving.

Now the sensitivity that decides whether you build it: at p = 0.40, cost = 0.012 + 0.004 + 0.060 = $0.076, only 24% saved. **The router's value is roughly linear in `p`, so the entire project's ROI hinges on measuring `p` before you build.** Measure it by running both models over your eval set and computing the fraction where cheap is correct. If `p` is under ~0.5, the engineering and operational complexity usually isn't worth it and you should instead work on whether a mid-tier model clears the floor outright.

**The costs people omit:**

- **Latency on escalated requests is additive** — cheap + verifier + expensive. Your p99 gets worse even as your mean cost improves. If 25% of requests take 3× as long, that's a UX regression you must price.
- **Two prompts to maintain, two evals to run, two models to migrate** when either is deprecated.
- **The verifier is another model that can be wrong**, in both directions. A verifier with a 5% false-accept rate silently ships 5% of the cheap model's errors.

**⚠ Trap:** routing on self-reported confidence ("rate your confidence 1–10"). Models are poorly calibrated at this and the scores cluster at 8–9 regardless of correctness. If you use a self-reported score, you must *measure its calibration* — bucket by reported confidence and plot actual accuracy per bucket. If the curve is flat, your router is a coin flip with extra steps.

### If I ask you in a design round which serving engine you'd use and what features you'd rely on, how do you avoid claiming something that isn't true today?

This is the "engine feature matrix" version of the volatility problem, and the answer has the same shape as the pricing one: **name the mechanism, name the engine, and explicitly flag the feature-availability claim as something you verify rather than assert.**

The durable landscape (**📅 Volatile — feature matrices move every release; verify before claiming**):

- **vLLM** — the de-facto default for self-hosted OSS serving. The paged KV cache (PagedAttention) is its own foundational contribution, and continuous batching — which it popularized rather than invented; the idea comes from the Orca serving paper — is the other half of why it is fast. It also carries prefix caching, tensor/pipeline parallelism, quantization support, LoRA adapter serving, structured-output backends, and speculative decoding. Broadest model coverage, fastest to support new architectures.
- **SGLang** — competitive throughput, with a radix-tree prefix cache designed for heavy prefix sharing (multi-turn agents, many requests over one long document) and a strong structured-output story.
- **TensorRT-LLM** — highest performance on NVIDIA silicon if you're willing to pay the compilation and operational friction. The right answer when you're squeezing a fixed fleet and the model set is stable.
- **Hosted inference providers** — the answer when you want open weights without owning the deployment, at a per-token price with someone else's utilization risk.

**The features whose availability you must verify rather than assume**, because they're the ones that get claimed casually in design rounds and are the most version-dependent: which quantization formats are supported for *your* architecture (not in general); whether the attention variant your model uses has a fast kernel path; which speculative-decoding modes are implemented and whether they compose with your other settings; which structured-output backend is wired in and what schema subset it supports; whether disaggregated prefill/decode is available; and whether multi-LoRA serving works with your quantization.

**How to say it in the room without hedging into uselessness:**

> "I'd default to vLLM — paged KV cache and continuous batching are the two things that actually determine throughput, and it has the widest model coverage. If the workload has heavy prefix sharing, like a document-QA product where thousands of requests share one long prefix, I'd benchmark SGLang against it because its radix prefix cache is built for exactly that. I'd want to verify the current release supports the quantization format and structured-output backend I'm assuming before I commit to those in a design — that matrix moves every few weeks and I've been burned assuming it."

That last sentence is the whole answer. It converts a potential wrong claim into evidence that you've operated this in production. **The failure mode is the opposite: confidently asserting that engine X supports feature Y, being wrong, and losing the room's trust on everything else you said.**

**⚠ Trap:** quoting a throughput benchmark from a blog post. Serving throughput is a function of your sequence-length distribution, your batch depth, your quantization, and your hardware. Any number not measured on your workload is marketing. If you cite one, cite it as "their published number on their workload" and say what you'd measure instead.

### I'm going to ask you the question you'll get in a real onsite: "What does it cost to run this at scale?" Do it live, out loud.

Take a concrete brief so the method is visible: **an AI support agent handling 100,000 tickets/day, resolving 60% without a human, with a hard requirement of a first response within 5 seconds.**

**Step 1 — state assumptions out loud and write them down.** This is the whole skill; the interviewer is grading the method, not the number.

- Average resolution trajectory: 8 API calls (retrieval, plan, 3–4 tool calls, draft, verify).
- System prompt + tool schemas: 6,000 tokens, frozen → cacheable.
- Retrieved context: 3,000 tokens/call, varies → not cacheable.
- Conversation growth: by call 8 the transcript adds ~4,000 tokens.
- Output: 400 tokens/call average, with the drafting call larger.
- Model: mid-tier at $3/Mtok in, $15/Mtok out. **📅 Volatile — I'd verify current pricing.**

**Step 2 — tokens per task.**
- Cacheable input: 6,000 × 8 calls = 48,000 tokens, ~90% cache-read after warm-up.
- Uncacheable input: (3,000 retrieved + growing transcript, average ~2,000) × 8 ≈ 40,000 tokens.
- Output: 400 × 8 = 3,200 tokens.

**Step 3 — cost per task.**
- Cached input: 48,000 × (0.9 × $0.30 + 0.1 × $3.75)/1e6 = 48,000 × $0.645/1e6 = **$0.0310**
- Uncached input: 40,000 × $3/1e6 = **$0.1200**
- Output: 3,200 × $15/1e6 = **$0.0480**
- **Total ≈ $0.199/task.** Call it $0.20.

**Step 4 — scale it and sanity-check.**
- 100,000 tickets/day × $0.20 = **$20,000/day** = **~$600,000/month**.
- Sanity check against value: a human handling a ticket costs perhaps $4–6 fully loaded. Deflecting 60% of 100k/day = 60,000 tickets × $5 = **$300,000/day of human cost avoided**, against $20,000/day of inference. **15:1.** The economics are not close, and saying so is more useful than the raw number.

**Step 5 — name the levers and size each one; the top two here are within 15% of each other, so you attack both.**
1. **Uncached input is 60% of the bill.** Cut retrieval from 3,000 to 1,200 tokens with a reranker: 1,800 tokens × 8 calls × $3/1e6 saves ~$0.043/task = **$130k/month**. Biggest single line item to attack.
2. **Turn count.** Getting from 8 calls to 6 cuts roughly 25% across the board = **$150k/month** — marginally larger than the retrieval lever, and the two compose.
3. **Tier the 40% that escalate to a human** — they don't need the full trajectory; detect early and hand off. If half of them can bail after 3 calls, that's ~$70k/month.
4. **Output tokens** — a verbosity instruction cutting 400 → 280 saves $0.0144/task = **$43k/month**.

**Step 6 — check the latency requirement, because a cost answer that violates the SLO is wrong.** 5-second first response with 8 calls in the trajectory means the *first response* cannot wait for the whole trajectory. Architecture: acknowledge immediately, stream the first substantive turn after retrieval (call 1–2, ~1.5 s), continue the trajectory behind the stream. If the design requires all 8 calls before any output, the SLO is unachievable at any price and that's the finding to report.

**🗣 Say this in the room:** "Roughly twenty cents a task, six hundred thousand a month at that volume — but the number I'd actually put in the doc is that we're spending $20k/day of inference to avoid $300k/day of human handling, and the biggest levers are trajectory length and retrieval size, not model choice. Every input above is an assumption I'd want to replace with a measurement in week one."

**🏋 Drill (20 minutes, timed, unaided, no calculator beyond arithmetic on paper):** given a product brief you invent, produce the six steps above with every number derived. Pass criterion: an assumptions list, a cost per task, a monthly figure, a value-side sanity check, and a ranked lever list where the top lever is justified by its share of the bill.

### Last one. Give me the whole stack for a product brief, and tell me how you keep this knowledge from going stale.

**Brief:** an enterprise knowledge assistant — 4,000 seats, questions over an internal corpus of 2M documents, must cite sources, EU customers, 3-second p95 to first token, launch in 10 weeks.

**The stack, decided in the framework's order:**

1. **Retention and residency first**, because they can veto everything else. EU customers → EU data residency, likely a regional endpoint or a cloud-partner deployment. That decision constrains the model list *and* the feature list (partner deployments are feature-lagged — verify whether batch, caching semantics, and native citations survive), so it's week-one work, not week-eight work.
2. **Architecture: retrieval, not long context.** 2M documents can't be stuffed, and the cost arithmetic is decisive: 300k tokens/question at $3/Mtok is $0.90/question versus ~$0.02 with retrieval. Embedding + reranking on self-hosted open models — this is the task class where open weights are at or above parity and 100× cheaper.
3. **Capability floor:** build the eval before touching a model. 200 real questions from pilot users, labeled with gold answers and gold citations. Threshold stated with two numbers: ≥90% answer accuracy, ≤2% unsupported claims. Descend from the frontier tier; take the cheapest model that clears with margin.
4. **Citations are a hard requirement**, so native citation support moves from "nice" to a selection axis with veto power. It converts groundedness from a judgment call into a check.
5. **Caching:** frozen system prompt with a breakpoint; retrieved context after it. At 4,000 seats × ~15 questions/day = 60,000 questions/day, a 6,000-token frozen prefix cached at 0.1× saves 60,000 × 6,000 × ($3.00 − $0.645)/1e6 ≈ **$848/day ≈ $25.4k/month** for one marker.
6. **Latency:** 3-second p95 TTFT with a retrieval hop means retrieval must be ~300 ms p95 and the model's TTFT ~1 s at your p95 input length. That budget rules out reasoning-tier models on the interactive path — if you want reasoning, it goes in a background "deep research" mode with different UX, not in the default path.
7. **Observability from day one:** per-call usage logging with trace/tenant/prompt-version, cost per resolved question as a tracked metric, cache hit rate as an alerting SLI, and a golden-set eval on a cron feeding the same dashboard as latency.
8. **Failure design:** a documented fallback model with a measured quality delta, capped retries, and an explicit "I don't have a grounded answer" path — which for a knowledge assistant is a *feature*, not a failure.

**Now the meta-question, which is what this section is really for.** The way you keep this from going stale is to **make the volatile layer an artifact you maintain rather than knowledge you hold**:

- **One file in your repo — a dated rate card and capability table.** Provider, model ID, context window, the five token rates, cache minimum, feature flags you depend on, and a `verified_on` date. Your cost function imports it. Your CI fails if `verified_on` is more than 30 days old. That single check converts staleness from an invisible risk into a build failure.
- **A scheduled job that pulls the live model list** from each provider's models endpoint and diffs against the file, opening a ticket on any change.
- **The golden-set eval on a cron**, so provider-side changes surface as a metric rather than a customer complaint.
- **A 90-minute refresh ritual before every loop** — lineups, prices, deprecations, parameter deltas, rate-limit tiers, engine matrices, open-weight releases, regulatory dates.

**🗣 Say this in the room, and mean it:** "I don't carry the model table in my head — I carry the framework and I re-resolve the table, because anything I memorized six months ago is wrong now. What's stable is the structure: output is about 5× input, cache reads about a tenth, batch about half, and the decision is always the cheapest model that clears a measured quality floor under a stated latency budget. If you want current numbers I'll pull them up, but the arithmetic is the same either way."

**🏋 Final drill (90 minutes, unaided, timed).** Write, from scratch: (a) the nine-axis selection rubric in order; (b) the five token rates with their approximate multiples and the caching break-even in calls for both TTLs; (c) a correct Anthropic tool loop with error handling and the five invariants; (d) the prefix-cache invalidation hierarchy table; (e) a per-request cost function; (f) the router break-even inequality with a worked example; (g) the EU AI Act tier names and the milestone you're closest to. **Pass criterion: no more than two factual errors, and every cost claim carries its arithmetic.** Then verify every volatile item against live docs and record the date — that verification pass *is* the habit this section exists to build.


---

## 6. Navigating the Process: Take-Homes, Work Trials, Proctoring and Scope

*Mastering this proves you know the artifact you are building toward from day one, rather than discovering the rubric at week six.*

### Before we talk tactics — what is a take-home actually testing that an interview can't?

The mental model that reframes everything: **a take-home is not a test of whether you can build the thing. It is a sample of your engineering judgment under an artificial scarcity constraint, and the scarcity is the point.** Everyone who reaches this stage can wire an embedding model to a vector store. What differs is what you chose to build *first*, what you deliberately did not build, and whether you can defend both. The assignment says "build a RAG system"; the rubric is measuring what you did with your fourth hour.

That reframe has an immediate consequence for how you spend time. A submission that implements retrieval, reranking, streaming, a cache, and a Next.js frontend but has no evaluation is *worse* than a submission with naive top-k retrieval, a 40-example labeled eval set, a results table, and a README paragraph saying "reranking is the next thing I'd add and here's the eval I'd use to justify it." The second candidate demonstrated the thing that's hard to hire for. The first demonstrated that they can follow a tutorial fast.

The second consequence is about the artifact you're aiming at. From day one of your prep, you should know that the deliverable at the end of nearly every AI-product and applied loop is: **a small working system, plus an eval that measures it, plus a written record of the decisions and their trade-offs, plus a 45–90 minute conversation where you defend all three.** Every hour of prep should build toward that composite artifact, not toward a body of knowledge. If you're eight weeks out and you haven't built one end-to-end, you are preparing for a different interview than the one you'll sit.

**🗣 Say this in the room** (when they ask you to walk your submission): "I scoped this to prove three things: that the pipeline works end to end, that I can measure whether it works, and that I know what I traded away. Let me start with the eval, because that's what drove every other decision."

**⚠ Trap:** treating the take-home as a portfolio piece and polishing the demo. The grader is usually a senior engineer with forty minutes, reading your README and skimming your repo before the call. Demo polish is invisible to them; a missing eval is not.

### Give me the honest distribution of take-home types. What am I most likely to get?

Across the pool of publicly-shared assignments from AI-engineering loops — roughly a hundred of them, spanning AI-product companies, big-tech applied teams, and enterprise AI orgs — the flavors cluster into five, and they overlap (a single assignment often counts in two buckets, which is why these sum past 100%):

- **Retrieval / RAG — about 40%.** "Here is a corpus (docs, PDFs, a wiki dump, support tickets). Build a system that answers questions over it with citations." The single most likely assignment you will receive.
- **Agentic — about 30%.** "Build an agent that can use these tools to accomplish X." Tool definitions, a loop, termination, error handling, and usually a cost or step budget.
- **Conversational — about 20%.** Multi-turn assistant with memory, state, and personality/policy constraints. Often fused with RAG.
- **Document processing — about 15%.** Extract structured fields from messy PDFs/scans/invoices/contracts into a schema, at some accuracy bar. Heaviest in enterprise and vertical-AI companies (legal, fintech, healthcare, insurance).
- **LLM-as-judge / evaluation — about 10%.** "Here are 200 model outputs. Build something that scores them and tell me how much I should trust it." Rarest, and the one where a backend engineer with eval discipline can most dramatically out-perform.

**📅 Volatile:** these proportions drift with the market. The direction of drift through 2025–2026 has been *away* from pure RAG and *toward* agentic and evaluation assignments, because RAG take-homes became too easy to complete with an AI coding assistant. Expect agentic to keep gaining share.

The practical planning move: **prepare one deep artifact and two shallow ones.** Build a genuinely good RAG system with a real eval harness — that's your deep one, and it covers 40% of the distribution and most of the conversational 20% too. Then build a small agent with three tools and a step budget, and a small LLM-judge with an agreement analysis against human labels. Those two can be weekend-scale. You now cover the realistic distribution with one week of build time rather than five.

**⚠ Trap:** preparing breadth by *reading* rather than *building*. You cannot write a defensible README about chunking trade-offs from a blog post. The defense round is specifically designed to detect the difference, and it does, within about six minutes.

### What's the realistic scope and time budget on these assignments — the stated number and the true one?

The stated scope clusters tightly, and so does the lie in it.

**Stated:** a core RAG or agentic assignment is stated at **2–4 hours**. Document-processing assignments are stated at **3–5 hours** (they're heavier because parsing is genuinely fiddly). Full-stack assignments with a UI are stated at **4–8 hours**. The deadline you're given is typically **2–7 days**, most commonly 3–5. The defense conversation is **45–90 minutes**.

**True:** the honest completion time is roughly **double the stated estimate** for a candidate who is doing it properly — meaning with an eval, tests, and a written decision log. A "4-hour" RAG assignment done to a hiring bar is 7–9 hours of real work. This is not because you're slow; it's because the estimate was written by someone who already had the corpus parsed and a template repo, and because the stated estimate implicitly excludes the eval that the rubric weights most heavily.

**💰 Math on the real cost to you:** a 4-hour-stated assignment at the true 8 hours, times an average of 2.5 take-homes in a live pipeline, is 20 hours. Add defense prep at 1.5 hours each — 3.75 more. That's ~24 hours, or three full working days, spent per pipeline. If you're running two pipelines a quarter while employed, that's six days of evenings and weekends. Budget it explicitly on a calendar or it will silently eat your systematic prep, which is the actual failure mode: candidates arrive at the onsite having done three take-homes and zero transformer-internals review.

The scoping decision that follows: **you do not get to spend 8 hours on all of them.** Pick. My rule is to give full effort to assignments from companies where you'd accept an offer and where the assignment arrives *after* a human conversation, and to time-box the rest to the stated estimate with an explicit README note: "I time-boxed this to the stated four hours; here is what I would build next and in what order." Graders respect a stated, defended time-box far more than they punish an incomplete feature. What they punish is silent incompleteness.

**🗣 Say this in the room:** "I held myself to the stated time budget and made the cut lines explicit in the README rather than delivering something half-built with no explanation. The three things I deferred are listed with the reason and the eval I'd use to decide if they're worth it."

### What clarifying questions do you ask before you start, and does asking make you look weak?

Asking makes you look senior, and not asking is one of the more reliable weak signals. The reason is structural: **these assignments are deliberately under-specified, and the under-specification is a test.** A candidate who starts coding immediately has revealed that they treat ambiguous requirements as an implementation detail; a candidate who sends four sharp questions has revealed that they know what an ambiguous requirement costs downstream.

There's a second, mercenary reason: **the answers roughly double your usable time.** If you ask "should I optimize for retrieval quality or for latency?" and they say "quality, latency isn't graded," you just deleted three hours of caching work from your plan.

The questions I actually send, in a single short email, within a few hours of receiving the assignment:

1. **"What's being weighted most heavily — is this primarily an architecture/judgment read, or do you want a working deployed thing?"** The answer tells you where to spend the marginal hour.
2. **"Is the stated N hours a budget or an estimate? I'd rather deliver something scoped and defended than something sprawling."** This legitimizes your time-box in advance, in writing.
3. **"Am I allowed to use AI coding assistants on this?"** Non-negotiable — ask it every time. See the AI-tool policy questions later in this section for why.
4. **"Any constraints on API providers, cost, or model choice? Do you have keys, or am I paying?"** Also flushes out whether they expect a specific stack.
5. **"Is there a preferred way to submit — repo link, zip, deployed URL?"** Trivial but it prevents a submission-mechanics failure.
6. Domain-specific: for RAG, **"how large is the real corpus in production and how often does it change?"** — this is the single best question you can ask, because it distinguishes a rebuild-the-index design from an incremental-ingestion design and shows you're thinking past the toy.

**⚠ Trap:** asking questions that reveal you haven't read the assignment, or asking twelve of them. Four to six crisp, decision-relevant questions in one message. Twelve questions in three messages reads as anxiety, and a question whose answer is in paragraph two of the brief reads as carelessness.

**⚠ Trap (the expensive one):** waiting for an answer before starting. Send the questions, then start on the parts that no answer would change — corpus parsing, the eval set, the repo skeleton. If they never reply (it happens perhaps a third of the time), note in your README: "I asked about X and Y; absent an answer I assumed Z, because [reason]." That documented assumption scores nearly as well as the answer would have.

### Walk me through what a passing RAG take-home actually contains. Assume four hours.

Since this is 40% of the distribution, it's worth being concrete. Here's the artifact I'd submit and the order I'd build it in.

**Hour 0–0.5: the eval set, before any retrieval code exists.** Read 15–20 documents from the corpus by hand. Write 30–50 question/expected-answer pairs, and for each one record the document ID (and ideally the passage) that contains the answer. Deliberately stratify: some single-hop factual, some requiring two documents, some whose answer is genuinely *not* in the corpus (these are your refusal tests, and they're the ones that separate submissions). Commit it as `eval/questions.jsonl`. This is the highest-leverage thirty minutes in the entire assignment.

**Hour 0.5–2: the boring pipeline, end to end.** Parse → chunk → embed → store → retrieve top-k → stuff into a prompt with citation instructions → generate. Use a well-known embedding model and a local vector store (FAISS, Chroma, or pgvector if Postgres is already in play — and pgvector is a defensible choice you can justify from operational familiarity). Do not build a reranker yet. Do not build a UI yet. Get an answer to come out of the far end.

**Hour 2–2.75: measure it.** Two families of metric, and you must report both:
- **Retrieval metrics** on your labeled set: recall@k (did the gold document appear in the top-k at all?) and MRR or precision@k. Recall@k is the one that matters, because if the passage isn't retrieved, no amount of generation quality saves you.
- **Answer metrics**: groundedness/faithfulness (is every claim supported by a retrieved chunk?) and correctness against your expected answers. An LLM judge is acceptable here *if* you spot-check a sample of its verdicts by hand and report the agreement rate.

Now you have a baseline table. This table is your submission's spine.

**Hour 2.75–3.5: one or two targeted improvements, each with a before/after number.** Look at what your baseline table says is broken. If recall@5 is 0.62, your problem is retrieval — try a different chunk size, or add BM25 and fuse. If recall@5 is 0.94 but correctness is 0.71, your problem is generation or context assembly — the passages are there and the model isn't using them. **Fix what the numbers say is broken, not what the blog posts say is fashionable.** This single behavior is the strongest senior signal in the whole exercise.

**Hour 3.5–4: the README.** Architecture in five sentences and one diagram, the results table with baseline vs improved, the decisions with their rationale, the known failure modes, and "what I'd do next with another day" in priority order.

**⚠ Trap:** the reflexive hybrid-search-plus-reranker-plus-HyDE stack, added before measuring anything. It's three hours of work, it *usually* helps a little, and it makes you unable to answer "how much did the reranker buy you?" — which is exactly what you'll be asked. An unmeasured optimization is not an optimization; in the defense round it's a liability.

**📐 Numbers you must know:** on a small clean corpus with sensible chunking, naive dense top-5 retrieval typically lands recall@5 somewhere in the 0.6–0.85 band; hybrid BM25 + dense fusion typically adds a handful of points; a cross-encoder reranker over top-50 typically adds several more on precision at the top of the list but costs latency. Do not quote these as facts about the world — quote *your* numbers from *your* table. The point of memorizing the rough band is to know when your own number is suspicious: a reported recall@5 of 0.99 on a real corpus almost always means your eval questions were written by paraphrasing the chunks, which is a leak.

### Same question for an agentic take-home. What separates a hire from a no-hire there?

The mental model: **a RAG take-home is graded on retrieval quality; an agentic take-home is graded on control.** Anyone can get a loop of `while not done: call model; execute tool` to produce a demo. The grader is looking for whether you treated the loop as an unbounded distributed system with an unreliable worker — which is exactly your backend instinct — or as a magic box.

The five things I make sure are visible in the code:

**1. A hard step and token budget, enforced in code, not in the prompt.** `max_steps=8`, and a running token counter that aborts and returns partial results with a reason. An agent without a budget is an unbounded loop calling a paid API — you would never ship that in a Celery worker and you should not ship it here. This one detail, present, moves you materially up the rubric.

**2. Explicit termination conditions, enumerated.** Success, budget exhausted, repeated no-progress (the same tool called with the same arguments twice), and unrecoverable tool error. Write them as an enum with a `TerminationReason` returned in the response. Interviewers notice.

**3. Tool errors as data, not exceptions.** When a tool call fails, the failure goes back into the conversation as a `tool_result` with `is_error` semantics so the model can recover — not a Python traceback that kills the loop. And there's a retry policy with a cap, because "just add a retry" on a model-driven loop multiplies cost.

**4. Structured, replayable traces.** Every step logged as a record: step index, tool name, arguments, result (truncated), token counts, latency, cumulative cost. Dump them as JSONL. This is a two-hour-cheap feature that makes your defense round trivially easy, because you can *show* what the agent did rather than describe it.

**5. An eval over trajectories, not just final answers.** Ten to twenty scripted tasks with known outcomes, scored on: did it reach the correct final state, how many steps did it take, how many tokens did it burn, how often did it call an unnecessary tool. Report mean and worst-case, not just mean — worst-case step count is what actually pages you in production.

**💰 Math you should put in the README:** with a 12-step cap, an agent whose context grows by ~1,200 tokens per step (tool schema echo + result) has cumulative input across the run of roughly 1,200 × (1+2+…+12) = 1,200 × 78 = **93,600 input tokens** for a single task, because every step re-sends the whole history. At $3/Mtok input that's $0.28 per task uncached; with prefix caching on the stable prefix at a 90% discount on cached tokens, the bulk of that collapses toward ~$0.05. At 50,000 tasks/month that's the difference between **$14,000 and $2,500 a month**. Putting that arithmetic in your README, with your own measured token counts, is worth more than any feature you could add in the same twenty minutes.

**🗣 Say this in the room:** "I treated the agent loop as a job runner with an unreliable worker: bounded steps, bounded tokens, typed termination reasons, tool errors fed back as data rather than raised, and a JSONL trace per run so failures are replayable. Then I evaluated trajectories, not just final answers, because an agent that gets the right answer in eleven steps is a different system from one that gets it in three."

### And a document-processing assignment — what's the shape and where do people lose it?

Document processing is the assignment where candidates most consistently mis-allocate, because the hard part is not where they expect. They spend four hours on prompt engineering for extraction and thirty minutes on parsing, and lose on parsing.

The mental model: **in document extraction, the model is rarely the bottleneck — the text you hand it is.** If your PDF parser silently drops the second column of a two-column layout, or flattens a table into a run-on line, or loses the page break that separated invoice 3 from invoice 4, then no prompt recovers the information. Garbage in the context window is not a model failure and no amount of temperature tuning fixes it.

So the shape of a strong submission:

**Establish parse quality first, with eyes on the actual text.** Dump the extracted text of five documents to files and read them. Note explicitly in the README which document classes your parser handles and which it mangles. If some documents are scans, say so and state whether you handled OCR or scoped it out. Naming the failure class you did not handle is a strength; silently producing garbage on it is the failure.

**Extract to a schema, with validation as a first-class output.** This is where your Pydantic fluency is a genuine advantage and you should make it visible: a model with typed fields, `Optional` where the field is genuinely optional, validators for formats (dates, currency, IDs, checksums), and a confidence or provenance field per extracted value. Use the provider's structured-output / constrained-decoding mode rather than parsing JSON out of prose — and say why in the README: a schema-constrained decode gives you a parse guarantee, whereas "respond in JSON" plus a `json.loads` gives you a retry loop and a tail of failures.

**Report per-field accuracy, not document accuracy.** A single number like "94% accurate" is nearly meaningless. The useful table has one row per field: `invoice_number 0.99 / vendor_name 0.91 / line_items 0.68 / total 0.97`, computed against 20–30 hand-labeled documents. That table immediately tells you and the grader that line items are the problem and everything else is fine, which is a *product* conclusion: ship it with line items flagged for human review.

**Then the escalation policy**, which is the senior move: define the confidence threshold below which a document routes to a human. State the arithmetic. "At a 0.85 threshold on the line-items field, 22% of documents route to review; at 100k documents/month and 90 seconds of human review each, that's 22,000 × 1.5 min = **550 human-hours/month**. Dropping the threshold to 0.7 halves the review volume but doubles the escaped-error rate from 1.1% to 2.3%." That paragraph is the difference between an engineer and an engineer who understands the business the system sits in.

**⚠ Trap:** the accuracy metric that ignores nulls. If your extractor returns `null` for `vendor_name` on 8% of documents and you score only the non-null cases, you'll report 0.97 for a field that's actually failing 8% of the time. Score misses as errors, and report coverage and accuracy as two separate numbers.

**📐 Numbers you must know:** a page of dense text is roughly **500–800 tokens** depending on tokenizer and language; a typical business PDF page with whitespace and tables lands nearer 300–600. So a 40-page contract is roughly 20k–30k tokens, which fits comfortably in a modern context window but costs real money at scale: 100k documents × 25k tokens = 2.5B input tokens; at $3/Mtok that's **$7,500 per full reprocess**. That number is why "just send the whole document every time" is a design decision requiring justification, not a default.

### The LLM-as-judge assignment is rarer but I want to be ready. What does a strong submission look like?

This is the assignment where your background gives you the largest relative advantage, because it is fundamentally a **measurement-instrument calibration problem**, and most candidates treat it as a prompt-writing problem.

The mental model to lead with: **a judge is a classifier you did not train and cannot inspect, so the only honest way to use it is to measure its agreement with the ground truth you do trust.** Nobody should believe a judge score until you've shown its agreement with human labels. That framing, stated in your first paragraph, is most of the grade.

The submission structure:

**1. Hand-label a subset yourself.** Take 60–100 of the provided outputs and label them personally against a written rubric. This is unglamorous and it is the whole assignment. You cannot calibrate an instrument with no reference standard.

**2. Write the rubric before the prompt.** Explicit criteria, explicit levels, and — critically — a definition of each level that a second person could apply and get the same answer. If your rubric says "3 = good," it's not a rubric. If it says "3 = answers the question and every factual claim is supported by the source, but omits at least one relevant qualification," it is.

**3. Report agreement with a chance-corrected statistic, not raw accuracy.** Raw agreement is inflated when the label distribution is skewed: if 80% of outputs are "good," a judge that says "good" every time scores 80%. Report Cohen's kappa (for a single human reference) or the confusion matrix in full. Read the scale correctly: kappa 0 is chance agreement and 1 is perfect, so a kappa of 0.4 on a 5-point scale is only *fair* agreement — meaningfully better than chance, but nowhere near reliable enough to gate an individual example — and saying exactly that is a *pass*, not a fail.

**4. Probe the known biases and show the results.** Three you should test explicitly because they're well documented: **position bias** (in a pairwise setup, swap A and B and measure how often the verdict flips — a flip rate well above chance means the judge is scoring position, not quality); **length bias** (correlate the judge's score with output length; a strong positive correlation means you built a verbosity detector); and **self-preference** (a judge tends to score outputs from its own model family more favorably, so if you're judging model X with model X, say so as a caveat).

**5. Give the deployment decision, with a threshold.** "This judge has kappa 0.61 against my labels, position-flip rate 9%, and length correlation r=0.31. That is good enough to use as a **regression detector** on aggregate scores across releases, and not good enough to use as a per-example gate that blocks a deploy. Here's the sample size I'd need for the aggregate use: to detect a 2-point drop with 80% power at α=0.05, roughly N examples per arm." Then show the calculation or state the assumption you used.

**⚠ Trap:** using a fine-grained 1–10 scale. Judges are poorly calibrated across ten levels and cluster on 7 and 8; the effective resolution is about three levels. Use a 3- or 5-point scale with anchored definitions, or better, pairwise comparison with a tie option, which is a much easier judgment for a model to make consistently.

**🗣 Say this in the room:** "I don't ask whether the judge is good. I ask what decision the judge is licensed to make. Mine had kappa 0.61 and a 9% position-flip rate, which licenses aggregate regression detection across a release, and does not license per-example gating. That's the sentence I'd put in the design doc."

### Conversational assistants — what's actually being graded there, since the pipeline is simple?

Conversational assignments look like the easiest of the five and they're graded on the thing that isn't in the assignment text: **state management across turns**, which is where all the interesting failure lives.

Lead with this: **a multi-turn assistant is a state machine whose transition function is a stochastic text generator, so every piece of state you let the model own is state you cannot assert on.** The design question is therefore what you keep *outside* the model. Slot values collected so far, the user's verified identity, whether a confirmation has been given, the cart contents, the escalation flag — all of that belongs in structured state your code owns, with the model reading from it and proposing updates that your code validates. Candidates who keep everything in the transcript and hope the model remembers produce systems that forget the user's name on turn nine and cannot be tested.

Concretely, in a 4-hour build I want to see:

- **A typed conversation state object** (a Pydantic model) persisted per session, separate from the message history.
- **A context assembly function** that builds the prompt from (system policy) + (compact state summary) + (last N turns) + (retrieved context if any) — with a token budget and a documented truncation policy. The truncation policy is what a grader looks for: what gets dropped first, and what is pinned and never dropped.
- **Policy constraints enforced outside the prompt where they matter.** If the assistant must not give medical advice or must not issue a refund over $200, the $200 check is an `if` statement in your code path, not a sentence in the system prompt. Any constraint that is only in the prompt is a constraint with a jailbreak-shaped hole in it, and saying that sentence out loud in the defense is a strong signal.
- **Multi-turn eval, which almost nobody does.** Scripted conversations of 5–8 turns with assertions at specific turns: "after turn 4 the state must contain a confirmed email"; "at turn 6, given this adversarial input, the assistant must refuse." A user-simulator (another model playing the customer against a persona and goal) makes this cheap to scale, and the fact that you built one is itself the differentiator.

**⚠ Trap:** the summarization-based memory that silently drops the constraint. Candidates compress old turns into a running summary to control tokens, then discover that the summary dropped "the customer said they're allergic to penicillin" or "the customer already declined the upsell twice." Pin critical facts into structured state and *never* route them through a lossy summarizer. Demonstrating that you thought about which facts are pinned is worth more than the summarizer itself.

**📐 Numbers you must know:** with a 1,500-token system prompt and ~250 tokens per turn-pair, a 20-turn conversation reaches 1,500 + 20 × 250 = **6,500 tokens of input on the final turn**, and the cumulative input across the whole conversation is 1,500×20 + 250×(1+2+…+20) = 30,000 + 52,500 = **82,500 tokens** — because the history is resent every turn. That quadratic-in-turns growth is the single most important cost fact about chat products, and the reason prefix caching is not optional at scale. At $3/Mtok that conversation costs $0.25 uncached; with the system prompt and stable history prefix cached at a 90% discount, roughly $0.04.

### The defense round — what actually gets graded in those 45 to 90 minutes?

The defense is where the take-home is scored. I want to be blunt about that: the code is read for maybe fifteen minutes before the call, and the grade is set during the conversation. Candidates who assume the artifact speaks for itself lose here routinely.

Four things are being probed, in roughly this order:

**1. Did you make the decisions, or did they happen to you?** The interviewer will point at an arbitrary choice — chunk size, k, the embedding model, the retry policy — and ask "why this?" There are exactly three acceptable answers: *"I measured it and here are the numbers"*, *"I didn't measure it, I picked a standard default because the marginal value was low relative to X, and here's the experiment I'd run if it mattered"*, and *"that's a bug, I'd change it."* All three pass. The failing answer is a post-hoc rationalization that you're visibly inventing in real time, which is extremely legible on a call.

**2. Do you know what's broken?** They will ask "what would break first if we put this in front of 10,000 users?" You should have this list already written in your README, ranked, with the reason. Having already written it converts a stress question into a rehearsed one.

**3. Can you extend it live?** Roughly half of these rounds include "let's add X" — a filter, a new tool, a caching layer — either as a discussion or as live coding in your own repo. This is why your code needs to be navigable *by you* under pressure: if you cannot find where retrieval happens in eight seconds, it reads as unfamiliarity with your own submission, which raises the question of who wrote it.

**4. Are you honest about the gaps?** "I didn't implement authentication because it wasn't in scope and it's well-understood; here's where it would go." Perfect answer. "It handles auth" when it doesn't, discovered by a follow-up question, is unrecoverable.

**🏋 Drill (45 minutes, unaided, one week before any real defense):** hand your submitted repo to a friend with three instructions — pick five arbitrary lines and ask "why?", ask "what breaks at 10k users?", and ask "add a per-tenant filter, code it now." Record yourself. Pass criterion: zero answers longer than 90 seconds, zero "um, I think I did that because...", and you find any file in your repo in under ten seconds.

**⚠ Trap:** rehearsing a narrative instead of re-reading the code. If you submitted five days ago, you have forgotten the details, and the details are what's asked about. Spend 30 minutes re-reading your own diff before the call. Every single time.
### Lay out the rubric. What are these submissions actually scored on, and how is the weight distributed?

The rubric is remarkably consistent across companies, which is the useful news: prepare once, submit everywhere. Eight criteria, and the weights are approximate but the *ordering* is stable.

1. **Functional correctness** — does it run from a clean clone and do the stated thing. Gate criterion; failing this ends the review.
2. **Code quality and architecture** — module boundaries, dependency direction, naming, absence of a 600-line `main.py`. You are already fluent here; this is where your backend background is worth free points.
3. **Evaluation methodology** — *the critical criterion*. How do you know the AI output is any good, what did you measure, on what data, and what did the numbers say. Weighted heaviest and skipped most often.
4. **Production readiness** — config, secrets, errors, timeouts, logging, cost tracking, graceful degradation.
5. **Performance against stated targets** — commonly p95 under 2 s, 100+ req/s sustained, cache hit rate above 40% when caching is in scope.
6. **Testing** — including the hard part, which is testing a nondeterministic dependency.
7. **Documentation** — README as design doc, decision log, known limitations.
8. **Demo / walkthrough** — the defense conversation, and sometimes a recorded demo.

If I had to put numbers on it from how these are actually graded: correctness and code quality together are maybe 30%, evaluation methodology alone is 20–25%, production readiness 15%, documentation 10%, testing 10%, performance 10%, demo the remainder. **The single distinguishing fact is that evaluation carries as much weight as the entire feature implementation, and it is the criterion most candidates score zero on.**

**🗣 Say this in the room** (opening your walkthrough, and it sets the frame for everything after): "The organizing decision was to build the evaluation harness before the retrieval pipeline, so that every subsequent change had a number attached. I'll show you the results table first and then how the architecture follows from it."

**⚠ Trap:** believing correctness is the bar. Correctness is the *gate*, not the bar. Roughly everyone who submits produces something that runs. Passing the gate earns you the right to be compared on criteria 2 through 8, and that comparison is where the offer is decided.

### You keep saying "eval-first." Make the case, and tell me exactly what to do in the first hour.

The case in one sentence: **for any system with a nondeterministic component, an unmeasured change is not an improvement — it is a rumor.** Your backend instincts already encode this. You would not accept "I optimized the query" without an EXPLAIN plan and a latency measurement. The exact same standard applies to "I improved the retrieval," and the only difference is that people forget to apply it because the output is prose and prose feels self-evidently good or bad.

There's a second argument that's specific to interviews and more mercenary: **every other criterion becomes cheap once the eval exists.** With a results table, your README writes itself, your defense answers are all "here's the number," your architecture decisions are all justified, and your "what I'd do next" is a ranked list derived from the table rather than a wish list. The eval is not an eighth of the work; it is the spine that makes the other seven-eighths defensible.

The first hour, concretely, for any of the five assignment types:

**Minutes 0–10: define the unit of evaluation and the success criterion in writing.** One sentence, with a number in it. "A response passes if every factual claim is supported by a retrieved chunk and the answer matches the reference on the key entity." Put it at the top of `eval/README.md`. If you can't write this sentence, you do not yet understand the assignment, and discovering that in minute five instead of hour three is worth the whole exercise.

**Minutes 10–40: build the dataset by hand.** 30–50 examples. Yes, by hand, from the real corpus. Yes, this is boring. It is also the only part of the submission that cannot be produced by an AI assistant in ninety seconds, which is precisely why it discriminates. Stratify deliberately: easy head cases, hard multi-hop cases, out-of-scope cases that should be refused, and at least three adversarial or malformed inputs.

**Minutes 40–60: build the runner.** A script that takes a system-under-test callable, runs all N examples (concurrently — you own asyncio, use it), scores each, and emits both a per-example JSONL and an aggregate table. Make the system-under-test a parameter so you can A/B two configurations in one run. Print a diff against the last run.

Here is a runner skeleton small enough to write from memory under time pressure:

```python
import asyncio, json, time
from dataclasses import dataclass, asdict

@dataclass
class Result:
    id: str; passed: bool; score: float; latency_ms: float; detail: str

async def run_eval(system, cases, scorer, concurrency=8):
    sem = asyncio.Semaphore(concurrency)
    async def one(case):
        async with sem:
            t0 = time.perf_counter()
            try:
                out = await system(case["question"])
                err = ""
            except Exception as e:                 # a crash is a failure, not a skip
                out, err = None, f"{type(e).__name__}: {e}"
            dt = (time.perf_counter() - t0) * 1000
            s = 0.0 if out is None else await scorer(case, out)
            return Result(case["id"], s >= 1.0, s, dt, err)
    results = await asyncio.gather(*(one(c) for c in cases))
    n = len(results)
    print(f"pass {sum(r.passed for r in results)}/{n}  "
          f"mean {sum(r.score for r in results)/n:.3f}  "
          f"p95 {sorted(r.latency_ms for r in results)[int(0.95*n)-1]:.0f}ms")
    with open("eval/last_run.jsonl", "w") as f:
        for r in results: f.write(json.dumps(asdict(r)) + "\n")
    return results
```

Forty lines, no framework, and it puts you in the top decile of submissions. Note the two details a grader will notice: **exceptions are scored as failures rather than swallowed** (a system that crashes on 10% of inputs must not report 100% on the remaining 90%), and **latency is captured in the same pass as quality**, because you will be asked about both.

**⚠ Trap:** generating your eval questions by asking a model to read your chunks and write questions about them. The questions come out phrased in the chunk's own vocabulary, so retrieval finds them trivially and you report recall@5 of 0.98. You have measured lexical overlap, not retrieval. Write them yourself from the *documents*, phrased the way a user would ask, and validate that at least a few are hard.

### What does "production readiness" mean inside a four-hour take-home? I can't deploy a real system.

Right — and the graders know that. Production readiness in this context is not "you deployed it," it's **"you demonstrated that you know what the gap is between this and production, and you closed the parts that are cheap."** It's a set of low-cost signals, and their absence is what's noticed.

The cheap ones, all together under thirty minutes:

- **Config and secrets from the environment.** `.env.example` committed, `.env` gitignored, a settings object (Pydantic settings is the obvious choice and reads as fluent), no API key in the repo. A committed API key is an instant and total failure at some companies, treated as a security-judgment signal rather than an oversight.
- **Timeouts and retries on every network call**, with a cap and jittered backoff, and an explicit note about which errors are retryable. Model APIs return 429s and 529-class overloads routinely; a client with no timeout will hang a request forever the first time the provider has a bad minute.
- **Structured logging with a request/trace ID** threaded through the whole call path, so a single request can be reconstructed from logs. Emit the token counts and the model name on every LLM call. This costs ten lines and it is the single most credible "I've operated this" signal available.
- **A cost counter.** Accumulate input/output tokens per request, multiply by a price table in config, expose it in the response metadata or a `/metrics` endpoint. Then put the per-request cost in the README. Almost nobody does this and everybody grading it cares.
- **Graceful degradation, stated.** What happens when the vector store is down (fall back to keyword search? return an honest error?), when the model API 429s (queue? fail fast? fall back to a smaller model?), when retrieval returns nothing (refuse, don't hallucinate). One paragraph in the README plus the code path for the important one.
- **Health and readiness endpoints** if there's a service, distinguishing "process is up" from "dependencies reachable."

**⚠ Trap:** confusing production readiness with infrastructure. A Kubernetes manifest, a Terraform module and a multi-stage Dockerfile in a 4-hour take-home read as *over-engineering*, not readiness — especially if the eval is missing. A single Dockerfile and a `docker compose up` that works is exactly the right amount. Effort spent on infra you didn't need is effort the grader knows you took from somewhere else.

**🗣 Say this in the room:** "I drew the production line at things that are cheap and that failing to do would be a judgment error: config hygiene, timeouts, trace IDs, token and cost accounting, and a documented degradation path per dependency. I explicitly did not containerize for multi-region or add a queue, because the assignment's traffic assumption didn't justify them and I'd rather spend that hour on the eval."

### The brief states p95 under 2 seconds and 100+ requests per second. How do you actually hit and demonstrate that?

First, decompose, because "p95 under 2s" is not a single budget — it's a chain, and you should say so out loud.

For a RAG endpoint the chain is: embed the query (30–80 ms to a hosted embedding API, ~5 ms local for a small model) → ANN search (1–20 ms for an in-memory index at small scale; tens of ms with a network hop and filtering) → optional rerank (a cross-encoder over 50 candidates is 50–200 ms and is often the largest non-LLM term) → prompt assembly (negligible) → **generation, which dominates everything else.**

That last term is where the honesty lives. Generation latency splits into **time-to-first-token** — driven by prefill over your prompt plus queueing at the provider — and **inter-token latency** times the number of output tokens. If your answer is 300 tokens and the model streams at 40 tokens/second, that's 7.5 seconds of generation alone and your 2-second p95 is unreachable *for the full response*. Which means the target must be re-read: **for a streaming interface, the meaningful SLO is TTFT, not total completion.** State that interpretation explicitly in your README; the grader is often checking whether you noticed.

**💰 Math, worked:** budget = 2,000 ms p95. Allocate: query embedding 80, retrieval 40, rerank 150, prompt assembly 10, network overhead 60 → 340 ms of non-generation. That leaves 1,660 ms for TTFT. A 4,000-token prompt on a mid-tier hosted model typically prefills in a few hundred milliseconds, so with typical queueing you have headroom — *unless* you stuff 30,000 tokens of context, in which case prefill alone can consume most of the budget. **The p95 target is therefore a context-length constraint in disguise,** and the way to hit it is to cut retrieved context, not to micro-optimize your vector search. Say that sentence in the defense and you've demonstrated the thing they were testing.

For **100 req/s**, do the arithmetic rather than hand-waving: at 100 req/s with a 3-second mean end-to-end latency, Little's Law gives 100 × 3 = **300 concurrent in-flight requests**. Your process must therefore be fully async with no blocking calls in the path (a synchronous embedding call or a blocking `psycopg2` cursor inside an async handler will collapse this, and that's a bug a grader will look for specifically), and — the actual constraint — your *provider* rate limit must support it. 100 req/s × 4,000 input tokens = **400,000 input tokens/second = 24M tokens/minute**, which is far above a standard account tier. The correct answer to "how do you hit 100 req/s" is therefore: "locally I hit it with a stubbed model client, and I've documented that the real constraint is the provider TPM ceiling, which requires tier escalation, batching, or a self-hosted engine. Here's the calculation."

**Demonstrate it, don't claim it.** Include a small load script (Locust, `hey`, or a 30-line asyncio driver) against a stubbed model client, and paste the output table — p50/p95/p99 and throughput — into the README. A measured 100 req/s against a stub plus an honest note about the real bottleneck beats an unmeasured claim by a wide margin.

**⚠ Trap:** benchmarking against the live provider and reporting the numbers as your system's performance. You'd be reporting the provider's queue depth on a Tuesday afternoon. Separate *your* overhead (which you control and should optimize) from *their* latency (which you can only budget around), and report the two separately.

### The brief says achieve a cache hit rate above 40%. Which cache, and how do you get there?

There are three distinct caches in an LLM system and conflating them is a common and costly error, so I'd disambiguate before answering.

**1. Provider-side prefix caching.** The provider retains the KV state for a prefix of your prompt; a subsequent request sharing that exact prefix skips prefill for it and is billed at a large discount on the cached portion. This is not a cache *you* implement; it's one you *design your prompt layout for*. The rule that follows is mechanical and it's the entire trick: **order your prompt static-to-dynamic.** System policy, then tool schemas, then few-shot examples, then long-lived retrieved context, and only then the user turn. One dynamic token near the front — a timestamp, a session ID, a shuffled document order — invalidates everything after it and takes your hit rate to zero.

**2. Exact-match response caching.** Hash the fully-rendered prompt (plus model, temperature, and all sampling parameters — leaving those out of the key is a real bug) and store the response in Redis. Hit rates depend entirely on traffic shape: a consumer product with a long-tail of unique questions might see 5–15%; an internal tool where people ask the same twelve things sees 40–60%.

**3. Semantic caching** — embed the query, and serve a cached response if the nearest neighbor is within a similarity threshold. Higher hit rates and **the most dangerous thing in this list**, because "what's our refund policy for EU customers?" and "what's our refund policy for US customers?" are extremely close in embedding space and have different answers. If you implement it, gate it behind a high threshold, never enable it for anything personalized or tenant-scoped, and log every hit for offline audit of exactly this failure. I would push back in review on any semantic cache without a documented false-hit audit.

**💰 Math, and this is the number to put in the README:** a 12,000-token system prompt + tool schemas, called 200,000 times/day, at $3/Mtok input. Uncached: 12,000 × 200,000 = 2.4B tokens/day = 2,400 × $3 = **$7,200/day**. With prefix caching at a 90% discount on the cached portion (**📅 Volatile** — cache read discounts and write surcharges differ by provider; re-verify) and a 90% hit rate: hits cost 0.9 × 2.4B × $0.30/Mtok = $648, misses cost 0.1 × 2.4B × $3/Mtok = $720, total ≈ **$1,368/day**. That's **$5,832/day saved, roughly $175k/month**, for the engineering work of not putting a timestamp at the top of your prompt. This is the highest return-per-line-of-code decision in applied LLM engineering and it is entirely a prompt-layout discipline.

**⚠ Trap:** reporting the cache hit rate without reporting what the cache *cost you in correctness*. A 60% hit rate on a semantic cache that serves the wrong tenant's answer 0.3% of the time is a catastrophe wearing a metric's clothes. Report hit rate *and* a false-hit audit on a sample.

### How do you test a system whose main dependency returns something different every time?

This is the question where your existing testing discipline transfers almost completely — the trick is knowing which layer gets which kind of test. I'd draw three tiers and be explicit that they answer different questions.

**Tier 1 — deterministic unit tests over everything that isn't the model.** Chunking, prompt rendering, citation parsing, schema validation, retry logic, token counting, cost calculation, context truncation. This is 80% of your code and it is fully deterministic. Test it normally. Property-based tests are especially strong here — "for any document, the concatenation of chunks contains every character of the original except at boundaries," or "the assembled prompt never exceeds the token budget for any input" — and Hypothesis in a take-home reads as genuine rigor.

**Tier 2 — recorded-fixture integration tests.** Capture real provider responses once, store them as JSON fixtures, replay them in CI. Your test suite runs offline, deterministically, in milliseconds, at zero cost. `vcrpy` or a hand-rolled fixture layer both work; the hand-rolled version is about twenty lines and lets you show the seam clearly:

```python
class RecordedClient:
    """Replays captured responses keyed by a hash of the request."""
    def __init__(self, path, live=None, record=False):
        self.path, self.live, self.record = path, live, record
        self.store = json.loads(path.read_text()) if path.exists() else {}

    async def complete(self, **kw):
        key = hashlib.sha256(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:16]
        if key in self.store:
            return self.store[key]
        if not self.record:
            raise KeyError(f"no fixture for {key}; run with RECORD=1")
        resp = await self.live.complete(**kw)
        self.store[key] = resp
        self.path.write_text(json.dumps(self.store, indent=2))
        return resp
```

The design point worth stating out loud: **the model client is an interface, and every test above it uses a fake.** That's the same dependency-inversion move you'd make for a payment gateway, and framing it that way in the defense is exactly right.

**Tier 3 — the eval suite, which is not a test.** Tests answer "did the code do what it's supposed to." Evals answer "is the output good enough." They have different pass criteria (binary vs a threshold on a distribution), different cadence (every commit vs nightly or pre-release), different cost (free vs dollars), and different failure semantics. **Conflating them is the classic mistake**: putting a live model call in your unit test suite makes CI slow, flaky, expensive, and dependent on a third party's uptime, and the first time it fails spuriously someone adds a retry, and the second time someone deletes the assertion.

What I do want in CI on every commit: tiers 1 and 2, plus a **tiny smoke eval** — five to ten examples against the live model, with a generous threshold, purely to catch "the API contract changed" or "someone broke the prompt template." Ten examples is a couple of cents and thirty seconds.

**⚠ Trap:** `assert "Paris" in response`. It passes for "Paris is the capital of France" and also for "I don't know anything about Paris." Assert on *structure* (valid JSON, required fields present, citations resolve to real chunk IDs, no citation to a chunk that wasn't retrieved) in tests, and leave *semantics* to the eval suite where you can express a threshold instead of a boolean.

**🗣 Say this in the room:** "I separate tests from evals on purpose. Tests are deterministic, run on every commit, and gate the merge — they use recorded fixtures so there's no live model in CI. Evals run nightly against the live model, produce a distribution rather than a boolean, and gate the release. Mixing them gives you a flaky CI that people learn to ignore, which is worse than having neither."

### Walk me through your README. What's in it and why does it matter this much?

The README is the highest-leverage file in the repository, and I'd argue that literally — it's the only artifact guaranteed to be read, it's read before the code, and it sets the frame the code is then read within. A grader with forty minutes reads the README fully and skims the code looking for confirmation.

The structure I use, in this order, because it's descending order of what the grader wants:

**1. What this is and how to run it — five lines, three commands.** `cp .env.example .env` → `docker compose up` → `curl localhost:8000/ask -d '{"q":"..."}'`. If your setup takes more than three commands or fails on a clean clone, nothing else in the README matters. Test this by cloning into a fresh directory and following your own instructions literally.

**2. Results, up front.** The table. Baseline vs final, per metric, with N and the eval set described in one sentence. Putting numbers above the fold is a deliberate act of framing: it says the system was measured, before the reader has formed any other impression.

**3. Architecture — one diagram and five sentences.** ASCII or Mermaid, in the file, not a linked image. Boxes and arrows for the data path and one sentence on why the boundaries fall where they do.

**4. Decisions and trade-offs — the section that gets you hired.** Six to ten entries, each: *what I chose / what I rejected / why / what it would take to change my mind.* "Chunking: 512 tokens with 64 overlap. Rejected semantic chunking — it cost 40 minutes to implement and my eval showed +0.02 recall@5, inside the noise on N=40. I'd revisit if the corpus had long structured sections, which this one doesn't." That is the paragraph that gets a hire recommendation, and it takes four minutes to write.

**5. Known limitations and what breaks first.** Ranked. Be specific and quantitative: "no incremental indexing — a corpus update requires a full rebuild, which takes 40 s at 5k documents and would take ~13 minutes at 100k, so at that scale I'd switch to an alias-swap reindex." Naming your own weaknesses precisely reads as confidence; a limitations section that says "could add more tests" reads as filler.

**6. What I'd do next, in priority order, with the reason.** Three to five items, each tied to a number from your results table. This directly seeds the "what would you improve" defense question with an answer you already wrote.

**7. Time spent, honestly.** "~5.5 hours against a stated 4; the overrun was in PDF parsing." Graders overwhelmingly respect this; it's a calibration signal.

**⚠ Trap:** the AI-generated README with six emoji headers, a features table, a badge row, and no numbers in it. It's instantly recognizable, it's the most common thing in the submission pile, and it actively hurts — because the contrast between a marketing-voiced README and a thin decisions section is the exact shape of "this person's assistant wrote the docs." Write it in your own voice, put numbers in it, keep it under two pages.

### What does repo hygiene actually mean here, and is anyone really looking at my commit history?

Yes, and more carefully than candidates expect — because commit history is one of the few remaining signals about *how* the work was done rather than what it produced.

What's actually inspected:

- **Commit granularity and messages.** A single commit titled "initial commit" containing 4,000 lines tells the grader nothing and raises the obvious question. Ten to twenty commits that tell a story — `add eval harness with 40 labeled questions`, `baseline retrieval: recall@5 = 0.62`, `chunk size 512→256, recall@5 0.62→0.71`, `add BM25 fusion` — do something better than avoid suspicion: they *are* a second decisions log, and a grader reading `git log --oneline` learns your process in thirty seconds. Put the metric in the commit message when a commit moved a metric. I do this in real work and it's the cheapest documentation that exists.
- **What's committed that shouldn't be.** `.env`, `__pycache__`, `.DS_Store`, `venv/`, a 200 MB index file, and above all an API key. A committed key is treated as a security-judgment failure, not a typo, and at some companies it is disqualifying on its own.
- **Dependency management.** A pinned lockfile (`uv.lock`, `poetry.lock`, or at minimum a `requirements.txt` with `==` versions) and a stated Python version. An unpinned `requirements.txt` that doesn't resolve on the grader's machine can fail you at the correctness gate before anyone reads a line of your code.
- **Layout.** `src/` or a package directory with real module boundaries, `tests/`, `eval/`, `scripts/`. Not eleven files at the root. Not `main.py` containing everything.
- **CI.** A GitHub Actions workflow running lint (ruff), types (mypy on the core modules at least), and the offline test tier is fifteen lines of YAML and a green badge that proves the tests actually pass on a machine that isn't yours. High return for the effort.

**⚠ Trap:** squashing everything into one commit to "look clean." It reads as hiding process. The messy-but-honest history — including a commit that says `revert semantic chunking, no measurable gain` — is strictly better, because it's evidence of the iteration the defense round is going to ask you about.

**🏋 Drill (20 minutes):** clone your own submission into a fresh directory on a machine with no environment set up. Follow your README literally, typing nothing that isn't written there. Time it. Pass criterion: working system in under five minutes with zero improvisation. Most candidates fail this the first time, and it's the cheapest possible failure to fix.

### Should I record a demo video? And what makes a walkthrough good rather than exhausting?

If they ask for one, yes, and keep it to **three to five minutes**. If they don't ask, a short one is usually a small positive and never a negative — with one condition: it must not be your only explanation. A video that duplicates your README adds nothing; a video that *shows the system failing* adds a lot.

The structure I'd record:

**0:00–0:30 — the problem and what you built, one sentence each.** No preamble, no "hi, so, um, in this video."
**0:30–1:30 — a working query, end to end, live.** Show the answer with its citations, and show the trace/log output alongside it so the retrieved chunks and the token counts are visible. Seeing the internals is what distinguishes this from a screenshot.
**1:30–2:30 — the eval running, and the results table.** This is the part that differentiates you. Run it live, let it print the table.
**2:30–3:30 — one deliberate failure.** Ask a question the corpus can't answer, and show it refusing rather than fabricating. Or feed it a malformed input and show the error path. **Demonstrating that you know where the edges are is more persuasive than any number of successes**, because every submission's happy path works.
**3:30–4:00 — the one thing you'd build next and why, referencing the table.**

**⚠ Trap:** the fifteen-minute unedited screen recording where you narrate reading your own code. Nobody watches past four minutes, and the grader's actual takeaway is a judgment about your ability to communicate concisely — which is a rated competency in every one of these loops.

**🗣 Say this in the room:** "The demo's third minute is a failure case on purpose. I'd rather you see the boundary I found than only the path I tuned."

### How much architecture is the right amount? I'm told over-engineering is a named failure but under-structuring is too.

The rule I enforce, in review and in take-homes: **structure that is load-bearing for a decision you actually made is good; structure that anticipates a decision you did not make is over-engineering.** The test is a single question — *can you name the specific change this abstraction makes cheap, and is that change plausible within the assignment's stated scope?* If yes, keep it. If the answer is "well, if we ever swap vector databases," delete it.

Concretely, in a 4-hour build, the abstractions that pay for themselves:

- **A `Retriever` protocol with one or two implementations.** Load-bearing because you *did* A/B dense vs hybrid in your eval, and the protocol is what made the A/B a one-line change. That's a real decision and the abstraction earned its place.
- **A model-client interface.** Load-bearing because your tests use a recorded fake. Real decision, real payoff.
- **A settings object.** Load-bearing because the eval sweeps chunk size and k.

And the ones that don't:

- A plugin registry with entry-point discovery, for two implementations.
- A repository layer over a repository layer, for one table.
- An abstract base class with a single subclass and no second one in sight.
- Kubernetes manifests, a service mesh, a message broker, and a multi-region story, in a take-home with no eval.

**⚠ Trap (the named critical error):** unjustified over-engineering is graded harshly not because complexity is bad but because **it's read as a proxy for judgment under constraint** — the exact thing the four-hour box was constructed to measure. A candidate who builds a Kafka pipeline for a 5,000-document corpus has told the grader that they will do the same thing on the team's roadmap. And there's a compounding effect: the hours went somewhere, and it's almost always the eval that got cut.

**🗣 Say this in the room** (when challenged on a simple choice): "I used an in-memory FAISS index because the corpus is 5,000 documents — at one 1,536-dimensional float32 embedding per document that's 5,000 × 1,536 × 4 bytes ≈ 30 MB, and a few hundred MB even chunked five ways — so it fits in RAM with room to spare. The interface is a protocol with one method, so swapping to pgvector or a hosted store when the corpus outgrows a single process is a contained change — and I've written down the threshold where I'd make that swap rather than making it prematurely." That answer is unattackable: it names the number, the decision, and the trigger for revisiting it.

### If I only have time to do three things exceptionally well, which three?

Ranked by marginal effect on the hire decision, and I'd be willing to defend this ordering in a debrief:

**1. The eval, with a results table that drove at least one decision.** It's the heaviest-weighted criterion, the most-skipped, and it makes every other answer in your defense concrete. If your submission has an eval and the median submission doesn't, you are compared on a different axis than everyone else in the pile.

**2. The decisions section of the README.** Six paragraphs, each naming a rejected alternative and the reason. It converts "the code does X" into "I chose X over Y" — which is the entire difference between demonstrating implementation and demonstrating engineering. Cost: twenty minutes.

**3. One deliberately-handled failure path, visible in both the code and the demo.** Empty retrieval → refuse rather than hallucinate. Provider 429 → backoff, then a documented fallback. Malformed document → skip, log, and count, with the count surfaced. Every submission's happy path works; almost none of them show what happens at the edge, and the edge is where production lives.

Notice what isn't on the list: a frontend, a reranker, a fine-tune, a deployment, or breadth of features. Those are how you *spend* the surplus after these three, and if there is no surplus you cut them and say so in the README.

**💰 Math on where the marginal hour goes:** hour 5 spent building a React UI moves you from "no UI" to "basic UI," which affects maybe one rubric line worth ~5%. Hour 5 spent going from zero eval to a 40-example eval with a baseline table affects the 20–25% criterion *and* improves your scores on documentation, decisions, and the defense conversation — call it 35% of the total grade influenced by one hour. That is a 7× difference in return, and it is the single most important allocation decision in the entire exercise.
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

**Fast regime — roughly 10–20 days end to end.** AI-native startups and scale-ups with a small hiring committee and a founder in the loop. Some run a median close to **11 days**; some frontier labs report a median near **19 days** (**📅 Volatile:** these medians come from self-reported company figures and shift with headcount pressure — use the 10–20 day band for planning rather than quoting either number). Characterized by: recruiter replies within a day, batched onsite, decision within 48 hours of the final round.

**Medium regime — roughly 3–5 weeks.** Mid-size AI companies and well-run enterprise AI teams. A take-home with a 5-day window plus scheduling friction plus a debrief cycle.

**Slow regime — 6–10 weeks, occasionally longer.** Big tech. The interviews are not the slow part; **the queues are** — recruiter-to-manager handoff, scheduling five interviewers across time zones, hiring committee convening weekly, then leveling review, then compensation approval, then an offer letter. Amazon-style loops add a bar-raiser scheduling constraint; Google-style loops add a hiring committee that meets on a fixed cadence and can request more information, which restarts a week. Some companies additionally require **manager references before an offer is extended**, which adds days and requires you to have warned your references in advance.

**📐 The scheduling arithmetic you must do:** if you want a big-tech offer (8 weeks) and a startup offer (2 weeks) to land in the same week, **you start the big-tech process six weeks before you start the startup process.** Not the same day. This one calculation is worth more leverage than any negotiation tactic, because leverage in an offer conversation is entirely a function of holding a competing deadline, and you cannot manufacture that after the fact.

Practical sequencing for a 12-week campaign: weeks 1–2 apply to the slow regime; weeks 5–6 apply to the medium regime; weeks 8–9 apply to the fast regime; weeks 10–12 everything converges. Front-load two low-stakes practice loops in weeks 3–4 so your first real onsite isn't your first onsite of the year.

**⚠ Trap:** letting a fast company's exploding offer set the clock. If a startup gives you 72 hours and your top choice is in week 5 of 8, the honest move is to tell the startup exactly that — "I have processes in flight that conclude in three weeks; I'd like to make a decision with complete information, and I'm genuinely interested." Good companies extend. Companies that refuse to extend a two-week ask are giving you a process signal, and it's the same signal as the two-week silence: they optimize their convenience over your decision quality.

**🗣 Say this in the room** (to a recruiter, in the first call, always): "What's the typical timeline from here to a decision, and are there any steps like references or committee review that add calendar time?" Asking this in call one lets you build the sequencing plan, and it reads as organized rather than presumptuous.

### How should I actually manage the day of a five-to-seven hour onsite?

Treat it as an endurance event, because that's what it is. **Plan on the assumption that your rounds four and five will be worse than your rounds one and two unless you actively manage for it** — the mechanism is accumulated cognitive load, hunger and dehydration, and the affect carried out of whatever round went badly, not a gap in what you know. (**📅 Volatile:** treat the size of that decay as a planning heuristic, not a measured effect — I have no defensible public number for it and neither will your interviewer.) This is a solvable engineering problem and most candidates don't even model it.

The mechanics that actually matter:

- **Ask for the schedule in advance and the order of rounds.** Then decide where your hardest round falls. If the live build or the system design is round five, that's worth flagging — "would it be possible to put the build round earlier in the day?" is a reasonable ask and is granted more often than people assume.
- **Eat before, and eat *during*.** A 5-hour onsite with one 15-minute break and no food degrades the last two rounds for no reason at all, and it is entirely self-inflicted. Bring something you can eat in four minutes. If it's virtual, this is free; if it's onsite, ask about the lunch break when you get the schedule.
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
### Give me the AI-tool policy map. Who bans assistants, who requires them, and what does "banned" actually mean?

There are three regimes and you must know which one you're in *before* the round starts, because the penalty for guessing wrong is disqualification, not a lower score.

**Regime 1 — prohibited in live rounds.** A cluster of companies explicitly bans AI coding assistants during live technical interviews: **Anthropic, DeepMind, xAI, HRT**, and a range of enterprise employers including names like **Marvell** and **Wolters Kluwer** have publicly stated or enforced AI-tool prohibitions in their interview processes. The rationale is uniform: they want to measure your unaided reasoning, and an assistant makes the signal unreadable. **📅 Volatile:** these policies are being revised constantly — Anthropic in particular has publicly reconsidered its stance as its own products became standard developer tooling. Verify with your recruiter for *your* loop; do not rely on what someone posted last year.

**Regime 2 — permitted or expected.** Many AI-product companies now assume you'll use assistants, because their engineers do all day. **Cursor's onsite project is the archetype**, and the reported hidden rubric there is not just "did you build it" but "do you actually use AI coding tools fluently" — meaning if you turn off Copilot to look pure, you may be failing the round you thought you were acing.

**Regime 3 — deliberately split.** Some employers run both in the same loop: an AI-assisted round and a raw round, measuring different things. **Microsoft** has been reported as running a split of this shape (**📅 Volatile:** tool policies at every large employer are being rewritten fast — treat any named company here as an example of the pattern, not a fact about their current loop, and confirm with your recruiter). This is the most honest design and I expect more companies to copy it, because the two skills genuinely have diverged: knowing how to drive an assistant well is a real, teachable, gradable competency, and so is being able to write correct code with an empty editor.

The practical instruction is one sentence: **ask, every single time, in writing, before the round.** "Am I permitted to use AI coding assistants in this round, and if so is that expected or merely allowed?" There is no downside to asking — it reads as professional — and the downside of assuming is total.

**⚠ Trap:** treating "permitted" as "unmonitored." Even where assistants are allowed, an interviewer watching you accept a 40-line completion without reading it is forming a judgment. The signal that scores well in regime 2 is *directed* use: you know what you want, you prompt for it specifically, you read the output, you reject the parts that are wrong, and you can explain every line you kept. The signal that scores badly is generate-and-hope.

**🗣 Say this in the room** (opening a permitted-tools round): "I'll use the assistant the way I do at work — for boilerplate and for the parts I'd otherwise look up — and I'll read and explain anything I accept. Tell me if you'd rather I turn it off for any portion."

### If a round is AI-assisted and graded on tool fluency, what does "using it well" look like to a grader?

The mental model: **an assistant is a very fast junior engineer with no context and no accountability, and you are the reviewer.** Every behavior that scores well follows from that framing, and every behavior that scores badly is a failure to review.

What scores well, concretely observable in a screen share:

- **You state the intent before you prompt.** "I need a chunker that splits on headings and falls back to a token window, preserving the source offsets" — then you prompt for that. The grader sees you thinking first, tool second.
- **You give it the constraint, not just the goal.** Type signatures, edge cases, the library version. A prompt that includes "must handle empty documents and must not split inside a code fence" produces code you don't have to fix.
- **You read the output and reject parts of it out loud.** "It's using a regex for the fence detection which will break on nested fences — I'll take the structure and rewrite that function." This single behavior is the highest-scoring one available, because it proves the review is real.
- **You use it for the boring 70% and hand-write the 30% that carries the design.** Handwrite the retrieval interface and the eval scoring logic; generate the CLI parsing, the fixture plumbing, the Dockerfile.
- **You verify rather than assume.** Run it, print the intermediate, check the shape. Assistants are confidently wrong about API signatures constantly, and catching that in ten seconds rather than debugging it in ten minutes is a visible skill.

What scores badly: accepting large blocks unread; prompting the same thing four times hoping for a different answer instead of reading the error; being unable to explain a line when asked; and — the one that ends the round — the code doesn't work and you don't know why, because you never had a model of it.

**⚠ Trap:** the fluency inversion. Engineers who use assistants heavily often *lose* the ability to write a nested comprehension or a decorator from an empty file, and they don't notice because they never do it. If any company on your list is in the prohibited regime, that atrophy is a live risk and it is measurable. Test it: open a blank file, no network, and implement a token-bucket rate limiter in twenty minutes. If that's uncomfortable, you have a specific, fixable gap.

**🏋 Drill (30 minutes, twice a week, for anyone with a prohibited-regime company on their list):** assistant disabled, editor autocomplete disabled, no browser. Implement one of: a fixed-size chunker with overlap that preserves character offsets; an async worker pool with a bounded queue and graceful shutdown; a token-bucket limiter; cosine similarity top-k over a list of vectors using only the standard library and `math`. Pass criterion: correct, runs first or second try, under 20 minutes, and you can state its complexity.

### Tell me the truth about proctoring. What's actually being monitored?

More than candidates assume, and the monitoring stack has hardened substantially since remote interviewing became default. Know the surface so you don't trip it accidentally.

**What's commonly in play:**
- **Camera-on requirements**, sometimes with the requirement that your full face and often your hands or workspace remain visible for the duration. Eye-movement patterns consistent with reading a second screen are a flagged behavior — this is the one that catches honest candidates who have notes open.
- **Browser lockdown or monitored-extension environments** for automated assessments (the CodeSignal/HackerRank family). These can detect tab switches, focus loss, copy-paste events, and paste volume. A paste of 200 lines into a coding assessment is logged and reviewed.
- **Screen sharing of the full desktop**, not a window, in live rounds — which means a notification preview from a chat app is visible to your interviewer.
- **Audio analysis** on some platforms, flagging a second voice in the room or audio consistent with a text-to-speech engine.
- **Keystroke and timing analytics**: burst-typing patterns, near-instantaneous production of a complex solution after a long pause, and identical solution structure across candidates.
- **Post-hoc plagiarism and similarity checks** across the candidate pool for automated assessments.

**What is explicitly prohibited essentially everywhere**, and what candidates most often rationalize: real-time interview-assistance tools — the "invisible" overlay products that transcribe the interviewer's question and generate an answer. Using one is not a gray area. It is grounds for immediate disqualification, it is increasingly detectable, and at least some companies share findings within their recruiting networks. The same applies to live transcription tools where the company hasn't consented; recording an interview without permission is a policy violation at most companies and in some jurisdictions a legal one.

**⚠ Trap — the honest candidate's false positive.** The most common way a legitimate candidate gets flagged is behavior that *looks* like cheating: glancing repeatedly off-camera at a second monitor (even to read the problem statement), typing a long block of code you wrote earlier from a scratch file, going silent for four minutes while thinking, or having a family member walk through the room and speak. Defenses are simple and worth doing: **single monitor if you can**, announce anything unusual before you do it ("I'm going to open the docs for the client library — is that OK?"), narrate while thinking so the silence isn't unexplained, and if you must reference notes, say so and ask.

**🗣 Say this in the room** (at the start of any proctored or recorded round): "Just so there's no ambiguity — I have the problem statement on my second screen and nothing else open, and I'll narrate as I go. Let me know if you'd prefer I share my entire desktop." That sentence costs eight seconds and eliminates the entire category of misunderstanding.

### The in-person share of interviews is rising. Why, and what does it mean for me if I'm interviewing from India?

The trend is real and directionally large: the share of technical interviews conducted in person has climbed from roughly **24% in 2022 to about 38% in 2025** (**📅 Volatile:** re-verify the current figure — the trend is what matters, and it is still moving). The cause is unambiguous — remote assessment stopped producing a usable signal once assistants and interview-assistance tooling became good and ubiquitous. In-person is the crude but effective fix, and the final onsite is where it's reappearing first.

Three consequences for a candidate interviewing across time zones:

**One: the loop bifurcates.** Screens and take-homes stay remote; the final onsite increasingly does not. That changes your planning — it means the last round may require travel, a visa appointment, and a week of lead time, and it means the calendar between "passed the technical" and "offer" is longer than the fast-regime numbers suggest. Ask in the recruiter screen: "is any part of the loop in person, and if so, do you support travel?" Ask in call one, not week six.

**Two: your remote rounds are scored under more suspicion than they were three years ago.** Not personally — structurally. The base rate of tool-assisted cheating went up, so graders discount ambiguous signals more than they used to. This is the strongest practical argument for the narration habit: a candidate who thinks out loud continuously is legible in a way that a silent candidate is not, and legibility is now worth points it didn't used to be worth.

**Three: for companies that will fly you in, the onsite is a genuine advantage and you should want it.** In-person rounds are where you can be memorable, where whiteboard reasoning reads better than it does over video, and where the "would I want this person around" judgment — which is a large part of every hire decision — resolves in your favor far more easily than over a laggy call at 11pm your time.

**⚠ Trap:** assuming an in-person requirement means they won't hire remotely. Those are separate questions and they're frequently conflated by candidates who then self-select out. Plenty of companies run an in-person final and then employ you remotely or relocate you later. Ask both questions separately: "is the loop in person?" and "is the role remote-eligible?"

### Should I disclose that I used an AI assistant on the take-home? And what if I used one heavily?

Yes, disclose, in one sentence in the README, and be specific rather than apologetic. Something like: *"I used an AI assistant for boilerplate — the FastAPI scaffolding, the Dockerfile, and the test fixtures. The retrieval design, the chunking strategy, the eval set, and the scoring logic are mine, and I've noted the reasoning for each in the decisions section."*

Three reasons this is the right move. **First, most companies now assume you used one**, so disclosure costs you nothing and non-disclosure is a small integrity risk if it comes up. **Second, it's a signal of exactly the judgment they're testing** — you drew a line between the parts where a tool is a productivity multiplier and the parts where the thinking is the deliverable, and you can state where the line is. **Third, and most practically, it pre-empts the defense round's worst moment.** If the grader suspects the submission is largely generated and you haven't addressed it, every question becomes an authorship probe and the conversation goes badly. If you named it up front, the probe never starts.

The harder case is when you used one *heavily* — the whole thing is essentially generated and you steered. My honest read: that submission will usually fail, not because of the tool but because of what's missing. Generated submissions have a recognizable shape: a feature-rich, plausibly-structured system, a README with marketing voice, and **no eval, no numbers, no rejected alternatives, and no known-limitations section that's specific.** Assistants produce implementations; they don't produce the labeled eval set you built by reading twenty documents, and they don't produce "I tried semantic chunking and reverted it because it bought 0.02 recall on N=40."

So the operational rule: **use the assistant for everything below the judgment line, and make sure the artifact contains at least three things it could not have produced.** The hand-built eval set. The results table with a reverted experiment in it. The decisions section with numbers and rejected alternatives. Those three are also, not coincidentally, the three highest-weighted things in the rubric.

**⚠ Trap:** the generated test suite. Assistants write tests that assert the code does what the code does — they'll happily generate forty tests that mock the thing under test into meaninglessness. A grader reading `test_retrieve_calls_embed` and thirty-eight siblings sees test *volume* with no test *value*, and it's a worse signal than five hand-written tests, because it suggests you don't know the difference.

### What does the take-home actually cost me in API spend, and how do I keep it sane?

Worth doing the arithmetic because candidates either burn real money or, worse, under-test to save money and submit something unverified.

**💰 Math for a typical RAG take-home.** Ingestion: a 5,000-document corpus at ~600 tokens per document is 3M tokens through an embedding model; at roughly $0.02–0.13/Mtok for hosted embeddings (**📅 Volatile**) that's **$0.06–$0.40 — negligible, and you'll re-run it maybe eight times, so under $4.** The generation side is where it adds up: a 40-example eval run at ~4,000 input and 400 output tokens per example is 160k in / 16k out per run. At $3/Mtok in and $15/Mtok out: 0.16 × $3 + 0.016 × $15 = $0.48 + $0.24 = **$0.72 per eval run.** Twenty-five runs over the build is **$18**. Add an LLM judge scoring each output at ~1,200 in / 150 out — 48k in / 6k out across 40 examples, so 0.048 × $3 + 0.006 × $15 = **$0.23 per run**, about $6 over twenty-five runs. **Realistic total: $25–40 for a thorough take-home.** That is cheap enough that cost should never be the reason you under-test — and if a company hasn't provided keys, "I spent about $30 of my own API credit on this" is a perfectly reasonable line in the submission note.

Where it *does* blow up is the agentic assignment, because an agent multiplies. **💰** A 12-step agent with ~1,200 tokens added per step re-sends its history each step, so one task is roughly 94k input tokens (the 1,200 × 78 triangular sum) plus outputs — call it $0.30–0.35 per task at $3/Mtok input. A 20-task eval run is **$6–7 per run**, and if you're iterating twenty times that's **$130**. This is precisely why the step budget and the prefix-cache-friendly prompt layout aren't just production concerns — they're your own take-home budget, and mentioning that you noticed is a genuinely good line in the defense.

The controls I'd put in place on day one, all of which double as production-readiness signals:

- **Cache every model response to disk during development**, keyed on the request hash. Your fifteenth eval run over unchanged inputs should cost nothing. This is the same fixture layer you'll use for tests, so it's not extra work.
- **A hard spend counter** that aborts the run past a threshold. Ten lines, and it's the same guard you'd want in production.
- **Iterate on a 10-example subset**, run the full 40 only when you think you've improved something.
- **Use the cheap tier for iteration and the target tier for the final numbers**, and say so in the README.

**⚠ Trap:** running your dev loop against the expensive model at high concurrency with no cap, and discovering at 2am that a bug caused a retry storm. The classic version is an exception handler that retries on *all* exceptions including the one your own scoring code raises, turning a 40-example run into 4,000 calls. Cap retries, cap spend, log cumulative cost after every run.

### I got rejected after a take-home with no feedback. Is there anything to extract from that?

Yes, and this is the highest-value thing you can do in a failed pipeline, because take-home rejections are the most diagnosable failures in the whole process — the artifact still exists and you can grade it yourself against a known rubric.

**Ask for feedback once, specifically, and make it easy to give.** Not "any feedback?" — that gets a template. Instead: "Totally understand. If it's possible, I'd find it genuinely useful to know whether the gap was in the implementation, the evaluation approach, or the scoping — even one word. I'm using these to calibrate." A narrow question with three options gets an answer maybe a third of the time, and one word is enough to redirect weeks of prep.

**Then self-grade against the eight criteria**, honestly, out of 5 each, and look at where you're below 3. From experience, the distribution of real answers among strong backend engineers is lopsided: the low scores are almost always **evaluation methodology** and **documented reasoning**, essentially never code quality. If that's your pattern, the fix isn't more building — it's applying the eval-first hour to the next one.

**Look for the pattern across rejections, not within one.** A single rejection is noise; the base rate at this compensation level means strong candidates lose loops routinely. Three rejections at the *same stage* is signal, and it localizes precisely: three at the screen means positioning, three at the take-home means the artifact, three at the defense means you're building things you can't defend, three at the final round means the values or the seniority read.

**⚠ Trap:** rewriting your rejected submission to be "better" and re-submitting it elsewhere unchanged. Different companies weight differently, and more importantly a submission tuned to a previous brief will visibly miss the new one's specifics. Reuse your *harness* — the eval runner, the fixture layer, the README template, the load script — as a personal toolkit you can drop into any assignment in ten minutes. Reusing infrastructure is legitimate and smart; reusing an answer is neither.

**📐 Numbers you must know:** budget for a base rate. At the target companies on this list, a strong senior candidate should expect roughly one offer per four to six full loops entered, and fewer than half of applications convert to a first call without a referral. That number is not a comment on you; it's the market's arithmetic, and knowing it is what keeps you from over-updating on a single rejection and quitting a process that's working.

### Put it together. What's the one artifact I should build starting this week, and what's the pass bar?

Build one thing, well, and make it the thing that 40% of assignments ask for and that the other 60% share the skeleton of: **a RAG system over a corpus you chose, with a real evaluation harness, in a repository you'd be happy to hand to a hiring manager.** Not a tutorial reproduction — a corpus you actually care about, so the eval questions are ones you can write from knowledge.

The specification, which doubles as your pass bar:

- **Corpus:** 1,000–10,000 real documents with genuine messiness — PDFs, inconsistent structure, some near-duplicates. Not a clean Q&A dataset.
- **Eval:** 50+ hand-written questions with gold document IDs, stratified into single-hop, multi-hop, and unanswerable-should-refuse. Committed as JSONL.
- **Metrics:** recall@k and MRR on retrieval; groundedness and correctness on answers; p50/p95 latency and cost per query, captured in the same run.
- **Results:** a table with at least three configurations compared, including one you tried and *reverted*, with the numbers that justified reverting.
- **Engineering:** async throughout, typed, a `Retriever` protocol with two implementations, recorded-fixture tests plus a smoke eval in CI, config from environment, structured logs with a trace ID, a token and cost counter, and a documented degradation path per dependency.
- **README:** run instructions that work from a fresh clone in under five minutes, results above the fold, one architecture diagram, eight decision entries each naming a rejected alternative, ranked known limitations with numbers, and next steps tied to the table.

**The pass bar, stated as an exam you can administer to yourself:** a senior engineer who has never seen your repo can clone it, run it, understand the architecture, and find the number that justifies your chunk size — in fifteen minutes, using only what's in the repository. If they have to ask you a question to get there, you haven't finished.

**🏋 Drill (the capstone, one week):** build it. Then, seven days after you finish, without re-reading anything, sit down with a timer and answer these ten aloud in under 90 seconds each: why that chunk size; why that k; why that embedding model; what your recall@5 is and what's in the miss cases; what happens when retrieval returns nothing; what happens when the provider 429s; what your p95 is and which component dominates it; what a query costs and how that scales to 1M queries/month; what breaks first at 100× the corpus; what you'd build next and what number would tell you it worked. Pass criterion: ten out of ten, no hedging, every quantitative answer a real number from your own table. **That is the interview.** Everything else in the loop is a variation on those ten questions, and if you can answer them about a system you actually built, you will not be surprised in the room.

**🗣 Say this in the room, to close a take-home defense:** "The thing I'd want you to take from this is the order I built it in. I built the measurement before the system, so every choice in here has a number behind it or an explicit note saying I didn't check and why. That's how I'd want to work on your team, and it's how I'd want to be held to account."
