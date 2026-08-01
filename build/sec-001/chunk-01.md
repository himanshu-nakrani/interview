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

The mental model I hold is an escalation ladder ordered by cost-to-try, and I climb it in order: (1) prompt and context assembly, (2) retrieval quality, (3) tool design and structured output contracts, (4) model routing — send the hard 10% to a stronger model, (5) distillation into a smaller model for cost, (6) parameter-efficient fine-tuning, (7) full fine-tuning. Each rung costs roughly an order of magnitude more engineering time than the one below it, and — this is the part candidates miss — each rung *adds a permanent maintenance obligation*. A fine-tune is not a change, it is an asset: you now own a checkpoint, an eval to prevent regression, a retraining trigger when the base model deprecates, and a deployment surface.

**💰 Math:** Compare interventions on a task doing 500k calls/month with an 8k-token prompt and 500-token output. At $3/Mtok input and $15/Mtok output, that is (8,000 × 3 + 500 × 15) / 1,000,000 = $0.024 + $0.0075 = $0.0315 per call, so 500,000 × $0.0315 = **$15,750/month**. A prompt/context fix costs a few engineer-days and can be reverted in one deploy. A distillation to a self-hosted 8B might cut inference to roughly $1,500–$3,000/month of GPU — real money — but costs a data-generation run, a training run, an eval harness, and a serving stack you now operate; call it 4–8 engineer-weeks up front plus ongoing ownership. At a $15.7k/month bill you are not saving your way to a fine-tune decision — you would need either a quality reason or 5–10× that volume for the payback to be obvious. **📅 Volatile:** per-token prices have been falling roughly an order of magnitude per generation-and-a-half; re-run this arithmetic with current numbers before you quote it.

The preconditions I'd name before agreeing to fine-tune: the failures are capability failures not context failures (verified by error analysis); you have ≥1,000 high-quality task-specific examples or a way to generate them; the task has a stable output format or style the base model resists; you have an eval that will catch regression; and either cost at your volume or latency justifies owning a serving stack. If I can't check most of those boxes, the answer is no.

**🗣 Say this in the room:** "Fine-tuning is the last rung, not the first. I'd want error analysis showing these are capability failures rather than context failures, a thousand-plus clean examples, and a regression eval before I'd sign up for owning a checkpoint — because a fine-tune isn't a change, it's a permanent asset with a maintenance cost."
