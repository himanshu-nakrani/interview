### Our support chatbot keeps giving wrong answers about our own product. My VP wants to fine-tune on our docs. What do you say in that meeting?

I say: fine-tuning is the wrong tool for that specific symptom, and I can tell you in one sentence why. Wrong answers about *our product* is a **knowledge** failure. Fine-tuning changes the model's *behavioural prior* — what shape of output it produces, in what register, with what default structure. It is a very poor mechanism for installing facts, and when you push facts through it, the failure mode is not "the model still doesn't know" but "the model now confidently asserts a plausible-looking wrong answer," which is strictly worse than the bug we started with. So we would spend six weeks and land somewhere worse than today.

Then I redirect, because "no" is not an answer. The mental model I want the room to leave with is a **ladder**, and the rule is that you climb it one rung at a time and you never skip a rung you have not measured. Rung one is the prompt and the context you assemble. Rung two is retrieval — actually putting the right doc in front of the model. Rung three is tool design. Rung four is constraining the output. Rung five is routing and model choice. Rung six is distillation. Rung seven, last, is fine-tuning. Every rung below fine-tuning is *reversible in an afternoon* and *observable in an eval run*; fine-tuning is a checkpoint you now own, version, re-train on every model upgrade, and cannot roll back with a config flag.

For this specific symptom I would bet money the fix is on rung two, and I would prove it in a day with the oracle-context experiment: take the fifty failing tickets, hand-paste the correct doc section into the prompt, and re-run. If the model answers correctly with the right document in context, the model is fine and your *retrieval* is broken. That is a chunking, embedding, or reranking bug, and it costs a sprint, not a quarter.

**🗣 Say this in the room:** "Wrong facts is a retrieval problem until proven otherwise. Fine-tuning teaches form, not facts — and training on facts the model doesn't already hold has been shown to *increase* hallucination rate on the facts it did hold. Before I'd approve a fine-tune I'd want the oracle-context ablation showing the model gets it right when the right document is in the window. If it does, we have a retrieval bug and we're about to spend six weeks fixing the wrong layer."

**⚠ Trap:** the single most common AI-Engineer rejection in the industry is reaching for fine-tuning by reflex. Interviewers at Sierra, Harvey, Glean and every big-tech applied team plant this question deliberately. It is not a knowledge test — it is a *judgment* test, and answering "we'd fine-tune on their support corpus" is close to an automatic no-hire even if the rest of your loop was strong, because it tells them you will burn a quarter of their engineering budget on the wrong layer.

### Lay out the escalation ladder for me, rung by rung, and tell me what each rung actually fixes.

The ladder in order, and — this is the part that matters — the *failure class* each rung is the correct treatment for:

**1. Better prompt and context.** Fixes: the model misunderstood the task, the output format is inconsistent, it doesn't know the constraints, it doesn't know the current date, it doesn't have the tacit rules a new hire would be told on day one. Iteration latency: minutes. Cost to try: one eval run.

**2. Better retrieval.** Fixes: the model doesn't have the fact. Includes chunking strategy, hybrid lexical+dense search, reranking, query rewriting, and — most underrated — simply retrieving *more* and letting a long-context model sort it out. Iteration latency: hours to days. Reversible.

**3. Better tool design.** Fixes: the capability is *verifiable and executable* — arithmetic, current inventory, a SQL aggregate, a code run, a date computation, an ACL check. Anything where a deterministic function gives the right answer, the model should be calling that function, not approximating it in weights. Iteration latency: days.

**4. Structured-output constraints.** Fixes: downstream parse failures, schema drift, invalid enum values, missing required fields. Constrained decoding / grammar-guided generation removes an entire class of production incident. Iteration latency: hours.

**5. Routing and model choice.** Fixes: cost and latency, and sometimes quality — a bigger model on the hard 20% and a cheap one on the easy 80%. Also includes "use the reasoning model with a thinking budget for this subtask." Iteration latency: days.

**6. Distillation.** Fixes: cost and latency on a pipeline whose *quality is already good enough*. You are compiling a working prompted system into a smaller one. Requires a teacher that already passes your eval. Iteration latency: weeks.

**7. Fine-tuning.** Fixes: tacit style and form the prompt cannot express, a long-tail output format, domain-specific decision boundaries with thousands of labeled examples behind them, and killing a 6,000-token system prompt for latency and cost. Iteration latency: weeks to months, plus permanent ownership.

The framing I use out loud: **rungs 1–5 change configuration; rung 6–7 change artifacts.** Configuration you can revert with a deploy. Artifacts you maintain forever.

**⚠ Trap:** people collapse rungs 1 and 2 into "prompt engineering" and then say "we tried prompt engineering, it didn't work." Those are different failure classes with different fixes. "We tried prompting" almost always means "we tried three system-prompt rewrites and never touched chunk size or added a reranker."

### Why that order specifically? Defend the ordering to someone who thinks it's arbitrary.

It is not arbitrary — it is sorted by a single quantity: **cost of being wrong**, which is roughly iteration latency multiplied by irreversibility.

Think of it exactly the way you'd think about a production database change. A config flag, a feature flag, a schema migration, a data migration, a re-shard — you order interventions by blast radius and rollback cost, not by how clever they are. Same discipline here. A prompt change is a config flag: you ship it, you watch the eval, you revert in thirty seconds if it regresses. A fine-tune is a re-shard: weeks of lead time, a new artifact in your registry, a serving path that now differs from everyone else's, and a rollback that means "go back to the previous checkpoint and re-run all of your evals."

The second ordering principle is **information gain per unit of effort**. Each lower rung teaches you something that makes the higher rungs cheaper if you eventually need them. Building the eval harness for rung 1 is the same harness you need to justify rung 7. Fixing retrieval gives you the labeled (query, correct-doc) pairs you'd need for a fine-tune anyway. Building the router at rung 5 tells you exactly which slice of traffic is hard, which is exactly the slice you'd distill or fine-tune for. Nothing on the lower rungs is wasted if you end up climbing. The reverse is emphatically not true: a fine-tune done before you fixed retrieval is thrown away when you fix retrieval, because your training data encoded the broken context format.

Third: **the lower rungs compose, the top rung doesn't.** You can have a better prompt *and* better retrieval *and* tools *and* routing simultaneously, and their gains are largely additive. A fine-tune is a fork in your artifact tree. Two fine-tunes for two purposes means two checkpoints, two eval suites, two serving lanes, or a merge that degrades both.

**🗣 Say this in the room:** "I order the ladder by rollback cost, not by sophistication. Everything below distillation is a config change I can revert in a deploy; everything at or above it is an artifact I own forever, including through every model upgrade. So I want proof — a measured plateau on a stable eval — before I take on permanent ownership."

### Give me the diagnostic. I hand you 200 failing production traces — how do you decide which rung the fix lives on?

I do not start by reading them all; I start by *classifying* them, because the whole game is turning 200 anecdotes into four numbers.

**🔍 Failure taxonomy — the decision procedure I actually run.** Sample 100 failures stratified by request type (never take the first 100 — they're skewed by whatever broke that morning). For each, answer four questions in order and stop at the first "no":

1. **Was the necessary information in the model's context window?** Check the actual assembled prompt, not the intent. If no → **retrieval bug (rung 2)**. This is the single largest bucket in every product system I have worked on; expect 40–60% of failures here.
2. **Given that the information was present, did the model use it correctly?** If it had the right doc and still answered wrong → now split further. Was the doc buried at position 40 of 50 retrieved chunks? That is still rung 2 (reranking / context ordering), and it is the "lost in the middle" effect, not a model deficiency.
3. **Was the failure a capability the model shouldn't be doing in its head at all?** Arithmetic off by a rounding, a stale date, a total that doesn't match the database. → **tool problem (rung 3)**.
4. **Did the pipeline crash, drop a field, or produce an unparseable blob?** → **structured-output problem (rung 4)**, and it should never have reached a quality review in the first place.
5. **Is the answer factually correct, complete, grounded — and still wrong for us?** Wrong tone, wrong register, wrong escalation decision, didn't follow the unwritten policy, over-hedged, too long, wrong idiom for the domain. → *now* you are in weights territory: **rung 5–7**.

That last bucket is the only one that justifies training. In my experience it is 5–15% of failures in a system that has never been tuned, and here is the kicker: about half of *that* bucket dissolves when you add three good few-shot examples, because "wrong register" is exactly what few-shots are for.

The output of this exercise is a table: `retrieval 52 / tool 14 / format 9 / genuinely-behavioural 18 / ambiguous-or-bad-label 7`. That table is the artifact you bring to the design review, and it is what converts "the VP wants a fine-tune" into "the data says 52% of our failures are a chunking bug." I have never seen a strong candidate skip this step, and I have never seen a weak one do it.

**⚠ Trap:** the "ambiguous-or-bad-label" bucket. If more than ~10% of your sampled failures are cases where two reasonable humans would disagree about the right answer, your *eval* is the broken component and every number downstream of it is noise. Fix that before you touch the model. Teams routinely fine-tune to chase an eval delta that was inside their own label noise.

### Take rung one seriously for a minute. What is actually still on the table in "write a better prompt" for a team that already thinks their prompt is good?

Almost always a lot, because "better prompt" is a bad name for what is actually **context engineering** — deciding what tokens are in the window, in what order, in what form, at what cost. The prompt string is maybe a third of that.

Concretely, what I look for when a team tells me their prompt is maxed out:

**Few-shot examples that are actually representative.** Most teams have three examples chosen when the feature was prototyped, all easy, all the same shape. Replacing them with 8–15 examples *mined from the failure set* is the highest-yield hour in this entire ladder. This is why LIMA matters as a reference: form and register are learned from a small number of high-quality demonstrations, and the prompt is a place you can put demonstrations for free.

**Task decomposition.** One prompt doing extract-then-decide-then-write is three prompts pretending to be one. Splitting it lets you eval each stage independently and usually lets you route the easy stages to a cheap model.

**Explicit negative space.** "If the retrieved documents do not contain the answer, say so and list what you'd need" is one line, and it converts a hallucination into an abstention. Teaching abstention through the prompt is the cheapest reliability win available.

**Order and cache-alignment.** Static content first (system prompt, tool schemas, few-shots), volatile content last. That is a quality-neutral change that makes the prefix cacheable and cuts input cost by ~90% on the static portion — so it pays for itself before it improves anything.

**Format of the context, not just the content.** Retrieved chunks as XML-tagged blocks with a source id and a date, rather than concatenated prose, measurably improves both grounding and citation accuracy, and it makes "which chunk did it use" observable.

**📄 Paper:** Zhou et al. (2023), *LIMA: Less Is More for Alignment* — 1,000 curated examples produced competitive instruction-following, supporting the superficial alignment hypothesis: alignment mostly teaches *format and style*, while knowledge comes from pretraining. The reason it belongs in a prompting answer is the corollary — if form is what's learnable from a handful of examples, put those examples in the prompt first and see if you still need weights.

**💰 Math:** reordering for prefix caching, using $3/Mtok uncached input and a 90% cache-read discount ($0.30/Mtok) as an illustrative frontier price: a 6,000-token static preamble costs 6,000 × $3/1e6 = **$0.018/call** uncached and 6,000 × $0.30/1e6 = **$0.0018/call** cached. At 300,000 calls/day that is $5,400/day vs $540/day — **$146k/month saved by moving one timestamp out of the system prompt.** **📅 Volatile:** verify current per-token and cache-read pricing before your loop; the ratio (~10×) has been more stable than the absolute numbers.

### How do you know you've exhausted retrieval? What's the experiment?

You run the **oracle-context ablation**, and it is the single most useful experiment in applied AI. It separates "the retriever failed" from "the generator failed" with one number, and it takes an afternoon.

Take 100–200 failing queries. For each, have a human (or a strong model with the full corpus, if you must) identify the chunk or document that *does* contain the answer. Now run the generation step twice: once with your real retriever's output, once with the oracle chunk pasted in at the top of the context. Three outcomes:

- **Oracle fixes it, real retrieval doesn't** → your ceiling is retrieval. Go work on chunking, hybrid search, reranking, query expansion. Nothing about the model is your problem. This is the majority case.
- **Oracle doesn't fix it either** → the model has the information and still gets it wrong. Now you are looking at reasoning, instruction-following, or format — rungs 3–7.
- **Real retrieval was already fine and the model ignored the right chunk** → context ordering / distraction. Fewer, better-ranked chunks usually beats more chunks. This is where a reranker earns its 80ms.

Alongside that, I want the standard retrieval metrics decomposed, because "RAG is bad" is not a diagnosis: **recall@k** of the gold chunk (is it in the candidate set at all?), and then **rank of the gold chunk** after reranking. If recall@50 is 0.94 but recall@5 is 0.61, you do not have a retrieval problem, you have a *ranking* problem, and a cross-encoder reranker fixes it for one added network hop.

The honest ceiling statement: retrieval is exhausted when recall@k is above ~0.95 on your gold set, the gold chunk lands in the top 3 after reranking, and the oracle ablation shows no further headroom. Until all three are true, "we should fine-tune" is not a supportable claim — you are proposing to spend six weeks on the 5% while leaving 40% on the table.

**⚠ Trap:** measuring retrieval quality by end-to-end answer score only. The end-to-end number confounds retriever and generator, so it moves when either changes and tells you nothing about which. I enforce in review that any RAG system reports retrieval metrics *separately* from generation metrics. Without that split you cannot climb this ladder at all — you are guessing which rung you're on.

**🏋 Drill:** given a RAG system and 50 known-bad queries, produce the recall@5 / recall@50 / oracle-ablation table in under 90 minutes and state which rung the fix lives on. Pass criterion: your rung assignment matches what a full week of investigation would have concluded, and you can name the specific next change (chunk size, reranker, query rewrite) rather than "improve retrieval."

### Explain why tool design is a rung on this ladder at all. Isn't a tool just a workaround for a weak model?

No — it is the opposite, and this is a framing I will defend hard. A tool is how you move a subproblem from a *probabilistic* system to a *deterministic* one. That is not a workaround; that is the entire history of reliable computing. You would never ask a model to compute a checksum, and you should feel the same way about asking it to total an invoice or recall today's inventory.

The rule I use: **if the capability is expressible as a function whose output you could unit-test, it belongs in a tool, not in weights.** Arithmetic, date math, currency conversion, an aggregate over a table, executing code, looking up an account balance, checking whether user U may read document D. Training a model to do these approximately is spending millions of parameters to get a worse answer than `sum()`.

The reason it sits at rung 3, above retrieval, is that tool design *subsumes* a lot of what people try to fix with retrieval or training. A team fighting to make the model remember 40,000 SKUs is solving with embeddings a problem that a `search_catalog(query, filters)` tool solves exactly. A team fine-tuning so the model "knows our escalation policy" often just needs a `get_policy(customer_tier, issue_type)` call that returns the current policy — which also means the policy can be edited by the ops team on a Tuesday without a training run.

Where tool design becomes real engineering — and this is where interviewers at agent-heavy companies push — is that the model's performance is dominated by the *interface*, not the implementation. Twelve overlapping tools with vague descriptions produce worse behaviour than four tools with sharp, mutually-exclusive contracts. Returning a 4,000-token raw API payload burns context and buries the answer; returning a 200-token normalized summary with a handle for drill-down does not. Error strings matter enormously: `{"error": "invalid_date_format", "hint": "use YYYY-MM-DD", "you_sent": "03/04/25"}` produces a successful retry; `500 Internal Server Error` produces a hallucinated retry or an infinite loop.

**🗣 Say this in the room:** "Before I consider changing the model, I ask whether the thing it's failing at is verifiable. If it is, it's a tool — deterministic, unit-testable, editable by non-engineers, and observable in traces. Most of what people want to teach a model is really an API they haven't written yet."

**⚠ Trap:** adding a tool without adding the *decision* of when not to use it. A model given a search tool will search when it already knows the answer, tripling latency and cost. Tool descriptions need explicit negative guidance ("do not call this for questions answerable from the conversation") and your eval needs a no-tool-needed slice, or you will ship a system that makes four API calls to answer "hello."

### Structured-output constraints are the fourth rung. What class of failure does constrained decoding actually eliminate, and what does it leave completely untouched?

It eliminates exactly one class and it eliminates it completely: **syntactic invalidity**. If you constrain generation to a grammar derived from your JSON schema, the sampler physically cannot emit a token that would make the output unparseable. Mechanically, at each decode step you compute the set of tokens permitted by the current state of a finite automaton compiled from the grammar, set the logits of every other token to −inf, then sample normally. The model's *relative* preferences among legal tokens are preserved; illegal ones have probability zero rather than small-but-nonzero. So your JSON-parse-failure rate goes to zero, not to 0.3%.

That is a bigger deal than it sounds. A 0.3% parse-failure rate at 500k calls/day is 1,500 failed requests a day, each triggering a retry that doubles your token spend on that request and adds a full round-trip of latency. Killing it with a grammar is a one-line config change on most engines. This is why it is rung 4 and not rung 7.

What it leaves completely untouched: **semantic correctness**. A schema-valid object can carry a hallucinated invoice number, a wrong enum choice, a plausible-but-false date. Constrained decoding guarantees shape, never truth. I have seen teams ship structured outputs and declare the hallucination problem solved because the error rate in their logs went to zero — the errors moved from "parse exception" (loud, alerting, counted) to "wrong value in a valid field" (silent, uncounted, in front of a customer). That is a monitoring regression disguised as a quality win.

**⚠ Trap:** the grammar can force the model to lie. If your schema says `category` is an enum of eight values and the true answer is "none of these," constrained decoding will make the model pick one of the eight with full confidence, because "I don't know" is not in the automaton. Every enum in a production schema needs an explicit `"unknown"` / `"other"` member and every required field needs a nullable escape, or you have engineered a system that cannot express uncertainty. I treat a schema with no abstention path as a review blocker.

**📄 Paper:** Willard & Louf (2023), *Efficient Guided Generation for Large Language Models* — reframed regex/CFG-constrained decoding as an index over the vocabulary and the automaton's states, making mask construction O(1) per step rather than a scan of the vocabulary; this is the mechanism behind Outlines and the family of engine-level guided-decoding implementations that followed.

A second, subtler cost: constrained decoding and prefix-cache-friendly prompting interact fine, but constrained decoding and *chain-of-thought* do not, if you force the model into JSON from token one. The fix is to let it think in free text inside a designated field or a preceding block and only constrain the final structured segment. Forcing structure over the whole generation measurably reduces reasoning quality on hard items, because you have removed the model's scratch space.

### Rung five is routing and model choice. Design me a router for a high-volume product feature and show me the money.

The mental model: a router is a **cache with a quality dimension**. You already know the pattern — serve the cheap path when it's sufficient, escalate on miss. The only new thing is that "miss" is not a boolean lookup, it is a confidence judgment, so the interesting engineering is in the escalation predicate, not the routing table.

The design I would actually ship for, say, a support-triage feature at 2M requests/month:

**Tier 0 — no model at all.** Exact-match and normalized-query cache, plus deterministic rules for the top intents. In most support workloads 10–25% of traffic is duplicate or template-matchable. Cost: ~$0.
**Tier 1 — small model, constrained output.** Handles classification, extraction, canned-response selection. Escalates when (a) the schema's confidence field is below threshold, (b) the classifier chose `other`, or (c) retrieval returned nothing above the score floor.
**Tier 2 — frontier model with full context and tools.** The hard tail.
**Tier 3 — human.** Non-negotiable for anything touching money, legal, or account deletion.

The escalation predicate is where candidates get separated. Do **not** ask the small model "are you confident?" — self-reported confidence from a small model is close to useless and is itself a learned style. Use signals that are external to the answer: retrieval score distribution, whether the top-2 retrieved chunks disagree, output length anomalies, tool-call failure, and — the strongest one — a cheap *verifier* pass (does the answer's every claim appear in the retrieved context?). A second small-model call as a verifier costs a fraction of one frontier call and is the highest-precision escalation signal available.

**💰 Math (illustrative prices — 📅 Volatile, re-verify):** frontier at $3/Mtok in, $15/Mtok out; small model at $0.25/Mtok in, $1.25/Mtok out. Request shape: 5,000-token static preamble (cacheable at 10%), 800-token dynamic context, 150 output tokens.

- All-frontier, cached preamble: (5,000 × $0.30 + 800 × $3 + 150 × $15)/1e6 = ($1,500 + $2,400 + $2,250)/1e6 = **$0.00615/req**. At 2M/month: **$12,300/month**.
- Routed: 20% tier-0 at $0; 56% tier-1 at (5,000 × $0.025 + 800 × $0.25 + 150 × $1.25)/1e6 = ($125 + $200 + $187.50)/1e6 = **$0.00051/req**; 24% tier-2 at $0.00615 plus the tier-1 attempt they already paid for ($0.00051).
- Blended: 0.20×0 + 0.56×$0.00051 + 0.24×($0.00666) = $0 + $0.000286 + $0.001598 = **$0.00188/req** → **$3,770/month**.

That is a **69% cost reduction for roughly one to two engineer-weeks**, with a rollback that is a feature flag. Compare that to a fine-tune's six weeks and permanent ownership, and you see why routing sits below distillation on the ladder. The honest caveat: you pay ~24% of requests twice (tier-1 attempt then tier-2), and you add tier-1's latency to every escalated request — typically 200–400ms — so if your p95 latency SLO is tight, cascade escalation is the wrong shape and you want a *predictive* router (a small classifier on the input that picks the tier up front) instead of a cascade.

### Why is distillation a separate rung from fine-tuning? They both end in a gradient step.

Because the *precondition* is completely different, and the precondition is the whole ladder.

Distillation begins from a system that **already passes your eval**. You have a prompted frontier pipeline hitting 87% on your suite, and the only remaining problem is that it costs $0.02 a call and takes 3 seconds. Distillation compiles that working system into a smaller one: you generate supervision from the teacher on your real input distribution, train a student, and accept a small quality loss for a large cost and latency win. The quality target is known in advance and the ceiling is the teacher. Risk is bounded — worst case, the student doesn't reach the bar and you keep serving the teacher.

Fine-tuning in the sense that gets people rejected begins from a system that **does not pass your eval**, and proposes to close the gap with gradients. You do not know the ceiling, you do not know if your data supports it, and the failure mode is silent: the model learns your training set's surface form and your eval — if it shares provenance with the training data — goes up while production goes sideways.

Put differently: distillation is a **cost transformation on a solved problem**. Fine-tuning-for-capability is a **bet on an unsolved one**. Those deserve different approval bars, so they are different rungs.

There is also a labour asymmetry worth stating. Distillation's data is nearly free — you already have production traffic, and the teacher generates the labels. A capability fine-tune's data is human-labeled or verifier-generated, which is where the real money and the real calendar time go: 5,000 human labels at $2 each is $10,000 and three weeks of vendor coordination, versus 5,000 teacher completions at $0.02 each is $100 and an afternoon.

**🗣 Say this in the room:** "I separate them by precondition. Distillation requires a teacher that already clears the bar — it's a cost move on a solved problem, and the ceiling is known. Fine-tuning for capability is a bet that gradients will close a gap I couldn't close with context, and I want a plateau measurement and a stable eval before I take that bet."
