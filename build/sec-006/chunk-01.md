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

**💰 Math on the real cost to you:** a 4-hour-stated assignment at the true 8 hours, times an average of 2.5 take-homes in a live pipeline, is 20 hours. Add defense prep at 1.5 hours each — 3.5 more. That's ~24 hours, or three full working days, spent per pipeline. If you're running two pipelines a quarter while employed, that's six days of evenings and weekends. Budget it explicitly on a calendar or it will silently eat your systematic prep, which is the actual failure mode: candidates arrive at the onsite having done three take-homes and zero transformer-internals review.

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

**3. Report agreement with a chance-corrected statistic, not raw accuracy.** Raw agreement is inflated when the label distribution is skewed: if 80% of outputs are "good," a judge that says "good" every time scores 80%. Report Cohen's kappa (for a single human reference) or the confusion matrix in full. A kappa of 0.4 on a 5-point scale means your judge is a coin flip with extra steps, and saying so is a *pass*, not a fail.

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
