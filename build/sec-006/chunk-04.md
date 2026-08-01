### Give me the AI-tool policy map. Who bans assistants, who requires them, and what does "banned" actually mean?

There are three regimes and you must know which one you're in *before* the round starts, because the penalty for guessing wrong is disqualification, not a lower score.

**Regime 1 — prohibited in live rounds.** A cluster of companies explicitly bans AI coding assistants during live technical interviews: **Anthropic, DeepMind, xAI, HRT**, and a range of enterprise employers including names like **Marvell** and **Wolters Kluwer** have publicly stated or enforced AI-tool prohibitions in their interview processes. The rationale is uniform: they want to measure your unaided reasoning, and an assistant makes the signal unreadable. **📅 Volatile:** these policies are being revised constantly — Anthropic in particular has publicly reconsidered its stance as its own products became standard developer tooling. Verify with your recruiter for *your* loop; do not rely on what someone posted last year.

**Regime 2 — permitted or expected.** Many AI-product companies now assume you'll use assistants, because their engineers do all day. **Cursor's onsite project is the archetype**, and the reported hidden rubric there is not just "did you build it" but "do you actually use AI coding tools fluently" — meaning if you turn off Copilot to look pure, you may be failing the round you thought you were acing.

**Regime 3 — deliberately split.** **Microsoft** runs both: an AI-assisted round and a raw round, in the same loop, measuring different things. This is the most honest design and I expect more companies to copy it, because the two skills genuinely have diverged: knowing how to drive an assistant well is a real, teachable, gradable competency, and so is being able to write correct code with an empty editor.

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

**💰 Math for a typical RAG take-home.** Ingestion: a 5,000-document corpus at ~600 tokens per document is 3M tokens through an embedding model; at roughly $0.02–0.13/Mtok for hosted embeddings (**📅 Volatile**) that's **$0.06–$0.40 — negligible, and you'll re-run it maybe eight times, so under $4.** The generation side is where it adds up: a 40-example eval run at ~4,000 input and 400 output tokens per example is 160k in / 16k out per run. At $3/Mtok in and $15/Mtok out: 0.16 × $3 + 0.016 × $15 = $0.48 + $0.24 = **$0.72 per eval run.** Twenty-five runs over the build is **$18**. Add an LLM judge scoring each output at ~1,200 in / 150 out: another $0.20 per run, $5 total. **Realistic total: $25–40 for a thorough take-home.** That is cheap enough that cost should never be the reason you under-test — and if a company hasn't provided keys, "I spent about $30 of my own API credit on this" is a perfectly reasonable line in the submission note.

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
