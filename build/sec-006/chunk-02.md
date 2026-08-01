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

**🗣 Say this in the room** (when challenged on a simple choice): "I used an in-memory FAISS index because the corpus is 5,000 documents, which is 30 MB of vectors, and it fits in RAM with room to spare. The interface is a protocol with one method, so swapping to pgvector or a hosted store when the corpus outgrows a single process is a contained change — and I've written down the threshold where I'd make that swap rather than making it prematurely." That answer is unattackable: it names the number, the decision, and the trigger for revisiting it.

### If I only have time to do three things exceptionally well, which three?

Ranked by marginal effect on the hire decision, and I'd be willing to defend this ordering in a debrief:

**1. The eval, with a results table that drove at least one decision.** It's the heaviest-weighted criterion, the most-skipped, and it makes every other answer in your defense concrete. If your submission has an eval and the median submission doesn't, you are compared on a different axis than everyone else in the pile.

**2. The decisions section of the README.** Six paragraphs, each naming a rejected alternative and the reason. It converts "the code does X" into "I chose X over Y" — which is the entire difference between demonstrating implementation and demonstrating engineering. Cost: twenty minutes.

**3. One deliberately-handled failure path, visible in both the code and the demo.** Empty retrieval → refuse rather than hallucinate. Provider 429 → backoff, then a documented fallback. Malformed document → skip, log, and count, with the count surfaced. Every submission's happy path works; almost none of them show what happens at the edge, and the edge is where production lives.

Notice what isn't on the list: a frontend, a reranker, a fine-tune, a deployment, or breadth of features. Those are how you *spend* the surplus after these three, and if there is no surplus you cut them and say so in the README.

**💰 Math on where the marginal hour goes:** hour 5 spent building a React UI moves you from "no UI" to "basic UI," which affects maybe one rubric line worth ~5%. Hour 5 spent going from zero eval to a 40-example eval with a baseline table affects the 20–25% criterion *and* improves your scores on documentation, decisions, and the defense conversation — call it 35% of the total grade influenced by one hour. That is a 7× difference in return, and it is the single most important allocation decision in the entire exercise.
