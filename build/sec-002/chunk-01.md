### You've got eight weeks and a full-time job. Walk me through how you'd decide what *not* to study.

The mental model is capacity planning, and you already run it every quarter: you do not add a feature to a saturated service, you find the thing already consuming the budget and delete it. Prep time is the saturated service. Almost every backend engineer who fails an AI loop failed it while working hard — they spent week three re-reading asyncio internals because that felt productive and safe, and arrived at the onsite unable to write multi-head attention on a whiteboard. **Effort was never the constraint. Allocation was.** So the first artifact you build is not a study schedule, it is a subtraction list.

The decision rule I use has exactly three tests, applied per topic:

1. **Does this topic appear in the loop at all?** If no interviewer will ask it, it is not prep, it is comfort. Generic behavioral STAR is the cleanest example — the archetypes you are targeting have replaced it with values rounds and customer-conversation simulations that STAR answers actively fail.
2. **If it appears, do you already clear the bar cold?** Not "do you know it" — *do you clear the bar under time pressure, unaided, at the depth this level demands*. That is what the 🧪 gates in this section are for. Three questions per skip, answered correctly out loud, or the skip is revoked.
3. **Is there a delta between your version and the AI version?** This is where most engineers get it wrong in both directions. Celery is a skip; *cancelling a 90-second streaming generation whose consumer disconnected* is not, because it has no Celery analogue. K8s is a skip; GPU device plugins, MIG partitioning and node-level fragmentation are not. The skip is almost never total — it is "skip the substrate, study the six-inch delta on top."

**📐 Numbers you must know:** this guide is 3,046 questions. The SKIP list removes roughly 350 of them — 11.5% of the corpus. At the mastery path's rate of 30 questions/day, 350 questions is 11.7 days. On an 8-week (56-day) runway that is **21% of your total prep time**, recovered and redirected into transformer internals, serving, evaluation, post-training and agent harnesses. That is the entire economic argument for this section: one-fifth of your runway, free, for the cost of one honest afternoon of self-assessment.

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

**📐 Numbers you must know:** weights in bytes = `params × bytes_per_param`, and bf16/fp16 is 2 bytes, so **the GB figure is just twice the parameter count in billions**. 8B → 16 GB, 70B → 140 GB, 405B → 810 GB. Halve it for fp8, quarter it for int4. Set that against 80 GB per H100-class card and the deployment shape falls straight out: 8B fits comfortably on one card with room for cache, 70B needs two in bf16 or one at 4-bit, 405B needs a full 8-GPU node and tensor parallelism. You should be able to go from a model name to a GPU count in about four seconds, because that is the pace at which this gets asked.

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

**What comes out.** Across the eight skips — asyncio, GIL/free-threading, Pydantic mechanics, SQLAlchemy/Alembic, Celery fundamentals, Docker/K8s basics, REST/gRPC design, generic STAR — the guide's own accounting is roughly 350 questions of the 3,046. But the honest number is larger than that, because these topics are also where you would have *voluntarily* over-invested: they are comfortable, they produce a feeling of progress, and they have infinite depth available. Call the recovered budget 350 questions of reading plus another 10–15 hours of comfort-rereading you now do not do.

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
