### Our corpus is 400,000 tokens and fits in the window. Make the case for and against just stuffing it, with numbers.

This is the honest edge case where long context is genuinely competitive, so I would do the arithmetic rather than reflexively defending retrieval.

**The stuffing architecture.** Every request sends the full 400k-token corpus plus the question. At $3/Mtok input: `400,000 × 3e-6 = $1.20` per query, uncached. Output at 500 tokens × $15/Mtok = `$0.0075`. **$1.21 per query.** TTFT: prefilling 400k tokens at 50–100 ms per 1k is **20–40 seconds**. Unusable interactively, at any price.

**With prompt caching**, and only if the corpus is byte-identical across requests: cache reads at $0.30/Mtok give `400,000 × 3e-7 = $0.12` per query — a 10× improvement — and TTFT drops to whatever the cached-prefix path costs, which is typically a small fraction of the prefill but is not zero and is provider-specific. **📅 Volatile:** cache write costs, TTLs (often measured in minutes, not hours) and hit semantics differ substantially by provider; verify all three before designing around this. The TTL matters enormously here: if the cache expires in five minutes and your traffic is one query per ten minutes, **you pay the write cost every time and never get a read hit**, and the whole economic argument collapses.

**The RAG architecture.** 8k assembled tokens: `8,000 × 3e-6 = $0.024` plus output `$0.0075` = **$0.032** per query, TTFT ~1.2 s including retrieval.

**The comparison at 100k queries/month:**

| | Cost/query | Monthly | TTFT |
|---|---|---|---|
| Stuff, uncached | $1.21 | $121,000 | 20–40 s |
| Stuff, cached (ideal) | $0.128 | $12,800 | seconds |
| RAG | $0.032 | $3,200 | ~1.2 s |

**RAG is 4× cheaper than the best case for stuffing and roughly 38× cheaper than the realistic one, and it is an order of magnitude faster.** Even at the cached price, the gap is `$9,600/month`, which pays for the entire retrieval stack many times over.

**The case *for* stuffing, stated fairly**, because there are three real ones. **First, no retrieval misses** — the ceiling is the model's ability to use the context, not your recall@k, and if your recall@8 is 0.85 then 15% of your queries are unanswerable by construction in the RAG design. **Second, engineering time**: stuffing is one afternoon and RAG is two weeks plus ongoing ownership of an index, a reindex pipeline, and an eval harness. At low volume that trade is obviously correct — at 500 queries/month, stuffing costs `500 × 0.128 = $64/month` and you should absolutely not build a retrieval system for that. **Third, cross-document reasoning**: "summarize the themes across all 400k tokens" is a question RAG structurally cannot answer.

**🗣 Say this in the room:** "The crossover is a volume question and I'd compute it rather than assert it. At a few hundred queries a month, stuff it — engineering time dominates and the retrieval system isn't worth owning. Above roughly ten thousand queries a month the cost and TTFT gap makes retrieval obviously correct. And in between, the deciding factor is usually whether the corpus is identical across users, because if it isn't, prompt caching does nothing for you and stuffing costs ten times more than the headline."

### What is cache-augmented generation, and when is it the right architecture?

**Mental model: CAG asks why you should re-prefill the same documents on every request when the KV cache from that prefill is a reusable artifact.** If the corpus is fixed and small, prefill it once, persist the KV cache, and make every subsequent request a decode-only operation over a warm cache. It converts "retrieval at query time" into "retrieval at deploy time."

The mechanism: run the model's forward pass over the corpus once, producing keys and values for every layer and position. Persist that KV tensor. On each request, load it, append the user's question's tokens, and decode. You have skipped the entire retrieval pipeline *and* the prefill. The framing was published under the "don't do RAG" banner in 2024–2025; **📅 treat the specific claims as fast-moving, and note that a hosted provider's prompt caching is essentially the productized version of this idea with the storage and eviction hidden from you.**

The constraints are severe and naming them is the whole answer:

**The corpus must fit in the context window.** Not "mostly fit" — the KV cache is over a specific token sequence, and there is no eviction policy you control.

**The corpus must be stable.** Any change invalidates the whole cache and requires a full re-prefill, because the KV values at position `n` depend on every token before `n`. **You cannot patch a document into the middle of a KV cache** — this is the property people miss, and it is why CAG is wrong for anything with meaningful write traffic.

**KV cache is large.** This is the number that decides it. The per-token KV size is `2 × n_layers × n_kv_heads × head_dim × bytes_per_element`. For a model with 32 layers, 8 KV heads (grouped-query attention), head_dim 128, in fp16: `2 × 32 × 8 × 128 × 2 = 131,072 bytes/token = 128 KiB/token`. **A 100k-token corpus is `100,000 × 131,072 = 13.1 GB` of KV cache** — more than the model weights for a 7B model. That is per-corpus, and if you have per-tenant corpora it is per-tenant.

**Where it is genuinely right:** a fixed, small, stable corpus consulted at high frequency by many users. A product manual. A policy handbook. A game's rulebook. A single API reference. **📅** And the pragmatic version for most teams is not to build it yourself: put the fixed corpus in the stable prefix of your prompt and let the provider's prompt caching do it, which gets you most of the benefit with none of the KV management.

**⚠ Trap:** proposing CAG for an enterprise knowledge base. Those are multi-gigabyte, permission-scoped per user, and change hourly. Each constraint independently disqualifies it. **The interviewer asking about CAG is usually checking whether you know it is a narrow special case rather than a RAG replacement**, and the answer that wins is naming the three constraints before naming a use case.

### Our long-context model scores 99% on needle-in-a-haystack at 500k tokens. My VP wants to delete the retrieval pipeline. What do you say?

I say the benchmark does not measure the capability we depend on, and I say it with a specific alternative measurement rather than a general objection — otherwise it reads as defending my own code.

**What needle-in-a-haystack actually tests:** insert one sentence into a long filler document, ask for it back. That is **single-fact exact retrieval with zero distractors** — the inserted needle is semantically alien to its surroundings, which makes it maximally easy to locate. Real queries are the opposite: the answer is surrounded by twelve *closely related* passages that are almost but not quite right, and the hard part is discrimination, not location. **Saturating NIAH tells you the model can find a red sentence in a green haystack. It tells you nothing about finding the right green sentence.**

The measurements I would run instead, and I would offer to run them that week:

**Multi-needle with distractors.** Four facts required for one answer, plus eight near-miss passages that state similar-but-different versions. This is what RULER-style benchmarks do, and it is where advertised context and effective context diverge.

**Your own corpus, your own queries.** Take your existing golden set, and instead of retrieving, put the answer's document plus 100k tokens of your real corpus in the context. Compare answer accuracy to the RAG pipeline. **If long context wins on your data, I will happily delete the pipeline — and having said that sincerely is what makes the rest of the objection credible.**

Then the three arguments that are true regardless of the benchmark outcome:

**Cost.** `500,000 × 3e-6 = $1.50` per query uncached versus `$0.024`. At 50k queries/month that is `$75,000` versus `$1,200`. **62× is not a number a VP overrides on a benchmark result.**

**Latency.** Seconds of prefill per turn in an interactive product.

**Auditability.** Discussed next, and in a regulated context it is not negotiable.

**🗣 Say this in the room:** "Needle-in-a-haystack is single-fact retrieval with no distractors, and our failure mode is discrimination among near-misses, not location. I'd run the multi-needle-with-distractors version on our own corpus before making the call — that's a two-day experiment. Separately, at 500k tokens per query the cost is about sixty times our current pipeline, so even if quality were equal the economics decide it."

### Retrieval keeps an advantage that long context can't match. Name it and defend it.

**Auditability.** A retrieval system produces a **bounded, enumerable, inspectable evidence set** — eight chunks with document IDs, versions, timestamps, and character offsets — and the answer is provably a function of exactly those. A long-context system produces an answer that is a function of 500,000 tokens, and "which part of it?" has no cheap answer.

This is not a nice-to-have and it is not primarily a UX argument. Concretely, retrieval gives you five things that a stuffed context does not:

**Verifiable citations at span granularity.** You can highlight the exact source characters. A reviewer confirms in two seconds. With full-context, "the model said this is in the corpus" is unfalsifiable without reading the corpus.

**Permission enforcement at the evidence boundary.** In a multi-tenant or ACL'd product, the retrieval filter *is* the security control — a document the user cannot see is never in the candidate set. In the stuffed design, the entire corpus is in the context and the only thing preventing leakage is the model's instruction-following. **That is not an access control; that is a request.** For Glean-shaped products this alone ends the debate.

**Reproducibility for incident review.** Six months later, "why did the system tell this customer they were entitled to a refund?" is answerable: here are the eight chunks it read, at these versions. Long-context gives you a prompt log and a shrug.

**Deletion and right-to-be-forgotten.** Remove a document from the index and it is unretrievable on the next query. In a persisted-KV-cache design you must invalidate and re-prefill. In a fine-tuned model you are, practically, stuck.

**Staleness control.** Retrieval returns the current version because the index was updated. A cached context returns whatever was cached, and if your TTL is long, silently.

**⚠ Trap:** treating auditability as a compliance checkbox rather than a debugging tool. **The evidence set is the single most useful artifact for diagnosing a bad answer**, and teams that stuff context give it up and then wonder why their failures are unfalsifiable. When an answer is wrong in a RAG system, you look at the retrieved chunks and within thirty seconds you know whether it is a retrieval failure or a generation failure. That fork is the beginning of every RAG debugging session, and long context deletes it.

**🗣 Say this in the room:** "The thing retrieval keeps is a bounded evidence set. That buys three separate things: span-level citations a human can verify in seconds, permission enforcement at the retrieval filter rather than in the prompt, and a debugging fork — was it retrieved, or was it read badly. In a regulated product the first two are requirements. The third is why I'd keep retrieval even if cost and quality were a wash."

### So what's the right answer — long context or RAG? And how would you prove it to a skeptical staff engineer?

The right answer is a **hybrid whose shape is set by measurement**, and the reason I say "usually hybrid" rather than picking a side is that the two techniques fix different failure modes: retrieval fixes "the evidence is not in the window," long context fixes "the evidence was in the corpus and retrieval missed it."

The hybrid I default to: **retrieve generously into a large window rather than precisely into a small one.** With a 4k window you needed the right 3 chunks — a precision problem. With a 128k window you can afford 40 chunks — a recall problem. So: hybrid retrieval with high recall, a reranker to order rather than to aggressively cut, top-20 to top-30 instead of top-5, and position-aware ordering. **You get retrieval's citations, permissions and cost profile, plus long context's tolerance for imperfect ranking.** That is 12k tokens, not 400k — 25× cheaper than stuffing and materially more robust than top-5.

Then the layered escalations: **cache the stable parts** (system prompt, and any fixed reference material) so their tokens cost a tenth; **route** the small fraction of genuinely whole-corpus questions to a map-reduce or precomputed-summary path, since neither retrieval nor stuffing handles those well; and **abstain** when retrieval confidence is low rather than escalating to a 400k-token prefill.

Now the proof, because "I would run an ablation" is not an answer and a staff engineer will not accept it:

**The ablation table, one variable at a time, on the same golden set, with matched budgets:**

| Config | Answer acc. | Faithfulness | Tokens/query | Cost/query | p95 TTFT |
|---|---|---|---|---|---|
| A. RAG top-5, no rerank | | | ~3k | | |
| B. RAG top-5 + rerank | | | ~3k | | |
| C. RAG top-25 + rerank | | | ~12k | | |
| D. RAG top-25 + rerank + position order | | | ~12k | | |
| E. Full corpus stuffed (subsample fitting the window) | | | ~400k | | |
| F. E + retrieval-ordered (best chunks first, rest after) | | | ~400k | | |

**The rules that make this credible:** one variable per row; the same 300-query golden set with the same judge and a fixed judge prompt; **bootstrap 95% confidence intervals on every delta**, because a 3-point accuracy difference on 300 queries is inside the noise and reporting it as a win is how these debates get won by the wrong side; and cost and latency reported in the same table as quality, so nobody can win on accuracy while quietly costing 40×.

**The row that usually settles it is C versus E.** If C is within the confidence interval of E, you have shown that 12k well-chosen tokens match 400k tokens at 1/33rd the cost, and the conversation is over. If E beats C by a wide margin, **your retrieval has a recall problem and you have learned something more valuable than the architecture answer** — go fix recall and re-run, because a retrieval ceiling below the long-context result is a bug, not an architectural verdict.

**🏋 Drill:** 90 minutes. Build rows A, C and E on any corpus you have, with a fixed LLM judge and bootstrap CIs. *Pass criterion:* you can state the cost-per-accuracy-point of moving from C to E and say whether you would pay it.

### Design a Glean-style enterprise assistant: Drive, Slack, Jira, Confluence, strict ACLs, 20,000 employees. Pick the tier and defend it.

I will take this end to end, because the interesting decisions are not the retrieval algorithm.

**Start with the constraint that dominates everything: permissions.** Twenty thousand employees with heterogeneous ACLs across four systems whose permission models do not agree — Drive has per-file sharing, Slack has channels and DMs, Jira has project roles, Confluence has space permissions. **This single fact rules out several architectures immediately:** no stuffed-corpus design (no enforcement boundary), no shared precomputed summaries across users (they would leak), no cross-user semantic answer cache (same reason), and no fine-tuning on the corpus (the weights would encode content the model cannot un-know per user).

The architecture:

**Ingestion.** Per-source connectors on a Celery-style queue, content-hash-based change detection, idempotent document IDs of the form `{source}:{native_id}:{version}`. Each chunk carries a **normalized ACL descriptor** — a set of principal IDs (users, groups) — resolved from the source system at ingest. Reindex is alias-swapped so a re-embed is atomic from the reader's perspective.

**The permission decision, and this is the one to get right in the room.** Two options: **materialize** ACLs into the index as a filterable field, or **resolve at query time** against the live permission service. Materialized is fast and goes stale — someone leaves a team and can still retrieve its documents until the next sync, which is a security incident with a latency measured in hours. Live resolution is correct and adds a hop.

**My answer is both, and the ordering matters: materialized ACLs as a pre-filter on the vector search for efficiency, then a live re-check on the final candidate set before assembly.** The pre-filter is a *performance* optimization that must be permissive-consistent — it may be stale in the direction of returning too much, never too little — and the post-check is the actual security boundary, run against ~50 candidates rather than 50 million documents, so it costs one batched call to the permission service, maybe 20 ms. **This is exactly the pattern you already use for cheap-index-then-authoritative-check in any permissioned search system, and saying so is the right register.**

**Retrieval.** Hybrid: dense for semantic, BM25 for the identifiers that dominate enterprise queries — ticket keys, error strings, doc titles, people's names. RRF at k=60. Then per-source weighting, tuned on a golden set: Confluence and Drive up-weighted for policy questions, Jira and Slack up-weighted for "why did we decide X" and "has anyone hit this." Cross-encoder rerank 100 → 10.

**Query understanding.** One structured call doing conversational rewrite, intent routing, and entity extraction into filters (person names, project keys, date ranges — "what did the payments team decide last quarter" is three filters and a query). Skipped entirely when the turn is self-contained and long, which is 50%+ of first turns.

**Which tier?** **Tier 2, deterministic pipeline — with exactly two tier-3 branches.** Branch one: a person-lookup route, because "who owns the payments service" should hit a directory and an ownership registry, not a text index. Branch two: an iterative loop for the small multi-hop slice. Everything else is a straight line, because with 20,000 employees the p50 query is "what's the VPN setup" and it must return in under a second.

**Freshness.** Slack and Jira change constantly; Confluence and Drive slowly. Different ingestion cadences — near-real-time CDC for the first two, hourly for the rest — and a recency prior in scoring for the conversational sources, since a 2023 Slack thread about a since-changed decision is actively harmful.

**💰 Math:** 20,000 employees × 3 queries/day × 22 working days = **1.32M queries/month**. At $0.032/query that is `$42,240/month` in generation. Prefix caching on a 2k stable prefix saves `1.32e6 × 2000 × 3e-6 × 0.9 = $7,128/month`. Head-of-distribution precomputation on the top 100 intents at 35% coverage saves another `1.32e6 × 0.35 × 0.9 × 0.032 = $13,305/month` and drops a third of traffic to sub-100 ms. **Combined, that is $20,433/month against a $42,240 baseline — the two cheapest optimizations in this guide, together, roughly halve the bill.** Self-hosting the reranker rather than paying per search unit saves a further `1.32e6 × ($0.002 − $0.000011) ≈ $2,625/month`, which is the point at which owning a GPU becomes obviously correct.

**⚠ Trap the interviewer is watching for:** proposing a cross-user answer cache. It is the obvious optimization and it is a data-leak generator — user A's answer, built from documents user B cannot see, served to user B on a semantically similar query. **Any cache in a permissioned system must be keyed on the permission scope, not just the query**, and the practical consequence is that cache hit rates are far lower than the naive estimate, which is exactly why head-of-distribution precomputation must be restricted to **globally-readable** documents.

### Design the RAG layer for a customer-support agent that also takes actions — refunds, plan changes, escalations. What's different?

What is different is that **retrieval is now feeding a decision, not a paragraph**, and a wrong retrieval becomes a wrong *action* rather than a wrong sentence. That single change reorders every priority.

**The architecture split I insist on: retrieval for the policy, structured data for the facts, and the action gated on both.** The question "can this customer get a refund" decomposes into a policy lookup (RAG over the refund policy, which is prose and changes rarely) and a fact lookup (this customer's plan, purchase date, prior refunds — which is a database query and must never come from a vector index). **Retrieving customer-specific facts from a text index is the single most common architectural error in support agents**, and it fails in the worst way: the top chunk is another customer's similar case, and the agent applies their circumstances to this one.

**Multi-turn is the norm, not the exception**, so conversational query rewriting is load-bearing rather than optional, and the history-versus-retrieval budget arbitration from earlier is a daily concern rather than an edge case. The structured rolling summary matters more here than anywhere: an agent that forgets the customer already said "I'm on the annual plan" and asks again is the most common complaint about these products.

**Confidence gating before action.** The retrieval evaluator is not an optimization here, it is a safety control: if the policy retrieval confidence is below threshold, the agent must **not** act — it escalates. The decision procedure I encode:

```
policy_confidence high  AND facts retrieved  AND action within limits  -> act
policy_confidence high  AND facts retrieved  AND action above limits   -> act with approval
policy_confidence low   OR  facts missing                              -> escalate to human
policy conflicts across retrieved chunks                               -> escalate to human
```

That last branch matters and is usually missing. If retrieval returns two policy chunks that disagree — typically a current and a superseded version — **the correct behaviour is escalation, not picking one.** Detect it by having the generator explicitly report conflicts, and by carrying timestamps and version metadata in every chunk so it *can*.

**Idempotency, which you already know but which acquires a twist.** The action must be idempotent on a key — fine, standard. The twist: **the LLM is not a reliable source of idempotency keys.** Generate the key deterministically from `(conversation_id, action_type, target_entity)` in your code, never from the model's output, or a retry produces a second refund with a new key. This is the exact place backend intuition transfers and the exact place people let the model do something it should not.

**💰 Math on the escalation threshold, because this is the trade-off the interviewer wants priced.** Suppose 100k conversations/month, 60% resolvable by the agent, human handling costs $4 per escalation, and an incorrect automated action costs $40 in remediation and goodwill. At a threshold escalating 25% of traffic with a 2% error rate on the rest: `25,000 × $4 = $100,000` in escalations plus `75,000 × 0.02 × $40 = $60,000` in errors = **$160,000**. Loosening the threshold to escalate only 15% but raising the error rate to 5%: `15,000 × $4 = $60,000` plus `85,000 × 0.05 × $40 = $170,000` = **$230,000.** **The tighter threshold is cheaper**, and that arithmetic — not intuition — is how the threshold should be set. **The general rule: when the cost of a wrong action exceeds the cost of a human touch by more than about 10×, bias hard toward escalation.**

### An engineer on my team wants to answer questions about our 3-million-line monorepo by putting the whole thing in a 1M-token context. What's wrong with that, and what do you build instead?

Start with the arithmetic, because it ends the proposal on its own: 3M lines of code is roughly **30–45M tokens** at ~10–15 tokens per line. It does not fit in a 1M-token window by a factor of 30–45. **The premise is false before we get to any interesting argument**, and noticing that first is the answer — a lot of long-context proposals dissolve on a token count nobody did.

Even for the subset that fits, the economics: 1M tokens at $3/Mtok is `$3.00` per query. An engineer asking 40 questions a day is `40 × 3 = $120/day = $2,640/month per engineer`. For a 200-engineer org, `$528,000/month`. **📅** Prices fall, but the ratio to a retrieval design — which serves the same query for around three cents — does not.

What I build instead, and code retrieval is genuinely different from document retrieval in ways worth naming:

**Chunk on syntax, not on characters.** Split by AST node — function, class, method — so a chunk is a complete unit with its signature intact. A function cut in half is worse than useless. Include the file path, the enclosing class, and the imports as a header on each chunk, because a method body without its class name is unattributable.

**Lexical retrieval is disproportionately important here.** Identifiers are the query. `getUserBillingContext` is exactly the kind of rare literal string that dense embeddings destroy and BM25 nails. In my experience code retrieval is **more** lexical-weighted than document retrieval — I would start the hybrid weights closer to even and tune from there rather than starting dense-dominant.

**Symbol index and call graph as first-class retrieval channels.** "Where is this function defined" and "who calls this" are graph queries with exact answers, not similarity queries. Route them to an LSP-style symbol index or a `tree-sitter`-derived graph. **A code assistant that answers "who calls X" with fuzzy retrieval is broken by design**, and the fix is not a better embedder, it is the right index.

**Retrieve by expansion, not by top-k alone.** Find the seed chunk, then pull in its direct dependencies — the functions it calls, the types it uses, the test that exercises it. This is a graph traversal from the retrieval result, and it is what makes code answers coherent. Bound it at one or two hops or you will pull in the world.

**Incremental reindex per commit.** Re-embedding a monorepo on every merge is not viable. Reindex only changed files, and their transitive dependents' *symbol* entries. Content-hash per file, and a job that runs in seconds on a typical PR.

**Where long context genuinely wins, and I would concede it:** once you have found the relevant 15 files, putting all of them in fully — not chunked — beats putting 40 fragments in. **The right design is retrieval for *file selection* and long context for *file content***, which is close to what good coding agents actually do. That is the hybrid answer with a specific shape, and it is more convincing than a general one.

**🗣 Say this in the room:** "Thirty to forty-five million tokens doesn't fit in a one-million-token window, so the premise fails on arithmetic. What I'd build is AST-level chunking with a heavier lexical weight than I'd use for prose, a symbol index for definition and reference queries, dependency expansion from the seed hit, and then long context for the selected files rather than for the repo. Retrieval picks the files, the window holds them whole."

### Give me the drill set and the whiteboard order-of-operations for this material.

Two things: the ordered procedure for a live design round, and the timed drills that make it automatic.

**The 45-minute RAG whiteboard, in order. Deviating from this order is what loses the round**, because candidates who start at the retrieval algorithm never get to evaluation and the interviewer scores them as someone who has read about RAG rather than shipped it.

1. **Clarify the corpus and the query distribution first — two minutes, four questions.** How many documents and what modality? How often do they change? Are they permissioned? What do the top 20 real queries look like? **You cannot design this system without those four answers and asking for them is scored.**
2. **State how you will know it works, before any architecture.** 100–300 golden queries with chunk-level relevance labels mined from real logs; recall@k and nDCG@k for retrieval, faithfulness and answer accuracy for generation, measured separately. **Open with this. It is the single highest-signal thing you can say in a RAG round.**
3. **Draw the ingestion path.** Connectors, parsing, structure-aware chunking with metadata and offsets, content-hash change detection, embedding queue, alias-swapped index.
4. **Draw the query path.** One structured query-understanding call (rewrite + route + filters), hybrid retrieve, RRF, rerank, dedup, position-aware assembly with citation markers and timestamps, generate with abstention.
5. **Name the tier and defend it.** "Tier 2, with two tier-3 branches, because X% of the query distribution is multi-hop and the rest isn't."
6. **Do the cost and latency arithmetic out loud.** Per-query cost broken into line items, TTFT budget by stage, and the two biggest levers with their monthly numbers.
7. **Name three failure modes and their detection.** Stale index, permission drift, dense-only failure on identifiers — each with the metric or alert that catches it.
8. **State what you would do differently at 100× scale**, unprompted.

**🏋 Drill 1 — the 100-line RAG (20 min, unaided).** Chunking, batched embedding with unit-normalization, brute-force search, prompt assembly with citation markers and an abstention instruction. *Pass:* runs, search is one matrix-vector product, and you can state the corpus size at which you would replace it with HNSW and why.

**🏋 Drill 2 — the query-understanding call (20 min, unaided).** Write the structured-output schema and prompt that does conversational rewrite, intent routing (including the no-retrieval branch), decomposition, and filter extraction in one call, plus the skip heuristic. *Pass:* one call not four, a closed enum for filters, a confidence field, and a low-confidence fallback to the boring path.

**🏋 Drill 3 — the cost model (10 min, no calculator).** Given 8k context, 400 output tokens, $3/$15 per Mtok, 1M queries/month: total monthly cost, the percentage that is input tokens, the saving from cutting context to 5k, and the saving from 90% prefix caching on a 2k stable prefix. *Pass:* `$32,000`, `~94%`, `$9,000`, `$5,400` — and you can say which one you would do first and why.

**🏋 Drill 4 — the multi-turn diagnosis (10 min, verbal).** "First turn works, second turn fails." Give the mechanism, the two-minute confirming check, the quantitative signature, the fix, and the wrong fix people try. *Pass:* you name the embedded string as the thing to inspect, recall-split-by-turn-index as the signature, rewriting as the fix, and concatenating full history as the wrong fix — in under 90 seconds.

**🏋 Drill 5 — the pushback (5 min, verbal, out loud).** Someone proposes agentic RAG over 200 documents. Deliver the diplomatic refusal with the arithmetic. *Pass:* you propose a cheap experiment rather than asserting, you have the 2× cost and 6× latency numbers ready, and the teammate would not feel dismissed.

**The meta-drill.** Be able to answer, in 60 seconds without notes: "when would you *not* build RAG?" The full answer is — when the corpus is small enough and stable enough that stuffing plus prompt caching is cheaper than owning an index; when the question is aggregate or whole-corpus and retrieval is structurally the wrong primitive, so text-to-SQL or precomputed rollups win; when the need is style or format rather than knowledge, where fine-tuning is the right tool and retrieval is not; and when volume is low enough that two weeks of engineering exceeds the lifetime inference bill. **A candidate who can enumerate when their favourite architecture is wrong is the one who gets the offer.**
