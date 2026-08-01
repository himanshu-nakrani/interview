### Walk me through your context assembly stage. You've got reranked chunks — now what?

**Mental model: assembly is a bin-packing problem with a fixed budget and several competing claimants, and almost nobody treats it as one.** The typical implementation is `"\n\n".join(chunks[:5])`, which silently makes four separate decisions — how many, in what order, with what metadata, at whose expense — none of them examined.

The budget declaration comes first, and it should be explicit in code:

```python
BUDGET = {
    "system":     2_000,   # stable, cacheable, never varies
    "tools":        800,   # stable
    "history":    3_000,   # recent turns, compacted beyond that
    "retrieved":  8_000,   # the variable part
    "question":     200,
    "reserve":    1_000,   # headroom so output isn't truncated
}
```

Then assembly in a fixed order, and the order is not arbitrary:

**1. Stable prefix first.** System prompt, tool definitions, any few-shot examples — everything byte-identical across requests goes at the very front, because prefix caching only matches from token zero. Get this wrong and you forfeit the single largest cost lever in the system.

**2. Filter and dedup the retrieved set** before counting tokens. Near-duplicate chunks are common — the same paragraph in a PDF and its HTML export, two versions of a runbook — and five copies of one paragraph occupying your window is the most wasteful failure in assembly.

**3. Allocate by group, not globally**, when the query was decomposed: a floor of 2–3 chunks per sub-question before filling the rest by score, or one sub-question's higher-scoring chunks starve another entirely.

**4. Order the survivors** for the position effects (next question).

**5. Wrap each chunk with the metadata the model needs to reason and cite:** a stable ID, source document title, section path, and **the timestamp** — the last one is non-optional if your corpus contains superseded content, which every corpus does.

**6. Count tokens with the real tokenizer** and trim by dropping whole chunks from the bottom, never by truncating mid-chunk. A half-chunk is a chunk whose claim has been separated from its qualifier, which is a hallucination generator.

**⚠ Trap:** estimating tokens as `len(text) / 4`. That heuristic is calibrated on English prose and is wildly wrong on the content RAG actually retrieves — code, JSON, tables, and non-Latin scripts run far denser in tokens per character, and the miss is one-directional: you under-count, exceed the window, and the provider truncates from wherever its policy says, usually silently. **Count with the actual tokenizer, cache the counts on the chunk record at index time, and add a hard assertion before the call.** The failure this prevents — silent truncation dropping the last chunk, which is often the reranker's top hit if you ordered ascending — is genuinely hard to diagnose from the outside because nothing errors.

### How does chunk ordering interact with prompt caching, and what does getting it wrong cost?

**Mental model: prefix caching is exact-prefix, byte-for-byte, from token zero — it is not an LRU over your content, it is a radix match on a token sequence.** One changed token at position 400 invalidates everything after it. That single property dictates the entire layout of your prompt.

The layout rule follows mechanically: **most stable content first, most variable content last.** System prompt → tool definitions → few-shot examples → *(cache breakpoint)* → retrieved chunks → conversation history → the user's question. Retrieved chunks vary per query, so nothing after them can ever be cached; therefore nothing that *could* be cached may sit after them.

**💰 Math, and this is the arithmetic to have ready.** System prompt + tools + examples = 3,000 tokens, stable. At $3/Mtok uncached and a 90%-discount cached rate of $0.30/Mtok (📅 verify per provider — some offer 50%, some 90%, and write costs differ):

- Uncached, every call: `3,000 × 3e-6 = $0.009`
- Cached: `3,000 × 3e-7 = $0.0009`
- Saving: **$0.0081 per call.** At 500k calls/month: `500,000 × 0.0081 = $4,050/month.`

Now the failure. Suppose someone puts a `"Current time: 2026-08-01T14:22:31Z"` line at the top of the system prompt, or sorts the retrieved chunks before the stable examples. **Cache hit rate goes to zero** and you pay the full `$4,050/month` you thought you had saved — plus the TTFT difference, since a cached prefix skips prefill: at roughly 50–100 ms per 1k tokens, those 3,000 tokens are **150–300 ms of TTFT** paid on every single request.

The rules I enforce in review:

- **No timestamps, request IDs, user names, or A/B flags in the stable prefix.** If the model needs the date, put it immediately before the question, at the end.
- **Deterministic chunk ordering.** If your retriever returns ties in nondeterministic order — and dictionary iteration or a parallel gather will do this — the assembled prompt differs run to run for the same query, and the retrieval-result cache and any downstream prefix cache both miss. Sort by `(-score, chunk_id)` so ties break deterministically.
- **Instrument the hit rate.** Providers return cache-read token counts in the usage payload. Emit that as a metric and alert on it. **A prompt-caching regression is invisible in every functional test and shows up only on the invoice 30 days later**, which is exactly the kind of failure that should be a monitored SLI.

**🗣 Say this in the room:** "Prefix caching is an exact-prefix match from token zero, so my prompt is laid out most-stable to most-variable, and I treat cache hit rate as a monitored SLI. The classic own-goal is a timestamp in the system prompt: it looks harmless, it invalidates the cache on every request, and on a 3k-token prefix at 500k calls a month that's about four thousand dollars and 200 milliseconds of TTFT you've thrown away with no test failing."

### Where in the context should the best chunk go, and why?

At the **beginning**, and if you have room for redundancy, restate the most important evidence at the end.

**📄 Paper:** Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni & Liang (2023) — *Lost in the Middle: How Language Models Use Long Contexts*. They varied the position of the gold document within a multi-document context and found a pronounced **U-shaped accuracy curve**: models retrieve information best from the very beginning and the very end of the context, and materially worse from the middle. It replaced the comfortable assumption that attention is position-agnostic over the input.

**Mechanism, honestly stated:** the cause is not fully settled and I would not pretend otherwise in a room. The contributing factors people cite are the training distribution — instructions and questions appear at the boundaries of training documents far more often than in the middle — and attention-sink behaviour at the sequence start, where the first tokens absorb a large share of attention mass regardless of content. Position-encoding extrapolation beyond training length contributes at very long contexts. **The effect is robust and reproducible even where the explanation is contested, and the honest answer is "the effect is well-replicated, the mechanism is partly training distribution and partly attention concentration at sequence boundaries."**

The practical assembly rule:

```python
def order_for_position(chunks):        # chunks sorted best-first by reranker
    """Best chunk at the start, second-best at the end, remainder in the middle
    descending. Puts your two strongest pieces of evidence at the two positions
    the model reads best."""
    if len(chunks) <= 2:
        return chunks
    return [chunks[0]] + chunks[2:] + [chunks[1]]
```

Some teams instead sort *ascending* by score so the best chunk lands last, adjacent to the question. That is also defensible and sometimes measures better — **which one wins is model-dependent and eval-set-dependent, and this is a genuine "run the ablation" case rather than a known answer.** It is a two-line change and a one-hour experiment; anyone asserting a universal rule here without having run it on their own model is guessing.

**⚠ Trap:** treating position effects as a reason to stuff more chunks in. The correct inference runs the other way. If the middle of the context is read poorly, then **the marginal chunk you added at position 12 is nearly free of benefit and fully priced** — you paid its tokens, its prefill latency, and its dilution of attention, and got very little back. Position effects are an argument for a *smaller, better-ranked* context, which is the argument for a reranker.

### Five of my eight retrieved chunks are near-duplicates of each other. How do you handle that at assembly time?

**Mental model: retrieval optimizes relevance per item, and relevance is not the same as the marginal value of adding an item to a set. Five copies of the right paragraph have the relevance of five chunks and the information content of one.** Every dedup technique is an attempt to score marginal contribution instead of individual relevance.

Three layers, applied in order, because they catch different things:

**Exact and near-exact duplicates** — the same content ingested twice via different connectors, a PDF and its HTML export, a wiki page and its mirror. Catch with content hashing at index time (a normalized hash: lowercase, collapse whitespace, strip boilerplate) and with MinHash or SimHash for near-duplicates above ~0.9 Jaccard. **This belongs in ingestion, not assembly** — carrying duplicates through embedding, ANN, and reranking wastes the whole pipeline's budget on them, and it inflates your index size and cost for nothing.

**Semantic redundancy** — different wording, same content. This is what MMR is for: greedily select the chunk maximizing `λ · relevance(c, q) − (1 − λ) · max_{s ∈ selected} sim(c, s)`. Two implementation details that people get wrong: relevance from a cross-encoder and similarity from cosine are **on different scales**, so you must rescale (min-max over the candidate set) before combining, or λ means nothing; and λ around **0.7** is a sane default — lower makes the context diverse but off-topic, which is worse than redundant.

**Overlap from chunking** — adjacent chunks with a 200-character overlap share text by construction. Collapse contiguous chunks from the same document into one span rather than shipping both with the seam duplicated. This is nearly free and cuts assembled tokens noticeably on corpora with generous overlap.

**⚠ Trap:** aggressive dedup destroys a genuine signal. If four independent documents all state the same policy, that repetition is **evidence of consensus**, and collapsing it to one can make the model hedge on something it should assert. Worse, if two of the four are *slightly* different — a superseded version and the current one — dedup may keep the wrong one. My rule: **dedup within a document freely; across documents, collapse to at most two representatives and preserve the count in the metadata** (`"3 other documents state this policy"`). The model can then use consensus without you spending tokens on it.

**💰 Math:** on a typical enterprise corpus I see 15–25% of retrieved tokens as redundant. At 8,000 assembled tokens and 20% redundancy, dedup saves `1,600 × 3e-6 = $0.0048` per query — `$4,800/month` at 1M queries — plus roughly 80–160 ms of prefill. And it *improves* quality by freeing those slots for chunks that were ranked 9 through 12. **Dedup is the rare optimization that is strictly Pareto-improving, which is why it is the first thing I add to a naive pipeline.**

### How do you get the model to produce citations that actually resolve to the right chunk?

**Mental model: the model can only cite what you gave it an unambiguous name for, and "the document above" is not a name.** Citation quality is overwhelmingly a function of your assembly format, not of the model's diligence.

The format that works, and each element earns its place:

```
[S3] doc: "Refund Policy v4" | section: Billing > Refunds > Annual | updated: 2025-11-04
Annual plans are refundable within 30 days of the renewal date...
```

**Short opaque IDs (`S1`…`S8`), not filenames.** Filenames are long, tokenize badly, and the model will paraphrase or truncate them. A two-token ID it copies exactly is reliable; a 40-character path is not. Maintain the ID→chunk mapping server-side for the request and resolve on the way out.

**Ask for the citation inline and immediately after the claim**, not as a bibliography at the end. Bibliographies are generated after the fact and are where fabricated attributions come from — the model has finished reasoning and is now decorating. Inline citation forces the attribution to be produced while the evidence is still driving the generation.

**Validate server-side, before streaming reaches the user.** Parse every `[Sn]`, confirm `n` is in the set you provided, and drop or flag unknown IDs. **Models will cite `[S9]` when you gave them eight sources.** This is a five-line check and it is not optional.

For anything regulated, escalate to **quote-then-answer**: require a verbatim quoted span from the source before each claim, then verify the quote by exact substring match against the chunk text. Non-matching quotes get flagged. That gives you character offsets you can actually highlight in a UI, and highlighting is what makes the citation checkable by a human in two seconds rather than thirty.

**⚠ Trap — the one that gets caught in a demo to a customer:** a citation that resolves to a real chunk that does not support the claim. The ID is valid, the link opens, the document is real, and the sentence is not in it. Every structural check passes. This is the failure mode that plausible-looking citations exist to hide, and the only defenses are quote verification (cheap, catches the blatant cases) and post-hoc entailment checking of each claim against its cited chunk (a small NLI model or a cheap judge call, ~50–150 ms per claim, worth it when the citation is the product).

**💰 Math:** post-hoc entailment over an average 4 claims per answer with a small hosted NLI model at ~30 ms each, run concurrently, adds ~50 ms wall clock and a per-call cost well under $0.001. Against an answer costing $0.030, that is under 3% for a measurable groundedness rate. **In legal, medical, or financial products I consider this mandatory; in an internal engineering assistant I consider it optional.** Say which product you are building before you say whether it is worth it.

### How many chunks should you put in the context? Give me the reasoning, not a number.

**Mental model: adding chunks trades recall for precision-of-attention, and the optimum is where the marginal chunk's probability of containing the answer stops exceeding its dilution cost.** That crossover is corpus- and model-specific, which is why anyone quoting "always use 5" is quoting a tutorial default.

The two curves. **Recall@k rises steeply and then flattens** — going 1→5 might take you from 0.55 to 0.88, and 5→20 from 0.88 to 0.94. Each additional chunk has sharply diminishing probability of being the one you needed. **Answer quality rises with recall but falls with dilution** — more distractors, worse position effects, more chance the model latches onto a topically-adjacent-but-wrong chunk. Multiply them and you get a hump.

How to actually find it, and this is the answer the interviewer wants: **plot both curves on your own eval set.** Measure recall@k for k in {1,3,5,8,12,20,30} — cheap, no generation needed, retrieval only. Then measure end-to-end answer quality at k in {3,5,8,12,20} — expensive, needs generation and judging. The recall curve tells you the ceiling at each k; the quality curve tells you where dilution starts costing more than coverage. **In most enterprise RAG systems I have measured this lands between 5 and 10 chunks at ~400 tokens each**, but I would state that as my prior and not as an answer.

Three factors that shift it:

**A reranker moves the optimum down.** With a good cross-encoder, recall@5 approaches what recall@20 was without it, so you get the same coverage with a quarter of the dilution and a quarter of the tokens. **That is the real argument for a reranker: not better ranking for its own sake, but a smaller context at the same recall**, which is simultaneously a quality, latency and cost win.

**Longer context windows move it up, but less than people expect** — see the position effects and the effective-context discussion. Window size raises the ceiling on what is possible, not on what is useful.

**Chunk size trades off against k directly.** Twenty 200-token chunks and five 800-token chunks are the same token budget with very different properties: more chunks means more documents represented and more boundary-cut claims; fewer, larger chunks means better local coherence and fewer sources. **Tune `k × chunk_size` as one budget, not two independent knobs**, and say that — it is the sentence that shows you have actually tuned this.

**🏋 Drill:** 30 minutes with a laptop and any labeled QA set. Produce the two-curve plot: recall@k for k ∈ {1,3,5,8,12,20,30}, and LLM-judged answer quality at k ∈ {3,5,8,12,20}, with assembled token count on a second axis. *Pass criterion:* you can point to the k where the recall curve's slope drops below the quality curve's decline and say "this is my operating point, and here is the token cost of moving one step right."

### Explain how LLMLingua-style prompt compression actually works.

**Mental model: natural language is enormously redundant, and an LLM's attention does not need grammatical sentences to extract facts. Compression exploits the gap between what is needed to *read* text and what is needed to *understand* it** — you can delete a large fraction of tokens and leave the semantic content mostly intact, because the deleted ones were predictable from the ones that remain.

**📄 Paper:** Jiang, Wu, Lin, Yang & Qiu (2023, Microsoft) — *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models*, with the follow-up **LongLLMLingua** targeting long-context RAG specifically. It replaced "summarize the context with an LLM" — which is generative, slow, and hallucination-prone — with a *selection* procedure over the existing tokens.

The mechanism, three parts:

**A small language model as an information detector.** Run a small model (GPT-2 or 7B-class) over the prompt and compute each token's conditional probability given its prefix. **A token the small model predicts confidently carries little information** — it is recoverable from context — so it is a candidate for deletion. High-surprisal tokens carry the content. This is a direct application of information theory: surprisal *is* information, measured in bits.

**A budget controller** allocating different compression ratios to different parts of the prompt: instructions and the question get compressed lightly or not at all, retrieved demonstrations and documents get compressed hard. Coarse-grained selection first (drop whole low-value passages), then fine-grained token pruning within survivors.

**Question-aware compression (LongLLMLingua)** conditions the importance estimate on the query, so tokens relevant to *this* question survive even if they are individually predictable. This also lets it reorder documents by relevance, which addresses the lost-in-the-middle problem in the same pass.

Reported compression ratios in the papers reach quite high multiples on some tasks; **📅 treat headline ratios as task-specific rather than as a number you can assume.** In my experience 2–4× on RAG context, with careful budget allocation and a quality check, is the range that survives contact with a real eval set.

**⚠ Trap:** compressed prompts are **not human-readable and not diffable**, which destroys your debugging workflow. When an answer is wrong, you now cannot tell whether the model misread the evidence or the compressor deleted the qualifier that changed the meaning — "the policy does **not** apply to annual plans" is one token away from its opposite, and `not` is often high-probability in context and therefore a deletion candidate. **Always log the uncompressed context alongside the compressed one**, at least on a sample, or you have traded token cost for permanently degraded observability.

### When does the compression call cost more than it saves? Show me the arithmetic.

This is the question that kills most compression proposals, and it is pure arithmetic, so do it before you build.

**The two-model case (the honest one).** You compress with a small hosted model and generate with a frontier model. Compressing 8,000 tokens down to 3,000 at 2.7×:

- Compression call: 8,000 input tokens on the small model. At $0.25/Mtok: `8,000 × 2.5e-7 = $0.002`.
- Generation savings: 5,000 fewer input tokens at $3/Mtok: `5,000 × 3e-6 = $0.015`.
- **Net: $0.013 saved per query, a 6.5× return.** At 1M queries/month, `$13,000/month`. The economics work, and they work because of the **price ratio between the compressor and the generator** — 12× here.

**Now break it.** Use a same-tier model as the compressor: `8,000 × 3e-6 = $0.024` to save `$0.015`. **You have lost $0.009 per query.** This is not a hypothetical — "use the same model to summarize the context first" is a common design, and it is strictly worse than doing nothing on cost, plus a full extra round trip on latency.

**Break it a second way: prefix caching.** If your context is largely cacheable — a stable document set, a long system prompt — those input tokens cost `$0.30/Mtok` rather than `$3/Mtok`. Now the savings are `5,000 × 3e-7 = $0.0015` against a `$0.002` compression cost. **Negative again.** And compression is *actively hostile* to caching: a compressor whose output varies with the query produces a different prefix every time, so it destroys the cache hit you already had. **The two techniques are in direct conflict and you must choose one.** That interaction is the sophisticated point in this answer and it is the one that lands.

**The latency ledger, separately.** Compression adds a serialized model call — 8,000 input tokens through a small model is not instant, budget 200–400 ms. It saves prefill on the generator: 5,000 fewer tokens at ~50–100 ms/1k is 250–500 ms. **Roughly a wash, possibly slightly positive.** So compression is a cost play, not a latency play, and anyone selling it as a TTFT improvement has not measured it.

**🗣 Say this in the room:** "Compression only pays when the compressor is roughly an order of magnitude cheaper per token than the generator and the context isn't cacheable. If the context is cacheable, compression destroys the cache hit and usually goes net-negative — at that point the cheaper move is a better reranker with a smaller k, which cuts tokens without any extra call and improves precision at the same time."

That last clause is the real recommendation. **My default is not to compress. My default is to retrieve less and rank better**, because it costs nothing, adds no latency, adds no failure mode, and improves quality rather than trading against it.

### We shipped context compression and faithfulness scores dropped. Walk me through the debugging.

Faithfulness dropping after compression is the expected outcome of a specific and diagnosable set of causes, so I would work the list rather than tuning the ratio blindly.

**First, confirm the direction of causality with a paired diff, not an aggregate.** Take 200 queries, run both compressed and uncompressed, and compare per query. You are looking for the subset that regressed, not the mean — the mean tells you it got worse, the subset tells you why. Group the regressions by query type and by which chunks were compressed hardest.

Then work the causes in order of likelihood:

**1. Negation and qualifier deletion.** Compression is token-level and token importance is estimated by surprisal, so short function words that flip meaning — `not`, `except`, `unless`, `only`, `prior to` — are exactly the low-surprisal tokens that get pruned. The regression signature is **answers that state the opposite of the source**. Test directly: build a slice of 30 queries whose ground-truth answer hinges on a negation or an exception, and measure it separately. **If this slice regresses more than the aggregate, you have found it**, and the fix is a protected-token list in the compressor rather than a lower ratio.

**2. Numbers, dates and identifiers.** Same mechanism. `$10,000` and `30 days` are content-critical and can be pruned as individually predictable in a numeric context. Signature: quantitative answers regress while qualitative ones do not. Fix: protect any token matching a numeric, date, currency, or identifier pattern — a regex allowlist, applied before pruning.

**3. Citation breakage.** If compression runs *after* you have inserted `[S3]` markers and chunk metadata, it will happily prune the markers or the metadata lines, and now the model cannot cite and your faithfulness judge — which typically checks claim-against-cited-chunk — scores everything as unsupported. **Signature: faithfulness collapses while answer relevancy is unchanged.** That specific divergence between the two RAGAS-style metrics is diagnostic. Fix: compress chunk *bodies* only, never the scaffolding, and reinsert markers after compression.

**4. The judge is measuring against the compressed context.** A subtle one. If your faithfulness evaluator receives the compressed context as the reference, then a claim that was correctly derived from a deleted sentence is now scored unfaithful — the *system* is fine and the *measurement* broke. **Always evaluate faithfulness against the uncompressed source text**, and check this before you conclude anything, because it is a 20-minute check that can invalidate the whole investigation.

**5. Over-compression on already-short contexts.** A fixed 3× ratio applied to a 2,000-token context leaves 660 tokens, and there was no redundancy to remove. Compression ratios must be **budget-driven, not fixed** — compress only the amount needed to fit the budget, and skip entirely when already under it.

**🔍 Failure taxonomy — the decision procedure:** faithfulness down and relevancy flat → citations or scaffolding broken. Both down on numeric queries only → number pruning. Both down on negation queries only → function-word pruning. Both down uniformly across all slices → ratio is simply too aggressive; back it off and re-measure. Nothing down on a re-run against uncompressed reference → your evaluator was the bug.

### In a long conversation, retrieved chunks and chat history compete for the same window. How do you arbitrate?

**Mental model: history and retrieval carry different kinds of information and degrade differently, so a single global budget managed by "drop the oldest" is the wrong policy for both.** History carries *intent and constraints* — what the user already told you, what you already ruled out. Retrieval carries *facts*. Dropping history loses the constraints and the assistant starts repeating itself or contradicting an earlier commitment; dropping retrieval loses the grounding and it hallucinates. They are not substitutable.

The policy I ship:

**Fixed floors for both, competing only over the remainder.** History gets a hard floor of the last 2 user turns plus the last assistant turn verbatim; retrieval gets a hard floor of the top 3 chunks. Whatever budget remains after floors is allocated by a simple rule: **if the current turn is a follow-up that references prior conversation, favour history; if it introduces a new topic, favour retrieval.** Your query-understanding call already classifies this, so it costs nothing extra.

**Compact history rather than truncating it.** Beyond the verbatim window, replace older turns with a rolling structured summary — not a prose summary, a structured one, because structure is what survives compaction:

```
Established: user is on Enterprise plan, EU region, migrating from v3.
Resolved: SSO config (answered turn 2), seat limits (turn 4).
Open: whether audit logs export to S3.
Constraints stated: cannot change the IdP, needs it done before Nov 30.
```

That is ~60 tokens carrying what 3,000 tokens of raw dialogue carried, and it preserves exactly the things that cause visible failures when lost. Regenerate it every N turns, not every turn, and **cache it** — it changes slowly, so it can sit in the cacheable region of the prompt if you regenerate on a boundary.

**Retrieve over the history itself once it exceeds the window.** Past turns become a per-conversation index; the current turn retrieves from it. That is the correct architecture for long-running assistants and it turns "how do I fit 200 turns in the window" into an ordinary retrieval problem with an ordinary answer.

**⚠ Trap:** letting retrieved chunks push out the system prompt or tool definitions. I have seen assembly code that trims from the front when over budget, which deletes the instructions and the abstention rule while preserving chunk 8. **Trimming order must be explicit and encoded**: drop retrieved chunks from the lowest-ranked upward, then compact history further, and **never** touch system, tools, or the current question. Assert on that invariant — an assertion here is worth more than a test, because the condition only fires under budget pressure that your tests do not reproduce.

### Everyone quotes big context windows. What actually degrades as you fill one, and what do you do about it?

Three distinct degradations, and lumping them together as "the model gets worse" is what a weak answer sounds like.

**1. Position-dependent retrieval accuracy (lost in the middle).** Discussed above: a U-shaped curve, evidence in the middle of a long context is used less reliably. This is about *where* information sits, not how much there is.

**2. Effective context is well below advertised context.** A model advertising a very large window will typically show accurate multi-fact retrieval and reasoning over a substantially smaller span. **📄 Paper:** Hsieh et al. (2024, NVIDIA) — *RULER: What's the Real Context Size of Your Long-Context Language Models?* It built synthetic tasks harder than needle-in-a-haystack (multi-needle, aggregation, variable tracking) and showed that models' effective lengths fell well short of their claimed ones. **📅 Volatile:** the specific per-model numbers move with every release; the *finding* — advertised ≫ effective — has held across generations, and that is what you cite.

**3. Distractor sensitivity, sometimes called context rot.** As you add more plausible-but-irrelevant material, accuracy falls even when the needed information is present and well-positioned. **📅 Volatile:** the "context rot" framing was popularized by a 2025 technical report from the Chroma team; treat the terminology as recent and the effect as well-supported by the broader long-context literature. The mechanism is intuitive: attention is a soft selection over everything present, so every distractor claims some probability mass, and the softmax has no notion of "ignore this entirely."

**What you do about it, in order:**

**Retrieve less, rank better.** Point 3 says distractors have a real cost, so a reranker that cuts 20 chunks to 6 is directly attacking the degradation, not merely saving tokens.

**Put the best evidence at the boundaries.** Directly attacks point 1, costs nothing.

**Measure your own effective context.** Do not trust the spec sheet. Build a 30-query set where the answer requires combining two facts, and run it with the same evidence embedded in 4k, 16k, 64k and 128k tokens of surrounding corpus. **The k where accuracy first drops meaningfully is your operating limit**, and it is usually a fraction of the advertised window. This experiment takes an afternoon and it is the single most useful long-context measurement you can make.

**Prefer structure over volume.** A 6k-token context of deduplicated, reranked, clearly-delimited chunks with metadata beats a 60k-token context of raw retrieval on quality *and* on cost *and* on latency. **The default assumption that more context is safer is exactly backwards**, and correcting it head-on is worth saying: more context is more distractors, more prefill, more dollars, and — past your effective limit — measurably worse answers.

### You're 400 ms over your TTFT SLO. Give me the order in which you drop things.

I want this as a rehearsed ladder, because in an incident you do not want to be reasoning from first principles, and in an interview a crisp ordered answer with numbers is worth more than a thoughtful ramble.

**1. Check prefix cache hit rate first (0 ms to fix, potentially 200–300 ms recovered).** If it regressed, something moved into or in front of the stable prefix. This is the highest-value check because the fix is free and the cause is usually a recent commit. Do this before touching architecture.

**2. Parallelize serialized LLM calls (saves 200–450 ms each).** If routing, rewriting, and filter extraction are three sequential calls, merge them into one structured-output call. If HyDE runs before the base retrieval, run them concurrently instead. **Most pipelines have at least one accidental serialization and this is usually where the 400 ms is.**

**3. Make transformation conditional (saves 250 ms on 60–80% of traffic).** Skip the rewrite when the turn is self-contained; skip HyDE unless base retrieval confidence is low.

**4. Cut assembled tokens (saves ~50–100 ms per 1k).** Going from 10k to 6k tokens is 200–400 ms of prefill. Dedup, tighter top-k, and a reranker all get you here, and unlike the options below they do not cost quality — they usually improve it.

**5. Truncate reranker input (saves 80–150 ms).** Score on the first 256 tokens of each chunk instead of 512. Halves cross-encoder compute at a modest ranking cost.

**6. Shrink the candidate set (saves 50–100 ms).** Rerank 50 candidates instead of 100. Measure recall@50 vs recall@100 first — if the gap is under a point, this is free.

**7. Start streaming earlier.** Not a real latency reduction, but TTFT is a *perceived* metric: emitting "Searching your documentation…" and then the retrieved source titles while generation is still prefilling changes the user's experience of the same wall clock. **📅** Product-dependent, and I would flag it as a UX lever rather than an engineering one so nobody thinks I am gaming the metric.

**8. Only now: a smaller or faster generation model.** Last, because it is the only lever on this list that trades away answer quality directly rather than trading away redundancy.

**💰 Math for the ladder:** on a pipeline running two serialized transformation calls (600 ms), a 12k context (900 ms prefill), and 100-candidate reranking at 512 tokens (220 ms), steps 2–6 recover `450 + 250×0.7 + 300 + 110 + 75 ≈ 1,110 ms` of p95 — nearly 3× the deficit — **without touching the model.** That is why "use a faster model" is the last resort and not the first: the architecture usually has a second of slack in it that nobody has looked for.
