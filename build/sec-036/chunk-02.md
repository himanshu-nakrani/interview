### Explain parent-document retrieval. When does it win and what does it cost?

Parent-document retrieval, also sold as "small-to-big," is the cleanest expression of the decoupling idea: **index the precise thing, return the complete thing.** You chunk each document twice. The child chunks are small — 128–256 tokens, one idea each — and those are the only things you embed and put in the vector index. Each child stores a pointer to its parent: a larger enclosing unit, typically 1,000–2,000 tokens, usually a section or a page or a fixed window. At query time you search over children, take the top-k, dereference to their distinct parents, dedupe, and hand the parents to the generator.

Why this works is worth stating precisely rather than hand-waving. A dense vector has fixed capacity. The signal-to-noise of the vector is roughly "how much of this text is about the one thing the query asks." A 128-token child that is entirely about the idle connection timeout produces a vector *about* the idle connection timeout. A 2,000-token parent containing that plus nine other configuration topics produces a vector about "configuration," which sits near a hundred other sections. **So you get the retrieval precision of small chunks and the context completeness of large ones, and the only thing you pay is a join.**

The implementation is a document store keyed by parent id sitting next to the vector index — which for you is trivially Postgres, and that adjacency is exactly the argument for keeping vectors in Postgres too when scale allows. Do not store parent text in the vector database's payload: you would be storing each parent once per child, an 8× duplication of your entire corpus in the most expensive storage tier you own.

```python
# ingest
for parent in split_by_section(doc, target=1500):
    pid = put_parent(parent)                      # Postgres row
    for child in chunk_text(parent.text, enc, max_tokens=200, overlap_tokens=0):
        index.add(embed(child), payload={"parent_id": pid, "doc_id": doc.id})

# query
hits = index.search(embed(query), k=20)
parent_ids = list(dict.fromkeys(h.payload["parent_id"] for h in hits))[:5]  # ordered dedupe
context = fetch_parents(parent_ids)
```

**💰 Math — the real cost is context tokens, not storage.** k=20 children at 200 tokens would be 4,000 tokens of context; dereferencing to 5 distinct parents at 1,500 tokens is 7,500. So you roughly double prompt size versus flat chunking at the same k. At $3/Mtok input, 7,500 vs 3,000 tokens is $0.0225 vs $0.009 per query — $13.50/day extra at 1,000 queries/day, $405/month. Against that: it is routinely the single largest end-to-end quality jump available in a naive RAG system, because "retrieved the right thing but not enough of it" is the most common failure in production. **I would pay $405/month for that every time**, and I would say the number out loud in the design review rather than saying "small extra cost."

**⚠ Trap:** the deduplication step. Twenty children frequently map to three parents, and if you forget `dict.fromkeys` (or equivalent ordered dedupe) you will send the same 1,500-token parent five times, blow your context budget, and — because models weight repeated content more heavily — bias the answer toward whichever section happened to have many matching children. I have reviewed this bug three separate times. It presents as "the model keeps talking about section 4."

**⚠ Trap, second one:** the parent boundary is now the boundary that matters, and it has all the same problems the child boundary had. If your parent is a fixed 1,500-token window rather than a real section, you have just moved the table-bisection problem up a level. Parents should be structural units.

### How is sentence-window retrieval different from parent-document retrieval, and when would you pick it?

They are cousins with a meaningfully different granularity contract. **Parent-document dereferences to a fixed structural unit; sentence-window dereferences to a symmetric neighborhood around the hit.**

In sentence-window retrieval, the indexed unit is a single sentence (or two). Each indexed sentence stores, as metadata, the *k* sentences before and after it — typically 3 each way. You embed only the center sentence. At query time you retrieve sentences and expand each hit to its stored window before assembling context.

The differences that matter:

**Window is centered on the hit; parent is not.** If a match lands at the very end of a section, parent-document gives you 1,500 tokens of preceding material and nothing after — and the continuation might be the actual answer. Sentence-window always gives you symmetric context. For corpora with weak or absent section structure (transcripts, long-form articles, chat logs, OCR'd text with no reliable headings) this is a real advantage, because there is no structural parent to dereference to.

**Windows overlap, parents do not.** Two adjacent sentence hits produce two windows sharing five of seven sentences. Deduplication is therefore not optional — you must merge overlapping windows into a union span rather than concatenating them. Parent-document's dedupe is a simple set operation; sentence-window's is an interval merge.

**Precision is higher and brittleness is higher.** A one-sentence embedding is the most precise retrieval unit available, which is great when the query is a specific factual lookup and terrible when the query is thematic ("summarize the risks discussed") — no single sentence is about the theme, so nothing scores well and you retrieve seven unrelated sentences.

```python
def merge_windows(hits, ctx=3):
    spans = sorted((max(0, h.sent_idx - ctx), h.sent_idx + ctx + 1) for h in hits)
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged
```

**My decision rule:** structured corpus with real headings → parent-document, because the structural unit is more meaningful than a symmetric window and the dedupe is simpler. Unstructured or flat corpus, or a corpus where the answer is a specific fact rather than a discussion → sentence-window. Question type skewed thematic/summarization → neither; you want a summary index or RAPTOR, covered below.

**⚠ Trap:** using sentence-window with a sentence splitter that has not been tuned for the domain. `nltk`'s Punkt or a naive `. ` split will shatter on "Fig. 3", "et al.", "U.S.C. § 1983", "v1.2.3", and decimal numbers. Your "sentences" become fragments, your embeddings become garbage, and the windows are misaligned. Use a proper segmenter (spaCy's sentencizer with domain abbreviation lists, or a rule set you actually tested) and — this is the part people skip — **assert a sentence-length distribution in CI**: if median sentence length drops below ~8 tokens, your splitter regressed.

### Explain contextual retrieval. What does it actually do and what are the reported numbers?

Contextual retrieval attacks the root cause of the anaphora problem directly: **before embedding a chunk, use an LLM to write a short passage that situates that chunk inside its document, and prepend it.** The chunk stops being a fragment and becomes a fragment plus its own briefing note.

Mechanically, for each chunk you make one small-model call with the whole document (or a large window of it) plus the chunk, and a prompt along the lines of "Here is a document. Here is a chunk from it. Write a short standalone context, 50–100 tokens, that situates this chunk within the document for search purposes. Answer only with the context." The output is prepended to the chunk text, and **the concatenation is what you embed and what you put in the BM25 index.** Both channels benefit, which is important: the generated context typically contains named entities, dates, section topics and document titles that the raw chunk lacked, and those are exactly the tokens that make lexical retrieval work.

A before/after makes the effect obvious. Raw chunk: *"Revenue grew 3% over the previous quarter."* Contextualized: *"This chunk is from ACME Corp's Q2 2023 SEC 10-Q filing; the previous quarter's revenue was $314 million. Revenue grew 3% over the previous quarter."* The first is retrievable by essentially no realistic query. The second is retrievable by "ACME Q2 2023 revenue growth," which is what someone will actually type.

**📄 Source:** Anthropic published this as an engineering write-up ("Introducing Contextual Retrieval," 2024) rather than a peer-reviewed paper, and the reported figures were: contextual embeddings alone reduced top-20 retrieval failure rate by ~35%; contextual embeddings combined with contextual BM25 reduced it by ~49%; adding a reranking stage on top brought the total reduction to ~67%. **Cite the mechanism confidently and the numbers as "reported by Anthropic in their engineering write-up"** — they are from one internal evaluation across several datasets, not an independent benchmark, and an interviewer who knows that will respect the hedge.

The reason it is a big deal is that it is one of very few techniques that improves the *dense and the lexical channel simultaneously* and requires no change at query time at all. Query latency is unaffected. All the cost moves to ingestion, where you can batch it, run it off-peak, and retry it freely. That asymmetry — pay at index time, free at query time — is the thing I look for in any RAG optimization, and it is why I rank contextual retrieval above most query-time tricks.

**⚠ Trap:** prepending context and then forgetting to also index it lexically. If you embed `context + chunk` but your BM25 index still holds the raw chunk, you have bought roughly the 35% number, not the 49% number, and you will not know why. The concatenated text must be the single canonical `chunk.text` that flows into both indexes and into the generator's context.

**⚠ Trap two:** letting the LLM restate the chunk instead of situating it. If the generated context paraphrases the chunk content, you have paid tokens to duplicate information and you have diluted the embedding with near-repetition. Constrain the prompt to *document-level* facts the chunk lacks — title, date, entity, section, what the surrounding discussion is about — and spot-check 50 outputs by hand before running it on 10 million.

### What does contextual retrieval cost at 10 million chunks? Show me the arithmetic and tell me whether you'd do it.

This is where the technique lives or dies and the arithmetic is the answer.

**The naive cost is catastrophic, and you should compute it first to show you understand why caching is not optional.** Each chunk's context call needs the whole document in the prompt. If the average document is 30,000 tokens and produces 60 chunks of 500 tokens, then processing that document naively is 60 calls × 30,000 input tokens = **1.8M input tokens to produce 60 × 80 = 4,800 output tokens.** You are re-sending the same document 60 times. For 10M chunks (≈167k documents at 60 chunks each): 167k × 1.8M = 3.0×10¹¹ input tokens. At even $0.25/Mtok that is **$75,000**. This is why people try the technique on 1,000 chunks, love it, and then abandon it.

**💰 With prompt caching, the same work collapses.** Cache the document prefix once per document; subsequent calls in that document's batch read the cached prefix at a large discount. Using representative numbers — $1.00/Mtok base input, cache *write* at 1.25× base, cache *read* at 0.1× base, $5.00/Mtok output **📅 Volatile: cache write/read multipliers and per-token prices change; re-verify against the provider's current pricing page before quoting these** — per document:

- Cache write, once: 30,000 tok × $1.00/M × 1.25 = **$0.0375**
- Cache reads, 60 calls: 60 × 30,000 × $1.00/M × 0.1 = **$0.180**
- Per-call chunk text (uncached suffix): 60 × 500 × $1.00/M = $0.030
- Output: 60 × 80 × $5.00/M = **$0.024**
- **Per document: ≈ $0.2715.** Per chunk: **≈ $0.0045.**

For 10M chunks: 167k documents × $0.2715 = **≈ $45,000.** Still not cheap. Now the levers, and this is the part that shows judgment:

1. **Use the cheapest capable model.** This is a summarization-with-context task, not reasoning. Dropping from a $1/$5 model to a $0.10/$0.40 class model cuts it ~10× to **≈ $4,500**. I would run a 200-chunk head-to-head and I would expect the cheap model to be indistinguishable here.
2. **Use a batch/offline tier**, typically ~50% off, for ingestion that does not need to be online: **≈ $2,250**.
3. **Don't contextualize everything.** Chunks that are already self-contained — ones that begin with a heading, contain the document title, or have no leading anaphora — gain little. Gate on the cheap dangling-reference regex from earlier plus a length threshold and you typically skip 40–60% of chunks: **≈ $1,000–1,300**.
4. **Truncate the document window.** Most chunks need the document *header* (title, date, parties, abstract) plus their own section, not all 30k tokens. Sending 4k instead of 30k cuts the dominant term by 7×.

**So the honest answer is: the naive implementation is $75k and the engineered one is low single-digit thousands, for a one-time backfill, plus a marginal per-document cost on new ingestion.** Ongoing: if you ingest 5,000 new documents/day at $0.03/doc (post-optimization) that is $150/day, $4,500/month, and *that* is the number that decides it — the backfill is a one-off, the run rate is forever.

**🗣 Say this in the room:** "Contextual retrieval is an index-time cost with zero query-time latency, which is the trade I want. Naively it's about 45 cents per hundred chunks because you re-send the document per chunk; with prompt caching on the document prefix, a small model, the batch tier, and skipping chunks that are already self-contained, I've seen it land near half a cent per chunk. For a 10M-chunk corpus that's a ~$1–2k backfill and a few thousand a month on new ingest. I'd fund that against a reported ~49% cut in top-20 retrieval failures — but I'd validate the 49% on our own golden set before committing the backfill."

**⚠ Trap:** forgetting that contextualized chunks are now **coupled to the document**. If the document changes, every chunk's context is potentially stale, not just the chunks whose text changed. Your incremental-reprocessing logic must treat a document-level edit as invalidating contexts corpus-wide *for that document*, which is a materially bigger reprocessing unit than "re-embed the two chunks that changed." Store the document content hash on every chunk so you can find them.

### Contrast contextual retrieval with late chunking. Which would you reach for?

They solve the same problem — chunk embeddings that lack document context — by opposite means, and the comparison is a good interview answer because it shows you can reason about cost structure rather than collect techniques.

**Contextual retrieval adds tokens.** It uses an LLM to generate new text that carries the missing context, prepends it, and embeds the enlarged chunk. The context becomes literal, visible, human-readable text that also helps BM25 and also helps the generator understand the chunk.

**Late chunking adds no tokens; it changes when you pool.** The trick, published by Jina AI (2024), exploits the fact that a long-context embedding model is a transformer: run the *entire* document (up to the model's context limit, e.g. 8k tokens) through the encoder once, producing one contextualized token embedding per token — each of which has attended to the whole document. *Then* apply your chunk boundaries and mean-pool the token embeddings within each chunk's span to get that chunk's vector. The chunk's vector is now informed by the whole document even though the chunk text is unchanged.

The comparison, on the axes that decide it:

| | Contextual retrieval | Late chunking |
|---|---|---|
| Ingest cost | One LLM call per chunk (cacheable, still real) | One encoder pass per document — *cheaper* than embedding chunks separately, since you encode each token once |
| Helps BM25? | Yes — new literal tokens enter the lexical index | No — chunk text is unchanged |
| Helps the generator? | Yes — the context is visible in the prompt | No — invisible, purely a vector effect |
| Requires | Any LLM | A long-context embedding model **and** an embedding API that exposes token-level output or self-hosting |
| Document length limit | Effectively none (you window the prompt) | Hard-capped at the encoder's context length; a 100-page PDF does not fit |

**The practical blocker for late chunking is the last row plus API access.** Most hosted embedding endpoints return one pooled vector per input, not per-token states — so late chunking usually means self-hosting the encoder. If you are already self-hosting embeddings on your own GPUs, late chunking is nearly free and I would absolutely take it. If you are calling a hosted embedding API, it is not available to you and the question is moot.

**They are also composable**, which is the answer I would actually give: contextual retrieval helps the lexical channel and the generator; late chunking helps the dense channel at near-zero cost. If I self-host embeddings I would run late chunking as the default and add contextualization only for the subset of chunks my dangling-reference detector flags.

**🗣 Say this in the room:** "They fix the same defect from opposite directions — contextual retrieval writes the missing context as tokens, late chunking recovers it from attention without adding tokens. Late chunking is cheaper but needs token-level access to a long-context encoder and can't span documents longer than its context window. Contextual retrieval is more expensive but also improves BM25 and the generator's understanding, which late chunking structurally cannot."

### Tell me about proposition-based indexing. When does decomposing into atomic facts backfire?

Propositions push granularity to its limit: **rewrite the corpus into standalone atomic factual statements, each self-contained and decontextualized, and index those.** "The Space Needle was completed in 1962 for the World's Fair" becomes two propositions: "The Space Needle was completed in 1962." and "The Space Needle was built for the 1962 World's Fair." Every pronoun is resolved, every reference is expanded, every compound sentence is split.

**📄 Paper:** Chen, Wang et al. (2023), *Dense X Retrieval: What Retrieval Granularity Should We Use?* — introduced propositions as a retrieval unit and released FactoidWiki, a proposition-level version of Wikipedia. The reported finding was that proposition-level retrieval outperformed sentence- and passage-level on several open-domain QA benchmarks, particularly for questions about rare entities, because a proposition's embedding is undiluted by neighboring facts.

When it works, the reason is clean: maximum precision per vector, and every unit is self-contained by construction so the anaphora problem is definitionally solved.

**Where it backfires, and this is the part interviewers probe:**

**It destroys everything that is not a fact.** Procedures are the clearest casualty. "Step 3: turn the valve clockwise" decomposed into a proposition loses its position in the sequence and its dependency on steps 1 and 2. Hand that to a model and it will tell a technician to turn a valve without depressurizing first. Any corpus that is instructions, workflows, code, legal reasoning chains, or argumentation is actively damaged by atomization, because the *relations between* statements are the content.

**It multiplies your index 3–10×.** A 500-token chunk yields maybe 15–25 propositions. Your 10M-chunk corpus becomes 150M+ vectors. That is a different infrastructure tier — memory, index build time, and cost all move by an order of magnitude.

**It requires an LLM pass over the entire corpus with a high error surface.** Proposition extraction hallucinates. It will invent a subject when resolving a pronoun ambiguously, and now you have a *fabricated fact indexed as ground truth*, unattributable to any source sentence, that will be retrieved and cited. This is qualitatively worse than a bad chunk boundary: a bad boundary loses information; a bad proposition *creates* it.

**Aggregation and counting break.** "How many exceptions does the policy list?" — the propositions each state one exception; nothing states the count; no top-k retrieval recovers it.

**My rule:** propositions are for encyclopedic, fact-dense, entity-centric corpora where questions are lookups — product catalogs, knowledge bases, structured reference material. They are wrong for anything procedural, argumentative, or sequential. And if I use them, I keep the source span on every proposition and **cite the original sentence, never the proposition**, so a hallucinated decontextualization is at least traceable and visible to the user.

**⚠ Trap:** the demo effect. Propositions look spectacular on a hand-picked set of entity lookups and quietly wreck your procedural queries. If your eval set is skewed toward factoid questions — which LLM-generated golden sets always are, because generating factoid questions is what models do by default — propositions will win your ablation and lose in production. **Stratify your eval set by question type before you believe any granularity result.**

### Walk me through RAPTOR. What problem does it solve that reranking cannot?

RAPTOR exists for a query class that flat retrieval structurally cannot serve: **questions whose answer is not in any chunk.** "What are the main themes across these 400 support tickets?" "How did the company's risk disclosure change between 2022 and 2024?" "Summarize the argument of this book." No top-k over leaf chunks answers these, because the answer is an aggregate. A reranker cannot help — it reorders candidates, and none of the candidates contains the answer.

**📄 Paper:** Sarthi et al. (2024), *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* (ICLR 2024). It replaced the assumption that the retrieval index must be a flat set of source chunks with a hierarchy of generated summaries.

**Mechanism, concretely:**
1. Chunk the corpus normally. These are the tree's leaves. Embed them.
2. **Cluster** the leaf embeddings. The paper uses soft clustering — dimensionality reduction with UMAP, then a Gaussian mixture model with the number of components chosen by BIC — so a chunk can belong to more than one cluster, which matters because documents genuinely span topics.
3. For each cluster, concatenate its members and **have an LLM summarize them.** That summary becomes a new node at level 1.
4. Embed the level-1 summaries and repeat: cluster, summarize, ascend. Stop when you have a handful of nodes or hit a depth cap.
5. **Index every node at every level in one flat vector index.** This is the "collapsed tree" retrieval mode, which the paper found outperformed traversing the tree top-down, and it is also far simpler: at query time you just search the whole index, and a thematic query naturally matches a high-level summary while a specific query matches a leaf.

The elegance is that step 5 requires no query classification. The vector space does the routing for you.

**💰 The cost, which is the question that follows.** Every level of the tree costs an LLM pass over (roughly) the whole corpus, at a compression ratio. With ~10:1 clustering, a 100M-token corpus generates ~10M tokens of level-1 summaries, ~1M at level 2, ~100k at level 3. Input tokens processed: 100M + 10M + 1M ≈ 111M; output ≈ 11.1M. At a small-model $0.10/$0.40 per Mtok **📅 Volatile**: 111 × $0.10 + 11.1 × $0.40 = $11.10 + $4.44 = **≈ $15.50** — cheap. At a $3/$15 frontier model: 111 × $3 + 11.1 × $15 = $333 + $167 = **$500**, and summarization quality is the whole value of the tree, so you cannot cheap out arbitrarily. Index size grows ~11%. The real cost is **rebuild semantics**: clusters are global, so adding documents invalidates the clustering. A truly incremental RAPTOR is an open problem; in practice you rebuild the tree on a schedule (nightly, weekly) and accept that summaries lag ingestion.

**⚠ Trap:** shipping RAPTOR on a corpus whose queries are all specific lookups. You will have paid for summary generation, added an 11% index tax, taken on a rebuild pipeline, and moved your metrics by nothing — because the leaves were always what matched. **Before proposing RAPTOR, classify your query log: what fraction are aggregative or thematic?** If it is under ~10%, the honest recommendation is "don't," and saying so is a stronger signal than implementing it.

**🗣 Say this in the room:** "RAPTOR indexes generated summaries alongside source chunks, so thematic queries can match a node that actually contains the aggregate answer. It's the right tool when a meaningful slice of your query log is 'summarize/compare/what are the themes,' and it's expensive theater when your queries are lookups. I'd measure that fraction from the query log first."

### What's a summary index, and what's the multi-vector trick where you index something other than the chunk text?

The general principle is the one that unifies half this section: **the thing you embed does not have to be the thing you return.** Once you internalize that, a family of techniques falls out.

**Summary index (document level).** Embed a generated summary of each whole document; the payload points at the document. Retrieval over summaries answers "which document is about X" — a routing question — and is dramatically better at it than retrieval over chunks, because a chunk from a 200-page manual tells you almost nothing about what the manual is for. The standard architecture is two-stage: retrieve documents by summary, then retrieve chunks *within* the selected documents. This is exactly the coarse-then-fine pattern of IVF, applied at the semantic layer, and describing it that way lands well.

**Hypothetical-question indexing.** For each chunk, generate 3–5 questions that the chunk answers, and index the questions rather than (or alongside) the chunk. The motivation is the **query-document asymmetry problem**: a user query is short and interrogative, a chunk is long and declarative, and even models trained with asymmetric prefixes only partially close that gap. Embedding a question puts your index in the same distributional neighborhood as the incoming query, so cosine similarity is measuring like against like. It is HyDE run at index time instead of query time — and index time is the better place, because it is paid once and adds zero query latency. Cost: ~5 extra LLM generations and 5 extra vectors per chunk, so a 5× index and one cheap-model call per chunk.

**Multi-vector, generally.** Store several vectors per chunk — the raw text, a summary, generated questions, extracted keywords — all pointing at the same payload. Retrieve as a union, dedupe by payload id, rerank. You are hedging across representations, and it works for the same reason ensembling works.

**⚠ Trap:** exploding your top-k budget. If a chunk has 6 vectors and you retrieve top-20, you might get 20 vectors covering 4 distinct chunks. **Retrieve top-k *after* deduplicating by payload id, not before** — which means over-fetching (say, 60 vectors) and collapsing. Every vector database makes this awkward and every team gets it wrong once.

**⚠ Trap two:** generated questions drift from real questions. Models generate well-formed, complete, grammatical questions; users type "revenue q3?" and "why 401 error". If your generated questions are all polished prose, the distributional match you were buying does not materialize. Seed the generator with 30 real queries from your logs as few-shot examples and the outputs get dramatically closer.

**💰 Math:** hypothetical-question indexing on 1M chunks at 500 tokens each: 1M calls × (500 in + 120 out). At $0.10/$0.40 per Mtok: 500M × $0.10/M = $50, plus 120M × $0.40/M = $48 → **$98 one-time**. Extra embeddings: 5M questions × ~25 tokens = 125M tokens at $0.02/M = **$2.50**. Extra index: 5M vectors × 1024 dim × 4 bytes = **20GB**, which at typical managed-vector-DB pricing is where the real recurring cost lands, not the generation. **Always price the storage, not just the tokens** — for multi-vector schemes storage dominates within a month.

### I've got a 400-page equipment manual. Users ask both "what's the torque spec for the M8 bolts" and "what's the general procedure for a hydraulic service." How do you index it?

One corpus, two query classes with opposite granularity needs, and the wrong answer is to pick a chunk size that splits the difference — it will serve both badly. The right answer is a **multi-granularity index with a single flat search surface.**

**Build three levels into one vector index**, each node tagged with its `level`:

- **Level 0 — leaves (~200 tokens).** Structure-aware: split on the manual's own numbered section hierarchy, never across a table or a procedure step list. Each leaf carries the full section path prefix (`5. Hydraulics > 5.3 Service > 5.3.2 Torque specifications`) in its text. The torque question hits here, and it hits precisely because the leaf is small enough that "M8" and "torque" and a number dominate the vector.
- **Level 1 — sections (~1,500 tokens, or a generated summary if longer).** One node per H2/H3 section. The "general procedure" question hits here, because a whole procedure is a section and its summary is *about* the procedure.
- **Level 2 — chapters.** One generated summary per chapter, ~300 tokens. Catches "what does this manual cover about hydraulics."

Search all levels in one query, then apply a **level-aware assembly rule**: if a level-0 hit's parent section is also in the result set, drop the leaf and keep the section (it subsumes it); if only leaves hit, expand each to its parent for context. This is parent-document retrieval generalized to a tree, and it is about forty lines of assembly logic.

**Then handle the part that is not retrieval at all.** "Torque spec for M8 bolts" is a *lookup*, not a search. In a manual, torque specs live in tables. I would extract every table into a structured store at ingest — a Postgres table of (doc, section, table_id, row_label, column_label, value, unit, page) — and route queries that look like spec lookups to a SQL/exact-match path with the vector path as fallback. **Retrieving a table chunk and asking the model to read the right cell is strictly worse than reading the cell yourself**, and the failure mode (adjacent row, wrong unit) is exactly the kind that gets someone hurt with a torque wrench. Structured extraction plus an exact path is the senior answer here and it is the one most candidates do not give.

**Also: units.** Manuals mix N·m and ft·lb. Normalize at extraction, store both, and render both in the answer with the source's original value cited. A unit error in a torque spec is a safety incident, and I would put a hard eval on it — 40 golden spec lookups, exact-value assertion, run on every parser or chunker change.

**🗣 Say this in the room:** "Two query classes with opposite granularity needs means two levels in the index, not a compromise chunk size — I'd index leaves, sections, and chapter summaries into one flat vector space with a level tag, and use a subsumption rule at assembly. And I'd pull the spec tables into a relational store with an exact lookup path, because reading a cell out of a retrieved markdown table is a hallucination waiting to happen."

### How does chunking change for legal contracts versus API docs versus Slack threads versus earnings-call transcripts?

The point of this question is whether you have a *procedure* for deriving strategy from corpus properties, rather than four memorized recipes. My procedure asks four things about any corpus: **What is the atomic semantic unit? What is the natural query? Where does meaning cross boundaries? What must never be split?** Then the strategy falls out.

**Legal contracts.** Atomic unit: the numbered clause. Natural query: "what does the agreement say about termination for convenience." Meaning crosses boundaries constantly — defined terms ("Confidential Information" is capital-D defined in §1 and used in §14), cross-references ("subject to Section 9.3"), and exhibits. Never split: a clause, or a definition from its term. **Strategy:** split on the clause numbering hierarchy, never by token count; prepend the full clause path and the contract's parties/date to every chunk; **build a defined-terms glossary at ingest and append the definitions of every capitalized defined term that appears in a chunk to that chunk.** That last move is contextual retrieval specialized to legal text and it is worth more than any generic technique here. Chunk sizes end up small and highly variable — 100 to 800 tokens — and that is correct.

**API documentation.** Atomic unit: the endpoint or the symbol. Natural query: "how do I paginate the list-invoices endpoint," or a literal like `POST /v1/invoices`. Meaning crosses boundaries into shared concepts (auth, pagination, error codes) referenced from every page. Never split: a code example from its explanation; a parameter table from its endpoint. **Strategy:** one chunk per endpoint/symbol, structure-derived. Duplicate shared concept text into chunks that reference it (yes, deliberately duplicate — auth is 80 tokens and it is worth repeating). **This corpus is dominated by exact literals**, so the lexical channel is doing most of the work and BM25 weighting should be high in your hybrid. Prepend the HTTP method and path as literal text.

**Slack threads / support tickets.** Atomic unit: the *thread*, not the message. Natural query: "has anyone hit this deploy error before." Meaning crosses boundaries via reply structure and implicit reference to shared state. Never split: a thread; a question from its accepted answer. **Strategy:** chunk by thread; if a thread exceeds budget, split at long time gaps rather than token counts. Prepend a synthesized header (channel, participants, date, and an LLM-generated one-line thread summary). Filter aggressively: bot messages, join/leave, standups, and threads with no reply are pure noise and typically 40–60% of the volume. **The dominant win here is filtering, not chunking**, and saying so is the mature answer.

**Earnings-call transcripts.** Atomic unit: the speaker turn, or the Q&A pair. Natural query: thematic and comparative — "what did the CFO say about margin pressure." Meaning crosses boundaries via extended answers and analyst follow-ups. Never split: a Q&A pair. **Strategy:** chunk on speaker turns, merge each analyst question with the full answer into one unit, prepend speaker name, role, company, quarter and date. This is the one corpus in the four where semantic chunking is a defensible fallback, because prepared remarks are long, boundary-free, and genuinely topic-shifting. Also the strongest candidate for RAPTOR, because "how has the margin story changed across four quarters" is a real and common query and is unanswerable from any leaf.

**🗣 Say this in the room:** "I don't have four recipes, I have four questions I ask about a corpus: what's the atomic unit, what's the natural query shape, where does meaning cross boundaries, and what must never be split. For contracts the atomic unit is the clause and the killer detail is injecting defined-term definitions; for Slack the atomic unit is the thread and the killer detail is that half the volume is noise you shouldn't index at all."

### Half my index is boilerplate — the same legal disclaimer, the same email signature, the same template header. What does that do and how do you handle it?

It does three specific kinds of damage and each has a different remedy, so name them separately.

**Damage 1: IDF destruction in the lexical index.** If 60% of chunks contain "This message is intended only for the named recipient," then every term in that sentence has document frequency ≈ 0.6N and IDF ≈ 0. Fine for those words — but the collateral damage is that any *legitimate* query containing "recipient" or "intended" now has a dead term, and if your query is short, you have lost a meaningful fraction of your scoring signal. Worse in your own domain: if every document says "Acme Corp," you have destroyed the discriminating power of your own most important entity.

**Damage 2: embedding-space collapse.** Shared text drags every chunk's vector toward a common component. The practical symptom is a compressed similarity distribution — your top-1 scores 0.84, your top-50 scores 0.81, and the ranking is nearly arbitrary within that band. **This is measurable and you should measure it:** sample 1,000 random chunk pairs and plot the cosine distribution. A healthy corpus has a broad distribution centered well below 0.5; a boilerplate-poisoned corpus has a narrow spike near 0.75+. That histogram is a diagnostic I run on every new corpus and it takes five minutes.

**Damage 3: wasted context and wasted money.** If a 120-token disclaimer rides along on every retrieved chunk and you retrieve 5, you spend 600 tokens per query on nothing. At 100k queries/day and $3/Mtok that is 600 × 100,000 × $3/1e6 = **$180/day, $5,400/month, to send a legal disclaimer to a language model.** That arithmetic ends the debate in any design review.

**Remedies, in order:**

**Exact near-duplicate detection at ingest.** Shingle each chunk into overlapping 5-grams, MinHash them, and bucket with LSH; anything above ~0.9 Jaccard is a duplicate family. For repeated *spans within* documents (as opposed to whole duplicate chunks), the cheaper approach is the frequency-based one from the header/footer question: hash normalized paragraphs across the corpus, and any paragraph appearing in more than ~30% of documents is boilerplate. Strip it, log it, and keep a sample so you can review what you removed.

**Keep one canonical copy.** If the disclaimer is genuinely queryable ("what's our standard confidentiality notice"), index it exactly once as its own document rather than 400,000 times as a passenger.

**Do not deduplicate before you look.** The failure I have made: aggressive dedup removed a 200-token "Safety Warnings" block that legitimately appeared in every equipment manual, and that block was the answer to a real class of user question. **Boilerplate removal must be reviewed by a human on a sample, once, before it runs on the corpus.** Print the top 50 repeated blocks with their document frequency and have someone read them. It takes twenty minutes and it prevents deleting the thing users ask about most.

**⚠ Trap:** deduplicating at the *chunk* level only. Boilerplate frequently appears mid-chunk, welded to unique content, so no chunk is a duplicate of another chunk and MinHash finds nothing while the pollution is total. You need span-level detection inside documents, not just chunk-level detection across them.

### You've got two weeks, a naive RAG system, and four techniques on the table: parent-document, contextual retrieval, RAPTOR, propositions. What order do you do them in and why?

This is a prioritization question and the answer they want is a decision procedure, not a ranking. Mine has one rule at the top: **do not implement anything until you have measured which failure mode you have.** So week one is not implementation.

**Days 1–3: build the golden set and the failure taxonomy.** 200 queries from logs, span-labeled. Then run the existing system and classify every failure into exactly one bucket:

- **A — the answer isn't in the corpus.** No retrieval technique fixes this. Escalate to a content problem. (Routinely 15–30% of "RAG failures" and nobody measures it.)
- **B — the right chunk was never retrieved** (not in top-50). Retrieval recall problem.
- **C — the right chunk was retrieved but ranked below the cutoff.** Ranking problem → reranker, not chunking.
- **D — the right chunk was retrieved and in context, but the answer was wrong or incomplete.** Context sufficiency problem.
- **E — the answer required aggregating across many chunks.** Structural problem.

**Now the four techniques map one-to-one onto buckets, and that mapping is the whole answer:**

- Bucket **D** dominant → **parent-document retrieval.** Half a day of work, no new model calls, no backfill, immediate measurable effect. This is always first if D is non-trivial, because it is the cheapest fix in the entire section.
- Bucket **B** dominant → **contextual retrieval** (after checking you have hybrid + a reranker at all; if you do not, add those first, they are cheaper and the reported 49% figure assumed hybrid). Days of work plus a real backfill cost, so it comes after the free thing.
- Bucket **E** non-trivial (>10% of queries) → **RAPTOR.** New pipeline, rebuild semantics, real ongoing ops. Justified only by a measured aggregative query share.
- Bucket **B** dominant *and* the corpus is encyclopedic/factoid *and* you have already tried contextual retrieval → **propositions.** Last, because it is the highest-risk (it can fabricate) and highest-cost (5–10× index).

**So the default two-week plan** if the taxonomy comes back with the usual shape (D heaviest, then B): days 1–3 eval, day 4 parent-document + measure, days 5–6 hybrid + reranker if absent + measure, days 7–10 contextual retrieval on a 10% sample, measure, then decide on the full backfill, days 11–14 backfill and re-measure. RAPTOR and propositions do not get implemented, and **the deliverable that gets you promoted is the document explaining why not, with the numbers.**

**🗣 Say this in the room:** "I'd spend the first three days not building anything — 200 labeled queries and a failure taxonomy that separates 'not retrieved' from 'retrieved but insufficient' from 'needs aggregation,' because those three have completely different fixes and the cheapest fix maps to the most common bucket. Parent-document expansion is usually the first thing I ship because it's half a day and no backfill. Contextual retrieval is second because it costs a backfill. RAPTOR and propositions I'd expect to argue *against* unless the taxonomy justifies them."

**⚠ Trap:** the enthusiasm trap, which is real and is being tested. Candidates who list every advanced technique they have read about signal that they have read, not shipped. The candidates who get hired name the measurement first and then decline to build three of the four things. Declining, with a number attached, is the senior move.
