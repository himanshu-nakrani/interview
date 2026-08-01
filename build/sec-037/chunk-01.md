### Everyone says "RAG" but the term has drifted. What did the original RAG paper actually propose, and how is that different from what we ship today?

The mental model: **the 2020 paper proposed a trained architecture; what we call RAG today is a prompt-assembly pattern.** Those are different things that share a name, and knowing the difference is a cheap signal of literacy.

**📄 Paper:** Lewis, Perez, Piktus, Petroni, Karpukhin, Goyal, Küttler, Lewis, Yih, Rocktäschel, Riedel & Kiela (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. It coined the term and, crucially, made retrieval **part of the model**: a DPR bi-encoder retrieves passages from a dense index over Wikipedia, and a BART seq2seq generator conditions on them, with the retriever's query encoder trained jointly by backpropagating the generation loss through the retrieval distribution. The document encoder and the index were frozen, because re-embedding 21M passages on every gradient step is not a thing you do.

Two variants are worth naming because interviewers who read the paper ask: **RAG-Sequence** marginalizes over documents at the sequence level — pick a document, generate the whole answer conditioned on it, then average over the top-k with the retriever's probabilities as weights. **RAG-Token** marginalizes per token, so different tokens in one answer can be sourced from different documents. RAG-Token is the more expressive one and the intuition worth carrying: it lets a single sentence fuse facts from two passages, which is exactly the multi-hop capability that naive prompt-stuffing gets accidentally and unreliably.

What we ship in 2026 is different in every respect except the shape of the idea. The generator is a frozen API model we cannot backprop through. The retriever is not trained on our generation loss; it is an off-the-shelf embedding model plus BM25 plus a cross-encoder. The "marginalization" is "concatenate the top-5 chunks into the prompt and hope attention sorts it out." **We traded end-to-end learned retrieval for zero training cost and a frontier-quality generator, and that trade was obviously correct** — but it means the retriever has no idea what the generator finds useful, which is the root cause of an entire failure class.

**🗣 Say this in the room:** "Lewis 2020 proposed retrieval as a differentiable component with the query encoder trained through the generation loss. What the industry calls RAG now is in-context retrieval against a frozen generator — no joint training, all the coupling lives in the prompt. The paper's real legacy is the framing of parametric versus non-parametric memory, not the architecture, which almost nobody runs."

### Explain Fusion-in-Decoder, and tell me why it still matters when nobody trains one.

FiD matters because it names the constraint that governs every context-assembly decision you will ever make: **how many passages you can attend over is bounded by attention cost, and FiD is the trick that made that cost linear instead of quadratic in the number of passages.**

**📄 Paper:** Izacard & Grave (2021) — *Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*. The mechanism: instead of concatenating the question and 100 passages into one long encoder input (quadratic in total length, because self-attention is), you encode each `(question, passage_i)` pair **independently** through the encoder. Each yields a sequence of hidden states. You then concatenate those encoder outputs — not the raw tokens — and let the decoder cross-attend over the union. The encoder cost is `n_passages × O(L²)` with `L` the per-passage length, which is linear in passage count; the fusion happens only in cross-attention, which is cheap.

The consequence they reported and that everyone remembers: performance kept improving as they scaled to 100 passages, whereas an extractive reader saturates almost immediately. Evidence for a claim I want you to internalize — **recall matters more than precision at the retrieval stage if your reader can actually handle the volume.**

Why it still matters with frozen API models: it explains why "just stuff 100 chunks in" performs worse than the FiD numbers suggest it should. When you concatenate 100 chunks into one prompt, every chunk's tokens attend to every other chunk's tokens. Chunk 47 is being contaminated by chunk 12's surface form. FiD structurally forbids that — passages never see each other during encoding, only during decoding. You get cross-passage interference for free in the modern pattern, and it is a real quality cost that people attribute to "the model got confused" without a mechanism.

**⚠ Trap:** candidates describe FiD as "an efficiency trick." It is also an *isolation* trick, and the isolation is half the benefit. The closest thing you can do with an API model is map-reduce: summarize each chunk independently against the question, then synthesize. That is FiD in spirit — independent encoding, late fusion — and it is why map-reduce beats stuffing on 50-document synthesis tasks despite costing more calls.

### What were REALM and Atlas doing that in-context RAG doesn't, and does it matter for my job?

These are the "retrieval belongs in pretraining" line, and they matter for exactly one interview purpose: proving you know that the industry's choice to bolt retrieval on at inference time was a pragmatic compromise, not a discovery that it is optimal.

**📄 Paper:** Guu, Lee, Tung, Pasupat & Chang (2020) — *REALM: Retrieval-Augmented Language Model Pre-Training*. It put a latent retriever inside masked-language-model pretraining: to predict a masked token, the model retrieves documents and the retrieval is trained by the signal "did retrieving this document help me fill the blank?" No relevance labels anywhere — the LM objective supervises the retriever. The engineering problem this creates is the interesting part: the document index is built from the very encoder you are training, so it goes stale after every update. Their answer was asynchronous index refresh — a background job re-embeds the corpus every few hundred steps while training continues against a slightly stale index. **That is exactly the stale-index problem you already know from search infrastructure, promoted into the training loop.**

**📄 Paper:** Izacard, Lewis, Lomeli, Hosseini, Petroni et al. (2022) — *Atlas: Few-shot Learning with Retrieval Augmented Language Models*. It jointly trained a retriever with an FiD reader and showed that an 11B-parameter retrieval-augmented model could beat a far larger purely-parametric model on knowledge-heavy QA with only tens of training examples. The durable takeaway: **parameters spent memorizing facts are parameters wasted; you can be an order of magnitude smaller if you can look things up.**

Does it matter for your job? Directly, no — you will not pretrain a retriever. Indirectly, twice. First, it is the honest citation when someone claims "fine-tuning will teach the model our docs": the literature's own answer to knowledge injection was retrieval, from the start, by the people best positioned to do it with weights. Second, it predicts where the field goes when open-weight models get cheap enough to post-train: retrieval-aware training, where the model learns *when* to retrieve rather than being told by a router. Self-RAG is the first commercially-legible step down that path.

### Draw me the tiers of RAG architecture and tell me how you decide which tier a project needs.

Three tiers, and the decision rule is the whole answer — the tiers themselves are common knowledge.

**Tier 1, naive.** Embed the query, ANN search, take top-k, concatenate into a prompt, generate. Two moving parts. This is the correct starting architecture for essentially every project, and it is what I build in the first day.

**Tier 2, advanced.** Deterministic pipeline with more stages: query rewriting for multi-turn, hybrid lexical+dense retrieval fused with RRF, cross-encoder reranking to cut 100 candidates to 8, metadata filtering, dedup, and a deliberate assembly order. Still a straight line — no branching, no model deciding what happens next. **This tier captures the overwhelming majority of the achievable quality gain**, and it is where 90% of shipped production RAG should live.

**Tier 3, modular / agentic.** Control flow decided at runtime by a model: routing across multiple indexes and tools, iterative retrieve-reason loops, self-correction, retrieval evaluators with fallback branches. The system may issue a variable number of retrievals per query.

The decision rule I actually enforce, in order:

1. **Build tier 1 and measure the retrieval ceiling first.** Recall@50 on a golden set of 100 real queries. If recall@50 is 0.95 and your answers are bad, retrieval is not your problem — go fix generation or chunking. Adding agentic retrieval to a pipeline with a 0.95 ceiling adds latency and buys nothing.
2. **Go to tier 2 when a per-slice ablation shows a specific stage earns its latency.** Not "reranking helps in general" — *your* reranker moved nDCG@10 by +0.08 on *your* golden set at +180 ms p95.
3. **Go to tier 3 only when the query distribution is structurally heterogeneous or multi-hop.** The test is: can a single retrieval, however well-formed, contain all the evidence needed? If yes for 95% of your traffic, tier 3 is architecture theater. If your users routinely ask "compare our Q3 policy to the Q1 one and tell me what changed," one retrieval structurally cannot do it, and you need a loop.

**🗣 Say this in the room:** "I'd default to hybrid retrieval plus a reranker — tier two — and I'd need evidence before going further. The upgrade trigger for agentic RAG isn't 'the answers are bad', it's 'a measurable share of my queries need evidence that no single retrieval can return.' Otherwise I've bought a 4× latency multiplier and a new class of nondeterministic failures to debug."

### Build me naive RAG from scratch. No LangChain, no vector database — just show me you understand the whole loop.

Here is the entire thing. The point of writing it unaided is that it makes the framework layer legible later: every abstraction you meet in LangChain maps onto one of these fifteen lines.

```python
import numpy as np, httpx
from typing import Iterable

EMB_URL = "https://api.example.com/v1/embeddings"   # any provider
DIM = 1024

def embed(texts: list[str], model: str = "embed-v3") -> np.ndarray:
    r = httpx.post(EMB_URL, json={"input": texts, "model": model}, timeout=60)
    r.raise_for_status()
    v = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)   # unit-norm => dot == cosine

def chunk(doc: str, size: int = 1600, overlap: int = 200) -> Iterable[str]:
    step = size - overlap
    for i in range(0, max(len(doc) - overlap, 1), step):
        yield doc[i : i + size]

class TinyRAG:
    def __init__(self):
        self.texts: list[str] = []
        self.meta: list[dict] = []
        self.M = np.zeros((0, DIM), dtype=np.float32)

    def index(self, docs: list[dict], batch: int = 96):
        chunks = [(c, {"doc_id": d["id"], "pos": i})
                  for d in docs for i, c in enumerate(chunk(d["text"]))]
        for i in range(0, len(chunks), batch):          # batch: provider rate limits
            part = chunks[i : i + batch]
            self.M = np.vstack([self.M, embed([c for c, _ in part])])
            self.texts += [c for c, _ in part]
            self.meta += [m for _, m in part]

    def search(self, query: str, k: int = 5) -> list[tuple[float, str, dict]]:
        q = embed([query])[0]
        scores = self.M @ q                              # (N,) cosine, one BLAS call
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self.texts[i], self.meta[i]) for i in top]

PROMPT = """Answer using only the sources below. Cite as [S1], [S2].
If the sources do not contain the answer, say "I don't have that documented."

{sources}

Question: {q}"""

def answer(rag: TinyRAG, q: str, llm) -> str:
    hits = rag.search(q, k=5)
    src = "\n\n".join(f"[S{i+1}] ({h[2]['doc_id']}) {h[1]}" for i, h in enumerate(hits))
    return llm(PROMPT.format(sources=src, q=q))
```

That is a complete, functioning RAG system in under 50 lines, and on a 200-document corpus it will beat a badly-configured LangChain pipeline. Brute-force `M @ q` over 50,000 chunks × 1024 dims is 51M multiply-adds — roughly 5 ms in NumPy on one core. **You do not need an ANN index below ~100k chunks and you should say so**; HNSW's value is sublinear scaling, and at 50k vectors linear scan is already inside your latency budget.

**⚠ Trap:** `np.argpartition` with `k >= len(scores)` raises, and unit-normalizing a zero vector produces NaNs that silently propagate to every score. Both are the kind of thing that passes review and blows up on the empty-document edge case in production. Guard them.

**🏋 Drill:** 20 minutes, no autocomplete, no reference. Reproduce the above from memory including batching and unit-normalization. *Pass criterion:* `search` is a single matrix-vector product with a partial sort, not a Python loop over documents, and the prompt contains an explicit abstention instruction and citation markers.

### Now show me the same thing in LangChain or LlamaIndex — and tell me when the framework actually earns its keep.

The equivalent is about eight lines, and I want to be precise about what those eight lines bought you.

```python
# LlamaIndex, roughly — verify exact import paths against your installed version
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter

Settings.chunk_size, Settings.chunk_overlap = 512, 64
docs  = SimpleDirectoryReader("./corpus").load_data()
index = VectorStoreIndex.from_documents(docs, transformations=[SentenceSplitter()])
qe    = index.as_query_engine(similarity_top_k=5, response_mode="compact")
print(qe.query("What is the escalation SLA for a P1?"))
```

**📅 Volatile:** import paths and class names in this ecosystem churn every few months — LlamaIndex's `llama_index.core` split and LangChain's LCEL/`langchain-core` reorganization both broke every tutorial in existence. Verify against the installed version before your loop; do not memorize signatures.

What the framework actually gives you, honestly: **connectors and parsers** (Confluence, Notion, Drive, S3, ~150 loaders you would otherwise write), **vector-store adapters** so swapping pgvector for Qdrant is a config change, **retriever composition** (hybrid, auto-merging, recursive) already wired, and **callback/tracing hooks** that plug into LangSmith or Phoenix. Those are real. The connector library alone can be worth a week.

What it costs you: control over the exact prompt string, which is the single highest-leverage variable in the whole system; a debugging experience where a bad answer requires reading framework source to discover that `response_mode="compact"` silently re-packs your chunks and dropped one; and a dependency graph large enough that `pip install langchain` has been known to pull in dozens of transitive packages.

**My rule in review:** use the framework for **ingestion and connectors**, write the **query path by hand**. Ingestion is heterogeneous, boring, and where the loaders live. The query path is 60 lines, is where all your quality lives, and is where you will be reading and re-reading the exact prompt at 2am. Haystack 2.x deserves a mention here because its explicit component-and-connection pipeline graph is more inspectable than LangChain's chains — if your org insists on a framework end-to-end, it is the one I would argue for.

**🗣 Say this in the room:** "I use LlamaIndex or Haystack for ingestion and connectors, and hand-write the retrieval and prompt-assembly path. The framework's abstraction over the query path hides the exact prompt, and the exact prompt is the thing I need to iterate on hourly. I'm not anti-framework — I'm anti-indirection between me and the string I'm sending to the model."

### A teammate proposes an agentic RAG system — a planner, a retriever tool, a critic, and a re-retrieval loop — over the company's 200-document internal FAQ. What do you say?

I say no, and I say it in a way that does not make the teammate look stupid, because the proposal is fashionable rather than foolish.

**⚠ Trap — and this is the named red flag for this section:** agentic RAG over a small, homogeneous corpus is the clearest signal in an interview that a candidate is pattern-matching on architecture blog posts rather than reasoning from constraints. Interviewers at Sierra, Glean and Harvey specifically probe for it, because they have all rejected candidates who reached for it.

The argument, with arithmetic. 200 documents at maybe 8 chunks each is **1,600 chunks**. Brute-force cosine over 1,600 × 1024 floats is 1.6M FLOPs — sub-millisecond. There is no scaling problem to solve. The realistic quality ceiling is set almost entirely by whether the answer is written down anywhere, and recall@20 over 1,600 chunks with hybrid retrieval will be somewhere near 0.95 — **you can put nearly the entire top-20 in the context window and be done.**

**💰 Math:** the agentic version costs a planner call, an average of 2.5 retrieval-tool round trips, a critic call, and a final synthesis — call it 5 model calls at ~1,500 input / 300 output tokens each. At $3/Mtok in and $15/Mtok out (📅 verify): `5 × (1500 × 3e-6 + 300 × 15e-6) = 5 × (0.0045 + 0.0045) = $0.045` per query, at roughly `5 × 900 ms = 4.5 s` of serialized latency. The tier-1 version is one call at 6,000 input / 300 output: `6000 × 3e-6 + 300 × 15e-6 = 0.018 + 0.0045 = $0.0225`, TTFT ~700 ms. So agentic is **2× the cost and ~6× the latency**, and on a 1,600-chunk corpus it is buying resolution of a retrieval problem that does not exist. At 50,000 queries/month that is `50,000 × $0.0225 = $1,125/month` of pure waste, plus a nondeterministic control flow that makes every quality regression a distributed-systems debugging exercise.

**🗣 Say this in the room — this is the diplomatic version, memorize it:** "I like the design for a corpus where a single retrieval genuinely can't contain the answer. For 200 FAQ documents I'd want to see the failure first — my prior is that hybrid retrieval plus a reranker gets recall@20 above 0.95 here, and at that point the loop is adding four hundred milliseconds and a nondeterministic control path to fix something that isn't broken. Can we build the simple one, measure the ceiling on fifty real queries, and let the number decide? If recall is already 0.95 the agent can't help, and if it's 0.6 we'll know exactly which failure mode to design against."

That framing wins because it does not assert; it proposes a cheap experiment whose outcome you are confident about. **The senior move is not being right — it is designing the two-hour experiment that makes being right unnecessary to argue about.**

### It's day one on a new RAG project. What do you build, and what are your explicit upgrade triggers?

Day one I build the boring version and, more importantly, **the measurement harness at the same time**, because every subsequent decision is unarguable without it.

The day-one system: structure-aware chunking at ~512 tokens with a heading path in the metadata; one good embedding model; Postgres with pgvector; BM25 in the same Postgres via `tsvector` or a small Elasticsearch if you already run one; RRF fusion at `k=60`; top-8 into a single prompt with citation markers and an abstention instruction. That is tier 2 minus the reranker, and it is achievable in two days.

Alongside it, non-negotiable: **50 to 100 real queries with labeled relevant chunks**, mined from support tickets, search logs, or by asking three internal experts for the questions they actually get. Plus per-request tracing that records the query, rewritten query, retrieved chunk IDs with scores, assembled prompt token count, and the final answer. Without those traces every future debugging session is archaeology.

Then the upgrade triggers, each stated as a measurement rather than a vibe:

| Add this | When this is true | Expected cost |
|---|---|---|
| Cross-encoder reranker | recall@50 ≫ recall@8 (retrieval finds it, ranking buries it) | +150–300 ms p95, ~$2/1k queries 📅 |
| Conversational query rewriting | >20% of traffic is multi-turn and follow-up recall drops vs first-turn | +250 ms, +$0.001/query |
| Multi-query + RRF | single-query recall@50 < 0.85 and failures are vocabulary-mismatch | +300 ms, +3 embed calls |
| Metadata filter extraction | queries contain constraints ("in the 2024 policy", "for EU customers") that retrieval ignores | +200 ms |
| Iterative retrieval loop | a labeled slice of queries is provably multi-hop | +2–4× latency and cost |
| Contextual chunk augmentation | chunks are ambiguous out of context (pronouns, "the above table") | one-time reindex cost |
| GraphRAG | whole-corpus sensemaking questions dominate | 10–100× indexing cost |

**⚠ Trap:** the trigger for a reranker is *not* "answers are bad." It is specifically **recall@50 much greater than recall@8** — the evidence is being retrieved and then buried at rank 30. If recall@50 is itself 0.6, a reranker can do nothing; you have a retrieval or a corpus problem, and reranking a candidate set that lacks the answer just reorders wrong things. I have seen a team spend a month on reranker selection while recall@100 sat at 0.55 because half the corpus never parsed.

### Walk me through the end-to-end latency budget of a full "advanced RAG" pipeline. Where does the time actually go?

Here is the budget I carry in my head, for a p95 target of 2.0 s to first token. Numbers are order-of-magnitude and you should measure your own, but the *shape* is stable and being able to draw it is the point.

| Stage | p50 | p95 | Notes |
|---|---|---|---|
| Query classification / routing | 120 ms | 300 ms | small model, ~40 output tokens |
| Conversational rewrite | 200 ms | 450 ms | only on follow-up turns |
| Embedding the query | 25 ms | 60 ms | one 1024-dim call; network dominates |
| ANN search (HNSW, 5M vectors) | 8 ms | 25 ms | genuinely cheap |
| BM25 search | 15 ms | 50 ms | |
| RRF fusion | <1 ms | <1 ms | pure Python on 100 items |
| Cross-encoder rerank, 100→8 | 90 ms | 220 ms | batched on one GPU, 256-token truncation |
| Assembly, dedup, token accounting | 5 ms | 15 ms | |
| **Prefill of ~6k tokens** | **300 ms** | **700 ms** | provider-dependent, cache-dependent |
| **Total TTFT** | **~760 ms** | **~1.8 s** | |

Three things jump out and each is a lesson. **First, retrieval is not the bottleneck.** ANN plus BM25 plus fusion is under 80 ms at p95. Engineers coming from search infrastructure over-optimize this and it is the smallest bar on the chart. **Second, every LLM call you add costs 200–450 ms of p95 — more than the entire retrieval stack.** That is why "add a query rewriter, add a critic, add a compressor" is not free architecture; each one is a quarter-second and each one can fail. **Third, prefill dominates**, and prefill is the one you control by controlling assembled token count, not by making retrieval faster.

**📐 Numbers you must know:** as a planning constant, budget **~200–400 ms of p95 for any additional LLM call** in the pre-retrieval path even with a small fast model, because you pay network round trip plus queueing plus prefill plus decode of a short output. And budget prefill at roughly **50–100 ms per 1,000 input tokens** on a hosted frontier model without cache hits — so a 12k-token context costs 600 ms–1.2 s of TTFT before a single output token appears. Both are worth verifying against your provider, but they are the right order for reasoning on a whiteboard.

**The design consequence:** LLM calls in the query path must be either (a) parallelized, (b) cached, or (c) conditional. Rewriting and metadata extraction can share one call. Multi-query variants can be embedded concurrently. Routing can be skipped when the classifier is confident from a cheap heuristic. **Serialized LLM calls in the pre-retrieval path are where RAG latency budgets die**, and the fix is almost always restructuring the call graph rather than picking a faster model.

### Give me a question that a single retrieval pass structurally cannot answer, and explain why.

"Which of our enterprise customers signed after our SOC 2 certification date?"

Walk through why this is impossible for one retrieval, because the reasoning generalizes. To retrieve the relevant customer contracts you would need to filter on signing date > X. But X — the SOC 2 certification date — is itself a fact stored somewhere in the corpus that you do not know at query time. **The retrieval query depends on the result of a prior retrieval.** No amount of query rewriting fixes this, because the rewriter also does not know the date. No amount of embedding quality fixes it, because the information required to formulate the query is not in the query.

That is the general form: **a question is structurally multi-hop when the identifier needed to retrieve the second piece of evidence appears only in the first piece of evidence.** Three families:

**Bridge questions.** "Who is the manager of the engineer who wrote the payments retry logic?" Hop 1: find the author. Hop 2: find that person's manager. The name is the bridge and it does not appear in the query.

**Comparison and aggregation.** "How does our current refund policy differ from the 2023 one?" Two independent retrievals are needed and then a comparison; a single retrieval returns chunks from both documents at best, with no guarantee it returns *matching sections* of both. Aggregations are worse — "how many incidents last quarter were caused by config changes?" requires touching every incident, and top-k retrieval is definitionally a sample, not a scan.

**Whole-corpus sensemaking.** "What are the recurring themes in customer complaints this quarter?" There is no top-k that answers this. The answer is a property of the whole set. This is the honest case for GraphRAG-style community summaries or a precomputed rollup — retrieval is the wrong primitive entirely.

**⚠ Trap:** the failure is silent and looks like a hallucination. The retriever returns chunks that are topically relevant — SOC 2 documents, contract documents — the LLM sees plausible material and produces a confident, wrong, well-cited answer. It cites real chunks. It just did the wrong operation over them. **This is the most dangerous RAG failure mode because every surface-level quality check passes:** citations resolve, retrieval scores are healthy, the answer is fluent. Only a domain expert catches it.

The diagnostic I use to size the problem: take 200 real queries and label each as single-hop, multi-hop, or aggregate. If multi-hop is 3% of traffic, route those to an iterative agent and leave the other 97% on the fast path. If it is 40%, you have a different product and should consider whether text retrieval is even the right substrate — that shape often wants text-to-SQL or a graph.

### Build me the cost model for a RAG request. I want to see every line item.

I want this on a napkin in an interview, so here is the structure. Assume a support assistant, 8k assembled context tokens, 400 output tokens, and hybrid retrieval with a reranker.

**Per-request generation cost.** At $3/Mtok input and $15/Mtok output (📅 verify against §5 before your loop): input `8,000 × 3e-6 = $0.024`, output `400 × 15e-6 = $0.006`. Generation subtotal **$0.030**.

**Query-side embedding.** One query embed at ~30 tokens: at $0.02/Mtok that is `30 × 2e-8 = $0.0000006`. Effectively free — **query embedding cost never appears in a real cost model and you should say so rather than including it for completeness.**

**Reranking.** Hosted rerankers are typically priced per search unit where one unit covers a query plus up to 100 documents; at roughly $2 per 1,000 searches (📅 volatile, and self-hosting a 110M cross-encoder changes this entirely), that is **$0.002/query**. Self-hosted on an L4 at ~$0.80/hr doing ~20 queries/s of 100-candidate reranking: `$0.80 / (20 × 3600) = $0.000011/query` — **180× cheaper**, which is the actual argument for self-hosting a reranker once you exceed a few million queries a month.

**Query transformation, if enabled.** A rewrite call at 800 input / 60 output on a cheap model at, say, $0.25/$1.25 per Mtok: `800 × 2.5e-7 + 60 × 1.25e-6 = 0.0002 + 0.000075 = $0.000275`. Cheap in dollars — **the cost of query transformation is latency, not money**, and conflating those two is a common analytical error.

**Amortized ingestion.** 2M chunks × 400 tokens = 800M tokens at $0.02/Mtok = `$16` per full reindex. If you reindex monthly and serve 1M queries/month, that is `$16/1e6 = $0.000016` per query. **Ingestion is a rounding error at query scale and a real number at reindex time** — the failure mode is the surprise $16 bill being fine but the 11-hour wall clock not being.

**Total: ~$0.032/query, of which 94% is generation input tokens.** That single fact determines every optimization priority: **cutting assembled context from 8k to 5k tokens saves `3,000 × 3e-6 = $0.009/query = 28% of total cost`**, while making retrieval twice as fast saves nothing. At 1M queries/month: `1e6 × 0.032 = $32,000/month`, and the context reduction is `1e6 × 0.009 = $9,000/month`. Prefix caching on a stable 2k-token system prompt at a 90% cache discount saves `2,000 × 3e-6 × 0.9 = $0.0054/query = $5,400/month` on top.

**🗣 Say this in the room:** "In almost every RAG system I've costed, 85–95% of the bill is input tokens on the generation call. So the cost levers, in order, are: reduce assembled context, get prefix caching to hit on the stable prefix, and route easy queries to a cheaper model. Making the vector search faster is a latency lever, not a cost lever, and I'd want to see the trace breakdown before anyone optimizes it."

### When people say "RAG is dead, just use long context," what's the actual technical claim and where does it fall down?

The claim deserves to be taken seriously rather than dismissed, because it is directionally right about a narrow case and wrong about the general one, and the difference is where the interview lives.

The steelman: context windows went from 4k to hundreds of thousands of tokens, per-token prices fell hard, prompt caching makes repeated prefixes cheap, and long-context models genuinely score near-perfectly on needle-in-a-haystack. If your entire corpus fits, retrieval is a lossy filter you inserted between the user and the evidence — and every retrieval miss is an answer the model *could* have gotten right. **Retrieval can only lose information relative to full context; it never adds any.** That is a real argument and I want you to state it before you rebut it.

Where it falls down, in order of how often it actually bites:

**Corpus size.** An enterprise wiki is 2–20M tokens. A codebase is 50M+. A legal discovery set is billions. "It fits" is true for a product spec and false for every corpus anyone builds a RAG system over. This is usually the end of the discussion and people skip it because it feels too obvious to say.

**Prefill economics.** 200k tokens of context at $3/Mtok is `200,000 × 3e-6 = $0.60` per call before you generate a single token. RAG's 8k context is `$0.024`. That is a **25× cost difference**; at 100k queries/day it is `100,000 × 0.576 = $57,600/day` of difference. Caching helps enormously if the context is *identical* across calls — and it is exactly zero help if each user has a different document set, which in any multi-tenant product they do.

**TTFT.** Prefilling 200k tokens takes seconds even on a good stack. Interactive products cannot spend that per turn.

**Effective vs advertised context.** Retrieval accuracy over long context degrades well before the advertised limit, and it degrades non-uniformly by position. That is the lost-in-the-middle and context-rot territory — a separate question below.

**Auditability.** Retrieval gives you a bounded, inspectable evidence set with document IDs. Full-context gives you "the answer is somewhere in the 200k tokens." For Harvey, for a bank, for anything regulated, **the citation is the product** and retrieval is the only architecture that makes span-level attribution cheap to verify.

**🗣 Say this in the room:** "Long context doesn't replace retrieval, it changes what retrieval has to be good at. When the window was 4k I needed precision — the right three chunks. With a 200k window I need recall — get the evidence in there somewhere and let attention do the ranking. So the architecture survives and the tuning target flips. And I'd point out that the cost and TTFT of a 200k prefill make it a batch-mode answer, not an interactive one, unless the context is identical across users and cacheable."
