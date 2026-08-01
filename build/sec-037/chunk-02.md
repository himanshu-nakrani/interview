### Our chatbot works perfectly on the first question and falls apart on the second. Debug it.

I would bet money on the diagnosis before looking at anything: **you are embedding the raw user turn, and the raw user turn in a multi-turn conversation is not a self-contained question.** This is the single most common production RAG bug, and it is invisible in every demo because demos are single-turn.

The mechanism is brutally simple once stated. Turn 1: "What's the refund policy for annual plans?" — embeds to a vector near refund-policy chunks, retrieval works, everyone is happy. Turn 2: "What about monthly?" That string embeds to a vector near… nothing useful. "What about monthly" has no lexical or semantic overlap with refund policy documents. Its nearest neighbours in embedding space are chunks about monthly billing cycles, monthly reports, monthly active users. **The retriever has no access to the conversation. It sees three words.** The LLM then receives irrelevant chunks plus a conversation history in which turn 1 was about refunds, and it does one of two things: hallucinates from parametric memory, or answers about monthly billing. Both look like "the model got confused."

Confirm it in two minutes with the trace: log the embedded query string alongside the retrieved chunk IDs. If the embedded string is the raw turn, you have found it. Then the quantitative version — split your eval set by turn index and compute recall@k for turn 1 versus turns 2+. **A 25–40 point recall gap between first turn and follow-up turns is the standard signature**, and having that plot ready is what separates "I think it's query rewriting" from "here is the number."

The fix is query rewriting, sometimes called conversational query reformulation or contextualization: before retrieval, call a small model with the last few turns and the current turn, and ask it to emit a standalone question. "What about monthly?" becomes "What is the refund policy for monthly plans?" which embeds correctly.

**⚠ Trap:** teams "fix" this by concatenating the whole conversation history into the embedding input. It is worse, not better, and predictably so. Embedding models are trained on short passages; feed them 900 tokens of dialogue and the pooled vector becomes a blurry centroid of every topic discussed, which retrieves the *average* of the conversation rather than the current question. It also gets monotonically worse as the conversation grows, which is a nasty property — quality degrades with engagement.

**🗣 Say this in the room:** "First thing I check is what string we actually embed. In multi-turn, the raw user turn is usually not self-contained — 'what about monthly?' has no retrievable signal. The fix is a rewrite step that resolves pronouns and elisions against the last few turns into a standalone query, and the way I'd prove it is splitting recall@k by turn index; a big first-turn-versus-follow-up gap is diagnostic."

### Write me the query rewriter, and tell me when you'd skip the call.

The prompt is more important than the code and both are short.

```python
REWRITE = """Rewrite the user's latest message into a standalone search query.
Resolve all pronouns and references using the conversation. Keep the user's own
terminology and any identifiers, codes or product names exactly as written.
If the latest message is already standalone, return it unchanged.
Return only the query, no explanation.

Conversation:
{history}

Latest message: {q}
Standalone query:"""

FOLLOWUP_HINTS = ("it", "that", "this", "they", "them", "those", "he", "she",
                  "the same", "what about", "and ", "why", "how about", "instead")

def needs_rewrite(q: str, history: list[str]) -> bool:
    if not history:
        return False
    ql = q.lower().strip()
    if len(ql.split()) <= 4:                 # terse turns are almost always elliptical
        return True
    return any(h in ql for h in FOLLOWUP_HINTS)

async def rewritten_query(q, history, small_llm, cache):
    if not needs_rewrite(q, history):
        return q
    key = sha1(("|".join(history[-4:]) + "||" + q).encode()).hexdigest()
    if (hit := await cache.get(key)):
        return hit
    out = (await small_llm(REWRITE.format(history="\n".join(history[-4:]), q=q))).strip()
    if not (3 <= len(out) <= 400):           # guard against a chatty model
        return q
    await cache.set(key, out, ex=86400)
    return out
```

Three design decisions worth defending. **Last 4 turns, not the whole history** — pronoun antecedents are almost always recent, and a longer window makes the rewriter drift toward summarizing the conversation rather than resolving the question. **"Keep identifiers exactly as written"** because rewriters love to normalize `ERR_TLS_CERT_ALTNAME_INVALID` into "the TLS certificate name error," which destroys your BM25 channel; this one clause has saved me more retrieval quality than any embedding upgrade. **A guard on the output**, because when the rewriter fails it fails by producing prose, and shipping prose to the retriever is worse than shipping the original turn — always fall back to the raw query rather than to a suspicious rewrite.

When to skip: first turn ever (no context to resolve), and any turn that passes the `needs_rewrite` heuristic as already-standalone. In production traffic I typically see **50–65% of turns skip the rewrite**, which halves the added latency at the aggregate level even though the p95 for follow-up turns still carries it.

**💰 Math:** at 1M queries/month, 40% needing rewrite, 800 in / 60 out on a small model at $0.25/$1.25 per Mtok: `400,000 × (800×2.5e-7 + 60×1.25e-6) = 400,000 × 0.000275 = $110/month`. That is nothing. The real cost is `400,000 × 250 ms` of added latency landing on exactly the turns where users are already impatient. **Judge query transformation on the latency budget, not the invoice.**

### How do you cache query rewrites, and what's the cache key?

The key is the interesting part, because the obvious choice is wrong.

The naive key is `hash(query)`. That is broken: "what about monthly?" rewrites to something completely different depending on the preceding conversation, so you would serve one user's context to another. The correct key is **`hash(last_n_turns + current_turn)`** — the rewrite is a pure function of exactly that input, so that is exactly what the key must cover. In the code above I use the last 4 turns joined, which is also what I feed the model; **the cache key must be derived from the model's actual input, not from a convenient subset.** If those two ever diverge you get a subtle correctness bug that looks like intermittent bad retrieval.

Hit rates in practice are modest and you should not oversell them. Conversation prefixes are nearly unique per user, so exact-match hit rates on rewrite caches run **low single digits to maybe 15%** — mostly from retries, refreshes, and the "user rephrased and resubmitted" pattern. The place caching actually pays is **first-turn queries**, which have no history and therefore key on the query alone; head-of-distribution first turns ("how do I reset my password") repeat constantly, and there a 40–60% hit rate is realistic.

**⚠ Trap:** semantic caching on query rewrites — embedding the query and serving the rewrite of a "close enough" neighbour. Do not. The whole point of the rewrite is precision about the current turn, and cosine 0.94 between "cancel my monthly plan" and "cancel my annual plan" is entirely achievable. **A semantic cache that returns a semantically-close but factually-wrong answer is the single most dangerous cache in an LLM system**, because the miss is invisible: no error, no latency spike, just a wrong answer served fast. If you use semantic caching at all, use it for full answers on a curated head-of-distribution set with a high threshold and human-reviewed entries, never for query understanding.

Where the caching money actually is: **cache the retrieval results keyed on the rewritten query**, not the rewrite itself. Retrieval is deterministic given the query and the index version, so key on `(index_version, rewritten_query, filters)`. That skips embedding, ANN, BM25 and reranking in one hop — 200–350 ms and the reranker's per-query cost — and it invalidates cleanly on index alias swap because the version is in the key.

### What is HyDE, and when does it make retrieval worse?

**Mental model: HyDE fixes an asymmetry.** Your query is a short question; your corpus is long declarative passages. Embedding models are trained mostly on symmetric or near-symmetric similarity, so a question vector and an answer-passage vector can sit surprisingly far apart even when the passage answers the question perfectly. HyDE closes the gap by making the query look like the corpus: **ask an LLM to hallucinate an answer, then embed the hallucination instead of the question.**

**📄 Paper:** Gao, Ma, Lin & Callan (2022) — *Precise Zero-Shot Dense Retrieval without Relevance Labels*. The contribution is that a *factually wrong* generated document still retrieves well, because retrieval only needs the right region of embedding space, not the right facts. It replaced the assumption that zero-shot dense retrieval needed labeled relevance data to work out-of-domain.

Mechanism, concretely: query → small LLM → a 100-word plausible passage → embed that → ANN search. Often you embed both the hypothetical document and the original query and average the vectors, or run both and fuse with RRF, which is more robust than replacing the query outright.

When it hurts, and this is the part interviewers want:

**When the query contains rare identifiers.** The generated document will paraphrase them away. `ERR_TLS_CERT_ALTNAME_INVALID` becomes "a certificate hostname mismatch error." You have just destroyed the exact-match signal that your BM25 channel depended on. **HyDE and lexical retrieval are actively in tension**, and if you run HyDE you must run BM25 on the *original* query, never the hypothetical.

**When the domain is out of the generator's knowledge.** For internal jargon — your product's feature names, your company's process names — the LLM generates a confident passage about something else entirely, and you retrieve that something else. HyDE's failure mode is not "no improvement," it is **retrieving coherently wrong material with high confidence**, which is worse than an obvious miss.

**When the query is already long and specific.** A 60-word detailed question is already in "passage" register; the asymmetry HyDE fixes does not exist, and the generation step just adds noise and 300 ms.

**When latency matters.** It is one more serialized LLM call before you can even embed. On a 1.5 s TTFT budget that is 20–30% of it.

My rule: **HyDE is a fallback, not a default.** Run normal hybrid retrieval; if the top reranker score is below your calibrated confidence threshold, *then* fire HyDE and re-retrieve. That gets the benefit on the queries that need it while paying the latency on maybe 15% of traffic. Modern embedding models with dedicated query/document instruction prefixes (`"query: "` / `"passage: "` style) also close much of the asymmetry HyDE was invented for — **the technique is less necessary in 2026 than in 2022, and saying so is a mark of currency.**

### Explain multi-query expansion, and implement the fusion step.

**Mental model: one query is one sample from the space of ways to ask the question, and retrieval is brittle to which sample you drew.** Multi-query expansion draws several samples and fuses the result sets, which converts a high-variance single shot into a lower-variance ensemble. It is bagging, applied to retrieval.

Mechanism: ask an LLM for N (typically 3–5) paraphrases that vary vocabulary and specificity — "How do I revoke an API key?" spawns "steps to delete an API token," "API key rotation procedure," "disable a compromised credential." Embed all N concurrently, run N retrievals, fuse the ranked lists with Reciprocal Rank Fusion.

```python
def rrf(ranked_lists: list[list[str]], k: int = 60, weights=None) -> list[tuple[str, float]]:
    """ranked_lists: each is doc_ids in descending relevance. Returns fused ranking."""
    w = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for wi, lst in zip(w, ranked_lists):
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + wi / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
```

**📄 Paper:** Cormack, Clarke & Buettcher (2009) — Reciprocal Rank Fusion. The reason it is the right tool here: it uses only ranks, never scores, so it needs no calibration across the N retrievals. Scores from five different query variants are *not* comparable — a variant that happens to be lexically close to many documents produces uniformly high cosines and would dominate a weighted-sum fusion for no good reason. `k = 60` damps the top ranks so that appearing at rank 3 in three lists (`3 × 1/63 = 0.0476`) beats appearing at rank 1 in one (`1/61 = 0.0164`). **That is the desired semantics: consensus across variants beats a single strong hit, because a single strong hit is exactly the high-variance event we are trying to average away.**

The cost, honestly: one LLM call (~250 ms) plus N embedding calls (concurrent, so ~40 ms) plus N ANN searches (concurrent, ~25 ms), and then a reranker candidate set that is up to N× larger, which is where the real latency lands. Budget **+300–400 ms p95 and roughly 2–3× the reranking cost**.

**⚠ Trap:** generating variants without constraining them. Left alone, a model produces variants that drift semantically — asked to paraphrase "how do I revoke an API key," it emits "what are API security best practices." That variant retrieves plausible, irrelevant material which then occupies slots in your fused top-k. Constrain the prompt: *"Rewrite the question in N different ways. Do not broaden or narrow the scope. Preserve every proper noun, identifier and number exactly."* And **always include the original query as one of the fused lists**, weighted equal or higher — it is the only variant guaranteed not to have drifted.

### What is step-back prompting and where does it belong in a RAG pipeline?

**Mental model: some questions are too specific to retrieve well, because the evidence you need is stated at a higher level of abstraction than the question.** Step-back prompting generates a deliberately more general question alongside the original, retrieves for both, and lets the generator use the general context to reason to the specific answer.

**📄 Paper:** Zheng et al. (2023, Google DeepMind) — *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*. The contribution is showing that eliciting a higher-level principle before answering improves reasoning on knowledge and STEM tasks; it is a prompting technique first and a retrieval technique second.

Concretely: "Can a user on the Starter plan invite external collaborators to a private project created before the March migration?" is a horrible retrieval query — it is so specific that no chunk matches it, because no document was written about exactly that intersection. The step-back version is "What are the collaborator permissions for the Starter plan?" That question *does* have a document. You retrieve both sets, and the model derives the specific answer from the general policy.

Where it belongs: **as a parallel branch, not a replacement.** Run the original query and the step-back query concurrently, fuse with RRF, weight the original higher. Cost is one small LLM call shared with your other query transformations plus one extra retrieval — under 100 ms if you batch it with the rewrite call, which you should.

When it hurts: when the answer really is stated specifically somewhere. Broadening a query that would have matched exactly now pulls in general policy documents that crowd out the specific one. The signature is a quality regression concentrated on your *easiest* queries, which is a horrible thing to ship — you broke the head of the distribution to fix the tail.

**⚠ Trap:** stacking step-back on top of HyDE on top of multi-query. Each of these transformations is individually defensible and they compose terribly: you now have 9 retrievals from queries that have each drifted a little, and the fused top-k is a soup of topically-adjacent material. **Each transformation is a decision to trade precision for recall, and they are not independent — running three of them is not three small trades, it is one large one.** Pick at most two, prove each with an ablation on a slice, and be able to name the queries each one is for.

### When do you decompose a question into sub-questions, and how do you keep it from making things worse?

Decomposition is right when the question **contains multiple independent information needs joined by a conjunction or a comparison** — not merely when it is long or complicated.

The clean case: "What's our data retention policy in the EU, and how does it differ from what we tell US customers?" Two retrievals, each of which has a good target document, then a comparison step. A single retrieval for the combined string lands in the middle of embedding space between two topics and reliably returns mediocre chunks for both — the classic centroid problem, where the average of two good vectors is close to neither.

The mechanism I ship:

```
1. classify: single | decomposable | multi-hop        (one small-model call)
2. if decomposable: emit 2-4 independent sub-questions, retrieve each in PARALLEL
3. assemble: group chunks under their sub-question, label them, dedup across groups
4. generate once, with the original question and the grouped evidence
```

The parallelism is the whole point and it is why decomposition is cheap while multi-hop is expensive. **Decomposable questions have independent sub-questions, so retrieval fans out concurrently: total latency is one classifier call plus max(retrievals), not sum.** Multi-hop questions have *dependent* sub-questions — hop 2's query needs hop 1's answer — and those serialize. Conflating decomposition with multi-hop is a common candidate error, and stating the difference crisply is a strong signal.

Keeping it from making things worse, three rules I enforce:

**Cap the fan-out at 4.** An unbounded decomposer will happily emit nine sub-questions for a two-part query, and now you have nine retrievals fighting for context budget.

**Allocate context budget per sub-question, not globally.** If you retrieve top-8 for each of four sub-questions and then take the global top-8 by score, you will often get all 8 from one sub-question because its scores happened to be higher, and silently drop an entire information need. This is a real and very common bug. **Reserve a floor — say 3 chunks minimum per sub-question — before filling the remainder by global score.**

**Label the groups in the prompt.** `## Evidence for: "EU retention policy"` followed by its chunks. Otherwise the model gets a flat pile and cannot tell which chunks address which half of the question, and the comparison degrades into a summary.

**⚠ Trap:** decomposing questions that only look compound. "What is the difference between a soft delete and a hard delete?" is a single information need with a conjunction in it — one document explains both, and decomposing it into two retrievals gets you two halves of the same page plus noise. The classifier must be trained or prompted on *your* query distribution with negative examples, or it will over-decompose. In review, I ask for the classifier's confusion matrix on 100 labeled real queries before it ships.

### Design the intent router for a production RAG assistant. Include the branch nobody remembers.

The branch nobody remembers is **"this is not a retrieval question at all,"** and forgetting it is why so many assistants answer "thanks!" by citing three documentation pages.

Here is the routing taxonomy I use for an internal-assistant product, which generalizes:

| Intent | Route | Why it matters |
|---|---|---|
| Factual lookup | hybrid retrieval → rerank → generate | the default, 60–70% of traffic |
| Multi-part / comparison | decompose → parallel retrieve | needs group-scoped budget |
| Multi-hop | iterative retrieve-reason loop | the only branch allowed to be slow |
| Aggregation / "how many" | text-to-SQL or a precomputed rollup | retrieval structurally cannot do this |
| Action / tool call | function calling, no retrieval | "reset my password" — do it, don't cite it |
| Chit-chat, thanks, greeting | direct response, **no retrieval** | the forgotten branch |
| Out of scope | abstain with a scoped explanation | protects against hallucination |
| Unsafe / policy | refusal path | |

The implementation should be a **single small-model call that returns structured output**, not a chain of classifiers, and it should be fused with your other pre-retrieval work:

```python
class QueryPlan(BaseModel):        # one call returns all of this
    intent: Literal["lookup","compare","multihop","aggregate","action","chitchat","oos"]
    standalone_query: str          # the conversational rewrite
    sub_questions: list[str] = []  # empty unless compare/multihop
    filters: dict[str, str] = {}   # {"product":"billing","year":"2024"}
    confidence: float
```

**One call, four jobs.** Routing, rewriting, decomposition and filter extraction are all "understand this query" tasks and a single structured-output call does them together for one round trip instead of four. **This is the highest-leverage structural change I make to naive advanced-RAG pipelines** — teams routinely build these as four sequential calls and pay 4 × 250 ms = 1 s of TTFT for work that fits in 300 ms.

The escape hatch matters too: when `confidence < 0.6`, route to the default lookup path rather than to anything exotic. **A router that is unsure should fall back to the boring branch, never to the expensive one** — the failure cost of an unnecessary retrieval is 200 ms, and the failure cost of an unnecessary agentic loop is four seconds and $0.05.

**🗣 Say this in the room:** "I'd do routing, conversational rewriting, sub-question decomposition and metadata-filter extraction in one structured-output call rather than four chained ones — same information, a quarter of the latency. And I'd make sure the taxonomy has a no-retrieval branch, because the most common embarrassing failure in these systems is citing documentation in response to 'thanks, that worked.'"

### Show me how you'd extract metadata filters from a natural-language query, and tell me how it goes wrong.

**Mental model: a filter is a hard constraint and retrieval similarity is a soft preference, so anything the user states as a constraint must be lifted out of the embedding and into the `WHERE` clause.** Cosine similarity cannot express "only 2024" — it can only express "somewhat about 2024," and it will happily return the 2022 policy at rank 2.

Mechanism: in the same structured-output call as routing, extract typed fields against a **closed schema you supply**, then apply them as pre-filters on the vector search.

```python
FILTER_SCHEMA = {
    "product":    ["billing", "auth", "search", "admin"],
    "doc_type":   ["policy", "runbook", "release_note", "faq"],
    "year":       "int (2019-2026) or null",
    "region":     ["us", "eu", "apac"],
}
# Prompt: "Extract only values explicitly stated or unambiguously implied.
#          Use exactly the allowed values. Omit any field you are not confident about."
```

Then: `filters` → your vector store's pre-filter (Qdrant payload filter, pgvector `WHERE` in the same query, etc.), with the crucial requirement that it be a **pre-filter or a filterable-HNSW traversal, not a post-filter** — post-filtering a selective constraint over top-k collapses recall, because the top-k was chosen before the filter was known.

How it goes wrong, in decreasing order of frequency:

**Over-extraction.** The user says "how does the new billing flow handle refunds" and the extractor emits `{"year": 2026}` because "new" felt temporal. Now the 2024 policy — which is the correct answer, unchanged since 2024 — is filtered out entirely. **A wrong filter is not a soft degradation; it is a hard zero.** The evidence is gone and no amount of reranking recovers it. This is why my prompt says "omit any field you are not confident about," and why I would rather under-extract by a wide margin.

**Vocabulary mismatch with the index.** The extractor returns `"authentication"`, the metadata field contains `"auth"`. Zero results. Fix: constrain to a closed enum in the prompt *and* validate the output against it server-side, dropping unknown values rather than passing them through.

**No results and no fallback.** If the filtered search returns fewer than `k` candidates, you must **retry without the filters and mark those results as unfiltered** rather than returning an empty set. The user asked a question; a filter you inferred should never be the reason they get nothing.

**⚠ Trap:** silently ANDing extracted filters. Three extracted fields, each 85% accurate, gives `0.85³ = 0.61` — a 39% chance that at least one filter is wrong and has excluded the answer. **Filter accuracy compounds multiplicatively and that arithmetic is the reason I cap extraction at the one or two fields with the highest measured precision** rather than shipping the full schema. Measure per-field precision on 100 labeled queries and only enable fields above ~0.95.

### Acronym expansion and spelling correction feel like 2010 search engineering. Do they still matter?

They matter more than most of the fashionable techniques on this list, and saying so is a differentiator, because everyone reaches for HyDE and nobody reaches for a synonym dictionary.

**The mechanism that makes them necessary is tokenization plus training distribution.** Internal acronyms — `PRD`, `TSR`, `CAB`, your company's three-letter project codes — are either out of distribution for the embedding model or, much worse, *in* distribution meaning something else entirely. `CAB` in your corpus means "change advisory board"; to a general embedding model it means a taxi. The vector lands in the wrong neighbourhood and there is no amount of reranking that recovers, because the candidate set never contained the right chunk.

The fix is unglamorous and deterministic: **a curated expansion dictionary applied at both index and query time.** Not an LLM call — a dictionary lookup with a few hundred entries, maintained by whoever owns the docs, applied as query augmentation (`"CAB approval"` → `"CAB (change advisory board) approval"`, expanding rather than replacing so the lexical channel keeps the literal). Sub-millisecond, zero cost, and in my experience worth more recall on internal corpora than upgrading the embedding model.

Spelling correction is the same shape. Dense retrieval degrades gracefully on typos — `"authentcation"` still lands near authentication — but **BM25 degrades catastrophically**, because a misspelled term simply is not in the inverted index and the highest-IDF term in the query contributes zero. In a hybrid system a typo therefore knocks out exactly the channel you added for precision. Fix with a corpus-derived correction: build a vocabulary from your own index, and for any query term absent from it with an edit distance of 1–2 from a high-frequency vocabulary term, add the correction as an OR-term rather than replacing.

**⚠ Trap:** using a general-purpose spellchecker. It "corrects" your product names, SKUs and error codes into English words — `Kubectl` → `Kubelet`, `EBITDA` → `EDITED`. **The dictionary must be derived from your corpus, not from English.** I have seen a general spellchecker silently destroy identifier retrieval and be blamed on the embedding model for six weeks.

**💰 Math:** these two cost approximately zero — a dictionary lookup and a symspell index in memory, maybe 2 ms combined — against the 250–400 ms of an LLM-based query rewrite. If an acronym dictionary buys +4 points of recall on 15% of traffic, its cost-per-point-of-recall is better than anything else in this section by three orders of magnitude. **Exhaust the deterministic query-understanding techniques before you spend a single LLM call**, and say that in the room; it reads as someone who has shipped search, not someone who has read about it.

### Give me your decision rule for when to skip query transformation entirely.

I want an explicit rule because the default failure is architectural accretion — every technique gets added, none gets removed, and TTFT drifts from 800 ms to 3 s over two quarters with no single commit to blame.

The rule, in order of evaluation:

**Skip everything if the query is a first turn, is longer than ~8 words, and contains no pronouns.** That query is already self-contained and specific; every transformation can only add drift. In my traffic that describes 45–60% of queries, and it is free to detect.

**Skip if a cheap confidence signal says retrieval already worked.** The strongest formulation is *retrieve first, transform only on failure*: run the base hybrid retrieval, look at the top reranker score against your calibrated threshold and the `top1 / mean(top10)` gap. If confident, generate. If not, *then* spend the LLM call on rewriting, HyDE or multi-query and retrieve again. This inverts the usual pipeline and it is the design I argue for in review: **you pay the transformation latency only on the queries that need it**, which is typically 15–25% of traffic, instead of on 100%.

The cost of that inversion is that the unlucky 20% pay retrieval *twice* — about +80 ms of retrieval plus the transformation call, so their p95 goes up. Whether that trade is right depends on whether your SLO is on the mean or the tail. **My rule: transform-on-failure when the SLO is on p50 and quality on the tail matters; transform-always when the SLO is on p95 and you cannot afford a bimodal latency distribution.** Say the trade-off out loud; interviewers are listening for whether you know it exists.

**Never skip conversational rewriting on a detected follow-up turn.** It is the one transformation whose absence causes a hard failure rather than a soft one — everything else degrades recall by a few points, and this one makes retrieval meaningless.

**📐 Numbers you must know:** the marginal budget. Each pre-retrieval LLM call is **200–450 ms p95**; a full retrieval round is **60–100 ms**; prefill of the final prompt is **50–100 ms per 1k tokens**. So on a 1.5 s TTFT budget with a 6k-token context (≈400 ms prefill) and 100 ms of retrieval, you have room for **two** serialized LLM calls, not four. Merge routing+rewrite+decomposition+filters into one structured call, and you have spent one of your two.

**🗣 Say this in the room:** "My default is one structured-output call that does routing, rewriting, decomposition and filter extraction together, fired only when a cheap heuristic says the turn isn't self-contained. Everything else — HyDE, multi-query, step-back — is conditional on low retrieval confidence, because I'd rather pay 400 milliseconds on the 20% of queries that failed than on the 100% that didn't."

### The corpus is our own product's chat history — users ask "what did we decide about the retry logic last week?" How is retrieving over conversation history different from retrieving over documents?

**Mental model: conversation is the worst-structured corpus you will ever index.** Documents are written to be read out of context by strangers; conversation is written assuming shared state that exists only in the participants' heads. Every property a retriever depends on is degraded.

The specific differences, each with its mitigation:

**Chunks are not self-contained by construction.** A message reading "yeah let's do that but cap it at 3" is meaningless in isolation and will embed to noise. This is the strongest case in the entire guide for **contextual chunk augmentation**: at index time, prepend an LLM-generated one-sentence summary of the surrounding conversation to each chunk before embedding it. Cost is one small-model call per chunk at index time — for 5M messages at ~250 in/40 out tokens on a cheap model, `5e6 × (250×2.5e-7 + 40×1.25e-6) = 5e6 × 0.0001125 = $562` one-time. **Compare that to the cost of the retrieval simply not working, which is unbounded.**

**Turn boundaries are the wrong chunk boundaries.** A decision spans eight messages over twenty minutes. Chunk by **conversation segment**, using time gaps (say, >30 min of silence) plus participant-set changes plus optionally an embedding-distance breakpoint, not by message. A segment of 5–15 messages with the thread title and participants in the metadata is the unit that actually answers "what did we decide."

**Recency and authority are load-bearing in a way they are not for documents.** "What did we decide" almost always means the *latest* decision, and conversation corpora contain the full history of superseded decisions, all of which are topically identical. Pure semantic retrieval on a 3-year Slack archive returns the 2023 decision and the 2024 reversal with indistinguishable scores. **You must apply a recency prior** — either a hard time filter extracted from the query ("last week" → a range filter, which is exactly the metadata-extraction path above) or a decay multiplier on the fused score.

**⚠ Trap — and this is the one that causes incidents:** conversation corpora are full of **superseded and contradicted statements**, and retrieval has no notion of which won. Someone proposes "let's use exponential backoff," someone else says "no, that breaks the SLA," and the retriever returns the proposal. The mitigation is not retrieval-side; it is prompt-side: instruct the generator to prefer later messages, surface timestamps *inside* the context so the model can reason about ordering, and have it state when sources conflict rather than silently picking one. **Timestamps in the assembled context are non-optional for conversational corpora** and everyone forgets them.

**Permissions are per-channel and change.** A DM indexed while someone was on a team must not be retrievable after they leave it. ACL-aware retrieval with filters resolved at query time from the live permission system — never from a snapshot copied into the index at ingestion — is the only correct design, and it is the reason Glean-shaped products are hard.
