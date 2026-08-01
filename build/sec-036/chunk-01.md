### Before we talk about chunk sizes — what is a chunk actually for? Why does the concept exist at all?

A chunk is the unit at which two completely different jobs collide, and almost every bad chunking decision comes from optimizing one of them while forgetting the other. **Job one: a chunk is the unit of retrieval — the thing that gets one embedding, one BM25 posting-list contribution, one relevance score. Job two: a chunk is the unit of context — the thing that gets pasted into the model's prompt and has to be self-contained enough to answer from.** These pull in opposite directions. Retrieval wants small chunks, because a single dense vector is a fixed-width lossy summary and the more text you cram behind it the more the specifics wash out into topical mush. Generation wants large chunks, because a fragment that says "this rate applies only in the second case" is useless without the paragraph that defined the cases.

The backend analogue that actually helps here: you have seen this exact tension in index design. A covering index makes reads self-sufficient at the cost of index size and write amplification; a narrow index is cheap to maintain but forces a heap lookup to get the rest of the row. Chunking is the same trade with a nastier twist — there is no heap lookup by default. Whatever the retriever returns is all the generator ever sees. If the row is incomplete, you do not get a slow query; you get a confidently wrong answer with a citation attached.

Once you see it that way, the entire advanced-chunking literature stops looking like a grab-bag of tricks and becomes one idea repeated: **decouple the retrieval unit from the context unit.** Parent-document retrieval, sentence-window retrieval, small-to-big, contextual retrieval, RAPTOR, propositions — every one of them is a different answer to "index something small and precise, but hand the model something large and complete." That is the sentence I want you to be able to say, because it reframes six named techniques as one design axis and interviewers notice when a candidate compresses instead of enumerating.

**🗣 Say this in the room:** "Chunking is choosing the granularity of two different things at once — what gets scored, and what gets read. Almost every technique past naive splitting exists to let those two be different sizes. So my first question about any chunking scheme is: what's the retrieval unit, what's the context unit, and are they the same object?"

**⚠ Trap:** treating chunking as a preprocessing detail you settle in week one and never revisit. Chunking is the highest-leverage knob in a RAG system per hour of engineering, because it sits upstream of everything — a chunk that cuts a table in half cannot be recovered by a better embedding model, a better reranker, or a better prompt. I have never seen a reranker upgrade recover more quality than fixing a boundary bug, and I have seen the reverse many times.

### Your take-home used 512-token chunks with 50 tokens of overlap. Why 512? Defend the number.

This is the question that ends most take-home defenses, and the losing answer is "512 is a common default" or worse, "it's what the tutorial used." The winning answer has three parts: what I would do if I had no data, what experiment I ran, and what the experiment said.

**With no data, my prior is 300–500 tokens for prose and I say so explicitly as a prior, not a fact.** The reasoning is mechanical rather than mystical. Most embedding models were contrastively trained on query-passage pairs where passages were roughly paragraph-length — MS MARCO passages average well under 100 words, BEIR corpora are mostly short passages. A model has seen very few 2000-token positives, so its representation quality degrades on inputs far longer than its training distribution even when its advertised max sequence length is 8192. Separately, the information-theoretic argument: one 1024-dim float32 vector is 4KB of budget for whatever you put behind it. Pack 2000 tokens in and the vector describes a topic; pack 200 in and it describes a claim. Retrieval is about claims.

**Then I run the experiment, because the prior is worth about as much as a guessed index fill-factor.** The protocol: build a golden set of 150–300 (query, relevant-passage) pairs from real query logs if I have them and hand-written if I do not. Index the same corpus at 128 / 256 / 512 / 1024 / 2048 tokens, holding embedding model, overlap fraction, and retriever identical. Measure two things separately, and this separation is the part candidates skip: **Recall@k on the retrieval side** (did the chunk containing the answer make the top-k at all?) and **end-to-end answer correctness with an LLM judge or human labels** (did the generator produce the right answer from what it got?). They disagree, and the disagreement is the finding. Small chunks usually win Recall@20 and lose end-to-end, because the retrieved fragment is precise but insufficient. Large chunks win end-to-end at small k and lose recall. The optimum is wherever those curves cross for *your* corpus, and it moves with your document type.

**📐 Numbers you must know:** the crossover I have observed most often for English technical prose sits at 300–600 tokens with 10–15% overlap. For legal contracts with numbered clauses I go smaller, 150–300, because the clause *is* the atomic unit and clause boundaries are hard boundaries. For conversational data (Slack, support tickets) I chunk by thread, not by token count, because a 40-token message is meaningless alone. For code I do not chunk by tokens at all — see the AST question later in this section. Quote these as priors you then test, never as answers.

**💰 Math:** the ablation is cheap and you should say so, because "I would run an ablation" sounds expensive until you price it. 50k documents × ~1200 tokens average = 60M tokens. Embedding all five configurations at a small-embedding-model price of $0.02/M tokens **📅 Volatile: verify current embedding prices before your loop** is 5 × 60M × $0.02/1M = **$6.00** of embedding spend, plus perhaps 90 minutes of wall clock at 1M tokens/minute of throughput. The LLM-judge pass over 300 queries × 5 configs = 1500 judgments at ~2000 input / 200 output tokens each: 3M input + 0.3M output. At $1/M in and $5/M out that is $3.00 + $1.50 = **$4.50**. **The entire experiment costs about eleven dollars and one afternoon.** There is no defensible reason to guess.

**🗣 Say this in the room:** "512 was my starting prior because it's near the passage length these encoders were trained on. But I didn't ship the prior — I swept 128 through 2048 against a 250-query golden set and measured Recall@20 and end-to-end correctness separately. 512 won end-to-end; 256 actually won Recall@20, which told me my failure mode was insufficient context rather than missed retrieval, and that's why I also added parent-document expansion."

**⚠ Trap:** running the sweep and reporting only end-to-end accuracy. If you only have one number you cannot tell whether a config lost because the right chunk was never retrieved or because it was retrieved and too fragmentary to use. Those failures have opposite fixes. Always instrument both stages.

### Implement fixed-size chunking with overlap. Then tell me what's wrong with the naive version everyone writes.

The naive version is four lines and wrong in four ways, so let me write the wrong one first because you will see it in every codebase.

```python
def bad_chunk(text, size=2000, overlap=200):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]
```

Wrong thing one: **it counts characters, not tokens.** Your budget is measured in tokens and the character-to-token ratio is not a constant — roughly 4 chars/token for English prose, closer to 2–3 for code with heavy punctuation, and as low as 1 char/token for CJK text or for text with many rare identifiers. A 2000-character "chunk" is 500 tokens of English and can be 1500 tokens of Japanese. If you sized against an 8192-token embedding limit using characters, some fraction of your corpus is being silently truncated by the encoder. Silently — the API does not error, it just drops the tail on some providers or errors on others, and either way you find out from a user.

Wrong thing two: **it cuts mid-word and mid-sentence**, which produces embeddings for strings like "...the applicable rate is 4.2% unless the coun". That is not a semantic unit and the encoder has never seen anything like it.

Wrong thing three: **the final chunk can be a 12-character orphan.** A chunk containing "ee below." gets an embedding, occupies an index slot, and will occasionally be retrieved because short strings sometimes land near queries in cosine space. Merge trailing chunks below a minimum size into their predecessor.

Wrong thing four: **`range(0, len(text), size - overlap)` with `overlap >= size` is an infinite-ish loop or a step of zero.** It will raise or hang, and it will do so only on the one config in your sweep where you fat-fingered the ratio.

Here is the version I would actually write, token-aware and boundary-aware, in about thirty lines:

```python
import tiktoken

def chunk_text(text, enc, max_tokens=512, overlap_tokens=64, min_tokens=64):
    assert 0 <= overlap_tokens < max_tokens
    # Split into sentence-ish atoms first so boundaries land somewhere legal.
    atoms = [s + " " for s in text.replace("\n\n", " \n\n").split(". ")]
    atom_toks = [len(enc.encode(a)) for a in atoms]

    chunks, cur, cur_len, i = [], [], 0, 0
    while i < len(atoms):
        if cur_len + atom_toks[i] > max_tokens and cur:
            chunks.append("".join(cur))
            # Walk backwards to build the overlap tail.
            back, taken = [], 0
            for a, n in zip(reversed(cur), reversed(atom_toks[i - len(cur):i])):
                if taken + n > overlap_tokens:
                    break
                back.append(a); taken += n
            cur, cur_len = list(reversed(back)), taken
        cur.append(atoms[i]); cur_len += atom_toks[i]; i += 1
    if cur:
        chunks.append("".join(cur))
    # Merge an undersized tail into its predecessor.
    if len(chunks) > 1 and len(enc.encode(chunks[-1])) < min_tokens:
        chunks[-2] += chunks[-1]; chunks.pop()
    return chunks
```

**⚠ Trap:** the subtle one is that a single atom can exceed `max_tokens` all by itself — a 900-token table row, a base64 blob, a minified JS line that survived your HTML cleaner. The loop above appends it anyway and emits an oversize chunk that the embedding API will reject or truncate. Production code needs an explicit hard-split fallback for any atom over the limit, and a counter on how often that fallback fires, because a spike in it means your parser changed behavior.

**📐 Numbers you must know:** ~4 characters per token for English (so 512 tokens ≈ 2,000 characters ≈ 350 words); ~1 token per 0.75 words. These are the two conversions you should be able to do in your head when an interviewer asks "how many chunks is a 40-page document?" — 40 pages × ~500 words/page = 20,000 words ≈ 27,000 tokens ≈ 53 chunks at 512 with no overlap, ~60 with 15% overlap.

### Walk me through recursive character splitting. Why is the separator list ordered the way it is, and where does it fail?

Recursive character splitting is a greedy, structure-respecting fallback ladder, and the mental model is: **try to break at the most semantically significant boundary available; only descend to a less significant one when the piece is still too big.** The separator list is ordered by how much meaning a break at that character destroys. The canonical order for prose is `["\n\n", "\n", ". ", " ", ""]` — paragraph, then line, then sentence, then word, then arbitrary character. You split on `\n\n` first; any resulting piece under the size limit is emitted; any piece still over the limit is recursively re-split on `\n`, and so on down. The empty-string separator at the end is the guarantee of termination — it can always split anything.

The algorithm is about twenty lines and worth being able to write:

```python
def recursive_split(text, seps, max_len, length_fn=len):
    if length_fn(text) <= max_len:
        return [text] if text.strip() else []
    if not seps:
        return [text[i:i+max_len] for i in range(0, len(text), max_len)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep) if sep else list(text)
    out, buf = [], ""
    for p in parts:
        cand = (buf + sep + p) if buf else p
        if length_fn(cand) <= max_len:
            buf = cand
        else:
            if buf:
                out.append(buf)
            buf = p if length_fn(p) <= max_len else ""
            if not buf:
                out.extend(recursive_split(p, rest, max_len, length_fn))
    if buf:
        out.append(buf)
    return out
```

Note the greedy merge in the middle: after splitting you *re-pack* adjacent small pieces up to the limit, which is what keeps you from emitting one chunk per paragraph in a document of two-sentence paragraphs. That packing step is the part people omit and then wonder why their chunk-length histogram has a spike at 40 tokens.

**Where it fails, specifically.** It fails whenever the character-level heuristic and the actual document structure disagree. A PDF converted to text has hard line wraps at column width, so `\n` appears mid-sentence and `\n\n` appears between visually-separated but semantically-continuous blocks — the ladder happily breaks a sentence in half at a wrap. A markdown table has one `\n` per row, so the ladder will split a table between rows and orphan the header. Code has `\n\n` between logically-coupled statements. Bullet lists get severed from their introducing sentence because the sentence ends with `:\n`.

**⚠ Trap:** believing that "recursive character splitter" implies semantic awareness because it splits on paragraphs. It has no model of the document. It is a *string* heuristic that happens to correlate with structure in clean prose and decorrelates completely on anything that went through a layout-to-text conversion. The rule I enforce in review: recursive character splitting is the fallback for text you could not parse structurally, never the primary strategy for a format that has structure you could have used. If the source is HTML, markdown, docx, or code, splitting on structure is strictly better and costs no more.

**🗣 Say this in the room:** "Recursive splitting is a good default for unstructured prose and a bad default for anything with real structure, which is most of an enterprise corpus. I'd reach for it only after asking what format the source actually is."

### Explain semantic chunking with embedding-distance breakpoints. Would you ship it?

The mechanism is genuinely elegant: instead of guessing where topics change, measure it. Split the document into sentences. Embed every sentence — often with a small sliding buffer of ±1 neighbouring sentence so each embedding has a little context. Then walk the sequence computing the cosine distance between consecutive sentence embeddings. Where the distance spikes, the topic changed; that is your boundary. Rather than a fixed distance threshold (which is uncalibrated across documents and models), the standard implementation uses a **percentile breakpoint**: take all consecutive distances in this document, and cut wherever the distance exceeds, say, the 95th percentile of that document's own distance distribution. That makes it self-normalizing per document.

```python
import numpy as np

def semantic_chunks(sentences, embed_fn, percentile=95, buffer=1):
    ctx = [" ".join(sentences[max(0, i - buffer): i + buffer + 1])
           for i in range(len(sentences))]
    E = np.array(embed_fn(ctx))                       # [n, d], L2-normalized
    d = 1.0 - np.sum(E[:-1] * E[1:], axis=1)          # cosine distance, len n-1
    thresh = np.percentile(d, percentile)
    cuts = [0] + [i + 1 for i, x in enumerate(d) if x > thresh] + [len(sentences)]
    return [" ".join(sentences[a:b]) for a, b in zip(cuts, cuts[1:])]
```

**Would I ship it? Usually no, and I want to give you the honest reason rather than a fashionable one.** Three problems.

First, **cost**. It requires embedding every sentence in the corpus, on top of embedding every final chunk. For a corpus of 60M tokens with average 25-token sentences that is 2.4M sentence embeddings; with a ±1 buffer each embedding covers ~75 tokens, so ~180M tokens of embedding calls versus 60M for the chunks themselves — a 4× increase in ingestion embedding cost and, more painfully, 4× the API round trips and rate-limit pressure. At $0.02/M that is $3.60 vs $1.20, which is nothing; at 500M tokens and a premium encoder at $0.13/M it is $195 vs $65 and it starts mattering, and at re-ingestion-every-week cadence it compounds.

Second, **the published ablations are unconvincing**. Multiple independent evaluations — including work out of Chroma on chunking strategies — have found semantic chunking's advantage over well-tuned fixed-size-with-overlap to be small and inconsistent across corpora, and sometimes negative. It is one of those techniques whose intuitive appeal exceeds its measured effect. I say this in interviews and it has never gone badly, because the interviewer usually knows.

Third, **it has no size control.** Percentile breakpoints produce a chunk-length distribution with a long right tail — a document with genuinely uniform topic produces one enormous chunk. Every production implementation ends up bolting a max-size hard split back on top, at which point you have fixed-size chunking with extra steps and extra cost.

**Where it does earn its keep:** genuinely unstructured, topic-shifting, boundary-free text where you have no structural signal at all. Meeting transcripts and long interview recordings are the honest use case — no headings, no paragraphs, real topic shifts. There, I have seen it beat fixed-size meaningfully.

**⚠ Trap:** the failure mode nobody mentions is that consecutive-sentence distance is enormously noisy for short sentences. "Yes." followed by "That's the third quarter number." has a huge cosine distance and zero topic change. Filter or merge sentences under ~5 tokens before computing distances, or your breakpoints land on every "Right." in the transcript.

**🗣 Say this in the room:** "Semantic chunking is a real technique with a weak track record against a properly tuned baseline, and it costs 3–5× in ingestion embeddings. I'd use it for transcripts where there's no structural signal, and I'd default to structure-aware splitting everywhere structure exists — which is most places."

### How do you chunk markdown and HTML properly? Show me what the chunk record looks like.

For any format with a document tree, you split the tree, not the string. The rule: **a chunk should be a contiguous span of the document that lives under exactly one path in the heading hierarchy, and that path travels with the chunk as metadata.**

For markdown, you walk the document maintaining a heading stack. Every `#`-level heading pops the stack down to its depth and pushes itself. The text between headings is the leaf content, and it gets split further only if it exceeds the token budget. Crucially, when you emit a chunk, **you prepend the heading path to the chunk text itself, not just to the metadata**. A chunk from deep inside a document that reads "Set the value to 30 seconds." is nearly unretrievable; the same chunk prefixed with `# Deployment Guide > ## Timeouts > ### Idle connection timeout` is trivially retrievable because the query "what's the idle connection timeout" now shares surface tokens *and* semantic content with the chunk. This is the single cheapest quality win in the entire section — it costs nothing at index time and nothing at query time.

```python
def markdown_chunks(md, max_tokens, enc):
    stack, buf, out = [], [], []
    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        path = " > ".join(h for _, h in stack)
        for piece in chunk_text(body, enc, max_tokens):        # from earlier
            out.append({"section_path": path,
                        "text": f"{path}\n\n{piece}" if path else piece})
        buf.clear()
    for line in md.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, line.lstrip("# ").strip()))
        else:
            buf.append(line)
    flush()
    return out
```

HTML is the same algorithm over a parsed DOM rather than a line scan, with two extra jobs: **strip chrome before you chunk** (nav, header, footer, cookie banner, sidebar, script/style) and **preserve semantics that survive to text** — convert `<table>` to markdown tables, `<li>` to `- `, `<a href>` to `text (url)` if the URL is load-bearing in your domain, and `<code>`/`<pre>` verbatim with fences. Readability-style main-content extraction (the `readability-lxml` or `trafilatura` approach) before chunking is worth its weight; without it every chunk in your index contains your own navigation menu and the encoder wastes representation capacity on it.

**⚠ Trap:** stripping tags with a regex or with `soup.get_text()` and calling it done. `get_text()` on a table produces a run of cell values with no row or column structure — `Q1 4.2 Q2 5.1 Q3 3.9` — which is not just unhelpful but actively wrong, because the model will happily attribute 4.2 to Q2. Any table that survives to your index without structure is a latent hallucination.

**🗣 Say this in the room:** "For anything with a document tree — markdown, HTML, docx, code — I split the tree and carry the heading path into the chunk text itself. Prefixing the section path is the highest-ROI thing in a chunking pipeline: zero marginal cost, and it makes deep fragments retrievable that otherwise carry no locating information at all."

### Chunking source code by character count — why is that malpractice, and what would you do instead?

Because code has an unambiguous, machine-readable structure and character splitting throws it away for no reason. A 512-token window cut through a Python file lands in the middle of a function body with probability near one. What you have then indexed is a fragment with no signature, no docstring, no imports, no class context — a body of statements referencing names that were defined in a chunk you did not retrieve. Its embedding encodes "some Python that does dict lookups," which matches ten thousand other fragments.

The correct unit is a **syntactic node**, and the tool is tree-sitter, which gives you an incremental, error-tolerant concrete syntax tree for ~40 languages behind one API. The chunking algorithm: parse the file, walk top-level declarations (functions, classes, methods, type definitions), and emit one chunk per declaration. If a declaration exceeds the budget, descend into it and split at statement boundaries within the body rather than at arbitrary offsets. If a declaration is tiny — a three-line helper — pack adjacent siblings together up to the budget rather than emitting 8-token chunks.

Then add the two enrichments that matter far more than the split points themselves:

**Enrichment one: a synthesized header on every chunk.** File path, language, enclosing class, the imports actually referenced by this chunk, and the signature of the enclosing function if you split inside one. Concretely, a chunk becomes:

```
# file: services/billing/invoice.py
# imports: from decimal import Decimal; from .tax import rate_for
# class: InvoiceBuilder
def add_line_item(self, sku: str, qty: int) -> None:
    ...
```

Now the query "how do we compute tax on an invoice line" can hit it via the path, the class name, and the import, none of which appear in the body.

**Enrichment two: a symbol index alongside the vector index.** An exact-match table mapping symbol name → definition location, built from the same tree-sitter pass. Half of all code queries are "where is `X` defined" or "who calls `X`," and those are lookups, not searches. Answering them with cosine similarity is the wrong data structure — this is the same instinct as reaching for a full-text search when you needed a unique index.

**⚠ Trap:** chunking a code repo and forgetting that the most valuable retrieval signal is often *not in the file at all* — it's the call graph. A function that is called from 200 places is more likely to be what the user meant than one called from zero. Ingest an import/call edge list during the same parse and use in-degree as a static boost, exactly as you would use PageRank-ish authority in web retrieval.

**📐 Numbers you must know:** tree-sitter parses on the order of tens of megabytes per second per core and is incremental — reparsing after an edit is proportional to the edit, not the file. That is what makes per-commit reindexing tractable: a 5M-LOC monorepo (~200MB of source) is a few seconds of parsing on a handful of cores for a full pass, and milliseconds for the 30 files a typical commit touches.

### Give me the experiment design. How do you actually choose chunk size with data rather than vibes?

Four things have to be nailed down before you run anything, and interviewers are checking whether you know that the eval design is harder than the sweep.

**One: the golden set, and where it comes from.** The best source is production query logs — take the real head of your query distribution, sample 200–300, and have a human (or a strong model with human spot-check) label which passages of which documents actually answer them. Second best, if you have no logs: generate questions *from* the corpus with an LLM — sample a chunk, ask for a question only that chunk can answer — then **filter aggressively**, because raw LLM-generated questions are lexically parasitic on their source chunk and will make any retriever look great. The filter I use: drop any generated question that shares more than ~40% of its content words with the source chunk, and drop any where a BM25-only retriever finds the source at rank 1, because those are testing string matching, not retrieval. Third best: hand-write 50. Fifty hand-written questions from someone who knows the domain beats 5,000 generated ones.

**Two: what stays fixed.** Embedding model, k, overlap fraction, normalization, and the generator prompt. Sweep one axis. If you change chunk size and embedding model together you have learned nothing and you will still ship a number.

**Three: the label granularity problem.** This is the subtle one. If you label "document 47, characters 8200–9100 answer this question," you can score any chunking configuration by asking whether a retrieved chunk *overlaps* the labeled span — no re-labeling per config. **If you label at the chunk level you have to re-label for every config, which makes the sweep 5× the human cost and is the reason most teams do it once and never again.** Label spans by character offset into the source document. This is a five-minute decision that determines whether your eval is reusable.

**Four: the two metrics, measured separately.**

- **Retrieval:** Recall@k for k ∈ {5, 10, 20}, where a hit means any retrieved chunk overlaps the labeled span by at least, say, 50% of the span. Also report nDCG if you have graded relevance.
- **End-to-end:** answer correctness against a reference, scored by an LLM judge with a rubric, on the *same* queries. Plus a groundedness/citation check — did the answer's claims appear in the retrieved context.

Then plot both against chunk size and read the shape.

**🔍 Failure taxonomy — how to read the result:**
- Recall@20 flat across sizes, end-to-end rising with size → your retriever is fine and your failures are context sufficiency. Fix with parent-document expansion or larger chunks, not a better encoder.
- Recall@20 falling as size grows, end-to-end also falling → chunks are too topically diluted; the vector is describing a document, not a claim. Go smaller and add the section-path prefix.
- Both flat across the whole sweep → chunk size is not your bottleneck. Stop sweeping. Your problem is lexical/dense mismatch, or parsing, or the queries are unanswerable from the corpus. Go check what fraction of your golden questions have *any* supporting passage.
- End-to-end high, groundedness low → the model is answering from parametric knowledge and your retrieval is decorative. This is common on general-knowledge corpora and it invalidates the whole sweep. Re-run with an adversarial subset where the answer is corpus-specific.

**🏋 Drill (90 minutes, unaided):** take any 2,000-document corpus you have. Write the span-offset labeling harness, hand-label 50 queries, sweep {256, 512, 1024} × {0%, 15% overlap}, and produce one plot with Recall@10 and judge-scored correctness on twin axes. **Pass criterion: you can state which config you would ship and name the one failure mode the plot rules out.** If you cannot name what the plot ruled out, you built a chart, not an experiment.

### How much overlap, and what does overlap actually cost me?

Overlap exists for exactly one reason: **a boundary is a coin flip on whether a fact and its context land together, and overlap buys you a second flip.** If the sentence "the limit is 30 requests per second" ends chunk 4 and the sentence that names which endpoint begins chunk 5, then with zero overlap neither chunk answers the question and no retriever can save you. With 15% overlap, the tail of chunk 4 appears at the head of chunk 5, and chunk 5 contains both.

That framing tells you how to size it: **overlap should be at least as long as the typical dependency distance in your text.** For prose, a sentence or two — 50–100 tokens on a 512-token chunk, i.e. 10–20%. For dense reference material where a definition governs the following paragraphs, more. For structure-aware chunking where boundaries land on real headings, often zero, because the boundary is no longer a coin flip — it is a genuine semantic break, and overlapping across it just duplicates.

**💰 Math — what it costs.** Overlap of fraction *f* multiplies chunk count by roughly `1/(1−f)`. At f=0.15, 1.18×. At f=0.25, 1.33×. At f=0.5, 2×. That multiplier hits four things:

- **Embedding cost at ingest:** 100M-token corpus at 15% overlap = 118M tokens embedded. At $0.13/M for a premium encoder: $15.34 vs $13.00. Trivial.
- **Index size and RAM:** 100M tokens / 512 = ~195k chunks becomes ~230k. At 1024-dim float32 that is 195k × 4KB = 800MB → 940MB, plus HNSW graph overhead of roughly M×2×4 bytes/vector (~128 bytes at M=16) — call it 25MB → 30MB. Still trivial at this scale; at 100× this scale (20M chunks) the 35k extra vectors per million start costing real RAM: 19.5M × 4KB = 78GB → 92GB, and a 14GB delta is a machine.
- **Duplicate results in the top-k.** This is the real cost and it is a *quality* cost, not a dollar one. With 25% overlap, adjacent chunks share a quarter of their text and score nearly identically. A top-5 can easily be three overlapping windows of the same paragraph, so your effective context diversity is 5 slots holding 2 paragraphs. **This is why overlap and MMR/near-duplicate suppression are the same conversation.**
- **Citation ambiguity:** the same sentence exists at two (doc, offset) locations, so "which chunk do I cite" needs a deterministic tiebreak.

**⚠ Trap:** cranking overlap to 50% "for safety." You have doubled your index, halved your effective top-k diversity, and made near-duplicate suppression mandatory rather than optional. The rule I enforce: overlap above 25% requires a dedup step in the same PR, or I do not approve it.

**🗣 Say this in the room:** "I treat overlap as insurance against boundary bisection, so I size it to the dependency distance in the text — 10–15% for prose — and I drop it to zero when the boundaries are structural rather than arbitrary. Past about 25% it stops buying recall and starts filling the top-k with near-duplicates, so it has to come with a dedup pass."

### A user asked about the Q3 revenue table and the model gave a confidently wrong number. You trace it to a chunk containing the bottom half of a table with no header row. Walk me through the fix.

This is the canonical ingestion incident and I want to walk it as a decision procedure, because the instinct — "increase chunk size so tables fit" — is the wrong fix and will be your first temptation.

**Why the failure is severe and silent.** A table's meaning lives entirely in the header row: the numbers are meaningless without the column labels. When you split a table, the top fragment is fine and the bottom fragment is a grid of numbers with row labels and no column labels. That bottom fragment still gets an embedding, still matches "Q3 revenue" reasonably well (the row labels probably mention quarters or line items), still gets retrieved, and the model then does exactly what you would do under pressure: it assumes column order and picks a number. There is no error, no low-confidence signal, no exception. It is a wrong number with a citation, which is worse than no answer, and it is the class of bug that gets a RAG product pulled from a finance team.

**The fix, in order of what I would actually do:**

**Step 1 — make tables atomic objects, not text.** In the parsing stage, tables come out as structured objects (list of rows, plus a header row, plus a caption, plus a bounding region and page number), never as a run of text handed to a character splitter. The chunker's contract becomes: *a table is never split by the generic splitter.* If a table fits the budget, it is one chunk. This is a parser-level architecture decision, not a chunker tweak, which is why "increase chunk size" does not fix it — a bigger budget just moves the bisection to a bigger table.

**Step 2 — for tables that genuinely exceed the budget, split by rows and repeat the header on every fragment.** Each fragment carries: the caption, the header row, a `rows 41–80 of 214` marker, and the section path. Repeating a 12-token header on 6 fragments costs 72 tokens total. That is the entire cost of eliminating this failure class.

**Step 3 — attach a linearized summary line for retrieval.** Tables retrieve badly as raw grids because embeddings of number grids are near-degenerate. Prepending "Quarterly revenue by business unit, FY2024, USD millions" — either the real caption or an LLM-generated one — makes the table findable by the query that actually gets asked. This is contextual retrieval applied to tables specifically, and it is the highest-value place to spend those tokens.

**Step 4 — the regression test, which is the part that gets you hired.** Add a golden query per critical table whose correct answer is a specific cell value, and assert the exact number in CI. Table extraction breaks on parser upgrades, and it breaks silently. A unit test that says "the Q3 EMEA figure is 4,182" will catch a PyMuPDF version bump that changed cell ordering. Without it, you will find out from a customer.

**🔍 Failure taxonomy for table ingestion**, as a checklist to run against any corpus:
1. Is any table split across chunks? Assert: no chunk contains a pipe-table fragment whose first row is not the header.
2. Is any table split across *pages* in the source, with the header on page 1 and rows on page 2? Detect by column-count continuity and merge before chunking.
3. Do merged cells exist? They serialize as blanks and shift row alignment. Forward-fill and flag.
4. Are numbers being read with locale-swapped separators — `1.234,56` parsed as `1.234`? This one silently changes magnitude by 1000×.
5. Does the table have a caption at all? If not, generate one; an uncaptioned table is unretrievable.

**💰 Math:** the alternative some teams choose — "just don't chunk tables, send whole documents" — costs you the entire context budget. A 40-page financial report is ~27k tokens. Sending it whole on every query at $3/Mtok input is $0.081/query; at 50k queries/day that is $4,050/day, $121k/month, versus retrieving three 600-token chunks at ~$0.005/query, $250/day. **The table fix costs 72 tokens; the "avoid the problem" fix costs $120k a month.**

### Every one of my chunks starts with "CONFIDENTIAL — Acme Corp — Page 7 of 40". How did that happen and how do you fix it?

This is header/footer pollution and it is worth understanding why it is more damaging than it looks. Assume a 40-token header/footer band on every page and 512-token chunks: you have donated **8% of every chunk's representation budget to a string that is identical across the entire corpus.** Two consequences. First, it dilutes every embedding toward a common corpus centroid — every chunk now shares a constant component, which compresses the spread of cosine similarities and makes your top-k rankings mushier. Second, and worse for hybrid retrieval, a query containing "Acme" now matches literally everything, and BM25's IDF for "Acme" collapses to near zero so it stops being a useful discriminator even where it should be one.

**How it happens:** naive extraction reads a PDF's text objects in content-stream order, page by page, and the page furniture is just more text objects. `PyMuPDF.get_text()` returns it. Concatenate 40 pages and you have 40 copies interleaved through your body text — and if you concatenated before chunking, the header now appears *mid-chunk*, splitting sentences that ran across the page break.

**The fix, three levels of effort:**

*Level 1, works surprisingly well:* frequency-based detection. Extract text per page with bounding boxes. Any text block whose normalized content (digits masked to `#`) appears on more than ~60% of pages **and** sits in the top or bottom 10% of the page's vertical extent is furniture. Drop it. The digit masking is what makes `Page 7 of 40` and `Page 8 of 40` collapse to the same key.

```python
from collections import Counter
import re

def find_furniture(pages, band=0.10, min_frac=0.6):
    """pages: list of [(text, y0, y1, page_h)] blocks per page."""
    counts, n = Counter(), len(pages)
    for blocks in pages:
        seen = set()
        for text, y0, y1, h in blocks:
            if y1 < band * h or y0 > (1 - band) * h:
                seen.add(re.sub(r"\d+", "#", text.strip()))
        counts.update(seen)
    return {k for k, c in counts.items() if c >= min_frac * n}
```

*Level 2:* use a layout model. Docling, Unstructured, Surya-based pipelines and Azure Document Intelligence all emit element types including `PageHeader`/`PageFooter`. If you are already running layout analysis, filtering on element type is free and more accurate than the heuristic.

*Level 3, and do this regardless:* **assert it in CI.** Compute the most common 5-gram across a random sample of 1,000 chunks. If any 5-gram appears in more than 20% of chunks, fail the build and print it. This one check catches header pollution, boilerplate legal disclaimers, template email signatures, "Click here to unsubscribe," and the Confluence page footer — an entire family of bugs with one assertion.

**⚠ Trap:** dropping the header without checking whether it carries the *only* copy of load-bearing metadata. In many enterprise PDFs the running header is the only place the document title, contract number, or effective date appears on every page. The right move is not to delete it — it is to **promote it to chunk metadata and remove it from chunk text.** Then a filter on `contract_id` still works, and it stops eating your embedding budget.

**🗣 Say this in the room:** "Repeated page furniture is a silent quality tax — it's 8% of every chunk's tokens spent on a string that's constant across the corpus, and it destroys the IDF of your own company name in the lexical index. I detect it by frequency plus vertical position, promote anything load-bearing into metadata, and I keep a CI assertion that no 5-gram appears in more than 20% of chunks."

### A retrieved chunk says "as noted above, this is contraindicated in such patients." What's broken and what do you do about it?

This is **anaphora across the chunk boundary** — the chunk contains a claim whose referents live outside it. "This," "such patients," "the above," "the aforementioned party," "Section 4.2 provides," "as previously defined." The text is grammatical, embeds fine, retrieves fine, and is *unusable and dangerous* in isolation: a model handed this chunk will either refuse or, more often, invent a plausible referent. In a medical or legal corpus that is not a quality issue, it is a liability issue.

**The important reframing:** this is not a chunking bug you can fix by moving the boundary. Dangling references exist at *every* scale — a section references an earlier section, a chapter references an earlier chapter. There is no chunk size at which reference resolution becomes unnecessary. So the fix has to be a resolution step, not a boundary step.

**Three fixes, in ascending cost:**

**Fix 1 — carry structural context into the chunk (free).** Section path prefix, plus the immediately preceding heading's first sentence. Resolves a surprising fraction of "such patients" cases because the population was usually defined in the section intro.

**Fix 2 — parent-document / sentence-window expansion at retrieval time (cheap).** You retrieved a 200-token chunk; you send the model its 1,200-token parent section. The "above" is now literally above. This is the highest ratio of fix-to-effort in the whole category and it is why small-to-big retrieval exists. Cost: your context per retrieved item goes from 200 to 1,200 tokens, so at k=5 you moved from 1k to 6k tokens per query — at $3/Mtok that is $0.003 → $0.018 per query, $15/day at 1k queries. Buy it.

**Fix 3 — contextual retrieval (expensive at index time, free at query time).** Have an LLM write 50–100 tokens of document-situating context for each chunk at ingestion and prepend it. "This chunk is from the contraindications section of the Drug X prescribing information; 'such patients' refers to patients with hepatic impairment defined in the preceding section." Now the chunk is self-contained *and* better-retrieved.

**⚠ Trap:** thinking increasing chunk size solves it. It moves the boundary; it does not eliminate boundaries. Worse, larger chunks make the retrieval side worse (topical dilution) while only probabilistically helping the anaphora side. I have watched a team go 512 → 2048 to "fix dangling references," lose 6 points of Recall@10, and still have dangling references.

**🔍 How I detect the problem at scale before a user does:** run a cheap classifier over the whole chunk index — a regex/keyword pass for `\b(this|these|those|such|the above|aforementioned|as noted|see section)\b` in the **first 15 tokens** of a chunk is a high-precision signal for "opens with a dangling reference." Measure what percentage of your index trips it. If it is over ~10%, your chunking strategy is structurally wrong for this corpus and you need parent expansion before you tune anything else. This is a two-minute script and it gives you a number to put in the design doc.

### What metadata do you attach to every chunk, and why do character offsets matter so much?

Here is the schema I would defend in a design review, field by field, with the reason each one exists — because "we store metadata" is not an answer and every field should have a job.

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str          # deterministic: sha256(doc_id | start | end | parser_ver)[:16]
    doc_id: str            # stable across re-ingestion; see the ID question later
    tenant_id: str         # partition/filter key — never optional, never inferred
    text: str              # what gets embedded AND what gets shown
    section_path: str      # "Deployment > Timeouts > Idle connection"
    page: int | None       # for PDFs; the citation target humans understand
    char_start: int        # offset into the *normalized source text*
    char_end: int
    source_uri: str        # where a human clicks to verify
    content_hash: str      # sha256 of normalized doc text at ingest
    parser_name: str       # "docling"
    parser_version: str    # "2.14.0" — you WILL need to diff by this
    chunker_config_hash: str  # hash of {strategy, size, overlap, separators}
    embed_model: str       # "text-embedding-3-large@3072"
    created_at: datetime
    doc_updated_at: datetime  # for recency scoring and staleness alerts
    acl: list[str]         # group ids; enforced at query time, not post-hoc
```

**Why character offsets are the field people skip and then rebuild in a panic.** Four reasons:

**Citation with highlighting.** Users trust an answer when they can click a citation and see the exact sentence highlighted in the original document. That requires (doc, char_start, char_end) mapped back onto the rendered source. Page number alone gets them to a page and makes them hunt. The difference in perceived trustworthiness between "page 7" and a highlighted span is enormous and it is the single most-requested feature in every RAG product I have worked on.

**Re-labeling-free evals.** As I said in the sweep question: label answers as spans in the source document, and any chunking config can be scored by span overlap. Without offsets, every config change invalidates your labels.

**Deduplication and provenance across configs.** When you run two chunkers in parallel (shadow index), offsets tell you that chunk A from config 1 and chunk B from config 2 are the same underlying text. Without them you are string-matching.

**Incremental reprocessing.** When a document changes, offsets let you determine *which* chunks changed rather than rewriting all of them — you diff the normalized text, map changed spans to chunks, and re-embed only those. On a corpus where documents get small edits frequently (a wiki, a docs site), this is the difference between re-embedding 20 chunks and re-embedding 400 per edit.

**⚠ Trap:** storing offsets into the *raw* file bytes rather than the *normalized extracted text*. Raw PDF bytes have no meaningful character positions at all; raw HTML offsets shift the moment you change your tag stripper. Define exactly one canonical normalized text per document, store it (it is cheap — it is smaller than the PDF), and make all offsets relative to it. Then a parser upgrade changes the normalized text and you know, from `parser_version`, exactly which chunks' offsets are now invalid.

**🗣 Say this in the room:** "Every chunk carries doc id, tenant, section path, page, character span into a stored canonical text, content hash, parser version, chunker config hash, and embedding model. The two that people skip and regret are the character span — because it's what makes citations clickable and evals reusable across configs — and the parser/chunker version fields, because they're how you scope a backfill when you upgrade something."
