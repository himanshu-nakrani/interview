### We have thousands of spreadsheets and CSV attachments, not a warehouse. How do you make tables retrievable — row-level or table-level chunking?

The mental model: **a table has two independent retrieval targets — "which table is this question about?" and "which rows matter?" — and they want opposite chunking.** Trying to serve both with one index is why table RAG usually disappoints.

**Table-level indexing.** One document per table: its title, its source file and sheet, a column-by-column description (name, inferred type, a few sample values, min/max for numerics, cardinality for categoricals), row count, and a short LLM-written summary of what the table is about. Embed that. This retrieves *the right table* for questions like "what's in the Q3 pricing sheet?" It cannot answer "what did we charge Acme?" because no cell values are in the index.

**Row-level indexing.** Each row becomes a document, serialized with its column names attached — `region: EMEA | product: Widget-A | units: 1200 | revenue: 48000` — plus the table title carried along. Now cell values are searchable, so entity lookups work. But you have exploded 40,000 rows into 40,000 embeddings that are nearly identical to each other (rows of the same table differ in a handful of tokens out of 40), which is a pathological case for dense retrieval: everything is similar to everything, and the ranking is dominated by whichever numeric literal happens to tokenize near the query. Row-level *lexical* retrieval works far better than row-level dense retrieval for exactly this reason — BM25 on `Acme` finds the Acme row; cosine similarity finds forty rows that are all "a row about a customer."

**The architecture I'd actually build — two stages:**

1. **Route to a table** with the table-level index (hybrid, since sheet names and column names are exact-match-y).
2. **Then don't retrieve rows at all — query them.** Load the table into DuckDB or pandas and run a generated query against it. You have reduced the problem to text-to-SQL over a single small, known schema, which is the *easy* case of text-to-SQL: 8 columns, no join ambiguity, no unstated business filters. Everything from the text-to-SQL section applies and works much better here than on a 4,000-table warehouse.

This is the thing to say in the room: **for anything numeric or aggregate, retrieval over rows is the wrong tool and computation over rows is the right one.** Row-level embedding is for the case where rows are genuinely prose-like — a support-ticket export where the useful column is a paragraph of free text. Then it isn't really a table, it's documents with metadata, and you should treat it that way.

**When you must stuff rows into the prompt** (small tables, mixed text+table documents), keep the *whole* table together if it fits — a table split across chunk boundaries loses its header, and a body row without column names is uninterpretable. Chunk-size-driven splitting cutting a table in half is one of the most common and most silent ingestion bugs.

**⚠ Trap:** embedding a row without its column headers. `EMEA | Widget-A | 1200 | 48000` retrieves poorly and, worse, is unusable in the prompt because the model has to guess which number is units and which is revenue. Always serialize key-value, never positionally.

### What serialization format do you use for tables in the prompt, and what does each cost you in tokens?

The trade is **token efficiency versus parsing reliability**, and the answer depends on table shape more than on model preference.

**Markdown pipe tables.** The header appears once; each row is `| a | b | c |`. Token overhead is the pipes and spaces — roughly 2–3 tokens per cell of pure structure. Models read these very well because they're ubiquitous in training data. Breaks down when cell content contains pipes or newlines, and becomes unreadable past ~15 columns because the model loses column alignment across a long row.

**CSV/TSV.** The cheapest — one delimiter token per cell, header once. But it is the most error-prone for the model on wide tables: counting to the eleventh comma to find the eleventh column is exactly the kind of positional task models are unreliable at, and one quoted field with an embedded comma silently shifts every subsequent column.

**JSON records** (`[{"region": "EMEA", "units": 1200}, ...]`). Most robust — every value is adjacent to its key, so column misalignment is impossible and the model never has to count. Also the most expensive: **the column names repeat on every row.** For a 12-column table, that's 12 key strings plus punctuation per row, easily 3–4× the token count of CSV.

**Key-value line format** (`region: EMEA | units: 1200 | ...`) — the same robustness as JSON at somewhat lower token cost, and it's what I use for row-level retrieval documents.

**💰 Math for a 10-column, 500-row table** with short values, rough per-row estimates:
- CSV: header ~15 tokens, then ~10 values × ~2 tokens + 9 commas ≈ **29 tokens/row** → 500 × 29 + 15 ≈ **14,500 tokens**
- Markdown: ~10 values × 2 + 11 pipes + spacing ≈ **35 tokens/row** → ≈ **17,500 tokens**
- JSON records: 10 keys × ~3 tokens + 10 values × ~2 + punctuation ≈ **62 tokens/row** → ≈ **31,000 tokens**

At $3/Mtok input, that's $0.044 vs $0.053 vs $0.093 per call. JSON costs **2.1× CSV**. On one call that's nothing. At 100,000 calls/day it's $4,400/day versus $9,300/day — a $4,900/day gap, **about $147k per month for a formatting choice**, which is the kind of number that makes this question worth asking.

**My rule:** ≤ 8 columns and ≤ 50 rows → markdown, it's cheap and readable. Wide tables (> 12 columns) → JSON records or key-value lines, because column misalignment errors cost more than tokens. Large tables → **don't serialize at all**; hand the model a query interface and let it compute.

**⚠ Trap:** serializing a merged-cell or multi-level-header table (common in real spreadsheets and PDF tables) into a flat format. The header hierarchy is destroyed and the model attributes values to the wrong column. Detect multi-level headers at parse time and either flatten them explicitly into compound column names (`Q3_2024_Revenue`) or refuse the table into a quarantine queue.

### A user asks "what was the average margin across these 200 rows?" and the model returns a number that's slightly off. Why, and what do you do about it?

Because **transformers do not compute; they pattern-match token sequences that look like computation.** There is no accumulator. Summing 200 four-digit numbers requires maintaining exact intermediate state across hundreds of steps of attention, and attention is a soft, lossy retrieval over positions — it will get the magnitude and the first couple of digits right and drift on the rest. That's why the answer is *slightly* off rather than absurd, which is precisely what makes it dangerous: an absurd number gets caught, a plausible one goes in the report.

The failure modes stack up in a specific order as tables grow:
- **Small tables (< 20 rows), simple ops:** usually correct, especially with chain-of-thought that writes intermediate sums.
- **Medium (20–200 rows):** arithmetic drift on sums and averages; ranking questions ("which region was third?") start failing because ordering 200 items requires a comparison sort the model can't perform reliably.
- **Large (> 200 rows):** unreliable for anything aggregate. Also, the model starts *sampling* — it computes over the rows it attended to and reports a confident answer as if it had seen all of them.
- **Any size, multi-step:** "margin = (revenue − cost) / revenue, then average, weighted by units" compounds error at every step.

**The fix is not a better prompt. It is to not ask the model to do arithmetic.** Route numeric questions to a code path:

```python
# The model writes code; the interpreter computes the answer.
resp = model.generate(
    system="You have a pandas DataFrame `df` with columns: " + schema_of(df) +
           "\nWrite Python that assigns the answer to `result`. Do not compute by hand.",
    user=question,
)
result = sandboxed_exec(resp.code, {"df": df})     # separate process, no net, timeout
answer = model.generate(f"Question: {question}\nComputed result: {result}\n"
                        f"Answer in one sentence, citing the computed value.")
```

This is the same architectural move as the semantic layer in text-to-SQL: **the model produces a specification, a deterministic engine produces the number.** Provider-hosted code-execution tools give you this without running a sandbox yourself; if you roll your own, it's a separate process with no network, a CPU and memory cap, and a wall-clock timeout — the usual untrusted-code checklist.

**Where the model is genuinely good and you should let it work:** reading a *single* value out of a table, comparing two values, explaining what a table shows qualitatively, deciding *which* computation answers the question, and interpreting the computed result. Those are language tasks. The arithmetic isn't.

**⚠ Trap:** believing chain-of-thought fixes this. It reduces the error rate on small tables and creates a much more convincing wrong answer on large ones, because now the model shows plausible working that contains an error at step 14 of 40. Visible reasoning is not verified reasoning.

**🗣 Say this in the room:** "I don't let the model do arithmetic over tables at all — it writes pandas or SQL and an interpreter computes. The model's job is choosing the computation and explaining the result. And I'd say the same thing about chain-of-thought here: it makes wrong answers more persuasive, not less frequent."

### Now teach me multimodal RAG from the ground up. What are the two fundamental architectures for retrieving over images and documents?

Start from the constraint: **retrieval requires a single scoring function between a query and a candidate. Text queries and image candidates live in different modalities, so you need to bridge them somewhere. There are exactly two places to put the bridge — at index time or at embed time.**

**Architecture A — separate indexes with a text bridge (bridge at index time).** Run every image through a captioning model or a VLM at ingestion, producing text: a description, extracted text, a summary of what a chart shows. Index that text in your ordinary text index alongside the document's prose. At query time everything is text-to-text retrieval — the machinery you already have, with BM25, rerankers, and hybrid fusion all working unchanged. The image itself is stored and referenced by ID; when a caption is retrieved, you can put the actual image into the generation prompt.

**Architecture B — a joint embedding space (bridge at embed time).** Use a model trained so that images and text land in the same vector space: CLIP, SigLIP, or a multimodal embedding model. Embed images at index time into that shared space; embed the text query into the same space; a single ANN search over one index returns images and text ranked together by cosine similarity. No captioning pass.

The trade-off is sharp and worth stating crisply:

| | Separate + captions (A) | Joint embedding (B) |
|---|---|---|
| Index cost | High — a VLM call per image | Low — one encoder forward pass |
| Recall on *specific* content | High: captions can contain the axis labels, the part number, the numbers in the chart | Low: a single 768-d vector cannot encode a table of numbers |
| Query flexibility | Text queries only (unless you also caption the query) | Native image-to-image and text-to-image search |
| Lexical matching | Works — captions go through BM25, so "part MX-4471" hits exactly | Doesn't work at all |
| Debuggability | Excellent — you can read why something was retrieved | Poor — it's a vector |
| Freshness of representation | Re-captioning is expensive but incremental | Model upgrade forces full re-embed |

**The honest default for enterprise document RAG is A**, and it surprises people. The reason is that most "images" in enterprise corpora are not photographs — they're charts, screenshots, diagrams, and scanned pages, i.e. **text rendered as pixels**. CLIP-style joint embeddings were trained on natural-image/alt-text pairs and are weak at reading dense text in an image; a caption from a good VLM captures the axis labels and the actual numbers, which is what users ask about.

**B wins** when the corpus is genuinely visual (product photos, real estate, stock imagery, medical imaging), when users search *with* images, or when you need to index millions of images and per-image VLM calls are prohibitive.

**And there is a third thing that isn't either** — late-interaction over page images (ColPali), which we'll get to, and which is increasingly the right answer for document-heavy corpora.

**🗣 Say this in the room:** "Two bridges: caption at index time and keep everything text, or use a joint embedding space and search vectors directly. For enterprise documents I default to captioning, because most 'images' are charts and screenshots — text rendered as pixels — and a 768-dimensional vector can't hold the numbers on a chart while a caption can. Joint embeddings win for genuinely visual corpora and for image-as-query."

### Teach me how CLIP is trained, from scratch. Write the loss.

CLIP is the reason a text query can retrieve an image at all, and the training objective is simpler than people expect.

**📄 Paper:** Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*. It replaced supervised ImageNet-style classification pretraining — which needs a fixed label set and human annotation — with contrastive learning over ~400M noisy (image, alt-text) pairs scraped from the web. That's the contribution: **natural language as a supervision signal, which gives you an open vocabulary for free.**

**Mechanism.** Two encoders — an image encoder (ViT or ResNet) and a text encoder (a transformer) — each projecting to a shared d-dimensional space (d = 512 or 768). Take a batch of N matched pairs. Embed all N images and all N texts, L2-normalize both. Compute the N×N matrix of cosine similarities, scaled by a learned temperature. The N diagonal entries are the true pairs; all N² − N off-diagonal entries are negatives constructed for free by pairing image *i* with text *j*. Then it's just cross-entropy in both directions: each image should pick its own caption out of the N texts, and each text should pick its own image.

```python
import torch, torch.nn.functional as F

def clip_loss(img_emb, txt_emb, logit_scale):
    # img_emb, txt_emb: [N, d];  logit_scale = exp(learned scalar), clamped <= 100
    i = F.normalize(img_emb, dim=-1)
    t = F.normalize(txt_emb, dim=-1)
    logits = logit_scale * i @ t.T           # [N, N] cosine sims
    labels = torch.arange(len(i), device=i.device)
    return 0.5 * (F.cross_entropy(logits, labels) +      # image -> text
                  F.cross_entropy(logits.T, labels))     # text -> image
```

That is the whole thing — about ten lines, and you should be able to write it unaided.

**📐 Numbers you must know:** CLIP was trained with a batch size of **32,768**, and that is not incidental. The number of negatives per example is N − 1, so contrastive quality scales with batch size; at N = 256 the task is easy and the embeddings are weak, at N = 32k it's genuinely discriminative. This is the same reason embedding-model training needs enormous batches, and it is why the softmax over an N×N matrix — memory O(N²) — becomes the engineering constraint that SigLIP later attacks.

**Why the temperature is learned, not fixed:** it controls the sharpness of the softmax, i.e. how hard the model pushes negatives apart. Fixed too low and gradients vanish; too high and the model over-separates on noise. CLIP parameterizes it as `exp(τ)` with τ learned, clamped to prevent blowup.

**What this buys at inference:** zero-shot classification (embed the class names as "a photo of a {class}", pick the nearest), and — the thing we care about — **text-to-image retrieval by cosine similarity in a single index.**

**⚠ Trap:** assuming CLIP embeddings are good at reading text inside images. They are not, in the sense that matters here. CLIP learned that images containing certain visual text-shapes co-occur with certain captions, so it has some OCR-ish capability, but it will not let you retrieve a page by a serial number printed on it. That limitation is precisely what ColPali exists to fix.

### What did SigLIP change, and does it matter for someone building a retrieval system rather than training one?

**📄 Paper:** Zhai et al. (2023), *Sigmoid Loss for Language Image Pre-Training* (SigLIP). The change is one line of the loss: replace the softmax-over-the-batch with an **independent sigmoid on every pair**.

Under CLIP's loss, the score for pair (i, j) is normalized across the whole row — computing the loss requires the full N×N similarity matrix and a global normalization, which means every device in a distributed run needs the batch's full set of embeddings gathered together. Under SigLIP, each of the N² pairs gets a binary label (1 for the diagonal, 0 elsewhere) and its own logistic loss with a learned temperature and bias:

```python
def siglip_loss(img_emb, txt_emb, logit_scale, logit_bias):
    i = F.normalize(img_emb, dim=-1)
    t = F.normalize(txt_emb, dim=-1)
    logits = logit_scale * i @ t.T + logit_bias        # [N, N]
    labels = 2 * torch.eye(len(i), device=i.device) - 1   # +1 diagonal, -1 elsewhere
    return -F.logsigmoid(labels * logits).mean()
```

Two consequences. **No global normalization** means the loss decomposes over pairs, so you can shard the similarity matrix across devices and only exchange chunks — memory stops being O(N²) on one device, and very large effective batches become tractable. And **it works much better at small batch sizes**, because a sigmoid loss doesn't depend on having many in-batch negatives to be meaningful. The bias term exists because with N² pairs and only N positives, the labels are wildly imbalanced; the learned bias absorbs that prior.

**Does it matter if you're not training?** Indirectly but concretely, in three ways:

1. **Better checkpoints for the same compute**, which is why SigLIP and SigLIP-2 backbones show up inside VLMs you'll actually use — the vision tower of many open multimodal models is a SigLIP encoder. When you pick a VLM for captioning, you're often picking a SigLIP-derived encoder without knowing it.
2. **If you do fine-tune an image-text retriever on domain pairs** — which is a very reasonable move for a specialized corpus, exactly as fine-tuning a text embedder is — the sigmoid loss means you don't need a 32k batch and 8 GPUs to get a usable signal. That changes the feasibility of the project on one machine.
3. It's a clean example of a pattern worth naming out loud: **an objective change that buys a systems property (shardability) rather than raw accuracy.** Interviewers like candidates who read papers for the engineering consequence.

**⚠ Trap:** treating "SigLIP > CLIP" as a benchmark claim you can assert. The reported wins depend on the compute regime and the eval; the durable, defensible statement is about the loss decomposing over pairs and behaving better at small batch. **📅 Volatile:** specific benchmark deltas — verify before quoting.

### Walk me through the caption-and-index approach concretely. What do you generate, what does it cost, and where does it fail?

The mechanism is a single ingestion-time decision — **what text do you write for this image?** — and the quality of the whole system rides on it.

**Do not generate a generic caption.** "A bar chart with blue bars" is retrievable by nobody. Generate a *structured, dense* description with a prompt that forces the specifics out:

> Describe this figure for a search index. Include: (1) the figure type; (2) the title and all axis labels with units; (3) every legend entry; (4) the key data values and the trend, with numbers; (5) any text visible in the image, transcribed verbatim; (6) what question this figure would answer. Be specific. Do not speculate about content you cannot see.

Then **index a composite document**, not just the caption: the caption, plus the figure's own caption text from the document ("Figure 3: Quarterly revenue by region"), plus the surrounding paragraph, plus the section heading path, plus any OCR'd text. Attach the image ID, page number, and bounding box as metadata. Now BM25 hits the part number that appears in the transcription, dense retrieval hits the semantic description, and your existing hybrid+rerank stack works with zero changes.

**Critically: pass the surrounding document context *into* the captioning call.** A VLM looking at a chart in isolation writes "a line chart showing an upward trend"; the same VLM given the section text writes "quarterly ARR growth for the EMEA enterprise segment, rising from $4.2M in Q1 to $7.8M in Q4." The second is retrievable. This is the single highest-leverage change to a captioning pipeline and it costs a few hundred extra input tokens.

**💰 Math for 200,000 figures.** Image token cost depends on resolution and provider — as a working model, Anthropic documents an estimate of roughly `(width × height) / 750` tokens per image, so a 1,100 × 850 page figure is about 1,250 tokens; providers cap the long edge (historically ~1,568 px, with newer high-resolution tiers going higher and costing proportionally more, up to several thousand tokens per image). **📅 Volatile — verify the current formula and caps before quoting.** Take ~1,300 image tokens + 500 context tokens in, 350 tokens out, on a mid-tier model at $3/$15 per Mtok:

- Input: 200,000 × 1,800 = 360M tokens → 360 × $3 = **$1,080**
- Output: 200,000 × 350 = 70M tokens → 70 × $15 = **$1,050**
- **Total ≈ $2,130**, halved to **~$1,065** on a 50% batch tier since captioning is entirely offline and latency-insensitive.

That's a real but very payable one-time cost, and it's the number to have ready. The recurring cost is what matters: at 2,000 new figures/day it's ~$21/day, ~$640/month.

**🔍 Where it fails:**
- **Dense tables rendered as images.** A caption cannot reproduce a 40-row table without becoming a 3,000-token document, and if you let it, your index fills with one enormous document that matches everything. Detect table-like figures and route them to a table-extraction path instead.
- **Fine detail below the model's effective resolution.** Small axis tick labels, dense schematics, engineering drawings. The caption confidently omits what it couldn't read.
- **Hallucinated specifics.** A VLM asked for "key data values" will sometimes invent them. Mitigate by prompting for verbatim-only transcription of text, and by spot-auditing: sample 100 captions, have a human check the numbers, and measure the fabrication rate. If it's above a couple of percent, drop the "key data values" instruction and keep only transcription.
- **Model drift on re-caption.** Upgrading the captioner changes every description, which changes every embedding — a full reindex, exactly like changing an embedding model.

### Here's a bug report: "the answer cites the right page but misses the chart that actually contains the number." What's the root cause pattern, and how do you prevent it?

**⚠ Trap:** this is the *images referenced by nearby text but never embedded* failure, and it is the most common silent defect in document RAG pipelines. The text says "as shown in Figure 4, EMEA revenue declined 12% in Q3." That sentence embeds beautifully, retrieves for a question about EMEA revenue, and lands in the context window. **Figure 4 is not in the context window** — it was dropped by the parser, or extracted to a file nobody indexed, or indexed as a separate document that didn't score highly enough to make top-k. The model sees a sentence that promises a figure and no figure, and it does one of three things: says the information isn't available (annoying), answers from the sentence alone (often fine, sometimes incomplete), or **fills in plausible numbers from the surrounding text** (an incident).

Three fixes, in increasing order of robustness:

**1. Bidirectional linkage at ingestion.** Every figure gets an ID; every text chunk records the figure IDs it references (detect `Figure N`, `Table N`, `see below`, image anchors in the source format); every figure records the chunk IDs that reference it. Store this as a real relation, not as a hope.

**2. Retrieval-time co-retrieval.** After ranking, expand the result set: for every retrieved chunk, pull in the figures it references; for every retrieved figure, pull in its referencing chunks and its own caption text. Do this *before* the context-packing budget is spent, and give co-retrieved figures a reserved slice of the budget so they can't be crowded out by higher-scoring text. This is the same parent-document/small-to-big pattern applied across modalities.

**3. Don't separate them in the first place.** For layout-heavy documents, chunk by *page region* and keep a figure with its caption and the paragraph that references it as one composite unit — or go all the way to page-image retrieval (ColPali), where the question doesn't arise because the page is the unit.

**The detection you should build regardless**, because this fails silently: **an ingestion-time assertion that every `Figure N` / `Table N` reference in the text resolves to an indexed asset.** Count unresolved references per document and alert when the rate crosses a threshold. In my experience this check finds problems in every pipeline it's first run against — a parser that dropped SVGs, a converter that inlined images as base64 the chunker then stripped, a scanned appendix nobody noticed. It costs an hour to write and it's the difference between "5% of our documents are quietly broken" and knowing which 5%.

**🗣 Say this in the room:** "This is the reference-without-referent bug. Text that mentions a figure retrieves well; the figure itself never made it into the index or the context. I fix it with bidirectional figure↔chunk links, co-retrieval with a reserved context budget for figures, and — most importantly — an ingestion assertion that every 'Figure N' reference resolves. That check finds broken documents in every pipeline I've run it on."

### Explain ColPali. What is it doing that a caption pipeline isn't, and what does it cost you in storage?

The mental model: **ColPali deletes the parsing stage.** Every other document-RAG pipeline converts pixels to text — OCR, layout analysis, table extraction, captioning — and every one of those stages loses information and introduces errors that then poison the index. ColPali's bet is that a vision-language model can embed the page *as an image* well enough to retrieve on directly, so the lossy text-extraction pipeline simply doesn't exist.

**📄 Paper:** Faysse et al. (2024), *ColPali: Efficient Document Retrieval with Vision Language Models*. It fine-tunes a VLM (PaliGemma in the original) to produce **multi-vector page embeddings** and scores them against multi-vector query embeddings with ColBERT-style late interaction. They also introduced **ViDoRe**, a visual document retrieval benchmark, because none of the existing text benchmarks measured this. The name you'll also hear is **ColQwen**, which is the same recipe with a Qwen-VL backbone instead of PaliGemma — treat "ColPali" as the method and the backbone as a swappable component, because that's how the family evolves.

**Mechanism, precisely.** Render each page to an image. The vision encoder splits it into a grid of patches, producing roughly 1,000 patch embeddings per page; a projection maps each to a low dimension (128 in the original). So a page is **~1,030 vectors of 128 dims**, not one vector. The query is tokenized and embedded into ~20 token vectors in the same space. Scoring is **MaxSim**, inherited from ColBERT: for each query token, take the maximum similarity against any page patch, then sum over query tokens.

**📄 Paper:** Khattab & Zaharia (2020), *ColBERT* — the origin of late interaction: keep per-token embeddings instead of pooling to one vector, and defer the interaction between query and document to a cheap MaxSim at scoring time. ColPali's contribution is applying it across modalities, with image patches playing the role of document tokens.

Why this works so well on documents: a single pooled vector must compress an entire page into 768 numbers, which destroys exactly the specific details (a part number, a value in a table cell, an axis label) that queries ask about. **Late interaction preserves locality** — one query token can match one patch of one table cell, and that match survives to the score. Meanwhile the vision encoder sees layout, so tables, figures, and reading order are handled implicitly rather than by a fragile parsing stage.

**💰 Storage math, which is the whole objection.** Per page: 1,030 vectors × 128 dims × 2 bytes (fp16) = **~264 KB/page**. Against single-vector: 768 dims × 4 bytes = 3 KB per chunk, maybe 3 chunks/page = **9 KB/page**. That's roughly **30× the storage**.

For **1 million pages**: 264 KB × 10⁶ = **264 GB** of vectors, versus ~9 GB single-vector. On memory-resident ANN at commodity cloud prices that's the difference between one machine and a small fleet — call it a few hundred dollars a month of RAM versus tens. Mitigations exist and matter: binary or int8 quantization on the patch vectors (4–8× reduction with modest recall loss, then rescore top candidates in full precision), patch pooling to cut the vector count per page, and PLAID-style centroid-based pruning from the ColBERT line of work so you don't MaxSim against every page.

**Latency** is the other cost: MaxSim over 1,030 × 20 similarities per candidate page is far more work than a dot product, so you need a two-stage design — cheap candidate generation, then late-interaction rescoring of the top few hundred.

**🗣 Say this in the room:** "ColPali is ColBERT late interaction with image patches as the document tokens — it skips OCR and layout parsing entirely by embedding the rendered page. You get big recall wins on tables, figures, and messy scans, and you pay roughly 30× the vector storage, about 264 KB per page at 128-dim fp16, plus a rescoring stage. I'd quantize the patch vectors and use it as a second-stage reranker rather than a flat index."

### When does ColPali lose? Give me the decision rule.

It loses in four specific places, and being able to name them is what separates having read the paper from having deployed it.

**1. Ordinary text documents.** If your corpus is markdown, HTML, or clean PDFs with a good text layer, you already have perfect text extraction. ColPali is solving a problem you don't have, at 30× the storage and a much slower scoring path. Plain hybrid retrieval on the extracted text wins on every axis.

**2. Cost and latency at scale.** Both the 264 KB/page storage and the MaxSim scoring cost. On a 50M-page corpus, 264 KB × 5×10⁷ = **13 TB** of vectors before quantization. That's an infrastructure project, not a config change.

**3. Generation, as opposed to retrieval.** ColPali retrieves *page images*. To answer, you now feed page images to a VLM, which costs image tokens (roughly `w×h/750` on the order of 1,000–4,800 tokens per page depending on resolution tier — **📅 Volatile**) versus a few hundred tokens for the extracted text of the same page. So a 5-page context that would have cost ~2,500 text tokens costs ~7,000+ image tokens. And **citation gets harder**: you can cite "page 14" but not "the sentence beginning 'net revenue…'", unless you run OCR anyway for the answer path.

**4. Exact lexical matching.** Late interaction is still semantic similarity between embeddings. A query for the literal string `MX-4471-B` is served better by an inverted index than by patch similarity. Keep BM25 in the mix.

**The decision rule I'd state:**

- **Text layer is clean and complete → don't use ColPali.** Extract, chunk, hybrid-retrieve.
- **Documents are scanned, or layout-critical (forms, financial statements, engineering drawings, slide decks), and OCR quality is measurably poor → ColPali is likely the biggest single retrieval win available**, because it routes around the failing stage rather than trying to fix it.
- **Mixed corpus → route per document at ingestion.** Detect whether a PDF has a usable text layer (extract text, compute characters-per-page; below a threshold it's a scan). Text-layer documents go the classic route; scans and layout-heavy documents go the visual route. Two indexes, fused at query time with RRF.
- **Any scale → treat it as a reranker.** Cheap first stage (BM25 over whatever OCR you have, plus single-vector dense over page-level captions) to get 200 candidates, then late-interaction rescoring to get the top 10. This bounds both storage-in-hot-tier and scoring cost.

**⚠ Trap:** running a bake-off on a clean-PDF benchmark, seeing ColPali roughly tie the text pipeline, and concluding it isn't worth it. Its advantage is concentrated on exactly the documents your text pipeline is failing on — so evaluate on your *hard* slice (scans, tables, forms), not on your average document, or you will measure the wrong thing and make the wrong call.

### Our OCR pipeline is 96% accurate and we thought that was fine. Explain how OCR errors poison the index, and how you'd detect and mitigate.

**⚠ Trap:** "96% character accuracy" sounds excellent and is a nearly useless number. Errors are not uniformly distributed over characters; they concentrate in exactly the tokens that carry retrieval weight — proper nouns, part numbers, dates, and figures in tables, because those have no linguistic redundancy for the OCR model's language prior to correct. At 96% character accuracy, a 12-character part number has a `0.96¹² ≈ 61%` chance of being fully correct, meaning **roughly 4 in 10 part numbers in your corpus are wrong**. And the wrongness is systematic: `0`↔`O`, `1`↔`l`↔`I`, `5`↔`S`, `rn`→`m`.

**How the poisoning propagates:**
- **Lexical retrieval dies exactly where you need it.** BM25 on `MX-4471` will never match indexed `MX-447l`. The user searches the number printed on the part in their hand and gets nothing.
- **Dense retrieval degrades quietly.** A corrupted token tokenizes into different subwords, shifting the chunk embedding slightly. The chunk doesn't disappear from results; it just ranks a bit lower, forever, on every query. You will never see this in a bug report.
- **The generator quotes the corruption.** The model faithfully reports the number in its context, so a user is told the part number is `MX-447l`. Grounded, cited, and wrong — the worst combination, because your faithfulness metrics say it's fine.
- **Chunking gets corrupted too.** OCR that garbles a heading breaks structure-aware splitting, and reading-order errors on multi-column pages interleave two columns into nonsense that embeds as nonsense.

**Detection — build these, they're cheap:**
1. **Per-page confidence from the OCR engine.** Tesseract and PaddleOCR both emit per-word confidence. Compute mean and the fraction of words below a threshold; quarantine pages in the bad tail rather than indexing them silently.
2. **Dictionary/perplexity sanity check.** Fraction of tokens that are not in a dictionary and not pattern-matched as identifiers. A page at 30% out-of-vocabulary is broken.
3. **Pattern validators for structured fields.** If part numbers match `^[A-Z]{2}-\d{4}(-[A-Z])?$`, run the regex over extracted text and flag near-misses (strings that match with one character substituted from the confusable set). This is a targeted, high-precision corruption detector.
4. **Round-trip check on a sample.** Take 200 pages, run a second, independent engine (or a VLM), diff. The disagreement rate is your real error estimate — far more honest than a vendor's benchmark number.
5. **Zero-result query monitoring in production.** A spike in queries returning nothing, especially queries that look like identifiers, is the operational signature of OCR corruption.

**Mitigation, in order of leverage:**
- **Use a VLM for parsing instead of classical OCR on hard layouts.** This is the 2026 default for anything with tables, multi-column layout, or poor scan quality, and it typically moves the error rate by a lot more than tuning Tesseract will.
- **Index both raw OCR and a normalized/corrected variant**, so a query can hit either.
- **Fuzzy lexical matching for identifier-shaped tokens** — index n-grams or a confusable-normalized form (map `0/O`, `1/l/I`, `5/S` to a canonical character) as a parallel field, so `MX-447l` and `MX-4471` collide.
- **Bypass OCR entirely for the worst slice** via page-image retrieval, which is precisely the argument for ColPali.
- **Quarantine and human-review** the bottom decile by confidence, rather than indexing it. A document that isn't indexed is a known gap; a document indexed as garbage is an unknown one.

**💰 The number that motivates the budget:** if 4% of your corpus is corrupted in ways that make it unretrievable, and your corpus supports a support team answering 50,000 tickets a year, that's 2,000 tickets a year where the answer existed and wasn't found. Price a support ticket at $8 of handling time and you have $16,000/year of value against a re-parsing job that costs a few thousand dollars of VLM inference.

### Chart and figure QA — what actually works, and what do you tell a stakeholder who wants "ask questions about our dashboards"?

The mental model: **there are three distinct things people mean by "chart QA," and they need three different systems.** Getting the requirement precise is most of the work.

**Type 1 — "which chart shows X?"** This is retrieval, and it's solved by the caption pipeline: dense structured descriptions plus the figure's own caption plus surrounding text, hybrid-indexed. Works well, cheap, and it's what most stakeholders actually want once you probe.

**Type 2 — "read a value off this chart."** The model is given the chart image and asked "what was Q3 EMEA revenue?" Modern VLMs are genuinely decent at this when the value is directly labeled, and unreliable when it must be *interpolated from the axis* — reading a bar's height against gridlines is a precise spatial-measurement task, and the failure is a confidently-stated wrong number. Rough guidance: labeled values, trust with verification; unlabeled values, treat as an estimate and say so in the output.

**Type 3 — "compute something across charts."** "What's the combined growth rate across these four regional charts?" Compounds Type-2 error four times and then does arithmetic on it. Don't build this on pixels.

**What actually works, in priority order:**

1. **Go to the source data.** The chart was rendered from a table, a query, or a dashboard definition. If you can reach that — the underlying CSV, the BI tool's API, the notebook that produced it — do that instead, and you're back to text-to-SQL or table QA, where the answer is *computed* rather than *perceived*. **This is the right answer and almost nobody asks the question.** The instinct to solve it visually is a symptom of not having asked where the pixels came from.
2. **Extract the data table from the chart at ingestion**, once, with a VLM prompted to emit structured data (`{series, x, y}` records) rather than prose. Store it alongside the image. Now Type-2 and Type-3 questions run against a table with code execution, and only the extraction is perceptual — and you can audit extraction quality offline, on a sample, instead of hoping per-query.
3. **Give the model tools to look closer.** Crop-and-zoom on a region, then re-read. Dense charts often fail simply because the tick labels are below the effective resolution after the provider downscales the image; a cropped re-read at native resolution fixes a meaningful share of failures.
4. **Show the chart alongside the answer.** Same principle as showing the SQL: make the error checkable by the human who cares.

**What I'd tell the stakeholder:** "Yes, and the accurate version of this reads the numbers behind the dashboard, not the picture of the dashboard. Point me at the BI tool's API or the underlying tables and I'll get you exact answers with citations. If we have to work from images — because they're in PDFs we don't own — I'll extract structured data at ingestion and be explicit in the UI about which numbers were read off a chart versus computed from data."

**⚠ Trap:** demoing Type-2 on three well-labeled charts, declaring success, and shipping. Build an eval set of 100 real charts from your corpus with ground-truth values, stratified by whether values are labeled, and measure. Expect the unlabeled slice to be far worse than the labeled slice, and expect the stakeholder's mental model to be calibrated entirely on the labeled slice.
