### Why is extracting text from a PDF hard? It's a text format, isn't it?

No, and this misconception is worth killing precisely because it drives bad architecture decisions. **A PDF is a page-description program, not a document.** It is a stream of drawing instructions: set the font, move the text cursor to (x=72.0, y=618.4), show these glyphs, move to (x=310.5, y=618.4), show these glyphs. There is no concept of a paragraph, a sentence, a column, a table, a heading, or a reading order in the file format. Those are things a human infers from spatial arrangement. A PDF is closer to an SVG than to HTML.

Everything painful follows from that one fact:

**There is no reading order.** The content stream order is whatever the producing application emitted, which for a two-column layout might be all of column one then all of column two, or might interleave them, or might be alphabetical by font. Extraction libraries that concatenate in stream order will produce interleaved gibberish on some documents and correct output on others, from the same generator, depending on version.

**There are no words or spaces, necessarily.** Many producers emit individual glyph-positioning operations with no space characters at all; the space between words is a horizontal displacement. Extractors reconstruct word boundaries by thresholding the gap between glyph bounding boxes against the font's advance width. Get the threshold wrong and you get `thequickbrown` or `t h e q u i c k`. Both happen in real corpora.

**There are no tables.** A table is some text and possibly some line-drawing operations. Recovering rows and columns means clustering glyph positions into a grid — and borderless tables have literally nothing but whitespace alignment to go on.

**The characters may not be characters.** A PDF maps glyph indices to Unicode via a `ToUnicode` CMap, which is optional. Subset-embedded fonts without a CMap extract as mojibake or as the glyph indices themselves. Ligatures (ﬁ, ﬄ) extract as single codepoints that break your search for "file". Some producers deliberately scramble the mapping.

**Text may not exist at all** — a scanned page is a single JPEG.

**🗣 Say this in the room:** "PDF is a page-description language, so it has coordinates and glyphs but no paragraphs, no reading order, and no tables — those are inferred by the extractor, not read from the file. That's why PDF ingestion is a computer-vision-flavored problem rather than a parsing problem, and why the parser choice is one of the two or three decisions that actually determine RAG quality."

**⚠ Trap:** benchmarking your parser on documents your own team produced. Word-exported and LaTeX-produced PDFs are the clean, well-behaved end of the distribution and every extractor handles them. The corpus that will break you is the one with 1990s scans, faxes of forms, third-party reports with three-column layouts, and Excel print-to-PDF with 60-column tables. **Sample your parser evaluation from the actual production corpus, stratified by source system and by year**, or your parser choice is uninformed.

### How do you tell programmatically whether a PDF has a real text layer, is scanned, or is a mix? And why does it matter?

It matters because it determines cost and pipeline path by a factor of a hundred. Extracting a text layer is microseconds and free. OCR-ing or VLM-parsing a page is hundreds of milliseconds to seconds and costs money. Routing every page through the expensive path because 8% of your corpus is scanned is a real budget error; routing scanned pages to the cheap path silently produces empty chunks, which is a real quality error.

**The detection heuristic, per page** — and per page is the point, because mixed documents are extremely common (a born-digital report with a scanned signature page, or a scanned document with a digitally-added cover sheet):

```python
def classify_page(page):                      # PyMuPDF-style pseudocode
    text = page.get_text().strip()
    n_chars = len(text)
    area = page.rect.width * page.rect.height
    images = page.get_images(full=True)
    img_cov = sum(bbox_area(page, x) for x in images) / area

    if n_chars > 200 and img_cov < 0.5:
        return "digital"
    if n_chars < 50 and img_cov > 0.6:
        return "scanned"
    if n_chars > 200 and img_cov > 0.6:
        return "digital_over_image"   # OCR'd already, or text over a scan
    return "sparse"                   # figure page, cover, or a broken text layer
```

Then add the checks that catch the nastier cases:

**Garbled-text detection.** A page can have a text layer that is *wrong* — bad CMap, scrambled encoding. Signal: compute the fraction of extracted characters that are outside the expected script's range, or the fraction of extracted "words" not in a dictionary. If more than ~30% of tokens are non-words, treat the text layer as untrustworthy and route to OCR even though `n_chars` is high. This one has burned me: a whole vendor's document family extracted as plausible-looking Latin-1 garbage, embedded fine, retrieved never, and nobody noticed for a month because the pipeline reported success.

**Pre-existing OCR quality.** `digital_over_image` means someone already OCR'd it, and their OCR may be from 2009 and terrible. Sample the confidence proxy — character-error signals like `l`/`1`/`I` confusion rates, or just word-dictionary hit rate — and re-OCR if it is below threshold. Modern engines are dramatically better than what most archives contain.

**Chars per page as a distribution, not a threshold.** Plot the histogram of `n_chars` across your whole corpus. You will see a bimodal shape: a spike near zero (scanned) and a broad hump around 1,500–3,500 (digital). Set your threshold in the valley, from your data, not from my number.

**💰 Math on why routing matters:** 5M pages, 8% scanned. All-VLM: 5M × $0.0055/page (derived later in this section) = **$27,500**. Routed: 400k scanned pages × $0.0055 = $2,200, plus 4.6M digital pages at essentially $0 for extraction = **$2,200**. **A ten-line classifier saves $25,300 on a single backfill**, and the same ratio every time you reprocess.

**⚠ Trap:** classifying at the document level. One scanned appendix in a 300-page digital report means either you OCR 300 pages needlessly or you lose the appendix entirely. **Route per page.** This is the single most common architectural mistake in document pipelines and it is invisible until someone asks about the appendix.

### Walk me through the PDF parser landscape. Which one do you pick and how do you decide?

I would group them into four tiers by what they actually do, because "which library is best" has no answer without the corpus.

**Tier 1 — raw text extractors.** `PyMuPDF` (the `fitz` binding to MuPDF) is the fast one: it will do tens to low hundreds of pages per second per core, gives you text with bounding boxes, images, and links, and is what I use for the classification pass above regardless of what does the real parsing. **License trap, and it is a real one: PyMuPDF is AGPL-3.0, with a commercial license from Artifex if you cannot comply.** In a company with a legal review process this can disqualify it outright, and knowing that unprompted is a credibility marker. The permissive alternative is `pypdfium2`, a binding to Google's PDFium (BSD-3), which is nearly as fast, and `pdfminer.six` (MIT), which is slow but pure Python and very tweakable. `pdfplumber` sits on pdfminer.six and adds genuinely good word/line/table geometry — it is the tool I reach for when I need to write custom table logic against a known document family, and it is far too slow for a million-page backfill.

**Tier 2 — layout-analysis pipelines.** These run a document-layout model to segment the page into typed regions (title, paragraph, list, table, figure, header, footer), recover reading order, and emit structured output. `Docling` (IBM, with a published technical report) and `Unstructured` are the two open frameworks people reach for; `Marker` converts PDFs to markdown using the `Surya` model family for layout/OCR/reading-order; `MinerU` is another open pipeline in this space. This tier is where you go when you need tables and figures as first-class objects rather than as text soup, which for RAG is nearly always.

**Tier 3 — hosted document-understanding APIs.** `Azure AI Document Intelligence` (formerly Form Recognizer) is the enterprise incumbent, with a general Layout model plus trainable custom-extraction models, and it is genuinely good at tables and key-value pairs on forms; Google Document AI and AWS Textract occupy the same slot. `LlamaParse` is the RAG-native hosted option. You choose this tier when you would rather buy accuracy than own a GPU pipeline, and when your legal posture allows sending documents to a third party — which, for the enterprise corpora that most need this tier, is frequently the blocker.

**Tier 4 — VLMs used directly as parsers.** Render the page to an image, hand it to a vision model, ask for markdown. Covered in its own question below.

**How I decide, as a procedure:**
1. Sample 100 pages stratified across source systems and years.
2. Hand-label ground truth for the things that matter to *your* queries — usually: is the reading order right, are the tables' cells correct, did the figures get captured, are headers stripped.
3. Run three or four candidates and score them on those axes, not on a generic benchmark.
4. **Then price it at your volume**, because the accuracy ordering is usually Tier 3/4 > Tier 2 > Tier 1 and the cost ordering is exactly the same, so the decision is where the accuracy curve flattens relative to your query needs.
5. Almost always the answer is a **hybrid router**: Tier 1 for clean digital pages (the bulk), Tier 2 or 4 for pages the classifier flags as complex — heavy table density, multi-column, scanned.

**📅 Volatile:** this landscape moves fast — capabilities, pricing and even project maintenance status change on a quarterly cadence. Verify what is current before your loop rather than quoting this list as of a date.

**🗣 Say this in the room:** "I wouldn't name a parser without seeing the corpus. What I'd commit to is the method: stratified 100-page sample, hand-labeled on reading order and table cell accuracy, three candidates scored on that, then a per-page router so I'm only paying for the expensive parser on the pages that need it. And I'd flag PyMuPDF's AGPL licensing before it becomes a legal surprise."

### A two-column academic paper came out as interleaved nonsense. What is reading-order recovery and how does it work?

The symptom — "The transformer architecture Our results show that consists of stacked the model converges" — is the extractor concatenating in content-stream or naive top-to-bottom order across a multi-column layout. Two lines that are vertically adjacent belong to different columns and different thoughts.

**The classical algorithm is the XY-cut** (recursive page segmentation), and it is worth being able to describe because it is simple and still the backbone of several pipelines. Project all text bounding boxes onto the vertical axis and find the horizontal whitespace bands; project onto the horizontal axis and find the vertical whitespace gutters. Cut on the widest gap. Recurse on each resulting region. For a two-column page, the widest gap is the vertical gutter between columns, so you cut there first, then recurse into each column and cut horizontally between paragraphs. The output is a tree; an in-order traversal gives you reading order. It fails on anything that spans the cut — a full-width figure or a table between columns creates a region that cannot be split cleanly, and the recursion produces the wrong order.

**The modern approach is a learned one.** A layout model (a detection model over the rendered page, or a multimodal encoder over text+position like the LayoutLM family) predicts region types and a reading-order sequence directly. This handles spanning elements, sidebars, callout boxes, footnotes and rotated text far better than geometry alone, and it is the reason Tier-2 pipelines beat Tier-1 extractors on real documents.

**How it still fails, as a taxonomy you can name:**
- **Spanning elements** — a full-width figure caption between two columns gets attached to the wrong column, or duplicated.
- **Footnotes** — pulled inline mid-sentence. Footnote markers become stray digits inside your chunk text (`the result was significant3 across`), which corrupts both the embedding and any numeric parsing downstream.
- **Marginalia and sidebars** — interleaved into body text.
- **Rotated pages** — landscape tables inside a portrait document. Detect via page rotation metadata *and* via text baseline angle, because the two disagree often.
- **Slides converted to PDF** — z-order is arbitrary and there is no meaningful reading order at all; treat each slide as an unordered bag with the title promoted, and stop trying.

**⚠ Trap:** you cannot detect this failure by looking at extraction success rates. The extractor succeeded. It produced text. The text is grammatically local and globally scrambled, it embeds to something plausible, and it retrieves occasionally. **The detection that works is a language-model perplexity check on a sample:** score extracted page text under a small LM and flag pages in the worst decile. Interleaved columns have wildly elevated perplexity because every ~10 tokens the sentence changes topic mid-clause. This is a cheap, general, corpus-agnostic quality gate and I put it in every ingestion pipeline I build. It also catches encoding corruption, OCR garbage, and header pollution in the same pass.

**🗣 Say this in the room:** "Reading order is inferred, not stored, so multi-column layouts are the classic failure. Geometric XY-cut handles simple cases; learned layout models handle spanning figures and sidebars. The important part is that this failure is silent — extraction 'succeeds' — so I gate it with a perplexity check on sampled pages and alert on the tail, which catches interleaving, encoding corruption and OCR garbage with one metric."

### Design layout-aware chunking for a PDF where tables and figures are first-class objects.

The architecture I would draw is a **two-stage pipeline with a typed intermediate representation**, and insisting on that intermediate representation is the whole design.

**Stage 1 — parse to a typed element stream.** Every page becomes an ordered list of elements, each with a type, its text or structured content, a bounding box, a page number, and a heading path derived from the running title/heading elements:

```python
Element = {
  "type": "title|heading|paragraph|list|table|figure|caption|header|footer|footnote|formula",
  "level": int | None,          # for headings
  "text": str | None,           # for text-ish types
  "table": {"header": [...], "rows": [[...]], "n_rows": int} | None,
  "image_ref": str | None,      # object store key for figures
  "bbox": (x0, y0, x1, y1),
  "page": int,
  "path": ["5. Hydraulics", "5.3 Service"],
}
```

**Stage 2 — assemble chunks from elements under explicit rules.** The rules, which I would write down in the design doc because they are the actual product decisions:

1. **Never split a table or a figure.** They are atomic. A table larger than the budget splits by rows with the header repeated and a `rows 41–80 of 214` marker; a figure never splits.
2. **Never split across a heading boundary.** A heading starts a new chunk, always.
3. **A caption belongs to its object.** Captions bind by proximity and by the "Figure N"/"Table N" reference; they are emitted as part of the object's chunk, never as a standalone paragraph. An orphaned caption chunk that says "Figure 7: Quarterly revenue by region" is a beautifully retrievable chunk containing no information.
4. **Headers and footers are dropped from text and promoted to metadata.**
5. **Footnotes are appended at the end of the chunk containing their reference marker**, not inline.
6. **Text elements pack greedily up to the budget** within a heading section, exactly as in the recursive splitter.
7. **A table or figure chunk carries its surrounding narrative** — the paragraph before and after — because "as Table 3 shows, margins compressed" is the sentence that makes Table 3 retrievable.

**For figures specifically, the chunk record is:** the caption, an LLM/VLM-generated description of the image (100–200 tokens: what kind of chart, what the axes are, the trend, any labeled values), the referencing sentences from the body, the object-store key for the image bytes, and the page + bbox for citation. **You embed the caption + description + referencing text.** The image itself is not in the vector index unless you are running a genuinely multimodal retriever; the generated description is the retrieval handle.

**⚠ Trap — the one this question is really testing:** images that are referenced in text but never ingested. The body says "as shown in Figure 4," Figure 4 is a chart with the actual numbers, your pipeline extracted text only, and now your index contains a pointer to information that does not exist in it. The model retrieves the referencing sentence, sees "Figure 4," and either says "I cannot see the figure" (best case) or invents what it shows (common case). **The detection is a two-line check: count `Figure \d+` / `Table \d+` references in extracted text, count figure/table elements extracted, and alert when the ratio is off.** If a document references 40 figures and you extracted 3, you have a parser problem, and this assertion finds it in every document, automatically, forever.

**💰 Math on figure descriptions:** 5M pages with ~0.4 figures/page = 2M figures. A VLM describing each at ~1,200 image tokens in and 150 tokens out, at $1/$5 per Mtok **📅 Volatile**: 2M × (1200 × $1/1e6 + 150 × $5/1e6) = 2M × ($0.0012 + $0.00075) = 2M × $0.00195 = **$3,900** one-time. Against a corpus where charts carry the numbers users ask about, that is obviously worth it; against a corpus of decorative stock photography it is $3,900 of nothing. **Sample 50 figures and classify them as informational vs decorative before funding this.**

### OCR engines versus VLM-based parsing — what's your decision rule in 2026?

The honest framing: these are not competitors on a single axis, they are different cost/accuracy points, and the correct system uses both.

**Classical OCR** — Tesseract (the old open standard, LSTM-based since v4, weak on layout, weak on tables, very good on clean scanned prose), PaddleOCR (Baidu; strong detection+recognition, good CJK, includes layout and table modules), and Surya (the model family behind Marker; text detection, recognition, layout and reading order, multilingual) — gives you **characters plus boxes.** It is fast, cheap, runs locally, is deterministic, and gives you per-word confidence scores. It does not understand the document; a separate layout/table stage has to do that.

**VLM parsing** — render the page, prompt a vision model for markdown — gives you **an interpreted document.** It handles multi-column reading order, borderless tables, handwriting, checkboxes, stamps, and mixed-language pages far better than an OCR+layout stack, because it is doing recognition and understanding jointly. The academic ancestor worth citing is **📄 Blecher et al. (2023), Nougat** — an encoder-decoder VLM that transcribed scientific PDFs directly to markup, replacing OCR-plus-layout pipelines for that domain and demonstrating both the promise and the characteristic failure mode (repetition collapse on out-of-distribution pages). Since then, purpose-built open document VLMs have proliferated, and general frontier VLMs became good enough to use directly.

**📅 Volatile:** the specific model to use and its price change every few months; the decision *rule* below is what is durable.

**My decision rule, in order:**
1. **Digital text layer present and trustworthy → extract it.** Free, exact, no hallucination risk. Never OCR a page you can read.
2. **Scanned, clean, single-column prose, high volume → classical OCR.** At millions of pages the cost gap is decisive and the accuracy gap on this page type is small.
3. **Scanned or digital but structurally hard — dense tables, multi-column, forms, mixed handwriting, poor scan quality, non-Latin scripts mixed with Latin → VLM.**
4. **Anything where a wrong number is expensive → VLM plus a validation pass**, or a hosted document-understanding API with confidence scores, plus human review on low confidence.

**💰 The arithmetic that drives this.** Per page:
- Classical OCR self-hosted: dominated by compute. A GPU-accelerated OCR stack does on the order of 5–20 pages/second on one modern GPU. At 10 pages/s and $1.50/hr for the instance: $1.50 / (10 × 3600) = **$0.0000417/page** — about 4 cents per thousand pages.
- VLM: a rendered page at ~150 DPI is roughly 1,000–1,600 image tokens depending on the provider's tiling; call it 1,300. Output markdown for a dense page is ~800 tokens. At $1/$5 per Mtok: (1300 × $1 + 800 × $5)/1e6 = $0.0013 + $0.0040 = **$0.0053/page**, i.e. **$5.30 per thousand pages**.
- **The ratio is roughly 100–130×.** For 5M pages: **$210 versus $26,500.**

**📐 Numbers you must know:** a rendered page at ~150 DPI costs roughly **1,000–1,600 image tokens** on current tiling schemes, and a dense page of markdown output is **~800 tokens** — those two numbers are all you need to price any VLM document pipeline on the spot. Derivation of the page count you will be asked about next: 500 tokens of text per average page, so **1 page ≈ 1 chunk at a 512-token chunk size**, and 1M pages ≈ 1M chunks ≈ 500M tokens.

That ratio is the entire argument for routing. If 10% of your pages are structurally hard, routing costs $0.10 × 5M × $0.0053 + 0.9 × 5M × $0.0000417 = $2,650 + $188 = **$2,838**, which is 89% cheaper than all-VLM and captures nearly all of the accuracy.

**🗣 Say this in the room:** "VLM parsing is roughly two orders of magnitude more expensive per page than self-hosted OCR — call it half a cent versus four thousandths of a cent — so the design is a router, not a choice. Digital text layer costs nothing and I always take it; clean scanned prose goes to OCR; tables, multi-column, forms and handwriting go to a VLM. And VLM output gets a validation pass, because unlike OCR it can hallucinate a plausible number that was never on the page."

### What are the specific failure modes of VLM-based document parsing, and how do you validate the output?

This question separates people who have run a VLM over a million pages from people who ran it over ten. **The defining difference from OCR is that OCR produces garbage when it fails and a VLM produces plausible fiction.** A garbled OCR string is visibly wrong; a hallucinated table cell reads as a perfectly reasonable number. That asymmetry drives everything about validation.

**The failure taxonomy:**

1. **Silent row/column omission.** On a 60-row table the model transcribes 47 rows, correctly, and stops. No error, no marker, valid markdown. This is the most dangerous one because everything about the output looks right.
2. **Repetition collapse.** The decoder falls into a loop and emits the same row or phrase hundreds of times. Nougat documented this and it persists across VLM parsers. Cheap to detect (n-gram repetition rate) and therefore inexcusable to miss.
3. **Cell fabrication.** A cell that is empty, illegible, or a merged-cell artifact gets filled with a plausible value inferred from the row/column pattern. Interpolation presented as transcription.
4. **Instruction leakage / commentary.** "Here is the markdown for the page:" or "I cannot read the bottom-left cell clearly, but it appears to be..." lands in your chunk text and gets embedded.
5. **Refusals and safety stops** on pages with faces, ID documents, medical records, or anything the model's policy layer treats as sensitive. In a corpus of scanned personnel files this can be a double-digit percentage of pages, and it presents as an empty result.
6. **Number normalization drift.** `1,234.56` becomes `1234.56`, `(1,234)` becomes `-1234` or `1234`, `1.234,56` (European) becomes `1.23456`. The last one is a 1000× magnitude error that reads as a perfectly normal number.
7. **Non-determinism.** The same page parsed twice gives slightly different output, so your content hashes churn and your incremental-reprocessing logic sees phantom changes. **Set temperature to 0 and still expect residual nondeterminism; hash the *normalized* output, not the raw string.**

**The validation layer, which is non-negotiable:**

- **Cross-check against cheap OCR.** Run classical OCR on the same page. Compare the *bag of numeric tokens* — every number the OCR found should appear in the VLM output and vice versa. A number in the VLM output that OCR never saw is a fabrication candidate; a number OCR saw that the VLM dropped is an omission. This single check catches failure modes 1 and 3, which are the expensive ones, and OCR costs 1% of the VLM call so it is basically free insurance.
- **Structural assertions on tables.** Every row has the same column count as the header. Row count is consistent with the number of detected horizontal text lines in the table's bbox. No cell is longer than N characters.
- **Repetition check.** Flag if any 8-gram occurs more than ~5 times in a page's output.
- **Length sanity.** Output token count within a plausible band of the page's ink coverage. A dense page producing 40 tokens of output means a refusal or a truncation.
- **Commentary filter.** Reject outputs containing first-person or meta-language.
- **Route failures to quarantine**, do not drop them and do not retry indefinitely.

**⚠ Trap:** treating the VLM parse as ground truth because it "looks clean." Clean markdown output is exactly what a hallucinating VLM produces. **The quality signal is agreement between two independent extractors, not the prettiness of one.**

**💰 Math:** adding the OCR cross-check costs $0.0000417/page against a $0.0053 VLM call — **0.8% overhead** — and catches the failure class that produces wrong numbers in customer-facing answers. There is no version of this trade-off I would decline.

### How do you serialize a table so an LLM can actually read it? Compare the formats and give me the token cost.

Three candidate serializations, and the differences in token cost are large enough that this is a real budget decision, not a stylistic one.

Take a 12-column × 40-row financial table with short cell values.

**Markdown pipe table.**
```
| Region | Q1 | Q2 | Q3 | ... |
|---|---|---|---|---|
| EMEA | 4,182 | 4,410 | 3,905 | ... |
```
Column headers appear once. Per row you pay the cells plus `n_cols + 1` pipe delimiters. Rough token accounting: 12 cells at ~3 tokens each = 36, plus 13 pipes at 1 token = 13, plus a newline ≈ **50 tokens/row.** Header + separator ≈ 60. Total ≈ 60 + 40 × 50 = **2,060 tokens.**

**JSON array of objects.**
```json
[{"Region": "EMEA", "Q1": "4,182", "Q2": "4,410", ...}, ...]
```
Every key is repeated on every row. Per row: 12 × (key ~3 tokens + `"` `"` `:` ~3 tokens + value ~3 tokens + `,` 1) ≈ 12 × 10 = 120, plus braces ≈ **123 tokens/row.** Total ≈ **4,920 tokens.** **Roughly 2.4× the markdown cost for identical information.**

**Row-wise natural language.**
```
For EMEA: Q1 was 4,182; Q2 was 4,410; Q3 was 3,905; ...
```
Per row: 12 × (label ~2 + "was" 1 + value ~3 + separator 1) ≈ 84, plus the prefix ≈ **90 tokens/row.** Total ≈ **3,600 tokens.**

**📐 Numbers you must know:** the ratios, which are stable regardless of table size because they come from per-cell overhead — **markdown 1×, row-wise natural language ~1.7×, JSON ~2.4×, HTML ~4×**, for identical information. Derivation: markdown pays ~1 delimiter token per cell, row-wise pays a repeated label (~3 tokens) per cell, JSON pays a repeated quoted key plus punctuation (~7 tokens) per cell, HTML pays open+close tags (~4–5 tokens) per cell.

**💰 So: markdown 2,060, row-wise 3,600, JSON 4,920.** At $3/Mtok input, per query where this table is retrieved: $0.0062 vs $0.0108 vs $0.0148. At 50k queries/day with one such table each: **$310/day vs $540/day vs $740/day** — $135k/year separating markdown from JSON, for the same table.

**Which do I use?** **Markdown, as the default, for tables that fit in a chunk.** It is the cheapest, and modern models are extremely well-practiced at reading it because it saturates their training data. Row-wise natural language wins in one specific case that matters for RAG: **when rows are the retrieval unit.** If your users ask "what were EMEA's numbers," and you want each row independently retrievable, a row-wise sentence is a self-contained chunk with its labels attached, whereas a markdown row is meaningless without the header. So the rule is: **table fits in one chunk → markdown; table must be split into row-level chunks → row-wise natural language with the header labels inlined.** JSON I use only when the table is going into a structured-output pipeline where a parser, not a model, reads it — and then it is not going in the prompt at all.

**⚠ Trap:** HTML table serialization. Some pipelines emit `<table><tr><td>` because that is what the parser produced. HTML is the most expensive of all — roughly 4–5 tokens of tag overhead per cell, so our 12×40 table lands near 8,000 tokens, nearly 4× markdown. Convert to markdown at the parsing stage, not at prompt-assembly time.

**⚠ Trap two:** losing the units and the scale. `4,182` in a table headed "USD millions" is meaningless if the header caption did not survive. **The caption, the units row, and any footnote asterisks must ride along with the table chunk**, or you will produce answers that are off by a factor of a million and read as confident.

### A table spans four pages with merged cells and a repeated header. What breaks and how do you handle it?

Three separate problems, each with its own fix, and naming them separately is the answer.

**Problem 1 — the table is fragmented by page.** Extractors work per page, so you get four table objects. Chunk them independently and pages 2–4 have no header (or a repeated header, depending on the document), the reader has no idea these are one table, and aggregate questions ("total across all regions") get answered from a quarter of the data — confidently.

*Fix:* a table-stitching pass that runs after per-page parsing and before chunking. Merge adjacent-page tables when: same column count, column x-positions align within a tolerance, the second table's first row either equals the first table's header (repeated header — drop it) or does not look like a header at all (continuation), and there is no intervening non-table content besides page furniture. This is thirty lines of heuristic and it is essential on any corpus of reports.

**Problem 2 — merged cells.** A cell spanning three rows extracts as a value in row 1 and empty strings in rows 2–3, or worse, shifts all subsequent cells left by one and silently misaligns the entire remainder of the table. This is the failure that produces wrong numbers with no visible symptom.

*Fix:* consume span metadata if your parser provides it (`rowspan`/`colspan` equivalents — Azure DI and the HTML output of several Tier-2 pipelines do give you this) and **forward-fill** merged values into the spanned cells so every row is complete and independently readable. If your parser does not give you spans, detect misalignment by asserting a constant column count per row and quarantine the table when it varies. **Do not silently pad short rows** — that is how a misalignment becomes permanent.

**Problem 3 — hierarchical headers.** Two header rows: `2024 | 2024 | 2025 | 2025` over `Q1 | Q2 | Q1 | Q2`. Flattened naively you get four columns named `Q1, Q2, Q1, Q2` and now half your columns are ambiguous, which means half the model's answers are coin flips.

*Fix:* detect multi-row headers (rows above the first numeric row) and flatten by concatenation: `2024 Q1`, `2024 Q2`, `2025 Q1`, `2025 Q2`. Store the original hierarchy in metadata for anyone who needs it.

**Then, the architectural point I would make in the interview:** **for tables that matter, do not rely on the model reading a serialized table at all.** Extract them into a relational store — `(doc_id, table_id, page, row_label, col_label, value_raw, value_numeric, unit)` — and route numeric and aggregate questions to SQL. A model asked to sum 40 rows from markdown will get it wrong at a rate you will not accept, and it will get it wrong differently each time. A `SELECT sum(value_numeric)` is exact, auditable, and free. **The LLM's job is to pick the table and write the filter; the database's job is arithmetic.** That division of labor is the whole reason text-to-SQL and semantic layers exist, and stating it here shows you know where the LLM's competence ends.

**⚠ Trap:** assuming the model can add. It cannot reliably, especially over more than a handful of numbers in context, and the errors are not random — they are plausible. If a user-facing answer contains a computed aggregate, it should come from code, not from the decoder.

### Formulas, handwriting, checkboxes and stamps — how do you handle the messy long tail?

Each of these is a distinct recognition problem and the honest answer includes where you would decline to solve it.

**Formulas.** The target representation is LaTeX, not Unicode. A displayed equation rendered to text becomes `E = mc2` or a soup of private-use glyphs, both of which are unretrievable and unreadable. Purpose-built formula recognizers exist (the Nougat lineage transcribed math to LaTeX; several open models and OCR toolkits ship dedicated math heads), and modern VLMs do this reasonably well when asked explicitly. The chunking rule: **a formula is atomic and belongs with its defining paragraph and its variable glossary.** The retrieval rule: **nobody queries in LaTeX.** Users ask "what's the formula for effective yield." So index a generated natural-language description alongside the LaTeX — "formula for effective annual yield in terms of nominal rate and compounding periods" — and store the LaTeX for rendering in the answer. Indexing raw LaTeX and expecting cosine similarity to find it is a common and complete waste.

**Handwriting.** Classical OCR is poor at it; this is squarely VLM/specialized-HTR territory and even there, accuracy on genuinely messy cursive is not good enough to be trusted silently. **The design decision is to attach a confidence and surface it.** If a handwritten field's transcription is low-confidence, the chunk carries a flag, the answer renders it as uncertain, and the citation links to the image crop so a human can check. **The failure I would refuse to ship is a low-confidence handwriting transcription flowing into an answer with the same confidence as printed text.** For a claims-processing or medical-records corpus, route low-confidence handwriting to human review; do not pretend the pipeline solved it.

**Checkboxes and form fields.** These are the highest-value, lowest-token elements in a form corpus and they are catastrophically easy to get wrong — a checked box that reads as unchecked inverts the meaning of a document. Text extraction sees nothing at all; you need either a form-understanding model (Azure DI's form models, Textract's `SELECTION_ELEMENT`) or explicit VLM prompting. **Serialize as an explicit key-value assertion, never as a symbol**: `Marital status: Married = CHECKED; Single = unchecked` rather than `☑ Married ☐ Single`, because glyph rendering of box characters is unreliable through the whole pipeline and a model reading `☐` versus `☑` at low resolution is a coin flip. Also: **if a PDF has an AcroForm layer, read the form field values directly from the PDF's field dictionary** — they are exact, structured, and free, and a shocking number of pipelines OCR a form whose values were sitting in the file as data.

**Stamps, signatures, watermarks.** Mostly you want to detect and *exclude* them — a "DRAFT" or "CONFIDENTIAL" diagonal watermark that OCRs into the middle of every page is header pollution with extra steps. Detect by rotation angle and by low-opacity rendering, strip from text, and promote to a document-level flag. Signature presence is occasionally the actual query ("is this contract executed?"), in which case detect-and-flag as boolean metadata rather than trying to transcribe.

**🗣 Say this in the room:** "The long tail is where I stop trying to solve everything and start attaching confidence. Formulas get transcribed to LaTeX and indexed via a generated natural-language description, because nobody searches in LaTeX. Handwriting gets a confidence score that propagates all the way into the answer's rendering — I won't ship a pipeline where an uncertain handwritten value is indistinguishable from printed text. And checkboxes come from the AcroForm dictionary when it exists, because that's exact data sitting in the file that most pipelines OCR instead of reading."

### Beyond PDFs — walk me through the traps in docx, pptx, xlsx, HTML and email.

Each format has one or two specific things that ruin an ingestion pipeline, and knowing them by name is the difference between a two-week and a two-month integration.

**docx.** It is a zip of XML, so extraction is easy and the traps are structural. **Tracked changes**: if you read the document XML naively you may get both the inserted and the deleted text concatenated, producing sentences that assert a thing and its negation. You must decide — and state — whether you index the accepted or the original version. **Comments** live in a separate part and are frequently the most interesting content in the document ("legal says we can't claim this") and are also frequently confidential; decide deliberately, do not default. **Footnotes/endnotes** are separate parts and are commonly dropped entirely. **Headers/footers** are separate parts, which is actually convenient — they never pollute your body text unless you go get them. **Embedded objects** (an Excel table pasted as an OLE object, an image of a table) are invisible to text extraction. And `python-docx` walks paragraphs but **skips text inside text boxes and shapes** by default, which in template-heavy corporate documents can be a large fraction of the content.

**pptx.** The trap is that **slides have no reading order** — shapes have z-order and position, and a text box at the bottom might be the title. **Speaker notes** are a separate part and are usually where the actual explanation lives; ingest them and mark them as notes. **A slide is the natural chunk**, and it is often too small on its own, so I merge slides into sections by title-slide boundaries or by deck outline. **Diagrams (SmartArt, grouped shapes) are text fragments with no relations** — "Ingest" "Embed" "Index" as three strings, with the arrows lost. If diagrams matter, render the slide and describe it with a VLM.

**xlsx.** The most misunderstood format in RAG. **A spreadsheet is not a document and chunking it as text is almost always wrong.** Traps: **formulas versus cached values** — you can read either, and the cached value is what a human sees but may be stale; **multiple sheets** with cross-sheet references; **merged cells and multi-row headers**, same as PDF tables; **hidden rows/columns and filtered views** (do you index hidden data? sometimes it is exactly what you must not surface); **dates as serial numbers** (`45,321`) if you read raw values; **number formatting** that displays `4.2%` while storing `0.042`. **My default: load it into a dataframe or a relational table and expose it via a structured/SQL path, not a text index.** If you must index it as text, index per-sheet summaries (sheet name, column headers, row count, a few sample rows, min/max of numeric columns) so retrieval can *find* the sheet, then route the actual question to a query engine.

**HTML.** Boilerplate is the whole game — nav, footer, cookie banner, related-articles, comment sections. Use main-content extraction (readability/trafilatura-style) before chunking. **JavaScript-rendered content is not in the HTML**; if the corpus is a modern SPA-based docs site you need a headless browser, and the naive fetch will silently return an empty shell — which looks like a successful ingest of a 300-byte document. **Assert a minimum extracted-text length per document** and this whole class becomes visible.

**Email (.eml/.msg).** **Quoted-reply chains are the dominant trap**: a 12-message thread means the last message contains all 11 previous ones, so naive ingestion indexes the same text 12 times with linearly-growing duplication. Strip quoted blocks (there are libraries for this; the heuristics are `>` prefixes, `On <date>, <person> wrote:` separators, and Outlook's `-----Original Message-----`), index each message once, and represent the thread as the parent unit. **Signatures and disclaimers** are boilerplate — strip them per the boilerplate question. **Attachments** are separate documents that need their own pipeline path and a parent-child link back to the message. **`.msg` is a proprietary OLE compound format** and needs a dedicated parser; **headers carry the metadata you actually want to filter on** (from, to, date, thread-id) and should be promoted, not embedded.

**⚠ Trap across all of them:** relying on file extension for routing. Enterprise corpora are full of `.doc` files that are actually RTF, `.xls` that are actually tab-separated text, `.pdf` that are actually images with a PDF wrapper, and files with no extension at all. **Route on content sniffing (libmagic-style) with the extension as a hint only**, and count and alert on mismatches — a spike in extension/content mismatches usually means an upstream export changed.

### The corpus includes 4,000 hours of recorded sales calls. How do you ingest audio for retrieval?

Audio is a document format with a time axis, and the design follows from that: **transcribe, segment on meaning rather than duration, and keep timestamps as first-class citations.**

**Transcription.** Whisper-family models (or a hosted equivalent) are the default; the choices that matter are (a) whether you get word-level timestamps — you want them, they are what makes citation-to-timestamp possible — and (b) **diarization**, speaker separation, which is a separate model and is the single biggest determinant of downstream quality on multi-party calls. A transcript without speaker labels is nearly useless for sales calls, because "we can do 20% off" means opposite things depending on who said it. Diarization is also the least reliable component in the stack — expect speaker-attribution errors at turn boundaries and on overlapping speech, and do not build features that assume it is exact.

**Chunking.** Not by duration and not by token count. **Chunk on speaker turns, merged into topical units.** A single "yeah" turn is not a chunk. My rule: merge consecutive turns until you reach the token budget or a long silence gap (> ~5 seconds, a natural topic boundary), never split mid-turn, and always prepend the speaker label and role. This is one of the corpora where semantic chunking on embedding-distance breakpoints genuinely earns its cost, because there is no other structural signal.

**Metadata that must ride along:** `call_id`, `start_ms`, `end_ms`, speaker label, speaker role (rep vs prospect, resolved from the CRM, not guessed), participant list, call date, account id. **`start_ms` is the citation target** — the answer should link to a player seeked to the moment, which is the audio equivalent of a highlighted span and produces the same trust effect.

**⚠ Trap — the one that ruins these systems:** **ASR errors on exactly the tokens users search for.** Product names, company names, people's names, and SKUs are the highest-value query terms and are the worst-transcribed, because they are out-of-vocabulary for the acoustic model. "Kubernetes" becomes "cuber netties." Your hybrid retrieval's lexical channel, which is supposed to save you on rare literals, is now indexing the wrong literal. **Two fixes, both cheap: (1) supply a domain vocabulary / initial prompt / biasing list to the ASR — most APIs support this and it materially improves proper-noun accuracy; (2) run a post-ASR normalization pass that fuzzy-matches transcript spans against your known entity list (product catalog, customer names from the CRM) and corrects them, storing both the raw and corrected text.** Skipping this is why so many call-recording search features feel broken.

**💰 Math:** 4,000 hours at a hosted ASR price of ~$0.006/minute **📅 Volatile** = 4,000 × 60 × $0.006 = **$1,440** one-time for transcription. Diarization roughly doubles it or is bundled. The transcripts: 4,000 hours × ~9,000 words/hour ≈ 36M words ≈ 48M tokens. Embedding at $0.02/M = **$0.96**. **The transcription dominates by three orders of magnitude, which tells you where to optimize — self-hosting Whisper on a GPU at ~10× real-time makes 4,000 hours about 400 GPU-hours, ~$600 at $1.50/hr, and gives you word timestamps and vocabulary biasing under your control.** For a one-time 4,000-hour backfill the hosted API is fine; for a continuous 500-hours-a-day feed ($4,320/month hosted) self-hosting pays back quickly.
