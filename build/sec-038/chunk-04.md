### We added images to our RAG context and the bill went up 6× while p95 latency doubled. Walk me through the token economics of multimodal context, and how caching behaves differently with media.

The mental model: **an image is not "a bit of extra context," it is a thousand-token chunk with none of a text chunk's compressibility.** Text you can truncate, summarize, or rerank away. An image is atomic — you send all of it or none of it — and its token cost is a function of pixel dimensions, which is a property of the file, not of how much information it carries.

**The mechanism.** Providers convert an image to tokens by dividing it into patches or tiles. Anthropic documents an estimate of roughly **`tokens ≈ (width × height) / 750`**, with the long edge capped (historically around 1,568 px; newer high-resolution tiers raise that to ~2,576 px and correspondingly raise the per-image ceiling to several thousand tokens). Other providers use a base-plus-per-tile formula over 512-px tiles. **📅 Volatile — verify the exact formula, caps, and per-tier ceilings for your provider before your loop; these changed materially in the last model generation.** The durable facts are: cost scales with area, there is a hard cap on the long edge (so oversized uploads are downscaled, silently costing you resolution), and a full page scan lands in the 1,000–5,000 token range depending on tier.

**💰 The arithmetic that explains the 6×.** Suppose your text-only context was 4,000 tokens per request and you now attach 5 page images at ~1,300 tokens each:

- Before: 4,000 tokens × $3/Mtok = **$0.012/request**
- After: 4,000 + 6,500 = 10,500 tokens × $3/Mtok = **$0.0315/request** — 2.6× on input alone

If you also moved to a higher-resolution tier at ~4,800 tokens/image: 4,000 + 24,000 = 28,000 tokens = **$0.084/request**, which is **7×**. That's your 6×. At 100,000 requests/day: $1,200/day → $8,400/day, i.e. **$36k/month → $252k/month**. This is a decision that needs to be made deliberately, with the number written down, not arrived at by adding a feature.

**Latency.** Image tokens are prefill tokens, and prefill is compute-bound, so TTFT scales roughly linearly with them. Going from 4k to 28k prefill tokens is a 7× increase in prefill work — that is your doubled p95, and no amount of streaming hides it because it all happens before the first token.

**Caching with media — the part that surprises people.** Images *can* carry a cache breakpoint like any other content block, so a fixed set of reference images in a stable prompt prefix caches normally and reads at roughly a tenth of the input price. But two things bite:

1. **Cache is a strict byte-prefix match, and retrieved images are per-query.** If your images arrive from retrieval, they differ every request, so they sit after any stable prefix and never cache. The 6× is on uncached tokens.
2. **Adding or removing images invalidates the message-level cache** even when tools and system prompt are untouched — the messages array changed. So a conversation that alternates between image-bearing and text-only turns thrashes its message cache.

**What I'd actually do**, in order:
- **Downscale before upload.** If your figures are legible at 1,100 × 850, sending 2,200 × 1,700 costs 4× the tokens for zero accuracy. Compute the token cost of your median image and check it against what you're actually sending; pipelines routinely ship originals.
- **Send text, not pixels, whenever the text exists.** If you already captioned and OCR'd at ingestion, the generation prompt should usually get the extracted text (a few hundred tokens) and only escalate to the image when the question is visual or the text is low-confidence.
- **Cap images per request** — 2–3, not 5–10 — and rerank so the ones you send are the ones that matter.
- **Restructure for caching:** stable system prompt and any fixed reference imagery first, behind a breakpoint; per-query retrieved content last.
- **Batch the offline work.** Captioning is not latency-sensitive; a 50% batch tier halves that bill.

**🗣 Say this in the room:** "Image tokens scale with pixel area — roughly width×height over 750 on Anthropic's estimate, capped at the long edge — so a page scan is one to five thousand tokens and five of them dwarf your text context. That's prefill, so it hits TTFT linearly too. And retrieved images can't sit in a cached prefix, so you pay full price. My first three moves are downscale to legibility, send extracted text instead of pixels when it exists, and cap images per request."

### How do you build RAG over audio and video? Be specific about chunking and about what you index.

The mental model: **audio and video are time-indexed streams, so the retrieval unit must carry a timestamp, and the timestamp is the citation.** Everything else follows from that.

**Audio / podcasts / meeting recordings.**

1. **Transcribe with word- or segment-level timestamps.** Any modern ASR gives you this; keep the timestamps, they are the whole point. Also keep **speaker diarization** — "who said it" is a filter users ask for constantly ("what did the CFO say about hiring?") and it's nearly free at transcription time.
2. **Do not chunk by the ASR's segments.** ASR segments break on silence, which is a terrible semantic boundary — you get 8-second fragments that split a sentence. Re-chunk the transcript by *semantic* units (paragraph-ish, 200–400 words) using the same recursive/semantic splitters you'd use on text, and carry `(start_time, end_time, speakers)` through as metadata by taking the min/max over the words in the chunk.
3. **Overlap generously.** Speech is repetitive and low-information-density per word; 20–25% overlap is reasonable, more than you'd use for dense prose.
4. **Index the transcript chunk, plus context**: episode/meeting title, date, participants, agenda item if you have one. Prepend a one-line document-level context to each chunk — the contextual-retrieval trick — because a chunk of a meeting transcript is often uninterpretable alone ("yeah, I think we should push it").
5. **Handle ASR error like OCR error.** Proper nouns and product names are the failure concentration. Supply a domain vocabulary/hotword list to the ASR if it supports one, and index a phonetic-normalized field alongside the raw transcript for fuzzy lexical matching.

**Video adds two more streams**, and the mistake is treating video as "audio plus lots of images."

- **The transcript is still the primary index** for most corpora — training videos, recorded talks, meetings. Do the audio pipeline above.
- **Frame sampling should be adaptive, not fixed-interval.** Sampling 1 frame/second over a 60-minute video is 3,600 frames, and if 55 of those minutes are a static slide you have paid to caption the same slide 3,300 times. Use **shot/scene-change detection** (frame-difference or histogram-based) to sample one keyframe per visually distinct segment, and cap frames per minute. On slide-based content this typically collapses 3,600 frames to 40–80 — a **50× reduction** in captioning cost.
- **Caption keyframes with the surrounding transcript as context**, exactly as with figures. A slide captioned in isolation reads "a slide with a diagram"; with the speaker's words it reads "the three-tier ingestion architecture, showing the queue between parsing and embedding."
- **Index keyframes as their own documents** with `(video_id, timestamp, ocr_text, caption)`, and link them bidirectionally to the transcript chunks they overlap in time. Retrieval on either side co-retrieves the other, exactly as with figures.
- **On-screen text is often the highest-value signal** in screen-recorded or slide-based video — OCR the keyframes and index that verbatim.

**💰 Cost model for 1,000 hours of video.** Naive 1 fps: 3.6M frames — a non-starter at any per-image price. Scene-change sampling at ~60 keyframes/hour: 60,000 frames. At ~1,300 image tokens + 300 context in, 250 out: input 60,000 × 1,600 = 96M tokens ≈ $288; output 15M ≈ $225. **~$513, or ~$257 on a batch tier.** Transcription at commodity ASR rates on the order of $0.006/minute is 60,000 minutes ≈ $360. So roughly **$600–900 to index 1,000 hours**, which is a very defensible number — and it is defensible *only* because of the scene-change sampling.

**⚠ Trap:** retrieving a transcript chunk and returning the whole video. The user has to scrub. Always return a deep link with the timestamp (`?t=1432`), and prefer a small window around the match rather than the exact start, because ASR boundaries are imprecise and starting mid-sentence feels broken.

### How do you cite back to a page region, a bounding box, or a timestamp? Plumb it through for me.

The mental model: **a citation is only as good as the coordinate system you preserved at ingestion. You cannot recover provenance you didn't record, and by the time you notice, reindexing is the only fix.** So this is an ingestion-schema decision made on day one, not a UI feature added in month six.

**The provenance record every retrieval unit carries.** Non-negotiable fields:

```
chunk_id, doc_id, doc_version, source_uri,
page_number,                       # 1-indexed, for paged formats
char_start, char_end,              # offsets into the *extracted text* of that page
bbox: [x0, y0, x1, y1],            # in PDF user-space points, plus page width/height
                                   # so the client can scale to any render resolution
time_start, time_end,              # for AV
extractor_name, extractor_version, # so you can invalidate on parser upgrade
ingested_at
```

Two details people get wrong. **Store bounding boxes in the source coordinate system with the page dimensions alongside**, not in the pixel coordinates of whatever DPI you happened to render at — otherwise a viewer rendering at a different zoom draws the box in the wrong place. And **record the coordinate origin**: PDF user space has the origin at the bottom-left, most image and web viewers use top-left, and flipped highlight boxes are the single most common bug in this feature.

**Getting the boxes.** Layout-aware parsers (PyMuPDF, pdfplumber, Docling, Unstructured, and the commercial document-intelligence services) return word- or block-level boxes. Keep them at word granularity if you can afford the storage — you can always union words into a span box, but you cannot split a block box back into words. For a chunk, the citation box is the union of its words' boxes, per page (a chunk spanning a page break has two boxes).

**Span-level citation in the answer**, which is what users actually want:
1. Ask the generator for structured output: claims with the supporting quote and the chunk ID it came from. Quote-then-answer prompting materially improves this.
2. **Verify the quote by string search in the cited chunk.** If the model's quote isn't literally present, the citation is unverified — mark it, don't render it as a confirmed source. This one check catches the "plausible citation pointing at the wrong chunk" failure that otherwise sails through review.
3. Map the located quote's character offsets to word boxes by walking the word list of that chunk, union the boxes, and emit `(page, bbox)` to the client.

**For visual retrieval (ColPali-style)** there is no character offset — the retrieval unit is a page. You have two options: cite at page granularity (honest and easy), or run OCR *on the retrieved page only* at answer time to recover word boxes for the specific span. The second is a good trade: you skip OCR for the whole corpus and pay for it on the handful of pages you actually cite.

**For AV**, the citation is `(media_id, time_start)` and the UI is a deep link plus an inline player seeked to a few seconds before the match. Include the transcript excerpt as text too — many users won't play the clip.

**⚠ Trap:** citations that don't survive reindexing. If `chunk_id` is a positional hash, re-chunking renumbers everything and every previously-saved citation in a report or a shared conversation now points at the wrong text. Use **content-derived, stable identifiers** — `doc_id + page + char_offset` — so a citation resolves against a document rather than against a chunking run, and version documents so an old citation can say "this cited v3; the current version is v7" rather than silently resolving to different text.

### Design a multimodal RAG system over 500,000 enterprise PDFs — contracts, financial reports, technical manuals, a lot of them scans. Take me through it.

I'd start by refusing to design one pipeline, because those three document classes have different failure modes and different value.

**Step 0 — characterize the corpus, before any architecture.** Sample 500 documents and measure: fraction with a usable text layer (extract text, compute chars/page; below ~200 chars/page it's a scan), fraction with tables, fraction with figures, page-count distribution, language mix. And sample 200 real user questions if any exist. **These two distributions determine every subsequent decision**, and I'd want a day to get them rather than guessing.

Assume it comes back: 55% clean text layer, 30% scanned, 15% mixed; contracts are text-heavy, financial reports are table-heavy, manuals are figure-heavy. 500k docs × ~25 pages = **12.5M pages**.

**Ingestion — routed by document class.**

A router at ingest time classifies each document (text-layer test + a lightweight classifier on the first page) and sends it down one of three paths:
- **Clean text →** layout-aware parse (PyMuPDF), structure-aware chunking on headings, tables extracted as first-class objects, figures extracted and queued for captioning.
- **Scanned →** VLM-based page parsing rather than classical OCR (this is where the quality difference is largest), producing text + layout + tables. Confidence-scored; the bottom decile goes to a quarantine queue for human review, not into the index.
- **Layout-critical (financial statements, forms) →** additionally rendered to page images for a visual index.

Everything is idempotent and content-hashed: `doc_id = hash(bytes)`, chunk IDs derived from `doc_id + page + offset`, so re-running the pipeline is a no-op on unchanged documents. Failures go to a dead-letter queue with the error, and **there's an alert on the per-day parse-failure rate**, because the default outcome otherwise is that 5% of documents silently fail and nobody knows for a year.

**Indexes — four, deliberately.**
1. **Text chunks**: hybrid (BM25 + dense), the workhorse. 12.5M pages × ~2.5 chunks = ~31M chunks.
2. **Tables**: table-level summaries for routing, with the structured table stored for query-time computation.
3. **Figures/captions**: VLM captions + surrounding text + OCR, as one composite document.
4. **Page images** (visual/late-interaction), scoped to the layout-critical and scanned subsets only — not all 12.5M pages. Say 30% = 3.75M pages. At 264 KB/page that's **990 GB**, so: int8 quantization (~4× → 250 GB), and it's a second-stage reranker over candidates from index 1, never a flat scan.

**Query path.**
Query understanding (rewrite against chat history, extract metadata filters like date/counterparty/document type) → parallel retrieval across indexes 1–3 with ACL filters applied **pre-filter, not post-filter** → RRF fusion → cross-encoder rerank to ~15 → visual rescoring for candidates from the scanned/layout set → dedup + MMR → budget-allocated packing with reserved slices for figures and tables → generation with structured citations → quote verification.

**Access control is the requirement that shapes everything.** Enterprise contracts have per-document ACLs. Filters must be applied *inside* the ANN search (filterable HNSW / a filtered index), because post-filtering a top-100 down to the 3 documents a user may see destroys recall — and a user seeing a chunk of a contract they're not entitled to is a legal incident, not a bug. I'd also make ACL correctness a first-class test in CI with synthetic users.

**💰 Cost model.** Parsing 3.75M scanned pages with a VLM at ~1,300 image tokens in / 800 out: input 4.9B tokens ≈ $14,600; output 3B ≈ $45,000 — **~$60k, or ~$30k on a batch tier**. That number will decide whether this project happens, so it goes on slide one, alongside the alternative (classical OCR at near-zero token cost with materially worse quality on exactly the documents users care about). Captioning figures: if 15% of pages have a figure, ~1.9M figures ≈ $20k batched. Embeddings for 31M chunks × ~400 tokens = 12.4B tokens at $0.10/Mtok ≈ $1,240. **Total initial index: roughly $50–85k**, with steady-state incremental ingest a small fraction of that.

**Evaluation, which I'd build before the pipeline.** 300 golden questions stratified by document class and by whether the answer lives in text, a table, or a figure. Measure retrieval recall@k separately per stratum — the figure and table strata will be much worse than text and that's the roadmap. Gate CI on it.

**🗣 Say this in the room:** "I'd route by document class at ingest rather than build one pipeline, because scans, tables, and prose fail differently. Four indexes, ACL filtering inside the ANN search rather than post-hoc, visual late-interaction as a reranker on the scanned subset only, and a parse-failure alert. The number I'd lead the design review with is the fifty-to-eighty thousand dollars of one-time VLM parsing, because that's what actually decides the architecture."

### A user reports that your text-to-SQL assistant told them Q3 revenue was $4.1M when finance says $3.8M. You have ten minutes. Go.

I want to localize the fault before I theorize, and there's an order that gets there fastest.

**Minute 0–1: get the artifacts.** Pull the trace: the question as asked, the retrieved schema subset, the generated SQL, the row count returned, the execution timestamp, and the user's tenant. If I can't get all of that from a log, that's the first bug and I fix it today — a text-to-SQL system without full query provenance is undebuggable.

**Minute 1–2: is it the same question?** Read the question as typed. Half of these are not defects. "Q3 revenue" — which fiscal quarter, in which timezone, gross or net, which entity, including or excluding intercompany? If finance's $3.8M is net-of-refunds for the parent entity in fiscal-Q3-ending-August and the user asked a calendar-Q3 question, the system is right and the *definition* is the defect. That is still a bug — an ambiguous question should have prompted a clarification, not an answer — but it's a different fix.

**Minute 2–4: run the SQL, then run finance's SQL.** Execute both against the same snapshot. If mine returns $4.1M and finance's returns $3.8M, diff the two queries structurally, not textually. Specifically check, in this order because this is the frequency order:
- **Filters present in theirs, absent in mine.** `is_test = false`, `status <> 'CANCELLED'`, `is_intercompany = false`. This is the number-one cause and it produces exactly this signature: mine is *higher* than theirs.
- **Join fan-out.** Does my query join to a one-to-many table and then `SUM`? Check `COUNT(*)` on the fact table alone versus after the join. A ratio above 1.0 is the smoking gun. This also produces a higher number.
- **Date boundary.** Compare the exact predicate. `>= '2024-07-01' AND < '2024-10-01'` in UTC against a business definition in US/Eastern shifts four hours of orders at each boundary.
- **Grain / dedup.** Does theirs have a `DISTINCT` or a pre-aggregation mine lacks?

A **higher** number points at a missing filter or a fan-out. A **lower** number points at an inner join that dropped rows or a filter that's too narrow. That polarity check alone usually halves the search space.

**Minute 4–6: check schema linking.** Did the retriever hand the model `fct_orders` or `stg_orders`? Is it querying a deprecated view? Log the retrieved table set and compare against what finance queries. If the model used a different table, the defect is in retrieval, not generation, and the fix is in the catalog (deprecate the old table's description, or exclude it) rather than in the prompt.

**Minute 6–8: check freshness.** When did the source table last load? If the query ran mid-load against a partially-written partition, both queries are "correct" and the data was in flux. This is why the answer surface needs a freshness stamp.

**Minute 8–10: classify and fix at the right layer.** The instinct will be to patch the prompt — "always exclude test orders." Resist it. **A prompt patch fixes one query and leaves the class open.** If the cause was a missing filter, the fix is a semantic-layer metric definition or a view that applies it, so no future query can omit it. If it was fan-out, the fix is pre-aggregation in the model layer. If it was schema linking, the fix is the catalog. And in every case, **the question goes into the golden set with finance's SQL as gold**, so the regression is caught in CI forever.

**🗣 Say this in the room:** "First I check whether it's actually the same question — half of these are definitional. Then I run both queries and use the polarity: too high means a missing filter or a join fan-out, too low means an inner join dropped rows. Then schema linking, then data freshness. And I fix it in the semantic layer, not the prompt, because a prompt patch fixes one query and leaves the whole class open — then the question goes in the golden set."

### Your GraphRAG system worked well at launch. Three months of incremental ingestion later, global-search answers have gotten vague and generic. Diagnose it.

"Vague and generic" is a specific symptom with a specific cause tree, and the shape of the degradation — global search worsening while local search holds up — points hard at the community layer.

**Hypothesis 1 (most likely): community drift from local-repair updates.** If new nodes were attached to a neighbor's community rather than triggering a Leiden recompute, the partition has been drifting away from a modularity-optimal one for three months. Communities become large, internally incoherent grab-bags; the LLM summarizing a grab-bag writes exactly the vague, generic report you're seeing ("this community covers various topics related to operations"). **Test:** recompute Leiden from scratch on the current graph and compare partitions — measure the normalized mutual information between the live partition and the fresh one, and the community-size distribution. If mean community size has grown or the size distribution has a long right tail that wasn't there at launch, this is it. **Fix:** scheduled full recompute with report regeneration for changed communities, plus a monitor on community-size percentiles.

**Hypothesis 2: hub nodes from entity over-merge.** Three months of entity resolution has been merging aggressively. If generic entities ("the system", "the customer", "the team") or two genuinely different organizations have collapsed into single high-degree nodes, those nodes glue unrelated subgraphs together and Leiden puts everything in one community. **Test:** plot the node-degree distribution now versus at launch. Look at the top 20 nodes by degree and read their names — if you see generic nouns, that's your answer. **Fix:** stoplist generic entity names at extraction, degree-cap or split hub nodes, audit and reverse recent merges (which requires that merges were recorded reversibly — if they weren't, that's the deeper bug).

**Hypothesis 3: stale reports.** Communities changed membership but their reports were never regenerated, so global search is map-reducing over summaries that describe the graph as it was in month one. **Test:** for each community, compare a hash of its current member set against the member-set hash stored on its report. Count mismatches. If a large fraction of reports are stale, that's mechanical and fixable. **Fix:** the dirty-flag discipline described earlier, plus a dashboard metric "% of reports whose member set has changed since generation."

**Hypothesis 4: the corpus itself changed.** Three months of new documents might be genuinely more heterogeneous — a new business line, a different document type — so the graph now spans topics that don't cluster cleanly. **Test:** run global search restricted to launch-era documents; if answers are good on the old slice and bad overall, the corpus, not the pipeline, is the cause. **Fix:** either partition the index by domain and run global search per-domain, or increase the community level's resolution so you get finer, more coherent clusters.

**Hypothesis 5: a level-selection mismatch.** Global search runs at a chosen hierarchy level. As the graph grew, the number of communities at that level grew, so more reports get packed into each map batch, each report gets less attention, and the reduce step is averaging over more partial answers — which produces blandness. **Test:** run the same query at a deeper (finer) level and compare. **Fix:** make the level adaptive to graph size, or increase the number of map batches.

**The order I'd actually check them:** community-size distribution and degree distribution first, because both are one query each and they discriminate between hypotheses 1, 2, and 4 immediately. Report staleness second, because it's a cheap hash comparison. Level selection last.

**⚠ Trap:** reaching for "the model got worse" or "let's regenerate all the reports with a better prompt." Regenerating reports over a bad partition produces better-written summaries of incoherent groups. **Fix the partition before you fix the prose.** And the meta-lesson: this whole failure was invisible because nobody was monitoring graph-structure metrics. Community-size percentiles, node-degree percentiles, report-staleness rate, and merge-rate belong on a dashboard from day one, exactly as index bloat and tombstone counts do for a vector store.

### An exec wants a monthly "what are the emerging themes across all customer feedback?" report over 200,000 support tickets. GraphRAG, long-context, or hybrid retrieval plus reranking? Choose and defend.

**None of them as stated**, and being able to say that crisply is the answer.

Let me price the three first, because the arithmetic settles two of them immediately. 200,000 tickets at ~400 tokens = **80M tokens**.

**Long-context.** 80M tokens does not fit in any context window (1M is the current frontier ceiling, so you'd need 80 sequential calls even to see it all). Even setting that aside: 80M tokens × $3/Mtok = **$240 per full pass** at input rates, before output, with no ability to reason across passes. And it would be a map-reduce anyway — you'd be reimplementing GraphRAG's reduce step with worse chunking. **Rejected on arithmetic.**

**Hybrid retrieval + reranking.** This is the census-versus-sample problem from the very first question. Top-50 tickets out of 200,000 is a 0.025% sample, selected by similarity to a query that doesn't name any theme. It will produce a confident, fluent, and **statistically meaningless** answer, and — because it looks exactly like a good answer — it is the most dangerous of the three. **Rejected on validity.**

**GraphRAG.** It's built for this. But: 80M tokens of indexing at the cost model from earlier is on the order of **$3,000–5,000 per full build**, tickets arrive continuously so the community layer needs regular recompute, and honestly, entity-relationship extraction is a poor fit for support tickets — the useful structure in a ticket isn't "Person X works at Org Y," it's "this customer hit failure mode Z in feature W."

**What I'd actually build, and why it's better than all three:**

**Embed → cluster → label → track.** This is the classical unsupervised-analysis pipeline, and it is the right shape for "what are the themes," because clustering *is* the theme-extraction algorithm.

1. Embed all 200,000 tickets: 80M tokens × $0.10/Mtok = **$8**. Not a typo.
2. Reduce dimensionality (UMAP) and cluster with **HDBSCAN**, which is the right choice because it doesn't require choosing k and it produces a noise class rather than forcing every ticket into a cluster — most tickets genuinely aren't part of an emerging theme.
3. For each cluster, sample 20 representative tickets (nearest the centroid, plus a few at the periphery for range) and make **one LLM call** to write a label, a description, and a representative quote. 400 clusters × ~4,000 tokens in / 400 out = 1.6M + 0.16M tokens ≈ **$7**.
4. Report cluster **size, growth rate versus last month, and sentiment**. "Emerging" is a derivative — a theme that was 40 tickets last month and 400 this month is the story, and no amount of summarization surfaces that. This is the part every LLM-first design misses.
5. Ship it as a monthly batch job with a human-readable report and drill-down links to the actual tickets.

**Total recurring cost: roughly $15–30/month**, versus thousands for GraphRAG, with output that is *quantitative* ("Theme: payment webhook timeouts — 412 tickets, up 8.2× MoM") rather than a fluent paragraph.

**When I'd flip to GraphRAG:** if the exec's real question is multi-hop and relational — "which customers who complained about latency also churned, and what did their account managers say?" — clustering can't answer that and a graph can. So the deciding question, which I'd ask before choosing anything: **"Show me three real questions you'd want answered."** If they're all "what are people complaining about," it's clustering. If they involve joining entities across sources, it's a graph.

**🗣 Say this in the room:** "Retrieval-plus-rerank is invalid here — top-50 out of 200,000 is a 0.025% sample of a census question, and it produces a confident answer that means nothing. Long context doesn't fit and costs $240 a pass. GraphRAG works but costs thousands and entity extraction is a poor fit for tickets. I'd embed, HDBSCAN cluster, label each cluster with one LLM call, and report size and month-over-month growth — about $15 a month, and it answers 'emerging' quantitatively, which none of the other three do."

### 🏋 Drill: the GraphRAG cost model, unaided, twenty minutes

**Setup.** No notes, no calculator beyond arithmetic on paper. Twenty minutes.

**The scenario.** A legal-tech company has 40,000 contracts averaging 12,000 tokens each. They want GraphRAG for cross-contract analysis ("which of our agreements have MFN clauses that conflict with the new distribution deal?"). Assume $3/Mtok input, $15/Mtok output, $0.10/Mtok embeddings, 800-token chunks, an extraction prompt of 1,500 tokens producing 600 output tokens per chunk, no gleanings. Assume extraction yields 4 entities per chunk with 30% duplication, and that Leiden produces communities averaging 8 members across all levels, with each community report reading 3,000 tokens and writing 700.

**Produce, in twenty minutes:**

1. Total corpus tokens, and the number of chunks.
2. Extraction cost, input and output shown separately.
3. Entity count after dedup, and the resulting community count across all levels.
4. Community-report generation cost.
5. Embedding cost.
6. Total index cost, and the multiple over plain chunk-and-embed vector RAG.
7. The monthly incremental cost if 800 new contracts arrive per month, stating which pipeline stages are incremental and which are not.
8. One sentence: at what query volume does this pay for itself against a paralegal at $60/hour taking 25 minutes per cross-contract question?

**Pass criteria.**
- Every stage priced with input and output separated. Combining them is an automatic fail — output is 5× input and that asymmetry drives the answer.
- Total corpus = 480M tokens; chunks = 600,000. If you didn't get these two, stop and restart; everything downstream is wrong.
- Your total lands between **$25,000 and $60,000** for the initial build. The band is wide because your entity/community assumptions are yours; what's graded is that the arithmetic is internally consistent and that you can point at which assumption drives the number.
- You correctly state that extraction and entity resolution are incremental (linear in new content) while **community detection and report regeneration are not** — and you give a strategy (scheduled recompute, regenerate only changed communities) with a cost.
- Your break-even calculation names the assumption you're least sure of.

**Common failure modes to check yourself against.** Forgetting that the extraction prompt is sent with *every* chunk (600,000 × 1,500 = 900M tokens of prompt alone — this often exceeds the corpus itself and it's the single most-missed term). Pricing output at input rates. Forgetting that community reports at higher hierarchy levels read their children's reports, so the community count is not just leaf communities. Quoting a total without saying which assumption it's most sensitive to.

**Do it again next week with gleanings enabled and a 50% batch discount**, and confirm you can state in one sentence which of those two changes moves the number more.

### 🏋 Drill: implement late-interaction MaxSim scoring from memory, twenty-five minutes

**Setup.** Blank editor, no autocomplete, no reference. Twenty-five minutes. This is the ColBERT/ColPali scoring core, and it comes up in from-scratch rounds at companies doing document AI.

**Part 1 (12 minutes) — the scorer.** Implement, in PyTorch or NumPy:

```
maxsim_score(Q, D) -> float
  Q: [n_q, d]  query token embeddings
  D: [n_d, d]  document (or page-patch) embeddings
```

Requirements: L2-normalize both, compute the full `[n_q, n_d]` similarity matrix, take the max over the document axis for each query token, sum over query tokens. Then write the **batched** version that scores one query against `B` documents of *ragged* length — meaning you must handle padding, and padded positions must not be able to win the max (mask with −inf before the max, not zero after).

**Part 2 (8 minutes) — the two-stage retrieval loop.** Write the function that, given a query and a corpus of `N` documents each with multi-vector embeddings, returns the top-k. It must not MaxSim against all `N`. Implement candidate generation first — e.g. ANN over all document token vectors flattened, collecting the union of parent document IDs of the top hits — then MaxSim rescoring of only those candidates.

**Part 3 (5 minutes) — answer these without looking anything up:**
1. Why max over document tokens and sum over query tokens, rather than the reverse or mean-over-both? What property does that asymmetry give you?
2. For 1M pages at 1,030 patch vectors of 128 dims in fp16, how much storage? Show the arithmetic.
3. What breaks if you L2-normalize the query but forget to normalize the document?
4. Name two ways to cut the storage and say what each costs you in recall.

**Pass criteria.**
- The scorer is correct and under 15 lines. The masking in the batched version is right — this is the part that's usually wrong, and the bug is silent (padded zeros score 0, which beats genuinely negative similarities, so a padded document can outrank a real one).
- Two-stage retrieval is present. A solution that scores all N documents is a fail regardless of correctness, because the entire engineering point of PLAID-style pruning is that you cannot afford the full scan.
- Q1: max-over-document means each query token finds its single best evidence anywhere in the document, which is what preserves locality and lets one term match one table cell; sum-over-query means every query term must be satisfied, which gives conjunctive behavior. Reversing it would reward documents that broadly resemble the query rather than documents that contain each specific term.
- Q2: 1,030 × 128 × 2 bytes = 263,680 bytes ≈ **264 KB/page**; × 10⁶ = **264 GB**. Stated in GB, with the arithmetic shown.
- Q3: unnormalized document vectors let magnitude dominate the dot product, so long or high-norm documents win regardless of direction — a silent, systematic ranking bias, not a crash.
- Q4: quantization (int8/binary — 4–8× smaller, costs some recall unless you rescore top candidates in full precision) and patch pooling / token pruning (fewer vectors per page, costs recall on fine detail, which is exactly the detail you adopted late interaction for). Naming the *cost* of each is what's being graded.

**Stretch, if you finish early:** implement binary quantization of the document vectors with Hamming-distance candidate generation followed by fp16 rescoring of the top 200, and measure the recall@10 delta against exact MaxSim on a small synthetic corpus. That two-stage quantize-then-rescore pattern is the single most useful thing to have implemented once, because it is how every production multi-vector system controls storage.
