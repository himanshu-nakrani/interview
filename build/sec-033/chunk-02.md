### How do you mine hard negatives, and what goes wrong when you do it naively?

A hard negative is a passage that **scores high but is wrong**, and it is the only kind of negative that carries gradient once your model is halfway decent. Random negatives teach the model "a support ticket is not a Python tutorial." Hard negatives teach it "the 2023 refund policy is not the 2024 refund policy," which is the distinction your users actually care about.

The standard mining loop, which you should be able to describe without notes:

1. Take your current retriever (or BM25, for the cold start).
2. For each training query, retrieve the top ~100 passages.
3. Remove the known positives.
4. Sample `k` negatives from a **rank window** — not the top of the list.
5. Train. Optionally re-mine with the improved model and repeat.

Step 4 is where everyone goes wrong, so let me be precise. Sampling from ranks 1–10 gives you the hardest possible negatives and it is the single most reliable way to destroy a fine-tune, because in a real corpus with incomplete labels, the top-ranked non-labeled passages are *overwhelmingly likely to be unlabeled positives*. You will spend your entire training budget pushing apart pairs that should be together. The working heuristic is to sample from a **middle window — roughly ranks 10 to 50, or 30 to 100** depending on how complete your labels are — plus a margin filter that discards any candidate whose score is within some epsilon of the positive's score.

Two named refinements worth citing. **ANCE** (Xiong et al., 2020) observed that negatives mined once at the start go stale as the model improves, and introduced asynchronously refreshed negatives from a periodically-rebuilt index — the model mines against itself, continuously. **RocketQA** (Qu et al., 2021) added cross-batch negatives and, crucially, used a **cross-encoder to denoise** the mined candidates: if a strong cross-encoder scores a candidate as relevant, throw it out of the negative pool rather than training against it.

**📐 Numbers you must know:** the practical dial is `k` = **4 to 8 hard negatives per query**, mined from ranks ~10–50, re-mined once or twice over the course of training. Below 4 you underuse the signal; above ~16 the memory cost per query grows linearly and you start over-representing the mining model's idiosyncrasies.

**⚠ Trap:** mining negatives with the same model you are about to evaluate against, then reporting improvement on a test set drawn from the same query pool. You have fit the mining artifacts. Mine from the training split only, and hold out queries *and* their positives before mining ever runs.

**🔍 Failure taxonomy — the fine-tune that made things worse.** If recall dropped after adding hard negatives, check in this order: (1) are your "negatives" actually unlabeled positives? Hand-inspect 30 of them — if more than ~10% look relevant, your window is too aggressive. (2) Did you drop batch size to fit the negatives? You may have traded 8k in-batch negatives for 8 hard ones. (3) Is the learning rate too high? Hard negatives sharpen the loss surface; the LR that worked with random negatives is often 2–3× too hot.

### Tell me about false-negative contamination and what GISTEmbed-style filtering does about it.

False-negative contamination is the specific, quantifiable damage done when a passage you labeled "negative" is in fact a correct answer to the query. It matters more than people expect because of how InfoNCE weights its gradients: the softmax concentrates negative pressure on the *highest-scoring* negatives, which are exactly the ones most likely to be mislabeled. So contamination doesn't add uniform noise — it precisely targets your gradient at your worst-labeled examples.

Where it comes from: retrieval datasets are built by annotating a *sample* of candidates. MS MARCO famously labels roughly one relevant passage per query out of a corpus of 8.8M, and there are demonstrably many other passages that answer the same question. Every one of them is a landmine in your negative pool. In your own corpus it is worse — if you generated synthetic queries from chunks, near-duplicate chunks are all correct answers and only one is labeled.

**📄 Paper:** Solatorio (2024), **GISTEmbed** — "Guided In-sample Selection of Training Negatives." The idea: use a stronger *guide* model to score every in-batch pair, and **mask out** of the loss any query-passage pair whose guide score exceeds the guide score of the query's own labeled positive. Those pairs are dropped from the softmax denominator entirely rather than being trained against. It replaced blind reliance on in-batch negatives being negative, and it is a small enough change that you can implement it in your own training loop in an afternoon.

The implementation in pseudocode, because the masking detail matters:

```
# guide is a frozen, stronger embedder
g_pos = sim(guide(q_i), guide(p_i))               # guide score of the true pair
g_all = sim(guide(q_i), guide(p_j)) for all j      # guide scores of all candidates
mask  = (g_all > g_pos) & (j != i)                 # suspected false negatives
logits[mask] = -inf                                # excluded from the denominator
loss = cross_entropy(logits, labels)
```

Setting the logit to `-inf` (not zero, not a large negative constant) removes the term from the softmax denominator cleanly. Zeroing it would leave `exp(0) = 1` in the denominator — a bug I have seen in a real training script.

**💰 Math on the cost:** the guide model runs over every batch, so if the guide is the same size as your student you roughly double training cost. If the guide is a 7B model and the student is 335M, you have made training ~20× more expensive and you should instead **precompute guide scores offline for your mined candidate pool** — mining is a one-time pass over `n_queries × 100` pairs, not a per-step cost. For 100k training queries × 100 candidates = 10M pairs at 400 tokens each = 4e9 tokens; on a 7B guide at `2 × 7e9 × 4e9` = 5.6e19 FLOPs, at 350 effective TFLOPS that is 160,000 s ≈ **44 GPU-hours ≈ $130 at $3/hr**. That is a rounding error against the value of not training on false negatives.

**⚠ Trap:** using the *same family* as the guide and student (e.g. `bge-large` guiding `bge-small`). The guide inherits the student's blind spots, so exactly the false negatives you most need to catch are the ones it also scores low. Use an architecturally different guide — ideally a cross-encoder, which sees the pair jointly and is far more accurate at this specific judgment.

### Your recall dropped roughly 30% after what should have been a no-op refactor. Nothing in the index changed. Walk me through the debug.

I would go straight for the instruction prefix, because in an embedding system this exact symptom — large, uniform recall degradation with no error, no exception, no latency change, and an index that was never touched — has one overwhelmingly likely cause.

Several major model families require a **task prefix baked into the text before tokenization**: the E5 family uses `"query: "` and `"passage: "`; BGE's English models use an instruction on the query side only, something like `"Represent this sentence for searching relevant passages: "`; the instruction-tuned LLM embedders take a full natural-language task description. These are not metadata. They are literal string concatenation, and the model was trained with them present.

Why this is so vicious: **it degrades gracefully.** If you drop the prefix, the query still embeds into roughly the right topical neighborhood. Nearest neighbors are still plausible. A smoke test that asserts "searching for 'refund policy' returns something about refunds" still passes. You just lose the fine discrimination that the asymmetry marker was providing, and recall@10 goes from 0.82 to 0.55.

My debug procedure, in order:

1. **Reproduce with a known pair.** Take one query and its known-correct chunk. Embed both, compute cosine. Compare that number to the same computation on the previous deploy. If the number moved, the embedding path changed.
2. **Diff the exact string fed to the tokenizer.** Not the input to your function — the string immediately before `tokenizer(...)`. Log it. In a refactor, prefixes get lost when someone consolidates two encode paths into one, or swaps `SentenceTransformer.encode` (which applies the configured prompt) for a raw `AutoModel` call (which does not).
3. **Check for asymmetry inversion.** A subtler version: the prefixes are present but swapped, or the ingest path used `"passage: "` and the new query path also uses `"passage: "`. Symmetric prefixes on an asymmetric model is a silent 20-point hit.
4. **Check pooling and normalization** as the next candidates — same class of silent failure, same debug method.

**⚠ Trap — and it is the marquee trap of this entire section:** believing that because prefixes are "just strings," forgetting them is a cosmetic issue. It is a 20-to-30-point recall regression that no unit test in a normal test suite will catch.

**The rule I enforce in review:** the prefix must live in exactly one place — a single `embed_query()` / `embed_passage()` pair of functions that nothing bypasses — and there must be a **golden-vector test**: check in 20 (text, expected_embedding_first_8_floats) fixtures generated by the reference implementation, and assert cosine > 0.9999 in CI. That test catches prefix loss, pooling drift, tokenizer version bumps, and dtype changes, all at once, in 200ms.

**🗣 Say this in the room:** "Large uniform recall loss with no errors and an untouched index means the embedding function changed, not the index. First thing I check is the instruction prefix — E5 needs `query:`/`passage:`, BGE needs its query instruction — because dropping it degrades gracefully enough that every test still passes. I'd catch it permanently with a golden-vector test in CI."

### Do task instructions in embeddings actually earn their keep, or is it cargo cult?

They earn their keep, but for a narrower reason than people assume, and the honest framing is worth having.

The mechanism: instruction-conditioned embedders concatenate a natural-language task description to the input, so the *same* text embeds to *different* points depending on what you want to retrieve. "Given a web search query, retrieve relevant passages" versus "Given a claim, retrieve documents that refute it" versus "Given a code function, retrieve its documentation." One model, many metric spaces, selected at inference time.

**📄 Paper:** Su et al. (2023), **INSTRUCTOR** — trained one embedder on ~330 tasks each paired with a natural-language instruction, showing a single instruction-conditioned model could beat task-specific models across a broad benchmark; it replaced the pattern of fine-tuning a separate embedder per task.

Where instructions genuinely help: (1) **disambiguating the relation**, as in the refute-vs-support example — no amount of topic matching gets you refutation retrieval, but an instruction can; (2) **asymmetry marking**, the `query:`/`passage:` case, which is really a degenerate one-bit instruction; (3) **multi-task indexes**, where you serve retrieval, clustering and classification off the same vectors.

Where they are close to cargo cult: writing a lovingly-crafted 40-word instruction for a plain "find the document that answers this question" task and expecting it to move the needle. It will move nDCG by a fraction of a point over the model's default prompt, and you spent an afternoon on it. Instruction *sensitivity* also varies enormously between models — some are trained on hundreds of instruction templates and generalize, others were trained on two prefixes and treat anything else as noise.

**⚠ Trap:** changing the instruction after you have indexed. The instruction is part of the embedding function. If you index 50M passages with instruction A and then start querying with instruction B on the *passage* side, you have effectively changed models. Query-side instruction changes are usually safe to iterate on (you re-embed one short string per request); passage-side changes are a full reindex. Design so that only the query side has a tunable instruction.

**💰 Math — the token cost nobody counts:** a 60-token instruction prepended to every passage in a 100M-chunk corpus adds `100e6 × 60` = **6e9 tokens** of pure overhead. At an API price of $0.13 per million tokens for a large embedding model that is `6e9 / 1e6 × $0.13` = **$780** of instruction text, plus roughly a 15% increase in indexing wall-clock if your chunks average 400 tokens. Keep passage-side instructions short or empty. **📅 Volatile:** verify current embedding prices before quoting them.

### Explain Matryoshka representation learning. What is actually different about the loss?

Matryoshka training makes a **single** embedding whose *prefixes* are themselves valid embeddings. Truncate a 3072-dim MRL vector to its first 256 dimensions and you get a usable 256-dim embedding — not a random projection, not a lossy compression you have to decode, just `v[:256]`, and it still ranks sensibly. That is a genuinely surprising property and the mechanism is simpler than the result suggests.

The trick is entirely in the loss. Instead of computing your contrastive loss once on the full `d`-dimensional vector, you compute it at several nested prefix widths and sum:

```
dims = [64, 128, 256, 512, 1024, 2048, 3072]
loss = 0
for d in dims:
    q_d = normalize(q[:, :d])          # slice, then re-normalize
    p_d = normalize(p[:, :d])
    loss += w_d * info_nce(q_d, p_d, tau)
```

That is the whole idea. Because the 64-dim prefix is penalized directly, gradient descent is forced to pack the most discriminative information into the earliest coordinates — you get an ordering of information by importance, learned rather than imposed. It behaves like a learned, task-aware PCA where the components come out sorted for free, except that unlike PCA it is optimized for your *retrieval* objective rather than for reconstruction variance.

The `w_d` weights are usually uniform or mildly favor the larger dims. Re-normalizing after slicing matters: the prefix of a unit vector is not a unit vector.

**📄 Paper:** Kusupati et al. (2022), **Matryoshka Representation Learning** — nested-prefix multi-loss training that yields one checkpoint usable at many dimensionalities; it replaced the practice of training and maintaining a separate model per embedding size.

This is why OpenAI's `text-embedding-3` family exposes a `dimensions` parameter, and why several open models ship "MRL-enabled" in the card. **⚠ Trap:** applying naive truncation to a model that was *not* MRL-trained. On an ordinary embedder the dimensions carry roughly equal information and are in no meaningful order — truncating 3072→256 there does not degrade gracefully, it degrades catastrophically. Check the model card for MRL support before you truncate anything. The corollary trap: after truncating an MRL vector you **must re-normalize**, or your cosine similarities are silently scaled by the (variable) norm of the retained prefix.

### Do the storage, latency and recall math for truncating 3072 to 256 dimensions on a 100M-chunk corpus.

This is the arithmetic I expect a candidate to do out loud, so let me do it fully.

**Raw storage.** 100M vectors × 3072 dims × 4 bytes (fp32) = `100e6 × 3072 × 4` = **1.229 TB**. At 256 dims: `100e6 × 256 × 4` = **102.4 GB**. A 12× reduction, and the difference between "this does not fit in RAM on any reasonable machine" and "this fits on one large instance."

**Index overhead.** HNSW adds roughly `M × 2 × 4` bytes per vector for the graph links (with `M = 16`, about 128 bytes/vector, plus the layer-0 doubling convention varies by implementation). For 100M that is ~12.8 GB either way — a rounding error at 3072 dims, a meaningful 12% surcharge at 256 dims. Worth noting because at very low dimension the *graph*, not the vectors, starts to dominate.

**Query latency.** Brute-force scan cost is linear in `d`, so 12× fewer bytes to stream is roughly a 12× cheaper distance computation. Under HNSW, the distance computation is not the only cost — graph traversal, cache misses and candidate-heap management contribute — so the realistic end-to-end speedup is smaller, typically **3–6×** on the search phase. But the memory-bandwidth story is the real one: at 3072 dims you touch 12 KB per candidate visited; at 256 dims you touch 1 KB, so far more of your working set stays in L2/L3 and your per-query cache miss rate falls dramatically.

**💰 Cost.** Memory is the dominant line item for an in-memory vector index. 1.23 TB of vectors plus graph needs roughly a 1.5–2 TB RAM footprint, which means several large memory-optimized instances — call it 2 × `r6i.32xlarge`-class at roughly $8/hr each = **$11,700/month**. At 256 dims, ~115 GB fits comfortably on a single 256 GB instance at roughly $2/hr = **$1,460/month**. That is an **$10,200/month saving**, or $122k/year, from one parameter.

**The recall cost.** This is the number you must actually measure, not quote. Published MRL results and vendor reports show graceful degradation — commonly single-digit percentage relative loss in nDCG@10 going from full to a quarter of the dimensions on general benchmarks — but the degradation is **corpus-dependent and gets worse as the corpus grows**, because with 100M candidates the fine distinctions you dropped start to matter. **📅 Volatile:** do not quote a specific published percentage as if it applies to your corpus. Measure it on your own 200-query benchmark at 3072, 1024, 512, 256, 128 and plot the curve; the elbow is usually visible and is your answer.

**🗣 Say this in the room:** "MRL truncation from 3072 to 256 is a 12× storage reduction — 1.23 TB down to 102 GB for 100M chunks — which is the difference between a multi-instance memory-optimized fleet at about $11.7k/month and a single box at $1.5k. The recall cost is a few points and it is corpus-specific, so I'd sweep dimension against nDCG@10 on my own benchmark and pick the elbow. And if I need the last couple of points back, I keep the full vectors on disk and rescore the top 100."

### Explain binary quantization with float rescoring. What recall should I expect?

Binary quantization is the most aggressive compression available and it works far better than it has any right to, for a reason worth internalizing: in high dimensions, **the sign pattern of a vector carries most of the angular information.** Keep one bit per dimension — `1` if the component is positive, `0` otherwise — and you have thrown away all magnitude but retained which orthant the vector points into, and with 1024 dimensions there are enough orthants that this is remarkably discriminative.

The compression: 1024 dims × 4 bytes = 4096 bytes → 1024 bits = **128 bytes**. That is **32×**.

The speed is the better story. Cosine similarity between two binary vectors reduces to **Hamming distance**, which is `popcount(a XOR b)` — on a modern CPU with AVX-512 or ARM NEON that is a handful of instructions per 512 bits, and it is integer work with no floating-point pipeline involved. Reported speedups on the distance kernel itself are in the tens of times.

The recall is where the design gets interesting, and the answer is **rescoring**. Binary search alone loses meaningful recall. So you run a two-stage retrieval entirely inside the vector store:

1. Search the binary index for the top `k × oversample` candidates — typically 100–400 when you want 10, using Hamming distance. Fast, tiny memory footprint.
2. Fetch the **full-precision (or int8) vectors for just those candidates** — from disk, from a separate column, from object storage — and rescore them exactly.
3. Return the true top `k`.

Step 2 touches maybe 200 vectors instead of 100M, so the cost of full precision is now negligible: `200 × 4096 bytes` = 800 KB of I/O. A refinement that costs nothing: keep the **query** in float and compare it against the binary document vectors asymmetrically, which recovers a chunk of the loss because you only quantized one side.

**📐 Numbers you must know:** binary = **32× compression** vs fp32, ~4× vs int8. Vendor and library reports for binary-plus-rescore on strong 1024-dim models cluster around **90–96% of full-precision recall** with an oversampling factor of 3–5×; without rescoring, expect a much larger drop. **📅 Volatile:** these are reported figures on public benchmarks — measure on your corpus, because the loss scales with corpus size and with how tightly clustered your documents are.

**⚠ Trap:** binary quantization on a *low*-dimensional embedding. At 1024 dims you have 1024 bits of signal; at 256 dims you have 256, and the Hamming distance distribution becomes so coarse that ties dominate — huge numbers of documents at identical distance, and your ranking degenerates. **The rule: if you are going to binarize, keep the dimensions high.** A 1024-dim binary vector (128 bytes) beats a 256-dim fp32 vector (1024 bytes) on both size and recall. This is the counterintuitive result that makes the technique worth knowing: **when you must shrink, quantize the precision before you truncate the dimension.**

### int8 or binary? Give me the decision rule.

The rule is a memory-budget ladder, and I would state it as one.

**int8 scalar quantization** maps each float dimension to one byte via a per-vector or per-dimension affine scale: `q = round((x − min) / (max − min) × 255)`. 4× compression from fp32. Distance is computed in int8 with integer SIMD (VNNI on x86 gives you dot products at very high throughput), and because you retain 256 levels per dimension rather than 2, the ranking is nearly indistinguishable from float — typically **>99% of full-precision recall** with no rescoring at all, on most corpora.

**Binary** is 32×, needs oversampling and rescoring to be competitive, and pays off only when memory is genuinely the binding constraint.

So the ladder:

| Corpus size (1024-dim) | fp32 | int8 | binary+rescore |
|---|---|---|---|
| 1M | 4.1 GB | 1.0 GB | 128 MB |
| 10M | 41 GB | 10 GB | 1.3 GB |
| 100M | 410 GB | 102 GB | 12.8 GB |
| 1B | 4.1 TB | 1.0 TB | 128 GB |

Read the decision off the table against your instance sizes. **Under ~10M vectors: use fp32 or int8 and stop optimizing** — 41 GB fits in RAM on a normal box and the engineering time you would spend on binary rescoring is worth more than the money. **10M–100M: int8 is the default**, with fp16 as a lazy alternative if your engine supports it. **Above ~100M, or if you need to serve many tenants' indexes hot simultaneously: binary with float rescoring**, because 12.8 GB versus 410 GB is a categorical change in what architecture is possible — it is the difference between a distributed sharded index and a single node.

**💰 Math:** at 100M vectors, fp32 needs ~512 GB of RAM with index overhead — roughly $16/hr of memory-optimized instance = **$11,700/month**. Binary+rescore needs ~20 GB hot plus the full vectors on cheap storage: one modest instance at ~$0.60/hr = **$440/month**, plus, say, 410 GB on object storage at $0.023/GB/month = **$9.4/month**. That is a **26× cost reduction** for a few points of recall you can partly buy back with a larger oversample.

**⚠ Trap:** quantizing and then never re-measuring the *end-to-end* answer quality. Recall@100 is nearly unchanged under int8, so the vector-store dashboard looks fine — but if your reranker only sees the top 20 and quantization reshuffled ranks 15–25, the chunk that actually contained the answer just fell out of the reranker's window. **Always measure quantization impact at the top-k your downstream stage actually consumes**, not at the k your vector store reports.

### Does a bigger embedding dimension mean a better model? Talk me through it.

No, and the confusion is a good filter question. Dimension bounds *capacity*, it does not confer *quality*. A 4096-dim embedding from a weakly-trained model will lose to a 768-dim embedding from a well-trained one, every time, and there are plenty of public results showing exactly that ordering.

The reasoning: what determines retrieval quality is the training objective, the data, and the amount of pretraining knowledge in the backbone. Dimension only determines how much information the pooling bottleneck *can* carry. Past a point, the bottleneck stops being the binding constraint and extra dimensions carry mostly redundant or noisy directions — which is precisely why MRL truncation works so well. If dimensions 256–3072 carried independent, load-bearing signal, cutting them would be catastrophic. It isn't. That is direct empirical evidence of redundancy.

There is also an active cost to high dimension beyond storage: **ANN recall degrades as dimension grows**. Every graph- and partition-based index relies on the geometry having enough distance contrast to prune the search space. As `d` grows, distances concentrate (the same effect that causes hubness), the contrast between the nearest and the tenth-nearest neighbor narrows, and the index has to explore more of the graph to find the true neighbors. Concretely, to hit the same recall@10 at 3072 dims that you hit at 768, you typically need a higher `efSearch`, which costs latency. So high dimension charges you twice: once in bytes, once in `ef`.

**The rule I use:** treat dimension as a **deployment parameter**, not a quality parameter. Pick the model on measured quality; pick the dimension by sweeping recall against your storage and latency budget on your own corpus. For most production RAG systems the sweet spot lands between **512 and 1024 dimensions**, and I would want a specific measured reason before going above that.

**🗣 Say this in the room:** "Dimension is capacity, not quality — a well-trained 768-dim model routinely beats a badly-trained 3072-dim one, and the fact that Matryoshka truncation degrades gracefully proves most of those extra dimensions are redundant. I choose the model on eval and the dimension on budget, and I'd want measured evidence before going above 1024 because high dimension costs me both bytes and `efSearch`."

### My documents are 5,000 tokens and the embedder maxes out at 512. What do I do?

First, name the failure mode precisely: nearly every embedding implementation **silently truncates** at `max_seq_length`. It does not raise. So a 5,000-token document becomes an embedding of its first 512 tokens, and 90% of your content is unindexed and unretrievable. Your system will confidently report having indexed the document.

The options, in the order I would consider them:

**1. Chunk properly (the right answer ~90% of the time).** Split the document into passages that fit, embed each, index each separately, and carry a `doc_id` + section path in the metadata so you can group or expand at retrieval time. This is not a workaround; it is what retrieval *wants*. A single vector for a 5,000-token document is an average of everything it says, which is a nearly useless representation — the document is about ten topics and the mean of ten topics is close to nothing in particular. Chunking gives you precision, and it gives you citation granularity.

**2. Chunk and pool at the document level for a two-stage design.** Keep chunk vectors for retrieval, and additionally store a mean-of-chunk-vectors document embedding for coarse filtering or clustering. Cheap, and it makes "which documents are about X" a fast query.

**3. Use a long-context embedder.** Several models support 8k+ token inputs (the Jina and Nomic families are the usual examples, and BGE-M3 supports long inputs). This is genuinely useful when the *unit of retrieval must be the whole document* — for example when you are retrieving code files and the caller needs the full file. **📅 Volatile:** verify current max sequence lengths on model cards.

**⚠ Trap — the one that catches people:** assuming a long-context embedder actually *uses* its long context well. A single 8,192-dimensional-input pooled vector still averages everything. Published analyses of long-context embedders consistently find that retrieval quality on a long document is dominated by the earlier portion of the text, and that chunking plus max-over-chunks beats whole-document embedding on most retrieval tasks. A model advertising 32k input length is telling you what it will accept, not what it will represent faithfully.

**The rule I enforce in review:** assert on length at ingestion. Tokenize, compare to `max_seq_length`, and either chunk or raise — never silently truncate. And log a counter for it, because "we truncated 4% of documents" is exactly the kind of thing that goes unnoticed for a year.

### BGE-M3 gives you dense, sparse and multi-vector outputs from one forward pass. How would you actually use that?

The mental model: **one encoder, three read-heads.** You pay for the transformer once and get three different retrieval signals out of the same activations — a pooled dense vector, a learned sparse term-weight vector over the vocabulary, and the full per-token matrix for late interaction. That is a genuinely elegant piece of engineering because the expensive part (the forward pass) is amortized across all three.

**📄 Paper:** Chen et al. (2024), **BGE-M3** — "multi-linguality, multi-functionality, multi-granularity": one model producing dense, lexical-sparse and ColBERT-style multi-vector representations, trained with a self-knowledge-distillation scheme where the three heads teach each other. It replaced the pattern of running a separate dense model, a separate SPLADE model and a separate ColBERT model to get hybrid retrieval.

How I would deploy it, and the honest cost accounting:

- **Dense head** → your normal HNSW/IVF index. 1024 dims. This is the baseline.
- **Sparse head** → an inverted index (Elasticsearch, Vespa, Qdrant's sparse vectors, or a `tsvector`-ish store). This is the head that saves you on exact identifiers — SKUs, error codes, version strings — which is the single most common dense-retrieval failure. Fusing dense and sparse with RRF gets you most of the hybrid benefit for one extra index.
- **Multi-vector / ColBERT head** → this is where you must be careful. Storing per-token vectors means roughly `n_tokens × d` per chunk. A 400-token chunk at 1024 dims fp32 is `400 × 1024 × 4` = **1.6 MB per chunk**, versus 4 KB for the dense vector — a **400× storage blowup**. For 100M chunks that is 160 TB. Nobody does that naively; ColBERT-style deployments use aggressive per-token quantization (2 bits/dim is typical in PLAID-style indexes) plus centroid-based pruning, which brings it down by ~50×, and you still would only apply it as a **reranking stage over ~100 candidates**, not as a first-stage index.

So the production shape I would actually ship: dense + sparse fused with RRF as the retrieval stage, multi-vector as an *optional* reranker where you already have the candidate set small. Which means you index two of the three heads and compute the third on demand.

**⚠ Trap:** enabling all three heads at ingestion because the API makes it a single boolean, and discovering three months later that your storage bill is dominated by ColBERT vectors nobody queries. Turn on what you have measured a use for.

### Design me a two-stage retrieval that exploits Matryoshka embeddings. Where does each vector live?

The design writes itself once you see it as a **coarse index in RAM, exact vectors on cheap storage** — structurally the same move as a covering index plus a heap fetch, which is a pattern you already know.

The architecture:

- **Stage 1 — shortlist.** A 256-dim (MRL-truncated) HNSW index, held entirely in RAM. For 100M chunks: `100e6 × 256 × 4` bytes = 102 GB of vectors + ~13 GB of graph = **115 GB**, which fits on one 256 GB instance. Search with a generous `efSearch` and return the top **200** candidates. Latency: single-digit milliseconds, because the whole working set is memory-resident and each distance touches 1 KB.
- **Stage 2 — exact rescore.** Fetch the full 3072-dim fp32 vectors for those 200 candidates only, and compute exact dot products. Those live wherever is cheapest — an on-disk column store, RocksDB, or S3 with a local LRU. `200 × 3072 × 4` = **2.4 MB** of reads per query. From local NVMe that is sub-millisecond; from object storage you would want a hot cache, because 200 individual GETs would destroy your p99. Return the true top 10.
- **Optional stage 3 — cross-encoder rerank** over the top 25.

Push the compression further and stage 1 becomes binary: 1024-dim binary is 128 bytes/vector, so 100M chunks is **12.8 GB** — it fits in the page cache of a machine you were going to run anyway.

**💰 Math on what this buys:** the naive design (100M × 3072 × 4 = 1.23 TB in RAM, plus graph ≈ 1.25 TB) needs roughly $11,700/month of memory-optimized instances. This design needs one 256 GB instance (~$2/hr = $1,460/month) plus 1.23 TB of NVMe or object storage (~$120/month for gp3-class, ~$28/month for S3). Total ≈ **$1,590/month versus $11,700** — a **7.4× reduction** — and the recall is *higher* than a pure 256-dim system because stage 2 restores exact scoring on the candidates that matter.

**⚠ Trap:** setting the stage-1 shortlist too small. The whole design rests on the assumption that the true top-10 under full precision is contained in the top-200 under truncated precision. That assumption is measurable and you must measure it: compute **recall of the exact top-10 within the truncated top-N** as you sweep N over {50, 100, 200, 500}, on your own corpus. If it is 0.94 at N=200 you have capped your system's recall at 0.94 no matter how good your reranker is. This "shortlist recall ceiling" is the number I ask for whenever someone proposes a multi-stage retriever, and most people have never computed it.

**🏋 Drill:** 30 minutes. Given any MRL-capable embedder and a 50k-document corpus, build the full-precision index and the 256-dim index, then plot shortlist-recall-at-N (N from 10 to 500) for the exact top-10. Pass criterion: you can state the smallest N that achieves ≥0.99 shortlist recall, and you can explain why that number will grow as the corpus grows.
