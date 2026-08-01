### Take me from a Common Crawl WARC file to a training document. What actually happens?

The mental model: Common Crawl is not a dataset, it is a very large tape archive of raw HTTP responses, and roughly 99% of the bytes in it are things you do not want — navigation chrome, cookie banners, ad markup, SEO spam, machine translation, and near-duplicates of the 1% you do want. Everything downstream is a funnel, and the funnel's job is to throw away 99% of the input without throwing away the tail of genuinely rare, high-value text.

Concretely, per monthly crawl you get on the order of 2–3 billion pages across ~90k WARC files, each a gzip-per-record archive of HTTP request/response pairs. Three formats exist: **WARC** (raw responses, ~100 TB compressed per crawl), **WAT** (metadata/links, JSON), and **WET** (Common Crawl's own naive text extraction). Almost nobody serious uses WET — its extraction is crude, keeps boilerplate, and loses structure. RefinedWeb and FineWeb both re-extract from WARC and both report that this alone is a large quality win.

The extraction step itself: given the HTML, identify the main content block and drop the rest. Two libraries dominate.

- **`trafilatura`** — Python, uses a cascade of heuristics plus readability-style fallbacks, and is generally regarded as the highest-quality open extractor. FineWeb uses it. Throughput is roughly on the order of tens of pages/second/core, which is the problem.
- **`resiliparse`** — C++-backed (part of the ChatNoir stack), roughly an order of magnitude faster, somewhat lower extraction quality. When you are processing 100 TB, that constant factor is the difference between a 2-day job and a 3-week job.

**💰 Math:** say you process 2.5e9 pages and `trafilatura` gives you 30 pages/sec/core. That is `2.5e9 / 30 = 8.3e7` core-seconds = **23,000 core-hours** per crawl. On 96-core spot instances at ~$1.50/hour for the machine, that is `23,000/96 × 1.50 ≈ $360` per crawl for extraction — genuinely cheap. But you want 20+ crawls, and you will re-run the pipeline five times as you tune filters, so budget 100× that: `~$36k`, plus egress and storage. The compute is not the bottleneck; the *iteration latency* is. Design for a 1% sample that runs in 20 minutes so you can actually ablate filter decisions.

**⚠ Trap:** treating extraction as a solved preprocessing step. Extractor choice materially changes model quality — it decides whether the model learns from an article or from an article wrapped in "Subscribe to our newsletter · Share on Facebook · Related posts." I insist on versioning the extractor in document metadata, because six months later "why did quality drop" is unanswerable if you cannot tell which extractor produced which shard.

Practical hygiene at this stage: record `url`, `crawl_id`, `warc_record_id`, `extractor_version`, and a content hash on every document. That provenance is what lets you answer a takedown request, debug a contamination report, and re-run a filter without re-extracting.

### Name the actual heuristic filters people use. Which would you keep and which are cargo cult?

There are two canonical rule sets, and being able to name their contents is a cheap credibility signal.

**📄 Paper:** Rae et al. (2021), *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* — the MassiveText quality filters, universally called "the Gopher rules." Applied per document:

- word count between 50 and 100,000
- mean word length between 3 and 10 characters
- fewer than 10% of lines starting with a bullet
- fewer than 30% of lines ending with an ellipsis
- symbol-to-word ratio (for `#` and `...`) below 0.1
- at least 80% of words containing at least one alphabetic character
- must contain at least two of the stop words `the, be, to, of, and, that, have, with`

Plus separate *repetition* filters: fraction of duplicate lines, duplicate paragraphs, and the fraction of characters in the top-k most frequent n-grams — these catch the spam pages that repeat a phrase 400 times.

**📄 Paper:** Raffel et al. (2020), *Exploring the Limits of Transfer Learning* (T5/C4) — the C4 heuristics: keep only lines ending in terminal punctuation; drop pages with fewer than 3 sentences; drop any line containing "javascript"; drop any page containing "lorem ipsum" or a curly brace `{`; and a bad-words list filter.

My opinions, since you asked which are cargo cult:

- **Keep, unconditionally:** the repetition filters. They are the highest-value rules in either set, they catch a genuinely pathological failure mode, and they are cheap. Also keep word-count and mean-word-length — they catch encoding disasters and link farms for free.
- **Keep, with modification:** the terminal-punctuation line filter. It works beautifully for prose and destroys code, tables and structured data. Gate it on document type.
- **Drop or heavily revise:** the C4 curly-brace rule (it deletes essentially all code and most JSON — C4 wanted English prose, you probably do not), and the bad-words list. The bad-words filter is the one I argue about in review most: it has been shown to disproportionately remove text about and by minority communities, including non-toxic text using reclaimed terms, so it damages both fairness and capability. If you need safety filtering, do it with a classifier that scores documents in context, not a substring list.
- **Do not stack them blindly.** Every filter has a false-positive rate. Stacking 15 filters with 2% FPR each removes ~26% of good documents, and the removed tail is systematically the unusual, high-information text you most wanted.

**⚠ Trap:** tuning heuristics by eyeballing the survivors. The only valid evaluation of a filter is a downstream ablation: train two 300M–1B models for a few billion tokens, one on each corpus, and compare on a fixed benchmark suite. Filters that "obviously" help routinely do nothing, and a filter that removes 40% of your data and gains 0.3 points is usually not worth the token loss.

### FineWeb-Edu and DCLM both use a learned classifier instead of rules. How do you build one, and why did it win?

Because heuristics can only express "this document is malformed," and the actual question is "is this document *worth learning from*," which is a semantic judgement no regex encodes. A learned filter is the same move as replacing a hand-tuned spam ruleset with a trained classifier — and it wins for the same reasons.

**📄 Paper:** Penedo et al. (2024), *The FineWeb Datasets* — FineWeb is ~15T tokens of filtered, deduplicated Common Crawl; **FineWeb-Edu** is the ~1.3T-token subset produced by an "educational value" classifier. The recipe is the important part: they had a strong LLM (Llama-3-70B-Instruct) score a few hundred thousand documents on a 0–5 educational-value scale, embedded those documents with a small embedding model, and trained a lightweight classification head on the embeddings. Then they ran that cheap classifier over the whole corpus and kept documents above a threshold. FineWeb-Edu substantially outperformed FineWeb on knowledge and reasoning benchmarks at equal token counts.

**📄 Paper:** Li et al. (2024), *DataComp-LM (DCLM)* — a benchmark where the model and compute are fixed and *only the data pipeline varies*, which is the right experimental design for this problem. Their winning **DCLM-baseline** (~3.8T tokens) used a fastText classifier whose positive class was instruction-formatted, high-quality text (OpenHermes-2.5 and highly-upvoted ELI5 posts) and whose negative class was random Common Crawl. Cheap, linear, extremely effective.

The generalizable recipe:

1. **Define the positive class by example, not by rule.** Pick a corpus that embodies what you want more of. This is where your judgement lives — "educational" and "instruction-like" are different targets and produce different models.
2. **Label at scale with an LLM, then distill into something cheap.** You cannot run a 70B model over 100 TB. You can run it over 500k documents and train a fastText or embedding+linear-head model that runs at ~100k documents/sec/core.
3. **Pick the threshold by ablation, not by intuition.** The threshold is a token-count/quality dial. Aggressive filtering can leave you data-constrained.

**💰 Math:** the distillation is the whole economic argument. Scoring 2e10 documents with a 70B model at even 1,000 tokens/doc is `2e13` tokens of inference — at `2 × 70e9 × 2e13 = 2.8e24` FLOPs, that is *four times* the training compute of the model you are building. Scoring 500k documents is `2.8e24 × (5e5/2e10) = 7e19` FLOPs — utterly negligible. The classifier then runs at fastText speeds: 2e10 documents at 100k/sec/core is `2e5` core-seconds ≈ **56 core-hours**. That asymmetry — expensive teacher on a sample, cheap student on everything — is the pattern.

**⚠ Trap:** the classifier learns your positive class's *surface form*, not its quality. If your positives are all Wikipedia-styled, you will filter for encyclopedic register and delete excellent conversational writing, forum answers and code comments. Check what a threshold sweep actually removes by reading 50 rejected documents at each level. I have seen an "educational" filter quietly delete almost all source code because code does not look like a textbook.

### At corpus scale, how do you handle language identification, PII and toxicity?

**Language ID.** The standard tool is a fastText language-identification model (the `lid.176` family covering 176 languages), which classifies a document in microseconds. You keep documents above a confidence threshold — 0.65 is a common English cutoff and the value matters a lot. The failure modes, in the order they will bite you:

- **Short documents are unreliable.** A 20-word document is near-coinflip between related languages. Filter by length *before* language ID, not after.
- **Code-switched and multilingual documents get dropped entirely.** A page with English commentary around a Hindi quotation may score below threshold in both. This systematically deletes exactly the data you would need for cross-lingual ability.
- **Code and math score as arbitrary languages.** If you run language ID over your code corpus you will delete most of it.
- **Threshold choice is a quality/quantity dial with a fairness edge.** A high English threshold preferentially removes non-standard varieties and non-native writing.

**PII.** At corpus scale you get one realistic option: high-precision regex/pattern detection for structured identifiers — emails, phone numbers, IP addresses, credit-card numbers (with Luhn validation to cut false positives), national ID formats — with **replacement by a type-preserving placeholder** rather than deletion, so you do not create ungrammatical text. Named-entity models for unstructured PII (names, addresses) exist but are far too slow and too imprecise to run over 100 TB, and names are not removable from web text anyway without destroying the corpus. Be honest about the limit: you are removing *contactable* identifiers, not anonymizing.

**⚠ Trap:** doing PII redaction *after* tokenization or after packing. Once documents are packed into fixed-length token sequences you have lost the document boundaries needed to reason about a record, and rewriting text means re-tokenizing everything downstream. PII redaction belongs immediately after extraction, before dedup — otherwise your dedup hashes are computed on text you are about to change, and your dedup becomes invalid.

**Toxicity and safety.** Two defensible positions and you should be able to argue both. Filter hard, and your model is safer out of the box but measurably worse at recognizing and refusing harmful content, because it has never seen any — this is a real result, and it also means your safety classifiers trained on that model are weaker. Filter lightly, and you rely on post-training to handle it, which is where alignment actually happens anyway. **My position:** filter aggressively for the genuinely illegal categories (CSAM detection via hash-matching against known-hash lists is non-negotiable), filter moderately for extreme-toxicity outliers, and leave the middle of the distribution alone, because refusal behavior is much more effectively taught in post-training than by data absence.

**🗣 Say this in the room:** "Language ID with fastText after a length filter, high-precision pattern-based PII redaction before dedup so the hashes stay valid, and safety filtering only at the extreme tail — because a model that has never seen harmful content can't recognize it, and refusal is a post-training problem."

### Explain exact-substring deduplication. Why does it need a suffix array?

The mental model: near-duplicate detection at the *document* level misses the most common form of duplication on the web, which is a shared *span* — the same three paragraphs of a press release embedded in 400 otherwise-different news pages. Document-level MinHash will not flag those because the documents are genuinely different overall. Exact-substring dedup finds and removes any span of at least `k` tokens that appears more than once anywhere in the corpus.

**📄 Paper:** Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better* — introduced the two-pronged approach (`ExactSubstr` via suffix arrays plus `NearDup` via MinHash), showed that C4 contains sentences repeated tens of thousands of times, and demonstrated that dedup reduces verbatim memorized emissions by an order of magnitude while *improving* held-out perplexity at fixed step count.

The mechanism: concatenate the entire tokenized corpus into one giant array `S`. Build a **suffix array** — the sorted order of all suffixes of `S`. Then any repeated substring appears as a set of *adjacent* entries in that sorted order, so you find all maximal repeats in a single linear scan comparing each adjacent pair's longest common prefix. The threshold in the original work is **50 BPE tokens**: any span of ≥50 tokens occurring more than once is removed from all but one occurrence.

The reason the suffix array matters is complexity. Naive pairwise comparison over `n` documents is `O(n²)` — at 2e10 documents that is 4e20 comparisons, which is not a big number, it is an impossible number. Suffix-array construction is `O(n log n)` in the total corpus length, and there are parallel/distributed constructions. It is still brutal: the array itself needs 8 bytes per token, so a 15T-token corpus needs **120 TB** just for the index, forcing a partitioned build with a merge — this is the real engineering content of the step and it is a distributed-sort problem you already know how to reason about.

**⚠ Trap:** setting `k` too low. At `k = 10` tokens you start deleting common idioms, license headers that legitimately belong on every file, and standard mathematical statements. At `k = 50` you are removing genuine duplication. Also: exact-substring removal *cuts documents mid-sentence*, leaving fragments. Either drop documents that lose more than some fraction of their content, or accept the fragments deliberately — do not discover it later in your samples.

### Derive the MinHash-LSH parameters for me. If I want to catch 80%-similar documents, what do I set?

The mental model in one line: MinHash converts "how similar are these two sets" into "how often do two cheap hashes agree," and LSH converts "compare everything to everything" into "bucket things so only plausible pairs are ever compared." It is the same trick as a bloom filter plus a hash join, and the math is fully derivable.

**Step 1 — shingling.** Represent each document as the set of its n-grams (word 5-grams is the standard choice; FineWeb and many others use 5).

**Step 2 — MinHash.** For a random permutation `h` of the universe of shingles, `min(h(S))` is equal for two sets with probability exactly the Jaccard similarity `J(A,B) = |A∩B| / |A∪B|`. Proof in one line: the minimum over `A ∪ B` is equally likely to be any element, and it agrees iff that element is in the intersection. So computing `P` independent min-hashes gives you an unbiased estimator of `J` with standard error `1/sqrt(P)`.

**Step 3 — LSH banding.** Split the `P` hashes into `b` bands of `r` rows each (`P = b·r`). Two documents become candidates if they match on *all* `r` hashes of *at least one* band. Probability of matching a given band: `s^r`. Probability of matching no band: `(1 − s^r)^b`. So:

```
P(candidate | Jaccard = s) = 1 − (1 − s^r)^b
```

This is an S-curve. Its inflection — the effective threshold — is approximately `t ≈ (1/b)^(1/r)`.

**Worked answer to your question.** Want `t = 0.8`. Pick `r = 13`, then `b = 1/t^r = 1/0.8^13 = 1/0.0550 = 18.2`, so `b = 18`, `P = 234` hashes. Check the curve: at `s = 0.9`, `0.9^13 = 0.2542`, `1 − (1−0.2542)^18 = 1 − 0.7458^18 = 1 − 0.0053 = 99.5%` caught. At `s = 0.7`, `0.7^13 = 0.00969`, `1 − 0.99031^18 = 1 − 0.8394 = 16%` caught. At `s = 0.5`, `0.5^13 = 1.22e-4`, `1 − (1−1.22e-4)^18 = 0.22%`. That is a sharp, well-behaved filter.

For reference, the commonly-used FineWeb-style configuration is `P = 112` hashes as `b = 14` bands of `r = 8`, giving `t ≈ (1/14)^(1/8) = e^(−ln14/8) = e^(−0.330) = 0.719` — a ~0.72 threshold, deliberately looser than 0.8.

```python
import numpy as np
def band_signature(minhashes, b, r):
    # minhashes: (n_docs, b*r) uint64
    sigs = minhashes.reshape(len(minhashes), b, r)
    # one bucket key per band; tuple-hash the r values
    return [ [hash(tuple(sigs[i, j])) for j in range(b)] for i in range(len(sigs)) ]

def curve(s, b, r):  # sanity-check your params before you run 100 TB through them
    return 1 - (1 - s**r)**b
```

**⚠ Trap:** raising `P` to "be more accurate" without changing `b` and `r`. Accuracy of the *estimate* improves, but the threshold is set entirely by the `b`/`r` split — going from 112 to 512 hashes at the same `b` changes `r` and therefore silently moves your threshold. Always compute the curve and plot it before you run.

**💰 Math:** cost is `P` hashes per document. At 2e10 documents and 234 hashes over ~1,000 shingles each, that is `2e10 × 234 × 1000 = 4.7e15` hash operations. At ~1e8 hashes/sec/core that is `4.7e7` core-seconds = **13,000 core-hours**, about $200–400 of spot compute. The expensive part is not the hashing, it is the shuffle: you must group ~`2e10 × 18 = 3.6e11` (band, bucket) keys globally. That is a distributed sort, and it is the reason this step is a Spark/Ray/`datatrove` job and not a Python script.

### What is SemDeDup, and when does it earn its keep over MinHash?

MinHash catches lexical near-duplicates: documents that share literal n-grams. It is blind to *semantic* duplication — the same news event written up by 200 outlets in 200 different phrasings, or the same StackOverflow answer restated across a dozen tutorial blogs. Those documents have low Jaccard similarity and near-identical information content, so the model learns nothing from the 199 extra copies but pays full compute for them.

**📄 Paper:** Abbas et al. (2023), *SemDeDup: Data-Efficient Learning at Web-Scale through Semantic Deduplication* — embed every document, cluster with k-means, then within each cluster remove one of every pair whose cosine similarity exceeds a threshold, keeping the member farther from the centroid (i.e. preferring the more distinctive example). They reported removing ~50% of LAION with no performance loss, and meaningful fractions of C4 with speedups at equal quality.

The reason it needs clustering is the same `O(n²)` problem as before: you cannot compute all pairwise cosines over 2e10 embeddings. K-means with, say, 50,000 centroids reduces it to `O(n²/k)` within-cluster comparisons — the same shard-then-compare structure as LSH, using a learned partition instead of a hash partition.

The honest cost accounting is why this is not universal:

**💰 Math:** you must embed the entire corpus. A small embedding model at ~100M params over 2e10 documents × 500 tokens = `1e13` tokens costs `2 × 1e8 × 1e13 = 2e21` FLOPs. On H100s at 40% MFU that is `2e21 / (989e12 × 0.4) = 5.06e6` GPU-seconds = **1,400 GPU-hours** ≈ $3,500 at $2.50/hr. Then storage: 2e10 embeddings at 768 dims × 2 bytes (fp16) = **30 TB** of vectors, which you then have to k-means. This is a real GPU-cluster job, unlike MinHash which is a CPU job. It earns its keep when you are data-*rich* and compute-constrained — you want to delete tokens to spend compute better — and it does not earn its keep when you are data-constrained, because it deletes tokens you need.

**My rule in review:** exact-substring → MinHash → then SemDeDup only if a downstream ablation at 1B scale shows a win. It is the highest-cost, lowest-certainty step in the dedup stack.

### Here's a design my team proposed: we shard the corpus by crawl month, run MinHash dedup per shard in parallel, then tokenize. Tear it apart.

This has two independent bugs and they are both the classic ones, so I am glad you asked it this way.

**Bug 1 — deduping shards independently does not deduplicate the corpus.** Dedup is a *global* set operation. If a document appears in 12 monthly crawls, per-shard dedup keeps one copy per shard: you have removed within-shard duplicates and kept 12 copies. Since the whole point of the web crawl is that popular pages are re-crawled repeatedly, the cross-shard duplication rate is *higher* than the within-shard rate. You have done the expensive part of the work and captured the smaller half of the benefit.

The fix is not "don't shard" — you must shard, the data does not fit anywhere else. The fix is to shard by a key derived from the *content*, so identical and near-identical content lands in the same shard by construction. For exact dedup, shard on `hash(document)`. For MinHash-LSH, the natural partition is the LSH bucket itself: emit `(band_id, bucket_hash) → doc_id` pairs from every worker, shuffle globally on that key, and do the candidate comparison inside the reducer. That is a standard map-shuffle-reduce and it is why every real dedup pipeline is a Spark or Ray job. The shuffle is the point; if there is no global shuffle in your design, your dedup is not global.

Second-order detail: after the reducer produces duplicate *pairs*, you still need a global decision about which copy to keep. Building the full connected components of the duplicate graph over 1e10 nodes is expensive; the standard cheat is to keep the document with the smallest ID in each cluster, computed with a union-find over the pair list or a couple of rounds of label propagation. Accept the approximation and move on.

**Bug 2 — deduping after tokenization, or letting tokenization precede filtering.** Tokenization is lossy for these purposes and it is the wrong representation:

- Two documents differing only in whitespace or Unicode normalization produce different token sequences, so token-level near-dedup under-detects relative to text-level.
- Any subsequent text edit (PII redaction, a filter change, boilerplate stripping) invalidates the token stream and forces full re-tokenization of everything.
- You lose the ability to re-run with a different tokenizer, which you will need, because tokenizer training should come *after* the mix is finalized so that the tokenizer's merges reflect the real distribution.

**⚠ Trap (the named version):** *dedup must be global, on text, before tokenization and after PII rewriting.* Any pipeline that violates that ordering has a silent correctness bug that shows up as unexplained memorization and an inflated token count.

**The corrected pipeline order:** extract → language ID + length filter → PII rewrite → heuristic filters → **global exact-substring dedup** → **global MinHash near-dedup** → quality classifier → decontamination → *then* tokenizer training on a stratified sample → tokenize → shard and pack.

**🗣 Say this in the room:** "Per-shard dedup isn't dedup — cross-crawl duplication is the majority of duplication. You have to shuffle globally on a content-derived key, which for MinHash is the LSH bucket. And it has to happen on text before tokenization, or every text edit invalidates your token shards."

### Give me the tour of the named public corpora. What is each one, and what did each prove?

This is a name-recognition question and the interviewer wants to hear that you know what each *contributed*, not just that it exists.

- **The Pile** — **📄 Gao et al. (2020)**, EleutherAI. 825 GiB across 22 curated subsets (PubMed Central, arXiv, GitHub, FreeLaw, StackExchange, Books3, and more). Its contribution was the thesis that **diversity of curated sources beats raw web volume**, and it was the first public dataset good enough that outside groups could train credible models. Its Books3 subset was later removed over copyright, which is itself an important lesson about corpus provenance.
- **RefinedWeb** — **📄 Penedo et al. (2023)**, TII/Falcon. Argued the opposite and won: **properly filtered and deduplicated web data alone can match or beat curated mixes**. 5T tokens produced, 600B publicly released. This flipped the field's default from "curate sources" to "filter the web hard."
- **Dolma** — **📄 Soldaini et al. (2024)**, AI2. ~3T tokens, and crucially released with **the full toolkit and every filtering decision documented** — its contribution is reproducibility and openness, which is why it is the corpus of choice for research on data itself (and it underpins the OLMo models).
- **FineWeb / FineWeb-Edu** — **📄 Penedo et al. (2024)**, HuggingFace. ~15T tokens, plus the ~1.3T educational-classifier subset. Contribution: an ablation-driven methodology — every filter justified by a small-model training run — and the demonstration that a learned quality classifier beats heuristics decisively.
- **DCLM / DCLM-baseline** — **📄 Li et al. (2024)**. A *benchmark* first and a dataset second: fix the architecture and compute, vary only the data. Contribution: made data curation a measurable, competitive research problem, and produced a ~3.8T-token fastText-filtered corpus that set the bar.
- **Nemotron-CC** — NVIDIA (2024). Contribution: showed you can **rescue** the documents your quality classifier rejects by having an LLM rewrite them, and combine real plus synthetically-rephrased data to get several trillion additional usable tokens rather than discarding them. It is the main public data point for "synthetic rephrasing at pretraining scale works." **📅 Volatile:** exact token counts vary by release; verify.
- **The Stack / The Stack v2** — **📄 Kocetkov et al. (2022)** and the BigCode follow-up. Permissively-licensed source code, with v2 built on the Software Heritage archive at far larger scale. Contribution: license-aware code data with an opt-out mechanism, which made "where did your code data come from" an answerable question.

**🗣 Say this in the room:** "The arc is Pile → RefinedWeb → FineWeb/DCLM: from 'curate diverse sources' to 'filter the web extremely hard' to 'let a learned classifier decide, and prove every filter with an ablation.' DCLM is the one I'd cite as changing the methodology, because it made data a controlled experiment."

### Design the data pipeline as infrastructure. 100 TB of WARC, a deadline, and my money. What do you build?

This is where your backend background is a genuine advantage over an ML-first candidate, so answer it like the distributed-systems problem it is.

**Shape of the job.** Almost every stage is embarrassingly parallel per document, punctuated by two global shuffles (exact dedup, MinHash dedup). So: a map-heavy pipeline with two barrier points. That immediately tells you the architecture — a batch framework with a real shuffle, not a queue of workers.

**Concrete stack I would defend:**

- **Storage:** object store (S3/GCS), documents as **compressed JSONL in ~256 MB–1 GB shards** or Parquet if you want columnar predicate pushdown for filtering. Never one file per document — you will spend more on request overhead than on compute. `zstd` over gzip: ~3–5× faster decompression at comparable ratio, and decompression is a real cost when you re-read 100 TB five times.
- **Compute:** Spark or Ray Data for the shuffle stages; plain spot-instance fan-out for the map stages. `datatrove` (HuggingFace) is the open pipeline that encodes most of the above and is a legitimate answer to "what would you use" — it has the Gopher/C4 filters, MinHash, and a Slurm/local executor built in.
- **Orchestration:** whatever you already run. This is a DAG of batch stages with checkpointed intermediate outputs; the important property is **every stage's output is durable and re-runnable independently**, because you will re-run stage 6 forty times.
- **Spot instances everywhere.** Every stage is idempotent and restartable at shard granularity, so preemption costs you one shard. This is a 60–70% cost saving and there is no reason not to take it.

**The three things I would insist on in design review:**

1. **A 0.1% sample that flows through the entire pipeline in under 30 minutes.** Your iteration speed on data decisions *is* your model quality. If testing a filter change takes a day, you will test five changes instead of a hundred.
2. **Per-stage document counters and byte counters, emitted as metrics.** Every stage reports in/out/dropped-by-reason. When your token count comes out 30% low, the answer must be a dashboard query, not an investigation. This is exactly the observability discipline you would apply to a Kafka pipeline, and it is routinely absent from data pipelines built by ML teams.
3. **Content-addressed, versioned outputs.** Stage outputs keyed by `(stage, config_hash, input_hash)`. You get free caching, free reproducibility, and the ability to answer "which corpus version trained checkpoint 4700."

**💰 Math:** rough end-to-end for 100 TB WARC → ~1T clean tokens. Extraction ~25k core-hours, filtering ~10k, MinHash ~15k plus shuffle, exact-substring dedup ~20k plus 100+ TB of intermediate index. Call it 80k core-hours; on 96-vCPU spot at ~$0.60/hr that is `80,000/96 × 0.60 ≈ $500` of CPU — but the shuffle I/O and intermediate storage dominate: 5 stages × 100 TB read+write, at S3 pricing plus a few hundred TB-months of intermediate storage, lands you in the **$15k–40k** range for one full pass. Multiply by the number of full passes you will actually do (five, if you are disciplined). The lesson: **storage and shuffle dominate, not CPU** — which is the opposite of what people budget for.

**⚠ Trap:** designing this as a streaming system. It is a batch system. Streaming buys you nothing (there is no latency requirement) and costs you the ability to do global operations, restart deterministically, and reproduce a corpus version.

### Our model jumped 12 points on a public benchmark after a data refresh. Everyone is excited. What do you do?

I assume contamination until proven otherwise, and I say so out loud before anyone puts it in a slide, because the base rate for "large unexplained jump on a public benchmark after ingesting more web data" is overwhelmingly leakage.

The reasoning: a 12-point jump from a *data* change with no architecture or compute change is a suspiciously large effect. Real data-quality wins at fixed compute look like 1–4 points across a *correlated set* of benchmarks. Contamination looks like a large jump on one or two specific benchmarks with nothing moving elsewhere. That signature — narrow, large, uncorrelated — is diagnostic.

**My decision procedure, in order:**

1. **Check the correlation structure.** Did the sibling benchmarks that measure the same capability move? If MMLU jumped 12 and ARC/HellaSwag/AGIEval are flat, that is not a capability change.
2. **Run the n-gram overlap scan.** Take every eval instance, and for each, check whether any 13-gram of the question (and separately, the *answer*) appears in the training corpus. GPT-3 used 13-gram overlap as its decontamination criterion; 8–13 grams is the normal range. Report the contaminated fraction, and re-score the benchmark on the clean subset only. If the jump is concentrated in contaminated items, you are done.
3. **Check for canaries.** Every serious eval set should embed a canary string — BIG-bench pioneered this with a fixed UUID that dataset authors were asked to include and corpus builders to filter on. If your corpus contains the canary GUID, the eval data is literally in your training set. This is a boolean test that takes minutes.
4. **Test for memorization directly.** Give the model the first half of a benchmark question and let it complete. If it reproduces the exact question *and the exact answer options in the exact order*, it memorized the item. Compare completion likelihood on the eval set against a held-out set drawn from the same distribution — a big log-prob gap on the eval set is a memorization signature. There is a clean published variant of this idea: check whether the model assigns higher likelihood to the *canonical ordering* of a multiple-choice item than to a shuffled ordering, which no non-contaminated model should do.
5. **Confirm with a private eval.** The reason you maintain an internal, never-published eval set is precisely this moment. If the private set did not move, the public jump is not real.

**⚠ Trap:** n-gram decontamination gives you false comfort. It catches verbatim copies and misses paraphrases, translations, reformatted versions, and solutions posted on a tutorial site under different wording. It is a floor on contamination, never a measurement of it. State this explicitly when you report a "decontaminated" number.

**🗣 Say this in the room:** "Before we celebrate, I'd check three things: whether correlated benchmarks moved, the 13-gram overlap rate between the eval set and the new corpus, and whether the private held-out set moved. A large narrow uncorrelated jump after a data refresh is leakage until proven otherwise, and n-gram decontamination is a floor on contamination, not a measurement."

### Walk me through building decontamination into the pipeline, including the part you can't fix.

The mental model: decontamination is a **join** between your corpus and the union of every eval set you will ever report, and like any join at this scale it is an index-build plus a probe. It is not conceptually hard; it is operationally hard because the right-hand side keeps growing and half of it is not public.

**Mechanism.**

1. **Assemble the eval registry.** Every benchmark you report — plus every benchmark your competitors report, because you will be compared on those — plus your internal held-out sets. Store questions, answers, and any few-shot exemplars.
2. **Build an n-gram index over the eval side.** For 13-grams over, say, 500k eval instances of 200 tokens each, that is ~1e8 n-grams — comfortably a bloom filter of a few hundred MB with a tiny false-positive rate, which is exactly the right structure since a false positive costs you one deleted document and a false negative costs you a contaminated benchmark. Size it: 1e8 items at 1% FPR needs ~10 bits/item = 120 MB. Take 0.1% and use 15 bits/item = 180 MB. Cheap.
3. **Probe during the filter stage.** For each training document, slide a 13-gram window and check the filter. Standard policy: if any eval n-gram hits, remove the *matching span* plus surrounding context, or drop the document if hits exceed a threshold. Dropping whole documents on a single 13-gram hit is too aggressive — a 13-gram from a Wikipedia sentence that a benchmark also quoted is a legitimate document.
4. **Insert your own canaries.** Any eval set you create gets a unique random GUID embedded in every instance, published with the set. Then your own future corpora can filter on it, and so can everyone else's. This costs nothing and it is the single highest-leverage thing an eval author does.
5. **Record and report.** Log contaminated-document counts per eval set as a first-class metric, and publish the decontamination method with any benchmark number.

**The part you cannot fix, and you should say this without hedging:**

- **Paraphrase and translation contamination** passes any n-gram filter. A GSM8K problem restated in different words, or in Spanish, is invisible to the join.
- **Solution contamination.** You filtered the question; the corpus contains a blog post working through the answer.
- **Future contamination.** You cannot decontaminate against a benchmark that will be published after your corpus is frozen — and every static public benchmark is contaminated within roughly a year of release, because the internet writes about it and the next crawl ingests that writing.
- **Provider contamination.** If any part of your pipeline used an API model to generate or filter data, that model's training set is unknown to you and may contain the evals.

The structural response is not better filtering, it is **eval hygiene**: a private held-out set that never touches the internet, rotating/regenerated benchmark instances, and evaluating on tasks created after the model's data cutoff. That is why time-stamped and continuously-refreshed evals became the credible ones.

**💰 Math on the cost of getting it wrong:** a contaminated benchmark leads to a wrong model-selection decision. If you ship the contaminated checkpoint over a genuinely better one, and the genuinely better one would have raised task success by 2 points on a feature handling 200k requests/day where each failure costs a $0.40 human fallback, that is `200,000 × 0.02 × 0.40 = $1,600/day = $584k/year` in avoidable cost. Decontamination is a bloom filter and a day of engineering. The ROI is absurd and it is still routinely skipped.
