### Give me your shortlist of embedding models and how you'd narrow it for a specific product.

I'll give you the map first, then the filter, because reciting names without a selection procedure is exactly the failure mode this question is designed to catch.

**Proprietary APIs.** OpenAI `text-embedding-3-small` and `-large` (1536 / 3072 dims, MRL-enabled via a `dimensions` parameter, extremely cheap, strong general-purpose baseline). Cohere `embed-v3`/`v4` (notable for native int8 and binary output types, and for a documented input-type distinction between search queries and search documents — the asymmetry marker as a first-class API field). Voyage (the family that specifically markets domain variants — code, legal, finance — and is a common pick when someone has already tried a general model and hit a domain ceiling). Google's Gemini embedding models. **📅 Volatile:** names, dimensions, prices and the relative ordering of all of these move every few months — verify against current docs before your loop.

**Open weights.** The **E5** family including multilingual-E5 (the `query:`/`passage:` prefix convention, the two-stage weak-then-supervised recipe). **BGE** including **BGE-M3** (dense + sparse + multi-vector from one pass, strong multilingual, long input). **GTE** (Alibaba). **Nomic Embed** (open training data and code, long context). **Jina** v2/v3 (long context, task-specific LoRA adapters on one backbone). **Qwen3-Embedding** and the E5-Mistral lineage as the LLM-based tier. **Stella** and **NV-Embed** as other leaderboard-visible LLM-derived embedders. **📅 Volatile:** same caveat, doubly so for the LLM-based tier where new checkpoints appear monthly.

Now the filter, which is the actual answer. I run six gates in this order and stop as soon as I have two or three survivors:

1. **License and data residency.** If the product is enterprise legal or health, an API that retains inputs may be disqualifying before quality is even discussed. This gate eliminates candidates faster than any other and people always run it last.
2. **Language coverage.** If 30% of the corpus is Japanese, English-only models are out — not "worse," out.
3. **Max sequence length vs. your chunk size.** A 512-token model and 800-token chunks is a silent-truncation bug waiting to happen.
4. **Dimension and quantization support**, against your storage budget from the corpus-size math.
5. **Asymmetry support** — does the model have a query/passage distinction or instruction conditioning? For retrieval, this matters more than a point of MTEB.
6. **Then, and only then, measured quality on my own 200-query benchmark.**

**🗣 Say this in the room:** "I don't pick from the leaderboard. I run license, language, max sequence length, dimension and asymmetry support as hard gates first, which usually leaves me two or three candidates, then I run all of them against my own in-domain benchmark and pick on measured nDCG@10 with cost per million chunks as the tiebreaker."

### How do you read MTEB without being fooled by it?

You read it as a **screening tool that tells you which models to test, and never as a ranking that tells you which model to ship.** The gap between those two uses is where careers go wrong.

What MTEB is: a broad benchmark aggregating dozens of datasets across retrieval, reranking, clustering, classification, STS, summarization and pair classification, with an overall average score. MMTEB extended it to a much larger multilingual set. It has been genuinely valuable — before it, model comparison was anecdotal.

**📄 Paper:** Muennighoff, Tazi, Magne & Reimers (2022), **MTEB** — the Massive Text Embedding Benchmark; it replaced the practice of evaluating sentence embedders on a handful of STS datasets, which had made models look interchangeable when they were not.

Now the four reasons the leaderboard misleads, which is what the interviewer is fishing for:

**1. Overfitting to the benchmark.** MTEB's constituent datasets are public and their training splits are in everyone's training mix. Some models are explicitly trained on the training portions of MTEB tasks. That is not cheating by the rules, but it means the score partially measures "was this dataset in your training data," which does not transfer to your private corpus.

**2. The average hides the shape.** The overall score aggregates classification and clustering tasks that have nothing to do with retrieval. A model can rank third overall and eleventh on retrieval. **If you are building RAG, look only at the retrieval sub-score, and ideally only at the BEIR-style zero-shot retrieval tasks.**

**3. Domain mismatch.** MTEB retrieval is heavily web, Wikipedia, scientific abstracts and forum text. If your corpus is insurance policy PDFs, internal Slack, or Terraform modules, the ranking has weak predictive power. This is exactly why BEIR was created — Thakur et al. (2021) built it as a *zero-shot* heterogeneous benchmark precisely to show that models tuned on MS MARCO do not generalize uniformly.

**4. It says nothing about your constraints.** A model 1.5 points ahead that costs 20× more to index and adds 40 ms to p99 is not the better model for your system.

**⚠ Trap:** the "we picked the #1 model on MTEB" answer in a design round. It signals you have never validated an embedder against your own data. The correction that gets you credit is to say what MTEB *is* good for: eliminating obviously-weak candidates and telling you where the frontier roughly is.

**📐 Numbers you must know:** the spread between the top model and the tenth model on MTEB retrieval is typically **1–3 points of nDCG@10**. The spread between a general model and the same model fine-tuned on in-domain pairs is routinely **5–15 points**. That ratio is the entire argument of this section: your time is better spent on domain adaptation than on leaderboard shopping.

### Walk me through building a 200-query in-domain benchmark. What exactly is in it and where do the labels come from?

This is the artifact that makes every other decision in the section possible, and building one is a two-day job that most teams never do. I would treat it as the first deliverable of any RAG project, before the retriever exists.

**Composition — 200 queries, deliberately stratified**, not sampled uniformly:

- **~80 head queries** taken from real logs (search logs, support tickets, the "what did people actually ask" table). If there are no logs yet, get them from the domain experts who will be your users.
- **~60 torso/tail queries** — the long tail is where retrieval actually breaks, and uniform sampling from logs will under-represent it because by definition each tail query is rare.
- **~30 adversarial queries** chosen to hit known dense-retrieval weaknesses: exact identifiers (SKU, error code, contract number), negation ("policies that do *not* require a deductible"), near-duplicate distinctions (2023 vs 2024 version of the same policy), acronyms, and multi-hop questions.
- **~30 "no good answer exists" queries** — this bucket is the one people skip and it is the one that catches hallucination. You need to know whether your system retrieves garbage confidently when the corpus genuinely does not contain the answer.

**Labels.** Aim for **graded relevance** (0 = irrelevant, 1 = partially relevant, 2 = fully answers), not binary, because nDCG needs grades to be meaningful. Get them by **pooling**: run three or four different retrievers (BM25, two dense models, a hybrid) over each query, take the union of the top 20 from each, dedupe, and have a human judge only that pool — typically 40–60 candidates per query. That is 200 × 50 = **10,000 judgments**, which is roughly 15–25 hours of a domain expert's time at 5–8 seconds each. That is the real cost and you should say it out loud, because it is what makes this a *decision* rather than a formality.

Two accelerants that are legitimate. **LLM-assisted pre-labeling:** have a strong model grade the pool, then have the human review only the disagreements and the borderline cases — cuts human time by roughly 60–70%. And **measure the LLM judge against the human on a 500-judgment sample** before trusting it; report the agreement rate.

**⚠ Trap:** generating both the queries *and* the labels with an LLM from your chunks, then reporting that your retriever gets 0.95 recall. You have built a benchmark that measures "can the retriever find the chunk the query was generated from," which is a much easier task than real retrieval and rewards exactly the lexical-overlap artifacts you were trying to avoid. Synthetic data is fine for *training*; the *evaluation* set needs real queries, or at minimum human-reviewed ones.

**⚠ Trap (the second one):** freezing the benchmark and never touching it. Query distributions shift. I re-sample 20% of the head queries from recent logs every quarter, keep the adversarial and no-answer buckets stable as a regression suite, and version the whole thing in git with the judgments as a checked-in JSONL.

**🗣 Say this in the room:** "Before I choose a model I build a 200-query in-domain set — 80 head from logs, 60 tail, 30 adversarial including exact identifiers and negation, 30 with no correct answer — graded 0/1/2 by pooling the top-20 from four different retrievers and having a domain expert judge the union. That's about 10,000 judgments and 20 hours of expert time, and it is the cheapest insurance in the whole project."

### Recall@k, MRR, nDCG — which one do you optimize and why?

You optimize the one that matches **what the next stage in your pipeline consumes**, and the reason people get this wrong is that they pick a metric before they have drawn their pipeline.

**Recall@k** — did any relevant document make it into the top `k`? This is the right metric for the **first stage** of a multi-stage retriever, because the first stage's only job is not to lose the answer. If you retrieve 100 and rerank to 5, then recall@100 is your ceiling: no reranker can recover a document that was never retrieved. I care about recall@100 to three decimal places and I barely care about the ordering within those 100.

**nDCG@10** — graded, position-discounted. This is the right metric for **the final ordering that reaches the LLM**, because it rewards putting the best chunk first and it uses graded relevance, so "partially relevant at rank 2" scores differently from "fully relevant at rank 2." It is the standard for comparing models.

**MRR@10** — reciprocal rank of the *first* relevant result. Right when there is exactly one correct answer and the user (or the model) reads top-down and stops — a "find the function definition" code search, or a FAQ lookup. Wrong when multiple documents are relevant, because it ignores everything after the first hit.

**The senior answer, and the one I would actually give:** I track **recall@k_retrieve** and **nDCG@k_context** as a pair, because they measure the two different failures. And then I say the uncomfortable part out loud: **none of these are the metric the business cares about.** The metric that matters is end-to-end answer correctness — whether the generated response was right and grounded. Retrieval metrics are a *proxy* that is fast and cheap to compute, and their value is that they let you iterate in seconds instead of dollars. But they can move in the wrong direction relative to answer quality: a change that raises nDCG by putting five near-duplicate chunks at the top can lower answer quality because the context window now contains one fact repeated five times.

**📐 Numbers you must know:** the standard production shape is **retrieve 50–100, rerank to 5–10**. So the two numbers you report are recall@100 (target: >0.95 on your benchmark — below 0.90 and your ceiling is the problem, not your reranker) and nDCG@10 after reranking.

**⚠ Trap:** reporting recall@5 for a system that reranks. You have measured the wrong stage. If your reranker sees 100 candidates, recall@5 of the first stage is irrelevant — it can be 0.4 and your system can still be excellent.

### Make the case: why would you fine-tune the embedder rather than upgrade the generator model?

Because in a RAG system, **retrieval failures are unrecoverable and generation failures are not.** That asymmetry is the whole argument and I would lead with it.

If the right chunk is in the context and the model is mediocre, you get a slightly worse-worded correct answer. If the right chunk is *not* in the context, the best model in the world produces a confident, fluent, wrong answer — or, if you have prompted it well, an "I don't know" on a question your corpus could have answered. No generator upgrade fixes a retrieval miss. The information is simply absent.

Now the numbers, which is what makes this an argument rather than an opinion.

**Quality delta.** Swapping a good frontier generator for a slightly better one moves end-to-end RAG answer quality by low single digits on most internal evals, because the generator was rarely the bottleneck. Fine-tuning a general embedder on a few thousand in-domain pairs routinely moves **recall@10 by 5–15 points** on a specialized corpus (legal, medical, internal jargon, code, non-English). Every point of recall is a query that goes from unanswerable to answerable.

**💰 Cost delta.** Fine-tuning a 335M encoder on 20k pairs: at batch 512 with GradCache, ~3 epochs, that is roughly 120 optimizer steps' worth of ~60k pair-encodings — call it 24M tokens of forward+backward at `6 × 0.335e9 × 24e6` = 4.8e16 FLOPs, well under **one GPU-hour**, so **under $5** of compute. Add the data-construction cost (synthetic generation over your corpus plus human review) and you are at maybe $500–2,000 all-in, mostly human time. Upgrading a generator from a $3/Mtok model to a $15/Mtok model on a system doing 200k calls/day at 4k input tokens each: `200e3 × 4e3` = 8e8 input tokens/day = 800 Mtok/day. At $3 that is $2,400/day; at $15 it is $12,000/day. The upgrade costs **$288,000/month more**. The embedder fine-tune costs a couple of thousand dollars, once.

**Latency delta.** The fine-tuned embedder is the same size — zero latency change. The bigger generator is slower on every single request.

**The honest caveats, which you must volunteer:** fine-tuning the embedder does not help if your failures are (a) generation-side — the model has the right chunk and still gets it wrong, (b) chunking-side — the answer is split across a boundary and no embedding fixes that, or (c) coverage-side — the document isn't in the corpus. **So the actual first move is always error analysis:** take 50 failing queries, and for each, check whether the correct chunk was in the retrieved set. That single tally tells you whether to work on retrieval or generation, and it takes an afternoon.

**🗣 Say this in the room:** "Retrieval failures are unrecoverable; generation failures usually aren't. Fine-tuning a small embedder on domain pairs costs a few hundred dollars and typically buys 5–15 points of recall on a specialized corpus, while a generator upgrade on a 200k-call-a-day system can cost six figures a month for low-single-digit end-to-end gains. But I'd run the error analysis first — 50 failures, tally whether the right chunk was retrieved — because if the answer is 'yes, and the model still got it wrong,' the embedder is not my problem."

### I have a corpus and no labeled query-passage pairs. Build me the training data.

This is the actual blocker in every real fine-tune, and the answer is **synthetic query generation with a hard-negative mining pass and a human-reviewed holdout.** Let me lay out the pipeline concretely.

**Step 1 — sample chunks, stratified.** Do not sample uniformly; you will get 40% boilerplate. Cluster your chunk embeddings (HDBSCAN or k-means with k≈200), sample proportionally-but-capped from each cluster so rare document types are represented. Target 10k–50k chunks.

**Step 2 — generate queries per chunk with an LLM.** The prompt design matters more than the model choice. Two rules: generate **3–5 queries per chunk with explicitly different styles** (a keyword-ish query, a full natural question, a question containing a specific entity or number from the chunk, and a vague/underspecified one), and **instruct the model to write queries a real user would type**, not queries that summarize the passage. The default failure is that the LLM writes "What does this document say about the refund policy for enterprise customers in the EMEA region?" — a query no human has ever typed, containing almost every content word from the passage. That trains lexical overlap, not semantics.

**Step 3 — filter with round-trip consistency.** Embed each generated query with your *current* model, retrieve top-10, and check whether the source chunk is there. Two useful signals: if the source chunk is at rank 1 with a huge margin, the query is trivially lexical — **drop it**, it teaches nothing. If the source chunk is not in the top 50, the query may be bad or unanswerable — inspect a sample, drop most. Keep the middle: queries where the source chunk is retrieved but not trivially.

**Step 4 — mine hard negatives** per the earlier procedure: top 10–50 from the current retriever, excluding the source chunk and anything from the same document, with a cross-encoder or guide-model denoising pass to drop suspected false negatives.

**Step 5 — human review of a sample, and a fully-human eval set.** Review 300 synthetic pairs by hand. If more than ~15% are bad, fix the prompt and regenerate — this is much cheaper than training on junk. And keep the *evaluation* set entirely separate and human-sourced.

**⚠ Trap — the one that invalidates the whole exercise:** deriving your test set from the same synthetic generator as your training set. You will show a beautiful improvement that is entirely the model learning your generator's stylistic tics. **Split by document, not by pair**, so no document contributes to both train and test, and evaluate on real queries.

**💰 Math on generation cost:** 20k chunks × 400 tokens input, 4 queries × ~20 output tokens each = 80 output tokens per chunk. Input: `20e3 × 400` = 8M tokens. Output: `20e3 × 80` = 1.6M tokens. At $3/Mtok in and $15/Mtok out that is `8 × $3 + 1.6 × $15` = $24 + $24 = **$48**. Run it through a batch API at ~50% and it is **$24**. Synthetic query generation is essentially free; the expensive part is the human review, and that is where you should spend.

### Give me the fine-tuning recipe. Loss, batch size, learning rate, epochs, and what improvement should I expect?

Here is the recipe I would actually run, with the reasoning attached to each number, because the numbers without reasoning is a recital.

**Loss:** `MultipleNegativesRankingLoss` (which is InfoNCE with in-batch negatives) — or its cached variant, `CachedMultipleNegativesRankingLoss`, if you need large batches on limited memory. With mined hard negatives, feed triples `(query, positive, negative_1..k)` so the loss sees both in-batch and explicit hard negatives.

**Batch size: as large as you can get, and use GradCache to get larger.** This is the single highest-leverage knob, for the reasons in the InfoNCE derivation — every additional in-batch item is a free negative. Target 512 minimum, 2048+ if you can. GradCache lets you hit 8192 on a single 80 GB card for a small encoder by doing two passes: one no-grad pass to compute all embeddings and the loss's gradient with respect to them, then a second pass in sub-batches that backprops through the encoder using the cached gradients.

**Learning rate: 1e-5 to 3e-5 for a full fine-tune of a small encoder**, with warmup over the first 10% of steps and linear or cosine decay. This is roughly 10× lower than you would use for training from scratch, and the reason is the whole point of domain adaptation: you want to *rotate* an existing good geometry, not rebuild it. **2e-5 is my default starting point.**

**Epochs: 1 to 3.** Embedding fine-tunes overfit fast. With 20k pairs at batch 512 that is ~40 steps per epoch — you are doing on the order of 100 optimizer steps total. If you find yourself training for 20 epochs, you have too little data, and the fix is more data, not more epochs.

**Temperature:** keep the model's original scale (0.05 for most, or `scale=20` in sentence-transformers terms, which is the same thing expressed as 1/τ). Do not tune it in the same experiment as the negatives.

**Evaluate every N steps against your held-out in-domain benchmark, on nDCG@10, and keep the best checkpoint.** Not on the loss. The loss will keep dropping past the point where retrieval quality peaks.

**📐 Expected gains, stated honestly as a range:** on a genuinely specialized corpus with domain jargon the general model does not know — legal citations, medical coding, internal product names, a low-resource language — expect **+5 to +15 points of recall@10**. On a corpus that is basically well-written English about ordinary topics, expect **+1 to +3**, sometimes zero, because the base model already handles it. Knowing which situation you are in *before* you spend the effort is what the error analysis is for.

**⚠ Trap:** the "improvement" that is actually your held-out set leaking. If your synthetic queries were generated from chunks, and near-duplicate chunks exist across your train/test split, the model has effectively seen the test positives. **Split by document ID and dedupe near-duplicates by MinHash before splitting**, not after.

### Full fine-tune or LoRA? And what do I lose on general retrieval?

For an embedder, I default to a **full fine-tune with a low learning rate**, which is the opposite of my default for generative LLMs, and the reason is size. A 335M encoder in bf16 with Adam needs `0.335e9 × 16` = 5.4 GB of optimizer and weight state — it fits on any GPU you have. LoRA exists to make training *fit*; when it already fits, LoRA is adding a constraint you do not need and it converges more slowly.

LoRA becomes correct in two situations. **One: the backbone is a 7B decoder-embedder**, where full fine-tuning needs `7e9 × 16` = 112 GB and LoRA at rank 16 over attention projections trains maybe 0.1% of parameters in a few GB. **Two: you need many domain adapters over one served backbone** — the Jina v3 design is exactly this, task-specific LoRA adapters swapped over a shared encoder — which lets you serve one set of base weights and switch behavior per tenant or per task. That second case is a genuinely good architecture for a multi-tenant product and worth proposing.

Now the part that matters more than the choice: **catastrophic forgetting is real and you must measure it.** A domain fine-tune reshapes the geometry toward your corpus, and the general retrieval ability the base model had can degrade. In a system where your corpus is the *only* thing you retrieve, that might be an acceptable trade. But it stops being acceptable the moment you add a second corpus, or your users start asking general questions, or you want to reuse the embedder for a different feature.

**The rule I enforce in review: every embedder fine-tune reports two numbers, not one.** In-domain nDCG@10 on your benchmark (should go up), and a general-retrieval score on a fixed public slice — a couple of BEIR datasets, or MTEB retrieval subset — held constant across every run (should not collapse). If in-domain goes +9 and general goes −12, you have built a model that is excellent right up until the product changes.

Three mitigations, in order of how often I use them: (1) **lower the learning rate** — most forgetting is just too much LR; (2) **mix ~20–30% of general-domain pairs into the training data** as a replay buffer, which is cheap and effective; (3) **LoRA with a small rank**, which limits how far the weights can move by construction.

**⚠ Trap:** evaluating only in-domain, shipping, and then six months later someone adds a second document source and retrieval quality on it is inexplicably terrible. Nobody connects it to the fine-tune from two quarters ago. That is why the general-retrieval regression number is a required field in the eval report, not a nice-to-have.

### How do you evaluate a fine-tuned embedder so you don't fool yourself?

By assuming you are fooling yourself and designing the evaluation to catch it. There are four specific ways an embedding fine-tune produces a fake improvement, and I check all four.

**1. Document-level leakage.** Split train/test by `doc_id`, never by pair. If chunk 4 of a document is in train and chunk 5 is in test, and they share a header, a defined term, and a page footer, the model has effectively memorized the test document's vocabulary. Additionally dedupe near-duplicates *before* splitting — MinHash or SimHash over shingles, threshold around 0.8 Jaccard — because enterprise corpora are full of documents that are 95% identical (last year's contract, the same policy in a different jurisdiction).

**2. Query-distribution leakage.** If train queries and test queries came from the same LLM with the same prompt, your test measures the generator's style, not retrieval. The test queries should be real, or at minimum generated by a *different* model with a *different* prompt and then human-filtered.

**3. Metric-stage mismatch.** Measure at the k your pipeline actually uses. And measure the **full pipeline**, not just the bi-encoder: it is entirely possible for a fine-tune to improve bi-encoder nDCG@10 while leaving end-to-end quality flat because your cross-encoder reranker was already fixing those cases. If the reranker was already covering the gap, you spent a week for nothing — and you would only know by running the ablation.

**4. Statistical significance.** With 200 queries, a 2-point nDCG difference is often noise. Run a **paired bootstrap**: resample the 200 queries with replacement 10,000 times, compute the metric difference on each resample, and report the 95% interval of the difference. If that interval straddles zero, you do not have a result. This takes ten lines of numpy and it is the single most under-used tool in applied retrieval work.

```python
import numpy as np
def paired_bootstrap(a, b, n=10000, rng=np.random.default_rng(0)):
    # a, b: per-query metric arrays for two systems, same queries, same order
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boots, [2.5, 97.5])
```

**⚠ Trap:** running 30 configurations, picking the best on the test set, and reporting that number. That is selecting on noise — with 30 comparisons at α=0.05 you expect roughly 1.5 spurious "wins" by chance. Use a dev set for selection and touch the test set once, or apply a multiple-comparison correction and say so.

**🗣 Say this in the room:** "I split by document not by pair, dedupe near-duplicates with MinHash before splitting, keep the evaluation queries human-sourced and separate from the synthetic training generator, measure at the k my reranker actually consumes, and report a paired bootstrap interval. If the 95% interval on the delta straddles zero I say I don't have a result, because with 200 queries a two-point nDCG move usually isn't one."

### When does fine-tuning the embedder *not* help? Talk me out of it.

Gladly — knowing when a technique doesn't apply is worth more than knowing the technique. There are five situations where I would refuse to fine-tune, and I would name them in a design review.

**One: your failures are lexical, not semantic.** If the errors are "user searched for error code `E_CONN_4417` and got nothing," no amount of contrastive training reliably fixes that — dense embeddings compress rare tokens away by construction. The fix is **hybrid retrieval with BM25 or a learned sparse head**, and it costs a day rather than a month. This is the most common misdiagnosis I see.

**Two: your failures are chunking failures.** If the answer spans a boundary, if a table got cut in half, if a definition is in a header that isn't in the chunk, the correct chunk *does not exist* in your index. Fine-tuning the model that searches an index with no answer in it is a category error.

**Three: your failures are coverage failures.** The document isn't in the corpus, or is in the corpus but failed to parse. Go count your parse failures — in most real ingestion pipelines it is 2–5% of documents and nobody has looked.

**Four: you don't have and can't get 5,000+ quality pairs.** Below roughly a few thousand pairs the fine-tune is mostly noise and the risk of degrading general performance outweighs the expected gain. Synthetic generation can get you there, but only if the corpus is large enough to sample from.

**Five: a reranker gets you there for less.** A cross-encoder reranker over the top 50 typically buys more nDCG than an embedder fine-tune, requires zero training, and can be swapped out on a config change. It costs latency and per-query compute, which the fine-tune does not — so the honest framing is a trade: **fine-tuning moves quality upstream at zero serving cost but requires data and a training pipeline; reranking buys quality immediately at a per-query cost.** If you are pre-product-market-fit, rerank. If you are at scale with a stable domain, fine-tune, and then consider distilling the reranker into the bi-encoder to get both.

**🔍 The decision procedure I would actually run:** take 50 failing queries. For each, check (a) was the correct chunk in the corpus at all? (b) if yes, was it in the top 100? (c) if yes, was it in the top 5 after reranking? Bucket (a)-failures → ingestion. Bucket (b)-failures → retrieval: check whether the query contains an exact identifier (→ hybrid) or domain jargon (→ fine-tune). Bucket (c)-failures → reranking. Anything that passed all three and still produced a bad answer → generation. That tally, on one page, is the most valuable artifact in a RAG project and it takes an afternoon.

### Multilingual: one model for all languages, or one per language?

One multilingual model, in almost every case, and the reason is not convenience — it is **cross-lingual alignment**, which per-language models cannot give you.

The mechanism: a multilingual embedder trained with parallel or translation-pair data places the same meaning at roughly the same point regardless of language. `"Wie setze ich mein Passwort zurück"` and an English document about password reset land near each other. That single property enables the thing enterprises actually want: a user queries in their language and retrieves from a corpus that is 70% English, without a translation step in the request path. Per-language models give you `n` disjoint vector spaces and you would have to route the query to the right index, which means you now need language detection on a 6-word query (unreliable) and you lose every cross-lingual hit.

Practical consequences worth stating:

- **One index, not `n`.** Mixing languages in a single index is correct under a multilingual model, and it means a single ANN structure to operate.
- **Tokenizer efficiency is a real cost.** Multilingual tokenizers spend more tokens per character on non-Latin scripts. The same Japanese passage may cost 1.5–2× the tokens of its English translation, which shows up directly in your indexing bill and in whether a chunk fits under `max_seq_length`. **Set chunk size in tokens, per language, not in characters.**
- **Quality is not uniform across languages,** and the model card's average hides it. Evaluate per language on your own data, weighted by your actual traffic mix.

**⚠ Trap:** assuming cross-lingual retrieval works because the model is "multilingual." Multilingual (handles many languages independently) and cross-lingual (aligns them in one space) are different properties, and not every multilingual model has strong alignment. **Test it directly:** take 100 known query-document pairs, translate the queries, and measure recall against the untranslated documents. If cross-lingual recall is far below same-language recall, the model is multilingual but not well-aligned, and you need either a different model or a translation step.

**When per-language *is* right:** when one language dominates and the domain is highly specialized in it — a Japanese-only legal corpus is better served by a strong Japanese-specialized model than by a multilingual generalist that treats Japanese as 6% of its training mix. And when a language is genuinely low-resource, a smaller dedicated model sometimes beats the generalist.

**📅 Volatile:** multilingual-E5, BGE-M3, and the Cohere and Voyage multilingual offerings are the usual candidates; verify current language coverage and per-language scores on MMTEB before committing.

### API or self-hosted embeddings? Show me the numbers.

The break-even is almost entirely about **volume and whether you re-embed**, and I can do it live.

Set up the scenario: 100M chunks at 400 tokens = **40 billion tokens** for a full index build, plus ongoing query traffic of 1M queries/day at 20 tokens = 20M tokens/day = **0.6 Btok/month**, plus 5M new/updated chunks/month = **2 Btok/month** of incremental ingestion.

**API path.** At a small-model price of $0.02 per million tokens: initial build `40,000 × $0.02` = **$800**. Monthly ongoing `(0.6 + 2) × 1000 × $0.02` = `2,600 × $0.02` = **$52/month**. At a large-model price of $0.13/Mtok: initial build **$5,200**, ongoing **$338/month**. **📅 Volatile:** verify current per-token embedding prices before quoting.

**Self-hosted path.** A 335M encoder: `2 × 0.335e9 × 40e9` = 2.68e19 FLOPs for the build. On an H100 at 990 bf16 TFLOPS with a realistic 20% MFU for short-sequence encoder work (~198 TFLOPS effective): `2.68e19 / 1.98e14` = 135,000 s = **37.5 GPU-hours**. At $3/hr on-demand that is **$113** — and the derived throughput is `40e9 / 135,000` ≈ **296k tokens/s**, which you should treat as an upper bound; measured TEI throughput is commonly 2–3× lower once tokenization, padding and HTTP overhead are counted, so plan at **~100k tokens/s per H100** and budget ~110 GPU-hours. Ongoing 2.6 Btok/month is `2.6e9 / 1e5` = 26,000 s ≈ 7 GPU-hours ≈ **$21/month of compute** — but you cannot buy 7 hours; you must keep a replica warm for query latency, so the real ongoing cost is **one always-on GPU**, roughly $2,190/month for an H100 or ~$400/month for an L4-class card that is plenty for a 335M model.

**So the honest comparison:** API costs $52–338/month; self-hosting costs $400–2,200/month in idle GPU plus an engineer's ongoing attention. **At this volume the API wins on cost, decisively.** The self-host case is made on other grounds:

1. **Data residency / no third-party egress** — the dominant real reason, and it is binary.
2. **You need a fine-tuned model.** You cannot fine-tune most API embedders. The moment domain adaptation is on the table, self-hosting is implied.
3. **Rate limits during backfill.** Re-embedding 100M chunks against an API means sustaining a huge request rate for hours; provider limits often make this take days, and you will be negotiating quota. Self-hosted, you just add GPUs.
4. **Version pinning.** An API model can be deprecated or silently updated; your index is a physical artifact of a specific model version and you need that version to exist for as long as the index does.

**🗣 Say this in the room:** "At 100M chunks the API embedding bill is a few hundred dollars a month and self-hosting costs more once you account for an always-on GPU — so cost is not the argument. I self-host for exactly three reasons: data residency, because I want to fine-tune, or because I need to control the model version so a deprecation doesn't invalidate my index."
