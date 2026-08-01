### Our RAG system works great in demos, but a customer searched for error code `ERR_TLS_CERT_ALTNAME_INVALID` and got back three generic pages about TLS. Walk me through what went wrong.

This is the single most common production RAG bug, and the answer is structural rather than a tuning problem: **a dense embedding is a lossy projection that throws away exactly the information that makes an identifier an identifier.** Your embedding model compressed a 30-token string down to 1024 floats. To do that it had to decide what to keep. It kept "this is about TLS, certificates, and an error." It discarded the precise character sequence, because during contrastive training nothing ever rewarded it for distinguishing `ALTNAME` from `CHAIN` — those two strings appeared in near-identical contexts, so the objective actively pulled them together.

Worse, that token string probably isn't even in the tokenizer vocabulary as a unit. `ERR_TLS_CERT_ALTNAME_INVALID` gets shredded into something like `ERR`, `_`, `TLS`, `_`, `CERT`, `_`, `ALT`, `NAME`, `_`, `INVALID` — ten-ish subword pieces, each of which is high-frequency and low-information. The pooled vector is dominated by the semantic gist. Meanwhile every one of your 400 TLS troubleshooting pages contains "TLS", "certificate", "invalid" and lands within a hair's breadth of the query in cosine space. Dense retrieval will confidently return the wrong three, with high similarity scores, and no signal at all that it failed.

BM25 has the opposite failure profile and therefore the exactly complementary strength. `ALTNAME` appears in maybe 2 documents out of 400,000. Its IDF is enormous. A single occurrence in the right doc dominates the whole score. Lexical retrieval does not "understand" the query at all, and that is precisely why it nails it.

**⚠ Trap:** the fix people reach for first is "use a better embedding model." It does not work, because the failure is not a quality gap — it is an information-theoretic one. No fixed-width dense vector reliably preserves rare literal strings. The fix is to add a lexical channel, not to buy a bigger encoder. I have watched teams burn three weeks upgrading from a 768-dim to a 3072-dim model and move this failure class approximately zero.

**🗣 Say this in the room:** "Dense retrieval fails on out-of-distribution literals — SKUs, error codes, version numbers, ticket IDs, proper names — because embedding is a lossy compression trained on semantic similarity, and those tokens carry no semantics, only identity. That's not a model-quality problem, it's a representation problem, so the fix is a hybrid lexical channel with BM25 or SPLADE. I would run that ablation before touching the embedder."

**📄 Paper:** Thakur, Reimers, Rücklé, Srivastava & Gurevych (2021) — BEIR, the zero-shot retrieval benchmark. Its most-cited finding is the uncomfortable one: BM25 beat most dense retrievers out-of-domain in 2021, and it remains the baseline every dense model has to earn its way past. It replaced the habit of reporting MS MARCO in-domain numbers and calling the problem solved.

### Derive TF-IDF for me. What problem is each of the two halves solving?

TF-IDF is **two independent corrections to the naive idea that "documents containing your words are relevant."** Start from that naive idea and watch it break twice.

Naive version: score a document by how many query terms it contains. Immediately broken — a document containing "the" fifty times scores enormously for the query "the cat." So you need to discount terms by how uninformative they are. That is the IDF half. Inverse document frequency is `log(N / df(t))` where `N` is the corpus size and `df(t)` is the number of documents containing `t`. It is, literally, the self-information of the event "a randomly drawn document contains `t`." A term in every document carries zero bits and gets weight ~0. A term in 1 document out of 10 million carries `log(10^7/1) ≈ 16` nats and dominates.

Second break: within a single document, is a term appearing 50 times really 50× more relevant than once? No — so you need a sublinear transform of raw count. The classic TF-IDF answer is `1 + log(tf)`, which turns 50 into ~4.9 and 1 into 1.0. Then you take `Σ_{t ∈ q} tf_weight(t,d) · idf(t)`, usually with cosine normalization over the document vector so long documents cannot win by mass.

The mental model I want you to carry: **IDF is the price of a term and TF is how many you bought.** A rare term is expensive currency; matching it is worth a lot. A stopword is worthless currency; matching it a hundred times still buys nothing.

**⚠ Trap:** `log(N/df)` goes negative-adjacent and behaves badly for terms present in nearly every document, and it is undefined for `df = 0`. Every real implementation uses a smoothed variant — Lucene uses `log(1 + (N − df + 0.5)/(df + 0.5))`, which is bounded below by 0 and never blows up. If you write the textbook formula on the whiteboard and the interviewer asks "what happens when df equals N," the correct answer is "I'd use the smoothed probabilistic IDF, because the raw one hits zero or goes negative and the score becomes uninterpretable."

TF-IDF matters today mostly as the thing BM25 fixed. You will not deploy it. You will be asked to derive it as the setup for the next question.

### Write BM25 on the whiteboard and tell me exactly what k1 and b do.

BM25 is **TF-IDF with the two hacks replaced by principled, tunable saturation and length normalization.** That is the whole story, and if you say it that way the rest follows.

```
score(q, D) = Σ_{t ∈ q}  IDF(t) · ( f(t,D) · (k1 + 1) ) / ( f(t,D) + k1 · (1 − b + b · |D| / avgdl) )

IDF(t) = ln( 1 + (N − n(t) + 0.5) / (n(t) + 0.5) )
```

where `f(t,D)` is the raw term frequency in the document, `|D|` is the document length in tokens, `avgdl` is the mean document length over the corpus, `N` is the number of documents, `n(t)` is the document frequency of `t`.

**`k1` controls term-frequency saturation.** Look at the TF factor in isolation, ignoring length: `f·(k1+1)/(f+k1)`. As `f → ∞` this asymptotes to `k1 + 1`. So the entire contribution of one term is *bounded*, no matter how many times it occurs. At `k1 = 1.2` (Lucene's default), a term occurring once contributes `1·2.2/(1+1.2) = 1.00` of its IDF; twice gives `2·2.2/(2+1.2) = 1.375`; ten times gives `10·2.2/11.2 = 1.96`; a hundred times gives `2.17`. Compute the marginal value explicitly: going from 99 to 100 occurrences moves the factor from `217.8/100.2 = 2.1737` to `220/101.2 = 2.1739` — an increment of `0.0002`, against the first occurrence's increment of `1.00`. **The 100th occurrence is worth roughly two-hundredths of a percent of what the 1st was worth.** That is term saturation, and it is the single most important property of BM25. Setting `k1 = 0` makes TF binary — presence/absence only. Setting `k1` large (say 8) makes it nearly linear in count, which is what you want for very long documents where repetition is genuinely evidential.

**`b` controls length normalization**, on a slider from 0 to 1. The denominator term `k1·(1 − b + b·|D|/avgdl)` is the effective saturation constant, scaled by how long this document is relative to average. At `b = 0` length is ignored entirely — a 50,000-word document competes on equal footing with a 200-word one, and long documents dominate because they contain everything. At `b = 1` you fully divide by relative length — a document twice as long needs twice the term frequency to score the same. Lucene's default is `b = 0.75`, a compromise that penalizes verbosity without punishing genuinely comprehensive documents.

**📐 Numbers you must know:** `k1 = 1.2`, `b = 0.75` are the Lucene/Elasticsearch/OpenSearch defaults and the numbers to quote. `k1 ∈ [1.2, 2.0]` and `b ∈ [0.3, 0.9]` is the tuning envelope. For RAG over uniform 512-token chunks, I lower `b` toward 0.3–0.5, because your chunks are already length-normalized by construction and aggressive length division just adds noise. That is a real, defensible tuning opinion and interviewers notice it.

**📄 Paper:** Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond* — the definitive retrospective on the Okapi BM25 line of work from the 1990s. Cite this rather than trying to name a single 1994 TREC paper; BM25 accreted across several.

### Implement BM25 from scratch. No libraries beyond the standard library.

Roughly 35 lines, and you should be able to type it without thinking. The structure that matters: precompute document frequencies and doc lengths at index time, score only documents in the posting lists of query terms at query time.

```python
import math
from collections import Counter, defaultdict

class BM25:
    def __init__(self, corpus_tokens, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        self.tf = [Counter(d) for d in corpus_tokens]        # per-doc term counts
        self.postings = defaultdict(list)                    # term -> [doc_id, ...]
        for i, counts in enumerate(self.tf):
            for term in counts:
                self.postings[term].append(i)
        self.idf = {
            t: math.log(1.0 + (self.N - len(ids) + 0.5) / (len(ids) + 0.5))
            for t, ids in self.postings.items()
        }

    def score(self, query_tokens, doc_id):
        s = 0.0
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
        for t in query_tokens:
            f = self.tf[doc_id].get(t, 0)
            if f:
                s += self.idf[t] * (f * (self.k1 + 1)) / (f + norm)
        return s

    def search(self, query_tokens, top_k=10):
        cands = set()
        for t in query_tokens:
            cands.update(self.postings.get(t, ()))          # only docs that match something
        scored = ((self.score(query_tokens, d), d) for d in cands)
        return sorted(scored, reverse=True)[:top_k]
```

Two details an interviewer will probe. First, `search` iterates only the union of posting lists, not all `N` documents — that is the entire reason an inverted index exists, and forgetting it is the most common way this implementation "works" on 1,000 docs and dies on 10 million. Second, `idf` is computed once at index time, not per query; recomputing it per query is an `O(vocab)` mistake I have seen ship.

**⚠ Trap:** BM25 scores are **not comparable across queries**. A two-term query and a six-term query produce sums over different numbers of terms with different IDFs; a score of 14.2 means nothing in absolute terms. Anyone who sets a global `score > 10.0` relevance threshold has built a system whose behavior silently varies with query length. Thresholds must be per-query (relative to the top hit) or learned, never absolute.

### What actually lives in an inverted index, and what makes query evaluation fast at ten million documents?

The inverted index is **exactly the Postgres GIN index you already know, and the mental model transfers wholesale**: a mapping from term → sorted list of document ids that contain it, plus positional and payload data. The retrieval work is a k-way merge over sorted integer lists, which is why a lexical search engine is cheap relative to anything neural.

A posting list entry is minimally `(doc_id, term_freq)`, and for phrase and proximity support also `positions[]`. Doc ids are stored delta-encoded and then variable-byte or PFOR-encoded, because the deltas are small and compress hard — a posting list for a common term goes from 4 bytes/entry to under 1. Lucene additionally stores per-field norms (an approximation of `|D|`) in a separate column so the BM25 length factor is a single lookup.

Three things make evaluation fast:

**Skip pointers.** A posting list is stored in blocks (Lucene uses 128 docs), with a skip list over block boundaries. When you are intersecting `rare_term AND common_term`, you advance the rare list one doc at a time and *seek* the common list to that doc id, skipping whole blocks without decoding them. This turns an `O(len(long list))` merge into roughly `O(len(short list) · log)`.

**Block-max WAND.** Each block stores the maximum possible BM25 contribution of any document in it. During top-k retrieval you maintain a heap of the current best `k` scores; if the sum of block maxima for the remaining lists cannot exceed the current `k`-th best score, you skip the entire block. On a top-10 query over a large index this routinely skips 90%+ of the postings without changing the result at all — it is exact, not approximate.

**📄 Paper:** Broder, Carmel, Herscovici, Soffer & Zien (2003) introduced WAND (Weak AND) — document-at-a-time top-k retrieval with an upper-bound pruning threshold. Ding & Suel (2011) added the per-block max-score refinement that everything modern uses.

**📐 Numbers you must know:** an inverted index typically costs 15–30% of the raw text size on disk after compression; the dense vector index for the same corpus at 1024-dim fp32 costs 4 KB *per chunk* regardless of chunk content. For 10M chunks of ~400 tokens (~2 KB text each, 20 GB raw), that is roughly 4–6 GB for the inverted index versus 40 GB for the flat dense vectors plus HNSW graph overhead. **The lexical half of your hybrid system is nearly free.** This is the number that ends the "can we afford to add BM25" conversation in about ten seconds.

### Talk me through stemming versus lemmatization, and tell me which one you would actually put in a production analyzer.

An analyzer is a deterministic text→token pipeline, and every stage of it is a **recall/precision trade you are making on behalf of every future query.** Character filters (HTML strip, accent fold) → tokenizer (split on Unicode boundaries) → token filters (lowercase, stopwords, stemming, synonyms). It runs identically at index time and query time, and the single most important invariant in lexical search is that **the query analyzer and the index analyzer must agree**, or you will search for tokens that were never written.

Stemming is a crude suffix-stripping heuristic — Porter, or its successor Snowball. `running → run`, `connection → connect`, `universities → univers`. It is fast, has no dictionary, and produces non-words. Lemmatization uses a dictionary plus part-of-speech tagging to map to the true dictionary form — `better → good`, `was → be`. It is 10–100× slower and needs a POS tagger, which needs sentence context, which you frequently do not have at query time for a three-word query.

**In production I use light stemming, almost never full lemmatization, and I use it on a separate field rather than replacing the raw one.** The reason is asymmetric cost. Over-stemming destroys precision irrecoverably: Porter maps both `universal` and `university` to `univers`, and in a legal or medical corpus that is a real error. Under-stemming just costs you a bit of recall, which your dense channel is already covering. So: index a `body` field with minimal analysis (lowercase, no stemming) and a `body.stemmed` field with Snowball, search both with the exact field boosted higher, and let the scorer prefer exact matches while still catching morphological variants.

**⚠ Trap:** stopword removal at query time. Elasticsearch's default English analyzer historically removed stopwords, and then the query `to be or not to be` becomes an empty token stream and returns nothing. Worse, `The Who` becomes nothing, and `vitamin A` loses the `A`. Modern practice is to **keep stopwords in the index** and rely on BM25's IDF to make them near-worthless automatically — which is exactly what IDF is for. The one exception is phrase queries over very large indexes where common-term posting lists dominate latency; there you use a common-grams filter, not deletion.

**🗣 Say this in the room:** "I keep stopwords and let IDF handle them, because deletion is a lossy irreversible decision made at index time and IDF is a soft weight recomputed against the actual corpus. And I stem into a secondary field rather than in place, so exact matches can outrank stemmed ones."

### Your search works fine in English. We just launched in Germany, Japan and Turkey. What breaks?

All three break, in three completely different ways, and none of them will show up in your English eval set. This is a question I have seen sink candidates who otherwise knew retrieval cold.

**German breaks on compounding.** `Lebensversicherungsgesellschaft` is one token meaning "life insurance company." A user searching for `Versicherung` gets zero lexical matches, because that substring is not a token. You need a decompounder — either dictionary-based (`hyphenation_decompounder` with a German dictionary) or the German analyzer's light stemming plus an n-gram fallback. Without it your German recall is catastrophically worse than your English recall and the metric average hides it.

**Japanese breaks on tokenization entirely.** There are no spaces. The Unicode-standard tokenizer that works for every European language emits either the whole sentence as one token or every character individually. Neither is right. You need a morphological analyzer — kuromoji for Japanese, and for Chinese either a dictionary segmenter or CJK bigrams. The default `standard` analyzer splits CJK into single characters, which gives you decent recall and appalling precision: searching for 京都 (Kyoto) matches 東京都 (Tokyo Metropolis) because both contain 京 and 都.

**Turkish breaks on lowercasing.** Turkish has dotted and dotless i as distinct letters. The uppercase of `i` is `İ`, and the lowercase of `I` is `ı`. If you lowercase with the invariant/English locale, `İSTANBUL` becomes `i̇stanbul` and never matches `istanbul`. This is the classic "Turkish I problem" and it is a locale-aware-lowercase filter, not a stemmer, that fixes it.

**🔍 Failure taxonomy — how multilingual lexical retrieval breaks, as a decision procedure:**
1. Recall is fine in English, near-zero in one language → wrong tokenizer for that script. Check CJK, Thai, Khmer first.
2. Recall is moderate everywhere but precision collapses in one language → character-level tokenization fallback firing (CJK bigrams, or n-gram field).
3. Exact-match queries fail on capitalized input only → locale-insensitive lowercase. Turkish, Azeri, Lithuanian.
4. Queries with accents match, queries without do not (or vice versa) → asymmetric `asciifolding` between index and query analyzer. This is the one that hides longest because it only affects some users.
5. Everything works, but relevance is subtly bad for one language → your `avgdl` is computed globally across a mixed-language corpus. German documents are longer in tokens than Chinese ones by a large factor, so the `b` length normalization is systematically punishing one language. Fix: per-language index or per-language field.

**⚠ Trap:** language detection at index time and *not* at query time. You detect the document language, route to the right analyzer, index correctly — and then at query time you have a three-word query that langdetect cannot classify, so it defaults to English and searches the English-analyzed field. Multilingual queries should be fanned out across all language fields and fused, not routed.

### How do you handle a document with a title, a body and tags? Do you just concatenate them?

No, and the reason is a genuinely subtle scoring bug. The naive approach — search each field separately and take the max or the sum of the per-field BM25 scores — **double-counts length normalization and IDF**, because each field computes its own `avgdl` and its own saturation curve. A query term appearing once in the title and once in the body gets two independent "first occurrence" bonuses, each near the steep part of the saturation curve, so it beats a document with the term twice in the body. That is usually wrong.

The principled fix is **BM25F**, which combines the fields *before* saturation rather than after. You compute a weighted pseudo-frequency across fields, normalize each field by its own field-specific length, and then apply the BM25 saturation once:

```
f̃(t, D) = Σ_fields  w_field · f(t, D_field) / (1 − b_field + b_field · |D_field| / avgdl_field)

score = Σ_t  IDF(t) · f̃(t,D) · (k1 + 1) / ( f̃(t,D) + k1 )
```

Note where the boost lives: `w_field` multiplies the frequency inside the saturation, not the final score. A title boost of 5 means "an occurrence in the title counts as five occurrences," which then saturates like any other frequency. A post-hoc score multiplier of 5 means "the title's entire contribution is 5× bigger," which is a different and much more aggressive claim. In Elasticsearch this is the difference between a `multi_match` with `type: best_fields` (per-field, max) and `type: cross_fields` (BM25F-flavored, treats fields as one). For a title/body/tags document I use `cross_fields` or `most_fields` and I set the title boost between 2 and 4.

**⚠ Trap:** setting field boosts by intuition and never validating. Title boost is the parameter most likely to be set to 10 by someone's gut and then silently ruin recall for long-tail queries, because a document whose title happens to contain a common query word now outranks a document that actually answers the question. Every field boost is a hyperparameter and belongs in the same grid search as `k1`, `b` and your hybrid alpha — tuned against a labeled set, reported with nDCG@10, and re-tuned when the corpus composition changes.

**💰 Math:** field boosts are also a latency question at scale. Searching four analyzed fields instead of one means four sets of posting lists to merge. On a 50M-document index where a single-field top-100 query runs at p95 ≈ 25 ms, a four-field `cross_fields` query typically lands at p95 ≈ 60–90 ms. In a hybrid pipeline whose total retrieval budget before the LLM is 300 ms, spending 90 of it on lexical to gain 1.5 nDCG points is a trade I would make; spending 200 ms on eight fields to gain 0.3 points is not.

### Give me the full taxonomy of query types where dense retrieval fails and lexical wins.

I want this as a memorized list, because it is the direct justification for hybrid and it comes up in nearly every applied-AI retrieval round.

**1. Identifiers and codes.** SKUs (`WH-1000XM5`), error codes, ticket ids (`ENG-4471`), CVE numbers, order numbers, IP addresses, commit SHAs. Zero semantic content, entirely identity. Dense retrieval has literally nothing to encode.

**2. Version numbers.** `pandas 2.1` versus `pandas 2.2` are near-identical in embedding space and completely different in answer-correctness. This one is vicious because dense retrieval returns something *plausible and wrong*, which is worse than returning nothing.

**3. Rare proper nouns.** A person's name that appears in three documents. Company names that collide with common words — searching for the company "Apron" or "Rippling" in a corpus about payroll. Dense embedding will return topically-related documents about the domain, not the entity.

**4. Domain jargon the encoder never saw.** Internal project codenames, in-house acronyms, chemical compound names, drug identifiers. Anything post-dating the embedder's training data — including your own product's feature names shipped last quarter.

**5. Negation and exclusion.** "Contracts *without* an auto-renewal clause." Encoders are notoriously bad at this: the embedding of "with X" and "without X" are extremely close, because the negation is one token out of thirty and pooling washes it out. Lexical does not fix this either — the honest answer is that negation needs a metadata filter or a structured predicate, not similarity search of any kind, and saying so is the mark of someone who has actually shipped this.

**6. Exact-phrase and quotation requirements.** Legal and compliance work where the user needs the document containing *this exact sentence*. Only a phrase query over a positional index can do this.

**7. Very short queries with high IDF.** A one-word query where that word is rare. Dense retrieval has almost no context to work with and produces a diffuse vector; BM25 has a single high-IDF posting list and is essentially exact.

**🗣 Say this in the room:** "The rule I use is: if the correct answer depends on a literal string rather than a meaning, dense retrieval is the wrong tool and no amount of model upgrading fixes it. Identifiers, versions, rare proper nouns, and jargon coined after the embedder's cutoff — those go to the lexical channel. Negation goes to a metadata filter, not to either retriever."

### And the reverse — where does BM25 fail and dense retrieval save you?

Symmetry matters here, because a candidate who only knows the pro-lexical argument sounds like they read one blog post. **BM25's fundamental limit is that it has no notion that two different strings can mean the same thing.** It is exact-match with clever weighting; every one of its failures is a vocabulary-mismatch failure.

**Synonymy.** The document says "myocardial infarction," the user types "heart attack." Zero term overlap, zero BM25 score, complete miss. You can patch this with a synonym filter, but synonym lists are a maintenance liability that grows without bound and encodes only the relations somebody remembered to write down.

**Paraphrase and question-to-answer asymmetry.** The user asks "how do I stop my container from being killed?" and the document says "OOMKilled occurs when a pod exceeds its memory limit; increase `resources.limits.memory`." Almost no lexical overlap. This is the dominant RAG query shape — natural-language questions against declarative documentation — and it is exactly where dense wins by a mile.

**Morphological and multi-word variation** that stemming misses: "cost reduction strategies" vs "how to reduce spend."

**Cross-lingual retrieval.** A query in Spanish against English documents. Lexical cannot do this at all without translation; a multilingual embedder does it natively.

**Conceptual and thematic queries.** "Documents about employee burnout" where the documents never use the word "burnout" but discuss attrition, overtime and exhaustion.

The correct summary is that the two methods have **nearly disjoint failure sets**, which is the entire mathematical justification for fusion: fusing two retrievers helps in proportion to how uncorrelated their errors are. Fusing two dense models with different checkpoints buys you very little because they fail on the same queries. Fusing BM25 and dense buys a lot. That framing — fusion gain ≈ error decorrelation — is what separates a real answer from "we use hybrid because it's best practice."

### Postgres already stores our documents. Can we just use `tsvector` and skip Elasticsearch?

Often yes, and I would push hard on trying it first — but you need to know the specific limitation you are accepting, or you will get caught.

Postgres full-text search gives you a proper inverted index (GIN over `tsvector`), stemming via configurable dictionaries, and phrase search. What it does **not** give you natively is BM25. `ts_rank` and `ts_rank_cd` are older cover-density and term-frequency heuristics that do not implement saturation or corpus-level IDF properly — `ts_rank` in particular does not use document frequency across the corpus at all, so a stopword-ish term contributes as much as a rare one. For short, homogeneous chunks in a RAG pipeline this matters less than you would think, because saturation and length normalization are both nearly no-ops when every chunk is 400–600 tokens. For heterogeneous documents it matters a lot.

Your options, in the order I would try them:

1. **`tsvector` + `ts_rank_cd`, fused with `pgvector` via RRF, all in one query.** One database, transactional consistency between your source rows and both indexes, no dual-write problem, no reindex-skew incident at 3am. For corpora under roughly 10M chunks this is genuinely the right answer and I will defend it in any design round.
2. **ParadeDB / `pg_search`**, which implements real BM25 inside Postgres on top of a Tantivy index. You get proper BM25 scoring and keep the single-database story, at the cost of an extension your ops team has to own.
3. **Elasticsearch/OpenSearch**, when you need per-field analyzers per language, aggressive analyzer experimentation, `search_after` deep pagination, or you already run it.

**🗣 Say this in the room:** "I start with Postgres — `tsvector` for lexical, `pgvector` for dense, RRF to fuse, one transaction, one backup story. I move to a dedicated search engine when I can name the specific capability I'm missing: real BM25 scoring, per-language analyzer chains, or write throughput my Postgres primary can't absorb. 'It doesn't scale' isn't a reason until I have the p95 that proves it."

**💰 Math:** the operational cost delta is not small. A managed Elasticsearch cluster sized for 10M documents with replication is commonly three nodes at roughly $200–400/month each (📅 Volatile — verify current instance pricing before your loop), so $7k–14k/year, plus the engineer-time cost of a second system to monitor, upgrade and keep in sync with Postgres. The `tsvector` column costs you disk on a database you already run and roughly 20–30% additional index size. In a design round, being the person who *didn't* add the second datastore is a better look than being the person who reached for Elasticsearch reflexively.

### Explain what a phrase query does at the index level, and why proximity scoring is not the same thing.

A phrase query — `"connection pool exhausted"` in quotes — is not a scoring adjustment. It is a **hard filter evaluated using the positions array in the posting lists**. Each posting is `(doc_id, tf, [positions])`. To evaluate a three-term phrase you intersect the three posting lists on `doc_id`, and then for each surviving document you check whether there exists a position `p` such that term1 is at `p`, term2 at `p+1`, term3 at `p+2`. That is a merge over sorted position lists — cheap, exact, and the only way to answer "does this exact string appear."

Proximity scoring is the soft version: a *span* or *slop* query that accepts the terms within `k` positions of each other, optionally scoring higher when they are closer. Elasticsearch's `match_phrase` with `slop: 3`, or a `span_near`. This is what you want for "these words are related in this document" rather than "these exact words in this exact order."

The reason to care in a RAG system: **positions cost storage and you can turn them off.** Lucene lets you set `index_options: docs` (doc ids only), `freqs`, `positions`, or `offsets`. Dropping to `freqs` cuts the index meaningfully — positions are typically the largest component of a posting list — but permanently disables phrase and proximity queries on that field. I have watched a team disable positions to save disk, ship it, and then discover six weeks later that legal's exact-quote workflow silently returned garbage, because a phrase query on a `freqs` field degrades to an unordered AND rather than erroring.

**⚠ Trap:** exposing quoted-phrase syntax in a RAG chat UI without checking that the field supports it. The system will happily return topically-similar results and the user will believe they got an exact match. If you support quotes, you must support them all the way down, including verifying that the phrase actually occurs in the returned chunk before you show it as an exact match.

### What is the honest ceiling on a well-tuned BM25 system, and how do you know when you have hit it?

The ceiling is set by **the vocabulary-overlap distribution of your query log against your corpus**, and you can measure it directly rather than guessing, which is the part most candidates miss.

Here is the procedure I run. Take 300 real queries from your logs with human-labeled relevant documents. For each query, compute recall@100 for BM25 alone. Then partition the failures into two buckets: (a) the relevant document contains at least one high-IDF query term but ranked below 100 — this is a *ranking* failure, tunable with `k1`, `b`, field boosts and phrase boosts; (b) the relevant document shares no meaningful terms with the query at all — this is a *vocabulary mismatch* failure, and **no amount of BM25 tuning will ever fix it.** Bucket (b) is your hard ceiling.

In my experience on natural-language question corpora, bucket (b) is 20–40% of queries, which is exactly why dense retrieval exists and why hybrid is not optional for question-answering workloads. On keyword-style corpora — internal code search, log search, product catalogs where users type identifiers — bucket (b) can be under 5%, and there dense retrieval is a rounding error you may not need at all.

**🏋 Drill (35 minutes, unaided):** implement the `BM25` class above from memory, index a 5,000-document corpus, and produce a table of recall@10 as you sweep `k1 ∈ {0, 0.6, 1.2, 2.0, 8.0}` and `b ∈ {0, 0.3, 0.75, 1.0}`. Pass criterion: the class is correct on first run, the sweep completes, and you can explain in one sentence why the `k1 = 0` row is nearly as good as the best row on short uniform chunks (answer: with 512-token chunks, terms rarely repeat, so binary presence and saturated frequency are nearly the same signal — which is also why `k1` tuning matters far less for RAG than for full-document search).
