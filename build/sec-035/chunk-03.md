### Explain the difference between a bi-encoder and a cross-encoder, and why the cross-encoder is fundamentally more accurate.

The distinction is **when the query and the document are allowed to see each other**, and everything else — cost, latency, index feasibility — falls out of that one architectural choice.

A **bi-encoder** encodes the query and the document *independently*. The document never sees the query; it is embedded once at index time into a fixed vector, and the query is embedded at query time, and relevance is a dot product between two vectors that were computed in total ignorance of each other. This is what makes ANN search possible: because the document vector doesn't depend on the query, you can precompute all 10 million of them and build an index. The price is that the entire interaction between query and document must be squeezed through a single scalar produced from two independently-compressed summaries.

A **cross-encoder** concatenates them — `[CLS] query [SEP] document [SEP]` — and runs a full transformer forward pass over the joint sequence, then reads a relevance score off a classification head. Now every query token attends to every document token in every layer. The model can notice that the query's "not" applies to the document's clause three sentences later, that the query's "2.2" must match the document's "2.2" and not its "2.1", that the document's pronoun refers to the entity the query asked about. **None of that survives independent encoding**, because at encoding time the document had no idea what would be asked of it.

The cost is that the score is not precomputable. There are `N` documents and each one needs its own forward pass with the query, so scoring the whole corpus is `O(N)` transformer passes. For 10M documents at ~10 ms each that is 27 hours per query. This is why cross-encoders are *exclusively* rerankers: you use a bi-encoder (and BM25) to get from 10M to 100, then the cross-encoder to get from 100 to 8.

**🗣 Say this in the room:** "A bi-encoder compresses the document without knowing the query, which is what makes the index possible and what caps the accuracy. A cross-encoder lets query and document attend to each other in every layer, which is why it's dramatically better and why it can only ever run on a candidate list. The whole retrieve-then-rerank architecture exists to buy cross-encoder accuracy at bi-encoder-plus-a-hundred cost."

**📄 Paper:** Nogueira & Cho (2019), *Passage Re-ranking with BERT* — monoBERT. It took the MS MARCO passage ranking leaderboard by a wide margin with the almost embarrassingly simple recipe of feeding `[CLS] query [SEP] passage [SEP]` into BERT and fine-tuning a binary relevance classifier. It replaced hand-engineered learning-to-rank feature pipelines with a single end-to-end model, and every reranker since is a descendant.

### Trace the reranker lineage for me — monoBERT to what we use today.

Worth knowing as a narrative because "which reranker do you use" is often followed by "why does that architecture exist."

**monoBERT (Nogueira & Cho, 2019)** — the cross-encoder as a binary classifier over `[CLS] q [SEP] d [SEP]`. Train with cross-entropy on relevant/non-relevant passage pairs; score with the positive-class probability. Simple, strong, and still the default architecture.

**duoBERT** — the same authors' pairwise follow-up: instead of scoring each document independently, feed *two* documents plus the query and ask which is more relevant. Better ordering, but `O(k²)` comparisons, so it was used as a second-stage rerank over monoBERT's top 30ish. It matters mostly as the intellectual ancestor of listwise LLM reranking.

**monoT5 (Nogueira, Jiang, Pradeep & Lin, 2020)** — reframed ranking as sequence-to-sequence: feed `Query: q Document: d Relevant:` to T5 and read the score as the model's probability of generating the token `true` versus `false`. The key result was that this generalizes far better in low-data and zero-shot settings than a classification head, because it reuses the pretrained language modeling objective rather than training a fresh head. This is the direct conceptual ancestor of LLM-as-reranker.

**Modern encoders — the bge-reranker family, Cohere Rerank, Voyage rerank, Jina reranker.** Architecturally these are still monoBERT: a cross-encoder producing a relevance score. What changed is the backbone (multilingual XLM-RoBERTa-class or larger), the training data (much more, much better hard negatives), and distillation from stronger teachers. `bge-reranker-v2-m3` is the open-weights workhorse — multilingual, built on the BGE-M3 backbone, and good enough that I use it as the baseline any commercial reranker has to beat before I pay for it.

**LLM rerankers (RankGPT and descendants, 2023→)** — a decoder LLM ranks a *list* of passages in one call, outputting a permutation. Different cost profile, different failure modes, covered separately below.

The through-line worth stating: **every advance moved information from "computed at index time" toward "computed at query time with the query present," and paid for it in latency.** That is the axis the whole field trades along.

### Give me the reranking latency math. I want to know whether 100 candidates fits inside a 300 ms budget.

This is exactly the arithmetic I want a candidate to be able to do out loud, so let me do it fully.

A transformer forward pass costs approximately `2 × N_params × N_tokens` FLOPs for the matmul-dominated part. Take `bge-reranker-base`, a BERT-base-class model at ~110M parameters, and a query-plus-passage sequence truncated to 512 tokens.

Per pair: `2 × 110e6 × 512 = 1.13e11 FLOPs = 0.113 TFLOP`.
For 100 candidates: `100 × 0.113 = 11.3 TFLOP`.

Now the hardware. An A10G does roughly 70 TFLOP/s of non-sparse bf16 in theory; at a realistic 35–45% model-FLOPs-utilization for a batched short-sequence encoder workload, call it **25 TFLOP/s effective**.

`11.3 / 25 = 0.45 seconds`. **100 candidates at 512 tokens does not fit in a 300 ms budget on one A10G.** That is the honest answer and the interesting part is what you do about it.

Three levers, in the order I pull them:

**Truncate.** Cost is linear in sequence length. Going from 512 to 256 tokens halves it to 226 ms. Your chunks are probably ~400 tokens; the reranker rarely needs all of them, because relevance is usually determined in the first couple of sentences. Measure the nDCG loss from truncation — in my experience truncating to 256 costs under 1 point on typical RAG chunks and buys you 2× throughput. That is a trade I take almost every time.

**Cut the candidate count.** 100 → 50 halves it again, to 113 ms at 256 tokens. Combined with truncation you are now comfortably inside budget. The recall cost of 100 → 50 is measurable on your ablation table and is usually small if your first stage is a good hybrid.

**Batch properly and use a smaller model.** All 50 pairs go in one batch — this is not optional, per-pair calls waste the GPU entirely. Sort by length and pad within the batch rather than to a global 512, which on a realistic length distribution saves another 20–35%. And consider a distilled 4- or 6-layer reranker: a 6-layer model at ~1/2 the params of BERT-base roughly halves FLOPs again.

**📐 Numbers you must know:** `2 × params × tokens` FLOPs per forward pass. `bge-reranker-base` ≈ 110M params. 100 pairs × 512 tokens ≈ 11.3 TFLOP ≈ 450 ms on one A10G at 25 TFLOP/s effective. **The memorizable version: roughly 4–5 ms per 512-token pair on a mid-range single GPU, so your candidate count times 4.5 ms is your rerank latency.** Sanity-check any vendor's latency claim against this.

**⚠ Trap:** measuring reranker latency with a warm single-request benchmark and then deploying behind a shared endpoint. At 20 QPS with 50 candidates each, you are asking for 1,000 pair-scorings per second, which is `1000 × 0.113 TFLOP = 113 TFLOP/s` — more than four A10Gs at full utilization. Reranking is a *throughput* problem in production and a *latency* problem in your benchmark, and the two sizing calculations give completely different answers.

### Write the production code for a rerank stage. Assume a self-hosted cross-encoder.

The mechanism matters more than any specific SDK, so here is the local-model version with the batching and truncation decisions made explicit.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "BAAI/bge-reranker-v2-m3"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, torch_dtype=torch.float16).eval().cuda()

@torch.inference_mode()
def rerank(query: str, docs: list[str], top_k: int = 8,
           max_length: int = 256, batch_size: int = 32) -> list[tuple[int, float]]:
    scores: list[float] = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        enc = tok([query] * len(batch), batch,            # pairwise encoding
                  padding=True, truncation="only_second", # never truncate the query
                  max_length=max_length, return_tensors="pt").to("cuda")
        logits = model(**enc).logits                       # [B, 1] for this family
        scores.extend(logits.squeeze(-1).float().tolist())
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    return [(i, scores[i]) for i in order[:top_k]]
```

The details that are load-bearing:

**`truncation="only_second"`** — truncate the document, never the query. The default `truncation=True` truncates the longest sequence, which for a long query and a short doc silently drops query tokens and destroys the score. This is a real bug I have found in shipped code.

**`padding=True` pads to the longest in the batch**, not to `max_length`. If you pass `padding="max_length"` you pay full 256-token cost on every pair regardless of actual length. On a realistic chunk-length distribution that is 25–40% wasted compute.

**Output shape varies by model family.** The BGE rerankers emit a single logit (higher is more relevant). Some cross-encoders emit two logits and you take the softmax positive class. Check `model.config.num_labels` rather than assuming; getting this wrong gives you a perfectly-inverted ranking that still "works" in the sense of not crashing.

**⚠ Trap:** cross-encoder scores are **logits, not probabilities, and not comparable across queries.** A score of 2.4 for query A and 2.4 for query B mean different things. If you want an absolute relevance threshold — "if nothing scores above X, say I don't know" — you must calibrate per model on a labeled set (fit a sigmoid, pick the threshold at your target precision), and re-calibrate whenever you change the model. Hardcoding `score > 0.5` after switching from a two-logit model to a single-logit model is how you end up returning zero results for every query.

### How do you choose between bge-reranker, Cohere Rerank, Voyage and Jina?

The axes, in the order I actually weigh them:

**Do you need to self-host?** This decides it more often than quality does. If you are handling regulated data — Harvey's legal corpora, a healthcare deployment, an on-prem enterprise install — the API rerankers are out and you are on the open-weights BGE/Jina line, full stop. Say this first in a design round; it demonstrates you know the constraint hierarchy.

**Quality on *your* data, measured.** All four vendors publish benchmark tables and all four tables show them winning. Ignore them. Run all four on 300 of your labeled queries and read the nDCG@10 and the p95 latency. The spread is usually smaller than the marketing implies, and occasionally one is dramatically better on your domain for reasons no benchmark predicted (multilingual, code, long documents).

**Language coverage.** If you serve non-English, this is a hard filter. `bge-reranker-v2-m3` and the Jina multilingual rerankers are built on multilingual backbones; several strong English rerankers degrade badly outside English and will not tell you so.

**Maximum document length.** Rerankers have context limits like anything else, and a reranker that truncates at 512 tokens on your 1,000-token chunks is scoring half your content. Some newer rerankers handle several thousand tokens, at proportionally higher cost.

**Cost model and its shape.** API rerankers typically bill per search where a "search" covers up to some number of documents, with long documents counting as multiple. **The pricing shape means your candidate count is a billing decision, which is a very different mental model from self-hosting where it's a latency decision.** Read the metering rules carefully before you build a pipeline that sends 100 candidates.

**💰 Math — the buy-vs-build crossover, which you should be able to compute live.** Assume a hosted reranker at $2.00 per 1,000 searches (📅 Volatile — verify current list pricing; the shape of the argument survives, the number will not).

- 1M queries/day = 1,000 thousand-search units/day × $2.00 = **$2,000/day = $60,000/month.**
- Self-hosted: from the arithmetic above, one A10G scores ~50 candidates at 256 tokens in ~113 ms, so ~8 QPS/GPU sustained. Average load is `1e6 / 86,400 = 11.6 QPS`; size for a 3× peak = 35 QPS → **5 GPUs**, call it 7 with headroom for failure domains. At roughly $1.00/GPU-hour on-demand: `7 × $1.00 × 730 = $5,110/month`, or under $3,000 with reserved/spot capacity.
- **Crossover: roughly $55k/month in favor of self-hosting at this volume**, against maybe 0.3–0.5 FTE of ops burden. At 10k queries/day the same math gives $20/month for the API versus $730/month for one always-on GPU, and the API wins decisively.

**🗣 Say this in the room:** "Use the hosted reranker until you cross roughly 100k reranked queries per day, then the GPU math flips hard — at 1M/day a hosted reranker at $2 per thousand searches is $60k/month against about $5k of A10G capacity. Below that threshold, paying the vendor is cheaper than the engineer-hours to run it."

### An LLM can rank passages too. Compare pointwise, listwise and RankGPT-style reranking against a cross-encoder.

An LLM reranker is **a cross-encoder with a much bigger brain and a much worse cost curve**, plus one genuinely new capability: it can see multiple candidates simultaneously and rank them *relative to each other*.

**Pointwise.** Ask the LLM, per candidate, "is this passage relevant to this query, 0–10?" This is monoT5's framing with a general LLM. Cost is `k` LLM calls, or one call per candidate — for 50 candidates at ~400 tokens each that is 20k input tokens minimum and 50 round trips unless you batch them into one request. Quality is decent, and the scores are on a comparable scale so you can threshold. The killer weakness: **LLM absolute relevance judgments are poorly calibrated and cluster** — you will get thirty 7s and no way to order them.

**Listwise / permutation generation (RankGPT).** Put 20 passages in one prompt, numbered, and ask the model to output the permutation: `[3] > [11] > [1] > ...`. Because the model sees them together, it makes *relative* judgments, which LLMs are far better at than absolute ones. For more than ~20 passages you use a sliding window: rank passages 1–20, keep the top, slide down by a stride, re-rank the overlapping window, repeat bottom-to-top. This is the RankGPT recipe and it produces genuinely excellent ordering.

**📄 Paper:** Sun, Yan, Ma, Ren, Yin & Ren (2023), *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents* — introduced permutation-generation listwise reranking with the sliding-window strategy and showed a general-purpose LLM could beat supervised cross-encoders on BEIR zero-shot. It also distilled the behavior into a small specialized ranker, which is the practically important half.

**The latency price, with arithmetic.** Take 50 candidates at 400 tokens each = 20,000 input tokens. Listwise with 20-passage windows and stride 10 needs 4–5 LLM calls. At a mid-tier model's throughput, each call has ~400–800 ms TTFT plus generation of a short permutation, so **2–4 seconds of added latency**, versus ~110 ms for a cross-encoder. Cost: `20,000 input tokens × ~4 windows-worth of overlap ≈ 40k tokens × $3/Mtok = $0.12 per query`. At 100k queries/day that is **$12,000/day**. Compare to a self-hosted cross-encoder at effectively fractions of a cent per query. **LLM reranking is 100–1000× more expensive than a cross-encoder.**

**When I actually use it:** offline or async workloads where 3 seconds is free — batch document processing, a research agent that already takes 40 seconds, generating training labels. And critically, **as a teacher**: use the LLM reranker to label 50k query-passage pairs, then distill that into a cross-encoder you can afford to serve. That is the standard play and it captures most of the quality at serving costs you can defend.

**⚠ Trap:** listwise LLM rerankers exhibit strong **position bias** — the model favors passages that appear early in the prompt, independent of content. The mitigation is to shuffle the input order and average across two or more permutations, which doubles your already-expensive call. If you evaluate a listwise reranker without shuffling, you are measuring your first-stage retriever's ordering as much as the LLM's judgment.

### Explain ColBERT and MaxSim. Why is late interaction a different point on the curve rather than just "a better bi-encoder"?

ColBERT sits **exactly between the bi-encoder and the cross-encoder**, and the design goal was to keep the precomputability of the former while recovering some of the token-level interaction of the latter.

The bi-encoder's information bottleneck is pooling: you take `[T, d]` contextual token vectors and crush them to `[d]`. Everything the model knew about individual tokens is gone. ColBERT's insight is: **don't pool.** Store all `T` token vectors per document, projected down to a small dimension (128 in the original). At query time, encode the query into its own `[Tq, 128]` token vectors, and score with **MaxSim**:

```
score(q, d) = Σ_{i=1}^{Tq}  max_{j=1..Td}  ( q_i · d_j )
```

For each query token, find its single best-matching document token, and sum those maxima over query tokens. That is it — the whole late-interaction mechanism is a max and a sum.

Why this is powerful: it is **soft term matching in embedding space.** BM25 matches the query token `TLS` against the document token `TLS` exactly. ColBERT matches the query token `TLS`'s contextual vector against whichever document token is closest — which might be `TLS`, or `SSL`, or `certificate`. You get per-token grounding (so rare identifiers get a strong signal from their own token, not a diluted pooled vector) plus semantic flexibility. It also has the property that the score is naturally *interpretable*: you can show which document token each query token matched, which is a real product advantage for citation UIs.

Why it is "late" interaction: the query and document representations are still computed independently — the document's token vectors are precomputable at index time — and the interaction happens only at the cheap MaxSim step. A cross-encoder's interaction happens in every attention layer, which is why it cannot be precomputed.

**📄 Paper:** Khattab & Zaharia (2020), *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. It introduced late interaction with MaxSim and showed it recovered most of the cross-encoder's accuracy at orders-of-magnitude lower query latency, replacing the assumption that you had to choose between a single-vector index and a full cross-encoder.

**⚠ Trap:** MaxSim is not symmetric and it is not a distance. It sums over *query* tokens only, so it does not penalize a document for containing lots of irrelevant content — a long document with the right tokens somewhere in it scores as well as a focused one. In practice this means ColBERT has a mild long-document bias and you should keep your chunks length-controlled rather than assuming the scorer will handle it.

### What does ColBERT cost to store, and how did ColBERTv2 and PLAID fix it?

The storage blowup is the reason ColBERT was a research curiosity for two years, and the numbers are worth memorizing because this is the standard objection you must be able to answer.

**Naive ColBERT v1 storage:** 128 dimensions × 2 bytes (fp16) = **256 bytes per token**. Take a realistic corpus: 10M chunks × 300 tokens = 3 billion token vectors.

`3e9 × 256 bytes = 768 GB.`

Compare to a single-vector dense index at 1024-dim fp32: `10e6 × 4 KB = 40 GB`. **ColBERT costs ~19× a dense index.** At that ratio nobody ships it, which is exactly what happened.

**ColBERTv2 (Santhanam et al., 2022)** fixed it with residual compression. Cluster all token vectors into centroids; store per token a centroid id plus a heavily quantized residual (1–2 bits per dimension). That is roughly `4 bytes (centroid id) + 128 dims × 2 bits / 8 = 32 bytes` ≈ **36 bytes per token**, with 1-bit residuals getting you closer to 20.

`3e9 × 36 bytes = 108 GB.` Now it is ~2.7× a dense index instead of 19×, which is a number you can actually take to an infrastructure review. ColBERTv2 also contributed *denoised supervision* — training with cross-encoder-distilled scores and hard negatives — which is where much of its quality gain came from, independent of the compression.

**PLAID (Santhanam et al., 2022)** is the serving engine. Naive ColBERT retrieval requires an ANN lookup per query token and then MaxSim against every candidate's full token set, which is slow. PLAID adds centroid-based pruning: use the centroid ids to cheaply identify which documents could plausibly score highly, aggressively prune, and only decompress residuals for the survivors. Reported speedups were large — several-fold on GPU and roughly an order of magnitude on CPU — turning ColBERT from "research latency" into a servable system.

**💰 Math — the RAM question that actually decides this.** 108 GB does not fit on one machine's RAM cheaply. An `r6i.4xlarge`-class instance (128 GB) runs roughly $1.00/hour = ~$730/month, and you need replication, so call it 2–3 of them: **$1,500–2,200/month** just to hold the index, versus a 40 GB dense index which fits comfortably on a single smaller node at maybe $250/month. (📅 Volatile — instance pricing changes; the ratio is the durable part.) That is a ~6–8× infrastructure cost multiplier, and it has to be justified by a quality delta on your ablation table.

**🗣 Say this in the room:** "ColBERT's honest cost is per-token storage. Uncompressed that's 256 bytes a token, about 19× a single-vector index, which is why v1 didn't ship. ColBERTv2's residual compression gets it to roughly 20–36 bytes a token — call it 2.7× dense — and PLAID makes the query side servable. I'd evaluate it as a *replacement* for hybrid-plus-reranker, not as an addition, and I'd want it to beat that pipeline on both nDCG and p95 before I take a 3× storage bill."

### Would you deploy ColBERT today, or hybrid plus a cross-encoder?

Hybrid plus a cross-encoder, in almost every case I have faced — and I want to give the reasons rather than just the verdict, because this is a live disagreement in the field and an interviewer may hold the other position.

**The case for hybrid + reranker:** every component is off-the-shelf and independently swappable. Your lexical index is Postgres or Elasticsearch, both of which your ops team already understands. Your dense index is pgvector or Qdrant. Your reranker is an HTTP call or a single GPU pod. Any one of them can be upgraded without touching the others, and each has a legible failure mode. Debugging is tractable: you can look at a BM25 score, a cosine, and a reranker logit and reason about each.

**The case for ColBERT:** it is one system instead of three, it gets much of the cross-encoder's accuracy without a per-query GPU forward pass over candidates, its latency is more predictable because it has no separate reranking stage, and the token-level match evidence is a genuinely nice product feature. For a team that is going to invest deeply in retrieval as core IP — a search company rather than a company with search in it — that consolidation is real.

**What decides it:** operational maturity and whether retrieval is your product. ColBERT's ecosystem is thinner. Vespa supports late interaction natively and is the most credible production path; there is real support in a few vector databases and in the RAGatouille/ColBERT reference stacks, but you are choosing a less-trodden road. If retrieval quality is the product — Perplexity, a legal-search company — that investment can pay. If retrieval is one component of a broader product, hybrid + reranker gets you 90–95% of the quality with a fraction of the operational novelty.

**⚠ Trap:** presenting this as settled. The genuinely honest framing is that late interaction, learned sparse (SPLADE) and hybrid-plus-cross-encoder all sit in a similar quality band, and published comparisons are heavily influenced by which one the authors built. **The decision rule I would state is: pick the one whose failure modes your team can debug at 3am, then prove it on your own labeled set.** Interviewers at strong companies respect "these are close and here's my tiebreaker" far more than a confident ranking you can't defend.

### Can you get cross-encoder quality without paying cross-encoder latency? Explain distillation into a bi-encoder.

Yes, partially, and this is the highest-leverage move in the whole area because it moves quality **upstream** — into the index, where it costs nothing at serving time.

The idea: a cross-encoder is an expensive oracle. Use it offline to produce soft relevance scores for a large number of (query, passage) pairs, then train your bi-encoder to reproduce those scores. The bi-encoder can never fully match the teacher — it has the pooling bottleneck — but it can learn a much better-shaped embedding space than it would from binary relevance labels alone, because the teacher's scores contain *graded* information about how relevant each negative is.

The canonical loss is **Margin-MSE**: rather than matching absolute scores (which are on incomparable scales), match the teacher's *margin* between a positive and a negative:

```
L = MSE( s_student(q, d+) − s_student(q, d−),
         s_teacher(q, d+) − s_teacher(q, d−) )
```

This is scale-free in a useful way: the student only has to reproduce the teacher's *ordering strength*, not its calibration. A KL-divergence variant over a softmax across multiple negatives works similarly and is what most modern recipes use.

**📄 Paper:** Hofstätter, Althammer, Schröder, Sertkan & Hanbury (2020), *Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation* — introduced Margin-MSE distillation from a cross-encoder teacher into bi-encoder and late-interaction students. It replaced the practice of training retrievers only on binary positive/negative labels, and it is why modern open embedding models are so much better than 2019-era DPR.

**The pipeline in practice:**
1. Mine hard negatives with your current retriever — for each training query, take documents ranked 10–100 that are not labeled relevant.
2. Score every (query, positive) and (query, hard-negative) pair with the cross-encoder teacher. **This is the expensive step and it is offline.**
3. Train the bi-encoder with Margin-MSE or KL against those scores.
4. Re-mine negatives with the improved retriever and repeat. Two rounds is typical; gains flatten fast after that.

**💰 Math:** teacher scoring for 100k queries × 20 candidates = 2M pairs at 512 tokens. At `2 × 110e6 × 512 = 0.113 TFLOP` per pair, that is `2e6 × 0.113 = 226,000 TFLOP`. On one A10G at 25 TFLOP/s effective: `226,000 / 25 = 9,040 seconds ≈ 2.5 GPU-hours` — call it **$3 of compute, plus the fine-tuning run.** For a permanent serving-time quality improvement with zero added query latency, that is the best ROI in this entire section.

**⚠ Trap:** distillation moves quality upstream but it does **not** eliminate the reranker's advantage, and teams conclude it does. The student still has the single-vector bottleneck; it still cannot do the token-level literal matching a cross-encoder does. What distillation buys is a better *candidate pool* — higher recall@50 — which makes your reranker more effective and sometimes lets you shrink the candidate count. Measure it as a recall improvement in the first stage, not as a reranker replacement.

### Our reranker helps on the eval set but users say results got worse. What are the likely causes?

This is the reranking-specific failure taxonomy, and I would work it in this order.

**1. Truncation is silently dropping the relevant passage.** Your chunks are 800 tokens, your reranker's `max_length` is 512, and the answer is in the last paragraph. The reranker scores the first 512 tokens, sees generic preamble, and ranks the correct chunk low. Symptom: reranking *hurts* specifically on long chunks. Diagnostic: bucket eval results by chunk length and look for a cliff. Fix: chunk shorter, or score with a sliding window and take the max, or use a longer-context reranker.

**2. The wrong field is being reranked.** You retrieve on `body` but pass `title + summary` to the reranker, or you pass the chunk without the contextual header that made it interpretable. The reranker sees text that does not contain what matched. Diagnostic: log the exact string sent to the reranker for a failing query and read it. I have found this bug more than once and it is always embarrassing when someone else finds it.

**3. Score-threshold drift after a model swap.** You had `if score < 0.5: drop`, calibrated for a model emitting sigmoid probabilities, and someone upgraded to a model emitting raw logits. Now everything is dropped or nothing is. Symptom: sudden, total, and correlated with a deploy — check this first if the regression is not gradual.

**4. Your eval set doesn't contain the queries users actually ask.** The reranker genuinely improves the queries you measured and degrades a class you didn't. The usual culprit is that eval sets are built from documentation and users ask about *recent* things, conversational follow-ups, or their own account. Diagnostic: sample 50 real failing queries from logs and check whether any of them resemble your eval set.

**5. Diversity collapse.** A reranker optimizes per-document relevance with no notion of the set. Five near-identical chunks all genuinely rank highest, so all five go into the context, and the LLM now has one fact repeated five times instead of five facts. Aggregate nDCG *improves* while answer quality drops, because nDCG is computed per-document and does not know your downstream consumer needs coverage. **This is the failure that most cleanly demonstrates why retrieval metrics are not answer metrics**, and it is the reason MMR exists.

**6. Latency pushed you past the abandonment threshold.** The results are better and 900 ms slower, and users who used to skim three results now bounce. If your "worse" signal is engagement rather than explicit thumbs-down, check the latency histogram before you touch relevance at all.

**🗣 Say this in the room:** "When a reranker helps offline and hurts online, my first three checks are truncation against chunk length, whether the string I send the reranker is the string I indexed, and whether nDCG improved while *set coverage* got worse — because a reranker scores documents independently and will happily return five copies of the same paragraph."

### How do you fit reranking inside a strict TTFT budget when you are streaming a response?

The constraint to state first: **reranking is on the critical path to the first token and cannot be overlapped with generation**, because the LLM cannot start until its context is assembled. Every millisecond of reranking is a millisecond of TTFT. That makes it structurally different from, say, a background enrichment call.

Budget arithmetic for a chat product with a 1.5 s TTFT target:

```
query embed + rewrite    :  120 ms   (skip the rewrite when there's no chat history)
hybrid retrieval (‖)     :   60 ms   (max of BM25 and ANN, run concurrently)
rerank 50 @ 256 tok      :  115 ms
dedup + MMR + assembly   :   15 ms
──────────────────────────────────
retrieval subtotal       :  310 ms
LLM prefill + first token: 400–900 ms depending on context length
──────────────────────────────────
total TTFT               : 710–1210 ms   ✓ inside budget
```

The levers when you blow the budget, in the order I use them:

**Run the retrievers concurrently.** `asyncio.gather` on the BM25 query and the ANN query. This is free and I am amazed how often it isn't done — the sequential version costs you the sum instead of the max, typically 30–40 ms wasted.

**Overlap the query embedding with the lexical query.** BM25 doesn't need the embedding. Fire it immediately.

**Truncate aggressively and measure the cost.** 512 → 256 tokens halves rerank latency for typically under 1 nDCG point.

**Two-tier reranking.** A tiny 4-layer distilled reranker scores all 100 in ~30 ms, cut to 20, then a large reranker scores those 20 in ~45 ms. Total 75 ms for close to full-model quality. This is the cascade pattern from classic web search and it is underused in RAG.

**Adaptive skipping.** If the fused top-1 RRF score is far above the top-2 — a confident, unambiguous retrieval — skip reranking entirely. On typical traffic this fires on 20–35% of queries, most of them head queries, and buys back the whole reranking budget for the ambiguous tail where it actually matters.

**Cache the whole retrieval result keyed on the normalized query.** Head queries repeat heavily. A Redis cache on `(normalized_query, corpus_version, acl_scope)` with a short TTL routinely serves 25–40% of traffic at 2 ms. Note the `acl_scope` in the key — omitting it is how you leak one tenant's results to another, and it is the single most dangerous cache bug in enterprise search.

**💰 Math:** if 30% of queries hit that cache, your reranker fleet shrinks by 30%. At the 1M queries/day sizing from earlier — 7 GPUs at $730/month = $5,110 — a 30% hit rate saves `0.30 × $5,110 = $1,533/month` and removes 310 ms of TTFT from a third of your traffic. That is a better return than almost any model-level optimization available to you.
