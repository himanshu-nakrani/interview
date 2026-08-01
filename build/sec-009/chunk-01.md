### Before you design anything, what's the checklist you run to decide whether this feature needs an LLM at all?

The mental model: an LLM is a probabilistic, non-idempotent, rate-limited, per-call-billed remote dependency with no schema guarantee and a p99 measured in seconds. You already have an instinct for what that kind of dependency costs you — you would never put one on a hot path without a very specific reason. The entire gate is just forcing yourself to name that reason before you write the prompt, because once a prompt exists nobody ever deletes it.

The checklist I actually run, in order, and I run it out loud in design review:

1. **Is the output space finite and known?** If the answer is one of N labels, one of N routes, or a number, you are describing a classifier or a query, not a generator. The LLM is only justified if N is large, open, or changes weekly.
2. **Is the input format stable?** Fixed-layout documents, well-formed JSON, a known CSV schema — a parser is exact, costs microseconds, and fails loudly. LLMs are for the long tail of format, not the head.
3. **Does a deterministic system already know the answer?** If the fact lives in Postgres, the correct architecture is SQL. An LLM that reads rows and re-derives an aggregate is a slow, expensive, occasionally-wrong replacement for `SUM()`.
4. **Is there a correctness authority other than human taste?** If a solver, a type checker, a test suite, a regex or a business rule can *verify* the answer, the LLM's job is to propose, not to decide.
5. **What is the cost of being wrong 3% of the time, and who eats it?** For an autocomplete suggestion, nothing. For a compliance decision, an enforcement action.
6. **Does the head of the distribution repeat?** If 40% of queries are 200 distinct strings, precompute them and the LLM only sees the tail.
7. **Do I have labels?** With 5,000 labeled examples, a supervised model is usually both cheaper and better. With 40 examples and an open output space, the LLM is genuinely the only thing that works.
8. **What is my latency budget and my QPS?** At 2,000 QPS with a 50 ms budget, the conversation is over — no frontier API meets that, and you are now discussing a distilled small model or a classical model.

**🗣 Say this in the room:** "My gate is: known finite output space, stable input format, an existing source of truth, and an external verifier. If three of those four are true, I ship a deterministic system and use the model only where the format is genuinely open — and I'd rather spend the model budget on the 8% tail than on the 92% that a parser handles exactly."

**⚠ Trap:** the failure mode this checklist prevents is not "we used an LLM unnecessarily." It is *entanglement*. Once the model is on the path, the schema becomes suggestive rather than enforced, retries become non-idempotent, your p99 becomes the provider's p99, and every downstream consumer starts defensive-parsing. Removing it six months later is a rewrite. That is why the gate runs before the prompt exists and not after.

### A PM wants an LLM to pull invoice number, date and total out of our vendor PDFs. The layout is identical every time. What do you tell them?

I tell them no, and then I tell them where the LLM *does* go, because "no" alone reads as incapacity.

The mental model: extraction from a fixed layout is a parsing problem, and parsers have a property no model has — they fail loudly on unexpected input instead of confidently inventing a plausible value. If a vendor changes their template, a regex returns `None` and your pipeline raises. A model returns an invoice number that looks exactly like an invoice number and is wrong. The first failure costs you an alert; the second costs you a wrong payment and a reconciliation project.

The mechanism I ship: text layer via `pdfplumber` or `pypdf` (or `pdftotext -layout` if the layout is column-sensitive), then anchored regex or positional extraction, then a validator — invoice number matches the vendor's known format, date parses and is within 400 days, total is a decimal that equals the sum of line items. Three layers, all deterministic.

```python
import re, datetime
INV = re.compile(r"Invoice\s*(?:No\.?|#)\s*[:\-]?\s*([A-Z]{2,4}-\d{6,8})", re.I)
TOT = re.compile(r"Total\s+Due\s*[:\-]?\s*\$?([\d,]+\.\d{2})", re.I)

def extract(text):
    inv, tot = INV.search(text), TOT.search(text)
    if not (inv and tot):
        raise ExtractionMiss(text[:400])       # loud, routed to the LLM fallback
    return {"invoice_no": inv.group(1),
            "total": Decimal(tot.group(1).replace(",", ""))}
```

Where the model earns its keep is the `ExtractionMiss` branch. Onboard a new vendor, or a template changes: the miss rate for that vendor spikes, the failures route to a vision-capable model with a strict JSON schema, a human confirms the first ten, and the confirmed extractions become the new anchors. That is the pattern I would defend in any design review — **deterministic head, model tail, and the miss rate is the metric that tells you when to re-cut the boundary.**

**💰 Math:** 200k invoices/month. Regex path: ~2 ms CPU each, effectively free — call it one small worker, $30/month. Model path at a frontier vision model, ~1,800 input tokens and 120 output tokens per invoice: at $3/Mtok in and $15/Mtok out that is 1,800 × 3e-6 + 120 × 15e-6 = $0.0054 + $0.0018 = **$0.0072/invoice**, so 200k × $0.0072 = **$1,440/month** if you route everything. With a 96% regex hit rate you route 8,000 invoices and pay **$58/month**, a 25× reduction — and more importantly your p50 goes from ~900 ms to ~3 ms. **📅 Volatile:** verify per-token prices before your loop.

**⚠ Trap:** "the model handles layout variation for free" is the seductive and wrong claim. It handles variation *silently*, which is the problem. With regex you can measure your miss rate exactly and it is a first-class metric on a dashboard. With a model your error rate is unmeasurable without a labeled sample, and nobody builds that labeled sample until after the incident.

### We currently route support tickets into one of twelve queues by calling a frontier model on every ticket. Talk me out of it — or into keeping it.

Twelve fixed labels is the textbook definition of a supervised classification problem, and you almost certainly have the training data already: every ticket ever routed, with its final queue after human correction, is a labeled example. If you have been running the LLM router for three months at even 5k tickets/day you are sitting on ~450k labeled rows. That is an embarrassment of data for a 12-class problem.

The mechanism, in ascending cost order. First, embed the ticket text with a small embedding model and fit multinomial logistic regression on top of the embeddings. That is genuinely ~15 lines and typically lands within a point or two of the LLM on a 12-class routing task, because routing is mostly topical and embeddings encode topic extremely well. Second, if the classes have strong lexical signal (product names, error codes), a TF-IDF + linear SVM baseline is often *better* than embeddings and costs nothing. Third, if you have per-queue exemplars rather than labels, embed the exemplars, take cosine similarity, and threshold — the nearest-centroid approach — which needs no training at all and gives you an abstention band for free.

```python
# embed once at ingest; you are probably already embedding for search
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
clf.fit(X_train_emb, y_train)                 # X: [N, 1024] float32
probs = clf.predict_proba(x_emb)              # [12]
top, conf = probs.argmax(), probs.max()
route = QUEUES[top] if conf >= 0.62 else "escalate_to_llm"
```

That last line is the whole design. You do not replace the LLM; you demote it to the abstention branch. Calibrate the threshold on held-out data to hit whatever routing accuracy the support org demands, and the fraction of traffic above threshold is your cost saving.

**💰 Math:** 5,000 tickets/day = 150k/month. LLM path at ~900 input tokens, 15 output tokens: 900 × 3e-6 + 15 × 15e-6 = $0.0027 + $0.000225 ≈ **$0.0029/ticket** → **$437/month**, p50 ~700 ms. Classifier path: embedding at ~$0.02/Mtok is 900 × 2e-8 = $0.000018/ticket → **$2.70/month**, and the logistic regression forward pass is a 1024×12 matmul, roughly 25 μs. If 85% clears the threshold, you pay $2.70 + 0.15 × $437 = **$68/month**. The dollars are small at this volume; the interesting number is latency — routing becomes synchronous and sub-millisecond instead of a 700 ms async job, which changes the product.

**🗣 Say this in the room:** "Twelve fixed labels with three months of human-corrected routing history is a supervised problem, not a generation problem. I'd fit a linear head on embeddings, calibrate an abstention threshold, and keep the LLM strictly for the low-confidence tail. The win isn't the $370 a month — it's that routing becomes deterministic, testable, and 25 μs."

**⚠ Trap:** do not let anyone tell you the classifier "can't handle a new queue." Adding a 13th queue means refitting a logistic regression on 450k rows, which takes about 40 seconds. The real constraint is *labels for the new class*, and the honest answer is: run the embedding-threshold or LLM path for the new class for two weeks, harvest labels, then fold it in. That is a boring, solved MLOps loop, and saying so is a seniority signal.

### Our "ask your data" feature works by dumping rows into the context window and asking the model for the answer. What would you ship instead?

The mental model that ends this discussion: a language model asked to compute `SUM(revenue) WHERE region='EMEA' AND quarter='Q3'` is performing arithmetic in a representation that was never designed for arithmetic, over a sample of rows that fit in the context, with no guarantee it saw all of them. Postgres does this exactly, over all rows, in milliseconds, and has done for thirty years. The model's job is **translation**, not computation.

The correct architecture has three deterministic components and one probabilistic one:

- **Schema-grounded generation.** The model sees the table DDL, column descriptions, a handful of gold query examples, and the question. It emits SQL — never an answer.
- **A validation gate.** Parse the generated SQL with `sqlglot`, assert it is a single `SELECT`, assert every referenced table and column exists in the schema catalog, assert the tenant predicate is present, reject anything with DDL/DML tokens. This is a whitelist, not a blacklist — blacklisting SQL injection patterns in generated SQL is how you get owned.
- **Execution under a leash.** Read-only role, `statement_timeout`, `LIMIT`, per-tenant row-level security. The database, not the prompt, enforces access.
- **Rendering.** The result set goes back through the model only to phrase it in English, with the numbers passed through verbatim, or better: rendered by a template so the model cannot perturb a digit.

Then there is the part everybody skips. For a real analytics product, the *head of the question distribution is tiny*. "What was revenue last quarter", "top 10 customers by ARR", "churn this month" — maybe 200 canonical questions cover 60% of traffic. Those become **precomputed metrics with hand-written, reviewed SQL**, matched by embedding similarity with a high threshold. Text-to-SQL only runs on the tail.

**🔍 Failure taxonomy — text-to-SQL in production, in the order these bite:**
1. *Semantic-not-syntactic errors.* The SQL runs, returns a number, and the number is wrong because the model joined on the wrong key or ignored a soft-delete flag. Undetectable without gold-query evals. This is 80% of your real error mass.
2. *Ambiguous business terms.* "Revenue" means bookings to sales and recognized revenue to finance. No model resolves this; a semantic layer does.
3. *Silent partial filters.* The tenant predicate gets dropped and you leak cross-tenant data. Mitigated only by RLS at the database, never by prompt instructions.
4. *Cost bombs.* An unbounded join across two fact tables. Mitigated by `statement_timeout` and a cost check via `EXPLAIN` before execution.
5. *Stale schema in the prompt.* A column is renamed; generation silently degrades. The schema block must be generated from the live catalog at request time, not pasted into a prompt file.

**🗣 Say this in the room:** "The model writes the query; the database computes the answer. I gate the generated SQL through a parser whitelist and run it as a read-only role with RLS and a statement timeout, and I evaluate on gold question-SQL pairs where I compare result sets, not query strings — because two different queries can both be right and string match will tell you nothing."

### Legal wants an LLM to decide whether a transaction is allowed under our sanctions policy. Push back.

This is the highest-value "no" in the whole section, and the way you say it matters as much as the content.

The mental model: a compliance gate is a *decision with an audit trail*. What a regulator asks for is not accuracy, it is **reproducibility and explanation** — the same input must produce the same decision today and in the deposition three years from now, and you must be able to name the rule that produced it. A model with temperature 0 is still not reproducible across a provider's silent version bump, and "the model said so" is not an explanation. That is a legal-defensibility argument, not an ML argument, and it is the one that wins the room.

So the architecture inverts. Deterministic rules make the decision; the model does the unstructured work that feeds them and the unstructured work that follows them:

- **Before the gate:** the model normalizes messy input — extracts the counterparty name, address and beneficial-owner text from a free-form document into a strict schema. Fuzzy name matching against a sanctions list is itself a classical problem (Jaro-Winkler, phonetic keys, an ML scorer trained on adjudicated matches), not an LLM problem.
- **The gate itself:** a rules engine over the normalized record. Versioned, unit-tested, with every rule traceable to a clause in the policy document. `BLOCK if counterparty in SDN_LIST or jurisdiction in EMBARGOED or (amount > threshold and kyc_tier < 2)`.
- **After the gate:** the model *drafts* the analyst's narrative — "flagged because the counterparty matched an SDN entry at 0.91 similarity" — for a human to review and sign. Drafting is a real, valuable, well-scoped LLM job.

**⚠ Trap:** "we'll use the LLM but log its reasoning for audit." Chain-of-thought text is a post-hoc narrative, not a causal trace of the computation, and treating it as an audit record is the most dangerous mistake in this entire domain. It is well documented that models produce explanations that do not correspond to the features actually driving the output. If your audit artifact is model-generated prose, you have an audit artifact that can be confidently, fluently wrong, and it will be read in court as if it were a log line.

**🗣 Say this in the room:** "I'd put the model on both sides of the gate and never inside it. It normalizes messy input into a schema and it drafts the analyst's rationale, but the block/allow decision comes from a versioned rules engine, because the regulator's requirement is reproducibility and a traceable rule — and generated reasoning text is a narrative, not a trace."

The general form of this pattern is worth naming, because it generalizes far past compliance: **the LLM handles the unstructured boundary; a deterministic core handles the decision.** Refunds above a threshold, medical dosing, access control, pricing, credit limits — same shape every time.

### Half our chat traffic is the same forty questions. How do you exploit that, and where does it go wrong?

The mental model: query distributions in every product I have measured are brutally Zipfian, and a Zipfian head is a caching problem — which you already know how to solve. The only new thing is that the cache key is fuzzy, and that fuzziness is where the danger lives.

Three tiers, and I would build them in this order.

**Tier 1 — curated answers for the true head.** Pull the top 200 queries by volume from a month of logs, cluster near-duplicates, and have a human write or approve the answer for each. These are not cached model outputs; they are content. Served from Postgres or Redis in single-digit milliseconds, versioned, reviewable, and correct by construction. For a docs assistant this routinely covers 30–50% of traffic.

**Tier 2 — exact-match cache.** Normalize (lowercase, strip punctuation, collapse whitespace) and hash. Zero risk of a wrong hit. Catches maybe another 5–10%.

**Tier 3 — semantic cache.** Embed the query, ANN-search the cache, and return the stored answer if cosine similarity exceeds a threshold. This is the one that gets people fired.

**⚠ Trap:** the semantic cache returns a *semantically close, factually wrong* answer. "How do I cancel my Pro plan?" and "How do I cancel my Team plan?" sit at cosine ~0.94 with most embedding models, and they have different answers. A threshold tuned on a similarity histogram rather than on labeled pairs will absolutely serve one for the other. Three rules I enforce: (a) the threshold is chosen on a hand-labeled set of near-miss pairs, not on the score distribution; (b) anything the query distinguishes by a *named entity* — plan name, product, version, region — is excluded from semantic caching entirely, or the entity is part of the cache key; (c) personalized or account-scoped answers are never semantically cached, only exact-cached within a user scope.

**💰 Math:** 1M chats/month, RAG-flavored, ~6k input tokens and 400 output tokens each. Uncached: 6,000 × 3e-6 + 400 × 15e-6 = $0.018 + $0.006 = **$0.024/chat** → **$24,000/month**. Now layer it: 35% curated (free), 8% exact-hit (free), 20% semantic-hit (embedding only, ~$0.0001) — that leaves 37% hitting the model, so 370,000 × $0.024 = **$8,880**, plus ~$100 of embedding, ≈ **$9,000/month**. A **62% reduction**, and the 43% served from tiers 1–2 comes back in ~8 ms instead of ~4 s.

And note what else you get: the curated tier is your regression suite. Those 200 questions with approved answers are exactly the eval set you were going to have to build anyway.

### Give me the full comparison: a 200-line scikit-learn model versus a frontier API, at one million requests a month. Cost, latency, accuracy — all of it.

Let me fix a concrete task so the numbers mean something: binary content classification — does this user-generated post violate policy — at 1M posts/month (~23 QPS average, ~70 QPS peak), with 1.2% positives and 50k human-labeled examples in hand.

**Cost.** The sklearn path is a gradient-boosted tree over TF-IDF plus a few hand features, or logistic regression over embeddings. Say embeddings: 1M × 300 tokens × $0.02/Mtok = 300M tokens × 2e-8 = **$6/month** for embedding, plus inference on two `c6i.large` instances for redundancy at ~$62/month each = **$130/month all-in**. The frontier path at ~450 input tokens (post + a compact policy prompt) and 8 output tokens: 450 × 3e-6 + 8 × 15e-6 = $0.00135 + $0.00012 = **$0.00147/post** → **$1,470/month**. With prompt caching on a 2,000-token policy preamble at a 90% cached-input discount, the cached portion costs 2,000 × 3e-7 = $0.0006 instead of $0.006 — but note the preamble was not in my 450 tokens, so realistically the honest comparison is **$1,470–$2,000/month vs $130/month, roughly 11–15×**. Using a small hosted model at ~$0.15/Mtok input instead: 450 × 1.5e-7 = $0.0000675 → **$68/month**, which is genuinely competitive. **📅 Volatile:** all four prices need re-verification.

**Latency.** GBDT over TF-IDF: feature hashing ~200 μs, tree ensemble forward ~50 μs, so **p99 well under 5 ms** end-to-end including network. Logistic-regression-over-embeddings adds an embedding call, ~30–60 ms p50, ~150 ms p99 — the embedding hop dominates entirely. Frontier API with 8 output tokens: TTFT ~250–500 ms, plus 8 tokens at ~25 ms/token ≈ 200 ms, so **p50 ~600 ms, p99 1.5–3 s**, and the p99 is not yours to control. That is a 100–600× latency ratio, and it decides whether moderation can be synchronous (block before publish) or must be asynchronous (publish then retract) — a *product* difference, not an infra one.

**Accuracy.** Here is where I refuse to be dogmatic. With 50k labels on a well-specified policy, a tuned GBDT will typically beat a zero-shot frontier model on the *head* of the policy — the common, lexically-signposted violations — often by a wide margin on PR-AUC, because it has learned your actual label distribution including your annotators' idiosyncrasies. The frontier model wins decisively on: novel phrasings it has never seen, multi-hop reasoning ("is this a threat given the referenced event?"), code-switched and low-resource languages, and *any policy that changed last Tuesday*, where the classifier has zero labels and the model needs a prompt edit. It also gives you a rationale string for the appeals queue, which the tree cannot.

**What I would actually ship**, and this is the answer: a cascade. GBDT scores everything at 5 ms. Confident-clean (say `p < 0.02`, ~88% of traffic) auto-passes. Confident-violation (`p > 0.85`, ~1.0%) auto-actions. The uncertain band — ~11% — goes to the frontier model, whose output both makes the decision and becomes a training label. Cost: $130 + 0.11 × $1,470 = **$292/month**. Latency: p50 5 ms, p89 5 ms, and only the ambiguous 11% pay the second. Accuracy: better than either alone, because the model is spending its capability on exactly the examples where capability matters.

**📐 Numbers you must know:** the cascade savings factor is just `1 / escalation_rate` on the LLM line. At 11% escalation you pay 9× less than routing everything; at 30% you pay 3.3× less. This means **the entire economics of a cascade live in the abstention threshold**, and tuning that threshold is a one-afternoon job with a labeled set. It is the highest-ROI hour in applied AI engineering and almost nobody does it deliberately.

**🗣 Say this in the room:** "I'd build both and cascade them. The classical model handles the 89% where it's confident at 5 ms and $130 a month; the frontier model handles the ambiguous 11% and its outputs become labels that shrink that band over time. That gets me sub-10ms p50, an order of magnitude less spend than routing everything, and a system that gets cheaper as it runs."

### How do you say "I would not use an LLM here" in an interview without sounding like you can't build with LLMs?

This is a real interview skill and it is graded, so let me be explicit about the mechanics.

The failure mode is answering with a refusal: "I wouldn't use an LLM for that." It reads as either dogma or inability, and the interviewer cannot distinguish the two. The fix is structural: **never lead with the negative, and never end without giving the LLM a job.**

The three-move pattern I use:

**Move 1 — name what the LLM is uniquely good at, so they know you know.** "The thing a frontier model gives me that nothing else does is handling unbounded input format and open output spaces without labeled data."

**Move 2 — show that this problem doesn't have that shape, with a specific property.** Not "it's simple" — a *property*: the output space is 12 fixed labels; the input is a fixed template; the answer is already in Postgres; there's a verifier. This is the move that signals seniority, because it shows you evaluated rather than pattern-matched.

**Move 3 — put the LLM somewhere real anyway.** Bootstrapping labels, the abstention branch, generating the eval set, drafting the human-reviewed rationale, handling the tail. There is almost always a genuine job for it, and offering one proves the "no" was analysis and not allergy.

Worked example, delivered end to end:

**🗣 Say this in the room:** "A model is what I reach for when the input format is unbounded or the output space is open and I don't have labels. Here neither is true — twelve fixed queues and three months of human-corrected routing history — so I'd fit a linear classifier on embeddings and calibrate an abstention threshold. I'd still use the model in two places: it labels the bootstrap set for a new queue before I have data, and it takes the low-confidence tail. That's a 25-microsecond p50 on 85% of traffic and the model spending its budget where it actually adds information."

Two calibration notes. First, **read the room's incentive**. If you are interviewing at a company whose product *is* an LLM product, the gate question is usually testing whether you know the boundary, not whether you will refuse — so lead with the LLM's genuine strengths and be crisp about the carve-out. Second, **if they push back, hold the line with a number, not with conviction.** "At 70 QPS peak with a 50 ms budget, no API round trip fits — that's the constraint, not a preference" ends the argument. Restating your opinion louder does not.

### Fine — flip it. What are the positive criteria that make an LLM genuinely the right call?

I want to be equally rigorous in the other direction, because an engineer who only knows how to say no is just as useless as one who says yes to everything.

An LLM is the right tool when **the input format is unbounded, the output space is open, and there is no cheap labeled dataset** — and it becomes the *only* tool when two or more of those hold simultaneously.

Concretely, the cases where I reach for a model without hesitation:

- **Open-ended generation with no single correct answer.** Drafting, summarizing, rewriting, explaining. There is no `SUM()` for "write a release note from this diff."
- **Zero-shot on a task that will be defined next week.** The killer property of a frontier model is that the spec lives in a prompt, editable in a deploy, with no retraining. When policy changes weekly, that is worth an enormous amount of latency and money.
- **Semantic understanding over long, messy, heterogeneous text.** Reading a 40-page contract and answering "does this have an auto-renewal clause" is not a regex problem and never was.
- **Long-tail coverage where per-class labels will never exist.** 4,000 intents with a power-law distribution: you will have labels for 30. The tail is the model's.
- **Compositional instruction-following.** "Extract these seven fields, but if the doc is an amendment, follow the parent contract's numbering." Encoding that as rules is possible and horrible; the maintenance cost dominates.
- **Anything requiring natural-language output for a human to read.** Even in a fully deterministic system, the last mile is often prose.
- **Code generation and transformation**, where the crucial property is that a compiler and a test suite verify the output — the LLM proposes, the toolchain disposes.

**⚠ Trap:** "it needs reasoning, so it needs an LLM." Most tasks people describe as reasoning are lookup plus arithmetic, both of which have exact solutions. The honest test for whether a task needs a model is: *write down the decision procedure*. If you can write it down completely, implement it — you just did the hard part. If you cannot write it down because the cases are unbounded, that is the genuine signal for an LLM, and it is a much narrower signal than it feels like.

### Someone says "only about 20% of a production agent system is the LLM." What's the other 80%, and why should I care that you know that?

The claim is a direct descendant of a well-known systems observation: **📄 Paper:** Sculley et al. (2015), "Hidden Technical Debt in Machine Learning Systems" — the famous diagram showing the ML code as a small box surrounded by configuration, data collection, serving infrastructure, monitoring and process management. The LLM era rediscovered it. The model call is a function invocation; the system around it is the product.

The other 80%, enumerated because vagueness here is a tell:

- **Retrieval and context assembly** — chunking, indexing, hybrid search, reranking, dedup, the token budget allocator that decides what gets dropped when the context overflows. This alone is often 30% of the code and 60% of the quality.
- **Tool layer** — schemas, argument validation, timeouts, retries with backoff, idempotency for side-effecting tools, permission scoping per user, and result truncation so a 40k-token API response doesn't blow the window.
- **Control flow and state** — the loop, the step budget, termination conditions, checkpointing so a 40-step trajectory can resume, and human-in-the-loop interrupts. This is a durable-workflow problem and you already know how to build it.
- **Output contracts** — schema-constrained decoding or validate-and-repair, and what you do on the third failure.
- **Evaluation** — the offline suite, the regression gate in CI, the LLM-judge with its own calibration set, online metrics, and the annotation pipeline that feeds all of it.
- **Observability** — per-step traces, token and cost attribution per tenant and per feature, prompt version stamped on every span.
- **Safety and policy** — input/output filters, prompt-injection defenses, PII redaction, rate limiting keyed on tokens rather than requests.
- **Cost and capacity control** — per-tenant token budgets, model routing, caching tiers, fallback when the provider degrades.

**🗣 Say this in the room:** "The model is a stateless function call with a bad p99 and no schema guarantee. Everything that makes it a product — retrieval, tools, the control loop, output contracts, evals, cost attribution — is ordinary distributed-systems engineering, which is exactly why my backend background transfers. I've watched teams spend a quarter tuning prompts when the actual defect was that their chunker split tables across boundaries."

Why interviewers care: this framing is the single fastest way to distinguish someone who has shipped from someone who has done tutorials. Tutorial-shaped answers are all prompt. Shipped-shaped answers spend most of their words on the surrounding machinery — and that machinery is where a senior backend engineer is already strong, which is the strategic reason to lead with it.

### I've got 400 labeled examples and a classification task. Few-shot a frontier model, or train something?

Both, in a specific order, and the order is the answer.

The mental model: 400 examples is squarely in the zone where the right move is not "pick one" but "use the model to escape the data-poverty trap." Labels are the scarce resource; compute is not.

The ladder I run:

**Step 1 — spend 100 of them on an eval set, immediately, before anything else.** Stratified by class, held out, never trained on, never looked at during prompt iteration except through an aggregate number. If you have 400 labels and you spend zero on evaluation, every subsequent decision is guesswork. This is the step people skip and it is the one that is graded.

**Step 2 — few-shot the model as the baseline.** 8–16 examples in the prompt, measure on the eval set. This is your reference point and it takes an hour. It also tells you whether the task is even well-defined: if a frontier model with 16 examples gets 61% on a 5-class problem, your *label definitions* are ambiguous, and no amount of modeling fixes that. Go re-adjudicate the taxonomy.

**Step 3 — fit a classical model on the 300 training examples over embeddings.** With 300 examples and a 1024-dim embedding, logistic regression with strong L2 is the right estimator — it will not be the best possible model but it will not overfit catastrophically either. Compare on the same eval set. My honest prior: below ~500 examples per class the LLM usually wins; above ~2,000 per class the linear head usually wins on the head classes and loses on the tail. Between those, it is genuinely task-dependent and you must measure. Anyone who tells you the crossover without knowing the task is guessing.

**Step 4 — if the LLM wins, use it to make more labels.** Run it over 20k unlabeled examples, keep only high-confidence predictions, hand-audit a 200-row sample of those to estimate pseudo-label precision, then train the classical model on the union. This is distillation-by-labeling and it is the highest-leverage move in the whole ladder. Then re-run step 3.

**⚠ Trap:** fitting on 400 examples and reporting cross-validated accuracy after you tuned the regularization strength on the same folds. You have now selected a hyperparameter on your evaluation data and your reported number is optimistic by several points — at n=400 the standard error on an accuracy estimate is already about `sqrt(0.85 × 0.15 / 100) ≈ 3.6%` on a 100-example test set, so a 2-point "improvement" is noise. Report a confidence interval or do not report a comparison.

**🏋 Drill:** take any public 5-class text dataset, subsample to 400 rows, and in 45 minutes produce a table with four rows — zero-shot, 8-shot, logistic-regression-on-embeddings, and LLM-pseudo-labeled-then-fit — each with accuracy, macro-F1 and a bootstrap 95% CI on a fixed 100-row eval split. Pass criterion: you can state which differences are real and which are inside the interval, without hedging.
