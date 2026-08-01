### Explain the TruLens RAG triad, and tell me where it overlaps with RAGAS.

The triad is a framing device more than a novel metric set, and its value is that it is memorable enough to reason with on a whiteboard. Three edges of a triangle whose vertices are **query**, **context**, and **response**:

**Context relevance** (query → context): is the retrieved material actually relevant to what was asked? This is the retrieval-side check, and it is the one that catches "we retrieved five chunks about billing when the user asked about SSO."

**Groundedness** (context → response): is every statement in the response supported by the retrieved context? This is the hallucination edge.

**Answer relevance** (query → response): does the response actually answer the question asked? This catches the evasive answer, the answer to a different question, and the padded non-answer.

The reason the triangle works pedagogically is that **all three must hold, and each fails in a way the other two cannot detect.** High groundedness with low context relevance = a faithful answer to the wrong material. High answer relevance with low groundedness = a fluent, on-topic hallucination — the most dangerous quadrant, and the one that user satisfaction scores reward. High context relevance and groundedness with low answer relevance = the system found the right material, quoted it accurately, and did not answer the question, which is what over-cautious "here is what the documentation says" prompting produces.

Overlap with RAGAS: groundedness ≈ faithfulness (same construct, both claim-decompose-and-entail), answer relevance ≈ answer relevancy (same construct), context relevance ≈ context precision minus the ranking-awareness. So roughly three of RAGAS's five, restated. **What the triad lacks and RAGAS has is context *recall*** — the triad has no edge for "was anything missing," because it only evaluates what was retrieved, never what should have been. That is a serious gap: a system that retrieves one perfectly-relevant chunk out of four needed scores 1.0 on all three triad edges and gives an incomplete answer.

How I actually use them: I do not run both frameworks. Pick one, pin the judge model and prompts, and **own your metric definitions** — the failure mode is a team that reports RAGAS faithfulness in Q1 and TruLens groundedness in Q2 and believes the numbers are comparable. They are not; different decomposition prompts produce differently-calibrated scores on the same data. If you switch, re-baseline everything.

**⚠ Trap:** believing these frameworks give you an absolute quality score you can put in an OKR. They do not. A faithfulness of 0.87 means "0.87 under this judge, this prompt, this decomposition, this dataset." It is useful as a *relative* signal between two configurations measured identically, and meaningless as a cross-team or cross-quarter absolute. I have watched a team celebrate a faithfulness improvement that was entirely caused by a judge model upgrade.

### What is ARES and when would you prefer a trained judge over a prompted one?

**📄 Paper:** Saad-Falcon, Khattab, Potts & Zaharia (2023/2024) — *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*. Its structure is worth knowing because it addresses the two things a naively-prompted LLM judge gets wrong.

The mechanism has three parts. **First**, generate synthetic training data from your own corpus — query/passage/answer triples with both positive and deliberately corrupted negative examples. **Second**, fine-tune small, cheap classifier-style judges on that synthetic data, one per axis (context relevance, answer faithfulness, answer relevance). **Third**, and this is the contribution that matters most, use a **small human-labeled validation set with prediction-powered inference** to produce statistical confidence intervals on the system's true scores rather than a bare point estimate. PPI is the machinery for combining many cheap noisy machine labels with a few expensive accurate human labels to get a *tighter, bias-corrected* interval than either alone.

When is a trained judge better than a prompted frontier model?

**Cost and throughput at scale.** A fine-tuned ~100M–1B classifier runs on cheap hardware at hundreds of items/second. A frontier judge at ~$0.02/judgment on 500k daily responses is 500,000 × $0.02 = **$10,000/day, $300k/month.** The same volume through a self-hosted classifier is one or two GPUs — call it $2–3/hour, ~$2,000/month. 📅 Volatile pricing. This is why online, every-request grounding checks in production are almost always small trained models and offline evaluation is frontier models.

**Consistency over time.** Your prompted judge's underlying model gets silently updated; your fine-tuned checkpoint does not. For a metric you plan to trend over four quarters, a pinned local checkpoint is the only honest instrument.

**Calibration on domain-specific notions of support.** In a legal or medical corpus, "supported by the context" has domain-specific meaning that a general judge gets wrong in a consistent direction. Training on domain-labeled examples fixes exactly that.

When is a prompted frontier judge better? When you have no labels, when the axis is subtle or novel, when you are evaluating a few hundred items offline, and when you want to be able to change the rubric next Tuesday without a training run. That is most teams most of the time, which is why prompted judges dominate despite being worse on every axis ARES optimizes.

**🗣 Say this in the room:** "The pattern I'd take from ARES is: generate synthetic judge training data from my own corpus, distil a small judge, and keep a few hundred human labels aside to calibrate it with a confidence interval rather than reporting a bare number. Prompted frontier judges offline, distilled judges online — the crossover is roughly when you're judging more than a few tens of thousands of items a day."

### Design span-level citations with verifiable character offsets. I want the whole pipeline.

The mental model: **a citation is only worth anything if a machine can verify it without a model.** Anything the LLM asserts about its own sourcing is unverified — it is a claim, not a citation. So the design goal is to make every citation reducible to a string-equality check against immutable stored text.

**Storage layer.** Chunks are immutable and carry provenance: `chunk_id`, `doc_id`, `content_version`, `start_char`, `end_char` into the *canonical extracted text* of the document, plus the display coordinates (page number, bounding box for PDFs, heading path for markdown). The canonical text is stored, never regenerated — if you re-extract, offsets shift, and every historical citation silently becomes wrong. That is why `content_version` is part of the chunk key.

**Prompt layer.** Present each chunk with an explicit, short, stable identifier, and instruct the model to emit citations referencing those identifiers plus a verbatim quote. Something like:

```
[C3] (doc: payments-runbook.md § Retries)
The webhook dispatcher retries a failed delivery 5 times with exponential
backoff starting at 2 seconds, and then moves the event to the DLQ.
```

with the instruction: *"After each factual sentence, cite as `[C<n>: "<exact quote from that chunk>"]`. The quote must appear verbatim in the cited chunk."*

**Verification layer — this is the part that makes it real.** For each emitted citation, locate the quote inside the cited chunk's stored text. Exact substring first; if that fails, normalized match (collapse whitespace, unify quotes and dashes, casefold); if that fails, a bounded fuzzy match with a similarity floor around 0.9. On success you now have `start_char = chunk.start_char + offset_within_chunk` and `end_char`, which are **verified absolute offsets into the source document**. On failure the citation is rejected. Rejection policy is a product decision: strip the citation and mark the sentence unsupported, retry generation once with the failure fed back, or refuse to render the answer. I default to strip-and-mark, plus a metric.

```python
def verify_citation(quote: str, chunk) -> tuple[int, int] | None:
    text = chunk.text
    i = text.find(quote)
    if i == -1:
        nq, nt, m = normalize(quote), *normalize_with_map(text)   # m: norm idx -> raw idx
        j = nt.find(nq)
        if j == -1:
            return None
        i, quote = m[j], text[m[j]: m[j + len(nq)]]
    return (chunk.start_char + i, chunk.start_char + i + len(quote))
```

**Render layer.** The UI receives `(doc_id, content_version, start_char, end_char)` and highlights the exact span in the rendered document, or draws the bounding box on the PDF page. The user can see the sentence the claim came from. That is the product feature; everything above is the machinery that makes it not a lie.

**📐 Numbers you must know:** track **citation verification rate** — verified / emitted. On a decent frontier model with a clear instruction I expect 0.92–0.98 with exact+normalized matching. Below ~0.90 the model is paraphrasing rather than quoting and you need to tighten the instruction or switch to an ID-only citation scheme with post-hoc entailment instead of quoting. Also track **coverage** — the fraction of factual sentences carrying at least one verified citation — which is a different and more important number, because a model can achieve 100% verification rate by citing once and asserting ten uncited sentences.

**⚠ Trap:** storing offsets into the *rendered* or *cleaned* text while the UI renders the original. Every whitespace normalization, header strip and ligature fix shifts offsets. Store the offset map from canonical to raw, or store offsets into raw and canonicalize only in memory. Getting this wrong produces highlights that are consistently off by 20–200 characters, which reads to users as the system citing the wrong sentence — indistinguishable from a hallucinated citation.

### What does quote-then-answer prompting buy you, and what does it cost?

Mental model: **you are forcing the model to do extraction before synthesis, in the output stream, where you can inspect it.** The model must first copy the relevant spans verbatim, then write the answer conditioned on its own copied spans. Two mechanisms make this help.

First, **copying is a much easier task than synthesizing**, and getting the evidence into the immediate left-context means the answer is generated with the evidence tokens a few hundred positions away rather than five thousand tokens back behind eight chunks. That directly attacks lost-in-the-middle and distractor interference: the model has effectively re-retrieved into a short, clean working set.

Second, **it produces an inspectable intermediate.** You can verify the quotes against the source before you render the answer, and you can measure whether the quotes actually support the answer, independently. Without the quote step, you have only the final text and a post-hoc guess about what it was based on.

The costs, honestly stated:

**Output tokens.** Quotes are typically 100–400 output tokens, and output tokens are the expensive ones. At $15/Mtok output, adding 300 output tokens is 300 × 15/1e6 = **$0.0045 per request**. At 200k requests/day that is $900/day, **$27,000/month**. 📅 Volatile. That is not a rounding error and it needs a quality number to justify it.

**Time to first *useful* token.** If the quotes stream first, the user watches a wall of quoted text before the answer starts. At ~60 output tokens/sec, 300 quote tokens is **5 seconds** before the answer begins. For a chat product that is unacceptable. Two fixes: emit quotes in a structured field the UI hides until the answer completes and renders as a footnote drawer, or invert the order (answer first, citations after) and lose the extraction-before-synthesis benefit. I generally choose the hidden-field version, which means committing to structured output.

**Copying is not free of error.** Models paraphrase while believing they are quoting, especially over long or list-heavy chunks. Which is precisely why the verification layer is mandatory rather than optional.

**When I use it:** high-stakes domains where a wrong claim is expensive (legal, medical, financial, compliance), and any surface where a human will act on the answer without opening the source. **When I skip it:** conversational surfaces with tight latency budgets, and low-stakes internal search where citation-by-chunk-ID with a post-hoc entailment check gets most of the benefit at a fraction of the output cost.

**🗣 Say this in the room:** "Quote-then-answer moves the evidence into the model's immediate left-context and gives me a verifiable intermediate. It costs roughly three hundred output tokens — about forty-five hundredths of a cent per call at fifteen dollars per million — and five seconds of perceived latency unless I hide it in a structured field. I'd ship it for anything legal or medical and skip it for conversational search."

### Implement post-hoc NLI entailment checking over an answer's claims. Give me the loop and the latency budget.

The mental model: **groundedness is a natural-language-inference problem, and NLI is a task with small, fast, well-trained models — you do not need a frontier LLM to ask "does this passage entail this sentence."**

The pipeline is three stages.

**Stage 1 — claim decomposition.** Split the answer into atomic, independently-checkable claims. Sentence splitting is the cheap version and it is wrong often enough to matter: "The retry limit is 5 and the backoff starts at 2 seconds" is two claims and can be half-true. The better version is an LLM decomposition pass into atomic facts, each rewritten to be self-contained (pronouns resolved, entities restated) — this is the mechanism from **📄 Paper:** Min et al. (2023) — *FActScore*, which decomposed long-form generations into atomic facts and scored each against a knowledge source, and established the practice.

**Stage 2 — evidence selection.** For each claim, pick candidate premises: the cited chunk if there is a citation, plus the top-3 retrieved chunks by embedding similarity to the claim. Do not use the whole context as one premise — NLI models have short input limits, and a long premise dilutes the signal.

**Stage 3 — entailment.** Run an NLI cross-encoder on each (premise, claim) pair, take the max entailment probability over premises, and threshold. **📄 Paper:** Laban, Schnabel, Bennett & Hearst (2022) — *SummaC* — showed that the way to make NLI models work for document-level consistency is exactly this: score at sentence-pair granularity and aggregate, rather than feeding the whole document as one premise, which is where earlier NLI-based approaches failed.

```python
def groundedness(answer: str, chunks: list[Chunk], nli, embed, tau: float = 0.5) -> dict:
    claims = decompose(answer)                       # LLM call, or sentence split
    cvecs, kvecs = embed(claims), embed([c.text for c in chunks])
    results = []
    for claim, cv in zip(claims, cvecs):
        top = [chunks[i] for i in topk_by_cosine(cv, kvecs, k=3)]
        scores = nli([(c.text, claim) for c in top])  # batched, one forward pass
        best = max(range(len(top)), key=lambda i: scores[i]["entailment"])
        results.append({
            "claim": claim,
            "p_entail": scores[best]["entailment"],
            "p_contradict": scores[best]["contradiction"],
            "premise_chunk": top[best].id,
            "supported": scores[best]["entailment"] >= tau,
        })
    return {"score": sum(r["supported"] for r in results) / max(len(results), 1),
            "claims": results}
```

**Latency budget.** For a 6-claim answer: decomposition is one LLM call, ~300 in / ~200 out, roughly 700–1,200 ms — this is the dominant cost and the first thing to drop if you need speed (fall back to sentence splitting, ~0 ms). Embedding 6 claims: one batched call, ~40 ms local or ~80 ms hosted. NLI on 18 pairs: a ~180M-parameter cross-encoder at ~256 tokens per pair, batched as one forward pass on a modern GPU, **15–40 ms**. Total ~800–1,300 ms with LLM decomposition, **~100 ms without it.**

That 100 ms number is the important one, because it means **NLI groundedness checking is affordable synchronously on every production response** if you skip LLM decomposition. You run it after generation completes, before final render, and you have a per-response groundedness score in production, on 100% of traffic, for the cost of a GPU you were probably already running for the reranker.

**⚠ Trap:** treating "not entailed" as "hallucinated." NLI models are conservative and label plenty of correct-but-implicit inferences as neutral. If the chunk says "retries occur at 2, 4, 8, 16 and 32 seconds" and the claim is "there are five retries," a strict NLI model may return neutral — that requires counting. So calibrate `tau` on labeled data and report the *distribution*, not a binary. The correct production use is: contradiction probability above a threshold is a **hard alarm** (that is a real, checkable defect), while low entailment is a **soft flag** for review. Contradiction is high-precision; non-entailment is not.

### The citation looks completely plausible but points at the wrong chunk. How do you catch that at scale?

This is the most insidious failure in the whole grounding stack, because **it defeats the exact defense it appears to satisfy.** The user sees a citation, clicks it maybe 5% of the time, and gains false confidence in an unverified claim. And it passes any check that only verifies "does the cited chunk exist."

Three mechanisms produce it. **One:** the model states a true fact it retrieved from chunk 7 and attaches the citation marker for chunk 2, because marker-to-content binding over eight chunks is a bookkeeping task models are mediocre at — the citation is a *fact* the model has to track, and it is competing with everything else in the context. **Two:** the model states a fact from parametric memory and attaches whichever citation looks topically closest, which is confabulation with a paper trail. **Three:** the claim genuinely draws on chunk 7 *and* chunk 2 and the model cited only one — under-citation rather than mis-citation, less harmful but it shows up in the same metric.

Detection, cheapest first:

**Quote verification** catches the crude cases for free (string match, sub-millisecond) — if the model must quote and the quote is not in the cited chunk, you have caught it deterministically. This is the strongest argument for quote-then-answer.

**Claim-to-cited-chunk entailment.** For each cited sentence, run NLI with *the cited chunk* as the premise. If entailment is low, the citation is unsupported. Then — and this is the diagnostic step — run NLI against *every other retrieved chunk*. If some other chunk entails it strongly, you have a **mis-attribution**: the fact is grounded, just cited wrong. If no chunk entails it, you have a **hallucination**. Two very different bugs with very different fixes, distinguished by one extra batched NLI pass over ~8 pairs, which is another ~20 ms.

**📄 Paper:** Gao, Yen, Yu & Chen (2023) — *Enabling Large Language Models to Generate Text with Citations* (the ALCE benchmark) — formalized exactly this as **citation precision** (of the citations emitted, how many actually support their claim) and **citation recall** (of the claims made, how many are fully supported by their citations), both computed with an NLI model. Those are the two numbers to put on your dashboard. They are also a genuinely useful thing to say in an interview, because most candidates only know "faithfulness."

The fixes, once you have measured it:

Reduce the bookkeeping load — fewer chunks in context (5 beats 15 for citation accuracy, independently of answer quality), and short stable IDs (`[C1]`, not a UUID). Structure the output so each claim carries its citation in a schema field rather than as inline markup the model has to interleave with prose. And for high-stakes surfaces, **repair rather than reject**: if NLI finds a different chunk that entails the claim, rewrite the citation to point there and log it. Never silently show the user a citation you have measured as wrong.

**💰 Math:** the entailment sweep over all retrieved chunks for a 6-claim answer with 8 chunks is 48 NLI pairs — one batched forward pass, ~40 ms on the GPU you already have for reranking, ~$0.00002 of amortized GPU time. Against the cost of a customer acting on a mis-attributed compliance answer, this is free. Run it on 100% of traffic in regulated domains and on a 10% sample everywhere else.

### Define attribution formally. What does "attributable to identified sources" actually mean?

**📄 Paper:** Rashkin et al. — *Measuring Attribution in Natural Language Generation Models* — introduced the **AIS** (Attributable to Identified Sources) framework, and its value is that it gives you a precise, human-annotatable definition where the industry otherwise uses "grounded" to mean six different things.

The definition, and it is worth stating carefully: a statement *s* made by a system is **attributable** to a source set *S* if a generic hearer would affirm the sentence *"According to S, s"* — given the surrounding context of the interaction. Three properties of that definition do real work.

**It is about the source's claim, not the truth.** If the retrieved document says the deadline is 30 days and it is wrong, an answer saying "the deadline is 30 days" is perfectly attributable and factually incorrect. Attribution and factuality are orthogonal axes. This is exactly the faithfulness-is-not-correctness point, given a formal home.

**It requires interpretability of the statement.** AIS annotation is a two-step protocol: first, is the statement even interpretable as a standalone claim in context? Ambiguous, elliptical, or context-dependent statements are un-annotatable and must be flagged before you ask about support. This is why claim decomposition must produce *self-contained* claims — an un-decontextualized claim ("it retries five times" — what does *it* refer to?) cannot be scored at all, and if you score it anyway you are generating noise.

**It is defined over the full statement**, so partial support is not attribution. "The webhook retries 5 times over 30 seconds" where the source says 5 retries but never mentions 30 seconds is *not* attributable, even though most of it is.

Why this matters operationally: it gives you the annotation guideline for building a human-labeled grounding set, which you need in order to calibrate any automatic checker. Without it, two annotators will disagree about half the time on "is this supported" and your gold labels will be worthless. With it, agreement is high enough to train against — and once you have a few hundred AIS-annotated examples, you can measure whether your NLI checker's threshold is set anywhere near a human's notion of support, which is the calibration step that nearly everyone skips.

**🗣 Say this in the room:** "Attribution in the AIS sense means a reader would accept 'according to this source, X' — it's about faithfulness to the source, explicitly not about the source being right. I keep those as two separate metrics, because a system can be perfectly attributable and consistently wrong if my corpus is stale, and only one of those two failures is fixable in the generation layer."

### Retrieval returned nothing useful and the model answered anyway, confidently and wrongly. What happened, and what do you do about it?

Mechanistically: **you asked a model that has parametric knowledge of your domain to answer a question, and you handed it some irrelevant text. Nothing in that setup instructs it to prefer silence.** The model's training overwhelmingly rewards producing a helpful answer. The retrieved context is not a constraint, it is just more tokens in the prompt; if it is unhelpful, the model falls back on what it knows, and what it knows about "SaaS retry policies" is a generic industry average that will sound exactly like your documentation.

This is the highest-severity RAG failure because the output is *maximally plausible*. A hallucinated citation can be checked. A hallucinated fact drawn from the model's genuine world knowledge is often *nearly* true — the standard retry count really is 3 or 5 at most companies — and it is indistinguishable from a retrieved fact by tone, specificity, or confidence.

Four defenses, in order of how much they actually buy:

**One — measure it first, with a designed slice.** You cannot manage this without a number. Build the unanswerable slice described earlier: take real questions, remove their supporting documents from the index, and measure the **answer rate on unanswerable queries.** A system that answers 70% of them has a serious problem that no user complaint will ever surface cleanly. This number is the KPI for this failure mode and almost nobody has it.

**Two — a retrieval-quality gate before generation.** If the top reranker score is below a calibrated threshold, or the score distribution is flat (top score not meaningfully above the rank-20 score), do not call the generator at all. Return "I could not find anything in the knowledge base about this." This is a *deterministic* control, which is why I like it: it does not depend on the model cooperating. Calibrate the threshold on your labeled set to a chosen precision/coverage point, and re-calibrate whenever the reranker changes, because raw scores are not comparable across models.

**Three — make abstention an explicit, exampled instruction.** Not "if you don't know, say so" buried in paragraph four of the system prompt. A structured decision: *"First determine whether the provided context contains sufficient information to answer. If it does not, respond only with INSUFFICIENT_CONTEXT and the specific information that is missing."* Give two few-shot examples of correct abstention. The specificity of the missing-information field matters — it forces the model to actually inspect the context rather than pattern-match on "am I confident."

**Four — post-hoc groundedness enforcement.** Run the NLI check; if the fraction of supported claims falls below a threshold, suppress the answer and abstain. This catches what slipped through, at the cost of latency and the awkwardness of retracting a streamed answer, which is a real UX problem worth designing around (stream to a buffer, render on pass).

**⚠ Trap:** believing "answer only from the context" in a system prompt solves this. It reduces the rate; it does not eliminate it, and the residual is the dangerous part. I have measured prompt-only defenses at somewhere between 50% and 80% reduction in unwarranted answering depending on model and domain — meaningful, and nowhere near a control. **Prompt instructions are a mitigation, not a guarantee, and the difference matters in a design review.** The deterministic score gate is the control.

### Design abstention. How do you get the system to say "I don't know" without it saying that all the time?

Abstention is a **precision/coverage trade-off with a tunable knob**, and the entire design question is where you put the knob and who gets to move it. Anyone who answers this without naming the trade-off has not shipped it.

The knob: as you make abstention more aggressive, you catch more unanswerable queries (good) and you also refuse more answerable ones (bad). Plot it explicitly — coverage on the x-axis (fraction of queries answered), accuracy-on-answered on the y-axis. Every abstention policy is a point on that curve, and the correct point depends on the cost ratio between a wrong answer and a non-answer. **In a legal or medical product a wrong answer costs 50× a non-answer, so you sit at 60% coverage and 97% accuracy. In an internal search tool a non-answer costs almost as much as a wrong one, so you sit at 95% coverage and 85% accuracy.** State the ratio out loud; it is the actual design input and it comes from the product, not from you.

The signals I combine, because no single one is sufficient:

**Retrieval score signals** — top reranker score, and the gap between rank 1 and rank 10. The gap is more robust than the absolute score because it is self-normalizing per query.
**Retrieval count** — how many chunks cleared the relevance floor. Zero or one is a strong abstention signal for questions needing synthesis.
**Model self-assessment** — the explicit sufficiency judgment, as a separate cheap call or as the first field of a structured output.
**Post-hoc groundedness** — fraction of claims entailed by context.
**Query-type routing** — an enumeration query ("list all…") over a filtered set where the filter matched 3 documents should abstain differently from a lookup query.

Combine them with a **small calibrated classifier, not a hand-tuned rule stack.** Logistic regression over five features, trained on a few hundred labeled (query, retrieval, answer-was-correct) triples. It takes an afternoon, it gives you a probability you can threshold anywhere on the curve, and unlike a rule stack it does not need re-tuning by hand every time a component changes. This is one of the few places where a classical ML model is unambiguously the right tool and saying so is a good signal in a room full of people reaching for another LLM call.

**The UX half, which is half the design.** Abstention is not a dead end; it is a branch. Good abstention offers: what *was* found and why it was insufficient ("I found the v2 policy but you asked about v4"), a suggested reformulation, an escalation path (open a ticket, ask a human, search the web), and a capture hook so the miss lands in your golden set. A bare "I don't know" is a product failure even when it is the epistemically correct output.

**🗣 Say this in the room:** "Abstention is a coverage-versus-precision curve, and the first thing I'd ask the product owner is the cost ratio between a wrong answer and a non-answer — that number picks the operating point. I'd combine retrieval-score gap, hit count, an explicit sufficiency check and post-hoc groundedness into a small calibrated classifier rather than a threshold stack, and I'd measure over-abstention as a paired regression on the answerable slice so I can see exactly what the safety knob costs."

### How do you measure over-abstention? Give me the experiment design.

You measure it as a **paired regression on the answerable slice**, and the word "paired" is doing all the work.

The setup. Your eval set has two disjoint slices: **answerable** (the golden chunks exist in the index and the question is answerable from them) and **unanswerable** (deliberately constructed by removing the supporting documents, plus real out-of-scope questions). Two configurations: baseline (current abstention policy) and candidate (the new, more aggressive one). Run both configurations on both slices, same queries, same order, same seed.

Four numbers come out, and you need all four together:

- **Abstention rate on unanswerable** — this should go UP. This is the benefit.
- **Abstention rate on answerable** — this is the **over-abstention rate**, and it should not go up much. This is the cost.
- **Accuracy on answered, answerable queries** — should go up slightly (you are filtering out the hard ones).
- **Effective useful-answer rate on answerable** = (1 − over-abstention) × accuracy-on-answered. This is the number that actually reflects user value on the queries the system is supposed to handle, and it is the one that catches a "safety improvement" that made the product worse.

Then the paired part. For each query in the answerable slice, record the outcome under both configs as a categorical: (answered-correct, answered-wrong, abstained). Build the **2×3 contingency of transitions**, and this is the artifact I actually want on the page:

```
                          candidate
baseline          correct   wrong   abstain
  correct            141       3        18   <-- 18 REGRESSIONS: was right, now refuses
  wrong                4      21        27   <-- 27 wins: was wrong, now honest
  abstain              2       1        33
```

Now you can say something precise: the candidate policy converted 27 wrong answers into honest abstentions and destroyed 18 correct answers, on 250 answerable queries. Whether that trade is good is the cost-ratio question — at a 3:1 cost ratio it is clearly good, at 1:1 it is a wash. A bare "abstention rate went from 8% to 19%" tells you none of that, and it is what most teams report.

Statistics: use **McNemar's test** on the discordant cells (18 versus 27). That is precisely the right test for paired binary outcomes and it uses only the cells where the two systems disagree, which is why paired designs are so much more sensitive here — the 141 queries both got right contribute nothing to the variance.

**⚠ Trap:** reporting abstention improvements measured only on the unanswerable slice. Any policy can drive unanswerable-abstention to 100% by abstaining always. **Over-abstention on the answerable slice is not a secondary metric, it is the denominator of the claim**, and a safety change presented without it should not pass review. I have seen a "hallucination reduction of 60%" that was a 22-point drop in answered queries.

### Should you show a user a retrieval confidence score? What is your position?

My position: **show a categorical confidence derived from a calibrated signal, never a raw number, and only if you have measured that the category correlates with accuracy.** Most implementations fail all three clauses.

Why not a raw number. A cosine similarity of 0.78 or an "87% confidence" means nothing to a user and, worse, it means nothing to *you* unless calibrated. Raw retrieval scores are not comparable across queries — score distributions depend on query length, vocabulary and language. A user who learns that 0.78 was good yesterday and bad today has learned that your number is noise. And an LLM asked for a self-reported confidence emits a plausible float with weak correlation to correctness; it is a hallucinated number wearing a lab coat.

What a calibrated categorical means, precisely: if you show "high confidence," the *measured* accuracy of answers in that bucket on your eval set should be, say, ≥95%, and you should be able to state that. "Medium" might be 80%. The calibration is the product promise. Build it by binning your retrieval-quality features, measuring accuracy per bin on the labeled set, and choosing bin boundaries to hit accuracy targets — then re-measure quarterly and after every component change, because calibration drifts the moment you touch the reranker.

The stronger design, and the one I usually argue for: **replace the confidence display with a legibility display.** Instead of "87% confident," show the citations, show the source document titles and dates, and highlight the cited spans. This gives the user something actionable — they can check — rather than something to defer to. Users are far better at judging "is this the right document" than at interpreting a probability, and it does not require you to defend a calibration curve to a regulator.

Where an explicit signal genuinely earns its place: **negative signaling.** "This answer is based on documents last updated in 2023" or "I found only partial information about this" or "these two sources disagree." Specific, actionable, and directly tied to a mechanism you can compute. That is worth building. A generic confidence percentage is not.

**⚠ Trap:** confidence theater increasing over-trust. A displayed "high confidence" badge measurably raises the rate at which users accept an answer without checking the citation. If the badge is not calibrated, you have built a mechanism that converts your errors into *acted-upon* errors. Uncalibrated confidence display is worse than no display, and that is the argument to make in the design review.

### Your summarizer merged facts from two different documents into a single sentence that is true of neither. Why does that happen, and how do you detect it?

Mechanism first, because the mechanism explains why this is not a "bug" you can prompt away. Attention operates over the whole context uniformly; there is no architectural notion of "document boundary." Once chunk A ("the enterprise plan has a 4-hour SLA") and chunk B ("the pro plan supports SSO") are in the same context, the model is free to attend to both while producing any token. When it generates a sentence about a plan's features, both chunks are live evidence and the generated sentence can inherit attributes from both: "the pro plan has a 4-hour SLA and supports SSO." Every individual fact is grounded. The **conjunction** is not.

This is why it survives naive faithfulness checking. A claim-level checker that decomposes into "the pro plan supports SSO" and "there is a 4-hour SLA" finds both entailed, and scores 1.0. **Decomposition destroys exactly the binding you needed to check.** This is the single most important thing to say about this failure, and it is a genuinely good interview answer because it shows you understand the limits of your own tooling.

Conditions that make it likely: near-duplicate chunks describing parallel entities (product tiers, regions, versions, customers) — parallel structure is precisely what invites cross-contamination; chunks stripped of their identifying header, so the model cannot tell which entity a paragraph is about; and any summarization or comparison prompt, which explicitly invites cross-document synthesis.

Detection, three layers:

**Entity-scoped decomposition.** Decompose into claims that *retain* their subject entity — "the Pro plan has a 4-hour SLA," not "there is a 4-hour SLA" — and require the entailing premise to contain the same entity. This is a decomposition-prompt change plus a filter, and it recovers most of the detection power.

**Single-source attribution requirement.** Require every claim to be entailed by a *single* chunk, not by the union of the context. Union-entailment is where merged facts hide. If a claim genuinely needs two chunks, that is a legitimate multi-hop inference — and it should be *flagged for review*, not silently accepted, because merged-fact errors and genuine multi-hop inferences are structurally identical from the outside.

**Structural prevention.** Prepend an explicit entity header to every chunk at ingest time (this is what contextual retrieval buys you beyond retrieval quality), and delimit chunks unambiguously in the prompt with their entity. Then, for parallel-entity corpora, prefer a per-entity map-reduce: answer per document independently, then combine the *answers* — which is FiD's isolation property implemented in the prompt layer, and it structurally prevents cross-contamination during the extraction step.

**🔍 Failure taxonomy — how to tell these apart in a trace:** claim entailed by a single chunk with matching entity → fine. Claim entailed by a single chunk but the entity in the claim does not appear in that chunk → **attribute transfer, this is the merge bug.** Claim entailed only by the union of two chunks → multi-hop, flag for review. Claim entailed by nothing → hallucination. Four outcomes, one entailment sweep plus an entity-match check.

### A PM asks whether you can just fine-tune the model on the company docs so you can drop the RAG pipeline. Answer them.

No, and I want to be precise about *why*, because "fine-tuning doesn't add knowledge" is repeated as a slogan and the slogan is slightly wrong in a way that gets people into arguments they lose.

**What fine-tuning actually does well:** it changes the *distribution* of outputs — style, format, tone, task framing, adherence to a schema, domain vocabulary, and the mapping from an input shape to an output shape. If your model writes in the wrong register or ignores your JSON schema, fine-tuning is an excellent and cheap fix.

**What it does badly:** installing specific, retrievable, updatable facts. Three independent reasons.

**Reason one — it is a terrible ratio.** Facts learned by gradient descent are distributed across billions of weights, entangled with everything else. Teaching the model one new fact reliably requires seeing it many times in many phrasings, and even then recall is probabilistic and query-phrasing-dependent. Retrieval gives you the fact deterministically, in one index write.

**Reason two — the empirical literature says so.** 📄 Ovadia et al. (2023) — *Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs* — compared unsupervised fine-tuning against RAG for injecting new knowledge and found RAG consistently stronger, with fine-tuning benefiting from augmenting the training data with paraphrases but still not closing the gap. And 📄 Gekhman et al. (2024) — *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* — found that examples introducing facts the model did not already know are learned slowly, and that as the model finally fits them, its tendency to hallucinate on *other* questions increases. Read that second result carefully: **fine-tuning on unknown facts can make the model worse at things it previously handled correctly**, because you are training it that confidently asserting unfamiliar specifics is the expected behavior.

**Reason three — the operational math, which is the argument that actually lands with a PM.** Our docs change how often? If the answer is anything under "annually," fine-tuning means retraining on every change. Update latency for RAG is the ingestion pipeline — minutes. For fine-tuning it is a training run, an eval cycle, and a deploy — days, and every one of those runs costs money and risks regressing everything else. There is no way to *delete* a fact from fine-tuned weights, which for a document that must be retracted for legal reasons is disqualifying. And you lose citations entirely: fine-tuned knowledge has no provenance, so you cannot show the user where an answer came from, which is usually the actual product requirement hiding behind "make it know our docs."

**🗣 Say this in the room:** "Fine-tuning changes behavior, retrieval changes knowledge. If the complaint is 'it answers in the wrong format or the wrong voice,' I'd fine-tune. If the complaint is 'it doesn't know our Q3 policy,' fine-tuning is the wrong tool — the literature shows it underperforms retrieval for knowledge injection and can increase hallucination on unrelated questions, I can't update or delete a fact once it's in the weights, and I lose citations, which is usually the requirement the ask is really about. The two compose fine, though: fine-tune the format, retrieve the facts."

**💰 Math:** to make the ops argument concrete for a 50k-document corpus updating ~200 docs/week. RAG incremental ingest: 200 docs × ~8k tokens = 1.6M tokens re-embedded weekly at $0.02/Mtok = **$0.032/week**, propagating in minutes. Fine-tuning: a full retrain to incorporate the delta, plus a full eval suite run, plus a staged deploy — call it $200–2,000 of compute and half an engineer-week of eval per cycle, propagating in days, with no deletion path. The RAG path is roughly four orders of magnitude cheaper *per update* and the update latency differs by three orders of magnitude. That is not a close call.
