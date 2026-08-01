### Flip it around — what is fine-tuning genuinely, uniquely good at? Give me the list and the mechanism for each.

I want to be equally clear on this side, because a candidate who can only say "don't fine-tune" is just as unhireable as one who reaches for it first. Fine-tuning is the correct and sometimes the *only* answer for five things.

**1. Tacit form and style the prompt cannot express.** There is a class of requirement that a domain expert can recognize instantly and cannot write down. "Write it the way our senior associates write it." "This is how a Bloomberg-terminal user expects a summary to read." "Our diffs look like this." You can approximate it with twenty few-shot examples and 3,000 tokens of style guide; you cannot *reliably* hit it, and you pay those 3,000 tokens on every call forever. The mechanism: gradient descent over thousands of examples fits a conditional distribution over surface form far more precisely than in-context learning over twenty, because in-context learning is a soft, attention-mediated approximation of the same fit and it is capacity-limited by what fits in the window.

**2. A long-tail output format with hard structure.** Not JSON — constrained decoding handles that. I mean domain formats with thousands of interacting conventions: a specific legal citation style, a particular patch/diff dialect, a DSL your company invented, structured clinical notes. These are learnable from data and unwritable as rules.

**3. Latency and cost, by deleting the prompt.** This is the underrated one and it is really distillation wearing a fine-tune's clothes. If your system prompt is 8,000 tokens of instructions and few-shots, baking it into weights removes 8,000 tokens of prefill from every request. That is both a token bill and a TTFT reduction, and on a self-hosted small model it is often a 3–5× total cost move.

**4. Decision boundaries with thousands of labels behind them.** Classification, routing, risk scoring, triage — tasks where you have historical labeled outcomes and the boundary is genuinely learned from data rather than statable as a policy. If you have 50,000 historical support tickets with their correct routing, a fine-tuned small model will beat a prompted frontier model on that task *and* cost 50× less. This is the strongest business case for fine-tuning that exists and it is nothing like "teach it our docs."

**5. Behaviour under a distribution the base model handles awkwardly.** Non-English domain jargon, a low-resource language, an unusual modality mix, a tool-calling dialect with a hundred internal tools. Here you are shifting a prior, not adding facts.

Notice what unifies all five: **form, format, latency, cost, and tacit style.** Not one of them is "the model will know new facts."

**📐 Numbers you must know:** the rough data floor per objective. Style/format transfer: 500–2,000 well-curated examples is usually enough, and LIMA is the reason we believe that. A learned decision boundary: 1,000 examples per class as a working minimum, and more if classes are imbalanced. Knowledge injection: there is no number that works, which is the point of the next question. And note the shape of the LIMA result — its power came from *curation*, not volume; 1,000 excellent examples beat 50,000 scraped ones, and the most common failure of a first fine-tune is training on a large dirty set.

### Explain to me, at the level of gradients, why fine-tuning doesn't reliably inject facts.

Start with where facts live. During pretraining, a fact like "the Model X-7 ships with a 4-year warranty" is not stored in one place; it is a distributed pattern across many MLP neurons in many layers, built up over *thousands of exposures in many surface forms* — the fact appeared as prose, as a table row, in a Q&A, in a forum reply, phrased forward and backward. The redundancy is what makes retrieval of that fact robust to how you ask.

Now look at what SFT does with your 5,000 QA pairs. Each fact appears once, in one phrasing, for one epoch or three. The cross-entropy loss on those tokens is minimized by the cheapest available adjustment, and the cheapest available adjustment is almost never "install a new distributed fact representation." It is **"learn the mapping from question-shape to answer-shape."** The model learns that after a question about warranties, an assertive sentence containing a number and the word "warranty" is high-probability. It learns the *template*, because the template is the highest-frequency, lowest-loss regularity in your dataset.

That is why the failure mode is not ignorance but **confident hallucination**. You have trained the model to produce answer-shaped strings in a domain where it does not reliably hold the answers. You have specifically trained *away* the hedging behaviour, because none of your 5,000 gold examples say "I don't have that information" — every single one confidently answers. You optimized the model to always answer. It complies. On the 30% of questions where the fact didn't take, it invents one in exactly the register you trained.

There is direct evidence for this. Gekhman et al. (2024) studied fine-tuning on examples containing knowledge the model did not already hold, and found two things that should change how you plan a project: examples carrying *new* knowledge are fitted much more slowly than examples the model already knows, and as the model does eventually fit them, its hallucination rate on questions it previously answered correctly **rises**. Slow to learn, and the learning is actively harmful to what was already there. That is close to the worst possible cost/benefit shape.

There is also a capacity argument. Work on knowledge capacity scaling reports that transformer language models store on the order of ~2 bits of factual knowledge per parameter under saturated training. Fine-tuning does not add parameters, so it cannot add capacity — anything you install displaces something. That is the same accounting that makes catastrophic forgetting inevitable rather than a bug you can tune away.

**📄 Paper:** Gekhman et al. (2024), *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* — showed that fine-tuning examples containing unfamiliar knowledge are learned slowly and, as they are learned, increase hallucination on previously-known facts. It replaced the folk belief that SFT is a general knowledge-injection mechanism.

This is the single most important sentence in this section and it is worth carrying into the retrieval material verbatim, because that is where the reflex actually fires: the meeting where someone proposes fine-tuning on the knowledge base is a retrieval meeting, not a training meeting.

**⚠ Trap:** "but the model *did* answer my new question correctly after fine-tuning." Check whether it answers a *paraphrase* of that question, and whether it answers the *reverse* direction ("which product has a 4-year warranty?"). Single-direction, single-phrasing recall is memorization of a string, not acquisition of a fact, and it does not survive real user phrasing. Building a paraphrase-held-out slice into your eval is how you catch this before production does.

**🗣 Say this in the room:** "Facts are stored redundantly across thousands of pretraining exposures in many surface forms. A fine-tune gives each fact one exposure, so the cheapest loss reduction available is learning the question-to-answer *template*, not the content. The measurable result is a model that has been trained never to say 'I don't know' in a domain where it frequently doesn't. That's a hallucination generator."

### But my teammate fine-tuned on 5,000 QA pairs from our internal wiki and the eval score went up 11 points. Is he wrong?

Probably, and I can tell you the three specific ways an 11-point gain here is usually an artifact — and the one way it might be real.

**Artifact 1: the eval and the training set share provenance.** If both were generated from the same wiki by the same script, they share phrasing, section ordering, and question templates. The model learned the template and your eval is measuring the template. The test: build a held-out slice where the questions were written by humans who never saw the training data, or at minimum paraphrased by a different model with different instructions. I have watched an 11-point gain become a 1-point gain under that treatment more than once.

**Artifact 2: he measured form and called it knowledge.** A large fraction of automatic eval gains after SFT are the model finally answering in the expected format — the right length, the right leading sentence, no preamble. If your grader is an exact-match or an LLM-judge with a rubric that rewards directness, format alone can move you ten points. Decompose the grader: score *groundedness* and *correctness* separately from *format compliance* and re-run.

**Artifact 3: the baseline was unfairly weak.** Was the prompted baseline given the same 5,000 documents via retrieval? Nine times out of ten the comparison is `fine-tuned model with no retrieval` vs `base model with no retrieval`, which is not the decision anyone faces. The real comparison is `fine-tuned, no retrieval` vs `base model + RAG`, and RAG usually wins on knowledge tasks. Ovadia et al. (2023) ran essentially this comparison — unsupervised fine-tuning for knowledge injection versus retrieval — and found retrieval the stronger mechanism, with fine-tuning improving mostly when augmented with many paraphrases of each fact.

**The way it might be real:** if the wiki content was *already* in pretraining (public docs, or a domain the model knows) and the fine-tune is teaching the model to *surface* what it already latently holds, in your format, with your terminology. That is not knowledge injection — that is elicitation, and it is legitimate. You can distinguish the two by testing on facts that postdate the model's cutoff or are genuinely private. If gains hold on private post-cutoff facts, something surprising happened and I'd want to see the paraphrase eval before believing it.

**⚠ Trap:** the most dangerous version of this is when the fine-tune *does* raise the aggregate score while raising the confident-wrong rate. Aggregate accuracy hides the shift from "abstained" to "wrong." Always report a three-way split — correct / incorrect / abstained — and treat a fall in abstention as a red flag even when accuracy rises. A system that goes 60/10/30 → 66/34/0 gained six points of accuracy and became far more dangerous.

### Is there ever a case where you'd put new knowledge into weights? What would that take?

Yes, and it is worth being able to describe precisely so you don't sound reflexively anti-training. Knowledge does go into weights — that is what pretraining is. The question is whether *your* knowledge, at *your* scale, justifies the mechanism that actually works, which is **continued pretraining with heavy augmentation**, not SFT on QA pairs.

What it actually takes:

- **Raw corpus, not QA pairs.** You train on the documents themselves with a language-modeling objective, the same way the knowledge got there originally.
- **Massive paraphrase augmentation.** Each fact rewritten many ways — different phrasings, different orderings, forward and reverse, embedded in different document types. This is the single biggest lever, and it is why the naive version fails: one exposure per fact is not how facts are learned.
- **A much lower learning rate than SFT** and a **replay mix** of general pretraining data (commonly 5–30%) to keep the model from forgetting everything else.
- **A capability-regression suite** run before and after, because you are moving weights that hold everything else too.
- **An SFT stage afterwards**, because continued pretraining on raw documents degrades instruction-following. You cannot ship the CPT checkpoint directly.

When is this worth it? Three conditions together: the corpus is **large** (tens of millions of tokens minimum — below that, retrieval is strictly better), the knowledge is **stable** (if it changes monthly you are signing up for monthly training runs), and the domain is **linguistically distant** from pretraining data so that the model needs new vocabulary and new idiom, not just new facts. Genomics notation, a proprietary hardware ISA, a low-resource language, decades of internal engineering documents in a house dialect. Bloomberg's finance-domain model and various code-domain CPT efforts are the archetype.

If any of those three fails — and for a normal product company all three usually fail — retrieval wins, and it is not close.

**💰 Math:** continued pretraining a 7B on 20B tokens: FLOPs ≈ 6·N·D = 6 × 7e9 × 2e10 = **8.4e20 FLOPs**. An H100 at ~400 TFLOP/s bf16 dense, at a realistic 40% MFU, delivers 1.6e14 FLOP/s. 8.4e20 / 1.6e14 = 5.25e6 GPU-seconds = **~1,460 H100-hours**. At an illustrative $3/H100-hour that is **~$4,400 of compute** — which sounds cheap, and is exactly why teams underestimate. The compute is never the cost. The corpus construction, dedup, augmentation, the SFT stage after, the regression suite, and the four failed runs are the cost, and they are engineer-months. **📅 Volatile:** GPU spot pricing moves constantly; verify.

### What are your named preconditions for approving a fine-tune? Be specific enough that I could put them in a design-review checklist.

Here is the checklist I actually use. All seven must be true; any "no" sends it back down the ladder.

1. **A stable, versioned eval exists and has been stable for at least two weeks.** Same inputs, same grader, same version pin, with variance measured by re-running it three times. If your eval moves ±4 points run-to-run, a 5-point fine-tune gain is unmeasurable. This is precondition zero — everything else is meaningless without it.
2. **A measured plateau on the lower rungs.** Documented: prompt variants tried, retrieval metrics at ceiling, tools in place, structured output enforced, routing evaluated. With numbers, in a doc, not "we tried stuff."
3. **≥1,000 high-quality labeled examples for the target behaviour**, with a real held-out test split that was *not* generated by the same process as training. For a decision-boundary task I want more — closer to 1,000 per class.
4. **The failure class is form, style, format, latency, or cost — not facts.** Stated explicitly, with the failure taxonomy table backing it.
5. **A capability-regression suite** covering the general behaviours the model must not lose: instruction-following, refusal calibration, tool-calling, format compliance on adjacent tasks, and basic reasoning. Fine-tuning is a global edit; you need a tripwire.
6. **A named owner and a re-training budget.** Someone owns this checkpoint through the next two model upgrades, and there is a line item for re-running the pipeline. If nobody will own it in six months, it will rot into a liability.
7. **A serving plan.** Where does this run, at what cost, with what fallback if the checkpoint misbehaves? A fine-tuned model with no canary path and no fallback to the prompted baseline is an outage waiting for traffic.

**🗣 Say this in the room:** "My gate is: a stable eval, a documented plateau on prompting and retrieval, at least a thousand clean labeled examples with a genuinely held-out split, a failure class that is form rather than facts, a regression suite, an owner, and a serving fallback. If I can't check all seven I don't approve the fine-tune — not because training is scary, but because without them I can't tell afterwards whether it worked."

**⚠ Trap:** precondition 1 is the one teams skip and it is the one that invalidates the project. A fine-tune evaluated against an eval you built *after* seeing the fine-tune's failures is unfalsifiable. Build the eval before the training set — always, no exceptions.

### "A measured plateau" — how do you actually prove the prompt is maxed out rather than just claiming it?

You prove it with an **ablation ladder** — a table where each row is a rung and the columns are your eval metrics. The claim "we plateaued" means "the last three rows added less than the noise floor."

The table I want to see in a design review looks like this, on the same 300-item eval set, same grader version:

```
config                                       acc   grounded  fmt-ok  p95 lat  $/1k req
base prompt, no retrieval                    41%     —        72%     1.9s     $3.10
+ rewritten prompt & 12 mined few-shots      58%     —        96%     2.4s     $4.80
+ retrieval, k=5, dense only                 69%     0.71     96%     2.7s     $5.90
+ hybrid BM25+dense, k=20 → rerank to 5      78%     0.86     97%     2.8s     $6.15
+ query rewriting                            81%     0.89     97%     3.0s     $6.40
+ tools (pricing, inventory, date)           86%     0.91     98%     3.1s     $6.80
+ constrained output                         86%     0.91    100%     3.1s     $6.80
+ ORACLE context (upper bound)               88%     0.97    100%      —         —
```

Two things make this table decisive. First, the last three real rows moved 5, 0, and 0 points — that is the plateau, and it is visible rather than asserted. Second, the **oracle row is the ceiling**: at 86% real versus 88% oracle, only 2 points remain accessible through better retrieval. Everything above 88% must come from somewhere else, and *that* is the honest opening for a fine-tune discussion.

You also need the **noise floor** on the same page: re-run the best config three times with temperature at your production setting and report the spread. If it is ±2.5 points, then the 86→88 gap is not reliably measurable and you should say so out loud rather than chase it.

Finally, run the error taxonomy on the residual failures at the best config. If the remaining 14% is 9% "ambiguous label," 3% "tone/register," and 2% "genuinely needs reasoning we don't have," then your fine-tune's realistic upside is the 3% tone bucket — and you can now say "a fine-tune buys us at most three points" instead of "a fine-tune will improve things." That sentence is the difference between a senior and a mid-level answer.

**🏋 Drill:** given a working RAG prototype and a 200-item eval, produce the full ablation ladder plus a noise-floor row in four hours. Pass criterion: every row is reproducible from a config diff, the oracle row is present, and you can state the maximum remaining headroom as a number.

### What is RAG genuinely good at, and be precise about where its ceiling is.

RAG's four genuine strengths, and each is a strength *fine-tuning structurally cannot have*:

**Changing knowledge.** The corpus updates on write. A doc edited at 10:04 is answerable at 10:05 after an index update — no training run, no eval cycle, no checkpoint. Anything that changes on a timescale shorter than your training cadence must be retrieved, full stop.

**Citations and auditability.** The system can say *which* chunk it used, and a human can click it. In regulated domains — legal, medical, financial — this is not a nice-to-have, it is the reason the product is deployable at all. A fine-tuned model's answer has no provenance; you cannot audit a weight. Harvey-class and Glean-class products are built on this property.

**Per-user permissions.** This is the one candidates forget and it is the sharpest. Retrieval can filter by ACL *at query time*: user U's search is scoped to the documents U may read. A fine-tuned model that memorized the corpus has no such filter — the knowledge is smeared across weights and will surface for anyone who asks the right question. **You cannot un-train a document from a checkpoint for one user.** In an enterprise deployment this alone disqualifies weight-based knowledge for anything multi-tenant or permissioned, before you even get to the quality argument.

**Deletability.** GDPR erasure, a customer offboarding, a document retracted by legal. Delete the row, reindex the shard, done. The weights equivalent is retraining.

Now the ceiling, honestly stated:

- **Multi-hop and aggregate questions.** "Which of our 400 contracts have both an auto-renew clause and a liability cap under $1M?" is not a top-k similarity problem. Retrieval returns the most similar chunks, not the *complete set satisfying a predicate*. Fixes are agentic (iterative retrieval, query decomposition) or structural (extract to a database and let the model write SQL — a rung-3 tool answer), not "better embeddings."
- **Tacit style and format.** Retrieval puts facts in the window. It does nothing for "write it like our associates write it."
- **Token cost and latency.** Every request carries 2,000–8,000 tokens of retrieved context. That is a per-request tax forever, and it is precisely the tax a fine-tune or a distilled model removes.
- **Distraction and the middle of the window.** More retrieved context is not monotonically better; irrelevant chunks measurably degrade answers, and information placed in the middle of a long context is recovered less reliably than at the ends — the "lost in the middle" effect (Liu et al., 2023). This is why reranking to fewer, better chunks beats stuffing k=50.
- **Retrieval recall is a hard ceiling.** If the gold chunk is not in the candidate set, no amount of generation quality saves you. Your end-to-end accuracy is bounded above by recall@k.

**📄 Paper:** Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — introduced the retriever+generator architecture trained jointly, establishing the pattern of conditioning generation on retrieved passages instead of relying on parametric memory alone.

### Give me the honest converse — when does RAG fail in a way that only training fixes?

Four cases, and I want to be able to name them because refusing to ever train is its own failure mode.

**1. The requirement is a style or register, not a fact.** No document contains "how our associates write." You can approximate with few-shots at a permanent token cost, but if the style is subtle and the volume is high, weights are the right home for it. This is the Harvey-shaped case: retrieve the statute, but the *drafting voice* is trained.

**2. The context cost is the product's economics.** If your RAG prompt is 8,000 tokens and you serve 5M requests a month, you are paying for 40 billion input tokens monthly. At $3/Mtok that is $120,000/month in context alone. Even with 90% prefix caching on the static half, you are carrying a structural cost that a distilled or fine-tuned small model deletes. At that volume, "fine-tune to remove the prompt" is a cost decision with a spreadsheet behind it, not a capability decision.

**3. The task is a learned decision boundary over history.** Routing 50,000 tickets, scoring risk, classifying transactions. There is no document to retrieve; the answer lives in the empirical distribution of past labeled outcomes. Retrieval of similar past examples (kNN-style few-shot) is a real competitor here and you should try it first — but with enough labels, a fine-tuned small model wins on both accuracy and cost.

**4. A tool-calling dialect with a large internal API surface.** A hundred internal tools with overlapping semantics is more than a prompt can disambiguate reliably, and the tool schemas alone can consume 10,000 tokens. Fine-tuning on tool-call trajectories teaches selection and argument-filling in a way that scales past what fits in the window.

Notice all four are form, cost, or learned-boundary — the same list as before. There is no fifth entry that reads "the model needs to know our facts."

**🗣 Say this in the room:** "The clean split I use: **facts go in context, behaviour goes in weights.** RAG owns anything that changes, needs a citation, or needs a permission check. Training owns tacit style, a learned decision boundary, and deleting a prompt you're paying for on every request. Most production systems need both, and the interesting design question is the seam between them."

### Design me a system that fine-tunes for structure and retrieves for facts. Make it concrete.

Take an enterprise research assistant of the Glean/Harvey shape: employees ask questions over an internal corpus with per-user ACLs, and the answers must be cited, in house format, and defensible.

**The seam I'd draw.** Weights own: output structure (a specific memo format — question restated, short answer, supporting analysis, citation block, confidence and caveats), the house register, tool-call selection across ~30 internal tools, and the abstention behaviour ("the retrieved documents do not support an answer; here is what would"). Context owns: every fact, every citation, every permission decision, every number.

**Serving path.**
1. Request arrives with the user's identity. Query rewriting (cheap model, or a rule) expands acronyms and resolves conversational references.
2. Hybrid retrieval — BM25 + dense — over an index **partitioned or filtered by ACL at query time**, k=50. ACL filtering happens in the retrieval layer where it is enforceable and auditable, never in the prompt.
3. Cross-encoder rerank 50 → 6. Assemble context as tagged blocks with `doc_id`, `title`, `last_modified`, `url`.
4. Generation by the fine-tuned model. Its training data taught it the memo format, the citation syntax `[doc_id §n]`, and the abstention rule — so the system prompt is now ~400 tokens instead of ~6,000.
5. A post-generation **citation verifier**: every `[doc_id]` in the output must appear in the retrieved set, and a cheap entailment check confirms each claim sentence is supported by its cited chunk. Failures escalate to the frontier model with the same context.

**Training data construction.** This is the part interviewers dig into. You need `(retrieved_context, question) → gold_memo` triples where the context is in *exactly the serving format*. Generate the memos with a frontier teacher using a 6,000-token style prompt, then filter: drop any memo whose citations don't verify, drop any where a second model judges the answer unsupported, and have humans spot-check 200. That is rejection-sampled distillation and it gets you 3,000–8,000 clean examples for a few hundred dollars of teacher tokens plus a week of pipeline work.

**⚠ Trap — the biggest one in this design:** if you train with a context format that differs at serving time — different tags, different chunk ordering, a different number of chunks, a citation field the serving path doesn't populate — the model's behaviour degrades in ways that look like random quality loss and are almost impossible to debug. I enforce that the *same code path* assembles context for training-data generation and for serving. Not "the same format"; the same function. This is the single most common cause of "the fine-tune was worse in production than in eval."

**⚠ Trap 2:** never let the fine-tuned model's parametric memory answer without retrieval. Train with a meaningful fraction (I use 15–25%) of examples where the retrieved context does *not* contain the answer and the gold output is an abstention. Without those, you have trained a model that always produces a confident memo, and it will produce one when retrieval returns garbage.

**💰 Math:** the prompt deletion alone. 6,000 → 400 tokens of system prompt at $3/Mtok input, 1M requests/month: (6,000 − 400) × 1e6 × $3/1e6 = **$16,800/month** if uncached. With a 90% cache read on the static preamble it is $1,680/month — so *if you already had prefix caching working, the prompt-deletion argument is worth 10× less than teams claim*. Check whether your preamble is actually cache-hitting before you use this as your justification; if your cache hit rate is 30% because a timestamp invalidates it, fix that first (rung 1) and the fine-tune's cost case may evaporate.

### Itemize the total cost of a fine-tune. I want the line items that don't show up on the GPU bill.

The GPU bill is almost never the number that matters, and a candidate who quotes only the training compute has not shipped one. Here is the full itemization for a realistic first fine-tune of a small open model, with the arithmetic.

**Direct compute — the small number.** LoRA on an 8B, 5,000 examples averaging 1,200 tokens, 3 epochs = 18M tokens. Training FLOPs ≈ 6·N·D = 6 × 8e9 × 1.8e7 = 8.6e17. One H100 at 400 TFLOP/s × 40% MFU = 1.6e14 FLOP/s → 5,400 seconds ≈ **1.5 GPU-hours**, call it $5. You will do 20–40 runs across hyperparameter sweeps and failures: **$100–$200.** This line item is a rounding error and it is the only one people quote.

**Data.** Either human-labeled — 5,000 × $2/label = **$10,000** plus 2–3 weeks of vendor onboarding, guideline writing, pilot rounds and adjudication — or teacher-generated: 5,000 completions × ~$0.02 = $100, ×4 for rejection sampling = **$400**, plus human spot-checking of 200 samples. Distillation is 25× cheaper on data and that is a large part of why it sits lower on the ladder.

**Eval infrastructure.** If it doesn't exist: **2–3 engineer-weeks.** At a fully-loaded senior cost of ~$300k/year ≈ $5,800/week, that is **$12,000–$17,000**. You need it regardless, so charge it to the program not the fine-tune — but you cannot start without it.

**Engineering time on the fine-tune itself.** Data pipeline, chat-template correctness, masking, packing, the sweep, the regression suite, the serving integration, the canary. **4–8 engineer-weeks = $23,000–$46,000.** This is the real cost and it dwarfs everything above.

**Serving.** A dedicated 8B on two L40S-class GPUs for redundancy at ~$1.50/GPU-hour: 2 × $1.50 × 730 = **$2,190/month standing**, whether or not traffic arrives. Plus you now own a second serving lane with its own scaling, monitoring, and on-call.

**Ongoing maintenance — the line nobody budgets.** Re-tune on every base-model upgrade (2–4× per year), each costing data regeneration + a sweep + a full eval cycle ≈ **1–2 engineer-weeks = $6,000–$12,000 per upgrade**, so **$12,000–$48,000/year**. Plus eval drift maintenance, plus the checkpoint's share of on-call.

**Total year one:** roughly **$50,000–$110,000** of loaded cost, of which under 0.5% is GPUs. That number is what you put next to the API bill you're trying to reduce, and it is why the break-even volume is higher than people expect.

**⚠ Trap:** quoting "$200 of compute" as the cost of a fine-tune. It is technically true and it is the tell that someone has read about fine-tuning rather than shipped one. The correct senior framing is that fine-tuning is an *organizational commitment* priced in engineer-quarters, not a GPU purchase.
