### One enterprise customer's extraction accuracy dropped 11 points after we switched models. Everyone else improved. Debug it.

The shape of this — everyone up, one customer sharply down, after a model swap — points hard at a tokenizer-driven cause, because model quality changes tend to move all cohorts in the same direction while tokenizer changes are *content-dependent* and therefore cohort-specific.

**My debug order:**

**1. Characterize the customer's content against everyone else's.** Pull 500 of their documents and 500 from the general population, and run the fertility profiler on both, under the old tokenizer and the new one. You are looking for a fertility ratio that moved sharply for this cohort and barely moved for the rest. Common causes: their documents are in a language the new tokenizer handles worse, or they are dominated by a structured format (dense XML, fixed-width tables, base64 attachments, chemical or drug names) that the new merges don't cover.

**2. Check whether they are now hitting truncation.** If their documents inflated 1.6× and your pipeline has a fixed token budget, they may have crossed a threshold that nobody else crossed. Query your truncation log by tenant. This is the single most common concrete answer and it will be visible in one query if you built the logging from the truncation-policy answer.

**3. Check chunk-boundary shifts in retrieval.** If they use RAG and you chunk by tokens, a fertility change silently re-chunks their entire corpus at different boundaries. If you re-embedded with a new embedder at the same time, that compounds. Compare recall@k for their queries before and after against a fixed gold set.

**4. Check for term fragmentation on their specific jargon.** Take their top 200 domain terms by frequency and tokenize each under both tokenizers. If a term that was 1–2 tokens is now 6, the model's handle on it genuinely weakened — a fragmented term has its meaning distributed across pieces that also appear in unrelated contexts.

**5. Only now consider that the new model is just worse at their task.** It happens, but it is the hypothesis that explains the *least* about the cohort-specific pattern, so it goes last.

**🔍 Failure taxonomy — "quality dropped after a model swap," ranked by how often I find each:**

| Evidence | Cause | Fix |
|---|---|---|
| Truncation rate for that tenant jumped | Fertility inflation crossed a fixed budget | Raise budget or chunk; make budgets adaptive to measured fertility |
| Rendered prompt differs byte-for-byte | Chat template changed between models | Re-render, diff, pin the template with the model |
| `cache_read_input_tokens` went to zero | New tokenizer invalidated all prefixes | Expected; warm the cache, re-baseline cost |
| Domain terms fragment 3×+ more | Vocabulary coverage regression | Different base model, or vocabulary extension if you own weights |
| Failures are uniformly distributed across cohorts | Genuine model capability change | Roll back or re-tune prompts |

**🗣 Say this in the room:** "A cohort-specific regression after a model swap is a tokenizer hypothesis until proven otherwise, because tokenizer effects are content-dependent and model-quality effects usually aren't. I'd measure fertility for that cohort under both tokenizers first, then check truncation rate by tenant — that combination has explained it every time I've seen this pattern."

### We fine-tuned on chat data and the model won't stop after its answer — it keeps generating a fake user turn. What did we do wrong?

You almost certainly got the **loss mask boundary** wrong, and specifically you masked by string length instead of by token position.

The mechanism. In SFT on chat data you want the loss computed only on assistant tokens — the prompt is context, not a target. So you build a `labels` tensor identical to `input_ids` but with `-100` (PyTorch's `ignore_index`) on every prompt token. The naive implementation:

```python
prompt_ids = tok(prompt_text).input_ids            # ← the bug lives here
full_ids   = tok(prompt_text + answer_text).input_ids
labels = full_ids.copy()
labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
```

This is wrong because **BPE merges across the concatenation point.** Tokenizing `prompt` alone and tokenizing `prompt + answer` do not agree at the seam: the last token of the prompt may merge with the first character of the answer into a single token that exists in neither of the separate encodings. So `len(prompt_ids)` is off by one, or the boundary token is half prompt and half answer. The result is that you either mask one answer token (mild) or *fail to mask* the final prompt token — which is often the assistant header — teaching the model to predict the structural scaffolding as if it were content.

Now the specific symptom you described. If the boundary is off such that the terminator is masked out — if `<|eot_id|>` at the end of the assistant turn falls in the ignored region, or if your data assembly dropped it entirely — **the model is never trained to emit end-of-turn.** It has learned to produce the answer and then, having seen conversation transcripts in pretraining, to continue with the next user turn, because nothing ever taught it that generation stops. That is exactly "it won't stop and writes a fake user turn."

**The correct implementation** uses the tokenizer's offset mapping to locate the boundary in token space:

```python
enc = tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
split_char = len(prompt_text)                       # char index where the answer starts
labels = list(enc.input_ids)
for i, (start, end) in enumerate(enc.offset_mapping):
    if end <= split_char:                           # token lies entirely in the prompt
        labels[i] = -100
# tokens straddling the boundary are trained on — deliberately, and now knowingly
```

The alternative, which I prefer for chat data, is to locate the assistant-header token *sequence* in the ID list and mask everything up to and including it — matching on token IDs, never on characters.

**⚠ Trap:** masking the EOT token itself. It is emitted by the assistant, it is the single most important token in the whole sequence for a usable chat model, and it sits at the very end where off-by-one errors live. **Assert it in your data loader:** for every example, `labels[-1] == eot_id`. One line, catches the entire class.

**⚠ Trap:** trusting the framework's collator without reading it. Most SFT libraries have a "response template" mechanism that finds the assistant marker and masks before it. They match on a *string* that they tokenize independently — and if your template's marker tokenizes differently in context than in isolation, the matcher silently finds nothing and masks nothing, so you train on the full sequence including the user turns. The tell is that the model starts hallucinating user messages and asking itself questions. Print the decoded non-masked span for three random training examples and eyeball it before launching a run. It takes four minutes and I have never regretted it.

### How would you run a tokenizer migration in production — new model, different tokenizer, live traffic?

Treat it exactly like a schema migration with an index rebuild, because that is structurally what it is: the token IDs are your encoding format, and changing them invalidates every derived artifact.

**What breaks, enumerated before you plan anything:**

| Artifact | Why it breaks | Rebuild cost |
|---|---|---|
| Prefix cache (KV blocks) | Keyed on token IDs; every prefix is a miss | Automatic, but cold — plan for full-price input and slow TTFT during warmup |
| Token budgets / rate limits | Same content is now a different token count | Recalibrate all per-tenant limits |
| Cost forecasts and margins | Fertility changed | Re-measure on your corpus, per language |
| RAG chunk boundaries | Chunks were sized in tokens | Re-chunk and re-embed the whole index if boundaries move |
| Truncation thresholds | Documents that fit may no longer | Re-derive from new fertility measurements |
| Eval baselines | Perplexity is not comparable across tokenizers | Convert to bits-per-byte or re-baseline task metrics |
| Fine-tuned adapters | Trained against the old vocabulary's IDs | Not portable. Retrain. |
| Logit-bias / banned-token configs | Configured by integer ID | Every ID must be remapped by string, not by number |

That last row is the one that bites hardest and nobody checks it. If you ban token ID `48219` because it was a glitch token, that ID means something completely different in the new vocabulary — you are now banning an ordinary word at random, and the failure is a subtle quality degradation with no error. **Store logit-bias and banned-token configs as strings, resolve to IDs at load, and assert the resolution succeeded.**

**The rollout procedure:**

1. **Measure before you move.** Fertility on your production corpus, per language and per content class, old versus new. This produces every number in the table above.
2. **Shadow.** Send a mirrored copy of production traffic to the new model, compare outputs offline on your eval set, and — importantly — record the *token count* distribution, not just quality. You want the p50/p95/p99 of input tokens under both, because your p99 is where truncation lives.
3. **Recalibrate budgets and thresholds** from the shadow data, before any real traffic moves.
4. **Rebuild derived artifacts.** Re-chunk and re-embed the RAG index into a *new* index, then alias-swap. Never mutate the live index — you already know why.
5. **Canary at 1–5%**, and monitor the tokenizer-specific signals, not just latency and error rate: truncation rate, `cache_read_input_tokens` ratio, tokens-per-request by tenant, and output-length distribution.
6. **Expect the cache to be cold** and do not read the canary's terrible TTFT as a model regression. Warm deliberately if you can — replay your top-N system prompts against the new deployment before shifting traffic.
7. **Keep the old path warm for rollback**, and remember that rolling back also cold-starts a cache.

**💰 Math on the cold-cache window.** 12,000-token system prompt, 200k calls/day, $3/Mtok, normally 92% cached. Steady state costs ~$1,238/day (from the caching arithmetic earlier). During the first hour after a cutover, hit rate is near zero, so that hour runs at `12,000 × (200,000/24) × 3e-6 = $300` instead of `$52` — an extra **$248 for one hour**, trivially affordable and worth budgeting for so nobody panics at the cost graph. The thing that is *not* trivially affordable is doing this accidentally every deploy because your tool serialization is non-deterministic.

### Why can't I compare perplexity between two models with different tokenizers, and what should I report instead?

Because perplexity is defined *per token*, and "a token" means something different in each model. `PPL = exp(mean NLL per token)`. The denominator is the tokenizer's unit, so you are comparing two quantities measured in different units and calling one better.

The direction of the bias is the part to be crisp about: **a coarser tokenizer (better compression, more characters per token) is penalized on perplexity.** Each of its tokens carries more information, so each is harder to predict, so its per-token NLL is higher — even when the model is *better* at modelling the text. A model with a 32k vocab will often show a lower perplexity than a strictly superior model with a 200k vocab on the same corpus. Reporting that as evidence of quality is a straightforwardly wrong conclusion, and it is a mistake I have seen make it into a design review.

The fix is to normalize by something tokenizer-invariant. **Bits per byte** is the standard choice:

```
BPB = ( Σ_i −log₂ p(t_i) ) / N_bytes
    = ( total_NLL_in_nats / ln 2 ) / N_bytes
```

Equivalently, `BPB = log₂(PPL) × N_tokens / N_bytes`. The numerator is the total information content the model assigns to the corpus — a property of the model's distribution over *text*, not over tokens. The denominator is the UTF-8 byte count of the corpus, which is a fixed property of the data. Both models are now measured in the same units and the comparison is valid.

```python
import math
def bits_per_byte(total_nll_nats: float, corpus: str) -> float:
    return (total_nll_nats / math.log(2)) / len(corpus.encode("utf-8"))
```

Bits-per-character works too and is common in older literature, but bytes are less ambiguous across scripts (a Devanagari character is 3 bytes; "per character" quietly favours models on scripts with dense characters).

**⚠ Trap:** comparing perplexity across *context lengths*, or across *stride settings* in a sliding-window evaluation, or on corpora with different amounts of the model's training data leaked in. Tokenizer mismatch is the most common invalidity but it is not the only one. My rule: perplexity is a within-model, within-tokenizer, within-eval-protocol *training diagnostic*. It is not a cross-model quality metric, and the moment you are comparing two different models you should be reporting task metrics — accuracy, pass@k, nDCG, format-compliance rate — that mean something to the product.

**🗣 Say this in the room:** "Perplexity is per token, and tokens aren't comparable across tokenizers — a coarser vocabulary gets penalized because each of its tokens carries more information. I'd report bits per byte for a cross-model language-modelling comparison, and honestly I'd report task metrics instead, because BPB still doesn't tell me whether the model does the job."

### What's the difference between a "fast" and "slow" tokenizer, and when does offset mapping actually matter?

"Fast" means the Rust-backed `tokenizers` implementation; "slow" means the pure-Python reference implementation. The speed difference is roughly an order of magnitude and it matters for throughput on ingest pipelines — but the *capability* difference is what actually decides the choice, and it is one feature: **offset mapping**.

A fast tokenizer can tell you, for every output token, the `(start_char, end_char)` span of the original string it came from. A slow tokenizer cannot; it gives you IDs and nothing else.

```python
enc = tok("Contract dated 2024-03-11.", return_offsets_mapping=True,
          add_special_tokens=False)
for tid, (s, e) in zip(enc.input_ids, enc.offset_mapping):
    print(repr(tok.decode([tid])), (s, e))
# also available on fast tokenizers: enc.word_ids(), enc.char_to_token(i),
# enc.token_to_chars(j)
```

**Where you genuinely need it:**

1. **Citation and highlight rendering.** Your RAG answer says "per §4.2 of the agreement," and the UI needs to highlight the exact source span. The model works in tokens; the browser works in characters. Offset mapping is the bridge, and without it you fall back to string-matching the quoted text against the source — which fails on any paraphrase, any whitespace normalization, and any Unicode variation.
2. **Token classification / extractive NER.** Labels are on character spans; the model predicts per token. You need alignment to build labels and to decode predictions back to spans. This is what `word_ids()` and `is_split_into_words=True` exist for.
3. **Precise loss masking**, exactly as in the SFT question above.
4. **Redaction and PII handling.** You detected a span in the original text and need to know which tokens to mask or which tokens produced it.

**The subtlety worth knowing** (and it is a good detail to volunteer): offsets are *character* offsets into the string you passed in, but the tokenizer may have applied a normalizer first — NFC/NFKC normalization, lowercasing, accent stripping. If the normalizer changed the string length, naive offsets can point into the normalized string rather than yours. Fast tokenizers track this and expose offsets against the original, but *only if the normalizer is offset-tracking*; custom normalizers can break the invariant. Test it: take a string with a combining accent and an em-dash, tokenize it, and assert `original[s:e]` round-trips for every token.

**⚠ Trap:** offsets for special tokens. `<|begin_of_text|>` corresponds to no characters in your input; its offset is typically `(0, 0)`. Code that assumes every token has a non-empty span will produce a zero-width highlight at position 0 or, worse, index into the wrong place. Filter special tokens before doing span work, or check `e > s`.

### Name three production bugs that look like model bugs but are actually tokenizer bugs, and tell me how you'd distinguish each.

This is the thesis of everything above, so let me be concrete rather than general.

**Bug 1: "The fine-tune barely helped."** Eval accuracy is strong, production accuracy is mediocre, no errors anywhere, and every hypothesis about data quality comes back clean. **Cause:** the chat template used during training differs from the one used at serving — a newline, an EOT variant, a system-message placement, or a double BOS. **Distinguishing test:** render one identical `messages` list through both code paths and diff the strings, then diff the token IDs. If they differ at all, that is your bug and you can stop investigating. **Why it masquerades:** every model output is still fluent and plausible, so there is no signal that anything structural is wrong; the degradation is a uniform few points across everything.

**Bug 2: "The model can't count / can't do arithmetic / can't reverse strings."** **Cause:** the input representation discarded the character structure the task requires, or the digit grouping misaligns place value. **Distinguishing test:** perform the identical task with the characters space-separated. If accuracy jumps from 40% to 95% with no other change, the model was never bad at the task — it could not see the input. Same test with numbers: comma-separate a multi-digit addition and re-measure. **Why it masquerades:** it presents as a reasoning failure, and reasoning failures are a plausible thing for an LLM to have, so people accept the explanation and start prompt-engineering. The fix is a tool call, not a prompt.

**Bug 3: "Costs tripled and latency got worse for our international users."** **Cause:** fertility inflation on non-Latin scripts, compounding into truncation and context starvation. **Distinguishing test:** measure tokens-per-character by language on production traffic and plot cost-per-request against language. If the ratio tracks fertility rather than request volume, it is the tokenizer. Check the truncation log by locale at the same time. **Why it masquerades:** it looks like a traffic-mix or capacity problem, and the quality half of it (shorter effective context → worse retrieval → worse answers) looks like a model quality problem for those users specifically.

I would offer a fourth if pressed, because it is the most operationally expensive: **"Our prompt cache hit rate is 4% and nobody knows why."** Cause is almost always a non-deterministic serialization in the prefix — unsorted JSON keys, a set, a timestamp, a UUID. Distinguishing test: capture two consecutive requests' full rendered prompts and diff them; the first differing byte is the culprit. It is a tokenizer-adjacent bug in that the cache is keyed on token IDs and any byte change is a token change.

**🗣 Say this in the room:** "The common thread is that all four fail *silently*. There's no exception, no 4xx, no anomalous log line — the model keeps producing fluent, well-formed output that's just worse or more expensive. That's why I put explicit assertions at the tokenizer boundary: a golden-fixture template parity test in CI, a BOS-count assertion on sampled traffic, a truncation-rate alarm, and a cache-hit-rate assertion. Four cheap checks that convert four silent failures into loud ones."

### I'm going to give you 45 minutes and a blank editor. Write me BPE — training and encoding — and then I'll ask you four things about it.

**🏋 Drill — Time: 45 minutes. No autocomplete, no reference material, no LLM assistance.** This is the exact format of an Anthropic or DeepMind live coding round on this topic.

**Part A (20 min) — training.** Implement `train_bpe(word_freqs: dict[str, int], num_merges: int) -> list[tuple[bytes, bytes]]` from scratch. Byte-level alphabet. Deterministic tie-breaking. Return the ordered merge list.

**Part B (10 min) — encoding.** Implement `encode(text, merges, vocab) -> list[int]`: a simple whitespace-and-punctuation pre-tokenizer, then merge-rank replay per chunk, then vocab lookup. Verify that `decode(encode(s)) == s` for a string containing an emoji, a Devanagari character, a tab, and a doubled backslash.

**Part C (15 min) — the follow-ups, spoken aloud while you work.** These are what the interviewer actually grades:

1. *What is the time complexity of your encoder, and what input makes it worst-case?* (Answer: O(n²) in pre-token length under the naive rescan; a long unbroken alphanumeric run — base64, a hex hash, a minified bundle.)
2. *You trained this twice on the same data and got different merge lists. Why?* (Tie-breaking on equal counts, resolved by dict insertion order, which depends on shard order.)
3. *Your vocabulary has 50,000 entries and you want 100,000. What changes in the model, and what does it cost?* (Two `V × d` matrices grow; compute the parameter delta for a stated `d`; discuss the sequence-length benefit against the bandwidth and softmax cost, and note the regime flip for small models.)
4. *Why does `" hello"` differ from `"hello"` and where does that bite in production?* (Leading-space merges; trailing-whitespace prompts, stop sequences, prefill boundaries.)

**Pass criteria:** all three parts complete and correct in 45 minutes; the round-trip test passes on the adversarial string; you answered all four follow-ups without needing to look anything up; and you volunteered the tie-breaking determinism problem *before* being asked. That last one is the difference between a "solid" and a "strong hire" on this question.

### Suppose we hand you an open-weight checkpoint and say "we're shipping this Friday." What would you check about its tokenizer, and how would you present it?

**🏋 Drill — Time: 90 minutes. Use any open-weight model you can load locally.** Produce a single script that emits a one-page report. This is a genuinely good portfolio artifact — it is small, it is legible, and it demonstrates production judgment rather than tutorial-following.

**The report must contain:**

1. **Identity block.** Tokenizer class, vocab size, tied/untied embeddings (verified by data pointer, not by config), BOS/EOS/PAD/UNK IDs, whether the chat template emits BOS, whether the model is fast or slow.
2. **Round-trip integrity.** `decode(encode(s)) == s` over a 10,000-document sample plus an adversarial set: emoji with skin-tone modifiers, combining diacritics, RTL text, a zero-width joiner, a 4-space and a tab indent, `\\n` inside a JSON string, a 5,000-character base64 blob. Report any failures with the offending bytes.
3. **Fertility table.** Tokens-per-character overall and broken out by content class (prose / code / tables / non-English / domain jargon), against your own corpus. Include the p50/p95/p99 of tokens-per-document, because p99 is where truncation lives.
4. **Undertrained-row scan.** The embedding-norm tail plus the unembedding-centrality probe from the glitch-token audit, with behavioural repeat-verbatim verification of the top 10 non-reserved candidates. Output a ban-list of token *strings* (not IDs).
5. **Template parity.** Render 20 golden `messages` fixtures with `tokenize=False`, hash the strings and the ID lists, and store them. This file becomes the CI fixture that catches the training/serving mismatch forever after.
6. **Special-token injection canary.** For every special token in the vocabulary, put its literal string inside a user message, render, tokenize, and assert it never appears as a control ID in the output.
7. **Cost projection.** Tokens per document × your monthly volume × the model's per-token price, with the arithmetic shown, for every candidate model you are comparing.

**Pass criteria:** the script runs end to end in under five minutes on 10k documents; every claim in the report is a computed number with the computation visible; and you can point at one line of the report and say "this is why we are choosing model X over model Y." If your report does not change a decision, it is a diagnostic, not an audit.

### You're the engineer who owns tokenization for a new in-house model. Walk me through the decisions you make, in order, and how you'd know each was right.

I would sequence it so that the irreversible decisions are made last and every one of them is backed by a measurement on our own corpus.

**Decision 1 — algorithm and library.** Byte-level BPE via the `tokenizers` library, unless the corpus is heavily weighted toward scripts without whitespace segmentation, in which case SentencePiece with byte fallback enabled. **How I'd know:** run both on a 10GB sample, compare fertility per language and per content class. The delta is usually small; if it is small, take BPE for the simpler deterministic encoder and the mature tooling.

**Decision 2 — training corpus for the tokenizer.** This is the decision most teams get wrong and it is nearly free to get right: **train the tokenizer on the same filtered, deduplicated corpus the model will be pretrained on, sampled to the same language and domain mix.** Every glitch token in history came from violating this. **How I'd know:** after training, run the undertrained-row scan; a clean pipeline produces essentially no non-reserved candidates.

**Decision 3 — pre-tokenization regex.** Digit chunks capped at 3 (or single-digit if arithmetic is a first-class product requirement and we can afford the token cost), leading-space attachment for compression, explicit whitespace-run tokens sized to our code indentation conventions, and a hard cap on pre-token length so no input can drive the encoder quadratic. **How I'd know:** fertility on code specifically, plus a synthetic arithmetic eval at 2/4/6/8 digits before and after a grouping change.

**Decision 4 — vocabulary size.** Sweep it. Measure fertility at 32k / 64k / 128k / 200k on our corpus, compute the parameter cost at our `d_model`, and pick the knee. **The decision rule:** long-context, multilingual, or code-heavy workloads justify a large vocabulary; small models and short-sequence workloads do not, because `V × d` becomes a punishing fraction of a small model and the compression buys nothing when sequences are 200 tokens. **How I'd know:** a table of (vocab size, tokens/doc, embedding params, % of total params, projected monthly cost) — the knee is visible.

**Decision 5 — tied or untied.** Tied below roughly 3B parameters or whenever `V × d` exceeds ~15% of the model; untied above. **How I'd know:** compute the fraction; if it is 25% of your model, the decision makes itself.

**Decision 6 — special tokens and reserved slots.** Reserve 200–256 unused slots up front. This is the cheapest insurance in the entire design: adding a token later means resizing embeddings on a trained model, and having pre-reserved slots means a future control token, a future tool-call delimiter, or a future modality marker costs nothing. Every serious model family does this and the reason is exactly this.

**Decision 7 — chat template.** Design it once, ship it in `tokenizer_config.json`, version it with the weights, and write the golden-fixture parity test on day one rather than after the first incident.

**Decision 8 — freeze and document.** After the model starts pretraining, the tokenizer is immutable. Publish the fertility table, the vocab size, the tie state, the special-token map, and the reserved-slot count in the model card, because every downstream team will need those numbers and if you don't publish them they will guess.

**🗣 Say this in the room:** "The through-line is that tokenization decisions are made once and are effectively permanent — you cannot change the tokenizer after pretraining starts without throwing away the run. So every decision gets a measurement on our own corpus first, the irreversible ones go last, and I reserve spare special-token slots because that's the one future-proofing that costs literally nothing. The failure mode I'm designing against isn't a bad choice, it's an *unmeasured* choice that nobody can revisit two months later."
