### Implement beam search with length normalization. Whiteboard level, no library.

Beam search is breadth-limited best-first search over the sequence lattice, scored in log space. The mental model that keeps the implementation honest: you are maintaining exactly `k` live hypotheses, and at every step you expand all of them, score all `k × V` extensions, and keep the best `k`. Everything else — length normalization, finished-beam handling, early stopping — is bookkeeping bolted onto that loop.

```python
@torch.no_grad()
def beam_search(model, input_ids, k=4, max_new=64, eos_id=2, alpha=0.7):
    # beams: list of (token_ids: list[int], logprob: float, past)
    beams = [(list(input_ids), 0.0, None)]
    finished = []

    for step in range(max_new):
        cands = []
        for toks, lp, past in beams:
            inp = torch.tensor([[toks[-1]]]) if past is not None \
                  else torch.tensor([toks])
            out = model(input_ids=inp, past_key_values=past, use_cache=True)
            logprobs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            top = torch.topk(logprobs, k)
            for score, tid in zip(top.values.tolist(), top.indices.tolist()):
                cands.append((toks + [tid], lp + score, out.past_key_values))

        cands.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for toks, lp, past in cands:
            if toks[-1] == eos_id:
                n = len(toks) - len(input_ids)
                finished.append((toks, lp / (n ** alpha)))       # <-- normalize
            else:
                beams.append((toks, lp, past))
            if len(beams) == k:
                break
        if not beams:
            break

    if not finished:                       # nothing hit EOS within budget
        finished = [(t, lp / (max(1, len(t) - len(input_ids)) ** alpha))
                    for t, lp, _ in beams]
    return max(finished, key=lambda f: f[1])[0]
```

Three points carry the answer. **Sum log-probs, never multiply probabilities** — 200 tokens at p≈0.5 each gives `2^-200 ≈ 6e-61`, which underflows fp32's ~1e-38 floor and silently becomes zero, at which point all beams tie at 0.0 and you are doing random search. **Length normalization is mandatory**, because log-probs are negative and every additional token makes the score worse, so an unnormalized beam search has a systematic bias toward the shortest sequence — it will emit EOS almost immediately. Dividing by `n^α` with `α ≈ 0.6–0.7` corrects it; the GNMT paper (Wu et al., 2016) used the smoother `lp(Y) = (5+|Y|)^α / 6^α`, which behaves better at very short lengths. **Finished beams leave the active set**, and you keep searching until you have `k` finished hypotheses or the best active beam's optimistic score can no longer beat the best finished one.

**💰 Math:** beam search costs `k×` the decode FLOPs and `k×` the KV cache. On a 70B model with GQA (8 KV heads, 128 head dim, 80 layers) at 4k context, one sequence's cache is `2 × 80 × 8 × 128 × 4096 × 2 bytes = 1.34 GB`. At `k=8` that is 10.7 GB of HBM for a *single request*, against 140 GB of weights on a 2×H100 node — you have just cut your maximum concurrency by roughly a factor of 8 for one user. That is why no chat product runs beam search, and it is a better answer than "it's slower."

**⚠ Trap:** beam search with sampling. They are different objects — search maximizes, sampling draws — and combining them ("sample within each beam") gives you neither the mode nor a valid sample from the sequence distribution. If an interviewer offers you `num_beams=4, do_sample=True`, the correct response is that this is stochastic beam search, it has no clean interpretation, and I would not ship it.

### So when is beam search actually the right call, and when is it wrong?

The decision rule is about the shape of the output space, not about the model. **Use search when the task has a small number of correct outputs and you are trying to find one of them. Use sampling when the task has many acceptable outputs and you are trying to produce a natural one.**

Beam search is right for machine translation, grammatical error correction, constrained code completion where a single-token mistake invalidates everything, and any task whose evaluation metric is a similarity score against a reference (BLEU, chrF, exact match). In all of these, the reference is close to the mode, so maximizing sequence likelihood is aligned with the metric. Translation is the canonical case, and it is where beam search was developed — the correct German rendering of an English sentence is nearly unique, so a slightly better-scoring path really is a better translation.

It is wrong for open-ended chat, story generation, brainstorming, and long-form assistant responses, and the reason is the finding from Holtzman et al. (2020) I mentioned earlier: **the mode of an open-ended language model's sequence distribution is degenerate.** Push the beam width up and quality gets *worse*, not better — output converges on short, bland, hedge-laden text and eventually on literal loops. The intuition is that a repetition is high-probability by construction (the model conditions on itself), so any maximization procedure is drawn toward repetition as an attractor. Beam search finds it faster than greedy does.

There is a third, decisive practical reason nobody talks about: beam search does not stream. You cannot emit token `t` until you know which beam wins, and a beam can be overtaken at step `t+40`. So beam search forces a full-response wait — you go from a 200 ms TTFT to a 4-second wall of silence on a 300-token answer. For an interactive product that is a non-starter regardless of quality.

**🗣 Say this in the room:** "Beam search when the output is near-unique and you're scored against a reference — translation, GEC, structured extraction. Sampling for anything open-ended, because the mode of an open-ended LM is degenerate and beam search finds the degenerate mode faster than greedy does. Also beam search can't stream, which alone rules it out for chat."

**📅 Volatile:** the hosted APIs from the major providers do not expose beam search at all — they expose temperature, top-p, and `n`. If you want search you are self-hosting. Verify the current parameter surface before your loop rather than asserting it.

### Repetition penalty, frequency penalty, presence penalty — write the formulas and tell me how they differ.

They are three different functions of "how many times has this token already appeared," and the difference between multiplicative and subtractive is the whole answer.

**Repetition penalty** (Keskar et al., 2019, the CTRL paper) is *multiplicative in log space* — it divides the logit of any already-seen token by `θ > 1`:

```
z_i  ←  z_i / θ      if token i has appeared and z_i > 0
z_i  ←  z_i × θ      if token i has appeared and z_i < 0
```

The sign split is not a detail, it is the whole implementation. Naively writing `z_i / θ` for a *negative* logit makes it *larger* (less negative), i.e. it rewards the repeated token. Every correct implementation, HuggingFace's included, branches on the sign. Typical `θ` is 1.0 (off) to 1.2.

**Presence penalty** and **frequency penalty** (the OpenAI-style pair) are *subtractive*:

```
z_i  ←  z_i − presence · 1[count_i > 0] − frequency · count_i
```

Presence is a one-time flat tax the moment a token appears at all; frequency scales linearly with the number of prior appearances. Both range roughly `[-2, 2]`, and negative values *encourage* repetition — occasionally useful for forcing format consistency. Because they are subtractive in log space, they are multiplicative on odds: a frequency penalty of 0.5 on a token seen 4 times subtracts 2.0 from its logit, cutting its odds by `e^2 = 7.4×`.

The behavioral difference that matters: **repetition penalty is scale-dependent and count-independent; frequency penalty is scale-independent and count-dependent.** Repetition penalty's effect on a token depends on the magnitude of its logit, which is a model-specific, uncalibrated quantity — the same `θ=1.1` bites differently on two models with different logit scales. Frequency penalty subtracts a fixed amount regardless of logit scale, which makes it portable across models but means a single appearance is penalized identically whether the token is "the" or "photosynthesis". Presence penalty saturates after one use, which is what you want for "cover more topics"; frequency penalty grows without bound, which is what you want for "stop saying this word."

**📐 Numbers you must know:** `frequency_penalty = 0.5` on a token seen 3 times subtracts 1.5 from its logit → odds ×`e^-1.5 = 0.223`, a 4.5× reduction, and that is true for *every* token regardless of its logit. `repetition_penalty = 1.1` on a logit of 8.0 gives 7.27, a logit reduction of 0.73 → odds ×`e^-0.73 = 0.48`. On a logit of 2.0 it gives 1.82, a reduction of 0.18 → odds ×`e^-0.18 = 0.83`. Same parameter, exactly 4× different strength (the reduction is `z(1 − 1/θ)`, linear in `z`), depending only on where the logit happened to sit. That is the argument for preferring frequency/presence when you have the choice.

### Explain mechanistically how each of those penalties destroys code generation.

This is my favorite question in the section, because the answer is not "penalties are bad" — it is that **code has a legitimately Zipfian, highly-repetitive token distribution, and every repetition penalty is a blunt instrument aimed exactly at the tokens code cannot do without.**

Count the repeats in a trivial Python function: `def`, `self`, `return`, `(`, `)`, `:`, `,`, newline, and four levels of indentation whitespace. In a 200-line file, `self` might appear 80 times and the 4-space indent token thousands of times. These are not stylistic redundancy; they are syntax.

**Repetition penalty** is the worst offender because it is unbounded in reach: once a token has appeared *anywhere* in the context — including in your prompt, if the implementation penalizes prompt tokens — it is permanently taxed for the rest of the generation. The failure mode is specific and I have watched it happen: the model writes a correct opening brace, then at closing time the `}` token has been penalized because braces appeared earlier, so the model emits something else — a newline, a comment, another statement. You get syntactically invalid code with a *missing closing delimiter*, which is the single most common LLM code-gen failure attributable to sampling. Same mechanism kills Python indentation: after 20 indented lines the indent token is taxed enough that the model dedents mid-block.

**Frequency penalty** degrades more gracefully but produces the "variable name drift" failure: after `user_id` has appeared eight times, `frequency_penalty = 0.3` has subtracted 2.4 from its logit — odds ×`e^-2.4 = 0.09` — and the model starts writing `userId`, `uid`, or `user_identifier` for the same variable. The code parses. It does not run. This is worse than a syntax error, because CI catches syntax errors and a NameError three call frames deep gets shipped.

**Presence penalty** is the mildest, since it saturates after one occurrence, but at high values it drives the model away from re-using imported names, so you get code that imports `json` and then hand-rolls a serializer.

**no_repeat_ngram_size** is catastrophic for code and I would reject it in review without discussion. Setting `no_repeat_ngram_size=3` makes it *structurally impossible* to emit any 3-gram twice. `for i in`, `if x ==`, `return None`, `) ->` — all banned after first use. This is not a probability shift; it is a hard mask. The model cannot write a second for-loop.

**🗣 Say this in the room:** "For code I set all repetition penalties to their identity values — `repetition_penalty=1.0`, `frequency_penalty=0`, `presence_penalty=0`, `no_repeat_ngram_size=0` — because code's token distribution is legitimately repetitive and every one of these penalizes syntax. If a code model is looping, the fix is upstream: the model, the prompt, or the stop sequences, not the sampler."

**🔍 Failure taxonomy — "generated code has an unbalanced delimiter":** (1) Check `repetition_penalty ≠ 1.0` — this is the cause more often than everything else combined. (2) Check `no_repeat_ngram_size ≠ 0`. (3) Check whether the penalty is being applied to prompt tokens as well as generated tokens (see the next question). (4) Check that a stop sequence isn't firing on a delimiter that appears inside a string literal. (5) Only then suspect the model, and confirm by re-running at `temperature=0` with all penalties off.

### Should repetition penalties see the prompt tokens, or only the generated tokens?

Only the generated tokens, in almost every case, and the fact that different stacks default differently is a real portability bug I have hit twice.

The argument is straightforward once you state the intent: a repetition penalty exists to stop the *model* from looping. Tokens in the prompt were not produced by the model's degenerate attractor — they are the input, and in many tasks the correct output legitimately reuses them heavily. Summarization reuses the source's proper nouns. RAG-grounded QA quotes retrieved passages verbatim — that is the desired behavior, that is what makes the answer citable. Code editing reproduces surrounding code. Structured extraction copies field values character-for-character out of the document. In every one of those, penalizing prompt tokens penalizes *correctness*.

The concrete disaster: a 6,000-token RAG context containing the answer, with `repetition_penalty=1.15` applied over the full sequence. Every distinctive term in the retrieved documents — the customer's name, the SKU, the dollar amount — has already "appeared," so each is taxed. The model paraphrases instead of quoting, drifts on the numbers, and your groundedness eval drops while your retrieval metrics stay green. You will spend a week looking at the retriever.

The one legitimate exception is a genuinely open-ended continuation task on a base model, where the prompt is a *seed* the model is meant to continue away from rather than a *source* it is meant to draw on. That is a narrow case.

**⚠ Trap:** HuggingFace's `RepetitionPenaltyLogitsProcessor` receives the full `input_ids` — prompt included — so by default it penalizes prompt tokens. Some serving engines scope penalties to the output window only, and some expose a flag. This is not documented prominently anywhere. The rule I enforce in review: if you set a nonzero repetition penalty, you must state in the PR description whether it covers the prompt, and you must have a test that generates against a long context and asserts a quoted string is reproduced verbatim.

### How would you force or ban specific tokens at decode time, and what are the limits of that?

`logit_bias` — a map from token id to an additive offset applied to the logits before temperature and truncation. It is the cheapest, most surgical control surface in decoding, and it is also the one most often misused.

The mechanics: a bias of `b` on token `i` multiplies that token's *odds* by `e^b` relative to every other token, because logits are log-odds up to a shared constant. So `+2` is a 7.4× odds boost, `+5` is 148×, `-100` is `e^-100 ≈ 3.7e-44`, which is an effective ban — the token cannot survive any realistic softmax. `+100` on a single token is an effective force. Note the asymmetry: banning is reliable because you only need to push one token below every other token; forcing is reliable only when you bias exactly one token, since biasing five tokens by +100 each leaves them competing with each other on their original relative logits.

Where it genuinely earns its place:

**Banning a token that leaks formatting.** A model that keeps opening with a markdown code fence when you want raw JSON: bias the ``` token id to −100. Zero latency cost, zero prompt tokens.

**Banning a stop-token to force continuation.** Bias `eos` to −100 to guarantee the model runs to `max_tokens`. Occasionally useful for a "keep expanding this outline" surface. Dangerous — see the trap.

**Restricting the output alphabet for classification.** Bias every token except your label tokens to −100, set `max_tokens=1`, and you have turned a generative model into a classifier that cannot produce an out-of-vocabulary answer. This is the single highest-value use and it deserves its own question.

The limits are real. You must know the *token ids*, which means you must run the exact tokenizer the model uses, and the ids change between model families and sometimes between checkpoint revisions — a hard-coded id list is a time bomb. Multi-token strings cannot be banned this way at all: banning "Paris" as a *word* means banning every token sequence that could spell it, including `" Paris"`, `"Par"+"is"`, and `"P"+"aris"`, and the leading-space variant is the one people forget. And **logit bias is not a safety control** — it shapes probability, it does not enforce a language. If you need a guarantee, you need constrained decoding with a grammar, not a bias.

**⚠ Trap:** banning EOS and then wondering why the model produces increasingly incoherent text past its natural stopping point. Once a model wants to stop, everything after the suppressed EOS is off-distribution — it has never seen training examples that continue past a natural completion. Quality falls off a cliff within 50 tokens. If you need longer output, ask for it in the prompt; do not force it in the sampler.

**📅 Volatile:** provider support for `logit_bias` is uneven and has been shrinking — several reasoning-model endpoints do not accept it, and some providers cap the bias magnitude. Verify against the current API reference before designing around it.

### Design a text classifier that uses a single forward pass and returns calibrated-ish probabilities.

This is the "classification by logit" pattern and it is one of the highest-leverage tricks in applied LLM work: it converts an open-ended generative call into something with the cost profile and the interface of a scikit-learn classifier. The mental model: you are not asking the model to *write* a label, you are reading the probability it assigns to each label token at exactly one position.

The design:

1. Prompt so the very next token is unambiguously the label. End with something like `"\nSentiment (positive/negative/neutral):"` — the trailing colon and the enumerated options make the next-token distribution sharply concentrated on the three options.
2. Choose **single-token, unambiguous labels.** Check with the actual tokenizer. `" positive"` with a leading space is usually one token; `"positive"` without may be too, and they are *different ids* — the model, having just emitted a colon, will produce the leading-space variant. Getting this wrong gives you a classifier that reads the wrong three logits and returns garbage that looks plausible. If your labels are multi-token, use single letters or digits (`A`/`B`/`C`, `1`/`2`/`3`) and map them.
3. Set `max_tokens=1`, `temperature=0`, and either `logit_bias = {id: 100 for id in label_ids}` to guarantee the answer is in-set, or request `logprobs` with `top_logprobs=20` and read the label logprobs directly.
4. Softmax over *only* the label logits to get a normalized distribution across your classes.

```python
import math
resp = client.chat.completions.create(
    model=MODEL, messages=msgs, max_tokens=1, temperature=0,
    logprobs=True, top_logprobs=20,
)
top = {t.token: t.logprob for t in resp.choices[0].logprobs.content[0].top_logprobs}
raw = {lbl: top.get(tok, -100.0) for lbl, tok in LABEL_TOKENS.items()}
m = max(raw.values())
exp = {k: math.exp(v - m) for k, v in raw.items()}      # stabilized softmax
Z = sum(exp.values())
probs = {k: v / Z for k, v in exp.items()}              # {'positive': 0.91, ...}
```

**💰 Math:** the win is output tokens. A generative classification call that emits `"The sentiment here is positive."` costs ~8 output tokens; this costs 1. At $15/Mtok output that is `8 × 15e-6 = $1.2e-4` vs `1.5e-5` per call — 8× on the output line. But the bigger win is latency: output tokens are serial, so 8 tokens at ~25 ms/token inter-token latency is 200 ms of decode versus 25 ms. At 5 M classifications/month you save `5e6 × 175 ms = 243 GPU-hours` of decode occupancy and roughly `5e6 × 1.05e-4 = $525/month` on output alone. And you get a *probability*, which means you can set a confidence threshold and route the uncertain 8% to a bigger model or a human — which is worth more than the cost saving.

**⚠ Trap:** treating those probabilities as calibrated. They are not, in the strict sense — instruction-tuned models are systematically overconfident, and RLHF makes it worse. They are *monotone-useful*: a 0.95 is genuinely more reliable than a 0.6, so thresholding works. But do not report them as "the model is 95% confident" to a user or a regulator. If you need calibration, fit a temperature-scaling parameter on a held-out labeled set — one scalar, fit in seconds, and it usually cuts expected calibration error substantially.

### Show me how logits-processor ordering can silently change the distribution.

The sampler is a pipeline of pure functions on a logit vector, and like any pipeline of non-commuting operations, order is semantics. Almost no one writes it down, and almost every stack orders it slightly differently.

The clean example is **temperature and top-k**. Consider three logits: `[10.0, 8.0, 6.0]`, `k=2`, `T=0.5`.

Order A — temperature first, then top-k. Divide: `[20, 16, 12]`. Top-2 keeps `[20, 16]`. Softmax: `e^0/(e^0+e^-4) = 1/(1+0.0183) = 0.982` and `0.018`.

Order B — top-k first, then temperature. Top-2 keeps `[10, 8]`. Divide by 0.5: `[20, 16]`. Softmax: identical, `0.982 / 0.018`.

These commute, because top-k is rank-based and temperature is rank-preserving. Now do the same with **top-p**, which is *mass*-based.

Logits `[10.0, 8.0, 6.0, 5.0]`, `p = 0.9`, `T = 2.0`.

Order A — temperature first: logits become `[5, 4, 3, 2.5]`. Exponentials `148.4, 54.60, 20.09, 12.18` sum to `235.3`, so probs are `[0.631, 0.232, 0.085, 0.052]`. Cumulative: `0.631, 0.863, 0.948`. Nucleus at p=0.9 is the first **three** tokens.

Order B — top-p first, at T=1: softmax of `[10,8,6,5]` → `e^10=22026, e^8=2981, e^6=403, e^5=148`, sum `25558`, probs `[0.862, 0.117, 0.0158, 0.0058]`. Cumulative: `0.862, 0.979`. Nucleus at p=0.9 is the first **two** tokens. Then apply T=2 to those two.

**Same parameters, different candidate sets — three tokens versus two.** Order A samples token 3 about 8% of the time; order B never samples it. That is not a rounding difference, it is a different model behavior, and it is invisible in any test that only checks "the output is valid JSON."

The general rule: **rank-based filters (top-k) commute with temperature; mass-based filters (top-p, min-p, typical, epsilon) do not.** The near-universal convention is temperature first, then truncation — HuggingFace, vLLM and the hosted APIs all effectively do this — which is order A. Min-p is the noted exception where several practitioners argue for the reverse.

Penalties make it sharper still, because a *subtractive* penalty does not commute with a *multiplicative* temperature. Frequency penalty before temperature gives `(z − f)/T`; after temperature it gives `z/T − f`. Those are equal only when `T = 1`. At `T = 0.5` the before-temperature ordering doubles the penalty's effective strength — `f = 0.5` becomes an effective `1.0` — which is exactly the kind of 2× discrepancy that makes a config behave differently on two engines that both claim to implement `frequency_penalty`. (Repetition penalty happens to commute with temperature, since both are multiplicative and temperature is sign-preserving, so the `θ` sign branch resolves the same way either way. That is a coincidence of the two being the same kind of operation, not a general rule.)

**🗣 Say this in the room:** "Ordering matters whenever a stage is mass-based rather than rank-based. Top-k commutes with temperature; top-p doesn't, and neither does a subtractive frequency penalty. The convention is penalties → temperature → truncation → softmax, and if I'm porting a config between two engines I diff the effective candidate set on a fixed prompt rather than trusting the parameter names to mean the same thing."

### Walk me through what actually happens when a stop sequence fires.

The mental model that resolves every stop-sequence bug: **stop sequences are strings, the model emits tokens, and the two do not align.** A stop sequence is not a token-level construct; it is a substring match applied to the decoded text, retrofitted onto a token-level generation loop. Everything awkward about them follows from that mismatch.

The mechanism in a serving engine: after each token is appended, the engine decodes the recent output and checks whether any stop string appears as a suffix of the accumulated text. If it does, generation halts, the stop string itself is *removed* from the returned text (this is the convention — the stop sequence is a delimiter, not content), and `finish_reason` is set to `"stop"`.

Three failure modes come out of the token/string mismatch.

**One: the stop string is a strict substring of a token.** If your stop is `"\n\n"` and the tokenizer has a single token for `"\n\nThe"`, the model emits that one token and you have overshot — the returned text contains `"The"` past your stop boundary, or the engine has to split a token, which it cannot do cleanly for the KV cache. Engines handle this by truncating the decoded string, so what you actually observe is that generation stopped *and* a fragment leaked. The fix is to choose stop strings that are token-aligned for your tokenizer, or to strip defensively on your side.

**Two: the stop string spans multiple tokens, so you cannot detect it until it is complete.** `"</answer>"` might be five tokens. During streaming, you have already sent `"</an"` to the user before you know a stop is coming. This is why any streaming implementation must **buffer at least `len(longest_stop_string) - 1` characters** before flushing to the client, and flush the buffer only on completion. Skip this and users see the closing tag flicker on screen before it disappears.

**Three: the stop string appears inside legitimate content.** Stopping on `"```"` in a code-generation surface means the first fenced block ends the response. Stopping on `"\n"` in a task that emits multi-line JSON truncates at line one. Stop sequences are a blunt tool; prefer a structural terminator the model was trained to emit (the chat template's own end-of-turn token) and reserve custom stops for genuinely delimited formats.

**⚠ Trap:** relying on a stop sequence for a hard guarantee. The model can, and eventually will, emit the stop string in a context where you meant it as content, and it can also fail to emit it at all — in which case you run to `max_tokens` and pay for every token. Stop sequences are an optimization, not a contract. The contract is `max_tokens` plus a parser that handles truncation.

### I'm streaming tokens to a browser and users occasionally see a U+FFFD replacement character in the middle of a word. Diagnose it.

That black-diamond replacement glyph is the tell for **decoding token ids one at a time.** Byte-level BPE tokens are sequences of bytes, not sequences of characters, and a single token can end in the middle of a multi-byte UTF-8 codepoint. Decode that token in isolation and you hand the decoder an incomplete byte sequence; with `errors="replace"` you get U+FFFD, with `errors="strict"` you get a `UnicodeDecodeError`.

This is not rare. Any non-ASCII content triggers it: an emoji is 4 UTF-8 bytes and is routinely split across two or three tokens; Devanagari, CJK, Cyrillic and accented Latin characters are 2–4 bytes each and split constantly. On an English-only demo you will never see it; the first Hindi or Japanese user finds it in ten seconds.

The fix is **incremental detokenization with a byte buffer and a committed offset**, which is what every real engine implements. The invariant: maintain the full byte string of everything generated so far; after each new token, re-decode the tail; emit only the portion that is a complete, valid UTF-8 prefix beyond what you have already emitted; keep the incomplete trailing bytes in the buffer for next round.

```python
class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids = []
        self.emitted = 0          # chars already sent to the client

    def push(self, token_id) -> str:
        self.ids.append(token_id)
        # decode the whole output each step (engines keep a sliding window
        # of the last ~8 tokens instead, for O(1) cost)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if len(text) <= self.emitted:
            return ""             # token added only incomplete bytes
        chunk, self.emitted = text[self.emitted:], len(text)
        return chunk
```

Two further subtleties that bite. **SentencePiece-family tokenizers represent a leading space as `▁`**, and whether the space is attached to the current token or the previous one differs between tokenizers — decoding a single token gives you `"▁hello"` and naive replacement gives you either a missing or a doubled space. Whole-string decoding avoids this entirely, which is why the reference implementation decodes the accumulated ids rather than the delta. **And re-decoding the full sequence every step is O(n²)** over a long generation; production engines keep a small sliding window of previously-decoded tokens as context for the incremental decode, exactly enough to get the boundary right, at O(1) per token.

**⚠ Trap:** doing the stop-sequence check on the token stream instead of on the accumulated decoded text. The two must be layered: detokenize incrementally, accumulate the text, check stop strings against the text, and buffer the last `len(stop)-1` characters before flushing. Teams that check stops against tokens miss every stop string that isn't token-aligned, then add a second string check on the client, then have two sources of truth.

### How do you handle `max_tokens` and stop reasons in production? This feels trivial and I suspect it isn't.

It isn't, and the reason is that **truncation is a normal, expected, silent outcome, and almost every codebase treats it as an error case that never happens.** Every response carries a stop reason, and the difference between `"stop"` and `"length"` is the difference between "the model finished" and "we cut it off mid-sentence and are about to parse it."

The three outcomes you must branch on, in Anthropic/OpenAI-shaped APIs: a natural end (`end_turn` / `stop`), a hit on your custom stop sequence (`stop_sequence` / `stop`), and hitting the token cap (`max_tokens` / `length`). There are others — tool use, content filtering, refusal — and the rule I enforce is that the handler must be exhaustive with an explicit `else: raise`, never a silent fall-through. A new stop reason appearing in a provider update should break your tests loudly, not corrupt your data quietly.

What goes wrong when you ignore it:

**Truncated JSON parses as a different valid object.** `{"items": [{"id": 1}, {"id": 2}` fails to parse — fine, loud. But `{"score": 8` truncated to `{"score": 8}` by a lenient repair library gives you a plausible wrong answer. If you run a JSON-repair step, you must check the stop reason *first* and refuse to repair a length-truncated response.

**Truncated content gets cached and embedded.** A summarization pipeline that hits `max_tokens`, writes the half-summary to Postgres, embeds it, and serves it for six months. The eval that would catch it doesn't exist because the eval set has short documents.

**Retrying a length-truncation multiplies cost without fixing anything.** The naive retry wrapper sees a "bad" response and retries with identical parameters, producing an identically-truncated response at 2× cost. The correct handling is to either raise the budget, ask for a shorter answer, or continue from the truncation point with the previous output as an assistant prefix.

**💰 Math:** set `max_tokens` deliberately, not defensively-high. Many stacks set it to the model's maximum "just to be safe" — but a very high `max_tokens` inflates the *reserved* KV budget in some schedulers, reducing how many requests the engine will admit concurrently. Concretely: a request reserving 8,192 output tokens on a 70B GQA model reserves `2 × 80 × 8 × 128 × 8192 × 2 bytes = 2.7 GB` of KV. On an 80 GB card with 70 GB of weights (fp8) you have ~10 GB for cache, i.e. **3 concurrent requests**. Set `max_tokens = 1024` and the reservation is 336 MB, giving ~30 concurrent — a 10× throughput difference from one integer. Whether your engine reserves eagerly or allocates on demand is engine-specific (paged allocators do the latter), so measure, but the "set it high to be safe" instinct is not free.

**🗣 Say this in the room:** "I branch exhaustively on the stop reason and treat `length` as a first-class outcome, not an error. Truncated output never gets parsed leniently, never gets cached, and never gets a naive retry — it gets a budget decision. And I set `max_tokens` to the real expected length plus headroom, because on some schedulers the reservation directly caps concurrency."
