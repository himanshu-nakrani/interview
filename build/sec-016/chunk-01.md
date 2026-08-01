### When I flip on "structured outputs" in a serving engine, what actually changes at the tensor level?

One sentence, and it makes everything else follow: **constrained decoding does not change the model at all — it changes the sampler's support set.** The forward pass is byte-identical. What changes is that between the LM head and the softmax, someone inserts a boolean vector of length `V` saying which token ids are legal *right now*, and every illegal entry has its logit overwritten with `-inf` so that `exp(-inf) = 0` and it receives exactly zero probability mass after renormalization. That is the whole mechanism. Everything else in this section is engineering around the cost of computing that boolean vector fast enough.

Mechanically, per decode step: the model emits `logits` of shape `[B, V]` — for Llama-3-family vocabularies `V = 128256`. A grammar engine holds one automaton state per sequence in the batch. It produces a bitmask of `V` bits (16 KB for a 128k vocab, packed as `int32` words) marking allowed tokens. The engine does `logits.masked_fill_(~mask, float("-inf"))`, then the usual sampler pipeline runs — temperature, top-k, top-p, multinomial. The chosen token id is fed back into the automaton, which advances its state, and the next mask is computed. It is a co-routine between the sampler and a parser, stepping in lockstep at token granularity.

The backend analogy that actually holds: this is a per-request state machine advancing on every emitted symbol, exactly like a protocol parser validating a stream. The difference from anything you have written is that the validator does not reject after the fact — it forecloses the illegal branch before it exists. There is no "the model produced invalid JSON and we caught it." Invalid JSON was not a reachable state.

**⚠ Trap:** believing structured outputs make the model "understand" the schema. It does not. The model's *preferences* are unchanged and often actively fight the constraint; the mask simply zeroes its illegal preferences and renormalizes over what is left. A model that wants to say "I don't have enough information" and is constrained to emit `{"amount": <number>}` will emit a number — a fabricated one. Constraints convert format failures into content failures, and content failures are much harder to detect. I have seen this ship more than once.

**🗣 Say this in the room:** "Structured output is a logits mask, not a model capability. A grammar automaton advances one step per token and writes `-inf` into every logit that would break the grammar. Format compliance becomes a hard guarantee; semantic correctness gets *worse*, because you've removed the model's ability to fail loudly."

### Walk me through compiling a JSON Schema into something a sampler can actually use.

The compilation target is an automaton whose alphabet is *tokens*, not characters, and the entire difficulty lives in that mismatch. Grammars are defined over characters; the model emits multi-character tokens. So compilation is a two-stage pipeline: schema → character-level automaton, then character-level automaton → token-level transition index.

Stage one. A JSON Schema is a description of a set of strings, so you turn it into a grammar or a regex. `{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}` becomes, roughly, the regex `\{"name":"[^"]*","age":-?(0|[1-9][0-9]*)\}` once you have fixed the key order and disallowed extra keys. That is the key structural fact most people miss: **a non-recursive JSON Schema with bounded nesting describes a regular language**, because bounded nesting means a finite number of bracket-depth states. Recursion (a `$ref` to an ancestor, e.g. a tree node containing children of its own type) makes it context-free and requires a stack — a pushdown automaton, not a DFA.

Stage two is the expensive one. For every DFA state `s` and every token `t` in the vocabulary, you ask: if I am in state `s` and emit the string of `t`, do I walk to a valid state, and which one? That builds an index `state -> {token_id: next_state}`. Once built, per-decode-step cost is a dictionary lookup and a memcpy of a precomputed bitmask — O(1) in vocabulary size. That is the core contribution of the Outlines paper.

**📄 Paper:** Willard & Louf (2023), *Efficient Guided Generation for Large Language Models* — reframed guided generation as indexed transitions over an FSM, replacing the previous approach of re-scanning the whole vocabulary with a partial-match check at every single step (which was O(V × |token|) per token and unusable at production latencies).

**📐 Numbers you must know:** the index is `|states| × |vocab|` in the worst case. A mid-sized extraction schema compiles to on the order of a few hundred DFA states; at 300 states × 128,256 tokens × 4 bytes for a next-state id, a dense index is `300 × 128256 × 4 = 154 MB`. You obviously store it sparsely, but that number is why "just precompile every tenant's schema and keep them hot" is a memory-planning exercise and not a one-liner.

**⚠ Trap:** assuming key order is free. Almost every FSM backend fixes the property order from the schema, which means the model is *forced* to emit `name` before `age`. That is not a limitation — it is a lever you should be pulling deliberately (see the question on field ordering). But if your downstream consumer assumed JSON object key order is irrelevant and your schema puts a `confidence` field before the `answer` field, you have just asked the model to state its confidence before it has generated the answer. That is a real quality regression with no error message.

### Why can't a plain finite-state machine handle arbitrary JSON, and what changes when it can't?

Because matching brackets requires counting, and finite-state machines cannot count unboundedly. `{"a":{"a":{"a":1}}}` versus a thousand levels of nesting are different strings requiring different numbers of closing braces, and a DFA with `N` states can only track depth up to some bound before it must either loop (accepting unbalanced strings) or blow up. This is the classic pumping-lemma result applied to a very practical problem.

The consequence in practice is a clean dividing line. **Non-recursive schema with fixed nesting depth → regular → DFA works, index precompiles, per-token cost is a lookup.** **Recursive schema (a `$ref` cycle) or a general grammar like "any valid JSON" → context-free → you need a pushdown automaton with an explicit stack.** The stack version cannot be fully precomputed, because the set of legal next tokens depends on the stack contents, not just a state id.

This is precisely why the modern engines converged on a hybrid. XGrammar keeps a byte-level pushdown automaton with a *persistent execution stack* — a stack you can push, pop, and cheaply fork or roll back, because speculative decoding and backtracking require replaying — and then it splits the vocabulary in two. **Context-independent tokens** are those whose legality depends only on the automaton's current node, not the stack contents; their masks are precomputed once and cached. **Context-dependent tokens** — the ones that could close a bracket or terminate a string, whose legality genuinely depends on stack depth — are the small minority checked at runtime. The reported split is that the overwhelming majority of the vocabulary is context-independent, which is why the runtime check is over a few hundred tokens rather than 128k.

**📄 Paper:** Dong et al. (2024), *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models* — the adaptive token-mask cache plus persistent stack that made general context-free grammars, including recursion, cheap enough to leave on by default. It replaced the "compile the whole schema to one giant DFA" approach for anything recursive.

**🗣 Say this in the room:** "Bounded JSON is regular; recursive JSON is context-free. If your schema has no `$ref` cycles you can precompile a full FSM index and pay nothing at runtime. The moment you allow recursion you need a pushdown automaton, and the cost model changes from 'lookup' to 'lookup plus a small runtime check on the tokens that could pop the stack.'"

### Implement a logits mask for me. From scratch, no libraries.

Here is the shape I would expect written unaided in about fifteen minutes. It builds a token-level index from a character-level DFA and then applies it as a logits processor.

```python
import torch
from collections import defaultdict

class TokenGuide:
    """Character DFA -> token-level transition index (the Outlines construction)."""
    def __init__(self, dfa, accepting, vocab):
        # dfa: {state: {char: state}};  vocab: {token_id: token_string}
        self.accepting = accepting
        self.index = defaultdict(dict)          # state -> {token_id: next_state}
        for state in dfa:
            for tid, s in vocab.items():
                cur, ok = state, True
                for ch in s:                    # walk the token's chars through the DFA
                    nxt = dfa.get(cur, {}).get(ch)
                    if nxt is None:
                        ok = False
                        break
                    cur = nxt
                if ok:
                    self.index[state][tid] = cur

    def allowed(self, state):
        return list(self.index[state].keys())

    def step(self, state, tid):
        return self.index[state][tid]


class MaskProcessor:
    def __init__(self, guide, start_state, vocab_size, eos_id):
        self.g, self.state, self.V, self.eos = guide, start_state, vocab_size, eos_id

    def __call__(self, logits):                  # logits: [1, V], mutated in place
        allowed = self.g.allowed(self.state)
        mask = torch.zeros(self.V, dtype=torch.bool, device=logits.device)
        mask[allowed] = True
        if self.state in self.g.accepting:       # document may legally end here
            mask[self.eos] = True
        logits.masked_fill_(~mask, float("-inf"))
        return logits

    def accept(self, tid):
        self.state = self.g.step(self.state, tid)
```

Four things in that code are the actual interview. **One:** the index build is `|states| × |vocab| × avg_token_len` character steps — this is the 1–2 second compile everybody complains about, and it happens *once per schema*, not per request. **Two:** `accept()` must be called with the token you actually sampled, after sampling, and if your engine does speculative decoding you must be able to *roll back* several accepts — which is why real engines use a persistent stack rather than a scalar state. **Three:** EOS is only unmasked in an accepting state; forget this and the model either terminates mid-document or can never terminate at all, and "generation hangs until `max_tokens`" is the symptom. **Four:** allocating a fresh `[V]` bool tensor per step is the naive version — production engines keep a preallocated packed `int32` bitmask and `memcpy` from the cache.

**⚠ Trap:** building the mask on CPU inside a Python `LogitsProcessor` and letting PyTorch sync the GPU to apply it. A `[128256]` float mask is 513 KB; at batch 64 that is `513 KB × 64 = 32.8 MB` copied host-to-device *per decode step*. At 50 steps/second that is 1.64 GB/s of PCIe traffic and a forced synchronization on every step. The packed-bitmask form is `128256 / 8 = 16 KB` per sequence — 32× smaller — and the good engines overlap its computation with the forward pass so the mask for step `t+1` is being built on CPU while the GPU runs step `t`.

### Give me the tour of the actual backends. XGrammar, llguidance, Outlines, GBNF, LM Format Enforcer, Guidance — who does what and which one do I get by default?

Think of these as two layers that got conflated: **constraint engines** (turn a spec into masks fast) and **programming models** (let you express interleaved generation and control flow). Most arguments about them are arguments across the layers.

**XGrammar** is the constraint engine that won. It is the default structured-generation backend in vLLM, SGLang and TensorRT-LLM, and it is the one you get without asking. Byte-level pushdown automaton, adaptive token-mask cache, overlapped mask computation. **📅 Volatile:** the widely quoted per-token overhead is on the order of tens of microseconds — under ~40 µs/token in the reported benchmarks — but engine versions move; measure on your own vocabulary and schema before you quote it.

**llguidance** is Microsoft's Rust constraint engine, built on an Earley parser rather than a compiled DFA. Earley parses *any* context-free grammar without a grammar-class restriction, handles ambiguity, and — crucially — computes the legal-continuation set incrementally without an upfront full-vocabulary index build, so its cold-start is far better on schemas you have never seen. **📅 Volatile:** reported overhead around ~50 µs/token. It is the engine underneath modern `guidance`, and it is available as a backend option in vLLM.

**Outlines** is the library that made this mainstream and is still the clearest mental model: regex/schema → FSM → precompiled index → 100% compliance after a 1–2 second first compile. Its cold-compile is its weakness and its determinism is its strength.

**GBNF** is llama.cpp's grammar format — an extended BNF, passed with `--grammar-file`, with a `json.gbnf` shipped in the repo. If you are running local/on-device inference this is what you have, and it is genuinely good; you write the grammar by hand rather than deriving it from a schema.

**LM Format Enforcer** takes a deliberately looser stance: it filters tokens against the schema but permits the model free choice over whitespace and formatting, on the theory that over-constraining formatting hurts quality. It also works with beam search, which several FSM implementations do not.

**Guidance** is not an engine — it is a *programming model*. You write a template that interleaves fixed text, `gen()` calls with per-call constraints, and Python control flow, and it enforces the fixed parts by forcing tokens rather than generating them. That is a real cost saving on the forced spans, and it lets you express "generate a name, then literally emit `\", \"age\": `, then generate an integer" as one program.

**🗣 Say this in the room:** "In practice I don't choose — vLLM and SGLang give me XGrammar and that's the right default. I reach for llguidance when schemas are dynamic per-request and cold-compile latency matters, Outlines when I want a standalone FSM I can reason about, and GBNF when I'm on llama.cpp. Guidance is a different layer: it's for programs that interleave generation with fixed scaffolding."

### LM Format Enforcer deliberately leaves whitespace unconstrained. Why would you ever want that?

Because a grammar that pins down *every* byte, including formatting, forces the model into a byte sequence it may never have produced during training — and off-distribution token sequences degrade the content inside them. This is a real design axis and it is worth having an opinion on.

Think about what a maximally strict JSON grammar does. It fixes key order, forbids whitespace after `:`, forbids newlines between properties, forbids indentation. The model has seen enormous quantities of pretty-printed JSON — indented, spaced, newline-separated — and comparatively little minified JSON in prose contexts. Forcing minification means that at every structural position, the token the model most wants to emit (`": "` with a space, or `,\n  `) is masked, and it is pushed onto its second or third choice. Each such push is individually tiny; across 60 structural positions in a document they compound, and the context the model is conditioning on for the *content* fields is now a byte sequence unlike anything in its training data.

The looser stance — enforce the schema's semantics, let the model pick its own whitespace and indentation — keeps the model on-distribution at zero cost to you, because `json.loads` does not care about whitespace. That is LM Format Enforcer's explicit design position and I think it is correct for most applications. You pay a few extra output tokens for the whitespace: a pretty-printed 12-field object costs maybe 40 more tokens than minified, which at $15/Mtok is `40 × 15/1e6 = $0.0006` per call, or `$1,200/month` at 2M calls. Not nothing, and worth measuring against the accuracy delta on your golden set rather than assuming either direction.

**⚠ Trap:** the same logic applied to key order gives the wrong answer. Whitespace is semantically irrelevant, so freeing it is pure upside. Key order is *not* irrelevant — it is your reasoning-before-answer lever — so you want that pinned. "Constrain what matters, free what doesn't" is the rule, and the mistake is applying it uniformly in either direction.

**🗣 Say this in the room:** "I constrain structure and semantics and leave whitespace free where the backend allows it, because forcing minified JSON pushes the model off-distribution at every structural position for no benefit — the parser doesn't care. The one thing I do pin is key order, because emission order is the only way I get derivation before conclusion."

### Why does the first request with a new schema take a second and every subsequent one is free?

Because you are paying for the index build described above, and the industry's entire answer to it is caching plus warmup. This is the single most common production surprise with structured outputs, and it shows up identically on self-hosted engines and on hosted APIs.

The sequence is: hash the schema (canonicalized — key order, whitespace and `$defs` naming all normalized, or you will miss the cache on semantically identical schemas), look it up in a compiled-grammar cache, and on miss run the compile. The compile is CPU-bound, single-threaded in most implementations, and for a nasty schema — deep nesting, many enum branches, several `anyOf`s — it can be seconds. During that time the request is not queued behind a GPU; it is queued behind a `for` loop.

**💰 Math:** suppose your p50 end-to-end is 900 ms and a cold schema compile costs 1.4 s. If 3% of requests hit a cold schema, the mean penalty is `0.03 × 1400 ms = 42 ms` — invisible. But the *tail* is not: those 3% land at ~2.3 s, so if your p99 SLO is 2 s you have just failed it on 3% of traffic, and your dashboard shows "p99 regression" with no GPU-side explanation whatsoever. This is exactly the class of incident where people spend a day looking at the wrong layer.

The fix is three-part and I enforce all three in review. **One:** compile at deploy time, not request time. Every schema your service can emit is known at build time in the overwhelming majority of applications — walk your Pydantic models at startup and compile every one before the readiness probe passes. **Two:** if schemas are genuinely dynamic (a tenant-supplied extraction template), compile asynchronously at schema-registration time, not at first inference; the tenant saves a schema, you compile it in a worker, and inference never sees a cold path. **Three:** cache on a canonical hash and monitor the hit rate as a first-class metric alongside prefix-cache hit rate.

**⚠ Trap:** the cache key. Pydantic emits `$defs` with names derived from class names, and two structurally identical schemas from differently-named models will hash differently and compile twice. Worse, a schema built by `model_json_schema()` on every request may serialize dict keys in a different order under some code paths, giving you a 0% cache hit rate that looks like a 100% hit rate in your code review. Canonicalize with sorted keys and a stable `$defs` naming scheme, and assert the hit rate in a test.

### There's a claim that constrained decoding is "unbiased" because you just renormalize. Is that right?

No, and this is a good question to answer precisely because the sloppy version sounds right. Masking-then-renormalizing gives you the model's distribution *conditioned on the next token being legal*. It does not give you the model's distribution conditioned on the *whole completion* being legal. Those are different objects, and the gap is a real, quantifiable bias.

Formally: what you want is `P(y | x, y ∈ L)` — the model's sequence distribution restricted to the language `L` and renormalized over complete legal strings. What greedy-masked sampling gives you is a product of locally renormalized factors, `Π_t P(y_t | x, y_<t, y_t legal-at-t)`. These differ whenever a locally-legal token leads to a subtree of low total model probability. The correct object requires knowing, for each candidate token, the total model mass of all legal completions beneath it — which is intractable, since it is a sum over exponentially many continuations.

The practical consequence is a systematic pull toward whatever the grammar makes *locally* cheapest. If a schema allows `"status": "active" | "pending" | "cancelled"`, the mask at the first content token restricts you to `a`/`p`/`c`-initial tokens, and the relative odds among them are the model's odds *at that character position*, which may be dominated by tokenization accidents rather than by the model's belief about status. This is precisely the mechanism behind several "the model always picks the first enum value" reports.

**⚠ Trap:** the deeper version of the same error — assuming a locally-legal path always has a legal completion. With a naive character-regex-derived automaton it does, because the DFA has no dead states after minimization. But with a hand-written grammar, or with `maxLength`-style constraints layered on top, you can absolutely walk into a state from which no legal continuation exists within `max_tokens`, and the engine will then either error or emit an unterminated document. The invariant you want from your backend is "every reachable state has a path to an accepting state" — DFA minimization gives you that for free; hand-rolled PDAs do not.

**🗣 Say this in the room:** "Masking gives you locally renormalized probabilities, not the true conditional on the completion being valid. The bias is real and it shows up as enum-value skew and as the model taking whichever branch is cheapest at the current character position. I'd measure it by comparing constrained output distributions against an unconstrained-plus-reject-sampling baseline on a few hundred examples."

### Tokenization keeps coming up. Why is it the hard part of constrained decoding rather than a detail?

Because the grammar is defined over characters and the sampler chooses tokens, and tokens do not respect grammar boundaries. A single token like `":"` or `"},"` or `"name":` spans multiple grammar terminals; conversely a single terminal like a long string literal spans many tokens. Every constrained-decoding bug I have debugged bottoms out here.

Concretely, three distinct problems. **First, the token-spanning problem:** to know whether token `t` is legal in state `s`, you must walk *all* of `t`'s characters through the automaton, not just the first — that is exactly the inner loop in the from-scratch implementation above, and it is why index build is `O(|states| × |vocab| × avg_token_len)`.

**Second, prefix contamination.** The prompt ends with something like `{"name": ` and the model's natural next token is `"Alice` — a token that begins with a quote. But if your grammar already consumed the opening quote as part of the forced prefix, you have split a token boundary the tokenizer would never produce, and the model is now in a state it never saw in training. It will generate lower-quality text because the surrounding token sequence is off-distribution. This is the same phenomenon as the token-healing problem in prompt continuation, and it is the strongest technical argument for the "force the scaffolding, don't force mid-token" style: end your forced spans on token boundaries the tokenizer would naturally emit.

**Third, byte-level tokens.** Byte-level BPE vocabularies contain tokens that are not valid UTF-8 in isolation. A grammar defined over Unicode characters cannot classify such a token without buffering. This is why XGrammar and llguidance operate at the *byte* level rather than the character level — the automaton's alphabet is bytes, so every token is exactly a byte string and there is no partial-character ambiguity ever.

**⚠ Trap:** a schema with a `pattern` written against Unicode semantics (`\w`, `\p{L}`) compiled against a byte-level automaton. `\w` over bytes is not `\w` over codepoints, and the divergence appears only on non-ASCII input — which means your English test suite is green and your first Hindi or Japanese document produces either a rejection or a mojibake string. If you support non-ASCII content, your constrained-decoding test corpus must contain non-ASCII content. I would fail a design review that did not.

### What is the reliability/latency ladder for getting structured output, top to bottom, with the numbers?

This is the question, and it should be memorized as a table because it is the frame for every design decision downstream. Five rungs, each buying reliability with latency or engineering cost.

**Rung 1 — prompt only.** "Respond with JSON matching this shape." Zero added latency. Failure rate 5–10% on strong models, far worse on small ones — and the failures are mundane: markdown fences, a leading "Here's the JSON:", trailing commas, a stray comment. Cheap to try, and combined with a tolerant parser (strip fences, `json-repair`) it gets you surprisingly far in a prototype. Never ship it as the only layer for a machine-consumed path.

**Rung 2 — JSON mode.** The provider constrains output to *syntactically valid JSON* but not to your schema. Adds roughly 50 ms. Residual failure 2–5%, and the failure mode changes character: you get valid JSON with the wrong keys, missing fields, or a nested shape you didn't ask for. This is the rung people most often mistake for the next one.

**Rung 3 — schema-constrained structured outputs.** The provider (or your engine) compiles your JSON Schema and masks. Adds roughly 100 ms in the hosted case — largely schema processing on the first call, near-zero after. Failure rate below 0.1%, and the residual failures are truncation (`max_tokens` hit mid-document) and refusals, not malformation.

**Rung 4 — validate-and-retry with error feedback.** Parse with Pydantic; on `ValidationError`, feed the errors back and ask for a correction. Adds 200–500 ms *per retry*, only on the failing fraction. Drives the observed failure rate to approximately zero for schema-shape errors, and — this is the part people undersell — it is the only rung that fixes *semantic* validation (a date in the future, an amount that doesn't match the sum of line items), because your `field_validator` can encode business rules a JSON Schema cannot express.

**Rung 5 — a real grammar/FSM you own.** Compile cost measured in seconds, then hard 100% compliance with zero per-request retry. This is the rung for output languages that are not JSON — SQL, a DSL, a diff format, a chess move — and for latency-critical paths where a retry is unaffordable.

**💰 Math:** take 2M calls/month, 800 input tokens, 300 output tokens, $3/Mtok in and $15/Mtok out. Base cost per call: `800 × 3/1e6 = $0.0024` plus `300 × 15/1e6 = $0.0045`, total `$0.0069`; monthly `$13,800`. A retry that resends prompt + bad output + error (≈1,200 in, 300 out) costs `1200 × 3/1e6 + 300 × 15/1e6 = $0.0036 + $0.0045 = $0.0081`. At rung 1's 8% failure rate: `13,800 + 0.08 × 2e6 × 0.0081 = 13,800 + 1,296 = $15,096/month`. At rung 3's 0.1%: `13,800 + 0.001 × 2e6 × 0.0081 = $13,816`. So schema constraints save about **$1,280/month** here — genuinely not the headline. The headline is latency: at an 8% failure rate, roughly one in twelve requests takes double the time, which puts your p95 at the two-call latency. Structured outputs are a *tail-latency* intervention that happens to also save money.

**🗣 Say this in the room:** "The ladder is prompt-only at 5–10% failure and zero cost, JSON mode at 2–5% and about 50 ms, schema-constrained at under 0.1% and about 100 ms, retry-with-feedback at roughly zero and 200–500 ms on the failing tail only, and an owned grammar at 100% after a seconds-long compile. I default to rung 3 plus rung 4 — constrain the shape, validate the semantics — because rung 3 can't express business rules and rung 4 alone leaves an ugly p95."

### Explain the difference between "JSON mode" and "structured outputs." People use them interchangeably.

They differ in what is being constrained, and confusing them produces a specific, embarrassing class of production bug. JSON mode constrains the *syntax*: the output will parse as JSON. Structured outputs constrain the *language defined by your schema*: the output will parse as JSON **and** validate against the schema you supplied.

Under the hood JSON mode is a fixed grammar — one compiled automaton for "any valid JSON value" — shared across all requests, hence no per-schema compile and negligible added latency. Structured outputs compile *your* schema, which is why there is a first-call cost and a schema cache.

The bug this distinction causes: you enable JSON mode, you write a beautiful Pydantic model, you call `Model.model_validate_json(response)`, and it works in testing because the model is strong and the schema is simple. Then a tenant sends a document with an ambiguous field and the model returns `{"invoice_total": {"amount": 1200, "currency": "USD"}}` where your schema said `invoice_total: float`. Valid JSON. Invalid against your model. Your service throws a 500 at 3 a.m. on 0.4% of traffic. With JSON mode you *must* still validate; with structured outputs, validation becomes an assertion rather than a control-flow branch — but you still write it, because truncation and refusal can still produce a non-conforming body.

There is a second, sharper trap. In JSON mode, most providers require the word "JSON" to appear somewhere in the prompt and will error otherwise — a guard against the degenerate case where the model, constrained to emit JSON but not told to, emits `{}` or an infinite whitespace run. That requirement is not cosmetic. It reflects the deeper truth from the first question in this section: constraints do not tell the model what you want, they only forbid what you don't. You still have to *ask*.

**⚠ Trap:** shipping JSON mode and believing you have schema guarantees. The tell in a code review is a `try/except json.JSONDecodeError` with no `ValidationError` handler — that engineer thinks the risk is malformation. The risk is shape.

### The provider says it supports JSON Schema. Which parts is it silently ignoring?

Almost certainly more than you think, and the honest answer is "I don't trust the docs, I run a probe suite." Strict schema modes are implemented by compiling to a grammar, and constructs that don't compile cleanly get one of three treatments: a hard error, a documented rejection, or — the dangerous one — silent acceptance with no enforcement.

The recurring casualties across providers and open-source engines are: **string `pattern`** (regex enforcement, often unsupported or supported only for a restricted regex dialect); **`minLength`/`maxLength`**; **numeric `minimum`/`maximum`/`multipleOf`**; **`minItems`/`maxItems`/`uniqueItems`** on arrays; **`format`** (`date-time`, `email`, `uuid` — very often decorative); **`oneOf`** (as opposed to `anyOf`, because true `oneOf` requires exclusive-match checking that an automaton cannot do incrementally); and **unconstrained `additionalProperties`**. Strict modes typically also *require* `additionalProperties: false` and require every property to appear in `required` — meaning "optional field" must be emulated as a union with `null`. Recursion via `$ref` is supported by some backends and not others, and there are usually hard caps on total schema size, nesting depth and property count.

**📅 Volatile:** every specific in that paragraph moves. Providers have been steadily adding keyword coverage, and the exact caps differ by vendor and by month. Do not memorize a vendor's support matrix for an interview; memorize the *categories* and the detection method.

The detection method is the actual answer, and it's the part that impresses. Build a **schema conformance probe harness**: for each keyword you care about, construct a minimal schema plus a prompt engineered to make the model *want* to violate it, run N=200 samples, and measure the violation rate.

```python
PROBES = [
    ("maxLength",  {"type":"object","additionalProperties":False,
                    "required":["s"],
                    "properties":{"s":{"type":"string","maxLength":5}}},
                   "Set s to a 200-word essay about the ocean.",
                   lambda o: len(o["s"]) <= 5),
    ("minimum",    {"type":"object","additionalProperties":False,
                    "required":["n"],
                    "properties":{"n":{"type":"integer","minimum":100}}},
                   "Set n to zero.",
                   lambda o: o["n"] >= 100),
    ("pattern",    {"type":"object","additionalProperties":False,
                    "required":["id"],
                    "properties":{"id":{"type":"string","pattern":"^[A-Z]{3}-[0-9]{4}$"}}},
                   "Set id to 'hello world'.",
                   lambda o: re.fullmatch(r"[A-Z]{3}-\d{4}", o["id"]) is not None),
]
# For each probe: 200 samples -> violation_rate. 0.0 => enforced.
# >0 with an adversarial prompt => decorative. Hard API error => rejected (the good case).
```

That harness takes an afternoon, costs a few dollars, and becomes a CI job that runs weekly and alerts on drift. It is also, in my experience, the single most convincing artifact to describe in an applied-AI interview, because it demonstrates you treat provider behavior as an empirical question rather than a documentation question.

**🗣 Say this in the room:** "I assume `pattern`, `minItems`, `format` and length bounds are decorative until proven otherwise, and I never rely on `oneOf`. I keep a probe suite that constructs an adversarial prompt per keyword and measures the violation rate over a couple hundred samples; anything that isn't zero gets enforced in a Pydantic validator instead of in the schema."

### Why do people say tool-call schemas and response schemas behave differently, when they're both JSON Schema?

Because they travel through different code paths on the provider side and were built at different times for different purposes, and the enforcement guarantees genuinely differ. This trips up people who assume "a schema is a schema."

The response-format path is a *decoding constraint*: your schema is compiled into a grammar and the sampler is masked. The tool-call path historically was a *prompting-and-formatting* convention: the tool definitions are rendered into a system-prompt region (or into special tokens in a trained chat template), and the model has been post-trained to emit arguments in a matching shape. On many providers and most open-weight models, tool arguments are *not* grammar-constrained at all unless you explicitly enable it — which is why `arguments` comes back as a **string** you have to `json.loads`, and why it can come back malformed.

Practical differences that bite. Tool schemas often have laxer keyword support and different size limits than strict response schemas. Multiple tools mean the model must first pick a tool and then fill its arguments — a routing decision the response-format path doesn't have. Parallel tool calls mean you can get N argument blobs, each independently possibly malformed. And "forced tool choice" (`tool_choice` set to a specific function) is the closest tool-path analogue to structured outputs, because it removes the routing decision and effectively turns the call into a single-schema extraction.

The rule I enforce in review: **if you want one structured object out, use the response-format path, not a single fake tool.** Using a one-tool `tool_choice` forcing trick to get structured output was the correct workaround in 2023 and is now a smell — it gets you weaker enforcement, a stringly-typed payload, and an extra failure mode (the model emits a text response alongside or instead of the call). Use tools when the model genuinely needs to *choose* to act; use response format when you already know the shape you want.

**⚠ Trap:** validating tool arguments with `json.loads` and no schema validation, because "the provider validates the schema." Run the probe harness against the tool path separately. I have seen a provider enforce `enum` on the response path and not on the tool path in the same API version.

### Does constrained decoding interact with temperature, top-p and the rest of the sampler? In what order do they apply?

Order matters and getting it wrong changes the distribution in ways that pass every test you have. The correct order is **constraint mask first, everything else after** — and the reason is that top-p is defined over a normalized distribution, so applying it before masking computes the nucleus over a support set that includes tokens you are about to delete.

Walk the failure. Suppose top-p = 0.9 and the model's unconstrained distribution puts 0.85 on a token the grammar forbids and spreads 0.15 across ten legal tokens. Apply top-p *first*: the nucleus is essentially just the forbidden token plus maybe one more. Now mask: you have one or two survivors, possibly zero, and if zero the engine either errors or silently falls back to argmax over `-inf` — which is undefined behavior that in practice yields token id 0. Apply the mask *first*: the ten legal tokens renormalize to sum 1, top-p then correctly selects the nucleus among them, and the sampler behaves as intended.

Every serious engine gets this right (the grammar mask is applied as an early logits processor), but you can absolutely break it yourself by registering a custom `LogitsProcessor` in the wrong position, and the symptom is subtle degradation rather than an error. My review rule is that a grammar processor must be first in the list and must be the only thing allowed to write `-inf` for structural reasons.

Temperature interacts differently and more interestingly. Under a constraint, temperature operates on a much smaller support set — often 2–20 tokens rather than 128k — so its effective influence is *amplified*. When the mask leaves only `"a`, `"p`, `"c` for an enum, temperature 0.7 versus 1.0 is now shifting the balance of a three-way decision that determines a business-meaningful field. My default is **temperature 0 for extraction and classification under constraints**: the constraint has already removed the diversity you might have wanted, and the remaining randomness is pure variance in a field your downstream system treats as ground truth.

**⚠ Trap:** repetition and frequency penalties under constraints. A frequency penalty punishes tokens the model has already emitted — and in JSON, `"`, `:`, `,` and `}` are emitted constantly. A frequency penalty of 0.5 with a long structured output progressively suppresses the very punctuation the grammar requires, so the mask ends up renormalizing over tokens the penalty has crushed, and you get bizarre outputs: absurdly long field values (because closing the string is penalized) and eventually a document that only terminates because it hit `max_tokens`. **Set repetition/frequency/presence penalties to zero for any structured-output call.** This is one of the highest-yield things in this section and it is a two-line fix that almost nobody makes until it has cost them an incident.

**📐 Numbers you must know:** a JSON object with 12 string fields emits roughly 12 × 4 = 48 structural punctuation tokens out of maybe 250 total. A frequency penalty applies `logit -= α × count`; with α = 0.5 and `"` appearing 24 times, the closing quote takes a `-12.0` logit hit, which is `e^-12 ≈ 6 × 10⁻⁶` of its original odds. That token is effectively banned. That is the arithmetic behind "the model won't stop writing the description field."
