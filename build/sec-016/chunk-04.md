### Design the structured-output layer for an enterprise document-extraction product. Whiteboard it.

Take the concrete case: contracts and invoices, tenants define their own extraction templates, 5M documents a month, p99 under 3 seconds for the interactive path, and a finance team downstream that will treat every number as authoritative. I would draw four layers and defend each one on numbers.

**Layer 1 — the contract layer.** Every template becomes a Pydantic model, generated from the tenant's template definition and cached. That model is the only source of truth: schema to the provider, parser for the response, validators for business rules, and the same class used in the eval harness. On the write path, a schema-admission validator enforces the rules that keep this affordable: max depth 3, max 60 properties, every string bounded, no recursion, enum cardinality capped, and a compile-time budget with a hard kill. Templates that fail admission are rejected at authoring time with a readable error, not at inference time with a 500.

**Layer 2 — the generation layer.** Constrained decoding on, temperature 0, all repetition/frequency/presence penalties explicitly zero, `max_tokens` set at p99-expected-output × 1.5 rather than at a round number. Schema ordering is `evidence → values → confidence`, and every numeric field is nullable with an explicit "null if not present in the document" in its description. For long or ambiguous documents, a reasoning field of 1,200 characters before the values.

**Layer 3 — the validation and repair layer.** `model_validate_json`, then business validators (line items sum to total within 2%, dates within a plausible range, currency consistent). Repair loop: at most two retries, appending rather than rewriting so the document prefix stays cached, error blob truncated to eight entries with `include_url=False`. A refusal detector short-circuits the loop. Anything still failing after two attempts goes to a human-review queue with the raw output and the validation errors attached — that queue is a product feature, not a failure path, and pretending it does not need to exist is the most common design-review failure on this question.

**Layer 4 — the measurement layer.** Format-compliance rate, repair rate and depth, per-field null and placeholder rate, enum distribution drift against a labelled reference, and a 400-example golden set run on every schema or model change. Alert on repair rate, not on compliance rate, because compliance is the lagging indicator.

**💰 The cost model, which is what makes this an answer rather than a diagram.** 5M docs/month, average 6k input tokens, 250 output tokens. At $3/Mtok in and $15/Mtok out: `5e6 × (6000 × 3/1e6 + 250 × 15/1e6) = 5e6 × ($0.018 + $0.00375) = 5e6 × $0.02175 = $108,750/month`. Prefix caching does not help much here because every document is different — the only cached region is the system prompt and schema, maybe 600 tokens, saving `5e6 × 600 × 2.7/1e6 = $8,100/month`. Repairs at 4% with two-thirds resolving on attempt one add roughly `0.04 × 5e6 × $0.021 ≈ $4,200/month`. Net around **$105,000/month**, and the single biggest lever is not the constraint layer at all — it is routing the 70% of easy, short, single-page documents to a cheaper model. If a model at $0.60/$2.40 per Mtok handles 70% at equal accuracy, that tier costs `3.5e6 × (6000 × 0.6/1e6 + 250 × 2.4/1e6) = 3.5e6 × $0.0042 = $14,700` and total drops to roughly `$14,700 + 0.3 × $108,750 = $47,325/month` — a **56% reduction**. Say that in the room. Constrained decoding is the reliability story; routing is the cost story; conflating them is a junior answer.

**⚠ Trap:** designing the human-review queue as an afterthought and sizing it wrong. At 5M docs and a 0.5% escalation rate, that is 25,000 documents/month to review — about 12 full-time reviewers at 2 minutes each. The escalation threshold is therefore a *staffing* decision, and it should be tunable per tenant and per field rather than a global constant.

### Design the structured-output layer for an agent with sixty tools.

Sixty tools changes the problem qualitatively, because now the dominant cost is not the output schema but the *input* — sixty tool definitions in every request — and the dominant quality risk is tool selection rather than argument formatting.

**The input cost first, because it dominates.** Sixty tools with descriptions and parameter schemas is realistically 8,000–15,000 tokens. **💰 Math:** at 12,000 schema tokens, $3/Mtok, and 500k agent steps/month, that is `12,000 × 3/1e6 × 5e5 = $18,000/month` just to describe the tools — before any actual work. Prefix caching is the entire game here: if the tool block is static and sits at the front of the prompt, a 90% cache discount takes it to `$1,800/month`. So the design rule is absolute: **tool definitions are a static prefix, byte-identical across requests, ordered deterministically, with no per-request interpolation.** A single templated tenant name inside a tool description costs you $16,200/month. I have seen exactly this.

**Tool selection.** Sixty options in one flat list degrades selection accuracy noticeably. Two mitigations, and I would present both with their trade-offs. *Retrieval over tools*: embed the tool descriptions, retrieve the top 10–15 for the current step, and present only those — this cuts input cost further but breaks the static prefix and therefore the cache, so it is a win only if the tool block is large relative to the rest of the context. *Hierarchical selection*: group tools into 6–8 namespaces, have the model pick a namespace and then a tool within it, as a two-level discriminated union. This preserves the cache and turns one 60-way decision into a 7-way followed by a ~9-way, which is measurably easier. I default to hierarchical.

**The output schema.** A discriminated union over tool names, with the discriminator emitted first so the grammar collapses to a single branch immediately. Every argument bounded. A `reasoning` field before the tool selection if the model is not a reasoning model, because tool choice is exactly the kind of decision that benefits from derivation-before-conclusion.

**The constraint-tax hazard, which is the senior point.** If you attach a hard JSON grammar from token zero, you may mask the model's trained tool-call opener and suppress tool calling entirely — the model fills your schema with a fabricated direct answer instead of calling the tool it should have called. On open-weight models this is a documented, reported effect. **📅 Volatile:** magnitude is model- and version-specific. The mitigation is to use the provider's native tool-calling channel rather than bolting a response-format grammar over it, and to include an explicit `no_tool_needed` variant in your union so "don't call a tool" is a *representable* action rather than something the grammar forbids. **A schema that makes the correct action unrepresentable will get you the incorrect action, silently, 100% of the time.**

**⚠ Trap:** parallel tool calls plus a strict union. If the model can emit N calls and you validate them as a list of a discriminated union, one malformed argument blob fails the whole list and your repair loop re-does all N. Validate each call independently, repair only the failing ones, and keep the successful ones — this cuts repair cost by roughly `(N-1)/N` on multi-call steps.

### When would you deliberately *not* use constrained decoding?

There are five cases, and being able to name them is what distinguishes judgment from cargo cult.

**One: open-ended reasoning where you only need the final answer.** If the task is "solve this math problem" and you need a number, constraining the whole generation costs you the derivation. Generate free-form and extract with a regex or a cheap second call. The extraction failure rate on a well-prompted final-answer convention is very low, and you keep full reasoning.

**Two: creative or conversational output.** A grammar over prose is either vacuous or harmful. If the only structure you need is "sometimes emit a tool call," use the native tool channel, not a response grammar.

**Three: when the schema would be enormous or dynamic per request.** Compile cost and cache-miss rate can exceed the value. If you have 50,000 distinct schemas and each is used twice, you are paying a 1.4 s compile per two requests. Either restructure to a small set of schemas plus data, or move to a backend with a cheap cold path, or drop to JSON mode plus validation.

**Four: when you need calibrated uncertainty.** A constraint removes the model's ability to decline, and it destroys the signal in the logits by renormalizing over a tiny support. If routing depends on confidence, use the logit-inspection approach described below instead.

**Five: when the output format is genuinely stable without it.** Strong models on simple flat schemas with a good prompt achieve very high compliance unconstrained, and if you are validating anyway, the marginal value of the constraint over prompt-plus-validate-plus-one-retry is small. Measure it: if prompt-only gives you 99.3% and structured gives 99.97%, the delta on 2M calls is 13,400 requests — real, but it should be a decision, not a reflex.

**🗣 Say this in the room:** "I don't constrain reasoning, creative output, or anything where I need the model to be able to decline. I constrain the machine-consumed final block. And I check whether the constraint is actually buying anything — on a strong model with a flat schema, prompt-plus-validate is often within a few tenths of a percent, and then the question is whether that delta is worth the compile-cache infrastructure."

### JSON isn't the only structured output. When do you write a real grammar, and what does it buy you?

You write a real grammar when the output language is not JSON and correctness of that language matters — SQL, a query DSL, a diff/patch format, a config file, a chess move, a regex, a chemical formula. JSON Schema cannot express any of these; you would be reduced to putting a SQL string inside a JSON field, which gives you a validated container around an unvalidated payload. That is not structure, it is packaging.

The canonical case, and the one most relevant to enterprise and data-platform companies: **text-to-SQL**. A grammar over a restricted SQL subset buys you three things a prompt cannot.

**Syntactic validity by construction.** No unbalanced parens, no missing `FROM`, no invented keyword. This alone removes a large fraction of text-to-SQL failures.

**Schema grounding.** This is the big one. The grammar can be *generated from your actual database catalog*, so the only table names the model can emit are your tables, and — with a context-sensitive grammar or a staged approach — the only column names it can emit after `FROM orders` are `orders`' columns. Hallucinated table and column names are the dominant text-to-SQL failure mode and a catalog-derived grammar makes them unreachable. There is no prompt that achieves this.

**Security as a structural property.** A grammar that admits only `SELECT` with a whitelisted set of tables *cannot express* `DROP`, `DELETE`, `INSERT`, or a semicolon-separated second statement. Compare that to a prompt saying "only generate read queries" plus a regex blacklist — a blacklist is a denylist over an infinite space and it will be bypassed. **A grammar is an allowlist enforced at generation time, which is the strongest form of this control available.** That framing lands well with anyone who has done security review, and it is the single best argument for grammars over schemas.

A sketch in GBNF-style EBNF, which is what you would write for llama.cpp or adapt for llguidance:

```
root      ::= "SELECT " cols " FROM " table (" WHERE " cond)? (" LIMIT " int)?
cols      ::= "*" | col ("," " " col)*
table     ::= "orders" | "customers" | "line_items"
col       ::= "id" | "customer_id" | "amount" | "created_at" | "status"
cond      ::= col " " op " " lit
op        ::= "=" | ">" | "<" | ">=" | "<="
lit       ::= int | "'" [a-zA-Z0-9_ -]{1,64} "'"
int       ::= [0-9]{1,9}
```

**⚠ Trap:** the grammar guarantees syntax and vocabulary, not *semantics*. `SELECT amount FROM customers` is grammatical under the above and nonsense against the real schema. Column-table correspondence needs either a context-sensitive grammar (one production per table, which explodes combinatorially) or a post-generation catalog check. Do the catalog check; it is twenty lines and it is where the remaining errors live. Be explicit about this limit when you present the design, because an interviewer who has built text-to-SQL will test whether you over-claim.

### How would you do classification without a grammar at all?

Read the logits. For a classification task with a small label set, constrained decoding is a heavy hammer and there is a much better tool: **restrict to one token, set `max_tokens=1`, request logprobs, and read the distribution over your label tokens directly.**

The construction: pick labels whose first token is unique and single-token if possible — `A`/`B`/`C`, or ` positive`/` negative`/` neutral` verified against the tokenizer. Then either bias every other token to `-100` with `logit_bias`, or simply take `top_logprobs` and renormalize over your label set.

```python
LABELS = {" positive": None, " negative": None, " neutral": None}   # verify each is 1 token
ids = {tokenizer.encode(k)[0]: k for k in LABELS}

resp = client.chat.completions.create(
    model=MODEL, messages=msgs, max_tokens=1, temperature=0,
    logprobs=True, top_logprobs=20,
    logit_bias={str(i): 100 for i in ids},      # OpenAI-style: range is -100..100
)
top = resp.choices[0].logprobs.content[0].top_logprobs
scores = {t.token: math.exp(t.logprob) for t in top if t.token in LABELS}
Z = sum(scores.values()) or 1.0
probs = {k: v / Z for k, v in scores.items()}   # calibrated-ish posterior over labels
```

Three things this buys you that a grammar does not. **Calibrated probabilities**, so you can threshold — route anything under 0.7 to a human or to a stronger model, which is the whole basis of a cascade. **Cost**: one output token instead of a 30-token JSON object. At $15/Mtok that is `1 × 15/1e6 = $0.000015` versus `30 × 15/1e6 = $0.00045` — **30× cheaper on output**, and at 10M classifications/month the difference is `$150` versus `$4,500`. **Latency**: one decode step instead of thirty; at 20 ms/token that is 20 ms versus 600 ms of decode, so TTFT dominates and the response is essentially as fast as the prefill.

**⚠ Trap:** assuming your label strings are single tokens. ` neutral` may tokenize as ` neut` + `ral`, in which case your logprob is over a prefix that is shared with other words and your probabilities are garbage. **Always verify with the actual tokenizer**, and prefer single-character labels (`A`, `B`, `C`) with a legend in the prompt when you cannot. Leading-space variants matter too: `positive` and ` positive` are different token ids, and requesting the wrong one gives you a probability of approximately zero for the correct answer.

**📐 Numbers you must know:** `logit_bias` on OpenAI-compatible APIs takes values in −100 to +100, where ±100 is effectively a hard ban or a hard force. Because bias is added in log space, +5 multiplies the odds by `e⁵ ≈ 148×`, not by 5%. That conversion — bias is log-odds, not probability — is the thing people get wrong, and it is why "I added a small bias of +10" is actually a `e¹⁰ ≈ 22,000×` odds change.

### Provider structured outputs or self-hosted grammar engine? Argue both sides and pick.

The honest framing is that this decision is downstream of a decision you have usually already made — hosted API versus self-hosted weights — and dressing it up as an independent choice is a tell. But there are real cases where it stands alone, and here is how I would reason.

**Take the provider's structured outputs when** you are already on a hosted API, your schemas are drawn from a bounded set, and your compliance requirement is "essentially always" rather than "provably always." You get zero infrastructure, a maintained schema cache, and — the underrated part — the provider's grammar is co-designed with their model, so the constraint-tax interaction with their thinking channel and tool channel is handled. What you give up: visibility (you cannot see the mask, cannot measure compile time, cannot know which keywords are enforced without probing), portability (schema-feature support differs per vendor, so a multi-provider abstraction leaks), and the ability to express anything that is not JSON Schema.

**Run your own grammar engine when** any one of four things is true. Your output language is not JSON — this alone decides it. You need provable enforcement for a security or compliance argument, where "the model cannot emit a DELETE" must be a property of the system rather than a measured rate. You are self-hosting on vLLM or SGLang anyway, in which case XGrammar is already there and using it costs you nothing. Or your schema volume and dynamism are high enough that you need control over compile scheduling and cache policy.

**The multi-provider case deserves its own note**, because it is where teams actually get hurt. If you abstract over three providers, your structured-output layer must degrade to the *intersection* of their schema-feature support, which is smaller than any individual provider's. My rule: define the contract in Pydantic, target the intersection in the schema you send, and enforce everything outside the intersection in validators. That way switching providers changes the compliance rate slightly and changes the repair rate slightly, and changes nothing about correctness — because your validators are the real contract and the schema is an optimization that reduces how often they fire.

**🗣 Say this in the room:** "If I'm on a hosted API with bounded schemas, I use the provider's structured outputs and enforce everything they don't support in Pydantic validators. If I'm self-hosting I use XGrammar because it's already in the engine. I'd only build something custom if the output language isn't JSON or if I need provable enforcement for a security argument — and in the multi-provider case I target the intersection of schema features and treat validators as the actual contract."

### Give me the failure taxonomy for structured outputs, as a decision procedure.

**🔍 Failure taxonomy.** Start from the symptom and walk down; each branch terminates in a fix, not an anecdote.

**Symptom: response does not parse.**
→ Check `finish_reason`. If `length`: `max_tokens` truncated mid-document. Cause is almost always an unbounded string or an unbounded array. Fix: bound the schema field, raise `max_tokens` to p99×1.5, alert on `length` as an error.
→ If `stop` and it still does not parse: constraints are not actually on. Verify the request payload carries the schema; verify the SDK path you are using enforces rather than hints. If constraints are on and output is malformed, that is an engine bug — pin versions and file it.
→ If the body has markdown fences or a preamble: you are on rung 1 or 2, not rung 3. Add the constraint or add a tolerant pre-parser.

**Symptom: parses but fails schema validation.**
→ Errors on shape (wrong types, extra keys): JSON mode is on, schema mode is not. Move up a rung.
→ Errors on `pattern`/`minItems`/length: the backend is not enforcing that keyword. Confirm with the probe harness, move the constraint into a Pydantic validator, keep the keyword in the schema as a prompt.
→ Errors from your own business validators: this is the system working. Route to repair, cap at two attempts.

**Symptom: validates but is wrong.**
→ All fields empty or placeholder: prompt lost its instructions, or the content is not in context, or the schema forbids null. Check in that order.
→ One enum always the same value: enum anchoring from local renormalization plus a weak prompt. Compare the distribution against a labelled sample; add a reasoning field before the enum; consider logit inspection instead.
→ Numbers plausible but wrong: fabrication under constraint. Add a required `evidence` field validated as a verbatim substring of the source, and make numeric fields nullable.
→ Accuracy dropped when constraints were enabled: constraint tax. Add a reasoning field before the answer, or split into two calls.
→ Tool calls stopped happening: the grammar masked the tool-call opener, or `no_tool_needed` is unrepresentable in your union. Add the variant; use the native tool channel.

**Symptom: it is slow.**
→ Bimodal TTFT: grammar-compile cache miss. Check compile count vs request count and canonical-hash count.
→ Uniformly slower under load only: mask computation on the CPU path, not overlapped. Check whether you have a Python logits processor in the hot loop.
→ p95 doubled but p50 flat: retry loop firing. Check repair rate — this is a *quality* regression showing up as a latency metric, which is the most commonly misdiagnosed shape in this entire section.

**⚠ Trap:** the meta-failure. Every branch above that ends in "validates but is wrong" is invisible to the metric most teams ship, which is parse rate. If your only alarm is on compliance, four of the five branches under "validates but is wrong" will run in production indefinitely.

### 🏋 Drill: implement the whole thing in forty-five minutes.

**Task.** Unaided, no autocomplete, no documentation. Write, from memory:

1. A function that converts a **non-recursive JSON Schema** with `type: object`, string/integer/boolean/enum properties, and `required` == all properties, into a Python regex string that matches exactly the legal serializations with fixed key order and no whitespace variation.
2. A `TokenGuide` that takes a compiled DFA over that regex and a `{token_id: str}` vocabulary and builds the `state -> {token_id: next_state}` index.
3. A `LogitsProcessor` that masks illegal tokens to `-inf`, unmasks EOS only in accepting states, and exposes `accept(token_id)` plus `rollback(n)`.
4. A test that generates 200 random objects conforming to the schema, encodes each with the tokenizer, replays the token sequence through the guide, and asserts every token was legal and the final state was accepting.

**Pass criteria.** All four in 45 minutes. The regex handles `"` escaping inside strings correctly (`[^"\\]|\\.`). The index build is a single nested loop you can explain the complexity of out loud: `O(|states| × |vocab| × avg_token_len)`. `rollback` works — if you implemented state as a scalar with no history, you have failed this criterion and you now understand why real engines use a persistent stack. The test catches a deliberately introduced bug: change one property's order in the regex and confirm the replay test fails.

**Why this drill.** It is the smallest program that forces you to hold the character/token mismatch, the accepting-state/EOS coupling, the index-cost model, and the rollback requirement simultaneously — which are exactly the four things an interviewer probes when they ask "how does structured output work." Doing it once means you will never again answer that question with "it masks invalid tokens" and stop there.

### 🏋 Drill: build the provider probe harness and report the results in ten sentences.

**Task.** Ninety minutes, budget $20 of API spend, one provider of your choice.

Build a harness that, for each of these keywords — `maxLength`, `minLength`, `pattern`, `minimum`, `maximum`, `multipleOf`, `minItems`, `maxItems`, `uniqueItems`, `format: date-time`, `enum`, `oneOf`, `additionalProperties: false`, and a self-referential `$ref` — constructs a minimal schema, an adversarial prompt engineered to make the model *want* to violate that keyword, and a Python predicate that checks the constraint. Run N=100 samples per keyword at temperature 1 (you want to see the tail). Classify each keyword as **enforced** (0 violations), **decorative** (>0 violations, no API error), or **rejected** (API returns an error on the schema — the good outcome, because it fails loudly).

Also measure, per keyword: the first-call latency versus the median of calls 2–100, which gives you the schema-compile cost directly.

**Pass criteria.** A table with fourteen rows and four columns (keyword, classification, violation rate, first-call latency delta), plus ten sentences of prose stating which keywords you will now enforce in Pydantic instead of in the schema, and what your schema-admission validator should therefore reject. Bonus criterion: wire it as a weekly CI job and diff against last week's table, because provider behavior drifts and you want to find out from a test rather than from a customer.

**Why this drill.** Two reasons. First, it is the answer to "how do you know which schema features are supported," and having actually run it converts a vague answer into a specific one with numbers. Second, it is a genuinely good portfolio artifact — a public repo with a weekly-updated support matrix across three providers is the kind of thing that gets a reply to a cold outreach, because it is a small amount of work that nobody else bothered to do and everyone needs.

### Last one. Sixty seconds: how do you guarantee an LLM returns valid, usable structured data?

Here is the compressed version, and I would deliver it roughly like this.

Structured output is a logits mask, not a model capability: a grammar automaton compiled from your JSON Schema advances one step per emitted token and writes `-inf` into every logit that would break the grammar. Non-recursive schemas are regular and compile to an FSM with a precomputed per-state token index; recursion needs a pushdown automaton, which is why the modern engines — XGrammar as the default in vLLM, SGLang and TensorRT-LLM, llguidance as the Earley-based alternative — use a persistent stack with cheap rollback for speculative decoding.

There is a ladder. Prompt-only is free and fails 5–10%. JSON mode adds ~50 ms and fails 2–5% on *shape*, not syntax. Schema-constrained adds ~100 ms and fails under 0.1%, and the residual failures are truncation and refusal. Retry-with-error-feedback adds 200–500 ms on the failing tail only and handles the semantic rules a schema cannot express. An owned grammar costs a seconds-long compile and then is exact. I default to schema-constrained plus Pydantic validation plus at most two repairs, because those two layers cover different failure classes.

Then the part that matters more than any of it: **constraints make format failures impossible and content failures invisible.** A model that could have refused now cannot; a required numeric field means a number comes back whether or not the document contains one. So the schema is designed for that — evidence before values, reasoning before answers, every string bounded, every uncertain field nullable with an explicit null-on-absence instruction, enums instead of free strings, and no unbounded anything. And the dashboard measures per-field null and placeholder rates, repair rate and depth, and accuracy on a golden set — not just compliance, because compliance goes to 100% precisely when the interesting failures start.

**🗣 Say this in the room** (the thirty-second version, if they cut you off): "It's a logits mask driven by a grammar compiled from the schema, so invalid tokens are unreachable rather than caught. I use provider structured outputs plus Pydantic validators for the business rules, with two repair attempts appended to the conversation so prefix caching still hits. The thing I watch isn't compliance rate — that goes to 100% and stays there. I watch per-field null rate and repair rate, because a constraint converts a visible format failure into a silent content failure, and the null rate dropping to zero after you add a constraint is an alarm, not a win."
