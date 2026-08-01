### You know Pydantic v2 cold. Show me how you'd make it the single source of truth for an LLM's output contract.

This is the highest-leverage thing a backend engineer brings to an AI team, and I would lead with it. The principle is that **exactly one artifact defines the contract**, and everything else — the schema sent to the provider, the parser, the business validation, the OpenAPI docs, the test fixtures — is derived from it. The moment the schema and the parser are two objects that can drift, you have a class of production bug that no eval will catch, because your evals also use one of the two.

```python
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Money = Annotated[float, Field(ge=0, le=1_000_000)]
ShortText = Annotated[str, StringConstraints(max_length=120, strip_whitespace=True)]

class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: ShortText = Field(description="Item name exactly as printed on the invoice")
    quantity: Annotated[int, Field(ge=1, le=10_000)]
    unit_price: Money

class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence: ShortText = Field(description="The line of the document the total was read from")
    vendor: ShortText
    currency: Literal["USD", "EUR", "GBP", "INR"]
    line_items: Annotated[list[LineItem], Field(min_length=1, max_length=200)]
    total: Money

    @field_validator("total")
    @classmethod
    def total_is_plausible(cls, v: float, info):
        items = info.data.get("line_items") or []
        computed = sum(i.quantity * i.unit_price for i in items)
        if computed and abs(computed - v) / max(computed, 1e-9) > 0.02:
            raise ValueError(f"total {v} disagrees with line-item sum {computed:.2f} by >2%")
        return v
```

`Invoice.model_json_schema()` is what you hand the provider. `Invoice.model_validate_json(response_text)` is what you hand the response. `ValidationError.errors()` is what you feed back on a repair attempt. One class, four uses, zero drift.

Three deliberate choices in that model that an interviewer should notice. `extra="forbid"` emits `additionalProperties: false`, which most strict schema modes *require* — so the config that makes your Python strict is the same config that makes your schema acceptable. Every string is bounded, because unbounded strings are the runaway-generation failure discussed later. And `evidence` comes first, which under an ordered FSM forces the model to quote the source line before it commits to numbers — a field-ordering trick that is free accuracy.

**⚠ Trap:** `Field(description=...)` is not documentation, it is prompt. Those descriptions are serialized into the JSON Schema you send to the model, so they are read by the model on every call and they cost tokens on every call. Write them as instructions to the model, not as notes to your teammates. Conversely, a schema with 40 verbose descriptions is silently adding hundreds of tokens to every request; at 400 schema tokens, 2M calls/month and $3/Mtok, that is `400 × 2e6 × 3/1e6 = $2,400/month` of pure schema overhead — worth measuring before you write an essay in a `description`.

**🗣 Say this in the room:** "The Pydantic model is the contract. `model_json_schema()` goes to the provider, `model_validate_json()` parses the response, `ValidationError.errors()` drives the repair prompt, and the field validators encode the business rules JSON Schema can't express. If the schema and the parser are ever two separate objects, they will drift and no eval will notice."

### Pydantic emits `$defs` and `$ref`. Providers sometimes choke on that. What do you do?

You canonicalize, and you decide deliberately whether to inline. Pydantic v2 factors every nested `BaseModel` into `$defs` with a `$ref` at the use site — correct JSON Schema, and necessary for recursive models, but a source of two concrete problems.

The first is **support**: some strict schema modes and some open-source grammar compilers handle `$ref` only partially (self-references especially), and the failure is often a compile error at the first request rather than at deploy time. The second is **cache-key instability**: `$defs` keys are derived from Python class names, so `LineItem` and `InvoiceLineItem` produce structurally identical schemas that hash differently and compile twice. If you have twenty tenants with template-generated models, that is twenty redundant grammar compiles and twenty cache slots.

My handling, in order. **Canonicalize first:** serialize with sorted keys, strip `title` fields (Pydantic adds them automatically and they are pure token cost — the model doesn't need `"title": "Vendor"` next to `"vendor"`), and hash that for the grammar cache. **Inline only when required:** write a small resolver that walks the schema, replaces each `$ref` with a deep copy of its `$defs` target, and drops `$defs` — but guard it, because a recursive model will make that walk diverge. Detect a `$ref` cycle before inlining and refuse rather than blowing the stack.

```python
def inline_defs(schema: dict) -> dict:
    defs = schema.get("$defs", {})
    seen = set()
    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                name = node["$ref"].split("/")[-1]
                if name in seen:
                    raise ValueError(f"recursive $ref to {name}; cannot inline")
                seen.add(name)
                out = walk(copy.deepcopy(defs[name]))
                seen.discard(name)
                return out
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk({k: v for k, v in schema.items() if k != "$defs"})
```

**⚠ Trap:** inlining a schema that references the same definition in fifteen places. You have just multiplied its token cost fifteen-fold in the request payload. A `$ref`-heavy schema that inlines to 3,000 tokens instead of 400 costs `2600 × 2e6 × 3/1e6 = $15,600/month` extra at 2M calls. Inline because a backend requires it, never as a default hygiene step, and measure the token delta both ways.

### How do you model "the agent picks one of eight actions" so that the schema does the work?

Discriminated unions, and this is where Pydantic v2's design pays off enormously compared to hand-rolling. The pattern is a `Literal` tag field per variant plus a `Field(discriminator=...)` on the union, and the payoff is threefold: the grammar can prune aggressively, Pydantic's error messages become useful, and validation is O(1) instead of trying every variant.

```python
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field, TypeAdapter

class SearchDocs(BaseModel):
    action: Literal["search_docs"]
    query: Annotated[str, Field(max_length=200)]
    top_k: Annotated[int, Field(ge=1, le=50)] = 10

class OpenFile(BaseModel):
    action: Literal["open_file"]
    path: Annotated[str, Field(max_length=400, pattern=r"^[\w./-]+$")]

class Finish(BaseModel):
    action: Literal["finish"]
    answer: Annotated[str, Field(max_length=2000)]

Step = Annotated[Union[SearchDocs, OpenFile, Finish], Field(discriminator="action")]
StepAdapter = TypeAdapter(Step)          # .json_schema() for the provider, .validate_json() for parsing
```

Why this matters to the *decoder*, not just to Python: with the discriminator first in the emission order, the grammar's state after the model emits `"action": "open_file"` collapses the union to exactly one branch. Every subsequent token is masked against `OpenFile` alone. Without a discriminator, an `anyOf` forces the automaton to track all branches simultaneously until they diverge, which makes the compiled grammar larger, the runtime state heavier, and the model's job harder — it can start down a branch and then be masked into a shape it did not intend.

Why it matters to *you*: without a discriminator, a Pydantic `ValidationError` on an eight-way union reports failures for all eight variants, producing a 60-line error blob. That blob is what you feed back in a repair loop, which means you have just spent 800 tokens telling the model about seven branches it never wanted. With a discriminator you get errors for the one branch it selected — typically three lines, ~40 tokens.

**💰 Math:** repair prompts on an undiscriminated 8-way union carry roughly 800 error tokens versus roughly 40 discriminated. If 4% of 2M monthly calls repair, that is `0.04 × 2e6 × 760 × 3/1e6 = $182/month` in pure error-blob tokens — small, but the real cost is that the model reads seven irrelevant branches and frequently "repairs" into the wrong one. I have watched a repair loop oscillate between two union variants for three attempts because the error message described both.

**⚠ Trap:** naming the discriminator field something the model has opinions about, like `type`. Models trained on tool-calling conventions have strong priors about `type`, `name` and `role`, and will occasionally emit a value from a different vocabulary. `action` or `kind` with an explicit `Literal` enum is safer; the grammar makes it impossible to be wrong, but you want the model's prior *aligned* with the mask rather than fighting it, because fighting shows up as degraded content in the other fields.

### Field validators as "repair hooks" — what does that mean concretely?

It means you stop treating validation as a binary gate and start treating it as a normalization layer with a rejection path, which is exactly how you already treat inbound API payloads. The LLM produces text with predictable, mechanical deviations; `mode="before"` validators absorb the ones that are unambiguous and let the genuinely ambiguous ones raise, so your expensive retry budget is spent only on real errors.

```python
from pydantic import BaseModel, field_validator
import re

class Extraction(BaseModel):
    amount: float
    invoice_date: date
    currency: Literal["USD","EUR","GBP","INR"]

    @field_validator("amount", mode="before")
    @classmethod
    def strip_money(cls, v):
        if isinstance(v, str):
            v = re.sub(r"[^\d.\-]", "", v.replace(",", ""))  # "$1,200.00 USD" -> "1200.00"
        return v

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, v):
        if isinstance(v, str):
            v = v.strip().upper()
            v = {"$":"USD", "US$":"USD", "€":"EUR", "£":"GBP", "RS":"INR", "₹":"INR"}.get(v, v)
        return v
```

The judgment call — and this is what the question is really probing — is **which deviations you absorb and which you reject**. My rule: absorb anything that is a *representation* difference with a unique correct answer (thousands separators, currency symbols, `"true"`/`"yes"`, ISO-vs-slash dates, leading/trailing whitespace, markdown fences around the whole body). Reject anything where absorbing it requires guessing at *meaning* — `"about 1200"`, `"1200 or 1500"`, `"N/A"`, an empty string in a required field. The first category is a parser bug on your side; the second is a model failure that must reach the retry loop or the human queue.

**⚠ Trap:** the silent-coercion slide. Someone adds a `before` validator that maps `"N/A"` to `0.0` because it was throwing in prod. Now every unreadable invoice records a total of zero, your validation error rate goes to zero, your dashboard turns green, and your finance team's ledger is quietly wrong. **A repair hook that can invent a value is not a repair hook, it is a data-corruption device.** The review rule I enforce: a `before` validator may reshape a value, never originate one. If it can return a value that was not derivable from the input, it belongs behind an explicit `Optional[...]` and a human-review flag.

**🗣 Say this in the room:** "I use `mode='before'` validators to absorb representation noise — currency symbols, thousands separators, date formats — because paying a 300 ms retry to fix a comma is absurd. But a validator is never allowed to invent a value; `'N/A'` has to raise and go to the retry loop or the review queue, otherwise you've converted a visible failure into a silent one."

### Implement a retry-with-error-feedback loop. Then tell me when to stop retrying.

The loop itself is fifteen lines; the interesting part is the stopping rule and the cost model.

```python
from pydantic import BaseModel, ValidationError

def extract(client, model_cls: type[BaseModel], messages, max_retries=2, **kw):
    schema = model_cls.model_json_schema()
    convo = list(messages)
    last_err = None
    for attempt in range(max_retries + 1):
        raw = call_llm(client, convo, schema=schema, temperature=0, **kw)
        try:
            return model_cls.model_validate_json(raw), attempt
        except ValidationError as e:
            last_err = e
            errs = e.errors(include_url=False)          # compact; no docs links
            brief = "\n".join(f"- {'.'.join(map(str,x['loc']))}: {x['msg']}" for x in errs[:8])
            convo = convo + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                 f"That response failed validation:\n{brief}\nReturn a corrected object. "
                 f"Change only the invalid fields."},
            ]
    raise last_err
```

Details that matter. `include_url=False` strips Pydantic's docs URLs, which are pure token waste in a prompt. Truncating to eight errors bounds the feedback size — a badly-shaped response can produce two hundred errors and you do not want to pay for them. Appending the assistant turn and a user correction (rather than rewriting the original prompt) means the prefix is unchanged, so **prefix caching still hits on the original prompt** and you only pay full price for the delta. And "change only the invalid fields" measurably reduces the case where the model rewrites a correct field into an incorrect one on retry.

Now the stopping rule, which is the senior half of the answer. **Retry at most twice, and only for errors of a type that retrying can fix.** A `missing` or `string_type` error is worth a retry — the model had the information and formatted it wrong. A business-rule violation from your own `field_validator` ("total disagrees with line-item sum") is worth exactly one retry, because if the model got the arithmetic wrong twice it is not going to get it right on attempt three; escalate to a stronger model or a human. And a *refusal* is worth zero retries: the model declined, and re-asking the same question with a validation error appended is a spend of tokens with a near-zero success rate. Detect refusals separately and route them out of the loop.

**💰 Math:** 2M calls/month, 800-token prompt, 300-token output, $3/Mtok in, $15/Mtok out. Base per call `$0.0069`. Retry payload ≈ 1,200 in + 300 out = `$0.0081`. At a 4% first-attempt failure with 80% of those fixed on attempt one:
- attempt-1 retries: `0.04 × 2e6 = 80,000 × $0.0081 = $648`
- attempt-2 retries: `0.2 × 80,000 = 16,000 × $0.0093` (payload grows again) `= $149`
- total overhead `$797/month` on a `$13,800` base — **5.8%**.

That is affordable. What is not affordable is the latency: those 80,000 requests take roughly 2× the p50, and 16,000 take 3×. If p50 is 900 ms, 4% at ~1.8 s and 0.8% at ~2.7 s puts your p99 above 2.5 s. **Retries are a cost non-event and a tail-latency event.** State it that way and you will sound like someone who has run this.

**⚠ Trap:** `max_retries` with no jitter or budget in a batch job. A pathological input — a scanned page that OCRs to noise — fails every attempt, and if your worker retries three times on 5% of a 400k-document backfill you have added 60,000 extra LLM calls. Put a per-job retry budget on top of the per-request retry count and alert when it saturates; this is the same discipline you already apply to Celery task retries, applied to a much more expensive unit of work.

### Strict schema modes demand every field be in `required`. How do you express an optional field?

As a union with `null`, and then you make Pydantic's `Optional` produce exactly that. This is a small mechanical point with an outsized effect on output quality, so it is worth being precise.

Strict modes require `required` to list every key in `properties`, because a grammar has to know deterministically whether the next token is a key or a closing brace — "this key may or may not appear" multiplies automaton states and creates ambiguity the compiler would rather refuse. The workaround is that the field is always *present* but may be `null`: `{"type": ["string", "null"]}` or `{"anyOf": [{"type":"string"}, {"type":"null"}]}`.

In Pydantic, `field: str | None` with a default of `None` produces exactly that `anyOf` and marks the field not-required. Under strict mode you want the union but you want it in `required` — so declare it as `field: str | None` with **no default**, which makes Pydantic treat it as required-but-nullable. The model must emit the key; it may emit `null`.

The quality consequence is the part people miss and it cuts both ways. Forcing the key to appear is *good* for recall: a model that can omit a field will omit it under uncertainty, and you lose extractions you would have gotten. But forcing it also invites the placeholder failure — the model, obliged to emit something, produces `""` or `"unknown"` rather than `null`. So pair nullability with an explicit instruction in the field description: `Field(description="The PO number printed on the invoice, or null if none appears")`. Then add a validator that maps the placeholder vocabulary to `None` and *counts* it, because the rate at which the model reaches for `"N/A"` instead of `null` is a real quality signal about your prompt.

**⚠ Trap:** `Optional[str] = None` in a strict-mode schema will typically be emitted by Pydantic as non-required, and the provider will then reject the schema or (worse) your compatibility shim will silently add it to `required` and change the meaning. Assert on the generated schema in a unit test — `assert set(M.model_json_schema()["required"]) == set(M.model_json_schema()["properties"])` — so the contract is enforced in CI rather than discovered in an API error.

### What makes a schema "LLM-friendly"? Give me the design rules.

The unifying principle: **the schema is a prompt with a grammar attached, and the model is filling it left to right with no ability to revise.** Every design rule falls out of that.

**Flat beats nested.** Each nesting level adds structural tokens the model must emit and track, and deep nesting is where models lose the thread — you get a correctly-shaped object with values assigned to the wrong parent. Three levels is my ceiling; beyond that I split into multiple calls or flatten with compound keys. Concretely, `{"buyer": {"address": {"city": ...}}}` becomes `{"buyer_city": ...}` unless the nesting carries real cardinality (a list of line items genuinely needs to be a list).

**Enums beat free strings.** `Literal["approved","rejected","needs_review"]` is a three-way masked decision; `status: str` is an open generation where the model will produce `"Approved"`, `"approved."`, `"APPROVED"` and `"approved (pending signature)"` across a thousand calls, and your downstream `if status == "approved"` misses a fraction of them. Under a grammar, an enum is enforced for free. Without a grammar, an enum in the schema is still a strong prompt.

**Bounded strings, always.** Covered at length in the next question; it is the single most expensive omission.

**Field order is emission order** — put context-establishing fields before decision fields.

**Field names are semantic hints.** `total_amount_usd` outperforms `t_amt`, because the name is read by a language model that has seen millions of invoices. Do not abbreviate to save tokens; the tokens you save are a rounding error and the accuracy you lose is not.

**One object per call unless the fields are genuinely coupled.** Splitting a 40-field extraction into three calls of 12–15 fields usually improves per-field accuracy and lets you route the easy call to a cheap model. The counter-argument is latency and the cost of re-sending the document; if the source document is 8k tokens and you split into three calls, you pay `3 × 8000 × 3/1e6 = $0.072` in input instead of `$0.024` — unless prefix caching covers it, at which point the marginal cost of the second and third call collapses by roughly an order of magnitude and the split becomes nearly free. **That is the actual decision rule: split the schema if and only if the document prefix is cached.**

**⚠ Trap:** `additionalProperties: true` as a "flexibility" escape hatch. Under a grammar, it is a licence for the model to emit arbitrary keys forever, and it is the second-most-common runaway-generation cause after unbounded strings. Under a strict mode, it is usually rejected outright. There is no version of this that is a good idea in a machine-consumed path.

### Tell me about unbounded string fields. Why is that a named trap?

Because it is the failure that costs real money, produces no error, and passes code review every single time. A field declared `summary: str` with no `maxLength` compiles to a grammar that permits any sequence of non-quote characters of any length. The mask never tells the model to stop. The only thing that stops it is EOS — which the model chooses — or `max_tokens`, which truncates mid-string and gives you an unparseable document.

The dynamics are worse than "sometimes it rambles," because of a positive feedback loop. Once the model is a few hundred tokens into a string, the local context is "I am in the middle of a long prose passage," which raises the probability of continuing prose and lowers the probability of the closing quote. Under a grammar the closing quote is legal at every position but competes against a distribution that has drifted toward continuation. If you have *also* set a frequency penalty, the `"` token has been suppressed proportional to how many quotes are already in the document — and now the string genuinely cannot end.

**💰 Math:** an extraction expected to emit ~200 output tokens. One unbounded `notes` field runs to `max_tokens = 4096`. At $15/Mtok output, intended cost is `200 × 15/1e6 = $0.0030`; the runaway is `4096 × 15/1e6 = $0.0614` — **20× the intended cost**, plus a truncated, unparseable body that triggers a retry, which may also run away. At 100k calls/day with a 2% runaway rate: `0.02 × 100,000 × ($0.0614 - $0.0030) = $117/day = $3,500/month` on a service whose intended spend is `100,000 × 30 × $0.0069 = $20,700/month` — a 17% cost overrun from one missing `max_length`. And the latency: 4096 tokens at 50 tok/s is 82 seconds. Your request timeout fires, the client retries, and you pay twice for a stream you never read.

The fixes, in order of preference. **Bound it in the schema** — `Annotated[str, StringConstraints(max_length=300)]` — and verify with the probe harness that your backend actually enforces `maxLength`, because many do not. **Bound it in the grammar directly** if you own the grammar; a repetition bound like `[^"]{0,300}` is trivially expressible in a regex-derived FSM and is genuinely enforced. **Bound it structurally** by replacing the free string with a bounded list of bounded strings, or with an enum, whenever the semantics permit. And as a backstop, **set `max_tokens` to a real budget** — 1.5× your p99 expected output, not 4096 — so the runaway is capped and, critically, *observable* as a `length` stop reason rather than as a mysterious cost line.

**🗣 Say this in the room:** "Every string field in every schema I ship has a length bound, and I alert on `finish_reason == 'length'` as a first-class error rather than treating it as normal. An unbounded string is a 20× cost tail and an unparseable body, and it never shows up in testing because your test documents are short."

### Does field order actually matter? Convince me with a mechanism, not a vibe.

It matters, and the mechanism is autoregression plus a fixed emission order. Under an ordered grammar the model must emit `properties` in the schema's declared sequence, and every token it emits is in the context for every later token. So a field placed early is *conditioning* for the fields after it, and a field placed late is *conditioned on* everything before it. There is no revision step. The model cannot compute the answer, then the reasoning, and reorder.

That gives you a free and quite large intervention: **put derivation before conclusion.** A schema of `{reasoning, answer}` lets the model spend tokens working the problem, and the `answer` token distribution is then conditioned on that work. A schema of `{answer, reasoning}` forces a snap judgment and then a post-hoc rationalization that is, by construction, unable to change the answer. This is chain-of-thought smuggled into the schema, and it is one of the highest-return schema edits available.

The same logic applies to `confidence` and to `evidence`. `{evidence, value, confidence}` is right: quote the source, extract from it, then rate it. `{confidence, value}` asks for a calibration on a number that does not exist yet, and what you get back is a number drawn from the model's prior over "what confidence values appear in JSON" — clustered at 0.85 and 0.95, uncorrelated with correctness. I have measured this; the correlation between a leading `confidence` field and actual accuracy is close to zero, and it becomes weakly but usefully positive when the field is moved after the value and after an evidence quote.

**⚠ Trap:** assuming your backend preserves order. Most FSM compilers use the schema's `properties` order, and Python dicts and Pydantic both preserve declaration order, so the chain usually holds. But a serializer that sorts keys — which you may have added yourself for cache-key canonicalization! — will reorder your properties alphabetically and silently destroy your reasoning-first design. Canonicalize for *hashing* on a copy; send the ordered schema. This is a genuinely nasty interaction between two individually-correct decisions and I have seen it ship.

**🏋 Drill:** take an extraction schema you have and run 200 documents twice — once with `{value, evidence}` and once with `{evidence, value}` — scoring exact-match accuracy on `value`. Pass criterion: you can state the delta with a confidence interval and explain the mechanism in one sentence without saying "the model thinks better." Budget: 45 minutes including the eval run.

### Your schema is recursive — a comment tree, a nested org chart. What breaks?

Three things break, in increasing order of annoyance. **Compilation:** a `$ref` cycle is not a regular language, so any backend that compiles to a DFA either refuses, silently truncates the recursion at some depth, or diverges trying to enumerate states. Backends with a pushdown automaton (XGrammar, llguidance) handle it, which is the concrete reason to know which engine you are on. **Termination:** a recursive grammar has no built-in depth bound, so nothing in the mask prevents the model from nesting forever; the only limiter is the model's own preference to stop plus `max_tokens`. This is the unbounded-string problem one level up, and it is worse because each level adds structural tokens. **Repair:** a `ValidationError` on a deeply nested structure produces `loc` paths like `("children", 3, "children", 7, "children", 0, "value")` that are expensive to feed back and hard for the model to act on.

My default is to **not use recursive schemas for LLM output**, and I would push back on a design that does. The alternative that works: emit a *flat* list of nodes with explicit ids and parent ids, then reconstruct the tree in Python.

```python
class Node(BaseModel):
    id: Annotated[int, Field(ge=0, le=999)]
    parent_id: int | None            # null for roots
    label: Annotated[str, Field(max_length=120)]

class Tree(BaseModel):
    nodes: Annotated[list[Node], Field(min_length=1, max_length=300)]

    @model_validator(mode="after")
    def acyclic_and_connected(self):
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate node ids")
        for n in self.nodes:
            if n.parent_id is not None and n.parent_id not in ids:
                raise ValueError(f"node {n.id} references missing parent {n.parent_id}")
        # cycle check omitted for brevity; walk parents with a visited set
        return self
```

This is strictly better in four ways: it compiles to a regular language so any backend handles it, `max_length=300` gives you a hard total-size bound, the model finds a flat list easier to produce correctly than deep nesting, and your structural invariants (acyclicity, connectedness, single root) become a `model_validator` you can unit-test rather than a grammar property you have to trust. The cost is one reconstruction function, twenty lines, that you write once.

**🗣 Say this in the room:** "I flatten recursion into an id/parent_id edge list and rebuild the tree in Python. It compiles to a regular grammar, it gives me a hard size bound, and structural invariants like acyclicity become a validator I can test instead of a grammar property I have to hope the backend implements."

### Name the failure class where the JSON is perfectly valid and completely useless.

I call it **"correct but unusable,"** and it is the characteristic failure of small models under constraint — the thing that happens the week after someone swaps a 70B for an 8B to save money and reports that "the format compliance rate went to 100%." It did. The extraction quality collapsed. Format compliance and content quality are orthogonal metrics and constraining hard makes the first one look perfect while hiding the second.

The taxonomy, because naming the variants is what makes this a senior answer:

**Schema echo.** The model emits the field's `description` or its type as the value: `{"vendor": "The name of the vendor", "amount": 0}`. The description was in its context; under a constraint it is a locally-plausible string. Detect with a substring check of each string value against the schema text.

**Placeholder flooding.** `"N/A"`, `"unknown"`, `"none"`, `"John Doe"`, `"example.com"`, `""`. The model was forced to emit a required string and had nothing. Detect with a placeholder regex and track the rate per field.

**Enum anchoring.** Every record gets the first enum value. This is partly the local-renormalization bias from constrained sampling and partly the model not actually deciding. Detect by comparing the enum distribution against a hand-labelled sample of 100 — if your labelled data is 40/35/25 and your production output is 92/5/3, you have anchoring, not a data shift.

**Numeric hallucination.** Constrained to a `number`, the model cannot say "I don't know," so it emits a plausible one. This is the most dangerous variant because it is undetectable by any format check and it is *specifically caused* by the constraint. Mitigate by making the field nullable and explicitly instructing null-on-absence, and by requiring an `evidence` string that must be a verbatim substring of the source — a validator you can actually enforce.

**Truncation-shaped output.** The model emits 3 of 14 line items and closes the array, because closing is always legal and it is racing the length budget. Detect by cross-checking a count field against `len(items)` — and put the count field *after* the list, so it is a report rather than a commitment.

The unifying detection strategy: **you cannot measure this with a parse-rate metric.** You need a per-field content eval — null rate, placeholder rate, value-distribution entropy against a labelled sample, and for numerics an evidence-groundedness check. My rule in review: if a dashboard shows format-compliance rate and nothing else, the service is unmonitored.

**⚠ Trap:** the celebratory rollout. Team ships constrained decoding, parse failures go from 6% to 0%, everyone declares victory, and nobody notices that the `amount` field's null rate went from 8% to 0% too — because the model can no longer decline. Those 8% did not become correct; they became fabricated. **A null rate that drops to zero when you add a constraint is an alarm, not a win.**

### How do you handle streaming when the output is constrained JSON? The user is staring at a spinner.

Streaming structured output is a genuinely different problem from streaming prose, because a partial JSON document is not a valid JSON document and `json.loads` on a prefix throws. You need an incremental parser plus a policy for when a field is safe to show.

The mechanism: buffer the token stream, and after each token attempt a *partial* parse that treats an unterminated document as complete-so-far by closing open brackets and, optionally, keeping trailing partial strings. Pydantic v2 ships this in `pydantic_core`:

```python
from pydantic_core import from_json

buf = ""
last_render = {}
for chunk in stream:
    buf += chunk
    try:
        obj = from_json(buf, allow_partial="trailing-strings")
    except ValueError:
        continue                      # not yet parseable; wait for more bytes
    render(obj)                       # obj is a dict with only the completed-enough keys
```

`allow_partial=True` drops an incomplete trailing string entirely; `allow_partial="trailing-strings"` keeps the partial string so you can stream text into a field as it arrives. That distinction is the whole UX decision.

Now the policy, which is the interesting part: **when is a field safe to render?** Three tiers, and I pick per field.

**Never-safe-until-closed:** anything the UI treats as a decision — an enum, a boolean, a numeric, an id. A partially-emitted number `12` may become `1200`; a partial enum `"appr` is unambiguous under a grammar but rendering it as `appr` looks broken. Wait for the field's closing token.

**Safe-to-stream-incrementally:** long prose fields — a summary, an explanation, an answer. Stream them character by character; this is where the perceived-latency win actually lives. Note that under `trailing-strings` you get the raw partial content and must be careful about rendering a half-emitted escape sequence (`\u00e` is not a character yet) — strip a trailing incomplete escape before rendering.

**Safe-to-append:** array elements. Each *completed* element can be appended to a list UI; do not render the in-progress element.

**💰 Math on why this is worth building:** a 400-token structured response at 50 tok/s takes 8 seconds end to end. If the user-visible answer field is the last field, non-streaming TTFB-to-useful-content is 8 s. Put the answer field first and stream it, and the user sees the first word at `TTFT + 20 ms ≈ 400 ms` — a **20× improvement in perceived latency** for zero change in total cost. The schema ordering decision and the streaming decision are the same decision.

**⚠ Trap:** rendering a field and then having it change. If you render `amount: 12` and the stream continues to `1200`, the number visibly mutates and users report it as a bug ("it showed the wrong total"). The rule is: **a field is rendered once, when it is closed** — with the sole exception of designated prose fields, where mutation-by-append is expected and reads as typing. Encode this as metadata on the Pydantic field (`json_schema_extra={"stream": "append"}`) so the frontend policy is derived from the same single source of truth as everything else.
