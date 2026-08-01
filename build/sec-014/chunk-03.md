### I want to add 5,000 domain terms to my model's vocabulary. Walk me through doing it correctly.

Before anything else: adding vocabulary is only possible if you own the weights. On a hosted API this question does not exist — you get the tokenizer the provider ships, full stop. So the premise is that you have an open-weight checkpoint and you are going to continue-pretrain or fine-tune it.

The mechanism is that you are appending `N` rows to the embedding matrix and (if untied) `N` rows to the unembedding matrix, and those rows arrive untrained. **A newly-added token is, by construction, a glitch token until you train it.** Every failure mode in this procedure follows from that one fact.

```python
# 1. Extend the tokenizer.
n_added = tok.add_tokens(new_terms)            # returns how many were actually new
old_v   = model.get_input_embeddings().weight.shape[0]

# 2. Capture the OLD sub-token decomposition BEFORE resizing, so we can
#    initialize each new row from the pieces it replaces. This is the good init.
pieces = {t: old_tok(t, add_special_tokens=False).input_ids for t in new_terms}

# 3. Resize. This grows both matrices (or the shared one, if tied).
model.resize_token_embeddings(len(tok))

# 4. Initialize new rows as the mean of their old sub-token embeddings.
import torch
E = model.get_input_embeddings().weight.data
with torch.no_grad():
    for t, ids in pieces.items():
        row = tok.convert_tokens_to_ids(t)
        E[row] = E[ids].mean(dim=0)
    # Any term with no usable decomposition falls back to the global mean.
```

**Why sub-token mean-init and not random.** A randomly initialized row — especially one drawn from the config's `initializer_range`, which is typically far smaller than the norm of trained embeddings — puts the new token at a point in embedding space the model has never visited. Its RMSNorm-scaled contribution to the residual stream is wrong in magnitude and meaningless in direction, and the first forward pass produces garbage that generates enormous gradients. Initializing to the mean of the pieces the term *used to* decompose into starts you at approximately the right place: the model already knows roughly what `indemn` + `ification` means, and the averaged vector is a decent first guess for the merged token. Training then only has to refine.

The weaker fallback — initialize every new row to the mean of all existing embeddings — is what recent versions of HuggingFace's `resize_token_embeddings` do by default (there is a `mean_resizing` behaviour; **verify what your installed version does rather than assuming**, because the default changed). Global mean is much better than random and noticeably worse than sub-token mean.

**The tie/untie decision.** If the model ties embeddings, resizing handles both sides in one shot and the new row must serve as both input representation and output classifier — which is fine, and it is what tying always meant. If untied, you must confirm `lm_head` was resized too and initialize *its* new rows separately (same sub-token-mean trick, over the `lm_head` rows). Assert both shapes after the resize. The silent-failure mode here is a framework that re-ties on resize and overwrites your trained `lm_head` with a transpose of `embed_tokens`.

**The unfreeze policy** — this is where most attempts fail. Three viable regimes:

1. **Embeddings-only warmup, then full unfreeze.** Freeze everything except `embed_tokens` and `lm_head`, train a few hundred steps at a normal LR to let the new rows find a sensible location, then unfreeze and continue-pretrain the whole model at a low LR. This is the safe default and the one I recommend.
2. **Full unfreeze from step zero, low LR, with warmup.** Works, but the untrained rows produce large early gradients that can perturb the trained weights. Gradient clipping is non-optional here.
3. **New rows only, everything else frozen forever.** Cheap, and it caps how good the result gets — the rest of the model never learns to *use* the new tokens, so they remain second-class.

**⚠ Trap:** LoRA plus vocabulary extension. LoRA adapts attention and MLP projections; it does not, by default, touch `embed_tokens` or `lm_head` at all. Add 5,000 tokens, train LoRA, and you have 5,000 rows that received exactly zero gradient — they are still glitch tokens, and your model now emits them (because they're in the vocab, so the sampler can pick them) with undefined behaviour. If you extend vocabulary you must include the embedding matrices in the trainable set: in PEFT that means `modules_to_save=["embed_tokens", "lm_head"]`, and it means your "small" adapter now carries `2 × V × d` full-precision parameters — for a 128k vocab at `d=4096`, that is 1.05B parameters in your "lightweight" adapter, roughly 4GB in fp32. People are surprised by this every single time.

### Should you add domain tokens? Give me the decision rule and the arithmetic.

Usually no. Here is how I decide.

**The gate — three conditions, all required:**
1. You own the weights and are already running a continued-pretraining or full-fine-tune job. If vocabulary extension would be the *reason* you start training, the answer is almost always no.
2. Your fertility profiling shows a concentrated, ranked list of terms where `tokens_per_occurrence × corpus_frequency` is large. Diffuse inflation across a long tail of terms is not fixable by adding 5,000 rows.
3. You have enough domain data to actually train the new rows. A rule of thumb I use: you want on the order of thousands of occurrences per new token in the continued-pretraining corpus. If a term appears 40 times in your whole dataset, its embedding will be undertrained and you have manufactured a glitch token to save four tokens.

**💰 Math — when it pays.** Legal-document pipeline, 1M documents/month, 6,000 tokens/document average. Profiling identifies 400 terms (statute formats, party-role terms, standard clause names) that average 4.2 tokens each and collectively occur ~250 times per document. Merging each to a single token saves `250 × 3.2 = 800` tokens/doc — 13.3%.

- Token savings: `800 × 1e6 = 800M` tokens/month. At $3/Mtok equivalent self-hosted compute cost, call it **$2,400/month**, plus 13% off prefill latency and 13% more document per context window.
- Parameter cost: `400 × 4096 × 2 = 3.3M` parameters. Negligible.
- Training cost: you were already doing a continued-pretrain run, so the marginal cost is the embeddings-only warmup — maybe an hour of GPU time.
- Risk cost: 400 rows that must be verified against the glitch-token audit before release.

At those numbers it is clearly worth it. Now change one input: **100** terms instead of 400, saving 3.3% of tokens, worth $600/month, and requiring you to spin up a training pipeline you did not otherwise have. Now it is clearly not worth it. The break-even is dominated by whether the training run was already happening.

**The alternatives I reach for first, because they are cheaper and reversible:**
- **Prompt-side abbreviation.** If a 6-token boilerplate phrase appears in every system prompt, define it once and use a 1-token alias. Zero training, zero risk.
- **Prefix caching.** If the expensive tokens are in the *fixed* part of the prompt, cache them at a ~90% discount and the inflation stops mattering. This solves the system-prompt half of the problem entirely.
- **Pick a different base model.** A model whose tokenizer already handles your domain (a code-trained tokenizer for a code product, a multilingual tokenizer for a multilingual product) gives you the compression for free with none of the risk.

**🗣 Say this in the room:** "Vocabulary extension is a real lever but it's the last one I'd pull. It requires owning the weights, a training run I'm already doing, enough domain data to actually train the new rows, and a concentrated head in the term-frequency distribution. If any of those is missing I'd get most of the benefit from prefix caching or from choosing a base model whose tokenizer already fits my corpus — and I'd have introduced zero new glitch tokens."

### I need to enforce a hard token budget per tenant. Do I count tokens locally or ask the provider? Where do they disagree?

Both, for different purposes, and understanding *why* they disagree is the whole answer.

**Local tokenization tells you the size of the content you control.** It is free, sub-millisecond, and synchronous, so you can use it in a hot path. It is exact for the raw text you pass in.

**The provider's count includes everything you don't see** — and that is where the gap lives. Every one of these adds tokens your local count of the message bodies will miss:

1. **Chat template overhead.** Role headers, turn delimiters, BOS. Typically 3–8 tokens *per message*, so a 20-turn conversation carries 60–160 tokens of pure scaffolding.
2. **Tool/function schemas.** Your tool definitions are serialized into the prompt and billed as input. Ten tools with decent JSON Schema and good descriptions is easily 1,500–3,000 tokens, on **every single call**, and it is completely invisible unless you go looking.
3. **Provider-side system additions.** Some providers inject their own preamble, safety text, or tool-use instructions. You cannot see it and you cannot count it locally.
4. **Multimodal content.** An image is billed at a token count computed from its dimensions by a provider-specific formula. There is no local tokenizer for a PNG.
5. **Structured-output/grammar scaffolding**, where the provider materializes the schema into the prompt.

So the design is: **local count for admission control, provider count for accounting, and the delta is a measured constant you monitor.**

For Anthropic there is a first-class endpoint that closes the gap: `client.messages.count_tokens(model=..., system=..., messages=..., tools=...)` returns the true `input_tokens` for exactly the request you would send, tools and all. Use it to *calibrate* — run it once per prompt-template version, record the overhead, and thereafter add that constant to your local estimate. Do not put a network round-trip in your per-request admission path.

**⚠ Trap:** using `tiktoken` to count tokens for a non-OpenAI model. It is a different tokenizer with a different vocabulary and different merges. It does not approximate other models' counts — it is simply the wrong function, typically undercounting by double-digit percentages on prose and much worse on code and non-English. If you are budgeting against a Claude or Llama model, use that model's tokenizer or that provider's counting endpoint. I have seen a rate limiter built on `tiktoken` against a non-OpenAI backend under-count by ~18%, which meant every "safe" request was over budget and the account hit provider-side 429s that the limiter believed were impossible.

**⚠ Trap:** counting input and forgetting output. `max_tokens` bounds output, but the model does not have to use it, and your *budget* must reserve it. Budget on `input_tokens + max_tokens` at admission and reconcile against actual usage after the response. Otherwise a tenant with a 100k budget can admit 100k of requests each reserving 4k of output and blow through by 4×.

### Design me the token-budget enforcement layer for a multi-tenant LLM gateway.

This is your Redis rate limiter with three changes, and naming those changes is the answer.

**Change 1: the unit is tokens, not requests.** A token-bucket in Redis keyed `tpm:{tenant}:{minute}`, refilled at the tenant's tokens-per-minute allowance. Requests differ in cost by three orders of magnitude, so counting requests tells you nothing about load.

**Change 2: the cost is unknown at admission time.** You know input tokens exactly; you know output tokens only after the fact. So it is a **reserve-then-reconcile** protocol, structurally identical to an authorization hold on a credit card:

```
admit:      reserve = count_input(req) + req.max_tokens
            atomically DECRBY the bucket; if it would go negative, reject 429
            record reservation_id -> reserve
respond:    actual = usage.input_tokens + usage.output_tokens
            atomically INCRBY the bucket by (reserve - actual)   # refund the unused hold
on error/timeout/disconnect: refund the entire reservation
```

The reconciliation must be idempotent and it must run on *every* exit path including client disconnect mid-stream, or your buckets leak reservations and tenants get throttled at 60% utilization with no explanation. Use a Lua script for the atomic check-and-decrement; you already know why.

**Change 3: two currencies, not one.** Input and output tokens have different prices (typically 4–5× apart) and different resource profiles — input is a prefill burst, output is sustained decode occupancy. Track both. Enforcement on a single blended number lets a tenant running long-generation workloads consume far more GPU-seconds than a tenant doing extraction, at the same "token" spend.

**The pieces that make it correct rather than approximately correct:**

- **Count with the right tokenizer, cached.** Load the tokenizer once per process (they are not cheap to construct), and version the cache key by tokenizer artifact hash so a model upgrade cannot silently change counts.
- **Discount cached-prefix input.** If your provider bills cached input tokens at ~10%, your budget should charge them at ~10%, or you are punishing tenants for the behaviour you want. This requires reading `cache_read_input_tokens` off the response and reconciling with it.
- **Add the measured template + tool overhead constant** from the calibration described in the previous answer.
- **Reject with the number.** A 429 that says "requested 8,400 tokens, 2,100 remaining in window, resets in 34s" is actionable; a bare 429 generates a support ticket.

**💰 Math — sizing the reservation policy.** Tenant at 200k TPM. Typical request: 3,000 input + `max_tokens=4000`, actual output 400. Reserving `max_tokens` means each request holds 7,000 and returns 3,600 — so at any instant you can admit `200,000 / 7,000 ≈ 28` concurrent requests, while actual consumption is `3,400 × 28 = 95,200` — you are running the tenant at **48% of their paid allowance** because of conservative holds. Fix by reserving a p95 of the tenant's *observed* output length instead of `max_tokens` (say 900 instead of 4,000), which raises admission to `200,000 / 3,900 ≈ 51` concurrent, ~87% utilization — and accepting that you will occasionally over-admit and must handle a bucket going negative gracefully rather than corrupting. That trade — conservative holds versus utilization — is the actual engineering judgment in this design, and it is what the interviewer wants to hear you reason about.

### My stop sequence isn't stopping. Explain the token-boundary problem.

Stop sequences are specified as **strings** and generation happens in **tokens**, and there is no guarantee the string aligns to a token boundary. Everything follows from that mismatch.

Three distinct failure modes:

**1. The stop string straddles tokens.** You stop on `"</answer>"`. The model emits `</`, `answer`, `>` as three tokens. No single token equals your stop string, so an engine that compares token-by-token never fires. Correct engines maintain a rolling decoded-text buffer and match on the text, not the tokens — which means they must buffer enough tokens to cover the longest stop sequence before releasing anything to the client. **If your stop sequence is long, you are adding buffering latency to every streamed token.**

**2. The stop string is a strict substring of an emitted token.** You stop on `"\n\n"`. The model emits a single token that decodes to `"\n\n\n"` (these exist in code-aware vocabularies). Your stop condition is satisfied *inside* that token. The engine must emit the prefix up to the match and discard the rest. Engines that only truncate at token boundaries leak an extra newline; engines that don't check within-token miss the stop entirely.

**3. The model never produces the exact string.** You stop on `"END"`. The model, in context, produces `" END"` — with the leading space, as one token — because that is how English tokenizes. Your stop never matches. This is the same leading-space phenomenon from the trailing-whitespace question, and it is the most common version of this bug in practice.

**⚠ Trap:** assuming the stop text is excluded from output. Providers differ on whether the matched stop sequence is stripped from the returned text or included, and they differ on what `stop_reason` / `finish_reason` value you get. **Read the response's stop reason and handle it explicitly** — `stop_sequence` vs `end_turn` vs `max_tokens` mean three completely different things about whether your output is complete, and code that parses `response.content` without checking is code that will one day parse a truncated JSON object as if it were whole.

**The controls I use, in order:**
- Prefer a **special token** as the terminator when you control the model — `<|eot_id|>` cannot straddle a boundary because it *is* a token. This is the structurally correct answer and it is why chat models have EOT tokens at all.
- If you must use a text stop, prefer something the model will emit at a natural token boundary and test it: generate 200 samples, grep for the intended stop string in the raw output, and confirm the stop actually fires.
- Use constrained decoding or structured-output mode when the real requirement is "stop when the JSON object closes." A grammar knows when the object is complete; a stop string is guessing.
- **Always** treat `max_tokens` as a possible outcome and validate before parsing.

### Users are seeing black-diamond question marks appear mid-stream in our chat UI, but the final saved message renders fine. Diagnose it.

Those are U+FFFD, the Unicode replacement character, and they mean something in your pipeline tried to UTF-8-decode a byte sequence that was a *partial* character.

The mechanism is byte-level BPE meeting multi-byte UTF-8. A token in a byte-level vocabulary is a sequence of bytes, and there is no rule that those bytes form complete characters. An emoji is 4 UTF-8 bytes; a Devanagari character is 3. The tokenizer may perfectly reasonably split those bytes across two tokens — token A carries the first two bytes, token B carries the last two. Decode token A in isolation and you have two bytes of a truncated 4-byte sequence: not valid UTF-8, so Python's `errors="replace"` hands you ``.

The final message is fine because when the stream ends you decode the *whole* token list at once and the bytes reassemble correctly. So the bug is exclusively in your incremental path.

**The wrong implementation** (which is what almost everyone writes first):

```python
for tok_id in stream:
    yield tokenizer.decode([tok_id])       # ← decodes each token in isolation. broken.
```

**The correct implementation** — decode a growing window and emit the diff:

```python
ids, emitted = [], ""
for tok_id in stream:
    ids.append(tok_id)
    text = tokenizer.decode(ids)           # full decode; bytes reassemble
    if text.endswith("�"):
        continue                           # incomplete char — hold, wait for more bytes
    yield text[len(emitted):]
    emitted = text
```

The `endswith("�")` guard is the key line: if the running decode ends in a replacement character, we are mid-character, so we buffer and wait for the next token instead of emitting a broken glyph. In production you would use an incremental byte-level decoder rather than re-decoding the whole list each step (that is O(n²) over the response), but the semantics are exactly this. HuggingFace's `TextIteratorStreamer` implements this pattern; most hand-rolled SSE handlers do not.

**🔍 Failure taxonomy — where the mojibake actually comes from:**

| Where you see it | Cause |
|---|---|
| Only mid-stream, final is clean | Per-token decode. Fix with the windowed diff above. |
| In the final message too | Real encoding bug — wrong codec somewhere in the transport, or bytes truncated. |
| Only on emoji / CJK / Indic | Confirms multi-byte splitting; same fix. |
| Only through your proxy, not direct | The proxy is re-chunking SSE frames on byte boundaries and splitting a UTF-8 sequence at the HTTP layer. Fix the proxy's buffering. |
| Only when a stop sequence is set | Stop-matching truncated mid-token; see the previous answer. |

**⚠ Trap:** "fixing" it by stripping `�` from the output. You have now silently deleted a character from the user's language instead of displaying a broken one. Buffer, don't filter.

### How do reasoning/thinking blocks and tool-call payloads interact with tokenization and cost? What do people get wrong?

They are all just tokens, and the whole point is that they are tokens you did not write and largely cannot see.

**Thinking tokens.** Extended-reasoning models emit a reasoning trace before the answer. Those tokens are generated by the model, so they are billed as **output** tokens — at output prices, which are typically 4–5× input. And on many current APIs the raw trace is not returned to you (you may get a summary, or nothing), so you are paying for tokens you cannot inspect. This is the single most surprising line item for teams adopting reasoning models: output token counts jump by a large multiple with no visible change in response length.

**📅 Volatile:** whether thinking is on by default, whether the raw trace is returned, how depth is controlled (a token budget parameter versus an effort level), and whether thinking blocks must be echoed back on subsequent turns all vary by model generation and have changed repeatedly. Read the current provider docs for the exact model you are shipping; do not carry forward what you remember. Verify before your loop.

The two operational consequences that are stable across all of that: **(a) `max_tokens` is a ceiling on thinking plus answer combined**, so a limit sized for your answer alone will truncate the answer after the thinking consumed the budget — the symptom is a `max_tokens` stop reason on a request that used to fit; and **(b)** if the API requires you to pass reasoning blocks back on the next turn for multi-turn tool use, those blocks re-enter as input tokens and grow your context every turn.

**Tool schemas.** Your tool definitions are serialized into the prompt on every request. They are input tokens, they are billed, and they are large. A tool with a nested JSON Schema, an enum, and a two-sentence description per field runs 150–400 tokens; twelve of them is 2,000–4,800 tokens on **every call in every turn of every conversation**.

**💰 Math — the invisible tool tax.** 12 tools × 250 tokens = 3,000 tokens of schema. 500k calls/day at $3/Mtok input: `3000 × 500,000 = 1.5e9` tokens/day → `$4,500/day` → **$135,000/month**, purely to re-transmit definitions that never change. With prefix caching at a 90% discount on cached input, that drops to **$13,500/month** — `$121,500/month saved` by making the tool block the stable prefix of your request. That is one config change. It is also why tool definitions must be **serialized deterministically** (sorted keys, stable ordering) — a Python `set` iteration order or an unsorted `json.dumps` changes the bytes, changes the prefix, and silently destroys the cache hit rate, turning that $13,500 back into $135,000 with no error anywhere.

**Tool *results*** are the other half and people forget them entirely. A tool that returns a 40KB JSON blob has injected ~10,000 tokens into the conversation, permanently, for every subsequent turn. **The rule I enforce in review: every tool returns a bounded, summarized payload, and the bound is asserted in code.** A search tool returns the top 5 with 200-character snippets, not the raw API response. If the agent needs the full document it can ask for it explicitly.

**🗣 Say this in the room:** "Thinking tokens bill as output at 4–5× input price and are often invisible to me, so I budget `max_tokens` for reasoning plus answer and I measure the ratio on my own workload. Tool schemas bill as input on every single turn, so they go in the cached prefix with deterministic serialization. And I cap tool result size in code — an unbounded tool result is a permanent context leak that compounds across the whole conversation."

### How does the tokenizer interact with prefix caching? What breaks the cache that people don't expect?

Prefix caching is keyed on the **token ID sequence**, not on your string. That single sentence explains every surprise in this area.

The cache stores KV tensors for a prefix of tokens and reuses them when a new request begins with the same token IDs. Matching is typically done on fixed-size blocks (16, 32 tokens) hashed by their contents, so the granularity of reuse is a block, not a token. Consequences:

**A change in byte 3 of a 12,000-token prompt invalidates 11,997 tokens of cache.** Not because caching is naive, but because attention is causal and every downstream key/value depends on every upstream token. There is no partial repair. This is why "put the timestamp at the top of the system prompt" is a five-figure mistake.

**Whitespace and serialization changes are token changes.** `json.dumps(d)` with unsorted keys, a Python `set` in your tool list, a `datetime.now()` in the system header, a trailing newline that a template sometimes emits and sometimes doesn't — each of these changes token IDs and each of them zeroes your hit rate silently. The observable is `cache_read_input_tokens` staying at 0 across requests that should obviously share a prefix.

**A tokenizer change invalidates everything, everywhere.** If you upgrade a model and the new one has a different tokenizer, every cached prefix in the fleet is dead — not stale, *dead*, because the same string now maps to different IDs. Plan the rollout for a cold cache: your first minutes at the new version will show TTFT at uncached levels and cost at full input price. If you do a canary rollout, the canary's cache is separate and will look terrible until it warms; do not read that as a regression in the new model.

**⚠ Trap:** a RoPE-base or position-encoding change in a "minor" model update. Cached KV tensors encode positional information; if the position encoding changed, the cached tensors are not merely stale, they are *wrong*, and a naive cache keyed only on token IDs would happily serve them. Any cache key must include the model/config version, not just the tokens.

**The design rule:** structure every prompt as `[frozen: tools + system + few-shots] → [semi-stable: conversation history] → [volatile: this turn's user input]`, in exactly that order, and put your cache breakpoint at the end of the frozen section. Then audit by asserting `cache_read_input_tokens > 0` on the second request of a fixed pair in CI. That assertion is worth more than any amount of care, because the failure is silent and cost-only — nothing breaks, you just pay 10× and nobody notices for a quarter.

**💰 Math:** 12,000-token system prompt, 200k calls/day, $3/Mtok input, 90% cache discount. Uncached: `12,000 × 200,000 × 3e-6 = $7,200/day = $216,000/month`. With a 92% hit rate: `0.08 × $7,200 + 0.92 × $720 = $576 + $662 = $1,238/day ≈ $37,100/month`. **~$179,000/month** rides on a serialization detail that a code review would catch in ten seconds if anyone were looking for it.

### How should I chunk documents for RAG — by characters or by tokens? Does it actually matter?

By tokens, and it matters more than people expect, for three reasons that are all downstream of fertility varying across content.

**Reason one: your budget is in tokens.** If your context allowance is "40 chunks × 400 tokens = 16,000 tokens," and you chunked by characters at 1,600 chars each, then an English prose chunk is ~400 tokens and a chunk of dense JSON or Hindi is 900–1,300 tokens. Your 40 chunks now cost 40,000 tokens and either blow the window or get silently truncated by your framework. Character-based chunking makes your token budget non-deterministic in exactly the cases where you can least afford it.

**Reason two: chunk *semantic* size becomes inconsistent.** The point of a fixed chunk size is that every chunk carries a comparable amount of information, so that embedding similarity is comparable across chunks. Character-chunking a mixed corpus gives you chunks holding wildly different amounts of meaning, which distorts retrieval scoring — a code chunk holds far less semantic content per character than prose.

**Reason three: it decouples you from the embedding model's own limit.** Embedding models have hard token limits (commonly 512 or 8192). Exceed it and most APIs *silently truncate* — you embed the first 512 tokens and the rest of your chunk contributes nothing to its vector, so the chunk is retrievable only by its opening. That is a silent recall failure and it is very hard to spot from the top because retrieval still returns results.

The correct implementation, with the detail people miss:

```python
def chunk(text, tok, size=400, overlap=50):
    ids = tok(text, add_special_tokens=False).input_ids
    step = size - overlap
    return [tok.decode(ids[i:i + size]) for i in range(0, len(ids), step)]
```

**⚠ Trap:** which tokenizer. You must chunk with the **embedding model's** tokenizer, not the generation model's, because the hard limit you are respecting belongs to the embedder. But your *context budget* is denominated in the generation model's tokens. These are different tokenizers with different fertility, so a 400-token chunk under the embedder may be 340 or 470 tokens under the generator. Measure the ratio on your corpus once and carry it as a constant; do not assume 1:1.

**⚠ Trap:** slicing token IDs mid-word and reassembling. `tok.decode(ids[i:i+size])` can start a chunk in the middle of a word, which is usually harmless but occasionally produces a leading fragment that embeds badly. The robust version chunks on *sentence or paragraph* boundaries and uses the token count only as a size constraint — semantic boundary first, token budget second. That is strictly better and it is what I would build.

### Talk to me about truncation. What's the right policy when input exceeds the window?

First principle: **truncation should be an explicit, logged, alarming decision — never a library default that fires silently.** The single worst configuration in this space is a tokenizer call with `truncation=True` and a `max_length` somewhere in a config file, because it will quietly discard the second half of a user's document and your pipeline will produce a confident, complete-looking, wrong answer.

The mechanical knobs, so you can talk about them precisely:

- **`truncation_side`** (`"right"` default, `"left"` available). Right-truncation drops the tail; left-truncation drops the head. For a chat transcript, left is usually correct — you want the recent turns and the system prompt, not the opening pleasantries. For a document you are summarizing, neither is correct and you should be chunking.
- **`padding_side`** — unrelated to truncation but the adjacent footgun: decoder-only models require **left** padding for batched generation, because right-padding puts pad tokens between the prompt and the generation position and the model attends to garbage at the position it is supposed to continue from.
- **Overflow / stride** — some tokenizer APIs will return the overflowed remainder as additional sequences with a configurable overlap, which is the right primitive for building a sliding-window pipeline rather than throwing content away.

**The policy I actually ship**, as a decision procedure:

1. **Count first, always.** Before the call, compute `input_tokens + max_tokens` and compare to the model's usable window (which is not the advertised window — see long-context degradation).
2. **If it fits, send it.** No truncation code path executes.
3. **If it doesn't fit and the content is a conversation:** drop whole turns from the middle, oldest first, never partial turns, and never the system prompt. A half-truncated tool result is worse than a missing one because the model will try to use it. Emit a marker (`[… 14 earlier turns omitted …]`) so the model knows history is missing.
4. **If it doesn't fit and the content is a document:** do not truncate at all. Chunk-and-map-reduce, or retrieve the relevant subset, or refuse with an actionable error. Truncating a contract and answering confidently about the half you kept is a correctness incident, not a capacity issue.
5. **Log every truncation event with tenant, request ID, and tokens dropped, and alert on the rate.** A truncation rate that climbs from 0.1% to 4% is telling you something changed upstream — a new customer with bigger documents, a language mix shift, a prompt template that grew.

**⚠ Trap:** truncating *after* rendering the chat template. If you truncate the rendered string you can cut a special token in half, or drop a `<|start_header_id|>` while keeping its matching close, and hand the model a structurally malformed transcript. Truncate at the **message** level, then render. The template is the last step, always.

**🗣 Say this in the room:** "I treat silent truncation as a correctness bug, not a resource limit. Counting happens before the call; if it doesn't fit, the response is either a structured drop of whole conversation turns with a marker, or a chunked pipeline, or an explicit error — never a `truncation=True` flag that eats half a document. And truncation happens on messages before templating, so I can never cut a special token in half."
