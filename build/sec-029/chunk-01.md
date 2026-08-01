### Start me at the beginning: when I set a cache breakpoint on a provider API, what is actually being cached?

The KV cache for the prefix — nothing more, nothing less. When the model prefills a prompt it computes, for every token position and every layer, a key vector and a value vector. That tensor is the entire state the model needs to keep generating; it is a pure function of the token sequence up to that point. So if two requests share a byte-identical prefix, the KV tensors for that prefix are *identical numbers*, and recomputing them is pure waste. Prompt caching is the provider storing those tensors keyed by a hash of the prefix and handing them back instead of re-running prefill.

That framing dictates everything else, and it is the single sentence that separates a candidate who has shipped this from one who has read the docs. Because the KV at position `t` depends causally on every token `0..t-1`, **the cache key is a prefix, not a set**. Change byte 12 of a 12,000-token prompt and positions 12 through 11,999 all have different keys and values. There is no partial credit, no diffing, no "it only changed a little." This is not an implementation shortcut the provider might fix later — it is what causal attention *means*.

Mechanically the provider hashes in fixed-size chunks (block-level hashing, typically 16–128 tokens, exactly like a content-addressed store) and walks forward from position 0 until the first chunk whose hash misses. Everything before the miss is a read; everything after is recomputed and written.

**⚠ Trap:** candidates describe prompt caching as "response caching" — the belief that identical prompts return identical cached *answers*. It is not. Output is still sampled fresh every call; the cache only removes prefill compute. You get the same latency and cost benefit on 1,000 different questions asked against the same cached 20k-token document. If your mental model were response caching, you would wrongly conclude that caching is useless for a system whose user turns always differ — which is precisely the workload where it pays the most.

**🗣 Say this in the room:** "Prompt caching stores the KV tensors for a prefix, keyed by a hash of the exact tokens. Because attention is causal, the key is a prefix — any byte change invalidates everything downstream. It saves prefill compute, not generation, so it cuts TTFT and input cost while leaving output tokens billed normally."

### Break down cache-write versus cache-read pricing for me, and derive the break-even.

The pricing is designed so that caching is a bet on reuse, and the bet has an explicit break-even you should be able to compute at a whiteboard. Three rates matter: base input, cache **write** (you paid to store the KV), and cache **read** (you paid to fetch it). On Anthropic's API the multipliers are 1.25× base for a 5-minute-TTL write, 2× base for a 1-hour-TTL write, and **0.1× base for a read** — a 90% discount on every cached token. **📅 Volatile:** multipliers and TTL options move; verify before your loop, and expect other providers to differ (some do implicit caching with no write premium and a smaller read discount).

**📐 Numbers you must know:** with a 1.25× write and 0.1× read, N requests over the same prefix cost `1.25 + 0.1(N−1)` units versus `N` units uncached. Set them equal: `1.25 + 0.1N − 0.1 = N` → `1.15 = 0.9N` → `N ≈ 1.28`. So **the second request already pays for the first** on a 5-minute TTL. For a 1-hour TTL write at 2×: `2 + 0.1(N−1) = N` → `1.9 = 0.9N` → `N ≈ 2.1`, so you need the **third** request inside the hour to break even.

That is the whole decision rule and it is unusually clean: if a prefix will be reused even twice within the TTL, cache it at 5 minutes. Only reach for the 1-hour TTL when your traffic has *gaps longer than five minutes* — a support tool that gets a burst at 09:00 and nothing until 09:20, or a nightly batch. The 1-hour TTL is not "better caching"; it is a longer lease at a doubled deposit.

**💰 Math:** a 12,000-token system prompt at $3/Mtok input costs 12,000 × $3/1,000,000 = **$0.036** per uncached call. A 5-minute write costs 1.25 × $0.036 = $0.045; each read costs 0.1 × $0.036 = **$0.0036**. At 200,000 calls/day with a 95% hit rate: writes 10,000 × $0.045 = $450, reads 190,000 × $0.0036 = $684, total **$1,134/day** versus 200,000 × $0.036 = **$7,200/day** uncached. That is $6,066/day, or **~$182,000/month**, from one field on one content block.

**⚠ Trap:** teams report the saving on the *whole* bill and get caught. Caching touches input tokens only. If your workload is 12k in / 2k out at $3/$15, output is 2,000 × $15/1e6 = $0.030 per call — nearly as much as the entire uncached input. Caching takes a 12k/2k request from $0.066 to $0.0336: a real 49% cut, not the 90% the input-side number implies. Always quote the blended figure.

### Walk me through the canonical worked example — an 8,000-token system prompt, cached and uncached.

This is the arithmetic I expect any applied-AI candidate to do out loud without a calculator, so let me do it in the units that actually appear on an invoice.

Take an 8,000-token system prompt: product instructions, tool schemas, a style guide, and six few-shot examples. At a representative $3/Mtok input price, that prefix costs 8,000 × $3 / 1,000,000 = **$0.024 per call**. Per thousand calls that is **$24**. With a 90% cache-read discount it becomes 8,000 × $0.30/Mtok = **$0.0024 per call**, or **$2.40 per thousand calls** once the prefix is warm — the canonical "~90% reduction." Amortizing the 1.25× write over a thousand calls adds 1.25 × $0.024 = $0.03 total, i.e. $0.00003/call, which rounds to nothing; that is why people quote the read price as if writes were free. **📅 Volatile:** the $3/Mtok anchor and the 0.1× read multiplier both move — re-derive with current numbers before you quote them.

Now scale it the way an interviewer will push you. At 500,000 calls/day: uncached 500,000 × $0.024 = **$12,000/day = $360,000/month**. Cached at a 92% hit rate: 40,000 misses × (1.25 × $0.024) = $1,200, plus 460,000 reads × $0.0024 = $1,104, total **$2,304/day ≈ $69,000/month**. You saved $291,000/month by ordering your prompt correctly.

The latency half matters as much at these companies. Prefill on an 8k prompt at, say, an effective 25,000 prefill tokens/sec of provider-side throughput is 8,000 / 25,000 = **320 ms** of TTFT you delete. For a Cursor-style inline completion with a 300 ms budget, that is not a cost optimization — it is the difference between shipping the feature and not.

**🗣 Say this in the room:** "An 8k-token preamble is $24 per thousand calls at $3/Mtok. Cached reads at 10% take it to $2.40 per thousand, and the 1.25× write amortizes to nothing past the first call. At half a million calls a day that is roughly $360k/month down to $70k/month, and it removes about 300 ms of TTFT. It is the highest-ROI change available without touching the model."

### How many cache breakpoints do I get, where do I put them, and what happens if I put one in the wrong place?

Think of breakpoints as explicit commit points in a write-ahead log: each one says "checkpoint the KV state as of exactly here." Anthropic allows **four per request** (**📅 Volatile**), and the render order is `tools` → `system` → `messages` — so a breakpoint on the last system block checkpoints tools *and* system together, and you have not spent a second breakpoint to do it.

The rule I enforce in review is **one breakpoint per stability tier, placed at the boundary where volatility increases.** A four-tier layout for an agent looks like: (1) tool schemas + core system prompt — changes on deploy; (2) tenant-level policy/persona — changes per customer; (3) retrieved documents or session context — changes per session; (4) the growing conversation — changes per turn. That is exactly four, and each tier's breakpoint is read by every request that shares that tier.

The wrong-place failure is subtle and expensive: **a breakpoint placed after volatile content writes a fresh cache entry on every single request and reads none.** You pay 1.25× forever and get 0% hit rate. This is the number-one caching bug I see. A team wants "maximum caching," so they mark the last block of the whole prompt — which includes the user's unique question — and every request mints a distinct entry. The bill goes *up* 25%.

There is a second gotcha that bites agent loops specifically: the lookback window. A breakpoint searches backward a bounded number of content blocks (20 on Anthropic — **📅 Volatile**) to find a prior cache entry. An agentic turn that emits eleven parallel `tool_use` blocks and eleven `tool_result` blocks has already blown past 20 blocks in one turn, so the next request's breakpoint finds nothing and silently misses despite a byte-identical prefix.

**⚠ Trap:** "silently" is the operative word. There is no error, no warning, no changed field except `cache_read_input_tokens: 0` buried in the usage object. The fix is to place an intermediate breakpoint roughly every 15 blocks inside long tool-heavy turns, and to assert on cache-read tokens in your integration tests rather than trusting that placement is still correct after someone refactors the prompt builder.

### There's a minimum cacheable prefix. Why does that exist, and how does it fail?

It exists because the cache has fixed overhead — a hash, an entry, a lookup, and the bandwidth to move KV tensors from wherever they are stored back into HBM — and below some prefix length that overhead exceeds the prefill you would have saved. So the provider simply declines to create an entry.

The failure mode is that it declines **silently**. You set `cache_control`, the request succeeds, and `cache_creation_input_tokens` comes back as `0` with no error. Nobody notices until someone asks why the cache hit rate is 0% on the classification endpoint.

The minimum is model-dependent and — this is the part people get wrong — **not monotonic across generations**. On Anthropic's current lineup it ranges from 512 tokens on the newest models up to 4,096 on some earlier ones (**📅 Volatile — verify against the current docs before your loop**). A 3,000-token prompt therefore caches on one model and silently does not on another *in the same family*, which means a model swap that looks like a one-line diff can quietly delete your caching.

The practical consequences are two. First, **short prompts are not worth caching anyway** — 1,000 tokens at $3/Mtok is $0.003, so the entire theoretical saving is $0.0027 a call, and you are burning a breakpoint (you only have four) plus review attention on it. Second, if you *want* a short prefix cached, the honest fix is usually to make it longer with content that earns its keep: move few-shot examples or a richer tool description into the prefix rather than padding.

**⚠ Trap:** a team sees the minimum, sizes their prompt at exactly the threshold, and then a prompt edit shaves 40 tokens off. Caching stops entirely, cost jumps ~10×, and the git blame points at a copy-editing PR. Keep a comfortable margin — I want the stable prefix at least 2× the model's minimum — and assert `cache_creation_input_tokens > 0` on the first call in a smoke test.

### Give me the architectural rule that follows from prefix matching, and show me how you'd structure a prompt builder around it.

The rule is one sentence: **order every prompt by descending stability, and never let a volatile byte appear before a stable one.** Static system prompt, then tool schemas, then few-shot examples, then tenant configuration, then retrieved documents, then conversation history, then the user's current turn. Volatility increases monotonically from left to right. That is not a style preference; it is the only layout under which prefix matching can work at all.

Backend translation: this is the same discipline as designing a composite index. `(tenant_id, created_at)` serves range scans; `(created_at, tenant_id)` does not, and no amount of hinting fixes it. Column order in a composite index and content order in a prompt are the same constraint — leftmost-prefix matching — and the same class of engineer gets both wrong for the same reason.

The implementation consequence is that **prompt assembly must be a single typed function, not string concatenation scattered across the codebase.** Here is the shape I insist on:

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class PromptLayers:
    tools: tuple            # tier 0 — deploy-scoped, sorted deterministically
    system: str             # tier 0
    tenant_policy: str      # tier 1 — customer-scoped
    session_context: str    # tier 2 — session-scoped
    turns: tuple = ()       # tier 3 — per-turn

def render(p: PromptLayers) -> dict:
    system = [
        {"type": "text", "text": p.system},
        {"type": "text", "text": p.tenant_policy,
         "cache_control": {"type": "ephemeral"}},   # breakpoint: tools+system+policy
    ]
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": p.session_context,
             "cache_control": {"type": "ephemeral"}},   # breakpoint: + session
        ]},
        *p.turns,
    ]
    return {"tools": sorted(p.tools, key=lambda t: t["name"]), 
            "system": system, "messages": messages}
```

Two details carry the weight. `sorted(..., key=name)` makes tool order deterministic — a `set` or a dict-ordering change would otherwise reshuffle position 0 of the entire prompt. And `frozen=True` means nobody mutates a layer after construction, which is how "current timestamp" ends up in a system prompt in the first place.

**🗣 Say this in the room:** "I make prompt assembly one function that takes stability tiers as separate typed fields and renders them in a fixed order, with breakpoints at tier boundaries. It's the same reasoning as composite index column order — leftmost-prefix matching means the order *is* the design. Ad-hoc f-strings scattered through the codebase is how a `datetime.now()` ends up at position zero."

### `cache_read_input_tokens` is zero across thousands of identical-looking requests. Walk me through the debug.

**🔍 Failure taxonomy** — run this as a decision procedure, top to bottom, because the cheap checks eliminate most of the field.

**Step 0 — confirm you're measuring the right thing.** `input_tokens` is the *uncached remainder only*. Total prompt size is `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. If `cache_creation` is also zero, you are below the minimum cacheable prefix or the breakpoint isn't reaching the API at all. If `cache_creation` is nonzero on *every* request, you are writing and never reading — jump to step 3.

**Step 1 — diff the rendered bytes.** Not the template, the *rendered prompt*, on two consecutive requests, serialized exactly as it goes on the wire. Ninety percent of cases die here, and the culprit is always one of:

| Invalidator | Why it kills the prefix |
|---|---|
| `datetime.now()` / "Today is …" in the system prompt | Position ~20 changes every request |
| A request ID, trace ID, or `uuid4()` interpolated early | Same, and every request is globally unique |
| `json.dumps(d)` without `sort_keys=True`, or iterating a `set` | Serialization order is nondeterministic across processes |
| Tool list built per-user (`build_tools(user)`) | Tools render at position 0; nothing shares across users |
| Conditional system sections (`if flag: system += ...`) | Each flag combination is a distinct prefix; N flags → 2^N caches |
| A retrieved document inserted *before* the preamble | The most common "we refactored RAG" regression |
| Trailing-whitespace or newline differences from a template engine | A one-byte diff is a total miss |

**Step 2 — check the model string.** Caches are model-scoped. A canary rolling 5% of traffic to a new model version is not "5% uncached"; it is two independent cache populations, each with lower reuse, and if the canary shares a load balancer with the stable pool the effective hit rate on *both* drops. This is also why an unpinned model alias is a caching liability, not just a reproducibility one.

**Step 3 — check breakpoint placement.** Writes with no reads means the breakpoint sits *after* volatile content. Move it earlier.

**Step 4 — check the lookback window and TTL.** If your agent emits more than ~20 content blocks per turn, the breakpoint's backward search never finds the prior entry. If your traffic is bursty with gaps over 5 minutes, entries expire between bursts — either pre-warm or switch to the 1-hour TTL and recompute the break-even (you now need three reads, not two).

**Step 5 — check for fan-out timing.** A cache entry only becomes readable once the first response *begins streaming*. N parallel requests fired simultaneously with the same prefix all miss, because none can read what the others are still writing. Fire one, await first token, then fan out the rest.

### Your team reorders the tool schemas in a config file and the bill triples overnight. Explain what happened and how you prevent it.

Tools render at **position 0** of the prompt, ahead of `system` and ahead of every message. Reordering them changes byte 0. Every downstream cache entry — the system prompt, the tenant policy, the entire conversation history for every in-flight session — is invalidated simultaneously, across every tenant, at once. You went from a ~95% hit rate to 0%, and because misses now pay the 1.25× *write* premium on re-population, your input cost is momentarily 1.25× the uncached baseline, i.e. about 12.5× the cached steady state. Triple is a conservative report.

The same blast radius applies to **adding or removing a single tool**, which is why "just add one tool for the new feature" is a cost incident, not a feature flag. It is also why per-user or per-plan tool sets are an anti-pattern: if enterprise customers get three extra tools, you have partitioned your cache by plan tier and destroyed cross-user sharing of the most expensive shared prefix you own.

Prevention, in the order I'd implement it:

1. **Sort tools deterministically at render time** — `sorted(tools, key=lambda t: t["name"])` — so config-file order is irrelevant and a merge conflict resolution can't reshuffle it.
2. **Serialize schemas canonically** — `json.dumps(schema, sort_keys=True, separators=(",", ":"))`. A Pydantic version bump that changes key emission order is otherwise a silent invalidator.
3. **Hash the rendered prefix in CI** and fail the build on an unexpected change. This turns a silent cost regression into a review conversation: "this PR invalidates the global prefix cache, here is the one-off repopulation cost, is it worth it?"
4. **Never vary the tool set per request.** If you need dynamic capability, use the mechanisms designed for it: deferred tool loading / tool search appends schemas rather than reordering the base set, and on models that support mid-conversation tool changes you send an addition or removal as a system message *after* the cached prefix instead of editing `tools`.

**⚠ Trap:** the team's instinct after an incident like this is to add more breakpoints. Breakpoints do not help — you cannot checkpoint *before* position 0. The only fix is determinism at the source.

**💰 Math:** at 200,000 calls/day with a 12k-token prefix at $3/Mtok, one day of 0% hit rate costs 200,000 × 1.25 × $0.036 = **$9,000** versus the cached steady state of ~$1,100. A single unsorted dict cost you **$7,900 in one day**, and it will not appear on any dashboard you currently have unless you are alerting on cache hit rate.

### What happens to your prompt cache when you change the model version, and how do you plan a migration around it?

Caches are keyed by model as well as by tokens, so a model change is a total flush by construction — the stored KV tensors were produced by different weights and are meaningless to the new model. There is no partial migration and no warm handoff. Plan for it as a **cold-start event with a known, computable cost**, exactly as you'd plan a Postgres index rebuild or a Redis flush.

The cost is: (number of distinct prefixes you need warm) × (prefix length) × (write multiplier) × (input price). If you serve 4,000 tenants each with a distinct 6,000-token tenant-policy layer, repopulating is 4,000 × 6,000 × 1.25 × $3/1e6 = **$90**. Trivial. If instead every *session* carries a distinct 60,000-token retrieved-document layer and you have 50,000 concurrent sessions, it is 50,000 × 60,000 × 1.25 × $3/1e6 = **$11,250** in one burst, plus the latency spike from every one of those sessions paying full prefill simultaneously. Which of those two you are is determined entirely by your prefix design, which is why I care about it at design time and not at migration time.

The migration playbook I use:

- **Pin model IDs explicitly.** An alias that silently rolls forward turns a vendor's release schedule into an unannounced cache flush on your production traffic. Pin, then upgrade deliberately.
- **Run the canary on a separate deployment**, not interleaved into the main pool. Interleaving gives you two half-warm caches instead of one warm and one cold, and it makes the A/B latency comparison meaningless because the control arm is also degraded.
- **Pre-warm before the cutover.** For the stable global prefix — tools plus system prompt — fire a warm-up request against the new model at deploy time. Anthropic supports `max_tokens: 0` for exactly this: the API runs prefill, writes the cache at your breakpoint, and returns immediately with empty content and zero billed output tokens (**📅 Volatile** — confirm the parameter is still supported, and note it is rejected alongside streaming, structured output, and forced tool choice).
- **Expect a TTFT regression window** the width of your TTL. Tell whoever owns the SLO dashboard before, not after.

**🗣 Say this in the room:** "Model version is part of the cache key, so a version bump is a full flush. I price the repopulation up front — distinct prefixes × prefix length × write multiplier × input rate — pin model IDs so it never happens by surprise, pre-warm the global prefix with a zero-output request at deploy, and warn the SLO owner about a TTFT bump for one TTL window."

### How does prompt caching work across a multi-turn conversation, and what does the cost curve actually look like?

Multi-turn is where caching goes from a nice saving to the thing that makes the product economically possible, because the alternative is quadratic.

Without caching, turn `n` re-prefills the entire history: turn 1 costs `L` tokens, turn 2 costs `2L`, turn `n` costs `nL`. Total across `N` turns is `L·N(N+1)/2` — **quadratic in conversation length**. With caching and a breakpoint on the last block of the most recent turn, turn `n` reads `(n−1)L` tokens at 0.1× and writes `L` new tokens at 1.25×, so the incremental cost per turn is roughly constant plus a slowly-growing read term. Total becomes `1.25LN + 0.1L·N(N+1)/2` — still quadratic, but with the quadratic term scaled by **0.1**, which is a 10× reduction on the dominant term.

**💰 Math:** a 20-turn conversation with 1,500 tokens per turn at $3/Mtok. Uncached: 1,500 × 20 × 21 / 2 = 315,000 tokens = **$0.945** per conversation. Cached: writes 1.25 × 1,500 × 20 = 37,500 effective tokens ($0.1125) plus reads 0.1 × 315,000 − 0.1 × 30,000 ≈ 28,500 effective tokens ($0.0855), total ≈ **$0.198**. That is a 4.8× cut, and at 100,000 conversations/day it is $94,500/day versus $19,800/day — **$2.24M/year**.

The placement rule for multi-turn: **put the breakpoint on the last content block of the most recently appended turn**, and move it forward each turn. Earlier breakpoints remain valid read points, so hits accumulate incrementally rather than requiring you to re-mark the whole history.

**⚠ Trap:** "the conversation grew past the context window, so we compact it" — and compaction rewrites history, which invalidates everything from the compaction point forward. That is correct and unavoidable, but you should *budget* for it: a compaction event is a full re-prefill of the compacted prefix, and if you compact every 30 turns you are paying a cold prefill every 30 turns. Compact less often with a larger window rather than more often with a small one, and place the compaction boundary so that the summary itself becomes the new stable prefix.

**⚠ Trap, sharper:** a fork operation — spawning a summarizer, a sub-agent, or a background classifier off the same conversation — that rebuilds `system` or `tools` even slightly differently misses the parent's cache entirely and pays full prefill on the whole history. Copy the parent's `system`, `tools`, and `model` verbatim, then append fork-specific content at the end. I have seen this single mistake double an agent's bill.

### How do extended-thinking / reasoning tokens interact with prompt caching?

Three interactions, and getting them wrong is expensive because thinking tokens are billed as output at output rates.

**First: enabling or disabling thinking invalidates the system and tools cache tiers but not messages** — on Anthropic's invalidation hierarchy, toggling thinking sits at the same level as `tool_choice` and images: it preserves the tools+system cache and invalidates from messages onward (**📅 Volatile** — the exact hierarchy is provider-specific and has changed). The practical rule is that you should not be flipping thinking on and off per request within a conversation anyway; pick a mode per route.

**Second, and this is the one people miss: thinking blocks become part of the prompt on the next turn.** In a multi-turn or agentic loop you echo the assistant's content — including its thinking blocks — back into `messages`. Those tokens were billed as output when generated, and they are now billed as *input* on every subsequent turn. A reasoning-heavy agent that emits 4,000 thinking tokens per turn is adding 4,000 tokens to the cached prefix each turn. With caching that costs 0.1× and is fine; **without** caching it is 4,000 tokens × $3/Mtok × (remaining turns) of pure re-prefill, and it compounds quadratically like any other history growth.

**Third: thinking is where context editing and caching collide.** Clearing thinking blocks from history to save context is a real technique, but each clear rewrites the prefix from the clear point forward and drops that cache. The rule I use: **clear on a boundary you were going to invalidate anyway** — at a compaction point, at a task boundary — never opportunistically mid-task.

**💰 Math:** an agent doing 15 tool-calling turns, 3,000 thinking tokens per turn, 8k static prefix. History at turn 15 is 8,000 + 15 × (3,000 thinking + 600 tool result) ≈ 62,000 tokens. Uncached re-prefill of that turn alone is 62,000 × $3/1e6 = **$0.186**; cached read is **$0.0186**. Across the 15-turn trajectory the difference is roughly $1.40 versus $0.14 per trajectory. At 40,000 trajectories/day: **$56,000/day versus $5,600/day**. Reasoning agents without prefix caching are not a slightly-worse design; they are an unshippable one.

**🗣 Say this in the room:** "Thinking tokens are billed as output once and then as input on every subsequent turn, so in an agent loop they're the fastest-growing part of your cached prefix. Caching turns that from a quadratic output-priced blowup into a 0.1×-priced read. I never clear thinking blocks opportunistically — only at a boundary where I'm invalidating the prefix anyway."

### Design the caching layout for an agent that makes 20 tool calls per task. Where do the breakpoints go?

Let me lay out the actual prompt shape first, because the answer falls out of it. An agent turn is: `tools` (fixed) → `system` (fixed) → user task → [assistant with `tool_use` blocks → user with `tool_result` blocks] × 20 → final answer. The prefix grows monotonically and every element of it is append-only, which is the ideal shape for prefix caching — nothing ever changes retroactively.

**Breakpoint 1** goes on the last system block, checkpointing tools + system. This is shared across every task and every tenant and is the single highest-value entry in the system. **Breakpoint 2** goes on the task specification — the user's initial request plus any retrieved context — because every one of the 20 iterations within this task shares it. **Breakpoints 3 and 4** are the interesting ones: they *slide forward* through the tool-call history as the trajectory grows, and this is where the 20-block lookback constraint bites.

The mechanic: each iteration appends an assistant message (possibly several `tool_use` blocks) and a user message (one `tool_result` per call). Parallel tool use makes this worse — five parallel calls is ten blocks in one iteration. If more than ~20 blocks accumulate between your last breakpoint and the new one, the backward search fails and you re-prefill the whole trajectory. So I re-place a sliding breakpoint roughly every 12–15 content blocks, keeping breakpoints 3 and 4 as a two-position window that advances: on each API call, put one on the last block of the previous turn and one on the last block of the current turn.

```python
def place_sliding_breakpoints(messages, stride=12):
    """Mark the last content block of every `stride`-th block position."""
    blocks = [(mi, bi) for mi, m in enumerate(messages)
                       for bi, _ in enumerate(m["content"])]
    for mi, bi in blocks[stride-1::stride][-2:]:      # keep only the last two
        messages[mi]["content"][bi]["cache_control"] = {"type": "ephemeral"}
```

**⚠ Trap:** a tool that returns a large, non-deterministic result — a timestamped log tail, a search API that reorders equally-ranked results, a `list_files` that doesn't sort — turns every subsequent iteration into a cache miss. The tool result is *inside* the prefix. I require every tool in an agent harness to return canonically-ordered, timestamp-free output for exactly this reason, and I treat "sort your tool output" as a caching requirement, not a cosmetic one.

**💰 Math:** 20 iterations, prefix reaching 40,000 tokens by the end, average prefix across iterations ≈ 24,000 tokens. Uncached: 20 × 24,000 × $3/1e6 = **$1.44/task**. Cached at 0.1× with writes on the delta: roughly $0.17/task. At 15,000 tasks/day that is $21,600/day versus $2,550/day — **$5.7M/year**. This is why "we'll add caching later" is not a defensible position for an agent product.

### Prefix caching cuts cost. What does it do to latency, and how do you quantify that separately?

Cost and latency are two different wins from the same mechanism and you should present them separately, because in an interview for Cursor or Perplexity the latency half is what they actually care about.

The latency saving is **prefill time on the cached span, and only that**. Prefill is compute-bound: it processes all prompt tokens in parallel through the network with arithmetic intensity high enough to saturate tensor cores. Its cost is roughly `2 · P · T_prompt` FLOPs for a dense model with `P` parameters and `T_prompt` prompt tokens. Reading a cache instead replaces that compute with a memory transfer of the KV tensors, which is dramatically cheaper on a decode-bound serving stack.

**📐 Numbers you must know:** for a 70B dense model, prefill of a 10,000-token prompt is 2 × 70×10⁹ × 10,000 = **1.4×10¹⁵ FLOPs**. On an H100 achieving ~400 effective TFLOP/s for prefill, that is 1.4e15 / 4e14 = **3.5 seconds** of pure GPU time. Even amortized across a batch, a cached read that avoids it removes hundreds of milliseconds from TTFT. Derive it this way in the room rather than quoting "caching makes it faster."

What it does **not** improve: inter-token latency, output token cost, or total generation time. If your p95 is dominated by generating 800 output tokens at 40 tok/s (20 seconds), removing 300 ms of prefill moves the needle by 1.5%. The decision rule: **prefix caching is a TTFT lever, and it only matters if TTFT is a meaningful fraction of your latency budget.** For a streaming chat answer with a long generation, it's marginal. For an autocomplete, a classifier, a router, a rerank call, or a tool-selection step — where output is 5–50 tokens and prefill is 10,000 — it is essentially the entire latency.

**⚠ Trap:** teams measure the caching win in aggregate p95 across all endpoints and conclude it "didn't help much," because the long-generation endpoints dominate the aggregate. Measure TTFT specifically, segmented by route, and segmented by hit/miss. The right dashboard is a two-line chart: p95 TTFT on cache hits and p95 TTFT on cache misses, on the same axis. If those lines are not visibly separated, your caching is not working, regardless of what the hit-rate counter says.

**🗣 Say this in the room:** "Caching removes prefill, so it moves TTFT and input cost — nothing else. On a 70B, 10k tokens of prefill is 1.4e15 FLOPs, about 3.5 GPU-seconds on an H100. If my route generates 800 tokens the win is noise; if it generates 20 tokens after a 10k-token prompt, prefill *is* the latency and caching is the whole optimization."
