### Token prices have fallen roughly 80% year over year and enterprise LLM spend has roughly doubled over the same period. Reconcile those.

Mental model: **this is Jevons' paradox with a two-week feedback loop.** When the unit cost of a capability collapses, the set of economically viable applications expands faster than the price falls, so aggregate consumption — and aggregate spend — rises. You have already lived this: cloud storage got 10× cheaper per GB and your storage bill went up, because at the new price you started keeping things you used to delete.

The mechanism specific to LLMs has three compounding legs, and naming all three is what separates an economics-flavored answer from a real one:

**Leg 1 — the capability floor moves down through the price tiers.** A task that only the frontier tier could do last year is done by the cheap tier this year. That's a per-request price cut. But it also means the task is now *worth automating at all* — a workflow that cost $0.40/item and saved $0.35 of human time was a non-starter; at $0.04/item it's a 9× ROI and gets built. **Price cuts don't just reduce the bill for existing work; they convert non-customers into customers.**

**Leg 2 — the architecture changed from one call to many.** The single biggest driver of enterprise spend growth is not traffic, it's **tokens per user-visible task**. A 2023 chat feature was one call, ~1,500 tokens. A 2026 agentic feature on the same user request is 15–40 calls with a growing transcript, tool results, and reasoning tokens — routinely 200k–2M tokens for one task. That's a 100–1000× increase in tokens per task, against an 80% price cut. Do the arithmetic: 0.2 × 300 = **60× more spend per task**, and this is the whole story.

**Leg 3 — reasoning tokens are a new line item that didn't exist.** They bill at the output rate, they're invisible in the response, and effort settings that improve quality multiply them.

**📐 Numbers you must know:** the orders of magnitude, derived rather than quoted. A 2023 single-call chat feature: **~1.5k tokens per user-visible task**. A 2026 agentic feature: 15–40 calls × a transcript that grows to tens of thousands of tokens plus reasoning = **200k–2M tokens per task**, i.e. **10²–10³×**. Against a roughly 5× per-token price cut over the same window, that is a net 20–200× increase in spend per task. Whenever someone tells you inference "got cheap," those two exponents are the counter-argument.

**💰 Math, so the shape is undeniable.** Feature X, 2024: 500k tasks/month × 1 call × (2,000 in + 400 out) at $10/$30 per Mtok = 500,000 × ($0.020 + $0.012) = **$16,000/month**. Same feature, 2026, rebuilt as an agent: 500k tasks × 18 calls × (average 9,000 cached-in + 700 out) at $3/$15 with a 90% cache-read rate. Input: 18 × 9,000 = 162,000 tokens/task; blended input rate ≈ 0.9 × $0.30 + 0.1 × $3.75 = $0.645/Mtok → 162,000 × $0.645/1e6 = $0.1045. Output: 18 × 700 = 12,600 × $15/1e6 = $0.189. Per task ≈ $0.294 → **$147,000/month**. The unit price fell ~70%, the quality is far higher, and the bill went up **9×**.

**What this means for you as an engineer, which is the part interviewers actually want:**

1. **Never forecast next year's budget by extrapolating this year's tokens at next year's prices.** You will be wrong by an order of magnitude in the expensive direction, because architecture changes dominate price changes.
2. **The unit that matters is cost per resolved task**, and it must be tracked as a first-class metric with a target, because it's the only quantity that stays comparable across an architecture change.
3. **Price cuts should be spent deliberately.** When a tier gets 60% cheaper you have a choice: bank it, or buy more quality (higher effort, more retrieval, a verification pass). Make that a decision with a number attached, not something that happens to you.

**🗣 Say this in the room:** "Prices per token fell ~80%, tokens per task rose two to three orders of magnitude because we went from one call to an agent loop with reasoning. Net, spend roughly doubled. So I don't track cost per token or cost per call — I track cost per resolved task, with a target, because that's the only unit that survives an architecture change and it's the one the business can price against."

### Your LLM bill tripled month over month. Traffic is flat. Debug it.

**🔍 Failure taxonomy, ordered by frequency, each with the diagnostic that confirms or eliminates it in one query.**

**1. Prefix cache stopped hitting.** Diagnostic: `sum(cache_read) / sum(cache_read + cache_write + input_uncached)` over time. If it fell, run the cache-debug procedure — hash the rendered prefix, look for a moved model alias, a dynamic tool set, a config-injected value, or traffic getting sparse enough to fall outside the TTL. This is the single most common cause and it can 4× your bill with zero code change.

**2. Output tokens per request grew.** Diagnostic: `mean(output_tokens)` by route, over time. Causes: a default flip that turned thinking on; an effort setting raised; a prompt change that removed a length constraint; a model upgrade that's simply more verbose. Output bills at 5× input, so a 2× output increase is often the entire delta.

**3. Turn count per task grew.** Diagnostic: `mean(api_calls_per_task)` — which requires you to have a task/trace ID on every call. If you don't, add it now; without it you cannot distinguish "each call got more expensive" from "we make more calls," and those have opposite fixes. Causes: a tool that started failing and triggering retries; a new tool that confuses selection; a loop cap raised; results split across messages killing parallel tool calls.

**4. Input length grew.** Diagnostic: `p50/p95(total_prompt_tokens)`, remembering that total = uncached + cache_read + cache_write, not `input_tokens` alone. Causes: retrieval `k` raised; a chunker change; conversation history not being trimmed; a document type that started arriving larger.

**5. Traffic mix shifted.** "Flat traffic" usually means flat request count. Diagnostic: spend by route and by tenant. One enterprise customer onboarding onto the expensive path can triple the bill while total requests move 3%.

**6. Retry / failover amplification.** Diagnostic: attempts per task, and spend attributable to non-first attempts. A provider degradation last month plus an uncapped retry ladder is a classic tripling.

**7. Someone changed the model.** Diagnostic: spend by the model ID *the API returned*. A route silently repointed from a $1/$5 tier to a $5/$25 tier is a 5× on that route.

**8. A batch job moved onto the sync path**, or a sync job lost its batch discount.

**The instrumentation that makes this a 20-minute investigation instead of a two-day one** — and the correct answer to "how would you have caught this sooner":

```python
# One structured log line per API call. Non-negotiable fields.
log.info("llm_call", extra={
    "trace_id": trace_id, "task_id": task_id, "route": route, "tenant_id": tenant,
    "model_requested": req_model, "model_served": resp.model,   # these can differ
    "prompt_version": prompt_hash, "attempt": attempt,
    "in_uncached": u.input_tokens, "cache_read": u.cache_read_input_tokens,
    "cache_write": u.cache_creation_input_tokens, "out": u.output_tokens,
    "stop_reason": resp.stop_reason, "cost_usd": cost(u, rates),
})
```

With that, every hypothesis above is a `GROUP BY` away. Without it, you are reading an invoice.

**⚠ Trap:** starting from the invoice. The provider bill is aggregated by model and day; it cannot tell you which route, tenant, or prompt version moved. **Cost must be attributed at the call site, in your own telemetry, at emit time** — exactly as you'd never debug a database cost problem from the RDS bill alone.

**🗣 Say this in the room:** "First question: did cost per call go up, or calls per task? Those have completely different fixes. I'd check cache hit rate, mean output tokens, and calls-per-task in that order — those three explain the large majority of surprise tripling, and all three are single queries if you log the usage breakdown with a trace ID at the call site."

### Design per-tenant cost attribution and budget enforcement for a multi-tenant AI product.

Mental model: **this is a metering and quota system, and you have built one before — the only genuinely new part is that the unit of consumption is not knowable until after the work is done.** You can't price the request at admission, because the model decides how many output and reasoning tokens to spend. So the standard "reserve, consume, settle" pattern applies, with a pessimistic reservation and a settlement.

**Layer 1 — attribution.** Every API call carries `tenant_id`, `user_id`, `feature`, `task_id`, `attempt`. Cost is computed at the call site from the response's `usage` object and the rate card for the exact model served. Emit as a structured event; aggregate in your warehouse. Two rules:

- **Attribute at the task level, not the call level.** A tenant consumed one "resolved ticket," which cost you 22 API calls. The task is what you bill or budget against; the calls are implementation detail.
- **Attribute failures and retries too.** A tenant whose data causes 40% tool failures costs you 1.6× the tokens for the same delivered value, and if your attribution only counts successes you will misprice them permanently.

**Layer 2 — enforcement.** A distributed token bucket per tenant, in Redis, keyed on **dollars or tokens per period**, not on requests. The reserve/settle shape:

```python
# Admission: reserve a pessimistic upper bound.
est = est_input_tokens * rates.input_uncached + max_tokens * rates.output
if not bucket.try_reserve(tenant, est):          # atomic Lua: check + debit
    raise BudgetExceeded(tenant)
try:
    resp = client.messages.create(..., max_tokens=max_tokens)
finally:
    actual = request_cost_usd(resp.usage, rates)
    bucket.settle(tenant, reserved=est, actual=actual)   # refund the difference
```

The reservation must be pessimistic (use `max_tokens`, not your guess) or a tenant can overrun by the ratio of actual to estimated. This is exactly the provider's own admission-control mechanism, and building it yourself for the same reason is a good thing to point out.

**Layer 3 — degradation, not rejection.** A hard 429 at the budget line is a bad product. The ladder: at 80% of budget, route to a cheaper tier and lower effort; at 95%, disable the expensive optional features (deep research, multi-agent fan-out); at 100%, queue non-interactive work and serve interactive work with a clear notice. Each step needs a measured quality delta so you know what you're spending in accuracy to save in dollars.

**Layer 4 — the controls that stop the bleeding.** Per-tenant caps on: `max_tokens` per call, turns per task, tool calls per task, retries per task, and concurrent tasks. Every one of those is a runaway-cost vector; a turn cap alone prevents the worst incident class.

**Layer 5 — reconciliation.** Nightly, sum your per-call attributed cost and compare to the provider's reported usage. Alert if they diverge by more than ~2%. Divergence means dropped telemetry, a rate card you didn't update, or traffic from a path you don't instrument — all three of which you want to know about before the invoice.

**💰 Math on why enforcement is not optional:** an agent with no turn cap on a task the tool set can't complete. 40 turns × 8,000 output tokens... realistically it loops: 200 turns × 800 output tokens = 160,000 output tokens at $15/Mtok = **$2.40 for one stuck task**. One tenant with a malformed integration generating 3,000 such tasks a day = **$7,200/day**. A turn cap of 25 bounds it at $0.30/task and $900/day, and a duplicate-call detector cuts it further. The cap is four lines of code.

**⚠ Trap:** building attribution on estimated tokens because "we need the number before the call." You need an *estimate* before the call for reservation, and the *actual* after for accounting. Conflating them gives you a ledger that drifts from the invoice and a finance team that stops trusting your dashboards.

### MCP has gone through several spec revisions. What changed, and why does the revision date matter operationally?

Mental model: **MCP is a wire protocol with dated revisions, and "we support MCP" is as meaningless a statement as "we support HTTP" without a version.** The revision string is the compatibility contract between your client and every server you connect to, and it has changed in ways that are not backwards-compatible.

**The revision history and what each changed (📅 Volatile — verify the current revision and any newer ones before your loop):**

- **2024-11-05** — the initial specification. JSON-RPC 2.0 over stdio or HTTP+SSE (two endpoints: one SSE stream for server→client, one POST endpoint for client→server). Core primitives: tools, resources, prompts.
- **2025-03-26** — the significant one. Replaced the two-endpoint HTTP+SSE transport with **Streamable HTTP** (a single endpoint that can optionally upgrade to SSE), which made MCP servers deployable behind ordinary infrastructure — stateless, load-balanced, serverless — rather than requiring a sticky long-lived connection. Added a comprehensive **OAuth 2.1-based authorization framework**, tool annotations, and audio content.
- **2025-06-18** — **removed JSON-RPC batching** (a genuine breaking change; clients that batched must stop), added **structured tool output**, **elicitation** (servers can request additional input from the user mid-call), and resource links. Formalized MCP servers as OAuth **resource servers** and required clients to send resource indicators so a token issued for one server can't be replayed at another. Made the `MCP-Protocol-Version` header required on HTTP transport.

**Why the date matters operationally, which is the actual question:**

1. **Transport compatibility.** A client that only speaks the original HTTP+SSE transport cannot talk to a Streamable-HTTP-only server, and vice versa. This is the number-one integration failure and it presents as a connection that hangs rather than an error.
2. **Auth model.** Pre-2025-03-26 there was no standardized authorization story, so servers rolled their own. Post-2025-06-18 there's a resource-indicator requirement designed specifically to prevent token-replay across servers. If you're integrating a third-party MCP server, "which revision do you implement" determines whether your security review passes.
3. **Breaking removals.** Batching removal means a client written against the earlier spec fails against a newer server, silently or loudly depending on implementation quality.
4. **Header negotiation.** The required version header on HTTP transport is how mismatches surface as a clean error instead of a mysterious hang — which is exactly why you should send it and reject servers that don't.

**The security posture you should raise unprompted**, because it's the thing product companies care about: an MCP server is **remote code you are granting tool access to your agent**. The risks are (a) a malicious or compromised server returning tool results crafted as prompt injections that redirect the agent, (b) token scope over-grant, (c) tool-name shadowing where a hostile server registers a tool name your agent trusts, and (d) supply chain — you're pulling a server someone else maintains. Mitigations: pin server versions, allowlist servers, keep credentials out of the sandbox (inject them at egress rather than handing them to the model's environment), treat every tool result as untrusted input, and require human confirmation for irreversible actions.

**🗣 Say this in the room:** "I'd never say 'we support MCP' without a revision date — the transport changed from dual-endpoint SSE to Streamable HTTP, batching was removed, and the authorization model went from unspecified to OAuth 2.1 with resource indicators. Those are hard compatibility boundaries. And operationally I treat every MCP server as untrusted remote code: allowlisted, version-pinned, credentials injected at egress rather than into the sandbox, and tool results treated as adversarial input."

### The EU AI Act — what actually applies to you as an applied AI engineer, and on what timeline?

Mental model: **the Act is risk-tiered and role-based. Your obligations are determined by two questions: what risk tier is your use case, and are you a *provider* or a *deployer* of the system?** Most product engineers are deployers of a general-purpose model inside a specific application, and that combination usually lands them in the transparency tier rather than the high-risk tier — but the exceptions are the ones that end careers.

**The tiers:**

- **Prohibited** — social scoring, certain biometric categorization, emotion inference in workplaces and education, untargeted facial-image scraping, some manipulative techniques. Not "regulated," *banned*.
- **High-risk** — Annex III use cases: employment and worker management (CV screening!), education access, essential private and public services including credit scoring, law enforcement, migration, justice, and critical infrastructure; plus AI as a safety component of already-regulated products. Obligations are heavy: risk management system, data governance, technical documentation, logging, human oversight, accuracy/robustness/cybersecurity, conformity assessment, registration.
- **Limited risk / transparency** — disclose that a user is interacting with an AI; label synthetic content and deepfakes.
- **Minimal risk** — most things. No specific obligations.
- **GPAI models** — a separate track with obligations on the model provider (technical documentation, copyright policy, training-content summary), plus additional systemic-risk obligations above a compute threshold.

**The milestone calendar (📅 Volatile — and genuinely contested; there has been active legislative discussion about delaying parts of the high-risk timeline. Verify the current state before your loop; do not assert a date you haven't checked this month):**

| Date | What applies |
|---|---|
| 1 Aug 2024 | Act enters into force |
| 2 Feb 2025 | Prohibited practices; AI-literacy obligations |
| 2 Aug 2025 | GPAI model obligations; governance structures; most penalty provisions |
| **2 Aug 2026** | **General applicability — including Annex III high-risk obligations and the Article 50 transparency obligations** |
| 2 Aug 2027 | High-risk as a safety component of regulated products; GPAI models placed on market before Aug 2025 must be brought into compliance |

Penalties scale to a percentage of global annual turnover, with the highest band for prohibited practices — which is why this gets executive attention rather than being a docs-page problem.

**What this means for your engineering work, concretely — this is the part that distinguishes a useful answer from a Wikipedia recital:**

1. **Classify the use case in the design doc, not at launch.** "Are we in Annex III?" is a five-minute question at design time and a three-month remediation at launch. Résumé screening, credit decisioning, and anything gating access to a service are the ones that surprise people.
2. **Transparency obligations are engineering work.** Disclosing AI interaction and marking synthetic content mean UI changes and, for generated media, machine-readable provenance marking. Budget for it.
3. **Logging and traceability are Act requirements for high-risk, and they're the same artifacts you want anyway** — trace IDs, model version, prompt version, inputs, outputs, and human-override records, retained. You are already building this for cost attribution and eval; scope it once.
4. **Human oversight must be real.** A human who rubber-stamps 400 decisions an hour is not oversight. It has to be a person with the information, the authority, and the time to override.
5. **The GPAI obligations mostly land on your model provider, and you inherit their documentation** — which is why "which provider" becomes a compliance question as well as a technical one.

**🗣 Say this in the room:** "As a deployer of a general-purpose model in a product, I'm usually in the transparency tier — disclose AI interaction, mark synthetic content. What I check first, at design time, is whether the use case falls in Annex III, because employment screening and credit decisioning are high-risk and that's a completely different obligation set: risk management, data governance, conformity assessment, real human oversight. The dates have been in motion legislatively, so I'd verify the current calendar rather than quote one from memory."

### A model you depend on is being retired in 60 days. Run the migration.

Mental model: **treat it as a dependency upgrade with a hard deadline and an unknown-magnitude behavioral change — closer to a database major-version upgrade than a library bump.** The specific hazard is that the API contract may be identical while the *behavior* is not, so your type checker and your integration tests both pass and your quality moves.

**Week 1 — inventory and classify.** Grep for the model ID across the repo and classify every hit, because the right action differs:

| Hit type | Action |
|---|---|
| Actual API call sites | Swap the ID **and** apply the breaking-change checklist |
| Model registries, routing configs, pricing catalogs, OpenAPI specs | The old entry may need to *stay* (the model is still served elsewhere). Add the new one alongside; never blind-replace |
| Capability gates (`if "opus-4" in model_id:`) | **Add** the new ID; don't replace, or you disable a feature for remaining old-model traffic |
| Test fixtures, seed data, registry assertions | Add alongside; verify the definer has an entry for the new model first |
| Suffixed variants (`-fast`, `-1024k`, dated snapshots) | These are deployment identifiers. Verify a new-model equivalent exists before assuming |

This classification step is the one people skip, and it's the one that causes "we migrated and prod broke in an unrelated service."

**Week 1–2 — read the migration guide and enumerate the breaking changes.** These are real and they 400: thinking-configuration parameters removed, sampling parameters (`temperature`/`top_p`/`top_k`) rejected on some newer models, assistant-turn prefills rejected, tool type/name pairs updated, parameter renames like `output_format` → `output_config.format`. Also enumerate the *silent* changes: default flips on thinking, tokenizer changes shifting token counts 1×–1.35%, changed defaults on reasoning visibility.

**Week 2–3 — re-baseline the numbers.** Re-run `count_tokens` against the new model on representative prompts. A tokenizer change means your `max_tokens` ceilings, context budgets, compaction triggers, and cost model are all off. **Do not apply a blanket multiplier**; measure.

**Week 3–4 — run the eval, both models, side by side.** Same eval set, same prompts. Report accuracy with confidence intervals, mean output tokens, p95 latency, and cost per resolved task. This is the artifact that says whether you can ship.

**Week 4–6 — adapt prompts.** Expect this to be needed and budget for it. Recent-generation models follow instructions more literally, calibrate verbosity to task complexity, and differ in tool-use eagerness and self-verification behavior. Concretely: prompts written to *overcome* an older model's reluctance ("CRITICAL: you MUST use this tool") over-trigger on a newer one; verification scaffolding you added ("double-check your work") can cause over-verification; severity filters in review prompts get followed more literally and depress measured recall.

**Week 6–8 — canary and cut over.** Route 5% → 25% → 100%, with the golden-set eval and cost-per-task on the dashboard at each step, and the old model still available for rollback until the retirement date passes.

**⚠ Trap:** doing the model-ID swap in week one and declaring the migration done because CI is green. Your test suite asserts shapes; the regression is in quality and cost, and neither is in your test suite unless you put an eval there. **The deliverable of a model migration is an eval comparison table, not a diff.**

**🗣 Say this in the room:** "I'd classify every hit on that model ID first — call sites get migrated, but registries and capability gates get the new model *added*, not substituted, or I break traffic that's still on the old one. Then breaking changes, then re-baseline token counts because the tokenizer probably moved, then a side-by-side eval with confidence intervals, then prompt adaptation, then a canary. The artifact I'd bring to review is the eval table, not the PR."

### Design a router across a cheap and an expensive model. Show me when it pays.

Mental model: **a router is a cascade with a confidence threshold, and its economics are governed by one number — the fraction of traffic the cheap model handles correctly — and one hazard: the cost of the escalation itself.** If escalation means re-running the whole task on the expensive model, you pay for both, and the router only pays if the cheap model handles a large majority.

**The three routing architectures, in increasing order of how much they actually work:**

1. **Pre-classification.** A tiny model or a classifier looks at the request and picks a tier. Cheap (one extra small call), but it's predicting difficulty from the input alone, which is genuinely hard — most requests don't look hard until you try them.
2. **Cascade with a confidence signal.** Run cheap first; escalate when a confidence signal is low. The confidence signal is the whole design: a self-reported score (weak, models are overconfident), token-level logprobs on the answer span (better where available), agreement across N samples at temperature (strong but N× the cost), or a cheap verifier model checking the cheap model's answer (usually the best cost/quality point).
3. **Deterministic routing on request features.** Route by tenant tier, feature, input length, or task type. Boring, transparent, no ML, and in my experience it captures most of the available savings — because the split is usually *by feature*, not by difficulty within a feature.

Start with (3). Add (2) where a feature has genuinely mixed difficulty. Reach for (1) rarely.

**💰 The math you must be able to do live.** Let `p` = fraction handled correctly by the cheap model, `C_c` = cheap cost/task, `C_e` = expensive cost/task, `C_v` = verifier cost/task.

```
cost_router      = C_c + C_v + (1 - p)·C_e
cost_all_expensive = C_e
router wins iff   C_c + C_v + (1 - p)·C_e  <  C_e
              ⟺   C_c + C_v  <  p·C_e
```

Plug numbers: `C_e` = $0.100, `C_c` = $0.012, `C_v` = $0.004. Condition: $0.016 < p × $0.100 → **p > 0.16**. So even a cheap model that only handles 16% of traffic breaks even. At p = 0.75: cost = 0.012 + 0.004 + 0.25 × 0.100 = **$0.041 vs $0.100 — a 59% saving**. At 5M tasks/month that's $500,000 → $205,000, a **$295k/month** saving.

Now the sensitivity that decides whether you build it: at p = 0.40, cost = 0.012 + 0.004 + 0.060 = $0.076, only 24% saved. **The router's value is roughly linear in `p`, so the entire project's ROI hinges on measuring `p` before you build.** Measure it by running both models over your eval set and computing the fraction where cheap is correct. If `p` is under ~0.5, the engineering and operational complexity usually isn't worth it and you should instead work on whether a mid-tier model clears the floor outright.

**The costs people omit:**

- **Latency on escalated requests is additive** — cheap + verifier + expensive. Your p99 gets worse even as your mean cost improves. If 25% of requests take 3× as long, that's a UX regression you must price.
- **Two prompts to maintain, two evals to run, two models to migrate** when either is deprecated.
- **The verifier is another model that can be wrong**, in both directions. A verifier with a 5% false-accept rate silently ships 5% of the cheap model's errors.

**⚠ Trap:** routing on self-reported confidence ("rate your confidence 1–10"). Models are poorly calibrated at this and the scores cluster at 8–9 regardless of correctness. If you use a self-reported score, you must *measure its calibration* — bucket by reported confidence and plot actual accuracy per bucket. If the curve is flat, your router is a coin flip with extra steps.

### If I ask you in a design round which serving engine you'd use and what features you'd rely on, how do you avoid claiming something that isn't true today?

This is the "engine feature matrix" version of the volatility problem, and the answer has the same shape as the pricing one: **name the mechanism, name the engine, and explicitly flag the feature-availability claim as something you verify rather than assert.**

The durable landscape (**📅 Volatile — feature matrices move every release; verify before claiming**):

- **vLLM** — the de-facto default for self-hosted OSS serving. Paged KV cache and continuous batching are its foundational contributions; it also carries prefix caching, tensor/pipeline parallelism, quantization support, LoRA adapter serving, structured-output backends, and speculative decoding. Broadest model coverage, fastest to support new architectures.
- **SGLang** — competitive throughput, with a radix-tree prefix cache designed for heavy prefix sharing (multi-turn agents, many requests over one long document) and a strong structured-output story.
- **TensorRT-LLM** — highest performance on NVIDIA silicon if you're willing to pay the compilation and operational friction. The right answer when you're squeezing a fixed fleet and the model set is stable.
- **Hosted inference providers** — the answer when you want open weights without owning the deployment, at a per-token price with someone else's utilization risk.

**The features whose availability you must verify rather than assume**, because they're the ones that get claimed casually in design rounds and are the most version-dependent: which quantization formats are supported for *your* architecture (not in general); whether the attention variant your model uses has a fast kernel path; which speculative-decoding modes are implemented and whether they compose with your other settings; which structured-output backend is wired in and what schema subset it supports; whether disaggregated prefill/decode is available; and whether multi-LoRA serving works with your quantization.

**How to say it in the room without hedging into uselessness:**

> "I'd default to vLLM — paged KV cache and continuous batching are the two things that actually determine throughput, and it has the widest model coverage. If the workload has heavy prefix sharing, like a document-QA product where thousands of requests share one long prefix, I'd benchmark SGLang against it because its radix prefix cache is built for exactly that. I'd want to verify the current release supports the quantization format and structured-output backend I'm assuming before I commit to those in a design — that matrix moves every few weeks and I've been burned assuming it."

That last sentence is the whole answer. It converts a potential wrong claim into evidence that you've operated this in production. **The failure mode is the opposite: confidently asserting that engine X supports feature Y, being wrong, and losing the room's trust on everything else you said.**

**⚠ Trap:** quoting a throughput benchmark from a blog post. Serving throughput is a function of your sequence-length distribution, your batch depth, your quantization, and your hardware. Any number not measured on your workload is marketing. If you cite one, cite it as "their published number on their workload" and say what you'd measure instead.

### I'm going to ask you the question you'll get in a real onsite: "What does it cost to run this at scale?" Do it live, out loud.

Take a concrete brief so the method is visible: **an AI support agent handling 100,000 tickets/day, resolving 60% without a human, with a hard requirement of a first response within 5 seconds.**

**Step 1 — state assumptions out loud and write them down.** This is the whole skill; the interviewer is grading the method, not the number.

- Average resolution trajectory: 8 API calls (retrieval, plan, 3–4 tool calls, draft, verify).
- System prompt + tool schemas: 6,000 tokens, frozen → cacheable.
- Retrieved context: 3,000 tokens/call, varies → not cacheable.
- Conversation growth: by call 8 the transcript adds ~4,000 tokens.
- Output: 400 tokens/call average, with the drafting call larger.
- Model: mid-tier at $3/Mtok in, $15/Mtok out. **📅 Volatile — I'd verify current pricing.**

**Step 2 — tokens per task.**
- Cacheable input: 6,000 × 8 calls = 48,000 tokens, ~90% cache-read after warm-up.
- Uncacheable input: (3,000 retrieved + growing transcript, average ~2,000) × 8 ≈ 40,000 tokens.
- Output: 400 × 8 = 3,200 tokens.

**Step 3 — cost per task.**
- Cached input: 48,000 × (0.9 × $0.30 + 0.1 × $3.75)/1e6 = 48,000 × $0.645/1e6 = **$0.0310**
- Uncached input: 40,000 × $3/1e6 = **$0.1200**
- Output: 3,200 × $15/1e6 = **$0.0480**
- **Total ≈ $0.199/task.** Call it $0.20.

**Step 4 — scale it and sanity-check.**
- 100,000 tickets/day × $0.20 = **$20,000/day** = **~$600,000/month**.
- Sanity check against value: a human handling a ticket costs perhaps $4–6 fully loaded. Deflecting 60% of 100k/day = 60,000 tickets × $5 = **$300,000/day of human cost avoided**, against $20,000/day of inference. **15:1.** The economics are not close, and saying so is more useful than the raw number.

**Step 5 — name the levers, in order of magnitude.**
1. **Uncached input is 60% of the bill.** Cut retrieval from 3,000 to 1,200 tokens with a reranker: saves ~$0.055/task = **$165k/month**. Biggest single lever.
2. **Turn count.** Getting from 8 calls to 6 cuts roughly 25% across the board = **$150k/month**.
3. **Tier the 40% that escalate to a human** — they don't need the full trajectory; detect early and hand off. If half of them can bail after 3 calls, that's ~$70k/month.
4. **Output tokens** — a verbosity instruction cutting 400 → 280 saves $0.0144/task = **$43k/month**.

**Step 6 — check the latency requirement, because a cost answer that violates the SLO is wrong.** 5-second first response with 8 calls in the trajectory means the *first response* cannot wait for the whole trajectory. Architecture: acknowledge immediately, stream the first substantive turn after retrieval (call 1–2, ~1.5 s), continue the trajectory behind the stream. If the design requires all 8 calls before any output, the SLO is unachievable at any price and that's the finding to report.

**🗣 Say this in the room:** "Roughly twenty cents a task, six hundred thousand a month at that volume — but the number I'd actually put in the doc is that we're spending $20k/day of inference to avoid $300k/day of human handling, and the biggest lever is retrieval size, not model choice. Every input above is an assumption I'd want to replace with a measurement in week one."

**🏋 Drill (20 minutes, timed, unaided, no calculator beyond arithmetic on paper):** given a product brief you invent, produce the six steps above with every number derived. Pass criterion: an assumptions list, a cost per task, a monthly figure, a value-side sanity check, and a ranked lever list where the top lever is justified by its share of the bill.

### Last one. Give me the whole stack for a product brief, and tell me how you keep this knowledge from going stale.

**Brief:** an enterprise knowledge assistant — 4,000 seats, questions over an internal corpus of 2M documents, must cite sources, EU customers, 3-second p95 to first token, launch in 10 weeks.

**The stack, decided in the framework's order:**

1. **Retention and residency first**, because they can veto everything else. EU customers → EU data residency, likely a regional endpoint or a cloud-partner deployment. That decision constrains the model list *and* the feature list (partner deployments are feature-lagged — verify whether batch, caching semantics, and native citations survive), so it's week-one work, not week-eight work.
2. **Architecture: retrieval, not long context.** 2M documents can't be stuffed, and the cost arithmetic is decisive: 300k tokens/question at $3/Mtok is $0.90/question versus ~$0.02 with retrieval. Embedding + reranking on self-hosted open models — this is the task class where open weights are at or above parity and 100× cheaper.
3. **Capability floor:** build the eval before touching a model. 200 real questions from pilot users, labeled with gold answers and gold citations. Threshold stated with two numbers: ≥90% answer accuracy, ≤2% unsupported claims. Descend from the frontier tier; take the cheapest model that clears with margin.
4. **Citations are a hard requirement**, so native citation support moves from "nice" to a selection axis with veto power. It converts groundedness from a judgment call into a check.
5. **Caching:** frozen system prompt with a breakpoint; retrieved context after it. At 4,000 seats × ~15 questions/day = 60,000 questions/day, a 6,000-token frozen prefix cached at 0.1× saves 60,000 × 6,000 × ($3.00 − $0.645)/1e6 ≈ **$424/day ≈ $12.7k/month** for one marker.
6. **Latency:** 3-second p95 TTFT with a retrieval hop means retrieval must be ~300 ms p95 and the model's TTFT ~1 s at your p95 input length. That budget rules out reasoning-tier models on the interactive path — if you want reasoning, it goes in a background "deep research" mode with different UX, not in the default path.
7. **Observability from day one:** per-call usage logging with trace/tenant/prompt-version, cost per resolved question as a tracked metric, cache hit rate as an alerting SLI, and a golden-set eval on a cron feeding the same dashboard as latency.
8. **Failure design:** a documented fallback model with a measured quality delta, capped retries, and an explicit "I don't have a grounded answer" path — which for a knowledge assistant is a *feature*, not a failure.

**Now the meta-question, which is what this section is really for.** The way you keep this from going stale is to **make the volatile layer an artifact you maintain rather than knowledge you hold**:

- **One file in your repo — a dated rate card and capability table.** Provider, model ID, context window, the five token rates, cache minimum, feature flags you depend on, and a `verified_on` date. Your cost function imports it. Your CI fails if `verified_on` is more than 30 days old. That single check converts staleness from an invisible risk into a build failure.
- **A scheduled job that pulls the live model list** from each provider's models endpoint and diffs against the file, opening a ticket on any change.
- **The golden-set eval on a cron**, so provider-side changes surface as a metric rather than a customer complaint.
- **A 90-minute refresh ritual before every loop** — lineups, prices, deprecations, parameter deltas, rate-limit tiers, engine matrices, open-weight releases, regulatory dates.

**🗣 Say this in the room, and mean it:** "I don't carry the model table in my head — I carry the framework and I re-resolve the table, because anything I memorized six months ago is wrong now. What's stable is the structure: output is about 5× input, cache reads about a tenth, batch about half, and the decision is always the cheapest model that clears a measured quality floor under a stated latency budget. If you want current numbers I'll pull them up, but the arithmetic is the same either way."

**🏋 Final drill (90 minutes, unaided, timed).** Write, from scratch: (a) the nine-axis selection rubric in order; (b) the five token rates with their approximate multiples and the caching break-even in calls for both TTLs; (c) a correct Anthropic tool loop with error handling and the five invariants; (d) the prefix-cache invalidation hierarchy table; (e) a per-request cost function; (f) the router break-even inequality with a worked example; (g) the EU AI Act tier names and the milestone you're closest to. **Pass criterion: no more than two factual errors, and every cost claim carries its arithmetic.** Then verify every volatile item against live docs and record the date — that verification pass *is* the habit this section exists to build.
