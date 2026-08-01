### Read me the rate-limit headers and then write the client that respects them.

Mental model: **an LLM provider's rate limiter is a multi-dimensional token bucket, and the dimension that kills you is not requests.** You are limited on requests per minute *and* input tokens per minute *and* output tokens per minute, independently. A workload of 10 RPM with 200k-token prompts saturates ITPM while sitting at 2% of your RPM ceiling — and the 429 you get back looks identical to a request-rate 429 unless you read the headers.

The headers, by provider family (**📅 Volatile — names drift; the structure doesn't**):

- **Anthropic:** `anthropic-ratelimit-requests-limit` / `-remaining` / `-reset`, plus the same triple for `input-tokens` and `output-tokens`, plus `retry-after` on a 429. `-reset` is an RFC 3339 timestamp, not a delta.
- **OpenAI:** `x-ratelimit-limit-requests` / `-remaining-requests` / `-reset-requests` and `-limit-tokens` / `-remaining-tokens` / `-reset-tokens`. Resets are durations like `6m0s`.

**📐 Numbers you must know:** the second mechanism, and the one that actually explains most surprise 429s — **admission control reserves against your declared `max_tokens`, not your actual output.** A request with `max_tokens: 128000` debits your output-token bucket by 128,000 at admission even if it generates 200 tokens; the difference is refunded after completion. So a service that lazily sets `max_tokens` to the model's ceiling everywhere will 429 at a fraction of its real throughput — and the fix is a one-line change to a sane per-route ceiling. This is one of the highest-value pieces of trivia in this section because it looks like a capacity problem and is actually a configuration problem.

```python
import time, random, anthropic

class BudgetedClient:
    def __init__(self, client, floor=0.05):
        self.c, self.floor = client, floor          # floor: pause below 5% remaining

    def call(self, **kw):
        for attempt in range(6):
            raw = self.c.messages.with_raw_response.create(**kw)
            h = raw.headers
            msg = raw.parse()
            # Proactive: back off before the bucket empties, don't wait for the 429.
            rem  = int(h.get("anthropic-ratelimit-input-tokens-remaining", 1 << 30))
            lim  = int(h.get("anthropic-ratelimit-input-tokens-limit", 1 << 30))
            if lim and rem / lim < self.floor:
                time.sleep(_secs_until(h.get("anthropic-ratelimit-input-tokens-reset")))
            return msg
        raise RuntimeError("exhausted retries")

def on_429(headers, attempt):
    ra = headers.get("retry-after")
    if ra:                                   # ALWAYS prefer the server's number
        return float(ra)
    return min(60.0, 2 ** attempt) + random.uniform(0, 1)   # jitter is mandatory
```

Three rules I enforce in review:

1. **Honor `retry-after` over your own backoff.** Your exponential curve is a guess; the header is the answer.
2. **Full jitter, always.** Synchronized retries from N workers reconverge into a thundering herd that keeps you 429'd. This is the same lesson as any distributed retry storm — you already know it, and people forget it here because the SDK "handles retries."
3. **Back off *before* the 429, using `-remaining`.** A 429 is a wasted round trip and, on some providers, still consumes a request slot. Treating `remaining/limit` as a health signal and shedding load at 5% is the difference between graceful degradation and a cliff.

**⚠ Trap:** relying on the SDK's built-in retries as your rate-limit strategy. They default to 2 retries and they're per-call — they do nothing about the fact that your 40 workers are collectively over budget. Rate limiting is a *fleet-level* concern; you need a shared token bucket (Redis, exactly as you'd build for any upstream quota) that all workers draw from, keyed on tokens rather than requests.

### You're launching in three weeks and your projected load is 6× your current TPM ceiling. What do you do?

This is a capacity-planning question wearing an LLM costume, and the strong answer treats it like one: **compute the requirement, secure the headroom, and design the degradation path — in that order, in parallel, because the first one gates the sales conversation.**

**Step 1: compute the requirement precisely, in the provider's units.** Not "6× traffic." Peak RPM, peak input-TPM, peak output-TPM, computed at your *p95* prompt length, not the mean, and at your peak minute, not your daily average. A product with a 4× peak-to-mean ratio and a 20% weekly growth curve needs headroom over peak, not over mean:

```
peak_input_TPM  = peak_RPM × p95_input_tokens
peak_output_TPM = peak_RPM × p95_max_tokens        # remember: admission reserves max_tokens
required        = peak × (1 + growth_to_launch) × safety_factor(1.5–2×)
```

**Step 2: escalate the tier — this has a lead time, so start it today.** Provider tiers advance on cumulative spend and account age, and there is a manual path above the self-serve ceiling. Both take time. The concrete asks: current tier and its limits, the thresholds for the next tier, and whether they'll grant a temporary or custom limit for a dated launch. Providers do this routinely for a credible launch plan with numbers attached — bring the arithmetic from step 1, not a request for "more."

**Step 3: reduce demand, which is usually faster than raising supply.**
- **Cache the prefix.** Cached-read tokens still count against ITPM on some providers but the *latency* relief is immediate and the cost relief is large; verify the quota treatment for yours.
- **Cut `max_tokens` per route.** Frees output-TPM headroom immediately, at zero quality cost if your outputs are actually short.
- **Move anything with a soft deadline to the batch tier**, which has separate quota. Nightly evals, backfills, enrichment jobs — off the interactive quota entirely.
- **Route the head of the distribution down a tier.** If 60% of traffic is a task your small model passes, that's 60% of your peak TPM moved to a different quota pool.

**Step 4: design the degradation path before you need it.** Priority classes: paying-user interactive traffic never sheds; free-tier traffic sheds first; background enrichment sheds first of all. Implement it as a shared token bucket with per-class reservations, not as "retry until it works."

**Step 5: secure a second provider as capacity, not just as failover.** Two providers with independent quotas is 2× headroom, and the eval work to qualify the second one is work you should have done anyway.

**💰 Math:** peak 900 RPM, p95 input 6,000 tokens, `max_tokens` 2,000. Input-TPM = 900 × 6,000 = 5.4M. Output-TPM reserved = 900 × 2,000 = 1.8M. If your tier caps ITPM at 2M you are **2.7× over on input alone** — and the fix might be entirely free: a prefix cache that makes 5,000 of those 6,000 tokens cache-reads (if they don't count against ITPM on your provider) drops you to 900k, under the cap, with no tier change and no code beyond one marker. **Always compute whether you have a quota problem or a prompt problem before you escalate.**

**⚠ Trap:** discovering your limits during the launch. Load-test against the real provider at real peak *two weeks out*, in the region you'll serve from. A synthetic test against a mock proves nothing about admission control.

### Running requests across multiple accounts to get more throughput — smart ops or a terms violation?

Give the honest, bounded answer; interviewers are testing your judgment about a genuinely grey area, and both "always fine" and "never do it" are wrong.

**The bright line:** creating multiple accounts *to evade a rate limit that a single account would enforce* is circumvention, and every major provider's terms prohibit it. Doing it will, at best, get your accounts merged or suspended; at worst it ends an enterprise relationship. Do not build a rotation over free-tier keys and call it architecture.

**What is legitimate and is normal enterprise practice:**

- **Multiple workspaces or projects under one organization**, each with its own key and its own limits, provisioned by the provider. This is a feature, not a workaround — it exists so you can isolate prod from staging, or team A's budget from team B's, and get per-workspace attribution and quotas.
- **Multiple deployment surfaces for the same model.** First-party API, plus the same model through a cloud partner (Bedrock / Vertex / Foundry), have **separate quota pools** and separate commercial relationships. Running primary on one and burst on another is a supported architecture, and it's how large deployments actually get capacity. It's also your best availability story, since the failure domains are genuinely different.
- **Multiple providers.** Obviously fine, and the strongest form.
- **Multiple regions** where the provider offers regional endpoints with independent capacity.

**The engineering costs you must name, because they're what makes this a real design question rather than a hack:**

1. **Prompt caches don't follow you.** Caches are scoped to the account/deployment and the model. Spreading traffic across pools means each pool maintains its own cache, so your hit rate degrades roughly as `1/N` for the same traffic volume — and a 90% → 60% hit rate can cost you more than the burst capacity was worth. Route with **cache affinity**: hash on the cache key (tenant, or system-prompt version) so the same prefix consistently lands on the same pool.
2. **Feature parity is not guaranteed.** Cloud-partner deployments are feature-lagged: batch endpoints, certain server-side tools, caching semantics, and newest-model availability differ. Your abstraction must degrade, and your eval must run against each surface.
3. **Model identity drifts.** The "same" model on a partner platform is a differently-versioned snapshot with a different ID scheme. Pin explicitly and eval per surface.
4. **Cost attribution fragments.** You now have N invoices with different rate cards.

**🗣 Say this in the room:** "Multi-account to dodge a limit is circumvention and I wouldn't build it. Multi-*workspace* under one org, and the same model across first-party plus a cloud partner, are supported and are how you actually get headroom — those are separate quota pools with separate failure domains. The engineering cost is that prompt caches are pool-scoped, so I route with cache affinity rather than round-robin, or I hand back in cache misses more than I gained in capacity."

### It's 3am. Your primary provider is returning 429s and 529s across the board. Design the failover.

**🔍 Failure taxonomy first — because the correct response differs by class, and reflexive failover on the wrong class makes things worse:**

| Signal | Class | Correct response |
|---|---|---|
| 429 with `retry-after`, your `-remaining` near zero | **You** are over quota | Shed load by priority class, back off; do NOT fail over (you'll just move the overload) |
| 429 with `-remaining` healthy | Provider-side capacity | Retry with jitter; fail over if it persists past ~30s |
| 529 / overloaded | Provider-side capacity | Retry with jitter, then fail over |
| 5xx on a subset of requests | Partial degradation | Retry; circuit-break per model, not per provider |
| Latency up, error rate flat | Degradation, not outage | Do **not** fail over on latency alone — you may be seeing your own long prompts |
| 400s spiking | **Your** bug, or a contract change | Never retry. Alert. (e.g. a retention-config or model-deprecation 400) |

The design, in layers:

**Layer 1 — request-level retry with jitter, honoring `retry-after`.** Cap at 2–3 attempts. Beyond that you're amplifying the outage; this is the "just add a retry" trap that costs you 4× on a bad day.

**Layer 2 — circuit breaker per (provider, model).** Standard breaker, but the half-open probe should be a *cheap* request, not a real user's. Break on error rate over a rolling window, not on consecutive failures — LLM traffic is bursty enough that consecutive-failure counting trips spuriously.

**Layer 3 — the failover ladder, in preference order.** This is where the LLM-specific judgment lives:

1. **Same model, different deployment surface** (first-party → cloud partner). Zero quality change; you keep your evals valid. Cost: cache is cold on the new surface, and feature parity may be partial.
2. **Same family, smaller model.** Predictable quality drop that you have *measured*, because you ran your eval against the fallback tier before the incident.
3. **Different provider.** Largest quality delta and the one requiring the most prep — a prompt tuned for one model is not tuned for another, so a cross-provider fallback needs its own prompt variant and its own eval run, maintained.
4. **Degrade the feature.** Cached/canned response, "try again shortly," queue-and-notify. For many features this is *better* than a bad answer, and saying so is a senior signal.

**The three things people forget:**

- **Fail over with a quality budget, not blindly.** If your fallback scores 78% where primary scores 94%, silently serving it during a 4-hour incident may cause more damage than a clean error — especially for anything writing to a database or sending a message. Gate destructive actions behind the primary; let read-only paths degrade.
- **Emit a signal that says which model served the request**, and put it on every log line and every eval sample. Otherwise your Monday quality dashboard shows an unexplained dip and you spend a day rediscovering that Saturday was an incident.
- **Bound the fan-out cost.** Failover multiplies attempts. Cap total attempts per user task, not per API call, or one bad hour becomes a 4× invoice.

**💰 Math:** a naive "retry 5 times across 3 providers" policy during a 2-hour partial outage on a service doing 40k tasks/hour at $0.12/task. Base spend for the window: 80,000 × $0.12 = $9,600. If 30% of requests trigger the full ladder and average 3.5 billed attempts: 24,000 × $0.12 × 3.5 + 56,000 × $0.12 = $10,080 + $6,720 = **$16,800** — a 75% overspend on an incident where you served *fewer* successful requests than normal. Cap attempts per task and count them.

### How do you tell a quality regression from an availability problem, when a provider ships a change you didn't ask for?

Mental model: **availability failures are loud and instrumented; quality failures are silent and only visible against a baseline you built in advance.** If you don't have a continuously-running eval, you have no ability to detect this class of incident at all — you'll learn about it from a customer, three weeks late, and you won't be able to prove it.

The threat model, concretely:

- **A stable alias repoints to a new snapshot.** You pinned `some-model-latest` and the underlying weights changed. Your evals move; your logs show nothing.
- **A default flips.** Thinking goes from off-by-default to on-by-default, or `display` changes, or effort's default level changes. Behavior and cost both move with no code change on your side.
- **A safety classifier tightens.** Requests that used to succeed start returning refusals. This shows up as an *error* class you may not be handling — on some providers a refusal is HTTP 200 with `stop_reason: "refusal"`, so if your code reads `content[0]` unconditionally you get an IndexError, and if it doesn't, you get an empty answer counted as a success.
- **Tokenizer change on a model revision.** Same text, 1×–1.35× the tokens. Your context budgets and cost model silently shift.

The detection system, which is the actual answer:

1. **Pin exact model IDs, never aliases, in production.** Aliases are for prototypes. This is the single highest-leverage control and it costs nothing.
2. **Run a golden-set eval continuously** — 100–300 examples, hourly or on a cron, against production config, with results in your normal metrics system alongside latency and error rate. Alert on a shift beyond your CI.
3. **Log a response fingerprint on every request**: exact model ID returned by the API (not the one you requested — providers may substitute), `stop_reason`, `usage` breakdown, and system-prompt version hash. A shift in the *distribution* of `stop_reason` or mean `output_tokens` is a leading indicator that fires before your eval does, because it's computed on 100% of traffic instead of a sample.
4. **Alert on token-per-request drift.** Mean output tokens jumping 40% overnight with flat traffic is a default flip or a model swap, and it's visible within minutes.
5. **Keep a canary pinned to the previous snapshot** during a migration window so you can A/B rather than argue.

**⚠ Trap:** treating "the model got worse" as unfalsifiable and moving on. It's entirely falsifiable if you have a golden set with confidence intervals, and entirely unfalsifiable if you don't. The senior behavior is to have the artifact ready *before* the incident, and to be able to say "our golden set moved from 93.1% ± 1.8 to 87.4% ± 2.1 between the 3rd and 4th, here are the twelve examples that flipped."

**🗣 Say this in the room:** "Availability I detect from error rates; quality I can only detect against a baseline, so I run a golden-set eval on a cron into the same dashboard as p99 latency, and I log the model ID the API actually returned on every request. Aliases never go to production — a silently-updated alias is a deploy I didn't do, and pinning is free."

### Give me the open-weight landscape. Families, and where each is honestly competitive.

Mental model: **the open-weight question is never "is it as good as the frontier model" — it's "is it above the capability floor for this task class, at a cost structure I control."** For a large fraction of production tasks the answer has been yes for a while, and for a shrinking-but-real set of tasks the answer is still clearly no. Knowing *which set* is the signal.

**The families (📅 Volatile — new releases every few weeks; verify the current flagship in each line before your loop):**

- **Llama (Meta).** The ecosystem default: the best tooling support, the most fine-tuning recipes, the widest deployment coverage. Community license, not open source — see the license question below.
- **Qwen (Alibaba).** Consistently the strongest open weights on coding and multilingual work, and a very wide size ladder from sub-1B to very large MoE, which matters because it lets you pick a size for your latency budget rather than accepting whatever the family ships. Mostly Apache-2.0, with per-size exceptions.
- **DeepSeek.** MoE architectures with strong reasoning; notable for permissive licensing on weights (MIT on the R1 line) and for publishing genuine architectural contributions (MLA, auxiliary-loss-free load balancing) rather than just weights.
- **Mistral.** Strong European option, mixed licensing — some models Apache-2.0, others under a research-only license that forbids commercial use. Read the specific model card every time; this family is where people most often get the license wrong.
- **Gemma (Google).** Small, efficient, good for on-device and cost-sensitive classification. Custom terms with a prohibited-use policy, not OSI-approved.
- **Specialist lines:** embedding models, rerankers, and code-specific models where a small open model is routinely *better* than a frontier general model at the same task and 100× cheaper.

**The honest capability gap, by task class:**

| Task class | Open weights vs frontier API |
|---|---|
| Classification, routing, sentiment, extraction with a schema | **Parity.** Frontier is waste here. |
| Embeddings and reranking | **Open wins.** Cheap, fast, self-hostable, task-specific. |
| Summarization of a provided document | **Near parity** at mid size. Differences show up on long inputs. |
| Single-file code completion / infill | **Near parity** with a good code model. |
| Structured multi-step tool use in an agent | **Gap, and it's the big one.** Selection accuracy, argument fidelity, error recovery, and knowing when to stop all degrade noticeably. |
| Long-horizon agentic coding across a repo | **Clear frontier advantage.** |
| Hard multi-step reasoning, math, competition-style problems | **Gap narrowing but real**, especially in the reasoning-trained frontier tier. |
| Very long context with multi-fact synthesis | **Frontier advantage**, and the usable-vs-advertised gap is wider on open models. |
| Anything where you need a vendor to be accountable | **API wins** by definition. |

**🗣 Say this in the room:** "My default architecture is a frontier API for the agentic and reasoning-heavy path and open weights for the high-volume mechanical path — classification, routing, embeddings, reranking, extraction. That split usually moves 70–90% of *request volume* onto self-hosted models while leaving the hard 10% on the API, and it's where the cost curve actually bends. The place I don't reach for open weights is multi-step tool use, where the gap is still real."

**⚠ Trap:** benchmarking an open model against a frontier model on a leaderboard and concluding parity. Leaderboards over-represent the tasks the open community optimizes for, and contamination is a live problem. Run your own eval. And if you quote a leaderboard number in an interview without having run your own, expect the follow-up "and what did *you* measure?"

### Walk me through the license traps in open weights. Be specific.

**⚠ Trap: "open weights" is not "open source."** Most of the popular families ship under custom licenses that are not OSI-approved and carry conditions a normal open-source license does not. Getting this wrong is a legal exposure, not a style violation, and an interviewer at any company with a legal department is testing whether you know that.

The specific clauses, by family (**📅 Volatile — licenses change between model generations within the same family; read the model card for the exact checkpoint you are deploying**):

**Llama — Community License.**
- **The 700M MAU clause.** If, on the release date of the version you use, your products or affiliates' products have **more than 700 million monthly active users**, you must request a separate license from Meta, which they may grant or withhold at their discretion. This is a real gate for a handful of companies and a non-issue for everyone else — but you must be able to answer it, because "we don't know our MAU" is not an answer in a procurement review.
- **Naming and attribution.** Derivative models must carry "Llama" at the start of the name; you must display "Built with Llama."
- **Acceptable Use Policy** incorporated by reference, and it is enforceable.
- **Output usage** — this clause has *changed between generations*. Earlier Llama licenses restricted using model outputs to improve other large language models; a later generation relaxed it, subject to the naming requirement. **Do not answer this from memory for a specific version; read that version's license text.** Being able to say "this clause changed between versions, so I'd check the exact checkpoint" is the correct answer.

**Qwen.** Most sizes ship Apache-2.0, which is genuinely permissive — but not uniformly across every size in every generation. Check the specific model card.

**Mistral.** Split licensing: several models are Apache-2.0; others ship under a research license that **prohibits commercial use** outright. This is the family where I've seen the most expensive mistakes, because "Mistral is Apache" is a widely-believed half-truth.

**Gemma.** Custom Terms of Use plus a Prohibited Use Policy. You must pass the terms through to downstream recipients, and Google reserves the ability to restrict use. Not OSI-approved.

**DeepSeek.** MIT on the R1-line weights — about as permissive as it gets. Note that permissive weight licensing does not automatically mean permissive *training-data* provenance, which is a separate question your legal team may ask.

**The cross-cutting trap — anti-distillation clauses on the API side.** This is separate from weight licenses and catches more people. Frontier API providers' terms generally prohibit using their outputs to develop a **competing model**. So the pipeline "call the frontier API on 500k prompts → fine-tune an open model on the outputs → ship it" has two independent legal questions: does the open model's license allow the derivative, and does the API provider's ToS allow you to use its outputs that way. The second one is where teams get surprised, and the answer is genuinely fact-specific: distilling to serve *your own* product's task is a different posture from distilling to ship a general-purpose model that competes with the provider.

**🗣 Say this in the room:** "I treat weight licenses as a procurement input, not a footnote. The three things I check on every checkpoint: is it actually OSI-approved or a custom license; is there a scale or field-of-use gate like Llama's 700-million-MAU clause; and are there naming, attribution, or output-usage conditions that follow the derivative downstream. And separately, if training data came from a frontier API, I check that provider's terms on using outputs to develop models — that's a different agreement and it's the one people forget."

### Legal comes to you: "Can we fine-tune Llama on our customer data and sell the resulting model as a product?" Answer them.

Answer in four separable questions, because conflating them is how this goes wrong. The interviewer is testing structured thinking under a legal-flavored question, not asking you to practice law — and you should say that out loud.

**1. Does the base model's license permit a commercial derivative?** For a Llama community-license model, generally yes, subject to: the MAU gate (check yours as of the version's release date), the naming requirement (your product's model name must begin with "Llama"), the "Built with Llama" attribution, passing the license and Acceptable Use Policy through to whoever you distribute to, and including the required notice file. So "sell it" is permitted but *conditioned*, and one of the conditions constrains your product branding — which is a real product decision, not a legal footnote, and worth raising early.

**2. Does your customer data permit this use?** Almost always the harder question. Your DPA and privacy policy have to actually authorize using customer content to train a model, and "improve our services" is frequently *not* read as covering "bake it into weights we sell to third parties." Then: is the data personal data under GDPR/CCPA, and if so what's your lawful basis, and how do you honor a deletion request against a model that has already memorized it? **Model weights are not a database you can DELETE FROM.** The engineering answer is to make it never enter the weights: per-tenant retrieval instead of per-tenant fine-tuning, PII stripping and canonicalization in the training pipeline, and a documented retention window on the training corpus.

**3. Does the training data have third-party provenance issues?** If any of the fine-tuning data is outputs from a frontier API, that provider's terms on using outputs to develop models applies independently of the Llama license — see the previous question. If it's scraped, that's its own analysis.

**4. What are you actually shipping?** Selling weights, offering it as a hosted service, and embedding it in your product are three different distribution modes with different obligations under the same license. Nail this down before answering anything else, because the naming and pass-through conditions bite hardest when you distribute weights.

**The engineering counter-proposal you should have ready:** in most cases where someone asks this, fine-tuning is not the right mechanism anyway. If the goal is "the model knows our customers' data," retrieval gives you per-tenant isolation, instant updates, an actual deletion story, and no license question about the derivative. If the goal is "the model follows our format and tone," a small fine-tune on *synthetic or internal* data — not customer data — usually gets you there. **Raising that alternative unprompted is the strongest thing you can do with this question**, because it shows you understand fine-tuning is the last rung of the escalation ladder rather than the first.

**🗣 Say this in the room:** "Four questions, and I'd want counsel on the middle two: does the base license allow a commercial derivative and under what conditions; does our DPA cover training on customer data and how do we honor deletion against weights; is there third-party provenance in the training set; and are we distributing weights or hosting a service. But before any of that — I'd push back on the design. If the requirement is per-customer knowledge, retrieval gives us tenant isolation, immediate updates, and a real deletion story, and it sidesteps the entire question."

### Build me the self-host versus API crossover model. Where's the break-even and what did you leave out?

Mental model: **API pricing is fully variable cost; self-hosting is mostly fixed cost with a utilization multiplier.** So the crossover is entirely a function of *sustained* utilization, and the number that kills self-hosting projects is never the GPU price — it's the fraction of the day those GPUs are idle.

**The API side** is trivial: `tokens × rate`, already built in the cost-function question.

**The self-host side**, and the discipline is in enumerating every term:

```
monthly_cost = GPUs × hours × $/GPU-hr        # compute
             + engineer_FTE_fraction × loaded_cost   # the term everyone omits
             + observability + load-balancing + storage + egress
             + idle_capacity_you_provisioned_for_peak
```

Then the throughput side, which decides how many GPUs you need:

```
tokens/sec/GPU  ← measured, at YOUR sequence lengths and YOUR batch size
sustainable_load = tokens/sec/GPU × GPUs × utilization_fraction
```

**💰 Math, worked.** Suppose 3B output tokens/month on a mid-size open model, and your measured serving throughput is 2,500 output tokens/sec/GPU at your batch depth on an 80GB card.

- Seconds of GPU time needed: 3e9 / 2,500 = **1.2M GPU-seconds** = 333 GPU-hours.
- At 100% utilization that's 333/730 ≈ **0.46 GPUs**. But you can't run at 100% — traffic is peaky. At a realistic 30% average utilization you need ~1.5 GPUs of capacity, and for redundancy you deploy 2 (or 4, if the model needs TP=2 and you want two replicas).
- 2 GPUs × 730 hours × $2.50/GPU-hr (**📅 Volatile**) = **$3,650/month** of compute.
- Plus 0.25 FTE at a $300k loaded cost = **$6,250/month**.
- Plus observability, LB, storage, on-call: call it **$1,000/month**.
- **Self-host total: ~$10,900/month.**

API side for the same 3B output tokens, assuming a comparable-tier hosted model at $5/Mtok output plus, say, 6B input tokens at $1/Mtok: (3,000 × $5) + (6,000 × $1) = $15,000 + $6,000 = **$21,000/month**.

So at this volume self-hosting wins by roughly 2×. Now change one number: drop the volume to 300M output tokens/month. API side falls to ~$2,100/month. Self-host side barely moves — you still need the GPUs for peak, you still need the engineer — call it $9,000. **Self-hosting is now 4× more expensive.** That's the whole shape: the API line is linear through the origin, the self-host line has a large intercept, and the crossover for a small open model against a small hosted model tends to sit somewhere in the high hundreds of millions to low billions of tokens per month.

**What people leave out, in order of how much it costs them:**

1. **Engineering time.** Always the largest term below ~1B tokens/month, and always the one omitted from the spreadsheet that gets self-hosting approved.
2. **Utilization.** Everyone models 100%. Real interactive traffic runs 20–40% average against peak-provisioned capacity. That's a 2.5–5× multiplier on your compute line.
3. **The quality delta.** If the open model needs 1.4× the output tokens to reach the same answer quality, or fails 6% more often and triggers a retry or a human, that's a cost you must add. Cost per *resolved task*, not per token.
4. **Peak headroom and redundancy.** One replica is not a deployment.
5. **Model upgrades.** The API upgrades for free; self-hosting means you own re-evaluating, re-tuning, and re-deploying every time a better checkpoint lands.

**🗣 Say this in the room:** "The crossover is set by sustained utilization, not by the GPU price. Below roughly a billion output tokens a month, the loaded cost of the engineer who owns the deployment dominates every other term, and the API wins even at a 5× per-token premium. Above it, self-hosting wins — but only for a workload with steady, predictable load and a task where the open model actually clears my quality floor. Bursty traffic on self-hosted GPUs is the worst of both worlds."

### Distillation from a frontier model into a small open model — what's the technical ceiling, and what's the legal ceiling?

**The technique.** You use a strong model to generate a training set — labels, reasoning traces, tool trajectories, preference pairs — and fine-tune a small open model on it. Two flavors: *hard-label* distillation (train on the teacher's sampled outputs, which is just supervised fine-tuning on synthetic data) and *soft-label* distillation (train on the teacher's full output distribution, minimizing KL to the teacher). Soft-label is more sample-efficient but requires logit access, which frontier APIs don't provide — so in practice API distillation is hard-label SFT on generated data.

**The technical ceiling, stated honestly:**

- **It works extremely well for narrow, well-specified tasks.** A 7B model distilled on 50k teacher-generated examples of one classification or extraction task routinely matches the teacher on that task, at 1–2% of the cost. This is the highest-ROI move in applied LLM engineering and it is underused.
- **It works poorly for general capability.** You are transferring *behavior on your data distribution*, not intelligence. Off-distribution the student falls off a cliff, and it falls off silently — the failure mode is confident wrongness on inputs your distillation set didn't cover.
- **The student inherits the teacher's errors and none of its uncertainty.** If the teacher is 94% accurate, 6% of your training labels are wrong, and the student learns them as ground truth. Filtering matters enormously: use the teacher's own agreement across samples, a verifier, or a second model as a judge to drop low-confidence labels. Distilling unfiltered teacher output is the most common reason a distillation project underdelivers.
- **Reasoning traces transfer better than you'd expect but need verification.** Training on teacher chains-of-thought helps — but only when you filter to traces whose *final answer was correct*. Traces with correct answers and broken reasoning teach broken reasoning.
- **Data volume beats data volume-of-model.** 50k well-filtered, diverse, on-distribution examples beats 500k unfiltered ones, consistently.

**The legal ceiling:** frontier providers' terms generally prohibit using outputs to develop a **competing** model. That word is load-bearing and the analysis is fact-specific. Distilling a task-specific classifier that runs inside your product is a materially different posture from training and releasing a general-purpose assistant. Separately, some open-weight licenses have their own output-usage clauses that have changed between generations. And if you plan to release the student's weights, you inherit both sets of conditions. **Get counsel; don't reason your way to a conclusion from the license text alone, and say so in the room** — knowing where the boundary of your competence is reads as senior, not evasive.

**💰 Math:** an extraction task at 20M requests/month, 1,200 in / 150 out. Frontier at $3/$15: (20e6 × 1200/1e6 × $3) + (20e6 × 150/1e6 × $15) = $72,000 + $45,000 = **$117,000/month**. Distillation project: 80k teacher-generated examples at ~$0.008 each = $640 one-time, plus ~$800 of fine-tuning compute, plus two engineer-weeks. Serving the 7B student self-hosted at, say, $4,500/month of GPU including utilization slack. **Payback on the one-time cost is under a day; the run-rate saving is ~$112k/month.** This is the single most compelling cost argument in applied LLM engineering — and the reason it isn't done everywhere is that it requires an eval good enough to prove the student is safe to ship, which most teams don't have.

**⚠ Trap:** distilling before you have a stable prompt and a trustworthy eval. You'd be freezing today's prompt into weights. Distill last, after the escalation ladder — prompt, context, retrieval, tools, structured output, routing — has been exhausted and the task has stopped moving.

### For each of these — a support-ticket router, a code-review assistant, and a legal-document Q&A — which model would you pick and why?

Answer each with the same shape: capability floor first, then the constraint that actually decides it. Interviewers use this format to test whether your framework survives contact with concrete cases.

**Support-ticket router** (classify into ~40 categories, extract 6 fields, route). High volume, low difficulty, latency matters only in aggregate.

- Floor: a small hosted model almost certainly clears it; a fine-tuned open model of 7–8B parameters clears it after distillation.
- Deciding constraint: **cost at volume**. At 20M/month this is exactly the distillation case from the previous question — $117k/month on frontier versus ~$5k self-hosted, with a two-week project.
- My pick: **small hosted tier to launch and to generate the distillation set; open-weight fine-tune once the taxonomy is stable.** Ship the escalation path too: anything the classifier scores below a confidence threshold goes to a bigger model, then to a human. That confidence-routed cascade is what makes the cheap tier safe.
- Structured output with a strict schema is mandatory here, and cheap.

**Code-review assistant** (comment on a diff, flag real bugs, don't nitpick).

- Floor: this is genuinely hard. Real-bug recall on unfamiliar code is where the open/frontier gap is widest, and it's a task where being wrong is expensive in a specific way: false positives train reviewers to ignore the tool, which kills adoption permanently.
- Deciding constraint: **precision and recall on real bugs**, and the fact that latency is soft — a review comment 40 seconds after the push is fine.
- My pick: **frontier reasoning-tier model, higher effort setting**, because latency is soft and the cost per review is bounded by diff size, not by traffic. A code review at 15k input tokens and 1,500 output tokens costs (15,000 × $5 + 1,500 × $25)/1e6 = $0.1125. At 3,000 reviews/day that's $338/day — **$10k/month**, trivially justified against engineer time.
- The non-obvious part: recent models follow severity filters *literally*, so a prompt saying "only report high-severity issues" makes measured recall drop even when bug-finding improved. Have it report everything with a confidence and severity, and filter downstream.

**Legal-document Q&A** (answer questions over a 400-page contract set, with citations).

- Floor: high, because the failure mode is a confident wrong answer about a contractual obligation.
- Deciding constraint: **grounding and citation fidelity**, then **data residency and retention**, then context.
- My pick: **retrieval plus a frontier model with native citation support**, not a long-context stuff. Two reasons: cost (400 pages ≈ 300k tokens × $3/Mtok = $0.90 per question versus ~$0.02 with retrieval), and quality (multi-fact synthesis over 300k tokens is exactly where usable-context falls short of advertised-context). Native citations turn "is this grounded?" from a judgment call into a check.
- Retention and residency likely decide the provider before anything else — this is the archetype of the case where legal picks your deployment surface and you eval whatever is left.

**🗣 Say this in the room:** "Router: cheap tier, strict schema, confidence-routed escalation, and distill once the taxonomy stops moving — it's a volume problem. Code review: frontier reasoning tier at high effort, because latency is soft and a false positive permanently costs adoption. Legal Q&A: retrieval plus native citations rather than long-context stuffing, and residency probably picks my provider before I run a single eval."
