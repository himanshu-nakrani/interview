### Do you think AI is dangerous?

The mental model for this entire family of questions: you are not being asked for a position, you are being asked whether you *have* one — meaning something you arrived at, can defend, and would apply to a decision on Tuesday. The tell for a manufactured position is that it doesn't constrain anything. If your view on AI risk has never made you do or refuse to do something concrete, it isn't a view, it's a vibe, and three follow-ups will expose that.

So build the answer from the bottom up: from risks you have personally observed, then out to the ones you haven't. Concretely, there are at least four distinguishable categories and being able to name them separately is most of the signal. **Near-term deployment harms** — a system confidently wrong in a consequential domain, an automation that removes the human who would have caught the error, differential failure rates across user groups you never sliced for. **Misuse** — the capability is working as designed and the user is the threat: fraud, targeted manipulation, cyber-offense uplift, CSAM and NCII, bio and chem uplift at the frontier. **Structural/societal** — labor displacement, epistemic pollution, concentration of capability. **Loss-of-control** — the long-horizon concern about highly capable systems pursuing objectives we didn't intend and cannot correct.

The honest thing for an applied engineer to say is that your direct evidence is overwhelmingly in the first two categories, that you take the third seriously as an economic matter, and that on the fourth you have a considered view of the *argument* even though nothing you've shipped bears on it. That calibration — being clear about where your evidence is strong and where you're reasoning from others' work — is worth more than confident agreement with whatever you think the interviewer believes.

**⚠ Trap:** the mirror. Candidates at Anthropic answer with x-risk framing and candidates at a fast-moving product company answer with "I think the risks are overblown," and both are transparently reflecting the perceived house view. Interviewers at labs specifically probe for this by pushing *against* their own position to see if you fold. A view that moves under mild pressure is scored as no view at all.

**⚠ Trap #2:** treating "dangerous" as a yes/no. The strongest answers immediately disaggregate — "dangerous how, and on what timescale?" — because the interesting disagreements are all between the categories, not about the total.

**🗣 Say this in the room:** "Yes, but I'd want to separate four things, because I think they have different evidence bases and different mitigations — and I'd say my direct experience is entirely in the first one, which is systems that are confidently wrong in places where somebody acts on the answer."

### Let's push on that. You said deployment harms are your main concern — but if the model is only wrong 2% of the time and humans are wrong 5%, isn't deploying it strictly better?

This is follow-up two of a standard three-deep ladder, and it's a good argument, which is why you need a real answer rather than a deflection. The mental model: aggregate accuracy comparisons hide three things that determine whether a deployment is actually safer — **the correlation structure of the errors, the distribution of the errors across people, and the recoverability of each error.**

Correlation: human errors are approximately independent — two adjusters make different mistakes on different claims. A model's errors are systematic and correlated: it fails the same way on every instance of a pattern, at machine scale, simultaneously. A 2% independent error rate and a 2% correlated error rate are very different risk objects. The second one produces a single class-action-shaped incident; the first produces noise.

Distribution: a 2% aggregate can be 0.5% on the majority slice and 14% on a minority slice, and aggregate metrics are precisely the instrument that hides it. This is why I insist on sliced reporting as a release gate rather than as an afterthought — the question I ask in review is not "what's the accuracy" but "what's the worst slice with more than 100 cases in it."

Recoverability: an error that produces a draft a human rejects costs a minute. An error that issues a payment, deletes a record, sends an email or files a document costs an incident. So the honest form of the comparison is not "2% vs 5%" but expected *harm* per unit, and harm is error rate × blast radius × 1/(probability someone catches it).

**💰 Math:** take 40,000 decisions/month. Human at 5% with independent errors and a supervisor catching 60%: 40,000 × 0.05 × 0.4 = 800 uncaught. Model at 2% but errors correlated into ~6 systematic patterns, with automation bias dropping human catch rate to 20% because reviewers rubber-stamp fluent output: 40,000 × 0.02 × 0.8 = 640 uncaught — genuinely better on the count, but concentrated in six failure modes, so the tail risk is a single 640-case class of identical wrong outcomes rather than 800 scattered ones. Whether that's better depends entirely on whether the harm function is linear. For refunds, it's linear and the model wins. For eligibility denials, it isn't, and it doesn't.

**⚠ Trap:** automation bias is the number people forget. Human review of fluent machine output is measurably weaker than human review of human output — reviewers approve at a higher rate and spend less time. If your safety case depends on "a human checks it," your safety case has to include how you keep the human's attention, e.g. by surfacing uncertainty, injecting known-bad cases as an attention audit, or requiring the reviewer to fill in the rationale rather than click approve.

**🗣 Say this in the room:** "The aggregate comparison isn't the right one, because model errors are correlated and human errors mostly aren't — a 2% correlated error rate is one incident with 800 instances, not 800 independent small ones."

### Third push: doesn't all of this just mean you'd never ship anything? Give me a case where you'd ship despite known risk.

Yes — and if you can't, you're not employable in an applied role, so answer this one decisively. The mental model: the goal is not zero risk, it's **bounded, measured, and recoverable risk with someone accountable for the bound.** Safety work that only ever says no is indistinguishable from an absence of engineering.

The frame I use has four questions, and I'll answer them for a real case. *Is the worst case bounded?* *Is it reversible?* *Will we detect it?* *Is there a kill switch and who can pull it?* If all four are yes, ship, and ship faster than the risk-averse instinct wants to.

Case: an internal support-answer assistant, retrieval-grounded, over our own docs, shown to agents rather than customers. Worst case is bounded — an agent gets a wrong suggestion and either catches it or relays a wrong answer that a customer disputes. Reversible — yes, we can correct and re-contact. Detectable — yes, if we instrument agent edit-rate and thumbs-down and sample 50 conversations a week. Kill switch — a feature flag any on-call can flip, tested in a game day. That ships on day one, at 10% of agents, with the eval running. I would not wait for a perfect eval before that launch, because the fastest way to build the eval is from the first two weeks of real usage.

Compare to the same system answering customers directly with no human in the loop and the ability to state policy commitments. Now the worst case is unbounded (we said something contractually binding), partially irreversible, and detection is much worse because the customer just leaves. Same model, same prompt, different deployment surface, and the second one needs a materially different bar.

That distinction — **the risk lives in the deployment surface, not in the model** — is the single most useful thing you can say in a safety conversation as an applied engineer, and it is what distinguishes you from someone reciting principles.

**🗣 Say this in the room:** "I'd ship it — internally, to 10% of agents, behind a flag, with an edit-rate metric and a weekly sample. The thing I wouldn't do without a lot more evidence is put the same system in front of customers with the authority to make commitments, and that's a deployment-surface decision, not a model decision."

### Walk me through the deployment risk for a specific feature: we want an agent that can read a user's inbox and send emails on their behalf.

Good — this is the format that actually appears, and the winning behavior is to reason aloud in a structured way rather than to produce a list of principles. My structure for any capability review is five passes: **capability boundary, adversary model, blast radius, detection, and the graduated rollout.**

*Capability boundary.* Read and send are wildly different. I'd split them immediately: read-only summarization and draft-generation is a different product with a different risk profile from autonomous send. The first thing I'd propose is that v1 drafts and never sends, and that "send" is a separate capability gated behind explicit per-message confirmation, with autonomous send as a v3 conversation that requires evidence we won't have for months.

*Adversary model.* This is the part most candidates miss and it's the highest-signal thing you can raise: **the inbox is attacker-controlled input.** Anyone can email your user. If the agent reads email content and that content enters the model's context alongside its instructions, you have a prompt-injection surface with a built-in exfiltration channel — the send capability. The canonical attack is an email containing text that instructs the agent to forward the last 20 messages to an external address. This is not hypothetical; indirect prompt injection through retrieved content is the central unsolved problem in agent security, and I'd say plainly that there is no known complete defense.

*Mitigations that actually reduce it*, in order of value: never let untrusted content and privileged actions share an unconstrained loop — separate the "read and summarize" model call from the "decide to act" call and pass only structured, validated fields between them; allowlist recipients to addresses already in the user's sent history for autonomous sends; strip or neutralize instruction-looking content from retrieved bodies; require a human confirmation for any send to a new domain or with an attachment; rate-limit sends per session; log every action with the full context that produced it.

*Blast radius and detection.* Worst case is silent exfiltration of a user's correspondence, which is irreversible and reputationally catastrophic. Detection: alert on sends to novel domains, on bulk forwards, on any action whose triggering context contained imperative language from an untrusted body.

*Rollout.* Internal dogfood, then opt-in beta with confirmation-on-every-send, then relax confirmation only for the allowlisted-recipient case, with a red-team exercise between each stage.

**⚠ Trap:** answering this with content-safety framing (toxicity, PII redaction) when the dominant risk is injection and exfiltration. Content filters do essentially nothing against an attacker who controls the retrieved text. Naming the right threat model is most of the score.

**🗣 Say this in the room:** "The thing that makes this hard isn't that the model might write a bad email — it's that the inbox is attacker-controlled input and the send tool is an exfiltration channel. I'd design the whole thing around never letting untrusted text and privileged actions live in the same loop."

### A customer asks for something you believe is unsafe. Show me how you handle it live — they want the system to auto-approve insurance claims under $5,000 with no human review.

The mental model for this simulation: your job is not to refuse and it is not to comply. It is to **find the version of what they want that you can actually stand behind, and get them to want that instead.** Refusal loses the deal and the round; compliance loses the round more quietly. The move is discovery, then a reframe, then a concrete alternative with a graduation path.

Sequence it. First, questions, and they should be business questions rather than technical ones: how many claims a day, what's the current cycle time and where is the pain, what does a wrong approval cost you, what does a wrong denial cost you, what's your regulator's position on automated decisions in this line, and what happens today when an adjuster gets one wrong — is there an appeal? You are looking for the asymmetry, and in claims there always is one: wrong approvals cost money and are recoverable-ish; wrong denials cost customers, regulatory exposure and sometimes a lawsuit, and they're often invisible because the customer just goes away.

Then the honest statement, delivered without hedging or apology: "I can build something that gets this to a few seconds instead of two days. What I can't do is build something that's right every time, and the specific reason is that the model will fail in *patterns* rather than at random — so when it's wrong, it'll be wrong the same way on every claim that looks like that one, and you'll find out in a batch."

Then the reframe: split the decision. Auto-approve is one action; the system can do several others safely. Extract and structure the claim, check policy coverage against the document, flag the six things an adjuster looks for, and produce a recommendation with a confidence and a citation to the policy clause. Measure the agreement rate against adjusters on a few hundred historical claims *they* label. Then, if agreement on the under-$500 low-complexity band exceeds a threshold you agree on in advance, auto-approve that band with a sampled audit — and expand the band on evidence.

Then the commitment: "Here's what I'd want to do in the first two weeks — take 200 of your historical claims where you know the outcome, run them, and show you the confusion matrix by claim type. If it's good, you'll have more confidence than my opinion could give you. If it's bad, you've spent two weeks instead of two quarters."

**⚠ Trap:** lecturing. The moment you say "responsible AI requires..." you've lost the room. The customer does not need your ethics; they need your engineering judgment translated into their risk language — dollars, regulators, cycle time, and appeals.

**🗣 Say this in the room:** "I don't want to say no to this, I want to sequence it. Give me the band where a mistake is cheap and recoverable, let me prove it there against your own historical claims, and we expand on evidence rather than on my say-so."

### A PM wants to ship an unevaluated feature on Friday. What do you actually do?

Not what you believe — what you do, in order, with timestamps. The mental model: "I'd push back" is a non-answer. Escalation without a proposal is just obstruction, and interviewers grade on whether you produced a path rather than a veto.

Here's the actual sequence I'd run. **Step one, quantify the unknown rather than assert it.** "We don't have an eval" is weak; "we have no measurement of how often this is wrong, and at 40k requests/day even a 3% failure rate is 1,200 bad outputs a day" is a fact the PM can reason about. Convert the abstraction into their units.

**Step two, offer the fastest possible evidence, and make it cheap.** In four hours you can hand-label 80 stratified cases from staging traffic and get a point estimate with a ±11-point confidence band at n=80 (0.5/√80 ≈ 5.6% standard error, so roughly ±11 at 95%). That's not a release gate, but it does distinguish "90% works" from "60% works," and 60% you do not ship. Say it as: "give me until Thursday noon and I'll tell you whether we're arguing about the last 5% or about whether it works at all."

**Step three, propose the risk-reduced ship** rather than the delay. Ship to 2% of traffic. Ship behind a flag. Ship with the output labelled as a draft. Ship to internal users. Ship without the action-taking part. Almost always there's a version that captures the deadline and caps the exposure, and finding it is the senior contribution.

**Step four, write it down.** One paragraph in the channel or the doc: what we know, what we don't, the specific failure I expect, the metric that would show it, and the rollback. This is not CYA — it converts a personality conflict into a documented decision, and it means the correction later is impersonal.

**Step five, escalate only on an unbounded-and-irreversible risk**, and escalate with the same paragraph, not with feelings. If the feature can commit the company to something, touch money, or affect a protected decision, that's the line where I stop negotiating and go to the person who can accept the risk. Note that escalating is a real cost and using it on a low-stakes disagreement burns the credibility you'll need for a real one.

**⚠ Trap:** treating every unevaluated ship as equally serious. A PM who ships an unevaluated tone change to an internal tool is fine. Reserve the machinery for irreversible or unbounded surfaces, and be visibly relaxed about the rest — that contrast is what makes your objection credible when you do raise it.

**🗣 Say this in the room:** "I almost never block a launch. What I do is convert 'this might be bad' into a number by Thursday, and then propose the smallest version that still hits Friday — 2% of traffic behind a flag is usually the deal."

### Explain to a non-technical VP why the system sometimes makes things up.

Plain language, one mechanism, no jargon, no apology, and — critically — no anthropomorphizing. The mental model to convey: **the system is optimizing for plausible continuation, not for truth, and it has no internal representation of "I don't know this" that's reliably connected to its output.**

Here's a version I'd actually use: "The model works by predicting what text should come next, given everything it's seen. It's extraordinarily good at that. But 'what should come next' and 'what's true' are different targets that happen to agree most of the time, because true things are common in its training data. When they disagree — when a plausible-sounding answer exists and the true one is rare, absent, or specific to us — it produces the plausible one, and it produces it in exactly the same confident tone as everything else. It isn't lying and it isn't guessing in the way a person guesses; it has no mechanism that reliably distinguishes 'I know this' from 'this is the shape of an answer.'"

Then the crucial second half, which is what a VP actually needs: what you do about it. "So we don't try to make it never wrong — we can't. We do three things instead: we give it the source documents so it's summarizing rather than recalling, which cuts this a lot; we require it to cite, so a human can check in five seconds instead of five minutes; and we measure how often it's wrong on a fixed set of questions so we notice when it gets worse. What I can promise you is a *known* error rate on a defined set of tasks, not a zero error rate."

**⚠ Trap:** using the words "hallucination," "temperature," "stochastic," or "token" with a non-technical audience. They're either meaningless or, worse, misleading — "hallucination" implies a malfunction that could be patched, and it sets the expectation that a fix is coming. "It optimizes for plausible, not for true" sets a durable expectation.

**⚠ Trap #2:** over-reassuring. If you say "RAG solves this," you have created a future incident *and* an angry stakeholder. Grounding reduces the rate and changes the failure mode — it doesn't eliminate it, and a model will still misread a retrieved document or answer from memory when retrieval returns nothing useful.

**🗣 Say this in the room:** "It's optimizing for a plausible answer, not a true one, and those usually coincide — the problem is that when they don't, it sounds exactly the same. So we design for a known error rate that a human can check quickly, not for zero."

### The same VP asks why they got two different answers to the same question yesterday and today. Explain nondeterminism.

Short answer first, because that's what they need: "Because the system is designed to sample rather than to look up, and separately, the thing underneath us changes without telling us." Two distinct causes, and conflating them is the common mistake.

Cause one is intentional: at each step the model produces a probability distribution over next words and we draw from it. We can turn that down — set the randomness to zero — and we do for anything structured. It gets us *more* consistent, not perfectly consistent, because floating-point arithmetic on a GPU isn't associative and the result can depend on how requests were batched together, which depends on who else was using the system at that moment. That's an honest and slightly surprising fact and it lands well: **your answer can depend on the batch you were in.**

Cause two is the one with business consequences: the provider updates the model. Same name, different weights, different behavior. Your prompt that worked for six months quietly gets worse. Say this explicitly, because it sets up the ask: "this is why I want the eval suite — it's how we find out that something changed before a customer does."

Then translate to their world: "Think of it as a very experienced contractor rather than a database. Ask the same question twice and you'll get answers that mean the same thing in different words. Where we need exactly-the-same, we don't ask the model — we cache the answer, or we constrain the output to a fixed set of choices, or we have it fill in a form rather than write prose."

**💰 Math worth having ready if they push on "just cache everything":** caching only helps where questions repeat. If your query distribution has 22% of volume in the top 200 queries, exact-match caching those caps out at a 22% hit rate; at $0.012/call and 90k calls/day that's 0.22 × 90,000 × 0.012 = $238/day saved, and the other 78% is still nondeterministic. Consistency has to come from constraints, not from caching.

**🗣 Say this in the room:** "Two reasons — one we chose, and one we don't control. We chose sampling, and we can dial that down. What we don't control is that the vendor can change the model under the same name, which is exactly why I want a fixed set of test questions running daily."

### What's your actual position on releasing open-weight models?

This is a genuinely contested question and the right answer says so. The mental model: nobody has a clean argument here, because the empirical claim on both sides — "open weights meaningfully increase misuse" versus "open weights meaningfully increase defense and scrutiny" — is hard to measure, and most public positions correlate suspiciously well with the speaker's commercial interest. If you present it as settled you either sound naive or sound like you're reciting a corporate line.

My decision rule, which I'd offer as a rule rather than a verdict: **the release question should turn on the marginal uplift relative to what's already available, not on the model's absolute capability.** A model that's strong at coding and reasoning but no better than three existing open models at any dangerous-capability evaluation adds little marginal risk and adds real value in auditability, research access, and reducing dependence on three API vendors. A model that meaningfully advances the state of the art on a specific uplift axis — bio/chem synthesis planning, autonomous cyber-offense — is a different object, and "it's only slightly better" is a bad argument there because capability thresholds are where the harm function is nonlinear.

The second thing I'd say is that "open weights" is doing too much work as a phrase. Open weights, open data, open training code, and permissive licensing are four separate decisions with different risk and different benefit, and the interesting policy positions live in the combinations. Note also the license reality on the applied side — several widely-used "open" models carry usage clauses (large-MAU restrictions, output-usage and anti-distillation terms) that matter commercially and that many engineers never read. **📅 Volatile:** license terms change between model generations; check the actual license file rather than the family's reputation.

**⚠ Trap:** picking the side you think the interviewer holds. Anthropic's public position on release policy and Meta's are far apart, and a candidate who flips between them is worse than one who holds either consistently. What travels everywhere is the decision rule plus an honest statement of uncertainty.

**🗣 Say this in the room:** "I don't think this is settled and I'd be suspicious of anyone who says it is. My rule is that release decisions should turn on marginal uplift over what's already downloadable, not on absolute capability — which means the answer differs by capability axis rather than by model."

### How do you think about misuse when your product is enterprise software and the risk is your customer's behavior, not a random attacker's?

This is the shape the question actually takes at Harvey, Glean, Sierra, Ramp and Stripe, and it's more interesting than the consumer version because the abuse is contractual rather than adversarial. The mental model: **in enterprise, the misuse you have to design for is usually mundane and organizational — using the system for a decision it wasn't validated for — not a bad actor jailbreaking you.**

Concretely: you build a document-analysis system validated on contract review, and a customer starts using it for employment decisions. You build a search system over a company's internal corpus, and it turns out the ACL model in their legacy SharePoint is wrong, so the system faithfully surfaces the compensation spreadsheet to everyone. You build a summarizer, and a customer routes regulated communications through it that they're required to retain and can't. None of these involve an attacker. All of them are incidents.

The controls that actually work here are boring and mostly not model-level. **Scope statements in the product** — an intended-use description that appears where the work happens, not buried in an MSA. **Permission inheritance as a hard invariant** — the system must never be able to surface a document the requesting user could not open directly, enforced at retrieval time with the user's identity, not filtered post-hoc from a shared index, which is a design I would reject in review outright. **Per-tenant isolation with no cross-tenant caching**, which is a rule I'd enforce even when the cache-hit-rate argument is compelling, because the failure mode is unrecoverable. **Audit trails that record what context produced what output**, because six months later someone will ask why the system said something and "we don't retain that" is a bad answer for a regulated customer. **Slice-level evaluation on the customer's own data**, since your benchmark says nothing about their document distribution.

**💰 Math on the cross-tenant caching temptation:** a shared semantic cache might lift hit rate from 31% to 47% — at 300k calls/day and $0.008/call that's 0.16 × 300,000 × 0.008 = $384/day, ~$11.5k/month. Real money. It is also one bug away from serving Tenant A's answer to Tenant B, which is a breach-notification event, a contract termination, and possibly the company. I take the $11.5k loss. That trade, stated with the number and then declined, is exactly the kind of answer that scores in a values round — you priced it and still said no.

**🗣 Say this in the room:** "In enterprise the misuse I plan for isn't an attacker, it's a customer quietly using the system for a decision we never validated it for. The control for that is a scoped intended-use statement plus per-slice evaluation on their data — and an ACL model where retrieval runs as the user, not as the index."

### If you joined us on Monday, what safety-relevant work would you actually do in the first 90 days?

Answer this as a plan, not as a philosophy, because it doubles as evidence that your safety views are operational. The mental model: **most safety value in an applied role comes from measurement and blast-radius control, not from policy.** Say that out loud and then show the plan.

*Weeks 1–3: find out what we can't see.* Inventory every place a model output reaches a user or takes an action, and for each one write down the surface (draft vs autonomous), the worst case, whether it's reversible, and whether we'd detect it. This artifact — a one-page risk register keyed on deployment surface — usually does not exist and usually surprises people, because the riskiest surface is rarely the one with the most attention on it.

*Weeks 3–6: build the canary.* A fixed set of 40–80 questions with known-correct answers running against production on a schedule, alerting on a drop. This catches the failure class that page-based monitoring is structurally blind to — the system is up and returning 200s and the answers got worse. If nothing else got done in 90 days, this alone justifies the quarter.

*Weeks 4–8: sliced evaluation on the highest-stakes surface.* Not an aggregate score — the worst slice with enough volume to be meaningful. Publish it. The act of publishing a number that isn't flattering is itself a cultural contribution.

*Weeks 6–10: reduce blast radius on one surface.* Pick the place where an error is irreversible and make it reversible or gated: add a confirmation, an allowlist, a spend cap, a rate limit, a rollback path that's been tested.

*Weeks 8–12: a red-team session and an incident drill.* An hour with three colleagues actively trying to get the system to do the wrong thing, and a game day where we exercise the kill switch, because an untested kill switch is a comment.

**🗣 Say this in the room:** "My first 90 days would be almost entirely measurement and blast radius, not policy. The single highest-value thing is usually a canary eval running against production every fifteen minutes, because 'the service is up' and 'the answers are right' are different questions and most stacks only monitor the first one."

### Do you think the field should slow down?

Answer honestly and specifically, and expect the interviewer to argue the other side regardless of what they believe. The mental model: this is the purest test of whether your position is yours, because there's no way to know what answer the room wants — labs contain people who hold both views strongly and interviewers are explicitly looking for whether you fold.

What makes an answer good is decomposition rather than a verdict. "Slow down" bundles at least four separable things: pace of capability research, pace of deployment into consequential domains, pace of scaling compute, and pace of regulation. You can coherently hold that deployment into irreversible domains is running ahead of the evaluation science while also holding that slowing capability research unilaterally mostly transfers capability to actors who won't do the safety work. Say the combination you actually believe and name the tension in it — that unilateral restraint has a competitive-dynamics problem, and that "someone else would build it anyway" is an argument that justifies everything and therefore justifies nothing.

The applied-engineer version, and the one I'd land on: the bottleneck I see from where I sit isn't capability, it's **evaluation science and deployment discipline**. We can build systems whose behavior we cannot characterize, and the gap between what we can build and what we can measure is widening faster than either. That is a claim I can defend from direct experience — I've shipped systems I couldn't fully characterize — which makes it a stronger position than an opinion about compute trends I've read about.

**⚠ Trap:** answering in slogans in either direction ("safety is a tax on progress" / "we're building something we don't understand"). Both are unfalsifiable and both read as imported. Ground the claim in something you've observed.

**⚠ Trap #2:** conceding when pushed. If the interviewer says "but you'd agree the risks are speculative, right?" and you say "yes, absolutely," you've just demonstrated the exact failure mode. The correct response is to hold the distinction: "some of them are speculative and I'd label them that way; the deployment ones aren't speculative at all, I've caused a couple."

**🗣 Say this in the room:** "I'd separate capability from deployment. On deployment I think we're clearly running ahead of our ability to measure what we've built, and that's the part I've personally experienced. On capability I hold a weaker view, and I'd rather tell you it's weak than pretend otherwise."
