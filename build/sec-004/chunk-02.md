### Take my target list — Cursor, Perplexity, Notion, Figma, Sierra, Harvey, Glean, Ramp, and then Meta, Google, Amazon, Microsoft, Databricks, Snowflake, Stripe — and sort it by what's actually achievable from India.

I will answer this as a **taxonomy plus a verification procedure**, not as a table of company facts, because company-by-company hiring policy is the most volatile information in this entire guide — offices open, remote policies flip, and a claim I make today is a liability in your loop next quarter. **📅 Volatile: verify every company-specific claim below against the company's own careers page and current job postings before you act on it.**

**Tier A — large employers with mature global entities and routine sponsorship.** Google, Microsoft, Amazon, Meta and similarly-sized companies operate substantial Indian engineering sites *and* sponsor relocation at volume. For these, the highest-probability path is not a US application at all: it is **apply to the Indian site, perform, transfer.** Their LCA footprint is enormous, their internal mobility programs are real, and their India AI orgs do genuine work. The failure mode here is being hired into a support or sustaining role and discovering there is no path from it to the AI org.

**Tier B — enterprise/data-platform companies with real Indian engineering.** Databricks, Snowflake, Stripe and peers have built Indian engineering sites that own production surfaces rather than being test farms. Same pattern as Tier A but with much smaller headcount and thus much more selective hiring — which is good news, because these sites hire senior backend engineers directly into product teams. Verify the *charter* of the Indian site before you apply: ask in the screen what the site owns end to end.

**Tier C — AI-product companies, mostly US-only and mostly in-person.** Cursor, Sierra, Harvey, Perplexity, Ramp and companies at that stage are typically concentrated in one or two offices with a strong in-person culture, and several are explicit that they do not hire remote. Some have opened or announced international offices; the announcements are recent and the reality lags them. For this tier the honest read is: **you are competing for relocation with sponsorship, or you are not competing.** Do not spend your application budget here without checking whether the specific req is remote-eligible, and do not assume an announced office means it is hiring engineers yet.

**Tier D — companies with genuine remote-international or EOR practice.** These exist across every tier and cannot be predicted from company size or fame; the empirical signal is a careers page that lists **specific approved countries** for remote hiring, or a job posting with an "India" or "Remote — Global" location. If India is on the approved-country list, they already have EOR infrastructure and hiring you is a solved problem for their people ops.

**The verification procedure, in order, per company (fifteen minutes each):**
1. Open the careers page; filter by location. Does India appear at all? For engineering, or only for sales/support?
2. If yes, open three engineering reqs at the Indian site and read what the team owns. "Own the X service end to end" versus "support the US team" is the entire difference.
3. If no India, check whether any req is remote-eligible and read the fine print for the country list.
4. Grep the DOL LCA disclosure data for the company's H-1B filing volume and titles.
5. Check LinkedIn for engineers at that company located in India — headcount and titles tell you what the site actually is.

**⚠ Trap: applying to a US-located req and hoping they'll make an exception.** They will not, because the exception is not the hiring manager's to make — it is a legal-entity question owned by people ops, and the answer is "we don't have an entity there" long before anyone reads your resume. The application is not rejected on merit; it is rejected on geography, and it costs you the ability to apply cleanly to a suitable req later.

**🗣 Say this in the room:** "I've sorted your postings by whether the Indian site owns product surfaces or supports US teams, and I'm applying to the former. If there's a US relocation path from that site within two to three years, that's the version of this role I'm most interested in."

### Contractor, employer-of-record, or the company's own subsidiary — what's the difference and which should you push for?

The mental model: **these three differ in who your legal employer is, and that single fact cascades into equity eligibility, benefits, termination protection, tax treatment, and — most importantly — whether you are inside or outside the org chart.**

**Independent contractor.** You invoice; they pay. Nobody withholds anything. In India you register a proprietorship or a company, you handle GST if applicable, and you take advance-tax obligations onto yourself. Advantages: highest gross rate (they are not paying employer-side costs), maximum flexibility, and you can serve multiple clients. Disadvantages that matter more than they sound: **no equity in the overwhelming majority of cases** (incentive stock options in the US are statutorily restricted to employees; contractors, if granted anything, get non-qualified options or nothing), no severance, no notice period, and a real legal risk on *their* side — many jurisdictions will reclassify a full-time-equivalent long-term contractor as a de facto employee, which is exactly why mature companies stop doing this.

**Employer of record.** A third party (Deel, Remote, Velocity Global, Globalization Partners and similar) has a legal entity in India and employs you there, invoicing the US company for your cost plus a fee. You get an Indian employment contract, statutory benefits, PF, payslips. The US company gets to treat you like an employee day to day without incorporating in India. **📅 Volatile:** the EOR vendor landscape consolidates constantly and specific vendors' India capabilities differ; the mechanism is durable, the names are not.

**Subsidiary employment.** The company has its own Indian entity and you are its employee. This is what a GCC is. Best benefits, cleanest equity story, and — the part that matters strategically — **you are an employee of the actual corporate group, which is what an intracompany transfer requires.**

**The ranking I would push for, and why:** subsidiary > EOR > contractor, and the gap between subsidiary and EOR is much larger than it looks. Subsidiary employment gives you the corporate relationship an L-1 needs; EOR does not, because your legal employer is the EOR, not the company. **An EOR employee cannot be intracompany-transferred, because there is no intracompany relationship.** That is the single most important and least-known fact in this question, and it means that accepting an EOR arrangement at a company you hoped to transfer within is quietly closing a door you thought you were opening.

**💰 Math on the EOR overhead you should know is being spent on you.** EOR vendors typically charge either a flat per-employee monthly fee or a percentage of payroll — commonly in the region of a few hundred dollars a month per head, or a mid-single-digit to low-double-digit percentage. **📅 Volatile:** pricing is negotiated and moves. So if your gross is $90,000/yr, the company's fully-loaded cost is that plus statutory India employer contributions plus the EOR margin — call it $100k–110k against a budget line that could have bought a US hire at $220k+ fully loaded. Know this number: it is your leverage, because it demonstrates you are the cheap option, and it is your risk, because a cost-cutting cycle cuts the line item that is easiest to cut and EOR contracts terminate more easily than employment ones.

**⚠ Trap: accepting "we'll convert you to full-time later."** Conversion requires the company to incorporate in India or to already have an entity — it is a corporate-structure decision, not an HR one, and no hiring manager can commit to it. Ask directly: "Do you have an Indian legal entity today?" If the answer is no, the conversion promise is a hope, and I would price the role as if the conversion never happens.

### You get a remote-from-India offer at a US AI startup. Build the comp model. Show me the arithmetic.

The mental model: **there are three reference bands, not one, and negotiation is a fight about which band the conversation anchors to.** The US band for the role, the India local-market band for the role, and the "blended" or "geo-adjusted" band the company invents to split the difference. Whoever names a band first usually wins, which is why the number you walk in with must be constructed, not felt.

**Step 1 — establish the US band.** Use levels.fyi for the company and level, cross-checked against DOL LCA disclosure wage data (which gives you the *base* the company actually certified for that title and location — a hard, filed number, not a self-report) and against Blind or Pave-derived aggregates. Say the model lands at $200k base + $150k/yr equity = **$350k total comp** for a senior AI engineer at a mid-stage US AI company. **📅 Volatile: AI-specific bands have moved sharply; re-benchmark within the month you negotiate.**

**Step 2 — establish the India band for the same work.** Senior engineers at strong Indian sites of US companies plausibly land ₹60–110 LPA all-in depending on company and level. At ₹85 LPA and ₹86/USD that is roughly **$99,000**. **📅 Volatile: both the band and the exchange rate move.**

**Step 3 — the gap is 3.5×, and that is the whole negotiation.** Companies justify the gap on cost-of-living, on local-market competitiveness, and on "we pay the market where you live." The counter-arguments that actually work are: (a) **you are not competing in the local market, you are competing for a US-team seat and delivering US-team output**; (b) **your replacement cost is the US hire they did not make**; and (c) if they want you to work US hours, they are buying your evenings, which is a real cost with a real price.

**Step 4 — construct the ask as a percentage of the US band with a stated rationale.** State the ask as a fraction of **US total compensation** and keep that denominator fixed for the whole negotiation, because the moment you let the conversation slide between "base," "cash" and "total comp" you have lost the ability to compare anything. A remote-from-India ask that lands is typically **45–65% of US total comp**, and the ones at the top of that range are the ones with a written justification. Against the $350k model: **$160–190k cash, plus the same equity grant as a US hire at that level** — that is 46–54% of total comp, and you say the percentage out loud so the concession is visible.

**💰 Worked example of a landed offer.** Ask: $165k cash. Their opening: $95k ("that's above the India market"). Your move — reframe the denominator: "I'm not asking for your US band. Your filed base for this title is $205k and levels.fyi puts total comp around $350k; $165k is 47% of that total, so I've already conceded more than half. At $165k I cost you less than a US hire's employer-side load alone, and I'm the one absorbing the time-zone cost." Realistic landing zone: **$130k–145k cash** — 37–41% of the $350k — plus equity, if you keep equity as a separate, non-tradeable ask. At ₹86/USD, $135k is roughly ₹1.16 crore, comfortably above any local band, which is exactly why this is worth twenty minutes of friction rather than accepting the first number.

**⚠ Trap: negotiating cash and equity as one number.** The company's cash budget for an EOR line item is often genuinely constrained, but equity comes from a different pool with different approvals. Negotiate them in separate conversations. Conceding on cash to "buy" equity is fine; conceding on equity to buy cash is usually a mistake, because the equity is the only part of the package with asymmetric upside and it is the part that geo-adjustment has the weakest justification for touching — a share of the company is worth the same regardless of where the holder sleeps.

**🗣 Say this in the room:** "I want to be straightforward about geography rather than dance around it. I understand you'll geo-adjust cash and I'm not going to argue for the full US band. What I will argue for is that the equity grant shouldn't be geo-adjusted — the strike price and the share count don't depend on where I live, and if I'm doing the same scope as a US hire, the ownership should reflect the scope."

### Am I even eligible for equity as a remote or EOR hire? What actually happens to the options?

Short answer: **usually yes for RSUs and non-qualified options, usually no for the US-tax-advantaged instruments, and always subject to a document you must read rather than assume.**

The mechanism. US companies grant equity under a **stock plan** whose eligibility language defines who can receive grants — typically "employees, directors and consultants of the Company and its subsidiaries." Two consequences follow directly. First, **incentive stock options (ISOs) are restricted by US tax law to employees**, so a contractor or an EOR-employed worker generally cannot receive ISOs and would receive **non-qualified stock options (NSOs)** or RSUs instead. For a non-US-resident this is mostly a non-issue — the ISO tax advantage is a US tax advantage you would not have benefited from anyway. Second, and this is the one that bites: **if your legal employer is an EOR and not the company or a subsidiary, you may fall outside the plan's eligibility definition entirely**, and the company has to grant to you as a "consultant" or amend the plan. Many do it routinely. Some discover mid-onboarding that they cannot, and the grant quietly disappears from the package.

**The rule I enforce: get the grant in the offer letter with share count, strike price (or grant value), vesting schedule, and the plan name — and then ask for the plan document and the form of award agreement before you sign.** You read a database migration before merging it; read this. The four things you are looking for in the award agreement:

1. **Eligibility** — does your engagement type qualify, in writing?
2. **Post-termination exercise window** — the standard is 90 days after termination, which for a non-US holder with a large exercise cost is often equivalent to forfeiture. A 7-to-10-year post-termination exercise window is a real and negotiable term and is worth more than a chunk of cash.
3. **Cliff and acceleration** — standard 1-year cliff, 4-year vest, and what happens on change of control (single- vs double-trigger).
4. **Transfer and settlement mechanics for a non-US holder** — can shares actually be issued to and held by someone in India, and through what broker? This is a real operational question and "we'll figure it out at liquidity" is a bad answer.

**⚠ Trap: assuming an offer letter's "equity: $150,000" line means anything by itself.** Without the share count and the current preferred/409A price you cannot compute the ownership percentage, and "$150k of equity" at a company that just raised at a stretched valuation may be worth a fraction of "$90k of equity" at an earlier-stage one. Ask for: total shares outstanding (fully diluted), your share count, the strike price, and the most recent preferred price. A company that refuses all four is telling you something.

**💰 The exercise-cost trap, with numbers.** Suppose you are granted 40,000 NSOs at a $2.00 strike and, four years later, the 409A is $12.00. Exercising all of them costs $80,000 cash, and in the US the $10.00/share spread would be ordinary income on exercise — for a non-US resident the treatment depends on the treaty and your residency, but the cash requirement does not go away. If your post-termination window is 90 days, leaving that job means finding $80,000 in ninety days or forfeiting the whole grant. **This is why the exercise window is the term I negotiate hardest on, and it costs the company nothing to give.**

### You're an Indian tax resident holding equity in a US company. What's the tax mechanism you need to understand — not as advice, as mechanism?

Stated up front: **this is mechanism, not tax advice, and the specifics change annually. Retain a chartered accountant who has actually handled foreign ESOPs before you do anything irreversible.** With that said, an engineer who cannot describe the mechanism will make an avoidable and expensive mistake, so here is the shape.

**The two taxable events.** For options, Indian tax treats the **exercise** as a perquisite — the difference between the fair market value at exercise and the exercise price is added to your salary income and taxed at your slab rate, with the employer typically obliged to withhold. Then the **sale** is a separate capital-gains event on the difference between the sale price and the FMV that was already taxed as a perquisite. For RSUs the analogous perquisite event happens at **vesting**, since there is no exercise price. The critical implication: **you can owe tax at exercise or vest on paper gains for shares you cannot sell**, which is a genuine cash-flow hazard with private-company equity and the reason "exercise early, exercise often" advice imported from US blogs can be actively dangerous for an Indian resident.

**Holding-period and rate treatment.** Foreign (unlisted-abroad) shares have their own holding-period threshold for long-term versus short-term capital gains treatment, distinct from the threshold for Indian listed shares. **📅 Volatile:** the thresholds and the rates were changed in recent Finance Acts; verify the current year's rules.

**Three compliance obligations people miss and get penalized for:**

1. **Schedule FA.** Indian residents must disclose foreign assets — including foreign shares, vested and unvested-where-applicable, and foreign bank/brokerage accounts — in the Foreign Assets schedule of the return. Non-disclosure exposes you to penalties under the black-money legislation that are grossly disproportionate to the asset value. This is the single most commonly missed obligation among Indian engineers with US equity, and it is not a rounding error.
2. **Foreign tax credit and Form 67.** If US tax is withheld, relief under the India-US double-taxation treaty is claimed through the foreign tax credit mechanism, which requires a timely filed form. Miss the filing and you can lose the credit and be taxed twice.
3. **The LRS ceiling on outbound remittance.** The Liberalised Remittance Scheme caps how much an Indian resident individual may remit abroad per financial year — a limit that becomes relevant if you need to *send dollars out* to exercise options. **📅 Volatile:** the ceiling and the TCS treatment on remittances have both been revised; verify the current figures.

**⚠ Trap: treating an EOR payslip as if the equity is handled.** The EOR administers your salary; it usually does *not* administer the US company's equity plan, and the perquisite-withholding chain that works cleanly for a subsidiary employee often has no owner in an EOR arrangement. Ask explicitly, in writing, before signing: **who computes and withholds the perquisite tax on my equity events, and who issues the documentation I need for my return?** If nobody can name an owner, you are the owner, and you should budget for it.

**💰 Why this is worth an afternoon.** On a $135,000 cash package plus a grant that produces a $60,000 perquisite event in a single year, the marginal Indian tax at the top slab plus surcharge and cess can exceed 40%. Getting the timing, the credit and the disclosure right is worth five figures a year; getting the Schedule FA disclosure wrong is worth a great deal more than that in penalty exposure.

### How do you benchmark compensation and turn a US band into a defensible remote ask?

The mental model: **an ask is defensible when it is derived rather than asserted.** Recruiters have heard every number; they have not heard many *derivations*, and a derivation is very hard to argue with because arguing requires attacking a source rather than attacking you.

**The four sources, ranked by how much weight they carry in an argument:**

1. **DOL LCA disclosure data** — the strongest, because it is the company's own filed wage for a specific title in a specific location, submitted under penalty of perjury. It is base salary only, it lags by a quarter or two, and it is a floor rather than a midpoint (companies file at or above the prevailing wage, and top-of-band hires often exceed it). But when you say "your filed base for Senior Software Engineer in San Francisco is $X," nobody argues.
2. **levels.fyi** — self-reported, so upward-biased at the top end, but it is the shared vocabulary. Its real value is **leveling**: mapping your scope onto the company's actual ladder so that you are negotiating within the right band rather than fighting the wrong one. Getting leveled one step higher is worth more than any negotiation tactic.
3. **Blind and community aggregates** — noisy, occasionally invaluable for a specific company's current behavior (are they matching? are they doing off-cycle refreshes?), and useless as a citation.
4. **The recruiter themself** — "what's the band for this level?" is a fair, normal question and many recruiters answer it, especially where pay-transparency laws require a posted range. Ranges posted under those laws are real data.

For India-side benchmarking, levels.fyi has usable India data for large companies, and beyond that you are largely relying on network. **📅 Volatile:** everything above is a point-in-time reading; re-pull within the month.

**The derivation template I would use, out loud:**

"For this role at your level, your filed base in [location] is $205k, levels.fyi puts total comp for that level around $340k, and your posted range for the equivalent US req is $190–240k base. I'm not asking for that. I'm asking for 50% of the total cash — $165k — on the reasoning that I'm delivering the same scope on your hours, from a location that saves you the employer-side and relocation cost of a US hire, and that this number is still well above what I'd be paid locally, which is what makes the arrangement stable for both of us."

That last clause matters more than people realize. **A remote arrangement priced far above the local band is a retention machine, and saying so out loud converts your ask from a cost into a risk mitigation.**

**⚠ Trap: benchmarking against the wrong level.** Backend engineers moving into AI roles routinely accept a level downgrade because they feel like beginners in the domain, and one level at these companies is commonly $80k–150k a year in total comp. Your leveling is set by scope of ownership, ambiguity handled and blast radius — all of which you have from backend work — not by how many months you have been writing eval harnesses. Fight the leveling conversation, not the number.

### The recruiter asks for your compensation expectations on the first call. You're in India. What do you actually say?

This is a scripted moment and you should have the script memorized, because the improvised version of this answer costs real money — anchoring at the India band on call one caps the whole negotiation, and refusing to answer at all reads as evasive from an international candidate whose geography the recruiter is already uncertain about.

**The mechanism you are managing:** the recruiter's job on call one is to disqualify you cheaply, and one of their two disqualifiers is comp mismatch (the other is work authorization). They need enough of a signal to proceed. Giving them nothing forces them to guess, and the guess is your local market. So you give them a number, but you give them the number attached to a frame.

**🗣 Say this in the room, verbatim:**

"Happy to be concrete so we don't waste rounds. I'm calibrating against the US band for this level rather than the Indian market, because the scope and the hours are the US role's — for context, your filed base for this title is around $X. I'm not asking for the full band; for a remote arrangement I'm targeting $160–180k cash — call it half of total comp — with a standard equity grant for the level. If your budget for a remote hire is structurally below that, I'd rather you tell me now."

Four things that script does. It **answers the question**, so you are not evasive. It **anchors on the US band** and names a source, so the anchor is not arbitrary. It **pre-concedes the geo-discount**, which makes you sound reasonable and removes their best argument. And it **invites early disqualification**, which sounds risky but is the strongest possible signal of confidence and saves you four rounds when the budget genuinely is not there.

**If you have no US band data for that specific company**, substitute: "For a senior AI engineer at a company at your stage, I'm calibrating to a $320–380k US total-comp band and targeting roughly half of the cash for a remote arrangement." Naming the band you are calibrating to is what makes the number defensible.

**⚠ Trap: answering with your current compensation.** In several US jurisdictions the recruiter is legally barred from asking for salary history, and even where they are not, an Indian current-comp number in the conversation is an anchor you will never escape — a ₹45 LPA current package translates to about $52k and every subsequent number will be measured against it. If asked directly for current comp: "I'd rather anchor on the market for the role than on my current package — the geographies aren't comparable. Here's what I'm targeting." Nobody pushes past that.

**⚠ Trap: the "what would make you say yes today" close.** Never answer that with a number on call one. "I'd want to see the full scope and meet the team before I could answer that responsibly" is the correct response and costs you nothing.

### How do you raise time zones in a recruiter screen without disqualifying yourself?

The mental model: **the interviewer's fear is not that you are in IST, it is that you will be unavailable when something breaks or when a decision needs three people in a room. Answer the fear, not the fact.** Candidates who lead with "I'm flexible on hours!" sound like they have not thought about it. Candidates who lead with a *specific committed overlap window and a written-first working style* sound like they have already run a distributed team, which as a backend engineer you effectively have.

**The mechanism to state:** IST is UTC+05:30. Against San Francisco that is **13.5 hours ahead of PST and 12.5 hours ahead of PDT**; against New York, **10.5 hours ahead of EST and 9.5 ahead of EDT**; against London, **5.5 hours ahead of GMT and 4.5 hours ahead of BST**. So a 9:00 a.m. Pacific standup is **9:30 p.m. IST** during US daylight time and **10:30 p.m. IST** in winter — and note that the offset *changes twice a year* because India does not observe daylight saving, which is a scheduling detail that breaks recurring meetings and that you should mention because almost no candidate does.

**🗣 Say this in the room:**

"On time zones — I'm UTC+5:30, so I'd commit to a hard overlap of 9:00 p.m. to 1:00 a.m. IST, which is 8:30 a.m. to 12:30 p.m. Pacific during daylight time. That covers your standup, any design review, and the first half of your day for anything urgent. Outside that window I work asynchronously and write things down — design docs, decision records, PR descriptions that someone can act on without me. The one thing I'd ask for is that decisions that need me get made inside that window rather than at 4 p.m. your time, and I'd flag that India doesn't do daylight saving, so the offset shifts by an hour twice a year and recurring invites need to account for it."

**Why that works:** it converts an objection into a demonstration of exactly the working style distributed teams say they want, and it names a concrete constraint you are asking them to respect — which reads as someone who has done this before, not someone who is grateful to be considered.

**⚠ Trap: promising full US-hours availability.** You will say it in the screen, do it for four months, and then quietly degrade — and the degradation is read as an attitude problem rather than as the predictable consequence of a sleep schedule nobody can hold. Promise four hours you can sustain for two years, not eight you can sustain for one quarter. I have seen more remote arrangements die from over-promising overlap than from having too little.

**⚠ Trap: raising it too late.** If you get to the onsite before geography comes up, you have wasted their time and yours, and the late reveal reads as concealment. Raise it in the first five minutes of call one, as a logistics item you have already solved, not as a confession.

### Design your actual working week against a San Francisco team. Give me the hours.

The mental model: **treat your calendar like a scheduler with two classes of work and a hard deadline, because that is what it is.** Synchronous work is latency-sensitive and must land inside a window you do not control. Deep work is throughput-oriented and should be scheduled where it is cheapest — which for you is a time of day that no US team can ever take from you. The mistake is running an undifferentiated day and letting the overlap window fragment it.

**The schedule I would actually run:**

- **09:30–13:30 IST — deep work block one.** This is 9:00 p.m.–1:00 a.m. Pacific the previous day; nobody is awake, nothing will interrupt you. This is where the hard implementation goes. It is the single most valuable block in a distributed engineer's week and it is a direct consequence of the time zone, not despite it.
- **13:30–15:00 IST — break, meal, exercise, sunlight.** Non-negotiable, because the evening block is what makes the whole thing sustainable and you cannot do a 9 p.m.–1 a.m. block on a depleted body.
- **15:00–18:00 IST — deep work block two / async communication.** Write the PR descriptions, the design docs, the decision records, the review comments. Everything you produce here lands in the US team's morning inbox, which is exactly the handoff property you want.
- **18:00–21:00 IST — off.** Dinner, family, a life. Protect this like a production SLO; it is the thing that fails first.
- **21:00–01:00 IST — the overlap block.** 8:30 a.m.–12:30 p.m. Pacific during PDT. Standup, design reviews, pairing, one-on-ones, incident response. Meetings only — do not schedule implementation here, because you will be interrupted and because your cognition is worse at midnight than at 10 a.m.

**💰 The arithmetic that makes this a feature, not a sacrifice.** That is 7 hours of protected deep work per day against a US-based engineer's realistically 3–4 hours between meetings — you get roughly **double the uninterrupted implementation time**, and your output lands overnight so the team wakes up to finished work. Over a quarter that is a visible throughput difference, and it is the single strongest argument available for renewing and expanding a remote arrangement. Say this explicitly in your first performance conversation; do not assume anyone notices.

**🔍 Failure taxonomy for the distributed week, as a decision procedure:**
- **Symptom: the overlap block creeps later.** Cause: US colleagues scheduling at their 1–3 p.m. Fix: publish your working hours in your calendar as a hard boundary and decline outside them once, early. If you decline once in month one it holds; if you accept for three months it never holds again.
- **Symptom: you are always the one being caught up.** Cause: decisions made in your night, in synchronous channels, without a written record. Fix: demand written decision records as a team norm — frame it as a benefit to *them* (onboarding, audit, the person on PTO), never as an accommodation for you.
- **Symptom: sleep debt accumulating, quality dropping in week 3 of every month.** Cause: a shifted schedule that never stabilizes because the weekend snaps you back. Fix: keep the *same* schedule on weekends for the first two months to let circadian entrainment happen, and treat any night past 1:00 a.m. IST as an incident that needs a root cause, not as commitment.
- **Symptom: invisible to leadership; skipped for promotion.** Cause: you are asleep during the hours when informal visibility happens. Fix: over-invest in written artifacts with your name on them — design docs, post-mortems, benchmark reports — because writing is the only visibility channel that works across a 12.5-hour offset.

### Which India-based employers actually do frontier-adjacent AI work, and how do you tell a real one from a wrapper shop?

The mental model: **evaluate an Indian AI job the way you would evaluate a service — by what it owns in production, not by what it is called.** The title "AI Engineer" in the Indian market spans everything from "owns the retrieval and serving stack for a product with millions of users" to "writes prompts for a client demo," and the résumé consequence of the second one is severe: two years of it makes you *less* hireable at your target companies, because it reads as prompt-tinkering rather than systems ownership.

**The categories that contain real work:**

1. **Global capability centres of US big tech with genuine research or AI-platform charters.** Google and Microsoft both operate long-standing research organizations in India with real publication records; Nvidia has substantial Indian engineering; Adobe, Amazon, Databricks, Stripe and others run Indian engineering sites that own production surfaces. **📅 Volatile:** which specific teams sit in India changes constantly — verify by asking what the site owns, not by trusting the brand.
2. **Indian foundation-model and AI-product companies.** A small number of Indian companies are genuinely training or heavily post-training models and building serving infrastructure rather than reselling API calls. **📅 Volatile:** this set turns over rapidly with funding cycles — evaluate the specific company at the time you apply.
3. **Indian engineering sites of US AI-native companies.** Several US AI companies have announced or opened India offices recently. The announcements outrun the engineering headcount by a year or more; check whether they are actually hiring engineers or only go-to-market.
4. **Deep-tech and quant firms.** Trading firms and a handful of infrastructure companies do genuinely hard ML systems work in India and pay at or above GCC bands.

**The five diagnostic questions I would ask in the interview to separate real from wrapper:**

1. **"What does this team own end to end in production — name the service."** A real answer names a system, a traffic level and an SLO. A wrapper answer names a client.
2. **"Do you have an eval harness, and who owns it?"** This is the single sharpest question in the set. Organizations doing real LLM work have an eval suite with a name, an owner and a CI integration. Organizations doing demo work have a spreadsheet, or nothing, and will say "we test manually" or "the client validates it."
3. **"Do you serve any model yourselves, or is everything a provider API?"** Neither answer is disqualifying — plenty of excellent applied work is pure API — but the answer tells you whether you will get serving-stack depth, and serving depth is the scarcest thing on an AI Engineer résumé.
4. **"What's your cost per resolved task, and has anyone driven it down?"** If nobody can answer, nobody is operating the system; they are shipping it.
5. **"How many engineers here have moved to a US or European site in the last two years?"** The answer is a direct measurement of whether the site is a real part of the company or a cost centre.

**⚠ Trap: taking a "GenAI Lead" title at a services company because the title is bigger.** Your target employers read titles from Indian services firms with heavy discounting and will interrogate scope instead. A senior IC role owning a real production system at a company with an eval culture is worth vastly more in your next loop than a lead title over a team of six building client demos — and the gap widens every year you stay.

**🗣 Say this in the room, when asked why you'd take an India-based role given your US ambitions:** "Because the fastest route to a US AI role is a track record of owning AI systems in production, and this site owns one. I'd rather spend two years shipping a real serving and eval stack here than two years applying from outside with nothing to point at."
