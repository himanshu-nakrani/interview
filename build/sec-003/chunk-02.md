### If you could only show me three things you've built, what would they be and why those three?

The mistake is picking three *impressive* things. Pick three things that **jointly span the axes the role is graded on**, because a portfolio is a coverage argument, not a highlight reel. Three artifacts that all demonstrate the same competence are worth barely more than one.

The three axes that matter for AI Engineer roles at your target companies are: (1) *do you understand the model layer from the inside*, (2) *can you measure quality rather than assert it*, and (3) *can you operate a system with real users, cost and latency*. So:

**Artifact 1 — the from-scratch mechanism piece.** A transformer implemented from nothing, or a KV-cache/attention-variant benchmark, or a small inference server. Its job is to make the "does this backend person actually understand the model" question resolve in ninety seconds without an interview. It does not need to be novel; it needs to be *correct, small, readable, and numerate* — i.e. it must report numbers, not just run.

**Artifact 2 — the eval harness.** A public, reproducible evaluation of something, with a task definition, a dataset you can distribute or regenerate, a grader whose agreement with human labels you measured, and a results table with confidence intervals. This is the single highest-converting artifact in this field and I'll defend that claim in the next question.

**Artifact 3 — the operated system.** Something with users, a cost line, and a post-mortem. For you, the guide plus its Next.js site is exactly this artifact, and it is stronger than a toy chatbot precisely because it has a real ingestion pipeline, real retrieval, real deployment and real traffic.

Notice what is absent: **a chatbot demo, an "AI agent that browses the web," and a LangChain tutorial reimplementation.** These are the three most common portfolio items in this market and their marginal information content is near zero, because the reviewer has seen forty of them and cannot distinguish "I understood this" from "I followed a video."

**⚠ Trap:** breadth as a substitute for depth. Six shallow repos read worse than two deep ones — they signal that you start things. Pin three, archive or unpin the rest. Your GitHub profile is a curated portfolio surface, not an archive, and treating it as an archive is a choice you are making by default.

**🗣 Say this in the room:** "I'd show you three things: an attention implementation with a KV-cache benchmark so you can see I understand the mechanism, an eval harness with a measured grader so you can see how I decide whether something works, and a system I actually operate with a cost model and a post-mortem. They're picked to cover different questions rather than to be three of the same thing."

### Why do you keep saying an eval harness converts better than a chatbot demo? Defend that.

Because of what each artifact is *evidence of*, and evidence is the only currency in a hiring decision.

A chatbot demo is evidence that you can call an API and wire a frontend. That was a differentiating skill in early 2023. It is now the baseline capability of a competent undergraduate with a weekend, and — critically — **the reviewer cannot tell your chatbot from a generated one.** The artifact does not discriminate, so it cannot convert. Worse, a demo invites the question "how do you know it's good?", and if the repo contains no answer, the artifact has actively harmed you.

An eval harness is evidence of the exact competency that the debrief rejects people for. Building one forces you to make and defend a chain of decisions that cannot be faked: what is the task, what is a correct answer, where does the data come from and is it contaminated, is the grader an exact-match, a rubric LLM judge, or a human, **what is the measured agreement between your grader and human labels**, how many examples do you need for the difference you're claiming to be real, and how do you report uncertainty. Every one of those is a five-minute interview conversation with a right answer, and having built one means you have already had the conversation with yourself.

There is a second, less obvious reason: **an eval harness produces a table, and a table is shareable.** A demo has to be experienced; a benchmark table can be screenshotted into a Slack channel by the engineer who found it. Artifacts that other people can forward are the ones that travel.

**📐 Numbers you must know — why your eval needs to be bigger than you think.** If two systems score 72% and 78% on n = 100 examples, is that real? The standard error of a proportion is √(p(1−p)/n) = √(0.75 × 0.25 / 100) ≈ 0.043, so each estimate carries roughly ±4.3 points of one-sigma noise, and the standard error of the *difference* on independent samples is about √(2) × 4.3 ≈ 6.1 points. A 6-point gap is one sigma. It is not a result. Getting that same 6-point gap to two sigma needs roughly n ≈ 400 per arm. **This single calculation, done out loud, marks you as someone who has actually run evals** — and it is why paired evaluation on the *same* items (where you compare per-item wins and losses rather than two independent means) is the standard trick, since it removes item difficulty as a variance source and buys you the same power at a fraction of the sample size.

**⚠ Trap:** publishing an eval where the grader is an LLM and you never validated the grader. An unvalidated judge is a random number generator with good grammar. The fix is cheap and it is the thing that makes your repo credible: hand-label 100 items, report the agreement between your judge and your labels (Cohen's κ or plain agreement percentage, stated), and publish the disagreements. Publishing your grader's failure cases is the strongest credibility signal available in this entire section, because nobody fakes that.

**🗣 Say this in the room:** "The thing I'd point at is the eval harness. Building it meant deciding what a correct answer is, measuring my judge against 100 human labels before trusting it, and being honest about how much of a gap is actually resolvable at my sample size. That's the reasoning I'd bring to your quality problem."

### I'm going to spend ninety seconds on your GitHub before the screen. What should I see?

Ninety seconds is roughly: profile page, read the top of the README of one pinned repo, glance at the file tree, maybe open one file. Design for exactly that.

**The profile page.** A one-line bio that says what you build, and up to six pinned repos ordered so the first one is the strongest. The green contribution square wall is not evidence of anything and I would ignore anyone who reads it as such; do not manufacture it with daily trivial commits, which is a visible and slightly embarrassing pattern.

**The README, which is the whole game.** The reviewer will not read your code before deciding whether to care. The README decides. It must answer, in the first screen — before any install instructions — **what this is, what question it answers, and what the result was.** My template:

1. One sentence: what it is and what it's for.
2. The result, as a table or three bullets with numbers. If it's a benchmark, the table goes here, at the top, above the fold.
3. "How to reproduce" in three commands maximum.
4. Then: the design section — the interesting decisions and their tradeoffs, written as a design doc.
5. Then: limitations, honestly. What it does not do, where the numbers are shaky, what you'd do with another week.

That fifth section is disproportionately powerful. A limitations section is the written form of calibration, and calibration is what senior hiring is actually screening for.

**The file tree.** A reviewer glancing at a tree is asking one question: does this look like a system or like a notebook someone exported? Signals that read well: a `tests/` directory with tests that actually assert something, a pinned dependency file (a lockfile, `pyproject.toml` with versions), a `Makefile` or `justfile` with the three commands from the README, a `results/` or `benchmarks/` directory with committed raw output. Signals that read badly: `notebook_final_v3_copy.ipynb`, a 900-line `main.py`, committed `.env` files, no tests at all.

**⚠ Trap:** the README that opens with "Installation." That structure assumes the reader has already decided to use your thing. Your reader has decided nothing; they are deciding whether you are worth an hour of their calendar. Result first, always.

**🏋 Drill:** open your top pinned repo, start a 90-second timer, and read only what a stranger would read in that window. Write down what you learned. Pass criterion: you can state what the project does, one number it produced, and one design decision the author made. If you can't, rewrite the README — not the code.

### Your repo has a benchmark table in the README. How do I know the numbers aren't garbage?

You don't, unless the repo makes it cheap to check — and **making it cheap to check is the entire value of the artifact.** An unfalsifiable benchmark table is decoration; a falsifiable one is a credential. The difference is a handful of specific practices, each of which is also a thing an interviewer might ask you about.

**Report the setup completely enough to reproduce.** Hardware (exact GPU or CPU, count), driver/CUDA version, library versions pinned, model identity including the exact revision or quantisation, batch size, sequence lengths, sampling parameters (temperature, top-p) and seeds. "Llama on an A100" is not a setup; "Llama-3.1-8B-Instruct at bf16, vLLM 0.x, one A100-80GB, input 1024 / output 128, batch 32, greedy" is. **📅 Volatile:** engine version numbers age fast — pin them in the repo and say the date you ran it.

**Report variance, not just the mean.** Three runs minimum, report median and spread. A single-run throughput number is a sample from a distribution whose width you have not measured, and on shared or virtualised hardware that width can be double digits of percent.

**Separate warm-up from steady state.** First-iteration numbers on a GPU include kernel autotuning, memory allocator warm-up, and possibly graph compilation. Anyone who has profiled a JIT'd system knows this reflex; the same discipline applies. Discard warm-up iterations explicitly and say you did.

**State the metric precisely.** "Latency" is ambiguous in this field in a way it isn't in backend: time-to-first-token, inter-token latency, end-to-end per request, and per-request-under-concurrency are four different numbers and the honest table names which. Throughput needs its concurrency level attached or it is meaningless.

**Commit the raw output.** Not just the summary table — the JSON or CSV your harness emitted, so a skeptic can recompute your aggregates. This is the single cheapest trust-building move available and almost nobody does it.

**Show a control.** If you're claiming your optimisation helped, the table must contain the unoptimised baseline measured the same day on the same box, not a number quoted from someone else's blog post.

**⚠ Trap: the accidental apples-to-oranges comparison.** The classic version is benchmarking two inference engines where one is doing continuous batching and the other is not, or where one is serving a quantised model and the other bf16, and reporting the throughput ratio as if it measured engine quality. If you publish that, someone knowledgeable will notice in public. The defensive habit: for every comparison, write one sentence listing what is held constant — and put that sentence in the README.

**💰 Math:** benchmarks cost money and the repo should say so. Ten configurations × three runs × four minutes a run = 120 minutes of GPU time; on a rented A100 at roughly $1.50–$2.50/hour that is about **$3–$5 of compute** for a table that can carry a portfolio. **📅 Volatile** on the hourly rate. The point to internalise is that the barrier to publishing a real benchmark is not cost — it is rigour, which is why doing it rigorously is a differentiator.

### Someone clones your repo and it doesn't run. What did you fail to do, and how do you make that impossible?

Treat "a stranger cannot run this in ten minutes" as a production incident, because the blast radius is the same: the artifact silently stops converting and you never find out. The debugging discipline here is the same one you'd apply to a flaky deploy — the bug is almost never in the code, it is in the **implicit environment** the code assumed.

**🔍 Failure taxonomy — why a portfolio repo fails to run, in order of frequency:**

1. **Unpinned dependencies.** It worked in March because a transitive dep hadn't broken yet. Fix: a lockfile committed, or `pyproject.toml` with upper bounds, plus the Python version stated. This ecosystem moves fast enough that unpinned means broken within months.
2. **An undocumented secret.** The code reads `OPENAI_API_KEY` and dies with a `KeyError` in a stack trace instead of a message. Fix: a committed `.env.example`, validate config at startup, and fail with a sentence a human can act on.
3. **Undocumented data.** The harness expects a dataset that lives on your laptop. Fix: either commit a small sample, ship a `make data` that downloads it, or generate synthetic data — and make the default path use the small sample so the smoke test runs for free.
4. **Requires a GPU you didn't say it requires.** Fix: state the hardware requirement in the first screen of the README, and provide a tiny-model or CPU path for the smoke test.
5. **Costs money to run and doesn't warn.** Someone runs your eval and burns $40 of API credit. Fix: print the estimated cost and token count before executing, and default to a `--limit 20` sample. This also reads as excellent engineering judgment on its own.
6. **The README drifted from the code.** The command in the README is two refactors old. Fix: this is the one that CI solves — make the README's commands the ones CI executes.

The structural solution is a **GitHub Actions workflow that runs the README's quickstart on a clean runner** on every push, with a mocked or tiny-model path so it doesn't need a GPU or credits. A green badge on a portfolio repo is not decoration; it is a machine-checked claim that a stranger can run this. That single workflow prevents four of the six failures above.

```yaml
# .github/workflows/smoke.yml — the highest-leverage 20 lines in a portfolio repo
name: smoke
on: [push, pull_request]
jobs:
  quickstart:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: make smoke        # the exact command the README tells a stranger to run
```

**⚠ Trap:** "it works on my machine" hidden behind a Dockerfile that you last built four months ago and that pulls `:latest` base images. A Dockerfile is not reproducibility unless something builds it on a schedule. Add a weekly `schedule:` trigger to that workflow and you'll learn about the breakage from an email rather than from a hiring manager's silence.

### You have forty hours. Design me the one portfolio project you'd build, end to end.

Forty hours is two weekends and it is enough — if the scope is chosen so that the *result* is the deliverable and the code is incidental. I'd build a **benchmark-plus-eval artifact that answers a question nobody has published a clean answer to**, because that produces a table, a blog post, and a repo from one body of work, and those three assets have different distribution channels.

Here's a concrete design I'd actually run, weighted toward your target archetypes.

**The question:** *For a retrieval-heavy assistant workload, how do the quality/cost/latency tradeoffs actually land across (a) frontier model with no retrieval and a large context, (b) small model with retrieval, (c) frontier model with retrieval, and (d) small model with retrieval plus a reranker?* This is a question every AI product team at Notion, Glean or Sierra has argued about internally with vibes rather than numbers.

**Hours 0–6: the dataset.** 200 questions over a public corpus you can redistribute — pick something with real structure and real ambiguity. Stratify deliberately: 60 single-hop, 60 multi-hop, 40 unanswerable-from-corpus (this slice is where systems embarrass themselves and where your artifact earns its keep), 40 requiring recency or aggregation. Label answers yourself. This is the expensive part and it is what nobody else will do.

**Hours 6–12: the grader, and validate it.** Rubric-based judge for correctness, plus a groundedness check (is each claim supported by a retrieved span). Then hand-label 100 items and report agreement. Publish the disagreement cases.

**Hours 12–24: the four systems.** Keep the code boring. One retrieval module, one generation module, a config file selecting the arm. Instrument tokens in/out per call, wall-clock per stage, and cost from a pricing table in config.

**Hours 24–32: run, three seeds, record raw output.** Report per-slice results, not just aggregate — the unanswerable and multi-hop slices are the story.

**Hours 32–40: write it up.** README with the table above the fold, a post with the narrative and the surprising result, limitations section, and the cost of running the whole thing.

**💰 Math on what this costs you to run:** 200 questions × 4 arms × 3 seeds = 2,400 generations. Say 6,000 input tokens (retrieved context) and 300 output. Frontier arms (half of them, 1,200 calls): 1,200 × (6,000 × $3/10⁶ + 300 × $15/10⁶) = 1,200 × ($0.018 + $0.0045) = **$27**. Small-model arms: 1,200 × (6,000 × $0.25/10⁶ + 300 × $1.25/10⁶) = 1,200 × ($0.0015 + $0.000375) = **$2.25**. Judge calls at 2,400 × 2 metrics × ~$0.004 ≈ **$19**. Embeddings for the corpus, a few dollars. **Total ≈ $50–$60 for an artifact that can anchor an entire job search.** That arithmetic — in the README — is itself a hiring signal. **📅 Volatile:** prices.

**⚠ Trap:** scoping a *product* instead of a *result* in forty hours. A half-finished product with no users is worth nothing; a small, complete, honestly-measured result is worth a lot. If your forty-hour plan includes "build auth," delete the plan.

### Everyone writes a from-scratch transformer. How do I make mine worth looking at?

You're right that the baseline version is worthless as a differentiator — there are thousands of them, most descended from the same handful of well-known educational repos, and a reviewer can't distinguish yours from a fork. But the underlying activity is still the single best way to actually learn the material, so the answer is not to skip it; it is to **attach a question to it.** A from-scratch implementation with a research question attached stops being a tutorial and becomes an experiment.

Question-attached variants that are genuinely interesting and are all inside a weekend or two:

- **Measure what the textbook asserts.** Train a small model twice, once with the 1/√d_k attention scaling and once without, and plot the loss curves and the attention-logit magnitudes. Now you have *evidence* for the answer you'd otherwise recite in an interview. Same treatment for pre-norm vs post-norm stability at increasing depth, or learned positional embeddings vs RoPE on length generalisation past the training length.
- **Implement one modern component honestly and benchmark it.** Multi-head vs grouped-query vs multi-query attention: implement all three in the same codebase, then report the actual KV-cache bytes per token and the decode throughput at several batch sizes. That table is directly the thing serving interviews ask about, and building it means you'll never fumble the KV-cache arithmetic.
- **Reproduce a paper's headline plot at 1/1000 scale and report where it breaks.** The honest write-up of *where the small-scale reproduction diverged from the paper* is more interesting than a successful reproduction, and it demonstrates research literacy, which is the scarce half of "AI engineer."

**📄 Paper:** Vaswani et al. (2017) — the encoder-decoder transformer built on scaled dot-product attention, which replaced recurrence as the sequence-modelling default. Reimplementing it is table stakes; *measuring one of its claims* is the differentiator.

**⚠ Trap:** the from-scratch repo that has no numbers in it at all. If the README says "an educational implementation of the transformer" and shows no loss curve, no throughput figure, no memory measurement, it is indistinguishable from every other one and it will be treated as such. **Numbers are what make an artifact yours.**

**🗣 Say this in the room:** "I wrote the transformer from scratch mainly to stop hand-waving, but the part I'd actually show you is the GQA comparison — same codebase, three attention variants, measured KV-cache bytes per token and decode throughput at batch 1, 8 and 32. It's where the memory-bandwidth story stopped being something I'd read and became something I'd measured."

### What makes a technical blog post actually reach the people who hire, rather than sinking?

Distribution in this field is not mysterious, but it is unforgiving: **posts that get read are posts that contain something the reader cannot get elsewhere.** Explanation is not that thing — the internet has an unlimited supply of "how attention works" posts, and yours is competing against ones written by people who built the systems. Three formats reliably clear the bar, and they map exactly onto the three artifact types.

**Format 1 — the post-mortem of a system you actually ran.** "We shipped X, here is what it cost, here is what broke, here is the number that made us change the design." Nobody else can write this post because nobody else has your incident. It is the most valuable and the least written, because it requires admitting something went wrong — which is exactly why it signals seniority. Structure it like an incident review: what we built, what we expected, what happened, the timeline, the root cause, what we changed, and what we'd do differently. Include the cost and latency numbers.

**Format 2 — the from-scratch reimplementation with numbers.** As above: the reimplementation is the excuse, the measurement is the content. Title it after the finding, not the activity: not "Building a Transformer from Scratch" but something that names what you measured.

**Format 3 — the benchmark nobody ran.** The most reliably shareable of the three, because it fills a gap people already feel. Pick a comparison that practitioners argue about with anecdotes: structured-output approaches under a strict schema, chunking strategies on a real corpus, reranker cost/benefit at various k, an inference engine comparison held genuinely constant. The rigour requirements from the benchmark question above apply in full — a widely-shared benchmark with a methodological hole is worse for you than no post.

**On distribution mechanics, briefly and honestly:** posts travel when someone with an audience shares them, and that happens when the post has a *specific, quotable finding* — a table, a number, a reversal of conventional wisdom. Title and first paragraph carry almost all the weight; state the finding in both. Post where practitioners in this field actually congregate rather than where you find it comfortable, and link it from your GitHub profile and your resume so it compounds even when it doesn't go anywhere on its own.

**⚠ Trap:** writing five explainer posts on topics you just learned. Beyond the near-zero distribution value, there is a real downside risk: an explainer written from a fresh understanding usually contains a subtle error, and the audience you want is exactly the audience that will spot it. **Write from what you measured, not from what you read this week.**

**📐 Numbers you must know:** the realistic conversion from writing is lumpy and low-frequency, and anyone promising otherwise is selling something. Assume most posts do nothing and that the value is (a) the artifact existing when someone Googles you, and (b) one post in five or ten reaching someone consequential. That expected value is still excellent, because the cost is one weekend and the payoff is a warm inbound — but plan the pipeline as though writing will produce zero interviews, and treat any that arrive as upside.

### Should I keep my demo hosted and live? What does that actually cost me?

Ask what the demo is *for*, and the answer is usually: to let a reviewer see, in under a minute, that a thing works. That goal has three implementations with wildly different costs, and the reflex choice — a live hosted app with a real model behind it — is usually the worst one.

**Option A — a 60-second screen recording, embedded at the top of the README.** Costs nothing, never breaks, never gets abused, works when the reviewer is on a plane, and takes the reviewer zero setup. For most portfolio purposes this is the correct answer and I'd default to it.

**Option B — hosted but keyless and rate-limited.** Correct when the interactivity *is* the point — a search/retrieval product where the interesting thing is trying your own query. This is the case for your guide's site, where a reviewer typing a query and getting a good result *is* the demonstration.

**Option C — hosted, unauthenticated, calling a frontier model on your key.** This is the one I'd push back on in review. You have built an open proxy to a metered API and put it on the public internet. It will be found by scrapers within days.

**💰 Math on why option C is a bad idea:** suppose a scraper hits your endpoint at a modest 2 requests/second for one day: 2 × 86,400 = 172,800 calls. At even 1,500 input and 500 output tokens on a mid-tier model at $3/$15 per million, that's 172,800 × ($0.0045 + $0.0075) = **$2,074 in one day.** The mitigation stack is exactly your day job — a per-IP token bucket, a global daily token budget enforced in Redis with a hard kill, a cheap small model behind the demo rather than a frontier one, a provider-side spend cap, and a billing alert. If you do host it, **put that mitigation stack in the README**; a reviewer seeing "the demo enforces a 50k-token/day global budget and degrades to a cached response beyond it" learns more about your engineering than the demo itself does.

**On keeping things alive long-term:** an idle serverless deployment of a Next.js app or a small FastAPI service is typically a few dollars a month or free on hobby tiers — that is not the cost that hurts. The costs that hurt are the model bill and your attention: a demo that has been broken for three months because a dependency moved is actively negative evidence. **📅 Volatile:** hosting tiers and free-tier limits change constantly.

**⚠ Trap:** the dead link on a resume. A portfolio URL that 404s or times out is worse than having no portfolio, because it converts a neutral prior into "does not check their own work." Put a weekly scheduled uptime check on every URL that appears on your resume — a five-line cron that curls each link and emails you on non-200. You would do this for a production endpoint without thinking; your resume links are a production endpoint.

### What do you actually look for in someone's commit history, and does commit hygiene matter for hiring?

Honestly: almost nobody reads commit history during a screen, so optimising for it is a poor use of time. But it matters in two specific, high-stakes moments, and both are worth ten minutes of preparation rather than a habit overhaul.

**Moment one: the reviewer who is deciding whether you wrote this.** When a repo looks suspiciously polished, a skeptical reviewer will glance at the history. A repo whose entire content arrived in one commit called "initial commit" three days before you applied is a legitimate flag — not proof of anything, but it removes a form of evidence you could have had for free. A history showing incremental construction over weeks, with commits that fix your own bugs, is quietly strong evidence of authorship. **This is the cheapest defence against the "did you just generate this?" question, and it costs nothing except committing as you go.**

**Moment two: your open-source pull requests.** Here hygiene is not cosmetic — it is the actual medium of the interaction. A maintainer decides how much of their scarce attention to give you partly from the shape of your PR: one logical change per commit, a message that says why rather than what, tests in the same PR, no unrelated formatting churn, and a description that states the problem, the approach and the verification. A PR that touches 40 files because your editor reformatted them will not be merged no matter how good the fix is.

The message convention I'd hold to on anything public: a short imperative subject under ~72 characters, a blank line, then a body that explains the *why* and any tradeoff considered. "Fix bug" is a wasted line; "Reject zero-length chunks before embedding — empty inputs produced NaN vectors that silently poisoned recall@10" is a line that makes a stranger trust you.

**⚠ Trap:** rewriting your history to look better before applying. Force-pushing a fabricated multi-week history is detectable (timestamps and their relationship to file content have a way of not matching) and the downside if noticed is catastrophic relative to the tiny upside. The honest version — "I built this over three weekends, here's the history" — is available to you for free if you simply start committing incrementally now.

**🗣 Say this in the room,** if a reviewer probes how a repo was built: "It's about three weekends of evenings — the history shows it. The first weekend was the dataset and the labelling, which was most of the work; the second was the four arms; the third was the write-up and fixing the two places where my judge disagreed with my own labels."
